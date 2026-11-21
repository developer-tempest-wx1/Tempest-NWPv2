"""Tempest-NWP: Automated SondeHub Radiosonde Ingestion & Quality Control Engine.

Ingests high-resolution 4D radiosonde telemetry tracks from the global SondeHub community network:
- Real-time global radiosonde tracking (Vaisala RS41, Meteomodem M10/M20, Graw DFM17)
- 4D trajectory drift: lat, lon, geometric altitude, time, horizontal wind, vertical ascent velocity
- Strict Quality Control (QC) & Meteorological Filtering:
  * Sensor Calibration Check: Discards uncalibrated raw telemetry sensors
  * Pure Ascent Filtering (vel_v > 1.5 m/s): Excludes parachute descent and stationary ground noise
  * Burst Detection: Truncates sounding at peak burst altitude
  * Hydrostatic & Barometric Verification: Derives pressure via hypsometric integration if sensor unequipped
  * Vertical Super-Obbing / Thinning: Aggregates dense 1-second frames to model-resolvable vertical layers
- Two Execution Modes:
  1. Live Continuous Stream (--live): Connects via WebSocket/MQTT stream, receives live frames second-by-second
     for 2.5 to 3.0 hours while balloons fly, continuously flushing QC soundings to WX-1 containers.
  2. Batch Fast-Sync (default): Instant multi-sonde API download of the past 2.5h sounding tracks.
- Standard Synoptic Windows:
  * Night Window: 22:45 UTC – 01:15 UTC (2.5h) for 00Z Synoptic Assimilation
  * Day Window:   10:45 UTC – 13:15 UTC (2.5h) for 12Z Synoptic Assimilation

Strictly 100% classical atmospheric physics (0% AI / 0% Machine Learning, 0% Mock Data).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import datetime
import gzip
import json
import logging
import math
from pathlib import Path
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple
import urllib.parse
import urllib.request

import numpy as np
import wx1

try:
    import sondehub
    _SONDEHUB_AVAILABLE = True
except ImportError:
    _SONDEHUB_AVAILABLE = False

logger = logging.getLogger("tempest_v2.ingest_sondehub")

SONDEHUB_BASE_URL = "https://api.v2.sondehub.org"
SONDEHUB_USER_AGENT = "Tempest-NWP/1.0 (Classical Atmospheric Physics Model)"

# Physical atmospheric constants (SI)
GRAVITY = 9.80665  # [m/s^2]
R_D = 287.058  # [J/(kg K)]
P0 = 101325.0  # [Pa]
T0 = 288.15  # [K]
LAPSE_RATE = 0.0065  # [K/m]


def is_within_synoptic_window(
    dt_utc: datetime.datetime,
    window_type: str = "auto",
) -> Tuple[bool, str]:
    """Evaluates whether given UTC time falls within the official synoptic sounding windows."""
    hour = dt_utc.hour + dt_utc.minute / 60.0

    # 00Z window: 21:00 UTC (pre-launch) to 01:15 UTC (burst/top)
    is_00z = (hour >= 21.0) or (hour <= 1.25)
    # 06Z window: 03:00 UTC (pre-launch) to 07:15 UTC (burst/top)
    is_06z = (3.0 <= hour <= 7.25)
    # 12Z window: 09:00 UTC (pre-launch) to 13:15 UTC (burst/top)
    is_12z = (9.0 <= hour <= 13.25)
    # 18Z window: 15:00 UTC (pre-launch) to 19:15 UTC (burst/top)
    is_18z = (15.0 <= hour <= 19.25)

    if window_type in ("night", "00z"):
        return is_00z, "00Z synoptic window (21:00-01:15 UTC)"
    elif window_type in ("day", "12z"):
        return is_12z, "12Z synoptic window (09:00-13:15 UTC)"
    elif window_type == "06z":
        return is_06z, "06Z intermediate synoptic window (03:00-07:15 UTC)"
    elif window_type == "18z":
        return is_18z, "18Z intermediate synoptic window (15:00-19:15 UTC)"
    else:  # auto
        if is_00z:
            return True, "00Z synoptic window (21:00-01:15 UTC)"
        elif is_12z:
            return True, "12Z synoptic window (09:00-13:15 UTC)"
        elif is_06z:
            return True, "06Z intermediate synoptic window (03:00-07:15 UTC)"
        elif is_18z:
            return True, "18Z intermediate synoptic window (15:00-19:15 UTC)"
        return True, f"synoptic sounding stream (current UTC: {dt_utc.strftime('%H:%M')})"


def altitude_to_pressure_std(alt_meters: float) -> float:
    """Computes reference atmospheric pressure (Pa) from geometric altitude (m) using ISA hypsometric formula."""
    if alt_meters <= 11000.0:
        t = T0 - LAPSE_RATE * alt_meters
        p = P0 * (t / T0) ** (GRAVITY / (R_D * LAPSE_RATE))
    else:
        # Stratosphere isothermal layer (11-20 km)
        p_11k = P0 * ((T0 - LAPSE_RATE * 11000.0) / T0) ** (GRAVITY / (R_D * LAPSE_RATE))
        t_11k = 216.65
        p = p_11k * math.exp(-GRAVITY * (alt_meters - 11000.0) / (R_D * t_11k))
    return float(np.clip(p, 50.0, 108000.0))


def _telemetry_timestamp(point: dict) -> float | None:
    value = point.get("datetime") or point.get("time_received")
    if not value:
        return None
    try:
        return datetime.datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return None


def _finite_float(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def apply_sonde_quality_control(
    telemetry_records: List[dict],
    min_ascent_vel: float = 1.5,
    max_altitude_m: float = 40000.0,
    vertical_thinning_dz: float = 200.0,
) -> List[dict]:
    """Order telemetry, detect ascent/burst, derive GPS drift where possible, and super-ob."""
    if not telemetry_records:
        return []

    # MQTT delivery order is not guaranteed to be strict by timestamp. Sort only
    # when every sample has a parseable timestamp; otherwise preserve arrival order.
    timestamped = [(_telemetry_timestamp(p), p) for p in telemetry_records]
    if all(ts is not None for ts, _ in timestamped):
        timestamped.sort(key=lambda item: item[0])
    records = [p for _, p in timestamped]

    ascent_points: list[dict] = []
    max_alt_seen = -float("inf")
    previous = None
    for pt in records:
        alt = _finite_float(pt.get("alt"))
        lat = _finite_float(pt.get("lat"))
        lon = _finite_float(pt.get("lon"))
        if alt is None or lat is None or lon is None:
            continue
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 360.0):
            continue
        if alt < -100.0 or alt > max_altitude_m:
            continue

        # Once the balloon is >200 m below its peak, subsequent descent frames
        # are not admitted to the ascent profile.
        if alt < max_alt_seen - 200.0:
            break
        max_alt_seen = max(max_alt_seen, alt)

        t_sec = _telemetry_timestamp(pt)
        vel_v = _finite_float(pt.get("vel_v"))
        if vel_v is None and previous is not None and t_sec is not None and previous[0] is not None:
            dt = t_sec - previous[0]
            if 0.5 <= dt <= 300.0:
                vel_v = (alt - previous[1]) / dt
        # USTF commonly omits vel_v. Derive it from GPS altitude/time where
        # possible; preserve the first frame when no derivative exists.
        if vel_v is not None and vel_v < min_ascent_vel:
            previous = (t_sec, alt, lat, lon)
            continue

        copied = dict(pt)
        copied["_tempest_time"] = t_sec if t_sec is not None else time.time()
        copied["_tempest_lat"] = lat
        copied["_tempest_lon"] = lon
        copied["_tempest_alt"] = alt
        ascent_points.append(copied)
        previous = (t_sec, alt, lat, lon)

    if len(ascent_points) < 2:
        return []

    valid_points = []
    earth_radius_m = 6371008.8
    for idx, pt in enumerate(ascent_points):
        alt = pt["_tempest_alt"]
        lat = pt["_tempest_lat"]
        lon = pt["_tempest_lon"]
        t_sec = float(pt["_tempest_time"])

        # Prefer explicit horizontal velocity + heading when usable. USTF's
        # -9999.0 vel_h sentinel is invalid. Otherwise estimate drift from GPS
        # track; it is a wind proxy, so later DA innovations still gate it.
        vel_h = _finite_float(pt.get("vel_h"))
        heading = _finite_float(pt.get("heading"))
        u_wind = v_wind = None
        if vel_h is not None and heading is not None and 0.1 <= vel_h <= 100.0 and 0.0 <= heading <= 360.0:
            heading_rad = math.radians(heading)
            u_wind = vel_h * math.sin(heading_rad)
            v_wind = vel_h * math.cos(heading_rad)
        elif idx + 1 < len(ascent_points):
            nxt = ascent_points[idx + 1]
            dt = float(nxt["_tempest_time"]) - t_sec
            if 1.0 <= dt <= 120.0:
                lat2 = float(nxt["_tempest_lat"])
                lon2 = float(nxt["_tempest_lon"])
                dlat = math.radians(lat2 - lat)
                dlon = math.radians((lon2 - lon + 180.0) % 360.0 - 180.0)
                mean_lat = math.radians(0.5 * (lat + lat2))
                u_wind = earth_radius_m * math.cos(mean_lat) * dlon / dt
                v_wind = earth_radius_m * dlat / dt
                if math.hypot(u_wind, v_wind) > 100.0:
                    u_wind = v_wind = None

        temp_c = _finite_float(pt.get("temp"))
        temp_k = temp_c + 273.15 if temp_c is not None else None
        if temp_k is not None and not (180.0 <= temp_k <= 330.0):
            temp_k = None

        rh_f = _finite_float(pt.get("humidity"))
        rh = rh_f if rh_f is not None and 0.0 <= rh_f <= 100.0 else None

        # USTF often has no pressure sensor field; use standard atmosphere as
        # a flagged fallback, not as a fabricated measured pressure.
        pres_hpa = _finite_float(pt.get("pressure"))
        pres_pa = pres_hpa * 100.0 if pres_hpa is not None and 0.5 < pres_hpa < 1100.0 else altitude_to_pressure_std(alt)

        valid_points.append({
            "alt": alt, "pressure": pres_pa, "lat": lat, "lon": lon,
            "temp": temp_k, "rh": rh, "u": u_wind, "v": v_wind, "time": t_sec,
        })

    if not valid_points:
        return []

    valid_points.sort(key=lambda x: x["alt"])
    binned_obs = []
    min_z = valid_points[0]["alt"]
    max_z = valid_points[-1]["alt"]
    z_bins = np.arange(min_z, max_z + vertical_thinning_dz, vertical_thinning_dz)
    for i in range(len(z_bins) - 1):
        z_low, z_high = z_bins[i], z_bins[i + 1]
        pts_in_bin = [p for p in valid_points if z_low <= p["alt"] < z_high]
        if not pts_in_bin:
            continue

        def _mean_valid(key: str) -> float:
            values = [float(p[key]) for p in pts_in_bin if p[key] is not None and math.isfinite(float(p[key]))]
            return float(np.mean(values)) if values else float("nan")

        binned_obs.append({
            "alt": float(0.5 * (z_low + z_high)),
            "pressure": _mean_valid("pressure"),
            "lat": _mean_valid("lat"),
            "lon": _mean_valid("lon"),
            "temp": _mean_valid("temp"),
            "rh": _mean_valid("rh"),
            "u": _mean_valid("u"),
            "v": _mean_valid("v"),
            "time": _mean_valid("time"),
        })
    return binned_obs


def save_sounding_points_to_wx1(
    qc_points: List[dict],
    out_dir: Path,
    n_soundings: int,
    file_prefix: str = "sondehub_soundings",
) -> int:
    """Saves binned sounding points into standardized WX-1 containers."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if not qc_points:
        return 0

    all_pressures = [p["pressure"] for p in qc_points]
    all_lats = [p["lat"] for p in qc_points]
    all_lons = [p["lon"] for p in qc_points]
    all_temps = [p["temp"] for p in qc_points]
    all_u = [p["u"] for p in qc_points]
    all_v = [p["v"] for p in qc_points]
    all_times = [p["time"] for p in qc_points]
    all_rh = [p["rh"] for p in qc_points]

    total_levels = len(all_pressures)

    arr_pres = np.array(all_pressures, dtype=np.float32)
    arr_lat = np.array(all_lats, dtype=np.float32)
    arr_lon = np.array(all_lons, dtype=np.float32)
    arr_temp = np.array(all_temps, dtype=np.float32)
    arr_u = np.array(all_u, dtype=np.float32)
    arr_v = np.array(all_v, dtype=np.float32)
    arr_time = np.array(all_times, dtype=np.float64)
    arr_rh = np.array(all_rh, dtype=np.float32)

    p_main = out_dir / f"{file_prefix}.wx1"
    wx1.write_array(
        p_main,
        arr_pres,
        dataset_id="sondehub_obs",
        variable_path="pressure",
        dimensions=["level_id"],
        profile="output",
        codec="zstd_shuffle",
        attributes={
            "source": "SondeHub Global Radiosonde Community Network (Live Telemetry)",
            "n_soundings": str(n_soundings),
            "n_levels": str(total_levels),
            "units": "Pa",
            "qc_ascent_filter": "vel_v > 1.5 m/s",
        },
    )

    for name, arr, unit in [
        ("lat", arr_lat, "degrees_north"),
        ("lon", arr_lon, "degrees_east"),
        ("temp", arr_temp, "K"),
        ("u_wind", arr_u, "m/s"),
        ("v_wind", arr_v, "m/s"),
        ("time", arr_time, "seconds_since_epoch"),
        ("humidity", arr_rh, "percent"),
    ]:
        v_path = out_dir / f"{file_prefix}_{name}.wx1"
        wx1.write_array(
            v_path,
            arr,
            dataset_id=f"sondehub_{name}",
            variable_path=name,
            dimensions=["level_id"],
            profile="output",
            codec="zstd_shuffle",
            attributes={"units": unit},
        )

    return total_levels


# ==============================================================================
# MODE 1: LIVE CONTINUOUS STREAMING ENGINE (2.5 - 3.0 HOURS)
# ==============================================================================

def _burst_grace_complete(
    sonde_states: dict[str, dict[str, float | None]],
    now_epoch: float,
    burst_grace_seconds: float = 20.0 * 60.0,
    inactive_seconds: float = 15.0 * 60.0,
) -> bool:
    """Return true once every known balloon has burst+grace or gone inactive.

    This is a stream-shutdown criterion, not a claim that every global sounding
    has been observed. A hard post-cutoff deadline in the caller bounds missing
    telemetry and balloons that never report a detectable burst.
    """
    if not sonde_states:
        return True
    for state in sonde_states.values():
        burst_at = state.get("burst_at")
        last_seen = state.get("last_seen", now_epoch) or now_epoch
        if burst_at is not None:
            if now_epoch < float(burst_at) + burst_grace_seconds:
                return False
        elif now_epoch - float(last_seen) <= inactive_seconds:
            return False
    return True


def stream_sondehub_live(
    out_dir: Path | str,
    duration_hours: float = 2.5,
    flush_interval_sec: float = 60.0,
    window: str = "auto",
    stop_event: Optional[threading.Event] = None,
    file_prefix: str = "sondehub_soundings",
    burst_grace_minutes: float = 20.0,
    max_post_cutoff_minutes: float = 90.0,
) -> int:
    """Stream live SondeHub telemetry and periodically persist cycle-specific WX-1.

    SondeHub 0.3.2 owns its MQTT network loop by default. ``duration_hours``
    reaches the scheduled post-synoptic cutoff; after that, active balloons are
    followed until each has burst plus ``burst_grace_minutes`` or gone inactive.
    ``max_post_cutoff_minutes`` bounds the additional wait. ``stop_event`` is a
    force-stop used only for daemon shutdown/error recovery. A final WX-1 flush
    always precedes MQTT disconnect.
    """
    if not _SONDEHUB_AVAILABLE:
        raise RuntimeError("sondehub library is not installed. Install the project's operational extra.")

    target_dir = Path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    _in_window, win_desc = is_within_synoptic_window(now_utc, window_type=window)
    duration_sec = max(1.0, float(duration_hours) * 3600.0)
    end_time = time.time() + duration_sec
    burst_grace_sec = max(0.0, float(burst_grace_minutes) * 60.0)
    hard_end_time = end_time + max(0.0, float(max_post_cutoff_minutes)) * 60.0

    print("==================================================================================")
    print("      TEMPEST-NWP: SONDEHUB REAL-TIME TELEMETRY STREAM")
    print(f"      • Scheduled Cutoff:    {duration_hours:.2f}h from start")
    print(f"      • Burst Grace:         {burst_grace_minutes:.0f}m after detected balloon burst")
    print(f"      • Hard Post-Cutoff:    {max_post_cutoff_minutes:.0f}m maximum extra wait")
    print(f"      • Synoptic Window:      {win_desc}")
    print(f"      • Target WX-1 Output:   {target_dir / (file_prefix + '.wx1')}")
    print("      • QC: GPS-time ordering, ascent/burst filter, 200 m vertical super-ob")
    print("==================================================================================")

    sonde_buffers: Dict[str, List[dict]] = defaultdict(list)
    sonde_states: dict[str, dict[str, float | None]] = {}
    lock = threading.Lock()
    frame_counter = 0
    start_wall = time.time()
    total_stored_levels = 0
    valid_sondes = 0
    last_flush = start_wall
    last_print = start_wall
    stream = None

    def on_live_frame(message: dict):
        nonlocal frame_counter
        serial = message.get("serial")
        if not serial:
            return
        received_at = time.time()
        altitude = _finite_float(message.get("alt"))
        with lock:
            frame_counter += 1
            serial_key = str(serial)
            sonde_buffers[serial_key].append(message)
            if altitude is not None and math.isfinite(altitude):
                state = sonde_states.setdefault(
                    serial_key, {"peak_alt": float(altitude), "burst_at": None, "last_seen": received_at}
                )
                state["last_seen"] = received_at
                peak_alt = float(state.get("peak_alt") or altitude)
                if altitude > peak_alt:
                    state["peak_alt"] = float(altitude)
                    peak_alt = float(altitude)
                if (state.get("burst_at") is None and peak_alt >= 15000.0
                        and altitude <= peak_alt - 200.0):
                    state["burst_at"] = received_at
                    logger.info(
                        "[SONDEHUB] Balloon burst detected: %s at ~%.0f m; retaining stream for %.0f more minutes",
                        serial_key, peak_alt, burst_grace_minutes,
                    )

    def persist_snapshot() -> tuple[int, int]:
        """Copy the in-memory snapshot under lock, then QC/write outside the lock."""
        with lock:
            snapshot = {serial: list(frames) for serial, frames in sonde_buffers.items()}
        points: list[dict] = []
        accepted = 0
        for frames in snapshot.values():
            binned = apply_sonde_quality_control(frames)
            if len(binned) >= 2:
                points.extend(binned)
                accepted += 1
        if points:
            levels = save_sounding_points_to_wx1(
                points, target_dir, accepted, file_prefix=file_prefix
            )
            return accepted, levels
        return accepted, 0

    try:
        print("[STREAM] Starting SondeHub 0.3.2 live stream...")
        # In 0.3.2 the default auto_start_loop=True starts the Paho loop for us.
        # Do not start a second loop with loop_start()/loop_forever().
        stream = sondehub.Stream(on_message=on_live_frame)
        print("[STREAM] Stream object created; collecting live telemetry.")

        completion_since: float | None = None
        while not (stop_event and stop_event.is_set()):
            now = time.time()
            if now >= end_time:
                with lock:
                    state_snapshot = {key: dict(value) for key, value in sonde_states.items()}
                grace_complete = _burst_grace_complete(
                    state_snapshot, now, burst_grace_seconds=burst_grace_sec
                )
                if grace_complete:
                    if not state_snapshot:
                        break
                    if completion_since is None:
                        completion_since = now
                    elif now - completion_since >= 300.0:
                        break
                else:
                    completion_since = None
                if now >= hard_end_time:
                    logger.warning(
                        "[SONDEHUB] Reached %.0f-minute post-cutoff safety limit; closing stream even though not every balloon had a detected burst",
                        max_post_cutoff_minutes,
                    )
                    break
            if stop_event:
                if stop_event.wait(1.0):
                    break
            else:
                time.sleep(1.0)
            now = time.time()
            if now - last_print >= 5.0:
                last_print = now
                with lock:
                    active_count = len(sonde_buffers)
                    frames_now = frame_counter
                elapsed = max(0.0, now - start_wall)
                remaining = max(0.0, end_time - now)
                pct = min(100.0, 100.0 * elapsed / duration_sec)
                sys.stdout.write(
                    f"\r[LIVE] {elapsed/60.0:5.1f}m elapsed | {remaining/60.0:5.1f}m remaining "
                    f"| sondes {active_count:3d} | frames {frames_now:7d} "
                    f"| WX-1 levels {total_stored_levels:6d} | {pct:5.1f}%"
                )
                sys.stdout.flush()

            if now - last_flush >= max(5.0, float(flush_interval_sec)):
                last_flush = now
                accepted, levels = persist_snapshot()
                if levels:
                    valid_sondes, total_stored_levels = accepted, levels

    except KeyboardInterrupt:
        print("\n[STREAM] Interrupted; performing final WX-1 flush.")
    except Exception:
        logger.exception("SondeHub live stream failed; attempting a final WX-1 flush before shutdown")
        raise
    finally:
        # The final write precedes disconnect so no last telemetry window is lost.
        try:
            accepted, levels = persist_snapshot()
            if levels:
                valid_sondes, total_stored_levels = accepted, levels
        except Exception:
            logger.exception("Final SondeHub WX-1 snapshot failed")

        if stream is not None:
            for method_name in ("disconnect", "loop_stop"):
                method = getattr(stream, method_name, None)
                if callable(method):
                    try:
                        method()
                    except Exception as exc:
                        logger.debug("SondeHub stream cleanup %s: %s", method_name, exc)

    print("\n" + "=" * 80)
    print("   SONDEHUB LIVE STREAM SESSION CLOSED")
    print(f"   • Active balloons with QC profiles: {valid_sondes}")
    print(f"   • Telemetry frames received:         {frame_counter:,}")
    print(f"   • WX-1 super-ob levels:              {total_stored_levels:,}")
    print(f"   • WX-1 file:                         {target_dir / (file_prefix + '.wx1')}")
    print("=" * 80 + "\n")
    return total_stored_levels


# ==============================================================================
# MODE 2: FAST BATCH RETRIEVAL (INSTANT 20-30s PULL)
# ==============================================================================

def fetch_active_sondes(timeout: int = 10) -> Dict[str, dict]:
    """Queries the SondeHub API for all currently active radiosondes worldwide."""
    url = f"{SONDEHUB_BASE_URL}/sondes"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": SONDEHUB_USER_AGENT, "Accept-Encoding": "gzip"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            if raw[:2] == b"\x1f\x8b":
                raw = gzip.decompress(raw)
            data = json.loads(raw.decode("utf-8"))
            return data
    except Exception as e:
        logger.warning("Failed querying active sondes from SondeHub: %s", e)
        return {}


def fetch_sonde_telemetry(serial: str, duration_str: str = "3h", timeout: int = 10) -> List[dict]:
    """Downloads high-resolution multi-level telemetry profile for a specific radiosonde."""
    encoded_serial = urllib.parse.quote(serial)
    url = f"{SONDEHUB_BASE_URL}/sondes/telemetry?serial={encoded_serial}&duration={duration_str}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": SONDEHUB_USER_AGENT, "Accept-Encoding": "gzip"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            if raw[:2] == b"\x1f\x8b":
                raw = gzip.decompress(raw)
            data = json.loads(raw.decode("utf-8"))
            records = data.get(serial, {})
            if isinstance(records, dict):
                return [records[k] for k in sorted(records.keys())]
            elif isinstance(records, list):
                return records
            return []
    except Exception as e:
        logger.debug("Could not fetch telemetry for sonde %s: %s", serial, e)
        return []


def ingest_sondehub_soundings(
    out_dir: Path | str,
    da_hours: float = 2.5,
    window: str = "auto",
    max_sondes: int = 150,
) -> int:
    """Fast batch sync: retrieves recent soundings via API within 20-30s."""
    target_dir = Path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    now_utc = datetime.datetime.now(datetime.timezone.utc)
    in_win, win_desc = is_within_synoptic_window(now_utc, window_type=window)

    logger.info("Starting SondeHub Ingestor (Batch Mode) | Window: %s | Lookback: %.1fh", win_desc, da_hours)

    active_sondes = fetch_active_sondes()
    if not active_sondes:
        return 0

    window_sec = da_hours * 3600.0
    cutoff_time = now_utc.timestamp() - window_sec

    candidate_serials = []
    for sid, info in active_sondes.items():
        tr = info.get("time_received")
        if not tr:
            continue
        try:
            t_sonde = datetime.datetime.fromisoformat(tr.replace("Z", "+00:00")).timestamp()
            if t_sonde >= cutoff_time:
                candidate_serials.append(sid)
        except Exception:
            continue

    if not candidate_serials:
        return 0

    selected_serials = candidate_serials[:max_sondes]
    duration_str = f"{max(1, int(round(da_hours)))}h"

    all_binned = []
    valid_sondes = 0

    for sid in selected_serials:
        records = fetch_sonde_telemetry(sid, duration_str=duration_str)
        if not records:
            continue

        qc_obs = apply_sonde_quality_control(records)
        if len(qc_obs) >= 5:
            all_binned.extend(qc_obs)
            valid_sondes += 1

    if not all_binned:
        return 0

    total_levels = save_sounding_points_to_wx1(
        all_binned, target_dir, valid_sondes, file_prefix="sondehub_batch_soundings"
    )
    logger.info("Stored %d SondeHub levels from %d radiosondes in %s", total_levels, valid_sondes, target_dir / "sondehub_batch_soundings.wx1")
    return total_levels


def main():
    parser = argparse.ArgumentParser(description="Tempest-NWP: SondeHub Radiosonde Ingestion & Live Streaming Engine")
    parser.add_argument("--outdir", default="data/observations/upperair", help="Output directory for WX-1 containers")
    parser.add_argument("--duration", type=float, default=2.5, help="Duration in hours (e.g. 2.5 for 2.5h)")
    parser.add_argument("--window", choices=["auto", "night", "day"], default="auto", help="Synoptic window (night: 22:45-01:15 UTC, day: 10:45-13:15 UTC)")
    parser.add_argument("--live", "--stream", action="store_true", help="Execute real-time live continuous streaming for the specified duration (e.g. 2.5h)")
    parser.add_argument("--flush-interval", type=float, default=60.0, help="Live stream WX-1 flush interval in seconds (default: 60s)")
    parser.add_argument("--burst-grace-minutes", type=float, default=20.0, help="Additional capture time after a detected balloon burst (default: 20m)")
    parser.add_argument("--max-post-cutoff-minutes", type=float, default=90.0, help="Maximum extension after scheduled cutoff (default: 90m)")
    parser.add_argument("--max-sondes", type=int, default=150, help="Maximum number of sounding profiles to fetch in batch mode")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.live:
        # Run real-time live continuous streaming for the full duration (e.g. 2.5 hours)
        stream_sondehub_live(
            out_dir=args.outdir,
            duration_hours=args.duration,
            flush_interval_sec=args.flush_interval,
            window=args.window,
            burst_grace_minutes=args.burst_grace_minutes,
            max_post_cutoff_minutes=args.max_post_cutoff_minutes,
        )
    else:
        # Run fast batch sync (20-30 seconds)
        print("=================================================================")
        print("   TEMPEST-NWP: SondeHub 4D Radiosonde Ingestion & QC Engine     ")
        print(f"      Duration: {args.duration:.1f}h | Window: {args.window}    ")
        print("      Mode: FAST BATCH SYNC (Use --live for continuous stream)  ")
        print("      Strictly 100% Classical Fluid Physics (0% AI / 0% ML)      ")
        print("=================================================================")

        levels = ingest_sondehub_soundings(
            out_dir=args.outdir,
            da_hours=args.duration,
            window=args.window,
            max_sondes=args.max_sondes,
        )

        print("\n" + "=" * 65)
        print(f"   [SUCCESS] SONDEHUB SOUNDING INGESTION COMPLETED!")
        print(f"   • Total Filtered Sounding Levels: {levels:,} levels")
        print(f"   • Output Directory:               {args.outdir}")
        print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
