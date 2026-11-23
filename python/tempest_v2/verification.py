"""Station Observation Loader, 3D-Var Driver, and Live KNMI/DWD Verification for Tempest-NWP v2.

Strictly 100% deterministic classical data assimilation & verification (0% AI / 0% ML, 0% mock data).
"""

from __future__ import annotations

import datetime
import logging
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import wx1

from tempest_v2.da.balance import BalanceOperator
from tempest_v2.da.lhn_radar import compute_lhn_tendencies
from tempest_v2.da.var3d import (
    RecursiveSpatialFilter,
    Var3DConfig,
    Var3DResult,
    VarObservation,
    VarObservationOperator,
    apply_analysis_increments_with_bounds,
    compute_iau_tendencies,
    solve_3dvar_pcg,
)
from tempest_v2.ic_builder import stagger_uv, unstagger_uv
from tempest_v2.projection import GRAVITY_M_S2, KAPPA, P0_PA, R_D, DomainSpec, VerticalGrid
from tempest_v2.titan_qc import run_titan_buddy_check

logger = logging.getLogger("tempest_v2.verification")


def compute_mslp_pa(
    spec: DomainSpec,
    hs_m: np.ndarray,
    state_3d: Dict[str, np.ndarray],
    *,
    theta_sfc_ref: float = 288.15,
    p_sfc_ref: float = 101_325.0,
    n_ref_hz: float = 0.01,
) -> np.ndarray:
    """Compute Mean Sea Level Pressure [Pa] hypsometrically from bottom model layer."""
    hs = np.asarray(hs_m, dtype=np.float64)
    z_0 = float(spec.vertical.zeta_s[0]) + float(spec.vertical.B_s[0]) * hs
    ref0 = VerticalGrid.reference_state(
        z_0[None, :, :],
        theta_sfc_k=theta_sfc_ref,
        p_sfc_pa=p_sfc_ref,
        brunt_vaisala_hz=n_ref_hz,
    )
    th0 = ref0["theta_bar"][0] + np.asarray(state_3d["theta_p"][0], dtype=np.float64)
    pi0 = np.maximum(1e-4, ref0["pi_bar"][0] + np.asarray(state_3d["pi_p"][0], dtype=np.float64))
    t0 = np.maximum(160.0, th0 * pi0)
    p0 = P0_PA * np.power(pi0, 1.0 / KAPPA)
    qv0 = np.maximum(0.0, np.asarray(state_3d.get("qv", np.zeros_like(t0))[0], dtype=np.float64))
    tv0 = t0 * (1.0 + 0.608 * qv0)
    return (p0 * np.exp((GRAVITY_M_S2 * z_0) / (R_D * np.maximum(tv0, 160.0)))).astype(np.float32)


_OBS_DIR_TS_RE = re.compile(r"^(\d{8})_(\d{2,4})Z$")


def _parse_obs_dir_utc(dirname: str) -> Optional[datetime.datetime]:
    """Parse `YYYYMMDD_HHMMZ` or `YYYYMMDD_HHZ` observation directory timestamp into UTC datetime."""
    m = _OBS_DIR_TS_RE.match(dirname.strip())
    if m is None:
        return None
    date_part, time_part = m.group(1), m.group(2)
    try:
        if len(time_part) == 4:
            dt = datetime.datetime.strptime(f"{date_part}_{time_part}", "%Y%m%d_%H%M")
        elif len(time_part) == 2:
            dt = datetime.datetime.strptime(f"{date_part}_{time_part}", "%Y%m%d_%H")
        else:
            return None
        return dt.replace(tzinfo=datetime.timezone.utc)
    except ValueError:
        return None


def _to_utc_dt(dt: datetime.datetime) -> datetime.datetime:
    return dt.astimezone(datetime.timezone.utc) if dt.tzinfo else dt.replace(tzinfo=datetime.timezone.utc)


def _expand_obs_search_dirs(
    obs_dir: Path,
    valid_time_utc: Optional[datetime.datetime] = None,
) -> List[Path]:
    roots = [obs_dir]
    runs_sub = obs_dir / "runs"
    ref_dt = _to_utc_dt(valid_time_utc) if valid_time_utc is not None else None
    for parent in (obs_dir, runs_sub):
        if parent.is_dir():
            ts_dirs = sorted(
                d for d in parent.iterdir() if d.is_dir() and len(d.name) >= 11 and d.name.endswith("Z")
            )
            if ts_dirs:
                if ref_dt is not None:
                    parsed: List[Tuple[float, Path]] = []
                    for d in ts_dirs:
                        dt_d = _parse_obs_dir_utc(d.name)
                        if dt_d is not None:
                            parsed.append((abs((dt_d - ref_dt).total_seconds()), d))
                    if parsed:
                        parsed.sort(key=lambda item: (item[0], item[1].name))
                        roots.extend(item[1] for item in parsed)
                    else:
                        roots.append(ts_dirs[-1])
                else:
                    roots.append(ts_dirs[-1])
    search_dirs: List[Path] = []
    seen: set[Path] = set()
    for r in roots:
        for sub in (r, r / "sfc", r / "upperair", r / "satellite", r / "marine"):
            if sub not in seen:
                seen.add(sub)
                search_dirs.append(sub)
    return search_dirs


def _load_obs_prefix_from_dir(d: Path, prefix: str) -> Optional[Dict[str, np.ndarray]]:
    cand = d / f"{prefix}.wx1"
    main_path = cand if cand.is_file() else None
    if main_path is None and not any(
        (d / f"{prefix}_{suf}.wx1").is_file()
        for suf in ("u_wind", "lat", "latitude", "temp", "pressure", "ztd", "ice_conc")
    ):
        return None

    all_keys = (
        "pressure",
        "lat",
        "lon",
        "latitude",
        "longitude",
        "elev",
        "temp",
        "temperature",
        "salinity",
        "depth",
        "u_wind",
        "v_wind",
        "time",
        "rh",
        "ztd",
        "sigma",
        "sst",
        "ice_conc",
        "ice_thick",
    )
    fields: Dict[str, np.ndarray] = {}
    if main_path is not None:
        with wx1.open_dataset(main_path) as ds:
            for k in all_keys:
                if k in ds:
                    fields[k] = np.asarray(ds[k].values)
            if not fields and len(ds.data_vars) == 1:
                only_var = next(iter(ds.data_vars))
                fields[str(only_var)] = np.asarray(ds[only_var].values)

    for k in (
        "pressure",
        "lat",
        "lon",
        "latitude",
        "longitude",
        "elev",
        "temp",
        "salinity",
        "depths",
        "u_wind",
        "v_wind",
        "time",
        "rh",
        "ztd",
        "sigma",
        "sst",
        "ice_conc",
        "ice_thick",
    ):
        if k not in fields:
            sub_p = d / f"{prefix}_{k}.wx1"
            if sub_p.is_file():
                with wx1.open_dataset(sub_p) as sds:
                    if k in sds:
                        fields[k] = np.asarray(sds[k].values)
                    elif len(sds.data_vars) > 0:
                        first = next(iter(sds.data_vars))
                        fields[k] = np.asarray(sds[first].values)

    if "lat" not in fields and "latitude" in fields:
        fields["lat"] = fields["latitude"]
    if "lon" not in fields and "longitude" in fields:
        fields["lon"] = fields["longitude"]
    if "temp" not in fields and "temperature" in fields:
        fields["temp"] = fields["temperature"]
    if "lat" not in fields or "lon" not in fields:
        return None
    return fields


def _read_obs_prefix(
    obs_dir: Path,
    prefix: str,
    *,
    valid_time_utc: Optional[datetime.datetime] = None,
    max_window_hours: Optional[float] = None,
) -> Optional[Dict[str, np.ndarray]]:
    search_dirs = _expand_obs_search_dirs(Path(obs_dir), valid_time_utc=valid_time_utc)
    ref_ts = _to_utc_dt(valid_time_utc).timestamp() if valid_time_utc is not None else None
    fallback_fields: Optional[Dict[str, np.ndarray]] = None

    for d in search_dirs:
        fields = _load_obs_prefix_from_dir(d, prefix)
        if fields is None:
            continue
        if fallback_fields is None:
            fallback_fields = fields
        if ref_ts is not None and max_window_hours is not None and "time" in fields:
            t_arr = np.asarray(fields["time"], dtype=np.float64)
            t_fin = t_arr[np.isfinite(t_arr) & (t_arr > 1e8)]
            if t_fin.size > 0 and float(np.min(np.abs(t_fin - ref_ts))) > float(max_window_hours) * 3600.0:
                continue
        return fields
    return fallback_fields


def _qsat_water_scalar(t_k: float, p_pa: float) -> float:
    tc = max(173.15, min(373.15, t_k)) - 273.15
    es = 611.21 * math.exp(17.502 * tc / (tc + 240.97))
    es = min(es, 0.90 * p_pa)
    return 0.622 * es / max(100.0, p_pa - 0.378 * es)


def load_station_obs_wx1(
    obs_dir: Path | str,
    spec: DomainSpec,
    *,
    valid_time_utc: Optional[datetime.datetime] = None,
    max_window_hours: float = 6.0,
    max_sfc_window_hours: float = 0.75,
    theta_sfc_ref: float = 288.15,
    p_sfc_ref: float = 101_325.0,
    n_ref_hz: float = 0.01,
    enable_titan_qc: bool = True,
) -> Tuple[List[VarObservation], List[Dict[str, Any]]]:
    """Load QC-filtered surface, upper-air, GNSS ZTD, satellite wind, and marine WX-1 observations inside `spec`.

    Raw-only containers (e.g. `satellite_atms_microwave_radiances.wx1`) are intentionally excluded
    from assimilation in compliance with AGENTS.md Rule 3.
    """
    odir = Path(obs_dir)
    margin_x = 2.0 * spec.dx_m
    margin_y = 2.0 * spec.dy_m
    x_min, x_max = float(spec.x_m[0]) + margin_x, float(spec.x_m[-1]) - margin_x
    y_min, y_max = float(spec.y_m[0]) + margin_y, float(spec.y_m[-1]) - margin_y

    eff_sfc_win_h = min(float(max_sfc_window_hours), float(max_window_hours))

    ref_ts: Optional[float] = None
    if valid_time_utc is not None:
        ref_ts = _to_utc_dt(valid_time_utc).timestamp()

    obs_list: List[VarObservation] = []
    stn_rows: List[Dict[str, Any]] = []

    def _in_domain_and_window(
        lat: float,
        lon: float,
        t_epoch: float,
        window_hours: Optional[float] = None,
    ) -> Optional[Tuple[float, float, float]]:
        if not (math.isfinite(lat) and math.isfinite(lon)):
            return None
        if spec.is_projected and spec.projection is not None:
            xm, ym = spec.projection.latlon_to_xy(lat, lon)
            if not (x_min <= float(xm) <= x_max and y_min <= float(ym) <= y_max):
                return None
        else:
            xm, ym = lon, lat
        eff_win_h = float(max_window_hours) if window_hours is None else float(window_hours)
        t_off = 0.0
        if ref_ts is not None and math.isfinite(t_epoch) and t_epoch > 1e8:
            t_off = float(t_epoch - ref_ts)
            if abs(t_off) > eff_win_h * 3600.0:
                return None
        return float(xm), float(ym), t_off

    # 1. Surface synoptic / AWS / METAR / WIS2 / Marine buoy observations
    surface_sources = (
        ("knmi", "knmi_insitu_obs"),
        ("dwd", "dwd_poi_obs"),
        ("wis2", "wis2_obs"),
        ("metar", "global_metars"),
        ("global_sfc", "global_surface_obs"),
        ("marine", "marine_buoy_obs"),
    )
    for provider, prefix in surface_sources:
        read_win_h = float(max_window_hours) if provider == "marine" else eff_sfc_win_h
        data = _read_obs_prefix(odir, prefix, valid_time_utc=valid_time_utc, max_window_hours=read_win_h)
        if data is None:
            continue
        n = len(data["lat"])
        lats = np.asarray(data["lat"], dtype=np.float64)
        lons = np.asarray(data["lon"], dtype=np.float64)
        elevs = np.asarray(data.get("elev", np.zeros(n)), dtype=np.float64)
        temps = np.asarray(data.get("temp", np.full(n, np.nan)), dtype=np.float64)
        press = np.asarray(data.get("pressure", np.full(n, np.nan)), dtype=np.float64)
        u_ws = np.asarray(data.get("u_wind", np.full(n, np.nan)), dtype=np.float64)
        v_ws = np.asarray(data.get("v_wind", np.full(n, np.nan)), dtype=np.float64)
        ssts = np.asarray(data.get("sst", np.full(n, np.nan)), dtype=np.float64)
        times = np.asarray(data.get("time", np.full(n, np.nan)), dtype=np.float64)

        step_sfc = max(1, n // 1500) if provider in ("metar", "global_sfc") and not spec.is_projected else 1
        best_by_loc: Dict[Tuple[float, float], Tuple[float, int, float, float, float]] = {}
        for idx in range(0, n, step_sfc):
            lat = float(lats[idx])
            lon = float(lons[idx])
            chk = _in_domain_and_window(lat, lon, float(times[idx]), window_hours=read_win_h)
            if chk is None:
                continue
            xm, ym, t_off = chk
            loc_key = (round(lat, 3), round(lon, 3))
            prev = best_by_loc.get(loc_key)
            if prev is None or abs(t_off) < prev[0]:
                best_by_loc[loc_key] = (abs(t_off), idx, xm, ym, t_off)

        for _, idx, xm, ym, t_off in sorted(best_by_loc.values(), key=lambda x: x[1]):
            lat = float(lats[idx])
            lon = float(lons[idx])

            elev = float(elevs[idx]) if math.isfinite(float(elevs[idx])) else 0.0
            elev = float(np.clip(elev, -10.0, 4000.0))
            sid = f"{provider}_{idx:04d}"

            sst_val = float(ssts[idx])
            if math.isfinite(sst_val) and 271.0 <= sst_val <= 310.0:
                obs_list.append(
                    VarObservation(
                        station_id=sid,
                        var_type="sst",
                        lat=lat,
                        lon=lon,
                        alt_m=0.0,
                        value=sst_val,
                        error_variance=0.8**2,
                        time_offset_sec=t_off,
                    )
                )

            # Enforce tight synoptic surface window on rapidly evolving atmospheric surface variables
            if abs(t_off) > eff_sfc_win_h * 3600.0:
                continue

            t_val = float(temps[idx])
            if math.isfinite(t_val) and 220.0 <= t_val <= 330.0:
                obs_list.append(
                    VarObservation(
                        station_id=sid,
                        var_type="t2m",
                        lat=lat,
                        lon=lon,
                        alt_m=elev,
                        value=t_val,
                        error_variance=1.2**2,
                        time_offset_sec=t_off,
                    )
                )
            else:
                t_val = float("nan")

            p_mslp = float(press[idx])
            if math.isfinite(p_mslp) and 92_000.0 <= p_mslp <= 106_500.0:
                t_col = t_val if math.isfinite(t_val) else 285.0
                p_stn = p_mslp * math.exp(-GRAVITY_M_S2 * elev / (R_D * t_col))
                obs_list.append(
                    VarObservation(
                        station_id=sid,
                        var_type="ps",
                        lat=lat,
                        lon=lon,
                        alt_m=elev,
                        value=p_stn,
                        error_variance=120.0**2,
                        time_offset_sec=t_off,
                    )
                )
            else:
                p_mslp = float("nan")

            u_v = float(u_ws[idx])
            v_v = float(v_ws[idx])
            if math.isfinite(u_v) and math.isfinite(v_v) and abs(u_v) <= 60.0 and abs(v_v) <= 60.0:
                obs_list.append(
                    VarObservation(
                        station_id=sid,
                        var_type="u_geo",
                        lat=lat,
                        lon=lon,
                        alt_m=elev,
                        value=u_v,
                        error_variance=1.8**2,
                        time_offset_sec=t_off,
                    )
                )
                obs_list.append(
                    VarObservation(
                        station_id=sid,
                        var_type="v_geo",
                        lat=lat,
                        lon=lon,
                        alt_m=elev,
                        value=v_v,
                        error_variance=1.8**2,
                        time_offset_sec=t_off,
                    )
                )
            else:
                u_v, v_v = float("nan"), float("nan")

            if provider in ("knmi", "dwd", "wis2", "metar", "marine"):
                stn_rows.append(
                    {
                        "provider": provider,
                        "station_id": sid,
                        "lat": lat,
                        "lon": lon,
                        "x_m": float(xm),
                        "y_m": float(ym),
                        "elev_m": elev,
                        "t2m_k": t_val,
                        "mslp_pa": p_mslp,
                        "u10_ms": u_v,
                        "v10_ms": v_v,
                    }
                )

    # 2. Upper-Air Radiosonde & SondeHub vertical soundings (theta_p, u_geo, v_geo, qv at pressure_pa)
    for provider, prefix in (("radiosonde", "radiosonde_obs"), ("sondehub", "sondehub_soundings")):
        data = _read_obs_prefix(odir, prefix, valid_time_utc=valid_time_utc, max_window_hours=max_window_hours)
        if data is None or "pressure" not in data:
            continue
        n = len(data["lat"])
        lats = np.asarray(data["lat"], dtype=np.float64)
        lons = np.asarray(data["lon"], dtype=np.float64)
        press = np.asarray(data["pressure"], dtype=np.float64)
        temps = np.asarray(data.get("temp", np.full(n, np.nan)), dtype=np.float64)
        rhs = np.asarray(data.get("rh", np.full(n, np.nan)), dtype=np.float64)
        u_ws = np.asarray(data.get("u_wind", np.full(n, np.nan)), dtype=np.float64)
        v_ws = np.asarray(data.get("v_wind", np.full(n, np.nan)), dtype=np.float64)
        times = np.asarray(data.get("time", np.full(n, np.nan)), dtype=np.float64)

        step_rad = max(1, n // 3000)
        n2_g = (float(n_ref_hz) * float(n_ref_hz)) / GRAVITY_M_S2
        for idx in range(0, n, step_rad):
            lat, lon, p_pa = float(lats[idx]), float(lons[idx]), float(press[idx])
            if not (math.isfinite(p_pa) and 5_000.0 <= p_pa <= 105_000.0):
                continue
            chk = _in_domain_and_window(lat, lon, float(times[idx]))
            if chk is None:
                continue
            _, _, t_off = chk
            sid = f"{provider}_{idx:05d}"

            # Analytical Brunt-Vaisala reference theta_bar(z) without allocating a dict per level
            z_approx = max(0.0, -7400.0 * math.log(max(p_pa / p_sfc_ref, 1e-4)))
            th_bar_lvl = float(theta_sfc_ref) * math.exp(n2_g * z_approx)

            t_val = float(temps[idx])
            if math.isfinite(t_val) and 180.0 <= t_val <= 330.0:
                pi_obs = (p_pa / P0_PA) ** KAPPA
                theta_p_obs = (t_val / max(pi_obs, 1e-4)) - th_bar_lvl
                obs_list.append(
                    VarObservation(
                        station_id=sid,
                        var_type="theta_p",
                        lat=lat,
                        lon=lon,
                        pressure_pa=p_pa,
                        value=theta_p_obs,
                        error_variance=1.5**2,
                        time_offset_sec=t_off,
                    )
                )
                rh_val = float(rhs[idx])
                if math.isfinite(rh_val) and 0.0 <= rh_val <= 100.0 and p_pa >= 25_000.0:
                    qv_obs = (0.01 * rh_val) * _qsat_water_scalar(t_val, p_pa)
                    obs_list.append(
                        VarObservation(
                            station_id=sid,
                            var_type="qv",
                            lat=lat,
                            lon=lon,
                            pressure_pa=p_pa,
                            value=qv_obs,
                            error_variance=0.0012**2,
                            time_offset_sec=t_off,
                        )
                    )

            u_v, v_v = float(u_ws[idx]), float(v_ws[idx])
            if math.isfinite(u_v) and math.isfinite(v_v) and abs(u_v) <= 110.0 and abs(v_v) <= 110.0:
                obs_list.append(
                    VarObservation(
                        station_id=sid,
                        var_type="u_geo",
                        lat=lat,
                        lon=lon,
                        pressure_pa=p_pa,
                        value=u_v,
                        error_variance=2.2**2,
                        time_offset_sec=t_off,
                    )
                )
                obs_list.append(
                    VarObservation(
                        station_id=sid,
                        var_type="v_geo",
                        lat=lat,
                        lon=lon,
                        pressure_pa=p_pa,
                        value=v_v,
                        error_variance=2.2**2,
                        time_offset_sec=t_off,
                    )
                )

    # 3. European E-GVAP Ground GNSS Zenith Total Delay (ZTD)
    ztd_data = _read_obs_prefix(odir, "egvap_gnss_ztd", valid_time_utc=valid_time_utc, max_window_hours=max_window_hours)
    if ztd_data is not None and "ztd" in ztd_data:
        n = len(ztd_data["lat"])
        lats = np.asarray(ztd_data["lat"], dtype=np.float64)
        lons = np.asarray(ztd_data["lon"], dtype=np.float64)
        elevs = np.asarray(ztd_data.get("elev", np.zeros(n)), dtype=np.float64)
        ztds = np.asarray(ztd_data["ztd"], dtype=np.float64)
        sigmas = np.asarray(ztd_data.get("sigma", np.full(n, 0.012)), dtype=np.float64)
        times = np.asarray(ztd_data.get("time", np.full(n, np.nan)), dtype=np.float64)
        for idx in range(n):
            lat, lon, z_val = float(lats[idx]), float(lons[idx]), float(ztds[idx])
            if not (math.isfinite(z_val) and 1.5 <= z_val <= 3.0):
                continue
            chk = _in_domain_and_window(lat, lon, float(times[idx]))
            if chk is None:
                continue
            _, _, t_off = chk
            sig = float(sigmas[idx]) if math.isfinite(float(sigmas[idx])) else 0.012
            sig = max(0.006, min(0.050, sig))
            obs_list.append(
                VarObservation(
                    station_id=f"egvap_{idx:04d}",
                    var_type="ztd",
                    lat=lat,
                    lon=lon,
                    alt_m=float(np.clip(elevs[idx] if math.isfinite(float(elevs[idx])) else 0.0, -10.0, 4000.0)),
                    value=z_val,
                    error_variance=sig**2,
                    time_offset_sec=t_off,
                )
            )

    # 4. Spaceborne AMV & Scatterometer Winds (EUMETSAT MTG, Metop ASCAT, GOES, Himawari)
    sat_sources = (
        ("eumetsat_amv", "eumetsat"),
        ("eumetsat_amv", "satellite_eumetsat_meteosat_amv_winds"),
        ("ascat", "eumetsat_ascat"),
        ("goes_amv", "satellite_goes_amv_winds"),
        ("himawari_amv", "satellite_jma_himawari9_winds"),
    )
    seen_sat_prefixes: set[str] = set()
    for provider, prefix in sat_sources:
        data = _read_obs_prefix(odir, prefix, valid_time_utc=valid_time_utc, max_window_hours=max_window_hours)
        if data is None:
            continue
        if provider == "eumetsat_amv" and "eumetsat_amv" in seen_sat_prefixes:
            continue
        seen_sat_prefixes.add(provider)
        n = len(data["lat"])
        lats = np.asarray(data["lat"], dtype=np.float64)
        lons = np.asarray(data["lon"], dtype=np.float64)
        press = np.asarray(data.get("pressure", np.full(n, np.nan)), dtype=np.float64)
        u_ws = np.asarray(data.get("u_wind", np.full(n, np.nan)), dtype=np.float64)
        v_ws = np.asarray(data.get("v_wind", np.full(n, np.nan)), dtype=np.float64)
        times = np.asarray(data.get("time", np.full(n, np.nan)), dtype=np.float64)
        step_sat = max(1, n // 2000)
        for idx in range(0, n, step_sat):
            lat, lon = float(lats[idx]), float(lons[idx])
            chk = _in_domain_and_window(lat, lon, float(times[idx]))
            if chk is None:
                continue
            _, _, t_off = chk
            u_v, v_v = float(u_ws[idx]), float(v_ws[idx])
            if not (math.isfinite(u_v) and math.isfinite(v_v) and abs(u_v) <= 110.0 and abs(v_v) <= 110.0):
                continue
            p_pa = float(press[idx])
            p_opt = p_pa if (math.isfinite(p_pa) and 5_000.0 <= p_pa <= 105_000.0) else None
            sid = f"{provider}_{idx:05d}"
            obs_list.append(
                VarObservation(
                    station_id=sid,
                    var_type="u_geo",
                    lat=lat,
                    lon=lon,
                    alt_m=10.0,
                    pressure_pa=p_opt,
                    value=u_v,
                    error_variance=2.5**2,
                    time_offset_sec=t_off,
                )
            )
            obs_list.append(
                VarObservation(
                    station_id=sid,
                    var_type="v_geo",
                    lat=lat,
                    lon=lon,
                    alt_m=10.0,
                    pressure_pa=p_opt,
                    value=v_v,
                    error_variance=2.5**2,
                    time_offset_sec=t_off,
                )
            )

    # 4b. Copernicus / Coriolis GDAC Argo float near-surface (5 m) in-situ temperature -> SST observations
    argo_data = _read_obs_prefix(odir, "argo_ocean_profiles", valid_time_utc=valid_time_utc, max_window_hours=max_window_hours)
    if argo_data is not None and "temp" in argo_data:
        a_lats = np.asarray(argo_data["lat"], dtype=np.float64)
        a_lons = np.asarray(argo_data["lon"], dtype=np.float64)
        a_temps = np.asarray(argo_data["temp"], dtype=np.float64)
        if a_temps.ndim == 2 and a_temps.shape[0] == len(a_lats) and a_temps.shape[1] >= 1:
            for idx in range(len(a_lats)):
                lat, lon = float(a_lats[idx]), float(a_lons[idx])
                chk = _in_domain_and_window(lat, lon, float("nan"))
                if chk is None:
                    continue
                t_top = float(a_temps[idx, 0])
                if math.isfinite(t_top) and 271.0 <= t_top <= 310.0:
                    obs_list.append(
                        VarObservation(
                            station_id=f"argo_{idx:04d}",
                            var_type="sst",
                            lat=lat,
                            lon=lon,
                            alt_m=0.0,
                            value=t_top,
                            error_variance=0.6**2,
                            time_offset_sec=0.0,
                        )
                    )

    # 5. Spatial TITAN Buddy Check QC on surface stations (t2m, ps/mslp, wind u/v)
    if enable_titan_qc and len(stn_rows) >= 4:
        lats_qc = np.array([r["lat"] for r in stn_rows], dtype=np.float64)
        lons_qc = np.array([r["lon"] for r in stn_rows], dtype=np.float64)
        elev_qc = np.array([r["elev_m"] for r in stn_rows], dtype=np.float64)
        sid_by_row = [r["station_id"] for r in stn_rows]
        rejected_pairs: set[Tuple[str, str]] = set()
        total_rejected = 0

        t2m_vals = np.array([r["t2m_k"] for r in stn_rows], dtype=np.float64)
        if np.count_nonzero(np.isfinite(t2m_vals)) >= 4:
            p_t, st_t = run_titan_buddy_check(lats_qc, lons_qc, elev_qc, t2m_vals, var_type="t2m")
            for idx_r, row in enumerate(stn_rows):
                if np.isfinite(t2m_vals[idx_r]) and not p_t[idx_r]:
                    rejected_pairs.add((sid_by_row[idx_r], "t2m"))
                    row["t2m_k"] = float("nan")
            total_rejected += int(st_t.get("rejected", 0))

        mslp_vals = np.array([r["mslp_pa"] for r in stn_rows], dtype=np.float64)
        if np.count_nonzero(np.isfinite(mslp_vals)) >= 4:
            p_p, st_p = run_titan_buddy_check(
                lats_qc, lons_qc, np.zeros_like(elev_qc), mslp_vals, var_type="psfc"
            )
            for idx_r, row in enumerate(stn_rows):
                if np.isfinite(mslp_vals[idx_r]) and not p_p[idx_r]:
                    rejected_pairs.add((sid_by_row[idx_r], "ps"))
                    row["mslp_pa"] = float("nan")
            total_rejected += int(st_p.get("rejected", 0))

        u_vals = np.array([r["u10_ms"] for r in stn_rows], dtype=np.float64)
        v_vals = np.array([r["v10_ms"] for r in stn_rows], dtype=np.float64)
        if np.count_nonzero(np.isfinite(u_vals) & np.isfinite(v_vals)) >= 4:
            p_u, st_u = run_titan_buddy_check(lats_qc, lons_qc, elev_qc, u_vals, var_type="u10m")
            p_v, st_v = run_titan_buddy_check(lats_qc, lons_qc, elev_qc, v_vals, var_type="v10m")
            for idx_r, row in enumerate(stn_rows):
                if np.isfinite(u_vals[idx_r]) and (not p_u[idx_r] or not p_v[idx_r]):
                    rejected_pairs.add((sid_by_row[idx_r], "u_geo"))
                    rejected_pairs.add((sid_by_row[idx_r], "v_geo"))
                    row["u10_ms"] = float("nan")
                    row["v10_ms"] = float("nan")
            total_rejected += max(int(st_u.get("rejected", 0)), int(st_v.get("rejected", 0)))

        if rejected_pairs:
            obs_list = [o for o in obs_list if (o.station_id, o.var_type) not in rejected_pairs]
            logger.info(
                "TITAN spatial QC filtered %d surface station-variable pairs across %d surface stations",
                total_rejected,
                len(stn_rows),
            )

    return obs_list, stn_rows


def apply_land_surface_oi_from_increments(
    surf_state: Dict[str, np.ndarray],
    dt0: np.ndarray,
    dqv0: np.ndarray,
    t0_bg: np.ndarray,
    qv0_bg: np.ndarray,
    p0_bg: np.ndarray,
    lsm: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Mahfouf (1991) / HARMONIE CANARI / ECMWF Cy50r1-style 2D-OI soil temperature & moisture update.

    Propagates 3D-Var 2m/lowest-level temperature (`dt0`) and humidity (`dqv0`) increments consistently
    into `t2m`, `q2m`, `t_skin`, top two soil temperature layers (`soil_t[0..1]`, 0-28 cm), and
    snow-free unfrozen soil moisture layers (`soil_m[0..2]`, 0-100 cm) over land (`lsm >= 0.5`).
    """
    ana_surf = {k: v.copy() for k, v in surf_state.items()}
    dt_c = np.clip(np.asarray(dt0, dtype=np.float64), -6.0, 6.0)
    dq_c = np.clip(np.asarray(dqv0, dtype=np.float64), -5.0e-3, 5.0e-3)
    is_land = np.asarray(lsm, dtype=np.float64) >= 0.5

    if "t2m" in ana_surf:
        t2_cur = np.asarray(ana_surf["t2m"], dtype=np.float64)
        if float(np.mean(np.abs(t2_cur))) > 150.0:
            ana_surf["t2m"] = np.clip(t2_cur + dt_c, 200.0, 340.0).astype(np.float32)
        else:
            ana_surf["t2m"] = np.clip(np.asarray(t0_bg, dtype=np.float64) + dt_c, 200.0, 340.0).astype(np.float32)
    if "q2m" in ana_surf:
        q2_cur = np.asarray(ana_surf["q2m"], dtype=np.float64)
        if float(np.mean(np.abs(q2_cur))) > 1e-6:
            ana_surf["q2m"] = np.clip(q2_cur + dq_c, 1e-7, 0.04).astype(np.float32)
        else:
            ana_surf["q2m"] = np.clip(np.asarray(qv0_bg, dtype=np.float64) + dq_c, 1e-7, 0.04).astype(np.float32)

    swe = np.asarray(ana_surf.get("snow_swe", np.zeros_like(dt_c)), dtype=np.float64)
    has_snow = swe > 0.1

    if "t_skin" in ana_surf:
        tsk_cur = np.asarray(ana_surf["t_skin"], dtype=np.float64)
        tsk_new = np.where(is_land, np.clip(tsk_cur + 0.85 * dt_c, 200.0, 340.0), tsk_cur)
        tsk_new = np.where(is_land & has_snow, np.minimum(tsk_new, 273.15), tsk_new)
        ana_surf["t_skin"] = tsk_new.astype(np.float32)

    if "soil_t" in ana_surf and np.asarray(ana_surf["soil_t"]).ndim == 3 and ana_surf["soil_t"].shape[0] >= 2:
        st_arr = np.asarray(ana_surf["soil_t"], dtype=np.float64).copy()
        for layer_idx, alpha_t in ((0, 0.65), (1, 0.25)):
            st_arr[layer_idx] = np.where(
                is_land,
                np.clip(st_arr[layer_idx] + alpha_t * dt_c, 210.0, 340.0),
                st_arr[layer_idx],
            )
        ana_surf["soil_t"] = st_arr.astype(np.float32)

    if "soil_m" in ana_surf and np.asarray(ana_surf["soil_m"]).ndim == 3 and ana_surf["soil_m"].shape[0] >= 3:
        t_bg = np.clip(np.asarray(t0_bg, dtype=np.float64), 173.15, 340.0)
        p_bg = np.maximum(np.asarray(p0_bg, dtype=np.float64), 1000.0)
        q_bg = np.clip(np.asarray(qv0_bg, dtype=np.float64), 1e-7, 0.05)

        def _qsat_vec(tk: np.ndarray, pk: np.ndarray) -> np.ndarray:
            tc = np.clip(tk - 273.15, -90.0, 60.0)
            es = np.minimum(611.21 * np.exp((17.502 * tc) / (tk - 32.19)), 0.90 * pk)
            return 0.62198 * es / np.maximum(pk - 0.37802 * es, 100.0)

        rh_bg = np.clip(q_bg / np.maximum(_qsat_vec(t_bg, p_bg), 1e-6), 0.0, 1.05)
        rh_ana = np.clip((q_bg + dq_c) / np.maximum(_qsat_vec(t_bg + dt_c, p_bg), 1e-6), 0.0, 1.05)
        drh = np.clip(rh_ana - rh_bg, -0.30, 0.30)

        tsk_ref = np.asarray(ana_surf.get("t_skin", t_bg), dtype=np.float64)
        active_sm = is_land & (~has_snow) & (tsk_ref > 274.15)
        sm_arr = np.asarray(ana_surf["soil_m"], dtype=np.float64).copy()
        # Mahfouf (1991) OI soil moisture gain coefficients (K_T < 0, K_RH > 0) for 0-7, 7-28, 28-100 cm
        for layer_idx, k_t, k_rh in ((0, -0.008, 0.08), (1, -0.005, 0.05), (2, -0.002, 0.02)):
            dw = np.clip(k_t * dt_c + k_rh * drh, -0.03, 0.03)
            sm_arr[layer_idx] = np.where(
                active_sm,
                np.clip(sm_arr[layer_idx] + dw, 0.05, 0.45),
                sm_arr[layer_idx],
            )
        ana_surf["soil_m"] = sm_arr.astype(np.float32)

    return ana_surf


def run_domain_3dvar(
    spec: DomainSpec,
    state_3d: Dict[str, np.ndarray],
    surf_state: Dict[str, np.ndarray],
    hs_m: np.ndarray,
    lsm: np.ndarray,
    obs_list: List[VarObservation],
    *,
    theta_sfc_ref: float = 288.15,
    p_sfc_ref: float = 101_325.0,
    n_ref_hz: float = 0.01,
    length_scale_h_m: float = 45_000.0,
    max_iterations: int = 25,
    iau_tau_sec: float = 900.0,
    trajectory: Optional[Dict[float, Dict[str, np.ndarray]]] = None,
    obs_dir: Optional[Path | str] = None,
    radar_rain_rate_mm_h: Optional[np.ndarray] = None,
    model_rain_rate_mm_h: Optional[np.ndarray] = None,
    dtheta_dt_mp: Optional[np.ndarray] = None,
    compute_iau: bool = True,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Optional[Dict[str, np.ndarray]], Var3DResult]:
    """Run balanced 3D-Var / 4D-FGAT + Radar LHN + OSI-SAF cryosphere update and return analysis + IAU forcing."""
    t0_var = time.perf_counter()
    hs = np.asarray(hs_m, dtype=np.float64)

    z_0 = float(spec.vertical.zeta_s[0]) + float(spec.vertical.B_s[0]) * hs
    ref0 = VerticalGrid.reference_state(
        z_0[None, :, :], theta_sfc_k=theta_sfc_ref, p_sfc_pa=p_sfc_ref, brunt_vaisala_hz=n_ref_hz
    )
    th0 = ref0["theta_bar"][0] + np.asarray(state_3d["theta_p"][0], dtype=np.float64)
    pi0 = np.maximum(1e-4, ref0["pi_bar"][0] + np.asarray(state_3d["pi_p"][0], dtype=np.float64))
    p0 = P0_PA * np.power(pi0, 1.0 / KAPPA)
    tv0 = np.maximum(160.0, th0 * pi0 * (1.0 + 0.608 * np.maximum(0.0, np.asarray(state_3d["qv"][0], dtype=np.float64))))
    dz_sfc = np.maximum(0.0, z_0 - hs)
    ps_bg = p0 * np.exp((GRAVITY_M_S2 * dz_sfc) / (R_D * tv0))

    dz_avg = float(spec.vertical.z_top_m) / max(1, spec.nz)

    def _build_fgat_trajectory(
        dom_spec: DomainSpec, bg_dict: Dict[str, np.ndarray], obs_seq: List[VarObservation]
    ) -> Optional[Dict[float, Dict[str, np.ndarray]]]:
        if not any(abs(float(o.time_offset_sec)) > 60.0 for o in obs_seq) or dom_spec.nx < 3 or dom_spec.ny < 3:
            return None
        t_offs = [float(o.time_offset_sec) for o in obs_seq]
        t_lo = min(-900.0, min(t_offs))
        t_hi = max(900.0, max(t_offs))
        dx_eff = np.maximum(float(dom_spec.dx_m) / np.maximum(dom_spec.map_factor, 0.05), 100.0)
        dy_eff = max(float(dom_spec.dy_m), 100.0)
        u_b = np.asarray(bg_dict["u"], dtype=np.float64)
        v_b = np.asarray(bg_dict["v"], dtype=np.float64)

        def _advect_state(dt_sec: float) -> Dict[str, np.ndarray]:
            if abs(dt_sec) < 1.0:
                return bg_dict
            out_t: Dict[str, np.ndarray] = {}
            for k_f, arr_f in bg_dict.items():
                a64 = np.asarray(arr_f, dtype=np.float64)
                if a64.ndim == 3:
                    dadx = np.gradient(a64, axis=2) / dx_eff[None, :, :]
                    dady = np.gradient(a64, dy_eff, axis=1)
                    cap = 0.25 * float(np.std(a64)) + 1e-4
                    adv = np.clip(-dt_sec * (u_b * dadx + v_b * dady), -cap, cap)
                    out_t[k_f] = (a64 + adv).astype(np.float32)
                elif a64.ndim == 2 and k_f == "ps":
                    dadx = np.gradient(a64, axis=1) / dx_eff
                    dady = np.gradient(a64, dy_eff, axis=0)
                    adv = np.clip(-dt_sec * (u_b[0] * dadx + v_b[0] * dady), -400.0, 400.0)
                    out_t[k_f] = (a64 + adv).astype(np.float32)
                else:
                    out_t[k_f] = arr_f
            return out_t

        return {t_lo: _advect_state(t_lo), 0.0: bg_dict, t_hi: _advect_state(t_hi)}

    # Courtier et al. (1994) Multi-Incremental 3D-Var / 4D-FGAT inner loop when horizontal grid >= 25k columns
    if spec.nx * spec.ny >= 25_000 and trajectory is None:
        from dataclasses import replace

        stride = max(2, int(round(math.sqrt((spec.nx * spec.ny) / 18_000.0))))
        y_idx = np.arange(0, spec.ny, stride, dtype=np.int64)
        x_idx = np.arange(0, spec.nx, stride, dtype=np.int64)
        c_spec = replace(
            spec,
            ny=len(y_idx),
            nx=len(x_idx),
            dx_m=spec.dx_m * stride,
            dy_m=spec.dy_m * stride,
            x_m=spec.x_m[x_idx].copy(),
            y_m=spec.y_m[y_idx].copy(),
            lat2d=spec.lat2d[np.ix_(y_idx, x_idx)].copy(),
            lon2d=spec.lon2d[np.ix_(y_idx, x_idx)].copy(),
            map_factor=spec.map_factor[np.ix_(y_idx, x_idx)].copy(),
            coriolis_f=spec.coriolis_f[np.ix_(y_idx, x_idx)].copy(),
        )
        c_hs = hs[np.ix_(y_idx, x_idx)].copy()
        c_lsm = np.asarray(lsm)[np.ix_(y_idx, x_idx)] >= 0.5
        u_st_in = np.asarray(state_3d["u"], dtype=np.float32)
        v_st_in = np.asarray(state_3d["v"], dtype=np.float32)
        u_c_sub = (
            0.5 * (u_st_in[:, y_idx, :][:, :, x_idx] + u_st_in[:, y_idx, :][:, :, np.minimum(x_idx + 1, u_st_in.shape[2] - 1)])
            if u_st_in.shape[2] == spec.nx + 1
            else u_st_in[:, y_idx, :][:, :, x_idx].copy()
        )
        v_c_sub = (
            0.5 * (v_st_in[:, y_idx, :][:, :, x_idx] + v_st_in[:, np.minimum(y_idx + 1, v_st_in.shape[1] - 1), :][:, :, x_idx])
            if v_st_in.shape[1] == spec.ny + 1
            else v_st_in[:, y_idx, :][:, :, x_idx].copy()
        )
        c_bg: Dict[str, np.ndarray] = {
            "u": u_c_sub,
            "v": v_c_sub,
            "theta_p": np.asarray(state_3d["theta_p"], dtype=np.float32)[:, y_idx, :][:, :, x_idx].copy(),
            "pi_p": np.asarray(state_3d["pi_p"], dtype=np.float32)[:, y_idx, :][:, :, x_idx].copy(),
            "qv": np.asarray(state_3d["qv"], dtype=np.float32)[:, y_idx, :][:, :, x_idx].copy(),
            "ps": ps_bg[np.ix_(y_idx, x_idx)].astype(np.float32),
        }
        if "sst_bulk" in surf_state:
            c_bg["sst"] = np.asarray(surf_state["sst_bulk"], dtype=np.float32)[np.ix_(y_idx, x_idx)].copy()

        c_traj = _build_fgat_trajectory(c_spec, c_bg, obs_list)
        c_hop = VarObservationOperator(
            c_spec, c_hs, c_bg, theta_sfc_ref=theta_sfc_ref, p_sfc_ref=p_sfc_ref, n_ref_hz=n_ref_hz
        )
        c_filter = RecursiveSpatialFilter(
            c_spec.dx_m, c_spec.dy_m, dz_m=dz_avg, length_scale_h_m=length_scale_h_m, length_scale_v_m=2_000.0, num_passes=2
        )
        c_bal = BalanceOperator(
            c_spec, c_hs, theta_sfc_ref=theta_sfc_ref, p_sfc_ref=p_sfc_ref, n_ref_hz=n_ref_hz
        )
        cfg = Var3DConfig(max_iterations=max_iterations, use_balance=True, iau_window_sec=iau_tau_sec)
        logger.info(
            "[%s] Running Multi-Incremental 3D-Var/4D-FGAT | grid=%dx%dx%d -> coarse %dx%dx%d (stride=%d) | obs=%d | fgat=%s",
            spec.name,
            spec.nz,
            spec.ny,
            spec.nx,
            c_spec.nz,
            c_spec.ny,
            c_spec.nx,
            stride,
            len(obs_list),
            c_traj is not None,
        )
        c_res = solve_3dvar_pcg(c_bg, obs_list, c_hop, c_filter, cfg, balance_op=c_bal, land_mask=c_lsm, trajectory=c_traj)

        # Memory-frugal separable 1D float32 prolongation one variable at a time (no simultaneous 5-var fine_inc/updated_c copies)
        fy = np.clip(np.arange(spec.ny, dtype=np.float32) / float(stride), 0.0, float(c_spec.ny - 1.00001))
        fx = np.clip(np.arange(spec.nx, dtype=np.float32) / float(stride), 0.0, float(c_spec.nx - 1.00001))
        j0_p = np.floor(fy).astype(np.int64)
        j1_p = np.minimum(j0_p + 1, c_spec.ny - 1)
        wy_p = (fy - j0_p).astype(np.float32)
        i0_p = np.floor(fx).astype(np.int64)
        i1_p = np.minimum(i0_p + 1, c_spec.nx - 1)
        wx_p = (fx - i0_p).astype(np.float32)

        def _prolong_field(arr_c: np.ndarray) -> np.ndarray:
            af = np.asarray(arr_c, dtype=np.float32)
            if af.ndim == 2:
                r = (1.0 - wy_p)[:, None] * af[j0_p, :] + wy_p[:, None] * af[j1_p, :]
                return ((1.0 - wx_p)[None, :] * r[:, i0_p] + wx_p[None, :] * r[:, i1_p]).astype(np.float32)
            r = (1.0 - wy_p)[None, :, None] * af[:, j0_p, :] + wy_p[None, :, None] * af[:, j1_p, :]
            return ((1.0 - wx_p)[None, None, :] * r[:, :, i0_p] + wx_p[None, None, :] * r[:, :, i1_p]).astype(np.float32)

        ana_state = dict(state_3d)
        inv_tau = np.float32(1.0 / max(1.0, float(iau_tau_sec)))
        iau_forcing: Optional[Dict[str, np.ndarray]] = {} if compute_iau else None

        # u & v staggered update
        for wkey, axis_s in (("u", 2), ("v", 1)):
            inc_c_w = c_res.analysis_increments.get(wkey)
            if inc_c_w is not None and np.any(inc_c_w):
                inc_f = _prolong_field(inc_c_w)
                if axis_s == 2 and ana_state["u"].shape[2] == spec.nx + 1:
                    inc_st = np.empty_like(ana_state["u"], dtype=np.float32)
                    inc_st[:, :, 1:-1] = 0.5 * (inc_f[:, :, :-1] + inc_f[:, :, 1:])
                    inc_st[:, :, 0] = 0.5 * (inc_f[:, :, -1] + inc_f[:, :, 0]) if spec.is_periodic_lon else inc_f[:, :, 0]
                    inc_st[:, :, -1] = inc_st[:, :, 0] if spec.is_periodic_lon else inc_f[:, :, -1]
                elif axis_s == 1 and ana_state["v"].shape[1] == spec.ny + 1:
                    inc_st = np.empty_like(ana_state["v"], dtype=np.float32)
                    inc_st[:, 1:-1, :] = 0.5 * (inc_f[:, :-1, :] + inc_f[:, 1:, :])
                    inc_st[:, 0, :] = inc_f[:, 0, :]
                    inc_st[:, -1, :] = inc_f[:, -1, :]
                else:
                    inc_st = inc_f
                del inc_f
                np.clip(inc_st, -25.0, 25.0, out=inc_st)
                if not compute_iau:
                    ana_state[wkey] += inc_st
                    np.clip(ana_state[wkey], -150.0, 150.0, out=ana_state[wkey])
                    del inc_st
                else:
                    ana_state[wkey] = np.clip(ana_state[wkey] + inc_st, -150.0, 150.0).astype(np.float32)
                    if iau_forcing is not None:
                        inc_st *= inv_tau
                        iau_forcing[wkey] = inc_st
            elif iau_forcing is not None:
                iau_forcing[wkey] = np.zeros_like(ana_state[wkey], dtype=np.float32)

        for skey in ("theta_p", "pi_p", "qv"):
            inc_c_s = c_res.analysis_increments.get(skey)
            if inc_c_s is not None and np.any(inc_c_s):
                inc_f = _prolong_field(inc_c_s)
                if skey == "qv":
                    if not compute_iau:
                        ana_state["qv"] += inc_f
                        np.clip(ana_state["qv"], 1e-8, 0.04, out=ana_state["qv"])
                        del inc_f
                    else:
                        ana_state["qv"] = np.clip(ana_state["qv"] + inc_f, 1e-8, 0.04).astype(np.float32)
                else:
                    if skey == "theta_p":
                        np.clip(inc_f, -12.0, 12.0, out=inc_f)
                    elif skey == "pi_p":
                        np.clip(inc_f, -0.004, 0.004, out=inc_f)
                    if not compute_iau:
                        ana_state[skey] += inc_f
                        del inc_f
                    else:
                        ana_state[skey] = (ana_state[skey] + inc_f).astype(np.float32)
                        if iau_forcing is not None:
                            inc_f *= inv_tau
                            iau_forcing[skey] = inc_f
            elif iau_forcing is not None and skey in ("theta_p", "pi_p"):
                iau_forcing[skey] = np.zeros_like(ana_state[skey], dtype=np.float32)

        if iau_forcing is not None:
            iau_forcing["w"] = np.zeros_like(ana_state["w"], dtype=np.float32)

        sst_updated: Optional[np.ndarray] = None
        if "sst" in c_res.analysis_increments and "sst_bulk" in surf_state and np.any(c_res.analysis_increments["sst"]):
            sst_updated = (np.asarray(surf_state["sst_bulk"], dtype=np.float32) + _prolong_field(c_res.analysis_increments["sst"])).astype(np.float32)

        res = c_res
    else:
        u_c, v_c = unstagger_uv(state_3d["u"], state_3d["v"])
        bg_centered: Dict[str, np.ndarray] = {
            "u": u_c,
            "v": v_c,
            "theta_p": np.asarray(state_3d["theta_p"], dtype=np.float32),
            "pi_p": np.asarray(state_3d["pi_p"], dtype=np.float32),
            "qv": np.asarray(state_3d["qv"], dtype=np.float32),
            "ps": ps_bg.astype(np.float32),
        }
        if "sst_bulk" in surf_state:
            bg_centered["sst"] = np.asarray(surf_state["sst_bulk"], dtype=np.float32)

        eff_traj = trajectory if trajectory is not None else _build_fgat_trajectory(spec, bg_centered, obs_list)
        hop = VarObservationOperator(
            spec, hs, bg_centered, theta_sfc_ref=theta_sfc_ref, p_sfc_ref=p_sfc_ref, n_ref_hz=n_ref_hz
        )
        spatial_filter = RecursiveSpatialFilter(
            spec.dx_m, spec.dy_m, dz_m=dz_avg, length_scale_h_m=length_scale_h_m, length_scale_v_m=2_000.0, num_passes=2
        )
        balance_op = BalanceOperator(
            spec, hs, theta_sfc_ref=theta_sfc_ref, p_sfc_ref=p_sfc_ref, n_ref_hz=n_ref_hz
        )
        cfg = Var3DConfig(max_iterations=max_iterations, use_balance=True, iau_window_sec=iau_tau_sec)
        logger.info(
            "[%s] Running 3D-Var/4D-FGAT | grid=%dx%dx%d | obs=%d | fgat=%s",
            spec.name,
            spec.nz,
            spec.ny,
            spec.nx,
            len(obs_list),
            eff_traj is not None,
        )
        res = solve_3dvar_pcg(
            bg_centered,
            obs_list,
            hop,
            spatial_filter,
            cfg,
            balance_op=balance_op,
            land_mask=(np.asarray(lsm) >= 0.5),
            trajectory=eff_traj,
        )
        inc_bounds = {
            "u": (-25.0, 25.0),
            "v": (-25.0, 25.0),
            "theta_p": (-12.0, 12.0),
            "pi_p": (-0.004, 0.004),
        }
        bounded_inc = {
            k: np.clip(v, *inc_bounds[k]) if k in inc_bounds else v
            for k, v in res.analysis_increments.items()
        }
        updated_c = apply_analysis_increments_with_bounds(bg_centered, bounded_inc)
        u_st, v_st = stagger_uv(updated_c["u"], updated_c["v"])
        ana_state = {k: v.copy() for k, v in state_3d.items()}
        ana_state["u"] = u_st
        ana_state["v"] = v_st
        ana_state["theta_p"] = updated_c["theta_p"]
        ana_state["pi_p"] = updated_c["pi_p"]
        ana_state["qv"] = updated_c["qv"]
        sst_updated = updated_c.get("sst")
        if compute_iau:
            inc_c = {
                "u": bounded_inc.get("u", np.zeros_like(u_c)),
                "v": bounded_inc.get("v", np.zeros_like(v_c)),
                "theta_p": bounded_inc.get("theta_p", np.zeros_like(ana_state["theta_p"])),
                "pi_p": bounded_inc.get("pi_p", np.zeros_like(ana_state["pi_p"])),
            }
            iau_raw = compute_iau_tendencies(inc_c, tau_sec=iau_tau_sec)
            iau_u_st, iau_v_st = stagger_uv(iau_raw["u"], iau_raw["v"])
            iau_forcing = {
                "u": iau_u_st,
                "v": iau_v_st,
                "w": np.zeros_like(ana_state["w"]),
                "theta_p": iau_raw["theta_p"],
                "pi_p": iau_raw["pi_p"],
            }
        else:
            iau_forcing = None

    # Adjust 2m temperature/moisture, skin temperature, soil_t/soil_m (Mahfouf 1991 / Cy50r1 OI), and SST
    th0_ana = ref0["theta_bar"][0] + np.asarray(ana_state["theta_p"][0], dtype=np.float64)
    pi0_ana = np.maximum(1e-4, ref0["pi_bar"][0] + np.asarray(ana_state["pi_p"][0], dtype=np.float64))
    t0_bg_2d = th0 * pi0
    t0_ana = th0_ana * pi0_ana
    dt0 = (t0_ana - t0_bg_2d).astype(np.float64)
    qv0_bg_2d = np.asarray(state_3d.get("qv", np.zeros_like(t0_bg_2d)), dtype=np.float64)
    if qv0_bg_2d.ndim == 3:
        qv0_bg_2d = qv0_bg_2d[0]
    qv0_ana_2d = np.asarray(ana_state.get("qv", np.zeros_like(t0_bg_2d)), dtype=np.float64)
    if qv0_ana_2d.ndim == 3:
        qv0_ana_2d = qv0_ana_2d[0]
    dqv0 = qv0_ana_2d - qv0_bg_2d
    p0_bg_2d = P0_PA * np.power(np.maximum(1e-4, pi0), 1.0 / KAPPA)
    ana_surf = apply_land_surface_oi_from_increments(
        surf_state, dt0, dqv0, t0_bg_2d, qv0_bg_2d, p0_bg_2d, lsm
    )
    if sst_updated is not None and "sst_bulk" in ana_surf:
        ana_surf["sst_bulk"] = sst_updated.astype(np.float32)

    # Apply OSI-SAF cryosphere sea-ice concentration / thickness on both projected (D2/D3) and global spherical (D1) grids
    if obs_dir is not None:
        osi = _read_obs_prefix(Path(obs_dir), "osi_saf_sea_ice")
        if osi is not None and "ice_conc" in osi:
            ic_vals = np.asarray(osi["ice_conc"], dtype=np.float64)
            it_vals = np.asarray(
                osi.get("ice_thick", np.clip(0.4 + 1.8 * (np.maximum(ic_vals, 0.0) ** 1.5), 0.1, 4.5)),
                dtype=np.float64,
            )
            ic_lats = np.asarray(osi["lat"], dtype=np.float64)
            ic_lons = np.asarray(osi["lon"], dtype=np.float64)
            ice_mask = np.isfinite(ic_vals) & (ic_vals >= 0.15) & np.isfinite(ic_lats) & np.isfinite(ic_lons)
            if np.any(ice_mask):
                if spec.is_projected and spec.projection is not None:
                    xm, ym = spec.projection.latlon_to_xy(ic_lats[ice_mask], ic_lons[ice_mask])
                    ii = np.round((xm - float(spec.x_m[0])) / spec.dx_m).astype(np.int64)
                    jj = np.round((ym - float(spec.y_m[0])) / spec.dy_m).astype(np.int64)
                else:
                    lon_m = ic_lons[ice_mask] % 360.0 if float(spec.x_m[-1]) > 180.0 else (((ic_lons[ice_mask] + 180.0) % 360.0) - 180.0)
                    dx_deg = float(spec.x_m[1] - spec.x_m[0]) if len(spec.x_m) > 1 else 1.0
                    dy_deg = float(spec.y_m[1] - spec.y_m[0]) if len(spec.y_m) > 1 else 1.0
                    ii = np.round((lon_m - float(spec.x_m[0])) / dx_deg).astype(np.int64)
                    jj = np.round((ic_lats[ice_mask] - float(spec.y_m[0])) / dy_deg).astype(np.int64)
                valid_ij = (ii >= 0) & (ii < spec.nx) & (jj >= 0) & (jj < spec.ny)
                ic_m = ic_vals[ice_mask][valid_ij]
                it_m = it_vals[ice_mask][valid_ij]
                for j_pt, i_pt, c_pt, h_pt in zip(jj[valid_ij], ii[valid_ij], ic_m, it_m):
                    if float(lsm[j_pt, i_pt]) < 0.5:
                        if "sst_bulk" in ana_surf:
                            ana_surf["sst_bulk"][j_pt, i_pt] = np.float32(271.35)
                        if "sea_ice_frac" in ana_surf:
                            ana_surf["sea_ice_frac"][j_pt, i_pt] = np.float32(min(0.997, max(0.0, c_pt)))
                        if "sea_ice_thick" in ana_surf:
                            ana_surf["sea_ice_thick"][j_pt, i_pt] = np.float32(max(0.15, h_pt))
                        if "sea_ice_tsfc" in ana_surf:
                            ana_surf["sea_ice_tsfc"][j_pt, i_pt] = np.float32(min(float(ana_surf["sea_ice_tsfc"][j_pt, i_pt]), 271.15))
                        if "sea_ice_albedo" in ana_surf:
                            ana_surf["sea_ice_albedo"][j_pt, i_pt] = np.float32(0.72)

    # Radar Latent Heat Nudging (LHN) via compute_lhn_tendencies
    if radar_rain_rate_mm_h is None and obs_dir is not None:
        for cand_d in _expand_obs_search_dirs(Path(obs_dir)):
            rad_p = cand_d / "radar_obs.wx1"
            if rad_p.is_file():
                with wx1.open_dataset(rad_p) as rds:
                    vname = "rain_rate_mm_h" if "rain_rate_mm_h" in rds else next(iter(rds.data_vars), None)
                    if vname is not None:
                        arr_r = np.asarray(rds[vname].values, dtype=np.float64)
                        if arr_r.shape == (spec.ny, spec.nx):
                            radar_rain_rate_mm_h = arr_r
                            break

    if radar_rain_rate_mm_h is not None:
        _, z_s, _ = spec.vertical.heights_3d(hs)
        mod_rr_mm_h = (
            np.asarray(model_rain_rate_mm_h, dtype=np.float64)
            if model_rain_rate_mm_h is not None
            else np.zeros((spec.ny, spec.nx), dtype=np.float64)
        )
        ref_full = VerticalGrid.reference_state(
            z_s, theta_sfc_k=theta_sfc_ref, p_sfc_pa=p_sfc_ref, brunt_vaisala_hz=n_ref_hz
        )
        dth_lhn, dqv_lhn = compute_lhn_tendencies(
            np.asarray(radar_rain_rate_mm_h, dtype=np.float64) / 3600.0,
            mod_rr_mm_h / 3600.0,
            z_s,
            hs,
            ref_full["rho_bar"],
            ana_state["theta_p"],
            ref_full["pi_bar"],
            ana_state["qv"],
            qc=ana_state.get("qc"),
            tau_lhn_sec=iau_tau_sec,
        )
        if iau_forcing is not None:
            iau_forcing["theta_p"] = (iau_forcing["theta_p"] + dth_lhn).astype(np.float32)
        ana_state["qv"] = np.clip(ana_state["qv"] + dqv_lhn * float(iau_tau_sec), 0.0, 0.045).astype(np.float32)

    logger.info(
        "[%s] 3D-Var/4D-FGAT done in %.2fs | J0=%.2f -> Jf=%.2f | iters=%d | converged=%s | iau=%s",
        spec.name,
        time.perf_counter() - t0_var,
        res.initial_cost_j0,
        res.final_cost_j,
        res.iterations,
        res.converged,
        iau_forcing is not None,
    )
    return ana_state, ana_surf, iau_forcing, res


def _bilinear_sample_2d(field: np.ndarray, spec: DomainSpec, x_pts: np.ndarray, y_pts: np.ndarray) -> np.ndarray:
    fx = (x_pts - float(spec.x_m[0])) / spec.dx_m
    fy = (y_pts - float(spec.y_m[0])) / spec.dy_m
    fx = np.clip(fx, 0.0, spec.nx - 1.000001)
    fy = np.clip(fy, 0.0, spec.ny - 1.000001)
    i0 = np.floor(fx).astype(np.int64)
    j0 = np.floor(fy).astype(np.int64)
    i1 = np.minimum(i0 + 1, spec.nx - 1)
    j1 = np.minimum(j0 + 1, spec.ny - 1)
    wx = fx - i0
    wy = fy - j0
    return (
        (1.0 - wy) * (1.0 - wx) * field[j0, i0]
        + (1.0 - wy) * wx * field[j0, i1]
        + wy * (1.0 - wx) * field[j1, i0]
        + wy * wx * field[j1, i1]
    )


def _metric_summary(obs: np.ndarray, mod: np.ndarray) -> Tuple[int, float, float, float]:
    valid = np.isfinite(obs) & np.isfinite(mod)
    if not np.any(valid):
        return 0, float("nan"), float("nan"), float("nan")
    diff = mod[valid] - obs[valid]
    return (
        int(valid.sum()),
        float(np.mean(diff)),
        float(np.sqrt(np.mean(diff * diff))),
        float(np.mean(np.abs(diff))),
    )


def verify_against_stations(
    spec: DomainSpec,
    state_3d: Dict[str, np.ndarray],
    surf_state: Dict[str, np.ndarray],
    hs_m: np.ndarray,
    station_rows: Sequence[Dict[str, Any]],
    *,
    theta_sfc_ref: float = 288.15,
    p_sfc_ref: float = 101_325.0,
    n_ref_hz: float = 0.01,
) -> Dict[str, Dict[str, float]]:
    """Evaluate bias, RMSE, and MAE against KNMI & DWD stations inside `spec`."""
    if not station_rows:
        empty = {
            "n_t2m": 0, "t2m_bias_k": float("nan"), "t2m_rmse_k": float("nan"), "t2m_mae_k": float("nan"),
            "n_mslp": 0, "mslp_bias_hpa": float("nan"), "mslp_rmse_hpa": float("nan"),
            "n_wind": 0, "wind_bias_ms": float("nan"), "wind_rmse_ms": float("nan"),
        }
        return {"all": dict(empty), "knmi": dict(empty), "dwd": dict(empty)}

    hs = np.asarray(hs_m, dtype=np.float64)
    z0 = float(spec.vertical.zeta_s[0]) + float(spec.vertical.B_s[0]) * hs
    ref0 = VerticalGrid.reference_state(
        z0[None, :, :], theta_sfc_k=theta_sfc_ref, p_sfc_pa=p_sfc_ref, brunt_vaisala_hz=n_ref_hz
    )
    th0 = ref0["theta_bar"][0] + np.asarray(state_3d["theta_p"][0], dtype=np.float64)
    pi0 = np.maximum(1e-4, ref0["pi_bar"][0] + np.asarray(state_3d["pi_p"][0], dtype=np.float64))
    t2m_grid = th0 * pi0
    mslp_grid = compute_mslp_pa(spec, hs, state_3d, theta_sfc_ref=theta_sfc_ref, p_sfc_ref=p_sfc_ref, n_ref_hz=n_ref_hz).astype(np.float64)

    uc, vc = unstagger_uv(state_3d["u"][:1], state_3d["v"][:1])
    u_grid, v_grid = uc[0].astype(np.float64), vc[0].astype(np.float64)
    if spec.is_projected and spec.projection is not None:
        u_geo_grid, v_geo_grid = spec.projection.grid_to_geo_wind(u_grid, v_grid, spec.lon2d)
    else:
        u_geo_grid, v_geo_grid = u_grid, v_grid

    x_pts = np.array([r["x_m"] for r in station_rows], dtype=np.float64)
    y_pts = np.array([r["y_m"] for r in station_rows], dtype=np.float64)
    elevs = np.array([r["elev_m"] for r in station_rows], dtype=np.float64)
    provs = np.array([r["provider"] for r in station_rows])

    hs_stn = _bilinear_sample_2d(hs, spec, x_pts, y_pts)
    mod_t2m = _bilinear_sample_2d(t2m_grid, spec, x_pts, y_pts) - 0.0065 * (elevs - hs_stn)
    mod_mslp_hpa = _bilinear_sample_2d(mslp_grid, spec, x_pts, y_pts) / 100.0
    mod_u = _bilinear_sample_2d(u_geo_grid, spec, x_pts, y_pts)
    mod_v = _bilinear_sample_2d(v_geo_grid, spec, x_pts, y_pts)
    mod_wspd = np.hypot(mod_u, mod_v)

    obs_t2m = np.array([r["t2m_k"] for r in station_rows], dtype=np.float64)
    obs_mslp_hpa = np.array([r["mslp_pa"] for r in station_rows], dtype=np.float64) / 100.0
    obs_wspd = np.hypot(
        np.array([r["u10_ms"] for r in station_rows], dtype=np.float64),
        np.array([r["v10_ms"] for r in station_rows], dtype=np.float64),
    )

    out: Dict[str, Dict[str, float]] = {}
    for key in ("all", "knmi", "dwd"):
        m = np.ones(len(station_rows), dtype=bool) if key == "all" else (provs == key)
        nt, tb, tr, tmae = _metric_summary(obs_t2m[m], mod_t2m[m])
        np_n, pb, pr, _ = _metric_summary(obs_mslp_hpa[m], mod_mslp_hpa[m])
        nw, wb, wr, _ = _metric_summary(obs_wspd[m], mod_wspd[m])
        out[key] = {
            "n_t2m": nt,
            "t2m_bias_k": tb,
            "t2m_rmse_k": tr,
            "t2m_mae_k": tmae,
            "n_mslp": np_n,
            "mslp_bias_hpa": pb,
            "mslp_rmse_hpa": pr,
            "n_wind": nw,
            "wind_bias_ms": wb,
            "wind_rmse_ms": wr,
        }
    return out


def compute_fair_crps(ensemble_values: np.ndarray, obs_value: np.ndarray | float) -> float:
    """Compute Ferro (2014) fair Continuous Ranked Probability Score across ensemble samples."""
    ens = np.asarray(ensemble_values, dtype=np.float64)
    if ens.ndim == 1:
        ens = ens[:, None]
    if ens.ndim != 2:
        raise ValueError(f"Expected 1D or 2D ensemble array, got shape {ens.shape}")
    obs = np.broadcast_to(np.asarray(obs_value, dtype=np.float64), (ens.shape[1],))

    scores = []
    for n in range(ens.shape[1]):
        y = obs[n]
        if not np.isfinite(y):
            continue
        x = ens[:, n]
        x = x[np.isfinite(x)]
        m = x.size
        if m == 0:
            continue
        term1 = float(np.mean(np.abs(x - y)))
        if m == 1:
            scores.append(term1)
            continue
        x_sorted = np.sort(x)
        weights = (2.0 * np.arange(m, dtype=np.float64) - float(m) + 1.0)
        term2 = float(np.dot(weights, x_sorted)) / float(m * (m - 1))
        scores.append(term1 - term2)
    return float(np.mean(scores)) if scores else float("nan")


def compute_brier_score(prob: np.ndarray, obs_event: np.ndarray) -> float:
    """Compute Brier score E[(p - o)^2] for probabilistic event forecasts vs binary obs."""
    p = np.asarray(prob, dtype=np.float64)
    o = np.asarray(obs_event, dtype=np.float64)
    valid = np.isfinite(p) & np.isfinite(o)
    if not np.any(valid):
        return float("nan")
    return float(np.mean((p[valid] - o[valid]) ** 2))


def verify_ensemble_against_stations(
    ens_dir: Path | str,
    lead_hour: float,
    spec: DomainSpec,
    stations: Sequence[Any],
    *,
    hs_m: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Evaluate fair CRPS, ensemble spread-skill ratio, CF RMSE, and Brier frost score."""
    ens_root = Path(ens_dir)
    if not ens_root.exists():
        raise FileNotFoundError(f"Ensemble directory not found: {ens_root}")

    tag = f"{int(round(lead_hour)):03d}" if abs(lead_hour - round(lead_hour)) < 1e-4 else f"{lead_hour:06.2f}"
    expected_name = f"forecast_{spec.name}_T+{tag}h.wx1"

    member_dirs: List[Path] = []
    cf_dir = ens_root / "cf"
    if cf_dir.is_dir():
        member_dirs.append(cf_dir)
    pf_dirs = sorted(
        d for d in ens_root.iterdir()
        if d.is_dir() and d.name.startswith("pf") and d.name[2:].isdigit()
    )
    member_dirs.extend(pf_dirs)

    member_files: List[Path] = []
    for d in member_dirs:
        cand = d / expected_name
        if cand.exists():
            member_files.append(cand)
        else:
            matches = sorted(d.glob(f"*T+{tag}h.wx1"))
            if matches:
                member_files.append(matches[0])

    if not member_files or not stations:
        return {
            "n_members": len(member_files),
            "n_stations": 0,
            "lead_hour": float(lead_hour),
            "t2m_crps_k": float("nan"),
            "t2m_ens_mean_rmse_k": float("nan"),
            "t2m_cf_rmse_k": float("nan"),
            "t2m_spread_k": float("nan"),
            "t2m_spread_skill_ratio": float("nan"),
            "mslp_crps_hpa": float("nan"),
            "wind10m_crps_ms": float("nan"),
            "brier_frost": float("nan"),
        }

    def _row_val(r: Any, key: str, default: float = float("nan")) -> float:
        if isinstance(r, dict):
            return float(r.get(key, default))
        return float(getattr(r, key, default))

    x_pts = np.array([_row_val(r, "x_m") for r in stations], dtype=np.float64)
    y_pts = np.array([_row_val(r, "y_m") for r in stations], dtype=np.float64)
    elevs = np.array([_row_val(r, "elev_m", 0.0) for r in stations], dtype=np.float64)
    obs_t2m = np.array([_row_val(r, "t2m_k") for r in stations], dtype=np.float64)
    obs_mslp_hpa = np.array([_row_val(r, "mslp_pa") for r in stations], dtype=np.float64) / 100.0
    obs_wspd = np.hypot(
        np.array([_row_val(r, "u10_ms") for r in stations], dtype=np.float64),
        np.array([_row_val(r, "v10_ms") for r in stations], dtype=np.float64),
    )

    hs_corr = (
        0.0065 * (elevs - _bilinear_sample_2d(np.asarray(hs_m, dtype=np.float64), spec, x_pts, y_pts))
        if hs_m is not None
        else np.zeros_like(elevs)
    )

    m_cnt = len(member_files)
    n_stn = len(stations)
    mod_t2m = np.empty((m_cnt, n_stn), dtype=np.float64)
    mod_mslp_hpa = np.empty((m_cnt, n_stn), dtype=np.float64)
    mod_wspd = np.empty((m_cnt, n_stn), dtype=np.float64)

    for m_idx, fp in enumerate(member_files):
        with wx1.open_dataset(fp) as ds:
            t2 = np.asarray(ds["t2m"].values, dtype=np.float64)
            psl = np.asarray(ds["mslp"].values, dtype=np.float64) / 100.0
            u10 = np.asarray(ds["u10m_geo"].values, dtype=np.float64)
            v10 = np.asarray(ds["v10m_geo"].values, dtype=np.float64)
        mod_t2m[m_idx] = _bilinear_sample_2d(t2, spec, x_pts, y_pts) - hs_corr
        mod_mslp_hpa[m_idx] = _bilinear_sample_2d(psl, spec, x_pts, y_pts)
        mod_wspd[m_idx] = np.hypot(
            _bilinear_sample_2d(u10, spec, x_pts, y_pts),
            _bilinear_sample_2d(v10, spec, x_pts, y_pts),
        )

    valid_t = np.isfinite(obs_t2m)
    if np.any(valid_t):
        t2m_crps_k = compute_fair_crps(mod_t2m[:, valid_t], obs_t2m[valid_t])
        ens_mean_t = np.mean(mod_t2m[:, valid_t], axis=0)
        t2m_ens_mean_rmse_k = float(np.sqrt(np.mean((ens_mean_t - obs_t2m[valid_t]) ** 2)))
        t2m_cf_rmse_k = float(np.sqrt(np.mean((mod_t2m[0, valid_t] - obs_t2m[valid_t]) ** 2)))
        t2m_spread_k = float(np.sqrt(np.mean(np.var(mod_t2m[:, valid_t], axis=0, ddof=1 if m_cnt > 1 else 0))))
        t2m_spread_skill_ratio = t2m_spread_k / max(t2m_ens_mean_rmse_k, 1e-12)
        p_frost = np.mean(mod_t2m[:, valid_t] < 273.15, axis=0)
        o_frost = (obs_t2m[valid_t] < 273.15).astype(np.float64)
        brier_frost = compute_brier_score(p_frost, o_frost)
    else:
        t2m_crps_k = float("nan")
        t2m_ens_mean_rmse_k = float("nan")
        t2m_cf_rmse_k = float("nan")
        t2m_spread_k = float("nan")
        t2m_spread_skill_ratio = float("nan")
        brier_frost = float("nan")

    mslp_crps_hpa = compute_fair_crps(mod_mslp_hpa, obs_mslp_hpa)
    wind10m_crps_ms = compute_fair_crps(mod_wspd, obs_wspd)

    return {
        "n_members": m_cnt,
        "n_stations": int(n_stn),
        "lead_hour": float(lead_hour),
        "t2m_crps_k": t2m_crps_k,
        "t2m_ens_mean_rmse_k": t2m_ens_mean_rmse_k,
        "t2m_cf_rmse_k": t2m_cf_rmse_k,
        "t2m_spread_k": t2m_spread_k,
        "t2m_spread_skill_ratio": t2m_spread_skill_ratio,
        "mslp_crps_hpa": mslp_crps_hpa,
        "wind10m_crps_ms": wind10m_crps_ms,
        "brier_frost": brier_frost,
    }
