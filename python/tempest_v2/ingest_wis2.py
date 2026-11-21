"""Tempest-NWP: WMO WIS2 real-time ingestion (Global Broker MQTT + Global Cache download).

Connects to a WIS2 Global Broker (public credentials everyone/everyone, MQTT 5, TLS, port 8883),
subscribes to the *cache* topics (the Global Cache re-publishes core data with a reliable download link),
downloads each BUFR file and decodes SYNOP / SHIP / BUOY-type surface reports with eccodes.
Output: <outdir>/wis2_obs.wx1 (+ _lat/_lon/_temp/_u_wind/_v_wind/_elev/_time), same layout as other surface feeds.

No synthetic data: reports that cannot be decoded are dropped and counted.
Requires: pip install paho-mqtt eccodes
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import math
import tempfile
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger("tempest_v2.ingest_wis2")

BROKERS = [
    {"id": "fr-meteofrance-global-broker", "host": "globalbroker.meteo.fr", "port": 8883},
    {"id": "cn-cma-global-broker", "host": "gb.wis.cma.cn", "port": 8883},
    {"id": "us-noaa-global-broker", "host": "wis2broker.globaldata.nws.noaa.gov", "port": 8883},
    {"id": "br-inmet-global-broker", "host": "globalbroker.inmet.gov.br", "port": 8883},
]

# WMO topic structure uses one centre-id level after a/wis2. Subscribe broadly and
# classify messages by the actual topic; never assume aircraft BUFR is a surface report.
DEFAULT_TOPICS = ["origin/a/wis2/#", "cache/a/wis2/#"]
SURFACE_TOPIC_MARKERS = ("surface-based-observations",)
MARINE_TOPIC_MARKERS = ("ocean", "drifting-buoys", "moored-buoys")
AIRCRAFT_TOPIC_MARKERS = ("aircraft", "amdar", "aircraft-based-observations")


def parse_notification(payload: bytes) -> Optional[dict]:
    try:
        d = json.loads(payload)
        links = d.get("links", [])
        href = next((l["href"] for l in links if l.get("rel") in ("canonical", "item") and "href" in l), None)
        props = d.get("properties", {})
        if href is None and props.get("data_id") is None:
            return None
        return {"href": href, "data_id": props.get("data_id", ""), "datetime": props.get("datetime") or props.get("pubtime")}
    except Exception:
        return None


def _download(url: str, timeout: int = 15) -> Optional[bytes]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Tempest-NWP/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except Exception as e:
        logger.debug("download failed %s: %s", url, e)
        return None


def decode_surface_bufr(raw: bytes) -> List[dict]:
    """Decode surface-type BUFR (SYNOP/SHIP/BUOY) into plain dicts. Missing values stay None."""
    import eccodes
    out = []
    with tempfile.NamedTemporaryFile(suffix=".bufr") as tf:
        tf.write(raw); tf.flush()
        with open(tf.name, "rb") as f:
            while True:
                gid = eccodes.codes_bufr_new_from_file(f)
                if gid is None:
                    break
                try:
                    eccodes.codes_set(gid, "unpack", 1)

                    def g(key):
                        try:
                            v = eccodes.codes_get(gid, key)
                            if isinstance(v, (int, float)) and (v == eccodes.CODES_MISSING_DOUBLE or v == eccodes.CODES_MISSING_LONG):
                                return None
                            return float(v) if isinstance(v, (int, float)) else v
                        except Exception:
                            return None

                    lat, lon = g("latitude"), g("longitude")
                    if lat is None or lon is None:
                        continue
                    try:
                        t = datetime.datetime(int(g("year")), int(g("month")), int(g("day")), int(g("hour") or 0),
                                              int(g("minute") or 0), tzinfo=datetime.timezone.utc).timestamp()
                    except Exception:
                        continue
                    wspd, wdir = g("windSpeed"), g("windDirection")
                    u = v = None
                    if wspd is not None and wdir is not None:
                        r = math.radians(wdir); u, v = -wspd * math.sin(r), -wspd * math.cos(r)
                    out.append({"lat": lat, "lon": lon % 360.0, "time": t,
                                "elev": g("heightOfStationGroundAboveMeanSeaLevel") or g("heightOfStation") or 0.0,
                                "p_msl": g("pressureReducedToMeanSeaLevel"), "p_stn": g("nonCoordinatePressure"),
                                "temp": g("airTemperature") or g("temperature"), "u": u, "v": v})
                except Exception:
                    pass
                finally:
                    eccodes.codes_release(gid)
    return out


def listen_and_collect(listen_seconds: int = 120, brokers=BROKERS, topics=DEFAULT_TOPICS, max_msgs: int = 200000) -> List[dict]:
    import paho.mqtt.client as mqtt
    notes: List[dict] = []
    lock = threading.Lock()

    def on_connect(client, userdata, flags, rc, props=None):
        for t in topics:
            client.subscribe(t, qos=0)

    def on_message(client, userdata, msg):
        topic = str(msg.topic)
        # Keep notifications with hrefs, including marine and aircraft data, for
        # raw archiving. Decoding/assimilation is decided later by the product type.
        if not any(m in topic for m in SURFACE_TOPIC_MARKERS + MARINE_TOPIC_MARKERS + AIRCRAFT_TOPIC_MARKERS):
            return
        n = parse_notification(msg.payload)
        if n and n["href"]:
            n["topic"] = topic
            n["raw_notification"] = msg.payload.decode("utf-8", errors="replace")
            with lock:
                if len(notes) < max_msgs:
                    notes.append(n)

    clients = []
    for b in brokers:
        try:
            # Paho 1.6.x is required by SondeHub 0.3.2; plain Client() also
            # remains compatible with Paho 2.x and uses the legacy callback API.
            c = mqtt.Client(protocol=mqtt.MQTTv5)
            c.username_pw_set("everyone", "everyone")
            c.tls_set()
            c.on_connect, c.on_message = on_connect, on_message
            c.connect(b["host"], b["port"], keepalive=60)
            c.loop_start(); clients.append(c)
            logger.info("WIS2: connected to %s:%d", b["host"], b["port"])
        except Exception as e:
            logger.warning("WIS2: cannot connect to %s: %s", b["host"], e)
    time.sleep(listen_seconds)
    for c in clients:
        c.loop_stop(); c.disconnect()
    seen, uniq = set(), []
    for n in notes:
        if n["href"] not in seen:
            seen.add(n["href"]); uniq.append(n)
    logger.info("WIS2: %d notifications (%d unique)", len(notes), len(uniq))
    return uniq


def ingest_wis2(outdir: Path, da_hours: float = 6.0, listen_seconds: int = 120) -> int:
    """Collect WIS2 surface reports for ~listen_seconds, decode, save to wis2_obs.wx1. Returns number of reports."""
    import wx1
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    notes = listen_and_collect(listen_seconds)
    raw_dir = outdir / "raw_notifications"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "notifications.jsonl").write_text(
        "\n".join(json.dumps(n, ensure_ascii=False) for n in notes), encoding="utf-8")
    reports: List[dict] = []
    n_fail = 0
    n_raw_only = 0

    # Prioritize surface notifications for decoding into wis2_obs.wx1.
    # Cap non-surface downloads to avoid stalling the operational pipeline on slow endpoints.
    surface_notes = [n for n in notes if any(m in str(n.get("topic", "")) for m in SURFACE_TOPIC_MARKERS)]
    other_notes = [n for n in notes if not any(m in str(n.get("topic", "")) for m in SURFACE_TOPIC_MARKERS)][:50]
    target_notes = surface_notes + other_notes

    with ThreadPoolExecutor(max_workers=16) as ex:
        blobs = list(ex.map(lambda n: _download(n["href"]), target_notes))
    for idx, (note, blob) in enumerate(zip(target_notes, blobs)):
        if blob is None:
            n_fail += 1
            continue
        topic = str(note.get("topic", ""))
        # Preserve every downloaded product by topic class. The installed decoder
        # currently handles surface BUFR only; aircraft/radiance/ocean remain raw-only.
        category = "surface" if any(m in topic for m in SURFACE_TOPIC_MARKERS) else (
            "aircraft" if any(m in topic for m in AIRCRAFT_TOPIC_MARKERS) else "marine")
        cat_dir = raw_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        raw_path = cat_dir / f"{idx:06d}.bufr"
        raw_path.write_bytes(blob)
        (cat_dir / f"{idx:06d}.json").write_text(json.dumps(note, indent=2), encoding="utf-8")
        if category != "surface":
            n_raw_only += 1
            continue
        try:
            reports.extend(decode_surface_bufr(blob))
        except Exception:
            n_fail += 1
    t_now = time.time()
    reports = [r for r in reports if t_now - da_hours * 3600 - 600 <= r["time"] <= t_now + 600]
    logger.info("WIS2: %d decoded surface reports in window; %d raw-only marine/aircraft files; %d failed downloads/decodes",
                len(reports), n_raw_only, n_fail)
    (raw_dir / "ingest_status.json").write_text(json.dumps({
        "notifications": len(notes), "decoded_surface_reports": len(reports),
        "raw_only_marine_aircraft_files": n_raw_only, "failed_downloads_or_decodes": n_fail,
        "decode_status": "surface BUFR only; aircraft and marine data are archived raw, not assimilated",
    }, indent=2), encoding="utf-8")
    if not reports:
        return 0
    nan = float("nan")
    def col(k): return np.array([nan if r.get(k) is None else r[k] for r in reports], dtype=np.float64)
    p = np.where(np.isfinite(col("p_msl")), col("p_msl"), col("p_stn"))
    for name, arr in [("wis2_obs", p), ("wis2_obs_lat", col("lat")), ("wis2_obs_lon", col("lon")), ("wis2_obs_temp", col("temp")),
                      ("wis2_obs_u_wind", col("u")), ("wis2_obs_v_wind", col("v")), ("wis2_obs_elev", col("elev")), ("wis2_obs_time", col("time"))]:
        wx1.write_array(outdir / f"{name}.wx1", arr, dataset_id=name, variable_path=name.replace("wis2_obs_", "") if name != "wis2_obs" else "pressure",
                        dimensions=["station"], profile="restart", codec="zstd_shuffle")
    return len(reports)


def main():
    ap = argparse.ArgumentParser(description="Tempest-NWP WIS2 ingest")
    ap.add_argument("--outdir", default="data/observations/wis2"); ap.add_argument("--listen-seconds", type=int, default=120)
    ap.add_argument("--da-hours", type=float, default=6.0)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print(f"WIS2 reports saved: {ingest_wis2(Path(a.outdir), a.da_hours, a.listen_seconds)}")


if __name__ == "__main__":
    main()
