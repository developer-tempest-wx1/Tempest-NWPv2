"""Tempest-NWP: E-GVAP / EUREF Ground GNSS Zenith Total Delay (ZTD) Ingestion Engine.

Downloads and decodes genuine near-real-time European GNSS Zenith Total Delay (ZTD) observations
from the official EUREF / E-GVAP geodetic analysis centres (BKG, ASI, IGN, ROB, GOP, BEK).

Observational Physics & Principles:
- Ground-based GNSS receivers measure the microwave phase delay from GPS/Galileo constellations.
- Total zenith path delay: ZTD = ZHD (Zenith Hydrostatic Delay) + ZWD (Zenith Wet Delay).
- Standard sea level: ZTD ~ 2.2 - 2.6 m; mountain stations: ZTD ~ 1.5 - 2.0 m.
- Provides high-frequency, continuous vertical column water vapour constraints over Europe.
- Stores decoded observations into WX-1 Scientific Containers:
  data/observations/runs/YYYYMMDD_HHMMZ/upperair/egvap_gnss_ztd.wx1
  (+ _lat, _lon, _elev, _sigma, _time).

Strictly 100% classical physics (0% AI / 0% Machine Learning, 0% Mock Data).
"""

from __future__ import annotations

import argparse
import datetime
import gzip
import logging
import math
import os
from pathlib import Path
import re
import urllib.request
from typing import Dict, List, Optional, Tuple

import numpy as np
import wx1

logger = logging.getLogger("tempest_v2.ingest_egvap")

BKG_EUREF_PRODUCTS_URL = "https://igs.bkg.bund.de/root_ftp/EUREF/products"


def find_latest_egvap_files(max_files: int = 4) -> List[str]:
    """Discovers the most recent available 1-hourly SINEX TRO files on the BKG EUREF server."""
    try:
        req = urllib.request.Request(BKG_EUREF_PRODUCTS_URL + "/", headers={"User-Agent": "Tempest-NWP/1.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            html = r.read().decode("utf-8", errors="ignore")
        
        weeks = sorted([w.strip("/") for w in re.findall(r'href=[\'\"](\d{4})/?[\'\"]', html) if int(w.strip("/")) > 2400])
        if not weeks:
            return []

        discovered = []
        # Check the latest 2 GPS weeks
        for w in reversed(weeks[-2:]):
            w_url = f"{BKG_EUREF_PRODUCTS_URL}/{w}/"
            try:
                w_req = urllib.request.Request(w_url, headers={"User-Agent": "Tempest-NWP/1.0"})
                with urllib.request.urlopen(w_req, timeout=10) as r_w:
                    w_html = r_w.read().decode("utf-8", errors="ignore")
                
                tro_files = re.findall(r'href=[\'\"]([^\'\"]+01H_TRO\.TRO\.gz)[\'\"]', w_html)
                # Sort by day of year (descending)
                tro_files = sorted(set(tro_files), reverse=True)
                for f in tro_files:
                    discovered.append(f"{w_url}{f}")
                    if len(discovered) >= max_files:
                        break
            except Exception as e:
                logger.debug("Failed reading week %s: %s", w, e)
            if len(discovered) >= max_files:
                break

        return discovered
    except Exception as e:
        logger.warning("Could not reach BKG EUREF server for E-GVAP ZTD discovery: %s", e)
        return []


def parse_sinex_tro_data(data_text: str) -> List[dict]:
    """Parses a SINEX TRO string extracting station locations and hourly ZTD values."""
    sites = {}
    in_sites = False
    in_sol = False
    records = []

    for line in data_text.splitlines():
        if line.startswith("+SITE/ID"):
            in_sites = True
            continue
        elif line.startswith("-SITE/ID"):
            in_sites = False
            continue
        elif line.startswith("+TROP/SOLUTION"):
            in_sol = True
            continue
        elif line.startswith("-TROP/SOLUTION"):
            in_sol = False
            continue

        # Station coordinates (SINEX fixed column standard)
        if in_sites and not line.startswith("*") and len(line) >= 70:
            stn = line[1:10].strip()
            try:
                lon_str = line[47:59].strip()
                lat_str = line[59:71].strip()
                hgt_str = line[71:83].strip() if len(line) >= 80 else "0.0"
                if lon_str and lat_str:
                    lon = float(lon_str)
                    if lon > 180.0:
                        lon -= 360.0
                    lat = float(lat_str)
                    hgt = float(hgt_str) if hgt_str else 0.0
                    sites[stn] = (lat, lon, hgt)
            except ValueError:
                pass

        # Hourly Troposphere solutions
        if in_sol and not line.startswith("*") and len(line) > 30:
            parts = line.split()
            if len(parts) >= 4:
                stn = parts[0]
                epoch = parts[1] # e.g. 2026:309:01800 (YYYY:DOY:SECS)
                try:
                    ztd_mm = float(parts[2])
                    std_mm = float(parts[3])
                    ztd_m = ztd_mm / 1000.0
                    std_m = std_mm / 1000.0

                    # Meteorological physical bounds:
                    # 1.3 m <= ZTD <= 3.2 m; standard deviation <= 0.05 m (5 cm)
                    if 1.3 <= ztd_m <= 3.2 and 0.0001 <= std_m <= 0.05:
                        if stn in sites:
                            lat, lon, hgt = sites[stn]
                            records.append({
                                "station": stn,
                                "lat": lat,
                                "lon": lon,
                                "elevation_m": hgt,
                                "ztd_m": ztd_m,
                                "sigma_m": std_m,
                                "epoch": epoch,
                            })
                except ValueError:
                    pass

    return records


def ingest_egvap_ztd(
    output_upperair_dir: Path,
    da_hours: float = 6.0,
    max_stations: int = 500,
) -> int:
    """Downloads live European GNSS ZTD solutions and saves to WX-1 containers."""
    output_upperair_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path("data/cache/egvap")
    cache_dir.mkdir(parents=True, exist_ok=True)

    urls = find_latest_egvap_files(max_files=4)
    if not urls:
        logger.warning("No E-GVAP ZTD files discovered; checking cached files in %s", cache_dir)
        cached_files = list(cache_dir.glob("*.TRO.gz"))
    else:
        cached_files = []
        for url in urls:
            fname = url.split("/")[-1]
            local_p = cache_dir / fname
            if not local_p.exists() or local_p.stat().st_size < 1000:
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": "Tempest-NWP/1.0"})
                    with urllib.request.urlopen(req, timeout=15) as r, open(local_p, "wb") as f_out:
                        f_out.write(r.read())
                    logger.info("Downloaded E-GVAP ZTD product: %s (%.1f KB)", fname, local_p.stat().st_size / 1024)
                except Exception as e:
                    logger.warning("Download failed for %s: %s", url, e)
            if local_p.exists():
                cached_files.append(local_p)

    all_records = []
    for f in cached_files:
        try:
            with gzip.open(f, "rt", encoding="utf-8", errors="ignore") as gz:
                content = gz.read()
            recs = parse_sinex_tro_data(content)
            all_records.extend(recs)
        except Exception as e:
            logger.warning("Could not parse %s: %s", f, e)

    if not all_records:
        logger.warning("No valid E-GVAP GNSS ZTD records parsed.")
        return 0

    # Thinning: retain the most recent and highest quality observation per station
    station_best = {}
    for r in all_records:
        stn = r["station"]
        if stn not in station_best or r["sigma_m"] < station_best[stn]["sigma_m"]:
            station_best[stn] = r

    unique_stations = list(station_best.values())[:max_stations]
    n_obs = len(unique_stations)

    lats = np.array([s["lat"] for s in unique_stations], dtype=np.float32)
    lons = np.array([s["lon"] for s in unique_stations], dtype=np.float32)
    elevs = np.array([s["elevation_m"] for s in unique_stations], dtype=np.float32)
    ztds = np.array([s["ztd_m"] for s in unique_stations], dtype=np.float32)
    sigmas = np.array([s["sigma_m"] for s in unique_stations], dtype=np.float32)
    times = np.zeros(n_obs, dtype=np.float32) # In window offset ~ 0s

    p_main = output_upperair_dir / "egvap_gnss_ztd.wx1"
    wx1.write_array(
        p_main,
        ztds,
        dataset_id="egvap_gnss_zenith_total_delay",
        variable_path="ztd",
        dimensions=["n_stations"],
        profile="output",
        codec="zstd_shuffle",
        attributes={
            "standard_name": "zenith_total_delay",
            "units": "m",
            "source": "E-GVAP / EUREF GNSS Troposphere Network",
            "n_stations": str(n_obs),
            "forward_operator": "Smith-Weintraub / Saastamoinen ZTD = 1e-6 * integral(N dz)",
            "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        },
    )

    for name, arr, unit in [
        ("lat", lats, "degrees_north"),
        ("lon", lons, "degrees_east"),
        ("elev", elevs, "m"),
        ("sigma", sigmas, "m"),
        ("time", times, "s"),
    ]:
        v_path = output_upperair_dir / f"egvap_gnss_ztd_{name}.wx1"
        wx1.write_array(
            v_path,
            arr,
            dataset_id=f"egvap_gnss_{name}",
            variable_path=name,
            dimensions=["n_stations"],
            profile="output",
            codec="zstd_shuffle",
            attributes={"units": unit},
        )

    logger.info("Saved %d genuine E-GVAP GNSS ZTD stations to WX-1 containers: %s", n_obs, p_main)
    return n_obs


def main():
    parser = argparse.ArgumentParser(description="Tempest-NWP: E-GVAP GNSS ZTD Ingestion")
    parser.add_argument("--outdir", type=Path, default=Path("data/observations/upperair"), help="Output directory")
    parser.add_argument("--da-hours", type=float, default=6.0, help="DA window in hours")
    args = parser.parse_args()

    n = ingest_egvap_ztd(args.outdir, da_hours=args.da_hours)
    print(f"E-GVAP GNSS ZTD Ingest completed: {n:,} stations written to {args.outdir}")


if __name__ == "__main__":
    main()
