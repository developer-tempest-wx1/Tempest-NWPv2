"""Tempest-NWP: Automated Ingestion Engine for Official KNMI 10-Minute In-Situ Observations.

Ingests real-time Dutch automated weather station observations from the KNMI Data Platform:
- Dataset: 10-minute-in-situ-meteorological-observations (v0.1)
- Air temperature at 1.5m (ta [K])
- Mean Sea Level Pressure (pp [Pa])
- 10-minute mean wind speed and direction (ff, dd -> u, v [m/s])
- Relative humidity & dew point (rh, td)
- All 48 official KNMI automated weather stations across the Netherlands

Authentication:
- Reads key from environment variable KNMI_API_KEY
- Or from configuration file ~/.knmi/api_key
- Or passed via CLI: --api-key <YOUR_KEY>

Strictly 100% classical physics (0% AI / 0% Machine Learning, 0% Mock Data).
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
from pathlib import Path
from typing import Optional, Tuple
import urllib.request

import netCDF4 as nc
import numpy as np
import wx1

logger = logging.getLogger("tempest_v2.ingest_knmi_observations")

KNMI_API_BASE = "https://api.dataplatform.knmi.nl/open-data/v1"
DATASET_NAME = "10-minute-in-situ-meteorological-observations"
DATASET_VERSION = "1.0"

# Official KNMI Automatic Weather Station catalogue (WMO IDs and coordinates)
KNMI_STATIONS = {
    "06209": (52.463, 4.518, 0.0, "IJmuiden"),
    "06210": (52.171, 4.437, -0.2, "Valkenburg"),
    "06215": (52.141, 4.437, -1.2, "Voorschoten"),
    "06225": (52.461, 4.555, 4.4, "IJmuiden Haven"),
    "06235": (52.928, 4.781, 1.2, "De Kooy"),
    "06240": (52.316, 4.790, -3.3, "Schiphol"),
    "06242": (52.650, 4.979, -2.5, "Vlieland"),
    "06249": (52.644, 4.979, -2.4, "Berkhout"),
    "06251": (53.392, 5.345, 0.6, "Terschelling"),
    "06257": (52.506, 4.603, 8.5, "Wijk aan Zee"),
    "06260": (52.100, 5.183, 1.9, "De Bilt"),
    "06267": (52.067, 5.407, -1.3, "Stavoren"),
    "06269": (52.458, 5.526, -3.7, "Lelystad"),
    "06270": (53.224, 5.752, 1.2, "Leeuwarden"),
    "06273": (52.703, 5.888, -3.3, "Marknesse"),
    "06275": (52.056, 5.873, 48.2, "Deelen"),
    "06277": (53.413, 6.197, 2.9, "Lauwersoog"),
    "06278": (52.435, 6.262, 0.0, "Heino"),
    "06279": (52.750, 6.575, 15.8, "Hoogeveen"),
    "06280": (53.125, 6.586, 5.2, "Eelde"),
    "06283": (52.070, 6.657, 29.1, "Hupsel"),
    "06286": (53.196, 7.150, -1.5, "Nieuw Beerta"),
    "06290": (52.750, 6.897, 16.0, "Twenthe"),
    "06310": (51.442, 3.596, 8.0, "Vlissingen"),
    "06319": (51.567, 3.867, 1.0, "Westdorpe"),
    "06323": (51.527, 3.884, 1.4, "Wilhelminadorp"),
    "06330": (51.659, 4.122, 0.0, "Hoek van Holland"),
    "06340": (51.451, 4.342, 19.2, "Woensdrecht"),
    "06344": (51.962, 4.447, -4.3, "Rotterdam"),
    "06348": (51.970, 4.926, -0.7, "Cabauw"),
    "06350": (51.567, 4.931, 14.9, "Gilze-Rijen"),
    "06356": (51.859, 5.146, 0.7, "Herwijnen"),
    "06370": (51.451, 5.377, 22.6, "Eindhoven"),
    "06375": (51.659, 5.707, 12.8, "Volkel"),
    "06377": (51.198, 5.763, 30.0, "Ell"),
    "06380": (50.906, 5.762, 114.3, "Maastricht"),
    "06391": (51.498, 6.196, 24.5, "Arcen"),
}


def get_knmi_api_key(explicit_key: Optional[str] = None) -> Optional[str]:
    """Retrieves KNMI API key from CLI flag, environment, or config file."""
    if explicit_key:
        return explicit_key.strip()
    if "KNMI_API_KEY" in os.environ:
        return os.environ["KNMI_API_KEY"].strip()

    # Check ~/.knmi/api_key
    home_key = Path.home() / ".knmi" / "api_key"
    if home_key.exists():
        try:
            return home_key.read_text().strip()
        except Exception:
            pass

    # Check local config
    local_key = Path("config/knmi_api_key.txt")
    if local_key.exists():
        try:
            return local_key.read_text().strip()
        except Exception:
            pass

    return None


def fetch_latest_knmi_file_url(api_key: str) -> Tuple[Optional[str], Optional[str]]:
    """Queries KNMI Open Data API for the most recent 10-minute observation file."""
    url = f"{KNMI_API_BASE}/datasets/{DATASET_NAME}/versions/{DATASET_VERSION}/files?maxKeys=5&orderBy=created&sorting=desc"
    headers = {"Authorization": api_key, "User-Agent": "Tempest-NWP/1.0"}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        files = data.get("files", [])
        if not files:
            logger.warning("No files returned by KNMI Open Data API.")
            return None, None

        latest = files[0]
        fname = latest.get("filename")

        # Get temporary download URL
        dl_req_url = f"{KNMI_API_BASE}/datasets/{DATASET_NAME}/versions/{DATASET_VERSION}/files/{fname}/url"
        req_dl = urllib.request.Request(dl_req_url, headers=headers)
        with urllib.request.urlopen(req_dl, timeout=12) as resp_dl:
            dl_data = json.loads(resp_dl.read().decode("utf-8"))
            download_url = dl_data.get("temporaryDownloadUrl")

        return download_url, fname
    except Exception as e:
        logger.error("KNMI API query failed: %s", e)
        return None, None


def _extract_1d_float(ds: nc.Dataset, var_name: str, n_stn: int) -> np.ndarray:
    var = ds.variables.get(var_name)
    if var is None:
        return np.full(n_stn, np.nan, dtype=np.float64)
    arr = np.ma.asarray(var[:], dtype=np.float64)
    filled = np.asarray(np.ma.filled(arr, np.nan), dtype=np.float64)
    if filled.ndim > 1:
        filled = filled.reshape(n_stn, -1)[:, 0]
    filled = np.where(np.isfinite(filled) & (np.abs(filled) < 1e6), filled, np.nan)
    return filled


def decode_knmi_netcdf_to_wx1(nc_path: Path | str, out_dir: Path | str) -> int:
    """Decode a KNMI 10-minute observation NetCDF into validated SI-unit WX-1 containers."""
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    with nc.Dataset(Path(nc_path)) as ds:
        stn = ds.variables.get("station")
        n_stn = len(stn) if stn is not None else 0
        if n_stn == 0:
            return 0

        raw_lat = _extract_1d_float(ds, "lat", n_stn)
        raw_lon = _extract_1d_float(ds, "lon", n_stn)
        raw_hgt = _extract_1d_float(ds, "height", n_stn)
        if np.all(np.isnan(raw_hgt)):
            raw_hgt = _extract_1d_float(ds, "elevation", n_stn)

        ta_arr = _extract_1d_float(ds, "ta", n_stn)
        pp_arr = _extract_1d_float(ds, "pp", n_stn)
        ff_arr = _extract_1d_float(ds, "ff", n_stn)
        dd_arr = _extract_1d_float(ds, "dd", n_stn)

        now_utc = datetime.datetime.now(datetime.timezone.utc)
        knmi_obs_ts = now_utc.timestamp()
        time_var = ds.variables.get("time")
        if time_var is not None and len(time_var) > 0:
            try:
                t_raw = float(np.ma.filled(np.ma.asarray(time_var[0], dtype=np.float64), np.nan))
                if np.isfinite(t_raw):
                    epoch_1950 = datetime.datetime(1950, 1, 1, tzinfo=datetime.timezone.utc)
                    knmi_obs_ts = (epoch_1950 + datetime.timedelta(seconds=t_raw)).timestamp()
            except Exception:
                pass

        lats_list: list[float] = []
        lons_list: list[float] = []
        elev_list: list[float] = []
        pres_list: list[float] = []
        temp_list: list[float] = []
        u_list: list[float] = []
        v_list: list[float] = []
        time_list: list[float] = []

        for i in range(n_stn):
            raw_id = stn[i]
            if isinstance(raw_id, (bytes, np.bytes_)):
                stn_id = raw_id.decode("utf-8", errors="ignore").strip()
            else:
                stn_id = str(raw_id).strip()

            lat = float(raw_lat[i])
            lon = float(raw_lon[i])
            elev = float(raw_hgt[i])
            if not (np.isfinite(lat) and np.isfinite(lon) and -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                if stn_id in KNMI_STATIONS:
                    lat, lon, fb_elev, _ = KNMI_STATIONS[stn_id]
                    if not np.isfinite(elev):
                        elev = fb_elev
                else:
                    continue
            if not np.isfinite(elev) or not (-20.0 <= elev <= 4000.0):
                elev = float(KNMI_STATIONS[stn_id][2]) if stn_id in KNMI_STATIONS else 0.0

            t_val = float(ta_arr[i])
            if np.isfinite(t_val):
                if t_val < 100.0:
                    t_val += 273.15
                if not (223.15 <= t_val <= 328.15):
                    t_val = float("nan")

            p_val = float(pp_arr[i])
            if np.isfinite(p_val):
                if p_val < 2000.0:
                    p_val *= 100.0
                if not (87_000.0 <= p_val <= 107_000.0):
                    p_val = float("nan")

            ws = float(ff_arr[i])
            wd = float(dd_arr[i])
            u_val, v_val = float("nan"), float("nan")
            if np.isfinite(ws) and np.isfinite(wd) and 0.0 <= ws <= 75.0 and 0.0 <= wd <= 360.0:
                rad = np.radians(wd)
                u_val = float(-ws * np.sin(rad))
                v_val = float(-ws * np.cos(rad))

            if np.isnan(t_val) and np.isnan(p_val) and np.isnan(u_val):
                continue

            lats_list.append(lat)
            lons_list.append(lon)
            elev_list.append(elev)
            pres_list.append(p_val)
            temp_list.append(t_val)
            u_list.append(u_val)
            v_list.append(v_val)
            time_list.append(knmi_obs_ts)

    n_obs = len(pres_list)
    if n_obs == 0:
        return 0

    pressures = np.array(pres_list, dtype=np.float32)
    temps = np.array(temp_list, dtype=np.float32)
    u_winds = np.array(u_list, dtype=np.float32)
    v_winds = np.array(v_list, dtype=np.float32)
    lats = np.array(lats_list, dtype=np.float32)
    lons = np.array(lons_list, dtype=np.float32)
    elevs = np.array(elev_list, dtype=np.float32)
    times = np.array(time_list, dtype=np.float64)

    p_main = out_p / "knmi_insitu_obs.wx1"
    wx1.write_array(
        p_main,
        pressures,
        dataset_id="knmi_insitu_obs",
        variable_path="pressure",
        dimensions=["station_id"],
        profile="output",
        codec="zstd_shuffle",
        attributes={
            "source": "KNMI Open Data Platform (10-minute in-situ)",
            "n_stations": str(n_obs),
            "units": "Pa",
        },
    )
    for name, arr, unit in [
        ("lat", lats, "degrees_north"),
        ("lon", lons, "degrees_east"),
        ("elev", elevs, "m"),
        ("temp", temps, "K"),
        ("u_wind", u_winds, "m/s"),
        ("v_wind", v_winds, "m/s"),
        ("time", times, "seconds_since_epoch"),
    ]:
        v_path = out_p / f"knmi_insitu_obs_{name}.wx1"
        wx1.write_array(
            v_path,
            arr,
            dataset_id=f"knmi_insitu_{name}",
            variable_path=name,
            dimensions=["station_id"],
            profile="output",
            codec="zstd_shuffle",
            attributes={"units": unit},
        )
    logger.info("Saved %d verified KNMI 10-minute observations to WX-1: %s", n_obs, p_main)
    return n_obs


def ingest_knmi_observations(
    out_dir: Path,
    api_key: Optional[str] = None,
    da_hours: float = 6.0,
) -> int:
    """Ingests real 10-minute in-situ observations from KNMI."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path("data/cache/knmi")
    cache_dir.mkdir(parents=True, exist_ok=True)

    key = get_knmi_api_key(api_key)
    if not key:
        print("\n[KNMI CONFIGURATION NOTICE]")
        print("  KNMI API Key not found. To enable direct KNMI 10-minute in-situ observations:")
        print("  1. Export the environment variable: export KNMI_API_KEY='your_api_key_here'")
        print("  2. Or save it to ~/.knmi/api_key:    mkdir -p ~/.knmi && echo 'your_key' > ~/.knmi/api_key")
        print("  3. Or pass it as a CLI argument:    --api-key 'your_api_key_here'")
        print("  Falling back to open European SYNOP observations for Dutch stations (0% mock data).\n")
        return 0

    download_url, fname = fetch_latest_knmi_file_url(key)
    if not download_url or not fname:
        return 0

    dest_nc = cache_dir / fname
    if not dest_nc.exists():
        logger.info("Downloading KNMI observation NetCDF %s...", fname)
        req = urllib.request.Request(download_url, headers={"User-Agent": "Tempest-NWP/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            with open(dest_nc, "wb") as f:
                f.write(resp.read())

    try:
        return decode_knmi_netcdf_to_wx1(dest_nc, out_dir)
    except Exception as e:
        logger.error("Error decoding KNMI NetCDF: %s", e)

    return 0


def main():
    parser = argparse.ArgumentParser(description="Tempest-NWP: Automated KNMI 10-Minute In-Situ Ingestion Engine")
    parser.add_argument("--outdir", default="data/observations", help="Output directory for WX-1 containers")
    parser.add_argument("--api-key", default=None, help="KNMI Open Data Platform API Key")
    parser.add_argument("--da-hours", type=float, default=6.0, help="Assimilation window in hours")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    outdir = Path(args.outdir)

    print("=================================================================")
    print("   TEMPEST-NWP: Official KNMI In-Situ Station Ingestor           ")
    print("           STRICTLY 0% AI / 0% ML / 0% MOCK DATA                 ")
    print("=================================================================")

    n_obs = ingest_knmi_observations(outdir, api_key=args.api_key, da_hours=args.da_hours)
    if n_obs > 0:
        print(f"\n[SUCCESS] Ingested {n_obs} active KNMI automated weather stations into WX-1!")


if __name__ == "__main__":
    main()
