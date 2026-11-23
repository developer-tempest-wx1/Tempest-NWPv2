"""Automated KNMI 10-Minute In-Situ Verification & Cartographic Diagnostics for Tempest-NWP v2.

Strictly 100% deterministic classical verification (0% AI / 0% ML, 0% mock data):
- Fetches or reads cached official KNMI 10-minute automated weather station NetCDFs
  (`10-minute-in-situ-meteorological-observations`, v1.0).
- Bilinearly samples `.wx1` forecast files (`t2m`, `td2m`, `rh2m`, `mslp`, `u10m_geo`,
  `v10m_geo`, `gust10m`, `vis_m`, `precip_rate`, `rain_acc`/`snow_acc`/`graupel_acc`)
  on the native LCC projection grid.
- Computes WMO continuous statistics (Bias, MAE, RMSE, Pearson r), circular wind-direction
  errors, and fog contingency metrics (< 1000 m POD, FAR, CSI), split across all stations,
  land-only stations, and independent lead times (excluding DA overlap & initial spin-up).
- Generates per-hour 2x2 verification maps and a run-wide lead-time dashboard PNG + CSV/JSON.
"""

from __future__ import annotations

import csv
import datetime
import json
import logging
import math
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import netCDF4 as nc
import numpy as np
from scipy.spatial import cKDTree
import xarray as xr
import wx1

from tempest_v2.ingest_knmi_observations import (
    DATASET_NAME,
    DATASET_VERSION,
    KNMI_API_BASE,
    KNMI_STATIONS,
    _extract_1d_float,
    get_knmi_api_key,
)
from tempest_v2.projection import get_domain_spec

logger = logging.getLogger("tempest_v2.knmi_verify")

_LEAD_RE = re.compile(r"_T\+([0-9]+(?:\.[0-9]+)?)h\.wx1$")
_IFS_SFC_RE = re.compile(r"(\d{8})_(\d{2})Z_sfc_step(\d+)")

HOURLY_VALIDATED_DATASET = "hourly-in-situ-meteorological-observations-validated"
HOURLY_VALIDATED_VERSION = "1.0"

_WMO_4680_HAIL_CODES = frozenset({74, 75, 76, 89, 93, 96, 99})
_WMO_4680_SHOWER_CODES = frozenset({80, 81, 82, 83, 84, 85, 86, 87, 89, 91, 92, 93, 94, 95, 96, 99})

# WMO IDs of Dutch North Sea platforms / buoys that do not represent inland land stations
_OFFSHORE_STATION_IDS = frozenset(
    {
        "06201",
        "06203",
        "06204",
        "06205",
        "06207",
        "06208",
        "06211",
        "06212",
        "06213",
        "06214",
        "06216",
        "06252",
        "06320",
        "06321",
    }
)
_OFFSHORE_KEYWORDS = (
    "platform",
    "lichteiland",
    "boei",
    "euro",
    "k13",
    "k14",
    "l9",
    "f3",
    "f16",
    "j6",
    "p11",
    "a12",
    "awg",
    "d15",
    "q1",
    "hoorn",
    "north cormorant",
    "auk",
)


def _to_utc(dt: datetime.datetime) -> datetime.datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=datetime.timezone.utc)
    return dt.astimezone(datetime.timezone.utc)


def _snap_10min_utc(dt: datetime.datetime) -> datetime.datetime:
    u = _to_utc(dt)
    return u.replace(minute=(u.minute // 10) * 10, second=0, microsecond=0)


def _snap_hourly_utc(dt: datetime.datetime, tolerance_s: float = 300.0) -> Optional[datetime.datetime]:
    u = _to_utc(dt)
    if u.minute == 0 and u.second == 0 and u.microsecond == 0:
        return u
    nearest = (u + datetime.timedelta(minutes=30)).replace(minute=0, second=0, microsecond=0)
    if abs((u - nearest).total_seconds()) <= float(tolerance_s):
        return nearest
    return None


def _normalize_knmi_station_id(raw_id: str) -> str:
    """Normalize WIGOS/WMO station IDs (e.g. `0-20000-0-06260` -> `06260`)."""
    s = str(raw_id).strip()
    if s.startswith("0-") and "-" in s:
        tail = s.rsplit("-", 1)[-1].strip()
        if len(tail) == 5 and tail.isdigit():
            return tail
    return s


def _knmi_vv_code_to_meters(code: float) -> float:
    """Convert KNMI WMO VV visibility code (0..89) to representative visibility [m]."""
    if not math.isfinite(code) or code < 0.0 or code > 89.0:
        return float("nan")
    c = int(round(code))
    if 0 <= c <= 49:
        return (c + 0.5) * 100.0
    if c == 50:
        return 5500.0
    if 56 <= c <= 79:
        return (c - 49.5) * 1000.0
    if 80 <= c <= 88:
        return (30.0 + (c - 80) * 5.0 + 2.5) * 1000.0
    if c == 89:
        return 75_000.0
    return float("nan")


def _decode_str_array(ds: nc.Dataset, var_name: str, n_stn: int) -> List[str]:
    var = ds.variables.get(var_name)
    if var is None:
        return ["" for _ in range(n_stn)]
    raw = var[:]
    out: List[str] = []
    for i in range(n_stn):
        item = raw[i]
        if isinstance(item, np.ma.MaskedArray):
            item = item.compressed().tobytes() if item.dtype.kind == "S" else item.data
        if isinstance(item, (bytes, np.bytes_)):
            s = item.decode("utf-8", errors="ignore").strip()
        elif isinstance(item, np.ndarray) and item.dtype.kind in ("S", "U"):
            if item.dtype.kind == "S":
                s = item.tobytes().decode("utf-8", errors="ignore").strip("\x00 ")
            else:
                s = "".join(str(x) for x in item.ravel()).strip()
        else:
            s = str(item).strip()
        out.append(s)
    return out


def _is_offshore_station(stn_id: str, stn_name: str) -> bool:
    if stn_id in _OFFSHORE_STATION_IDS:
        return True
    low = stn_name.lower()
    return any(kw in low for kw in _OFFSHORE_KEYWORDS)


def _download_knmi_dataset_file(
    dataset_name: str,
    dataset_version: str,
    fname: str,
    cache_dir: Path | str,
    *,
    api_key: Optional[str] = None,
) -> Optional[Path]:
    """Download (or reuse cached) a NetCDF file from KNMI Open Data API."""
    cdir = Path(cache_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    out_p = cdir / fname
    if out_p.is_file() and out_p.stat().st_size > 1024:
        return out_p

    key = get_knmi_api_key(api_key)
    if not key:
        logger.warning("No KNMI API key configured and %s is not cached.", fname)
        return None

    headers = {"Authorization": key, "User-Agent": "Tempest-NWP/1.0"}
    meta_url = f"{KNMI_API_BASE}/datasets/{dataset_name}/versions/{dataset_version}/files/{fname}/url"
    req = urllib.request.Request(meta_url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            dl_url = json.loads(resp.read().decode("utf-8")).get("temporaryDownloadUrl")
        if not dl_url:
            return None
        tmp_p = cdir / f".{fname}.{os.getpid()}.tmp"
        urllib.request.urlretrieve(dl_url, tmp_p)
        os.replace(tmp_p, out_p)
        return out_p
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            logger.debug("KNMI %s file not available (404): %s", dataset_name, fname)
            return None
        if exc.code in (401, 403):
            raise PermissionError(f"KNMI API authentication failed (HTTP {exc.code}) for {fname}") from exc
        logger.warning("KNMI API HTTP %d for %s: %s", exc.code, fname, exc)
        return None
    except Exception as exc:
        logger.warning("KNMI download failed for %s: %s", fname, exc)
        return None


def fetch_knmi_10min_netcdf(
    valid_time_utc: datetime.datetime,
    cache_dir: Path | str,
    *,
    api_key: Optional[str] = None,
) -> Optional[Path]:
    """Download (or reuse cached) official KNMI 10-minute NetCDF for `valid_time_utc`."""
    snap = _snap_10min_utc(valid_time_utc)
    fname = f"KMDS__OPER_P___10M_OBS_L2_{snap:%Y%m%d%H%M}.nc"
    return _download_knmi_dataset_file(DATASET_NAME, DATASET_VERSION, fname, cache_dir, api_key=api_key)


def fetch_knmi_hourly_validated_netcdf(
    valid_time_utc: datetime.datetime,
    cache_dir: Path | str,
    *,
    api_key: Optional[str] = None,
) -> Optional[Path]:
    """Download (or reuse cached) official KNMI hourly-validated NetCDF (1951-present)."""
    snap = _snap_hourly_utc(valid_time_utc)
    if snap is None:
        return None
    fname = f"hourly-observations-validated-{snap:%Y%m%d-%H}.nc"
    return _download_knmi_dataset_file(
        HOURLY_VALIDATED_DATASET, HOURLY_VALIDATED_VERSION, fname, cache_dir, api_key=api_key
    )


def _extract_knmi_station_meta(ds: nc.Dataset, n_stn: int) -> Tuple[List[str], List[str], np.ndarray, np.ndarray, np.ndarray, Optional[datetime.datetime]]:
    raw_ids = _decode_str_array(ds, "station", n_stn)
    wsi_ids = _decode_str_array(ds, "wsi", n_stn) if "wsi" in ds.variables else [""] * n_stn
    ids = [_normalize_knmi_station_id(raw_ids[i] or wsi_ids[i]) for i in range(n_stn)]
    names = _decode_str_array(ds, "stationname", n_stn)
    lats = _extract_1d_float(ds, "lat", n_stn)
    lons = _extract_1d_float(ds, "lon", n_stn)
    hgts = _extract_1d_float(ds, "height", n_stn)
    if np.all(np.isnan(hgts)):
        hgts = _extract_1d_float(ds, "elevation", n_stn)

    obs_dt: Optional[datetime.datetime] = None
    tvar = ds.variables.get("time")
    if tvar is not None and len(tvar) > 0:
        try:
            t_raw = float(np.ma.filled(np.ma.asarray(tvar[0], dtype=np.float64), np.nan))
            if np.isfinite(t_raw):
                obs_dt = datetime.datetime(1950, 1, 1, tzinfo=datetime.timezone.utc) + datetime.timedelta(
                    seconds=t_raw
                )
        except Exception:
            pass
    return ids, names, lats, lons, hgts, obs_dt


def _build_knmi_station_row(
    *,
    sid: str,
    raw_name: str,
    lat: float,
    lon: float,
    elev: float,
    obs_dt: Optional[datetime.datetime],
    t_raw: float,
    td_raw: float,
    rh_raw: float,
    p_raw: float,
    ws_raw: float,
    wd_raw: float,
    gust_raw: float,
    vis_v: float,
    precip_mmh: float,
    precip_1h_mm: float,
    sw_down_wm2: float,
    tcc: float,
    obs_source: str,
    obs_fog_w1: Optional[int] = None,
    obs_rain_w2: Optional[int] = None,
    obs_snow_w3: Optional[int] = None,
    obs_thunder_w5: Optional[int] = None,
    obs_ice_w6: Optional[int] = None,
    obs_ww_code: Optional[int] = None,
    obs_hail: Optional[int] = None,
    obs_convective_shower: Optional[int] = None,
    reject_calm_dir: bool = False,
) -> Optional[Dict[str, Any]]:
    sname = raw_name or (KNMI_STATIONS[sid][3] if sid in KNMI_STATIONS else sid)
    if not (math.isfinite(lat) and math.isfinite(lon) and -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        if sid in KNMI_STATIONS:
            lat, lon, fb_elev, fb_name = KNMI_STATIONS[sid]
            sname = sname or fb_name
            if not math.isfinite(elev):
                elev = fb_elev
        else:
            return None
    if not math.isfinite(elev):
        elev = float(KNMI_STATIONS[sid][2]) if sid in KNMI_STATIONS else 0.0

    t2m_c = float(t_raw)
    if math.isfinite(t2m_c) and t2m_c > 150.0:
        t2m_c -= 273.15
    if not (-55.0 <= t2m_c <= 55.0):
        t2m_c = float("nan")

    td2m_c = float(td_raw)
    if math.isfinite(td2m_c) and td2m_c > 150.0:
        td2m_c -= 273.15
    if not (-65.0 <= td2m_c <= 45.0):
        td2m_c = float("nan")

    rh_pct = float(rh_raw) if (math.isfinite(rh_raw) and 0.0 <= rh_raw <= 105.0) else float("nan")

    mslp_hpa = float(p_raw)
    if math.isfinite(mslp_hpa) and mslp_hpa > 2000.0:
        mslp_hpa *= 0.01
    if not (870.0 <= mslp_hpa <= 1075.0):
        mslp_hpa = float("nan")

    ws = float(ws_raw)
    wd = float(wd_raw)
    wd_valid = math.isfinite(wd) and ((0.0 < wd <= 360.0) if reject_calm_dir else (0.0 <= wd <= 360.0))
    if not wd_valid:
        wd = float("nan")

    u10, v10 = float("nan"), float("nan")
    if math.isfinite(ws) and 0.0 <= ws <= 75.0:
        if math.isfinite(wd):
            rad = math.radians(wd)
            u10 = -ws * math.sin(rad)
            v10 = -ws * math.cos(rad)
    else:
        ws = float("nan")

    gust = float(gust_raw) if (math.isfinite(gust_raw) and 0.0 <= gust_raw <= 100.0) else float("nan")
    if all(math.isnan(x) for x in (t2m_c, mslp_hpa, ws, rh_pct, vis_v, precip_mmh, sw_down_wm2, tcc)):
        return None

    return {
        "station_id": sid,
        "station_name": sname,
        "lat": lat,
        "lon": lon,
        "elev_m": elev,
        "is_offshore": _is_offshore_station(sid, sname),
        "obs_time_utc": obs_dt.isoformat() if obs_dt is not None else "",
        "obs_source": obs_source,
        "t2m_c": t2m_c,
        "td2m_c": td2m_c,
        "rh_pct": rh_pct,
        "mslp_hpa": mslp_hpa,
        "ws10_ms": ws,
        "wd10_deg": wd,
        "u10_ms": u10,
        "v10_ms": v10,
        "gust10_ms": gust,
        "vis_m": vis_v,
        "obs_fog_w1": obs_fog_w1,
        "obs_rain_w2": obs_rain_w2,
        "obs_snow_w3": obs_snow_w3,
        "obs_thunder_w5": obs_thunder_w5,
        "obs_ice_w6": obs_ice_w6,
        "obs_ww_code": obs_ww_code,
        "obs_hail": obs_hail,
        "obs_convective_shower": obs_convective_shower,
        "precip_mmh": precip_mmh,
        "precip_1h_mm": precip_1h_mm,
        "sw_down_wm2": sw_down_wm2,
        "tcc": tcc,
    }


def load_knmi_10min_stations(nc_path: Path | str) -> List[Dict[str, Any]]:
    """Parse KNMI 10-minute station NetCDF into clean SI/meteorological station dicts."""
    rows: List[Dict[str, Any]] = []
    with nc.Dataset(Path(nc_path)) as ds:
        stn_var = ds.variables.get("station")
        n_stn = len(stn_var) if stn_var is not None else 0
        if n_stn == 0:
            return rows

        ids, names, lats, lons, hgts, obs_dt = _extract_knmi_station_meta(ds, n_stn)
        ta = _extract_1d_float(ds, "ta", n_stn)
        td = _extract_1d_float(ds, "td", n_stn)
        rh = _extract_1d_float(ds, "rh", n_stn)
        pp = _extract_1d_float(ds, "pp", n_stn)
        ff = _extract_1d_float(ds, "ff", n_stn)
        dd = _extract_1d_float(ds, "dd", n_stn)
        fx = _extract_1d_float(ds, "fx", n_stn)
        zm = _extract_1d_float(ds, "zm", n_stn)
        vv = _extract_1d_float(ds, "vv", n_stn)
        rg = _extract_1d_float(ds, "rg", n_stn)
        qg = _extract_1d_float(ds, "qg", n_stn)

        for i in range(n_stn):
            zm_i, vv_i = float(zm[i]), float(vv[i])
            vis_v = zm_i if (math.isfinite(zm_i) and 0.0 <= zm_i <= 100_000.0) else float("nan")
            if not math.isfinite(vis_v) and math.isfinite(vv_i) and 0.0 <= vv_i <= 100_000.0:
                vis_v = vv_i
            rg_i = float(rg[i])
            rg_mmh = rg_i if (math.isfinite(rg_i) and 0.0 <= rg_i <= 300.0) else float("nan")
            qg_i = float(qg[i])
            qg_wm2 = qg_i if (math.isfinite(qg_i) and 0.0 <= qg_i <= 1400.0) else float("nan")

            row = _build_knmi_station_row(
                sid=ids[i],
                raw_name=names[i],
                lat=float(lats[i]),
                lon=float(lons[i]),
                elev=float(hgts[i]),
                obs_dt=obs_dt,
                t_raw=float(ta[i]),
                td_raw=float(td[i]),
                rh_raw=float(rh[i]),
                p_raw=float(pp[i]),
                ws_raw=float(ff[i]),
                wd_raw=float(dd[i]),
                gust_raw=float(fx[i]),
                vis_v=vis_v,
                precip_mmh=rg_mmh,
                precip_1h_mm=float("nan"),
                sw_down_wm2=qg_wm2,
                tcc=float("nan"),
                obs_source="10min",
                obs_fog_w1=None,
                reject_calm_dir=False,
            )
            if row is not None:
                rows.append(row)
    return rows


def load_knmi_hourly_validated_stations(nc_path: Path | str) -> List[Dict[str, Any]]:
    """Parse KNMI hourly validated NetCDF (`hourly-in-situ-meteorological-observations-validated` v1.0)."""
    rows: List[Dict[str, Any]] = []
    with nc.Dataset(Path(nc_path)) as ds:
        stn_var = ds.variables.get("station")
        n_stn = len(stn_var) if stn_var is not None else 0
        if n_stn == 0:
            return rows

        ids, names, lats, lons, hgts, obs_dt = _extract_knmi_station_meta(ds, n_stn)
        t_arr = _extract_1d_float(ds, "T", n_stn)
        td_arr = _extract_1d_float(ds, "TD", n_stn)
        u_arr = _extract_1d_float(ds, "U", n_stn)
        p_arr = _extract_1d_float(ds, "P", n_stn)
        ff_arr = _extract_1d_float(ds, "FF", n_stn)
        fh_arr = _extract_1d_float(ds, "FH", n_stn)
        fx_arr = _extract_1d_float(ds, "FX", n_stn)
        dd_arr = _extract_1d_float(ds, "DD", n_stn)
        rh_mm_arr = _extract_1d_float(ds, "RH", n_stn)  # hourly precip amount [mm]
        q_jcm2_arr = _extract_1d_float(ds, "Q", n_stn)  # hourly global solar radiation [J/cm^2]
        n_okta_arr = _extract_1d_float(ds, "N", n_stn)  # cloud cover [0..8 okta, 9=obscured]
        vv_code_arr = _extract_1d_float(ds, "VV", n_stn)
        w1_arr = _extract_1d_float(ds, "W1", n_stn)
        w2_arr = _extract_1d_float(ds, "W2", n_stn)
        w3_arr = _extract_1d_float(ds, "W3", n_stn)
        w5_arr = _extract_1d_float(ds, "W5", n_stn)
        w6_arr = _extract_1d_float(ds, "W6", n_stn)
        ww_arr = _extract_1d_float(ds, "WW", n_stn)

        def _flag01(val: float) -> Optional[int]:
            return int(round(val)) if (math.isfinite(val) and val in (0.0, 1.0)) else None

        for i in range(n_stn):
            ws_i = float(ff_arr[i]) if math.isfinite(float(ff_arr[i])) else float(fh_arr[i])

            rh_raw = float(rh_mm_arr[i])
            if math.isfinite(rh_raw) and -0.1 <= rh_raw < 0.0:
                rh_1h_mm = 0.025
            elif math.isfinite(rh_raw) and 0.0 <= rh_raw <= 300.0:
                rh_1h_mm = rh_raw
            else:
                rh_1h_mm = float("nan")

            q_raw = float(q_jcm2_arr[i])
            sw_wm2 = (q_raw * (10_000.0 / 3600.0)) if (math.isfinite(q_raw) and 0.0 <= q_raw * (10_000.0 / 3600.0) <= 1400.0) else float("nan")

            n_raw = float(n_okta_arr[i])
            tcc_val = (n_raw / 8.0) if (math.isfinite(n_raw) and 0.0 <= n_raw <= 8.0) else float("nan")

            obs_fog_w1 = _flag01(float(w1_arr[i]))
            obs_rain_w2 = _flag01(float(w2_arr[i]))
            obs_snow_w3 = _flag01(float(w3_arr[i]))
            obs_thunder_w5 = _flag01(float(w5_arr[i]))
            obs_ice_w6 = _flag01(float(w6_arr[i]))
            ww_raw = float(ww_arr[i])
            obs_ww = int(round(ww_raw)) if (math.isfinite(ww_raw) and 0.0 <= ww_raw <= 99.0) else None
            if obs_ww is not None:
                obs_hail: Optional[int] = 1 if obs_ww in _WMO_4680_HAIL_CODES else 0
                obs_shower: Optional[int] = 1 if obs_ww in _WMO_4680_SHOWER_CODES else 0
            else:
                obs_hail = None
                obs_shower = 1 if obs_thunder_w5 == 1 else None

            vis_v = _knmi_vv_code_to_meters(float(vv_code_arr[i]))
            if not math.isfinite(vis_v) and obs_fog_w1 is not None:
                vis_v = 500.0 if obs_fog_w1 == 1 else 10_000.0

            row = _build_knmi_station_row(
                sid=ids[i],
                raw_name=names[i],
                lat=float(lats[i]),
                lon=float(lons[i]),
                elev=float(hgts[i]),
                obs_dt=obs_dt,
                t_raw=float(t_arr[i]),
                td_raw=float(td_arr[i]),
                rh_raw=float(u_arr[i]),
                p_raw=float(p_arr[i]),
                ws_raw=ws_i,
                wd_raw=float(dd_arr[i]),
                gust_raw=float(fx_arr[i]),
                vis_v=vis_v,
                precip_mmh=rh_1h_mm,
                precip_1h_mm=rh_1h_mm,
                sw_down_wm2=sw_wm2,
                tcc=tcc_val,
                obs_source="hourly-validated",
                obs_fog_w1=obs_fog_w1,
                obs_rain_w2=obs_rain_w2,
                obs_snow_w3=obs_snow_w3,
                obs_thunder_w5=obs_thunder_w5,
                obs_ice_w6=obs_ice_w6,
                obs_ww_code=obs_ww,
                obs_hail=obs_hail,
                obs_convective_shower=obs_shower,
                reject_calm_dir=True,
            )
            if row is not None:
                rows.append(row)
    return rows


def load_knmi_stations(nc_path: Path | str) -> List[Dict[str, Any]]:
    """Auto-detect NetCDF schema (`hourly-validated` vs `10min`) and parse station dicts."""
    p = Path(nc_path)
    with nc.Dataset(p) as ds:
        is_hv = ("T" in ds.variables and "ta" not in ds.variables) or (
            "hourly-observations-validated" in p.name and "ta" not in ds.variables
        )
    return load_knmi_hourly_validated_stations(p) if is_hv else load_knmi_10min_stations(p)


def load_knmi_hourly_precip_mm(
    valid_time_utc: datetime.datetime,
    cache_dir: Path | str,
    *,
    api_key: Optional[str] = None,
    fetcher: Optional[Callable[[datetime.datetime, Path | str], Optional[Path]]] = None,
) -> Dict[str, float]:
    """Accumulate six 10-minute KNMI gauge intervals (`-50m .. 0m`) into exact 1-hour rainfall [mm]."""
    fetch_fn = fetcher if fetcher is not None else (lambda dt, cdir: fetch_knmi_10min_netcdf(dt, cdir, api_key=api_key))
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    end_dt = _snap_10min_utc(valid_time_utc)
    for back_min in (50, 40, 30, 20, 10, 0):
        t_slice = end_dt - datetime.timedelta(minutes=back_min)
        nc_p = fetch_fn(t_slice, cache_dir)
        if nc_p is None:
            continue
        for r in load_knmi_10min_stations(nc_p):
            rg = r["precip_mmh"]
            if math.isfinite(rg) and rg >= 0.0:
                sid = r["station_id"]
                sums[sid] = sums.get(sid, 0.0) + rg / 6.0
                counts[sid] = counts.get(sid, 0) + 1
    return {sid: val for sid, val in sums.items() if counts.get(sid, 0) >= 3}


def discover_forecast_wx1(
    input_path: Path | str,
) -> Tuple[List[Tuple[float, datetime.datetime, Path]], Optional[Dict[str, Any]]]:
    """Locate sorted `(lead_hours, valid_time_utc, wx1_path)` entries from a run dir or `.wx1` dir."""
    root = Path(input_path)
    if not root.exists():
        raise FileNotFoundError(f"Forecast directory does not exist: {root}")

    manifest: Optional[Dict[str, Any]] = None
    for cand_mf in (root / "manifest.json", root.parent / "manifest.json"):
        if cand_mf.is_file():
            try:
                manifest = json.loads(cand_mf.read_text(encoding="utf-8"))
                break
            except Exception:
                pass

    wx1_files: List[Path] = []
    for sub in (root / "d3", root / "d2", root / "d1", root):
        if sub.is_dir():
            hits = sorted(sub.glob("forecast_*_T+*h.wx1"))
            if hits:
                wx1_files = hits
                break
    if not wx1_files:
        raise FileNotFoundError(f"No forecast_*_T+*h.wx1 files found under {root}")

    items: List[Tuple[float, datetime.datetime, Path]] = []
    for p in wx1_files:
        with wx1.open_dataset(p) as ds:
            raw_lead = ds.attrs.get("lead_hours")
            if raw_lead is not None:
                lead_h = float(raw_lead)
            else:
                m = _LEAD_RE.search(p.name)
                lead_h = float(m.group(1)) if m else 0.0
            raw_v = str(ds.attrs.get("valid_time_utc", "")).strip()
            if not raw_v:
                raise ValueError(f"Missing valid_time_utc attribute in {p}")
            v_dt = _to_utc(datetime.datetime.fromisoformat(raw_v.replace("Z", "+00:00")))
        items.append((lead_h, v_dt, p))

    items.sort(key=lambda it: (it[0], str(it[2])))
    return items, manifest


def _sample_2d_bilinear(
    arr2d: np.ndarray,
    j0: np.ndarray,
    j1: np.ndarray,
    i0: np.ndarray,
    i1: np.ndarray,
    wy: np.ndarray,
    wx: np.ndarray,
) -> np.ndarray:
    a = np.asarray(arr2d, dtype=np.float64)
    return (
        (1.0 - wy) * ((1.0 - wx) * a[j0, i0] + wx * a[j0, i1])
        + wy * ((1.0 - wx) * a[j1, i0] + wx * a[j1, i1])
    )


def sample_wx1_at_stations(
    wx1_path: Path | str,
    stations: Sequence[Dict[str, Any]],
    *,
    prev_total_precip: Optional[np.ndarray] = None,
    prev_precip_parts: Optional[Dict[str, np.ndarray]] = None,
    interval_h: float = 1.0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Sample a forecast `.wx1` at KNMI station coordinates via exact projected bilinear interpolation."""
    with wx1.open_dataset(Path(wx1_path)) as ds:
        domain_name = str(ds.attrs.get("domain_name", "")).strip()
        lat2d = np.asarray(ds["lat"].values, dtype=np.float64)
        lon2d = np.asarray(ds["lon"].values, dtype=np.float64)
        t2m_c = np.asarray(ds["t2m"].values, dtype=np.float64) - 273.15
        mslp_hpa = np.asarray(ds["mslp"].values, dtype=np.float64) * 0.01
        u10 = np.asarray(ds["u10m_geo"].values, dtype=np.float64)
        v10 = np.asarray(ds["v10m_geo"].values, dtype=np.float64)
        rh_pct = np.asarray(ds["rh2m"].values, dtype=np.float64) if "rh2m" in ds else np.full_like(t2m_c, np.nan)
        td2m_c = (
            (np.asarray(ds["td2m"].values, dtype=np.float64) - 273.15)
            if "td2m" in ds
            else np.full_like(t2m_c, np.nan)
        )
        gust10 = np.asarray(ds["gust10m"].values, dtype=np.float64) if "gust10m" in ds else np.full_like(t2m_c, np.nan)
        vis_m = np.asarray(ds["vis_m"].values, dtype=np.float64) if "vis_m" in ds else np.full_like(t2m_c, np.nan)
        sw_down = np.asarray(ds["sw_down"].values, dtype=np.float64) if "sw_down" in ds else np.full_like(t2m_c, np.nan)
        tcc = np.asarray(ds["tcc"].values, dtype=np.float64) if "tcc" in ds else np.full_like(t2m_c, np.nan)
        cape_mu = np.asarray(ds["cape_mu"].values, dtype=np.float64) if "cape_mu" in ds else np.full_like(t2m_c, np.nan)
        cin_sb = np.asarray(ds["cin_sb"].values, dtype=np.float64) if "cin_sb" in ds else np.full_like(t2m_c, np.nan)
        iwp = np.asarray(ds["iwp"].values, dtype=np.float64) if "iwp" in ds else np.full_like(t2m_c, np.nan)
        frzlvl_m = (
            np.asarray(ds["freezing_level_m"].values, dtype=np.float64)
            if "freezing_level_m" in ds
            else np.full_like(t2m_c, np.nan)
        )
        wx1_attrs = {
            str(k): str(ds.attrs[k])
            for k in (
                "model",
                "git_commit",
                "nz",
                "dt_s",
                "physics_every",
                "cy46_physics",
                "pbl_last",
                "enable_ocean_si3",
                "lbc_rim_width",
                "lbc_tau_max_s",
            )
            if k in ds.attrs
        }

        rain_acc = np.asarray(ds["rain_acc"].values, dtype=np.float64) if "rain_acc" in ds else np.zeros_like(t2m_c)
        snow_acc = np.asarray(ds["snow_acc"].values, dtype=np.float64) if "snow_acc" in ds else np.zeros_like(t2m_c)
        graup_acc = (
            np.asarray(ds["graupel_acc"].values, dtype=np.float64) if "graupel_acc" in ds else np.zeros_like(t2m_c)
        )
        tot_precip_mm = np.maximum(0.0, rain_acc + snow_acc + graup_acc)

        if "precip_rate" in ds:
            precip_mmh = np.maximum(0.0, np.asarray(ds["precip_rate"].values, dtype=np.float64) * 3600.0)
        elif prev_total_precip is not None and prev_total_precip.shape == tot_precip_mm.shape:
            precip_mmh = np.maximum(0.0, (tot_precip_mm - prev_total_precip) / max(float(interval_h), 1e-6))
        else:
            precip_mmh = np.zeros_like(tot_precip_mm)

        if prev_total_precip is not None and prev_total_precip.shape == tot_precip_mm.shape:
            precip_step_mm = np.maximum(0.0, tot_precip_mm - prev_total_precip)
        else:
            precip_step_mm = np.zeros_like(tot_precip_mm)

        def _part_delta(cur: np.ndarray, key: str) -> np.ndarray:
            if prev_precip_parts is not None and key in prev_precip_parts:
                prev = prev_precip_parts[key]
                if isinstance(prev, np.ndarray) and prev.shape == cur.shape:
                    return np.maximum(0.0, cur - prev)
            return np.zeros_like(cur)

        rain_step_mm = _part_delta(rain_acc, "rain_acc")
        snow_step_mm = _part_delta(snow_acc, "snow_acc")
        graup_step_mm = _part_delta(graup_acc, "graupel_acc")

        x_coords = np.asarray(ds.coords["x"].values, dtype=np.float64) if "x" in ds.coords else None
        y_coords = np.asarray(ds.coords["y"].values, dtype=np.float64) if "y" in ds.coords else None

    ny, nx = lat2d.shape
    st_lats = np.array([float(s["lat"]) for s in stations], dtype=np.float64)
    st_lons = np.array([float(s["lon"]) for s in stations], dtype=np.float64)

    use_proj = False
    if x_coords is not None and y_coords is not None and len(x_coords) == nx and len(y_coords) == ny and nx > 1 and ny > 1:
        try:
            spec = get_domain_spec(domain_name, nz=16)
            if spec.is_projected and spec.projection is not None:
                sx, sy = spec.projection.latlon_to_xy(st_lats, st_lons)
                use_proj = True
        except Exception:
            use_proj = False

    fields_2d = {
        "mod_t2m_c": t2m_c,
        "mod_td2m_c": td2m_c,
        "mod_rh_pct": rh_pct,
        "mod_mslp_hpa": mslp_hpa,
        "mod_u10_ms": u10,
        "mod_v10_ms": v10,
        "mod_gust10_ms": gust10,
        "mod_vis_m": vis_m,
        "mod_sw_down_wm2": sw_down,
        "mod_tcc": tcc,
        "mod_precip_mmh": precip_mmh,
        "mod_precip_step_mm": precip_step_mm,
        "mod_rain_step_mm": rain_step_mm,
        "mod_snow_step_mm": snow_step_mm,
        "mod_graupel_step_mm": graup_step_mm,
        "mod_cape_mu": cape_mu,
        "mod_cin_sb": cin_sb,
        "mod_iwp": iwp,
        "mod_freezing_level_m": frzlvl_m,
    }

    if use_proj:
        assert x_coords is not None and y_coords is not None
        x_asc = bool(x_coords[-1] > x_coords[0])
        y_asc = bool(y_coords[-1] > y_coords[0])
        xc = x_coords if x_asc else x_coords[::-1]
        yc = y_coords if y_asc else y_coords[::-1]
        inside = (sx >= xc[0]) & (sx <= xc[-1]) & (sy >= yc[0]) & (sy <= yc[-1])

        i0_a = np.clip(np.searchsorted(xc, sx, side="right") - 1, 0, nx - 2)
        i1_a = i0_a + 1
        wx = np.clip((sx - xc[i0_a]) / np.maximum(xc[i1_a] - xc[i0_a], 1e-12), 0.0, 1.0)
        i0, i1 = (i0_a, i1_a) if x_asc else ((nx - 1) - i0_a, (nx - 1) - i1_a)

        j0_a = np.clip(np.searchsorted(yc, sy, side="right") - 1, 0, ny - 2)
        j1_a = j0_a + 1
        wy = np.clip((sy - yc[j0_a]) / np.maximum(yc[j1_a] - yc[j0_a], 1e-12), 0.0, 1.0)
        j0, j1 = (j0_a, j1_a) if y_asc else ((ny - 1) - j0_a, (ny - 1) - j1_a)

        sampled = {fk: _sample_2d_bilinear(arr, j0, j1, i0, i1, wy, wx) for fk, arr in fields_2d.items()}
    else:
        tree = cKDTree(np.column_stack([lat2d.ravel(), lon2d.ravel()]))
        dist, idx = tree.query(np.column_stack([st_lats, st_lons]))
        inside = dist <= 0.15
        jj = idx // nx
        ii = idx % nx
        sampled = {fk: arr[jj, ii] for fk, arr in fields_2d.items()}

    matched: List[Dict[str, Any]] = []
    for k, stn in enumerate(stations):
        if not bool(inside[k]):
            continue
        mu, mv = float(sampled["mod_u10_ms"][k]), float(sampled["mod_v10_ms"][k])
        mws = math.hypot(mu, mv) if (math.isfinite(mu) and math.isfinite(mv)) else float("nan")
        mwd = (math.degrees(math.atan2(-mu, -mv)) % 360.0) if (math.isfinite(mu) and math.isfinite(mv) and mws > 1e-4) else float("nan")
        row = dict(stn)
        for fk, arr in sampled.items():
            row[fk] = float(arr[k])
        row["mod_ws10_ms"] = mws
        row["mod_wd10_deg"] = mwd
        matched.append(row)

    rw = min(16, max(1, ny // 6), max(1, nx // 6))
    rim_mask = np.ones((ny, nx), dtype=bool)
    rim_mask[rw : ny - rw, rw : nx - rw] = False
    nl_mask = (lat2d >= 50.6) & (lat2d <= 53.6) & (lon2d >= 3.2) & (lon2d <= 7.3)
    grid_bundle = {
        "domain_name": domain_name or "tempest_lam",
        "wx1_attrs": wx1_attrs,
        "lat": lat2d,
        "lon": lon2d,
        "t2m_c": t2m_c,
        "mslp_hpa": mslp_hpa,
        "u10_geo": u10,
        "v10_geo": v10,
        "rh_pct": rh_pct,
        "vis_m": vis_m,
        "precip_mmh": precip_mmh,
        "tot_precip_mm": tot_precip_mm,
        "precip_step_mm": precip_step_mm,
        "rain_acc": rain_acc,
        "snow_acc": snow_acc,
        "graupel_acc": graup_acc,
        "rim_max_mmh": float(np.max(precip_mmh[rim_mask])),
        "int_max_mmh": float(np.max(precip_mmh[~rim_mask])),
        "nl_max_mmh": float(np.max(precip_mmh[nl_mask])) if np.any(nl_mask) else float("nan"),
        "rim_tot_max_mm": float(np.max(tot_precip_mm[rim_mask])),
        "int_tot_max_mm": float(np.max(tot_precip_mm[~rim_mask])),
        "n_gt5_all": int(np.sum(precip_mmh > 5.0)),
        "n_gt5_int": int(np.sum(precip_mmh[~rim_mask] > 5.0)),
    }
    return matched, grid_bundle


def compute_continuous_stats(mod: Sequence[float], obs: Sequence[float]) -> Dict[str, Any]:
    """Compute WMO bias (mod - obs), MAE, RMSE, and Pearson correlation across finite pairs."""
    m = np.asarray(mod, dtype=np.float64)
    o = np.asarray(obs, dtype=np.float64)
    valid = np.isfinite(m) & np.isfinite(o)
    n = int(np.sum(valid))
    if n == 0:
        return {
            "n": 0,
            "bias": float("nan"),
            "mae": float("nan"),
            "rmse": float("nan"),
            "corr": float("nan"),
            "obs_mean": float("nan"),
            "mod_mean": float("nan"),
        }
    mv, ov = m[valid], o[valid]
    err = mv - ov
    corr = float("nan")
    if n >= 3 and float(np.std(mv)) > 1e-9 and float(np.std(ov)) > 1e-9:
        corr = float(np.corrcoef(mv, ov)[0, 1])
    return {
        "n": n,
        "bias": float(np.mean(err)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(math.sqrt(float(np.mean(err * err)))),
        "corr": corr,
        "obs_mean": float(np.mean(ov)),
        "mod_mean": float(np.mean(mv)),
    }


def compute_wind_dir_stats_deg(
    mod_deg: Sequence[float],
    obs_deg: Sequence[float],
    obs_ws_ms: Sequence[float],
    min_ws_ms: float = 3.0,
) -> Dict[str, Any]:
    """Compute circular shortest-arc wind direction error [deg] for wind speeds >= `min_ws_ms`."""
    m = np.asarray(mod_deg, dtype=np.float64)
    o = np.asarray(obs_deg, dtype=np.float64)
    w = np.asarray(obs_ws_ms, dtype=np.float64)
    valid = np.isfinite(m) & np.isfinite(o) & np.isfinite(w) & (w >= float(min_ws_ms))
    n = int(np.sum(valid))
    if n == 0:
        return {
            "n": 0,
            "min_ws_ms": float(min_ws_ms),
            "bias_deg": float("nan"),
            "mae_deg": float("nan"),
            "rmse_deg": float("nan"),
        }
    diff = ((m[valid] - o[valid] + 180.0) % 360.0) - 180.0
    return {
        "n": n,
        "min_ws_ms": float(min_ws_ms),
        "bias_deg": float(np.mean(diff)),
        "mae_deg": float(np.mean(np.abs(diff))),
        "rmse_deg": float(math.sqrt(float(np.mean(diff * diff)))),
    }


def compute_fog_contingency(
    mod_vis_m: Sequence[float],
    obs_vis_m: Sequence[float],
    threshold_m: float = 1000.0,
) -> Dict[str, Any]:
    """Compute WMO 2x2 fog (< `threshold_m`) contingency counts, POD, FAR, and CSI."""
    m = np.asarray(mod_vis_m, dtype=np.float64)
    o = np.asarray(obs_vis_m, dtype=np.float64)
    valid = np.isfinite(m) & np.isfinite(o)
    n = int(np.sum(valid))
    if n == 0:
        return {
            "n": 0,
            "obs_fog": 0,
            "mod_fog": 0,
            "hits": 0,
            "misses": 0,
            "false_alarms": 0,
            "correct_negatives": 0,
            "pod": float("nan"),
            "far": float("nan"),
            "csi": float("nan"),
        }
    mf = m[valid] < float(threshold_m)
    of = o[valid] < float(threshold_m)
    hits = int(np.sum(mf & of))
    misses = int(np.sum((~mf) & of))
    fa = int(np.sum(mf & (~of)))
    cn = int(np.sum((~mf) & (~of)))
    pod = float(hits / (hits + misses)) if (hits + misses) > 0 else float("nan")
    far = float(fa / (hits + fa)) if (hits + fa) > 0 else float("nan")
    csi = float(hits / (hits + misses + fa)) if (hits + misses + fa) > 0 else float("nan")
    return {
        "n": n,
        "obs_fog": hits + misses,
        "mod_fog": hits + fa,
        "hits": hits,
        "misses": misses,
        "false_alarms": fa,
        "correct_negatives": cn,
        "pod": pod,
        "far": far,
        "csi": csi,
    }


def compute_binary_contingency(
    mod_yes: Sequence[Optional[bool]],
    obs_yes: Sequence[Optional[bool]],
) -> Dict[str, Any]:
    """Compute 2x2 dichotomous weather-event contingency counts, POD, FAR, CSI, and frequency bias."""
    hits = misses = fa = cn = n = 0
    for m, o in zip(mod_yes, obs_yes):
        if m is None or o is None:
            continue
        mb = bool(m)
        ob = bool(o)
        n += 1
        if mb and ob:
            hits += 1
        elif (not mb) and ob:
            misses += 1
        elif mb and (not ob):
            fa += 1
        else:
            cn += 1
    pod = float(hits / (hits + misses)) if (hits + misses) > 0 else float("nan")
    far = float(fa / (hits + fa)) if (hits + fa) > 0 else float("nan")
    csi = float(hits / (hits + misses + fa)) if (hits + misses + fa) > 0 else float("nan")
    fb = float((hits + fa) / (hits + misses)) if (hits + misses) > 0 else float("nan")
    return {
        "n": n,
        "obs_yes": hits + misses,
        "mod_yes": hits + fa,
        "hits": hits,
        "misses": misses,
        "false_alarms": fa,
        "correct_negatives": cn,
        "pod": pod,
        "far": far,
        "csi": csi,
        "freq_bias": fb,
    }


def _mod_thunder_indicator(row: Dict[str, Any]) -> Optional[bool]:
    """Classical microphysics+instability thunder diagnostic (graupel+CAPE/precip or deep IWP+CAPE)."""
    gr = float(row.get("mod_graupel_step_mm", float("nan")))
    cape = float(row.get("mod_cape_mu", float("nan")))
    iwp = float(row.get("mod_iwp", float("nan")))
    pr = float(row.get("mod_precip_cmp", row.get("mod_precip_step_mm", float("nan"))))
    if not math.isfinite(gr) and not math.isfinite(cape):
        return None
    if math.isfinite(gr) and gr >= 0.05 and ((math.isfinite(cape) and cape >= 50.0) or (math.isfinite(pr) and pr >= 0.5)):
        return True
    if math.isfinite(cape) and cape >= 300.0 and math.isfinite(iwp) and iwp >= 0.25 and math.isfinite(pr) and pr >= 1.5:
        return True
    return False


def compute_fractions_skill_score(
    mod_mm: np.ndarray,
    obs_mm: np.ndarray,
    *,
    dx_km: float = 2.5,
    thresholds_mm: Sequence[float] = (0.1, 0.5, 1.0, 2.0, 5.0),
    scales_km: Sequence[float] = (2.5, 10.0, 25.0, 50.0),
) -> Dict[str, Any]:
    """Roberts & Lean (2008, MWR 136, 78-97) Fractions Skill Score (FSS) across thresholds & scales.

    Uses exact O(N) 2D summed-area tables restricted to valid finite `(mod_mm, obs_mm)` pixels so
    out-of-radar-coverage NaNs never bias neighbourhood fractions.
    """
    m = np.asarray(mod_mm, dtype=np.float64)
    o = np.asarray(obs_mm, dtype=np.float64)
    if m.shape != o.shape or m.ndim != 2:
        raise ValueError(f"Expected matching 2D grids for FSS, got {m.shape} vs {o.shape}")

    valid = np.isfinite(m) & np.isfinite(o)
    n_valid = int(np.sum(valid))
    if n_valid == 0:
        return {"n_valid": 0, "dx_km": float(dx_km), "thresholds": {}}

    ny, nx = m.shape

    def _box_sum(arr2d: np.ndarray, rad: int) -> np.ndarray:
        if rad <= 0:
            return arr2d
        sat = np.zeros((ny + 1, nx + 1), dtype=np.float64)
        sat[1:, 1:] = np.cumsum(np.cumsum(arr2d, axis=0), axis=1)
        j0 = np.clip(np.arange(ny) - rad, 0, ny)
        j1 = np.clip(np.arange(ny) + rad + 1, 0, ny)
        i0 = np.clip(np.arange(nx) - rad, 0, nx)
        i1 = np.clip(np.arange(nx) + rad + 1, 0, nx)
        return sat[j1[:, None], i1[None, :]] - sat[j0[:, None], i1[None, :]] - sat[j1[:, None], i0[None, :]] + sat[j0[:, None], i0[None, :]]

    v_f = valid.astype(np.float64)
    scale_rads = [(float(sc), max(0, int(round(0.5 * (max(1.0, float(sc) / max(float(dx_km), 1e-3)) - 1.0))))) for sc in scales_km]
    v_counts = {sc: _box_sum(v_f, rad) for sc, rad in scale_rads}

    out_th: Dict[str, Any] = {}
    for thr in thresholds_mm:
        thr_f = float(thr)
        obs_bin = np.where(valid & (o >= thr_f), 1.0, 0.0)
        mod_bin = np.where(valid & (m >= thr_f), 1.0, 0.0)
        f_obs = float(np.sum(obs_bin) / n_valid)
        f_mod = float(np.sum(mod_bin) / n_valid)
        fss_uniform = 0.5 + 0.5 * f_obs
        sc_dict: Dict[str, Any] = {}
        for sc, rad in scale_rads:
            vc = v_counts[sc]
            ok = valid & (vc > 0.0)
            o_frac = np.where(ok, _box_sum(obs_bin, rad) / np.maximum(vc, 1.0), 0.0)
            m_frac = np.where(ok, _box_sum(mod_bin, rad) / np.maximum(vc, 1.0), 0.0)
            diff = o_frac[ok] - m_frac[ok]
            fbs = float(np.sum(diff * diff))
            fbs_ref = float(np.sum(o_frac[ok] ** 2 + m_frac[ok] ** 2))
            fss = (1.0 - fbs / fbs_ref) if fbs_ref > 1e-12 else float("nan")
            sc_dict[f"{sc:g}km"] = {
                "scale_km": sc,
                "window_cells": 2 * rad + 1,
                "fbs": fbs,
                "fbs_ref": fbs_ref,
                "fss": fss,
            }
        out_th[f"{thr_f:g}mm"] = {
            "threshold_mm": thr_f,
            "f_obs": f_obs,
            "f_mod": f_mod,
            "freq_bias": (f_mod / f_obs) if f_obs > 1e-12 else float("nan"),
            "fss_uniform": fss_uniform,
            "scales": sc_dict,
        }
    return {"n_valid": n_valid, "dx_km": float(dx_km), "thresholds": out_th}


def _aggregate_fss_records(records: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    valid_recs = [r for r in records if isinstance(r, dict) and r.get("n_valid", 0) > 0 and r.get("thresholds")]
    if not valid_recs:
        return None
    first_th = valid_recs[0]["thresholds"]
    agg_th: Dict[str, Any] = {}
    for th_k, th_v in first_th.items():
        n_tot = sum(int(r["n_valid"]) for r in valid_recs)
        f_obs_avg = sum(float(r["thresholds"][th_k]["f_obs"]) * int(r["n_valid"]) for r in valid_recs) / max(n_tot, 1)
        f_mod_avg = sum(float(r["thresholds"][th_k]["f_mod"]) * int(r["n_valid"]) for r in valid_recs) / max(n_tot, 1)
        sc_agg: Dict[str, Any] = {}
        for sc_k, sc_v in th_v["scales"].items():
            fbs_sum = sum(float(r["thresholds"][th_k]["scales"][sc_k]["fbs"]) for r in valid_recs)
            ref_sum = sum(float(r["thresholds"][th_k]["scales"][sc_k]["fbs_ref"]) for r in valid_recs)
            sc_agg[sc_k] = {
                "scale_km": sc_v["scale_km"],
                "window_cells": sc_v["window_cells"],
                "fbs": fbs_sum,
                "fbs_ref": ref_sum,
                "fss": (1.0 - fbs_sum / ref_sum) if ref_sum > 1e-12 else float("nan"),
            }
        agg_th[th_k] = {
            "threshold_mm": th_v["threshold_mm"],
            "f_obs": f_obs_avg,
            "f_mod": f_mod_avg,
            "freq_bias": (f_mod_avg / f_obs_avg) if f_obs_avg > 1e-12 else float("nan"),
            "fss_uniform": 0.5 + 0.5 * f_obs_avg,
            "scales": sc_agg,
        }
    return {"n_hours": len(valid_recs), "dx_km": valid_recs[0]["dx_km"], "thresholds": agg_th}


def _summarize_matched_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    t2m = compute_continuous_stats([r["mod_t2m_c"] for r in rows], [r["t2m_c"] for r in rows])
    td2m = compute_continuous_stats([r["mod_td2m_c"] for r in rows], [r["td2m_c"] for r in rows])
    rh = compute_continuous_stats([r["mod_rh_pct"] for r in rows], [r["rh_pct"] for r in rows])
    mslp = compute_continuous_stats([r["mod_mslp_hpa"] for r in rows], [r["mslp_hpa"] for r in rows])
    ws10 = compute_continuous_stats([r["mod_ws10_ms"] for r in rows], [r["ws10_ms"] for r in rows])
    gust = compute_continuous_stats([r["mod_gust10_ms"] for r in rows], [r["gust10_ms"] for r in rows])
    precip = compute_continuous_stats(
        [r.get("mod_precip_cmp", float("nan")) for r in rows],
        [r.get("obs_precip_cmp", float("nan")) for r in rows],
    )
    sw_down = compute_continuous_stats(
        [r.get("mod_sw_down_wm2", float("nan")) for r in rows],
        [r.get("sw_down_wm2", float("nan")) for r in rows],
    )
    tcc = compute_continuous_stats(
        [r.get("mod_tcc", float("nan")) for r in rows],
        [r.get("tcc", float("nan")) for r in rows],
    )
    wdir = compute_wind_dir_stats_deg(
        [r["mod_wd10_deg"] for r in rows],
        [r["wd10_deg"] for r in rows],
        [r["ws10_ms"] for r in rows],
    )
    fog = compute_fog_contingency([r["mod_vis_m"] for r in rows], [r["vis_m"] for r in rows])

    def _ge(val: Any, thr: float) -> Optional[bool]:
        fv = float(val) if val is not None else float("nan")
        return (fv >= thr) if math.isfinite(fv) else None

    def _obs_flag(r: Dict[str, Any], key: str, *, need_positive_lead: bool = False) -> Optional[bool]:
        if need_positive_lead and float(r.get("lead_hours", 0.0)) <= 0.0:
            return None
        v = r.get(key)
        return (int(v) == 1) if v is not None else None

    def _mod_species(r: Dict[str, Any], key: str, thr: float) -> Optional[bool]:
        if float(r.get("lead_hours", 0.0)) <= 0.0:
            return None
        return _ge(r.get(key), thr)

    def _mod_freezing_rain(r: Dict[str, Any]) -> Optional[bool]:
        if float(r.get("lead_hours", 0.0)) <= 0.0:
            return None
        mt = float(r.get("mod_t2m_c", float("nan")))
        mr = float(r.get("mod_rain_step_mm", float("nan")))
        if not (math.isfinite(mt) and math.isfinite(mr)):
            return None
        return (mt <= 0.5) and (mr >= 0.05)

    events = {
        "rain_0p1mm": compute_binary_contingency(
            [_ge(r.get("mod_precip_cmp"), 0.1) for r in rows],
            [_ge(r.get("obs_precip_cmp"), 0.1) for r in rows],
        ),
        "rain_2mm": compute_binary_contingency(
            [_ge(r.get("mod_precip_cmp"), 2.0) for r in rows],
            [_ge(r.get("obs_precip_cmp"), 2.0) for r in rows],
        ),
        "gust_15ms": compute_binary_contingency(
            [_ge(r.get("mod_gust10_ms"), 15.0) for r in rows],
            [_ge(r.get("gust10_ms"), 15.0) for r in rows],
        ),
        "gust_20ms": compute_binary_contingency(
            [_ge(r.get("mod_gust10_ms"), 20.0) for r in rows],
            [_ge(r.get("gust10_ms"), 20.0) for r in rows],
        ),
        "snow": compute_binary_contingency(
            [_mod_species(r, "mod_snow_step_mm", 0.05) for r in rows],
            [_obs_flag(r, "obs_snow_w3", need_positive_lead=True) for r in rows],
        ),
        "hail_graupel": compute_binary_contingency(
            [_mod_species(r, "mod_graupel_step_mm", 0.05) for r in rows],
            [_obs_flag(r, "obs_hail", need_positive_lead=True) for r in rows],
        ),
        "thunder": compute_binary_contingency(
            [(_mod_thunder_indicator(r) if float(r.get("lead_hours", 0.0)) > 0.0 else None) for r in rows],
            [_obs_flag(r, "obs_thunder_w5", need_positive_lead=True) for r in rows],
        ),
        "ice_freezing_rain": compute_binary_contingency(
            [_mod_freezing_rain(r) for r in rows],
            [_obs_flag(r, "obs_ice_w6", need_positive_lead=True) for r in rows],
        ),
    }

    def _nanmax_vals(key: str) -> float:
        vals = [float(r[key]) for r in rows if r.get(key) is not None and math.isfinite(float(r[key]))]
        return max(vals) if vals else float("nan")

    convection_diag = {
        "mod_cape_mu_max": _nanmax_vals("mod_cape_mu"),
        "mod_graupel_max_mm": _nanmax_vals("mod_graupel_step_mm"),
        "mod_snow_max_mm": _nanmax_vals("mod_snow_step_mm"),
        "obs_thunder_hrs": sum(1 for r in rows if r.get("obs_thunder_w5") == 1),
        "obs_snow_hrs": sum(1 for r in rows if r.get("obs_snow_w3") == 1),
        "obs_hail_hrs": sum(1 for r in rows if r.get("obs_hail") == 1),
        "obs_ice_hrs": sum(1 for r in rows if r.get("obs_ice_w6") == 1),
        "obs_shower_hrs": sum(1 for r in rows if r.get("obs_convective_shower") == 1),
    }

    n_station_hours = len(rows)
    unique_ids = {str(r["station_id"]) for r in rows if r.get("station_id")}
    n_unique_stations = len(unique_ids) if unique_ids else n_station_hours
    return {
        "n_stations": n_unique_stations,
        "n_unique_stations": n_unique_stations,
        "n_station_hours": n_station_hours,
        "t2m": t2m,
        "td2m": td2m,
        "rh": rh,
        "mslp": mslp,
        "ws10": ws10,
        "gust10": gust,
        "wdir": wdir,
        "fog": fog,
        "precip": precip,
        "sw_down": sw_down,
        "tcc": tcc,
        "events": events,
        "convection_diag": convection_diag,
    }


def _auto_discover_ifs_dir(run_base: Path, manifest: Optional[Dict[str, Any]]) -> Optional[Path]:
    """Locate cached ECMWF IFS GRIB2 directory matching a run's `manifest.json`."""
    candidates: List[Path] = []
    lbc = manifest.get("lbc") if isinstance(manifest, dict) else None
    if isinstance(lbc, dict) and lbc.get("ecmwf_cycle_utc"):
        try:
            cyc_dt = _to_utc(datetime.datetime.fromisoformat(str(lbc["ecmwf_cycle_utc"]).replace("Z", "+00:00")))
            cyc_tag = f"{cyc_dt:%Y%m%d_%H}Z"
            resol = str(lbc.get("resol") or "").strip()
            tags = [f"{cyc_tag}_{resol}"] if (resol and resol != "auto") else []
            for base_cache in (
                run_base.parent / "ecmwf_lbc_cache",
                run_base.parent.parent / "ecmwf_lbc_cache",
                Path("runs") / "ecmwf_lbc_cache",
            ):
                for tag in tags:
                    candidates.append(base_cache / tag)
                if base_cache.is_dir():
                    for p in sorted(base_cache.glob(f"{cyc_tag}*")):
                        if p.is_dir():
                            candidates.append(p)
            candidates.append(run_base / "raw_ecmwf")
        except Exception:
            pass
    for cand in candidates:
        if cand.is_dir() and any(cand.glob("*_sfc_*.grib2")):
            return cand
    return None


def discover_ifs_grib_by_valid_time(ifs_dir: Path | str) -> Dict[str, Path]:
    """Index `*_sfc_*.grib2` files by UTC valid-time key `YYYY-MM-DDTHH:MMZ`."""
    root = Path(ifs_dir)
    if not root.is_dir():
        return {}
    out: Dict[str, Path] = {}
    for p in sorted(root.glob("*_sfc_*.grib2")):
        v_dt: Optional[datetime.datetime] = None
        m = _IFS_SFC_RE.search(p.name)
        if m:
            ymd, hh, step_s = m.groups()
            cyc_dt = datetime.datetime.strptime(f"{ymd}{hh}", "%Y%m%d%H").replace(tzinfo=datetime.timezone.utc)
            v_dt = cyc_dt + datetime.timedelta(hours=int(step_s))
        else:
            try:
                from tempest_v2.ic_builder import _read_cfgrib_var_map

                vmap = _read_cfgrib_var_map(p, use_cache=False)
                for da, _la, _lo in vmap.values():
                    if "valid_time" in da.coords:
                        ts = int(np.datetime64(da.coords["valid_time"].values, "s").astype(np.int64))
                        v_dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
                        break
            except Exception:
                v_dt = None
        if v_dt is not None:
            out[v_dt.strftime("%Y-%m-%dT%H:%MZ")] = p
    return out


def _bilinear_sample_latlon_2d(
    field: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    st_lats: np.ndarray,
    st_lons: np.ndarray,
) -> np.ndarray:
    """Bilinearly interpolate a regular lat-lon 2D field at station coordinates."""
    arr = np.asarray(field, dtype=np.float64)
    while arr.ndim > 2:
        arr = arr[0]
    la = np.asarray(lats, dtype=np.float64)
    lo = np.asarray(lons, dtype=np.float64)
    if la.size < 2 or lo.size < 2:
        return np.full(len(st_lats), np.nan, dtype=np.float64)
    if la[-1] < la[0]:
        la = la[::-1]
        arr = arr[::-1, :]
    if lo[-1] < lo[0]:
        lo = lo[::-1]
        arr = arr[:, ::-1]

    slat = np.asarray(st_lats, dtype=np.float64)
    slon_raw = np.asarray(st_lons, dtype=np.float64)
    if float(np.nanmax(lo)) > 180.0:
        slon = np.mod(slon_raw, 360.0)
    else:
        slon = ((slon_raw + 180.0) % 360.0) - 180.0

    inside = (
        np.isfinite(slat)
        & np.isfinite(slon)
        & (slat >= la[0])
        & (slat <= la[-1])
        & (slon >= lo[0])
        & (slon <= lo[-1])
    )
    ny, nx = arr.shape
    slat_c = np.clip(slat, la[0], la[-1])
    slon_c = np.clip(slon, lo[0], lo[-1])
    j0 = np.clip(np.searchsorted(la, slat_c, side="right") - 1, 0, ny - 2)
    j1 = j0 + 1
    wy = np.clip((slat_c - la[j0]) / np.maximum(la[j1] - la[j0], 1e-12), 0.0, 1.0)
    i0 = np.clip(np.searchsorted(lo, slon_c, side="right") - 1, 0, nx - 2)
    i1 = i0 + 1
    wx = np.clip((slon_c - lo[i0]) / np.maximum(lo[i1] - lo[i0], 1e-12), 0.0, 1.0)
    vals = _sample_2d_bilinear(arr, j0, j1, i0, i1, wy, wx)
    return np.where(inside, vals, np.nan)


def sample_ifs_grib_at_stations(
    grib_path: Path | str,
    stations: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Sample ECMWF IFS surface GRIB2 variables at KNMI station coordinates."""
    if not stations:
        return []
    from tempest_v2.ic_builder import _read_cfgrib_var_map

    vmap = _read_cfgrib_var_map(Path(grib_path), use_cache=False)
    st_lats = np.array([float(s["lat"]) for s in stations], dtype=np.float64)
    st_lons = np.array([float(s["lon"]) for s in stations], dtype=np.float64)
    n = len(stations)

    def _sample_key(aliases: Tuple[str, ...]) -> np.ndarray:
        for k in aliases:
            if k in vmap:
                da, la, lo = vmap[k]
                return _bilinear_sample_latlon_2d(np.asarray(da.values, dtype=np.float64), la, lo, st_lats, st_lons)
        return np.full(n, np.nan, dtype=np.float64)

    t2m_c = _sample_key(("t2m", "2t")) - 273.15
    td2m_c = _sample_key(("d2m", "2d")) - 273.15
    mslp_hpa = _sample_key(("msl", "mslp")) * 0.01
    u10 = _sample_key(("u10", "10u"))
    v10 = _sample_key(("v10", "10v"))

    rh_pct = np.full(n, np.nan, dtype=np.float64)
    valid_td = np.isfinite(t2m_c) & np.isfinite(td2m_c)
    if np.any(valid_td):
        tc = t2m_c[valid_td]
        td = np.minimum(td2m_c[valid_td], tc)
        rh_pct[valid_td] = np.clip(
            100.0 * np.exp((17.625 * td) / (243.04 + td) - (17.625 * tc) / (243.04 + tc)),
            0.0,
            100.0,
        )

    matched: List[Dict[str, Any]] = []
    for k, stn in enumerate(stations):
        mu, mv = float(u10[k]), float(v10[k])
        has_wind = math.isfinite(mu) and math.isfinite(mv)
        mws = math.hypot(mu, mv) if has_wind else float("nan")
        mwd = (math.degrees(math.atan2(-mu, -mv)) % 360.0) if (has_wind and mws > 1e-4) else float("nan")
        row = dict(stn)
        row.update(
            {
                "mod_t2m_c": float(t2m_c[k]),
                "mod_td2m_c": float(td2m_c[k]),
                "mod_rh_pct": float(rh_pct[k]),
                "mod_mslp_hpa": float(mslp_hpa[k]),
                "mod_u10_ms": mu,
                "mod_v10_ms": mv,
                "mod_ws10_ms": mws,
                "mod_wd10_deg": mwd,
                "mod_gust10_ms": float("nan"),
                "mod_vis_m": float("nan"),
                "mod_sw_down_wm2": float("nan"),
                "mod_tcc": float("nan"),
                "mod_precip_mmh": float("nan"),
                "mod_precip_step_mm": float("nan"),
                "obs_precip_cmp": float("nan"),
                "mod_precip_cmp": float("nan"),
            }
        )
        matched.append(row)
    return matched


def compute_skill_score_vs_ifs(tempest_stats: Dict[str, Any], ifs_stats: Dict[str, Any]) -> Dict[str, Any]:
    """Compute RMSE Skill Score (SS = 1 - RMSE_Tempest / RMSE_Ref; >0 means Tempest beats IFS/ERA5)."""
    out: Dict[str, Any] = {}
    for key in ("t2m", "td2m", "rh", "mslp", "ws10", "gust10", "sw_down", "tcc", "precip"):
        tb = tempest_stats.get(key, {})
        ib = ifs_stats.get(key, {})
        rt = float(tb.get("rmse", float("nan")))
        ri = float(ib.get("rmse", float("nan")))
        bt = float(tb.get("bias", float("nan")))
        bi = float(ib.get("bias", float("nan")))
        ss = (1.0 - rt / ri) if (math.isfinite(rt) and math.isfinite(ri) and ri > 1e-12) else float("nan")
        out[key] = {
            "n_tempest": int(tb.get("n", 0)),
            "n_ifs": int(ib.get("n", 0)),
            "skill_score": ss,
            "delta_rmse": (rt - ri) if (math.isfinite(rt) and math.isfinite(ri)) else float("nan"),
            "delta_abs_bias": (abs(bt) - abs(bi)) if (math.isfinite(bt) and math.isfinite(bi)) else float("nan"),
        }
    wt = tempest_stats.get("wdir", {})
    wi = ifs_stats.get("wdir", {})
    mt = float(wt.get("mae_deg", float("nan")))
    mi = float(wi.get("mae_deg", float("nan")))
    out["wdir"] = {
        "n_tempest": int(wt.get("n", 0)),
        "n_ifs": int(wi.get("n", 0)),
        "skill_score": (1.0 - mt / mi) if (math.isfinite(mt) and math.isfinite(mi) and mi > 1e-12) else float("nan"),
        "delta_mae_deg": (mt - mi) if (math.isfinite(mt) and math.isfinite(mi)) else float("nan"),
    }
    return out


def _align_rows_with_ifs(
    tempest_rows: Sequence[Dict[str, Any]],
    ifs_rows: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Align Tempest & IFS/ERA5 station rows per `(valid_time_utc, station_id)` and mask missing ref vars."""
    ifs_idx = {
        (str(r.get("valid_time_utc", "")), str(r.get("station_id", ""))): r
        for r in ifs_rows
    }
    t_aligned: List[Dict[str, Any]] = []
    i_aligned: List[Dict[str, Any]] = []
    mask_keys = (
        "mod_t2m_c",
        "mod_td2m_c",
        "mod_rh_pct",
        "mod_mslp_hpa",
        "mod_ws10_ms",
        "mod_wd10_deg",
        "mod_gust10_ms",
        "mod_sw_down_wm2",
        "mod_tcc",
        "mod_precip_cmp",
    )
    for tr in tempest_rows:
        key = (str(tr.get("valid_time_utc", "")), str(tr.get("station_id", "")))
        ir = ifs_idx.get(key)
        if ir is None:
            continue
        tc = dict(tr)
        ic = dict(ir)
        for copy_k in ("lead_hours", "obs_precip_cmp", "precip_unit"):
            if copy_k in tr and (copy_k not in ic or (isinstance(ic[copy_k], float) and not math.isfinite(ic[copy_k]))):
                ic[copy_k] = tr[copy_k]
        for mod_k in mask_keys:
            raw_iv = ic.get(mod_k, float("nan"))
            iv = float(raw_iv) if raw_iv is not None else float("nan")
            if not math.isfinite(iv):
                tc[mod_k] = float("nan")
        t_aligned.append(tc)
        i_aligned.append(ic)
    return t_aligned, i_aligned


def _select_era5_time_slice(era5_ds: xr.Dataset, valid_dt: Optional[datetime.datetime] = None) -> xr.Dataset:
    if "time" not in era5_ds.dims:
        return era5_ds
    if era5_ds.sizes["time"] <= 1 or valid_dt is None:
        return era5_ds.isel(time=0)
    t_target = np.datetime64(_to_utc(valid_dt).replace(tzinfo=None), "s")
    return era5_ds.sel(time=t_target, method="nearest")


def sample_era5_at_stations(
    era5_ds: xr.Dataset,
    stations: Sequence[Dict[str, Any]],
    *,
    valid_dt: Optional[datetime.datetime] = None,
) -> List[Dict[str, Any]]:
    """Sample ARCO-ERA5 0.25-deg surface fields bilinearly at KNMI station coordinates."""
    if not stations:
        return []
    ds_t = _select_era5_time_slice(era5_ds, valid_dt)
    lat_k = next(c for c in ("latitude", "lat") if c in ds_t.coords or c in ds_t.variables)
    lon_k = next(c for c in ("longitude", "lon") if c in ds_t.coords or c in ds_t.variables)
    lats = np.asarray(ds_t[lat_k].values, dtype=np.float64)
    lons = np.asarray(ds_t[lon_k].values, dtype=np.float64)
    st_lats = np.array([float(s["lat"]) for s in stations], dtype=np.float64)
    st_lons = np.array([float(s["lon"]) for s in stations], dtype=np.float64)
    n = len(stations)

    def _sample_key(aliases: Tuple[str, ...]) -> np.ndarray:
        for k in aliases:
            if k in ds_t.data_vars:
                arr = np.asarray(ds_t[k].values, dtype=np.float64)
                while arr.ndim > 2:
                    arr = arr[0]
                return _bilinear_sample_latlon_2d(arr, lats, lons, st_lats, st_lons)
        return np.full(n, np.nan, dtype=np.float64)

    t2m_c = _sample_key(("2m_temperature", "t2m")) - 273.15
    td2m_c = _sample_key(("2m_dewpoint_temperature", "d2m")) - 273.15
    mslp_hpa = _sample_key(("mean_sea_level_pressure", "msl")) * 0.01
    u10 = _sample_key(("10m_u_component_of_wind", "u10"))
    v10 = _sample_key(("10m_v_component_of_wind", "v10"))
    gust10 = _sample_key(("10m_wind_gust_since_previous_post_processing", "fg10", "instantaneous_10m_wind_gust", "i10fg"))
    tcc_raw = _sample_key(("total_cloud_cover", "tcc"))
    tcc = np.where(np.isfinite(tcc_raw), np.clip(tcc_raw, 0.0, 1.0), np.nan)
    ssrd_raw = _sample_key(("surface_solar_radiation_downwards", "ssrd"))
    sw_down = np.where(np.isfinite(ssrd_raw), np.maximum(0.0, ssrd_raw / 3600.0), np.nan)
    tp_raw = _sample_key(("total_precipitation", "tp"))
    tp_mm = np.where(np.isfinite(tp_raw), np.maximum(0.0, tp_raw * 1000.0), np.nan)

    rh_pct = np.full(n, np.nan, dtype=np.float64)
    valid_td = np.isfinite(t2m_c) & np.isfinite(td2m_c)
    if np.any(valid_td):
        tc = t2m_c[valid_td]
        td = np.minimum(td2m_c[valid_td], tc)
        rh_pct[valid_td] = np.clip(
            100.0 * np.exp((17.625 * td) / (243.04 + td) - (17.625 * tc) / (243.04 + tc)),
            0.0,
            100.0,
        )

    matched: List[Dict[str, Any]] = []
    for k, stn in enumerate(stations):
        mu, mv = float(u10[k]), float(v10[k])
        has_wind = math.isfinite(mu) and math.isfinite(mv)
        mws = math.hypot(mu, mv) if has_wind else float("nan")
        mwd = (math.degrees(math.atan2(-mu, -mv)) % 360.0) if (has_wind and mws > 1e-4) else float("nan")
        row = dict(stn)
        row.update(
            {
                "mod_t2m_c": float(t2m_c[k]),
                "mod_td2m_c": float(td2m_c[k]),
                "mod_rh_pct": float(rh_pct[k]),
                "mod_mslp_hpa": float(mslp_hpa[k]),
                "mod_u10_ms": mu,
                "mod_v10_ms": mv,
                "mod_ws10_ms": mws,
                "mod_wd10_deg": mwd,
                "mod_gust10_ms": float(gust10[k]),
                "mod_vis_m": float("nan"),
                "mod_sw_down_wm2": float(sw_down[k]),
                "mod_tcc": float(tcc[k]),
                "mod_precip_mmh": float(tp_mm[k]),
                "mod_precip_step_mm": float(tp_mm[k]),
                "obs_precip_cmp": float("nan"),
                "mod_precip_cmp": float(tp_mm[k]),
            }
        )
        matched.append(row)
    return matched


def compute_upper_air_vs_era5(
    wx1_path: Path | str,
    era5_ds: xr.Dataset,
    *,
    rim_width: int = 16,
    valid_dt: Optional[datetime.datetime] = None,
) -> Dict[str, Any]:
    """Compare interior Tempest `.wx1` synoptic & 850/500 hPa fields directly against ARCO-ERA5 0.25-deg."""
    with wx1.open_dataset(Path(wx1_path)) as ds_m:
        lat2d = np.asarray(ds_m["lat"].values, dtype=np.float64)
        lon2d = np.asarray(ds_m["lon"].values, dtype=np.float64)
        ny, nx = lat2d.shape
        rw = min(int(rim_width), max(0, ny // 6), max(0, nx // 6))
        if (ny - 2 * rw) < 2 or (nx - 2 * rw) < 2:
            rw = 0
        sy, sx = slice(rw, ny - rw), slice(rw, nx - rw)

        lat_int = lat2d[sy, sx].ravel()
        lon_int = lon2d[sy, sx].ravel()
        mslp_int = np.asarray(ds_m["mslp"].values, dtype=np.float64)[sy, sx].ravel() * 0.01
        t2m_int = np.asarray(ds_m["t2m"].values, dtype=np.float64)[sy, sx].ravel() - 273.15

        plev_k = next((c for c in ("plev", "level") if c in ds_m.coords or c in ds_m.variables), None)
        t850_m = gh500_m = w850_m = w500_m = q850_m = None
        if plev_k is not None:
            plev_raw = np.asarray(ds_m[plev_k].values, dtype=np.float64)
            plev_pa = plev_raw * 100.0 if (plev_raw.size > 0 and float(np.max(plev_raw)) < 2000.0) else plev_raw
            i850 = int(np.argmin(np.abs(plev_pa - 85_000.0)))
            i500 = int(np.argmin(np.abs(plev_pa - 50_000.0)))
            has850 = abs(float(plev_pa[i850]) - 85_000.0) <= 2500.0
            has500 = abs(float(plev_pa[i500]) - 50_000.0) <= 2500.0

            def _pl2(name: str, kidx: int) -> Optional[np.ndarray]:
                if name not in ds_m:
                    return None
                arr = np.asarray(ds_m[name].values, dtype=np.float64)
                while arr.ndim > 3:
                    arr = arr[0]
                return arr[kidx, sy, sx].ravel()

            if has850:
                t850_m = _pl2("t_pl", i850)
                u850, v850 = _pl2("u_pl", i850), _pl2("v_pl", i850)
                if u850 is not None and v850 is not None:
                    w850_m = np.hypot(u850, v850)
                q850_raw = _pl2("q_pl", i850)
                if q850_raw is not None:
                    q850_m = q850_raw * 1000.0
            if has500:
                gh500_m = _pl2("gh_pl", i500)
                if gh500_m is None:
                    z500_raw = _pl2("z_pl", i500)
                    if z500_raw is not None:
                        gh500_m = z500_raw / 9.80665
                u500, v500 = _pl2("u_pl", i500), _pl2("v_pl", i500)
                if u500 is not None and v500 is not None:
                    w500_m = np.hypot(u500, v500)

    ds_e = _select_era5_time_slice(era5_ds, valid_dt)
    elat_k = next(c for c in ("latitude", "lat") if c in ds_e.coords or c in ds_e.variables)
    elon_k = next(c for c in ("longitude", "lon") if c in ds_e.coords or c in ds_e.variables)
    e_lats = np.asarray(ds_e[elat_k].values, dtype=np.float64)
    e_lons = np.asarray(ds_e[elon_k].values, dtype=np.float64)

    def _regrid_e2(aliases: Tuple[str, ...]) -> Optional[np.ndarray]:
        for ak in aliases:
            if ak in ds_e.data_vars:
                arr = np.asarray(ds_e[ak].values, dtype=np.float64)
                while arr.ndim > 2:
                    arr = arr[0]
                return _bilinear_sample_latlon_2d(arr, e_lats, e_lons, lat_int, lon_int)
        return None

    e_mslp_raw = _regrid_e2(("mean_sea_level_pressure", "msl"))
    e_t2m_raw = _regrid_e2(("2m_temperature", "t2m"))
    empty_stat = compute_continuous_stats([], [])
    out: Dict[str, Any] = {
        "n_points": int(lat_int.size),
        "mslp_hpa": compute_continuous_stats(mslp_int, e_mslp_raw * 0.01) if e_mslp_raw is not None else dict(empty_stat),
        "t2m_c": compute_continuous_stats(t2m_int, e_t2m_raw - 273.15) if e_t2m_raw is not None else dict(empty_stat),
        "t850_k": dict(empty_stat),
        "gh500_m": dict(empty_stat),
        "wspd850_ms": dict(empty_stat),
        "wspd500_ms": dict(empty_stat),
        "q850_gkg": dict(empty_stat),
    }

    elvl_k = next((c for c in ("level", "plev") if c in ds_e.coords or c in ds_e.variables), None)
    if elvl_k is not None:
        elvl_raw = np.asarray(ds_e[elvl_k].values, dtype=np.float64)
        elvl_hpa = elvl_raw * 0.01 if (elvl_raw.size > 0 and float(np.max(elvl_raw)) > 2000.0) else elvl_raw
        ei850 = int(np.argmin(np.abs(elvl_hpa - 850.0)))
        ei500 = int(np.argmin(np.abs(elvl_hpa - 500.0)))
        ehas850 = abs(float(elvl_hpa[ei850]) - 850.0) <= 25.0
        ehas500 = abs(float(elvl_hpa[ei500]) - 500.0) <= 25.0

        def _regrid_e3(aliases: Tuple[str, ...], kidx: int) -> Optional[np.ndarray]:
            for ak in aliases:
                if ak in ds_e.data_vars:
                    arr = np.asarray(ds_e[ak].values, dtype=np.float64)
                    while arr.ndim > 3:
                        arr = arr[0]
                    return _bilinear_sample_latlon_2d(arr[kidx], e_lats, e_lons, lat_int, lon_int)
            return None

        if ehas850:
            if t850_m is not None:
                e_t850 = _regrid_e3(("temperature", "t"), ei850)
                if e_t850 is not None:
                    out["t850_k"] = compute_continuous_stats(t850_m, e_t850)
            if w850_m is not None:
                eu850 = _regrid_e3(("u_component_of_wind", "u"), ei850)
                ev850 = _regrid_e3(("v_component_of_wind", "v"), ei850)
                if eu850 is not None and ev850 is not None:
                    out["wspd850_ms"] = compute_continuous_stats(w850_m, np.hypot(eu850, ev850))
            if q850_m is not None:
                e_q850 = _regrid_e3(("specific_humidity", "q"), ei850)
                if e_q850 is not None:
                    out["q850_gkg"] = compute_continuous_stats(q850_m, e_q850 * 1000.0)
        if ehas500:
            if gh500_m is not None:
                e_z500 = _regrid_e3(("geopotential", "z"), ei500)
                if e_z500 is not None:
                    out["gh500_m"] = compute_continuous_stats(gh500_m, e_z500 / 9.80665)
            if w500_m is not None:
                eu500 = _regrid_e3(("u_component_of_wind", "u"), ei500)
                ev500 = _regrid_e3(("v_component_of_wind", "v"), ei500)
                if eu500 is not None and ev500 is not None:
                    out["wspd500_ms"] = compute_continuous_stats(w500_m, np.hypot(eu500, ev500))
    return out


_ERA5_VERIFY_3D_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("geopotential", "z"),
    ("temperature", "t"),
    ("specific_humidity", "q"),
    ("u_component_of_wind", "u"),
    ("v_component_of_wind", "v"),
)
_ERA5_VERIFY_2D_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("mean_sea_level_pressure", "msl"),
    ("2m_temperature", "t2m"),
    ("2m_dewpoint_temperature", "d2m"),
    ("10m_u_component_of_wind", "u10"),
    ("10m_v_component_of_wind", "v10"),
    ("10m_wind_gust_since_previous_post_processing", "fg10"),
    ("total_cloud_cover", "tcc"),
    ("surface_solar_radiation_downwards", "ssrd"),
    ("total_precipitation", "tp"),
)


def _load_era5_verify_slice(
    valid_dt: datetime.datetime,
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    era5_cache_dir: Optional[Path | str],
    opened_era5_ds: Optional[xr.Dataset],
    *,
    allow_cache_write: bool = True,
) -> Tuple[Optional[xr.Dataset], Optional[xr.Dataset]]:
    """Load or lazily slice a lightweight ARCO-ERA5 verification subset (`sfc + 850/500 hPa`)."""
    v_dt = _to_utc(valid_dt)
    cdir = Path(era5_cache_dir) if era5_cache_dir is not None else None
    verify_cache = (cdir / f"era5_verify_0p25_{v_dt:%Y%m%d_%H}Z.wx1") if cdir is not None else None
    if verify_cache is not None and verify_cache.is_file() and verify_cache.stat().st_size > 1024:
        try:
            with wx1.open_dataset(verify_cache) as ds_c:
                return ds_c.load(), opened_era5_ds
        except Exception:
            pass

    ds_src = opened_era5_ds
    if ds_src is None:
        from tempest_v2.ingest_era5 import open_arco_era5_dataset

        ds_src = open_arco_era5_dataset()

    from tempest_v2.ingest_era5 import _extract_lon_wrapped_subset

    pairs: List[Tuple[str, str]] = []
    for long_k, short_k in _ERA5_VERIFY_3D_PAIRS + _ERA5_VERIFY_2D_PAIRS:
        if long_k in ds_src.data_vars:
            pairs.append((long_k, short_k))
        elif short_k in ds_src.data_vars:
            pairs.append((short_k, short_k))
    if not pairs:
        return None, ds_src

    ds_sel = ds_src[[src for src, _ in pairs]]
    if "time" in ds_sel.dims:
        t_target = np.datetime64(v_dt.replace(tzinfo=None), "s")
        ds_t = ds_sel.sel(time=t_target, method="nearest")
        sel_sec = int(np.datetime64(ds_t["time"].values, "ns").astype("datetime64[s]").astype(np.int64))
        if abs(sel_sec - int(v_dt.timestamp())) > 1800:
            raise ValueError(f"ERA5 archive does not contain {v_dt.isoformat()}")
    else:
        ds_t = ds_sel

    lat_min = max(-90.0, float(np.min(lat2d)) - 2.0)
    lat_max = min(90.0, float(np.max(lat2d)) + 2.0)
    lon_180 = ((np.asarray(lon2d, dtype=np.float64) + 180.0) % 360.0) - 180.0
    lon_min = max(-180.0, float(np.min(lon_180)) - 2.0)
    lon_max = min(180.0, float(np.max(lon_180)) + 2.0)
    ds_sub = _extract_lon_wrapped_subset(ds_t, lat_min, lat_max, lon_min, lon_max)

    elvl_k = next((c for c in ("level", "plev") if c in ds_sub.coords), None)
    if elvl_k is not None:
        lvls = np.asarray(ds_sub[elvl_k].values, dtype=np.float64)
        pick = [lvls[int(np.argmin(np.abs(lvls - tgt)))] for tgt in (850.0, 500.0) if np.min(np.abs(lvls - tgt)) <= 25.0]
        if pick:
            ds_sub = ds_sub.sel({elvl_k: sorted(set(pick), reverse=True)})

    ds_loaded = ds_sub.load()
    rename_map = {src: short for src, short in pairs if src != short and src in ds_loaded.data_vars}
    if rename_map:
        ds_loaded = ds_loaded.rename(rename_map)

    if allow_cache_write and verify_cache is not None:
        try:
            verify_cache.parent.mkdir(parents=True, exist_ok=True)
            wx1.write_dataset(
                verify_cache,
                ds_loaded,
                dataset_id=f"era5_verify_{v_dt:%Y%m%d_%H}Z",
                codec="zstd_shuffle",
                workers=2,
            )
        except Exception as exc:
            logger.debug("Skipping ERA5 verify cache write (%s): %s", verify_cache, exc)
    return ds_loaded, ds_src


def _aggregate_upper_air_records(records: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    valid = [r for r in records if isinstance(r, dict) and r]
    if not valid:
        return None
    out: Dict[str, Any] = {"n_hours": len(valid)}
    keys = ("mslp_hpa", "t2m_c", "t850_k", "gh500_m", "wspd850_ms", "wspd500_ms", "q850_gkg")
    for k in keys:
        biases = [float(r[k]["bias"]) for r in valid if k in r and math.isfinite(float(r[k].get("bias", float("nan"))))]
        rmses = [float(r[k]["rmse"]) for r in valid if k in r and math.isfinite(float(r[k].get("rmse", float("nan"))))]
        ns = sum(int(r[k].get("n", 0)) for r in valid if k in r)
        out[k] = {
            "n_points": ns,
            "n_hours": len(rmses),
            "bias": float(np.mean(biases)) if biases else float("nan"),
            "rmse": math.sqrt(float(np.mean([x * x for x in rmses]))) if rmses else float("nan"),
        }
    return out


def plot_hourly_verification_map(
    out_png: Path | str,
    lead_h: float,
    valid_dt: datetime.datetime,
    grid: Dict[str, Any],
    matched_rows: Sequence[Dict[str, Any]],
    hour_stats: Dict[str, Any],
    *,
    land_only: bool = False,
    obs_label: str = "KNMI 10-min",
) -> Path:
    """Render a 2x2 Netherlands verification PNG comparing model fields against KNMI in-situ obs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in matched_rows if (not land_only or not r["is_offshore"])]
    out_p = Path(out_png)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    use_cartopy = True
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except Exception:
        use_cartopy = False

    nl_extent = [3.1, 7.45, 50.65, 53.65]
    lat = grid["lat"]
    lon = grid["lon"]
    nl_box = (lat >= 50.4) & (lat <= 53.9) & (lon >= 2.8) & (lon <= 7.7)

    fig = plt.figure(figsize=(13.5, 11.2), dpi=140)
    proj_kw = {"subplot_kw": {"projection": ccrs.PlateCarree()}} if use_cartopy else {}

    def _prep_ax(pos: int, title: str):
        ax = fig.add_subplot(2, 2, pos, **proj_kw.get("subplot_kw", {}))
        if use_cartopy:
            ax.set_extent(nl_extent, crs=ccrs.PlateCarree())
            try:
                ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.7, edgecolor="#333333")
                ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.6, linestyle="--", edgecolor="#555555")
            except Exception:
                pass
        else:
            ax.set_xlim(nl_extent[0], nl_extent[1])
            ax.set_ylim(nl_extent[2], nl_extent[3])
            ax.set_aspect(1.0 / math.cos(math.radians(52.1)))
        ax.set_title(title, fontsize=10.5, fontweight="bold")
        return ax

    xform = {"transform": ccrs.PlateCarree()} if use_cartopy else {}
    st_lon = np.array([r["lon"] for r in rows], dtype=np.float64)
    st_lat = np.array([r["lat"] for r in rows], dtype=np.float64)

    # Panel 1: 2m Temperature (model shading + KNMI circles)
    ax1 = _prep_ax(1, "2m Temperatuur (°C) — Model (vlak) vs KNMI (cirkels)")
    t_vals = grid["t2m_c"][nl_box]
    obs_t = np.array([r["t2m_c"] for r in rows], dtype=np.float64)
    all_t = np.concatenate([t_vals[np.isfinite(t_vals)], obs_t[np.isfinite(obs_t)]])
    vmin_t = math.floor(float(np.min(all_t)) - 0.5) if all_t.size else -5.0
    vmax_t = math.ceil(float(np.max(all_t)) + 0.5) if all_t.size else 15.0
    if vmax_t <= vmin_t:
        vmax_t = vmin_t + 2.0
    pm1 = ax1.pcolormesh(lon, lat, grid["t2m_c"], cmap="RdYlBu_r", vmin=vmin_t, vmax=vmax_t, shading="auto", **xform)
    m_t = np.isfinite(obs_t)
    if np.any(m_t):
        ax1.scatter(
            st_lon[m_t],
            st_lat[m_t],
            c=obs_t[m_t],
            cmap="RdYlBu_r",
            vmin=vmin_t,
            vmax=vmax_t,
            s=55,
            edgecolors="black",
            linewidths=0.9,
            zorder=5,
            **xform,
        )
    fig.colorbar(pm1, ax=ax1, fraction=0.042, pad=0.03, label="°C")

    # Panel 2: MSLP contours + 10m Wind vectors & KNMI speed circles
    ax2 = _prep_ax(2, "MSLP (hPa) & 10m Wind (m/s) — Model vs KNMI")
    mslp = grid["mslp_hpa"]
    ws_grid = np.hypot(grid["u10_geo"], grid["v10_geo"])
    pm2 = ax2.pcolormesh(lon, lat, ws_grid, cmap="YlGnBu", vmin=0.0, vmax=max(12.0, float(np.nanpercentile(ws_grid[nl_box], 98))), shading="auto", **xform)
    pmin = math.floor(float(np.nanmin(mslp[nl_box])))
    pmax = math.ceil(float(np.nanmax(mslp[nl_box])))
    if pmax > pmin:
        lvls = np.arange(pmin, pmax + 1, 1 if (pmax - pmin) <= 18 else 2)
        cs = ax2.contour(lon, lat, mslp, levels=lvls, colors="black", linewidths=0.8, **xform)
        ax2.clabel(cs, fmt="%d", fontsize=7.5)
    stride = max(4, min(grid["lat"].shape) // 45)
    ax2.quiver(
        lon[::stride, ::stride],
        lat[::stride, ::stride],
        grid["u10_geo"][::stride, ::stride],
        grid["v10_geo"][::stride, ::stride],
        scale=180.0,
        width=0.0025,
        color="#1a1a1a",
        alpha=0.7,
        **xform,
    )
    obs_ws = np.array([r["ws10_ms"] for r in rows], dtype=np.float64)
    m_ws = np.isfinite(obs_ws)
    if np.any(m_ws):
        ax2.scatter(
            st_lon[m_ws],
            st_lat[m_ws],
            c=obs_ws[m_ws],
            cmap="YlGnBu",
            vmin=0.0,
            vmax=max(12.0, float(np.nanpercentile(ws_grid[nl_box], 98))),
            s=48,
            edgecolors="black",
            linewidths=0.9,
            zorder=5,
            **xform,
        )
    fig.colorbar(pm2, ax=ax2, fraction=0.042, pad=0.03, label="m/s")

    # Panel 3: RH (%) & Fog (< 1 km) hits / misses / false alarms
    ax3 = _prep_ax(3, "Relatieve Vochtigheid (%) & Mist (< 1 km)")
    pm3 = ax3.pcolormesh(lon, lat, grid["rh_pct"], cmap="PuBu", vmin=50.0, vmax=100.0, shading="auto", **xform)
    obs_v = np.array([r["vis_m"] for r in rows], dtype=np.float64)
    mod_v = np.array([r["mod_vis_m"] for r in rows], dtype=np.float64)
    valid_v = np.isfinite(obs_v) & np.isfinite(mod_v)
    hits = valid_v & (obs_v < 1000.0) & (mod_v < 1000.0)
    misses = valid_v & (obs_v < 1000.0) & (mod_v >= 1000.0)
    fas = valid_v & (obs_v >= 1000.0) & (mod_v < 1000.0)
    clears = valid_v & (obs_v >= 1000.0) & (mod_v >= 1000.0)
    if np.any(clears):
        ax3.scatter(st_lon[clears], st_lat[clears], s=20, c="#777777", alpha=0.65, label="Geen mist", zorder=4, **xform)
    if np.any(fas):
        ax3.scatter(st_lon[fas], st_lat[fas], s=60, marker="s", c="#ff8c00", edgecolors="black", linewidths=0.8, label=f"Vals alarm ({int(np.sum(fas))})", zorder=5, **xform)
    if np.any(misses):
        ax3.scatter(st_lon[misses], st_lat[misses], s=75, marker="^", c="#d62728", edgecolors="black", linewidths=0.8, label=f"Gemist ({int(np.sum(misses))})", zorder=6, **xform)
    if np.any(hits):
        ax3.scatter(st_lon[hits], st_lat[hits], s=80, marker="o", c="#2ca02c", edgecolors="black", linewidths=0.9, label=f"Raak ({int(np.sum(hits))})", zorder=7, **xform)
    ax3.legend(loc="upper left", fontsize=7.5, framealpha=0.88)
    fig.colorbar(pm3, ax=ax3, fraction=0.042, pad=0.03, label="%")

    # Panel 4: Precipitation rate/1h vs KNMI rain gauges
    ax4 = _prep_ax(4, f"Neerslag ({rows[0]['precip_unit'] if rows else 'mm/u'}) — Model vs KNMI")
    p_field = np.ma.masked_less(grid["precip_mmh"], 0.05)
    vmax_p = max(5.0, float(np.nanmax(grid["precip_mmh"][nl_box])) if np.any(nl_box) else 5.0)
    vmax_p = min(vmax_p, 25.0)
    ax4.set_facecolor("#f3f4f6")
    pm4 = ax4.pcolormesh(lon, lat, p_field, cmap="turbo", vmin=0.0, vmax=vmax_p, shading="auto", **xform)
    obs_p = np.array([r["obs_precip_cmp"] for r in rows], dtype=np.float64)
    m_p = np.isfinite(obs_p)
    if np.any(m_p):
        ax4.scatter(
            st_lon[m_p],
            st_lat[m_p],
            c=obs_p[m_p],
            cmap="turbo",
            vmin=0.0,
            vmax=vmax_p,
            s=52,
            edgecolors="black",
            linewidths=0.9,
            zorder=5,
            **xform,
        )
    ax4.text(
        0.02,
        0.02,
        f"NL max: {grid['nl_max_mmh']:.2f} mm/u | Binnen max: {grid['int_max_mmh']:.2f} | Rand max: {grid['rim_max_mmh']:.2f}",
        transform=ax4.transAxes,
        fontsize=7.5,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc", "pad": 2.5},
    )
    fig.colorbar(pm4, ax=ax4, fraction=0.042, pad=0.03, label="mm/u")

    s = hour_stats
    fig.suptitle(
        f"Tempest-NWP ({grid['domain_name']}) vs {obs_label} | T+{lead_h:02.0f}h | Valid: {valid_dt:%Y-%m-%d %H:%MZ}\n"
        f"T2m: bias {s['t2m']['bias']:+.2f} K, RMSE {s['t2m']['rmse']:.2f} K | "
        f"MSLP: bias {s['mslp']['bias']:+.2f} hPa, RMSE {s['mslp']['rmse']:.2f} hPa | "
        f"RH: bias {s['rh']['bias']:+.1f}% | "
        f"Wind: bias {s['ws10']['bias']:+.2f} m/s, RMSE {s['ws10']['rmse']:.2f} m/s | "
        f"Mist H/M/FA: {s['fog']['hits']}/{s['fog']['misses']}/{s['fog']['false_alarms']}",
        fontsize=11,
        fontweight="bold",
        y=0.985,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.935))
    fig.savefig(out_p)
    plt.close(fig)
    return out_p


def plot_verification_timeseries(
    out_png: Path | str,
    hourly_records: Sequence[Dict[str, Any]],
    *,
    independent_after_hour: float = 4.0,
    domain_name: str = "Tempest LAM",
    use_land_stats: bool = True,
    obs_label: str = "KNMI 10-min",
) -> Path:
    """Render a 2x3 lead-time dashboard PNG showing Bias & RMSE across all verified forecast hours."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_p = Path(out_png)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    bucket = "land" if use_land_stats else "all"

    leads = np.array([r["lead_hours"] for r in hourly_records], dtype=np.float64)

    def _series(metric: str, field: str) -> np.ndarray:
        return np.array([r[bucket][metric][field] for r in hourly_records], dtype=np.float64)

    ifs_key = "ifs_land" if use_land_stats else "ifs_all"

    def _ifs_series(metric: str, field: str) -> np.ndarray:
        vals = []
        for r in hourly_records:
            blk = r.get(ifs_key)
            vals.append(float(blk[metric][field]) if isinstance(blk, dict) and metric in blk else float("nan"))
        return np.array(vals, dtype=np.float64)

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.8), dpi=140, sharex=True)
    ax_list = axes.ravel()

    for ax in ax_list:
        if len(leads) and leads[0] < independent_after_hour:
            ax.axvspan(float(leads[0]) - 0.2, float(independent_after_hour) - 0.1, color="#ffe8cc", alpha=0.55)
        ax.grid(True, alpha=0.3)

    # 1. T2m & Td2m
    ax = ax_list[0]
    ax.axhline(0.0, color="black", linewidth=0.7)
    ax.plot(leads, _series("t2m", "bias"), "o-", label="T2m Bias (K)")
    ax.plot(leads, _series("t2m", "rmse"), "s--", label="T2m RMSE (K)")
    ax.plot(leads, _series("td2m", "bias"), "^:", label="Td2m Bias (K)")
    ifs_t2m = _ifs_series("t2m", "rmse")
    if np.any(np.isfinite(ifs_t2m)):
        m_i = np.isfinite(ifs_t2m)
        ax.plot(leads[m_i], ifs_t2m[m_i], "kD--", linewidth=1.2, markersize=5, label="IFS 0.1° T2m RMSE")
    ax.set_title("2m Temperatuur & Dauwpunt (K)")
    ax.legend(fontsize=8)

    # 2. MSLP
    ax = ax_list[1]
    ax.axhline(0.0, color="black", linewidth=0.7)
    ax.plot(leads, _series("mslp", "bias"), "o-", color="#1f77b4", label="MSLP Bias (hPa)")
    ax.plot(leads, _series("mslp", "rmse"), "s--", color="#ff7f0e", label="MSLP RMSE (hPa)")
    ifs_mslp = _ifs_series("mslp", "rmse")
    if np.any(np.isfinite(ifs_mslp)):
        m_i = np.isfinite(ifs_mslp)
        ax.plot(leads[m_i], ifs_mslp[m_i], "kD--", linewidth=1.2, markersize=5, label="IFS 0.1° MSLP RMSE")
    ax.set_title("Luchtdruk op zeeniveau (hPa)")
    ax.legend(fontsize=8)

    # 3. RH2m
    ax = ax_list[2]
    ax.axhline(0.0, color="black", linewidth=0.7)
    ax.plot(leads, _series("rh", "bias"), "o-", color="#2ca02c", label="RH Bias (%)")
    ax.plot(leads, _series("rh", "rmse"), "s--", color="#9467bd", label="RH RMSE (%)")
    ifs_rh = _ifs_series("rh", "rmse")
    if np.any(np.isfinite(ifs_rh)):
        m_i = np.isfinite(ifs_rh)
        ax.plot(leads[m_i], ifs_rh[m_i], "kD--", linewidth=1.2, markersize=5, label="IFS 0.1° RH RMSE")
    ax.set_title("Relatieve Vochtigheid 2m (%)")
    ax.legend(fontsize=8)

    # 4. Wind speed & direction
    ax = ax_list[3]
    ax.axhline(0.0, color="black", linewidth=0.7)
    ax.plot(leads, _series("ws10", "bias"), "o-", label="10m Wind Bias (m/s)")
    ax.plot(leads, _series("ws10", "rmse"), "s--", label="10m Wind RMSE (m/s)")
    ifs_ws = _ifs_series("ws10", "rmse")
    if np.any(np.isfinite(ifs_ws)):
        m_i = np.isfinite(ifs_ws)
        ax.plot(leads[m_i], ifs_ws[m_i], "kD--", linewidth=1.2, markersize=5, label="IFS 0.1° Wind RMSE")
    ax_dir = ax.twinx()
    ax_dir.plot(leads, _series("wdir", "mae_deg"), "d:", color="#8c564b", label="Richting MAE (°)")
    ax_dir.set_ylabel("Richting MAE (°)", fontsize=8.5, color="#8c564b")
    ax.set_title("10m Wind & Richting")
    ax.set_xlabel("Forecast Lead (uur)")
    ax.legend(loc="upper left", fontsize=8)

    # 5. Fog contingency counts
    ax = ax_list[4]
    w_bar = 0.25
    ax.bar(leads - w_bar, [r[bucket]["fog"]["hits"] for r in hourly_records], width=w_bar, color="#2ca02c", label="Hits")
    ax.bar(leads, [r[bucket]["fog"]["misses"] for r in hourly_records], width=w_bar, color="#d62728", label="Misses")
    ax.bar(leads + w_bar, [r[bucket]["fog"]["false_alarms"] for r in hourly_records], width=w_bar, color="#ff8c00", label="False Alarms")
    ax.set_title("Mist (< 1000 m) — Hits / Misses / FA")
    ax.set_xlabel("Forecast Lead (uur)")
    ax.legend(fontsize=8)

    # 6. Precipitation: domain rim max vs interior max vs KNMI max
    ax = ax_list[5]
    ax.plot(leads, [r["grid_precip"]["nl_max_mmh"] for r in hourly_records], "o-", label="NL-box max (mm/u)")
    ax.plot(leads, [r["grid_precip"]["int_max_mmh"] for r in hourly_records], "s--", label="Binnen max (mm/u)")
    ax.plot(leads, [r["grid_precip"]["rim_max_mmh"] for r in hourly_records], "^:", color="#d62728", label="Rand max (mm/u)")
    ax.plot(
        leads,
        [max([x["obs_precip_cmp"] for x in r["obs_rows_land"] if math.isfinite(x["obs_precip_cmp"])], default=0.0) for r in hourly_records],
        "k.-",
        label="KNMI max (mm/u)",
    )
    ax.set_title("Neerslagpiek (Binnen vs Rand vs KNMI)")
    ax.set_xlabel("Forecast Lead (uur)")
    ax.legend(fontsize=8)

    scope_lbl = "Landstations" if use_land_stats else "Alle KNMI stations"
    fig.suptitle(
        f"Tempest-NWP ({domain_name}) — Uurlijkse {obs_label} Verificatie ({scope_lbl})\n"
        f"(Oranje band: T < +{independent_after_hour:.0f}u = DA-overlap / spin-up)",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.925))
    fig.savefig(out_p)
    plt.close(fig)
    return out_p


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def verify_run_against_knmi(
    input_path: Path | str,
    *,
    outdir: Optional[Path | str] = None,
    knmi_cache_dir: Optional[Path | str] = None,
    api_key: Optional[str] = None,
    obs_source: str = "auto",
    max_lead_hours: Optional[float] = None,
    independent_after_hour: Optional[float] = None,
    hourly_precip_6x: bool = False,
    land_only_plots: bool = False,
    make_maps: bool = True,
    fetcher: Optional[Callable[[datetime.datetime, Path | str], Optional[Path]]] = None,
    ifs_grib_dir: Optional[Path | str] = None,
    compare_ifs: bool = False,
    ifs_sampler: Optional[Callable[[Path | str, Sequence[Dict[str, Any]]], List[Dict[str, Any]]]] = None,
    compare_era5: bool = False,
    era5_cache_dir: Optional[Path | str] = None,
    era5_ds: Optional[xr.Dataset] = None,
    era5_step_hours: float = 3.0,
    verify_radar_fss: bool = False,
    radar_cache_dir: Optional[Path | str] = None,
    radar_hourly_fetcher: Optional[Callable[[datetime.datetime, Any], Optional[np.ndarray]]] = None,
) -> Dict[str, Any]:
    """Verify every available forecast `.wx1` step against KNMI in-situ observations (`10min` or `hourly-validated`)."""
    if obs_source not in ("auto", "10min", "hourly-validated"):
        raise ValueError(f"Unsupported obs_source {obs_source!r}; expected 'auto', '10min', or 'hourly-validated'")

    items, manifest = discover_forecast_wx1(input_path)
    if max_lead_hours is not None:
        items = [it for it in items if it[0] <= float(max_lead_hours) + 1e-6]

    root = Path(input_path)
    run_base = root if (root / "manifest.json").is_file() else (root.parent if root.name in ("d1", "d2", "d3") else root)
    eff_outdir = Path(outdir) if outdir is not None else (run_base / "verification" / "knmi_hourly")
    eff_outdir.mkdir(parents=True, exist_ok=True)
    eff_cache = Path(knmi_cache_dir) if knmi_cache_dir is not None else (eff_outdir / "knmi_cache")
    maps_dir = eff_outdir / "maps"

    if independent_after_hour is None:
        has_da = bool(manifest and isinstance(manifest.get("da"), dict) and manifest["da"].get("var3d"))
        indep_h = 4.0 if has_da else 2.0
    else:
        indep_h = float(independent_after_hour)

    def _fetch_obs_for_valid_time(valid_dt: datetime.datetime, cdir: Path) -> Optional[Path]:
        if fetcher is not None:
            return fetcher(valid_dt, cdir)
        if obs_source == "hourly-validated":
            return fetch_knmi_hourly_validated_netcdf(valid_dt, cdir, api_key=api_key)
        if obs_source == "10min":
            return fetch_knmi_10min_netcdf(valid_dt, cdir, api_key=api_key)

        u_dt = _to_utc(valid_dt)
        hr_snap = _snap_hourly_utc(u_dt)
        if hr_snap is not None:
            hv_p = cdir / f"hourly-observations-validated-{hr_snap:%Y%m%d-%H}.nc"
            if hv_p.is_file() and hv_p.stat().st_size > 1024:
                return hv_p
        m10_p = cdir / f"KMDS__OPER_P___10M_OBS_L2_{_snap_10min_utc(u_dt):%Y%m%d%H%M}.nc"
        if m10_p.is_file() and m10_p.stat().st_size > 1024:
            return m10_p

        age_h = (datetime.datetime.now(datetime.timezone.utc) - u_dt).total_seconds() / 3600.0
        if age_h >= 24.0 and hr_snap is not None:
            p = fetch_knmi_hourly_validated_netcdf(u_dt, cdir, api_key=api_key)
            if p is not None:
                return p
            return fetch_knmi_10min_netcdf(u_dt, cdir, api_key=api_key)
        p = fetch_knmi_10min_netcdf(u_dt, cdir, api_key=api_key)
        if p is not None:
            return p
        if hr_snap is not None:
            return fetch_knmi_hourly_validated_netcdf(u_dt, cdir, api_key=api_key)
        return None

    eff_ifs_dir = Path(ifs_grib_dir) if ifs_grib_dir is not None else (_auto_discover_ifs_dir(run_base, manifest) if compare_ifs else None)
    ifs_by_time = discover_ifs_grib_by_valid_time(eff_ifs_dir) if eff_ifs_dir is not None else {}
    ifs_sample_fn = ifs_sampler if ifs_sampler is not None else sample_ifs_grib_at_stations
    lbc_manifest = manifest.get("lbc") if isinstance(manifest, dict) else None
    if era5_cache_dir is not None:
        eff_era5_cache: Optional[Path] = Path(era5_cache_dir)
    elif isinstance(lbc_manifest, dict) and lbc_manifest.get("era5_cache_dir"):
        eff_era5_cache = Path(str(lbc_manifest["era5_cache_dir"]))
    else:
        eff_era5_cache = eff_outdir / "era5_verify_cache"
    opened_era5_ds = era5_ds
    allow_era5_cache_write = era5_ds is None

    eff_radar_cache = Path(radar_cache_dir) if radar_cache_dir is not None else (eff_outdir / "radar_cache")
    fss_all_recs: List[Dict[str, Any]] = []
    fss_indep_recs: List[Dict[str, Any]] = []
    radar_spec_cache: Dict[str, Any] = {}
    if (compare_ifs or ifs_grib_dir is not None) and not ifs_by_time:
        logger.warning("IFS comparison requested, but no *_sfc_*.grib2 files found (dir=%s).", eff_ifs_dir)

    all_pairs_all: List[Dict[str, Any]] = []
    all_pairs_land: List[Dict[str, Any]] = []
    indep_pairs_all: List[Dict[str, Any]] = []
    indep_pairs_land: List[Dict[str, Any]] = []
    ifs_t_all: List[Dict[str, Any]] = []
    ifs_i_all: List[Dict[str, Any]] = []
    ifs_t_land: List[Dict[str, Any]] = []
    ifs_i_land: List[Dict[str, Any]] = []
    ifs_indep_t_all: List[Dict[str, Any]] = []
    ifs_indep_i_all: List[Dict[str, Any]] = []
    ifs_indep_t_land: List[Dict[str, Any]] = []
    ifs_indep_i_land: List[Dict[str, Any]] = []
    ifs_matched_leads: List[float] = []
    era5_t_all: List[Dict[str, Any]] = []
    era5_e_all: List[Dict[str, Any]] = []
    era5_t_land: List[Dict[str, Any]] = []
    era5_e_land: List[Dict[str, Any]] = []
    era5_indep_t_all: List[Dict[str, Any]] = []
    era5_indep_e_all: List[Dict[str, Any]] = []
    era5_indep_t_land: List[Dict[str, Any]] = []
    era5_indep_e_land: List[Dict[str, Any]] = []
    era5_matched_leads: List[float] = []
    era5_upper_all: List[Dict[str, Any]] = []
    era5_upper_indep: List[Dict[str, Any]] = []
    hourly_records: List[Dict[str, Any]] = []
    station_buckets: Dict[str, List[Dict[str, Any]]] = {}

    prev_tot_precip: Optional[np.ndarray] = None
    prev_precip_parts: Optional[Dict[str, np.ndarray]] = None
    prev_lead_h: Optional[float] = None
    domain_name = "tempest_lam"
    first_wx1_attrs: Dict[str, str] = {}
    used_sources: set[str] = set()
    map_paths: List[Path] = []
    skipped_hours: List[Dict[str, Any]] = []

    for lead_h, valid_dt, wx1_path in items:
        nc_path = _fetch_obs_for_valid_time(valid_dt, eff_cache)
        if nc_path is None:
            skipped_hours.append({"lead_hours": lead_h, "valid_time_utc": valid_dt.isoformat(), "reason": "no_knmi_obs"})
            with wx1.open_dataset(wx1_path) as ds_p:
                r = np.asarray(ds_p["rain_acc"].values, dtype=np.float64) if "rain_acc" in ds_p else None
                s = np.asarray(ds_p["snow_acc"].values, dtype=np.float64) if "snow_acc" in ds_p else None
                g = np.asarray(ds_p["graupel_acc"].values, dtype=np.float64) if "graupel_acc" in ds_p else None
                if r is not None:
                    prev_tot_precip = np.maximum(0.0, r + (s if s is not None else 0.0) + (g if g is not None else 0.0))
                    prev_precip_parts = {
                        "rain_acc": r,
                        "snow_acc": s if s is not None else np.zeros_like(r),
                        "graupel_acc": g if g is not None else np.zeros_like(r),
                    }
                    prev_lead_h = lead_h
            continue

        stations = load_knmi_stations(nc_path)
        src_kind = (
            stations[0].get("obs_source", "10min")
            if stations
            else ("hourly-validated" if "hourly-observations-validated" in nc_path.name else "10min")
        )
        used_sources.add(src_kind)

        had_prev_precip = prev_tot_precip is not None
        obs_1h_map: Dict[str, float] = {}
        if src_kind == "10min" and hourly_precip_6x and had_prev_precip:
            obs_1h_map = load_knmi_hourly_precip_mm(valid_dt, eff_cache, api_key=api_key, fetcher=fetcher)

        dt_h = (lead_h - prev_lead_h) if (prev_lead_h is not None and lead_h > prev_lead_h) else 1.0
        matched, grid = sample_wx1_at_stations(
            wx1_path,
            stations,
            prev_total_precip=prev_tot_precip,
            prev_precip_parts=prev_precip_parts,
            interval_h=dt_h,
        )
        domain_name = grid["domain_name"]
        if not first_wx1_attrs and isinstance(grid.get("wx1_attrs"), dict):
            first_wx1_attrs = dict(grid["wx1_attrs"])
        prev_tot_precip = grid["tot_precip_mm"]
        prev_precip_parts = {
            "rain_acc": grid["rain_acc"],
            "snow_acc": grid["snow_acc"],
            "graupel_acc": grid["graupel_acc"],
        }
        prev_lead_h = lead_h

        for row in matched:
            row["lead_hours"] = lead_h
            row["valid_time_utc"] = valid_dt.isoformat()
            if (
                src_kind == "hourly-validated"
                and lead_h > 0.0
                and had_prev_precip
                and abs(dt_h - 1.0) < 0.25
                and math.isfinite(row.get("precip_1h_mm", float("nan")))
            ):
                row["obs_precip_cmp"] = row["precip_1h_mm"]
                row["mod_precip_cmp"] = row["mod_precip_step_mm"]
                row["precip_unit"] = "mm/1h"
            elif obs_1h_map and row["station_id"] in obs_1h_map and lead_h > 0.0:
                row["obs_precip_cmp"] = obs_1h_map[row["station_id"]]
                row["mod_precip_cmp"] = row["mod_precip_step_mm"]
                row["precip_unit"] = "mm/1h"
            else:
                row["obs_precip_cmp"] = row["precip_mmh"]
                row["mod_precip_cmp"] = row["mod_precip_mmh"]
                row["precip_unit"] = "mm/u"
            station_buckets.setdefault(row["station_id"], []).append(row)

        land_rows = [r for r in matched if not r["is_offshore"]]
        all_stats = _summarize_matched_rows(matched)
        land_stats = _summarize_matched_rows(land_rows)
        is_indep = bool(lead_h + 1e-6 >= indep_h)

        all_pairs_all.extend(matched)
        all_pairs_land.extend(land_rows)
        if is_indep:
            indep_pairs_all.extend(matched)
            indep_pairs_land.extend(land_rows)

        rec: Dict[str, Any] = {
            "lead_hours": lead_h,
            "valid_time_utc": valid_dt.isoformat(),
            "independent": is_indep,
            "all": all_stats,
            "land": land_stats,
            "grid_precip": {
                "nl_max_mmh": grid["nl_max_mmh"],
                "int_max_mmh": grid["int_max_mmh"],
                "rim_max_mmh": grid["rim_max_mmh"],
                "rim_tot_max_mm": grid["rim_tot_max_mm"],
                "int_tot_max_mm": grid["int_tot_max_mm"],
                "n_gt5_all": grid["n_gt5_all"],
                "n_gt5_int": grid["n_gt5_int"],
            },
            "obs_rows_land": land_rows,
        }

        v_key = _to_utc(valid_dt).strftime("%Y-%m-%dT%H:%MZ")
        ifs_path = ifs_by_time.get(v_key)
        if ifs_path is not None:
            ifs_matched = ifs_sample_fn(ifs_path, stations)
            for ir in ifs_matched:
                ir["lead_hours"] = lead_h
                ir["valid_time_utc"] = valid_dt.isoformat()
            ifs_land_raw = [r for r in ifs_matched if not r["is_offshore"]]
            t_all_a, i_all_a = _align_rows_with_ifs(matched, ifs_matched)
            t_land_a, i_land_a = _align_rows_with_ifs(land_rows, ifs_land_raw)
            ifs_all_stats = _summarize_matched_rows(i_all_a)
            ifs_land_stats = _summarize_matched_rows(i_land_a)
            t_land_for_skill = _summarize_matched_rows(t_land_a)
            rec["ifs_all"] = ifs_all_stats
            rec["ifs_land"] = ifs_land_stats
            rec["skill_vs_ifs_land"] = compute_skill_score_vs_ifs(t_land_for_skill, ifs_land_stats)
            ifs_t_all.extend(t_all_a)
            ifs_i_all.extend(i_all_a)
            ifs_t_land.extend(t_land_a)
            ifs_i_land.extend(i_land_a)
            if is_indep:
                ifs_indep_t_all.extend(t_all_a)
                ifs_indep_i_all.extend(i_all_a)
                ifs_indep_t_land.extend(t_land_a)
                ifs_indep_i_land.extend(i_land_a)
            ifs_matched_leads.append(lead_h)

        era5_cadence_ok = (float(era5_step_hours) <= 1.0) or (
            abs(lead_h - round(lead_h / float(era5_step_hours)) * float(era5_step_hours)) < 1e-3
        )
        if compare_era5 and era5_cadence_ok:
            try:
                ds_e_slice, opened_era5_ds = _load_era5_verify_slice(
                    valid_dt,
                    grid["lat"],
                    grid["lon"],
                    eff_era5_cache,
                    opened_era5_ds,
                    allow_cache_write=allow_era5_cache_write,
                )
                if ds_e_slice is not None:
                    try:
                        era5_matched = sample_era5_at_stations(ds_e_slice, stations, valid_dt=valid_dt)
                        for er in era5_matched:
                            er["lead_hours"] = lead_h
                            er["valid_time_utc"] = valid_dt.isoformat()
                        era5_land_raw = [r for r in era5_matched if not r["is_offshore"]]
                        t_all_e, e_all_a = _align_rows_with_ifs(matched, era5_matched)
                        t_land_e, e_land_a = _align_rows_with_ifs(land_rows, era5_land_raw)
                        era5_all_stats = _summarize_matched_rows(e_all_a)
                        era5_land_stats = _summarize_matched_rows(e_land_a)
                        t_land_e_skill = _summarize_matched_rows(t_land_e)
                        rec["era5_all"] = era5_all_stats
                        rec["era5_land"] = era5_land_stats
                        rec["skill_vs_era5_land"] = compute_skill_score_vs_ifs(t_land_e_skill, era5_land_stats)
                        era5_t_all.extend(t_all_e)
                        era5_e_all.extend(e_all_a)
                        era5_t_land.extend(t_land_e)
                        era5_e_land.extend(e_land_a)
                        if is_indep:
                            era5_indep_t_all.extend(t_all_e)
                            era5_indep_e_all.extend(e_all_a)
                            era5_indep_t_land.extend(t_land_e)
                            era5_indep_e_land.extend(e_land_a)
                        era5_matched_leads.append(lead_h)

                        ua_rec = compute_upper_air_vs_era5(wx1_path, ds_e_slice, valid_dt=valid_dt)
                        rec["era5_upper_air"] = ua_rec
                        era5_upper_all.append(ua_rec)
                        if is_indep:
                            era5_upper_indep.append(ua_rec)
                    finally:
                        if era5_ds is None:
                            ds_e_slice.close()
            except Exception as exc:
                logger.warning("ERA5 validation failed at T+%.1fh (%s): %s", lead_h, valid_dt.isoformat(), exc)

        if verify_radar_fss and lead_h > 0.0 and abs(dt_h - 1.0) < 0.25:
            try:
                if domain_name not in radar_spec_cache:
                    radar_spec_cache[domain_name] = get_domain_spec(domain_name, nz=16)
                r_spec = radar_spec_cache[domain_name]
                if radar_hourly_fetcher is not None:
                    obs_rad_1h = radar_hourly_fetcher(valid_dt, r_spec)
                else:
                    from tempest_v2.ingest_radar_qpe import fetch_knmi_radar_hourly_mm

                    obs_rad_1h = fetch_knmi_radar_hourly_mm(
                        valid_dt, r_spec, api_key=api_key, cache_dir=eff_radar_cache
                    )
                if obs_rad_1h is not None and obs_rad_1h.shape == grid["precip_step_mm"].shape:
                    fss_rec = compute_fractions_skill_score(
                        grid["precip_step_mm"],
                        obs_rad_1h,
                        dx_km=float(r_spec.dx_m) / 1000.0,
                    )
                    rec["radar_fss"] = fss_rec
                    fss_all_recs.append(fss_rec)
                    if is_indep:
                        fss_indep_recs.append(fss_rec)
            except Exception as exc:
                logger.warning("Radar FSS verification failed at T+%.1fh (%s): %s", lead_h, valid_dt.isoformat(), exc)

        hourly_records.append(rec)

        if make_maps:
            tag = f"{int(round(lead_h)):03d}" if abs(lead_h - round(lead_h)) < 1e-4 else f"{lead_h:06.2f}"
            obs_label_hr = "KNMI hourly-validated" if src_kind == "hourly-validated" else "KNMI 10-min"
            mp = plot_hourly_verification_map(
                maps_dir / f"knmi_map_T+{tag}h.png",
                lead_h,
                valid_dt,
                grid,
                matched,
                land_stats if land_only_plots else all_stats,
                land_only=land_only_plots,
                obs_label=obs_label_hr,
            )
            map_paths.append(mp)

    # Hourly CSV export
    has_ifs_rows = bool(ifs_matched_leads)
    hourly_csv_rows: List[Dict[str, Any]] = []
    for rec in hourly_records:
        st_b = rec["land"]
        gp = rec["grid_precip"]
        csv_row: Dict[str, Any] = {
            "lead_hours": rec["lead_hours"],
            "valid_time_utc": rec["valid_time_utc"],
            "independent": rec["independent"],
            "n_land": st_b["n_stations"],
            "n_all": rec["all"]["n_stations"],
            "land_t2m_bias_k": round(st_b["t2m"]["bias"], 3),
            "land_t2m_rmse_k": round(st_b["t2m"]["rmse"], 3),
            "land_t2m_corr": round(st_b["t2m"]["corr"], 3),
            "land_td2m_bias_k": round(st_b["td2m"]["bias"], 3),
            "land_rh_bias_pct": round(st_b["rh"]["bias"], 2),
            "land_mslp_bias_hpa": round(st_b["mslp"]["bias"], 3),
            "land_mslp_rmse_hpa": round(st_b["mslp"]["rmse"], 3),
            "land_ws10_bias_ms": round(st_b["ws10"]["bias"], 3),
            "land_ws10_rmse_ms": round(st_b["ws10"]["rmse"], 3),
            "land_gust10_bias_ms": round(st_b["gust10"]["bias"], 3),
            "land_wdir_mae_deg": round(st_b["wdir"]["mae_deg"], 1),
            "land_sw_down_bias_wm2": round(st_b["sw_down"]["bias"], 1),
            "land_sw_down_rmse_wm2": round(st_b["sw_down"]["rmse"], 1),
            "land_tcc_bias": round(st_b["tcc"]["bias"], 3),
            "land_tcc_rmse": round(st_b["tcc"]["rmse"], 3),
            "land_fog_hits": st_b["fog"]["hits"],
            "land_fog_misses": st_b["fog"]["misses"],
            "land_fog_fa": st_b["fog"]["false_alarms"],
            "grid_nl_precip_max_mmh": round(gp["nl_max_mmh"], 2),
            "grid_int_precip_max_mmh": round(gp["int_max_mmh"], 2),
            "grid_rim_precip_max_mmh": round(gp["rim_max_mmh"], 2),
        }
        if has_ifs_rows:
            ib = rec.get("ifs_land")
            sk = rec.get("skill_vs_ifs_land")
            csv_row.update(
                {
                    "ifs_t2m_bias_k": round(ib["t2m"]["bias"], 3) if ib else float("nan"),
                    "ifs_t2m_rmse_k": round(ib["t2m"]["rmse"], 3) if ib else float("nan"),
                    "t2m_skill_vs_ifs": round(sk["t2m"]["skill_score"], 3) if sk else float("nan"),
                    "ifs_mslp_bias_hpa": round(ib["mslp"]["bias"], 3) if ib else float("nan"),
                    "ifs_mslp_rmse_hpa": round(ib["mslp"]["rmse"], 3) if ib else float("nan"),
                    "ifs_ws10_rmse_ms": round(ib["ws10"]["rmse"], 3) if ib else float("nan"),
                }
            )
        if era5_matched_leads:
            eb = rec.get("era5_land")
            esk = rec.get("skill_vs_era5_land")
            csv_row.update(
                {
                    "era5_t2m_rmse_k": round(eb["t2m"]["rmse"], 3) if eb else float("nan"),
                    "t2m_skill_vs_era5": round(esk["t2m"]["skill_score"], 3) if esk else float("nan"),
                    "era5_mslp_rmse_hpa": round(eb["mslp"]["rmse"], 3) if eb else float("nan"),
                    "era5_ws10_rmse_ms": round(eb["ws10"]["rmse"], 3) if eb else float("nan"),
                }
            )
        hourly_csv_rows.append(csv_row)
    hourly_csv_p = eff_outdir / "knmi_hourly_metrics.csv"
    if hourly_csv_rows:
        _write_csv(hourly_csv_p, hourly_csv_rows, list(hourly_csv_rows[0].keys()))

    # Per-station CSV export (over independent hours if any, else all hours)
    stn_rows_out: List[Dict[str, Any]] = []
    for sid, s_list in sorted(station_buckets.items()):
        sub = [x for x in s_list if x["lead_hours"] + 1e-6 >= indep_h] or s_list
        s_sum = _summarize_matched_rows(sub)
        stn_rows_out.append(
            {
                "station_id": sid,
                "station_name": sub[0]["station_name"],
                "is_offshore": sub[0]["is_offshore"],
                "lat": round(sub[0]["lat"], 3),
                "lon": round(sub[0]["lon"], 3),
                "n_hours": len(sub),
                "t2m_bias_k": round(s_sum["t2m"]["bias"], 3),
                "t2m_rmse_k": round(s_sum["t2m"]["rmse"], 3),
                "rh_bias_pct": round(s_sum["rh"]["bias"], 2),
                "mslp_bias_hpa": round(s_sum["mslp"]["bias"], 3),
                "ws10_bias_ms": round(s_sum["ws10"]["bias"], 3),
                "ws10_rmse_ms": round(s_sum["ws10"]["rmse"], 3),
                "wdir_mae_deg": round(s_sum["wdir"]["mae_deg"], 1),
            }
        )
    stn_rows_out.sort(key=lambda r: (math.isnan(r["t2m_rmse_k"]), -(r["t2m_rmse_k"] if math.isfinite(r["t2m_rmse_k"]) else 0.0)))
    stn_csv_p = eff_outdir / "knmi_station_metrics.csv"
    if stn_rows_out:
        _write_csv(stn_csv_p, stn_rows_out, list(stn_rows_out[0].keys()))

    if used_sources == {"hourly-validated"}:
        obs_label = "KNMI hourly-validated"
    elif used_sources == {"10min"} or not used_sources:
        obs_label = "KNMI 10-min"
    else:
        obs_label = "KNMI in-situ"

    ts_png: Optional[Path] = None
    if hourly_records:
        ts_png = plot_verification_timeseries(
            eff_outdir / "overview_timeseries.png",
            hourly_records,
            independent_after_hour=indep_h,
            domain_name=domain_name,
            use_land_stats=True,
            obs_label=obs_label,
        )

    # Strip large per-station lists before JSON serialization
    hourly_json = [
        {k: v for k, v in rec.items() if k != "obs_rows_land"}
        for rec in hourly_records
    ]

    def _build_ref_comparison(
        src_key: str,
        src_label_k: str,
        src_label_val: str,
        matched_leads: List[float],
        t_all: Sequence[Dict[str, Any]],
        r_all: Sequence[Dict[str, Any]],
        t_land: Sequence[Dict[str, Any]],
        r_land: Sequence[Dict[str, Any]],
        indep_t_all: Sequence[Dict[str, Any]],
        indep_r_all: Sequence[Dict[str, Any]],
        indep_t_land: Sequence[Dict[str, Any]],
        indep_r_land: Sequence[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not matched_leads:
            return None

        def _cmp_block(t_rows: Sequence[Dict[str, Any]], r_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
            ts = _summarize_matched_rows(t_rows)
            rs = _summarize_matched_rows(r_rows)
            return {"tempest": ts, src_key: rs, "skill": compute_skill_score_vs_ifs(ts, rs)}

        return {
            src_label_k: src_label_val,
            "matched_hours": len(matched_leads),
            "matched_leads": matched_leads,
            "overall_all": _cmp_block(t_all, r_all),
            "overall_land": _cmp_block(t_land, r_land),
            "independent_all": _cmp_block(indep_t_all, indep_r_all),
            "independent_land": _cmp_block(indep_t_land, indep_r_land),
        }

    ifs_comparison = _build_ref_comparison(
        "ifs",
        "ifs_dir",
        str(eff_ifs_dir) if eff_ifs_dir is not None else "",
        ifs_matched_leads,
        ifs_t_all,
        ifs_i_all,
        ifs_t_land,
        ifs_i_land,
        ifs_indep_t_all,
        ifs_indep_i_all,
        ifs_indep_t_land,
        ifs_indep_i_land,
    )
    era5_comparison = _build_ref_comparison(
        "era5",
        "era5_cache_dir",
        str(eff_era5_cache) if eff_era5_cache is not None else "",
        era5_matched_leads,
        era5_t_all,
        era5_e_all,
        era5_t_land,
        era5_e_land,
        era5_indep_t_all,
        era5_indep_e_all,
        era5_indep_t_land,
        era5_indep_e_land,
    )
    era5_upper_air = (
        {
            "overall": _aggregate_upper_air_records(era5_upper_all),
            "independent": _aggregate_upper_air_records(era5_upper_indep),
        }
        if era5_upper_all
        else None
    )

    run_metadata: Dict[str, Any] = {}
    if isinstance(manifest, dict):
        for mk in (
            "git_commit",
            "mode",
            "domain",
            "init_time_utc",
            "ic_source",
            "warm_start",
            "forecast_hours",
            "nz",
            "dt_s",
            "physics_every",
            "cy46_physics",
            "pbl_last",
        ):
            if mk in manifest:
                run_metadata[mk] = manifest[mk]
        da_blk = manifest.get("da")
        run_metadata["da_enabled"] = bool(isinstance(da_blk, dict) and da_blk.get("var3d"))
    for ak, av in first_wx1_attrs.items():
        if ak not in run_metadata:
            run_metadata[ak] = av

    summary: Dict[str, Any] = {
        "domain_name": domain_name,
        "obs_source_requested": obs_source,
        "obs_sources_used": sorted(used_sources),
        "run_metadata": run_metadata,
        "input_path": str(root),
        "outdir": str(eff_outdir),
        "independent_after_hour": indep_h,
        "verified_hours": len(hourly_records),
        "skipped_hours": skipped_hours,
        "overall_all": _summarize_matched_rows(all_pairs_all),
        "overall_land": _summarize_matched_rows(all_pairs_land),
        "independent_all": _summarize_matched_rows(indep_pairs_all),
        "independent_land": _summarize_matched_rows(indep_pairs_land),
        "ifs_comparison": ifs_comparison,
        "era5_comparison": era5_comparison,
        "era5_upper_air": era5_upper_air,
        "radar_fss": (
            {
                "overall": _aggregate_fss_records(fss_all_recs),
                "independent": _aggregate_fss_records(fss_indep_recs),
            }
            if fss_all_recs
            else None
        ),
        "hourly": hourly_json,
        "worst_t2m_land_stations": [r for r in stn_rows_out if not r["is_offshore"]][:10],
        "artifacts": {
            "summary_json": str(eff_outdir / "knmi_verification_summary.json"),
            "hourly_csv": str(hourly_csv_p),
            "station_csv": str(stn_csv_p),
            "timeseries_png": str(ts_png) if ts_png is not None else None,
            "map_pngs": [str(p) for p in map_paths],
        },
    }
    (eff_outdir / "knmi_verification_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    return summary
