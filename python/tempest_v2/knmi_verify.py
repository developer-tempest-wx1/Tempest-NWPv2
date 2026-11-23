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


def fetch_knmi_10min_netcdf(
    valid_time_utc: datetime.datetime,
    cache_dir: Path | str,
    *,
    api_key: Optional[str] = None,
) -> Optional[Path]:
    """Download (or reuse cached) official KNMI 10-minute NetCDF for `valid_time_utc`."""
    snap = _snap_10min_utc(valid_time_utc)
    fname = f"KMDS__OPER_P___10M_OBS_L2_{snap:%Y%m%d%H%M}.nc"
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
    meta_url = f"{KNMI_API_BASE}/datasets/{DATASET_NAME}/versions/{DATASET_VERSION}/files/{fname}/url"
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
            logger.debug("KNMI 10-min file not available (404): %s", fname)
            return None
        if exc.code in (401, 403):
            raise PermissionError(f"KNMI API authentication failed (HTTP {exc.code}) for {fname}") from exc
        logger.warning("KNMI API HTTP %d for %s: %s", exc.code, fname, exc)
        return None
    except Exception as exc:
        logger.warning("KNMI download failed for %s: %s", fname, exc)
        return None


def load_knmi_10min_stations(nc_path: Path | str) -> List[Dict[str, Any]]:
    """Parse KNMI 10-minute station NetCDF into clean SI/meteorological station dicts."""
    p = Path(nc_path)
    rows: List[Dict[str, Any]] = []
    with nc.Dataset(p) as ds:
        stn_var = ds.variables.get("station")
        n_stn = len(stn_var) if stn_var is not None else 0
        if n_stn == 0:
            return rows

        ids = _decode_str_array(ds, "station", n_stn)
        names = _decode_str_array(ds, "stationname", n_stn)
        lats = _extract_1d_float(ds, "lat", n_stn)
        lons = _extract_1d_float(ds, "lon", n_stn)
        hgts = _extract_1d_float(ds, "height", n_stn)
        if np.all(np.isnan(hgts)):
            hgts = _extract_1d_float(ds, "elevation", n_stn)

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

        for i in range(n_stn):
            sid = ids[i]
            sname = names[i] or (KNMI_STATIONS[sid][3] if sid in KNMI_STATIONS else sid)
            lat = float(lats[i])
            lon = float(lons[i])
            elev = float(hgts[i])
            if not (math.isfinite(lat) and math.isfinite(lon) and -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                if sid in KNMI_STATIONS:
                    lat, lon, fb_elev, fb_name = KNMI_STATIONS[sid]
                    sname = sname or fb_name
                    if not math.isfinite(elev):
                        elev = fb_elev
                else:
                    continue
            if not math.isfinite(elev):
                elev = float(KNMI_STATIONS[sid][2]) if sid in KNMI_STATIONS else 0.0

            t2m_c = float(ta[i])
            if math.isfinite(t2m_c) and t2m_c > 150.0:
                t2m_c -= 273.15
            if not (-55.0 <= t2m_c <= 55.0):
                t2m_c = float("nan")

            td2m_c = float(td[i])
            if math.isfinite(td2m_c) and td2m_c > 150.0:
                td2m_c -= 273.15
            if not (-65.0 <= td2m_c <= 45.0):
                td2m_c = float("nan")

            rh_pct = float(rh[i])
            if not (0.0 <= rh_pct <= 105.0):
                rh_pct = float("nan")

            mslp_hpa = float(pp[i])
            if math.isfinite(mslp_hpa) and mslp_hpa > 2000.0:
                mslp_hpa *= 0.01
            if not (870.0 <= mslp_hpa <= 1075.0):
                mslp_hpa = float("nan")

            ws = float(ff[i])
            wd = float(dd[i])
            u10, v10 = float("nan"), float("nan")
            if math.isfinite(ws) and 0.0 <= ws <= 75.0:
                if math.isfinite(wd) and 0.0 <= wd <= 360.0:
                    rad = math.radians(wd)
                    u10 = -ws * math.sin(rad)
                    v10 = -ws * math.cos(rad)
            else:
                ws = float("nan")
            if not (math.isfinite(wd) and 0.0 <= wd <= 360.0):
                wd = float("nan")

            gust = float(fx[i]) if (math.isfinite(float(fx[i])) and 0.0 <= float(fx[i]) <= 100.0) else float("nan")
            vis_v = float(zm[i]) if (math.isfinite(float(zm[i])) and 0.0 <= float(zm[i]) <= 100_000.0) else float("nan")
            if not math.isfinite(vis_v):
                vis_v = float(vv[i]) if (math.isfinite(float(vv[i])) and 0.0 <= float(vv[i]) <= 100_000.0) else float("nan")
            rg_mmh = float(rg[i]) if (math.isfinite(float(rg[i])) and 0.0 <= float(rg[i]) <= 300.0) else float("nan")
            qg_wm2 = float(qg[i]) if (math.isfinite(float(qg[i])) and 0.0 <= float(qg[i]) <= 1400.0) else float("nan")

            if all(math.isnan(x) for x in (t2m_c, mslp_hpa, ws, rh_pct, vis_v, rg_mmh)):
                continue

            rows.append(
                {
                    "station_id": sid,
                    "station_name": sname,
                    "lat": lat,
                    "lon": lon,
                    "elev_m": elev,
                    "is_offshore": _is_offshore_station(sid, sname),
                    "obs_time_utc": obs_dt.isoformat() if obs_dt is not None else "",
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
                    "precip_mmh": rg_mmh,
                    "sw_down_wm2": qg_wm2,
                }
            )
    return rows


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
        if not x_asc:
            i0, i1 = (nx - 1) - i0_a, (nx - 1) - i1_a
        else:
            i0, i1 = i0_a, i1_a

        j0_a = np.clip(np.searchsorted(yc, sy, side="right") - 1, 0, ny - 2)
        j1_a = j0_a + 1
        wy = np.clip((sy - yc[j0_a]) / np.maximum(yc[j1_a] - yc[j0_a], 1e-12), 0.0, 1.0)
        if not y_asc:
            j0, j1 = (ny - 1) - j0_a, (ny - 1) - j1_a
        else:
            j0, j1 = j0_a, j1_a

        m_t2m = _sample_2d_bilinear(t2m_c, j0, j1, i0, i1, wy, wx)
        m_td2m = _sample_2d_bilinear(td2m_c, j0, j1, i0, i1, wy, wx)
        m_rh = _sample_2d_bilinear(rh_pct, j0, j1, i0, i1, wy, wx)
        m_mslp = _sample_2d_bilinear(mslp_hpa, j0, j1, i0, i1, wy, wx)
        m_u10 = _sample_2d_bilinear(u10, j0, j1, i0, i1, wy, wx)
        m_v10 = _sample_2d_bilinear(v10, j0, j1, i0, i1, wy, wx)
        m_gust = _sample_2d_bilinear(gust10, j0, j1, i0, i1, wy, wx)
        m_vis = _sample_2d_bilinear(vis_m, j0, j1, i0, i1, wy, wx)
        m_pr = _sample_2d_bilinear(precip_mmh, j0, j1, i0, i1, wy, wx)
        m_p1h = _sample_2d_bilinear(precip_step_mm, j0, j1, i0, i1, wy, wx)
    else:
        tree = cKDTree(np.column_stack([lat2d.ravel(), lon2d.ravel()]))
        dist, idx = tree.query(np.column_stack([st_lats, st_lons]))
        inside = dist <= 0.15
        jj = idx // nx
        ii = idx % nx
        m_t2m = t2m_c[jj, ii]
        m_td2m = td2m_c[jj, ii]
        m_rh = rh_pct[jj, ii]
        m_mslp = mslp_hpa[jj, ii]
        m_u10 = u10[jj, ii]
        m_v10 = v10[jj, ii]
        m_gust = gust10[jj, ii]
        m_vis = vis_m[jj, ii]
        m_pr = precip_mmh[jj, ii]
        m_p1h = precip_step_mm[jj, ii]

    matched: List[Dict[str, Any]] = []
    for k, stn in enumerate(stations):
        if not bool(inside[k]):
            continue
        mu, mv = float(m_u10[k]), float(m_v10[k])
        mws = math.hypot(mu, mv) if (math.isfinite(mu) and math.isfinite(mv)) else float("nan")
        mwd = (math.degrees(math.atan2(-mu, -mv)) % 360.0) if (math.isfinite(mu) and math.isfinite(mv) and mws > 1e-4) else float("nan")
        row = dict(stn)
        row.update(
            {
                "mod_t2m_c": float(m_t2m[k]),
                "mod_td2m_c": float(m_td2m[k]),
                "mod_rh_pct": float(m_rh[k]),
                "mod_mslp_hpa": float(m_mslp[k]),
                "mod_u10_ms": mu,
                "mod_v10_ms": mv,
                "mod_ws10_ms": mws,
                "mod_wd10_deg": mwd,
                "mod_gust10_ms": float(m_gust[k]),
                "mod_vis_m": float(m_vis[k]),
                "mod_precip_mmh": float(m_pr[k]),
                "mod_precip_step_mm": float(m_p1h[k]),
            }
        )
        matched.append(row)

    rw = min(16, max(1, ny // 6), max(1, nx // 6))
    rim_mask = np.ones((ny, nx), dtype=bool)
    rim_mask[rw : ny - rw, rw : nx - rw] = False
    nl_mask = (lat2d >= 50.6) & (lat2d <= 53.6) & (lon2d >= 3.2) & (lon2d <= 7.3)
    grid_bundle = {
        "domain_name": domain_name or "tempest_lam",
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
    min_ws_ms: float = 1.5,
) -> Dict[str, Any]:
    """Compute circular shortest-arc wind direction error [deg] for wind speeds >= `min_ws_ms`."""
    m = np.asarray(mod_deg, dtype=np.float64)
    o = np.asarray(obs_deg, dtype=np.float64)
    w = np.asarray(obs_ws_ms, dtype=np.float64)
    valid = np.isfinite(m) & np.isfinite(o) & np.isfinite(w) & (w >= float(min_ws_ms))
    n = int(np.sum(valid))
    if n == 0:
        return {"n": 0, "bias_deg": float("nan"), "mae_deg": float("nan"), "rmse_deg": float("nan")}
    diff = ((m[valid] - o[valid] + 180.0) % 360.0) - 180.0
    return {
        "n": n,
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


def _summarize_matched_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    t2m = compute_continuous_stats([r["mod_t2m_c"] for r in rows], [r["t2m_c"] for r in rows])
    td2m = compute_continuous_stats([r["mod_td2m_c"] for r in rows], [r["td2m_c"] for r in rows])
    rh = compute_continuous_stats([r["mod_rh_pct"] for r in rows], [r["rh_pct"] for r in rows])
    mslp = compute_continuous_stats([r["mod_mslp_hpa"] for r in rows], [r["mslp_hpa"] for r in rows])
    ws10 = compute_continuous_stats([r["mod_ws10_ms"] for r in rows], [r["ws10_ms"] for r in rows])
    gust = compute_continuous_stats([r["mod_gust10_ms"] for r in rows], [r["gust10_ms"] for r in rows])
    precip = compute_continuous_stats([r["mod_precip_cmp"] for r in rows], [r["obs_precip_cmp"] for r in rows])
    wdir = compute_wind_dir_stats_deg(
        [r["mod_wd10_deg"] for r in rows],
        [r["wd10_deg"] for r in rows],
        [r["ws10_ms"] for r in rows],
    )
    fog = compute_fog_contingency([r["mod_vis_m"] for r in rows], [r["vis_m"] for r in rows])
    return {
        "n_stations": len(rows),
        "t2m": t2m,
        "td2m": td2m,
        "rh": rh,
        "mslp": mslp,
        "ws10": ws10,
        "gust10": gust,
        "wdir": wdir,
        "fog": fog,
        "precip": precip,
    }


def plot_hourly_verification_map(
    out_png: Path | str,
    lead_h: float,
    valid_dt: datetime.datetime,
    grid: Dict[str, Any],
    matched_rows: Sequence[Dict[str, Any]],
    hour_stats: Dict[str, Any],
    *,
    land_only: bool = False,
) -> Path:
    """Render a 2x2 Netherlands verification PNG comparing model fields against KNMI 10-min obs."""
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
        f"Tempest-NWP ({grid['domain_name']}) vs KNMI 10-min | T+{lead_h:02.0f}h | Valid: {valid_dt:%Y-%m-%d %H:%MZ}\n"
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
    ax.set_title("2m Temperatuur & Dauwpunt (K)")
    ax.legend(fontsize=8)

    # 2. MSLP
    ax = ax_list[1]
    ax.axhline(0.0, color="black", linewidth=0.7)
    ax.plot(leads, _series("mslp", "bias"), "o-", color="#1f77b4", label="MSLP Bias (hPa)")
    ax.plot(leads, _series("mslp", "rmse"), "s--", color="#ff7f0e", label="MSLP RMSE (hPa)")
    ax.set_title("Luchtdruk op zeeniveau (hPa)")
    ax.legend(fontsize=8)

    # 3. RH2m
    ax = ax_list[2]
    ax.axhline(0.0, color="black", linewidth=0.7)
    ax.plot(leads, _series("rh", "bias"), "o-", color="#2ca02c", label="RH Bias (%)")
    ax.plot(leads, _series("rh", "rmse"), "s--", color="#9467bd", label="RH RMSE (%)")
    ax.set_title("Relatieve Vochtigheid 2m (%)")
    ax.legend(fontsize=8)

    # 4. Wind speed & direction
    ax = ax_list[3]
    ax.axhline(0.0, color="black", linewidth=0.7)
    ax.plot(leads, _series("ws10", "bias"), "o-", label="10m Wind Bias (m/s)")
    ax.plot(leads, _series("ws10", "rmse"), "s--", label="10m Wind RMSE (m/s)")
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
        f"Tempest-NWP ({domain_name}) — Uurlijkse KNMI 10-min Verificatie ({scope_lbl})\n"
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
    max_lead_hours: Optional[float] = None,
    independent_after_hour: Optional[float] = None,
    hourly_precip_6x: bool = False,
    land_only_plots: bool = False,
    make_maps: bool = True,
    fetcher: Optional[Callable[[datetime.datetime, Path | str], Optional[Path]]] = None,
) -> Dict[str, Any]:
    """Verify every available forecast `.wx1` step of a run against KNMI 10-minute observations."""
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

    fetch_fn = fetcher if fetcher is not None else (lambda dt, cdir: fetch_knmi_10min_netcdf(dt, cdir, api_key=api_key))

    all_pairs_all: List[Dict[str, Any]] = []
    all_pairs_land: List[Dict[str, Any]] = []
    indep_pairs_all: List[Dict[str, Any]] = []
    indep_pairs_land: List[Dict[str, Any]] = []
    hourly_records: List[Dict[str, Any]] = []
    station_buckets: Dict[str, List[Dict[str, Any]]] = {}

    prev_tot_precip: Optional[np.ndarray] = None
    prev_lead_h: Optional[float] = None
    domain_name = "tempest_lam"
    map_paths: List[Path] = []
    skipped_hours: List[Dict[str, Any]] = []

    for lead_h, valid_dt, wx1_path in items:
        nc_path = fetch_fn(valid_dt, eff_cache)
        if nc_path is None:
            skipped_hours.append({"lead_hours": lead_h, "valid_time_utc": valid_dt.isoformat(), "reason": "no_knmi_obs"})
            # Advance previous precipitation tracker so later hourly deltas stay exact
            with wx1.open_dataset(wx1_path) as ds_p:
                r = np.asarray(ds_p["rain_acc"].values, dtype=np.float64) if "rain_acc" in ds_p else 0.0
                s = np.asarray(ds_p["snow_acc"].values, dtype=np.float64) if "snow_acc" in ds_p else 0.0
                g = np.asarray(ds_p["graupel_acc"].values, dtype=np.float64) if "graupel_acc" in ds_p else 0.0
                if isinstance(r, np.ndarray):
                    prev_tot_precip = np.maximum(0.0, r + s + g)
                    prev_lead_h = lead_h
            continue

        stations = load_knmi_10min_stations(nc_path)
        obs_1h_map: Dict[str, float] = {}
        if hourly_precip_6x and prev_tot_precip is not None:
            obs_1h_map = load_knmi_hourly_precip_mm(valid_dt, eff_cache, api_key=api_key, fetcher=fetcher)

        dt_h = (lead_h - prev_lead_h) if (prev_lead_h is not None and lead_h > prev_lead_h) else 1.0
        matched, grid = sample_wx1_at_stations(
            wx1_path,
            stations,
            prev_total_precip=prev_tot_precip,
            interval_h=dt_h,
        )
        domain_name = grid["domain_name"]
        prev_tot_precip = grid["tot_precip_mm"]
        prev_lead_h = lead_h

        for row in matched:
            row["lead_hours"] = lead_h
            row["valid_time_utc"] = valid_dt.isoformat()
            if obs_1h_map and row["station_id"] in obs_1h_map and lead_h > 0.0:
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

        rec = {
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
        hourly_records.append(rec)

        if make_maps:
            tag = f"{int(round(lead_h)):03d}" if abs(lead_h - round(lead_h)) < 1e-4 else f"{lead_h:06.2f}"
            mp = plot_hourly_verification_map(
                maps_dir / f"knmi_map_T+{tag}h.png",
                lead_h,
                valid_dt,
                grid,
                matched,
                land_stats if land_only_plots else all_stats,
                land_only=land_only_plots,
            )
            map_paths.append(mp)

    # Hourly CSV export
    hourly_csv_rows: List[Dict[str, Any]] = []
    for rec in hourly_records:
        st_b = rec["land"]
        gp = rec["grid_precip"]
        hourly_csv_rows.append(
            {
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
                "land_wdir_mae_deg": round(st_b["wdir"]["mae_deg"], 1),
                "land_fog_hits": st_b["fog"]["hits"],
                "land_fog_misses": st_b["fog"]["misses"],
                "land_fog_fa": st_b["fog"]["false_alarms"],
                "grid_nl_precip_max_mmh": round(gp["nl_max_mmh"], 2),
                "grid_int_precip_max_mmh": round(gp["int_max_mmh"], 2),
                "grid_rim_precip_max_mmh": round(gp["rim_max_mmh"], 2),
            }
        )
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

    ts_png: Optional[Path] = None
    if hourly_records:
        ts_png = plot_verification_timeseries(
            eff_outdir / "overview_timeseries.png",
            hourly_records,
            independent_after_hour=indep_h,
            domain_name=domain_name,
            use_land_stats=True,
        )

    # Strip large per-station lists before JSON serialization
    hourly_json = [
        {k: v for k, v in rec.items() if k != "obs_rows_land"}
        for rec in hourly_records
    ]
    summary: Dict[str, Any] = {
        "domain_name": domain_name,
        "input_path": str(root),
        "outdir": str(eff_outdir),
        "independent_after_hour": indep_h,
        "verified_hours": len(hourly_records),
        "skipped_hours": skipped_hours,
        "overall_all": _summarize_matched_rows(all_pairs_all),
        "overall_land": _summarize_matched_rows(all_pairs_land),
        "independent_all": _summarize_matched_rows(indep_pairs_all),
        "independent_land": _summarize_matched_rows(indep_pairs_land),
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
