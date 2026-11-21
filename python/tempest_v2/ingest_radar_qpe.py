"""Classical Live Radar QPE Ingestor (KNMI RTCOR-5m & DWD RADOLAN RW) -> WX-1.

Strictly 100% deterministic geospatial decoding & reprojection (0% AI / 0% ML, 0% mock data):
- KNMI Open Data Platform (`nl_rdr_data_rtcor_5m` v1.0 HDF5, 765x700 @ 1 km polar stereographic).
- DWD Open Data RADOLAN RW (`raa01-rw_10000-*-dwd---bin.gz`, 900x900 @ 1 km polar stereographic).
- Outputs lossless `zstd_shuffle` WX-1 container `radar_obs.wx1` (`rain_rate_mm_h` [ny, nx], float32, mm/h)
  ready for Stefan et al. (2008) Latent Heat Nudging (`compute_lhn_tendencies`).
"""

from __future__ import annotations

import datetime
import gzip
import json
import logging
import re
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import h5py
import numpy as np
import pyproj
from scipy.ndimage import map_coordinates
import wx1

from tempest_v2.ingest_knmi_observations import KNMI_API_BASE, get_knmi_api_key
from tempest_v2.projection import DomainSpec

logger = logging.getLogger("tempest_v2.ingest_radar_qpe")

KNMI_RADAR_DATASET = "nl_rdr_data_rtcor_5m"
KNMI_RADAR_VERSION = "1.0"
DWD_RADOLAN_RW_BASE = "https://opendata.dwd.de/climate_environment/CDC/grids_germany/hourly/radolan/recent/bin/"

KNMI_RADAR_PROJ4_DEFAULT = (
    "+proj=stere +lat_0=90 +lon_0=0 +lat_ts=60 +a=6378137 +b=6356752 +x_0=0 +y_0=0 +units=km"
)
# Official DWD RADOLAN polar stereographic sphere (R = 6370.04 km, true at 60N, lon_0=10E)
DWD_RADOLAN_PROJ4 = "+proj=stere +lat_0=90 +lat_ts=60 +lon_0=10 +a=6370040 +b=6370040 +units=km"
DWD_RADOLAN_X0_KM = -523.4622
DWD_RADOLAN_Y0_KM = -4658.6447


def _reproject_radar_to_domain(
    src_rain_mm_h: np.ndarray,
    src_proj4: str,
    x_origin_km: float,
    y_origin_km: float,
    dx_km: float,
    dy_km: float,
    spec: DomainSpec,
    *,
    y_ascending: bool = True,
    geo_crs: str = "EPSG:4326",
) -> np.ndarray:
    """Bilinearly sample radar rain rate [mm/h] onto `spec` preserving NaN outside radar coverage."""
    transformer = pyproj.Transformer.from_crs(geo_crs, src_proj4, always_xy=True)
    x_km, y_km = transformer.transform(spec.lon2d, spec.lat2d)

    cols = (x_km - x_origin_km) / dx_km - 0.5
    if y_ascending:
        rows = (y_km - y_origin_km) / dy_km - 0.5
    else:
        rows = (y_origin_km - y_km) / abs(dy_km) - 0.5

    ny_s, nx_s = src_rain_mm_h.shape
    in_bounds = (rows >= 0.0) & (rows <= ny_s - 1.0) & (cols >= 0.0) & (cols <= nx_s - 1.0)

    valid_mask = np.isfinite(src_rain_mm_h)
    clean_src = np.where(valid_mask, np.maximum(0.0, src_rain_mm_h), 0.0).astype(np.float64)
    w_src = valid_mask.astype(np.float64)

    coords = np.array([np.clip(rows.ravel(), 0.0, ny_s - 1.0), np.clip(cols.ravel(), 0.0, nx_s - 1.0)])
    val_interp = map_coordinates(clean_src, coords, order=1, mode="nearest").reshape(spec.ny, spec.nx)
    w_interp = map_coordinates(w_src, coords, order=1, mode="nearest").reshape(spec.ny, spec.nx)

    out = np.full((spec.ny, spec.nx), np.nan, dtype=np.float32)
    ok = in_bounds & (w_interp > 0.5)
    out[ok] = np.clip(val_interp[ok] / w_interp[ok], 0.0, 250.0).astype(np.float32)
    return out


def decode_knmi_radar_hdf5(h5_path: Path | str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Decode KNMI RTCOR-5m HDF5 into 2D rain rate [mm/h] and projection metadata."""
    with h5py.File(Path(h5_path), "r") as hf:
        raw = np.asarray(hf["image1/image_data"][:], dtype=np.float64)
        missing_val = 65534.0
        out_val = 65535.0
        scale = 0.01
        offset = 0.0
        if "image1/calibration" in hf:
            cattrs = hf["image1/calibration"].attrs
            if "calibration_missing_data" in cattrs:
                missing_val = float(np.asarray(cattrs["calibration_missing_data"]).ravel()[0])
            if "calibration_out_of_image" in cattrs:
                out_val = float(np.asarray(cattrs["calibration_out_of_image"]).ravel()[0])
            formula = cattrs.get("calibration_formulas", b"GEO=0.01*PV+0")
            if isinstance(formula, bytes):
                formula = formula.decode("ascii", errors="ignore")
            m = re.search(r"([0-9eE.+_-]+)\s*\*\s*PV\s*([+-][0-9eE.+_]+)?", str(formula))
            if m:
                scale = float(m.group(1))
                offset = float(m.group(2)) if m.group(2) else 0.0

        proj4 = KNMI_RADAR_PROJ4_DEFAULT
        if "geographic/map_projection" in hf:
            p_raw = hf["geographic/map_projection"].attrs.get("projection_proj4_params")
            if p_raw is not None:
                proj4 = p_raw.decode("ascii", errors="ignore") if isinstance(p_raw, bytes) else str(p_raw)

        gattrs = hf["geographic"].attrs if "geographic" in hf else {}
        col_off = float(np.asarray(gattrs.get("geo_column_offset", [0.0])).ravel()[0])
        row_off = float(np.asarray(gattrs.get("geo_row_offset", [3650.0])).ravel()[0])
        px_x = float(np.asarray(gattrs.get("geo_pixel_size_x", [1.0])).ravel()[0])
        px_y = float(np.asarray(gattrs.get("geo_pixel_size_y", [-1.0])).ravel()[0])

        # Convert 5-minute accumulation [mm] to instantaneous rate [mm/h]
        rain_mm_h = np.full(raw.shape, np.nan, dtype=np.float64)
        valid = np.isfinite(raw) & (raw < min(missing_val, out_val)) & (raw >= 0.0)
        rain_mm_h[valid] = np.maximum(0.0, (scale * raw[valid] + offset) * 12.0)

        meta = {
            "proj4": proj4,
            "x_origin_km": col_off,
            "y_origin_km": -row_off if row_off > 0 else row_off,
            "dx_km": abs(px_x),
            "dy_km": abs(px_y),
            "y_ascending": False,
        }
        return rain_mm_h, meta


def fetch_knmi_radar_qpe_5m(
    spec: DomainSpec,
    *,
    api_key: Optional[str] = None,
    cache_dir: Path | str = "data/cache/radar",
) -> Optional[Tuple[np.ndarray, str]]:
    """Fetch live KNMI RTCOR-5m radar QPE and reproject onto `spec` [mm/h]."""
    key = get_knmi_api_key(api_key)
    if not key:
        return None
    cdir = Path(cache_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": key, "User-Agent": "Tempest-NWP/1.0"}
    list_url = (
        f"{KNMI_API_BASE}/datasets/{KNMI_RADAR_DATASET}/versions/{KNMI_RADAR_VERSION}/files"
        "?maxKeys=2&orderBy=created&sorting=desc"
    )
    req = urllib.request.Request(list_url, headers=headers)
    with urllib.request.urlopen(req, timeout=12) as resp:
        files = json.loads(resp.read().decode("utf-8")).get("files", [])
    if not files:
        return None
    fname = files[0]["filename"]
    local_h5 = cdir / fname
    if not local_h5.exists() or local_h5.stat().st_size == 0:
        dl_api = (
            f"{KNMI_API_BASE}/datasets/{KNMI_RADAR_DATASET}/versions/{KNMI_RADAR_VERSION}"
            f"/files/{fname}/url"
        )
        dl_url = json.loads(
            urllib.request.urlopen(urllib.request.Request(dl_api, headers=headers), timeout=12)
            .read()
            .decode("utf-8")
        )["temporaryDownloadUrl"]
        local_h5.write_bytes(urllib.request.urlopen(urllib.request.Request(dl_url), timeout=25).read())

    rain_src, meta = decode_knmi_radar_hdf5(local_h5)
    grid = _reproject_radar_to_domain(
        rain_src,
        meta["proj4"],
        meta["x_origin_km"],
        meta["y_origin_km"],
        meta["dx_km"],
        meta["dy_km"],
        spec,
        y_ascending=bool(meta["y_ascending"]),
    )
    return grid, fname


def parse_dwd_radolan_rw_bytes(raw_bytes: bytes) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Decode uncompressed DWD RADOLAN RW binary payload into 2D rain rate [mm/h]."""
    etx_pos = raw_bytes.index(0x03)
    hdr = raw_bytes[:etx_pos].decode("ascii", errors="replace")
    payload = raw_bytes[etx_pos + 1 :]

    ny, nx = 900, 900
    m_gp = re.search(r"GP\s*(\d+)x\s*(\d+)", hdr)
    if m_gp:
        ny, nx = int(m_gp.group(1)), int(m_gp.group(2))

    # Header precision factor: E-01 => 0.1 mm/h, E-02 => 0.01 mm/h
    prec = 0.1
    m_pr = re.search(r"PR\s*E([+-]?\d+)", hdr)
    if m_pr:
        prec = 10.0 ** int(m_pr.group(1))

    int_min = 60.0
    m_int = re.search(r"INT\s*(\d+)", hdr)
    if m_int:
        int_min = max(1.0, float(m_int.group(1)))

    arr_u16 = np.frombuffer(payload, dtype="<u2").reshape(ny, nx)
    nodata = (arr_u16 & 0x2000) != 0
    raw_val = (arr_u16 & 0x0FFF).astype(np.float64)

    rain_mm_h = np.full((ny, nx), np.nan, dtype=np.float64)
    rate_scale = prec * (60.0 / int_min)
    rain_mm_h[~nodata] = raw_val[~nodata] * rate_scale
    return rain_mm_h, {"header": hdr, "ny": ny, "nx": nx, "precision": prec, "interval_min": int_min}


def fetch_dwd_radolan_rw(
    spec: DomainSpec,
    *,
    cache_dir: Path | str = "data/cache/radar",
) -> Optional[Tuple[np.ndarray, str]]:
    """Fetch latest DWD RADOLAN RW composite from DWD Open Data and reproject onto `spec`."""
    cdir = Path(cache_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    # Try direct candidate timestamps for the past 4 hours first (fast HTTP GET vs 20 MB HTML index)
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    raw_gz: Optional[bytes] = None
    fname: Optional[str] = None
    for h_back in range(0, 5):
        t_cand = now_utc - datetime.timedelta(hours=h_back)
        for minute_tag in ("50", "10"):
            cand_name = f"raa01-rw_10000-{t_cand.strftime('%y%m%d%H')}{minute_tag}-dwd---bin.gz"
            cand_path = cdir / cand_name
            if cand_path.is_file() and cand_path.stat().st_size > 1000:
                raw_gz = cand_path.read_bytes()
                fname = cand_name
                break
            url = f"{DWD_RADOLAN_RW_BASE}{cand_name}"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Tempest-NWP/1.0"})
                with urllib.request.urlopen(req, timeout=8) as resp:
                    raw_gz = resp.read()
                cand_path.write_bytes(raw_gz)
                fname = cand_name
                break
            except Exception:
                continue
        if raw_gz is not None:
            break

    if raw_gz is None:
        req = urllib.request.Request(DWD_RADOLAN_RW_BASE, headers={"User-Agent": "Tempest-NWP/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
        matches = sorted(re.findall(r"raa01-rw_10000-\d{10}-dwd---bin\.gz", html))
        if not matches:
            return None
        fname = matches[-1]
        cand_path = cdir / fname
        if not cand_path.is_file() or cand_path.stat().st_size == 0:
            raw_gz = urllib.request.urlopen(
                urllib.request.Request(f"{DWD_RADOLAN_RW_BASE}{fname}", headers={"User-Agent": "Tempest-NWP/1.0"}),
                timeout=15,
            ).read()
            cand_path.write_bytes(raw_gz)
        else:
            raw_gz = cand_path.read_bytes()

    rain_src, _ = parse_dwd_radolan_rw_bytes(gzip.decompress(raw_gz))
    grid = _reproject_radar_to_domain(
        rain_src,
        DWD_RADOLAN_PROJ4,
        DWD_RADOLAN_X0_KM,
        DWD_RADOLAN_Y0_KM,
        1.0,
        1.0,
        spec,
        y_ascending=True,
        geo_crs="+proj=longlat +a=6370040 +b=6370040 +no_defs",
    )
    return grid, str(fname)


def ingest_radar_qpe_to_wx1(
    out_dir: Path | str,
    spec: DomainSpec,
    *,
    api_key: Optional[str] = None,
    cache_dir: Path | str = "data/cache/radar",
) -> Optional[Path]:
    """Download KNMI RTCOR-5m + DWD RADOLAN RW composites, reproject onto `spec`, and save `radar_obs.wx1`."""
    odir = Path(out_dir)
    odir.mkdir(parents=True, exist_ok=True)

    knmi_grid: Optional[np.ndarray] = None
    dwd_grid: Optional[np.ndarray] = None
    sources: list[str] = []

    try:
        res_k = fetch_knmi_radar_qpe_5m(spec, api_key=api_key, cache_dir=cache_dir)
        if res_k is not None:
            knmi_grid, k_name = res_k
            sources.append(f"KNMI:{k_name}")
    except Exception as exc:
        logger.warning("KNMI radar QPE ingest notice: %s", exc)

    try:
        res_d = fetch_dwd_radolan_rw(spec, cache_dir=cache_dir)
        if res_d is not None:
            dwd_grid, d_name = res_d
            sources.append(f"DWD:{d_name}")
    except Exception as exc:
        logger.warning("DWD RADOLAN QPE ingest notice: %s", exc)

    if knmi_grid is None and dwd_grid is None:
        return None

    if knmi_grid is not None and dwd_grid is not None:
        # Prefer KNMI radar over NL/BE, fill outside/masked areas with DWD RADOLAN over DE/border
        composite = np.where(np.isfinite(knmi_grid), knmi_grid, dwd_grid)
    elif knmi_grid is not None:
        composite = knmi_grid
    else:
        composite = dwd_grid

    composite_clean = np.where(np.isfinite(composite), np.maximum(0.0, composite), 0.0).astype(np.float32)
    out_path = odir / "radar_obs.wx1"
    wx1.write_array(
        out_path,
        composite_clean,
        dataset_id=f"radar_qpe_{spec.name}",
        variable_path="rain_rate_mm_h",
        dimensions=["y", "x"],
        profile="output",
        codec="zstd_shuffle",
        attributes={
            "units": "mm h-1",
            "sources": ",".join(sources),
            "domain": spec.name,
            "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        },
    )
    logger.info(
        "Saved composite radar QPE (%s) to WX-1: %s (max=%.2f mm/h)",
        ",".join(sources),
        out_path,
        float(composite_clean.max()),
    )
    return out_path
