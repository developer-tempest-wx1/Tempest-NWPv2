"""Deterministic ESA WorldCover 10 m & ISRIC SoilGrids 250 m Static Surface Engine.

Builds seamless pan-European land-cover (`z0m_veg`, `albedo`, `veg_frac`, DSM canopy/building
height offset) and soil texture / organic peat (class 14) fields on any Tempest-NWP `DomainSpec`,
cached in lossless `zstd_shuffle` WX-1 containers.

Strictly 100% classical deterministic geophysical preprocessing (0% AI / 0% ML), SI units only.
"""

from __future__ import annotations

import logging
import math
import os
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt
import xarray as xr

import wx1

from tempest_v2.projection import DomainSpec

logger = logging.getLogger("tempest_v2.land_static")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = PROJECT_ROOT / "data" / "static"

_WORLDCOVER_SCHEMA_VERSION = "2"
_SOILGRIDS_SCHEMA_VERSION = "1"

# ESA WorldCover 2021 v200 discrete class codes -> physical SI parameters:
# code: (z0m_m, albedo, veg_frac, dsm_height_offset_m)
WORLDCOVER_CLASS_PHYSICS: Dict[int, Tuple[float, float, float, float]] = {
    10: (1.00, 0.13, 0.90, 6.0),    # Tree cover
    20: (0.20, 0.18, 0.60, 0.5),    # Shrubland
    30: (0.03, 0.21, 0.80, 0.0),    # Grassland
    40: (0.08, 0.20, 0.65, 0.0),    # Cropland
    50: (0.70, 0.16, 0.15, 1.5),    # Built-up
    60: (0.015, 0.28, 0.05, 0.0),   # Bare / sparse vegetation
    70: (0.002, 0.72, 0.00, 0.0),   # Snow and ice
    80: (0.0001, 0.06, 0.00, 0.0),  # Permanent water bodies
    90: (0.05, 0.15, 0.75, 0.0),    # Herbaceous wetland
    95: (0.80, 0.14, 0.85, 4.0),    # Mangroves
    100: (0.02, 0.22, 0.50, 0.0),   # Moss and lichen
}

_WC_TILE_RE = re.compile(r"_([NS])(\d{2})([EW])(\d{3})_Map\.tif$")


def resolve_worldcover_root(local_root: Optional[Path | str] = None) -> Optional[Path]:
    """Locate ESA WorldCover 2021 GeoTIFF directory if available locally."""
    candidates = []
    if local_root is not None:
        candidates.append(Path(local_root))
    env_dir = os.environ.get("TEMPEST_WORLDCOVER_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.append(STATIC_DIR / "esa_worldcover_2021")
    for cand in candidates:
        if cand.exists() and any(cand.glob("ESA_WorldCover_*.tif")):
            return cand
    return None


def resolve_soilgrids_root(local_root: Optional[Path | str] = None) -> Optional[Path]:
    """Locate SoilGrids 250 m GeoTIFF directory if available locally."""
    candidates = []
    if local_root is not None:
        candidates.append(Path(local_root))
    env_dir = os.environ.get("TEMPEST_SOILGRIDS_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.append(STATIC_DIR / "soilgrids_250m")
    for cand in candidates:
        if cand.exists() and (cand / "SoilGrids_clay_0-5cm_Q0p5.tif").is_file():
            return cand
    return None


def _domain_latlon_bbox(spec: DomainSpec, pad_deg: float = 0.05) -> Tuple[float, float, float, float]:
    """Return `(lon_min, lat_min, lon_max, lat_max)` in `[-180, 180]` padded by `pad_deg`."""
    lat = np.asarray(spec.lat2d, dtype=np.float64)
    lon = ((np.asarray(spec.lon2d, dtype=np.float64) + 180.0) % 360.0) - 180.0
    lat_min = max(-90.0, float(lat.min()) - float(pad_deg))
    lat_max = min(90.0, float(lat.max()) + float(pad_deg))
    if spec.is_periodic_lon or (float(lon.max()) - float(lon.min())) > 300.0:
        return -180.0, lat_min, 180.0, lat_max
    lon_min = max(-180.0, float(lon.min()) - float(pad_deg))
    lon_max = min(180.0, float(lon.max()) + float(pad_deg))
    return lon_min, lat_min, lon_max, lat_max


def _latlon_to_grid_flat_idx(
    spec: DomainSpec,
    lat_pts: np.ndarray,
    lon_pts: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Map 2D/1D geographic `(lat, lon)` points to in-bounds flattened `(y * nx + x)` indices."""
    ny, nx = spec.ny, spec.nx
    if spec.is_projected and spec.projection is not None:
        x_m, y_m = spec.projection.latlon_to_xy(lat_pts, lon_pts)
        ii = np.rint((x_m - float(spec.x_m[0])) / float(spec.dx_m)).astype(np.int64)
        jj = np.rint((y_m - float(spec.y_m[0])) / float(spec.dy_m)).astype(np.int64)
    else:
        lat_1d = np.asarray(spec.lat2d[:, 0], dtype=np.float64)
        lon_1d = np.asarray(spec.lon2d[0, :], dtype=np.float64)
        lon_q = np.mod(lon_pts, 360.0) if float(lon_1d.max()) > 180.0 else lon_pts
        dlat = (float(lat_1d[-1]) - float(lat_1d[0])) / max(len(lat_1d) - 1, 1)
        dlon = (float(lon_1d[-1]) - float(lon_1d[0])) / max(len(lon_1d) - 1, 1)
        jj = np.rint((lat_pts - float(lat_1d[0])) / dlat).astype(np.int64)
        ii = np.rint((lon_q - float(lon_1d[0])) / dlon).astype(np.int64)
        if spec.is_periodic_lon:
            ii = np.mod(ii, nx)
    valid = (jj >= 0) & (jj < ny) & (ii >= 0) & (ii < nx)
    flat = (jj[valid] * nx + ii[valid]).astype(np.int64)
    return flat, valid


def _window_latlon_grid(transform, col_indices: np.ndarray, row_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute 2D `(lat, lon)` arrays at pixel centers `(row_indices, col_indices)` for an affine GeoTIFF."""
    if abs(float(transform.b)) < 1e-12 and abs(float(transform.d)) < 1e-12:
        lon_1d = float(transform.c) + (col_indices + 0.5) * float(transform.a)
        lat_1d = float(transform.f) + (row_indices + 0.5) * float(transform.e)
        return lat_1d[:, None], lon_1d[None, :]
    c2, r2 = np.meshgrid(col_indices + 0.5, row_indices + 0.5)
    lon2, lat2 = transform * (c2, r2)
    return np.asarray(lat2, dtype=np.float64), np.asarray(lon2, dtype=np.float64)


def _worldcover_luts(z_blend_m: float) -> Dict[str, np.ndarray]:
    """Build 256-entry uint8 lookup tables for fast vectorized WorldCover block aggregation."""
    lut_valid = np.zeros(256, dtype=np.bool_)
    lut_land = np.zeros(256, dtype=np.bool_)
    lut_water = np.zeros(256, dtype=np.bool_)
    lut_tree = np.zeros(256, dtype=np.bool_)
    lut_built = np.zeros(256, dtype=np.bool_)
    lut_cd = np.zeros(256, dtype=np.float32)
    lut_alb = np.zeros(256, dtype=np.float32)
    lut_veg = np.zeros(256, dtype=np.float32)
    lut_dh = np.zeros(256, dtype=np.float32)

    zb = max(float(z_blend_m), 5.0)
    for code, (z0m, alb, veg, dh) in WORLDCOVER_CLASS_PHYSICS.items():
        lut_valid[code] = True
        if code == 80:
            lut_water[code] = True
        else:
            lut_land[code] = True
            lut_cd[code] = float(1.0 / (math.log(zb / max(z0m, 1e-4)) ** 2))
            lut_alb[code] = float(alb)
            lut_veg[code] = float(veg)
            lut_dh[code] = float(dh)
        if code == 10:
            lut_tree[code] = True
        elif code == 50:
            lut_built[code] = True

    return {
        "valid": lut_valid,
        "land": lut_land,
        "water": lut_water,
        "tree": lut_tree,
        "built": lut_built,
        "cd": lut_cd,
        "alb": lut_alb,
        "veg": lut_veg,
        "dh": lut_dh,
    }


def sample_worldcover_on_grid(
    spec: DomainSpec,
    local_root: Optional[Path | str] = None,
    *,
    z_blend_m: float = 50.0,
) -> Dict[str, np.ndarray]:
    """Aggregate 10 m ESA WorldCover 2021 tiles onto `spec` via drag-conserving blending-height averaging."""
    import rasterio
    from rasterio.windows import Window, from_bounds

    wc_root = resolve_worldcover_root(local_root)
    if wc_root is None:
        raise FileNotFoundError("ESA WorldCover 2021 directory not found locally.")

    lon_min, lat_min, lon_max, lat_max = _domain_latlon_bbox(spec, pad_deg=0.05)
    tiles = []
    for path in sorted(wc_root.glob("ESA_WorldCover_*.tif")):
        m = _WC_TILE_RE.search(path.name)
        if m:
            la = int(m.group(2)) * (1 if m.group(1) == "N" else -1)
            lo = int(m.group(4)) * (1 if m.group(3) == "E" else -1)
            if lo + 3.0 < lon_min or lo > lon_max or la + 3.0 < lat_min or la > lat_max:
                continue
        tiles.append(path)

    ny, nx = spec.ny, spec.nx
    n_cells = ny * nx
    sum_tot = np.zeros(n_cells, dtype=np.float64)
    sum_land = np.zeros(n_cells, dtype=np.float64)
    sum_water = np.zeros(n_cells, dtype=np.float64)
    sum_tree = np.zeros(n_cells, dtype=np.float64)
    sum_built = np.zeros(n_cells, dtype=np.float64)
    sum_cd = np.zeros(n_cells, dtype=np.float64)
    sum_alb = np.zeros(n_cells, dtype=np.float64)
    sum_veg = np.zeros(n_cells, dtype=np.float64)
    sum_dh = np.zeros(n_cells, dtype=np.float64)

    luts = _worldcover_luts(z_blend_m)
    block = 10
    strip_rows = 500

    for tile_path in tiles:
        with rasterio.open(tile_path) as ds:
            b = ds.bounds
            x0, y0 = max(lon_min, float(b.left)), max(lat_min, float(b.bottom))
            x1, y1 = min(lon_max, float(b.right)), min(lat_max, float(b.top))
            if x1 <= x0 or y1 <= y0:
                continue
            win = from_bounds(x0, y0, x1, y1, ds.transform).round_offsets().round_lengths()
            col_off = max(0, int(win.col_off))
            row_off = max(0, int(win.row_off))
            width = min(ds.width - col_off, int(win.width))
            height = min(ds.height - row_off, int(win.height))
            width -= width % block
            height -= height % block
            if width < block or height < block:
                continue
            bw = width // block
            col_centers = col_off + np.arange(bw, dtype=np.float64) * block + 0.5 * (block - 1)

            for r_rel in range(0, height, strip_rows):
                h_cur = min(strip_rows, height - r_rel)
                h_cur -= h_cur % block
                if h_cur < block:
                    continue
                bh = h_cur // block
                r_abs = row_off + r_rel
                strip = ds.read(1, window=Window(col_off, r_abs, width, h_cur))
                row_centers = r_abs + np.arange(bh, dtype=np.float64) * block + 0.5 * (block - 1)
                lat_c, lon_c = _window_latlon_grid(ds.transform, col_centers, row_centers)
                if lat_c.shape != (bh, bw):
                    lat_c = np.broadcast_to(lat_c, (bh, bw))
                    lon_c = np.broadcast_to(lon_c, (bh, bw))
                flat_idx, in_dom = _latlon_to_grid_flat_idx(spec, lat_c, lon_c)
                if flat_idx.size == 0:
                    continue

                blk_valid = luts["valid"][strip].reshape(bh, block, bw, block).sum(axis=(1, 3), dtype=np.uint16)
                blk_land = luts["land"][strip].reshape(bh, block, bw, block).sum(axis=(1, 3), dtype=np.uint16)
                blk_water = luts["water"][strip].reshape(bh, block, bw, block).sum(axis=(1, 3), dtype=np.uint16)
                blk_tree = luts["tree"][strip].reshape(bh, block, bw, block).sum(axis=(1, 3), dtype=np.uint16)
                blk_built = luts["built"][strip].reshape(bh, block, bw, block).sum(axis=(1, 3), dtype=np.uint16)
                blk_cd = luts["cd"][strip].reshape(bh, block, bw, block).sum(axis=(1, 3), dtype=np.float32)
                blk_alb = luts["alb"][strip].reshape(bh, block, bw, block).sum(axis=(1, 3), dtype=np.float32)
                blk_veg = luts["veg"][strip].reshape(bh, block, bw, block).sum(axis=(1, 3), dtype=np.float32)
                blk_dh = luts["dh"][strip].reshape(bh, block, bw, block).sum(axis=(1, 3), dtype=np.float32)

                sum_tot += np.bincount(flat_idx, weights=blk_valid[in_dom], minlength=n_cells)
                sum_land += np.bincount(flat_idx, weights=blk_land[in_dom], minlength=n_cells)
                sum_water += np.bincount(flat_idx, weights=blk_water[in_dom], minlength=n_cells)
                sum_tree += np.bincount(flat_idx, weights=blk_tree[in_dom], minlength=n_cells)
                sum_built += np.bincount(flat_idx, weights=blk_built[in_dom], minlength=n_cells)
                sum_cd += np.bincount(flat_idx, weights=blk_cd[in_dom], minlength=n_cells)
                sum_alb += np.bincount(flat_idx, weights=blk_alb[in_dom], minlength=n_cells)
                sum_veg += np.bincount(flat_idx, weights=blk_veg[in_dom], minlength=n_cells)
                sum_dh += np.bincount(flat_idx, weights=blk_dh[in_dom], minlength=n_cells)

    sum_tot = sum_tot.reshape(ny, nx)
    sum_land = sum_land.reshape(ny, nx)
    sum_water = sum_water.reshape(ny, nx)
    sum_tree = sum_tree.reshape(ny, nx)
    sum_built = sum_built.reshape(ny, nx)
    sum_cd = sum_cd.reshape(ny, nx)
    sum_alb = sum_alb.reshape(ny, nx)
    sum_veg = sum_veg.reshape(ny, nx)
    sum_dh = sum_dh.reshape(ny, nx)

    has_tile = sum_tot > 0.0
    has_land = sum_land > 0.0
    denom_land = np.maximum(sum_land, 1.0)
    denom_tot = np.maximum(sum_tot, 1.0)

    cd_fallback = 1.0 / (math.log(max(float(z_blend_m), 5.0) / 0.15) ** 2)
    cd_mean = np.where(has_land, sum_cd / denom_land, cd_fallback)
    z0m_veg = np.where(
        has_land,
        np.clip(float(z_blend_m) * np.exp(-1.0 / np.sqrt(np.maximum(cd_mean, 1e-6))), 0.002, 1.5),
        1.0e-4,
    )

    return {
        "wc_valid": has_tile.astype(np.float32),
        "wc_land_frac": np.where(has_tile, sum_land / denom_tot, 0.0).astype(np.float32),
        "z0m_veg": z0m_veg.astype(np.float32),
        "albedo_land": np.where(has_land, np.clip(sum_alb / denom_land, 0.05, 0.85), 0.19).astype(np.float32),
        "veg_frac": np.where(has_land, np.clip(sum_veg / denom_land, 0.0, 1.0), 0.0).astype(np.float32),
        "dsm_height_offset_m": np.where(has_land, np.clip(sum_dh / denom_land, 0.0, 20.0), 0.0).astype(np.float32),
        "tree_frac": np.where(has_land, np.clip(sum_tree / denom_land, 0.0, 1.0), 0.0).astype(np.float32),
        "built_frac": np.where(has_land, np.clip(sum_built / denom_land, 0.0, 1.0), 0.0).astype(np.float32),
        "water_frac": np.where(has_tile, np.clip(sum_water / denom_tot, 0.0, 1.0), 1.0).astype(np.float32),
    }


def ensure_domain_worldcover_static(
    spec: DomainSpec,
    *,
    output_dir: Path | str = STATIC_DIR,
    local_root: Optional[Path | str] = None,
    z_blend_m: float = 50.0,
    force: bool = False,
) -> Path:
    """Build or load cached ESA WorldCover 10 m WX-1 container for `spec`."""
    out_path = Path(output_dir) / f"landcover_worldcover_{spec.name}.wx1"
    if out_path.is_file() and not force:
        with wx1.open_dataset(out_path) as ds:
            if ds.attrs.get("worldcover_schema_version") == _WORLDCOVER_SCHEMA_VERSION:
                return out_path

    fields = sample_worldcover_on_grid(spec, local_root=local_root, z_blend_m=z_blend_m)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    y_dim, x_dim = ("y", "x") if spec.is_projected else ("latitude", "longitude")
    unit_y, unit_x = ("m", "m") if spec.is_projected else ("degrees_north", "degrees_east")
    meta_units = {
        "wc_valid": "1",
        "wc_land_frac": "1",
        "z0m_veg": "m",
        "albedo_land": "1",
        "veg_frac": "1",
        "dsm_height_offset_m": "m",
        "tree_frac": "1",
        "built_frac": "1",
        "water_frac": "1",
    }
    ds = xr.Dataset(
        data_vars={
            k: ((y_dim, x_dim), arr.astype(np.float32), {"units": meta_units[k]})
            for k, arr in fields.items()
        },
        coords={
            y_dim: (y_dim, np.asarray(spec.y_m, dtype=np.float64), {"units": unit_y}),
            x_dim: (x_dim, np.asarray(spec.x_m, dtype=np.float64), {"units": unit_x}),
            "lat2d": ((y_dim, x_dim), np.asarray(spec.lat2d, dtype=np.float64), {"units": "degrees_north"}),
            "lon2d": ((y_dim, x_dim), np.asarray(spec.lon2d, dtype=np.float64), {"units": "degrees_east"}),
        },
        attrs={
            "domain": spec.name,
            "source": "ESA WorldCover 2021 v200 (10 m) drag-conserving blending-height aggregation",
            "z_blend_m": f"{float(z_blend_m):.1f}",
            "worldcover_schema_version": _WORLDCOVER_SCHEMA_VERSION,
            "ai_ml_usage": "0%",
        },
    )
    wx1.write_dataset(out_path, ds, dataset_id=f"worldcover_static_{spec.name}", codec="zstd_shuffle", workers=4)
    return out_path


def _read_clipped_geotiff(path: Path, lon_min: float, lat_min: float, lon_max: float, lat_max: float):
    """Read intersecting GeoTIFF window and return `(array, flat_grid_fn, window_bounds)`."""
    import rasterio
    from rasterio.windows import Window, from_bounds

    with rasterio.open(path) as ds:
        b = ds.bounds
        x0, y0 = max(lon_min, float(b.left)), max(lat_min, float(b.bottom))
        x1, y1 = min(lon_max, float(b.right)), min(lat_max, float(b.top))
        if x1 <= x0 or y1 <= y0:
            return None, None, None, b
        win = from_bounds(x0, y0, x1, y1, ds.transform).round_offsets().round_lengths()
        c0 = max(0, int(win.col_off))
        r0 = max(0, int(win.row_off))
        w = min(ds.width - c0, int(win.width))
        h = min(ds.height - r0, int(win.height))
        if w <= 0 or h <= 0:
            return None, None, None, b
        arr = ds.read(1, window=Window(c0, r0, w, h))
        lat_px, lon_px = _window_latlon_grid(
            ds.transform,
            c0 + np.arange(w, dtype=np.float64),
            r0 + np.arange(h, dtype=np.float64),
        )
        return arr, lat_px, lon_px, b


def _classify_usda_vectorized(sand_pct: np.ndarray, silt_pct: np.ndarray, clay_pct: np.ndarray) -> np.ndarray:
    """Vectorized USDA 13-class soil classification matching `hwsd22_soil._texture_class`."""
    cls = np.where(sand_pct >= 52.0, 11, 9)
    cls = np.where(silt_pct >= 50.0, np.where((silt_pct >= 80.0) & (clay_pct < 12.0), 6, 7), cls)
    cls = np.where(sand_pct >= 70.0, np.where(clay_pct >= 15.0, 11, np.where(sand_pct >= 85.0, 13, 12)), cls)
    cls = np.where(clay_pct >= 27.0, np.where(sand_pct >= 45.0, 10, np.where(silt_pct >= 28.0, 4, 5)), cls)
    cls = np.where(clay_pct >= 40.0, np.where(sand_pct >= 45.0, 8, np.where(silt_pct >= 40.0, 2, 1)), cls)
    return cls.astype(np.int32)


def sample_soilgrids_on_grid(
    spec: DomainSpec,
    lsm_land: np.ndarray,
    local_root: Optional[Path | str] = None,
) -> Dict[str, np.ndarray]:
    """Sample ISRIC SoilGrids 250 m (top 0-30 cm texture, SOC, bulk density, WRB Histosols) onto `spec`."""
    sg_root = resolve_soilgrids_root(local_root)
    if sg_root is None:
        raise FileNotFoundError("SoilGrids 250 m directory not found locally.")

    lon_min, lat_min, lon_max, lat_max = _domain_latlon_bbox(spec, pad_deg=0.05)
    ny, nx = spec.ny, spec.nx
    n_cells = ny * nx
    is_land = np.asarray(lsm_land, dtype=np.float64) >= 0.5

    # Read 0-30 cm texture layers (weights: 5 cm, 10 cm, 15 cm -> total 30 cm) on SoilGrids 250 m metric grid.
    depths_w = (("0-5cm", 5.0 / 30.0), ("5-15cm", 10.0 / 30.0), ("15-30cm", 15.0 / 30.0))
    clay_w = sand_w = silt_w = None
    c0_ref = s0_ref = z0_ref = None
    lat_sg = lon_sg = None
    sg_bounds = None

    for d_name, w_d in depths_w:
        c_arr, lat_p, lon_p, b = _read_clipped_geotiff(sg_root / f"SoilGrids_clay_{d_name}_Q0p5.tif", lon_min, lat_min, lon_max, lat_max)
        s_arr, _, _, _ = _read_clipped_geotiff(sg_root / f"SoilGrids_sand_{d_name}_Q0p5.tif", lon_min, lat_min, lon_max, lat_max)
        z_arr, _, _, _ = _read_clipped_geotiff(sg_root / f"SoilGrids_silt_{d_name}_Q0p5.tif", lon_min, lat_min, lon_max, lat_max)
        sg_bounds = b
        if c_arr is None or s_arr is None or z_arr is None:
            break
        if c0_ref is None:
            c0_ref, s0_ref, z0_ref = c_arr, s_arr, z_arr
            lat_sg, lon_sg = lat_p, lon_p
            clay_w = w_d * c_arr.astype(np.float64)
            sand_w = w_d * s_arr.astype(np.float64)
            silt_w = w_d * z_arr.astype(np.float64)
        else:
            clay_w += w_d * c_arr.astype(np.float64)
            sand_w += w_d * s_arr.astype(np.float64)
            silt_w += w_d * z_arr.astype(np.float64)

    clay_cell = np.zeros(n_cells, dtype=np.float64)
    sand_cell = np.zeros(n_cells, dtype=np.float64)
    silt_cell = np.zeros(n_cells, dtype=np.float64)
    soc0_cell = np.zeros(n_cells, dtype=np.float64)
    soc15_cell = np.zeros(n_cells, dtype=np.float64)
    bd0_cell = np.zeros(n_cells, dtype=np.float64)
    bd15_cell = np.zeros(n_cells, dtype=np.float64)
    n_soil_cell = np.zeros(n_cells, dtype=np.float64)
    n_soc_cell = np.zeros(n_cells, dtype=np.float64)
    n_bd_cell = np.zeros(n_cells, dtype=np.float64)

    if clay_w is not None and lat_sg is not None and lon_sg is not None:
        soc0_arr, _, _, _ = _read_clipped_geotiff(sg_root / "SoilGrids_soc_0-5cm_Q0p5.tif", lon_min, lat_min, lon_max, lat_max)
        soc15_arr, _, _, _ = _read_clipped_geotiff(sg_root / "SoilGrids_soc_15-30cm_Q0p5.tif", lon_min, lat_min, lon_max, lat_max)
        bd0_arr, _, _, _ = _read_clipped_geotiff(sg_root / "SoilGrids_bdod_0-5cm_Q0p5.tif", lon_min, lat_min, lon_max, lat_max)
        bd15_arr, _, _, _ = _read_clipped_geotiff(sg_root / "SoilGrids_bdod_15-30cm_Q0p5.tif", lon_min, lat_min, lon_max, lat_max)

        # Exclude both ocean and zero-masked urban cores from texture averages.
        valid_tex = (c0_ref.astype(np.int32) + s0_ref.astype(np.int32) + z0_ref.astype(np.int32) > 0) & (
            (clay_w + sand_w + silt_w) > 0.0
        )
        lat_2 = np.broadcast_to(lat_sg, valid_tex.shape)
        lon_2 = np.broadcast_to(lon_sg, valid_tex.shape)
        flat_tex, in_dom_tex = _latlon_to_grid_flat_idx(spec, lat_2[valid_tex], lon_2[valid_tex])
        if flat_tex.size > 0:
            n_soil_cell = np.bincount(flat_tex, minlength=n_cells).astype(np.float64)
            clay_cell = np.bincount(flat_tex, weights=clay_w[valid_tex][in_dom_tex], minlength=n_cells)
            sand_cell = np.bincount(flat_tex, weights=sand_w[valid_tex][in_dom_tex], minlength=n_cells)
            silt_cell = np.bincount(flat_tex, weights=silt_w[valid_tex][in_dom_tex], minlength=n_cells)

        if soc0_arr is not None and soc15_arr is not None:
            valid_soc = valid_tex & (soc0_arr > 0) & (soc15_arr > 0)
            flat_soc, in_soc = _latlon_to_grid_flat_idx(spec, lat_2[valid_soc], lon_2[valid_soc])
            if flat_soc.size > 0:
                n_soc_cell = np.bincount(flat_soc, minlength=n_cells).astype(np.float64)
                soc0_cell = np.bincount(flat_soc, weights=soc0_arr[valid_soc][in_soc], minlength=n_cells)
                soc15_cell = np.bincount(flat_soc, weights=soc15_arr[valid_soc][in_soc], minlength=n_cells)

        if bd0_arr is not None and bd15_arr is not None:
            valid_bd = valid_tex & (bd0_arr > 0) & (bd15_arr > 0)
            flat_bd, in_bd = _latlon_to_grid_flat_idx(spec, lat_2[valid_bd], lon_2[valid_bd])
            if flat_bd.size > 0:
                n_bd_cell = np.bincount(flat_bd, minlength=n_cells).astype(np.float64)
                bd0_cell = np.bincount(flat_bd, weights=bd0_arr[valid_bd][in_bd], minlength=n_cells)
                bd15_cell = np.bincount(flat_bd, weights=bd15_arr[valid_bd][in_bd], minlength=n_cells)

    # Read WRB MostProbable separately using its own transform (10800x6000 grid differs from 9953x5230 properties).
    n_wrb_cell = np.zeros(n_cells, dtype=np.float64)
    n_hist_cell = np.zeros(n_cells, dtype=np.float64)
    wrb_path = sg_root / "SoilGrids_WRB_MostProbable.tif"
    if wrb_path.is_file():
        wrb_arr, lat_wrb, lon_wrb, _ = _read_clipped_geotiff(wrb_path, lon_min, lat_min, lon_max, lat_max)
        if wrb_arr is not None and lat_wrb is not None:
            valid_wrb = wrb_arr > 0
            lat_w2 = np.broadcast_to(lat_wrb, wrb_arr.shape)
            lon_w2 = np.broadcast_to(lon_wrb, wrb_arr.shape)
            flat_wrb, in_wrb = _latlon_to_grid_flat_idx(spec, lat_w2[valid_wrb], lon_w2[valid_wrb])
            if flat_wrb.size > 0:
                n_wrb_cell = np.bincount(flat_wrb, minlength=n_cells).astype(np.float64)
                is_hist = (wrb_arr[valid_wrb][in_wrb] == 14).astype(np.float64)
                n_hist_cell = np.bincount(flat_wrb, weights=is_hist, minlength=n_cells)

    n_soil = n_soil_cell.reshape(ny, nx)
    has_soil = n_soil > 0.0
    den_soil = np.maximum(n_soil, 1.0)
    clay_m = np.where(has_soil, clay_cell.reshape(ny, nx) / den_soil, 0.0)
    sand_m = np.where(has_soil, sand_cell.reshape(ny, nx) / den_soil, 0.0)
    silt_m = np.where(has_soil, silt_cell.reshape(ny, nx) / den_soil, 0.0)

    has_soc = n_soc_cell.reshape(ny, nx) > 0.0
    den_soc = np.maximum(n_soc_cell.reshape(ny, nx), 1.0)
    soc0_m = np.where(has_soc, soc0_cell.reshape(ny, nx) / den_soc, 0.0)
    soc15_m = np.where(has_soc, soc15_cell.reshape(ny, nx) / den_soc, 0.0)

    has_bd = n_bd_cell.reshape(ny, nx) > 0.0
    den_bd = np.maximum(n_bd_cell.reshape(ny, nx), 1.0)
    bd0_m = np.where(has_bd, bd0_cell.reshape(ny, nx) / den_bd, 0.0)
    bd15_m = np.where(has_bd, bd15_cell.reshape(ny, nx) / den_bd, 0.0)

    has_wrb = n_wrb_cell.reshape(ny, nx) > 0.0
    histosol_frac = np.where(has_wrb, n_hist_cell.reshape(ny, nx) / np.maximum(n_wrb_cell.reshape(ny, nx), 1.0), 0.0)

    lon_180 = ((np.asarray(spec.lon2d, dtype=np.float64) + 180.0) % 360.0) - 180.0
    lat_2d = np.asarray(spec.lat2d, dtype=np.float64)
    if sg_bounds is not None:
        in_sg_bbox = (
            (lon_180 >= float(sg_bounds.left))
            & (lon_180 <= float(sg_bounds.right))
            & (lat_2d >= float(sg_bounds.bottom))
            & (lat_2d <= float(sg_bounds.top))
        )
    else:
        in_sg_bbox = np.zeros((ny, nx), dtype=bool)

    # Nearest-neighbor fill for dense urban cores inside the SoilGrids bbox where all 250 m pixels are 0-masked.
    need_fill = in_sg_bbox & is_land & (~has_soil)
    if np.any(need_fill) and np.any(has_soil):
        dist_cells, (idx_j, idx_i) = distance_transform_edt(~has_soil, return_indices=True)
        max_fill_cells = max(4.0, 25_000.0 / max(float(spec.dx_m), 100.0))
        fill_mask = need_fill & (dist_cells <= max_fill_cells)
        for arr in (clay_m, sand_m, silt_m, soc0_m, soc15_m, bd0_m, bd15_m, histosol_frac):
            arr[fill_mask] = arr[idx_j[fill_mask], idx_i[fill_mask]]
        has_soil = has_soil | fill_mask

    tex_tot = clay_m + sand_m + silt_m
    sg_valid = (in_sg_bbox & has_soil & (tex_tot > 0.0)).astype(np.float32)
    safe_tot = np.maximum(tex_tot, 1.0)
    clay_pct = np.where(sg_valid >= 0.5, 100.0 * clay_m / safe_tot, 0.0)
    sand_pct = np.where(sg_valid >= 0.5, 100.0 * sand_m / safe_tot, 0.0)
    silt_pct = np.where(sg_valid >= 0.5, 100.0 * silt_m / safe_tot, 0.0)

    usda_cls = _classify_usda_vectorized(sand_pct, silt_pct, clay_pct)
    soc_0_30 = (5.0 * soc0_m + 25.0 * soc15_m) / 30.0
    bdod_0_30 = 0.35 * bd0_m + 0.65 * bd15_m
    is_peat = (
        (histosol_frac >= 0.35)
        | ((soc_0_30 >= 550.0) & (bdod_0_30 > 0.0) & (bdod_0_30 <= 95.0))
        | (soc15_m >= 500.0)
    )
    soil_cls = np.where(is_peat, 14, usda_cls)
    soil_type = np.where(is_land & (sg_valid >= 0.5), soil_cls.astype(np.float32), 0.0).astype(np.float32)

    return {
        "soil_type": soil_type,
        "clay_pct": clay_pct.astype(np.float32),
        "sand_pct": sand_pct.astype(np.float32),
        "silt_pct": silt_pct.astype(np.float32),
        "soc_gkg": (0.1 * soc_0_30).astype(np.float32),
        "bdod_gcm3": (0.01 * bdod_0_30).astype(np.float32),
        "peat_frac": histosol_frac.astype(np.float32),
        "sg_valid": sg_valid,
    }


def ensure_domain_soilgrids_static(
    spec: DomainSpec,
    lsm_land: np.ndarray,
    *,
    output_dir: Path | str = STATIC_DIR,
    local_root: Optional[Path | str] = None,
    force: bool = False,
) -> Path:
    """Build or load cached SoilGrids 250 m soil texture + peat WX-1 container for `spec`."""
    out_path = Path(output_dir) / f"soil_type_{spec.name}_soilgrids250m.wx1"
    if out_path.is_file() and not force:
        with wx1.open_dataset(out_path) as ds:
            if ds.attrs.get("soilgrids_schema_version") == _SOILGRIDS_SCHEMA_VERSION:
                return out_path

    fields = sample_soilgrids_on_grid(spec, lsm_land=lsm_land, local_root=local_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    y_dim, x_dim = ("y", "x") if spec.is_projected else ("latitude", "longitude")
    unit_y, unit_x = ("m", "m") if spec.is_projected else ("degrees_north", "degrees_east")
    meta_units = {
        "soil_type": "1",
        "clay_pct": "%",
        "sand_pct": "%",
        "silt_pct": "%",
        "soc_gkg": "g kg-1",
        "bdod_gcm3": "g cm-3",
        "peat_frac": "1",
        "sg_valid": "1",
    }
    ds = xr.Dataset(
        data_vars={
            k: ((y_dim, x_dim), arr.astype(np.float32), {"units": meta_units[k]})
            for k, arr in fields.items()
        },
        coords={
            y_dim: (y_dim, np.asarray(spec.y_m, dtype=np.float64), {"units": unit_y}),
            x_dim: (x_dim, np.asarray(spec.x_m, dtype=np.float64), {"units": unit_x}),
            "lat2d": ((y_dim, x_dim), np.asarray(spec.lat2d, dtype=np.float64), {"units": "degrees_north"}),
            "lon2d": ((y_dim, x_dim), np.asarray(spec.lon2d, dtype=np.float64), {"units": "degrees_east"}),
        },
        attrs={
            "domain": spec.name,
            "source": "ISRIC SoilGrids 250 m (0-30 cm Q0.5 + WRB MostProbable)",
            "soilgrids_schema_version": _SOILGRIDS_SCHEMA_VERSION,
            "ai_ml_usage": "0%",
        },
    )
    wx1.write_dataset(out_path, ds, dataset_id=f"soilgrids_static_{spec.name}", codec="zstd_shuffle", workers=4)
    return out_path
