"""Offline Copernicus DEM (GLO-30 / GLO-90) & Water Body Mask (WBM) Geophysical Engine.

Reads local Copernicus DSM COG DEM & WBM GeoTIFFs from disk (ZERO network calls in offline mode):
- dem30_europe/DEM/Copernicus_DSM_COG_10_{lat}_{lon}_DEM.tif + dem30_europe/WBM/..._WBM.tif (30 m)
- dem90_world/DEM/Copernicus_DSM_COG_30_{lat}_{lon}_DEM.tif + dem90_world/WBM/..._WBM.tif (90 m)

WBM codes: 0 = Land, 1 = Ocean/Sea, 2 = Lake/Reservoir, 3 = River.
Preserves below-sea-level polders (e.g. Flevoland / Zuidplaspolder) and zeroes ocean cells (0.0 m).
Outputs lossless WX-1 v0.1.8 containers with coordinates and SI units (0% AI / 0% ML).
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import uniform_filter
import xarray as xr

import wx1

logger = logging.getLogger("tempest_v2.copernicus_dem")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCAL_DEM_ROOT = PROJECT_ROOT / "data" / "static" / "copernicus_dem_raw"
STATIC_DIR = PROJECT_ROOT / "data" / "static"


def resolve_local_dem_root(local_root: Optional[Path | str] = None) -> Path:
    if local_root is not None:
        return Path(local_root)
    env_dir = os.environ.get("TEMPEST_COPERNICUS_DEM_DIR")
    if env_dir:
        return Path(env_dir)
    if DEFAULT_LOCAL_DEM_ROOT.exists():
        return DEFAULT_LOCAL_DEM_ROOT
    fallback = Path("/mnt/d/AI/weer-ai/Tempest/COPERNICUS_DEM")
    return fallback


def format_tile_stem(lat: int, lon: int, tag: str) -> Tuple[str, str]:
    """Return official Copernicus DSM DEM and WBM stems for integer SW corner (lat, lon)."""
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    base = f"Copernicus_DSM_COG_{tag}_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00"
    return f"{base}_DEM.tif", f"{base}_WBM.tif"


def is_known_land_tile(lat: int, lon: int) -> bool:
    """Integrity guard: identifies European continental tiles known to contain land."""
    if lat == 51 and 3 <= lon <= 15:
        return True
    if 52 <= lat <= 53 and 5 <= lon <= 14:
        return True
    if 48 <= lat <= 50 and 2 <= lon <= 15:
        return True
    if lat == 47 and 6 <= lon <= 15:
        return True
    return False


def locate_local_tile_files(
    lat: int,
    lon: int,
    local_root: Optional[Path | str] = None,
    prefer_glo30: bool = True,
) -> Optional[Tuple[Path, Optional[Path], str]]:
    """Locate local DEM & WBM GeoTIFFs, preferring GLO-30 Europe and falling back to GLO-90 World."""
    root = resolve_local_dem_root(local_root)
    candidates = []
    if prefer_glo30:
        candidates.append((root / "dem30_europe", "10", "GLO-30"))
    candidates.append((root / "dem90_world", "30", "GLO-90"))

    for base_dir, tag, res_name in candidates:
        dem_name, wbm_name = format_tile_stem(lat, lon, tag)
        dem_path = base_dir / "DEM" / dem_name
        wbm_path = base_dir / "WBM" / wbm_name
        if dem_path.is_file() and dem_path.stat().st_size > 1000:
            return dem_path, (wbm_path if wbm_path.is_file() else None), res_name
    return None


def shapiro_smooth_orography(
    zs_m: np.ndarray,
    lsm_frac: np.ndarray,
    passes: int = 1,
    weight: float = 0.125,
) -> np.ndarray:
    """Gentle 2D 5-point filter removing 2*dx checkerboard noise while keeping pure ocean = 0.0 m."""
    out = np.asarray(zs_m, dtype=np.float64).copy()
    land = np.asarray(lsm_frac, dtype=np.float64) >= 0.05
    w_center = 1.0 - 4.0 * weight
    for _ in range(max(0, int(passes))):
        padded = np.pad(out, 1, mode="edge")
        filt = (
            w_center * padded[1:-1, 1:-1]
            + weight * (padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:])
        )
        out = np.where(land, filt, 0.0)
    return out.astype(np.float32)


def sample_copernicus_on_2d_grid(
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    dx_m: float,
    dy_m: float,
    *,
    local_root: Optional[Path | str] = None,
    prefer_glo30: bool = True,
    offline: bool = True,
    compute_subgrid_sso: bool = True,
    smooth_passes: int = 1,
) -> Dict[str, np.ndarray]:
    """Sample offline Copernicus DEM (GLO-30 with GLO-90 fallback) & WBM onto arbitrary 2D (lat, lon) grid."""
    import rasterio

    if not offline:
        raise ValueError("Online AWS fetching is disabled in v2; use local_root offline Copernicus tiles.")

    lat_arr = np.asarray(lat2d, dtype=np.float64)
    lon_arr = ((np.asarray(lon2d, dtype=np.float64) + 180.0) % 360.0) - 180.0
    if lat_arr.shape != lon_arr.shape or lat_arr.ndim != 2:
        raise ValueError("lat2d and lon2d must be matching 2D arrays")

    ny, nx = lat_arr.shape
    oro = np.zeros((ny, nx), dtype=np.float32)
    oro_std = np.zeros((ny, nx), dtype=np.float32)
    lsm_frac = np.zeros((ny, nx), dtype=np.float32)
    lake_frac = np.zeros((ny, nx), dtype=np.float32)
    river_frac = np.zeros((ny, nx), dtype=np.float32)

    tile_lat_idx = np.floor(lat_arr).astype(np.int32)
    tile_lon_idx = np.floor(lon_arr).astype(np.int32)
    unique_lat_min = int(tile_lat_idx.min())
    unique_lat_max = int(tile_lat_idx.max())
    unique_lon_min = int(tile_lon_idx.min())
    unique_lon_max = int(tile_lon_idx.max())

    root = resolve_local_dem_root(local_root)
    if not root.exists():
        raise FileNotFoundError(
            f"Local Copernicus DEM directory not found: {root}. "
            f"Set TEMPEST_COPERNICUS_DEM_DIR or symlink data/static/copernicus_dem_raw."
        )

    # Approximate angular cell footprint [deg]
    dlat_cell = dy_m / 111_195.0
    n_glo30, n_glo90, n_ocean = 0, 0, 0

    for tlat in range(unique_lat_min, unique_lat_max + 1):
        dlon_cell = dx_m / (111_195.0 * max(0.2, math.cos(math.radians(tlat + 0.5))))
        for tlon in range(unique_lon_min, unique_lon_max + 1):
            mask = (tile_lat_idx == tlat) & (tile_lon_idx == tlon)
            if not np.any(mask):
                continue

            found = locate_local_tile_files(tlat, tlon, root, prefer_glo30=prefer_glo30)
            if found is None:
                if is_known_land_tile(tlat, tlon):
                    raise FileNotFoundError(
                        f"Missing local Copernicus DEM tile for known continental land ({tlat}, {tlon}) under {root}"
                    )
                # Pure open ocean tile: elevation=0, lsm=0, lake=0, river=0
                n_ocean += 1
                continue

            dem_path, wbm_path, res_name = found
            if res_name == "GLO-30":
                n_glo30 += 1
            else:
                n_glo90 += 1

            with rasterio.open(dem_path) as src_dem:
                dem = src_dem.read(1).astype(np.float64)
                inv_tr = ~src_dem.transform
                px_w = abs(float(src_dem.transform.a))
                px_h = abs(float(src_dem.transform.e))
                if src_dem.nodata is not None:
                    dem = np.where(dem == src_dem.nodata, 0.0, dem)

            h_px, w_px = dem.shape
            if wbm_path is not None:
                with rasterio.open(wbm_path) as src_wbm:
                    wbm = src_wbm.read(1)
                if wbm.shape != (h_px, w_px):
                    jj_s = np.clip(np.linspace(0, wbm.shape[0] - 1, h_px).astype(np.int64), 0, wbm.shape[0] - 1)
                    ii_s = np.clip(np.linspace(0, wbm.shape[1] - 1, w_px).astype(np.int64), 0, wbm.shape[1] - 1)
                    wbm = wbm[np.ix_(jj_s, ii_s)]
            else:
                wbm = np.where(dem == 0.0, 1, 0).astype(np.uint8)

            # Force ocean surface strictly 0.0 m (WBM == 1); preserve below-sea-level polders (WBM == 0)
            dem = np.where(wbm == 1, 0.0, dem)
            is_land = (wbm == 0).astype(np.float64)
            is_lake = (wbm == 2).astype(np.float64)
            is_river = (wbm == 3).astype(np.float64)

            box_r = max(1, int(round(dlat_cell / px_h)))
            box_c = max(1, int(round(dlon_cell / px_w)))

            if box_r > 1 or box_c > 1:
                land_f = uniform_filter(is_land, size=(box_r, box_c), mode="nearest")
                lake_f = uniform_filter(is_lake, size=(box_r, box_c), mode="nearest")
                river_f = uniform_filter(is_river, size=(box_r, box_c), mode="nearest")
                land_dem_sum = uniform_filter(dem * is_land, size=(box_r, box_c), mode="nearest")
                dem_mean = np.where(land_f > 0.05, land_dem_sum / np.maximum(land_f, 1e-6), 0.0)
                if compute_subgrid_sso:
                    dem2_sum = uniform_filter((dem * dem) * is_land, size=(box_r, box_c), mode="nearest")
                    dem2_mean = np.where(land_f > 0.05, dem2_sum / np.maximum(land_f, 1e-6), 0.0)
                    dem_std = np.sqrt(np.maximum(0.0, dem2_mean - dem_mean * dem_mean))
                else:
                    dem_std = np.zeros_like(dem_mean)
            else:
                land_f = is_land
                lake_f = is_lake
                river_f = is_river
                dem_mean = np.where(is_land > 0.5, dem, 0.0)
                dem_std = np.zeros_like(dem_mean)

            cols_f, rows_f = inv_tr @ (lon_arr[mask], lat_arr[mask])
            r_idx = np.clip(np.rint(rows_f).astype(np.int64), 0, h_px - 1)
            c_idx = np.clip(np.rint(cols_f).astype(np.int64), 0, w_px - 1)

            lf_pts = np.clip(land_f[r_idx, c_idx], 0.0, 1.0)
            oro[mask] = np.where(lf_pts >= 0.05, dem_mean[r_idx, c_idx], 0.0).astype(np.float32)
            oro_std[mask] = np.where(lf_pts >= 0.05, dem_std[r_idx, c_idx], 0.0).astype(np.float32)
            lsm_frac[mask] = lf_pts.astype(np.float32)
            lake_frac[mask] = np.clip(lake_f[r_idx, c_idx], 0.0, 1.0).astype(np.float32)
            river_frac[mask] = np.clip(river_f[r_idx, c_idx], 0.0, 1.0).astype(np.float32)

    oro_smooth = shapiro_smooth_orography(oro, lsm_frac, passes=smooth_passes) if smooth_passes > 0 else oro.copy()
    logger.info(
        "Copernicus offline composite (%dx%d): %d GLO-30 tiles, %d GLO-90 tiles, %d ocean tiles | oro [%.1f, %.1f] m",
        ny, nx, n_glo30, n_glo90, n_ocean, float(oro.min()), float(oro.max()),
    )
    return {
        "orography_m": oro,
        "orography_smooth_m": oro_smooth,
        "orography_std_m": oro_std,
        "lsm_frac": lsm_frac,
        "lake_frac": lake_frac,
        "river_frac": river_frac,
    }


def write_copernicus_static_wx1(
    output_path: Path | str,
    fields: Dict[str, np.ndarray],
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    *,
    domain_name: str,
    is_projected: bool = True,
) -> Path:
    """Save Copernicus DEM + WBM composite into a single WX-1 v0.1.8 container with full coordinates."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    y_dim, x_dim = ("y", "x") if is_projected else ("latitude", "longitude")
    unit_xy = "m" if is_projected else ("degrees_north", "degrees_east")

    ds = xr.Dataset(
        data_vars={
            "orography_m": ((y_dim, x_dim), fields["orography_m"].astype(np.float32), {"units": "m", "long_name": "Surface elevation above MSL"}),
            "orography_smooth_m": ((y_dim, x_dim), fields["orography_smooth_m"].astype(np.float32), {"units": "m", "long_name": "2dx filtered surface elevation"}),
            "orography_std_m": ((y_dim, x_dim), fields["orography_std_m"].astype(np.float32), {"units": "m", "long_name": "Sub-grid orography standard deviation"}),
            "lsm_frac": ((y_dim, x_dim), fields["lsm_frac"].astype(np.float32), {"units": "1", "long_name": "Copernicus WBM land fraction"}),
            "lake_frac": ((y_dim, x_dim), fields["lake_frac"].astype(np.float32), {"units": "1", "long_name": "Copernicus WBM freshwater lake fraction"}),
            "river_frac": ((y_dim, x_dim), fields["river_frac"].astype(np.float32), {"units": "1", "long_name": "Copernicus WBM river fraction"}),
        },
        coords={
            y_dim: (y_dim, np.asarray(y_coords, dtype=np.float64), {"units": unit_xy[0] if isinstance(unit_xy, tuple) else unit_xy}),
            x_dim: (x_dim, np.asarray(x_coords, dtype=np.float64), {"units": unit_xy[1] if isinstance(unit_xy, tuple) else unit_xy}),
            "lat2d": ((y_dim, x_dim), np.asarray(lat2d, dtype=np.float64), {"units": "degrees_north", "standard_name": "latitude"}),
            "lon2d": ((y_dim, x_dim), np.asarray(lon2d, dtype=np.float64), {"units": "degrees_east", "standard_name": "longitude"}),
        },
        attrs={
            "domain": domain_name,
            "source": "ESA Copernicus DSM (GLO-30 Europe + GLO-90 World) offline composite",
            "ai_ml_usage": "0%",
        },
    )
    wx1.write_dataset(out, ds, dataset_id=f"copernicus_static_{domain_name}", codec="zstd_shuffle", workers=4)
    return out
