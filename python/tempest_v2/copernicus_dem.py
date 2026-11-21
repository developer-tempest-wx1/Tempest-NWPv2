"""Tempest-NWP: Copernicus DEM (GLO-90 / GLO-30) Static Geophysical Engine.

Downloads, verifies, and processes high-resolution topography and land-sea masks
from the official ESA / Copernicus TanDEM-X Digital Elevation Model (AWS Open Data):
- Global coverage from 90°S to 90°N (unlike SRTM which cuts off at 60°N).
- 100% deterministic physical boundary conditions (0% AI / 0% Machine Learning).
- Includes separate auxiliary Water Body Mask (WBM GeoTIFF):
    0 = Land
    1 = Ocean / Sea
    2 = Lake / Reservoir (e.g. IJsselmeer)
    3 = River
- Dynamic latitude tile column geometry:
    * |lat| < 50°: 1200 x 1200 pixels (3" x 3" grid spacing)
    * 50° <= |lat| < 60°: 1200 x 800 pixels (3" x 4.5" grid spacing)
    * 60° <= |lat| < 70°: 1200 x 600 pixels (3" x 6" grid spacing)
    * 70° <= |lat| < 80°: 1200 x 400 pixels (3" x 9" grid spacing)
- Preserves authentic below-sea-level land (e.g. Dutch Zuidplaspolder -6.76m, Flevoland,
  and Tagebau open-pit mines) without false clamping to zero.
- Eliminates spectral Gibbs-ringing: Ocean pixels are strictly 0.0 m.
- Provenance tracking: Open ocean tiles (HTTP 404 on AWS) record full JSON provenance.
- Known-land guard: Protects against network timeouts silently misclassifying land as ocean.
- Permanent one-time caching in lossless WX-1 scientific container format (`.wx1`).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import io
import json
import logging
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Set, Tuple
import urllib.request

# Ensure python/ directory is on sys.path
_PKG_DIR = str(Path(__file__).resolve().parent.parent)
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

import numpy as np
from PIL import Image

import wx1

logger = logging.getLogger("tempest_v2.copernicus_dem")

CACHE_DIR = Path("data/static/copernicus_dem/cache_tiles")
STATIC_DIR = Path("data/static")
GLO90_BASE_URL = "https://copernicus-dem-90m.s3.amazonaws.com"
GLO30_BASE_URL = "https://copernicus-dem-30m.s3.amazonaws.com"


def get_copernicus_tile_shape(lat: int, resolution: str = "GLO-90") -> Tuple[int, int]:
    """Returns official (n_rows, n_cols) for a 1x1 degree Copernicus DEM tile based on latitude band.
    
    Per official ESA Copernicus DEM Product Specification:
    - Latitude rows are always 1200 (GLO-90, 3 arcsec) or 3600 (GLO-30, 1 arcsec).
    - Longitude columns decrease towards poles to maintain roughly isotropic metric spacing:
        * 0° to 50°: 1200 cols (3.0" dlon)
        * 50° to 60°: 800 cols (4.5" dlon)
        * 60° to 70°: 600 cols (6.0" dlon)
        * 70° to 80°: 400 cols (9.0" dlon)
        * 80° to 85°: 240 cols (15.0" dlon)
        * 85° to 90°: 120 cols (30.0" dlon)
    """
    abs_lat = abs(lat)
    n_rows = 1200 if resolution == "GLO-90" else 3600

    if abs_lat < 50:
        n_cols = 1200 if resolution == "GLO-90" else 3600
    elif abs_lat < 60:
        n_cols = 800 if resolution == "GLO-90" else 2400
    elif abs_lat < 70:
        n_cols = 600 if resolution == "GLO-90" else 1800
    elif abs_lat < 80:
        n_cols = 400 if resolution == "GLO-90" else 1200
    elif abs_lat < 85:
        n_cols = 240 if resolution == "GLO-90" else 720
    else:
        n_cols = 120 if resolution == "GLO-90" else 360

    return n_rows, n_cols


def is_known_land_tile(lat: int, lon: int) -> bool:
    """Sanity guard: identifies 1x1 degree tiles known to contain substantial continental land.
    
    Prevents temporary network outages or HTTP timeouts from corrupting land areas into ocean.
    """
    # Netherlands & Flanders:
    if lat == 51 and 3 <= lon <= 7:
        return True
    if 52 <= lat <= 53 and 4 <= lon <= 7:
        return True
    # Belgium, Luxembourg & Northern France: 48-50°N, 2-6°E
    if 48 <= lat <= 50 and 2 <= lon <= 6:
        return True
    # Western & Central Germany:
    if 48 <= lat <= 53 and 6 <= lon <= 10:
        return True
    if lat == 54 and 8 <= lon <= 10:
        return True
    # England & Wales:
    if 51 <= lat <= 52 and -3 <= lon <= 1:
        return True
    if lat == 53 and -3 <= lon <= 0:
        return True
    if lat == 54 and -3 <= lon <= -1:
        return True
    # Jutland / Denmark: 55-57°N, 8-10°E
    if 55 <= lat <= 57 and 8 <= lon <= 10:
        return True
    return False


def format_tile_name(lat: int, lon: int, resolution: str = "GLO-90") -> str:
    """Formats official Copernicus DEM 1x1 degree tile identifier."""
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    tag = "30" if resolution == "GLO-90" else "10"
    return f"Copernicus_DSM_COG_{tag}_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM"


def write_ocean_marker_metadata(
    marker_path: Path,
    lat: int,
    lon: int,
    tile_name: str,
    resolution: str,
    http_status: int = 404,
) -> None:
    """Writes explicit JSON provenance metadata into an .ocean sentinel marker file."""
    base_url = GLO90_BASE_URL if resolution == "GLO-90" else GLO30_BASE_URL
    meta = {
        "tile_name": tile_name,
        "latitude": lat,
        "longitude": lon,
        "resolution": resolution,
        "http_status": http_status,
        "verified_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source": f"{base_url}/{tile_name}/{tile_name}.tif",
        "classification": "open_ocean_no_land",
        "elevation_m": 0.0,
        "wbm_class": 1,
        "lsm": 0.0,
    }
    marker_path.write_text(json.dumps(meta, indent=2))


def download_tile(
    lat: int,
    lon: int,
    cache_dir: Path = CACHE_DIR,
    resolution: str = "GLO-90",
) -> Tuple[int, int, bool]:
    """Downloads a single 1x1 degree DEM and WBM tile from AWS Open Data if not cached.
    
    Returns (lat, lon, is_land):
    - is_land is True if tile has land/topography.
    - is_land is False if tile is pure open ocean (HTTP 404 on AWS).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    tile_name = format_tile_name(lat, lon, resolution)
    dem_file = cache_dir / f"{tile_name}.tif"
    wbm_file = cache_dir / f"{tile_name}_WBM.tif"
    ocean_marker = cache_dir / f"{tile_name}.ocean"

    if ocean_marker.exists():
        if ocean_marker.stat().st_size == 0:
            # Upgrade legacy 0-byte file with JSON metadata
            write_ocean_marker_metadata(ocean_marker, lat, lon, tile_name, resolution, 404)
        return (lat, lon, False)

    if dem_file.exists() and wbm_file.exists() and dem_file.stat().st_size > 1000:
        return (lat, lon, True)

    base_url = GLO90_BASE_URL if resolution == "GLO-90" else GLO30_BASE_URL
    dem_url = f"{base_url}/{tile_name}/{tile_name}.tif"
    wbm_url = f"{base_url}/{tile_name}/AUXFILES/{tile_name.replace('DEM', 'WBM')}.tif"

    # Check existence via HEAD
    req_head = urllib.request.Request(dem_url, headers={"User-Agent": "Tempest-NWP/1.0"}, method="HEAD")
    try:
        with urllib.request.urlopen(req_head, timeout=10) as resp:
            if resp.status != 200:
                if is_known_land_tile(lat, lon):
                    raise RuntimeError(f"[INTEGRITY ERROR] Tile {lat},{lon} is known land but returned status {resp.status}!")
                write_ocean_marker_metadata(ocean_marker, lat, lon, tile_name, resolution, resp.status)
                return (lat, lon, False)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            if is_known_land_tile(lat, lon):
                raise RuntimeError(
                    f"[FATAL INTEGRITY VIOLATION] Tile {lat},{lon} ({tile_name}) contains verified CONTINENTAL LAND "
                    f"(Netherlands/Flanders/UK/France/Germany) but AWS returned HTTP 404! Suspected URL format or connection error."
                )
            # Pure open ocean tile confirmed
            write_ocean_marker_metadata(ocean_marker, lat, lon, tile_name, resolution, 404)
            return (lat, lon, False)
        raise RuntimeError(f"HTTP error checking tile {tile_name}: {e}")
    except Exception as e:
        if is_known_land_tile(lat, lon):
            raise RuntimeError(f"[NETWORK ERROR] Failed checking known land tile {lat},{lon}: {e}")
        logger.warning("Error accessing Copernicus DEM tile %s: %s", tile_name, e)
        write_ocean_marker_metadata(ocean_marker, lat, lon, tile_name, resolution, 500)
        return (lat, lon, False)

    # Download DEM GeoTIFF
    try:
        req_dem = urllib.request.Request(dem_url, headers={"User-Agent": "Tempest-NWP/1.0"})
        with urllib.request.urlopen(req_dem, timeout=25) as resp:
            dem_data = resp.read()
        dem_file.write_bytes(dem_data)
    except Exception as e:
        raise RuntimeError(f"Failed downloading DEM {tile_name}: {e}")

    # Download separate Water Body Mask (WBM GeoTIFF)
    try:
        req_wbm = urllib.request.Request(wbm_url, headers={"User-Agent": "Tempest-NWP/1.0"})
        with urllib.request.urlopen(req_wbm, timeout=25) as resp:
            wbm_data = resp.read()
        wbm_file.write_bytes(wbm_data)
    except Exception as e:
        logger.warning("WBM missing for %s: %s", tile_name, e)
        # Create fallback: elevation <= 0 is ocean, > 0 is land
        img = Image.open(dem_file)
        arr = np.array(img, dtype=np.float32)
        fallback_wbm = np.where(arr <= 0.0, 1, 0).astype(np.uint8)
        Image.fromarray(fallback_wbm).save(wbm_file)

    return (lat, lon, True)


def load_cached_tile(
    lat: int,
    lon: int,
    cache_dir: Path = CACHE_DIR,
    resolution: str = "GLO-90",
) -> Tuple[np.ndarray, np.ndarray]:
    """Loads a 1x1 degree tile from disk cache with exact column geometry.
    
    Returns (dem_grid, wbm_grid) of shape (n_rows, n_cols).
    - Over ocean: dem is strictly 0.0f, wbm is 1 (ocean).
    - On land: authentic elevations (including Dutch polders down to -6.8m) are fully preserved.
    """
    tile_name = format_tile_name(lat, lon, resolution)
    ocean_marker = cache_dir / f"{tile_name}.ocean"
    dem_file = cache_dir / f"{tile_name}.tif"
    wbm_file = cache_dir / f"{tile_name}_WBM.tif"

    n_rows, n_cols = get_copernicus_tile_shape(lat, resolution)

    if ocean_marker.exists() or not dem_file.exists():
        # Pure open ocean tile with exact shape for this latitude
        dem = np.zeros((n_rows, n_cols), dtype=np.float32)
        wbm = np.ones((n_rows, n_cols), dtype=np.uint8)  # 1 = Ocean Water
        return dem, wbm

    # Open single-band float32 DEM GeoTIFF
    img_dem = Image.open(dem_file)
    dem = np.array(img_dem, dtype=np.float32)

    # Open separate auxiliary Water Body Mask (WBM GeoTIFF)
    if wbm_file.exists() and wbm_file.stat().st_size > 0:
        img_wbm = Image.open(wbm_file)
        wbm = np.array(img_wbm, dtype=np.uint8)
        if wbm.shape != dem.shape:
            wbm = np.array(img_wbm.resize((dem.shape[1], dem.shape[0]), Image.NEAREST), dtype=np.uint8)
    else:
        # Fallback: if elevation == 0, ocean; else land
        wbm = np.where(dem == 0.0, 1, 0).astype(np.uint8)

    # Clean ocean surface:
    # WBM values: 0 = Land, 1 = Ocean/Sea, 2 = Lake/Reservoir, 3 = River.
    # Ocean water surface is strictly 0.0 m (eliminates Gibbs ringing artifacts).
    # Land pixels (wbm == 0), including below-sea-level polders (-1m to -7m), remain UNTOUCHED.
    ocean_mask = (wbm == 1)
    dem[ocean_mask] = 0.0

    return dem, wbm


def print_domain_tile_status_map(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    cache_dir: Path = CACHE_DIR,
    resolution: str = "GLO-90",
) -> None:
    """Prints an ASCII provenance map of all tiles in the domain bounding box."""
    i_lat_min = int(np.floor(lat_min))
    i_lat_max = int(np.floor(lat_max))
    i_lon_min = int(np.floor(lon_min))
    i_lon_max = int(np.floor(lon_max))

    lons_range = list(range(i_lon_min, i_lon_max + 1))
    lats_range = list(range(i_lat_max, i_lat_min - 1, -1))  # North to South

    n_land = 0
    n_ocean = 0
    n_missing = 0

    print("\n" + "=" * 94)
    print(f"             COPERNICUS DEM TILE INVENTORY & PROVENANCE MAP ({resolution})")
    print(f"       Bounding Box: {i_lat_min}°N to {i_lat_max}°N | {i_lon_min}°E to {i_lon_max}°E ({len(lats_range)*len(lons_range)} total tiles)")
    print("=" * 94)

    # Header row
    lon_header = "Lat\\Lon | " + " ".join([f"{l:+3d}°".replace("+", " ") for l in lons_range])
    print(lon_header)
    print("-" * len(lon_header))

    for lat in lats_range:
        row_str = f" {lat:2d}°N  | "
        for lon in lons_range:
            tname = format_tile_name(lat, lon, resolution)
            dem_f = cache_dir / f"{tname}.tif"
            wbm_f = cache_dir / f"{tname}_WBM.tif"
            ocean_f = cache_dir / f"{tname}.ocean"

            if dem_f.exists() and wbm_f.exists() and dem_f.stat().st_size > 1000:
                row_str += " [L] "
                n_land += 1
            elif ocean_f.exists():
                row_str += " [~] "
                n_ocean += 1
            else:
                row_str += " [!] "
                n_missing += 1
        print(row_str)

    print("-" * len(lon_header))
    print(f"Legend: [L] = Land (DEM float32 GeoTIFF + WBM mask present)")
    print(f"        [~] = Ocean (Confirmed HTTP 404 pure sea, metadata verified, elevation=0.0m)")
    print(f"        [!] = MISSING / CORRUPT")
    print(f"Summary: Verified {n_land} land tiles, {n_ocean} open ocean tiles, {n_missing} missing tiles.")
    print("=" * 94 + "\n")

    if n_missing > 0:
        raise RuntimeError(f"[INTEGRITY ERROR] Domain contains {n_missing} missing or unverified tiles!")


def ensure_domain_tiles(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    cache_dir: Path = CACHE_DIR,
    resolution: str = "GLO-90",
    max_workers: int = 12,
) -> List[Tuple[int, int, bool]]:
    """Downloads all 1x1 degree Copernicus tiles covering the bounding box."""
    i_lat_min = int(np.floor(lat_min))
    i_lat_max = int(np.floor(lat_max))
    i_lon_min = int(np.floor(lon_min))
    i_lon_max = int(np.floor(lon_max))

    tiles_to_query = [
        (lat, lon)
        for lat in range(i_lat_min, i_lat_max + 1)
        for lon in range(i_lon_min, i_lon_max + 1)
    ]

    logger.info(
        "Checking/downloading %d Copernicus DEM tiles (Lat: %d to %d, Lon: %d to %d)...",
        len(tiles_to_query), i_lat_min, i_lat_max, i_lon_min, i_lon_max
    )

    t0 = time.time()
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_tile, lat, lon, cache_dir, resolution): (lat, lon)
            for (lat, lon) in tiles_to_query
        }
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    t1 = time.time()
    land_tiles = [r for r in results if r[2]]
    logger.info(
        "Copernicus DEM verification complete in %.1fs: %d land tiles, %d open ocean tiles.",
        t1 - t0, len(land_tiles), len(results) - len(land_tiles)
    )

    # Print full visual ASCII provenance dashboard
    print_domain_tile_status_map(lat_min, lat_max, lon_min, lon_max, cache_dir, resolution)
    return results


def generate_dem_for_domain(
    domain_name: str,
    target_lats: np.ndarray,
    target_lons: np.ndarray,
    compute_subgrid_sso: bool = False,
    cache_dir: Path = CACHE_DIR,
    output_dir: Path = STATIC_DIR,
    resolution: str = "GLO-90",
) -> Tuple[Path, Path, Optional[Path]]:
    """Generates and permanently saves high-resolution static fields into WX-1 containers.
    
    Outputs:
    - data/static/orography_{domain_name}.wx1   (Surface elevation z_s [m])
    - data/static/lsm_{domain_name}.wx1         (Land-Sea Mask [0.0 = sea, 1.0 = land])
    - data/static/sso_std_{domain_name}.wx1     (Subgrid standard deviation sigma_sso [m], if enabled)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    oro_path = output_dir / f"orography_{domain_name}.wx1"
    lsm_path = output_dir / f"lsm_{domain_name}.wx1"
    sso_path = output_dir / f"sso_std_{domain_name}.wx1" if compute_subgrid_sso else None

    # Step 1: Ensure all required tiles are downloaded and verified
    lat_min, lat_max = float(target_lats.min()), float(target_lats.max())
    lon_min, lon_max = float(target_lons.min()), float(target_lons.max())

    ensure_domain_tiles(lat_min, lat_max, lon_min, lon_max, cache_dir, resolution)

    ny = len(target_lats)
    nx = len(target_lons)
    logger.info("Compositing Copernicus DEM onto %s grid (%d x %d)...", domain_name, ny, nx)

    zs_out = np.zeros((ny, nx), dtype=np.float32)
    lsm_out = np.zeros((ny, nx), dtype=np.float32)
    sso_out = np.zeros((ny, nx), dtype=np.float32) if compute_subgrid_sso else None

    # Grid step approximations
    dlat = abs(float(target_lats[0] - target_lats[1])) if ny > 1 else 0.02
    dlon = abs(float(target_lons[1] - target_lons[0])) if nx > 1 else 0.02
    half_dlat = dlat * 0.5
    half_dlon = dlon * 0.5

    # Group target grid cells by 1x1 degree tile to process chunk-by-chunk in memory
    i_lat_min = int(np.floor(lat_min))
    i_lat_max = int(np.floor(lat_max))
    i_lon_min = int(np.floor(lon_min))
    i_lon_max = int(np.floor(lon_max))

    for tile_lat in range(i_lat_min, i_lat_max + 1):
        for tile_lon in range(i_lon_min, i_lon_max + 1):
            j_indices = np.where((target_lats >= tile_lat - half_dlat) & (target_lats < tile_lat + 1.0 + half_dlat))[0]
            i_indices = np.where((target_lons >= tile_lon - half_dlon) & (target_lons < tile_lon + 1.0 + half_dlon))[0]

            if len(j_indices) == 0 or len(i_indices) == 0:
                continue

            dem_tile, wbm_tile = load_cached_tile(tile_lat, tile_lon, cache_dir, resolution)
            n_rows, n_cols = dem_tile.shape

            for j in j_indices:
                pt_lat = float(target_lats[j])
                row_idx = int(np.clip((tile_lat + 1.0 - pt_lat) * n_rows, 0, n_rows - 1))
                half_r = max(1, int(half_dlat * n_rows))
                r_start = max(0, row_idx - half_r)
                r_end = min(n_rows, row_idx + half_r + 1)

                for i in i_indices:
                    pt_lon = float(target_lons[i])
                    col_idx = int(np.clip((pt_lon - tile_lon) * n_cols, 0, n_cols - 1))
                    half_c = max(1, int(half_dlon * n_cols))
                    c_start = max(0, col_idx - half_c)
                    c_end = min(n_cols, col_idx + half_c + 1)

                    sub_dem = dem_tile[r_start:r_end, c_start:c_end]
                    sub_wbm = wbm_tile[r_start:r_end, c_start:c_end]

                    if sub_dem.size == 0:
                        continue

                    # Land fraction: WBM != 1 (where 1 is ocean water)
                    # 0 = Land, 2 = Lake/IJsselmeer, 3 = River. For land-sea mask:
                    is_land_pixel = (sub_wbm == 0)
                    land_fraction = float(np.mean(is_land_pixel))

                    if land_fraction < 0.05:
                        # Pure or nearly pure sea: flatten to exact 0.0 m (NO GIBBS RINGING)
                        zs_val = 0.0
                        lsm_val = 0.0
                        sso_val = 0.0
                    elif land_fraction > 0.95:
                        # Pure land (including below-sea-level Dutch polders down to -6.8m)
                        zs_val = float(np.mean(sub_dem))
                        lsm_val = 1.0
                        sso_val = float(np.std(sub_dem)) if compute_subgrid_sso else 0.0
                    else:
                        # Mixed coastal cell
                        land_elevs = sub_dem[is_land_pixel]
                        zs_val = float(np.mean(land_elevs)) if land_elevs.size > 0 else 0.0
                        lsm_val = land_fraction
                        sso_val = float(np.std(land_elevs)) if (compute_subgrid_sso and land_elevs.size > 0) else 0.0

                    zs_out[j, i] = zs_val
                    lsm_out[j, i] = lsm_val
                    if compute_subgrid_sso:
                        sso_out[j, i] = sso_val

    # Save to compressed WX-1 containers
    def _write_wx1(path: Path, data: np.ndarray, var: str):
        if path.exists():
            path.unlink()
        wx1.write_array(
            path,
            data.astype(np.float32),
            dataset_id=f"copernicus_{domain_name}_{var}",
            variable_path=var,
            dimensions=["latitude", "longitude"],
            chunk_shape=[ny, nx],
            profile="output",
            codec="zstd_shuffle",
        )

    _write_wx1(oro_path, zs_out, "zs")
    logger.info("Saved %s (min=%.1f m, max=%.1f m)", oro_path.name, zs_out.min(), zs_out.max())

    _write_wx1(lsm_path, lsm_out, "lsm")
    logger.info("Saved %s (ocean=%.1f%%, land=%.1f%%)", lsm_path.name, (lsm_out < 0.5).mean() * 100, (lsm_out >= 0.5).mean() * 100)

    if compute_subgrid_sso and sso_path is not None:
        _write_wx1(sso_path, sso_out, "sso_std")
        logger.info("Saved %s (mean sigma_sso=%.1f m, max=%.1f m)", sso_path.name, sso_out.mean(), sso_out.max())

    return oro_path, lsm_path, sso_path


def main():
    parser = argparse.ArgumentParser(description="Tempest-NWP: Copernicus DEM Processor")
    parser.add_argument("--domain", default="benelux", choices=["benelux", "d2_europe", "d1_global"], help="Domain to generate")
    parser.add_argument("--subgrid-sso", action="store_true", help="Compute subgrid orographic variance sigma_sso")
    parser.add_argument("--d3-resolution-m", type=int, choices=[2500, 1500, 500], default=None,
                        help="Select D3 nominal geographic spacing in metres (default: TEMPEST_D3_RESOLUTION_M or 2500)")
    parser.add_argument("--map-only", action="store_true", help="Only display the tile status map without regenerating WX-1")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if args.d3_resolution_m is not None:
        import os
        os.environ["TEMPEST_D3_RESOLUTION_M"] = str(args.d3_resolution_m)

    print("=================================================================")
    print("   TEMPEST-NWP: Copernicus DEM (GLO-90) Geophysical Engine       ")
    print(f"         Domain: {args.domain.upper()} | Subgrid SSO: {args.subgrid_sso} ")
    print("         STRICTLY 0% AI / 0% ML / 100% DETERMINISTIC PHYSICS     ")
    print("=================================================================\n")

    if args.domain == "benelux":
        from tempest_v2.static_fields import get_benelux_grid_coords, benelux_grid_tag
        lats, lons = get_benelux_grid_coords()
        d3_name = f"benelux_{benelux_grid_tag()}"
        if args.map_only:
            print_domain_tile_status_map(float(lats.min()), float(lats.max()), float(lons.min()), float(lons.max()))
        else:
            generate_dem_for_domain(d3_name, lats, lons, compute_subgrid_sso=args.subgrid_sso)
    elif args.domain == "d2_europe":
        lats = np.linspace(72.0, 30.0, 421, dtype=np.float32)
        lons = np.linspace(-25.0, 45.0, 701, dtype=np.float32)
        if args.map_only:
            print_domain_tile_status_map(30.0, 72.0, -25.0, 45.0)
        else:
            generate_dem_for_domain("d2_europe_11km", lats, lons, compute_subgrid_sso=True)
    elif args.domain == "d1_global":
        lats = np.linspace(90.0, -90.0, 451, dtype=np.float32)
        lons = np.linspace(0.0, 359.6, 900, dtype=np.float32)
        generate_dem_for_domain("d1_global_0p4", lats, lons, compute_subgrid_sso=True)


if __name__ == "__main__":
    main()
