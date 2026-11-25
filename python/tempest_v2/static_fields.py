"""Unified Static Geophysical Fields Manager for Tempest-NWP v2.

Combines deterministic offline geophysical sources onto any DomainSpec (D1/D2/D3):
1. Topography & Water Body Masks from local Copernicus DEM (30 m Europe + 90 m World fallback).
2. Soil Texture & 4-Layer Profiles from FAO HWSD v2.2 on 2D LCC or lat/lon grids.
3. Sea-floor Bathymetry & Wet Mask from GEBCO 2026 NetCDF/GeoTIFF (strict MSL datum).

Stored in WX-1 v0.1.8 containers with zstd_shuffle. Strictly 0% AI / 0% ML.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import wx1

from tempest_v2.copernicus_dem import (
    resolve_local_dem_root,
    sample_copernicus_on_2d_grid,
    write_copernicus_static_wx1,
)
from tempest_v2.projection import DomainSpec, get_domain_spec

logger = logging.getLogger("tempest_v2.static_fields")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = PROJECT_ROOT / "data" / "static"


def get_benelux_grid_coords(resolution_m: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Compatibility helper returning (lats, lons) 1D bounds or D3 Benelux grid axes."""
    if resolution_m is None:
        resolution_m = int(os.environ.get("TEMPEST_D3_RESOLUTION_M", "2500"))
    if resolution_m == 2500:
        spec = get_domain_spec("d3")
        return spec.lat2d[:, spec.nx // 2].astype(np.float32), spec.lon2d[spec.ny // 2, :].astype(np.float32)
    if resolution_m == 1500:
        spec = get_domain_spec("d3_benelux_de_1p5km")
        return spec.lat2d[:, spec.nx // 2].astype(np.float32), spec.lon2d[spec.ny // 2, :].astype(np.float32)
    scale = 2500.0 / float(resolution_m)
    ny = int(round(420 * scale))
    nx = int(round(360 * scale))
    return (
        np.linspace(48.0, 57.5, ny, dtype=np.float32),
        np.linspace(-1.5, 10.5, nx, dtype=np.float32),
    )


def benelux_grid_tag() -> str:
    metres = int(os.environ.get("TEMPEST_D3_RESOLUTION_M", "2500"))
    return {2500: "2p5km", 1500: "1p5km", 500: "500m"}.get(metres, f"{metres}m")


def ensure_domain_copernicus_static(
    spec: DomainSpec,
    *,
    output_dir: Path | str = STATIC_DIR,
    local_dem_root: Optional[Path | str] = None,
    prefer_glo30: bool = True,
    smooth_passes: int = 1,
    sso_prefilter_m: float = 250.0,
    force: bool = False,
) -> Path:
    """Build or load cached Copernicus DEM + WBM static WX-1 container for any DomainSpec."""
    out_dir = Path(output_dir)
    out_path = out_dir / f"static_copernicus_{spec.name}.wx1"
    if out_path.is_file() and not force:
        with wx1.open_dataset(out_path) as ds:
            has_prefilter = "sso_prefilter_m" in ds.attrs
        if has_prefilter or not resolve_local_dem_root(local_dem_root).exists():
            return out_path
    fields = sample_copernicus_on_2d_grid(
        spec.lat2d,
        spec.lon2d,
        spec.dx_m,
        spec.dy_m,
        local_root=local_dem_root,
        prefer_glo30=prefer_glo30,
        offline=True,
        compute_subgrid_sso=True,
        smooth_passes=smooth_passes,
        sso_prefilter_m=sso_prefilter_m,
    )
    return write_copernicus_static_wx1(
        out_path,
        fields,
        spec.x_m,
        spec.y_m,
        spec.lat2d,
        spec.lon2d,
        domain_name=spec.name,
        is_projected=spec.is_projected,
        sso_prefilter_m=sso_prefilter_m,
    )


def load_static_fields_dict(static_wx1_path: Path | str) -> Dict[str, np.ndarray]:
    """Load Copernicus static fields from WX-1 v0.1.8 container."""
    ds = wx1.open_dataset(static_wx1_path)
    try:
        out = {
            name: np.asarray(ds[name].values, dtype=np.float32)
            for name in ("orography_m", "orography_smooth_m", "orography_std_m", "lsm_frac", "lake_frac", "river_frac")
            if name in ds
        }
    finally:
        ds.close()
    return out
