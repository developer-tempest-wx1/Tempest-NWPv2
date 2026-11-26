"""Unified Offline Static Fields, ECMWF/Aurora IC Builder, and Lossless WX-1 Restart I/O.

Strictly 100% deterministic meteorological preprocessing (0% AI / 0% ML) in SI units:
- Combines Copernicus DEM 30m/90m, GEBCO 2026 bathymetry, and FAO HWSD v2.2 soil classes/profiles.
- Converts live ECMWF IFS GRIB2 analyses (`0p1` / `0p25`) or Microsoft Aurora 1.5 `.wx1` snapshots
  onto staggered C-grid LCC + Klemp (2011) `nz=65`/`90`/`137` prognostic state and 4-layer LSM/COARE/FLake surface state.
- Saves/loads bit-exact `zstd_shuffle` WX-1 v0.1.8 restart checkpoints and forecasts.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt
import xarray as xr
import wx1

from tempest_v2.aurora_adapter import aurora_wx1_to_domain_state
from tempest_v2.bathymetry import load_bathymetry, prepare_bathymetry_on_2d_grid
from tempest_v2.cuda_bridge import NonHydrostaticGPUSolver
from tempest_v2.hwsd22_soil import sample_hwsd22_on_grid
from tempest_v2.land_static import (
    ensure_domain_soilgrids_static,
    ensure_domain_worldcover_static,
    resolve_soilgrids_root,
    resolve_worldcover_root,
)
from tempest_v2.soil_model import remap_ecmwf_swvl_via_smi
from tempest_v2.projection import (
    CP_D,
    GRAVITY_M_S2,
    KAPPA,
    P0_PA,
    R_D,
    DomainSpec,
    VerticalGrid,
)
from tempest_v2.static_fields import (
    STATIC_DIR,
    ensure_domain_copernicus_static,
    load_static_fields_dict,
)

logger = logging.getLogger("tempest_v2.ic_builder")

def _resolve_default_static_dir(env_name: str, local_sub: str) -> Path:
    env_val = os.environ.get(env_name)
    if env_val:
        return Path(env_val)
    return STATIC_DIR / local_sub


def _pick_gebco_elevation_nc(gebco_root: Path) -> Optional[Path]:
    """Select primary GEBCO elevation NetCDF under `gebco_root`, skipping TID masks."""
    candidates = [
        p
        for p in sorted(Path(gebco_root).rglob("*.nc"))
        if "_tid" not in p.name.lower()
        and "tid" not in p.stem.lower().split("_")
        and "tid" not in p.parent.name.lower().split("_")
    ]
    if not candidates:
        return None
    for preferred in ("gebco_2026.nc", "gebco_2025.nc"):
        for p in candidates:
            if p.name.lower() == preferred and "sub_ice" not in p.name.lower():
                return p
    non_sub_ice = [p for p in candidates if "sub_ice" not in p.name.lower()]
    return non_sub_ice[0] if non_sub_ice else candidates[0]


DEFAULT_COPERNICUS_ROOT = _resolve_default_static_dir("TEMPEST_COPERNICUS_DEM_DIR", "copernicus_dem_raw")
DEFAULT_GEBCO_ROOT = _resolve_default_static_dir("TEMPEST_BATHY_DIR", "bathymetry_raw")


def stagger_uv(u_c: np.ndarray, v_c: np.ndarray, *, periodic_x: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate cell-centered 3D winds `[nz, ny, nx]` onto Arakawa C-grid faces."""
    u = np.asarray(u_c, dtype=np.float32)
    v = np.asarray(v_c, dtype=np.float32)
    nz, ny, nx = u.shape
    u_st = np.empty((nz, ny, nx + 1), dtype=np.float32)
    u_st[:, :, 1:-1] = 0.5 * (u[:, :, :-1] + u[:, :, 1:])
    if periodic_x:
        u_seam = 0.5 * (u[:, :, -1] + u[:, :, 0])
        u_st[:, :, 0] = u_seam
        u_st[:, :, -1] = u_seam
    else:
        u_st[:, :, 0] = u[:, :, 0]
        u_st[:, :, -1] = u[:, :, -1]

    v_st = np.empty((nz, ny + 1, nx), dtype=np.float32)
    v_st[:, 1:-1, :] = 0.5 * (v[:, :-1, :] + v[:, 1:, :])
    v_st[:, 0, :] = v[:, 0, :]
    v_st[:, -1, :] = v[:, -1, :]
    return u_st, v_st


def unstagger_uv(u_st: np.ndarray, v_st: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate C-grid staggered winds back onto scalar cell centers `[nz, ny, nx]`."""
    u = np.asarray(u_st, dtype=np.float32)
    v = np.asarray(v_st, dtype=np.float32)
    u_c = 0.5 * (u[:, :, :-1] + u[:, :, 1:])
    v_c = 0.5 * (v[:, :-1, :] + v[:, 1:, :])
    return u_c.astype(np.float32), v_c.astype(np.float32)


def _fill_nan_nearest(arr: np.ndarray, fallback: float) -> np.ndarray:
    """Fill NaNs cleanly via nearest finite neighbor to avoid coastal edge holes."""
    out = np.asarray(arr, dtype=np.float64).copy()
    mask = ~np.isfinite(out)
    if np.all(mask):
        out.fill(fallback)
        return out
    if np.any(mask):
        _, indices = distance_transform_edt(mask, return_indices=True)
        out[mask] = out[tuple(indices)][mask]
    return out


def build_domain_static_fields(
    spec: DomainSpec,
    *,
    static_dir: Path | str = STATIC_DIR,
    dem_root: Optional[Path | str] = None,
    gebco_root: Optional[Path | str] = DEFAULT_GEBCO_ROOT,
    worldcover_root: Optional[Path | str] = None,
    soilgrids_root: Optional[Path | str] = None,
    use_worldcover: bool = True,
    use_soilgrids: bool = True,
    force: bool = False,
    allow_synthetic_fallback: bool = True,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Build or load cached Copernicus DEM + GEBCO 2026 + HWSD v2.2 / SoilGrids / WorldCover static fields on `spec`."""
    sdir = Path(static_dir)
    sdir.mkdir(parents=True, exist_ok=True)
    ny, nx = spec.ny, spec.nx

    resolved_dem = Path(dem_root) if dem_root is not None else (DEFAULT_COPERNICUS_ROOT if DEFAULT_COPERNICUS_ROOT.exists() else None)
    cop_cache = sdir / f"static_copernicus_{spec.name}.wx1"
    used_fast_global_gebco = False
    if not spec.is_projected and spec.dx_m >= 15_000.0 and not cop_cache.is_file() and gebco_root is not None and Path(gebco_root).exists():
        gebco_nc = _pick_gebco_elevation_nc(Path(gebco_root))
        if gebco_nc is not None:
            try:
                bathy_fast = sdir / f"bathymetry_{spec.name}.wx1"
                if not bathy_fast.is_file() or force:
                    prepare_bathymetry_on_2d_grid(
                        gebco_nc, spec.name, spec.lat2d, spec.lon2d, vertical_datum="msl", output_path=bathy_fast
                    )
                with xr.open_dataset(gebco_nc) as gds:
                    sy = max(1, int(gds["elevation"].sizes["lat"]) // max(2 * ny, 720))
                    sx = max(1, int(gds["elevation"].sizes["lon"]) // max(2 * nx, 1440))
                    sub = gds["elevation"].isel(lat=slice(None, None, sy), lon=slice(None, None, sx))
                    slat = np.asarray(sub.coords["lat"].values, dtype=np.float64)
                    slon = np.asarray(sub.coords["lon"].values, dtype=np.float64)
                    selev = np.asarray(sub.values, dtype=np.float64)
                if slat[0] > slat[-1]:
                    slat = slat[::-1]
                    selev = selev[::-1, :]
                slon_w = np.append(slon, slon[0] + 360.0)
                selev_w = np.concatenate([selev, selev[:, :1]], axis=1)
                tgt_lon = ((spec.lon2d + 180.0) % 360.0) - 180.0
                tgt_lon = np.where(tgt_lon < slon_w[0], tgt_lon + 360.0, tgt_lon)
                itp_e = RegularGridInterpolator((slat, slon_w), selev_w, method="linear", bounds_error=False, fill_value=0.0)
                elev2d = itp_e(np.column_stack([spec.lat2d.ravel(), tgt_lon.ravel()])).reshape(ny, nx)
                hs_m = np.maximum(elev2d, 0.0).astype(np.float64)
                lsm_frac = np.where(elev2d > 0.0, 1.0, 0.0).astype(np.float32)
                # 1-pass 2*dx Shapiro filter matching D2/D3 Copernicus DEM path (keeps ocean = 0.0 m, wraps periodic longitude)
                w_s = 0.125
                hs_pad_y = np.pad(hs_m, ((1, 1), (0, 0)), mode="edge")
                hs_pad = np.pad(hs_pad_y, ((0, 0), (1, 1)), mode="wrap" if spec.is_periodic_lon else "edge")
                hs_filt = (1.0 - 4.0 * w_s) * hs_m + w_s * (
                    hs_pad[:-2, 1:-1] + hs_pad[2:, 1:-1] + hs_pad[1:-1, :-2] + hs_pad[1:-1, 2:]
                )
                hs_m = np.where(lsm_frac >= 0.05, hs_filt, 0.0)
                lake_frac = np.zeros((ny, nx), dtype=np.float32)
                oro_std = np.zeros((ny, nx), dtype=np.float32)
                used_fast_global_gebco = True
            except Exception as exc:
                logger.warning("Fast global GEBCO static fallback skipped (%s); trying Copernicus DEM.", exc)

    if used_fast_global_gebco:
        pass
    elif resolved_dem is not None and resolved_dem.exists():
        try:
            cop_path = ensure_domain_copernicus_static(
                spec, output_dir=sdir, local_dem_root=resolved_dem, prefer_glo30=True, smooth_passes=1, force=force
            )
            cop = load_static_fields_dict(cop_path)
            hs_m = np.asarray(cop["orography_smooth_m"], dtype=np.float64)
            lsm_frac = np.asarray(cop["lsm_frac"], dtype=np.float32)
            lake_frac = np.asarray(cop["lake_frac"], dtype=np.float32)
            oro_std = np.asarray(cop.get("orography_std_m", np.zeros((ny, nx), dtype=np.float32)), dtype=np.float32)
        except FileNotFoundError:
            if not allow_synthetic_fallback:
                raise
            logger.warning("Incomplete local Copernicus DEM coverage for %s; falling back to smooth flat terrain.", spec.name)
            hs_m = np.zeros((ny, nx), dtype=np.float64)
            lsm_frac = np.ones((ny, nx), dtype=np.float32)
            lake_frac = np.zeros((ny, nx), dtype=np.float32)
            oro_std = np.zeros((ny, nx), dtype=np.float32)
    else:
        if not allow_synthetic_fallback:
            raise FileNotFoundError(f"Copernicus DEM root not found: {resolved_dem}")
        hs_m = np.zeros((ny, nx), dtype=np.float64)
        lsm_frac = np.ones((ny, nx), dtype=np.float32)
        lake_frac = np.zeros((ny, nx), dtype=np.float32)
        oro_std = np.zeros((ny, nx), dtype=np.float32)

    is_land = lsm_frac >= 0.5
    is_lake = (~is_land) & (lake_frac >= 0.35)
    lsm_bin = np.where(is_land, 1.0, 0.0).astype(np.float32)
    lake_bin = np.where(is_lake, 1.0, 0.0).astype(np.float32)

    # GEBCO 2026 bathymetry (positive-down water depth [m])
    bathy_cache = sdir / f"bathymetry_{spec.name}.wx1"
    depth = np.where(is_lake, 4.5, np.where(~is_land, 35.0, 0.0)).astype(np.float32)
    if bathy_cache.is_file() and not force:
        loaded_b = load_bathymetry(bathy_cache, expected_shape=(ny, nx))
        if loaded_b is not None:
            depth = np.asarray(loaded_b[0], dtype=np.float32)
    elif gebco_root is not None and Path(gebco_root).exists():
        gebco_nc = _pick_gebco_elevation_nc(Path(gebco_root))
        if gebco_nc is not None:
            prepare_bathymetry_on_2d_grid(
                gebco_nc, spec.name, spec.lat2d, spec.lon2d, vertical_datum="msl", output_path=bathy_cache
            )
            loaded_b = load_bathymetry(bathy_cache, expected_shape=(ny, nx))
            if loaded_b is not None:
                depth = np.asarray(loaded_b[0], dtype=np.float32)
    depth = np.where(is_land, 0.0, np.maximum(depth, np.where(is_lake, 4.5, 5.0))).astype(np.float32)

    # FAO HWSD v2.2 USDA soil texture class (0 water, 1..13 mineral/organic classes)
    soil_cache = sdir / f"soil_type_{spec.name}_hwsd22.wx1"
    soil_type = np.where(is_land, 9.0, 0.0).astype(np.float32)
    if soil_cache.is_file() and not force:
        with wx1.open_dataset(soil_cache) as sds:
            soil_type = np.asarray(sds["soil_type"].values, dtype=np.float32)
    else:
        try:
            soil_cache = sample_hwsd22_on_grid(spec.lat2d, spec.lon2d, spec.name, output_path=soil_cache)
            with wx1.open_dataset(soil_cache) as sds:
                soil_type = np.asarray(sds["soil_type"].values, dtype=np.float32)
        except Exception as exc:
            if not allow_synthetic_fallback:
                raise
            logger.warning("HWSD v2.2 lookup skipped (%s); using loam (class 9) over land.", exc)
    soil_type = np.where(is_land, np.where(soil_type >= 1.0, soil_type, 9.0), 0.0).astype(np.float32)

    # SoilGrids 250 m seamless European override (texture classes 1..13 + organic peat class 14)
    if use_soilgrids and resolve_soilgrids_root(soilgrids_root) is not None:
        try:
            sg_cache = ensure_domain_soilgrids_static(
                spec,
                lsm_land=lsm_bin,
                output_dir=sdir,
                local_root=soilgrids_root,
                force=force,
            )
            with wx1.open_dataset(sg_cache) as sgds:
                sg_valid = np.asarray(sgds["sg_valid"].values, dtype=np.float32) >= 0.5
                sg_stype = np.asarray(sgds["soil_type"].values, dtype=np.float32)
            soil_type = np.where(is_land & sg_valid & (sg_stype >= 1.0), sg_stype, soil_type)
        except Exception as exc:
            logger.warning("SoilGrids 250 m lookup skipped (%s); keeping HWSD v2.2.", exc)
    soil_type = np.where(is_land, np.where(soil_type >= 1.0, soil_type, 9.0), 0.0).astype(np.float32)

    # Roughness length z0m, albedo, greenness fraction (ESA WorldCover 10 m when available, else elevation fallback)
    z0m_land_fb = np.clip(0.15 + 0.0015 * oro_std + 0.08 * np.clip(hs_m / 600.0, 0.0, 1.0), 0.05, 1.2)
    albedo_fb = np.where(hs_m > 1800.0, 0.35, 0.19)
    veg_fb = np.clip(0.65 - 0.35 * np.clip((hs_m - 600.0) / 1400.0, 0.0, 1.0), 0.15, 0.75)

    z0m_land = z0m_land_fb
    albedo_land = albedo_fb
    veg_land = veg_fb
    if use_worldcover and resolve_worldcover_root(worldcover_root) is not None:
        try:
            wc_cache = ensure_domain_worldcover_static(
                spec,
                output_dir=sdir,
                local_root=worldcover_root,
                force=force,
            )
            with wx1.open_dataset(wc_cache) as wcds:
                wc_valid = np.asarray(wcds["wc_valid"].values, dtype=np.float32) >= 0.5
                wc_land_frac = np.asarray(wcds["wc_land_frac"].values, dtype=np.float32)
                wc_z0m_veg = np.asarray(wcds["z0m_veg"].values, dtype=np.float64)
                wc_alb = np.asarray(wcds["albedo_land"].values, dtype=np.float64)
                wc_veg = np.asarray(wcds["veg_frac"].values, dtype=np.float64)
                wc_dh = np.asarray(wcds["dsm_height_offset_m"].values, dtype=np.float64)

            wc_land_ok = is_land & wc_valid & (wc_land_frac > 0.0)
            hs_m = np.where(wc_land_ok, hs_m - wc_dh, hs_m)
            hs_m = np.where(is_land, hs_m, 0.0)

            z_b = 50.0
            inv_ln2_veg = 1.0 / np.square(np.log(z_b / np.clip(wc_z0m_veg, 0.002, 1.5)))
            sigma_eff = np.maximum(0.0, np.asarray(oro_std, dtype=np.float64) - 15.0)
            inv_ln2_tot = inv_ln2_veg + 1.5 * np.square(sigma_eff / 1000.0)
            z0m_wc = np.clip(z_b * np.exp(-1.0 / np.sqrt(np.maximum(inv_ln2_tot, 1e-6))), 0.002, 1.5)

            z0m_land = np.where(wc_land_ok, z0m_wc, z0m_land_fb)
            albedo_land = np.where(wc_land_ok, wc_alb, albedo_fb)
            veg_land = np.where(wc_land_ok, wc_veg, veg_fb)
        except Exception as exc:
            logger.warning("ESA WorldCover 10 m lookup skipped (%s); using elevation fallback.", exc)

    z0m_base = np.where(is_land, z0m_land, 1.0e-4).astype(np.float32)
    albedo_base = np.where(is_land, albedo_land, 0.06).astype(np.float32)
    veg_frac = np.where(is_land, veg_land, 0.0).astype(np.float32)

    surface_static = {
        "lat_deg": spec.lat2d.astype(np.float32),
        "lon_deg": spec.lon2d.astype(np.float32),
        "lsm": lsm_bin,
        "lake_mask": lake_bin,
        "soil_type": soil_type,
        "bathy_depth_m": depth,
        "z0m_base": z0m_base,
        "albedo_base": albedo_base,
        "veg_frac": veg_frac,
    }
    return hs_m, surface_static


def _regrid_latlon_field(
    src: np.ndarray,
    src_lats: np.ndarray,
    src_lons: np.ndarray,
    target_spec: DomainSpec,
    *,
    fill_nan_value: Optional[float] = None,
) -> np.ndarray:
    src_raw = np.asarray(src)
    lats_raw = np.asarray(src_lats, dtype=np.float64)
    lons_raw = np.asarray(src_lons, dtype=np.float64)

    # Crop global lat/lon inputs (e.g. ECMWF 0p1 1801x3600) to regional bbox + 1.5 deg before float64 copy.
    cropped = False
    tgt_lon_180 = ((np.asarray(target_spec.lon2d, dtype=np.float64) + 180.0) % 360.0) - 180.0
    lat_min = max(-90.0, float(np.min(target_spec.lat2d)) - 1.5)
    lat_max = min(90.0, float(np.max(target_spec.lat2d)) + 1.5)
    lon_min = float(np.min(tgt_lon_180)) - 1.5
    lon_max = float(np.max(tgt_lon_180)) + 1.5
    is_regional = (
        not target_spec.is_periodic_lon
        and (lon_max - lon_min) < 180.0
        and lon_min > -179.5
        and lon_max < 179.5
        and len(lats_raw) > 4
        and len(lons_raw) > 4
    )
    if is_regional:
        lat_mask = (lats_raw >= lat_min) & (lats_raw <= lat_max)
        lat_idx = np.where(lat_mask)[0]
        if lat_idx.size >= 2:
            j0 = max(0, int(lat_idx[0]) - 1)
            j1 = min(len(lats_raw) - 1, int(lat_idx[-1]) + 1)
            lats_c = lats_raw[j0 : j1 + 1]
            sub_y = src_raw[..., j0 : j1 + 1, :]

            idx_sorted: Optional[np.ndarray] = None
            lons_c: Optional[np.ndarray] = None
            if lons_raw.max() > 180.0 and lons_raw.min() >= 0.0:
                if lon_min < 0.0 <= lon_max:
                    iw = np.where(lons_raw >= 360.0 + lon_min)[0]
                    ie = np.where(lons_raw <= lon_max)[0]
                    if iw.size >= 1 and ie.size >= 1:
                        iw0, iw1 = max(0, int(iw[0]) - 1), min(len(lons_raw) - 1, int(iw[-1]) + 1)
                        ie0, ie1 = max(0, int(ie[0]) - 1), min(len(lons_raw) - 1, int(ie[-1]) + 1)
                        idx_comb = np.concatenate([np.arange(iw0, iw1 + 1), np.arange(ie0, ie1 + 1)])
                        _, uniq_pos = np.unique(idx_comb, return_index=True)
                        idx_comb = idx_comb[np.sort(uniq_pos)]
                        lons_wrapped = ((lons_raw[idx_comb] + 180.0) % 360.0) - 180.0
                        order = np.argsort(lons_wrapped)
                        idx_sorted = idx_comb[order]
                        lons_c = lons_wrapped[order]
                elif lon_min >= 0.0:
                    il = np.where((lons_raw >= lon_min) & (lons_raw <= lon_max))[0]
                    if il.size >= 2:
                        i0, i1 = max(0, int(il[0]) - 1), min(len(lons_raw) - 1, int(il[-1]) + 1)
                        idx_sorted = np.arange(i0, i1 + 1)
                        lons_c = lons_raw[idx_sorted]
                else:  # lon_max < 0.0
                    il = np.where((lons_raw >= 360.0 + lon_min) & (lons_raw <= 360.0 + lon_max))[0]
                    if il.size >= 2:
                        i0, i1 = max(0, int(il[0]) - 1), min(len(lons_raw) - 1, int(il[-1]) + 1)
                        idx_sorted = np.arange(i0, i1 + 1)
                        lons_c = lons_raw[idx_sorted] - 360.0
            else:
                il = np.where((lons_raw >= lon_min) & (lons_raw <= lon_max))[0]
                if il.size >= 2:
                    i0, i1 = max(0, int(il[0]) - 1), min(len(lons_raw) - 1, int(il[-1]) + 1)
                    idx_sorted = np.arange(i0, i1 + 1)
                    lons_c = lons_raw[idx_sorted]

            if idx_sorted is not None and lons_c is not None and len(lons_c) >= 2 and np.all(np.diff(lons_c) > 0):
                data = np.asarray(sub_y[..., :, idx_sorted], dtype=np.float64)
                lats = lats_c
                lons = lons_c
                cropped = True

    if not cropped:
        data = np.asarray(src_raw, dtype=np.float64)
        lats = lats_raw
        lons = lons_raw

    if lats[0] > lats[-1]:
        lats = lats[::-1]
        data = data[..., ::-1, :]

    if not cropped:
        if lons.max() > 180.0 and lons.min() >= 0.0:
            # Unwrap 0..360 -> -180..180 sorted cleanly if global/Benelux spans 0E
            lons_w = ((lons + 180.0) % 360.0) - 180.0
            order = np.argsort(lons_w)
            lons = lons_w[order]
            data = data[..., :, order]

        if len(lons) >= 2 and (lons[-1] - lons[0]) > 350.0:
            lons = np.append(lons, lons[0] + 360.0)
            data = np.concatenate([data, data[..., :, :1]], axis=-1)

    tgt_lon = tgt_lon_180
    if len(lons) >= 2 and lons[-1] > 180.0:
        tgt_lon = np.where(tgt_lon < lons[0], tgt_lon + 360.0, tgt_lon)
    pts = np.column_stack([target_spec.lat2d.ravel(), tgt_lon.ravel()])

    def _interp_2d(slc: np.ndarray) -> np.ndarray:
        if fill_nan_value is not None:
            slc = _fill_nan_nearest(slc, fill_nan_value)
        itp = RegularGridInterpolator((lats, lons), slc, method="linear", bounds_error=False, fill_value=None)
        return itp(pts).reshape(target_spec.ny, target_spec.nx)

    if data.ndim == 2:
        return _interp_2d(data)
    out = np.empty((data.shape[0], target_spec.ny, target_spec.nx), dtype=np.float64)
    for k in range(data.shape[0]):
        out[k] = _interp_2d(data[k])
    return out


_CFGRIB_CACHE: Dict[Tuple[str, float], Dict[str, Tuple[xr.DataArray, np.ndarray, np.ndarray]]] = {}


def clear_cfgrib_cache() -> None:
    """Release cached cfgrib DataArrays to reclaim host RAM after IC loading."""
    _CFGRIB_CACHE.clear()


def _read_cfgrib_var_map(
    grib_path: Path, *, use_cache: bool = True
) -> Dict[str, Tuple[xr.DataArray, np.ndarray, np.ndarray]]:
    import warnings
    import cfgrib

    key = (str(grib_path.resolve()), grib_path.stat().st_mtime if grib_path.exists() else 0.0)
    if use_cache and key in _CFGRIB_CACHE:
        return _CFGRIB_CACHE[key]
    if use_cache and len(_CFGRIB_CACHE) >= 4:
        _CFGRIB_CACHE.pop(next(iter(_CFGRIB_CACHE)))

    found: Dict[str, Tuple[xr.DataArray, np.ndarray, np.ndarray]] = {}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        for ds in cfgrib.open_datasets(str(grib_path), backend_kwargs={"indexpath": ""}):
            lat_name = next((c for c in ("latitude", "lat") if c in ds.coords), None)
            lon_name = next((c for c in ("longitude", "lon") if c in ds.coords), None)
            if lat_name is None or lon_name is None:
                ds.close()
                continue
            lats = np.asarray(ds.coords[lat_name].values, dtype=np.float64)
            lons = np.asarray(ds.coords[lon_name].values, dtype=np.float64)
            for vname in ds.data_vars:
                da = ds[vname].load()
                found[str(vname)] = (da, lats, lons)
            ds.close()
    if use_cache:
        _CFGRIB_CACHE[key] = found
    return found


def _q_from_dewpoint_pa(
    td_k: np.ndarray,
    p_pa: np.ndarray,
    t_k: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Convert 2 m dewpoint temperature [K] and surface pressure [Pa] to specific humidity [kg/kg] via Bolton/Magnus."""
    td_eff = np.asarray(td_k, dtype=np.float64)
    if t_k is not None:
        td_eff = np.minimum(td_eff, np.asarray(t_k, dtype=np.float64))
    tc = np.clip(td_eff - 273.15, -90.0, 60.0)
    es = 611.2 * np.exp((17.67 * tc) / (tc + 243.5))
    p_safe = np.maximum(np.asarray(p_pa, dtype=np.float64), 1000.0)
    es = np.minimum(es, 0.90 * p_safe)
    return np.clip((0.62198 * es) / np.maximum(p_safe - 0.37802 * es, 100.0), 1e-7, 0.04)


def _vertical_map_isobaric_to_klemp(
    spec: DomainSpec,
    hs_m: np.ndarray,
    p_levels_pa: np.ndarray,
    z_h: np.ndarray,
    t_h: np.ndarray,
    u_geo_h: np.ndarray,
    v_geo_h: np.ndarray,
    q_h: np.ndarray,
    ps_child: np.ndarray,
    *,
    theta_sfc_ref: float,
    p_sfc_ref: float,
    n_ref_hz: float,
    t2m_h: Optional[np.ndarray] = None,
    q2m_h: Optional[np.ndarray] = None,
    u10_geo_h: Optional[np.ndarray] = None,
    v10_geo_h: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Interpolate isobaric fields onto Klemp coordinates with C-grid winds and hydrostatic Exner balance (row-batched float32)."""
    nz, ny, nx = spec.nz, spec.ny, spec.nx
    if z_h[-1, 0, 0] < z_h[0, 0, 0]:
        p_levels_pa = p_levels_pa[::-1]
        z_h = z_h[::-1]
        t_h = t_h[::-1]
        u_geo_h = u_geo_h[::-1]
        v_geo_h = v_geo_h[::-1]
        q_h = q_h[::-1]

    if spec.is_projected and spec.projection is not None:
        u_grid_h, v_grid_h = spec.projection.geo_to_grid_wind(u_geo_h, v_geo_h, spec.lon2d[None, :, :])
    else:
        u_grid_h, v_grid_h = u_geo_h, v_geo_h

    if u10_geo_h is not None and v10_geo_h is not None:
        u10_in = np.asarray(u10_geo_h, dtype=np.float64)
        v10_in = np.asarray(v10_geo_h, dtype=np.float64)
        if spec.is_projected and spec.projection is not None:
            u10_grid_h, v10_grid_h = spec.projection.geo_to_grid_wind(u10_in, v10_in, spec.lon2d)
        else:
            u10_grid_h, v10_grid_h = u10_in, v10_in
    else:
        u10_grid_h, v10_grid_h = None, None
    t2m_2d = None if t2m_h is None else np.asarray(t2m_h, dtype=np.float64)
    q2m_2d = None if q2m_h is None else np.asarray(q2m_h, dtype=np.float64)

    eps_v = (461.5 / R_D) - 1.0  # Matches EPS_V in nh_internal.cuh (~0.60769)
    nlev = len(p_levels_pa)
    ln_p = np.log(p_levels_pa)

    theta_p_out = np.empty((nz, ny, nx), dtype=np.float32)
    pi_p_out = np.empty((nz, ny, nx), dtype=np.float32)
    qv_out = np.empty((nz, ny, nx), dtype=np.float32)
    u_c_out = np.empty((nz, ny, nx), dtype=np.float32)
    v_c_out = np.empty((nz, ny, nx), dtype=np.float32)

    zeta_s = spec.vertical.zeta_s[:, None, None].astype(np.float64)
    b_s = spec.vertical.B_s[:, None, None].astype(np.float64)
    zeta_w = spec.vertical.zeta_w[:, None, None].astype(np.float64)
    b_w = spec.vertical.B_w[:, None, None].astype(np.float64)
    db_w = spec.vertical.dB_dzeta_w[1:-1, None, None].astype(np.float64)
    dzeta = spec.vertical.dzeta[:, None, None].astype(np.float64)
    dz_w_base = 0.5 * (dzeta[1:] + dzeta[:-1])

    hs_2d = np.asarray(hs_m, dtype=np.float64)
    ps_2d = (
        np.asarray(ps_child, dtype=np.float64)
        if ps_child is not None
        else np.full((ny, nx), np.nan, dtype=np.float64)
    )
    n2 = float(n_ref_hz) * float(n_ref_hz)
    g2_cp = (GRAVITY_M_S2 * GRAVITY_M_S2) / (CP_D * float(theta_sfc_ref) * max(n2, 1e-12))
    pi_sfc = (float(p_sfc_ref) / P0_PA) ** KAPPA

    chunk_ny = 64
    for j0 in range(0, ny, chunk_ny):
        j1 = min(ny, j0 + chunk_ny)
        ny_c = j1 - j0
        n_col = ny_c * nx
        hs_c = hs_2d[None, j0:j1, :]
        z_chunk = zeta_s + b_s * hs_c
        exp_fac = np.exp((n2 / GRAVITY_M_S2) * z_chunk)
        theta_bar_c = float(theta_sfc_ref) * exp_fac
        pi_bar_c = np.maximum(1e-4, pi_sfc - g2_cp * (1.0 - 1.0 / exp_fac))

        zs_b = z_h[:, j0:j1, :].reshape(nlev, n_col)
        zt_b = z_chunk.reshape(nz, n_col)
        t_b = t_h[:, j0:j1, :].reshape(nlev, n_col)
        u_b = u_grid_h[:, j0:j1, :].reshape(nlev, n_col)
        v_b = v_grid_h[:, j0:j1, :].reshape(nlev, n_col)
        q_b = q_h[:, j0:j1, :].reshape(nlev, n_col)

        # Vectorized vertical bracket index across all columns in chunk
        k_hi = np.clip(np.sum(zs_b[:, None, :] < zt_b[None, :, :], axis=0), 1, nlev - 1)
        k_lo = k_hi - 1
        col_idx = np.arange(n_col)[None, :]
        z_lo = zs_b[k_lo, col_idx]
        z_hi = zs_b[k_hi, col_idx]
        wt = np.clip((zt_b - z_lo) / np.maximum(z_hi - z_lo, 1e-6), 0.0, 1.0)

        t_lo = t_b[k_lo, col_idx]
        t_hi = t_b[k_hi, col_idx]
        q_lo = q_b[k_lo, col_idx]
        q_hi = q_b[k_hi, col_idx]

        # Interpolate temperature T(z) linearly in z (avoids convex theta chord bias)
        t_i = np.maximum(150.0, (1.0 - wt) * t_lo + wt * t_hi)
        u_i = (1.0 - wt) * u_b[k_lo, col_idx] + wt * u_b[k_hi, col_idx]
        v_i = (1.0 - wt) * v_b[k_lo, col_idx] + wt * v_b[k_hi, col_idx]
        q_i = np.clip((1.0 - wt) * q_lo + wt * q_hi, 1e-7, 0.04)

        # Exact linear-Tv(z) hypsometric pressure inside each isobaric bracket
        tv_lo = np.maximum(150.0, t_lo * (1.0 + eps_v * np.clip(q_lo, 0.0, 0.04)))
        tv_hi = np.maximum(150.0, t_hi * (1.0 + eps_v * np.clip(q_hi, 0.0, 0.04)))
        tv_i = np.maximum(150.0, t_i * (1.0 + eps_v * q_i))
        d_tv_br = tv_hi - tv_lo
        safe_denom = np.where(np.abs(d_tv_br) > 1e-3, np.log(tv_hi / tv_lo), 1.0)
        frac_p = np.where(np.abs(d_tv_br) > 1e-3, np.log(tv_i / tv_lo) / safe_denom, wt)
        frac_p = np.clip(frac_p, 0.0, 1.0)
        p_i = np.exp(ln_p[k_lo] + frac_p * (ln_p[k_hi] - ln_p[k_lo]))

        above = zt_b > zs_b[-1:, :]
        if np.any(above):
            t_top = np.maximum(180.0, t_b[-1:, :])
            q_top = np.clip(q_b[-1:, :], 0.0, 0.04)
            tv_top = t_top * (1.0 + eps_v * q_top)
            p_abv = p_levels_pa[-1] * np.exp(-GRAVITY_M_S2 * (zt_b - zs_b[-1:, :]) / (R_D * tv_top))
            p_i = np.where(above, p_abv, p_i)
            t_i = np.where(above, t_top, t_i)

        below = zt_b < zs_b[0:1, :]
        if np.any(below):
            gamma_std = 0.0065  # Fallback standard tropospheric lapse rate [K/m]
            dz_down = np.maximum(0.0, zs_b[0:1, :] - zt_b)
            t_bot = np.maximum(180.0, t_b[0:1, :])
            q_bot = np.clip(q_b[0:1, :], 0.0, 0.04)
            tv_bot = t_bot * (1.0 + eps_v * q_bot)

            ps_flat = ps_2d[j0:j1, :].reshape(1, n_col)
            hs_flat = hs_c.reshape(1, n_col)
            valid_hypso = (
                np.isfinite(ps_flat)
                & (ps_flat > p_levels_pa[0] + 150.0)
                & (zs_b[0:1, :] > hs_flat + 20.0)
            )
            log_ratio = np.where(
                valid_hypso,
                np.log(np.maximum(ps_flat, p_levels_pa[0] + 150.0) / p_levels_pa[0]),
                1.0,
            )
            tv_mean = np.where(
                valid_hypso,
                (GRAVITY_M_S2 * (zs_b[0:1, :] - hs_flat)) / np.maximum(R_D * log_ratio, 1e-3),
                tv_bot,
            )
            tv_sfc_est = np.clip(2.0 * tv_mean - tv_bot, tv_bot - 15.0, tv_bot + 15.0)
            t_sfc_est = tv_sfc_est / (1.0 + eps_v * q_bot)
            span_hypso = np.maximum(zs_b[0:1, :] - hs_flat, 1.0)
            wb = np.clip((zt_b - hs_flat) / span_hypso, 0.0, 1.0)
            t_hypso = (1.0 - wb) * t_sfc_est + wb * t_bot

            t_std = t_bot + gamma_std * dz_down
            t_bel = np.maximum(180.0, np.where(valid_hypso, t_hypso, t_std))
            tv_bel = t_bel * (1.0 + eps_v * q_bot)
            p_bel = p_levels_pa[0] * np.exp(GRAVITY_M_S2 * dz_down / (R_D * 0.5 * (tv_bot + tv_bel)))
            p_i = np.where(below, p_bel, p_i)
            t_i = np.where(below, t_bel, t_i)

        if (t2m_2d is not None) or (q2m_2d is not None) or (u10_grid_h is not None and v10_grid_h is not None):
            hs_flat_sbl = hs_c.reshape(1, n_col)
            col_1d = np.arange(n_col)
            k_top_sbl = np.clip(np.sum(zs_b <= (hs_flat_sbl + 15.0), axis=0), 0, nlev - 1)
            z_top_sbl = zs_b[k_top_sbl, col_1d][None, :]
            in_sbl = (zt_b < z_top_sbl) & (z_top_sbl > hs_flat_sbl + 15.0)
            if np.any(in_sbl):
                z_sfc_t = hs_flat_sbl + 2.0
                wt_t = np.clip((zt_b - z_sfc_t) / np.maximum(z_top_sbl - z_sfc_t, 1.0), 0.0, 1.0)
                if t2m_2d is not None:
                    t2_f = t2m_2d[j0:j1, :].reshape(1, n_col)
                    ok_t2 = in_sbl & np.isfinite(t2_f) & (t2_f > 180.0) & (t2_f < 350.0)
                    t_top_sbl = t_b[k_top_sbl, col_1d][None, :]
                    t_sbl = np.maximum(150.0, (1.0 - wt_t) * t2_f + wt_t * t_top_sbl)
                    t_i = np.where(ok_t2, t_sbl, t_i)
                if q2m_2d is not None:
                    q2_f = q2m_2d[j0:j1, :].reshape(1, n_col)
                    ok_q2 = in_sbl & np.isfinite(q2_f) & (q2_f > 0.0) & (q2_f < 0.05)
                    q_top_sbl = q_b[k_top_sbl, col_1d][None, :]
                    q_sbl = np.clip((1.0 - wt_t) * q2_f + wt_t * q_top_sbl, 1e-7, 0.04)
                    q_i = np.where(ok_q2, q_sbl, q_i)
                if u10_grid_h is not None and v10_grid_h is not None:
                    u10_f = u10_grid_h[j0:j1, :].reshape(1, n_col)
                    v10_f = v10_grid_h[j0:j1, :].reshape(1, n_col)
                    ok_u10 = in_sbl & np.isfinite(u10_f) & np.isfinite(v10_f)
                    z_sfc_u = hs_flat_sbl + 10.0
                    wt_u = np.clip((zt_b - z_sfc_u) / np.maximum(z_top_sbl - z_sfc_u, 1.0), 0.0, 1.0)
                    u_top_sbl = u_b[k_top_sbl, col_1d][None, :]
                    v_top_sbl = v_b[k_top_sbl, col_1d][None, :]
                    u_i = np.where(ok_u10, (1.0 - wt_u) * u10_f + wt_u * u_top_sbl, u_i)
                    v_i = np.where(ok_u10, (1.0 - wt_u) * v10_f + wt_u * v_top_sbl, v_i)
                # Recompute preliminary p_i below z_h[0] using the surface-anchored virtual temperature
                if np.any(below):
                    dz_down_sbl = np.maximum(0.0, zs_b[0:1, :] - zt_b)
                    tv_bot_sbl = np.maximum(180.0, t_b[0:1, :]) * (1.0 + eps_v * np.clip(q_b[0:1, :], 0.0, 0.04))
                    tv_bel_sbl = np.maximum(150.0, t_i) * (1.0 + eps_v * np.clip(q_i, 0.0, 0.04))
                    p_bel_sbl = p_levels_pa[0] * np.exp(GRAVITY_M_S2 * dz_down_sbl / (R_D * 0.5 * (tv_bot_sbl + tv_bel_sbl)))
                    p_i = np.where(below, p_bel_sbl, p_i)

        t_3d = t_i.reshape(nz, ny_c, nx)
        p_3d = p_i.reshape(nz, ny_c, nx)
        q_3d = np.clip(q_i.reshape(nz, ny_c, nx), 1e-7, 0.04)

        es_w = 611.2 * np.exp(17.67 * (t_3d - 273.15) / np.maximum(t_3d - 29.65, 1.0))
        qs_w = 0.62198 * es_w / np.maximum(p_3d - 0.378 * es_w, 10.0)
        q_3d = np.clip(np.minimum(q_3d, 0.995 * qs_w), 1e-7, 0.04)

        # Anchor lowest model level (k=0) to ps_child on lowlands / below bottom isobaric level
        ps_c = ps_2d[j0:j1, :]
        hs_2c = hs_2d[j0:j1, :]
        tv0 = t_3d[0] * (1.0 + eps_v * q_3d[0])
        dz0 = np.maximum(0.0, z_chunk[0] - hs_2c)
        p0_from_ps = np.where(
            np.isfinite(ps_c),
            ps_c * np.exp(-(GRAVITY_M_S2 * dz0) / (R_D * np.maximum(tv0, 180.0))),
            p_3d[0],
        )
        ps_ok = np.isfinite(ps_c) & (ps_c > 20_000.0) & (ps_c < 120_000.0)
        w_ps = np.where(ps_ok, np.where(z_chunk[0] < z_h[0, j0:j1, :], 1.0, np.clip(1.0 - hs_2c / 300.0, 0.0, 1.0)), 0.0)
        p0_anchor = w_ps * p0_from_ps + (1.0 - w_ps) * p_3d[0]
        pi0_anchor = np.power(np.maximum(p0_anchor, 10.0) / P0_PA, KAPPA)

        # Solve exact discrete hydrostatic balance matching nh_core.cu (3 fixed-point passes)
        pi_cur = np.power(np.maximum(p_3d, 10.0) / P0_PA, KAPPA)
        if nz > 1:
            z_w_c = zeta_w + b_w * hs_c
            dz_s_c = np.diff(z_w_c, axis=0)
            wt_w = dz_s_c[:-1] / (dz_s_c[1:] + dz_s_c[:-1])
            dz_w_cuda = (1.0 + db_w * hs_c) * dz_w_base
            thb_w = float(theta_sfc_ref) * np.exp((n2 / GRAVITY_M_S2) * z_w_c[1:-1])
            vf_c = 1.0 + eps_v * q_3d
            for _ in range(3):
                th_cur = np.maximum(50.0, t_3d / pi_cur)
                dthv = (th_cur - theta_bar_c) * vf_c + theta_bar_c * (vf_c - 1.0)
                dthv_w = (1.0 - wt_w) * dthv[:-1] + wt_w * dthv[1:]
                thv_w = np.maximum(50.0, thb_w + dthv_w)
                dpi_p = (GRAVITY_M_S2 * dthv_w / (CP_D * thv_w * thb_w)) * dz_w_cuda
                pip_new = np.empty_like(pi_cur)
                pip_new[0] = pi0_anchor - pi_bar_c[0]
                pip_new[1:] = pip_new[0:1] + np.cumsum(dpi_p, axis=0)
                pi_cur = np.maximum(1e-4, pi_bar_c + pip_new)
        else:
            pi_cur[0] = np.maximum(1e-4, pi0_anchor)

        th_cur = np.maximum(50.0, t_3d / pi_cur)
        p_cur = P0_PA * np.power(pi_cur, 1.0 / KAPPA)
        es_w = 611.2 * np.exp(17.67 * (t_3d - 273.15) / np.maximum(t_3d - 29.65, 1.0))
        qs_w = 0.62198 * es_w / np.maximum(p_cur - 0.378 * es_w, 10.0)
        q_3d = np.clip(np.minimum(q_3d, 0.995 * qs_w), 1e-7, 0.04)

        theta_p_out[:, j0:j1, :] = (th_cur - theta_bar_c).astype(np.float32)
        pi_p_out[:, j0:j1, :] = (pi_cur - pi_bar_c).astype(np.float32)
        qv_out[:, j0:j1, :] = q_3d.astype(np.float32)
        u_c_out[:, j0:j1, :] = u_i.reshape(nz, ny_c, nx).astype(np.float32)
        v_c_out[:, j0:j1, :] = v_i.reshape(nz, ny_c, nx).astype(np.float32)

    u_st, v_st = stagger_uv(u_c_out, v_c_out, periodic_x=spec.is_periodic_lon)
    del u_c_out, v_c_out

    out: Dict[str, np.ndarray] = {
        "u": u_st,
        "v": v_st,
        "w": np.zeros((nz + 1, ny, nx), dtype=np.float32),
        "theta_p": theta_p_out,
        "pi_p": pi_p_out,
        "qv": qv_out,
    }
    for sp in ("qc", "qr", "qi", "qs", "qg"):
        out[sp] = np.zeros((nz, ny, nx), dtype=np.float32)
    return out


def _empty_surface_state_for_spec(spec: DomainSpec) -> Dict[str, np.ndarray]:
    s2 = (spec.ny, spec.nx)
    s4 = (4, spec.ny, spec.nx)
    out = {k: np.zeros(s2, dtype=np.float32) for k in NonHydrostaticGPUSolver.SURFACE_2D_FIELDS}
    out["soil_t"] = np.full(s4, 285.0, dtype=np.float32)
    out["soil_m"] = np.full(s4, 0.28, dtype=np.float32)
    out["t_skin"].fill(286.0)
    out["sst_bulk"].fill(284.0)
    out["lake_tbot"].fill(281.0)
    out["lake_hmld"].fill(2.5)
    out["lake_ctheta"].fill(0.65)
    out["sea_ice_tsfc"].fill(271.35)
    out["sea_ice_albedo"].fill(0.64)
    return out


def ecmwf_grib_to_domain_state(
    sfc_grib: Path | str,
    pl_grib: Path | str,
    spec: DomainSpec,
    hs_m: np.ndarray,
    *,
    soil_grib: Optional[Path | str] = None,
    meta_json: Optional[Path | str] = None,
    surface_static: Optional[Dict[str, np.ndarray]] = None,
    theta_sfc_ref: float = 288.15,
    p_sfc_ref: float = 101_325.0,
    n_ref_hz: float = 0.01,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], datetime.datetime]:
    """Decode live ECMWF IFS GRIB2 step-0 files into balanced C-grid 3D + 4-layer surface IC."""
    sfc_p = Path(sfc_grib)
    pl_p = Path(pl_grib)
    if soil_grib is None:
        cand_soil = Path(str(sfc_p).replace("_sfc_", "_soil_"))
        if cand_soil.is_file():
            soil_grib = cand_soil
    if meta_json is None:
        cand_meta = Path(str(sfc_p).replace("_sfc_", "_").replace(".grib2", "_metadata.json"))
        if cand_meta.is_file():
            meta_json = cand_meta

    valid_dt: Optional[datetime.datetime] = None
    if meta_json is not None and Path(meta_json).is_file():
        meta = json.loads(Path(meta_json).read_text(encoding="utf-8"))
        v_iso = meta.get("valid_time_utc")
        if v_iso:
            valid_dt = datetime.datetime.fromisoformat(v_iso.replace("Z", "+00:00")).astimezone(datetime.timezone.utc)

    # 1. Decode 3D pressure levels
    pl_vars = _read_cfgrib_var_map(pl_p)
    t_da, pl_lats, pl_lons = pl_vars["t"]
    p_coord = next(c for c in ("isobaricInhPa", "level") if c in t_da.coords)
    p_hpa = np.asarray(t_da.coords[p_coord].values, dtype=np.float64)
    p_levels_pa = p_hpa * 100.0

    def _squeeze_pl(name: str) -> np.ndarray:
        arr = np.asarray(pl_vars[name][0].values, dtype=np.float64)
        while arr.ndim > 3:
            arr = arr[0]
        return arr

    if valid_dt is None and "valid_time" in t_da.coords:
        vt_raw = t_da.coords["valid_time"].values
        ts = int(np.datetime64(vt_raw, "s").astype(np.int64))
        valid_dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
    if valid_dt is None:
        valid_dt = datetime.datetime.now(datetime.timezone.utc).replace(minute=0, second=0, microsecond=0)

    t_h = _regrid_latlon_field(_squeeze_pl("t"), pl_lats, pl_lons, spec, fill_nan_value=260.0)
    u_geo_h = _regrid_latlon_field(_squeeze_pl("u"), pl_lats, pl_lons, spec, fill_nan_value=0.0)
    v_geo_h = _regrid_latlon_field(_squeeze_pl("v"), pl_lats, pl_lons, spec, fill_nan_value=0.0)
    q_h = np.clip(_regrid_latlon_field(_squeeze_pl("q"), pl_lats, pl_lons, spec, fill_nan_value=1e-4), 1e-7, 0.04)
    gh_name = "gh" if "gh" in pl_vars else "z"
    gh_arr = _squeeze_pl(gh_name)
    if gh_name == "z":
        gh_arr = gh_arr / GRAVITY_M_S2
    z_h = _regrid_latlon_field(gh_arr, pl_lats, pl_lons, spec, fill_nan_value=0.0)

    # 2. Decode 2D surface fields
    sfc_vars = _read_cfgrib_var_map(sfc_p)

    def _get_sfc(aliases: Tuple[str, ...], default_val: float) -> np.ndarray:
        for k in aliases:
            if k in sfc_vars:
                da, la, lo = sfc_vars[k]
                arr = np.asarray(da.values, dtype=np.float64)
                while arr.ndim > 2:
                    arr = arr[0]
                return _regrid_latlon_field(arr, la, lo, spec, fill_nan_value=default_val)
        return np.full((spec.ny, spec.nx), default_val, dtype=np.float64)

    msl_h = _get_sfc(("msl", "mslp"), 101_325.0)
    has_t2m = any(k in sfc_vars for k in ("t2m", "2t"))
    t2m_h = _get_sfc(("t2m", "2t"), 285.0)
    d2m_key = next((k for k in ("d2m", "2d") if k in sfc_vars), None)
    d2m_h = _get_sfc((d2m_key,), float(np.mean(t2m_h))) if d2m_key is not None else None
    u10_key = next((k for k in ("u10", "10u") if k in sfc_vars), None)
    v10_key = next((k for k in ("v10", "10v") if k in sfc_vars), None)
    u10_geo_h: Optional[np.ndarray] = None
    v10_geo_h: Optional[np.ndarray] = None
    u10_grid_h: Optional[np.ndarray] = None
    v10_grid_h: Optional[np.ndarray] = None
    if u10_key is not None and v10_key is not None:
        u10_geo_h = _get_sfc((u10_key,), 0.0)
        v10_geo_h = _get_sfc((v10_key,), 0.0)
        if spec.is_projected and spec.projection is not None:
            u10_grid_h, v10_grid_h = spec.projection.geo_to_grid_wind(u10_geo_h, v10_geo_h, spec.lon2d)
        else:
            u10_grid_h, v10_grid_h = u10_geo_h, v10_geo_h
    skt_key = next((k for k in ("skt", "t") if k in sfc_vars), None)
    skt_h = _get_sfc((skt_key,), float(np.mean(t2m_h))) if skt_key is not None else np.full((spec.ny, spec.nx), float(np.mean(t2m_h)), dtype=np.float64)
    if skt_key is not None and "lsm" in sfc_vars:
        skt_da, skt_la, skt_lo = sfc_vars[skt_key]
        lsm_da, _, _ = sfc_vars["lsm"]
        skt_raw = np.asarray(skt_da.values, dtype=np.float64)
        lsm_raw = np.asarray(lsm_da.values, dtype=np.float64)
        while skt_raw.ndim > 2:
            skt_raw = skt_raw[0]
        while lsm_raw.ndim > 2:
            lsm_raw = lsm_raw[0]
        if skt_raw.shape == lsm_raw.shape:
            skt_water_raw = np.where(lsm_raw < 0.5, skt_raw, np.nan)
            water_fallback = max(float(np.mean(t2m_h)), 278.0)
            sst_water_h = _regrid_latlon_field(skt_water_raw, skt_la, skt_lo, spec, fill_nan_value=water_fallback)
        else:
            sst_water_h = skt_h
    else:
        sst_water_h = skt_h
    sd_h = _get_sfc(("sd", "snow"), 0.0)
    # ECMWF IFS GRIB2 snow depth is in meters of water equivalent -> kg/m^2 = mm
    if "sd" in sfc_vars and "m" in str(sfc_vars["sd"][0].attrs.get("units", "")).lower():
        sd_h = sd_h * 1000.0
    sd_h = np.clip(sd_h, 0.0, 2000.0)

    hs = np.asarray(hs_m, dtype=np.float64)
    ps_child = msl_h * np.exp(-(GRAVITY_M_S2 * hs) / (R_D * np.maximum(t2m_h, 200.0)))
    q2m_ic = _q_from_dewpoint_pa(d2m_h, ps_child, t_k=t2m_h) if d2m_h is not None else None

    state_3d = _vertical_map_isobaric_to_klemp(
        spec,
        hs,
        p_levels_pa,
        z_h,
        t_h,
        u_geo_h,
        v_geo_h,
        q_h,
        ps_child,
        theta_sfc_ref=theta_sfc_ref,
        p_sfc_ref=p_sfc_ref,
        n_ref_hz=n_ref_hz,
        t2m_h=t2m_h if has_t2m else None,
        q2m_h=q2m_ic,
        u10_geo_h=u10_geo_h,
        v10_geo_h=v10_geo_h,
    )

    surf = _empty_surface_state_for_spec(spec)
    surf["t_skin"] = np.clip(skt_h, 210.0, 330.0).astype(np.float32)
    surf["t2m"] = np.clip(t2m_h, 210.0, 330.0).astype(np.float32)
    if d2m_h is not None:
        surf["q2m"] = _q_from_dewpoint_pa(d2m_h, ps_child, t_k=t2m_h).astype(np.float32)
    else:
        surf["q2m"] = np.clip(state_3d["qv"][0], 1e-7, 0.04).astype(np.float32)
    if u10_grid_h is not None and v10_grid_h is not None:
        surf["u10m"] = np.clip(u10_grid_h, -100.0, 100.0).astype(np.float32)
        surf["v10m"] = np.clip(v10_grid_h, -100.0, 100.0).astype(np.float32)
    surf["snow_swe"] = sd_h.astype(np.float32)
    surf["sst_bulk"] = np.clip(sst_water_h, 271.15, 310.0).astype(np.float32)
    if surface_static is not None and "lsm" in surface_static:
        is_ocean = (np.asarray(surface_static["lsm"]) < 0.5) & (np.asarray(surface_static.get("lake_mask", 0.0)) < 0.5)
        surf["t_skin"] = np.where(is_ocean, surf["sst_bulk"], surf["t_skin"]).astype(np.float32)
    surf["lake_tbot"] = np.clip(skt_h - 1.5, 273.15, 305.0).astype(np.float32)

    if soil_grib is not None and Path(soil_grib).is_file():
        sol_vars = _read_cfgrib_var_map(Path(soil_grib))
        sot_key = next((k for k in ("sot", "stl1", "st") if k in sol_vars), None)
        vsw_key = next((k for k in ("vsw", "swvl1", "swvl") if k in sol_vars), None)
        if sot_key is not None:
            da, la, lo = sol_vars[sot_key]
            arr = np.asarray(da.values, dtype=np.float64)
            while arr.ndim > 3:
                arr = arr[0]
            if arr.ndim == 2:
                arr = np.broadcast_to(arr[None, :, :], (4, arr.shape[0], arr.shape[1]))
            arr = np.where((arr < 150.0) | (arr > 360.0), np.nan, arr)
            surf["soil_t"] = np.clip(_regrid_latlon_field(arr[:4], la, lo, spec, fill_nan_value=float(np.mean(t2m_h))), 210.0, 340.0).astype(np.float32)
        if vsw_key is not None:
            da, la, lo = sol_vars[vsw_key]
            arr = np.asarray(da.values, dtype=np.float64)
            while arr.ndim > 3:
                arr = arr[0]
            if arr.ndim == 2:
                arr = np.broadcast_to(arr[None, :, :], (4, arr.shape[0], arr.shape[1]))
            arr = np.where((arr < 0.0) | (arr > 0.9), np.nan, arr)
            if "lsm" in sfc_vars:
                lsm_sol = np.asarray(sfc_vars["lsm"][0].values, dtype=np.float64)
                while lsm_sol.ndim > 2:
                    lsm_sol = lsm_sol[0]
                if lsm_sol.shape == arr.shape[1:] and np.any(lsm_sol >= 0.5):
                    arr = np.where(lsm_sol[None, :, :] >= 0.5, arr, np.nan)
            raw_sm = np.clip(_regrid_latlon_field(arr[:4], la, lo, spec, fill_nan_value=0.28), 0.01, 0.85)
            if surface_static is not None and "soil_type" in surface_static:
                slt_key = next((k for k in ("slt", "soil_type") if k in sfc_vars), None)
                slt_child = None
                if slt_key is not None:
                    s_da, s_la, s_lo = sfc_vars[slt_key]
                    s_arr = np.asarray(s_da.values, dtype=np.float64)
                    while s_arr.ndim > 2:
                        s_arr = s_arr[0]
                    slt_child = np.rint(_regrid_latlon_field(s_arr, s_la, s_lo, spec, fill_nan_value=2.0))
                surf["soil_m"] = remap_ecmwf_swvl_via_smi(raw_sm, surface_static["soil_type"], src_soil_type=slt_child)
            else:
                surf["soil_m"] = np.clip(raw_sm, 0.02, 0.77).astype(np.float32)
    else:
        for layer in range(4):
            surf["soil_t"][layer] = surf["t_skin"]

    if surface_static is not None and "bathy_depth_m" in surface_static:
        d = np.asarray(surface_static["bathy_depth_m"], dtype=np.float32)
        surf["lake_hmld"] = np.clip(0.6 * np.where(d > 0.5, d, 4.5), 0.2, 20.0).astype(np.float32)

    return state_3d, surf, valid_dt


def aurora_wx1_to_ic(
    wx1_path: Path | str,
    spec: DomainSpec,
    hs_m: np.ndarray,
    *,
    theta_sfc_ref: float = 288.15,
    p_sfc_ref: float = 101_325.0,
    n_ref_hz: float = 0.01,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Convert an Aurora 1.5 WX-1 analysis/step-0 snapshot into balanced C-grid 3D + surface IC."""
    raw = aurora_wx1_to_domain_state(
        str(wx1_path),
        spec,
        hs_m,
        theta_sfc_ref=theta_sfc_ref,
        p_sfc_ref=p_sfc_ref,
        n_ref_hz=n_ref_hz,
    )
    u_st, v_st = stagger_uv(raw["u"], raw["v"], periodic_x=spec.is_periodic_lon)
    nz, ny, nx = spec.nz, spec.ny, spec.nx
    state_3d: Dict[str, np.ndarray] = {
        "u": u_st,
        "v": v_st,
        "w": raw["w"].astype(np.float32),
        "theta_p": raw["theta_p"].astype(np.float32),
        "pi_p": raw["pi_p"].astype(np.float32),
        "qv": raw["qv"].astype(np.float32),
    }
    for sp in ("qc", "qr", "qi", "qs", "qg"):
        state_3d[sp] = np.zeros((nz, ny, nx), dtype=np.float32)

    t2m = raw.get("t2m", np.full((ny, nx), 285.0, dtype=np.float32))
    surf = _empty_surface_state_for_spec(spec)
    surf["t_skin"] = t2m.astype(np.float32)
    surf["t2m"] = t2m.astype(np.float32)
    surf["sst_bulk"] = np.clip(t2m, 271.15, 310.0).astype(np.float32)
    for m in range(4):
        surf["soil_t"][m] = t2m.astype(np.float32)
    return state_3d, surf


def save_tempest_restart_wx1(
    path: Path | str,
    valid_time_utc: datetime.datetime,
    spec: DomainSpec,
    hs_m: np.ndarray,
    state_3d: Dict[str, np.ndarray],
    surf_state: Dict[str, np.ndarray],
    *,
    sim_time_sec: float = 0.0,
    theta_sfc_ref: float = 288.15,
    p_sfc_ref: float = 101_325.0,
    n_ref_hz: float = 0.01,
) -> Path:
    """Save lossless bit-exact prognostic 3D + 2D + 4-layer soil checkpoint to WX-1."""
    out_p = Path(path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    v_dt = (
        valid_time_utc.astimezone(datetime.timezone.utc)
        if valid_time_utc.tzinfo
        else valid_time_utc.replace(tzinfo=datetime.timezone.utc)
    )
    t_coord = np.array([np.datetime64(int(v_dt.timestamp()), "s")], dtype="datetime64[s]")

    data_vars: Dict[str, Any] = {
        "u": (("z", "y", "x_u"), np.asarray(state_3d["u"], dtype=np.float32), {"units": "m s-1"}),
        "v": (("z", "y_v", "x"), np.asarray(state_3d["v"], dtype=np.float32), {"units": "m s-1"}),
        "w": (("z_w", "y", "x"), np.asarray(state_3d["w"], dtype=np.float32), {"units": "m s-1"}),
        "theta_p": (("z", "y", "x"), np.asarray(state_3d["theta_p"], dtype=np.float32), {"units": "K"}),
        "pi_p": (("z", "y", "x"), np.asarray(state_3d["pi_p"], dtype=np.float32), {"units": "1"}),
        "hs_m": (("y", "x"), np.asarray(hs_m, dtype=np.float64), {"units": "m"}),
    }
    for sp in NonHydrostaticGPUSolver.SPECIES:
        if sp in state_3d:
            data_vars[sp] = (("z", "y", "x"), np.asarray(state_3d[sp], dtype=np.float32), {"units": "kg kg-1"})
    if "tke" in state_3d:
        data_vars["tke"] = (("z", "y", "x"), np.asarray(state_3d["tke"], dtype=np.float32), {"units": "m2 s-2"})

    for k in NonHydrostaticGPUSolver.SURFACE_STATE_STRUCT_ORDER:
        if k not in surf_state:
            continue
        dt = np.float64 if k in NonHydrostaticGPUSolver.SURFACE_FP64_FIELDS else np.float32
        arr = np.asarray(surf_state[k], dtype=dt)
        if arr.ndim == 3:
            data_vars[k] = (("soil_layer", "y", "x"), arr)
        else:
            data_vars[k] = (("y", "x"), arr)

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": ("time", t_coord),
            "z": ("z", spec.vertical.zeta_s.astype(np.float64)),
            "z_w": ("z_w", spec.vertical.zeta_w.astype(np.float64)),
            "soil_layer": ("soil_layer", np.array([0, 1, 2, 3], dtype=np.int32)),
            "y": ("y", spec.y_m.astype(np.float64)),
            "y_v": ("y_v", np.arange(spec.ny + 1, dtype=np.int32)),
            "x": ("x", spec.x_m.astype(np.float64)),
            "x_u": ("x_u", np.arange(spec.nx + 1, dtype=np.int32)),
        },
        attrs={
            "kind": "tempest_v2_restart",
            "domain_name": spec.name,
            "valid_time_utc": v_dt.isoformat(),
            "sim_time_sec": f"{float(sim_time_sec):.6f}",
            "theta_sfc_ref": f"{float(theta_sfc_ref):.6f}",
            "p_sfc_ref": f"{float(p_sfc_ref):.6f}",
            "n_ref_hz": f"{float(n_ref_hz):.6f}",
        },
    )
    kz_chunk = max(1, min(int(spec.nz), (16 * 1024 * 1024) // max(1, spec.ny * (spec.nx + 1) * 4)))
    chunks_map: Dict[str, List[int]] = {}
    for vn, da in ds.data_vars.items():
        if da.ndim == 3 and da.shape[0] >= spec.nz:
            chunks_map[str(vn)] = [min(int(da.shape[0]), kz_chunk), int(da.shape[1]), int(da.shape[2])]
    wx1.write_dataset(
        out_p,
        ds,
        dataset_id=f"restart_{spec.name}",
        codec="zstd_shuffle",
        chunk_shapes=chunks_map,
        workers=4,
    )
    return out_p


def load_tempest_restart_wx1(
    path: Path | str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Any]]:
    """Load lossless WX-1 restart checkpoint returning `(state_3d, surf_state, metadata)`."""
    state_keys = ("u", "v", "w", "theta_p", "pi_p") + NonHydrostaticGPUSolver.SPECIES + ("tke",)
    surf_keys = set(NonHydrostaticGPUSolver.SURFACE_STATE_STRUCT_ORDER)
    state_3d: Dict[str, np.ndarray] = {}
    surf_state: Dict[str, np.ndarray] = {}
    meta: Dict[str, Any] = {}

    with wx1.open_dataset(path) as ds:
        meta = dict(ds.attrs)
        if "hs_m" in ds:
            meta["hs_m"] = np.asarray(ds["hs_m"].values, dtype=np.float64)
        for k in state_keys:
            if k in ds:
                state_3d[k] = np.asarray(ds[k].values, dtype=np.float32)
        for k in NonHydrostaticGPUSolver.SURFACE_STATE_STRUCT_ORDER:
            if k in surf_keys and k in ds:
                dt = np.float64 if k in NonHydrostaticGPUSolver.SURFACE_FP64_FIELDS else np.float32
                surf_state[k] = np.asarray(ds[k].values, dtype=dt)
    return state_3d, surf_state, meta
