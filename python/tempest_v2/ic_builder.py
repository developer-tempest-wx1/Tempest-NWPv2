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
from tempest_v2.projection import (
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

DEFAULT_COPERNICUS_ROOT = Path("/mnt/d/AI/weer-ai/Tempest/COPERNICUS_DEM")
DEFAULT_GEBCO_ROOT = Path("/mnt/d/AI/weer-ai/Tempest/bathymetrie")


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
    force: bool = False,
    allow_synthetic_fallback: bool = True,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Build or load cached Copernicus DEM + GEBCO 2026 + HWSD v2.2 static fields on `spec`."""
    sdir = Path(static_dir)
    sdir.mkdir(parents=True, exist_ok=True)
    ny, nx = spec.ny, spec.nx

    resolved_dem = Path(dem_root) if dem_root is not None else (DEFAULT_COPERNICUS_ROOT if DEFAULT_COPERNICUS_ROOT.exists() else None)
    cop_cache = sdir / f"static_copernicus_{spec.name}.wx1"
    used_fast_global_gebco = False
    if not spec.is_projected and spec.dx_m >= 15_000.0 and not cop_cache.is_file() and gebco_root is not None and Path(gebco_root).exists():
        nc_candidates = sorted(Path(gebco_root).rglob("*.nc"))
        if nc_candidates:
            try:
                bathy_fast = sdir / f"bathymetry_{spec.name}.wx1"
                if not bathy_fast.is_file() or force:
                    prepare_bathymetry_on_2d_grid(
                        nc_candidates[0], spec.name, spec.lat2d, spec.lon2d, vertical_datum="msl", output_path=bathy_fast
                    )
                with xr.open_dataset(nc_candidates[0]) as gds:
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
        nc_candidates = sorted(Path(gebco_root).rglob("*.nc"))
        if nc_candidates:
            prepare_bathymetry_on_2d_grid(
                nc_candidates[0], spec.name, spec.lat2d, spec.lon2d, vertical_datum="msl", output_path=bathy_cache
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

    # Roughness length z0m, albedo, greenness fraction derived physically from elevation & terrain sigma
    z0m_land = np.clip(0.05 + 0.0015 * oro_std + 0.08 * np.clip(hs_m / 600.0, 0.0, 1.0), 0.03, 1.2)
    z0m_base = np.where(is_land, z0m_land, 1.0e-4).astype(np.float32)
    albedo_base = np.where(is_land, np.where(hs_m > 1800.0, 0.35, 0.19), 0.06).astype(np.float32)
    veg_frac = np.where(is_land, np.clip(0.65 - 0.35 * np.clip((hs_m - 600.0) / 1400.0, 0.0, 1.0), 0.15, 0.75), 0.0).astype(np.float32)

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
    data = np.asarray(src, dtype=np.float64)
    lats = np.asarray(src_lats, dtype=np.float64)
    lons = np.asarray(src_lons, dtype=np.float64)

    if lats[0] > lats[-1]:
        lats = lats[::-1]
        data = data[..., ::-1, :]

    if lons.max() > 180.0 and lons.min() >= 0.0:
        # Unwrap 0..360 -> -180..180 sorted cleanly if global/Benelux spans 0E
        lons_w = ((lons + 180.0) % 360.0) - 180.0
        order = np.argsort(lons_w)
        lons = lons_w[order]
        data = data[..., :, order]

    if len(lons) >= 2 and (lons[-1] - lons[0]) > 350.0:
        lons = np.append(lons, lons[0] + 360.0)
        data = np.concatenate([data, data[..., :, :1]], axis=-1)

    tgt_lon = ((target_spec.lon2d + 180.0) % 360.0) - 180.0
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


def _read_cfgrib_var_map(grib_path: Path) -> Dict[str, Tuple[xr.DataArray, np.ndarray, np.ndarray]]:
    import warnings
    import cfgrib

    key = (str(grib_path.resolve()), grib_path.stat().st_mtime if grib_path.exists() else 0.0)
    if key in _CFGRIB_CACHE:
        return _CFGRIB_CACHE[key]
    if len(_CFGRIB_CACHE) >= 4:
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
    _CFGRIB_CACHE[key] = found
    return found


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

    exner_lvl = (p_levels_pa / P0_PA) ** KAPPA
    theta_h = t_h / exner_lvl[:, None, None]
    nlev = len(p_levels_pa)
    ln_p = np.log(p_levels_pa)

    theta_p_out = np.empty((nz, ny, nx), dtype=np.float32)
    pi_p_out = np.empty((nz, ny, nx), dtype=np.float32)
    qv_out = np.empty((nz, ny, nx), dtype=np.float32)
    u_c_out = np.empty((nz, ny, nx), dtype=np.float32)
    v_c_out = np.empty((nz, ny, nx), dtype=np.float32)

    zeta_s = spec.vertical.zeta_s[:, None, None].astype(np.float64)
    b_s = spec.vertical.B_s[:, None, None].astype(np.float64)
    hs_2d = np.asarray(hs_m, dtype=np.float64)
    n2 = float(n_ref_hz) * float(n_ref_hz)
    g2_cp = (GRAVITY_M_S2 * GRAVITY_M_S2) / (1004.64 * float(theta_sfc_ref) * max(n2, 1e-12))
    pi_sfc = (float(p_sfc_ref) / P0_PA) ** KAPPA

    chunk_ny = 64
    for j0 in range(0, ny, chunk_ny):
        j1 = min(ny, j0 + chunk_ny)
        ny_c = j1 - j0
        n_col = ny_c * nx
        z_chunk = zeta_s + b_s * hs_2d[None, j0:j1, :]
        exp_fac = np.exp((n2 / GRAVITY_M_S2) * z_chunk)
        theta_bar_c = float(theta_sfc_ref) * exp_fac
        pi_bar_c = np.maximum(1e-4, pi_sfc - g2_cp * (1.0 - 1.0 / exp_fac))

        zs_b = z_h[:, j0:j1, :].reshape(nlev, n_col)
        zt_b = z_chunk.reshape(nz, n_col)
        th_b = theta_h[:, j0:j1, :].reshape(nlev, n_col)
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

        th_i = (1.0 - wt) * th_b[k_lo, col_idx] + wt * th_b[k_hi, col_idx]
        u_i = (1.0 - wt) * u_b[k_lo, col_idx] + wt * u_b[k_hi, col_idx]
        v_i = (1.0 - wt) * v_b[k_lo, col_idx] + wt * v_b[k_hi, col_idx]
        q_i = (1.0 - wt) * q_b[k_lo, col_idx] + wt * q_b[k_hi, col_idx]
        p_i = np.exp((1.0 - wt) * ln_p[k_lo] + wt * ln_p[k_hi])

        above = zt_b > zs_b[-1:, :]
        if np.any(above):
            t_top = np.maximum(180.0, th_b[-1:, :] * ((p_levels_pa[-1] / P0_PA) ** KAPPA))
            p_abv = p_levels_pa[-1] * np.exp(-GRAVITY_M_S2 * (zt_b - zs_b[-1:, :]) / (R_D * t_top))
            th_abv = t_top / ((np.maximum(p_abv, 1.0) / P0_PA) ** KAPPA)
            p_i = np.where(above, p_abv, p_i)
            th_i = np.where(above, th_abv, th_i)

        pi_i = np.power(np.maximum(p_i, 10.0) / P0_PA, KAPPA)
        theta_p_out[:, j0:j1, :] = (th_i.reshape(nz, ny_c, nx) - theta_bar_c).astype(np.float32)
        pi_p_out[:, j0:j1, :] = (pi_i.reshape(nz, ny_c, nx) - pi_bar_c).astype(np.float32)
        qv_out[:, j0:j1, :] = np.clip(q_i.reshape(nz, ny_c, nx), 1e-7, 0.04).astype(np.float32)
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
    t2m_h = _get_sfc(("t2m", "2t"), 285.0)
    skt_h = _get_sfc(("skt", "t"), float(np.mean(t2m_h)))
    sd_h = _get_sfc(("sd", "snow"), 0.0)
    # ECMWF IFS GRIB2 snow depth is in meters of water equivalent -> kg/m^2 = mm
    if "sd" in sfc_vars and "m" in str(sfc_vars["sd"][0].attrs.get("units", "")).lower():
        sd_h = sd_h * 1000.0
    sd_h = np.clip(sd_h, 0.0, 2000.0)

    hs = np.asarray(hs_m, dtype=np.float64)
    ps_child = msl_h * np.exp(-(GRAVITY_M_S2 * hs) / (R_D * np.maximum(t2m_h, 200.0)))

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
    )

    surf = _empty_surface_state_for_spec(spec)
    surf["t_skin"] = np.clip(skt_h, 210.0, 330.0).astype(np.float32)
    surf["t2m"] = np.clip(t2m_h, 210.0, 330.0).astype(np.float32)
    surf["snow_swe"] = sd_h.astype(np.float32)
    surf["sst_bulk"] = np.clip(skt_h, 271.15, 310.0).astype(np.float32)
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
            surf["soil_m"] = np.clip(_regrid_latlon_field(arr[:4], la, lo, spec, fill_nan_value=0.28), 0.03, 0.50).astype(np.float32)
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

    for k in NonHydrostaticGPUSolver.SURFACE_STATE_STRUCT_ORDER:
        if k not in surf_state:
            continue
        arr = np.asarray(surf_state[k], dtype=np.float32)
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
    state_keys = ("u", "v", "w", "theta_p", "pi_p") + NonHydrostaticGPUSolver.SPECIES
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
                surf_state[k] = np.asarray(ds[k].values, dtype=np.float32)
    return state_3d, surf_state, meta
