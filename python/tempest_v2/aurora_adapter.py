"""Aurora 1.5 WX-1 Input Adapter for Tempest-NWP v2.

Reads Microsoft Aurora 1.5 forecast/analysis snapshots saved in WX-1 format
(from aurora15_colab_fixed_v4.ipynb) using tempest-wx1 v0.1.8 and maps:
- 3D isobaric fields (u, v, t, z, q on 1000..50 hPa)
- 2D surface fields (2t/t2m, 10u/u10, 10v/v10, msl, sp)
onto any DomainSpec (D2 LCC or D3 1.5 km LCC) + Klemp VerticalGrid (nz=65, 90, 137)
with hydrostatic Exner perturbations and surface-pressure hypsometric elevation adjustment.

NOTE (0% AI inside NWP): Aurora outputs act strictly as external IC/LBC background files,
never inside Tempest's solver or physics kernels. Soil (sot/vsw) is not output by Aurora
and should come from ECMWF IFS step-0 analysis or HWSD climatology.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import wx1

from tempest_v2.projection import (
    CP_D,
    GRAVITY_M_S2,
    KAPPA,
    P0_PA,
    R_D,
    DomainSpec,
    VerticalGrid,
)

logger = logging.getLogger("tempest_v2.aurora_adapter")


def _extract_pressure_stack(ds, prefix: str, levels_hpa: list[int]) -> Optional[np.ndarray]:
    """Assemble [n_levels, ny, nx] stack matching `prefix_{lev}` in ds."""
    slices = []
    for lev in levels_hpa:
        cand = next((k for k in (f"{prefix}_{lev}", f"{prefix}{lev}", f"{prefix}_{lev}hPa") if k in ds), None)
        if cand is None:
            return None
        arr = np.asarray(ds[cand].values, dtype=np.float64)
        while arr.ndim > 2:
            arr = arr[0]
        slices.append(arr)
    return np.stack(slices, axis=0)


def aurora_wx1_to_domain_state(
    wx1_path: str,
    target_spec: DomainSpec,
    target_hs_m: np.ndarray,
    *,
    theta_sfc_ref: float = 288.15,
    p_sfc_ref: float = 101_325.0,
    n_ref_hz: float = 0.01,
) -> Dict[str, np.ndarray]:
    """Convert an Aurora 1.5 `.wx1` snapshot to balanced Tempest-NWP v2 prognostic state on DomainSpec."""
    levels_hpa = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

    with wx1.open_dataset(wx1_path) as ds:
        lat_name = next((c for c in ("lat", "latitude") if c in ds.coords or c in ds.variables), None)
        lon_name = next((c for c in ("lon", "longitude") if c in ds.coords or c in ds.variables), None)
        if lat_name is None or lon_name is None:
            raise ValueError(f"Aurora WX-1 dataset {wx1_path} lacks lat/lon coordinates")

        src_lats = np.asarray(ds[lat_name].values, dtype=np.float64)
        src_lons = np.asarray(ds[lon_name].values, dtype=np.float64)

        def _get_2d(names: tuple[str, ...], default_val: float = 0.0) -> np.ndarray:
            for n in names:
                if n in ds:
                    a = np.asarray(ds[n].values, dtype=np.float64)
                    while a.ndim > 2:
                        a = a[0]
                    return a
            return np.full((len(src_lats), len(src_lons)), default_val, dtype=np.float64)

        t_pl = _extract_pressure_stack(ds, "t", levels_hpa)
        u_pl = _extract_pressure_stack(ds, "u", levels_hpa)
        v_pl = _extract_pressure_stack(ds, "v", levels_hpa)
        q_pl = _extract_pressure_stack(ds, "q", levels_hpa)
        z_pl = _extract_pressure_stack(ds, "z", levels_hpa)

        if t_pl is None or u_pl is None or v_pl is None:
            raise ValueError(f"Aurora WX-1 file {wx1_path} missing required upper-air stacks (t/u/v)")

        msl = _get_2d(("msl", "mslp"), 101_325.0)
        sp_in = _get_2d(("sp", "ps"), np.nan)
        t2m = _get_2d(("2t", "t2m"), 285.0)

    # Sort ascending latitude and normalize longitudes onto target range
    if src_lats[0] > src_lats[-1]:
        src_lats = src_lats[::-1]
        t_pl = t_pl[:, ::-1, :]
        u_pl = u_pl[:, ::-1, :]
        v_pl = v_pl[:, ::-1, :]
        if q_pl is not None:
            q_pl = q_pl[:, ::-1, :]
        if z_pl is not None:
            z_pl = z_pl[:, ::-1, :]
        msl = msl[::-1, :]
        sp_in = sp_in[::-1, :]
        t2m = t2m[::-1, :]

    if src_lons.max() > 180.0:
        tgt_lon_query = np.mod(target_spec.lon2d, 360.0)
    else:
        tgt_lon_query = ((target_spec.lon2d + 180.0) % 360.0) - 180.0

    pts = np.column_stack([target_spec.lat2d.ravel(), tgt_lon_query.ravel()])

    def _regrid_2d(src2d: np.ndarray) -> np.ndarray:
        itp = RegularGridInterpolator((src_lats, src_lons), src2d, method="linear", bounds_error=False, fill_value=None)
        return itp(pts).reshape(target_spec.ny, target_spec.nx)

    def _regrid_stack(stack: np.ndarray) -> np.ndarray:
        out = np.empty((stack.shape[0], target_spec.ny, target_spec.nx), dtype=np.float64)
        for k in range(stack.shape[0]):
            out[k] = _regrid_2d(stack[k])
        return out

    t_h = _regrid_stack(t_pl)
    u_geo_h = _regrid_stack(u_pl)
    v_geo_h = _regrid_stack(v_pl)
    q_h = np.clip(_regrid_stack(q_pl), 1e-7, 0.04) if q_pl is not None else np.full_like(t_h, 1e-4)

    msl_h = _regrid_2d(msl)
    t2m_h = _regrid_2d(t2m)
    hs = np.asarray(target_hs_m, dtype=np.float64)

    # Derive surface pressure adjusted hypsometrically to target Copernicus DEM topography
    ps_child = msl_h * np.exp(-(GRAVITY_M_S2 * hs) / (R_D * np.maximum(t2m_h, 200.0)))

    # Heights of pressure levels [m]: use geopotential z/g if available, else hypsometric
    if z_pl is not None:
        z_h = _regrid_stack(z_pl) / GRAVITY_M_S2
    else:
        p_pa = np.array(levels_hpa, dtype=np.float64)[:, None, None] * 100.0
        z_h = (R_D * 255.0 / GRAVITY_M_S2) * np.log(msl_h[None, :, :] / p_pa)

    # Rotate horizontal winds onto LCC grid
    if target_spec.is_projected and target_spec.projection is not None:
        u_grid_h, v_grid_h = target_spec.projection.geo_to_grid_wind(
            u_geo_h, v_geo_h, target_spec.lon2d[None, :, :]
        )
    else:
        u_grid_h, v_grid_h = u_geo_h, v_geo_h

    _, z_child_s, _ = target_spec.vertical.heights_3d(hs)
    ref = VerticalGrid.reference_state(z_child_s, theta_sfc_k=theta_sfc_ref, p_sfc_pa=p_sfc_ref, brunt_vaisala_hz=n_ref_hz)

    ny, nx, nz = target_spec.ny, target_spec.nx, target_spec.nz
    p_levels_pa = np.array(levels_hpa, dtype=np.float64) * 100.0
    exner_levels = (p_levels_pa / P0_PA) ** KAPPA
    theta_h = t_h / exner_levels[:, None, None]

    theta_3d = np.empty((nz, ny, nx), dtype=np.float64)
    u_3d = np.empty((nz, ny, nx), dtype=np.float64)
    v_3d = np.empty((nz, ny, nx), dtype=np.float64)
    qv_3d = np.empty((nz, ny, nx), dtype=np.float64)
    p_3d = np.empty((nz, ny, nx), dtype=np.float64)

    # Vectorized columnwise vertical interpolation in height z (ascending order ensured)
    if z_h[-1, 0, 0] < z_h[0, 0, 0]:
        z_h = z_h[::-1]
        theta_h = theta_h[::-1]
        u_grid_h = u_grid_h[::-1]
        v_grid_h = v_grid_h[::-1]
        q_h = q_h[::-1]
        p_levels_pa = p_levels_pa[::-1]

    ln_p_levels = np.log(p_levels_pa)
    z_col_flat = z_h.reshape(len(levels_hpa), ny * nx)
    zc_flat = z_child_s.reshape(nz, ny * nx)
    th_flat = theta_h.reshape(len(levels_hpa), ny * nx)
    u_flat = u_grid_h.reshape(len(levels_hpa), ny * nx)
    v_flat = v_grid_h.reshape(len(levels_hpa), ny * nx)
    q_flat = q_h.reshape(len(levels_hpa), ny * nx)

    th_out = np.empty((nz, ny * nx), dtype=np.float64)
    u_out = np.empty((nz, ny * nx), dtype=np.float64)
    v_out = np.empty((nz, ny * nx), dtype=np.float64)
    q_out = np.empty((nz, ny * nx), dtype=np.float64)
    p_out = np.empty((nz, ny * nx), dtype=np.float64)

    for col in range(ny * nx):
        zs_col = z_col_flat[:, col]
        zt_col = zc_flat[:, col]
        th_out[:, col] = np.interp(zt_col, zs_col, th_flat[:, col])
        u_out[:, col] = np.interp(zt_col, zs_col, u_flat[:, col])
        v_out[:, col] = np.interp(zt_col, zs_col, v_flat[:, col])
        q_out[:, col] = np.interp(zt_col, zs_col, q_flat[:, col])
        p_out[:, col] = np.exp(np.interp(zt_col, zs_col, ln_p_levels))
        # Above 50 hPa top of Aurora, relax smoothly via hydrostatic scale height
        above = zt_col > zs_col[-1]
        if np.any(above):
            t_top = th_flat[-1, col] * ((p_levels_pa[-1] / P0_PA) ** KAPPA)
            p_out[above, col] = p_levels_pa[-1] * np.exp(-GRAVITY_M_S2 * (zt_col[above] - zs_col[-1]) / (R_D * max(180.0, t_top)))
            ex_above = np.power(np.maximum(p_out[above, col], 1.0) / P0_PA, KAPPA)
            th_out[above, col] = t_top / ex_above

    theta_3d = th_out.reshape(nz, ny, nx)
    u_3d = u_out.reshape(nz, ny, nx)
    v_3d = v_out.reshape(nz, ny, nx)
    qv_3d = np.clip(q_out.reshape(nz, ny, nx), 1e-7, 0.04)
    p_3d = p_out.reshape(nz, ny, nx)

    pi_3d = np.power(np.maximum(p_3d, 10.0) / P0_PA, KAPPA)
    theta_p = (theta_3d - ref["theta_bar"]).astype(np.float32)
    pi_p = (pi_3d - ref["pi_bar"]).astype(np.float32)

    return {
        "u": u_3d.astype(np.float32),
        "v": v_3d.astype(np.float32),
        "w": np.zeros((nz + 1, ny, nx), dtype=np.float32),
        "theta_p": theta_p,
        "pi_p": pi_p,
        "qv": qv_3d.astype(np.float32),
        "ps": ps_child.astype(np.float32),
        "t2m": t2m_h.astype(np.float32),
    }
