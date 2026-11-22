"""Lateral Boundary Condition (LBC) Manager for Tempest-NWP v2 Nested Grids.

Strictly 100% Classical Atmospheric Dynamics (0% AI / 0% Machine Learning).
Supports full non-hydrostatic prognostic LBC fields (u, v, w, theta_p, pi_p, ps, qv,
plus optional condensed hydrometeors qc..qg) with:
- Horizontal regridding onto LCC 2D child grids
- Exact earth-relative <-> LCC-grid wind vector rotation
- Vertical interpolation onto Klemp (2011) terrain-following height levels z_s
- Smooth temporal linear interpolation and WX-1 v0.1.8 lossless storage.
"""

from __future__ import annotations

import datetime
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import xarray as xr

import wx1

from tempest_v2.projection import DomainSpec

logger = logging.getLogger("tempest_v2.lbc_manager")

LBC_3D_VARS = ("u", "v", "theta_p", "pi_p", "qv")
LBC_HYDROMETEORS = ("qc", "qr", "qi", "qs", "qg")
_LBC_PLAN_CACHE: Dict[Tuple[Any, ...], Dict[str, Any]] = {}


def _compute_regrid_stencil(
    parent_spec: DomainSpec,
    child_spec: DomainSpec,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Precompute 2D bilinear index/weight stencil from parent grid to child grid once."""
    ny_p, nx_p = int(parent_spec.ny), int(parent_spec.nx)
    if parent_spec.is_projected and parent_spec.projection is not None:
        tx, ty = parent_spec.projection.latlon_to_xy(child_spec.lat2d, child_spec.lon2d)
        py_axis = np.asarray(parent_spec.y_m, dtype=np.float64)
        px_axis = np.asarray(parent_spec.x_m, dtype=np.float64)
        periodic_lon = False
    else:
        py_axis = np.asarray(parent_spec.y_m, dtype=np.float64)
        px_axis = np.asarray(parent_spec.x_m, dtype=np.float64)
        ty = np.asarray(child_spec.lat2d, dtype=np.float64)
        lon_tgt = np.asarray(child_spec.lon2d, dtype=np.float64)
        if parent_spec.is_periodic_lon and px_axis[0] >= -1e-6 and np.nanmax(px_axis) > 180.0:
            tx = np.mod(lon_tgt, 360.0)
        else:
            tx = ((lon_tgt + 180.0) % 360.0) - 180.0
        periodic_lon = bool(parent_spec.is_periodic_lon and len(px_axis) >= 2)

    flip_y = bool(len(py_axis) >= 2 and py_axis[0] > py_axis[-1])
    y_asc = py_axis[::-1] if flip_y else py_axis
    if len(y_asc) >= 2:
        j0_a = np.clip(np.searchsorted(y_asc, ty, side="right") - 1, 0, ny_p - 2)
        j1_a = j0_a + 1
        wy = np.clip((ty - y_asc[j0_a]) / np.maximum(y_asc[j1_a] - y_asc[j0_a], 1e-12), 0.0, 1.0).astype(np.float32)
        if flip_y:
            j0 = (ny_p - 1) - j0_a
            j1 = (ny_p - 1) - j1_a
        else:
            j0, j1 = j0_a, j1_a
    else:
        j0 = np.zeros_like(ty, dtype=np.int64)
        j1 = j0
        wy = np.zeros_like(ty, dtype=np.float32)

    if len(px_axis) >= 2:
        if periodic_lon:
            px_ext = np.append(px_axis, px_axis[0] + 360.0)
            tx_adj = np.where(tx < px_axis[0], tx + 360.0, tx)
            i0 = np.clip(np.searchsorted(px_ext, tx_adj, side="right") - 1, 0, nx_p - 1)
            i1_ext = i0 + 1
            wx = np.clip((tx_adj - px_ext[i0]) / np.maximum(px_ext[i1_ext] - px_ext[i0], 1e-12), 0.0, 1.0).astype(np.float32)
            i1 = np.where(i1_ext >= nx_p, 0, i1_ext)
        else:
            i0 = np.clip(np.searchsorted(px_axis, tx, side="right") - 1, 0, nx_p - 2)
            i1 = i0 + 1
            wx = np.clip((tx - px_axis[i0]) / np.maximum(px_axis[i1] - px_axis[i0], 1e-12), 0.0, 1.0).astype(np.float32)
    else:
        i0 = np.zeros_like(tx, dtype=np.int64)
        i1 = i0
        wx = np.zeros_like(tx, dtype=np.float32)

    return j0.astype(np.int64), j1.astype(np.int64), i0.astype(np.int64), i1.astype(np.int64), wy, wx


def _horizontal_regrid_3d_to_child(
    field_3d: np.ndarray,
    parent_spec: DomainSpec,
    child_spec: DomainSpec,
    stencil: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None,
) -> np.ndarray:
    """Vectorized bilinear regridding of parent `[nk, ny_p, nx_p]` onto child `[nk, ny_c, nx_c]` in float32."""
    nk = field_3d.shape[0]
    j0, j1, i0, i1, wy, wx = stencil if stencil is not None else _compute_regrid_stencil(parent_spec, child_spec)
    out = np.empty((nk, child_spec.ny, child_spec.nx), dtype=np.float32)
    om_wy = (np.float32(1.0) - wy)[None, :, :]
    wy_b = wy[None, :, :]
    om_wx = (np.float32(1.0) - wx)[None, :, :]
    wx_b = wx[None, :, :]

    for k0 in range(0, nk, 32):
        k1 = min(nk, k0 + 32)
        sl = np.asarray(field_3d[k0:k1], dtype=np.float32)
        out[k0:k1] = (
            om_wy * (om_wx * sl[:, j0, i0] + wx_b * sl[:, j0, i1])
            + wy_b * (om_wx * sl[:, j1, i0] + wx_b * sl[:, j1, i1])
        )
    return out


def _compute_vertical_stencil_1d(
    src_z_2d: np.ndarray,
    dst_z_2d: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute vectorized binary-search linear vertical interpolation indices/weights for `[nz, n_col]` columns."""
    zs = np.asarray(src_z_2d, dtype=np.float32)
    zd = np.asarray(dst_z_2d, dtype=np.float32)
    nz_src, n_col = zs.shape
    if nz_src <= 1:
        z_i = np.zeros_like(zd, dtype=np.int16)
        return z_i, z_i, np.zeros_like(zd, dtype=np.float32)

    c_idx = np.arange(n_col, dtype=np.int32)[None, :]
    lo = np.zeros_like(zd, dtype=np.int32)
    hi = np.full_like(zd, nz_src, dtype=np.int32)
    n_bits = int(math.ceil(math.log2(max(2, nz_src + 1))))
    for _ in range(n_bits):
        mid = np.minimum((lo + hi) >> 1, nz_src - 1)
        cond = zs[mid, c_idx] < zd
        lo = np.where(cond, mid + 1, lo)
        hi = np.where(cond, hi, mid)

    k_hi = np.clip(lo, 1, max(1, nz_src - 1)).astype(np.int16)
    k_lo = (k_hi - 1).astype(np.int16)
    z_lo = zs[k_lo, c_idx]
    z_hi = zs[k_hi, c_idx]
    w = np.clip((zd - z_lo) / np.maximum(z_hi - z_lo, np.float32(1e-5)), 0.0, 1.0).astype(np.float32)
    return k_lo, k_hi, w


def _apply_vertical_stencil_1d(
    src_vals_2d: np.ndarray,
    k_lo: np.ndarray,
    k_hi: np.ndarray,
    w: np.ndarray,
    clamp_non_negative: bool = False,
) -> np.ndarray:
    """Apply precomputed vertical interpolation stencil `(k_lo, k_hi, w)` to `[nz_src, n_col]` float32 values."""
    sf = np.asarray(src_vals_2d, dtype=np.float32)
    c_idx = np.arange(sf.shape[1], dtype=np.int32)[None, :]
    out = ((np.float32(1.0) - w) * sf[k_lo, c_idx] + w * sf[k_hi, c_idx]).astype(np.float32, copy=False)
    if clamp_non_negative:
        np.maximum(out, np.float32(0.0), out=out)
    return out


def _vertical_interp_columnwise(
    src_vals: np.ndarray,
    src_z: np.ndarray,
    dst_z: np.ndarray,
    clamp_non_negative: bool = False,
) -> np.ndarray:
    """Linearly interpolate 3D `[nz_src, ny, nx]` in height src_z onto dst_z `[nz_dst, ny, nx]` via binary search."""
    nz_src, ny, nx = src_vals.shape
    nz_dst = dst_z.shape[0]
    out = np.empty((nz_dst, ny, nx), dtype=np.float32)

    chunk_ny = 64
    for j0 in range(0, ny, chunk_ny):
        j1 = min(ny, j0 + chunk_ny)
        n_col = (j1 - j0) * nx
        sf = np.asarray(src_vals[:, j0:j1, :], dtype=np.float32).reshape(nz_src, n_col)
        zs = np.asarray(src_z[:, j0:j1, :], dtype=np.float32).reshape(nz_src, n_col)
        zd = np.asarray(dst_z[:, j0:j1, :], dtype=np.float32).reshape(nz_dst, n_col)
        k_lo, k_hi, w = _compute_vertical_stencil_1d(zs, zd)
        out[:, j0:j1, :] = _apply_vertical_stencil_1d(sf, k_lo, k_hi, w, clamp_non_negative=clamp_non_negative).reshape(
            nz_dst, j1 - j0, nx
        )
    return out


def _get_lbc_plan(
    parent_spec: DomainSpec,
    parent_hs_m: np.ndarray,
    child_spec: DomainSpec,
    child_hs_m: np.ndarray,
    rim_width: Optional[int],
) -> Dict[str, Any]:
    """Build and cache horizontal+vertical LBC regridding plan (rim-only on large domains)."""
    use_rim = (
        rim_width is not None
        and int(rim_width) > 0
        and child_spec.ny > 2 * int(rim_width) + 8
        and child_spec.nx > 2 * int(rim_width) + 8
    )
    rw = int(rim_width) if use_rim else 0
    key = (
        parent_spec.name,
        int(parent_spec.nz),
        int(parent_spec.ny),
        int(parent_spec.nx),
        child_spec.name,
        int(child_spec.nz),
        int(child_spec.ny),
        int(child_spec.nx),
        rw,
        float(np.asarray(parent_hs_m, dtype=np.float64).sum()),
        float(np.asarray(child_hs_m, dtype=np.float64).sum()),
    )
    cached = _LBC_PLAN_CACHE.get(key)
    if cached is not None:
        return cached

    stencil = _compute_regrid_stencil(parent_spec, child_spec)
    j0, j1, i0, i1, wy, wx = stencil

    cos_p: Optional[np.ndarray] = None
    sin_p: Optional[np.ndarray] = None
    if parent_spec.is_projected and parent_spec.projection is not None:
        gp = parent_spec.projection.wind_rotation_angle_rad(parent_spec.lon2d)
        cos_p = np.cos(gp).astype(np.float32)
        sin_p = np.sin(gp).astype(np.float32)

    p_hs = np.asarray(parent_hs_m, dtype=np.float32)
    c_hs = np.asarray(child_hs_m, dtype=np.float32)
    pv = parent_spec.vertical
    cv = child_spec.vertical

    if use_rim:
        mask = np.zeros((child_spec.ny, child_spec.nx), dtype=bool)
        mask[:rw, :] = True
        mask[-rw:, :] = True
        mask[:, :rw] = True
        mask[:, -rw:] = True
        rj, ri = np.nonzero(mask)
        rim_flat = (rj * child_spec.nx + ri).astype(np.int64)
        j0_r = j0[rj, ri]
        j1_r = j1[rj, ri]
        i0_r = i0[rj, ri]
        i1_r = i1[rj, ri]
        wy_r = wy[rj, ri].astype(np.float32)
        wx_r = wx[rj, ri].astype(np.float32)
        w00 = (np.float32(1.0) - wy_r) * (np.float32(1.0) - wx_r)
        w01 = (np.float32(1.0) - wy_r) * wx_r
        w10 = wy_r * (np.float32(1.0) - wx_r)
        w11 = wy_r * wx_r

        hs_p_rim = w00 * p_hs[j0_r, i0_r] + w01 * p_hs[j0_r, i1_r] + w10 * p_hs[j1_r, i0_r] + w11 * p_hs[j1_r, i1_r]
        hs_c_rim = c_hs[rj, ri]
        z_p = (pv.zeta_s[:, None].astype(np.float32) + pv.B_s[:, None].astype(np.float32) * hs_p_rim[None, :])
        z_cs = (cv.zeta_s[:, None].astype(np.float32) + cv.B_s[:, None].astype(np.float32) * hs_c_rim[None, :])
        z_cw = (cv.zeta_w[:, None].astype(np.float32) + cv.B_w[:, None].astype(np.float32) * hs_c_rim[None, :])
        v_sten_s = _compute_vertical_stencil_1d(z_p, z_cs)
        v_sten_w = _compute_vertical_stencil_1d(z_p, z_cw)

        cos_c: Optional[np.ndarray] = None
        sin_c: Optional[np.ndarray] = None
        if child_spec.is_projected and child_spec.projection is not None:
            gc = child_spec.projection.wind_rotation_angle_rad(child_spec.lon2d[rj, ri])
            cos_c = np.cos(gc).astype(np.float32)
            sin_c = np.sin(gc).astype(np.float32)

        plan: Dict[str, Any] = {
            "use_rim": True,
            "rim_width": rw,
            "rim_flat": rim_flat,
            "rj": rj,
            "ri": ri,
            "j0": j0_r,
            "j1": j1_r,
            "i0": i0_r,
            "i1": i1_r,
            "w00": w00,
            "w01": w01,
            "w10": w10,
            "w11": w11,
            "v_sten_s": v_sten_s,
            "v_sten_w": v_sten_w,
            "cos_p": cos_p,
            "sin_p": sin_p,
            "cos_c": cos_c,
            "sin_c": sin_c,
        }
    else:
        n_col = child_spec.ny * child_spec.nx
        hs_p_full = _horizontal_regrid_3d_to_child(p_hs[None, :, :], parent_spec, child_spec, stencil=stencil)[0]
        _, z_p_3d, _ = pv.heights_3d(hs_p_full)
        z_cw_3d, z_cs_3d, _ = cv.heights_3d(c_hs)
        z_p_2d = z_p_3d.astype(np.float32).reshape(parent_spec.nz, n_col)
        v_sten_s = _compute_vertical_stencil_1d(z_p_2d, z_cs_3d.astype(np.float32).reshape(child_spec.nz, n_col))
        v_sten_w = _compute_vertical_stencil_1d(z_p_2d, z_cw_3d.astype(np.float32).reshape(child_spec.nz + 1, n_col))

        cos_c_full: Optional[np.ndarray] = None
        sin_c_full: Optional[np.ndarray] = None
        if child_spec.is_projected and child_spec.projection is not None:
            gc = child_spec.projection.wind_rotation_angle_rad(child_spec.lon2d)
            cos_c_full = np.cos(gc).astype(np.float32)
            sin_c_full = np.sin(gc).astype(np.float32)

        plan = {
            "use_rim": False,
            "rim_width": 0,
            "stencil": stencil,
            "v_sten_s": v_sten_s,
            "v_sten_w": v_sten_w,
            "cos_p": cos_p,
            "sin_p": sin_p,
            "cos_c": cos_c_full,
            "sin_c": sin_c_full,
        }

    if len(_LBC_PLAN_CACHE) >= 4:
        del _LBC_PLAN_CACHE[next(iter(_LBC_PLAN_CACHE))]
    _LBC_PLAN_CACHE[key] = plan
    return plan


def export_child_lbc_step(
    output_dir: Path | str,
    step_index: int,
    sim_time_sec: float,
    valid_time_utc: datetime.datetime,
    parent_spec: DomainSpec,
    parent_state: Dict[str, np.ndarray],
    parent_hs_m: np.ndarray,
    child_spec: DomainSpec,
    child_hs_m: np.ndarray,
    *,
    include_hydrometeors: bool = False,
    rim_width: Optional[int] = 16,
) -> Path:
    """Regrid parent 3D state onto child LCC + Klemp terrain-following coordinates and store as WX-1 v0.1.8."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"lbc_{child_spec.name}_step_{step_index:04d}.wx1"

    plan = _get_lbc_plan(parent_spec, parent_hs_m, child_spec, child_hs_m, rim_width)
    use_rim = bool(plan["use_rim"])

    u_src = np.asarray(parent_state["u"], dtype=np.float32)
    v_src = np.asarray(parent_state["v"], dtype=np.float32)
    if u_src.shape[2] == parent_spec.nx + 1:
        u_src = 0.5 * (u_src[:, :, :-1] + u_src[:, :, 1:])
    if v_src.shape[1] == parent_spec.ny + 1:
        v_src = 0.5 * (v_src[:, :-1, :] + v_src[:, 1:, :])

    if plan["cos_p"] is not None:
        cp, sp = plan["cos_p"][None, :, :], plan["sin_p"][None, :, :]
        u_geo = u_src * cp + v_src * sp
        v_geo = -u_src * sp + v_src * cp
    else:
        u_geo, v_geo = u_src, v_src

    w_src = np.asarray(parent_state["w"], dtype=np.float32)
    if w_src.shape[0] == parent_spec.nz + 1:
        w_src = 0.5 * (w_src[:-1] + w_src[1:])

    ps_p = np.asarray(parent_state.get("ps", np.full((parent_spec.ny, parent_spec.nx), 101_325.0, dtype=np.float32)), dtype=np.float32)
    v_sten_s = plan["v_sten_s"]
    v_sten_w = plan["v_sten_w"]
    nz_c, ny_c, nx_c = int(child_spec.nz), int(child_spec.ny), int(child_spec.nx)

    if use_rim:
        j0_r, j1_r, i0_r, i1_r = plan["j0"], plan["j1"], plan["i0"], plan["i1"]
        w00, w01, w10, w11 = plan["w00"][None, :], plan["w01"][None, :], plan["w10"][None, :], plan["w11"][None, :]
        rim_flat = plan["rim_flat"]
        n_rim = len(rim_flat)

        def _gather_rim_horiz(arr3: np.ndarray) -> np.ndarray:
            a = np.asarray(arr3, dtype=np.float32)
            nk = a.shape[0]
            h_rim = np.empty((nk, n_rim), dtype=np.float32)
            for k0 in range(0, nk, 32):
                k1 = min(nk, k0 + 32)
                sl = a[k0:k1]
                h_rim[k0:k1] = (
                    w00 * sl[:, j0_r, i0_r]
                    + w01 * sl[:, j0_r, i1_r]
                    + w10 * sl[:, j1_r, i0_r]
                    + w11 * sl[:, j1_r, i1_r]
                )
            return h_rim

        def _regrid_rim_packed(arr3: np.ndarray, v_sten: Tuple[np.ndarray, np.ndarray, np.ndarray], clamp_nn: bool = False) -> np.ndarray:
            return _apply_vertical_stencil_1d(_gather_rim_horiz(arr3), *v_sten, clamp_non_negative=clamp_nn)

        u_geo_r = _gather_rim_horiz(u_geo)
        v_geo_r = _gather_rim_horiz(v_geo)
        if plan["cos_c"] is not None:
            cc, sc = plan["cos_c"][None, :], plan["sin_c"][None, :]
            u_ch_r = u_geo_r * cc - v_geo_r * sc
            v_ch_r = u_geo_r * sc + v_geo_r * cc
        else:
            u_ch_r, v_ch_r = u_geo_r, v_geo_r

        u_rim = _apply_vertical_stencil_1d(u_ch_r, *v_sten_s)
        v_rim = _apply_vertical_stencil_1d(v_ch_r, *v_sten_s)
        w_rim = _regrid_rim_packed(w_src, v_sten_w)
        w_rim[-1] = 0.0
        theta_rim = _regrid_rim_packed(parent_state["theta_p"], v_sten_s)
        pi_rim = _regrid_rim_packed(parent_state["pi_p"], v_sten_s)
        qv_rim = _regrid_rim_packed(parent_state["qv"], v_sten_s, clamp_nn=True)

        ps_rim = (
            w00[0] * ps_p[j0_r, i0_r]
            + w01[0] * ps_p[j0_r, i1_r]
            + w10[0] * ps_p[j1_r, i0_r]
            + w11[0] * ps_p[j1_r, i1_r]
        ).astype(np.float32, copy=False)

        v_dt = valid_time_utc.astimezone(datetime.timezone.utc) if valid_time_utc.tzinfo else valid_time_utc.replace(tzinfo=datetime.timezone.utc)
        t_coord = np.array([np.datetime64(int(v_dt.timestamp()), "s")], dtype="datetime64[s]")

        data_vars = {
            "u": (("z", "rim"), u_rim, {"units": "m s-1"}),
            "v": (("z", "rim"), v_rim, {"units": "m s-1"}),
            "w": (("z_w", "rim"), w_rim, {"units": "m s-1"}),
            "theta_p": (("z", "rim"), theta_rim, {"units": "K"}),
            "pi_p": (("z", "rim"), pi_rim, {"units": "1"}),
            "qv": (("z", "rim"), qv_rim, {"units": "kg kg-1"}),
            "ps": (("rim",), ps_rim, {"units": "Pa"}),
            "rim_j": (("rim",), plan["rj"].astype(np.int32), {"units": "1"}),
            "rim_i": (("rim",), plan["ri"].astype(np.int32), {"units": "1"}),
        }
        if include_hydrometeors:
            for qname in LBC_HYDROMETEORS:
                if qname in parent_state:
                    data_vars[qname] = (
                        ("z", "rim"),
                        _regrid_rim_packed(parent_state[qname], v_sten_s, clamp_nn=True),
                        {"units": "kg kg-1"},
                    )

        ds = xr.Dataset(
            data_vars=data_vars,
            coords={
                "time": ("time", t_coord),
                "z": ("z", child_spec.vertical.zeta_s.astype(np.float32), {"units": "m"}),
                "z_w": ("z_w", child_spec.vertical.zeta_w.astype(np.float32), {"units": "m"}),
                "rim": ("rim", np.arange(n_rim, dtype=np.int32)),
            },
            attrs={
                "parent_domain": parent_spec.name,
                "child_domain": child_spec.name,
                "step_index": str(int(step_index)),
                "sim_time_sec": f"{sim_time_sec:.3f}",
                "valid_time_utc": v_dt.isoformat(),
                "lbc_storage": "rim_packed",
                "lbc_rim_width": str(int(rim_width)),
                "child_ny": str(ny_c),
                "child_nx": str(nx_c),
            },
        )
        chunks_map = {
            str(vn): [min(int(da.shape[0]), 64), min(int(da.shape[1]), 65536)]
            for vn, da in ds.data_vars.items()
            if da.ndim == 2
        }
    else:
        stencil = plan["stencil"]
        n_col = ny_c * nx_c

        def _regrid_full_3d(arr3: np.ndarray, v_sten: Tuple[np.ndarray, np.ndarray, np.ndarray], nz_out: int, clamp_nn: bool = False) -> np.ndarray:
            h_f = _horizontal_regrid_3d_to_child(np.asarray(arr3, dtype=np.float32), parent_spec, child_spec, stencil=stencil)
            return _apply_vertical_stencil_1d(h_f.reshape(h_f.shape[0], n_col), *v_sten, clamp_non_negative=clamp_nn).reshape(
                nz_out, ny_c, nx_c
            )

        u_geo_h = _horizontal_regrid_3d_to_child(u_geo, parent_spec, child_spec, stencil=stencil)
        v_geo_h = _horizontal_regrid_3d_to_child(v_geo, parent_spec, child_spec, stencil=stencil)
        if plan["cos_c"] is not None:
            cc, sc = plan["cos_c"][None, :, :], plan["sin_c"][None, :, :]
            u_ch_h = u_geo_h * cc - v_geo_h * sc
            v_ch_h = u_geo_h * sc + v_geo_h * cc
        else:
            u_ch_h, v_ch_h = u_geo_h, v_geo_h

        u_child = _apply_vertical_stencil_1d(u_ch_h.reshape(u_ch_h.shape[0], n_col), *v_sten_s).reshape(nz_c, ny_c, nx_c)
        v_child = _apply_vertical_stencil_1d(v_ch_h.reshape(v_ch_h.shape[0], n_col), *v_sten_s).reshape(nz_c, ny_c, nx_c)
        w_child = _regrid_full_3d(w_src, v_sten_w, nz_c + 1)
        w_child[-1] = 0.0
        theta_child = _regrid_full_3d(parent_state["theta_p"], v_sten_s, nz_c)
        pi_child = _regrid_full_3d(parent_state["pi_p"], v_sten_s, nz_c)
        qv_child = _regrid_full_3d(parent_state["qv"], v_sten_s, nz_c, clamp_nn=True)
        ps_child = _horizontal_regrid_3d_to_child(ps_p[None, :, :], parent_spec, child_spec, stencil=stencil)[0]

        v_dt = valid_time_utc.astimezone(datetime.timezone.utc) if valid_time_utc.tzinfo else valid_time_utc.replace(tzinfo=datetime.timezone.utc)
        t_coord = np.array([np.datetime64(int(v_dt.timestamp()), "s")], dtype="datetime64[s]")

        data_vars = {
            "u": (("z", "y", "x"), u_child, {"units": "m s-1"}),
            "v": (("z", "y", "x"), v_child, {"units": "m s-1"}),
            "w": (("z_w", "y", "x"), w_child, {"units": "m s-1"}),
            "theta_p": (("z", "y", "x"), theta_child, {"units": "K"}),
            "pi_p": (("z", "y", "x"), pi_child, {"units": "1"}),
            "qv": (("z", "y", "x"), qv_child, {"units": "kg kg-1"}),
            "ps": (("y", "x"), ps_child, {"units": "Pa"}),
        }

        if include_hydrometeors:
            for qname in LBC_HYDROMETEORS:
                if qname in parent_state:
                    data_vars[qname] = (
                        ("z", "y", "x"),
                        _regrid_full_3d(parent_state[qname], v_sten_s, nz_c, clamp_nn=True),
                        {"units": "kg kg-1"},
                    )

        ds = xr.Dataset(
            data_vars=data_vars,
            coords={
                "time": ("time", t_coord),
                "z": ("z", child_spec.vertical.zeta_s.astype(np.float32), {"units": "m"}),
                "z_w": ("z_w", child_spec.vertical.zeta_w.astype(np.float32), {"units": "m"}),
                "y": ("y", child_spec.y_m.astype(np.float64), {"units": "m"}),
                "x": ("x", child_spec.x_m.astype(np.float64), {"units": "m"}),
            },
            attrs={
                "parent_domain": parent_spec.name,
                "child_domain": child_spec.name,
                "step_index": str(int(step_index)),
                "sim_time_sec": f"{sim_time_sec:.3f}",
                "valid_time_utc": v_dt.isoformat(),
            },
        )
        kz_chunk = max(1, min(nz_c, (16 * 1024 * 1024) // max(1, ny_c * nx_c * 4)))
        chunks_map = {
            str(vn): [min(int(da.shape[0]), kz_chunk), int(da.shape[1]), int(da.shape[2])]
            for vn, da in ds.data_vars.items()
            if da.ndim == 3
        }

    wx1.write_dataset(
        out_path,
        ds,
        dataset_id=f"lbc_{child_spec.name}_{step_index:04d}",
        codec="zstd_shuffle",
        chunk_shapes=chunks_map,
        workers=4,
    )
    return out_path


def _unpack_rim(
    packed: Dict[str, np.ndarray],
    meta: Tuple[int, int, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Expand compact `rim_packed` arrays onto full child grid shapes."""
    ny_c, nx_c, rim_flat = meta
    n_xy = ny_c * nx_c
    out: Dict[str, np.ndarray] = {}
    for k, arr in packed.items():
        if arr.ndim == 2:
            nk = arr.shape[0]
            full = np.zeros((nk, ny_c, nx_c), dtype=np.float32)
            full.reshape(nk, n_xy)[:, rim_flat] = arr
            out[k] = full
        elif arr.ndim == 1:
            full2 = np.zeros((ny_c, nx_c), dtype=np.float32)
            full2.reshape(n_xy)[rim_flat] = arr
            out[k] = full2
        else:
            out[k] = arr.copy()
    return out


class LBCInterpolator:
    """Loads time-indexed WX-1 v0.1.8 LBC snapshots and interpolates linearly at every model timestep."""

    def __init__(self, lbc_dir: Path | str, domain_name: str):
        self.lbc_dir = Path(lbc_dir)
        self.domain_name = domain_name
        files = sorted(self.lbc_dir.glob(f"lbc_{domain_name}_step_*.wx1"))
        if not files:
            raise FileNotFoundError(f"No LBC WX-1 files found for {domain_name} in {self.lbc_dir}")
        self.steps: list[Tuple[float, Path]] = []
        for f in files:
            ds = wx1.open_dataset(f)
            try:
                t_sec = float(ds.attrs.get("sim_time_sec", "0.0"))
            finally:
                ds.close()
            self.steps.append((t_sec, f))
        self.steps.sort(key=lambda pair: pair[0])
        self._cache: Dict[Path, Dict[str, np.ndarray]] = {}
        self._rim_meta: Dict[Path, Optional[Tuple[int, int, np.ndarray]]] = {}

    def _load(self, path: Path) -> Dict[str, np.ndarray]:
        if path not in self._cache:
            if len(self._cache) >= 4:
                oldest = next(iter(self._cache))
                del self._cache[oldest]
                self._rim_meta.pop(oldest, None)
            ds = wx1.open_dataset(path)
            try:
                if ds.attrs.get("lbc_storage") == "rim_packed":
                    ny_c = int(ds.attrs["child_ny"])
                    nx_c = int(ds.attrs["child_nx"])
                    rj = np.asarray(ds["rim_j"].values, dtype=np.int64)
                    ri = np.asarray(ds["rim_i"].values, dtype=np.int64)
                    self._rim_meta[path] = (ny_c, nx_c, rj * nx_c + ri)
                    self._cache[path] = {
                        name: np.asarray(ds[name].values, dtype=np.float32)
                        for name in ds.data_vars
                        if name not in ("rim_j", "rim_i")
                    }
                else:
                    self._rim_meta[path] = None
                    self._cache[path] = {
                        name: np.asarray(ds[name].values, dtype=np.float32)
                        for name in ds.data_vars
                    }
            finally:
                ds.close()
        return self._cache[path]

    def get_lbc_at_time(self, sim_time_sec: float) -> Dict[str, np.ndarray]:
        if len(self.steps) == 1 or sim_time_sec <= self.steps[0][0]:
            p = self.steps[0][1]
            d = self._load(p)
            m = self._rim_meta.get(p)
            return _unpack_rim(d, m) if m is not None else {k: v.copy() for k, v in d.items()}
        if sim_time_sec >= self.steps[-1][0]:
            p = self.steps[-1][1]
            d = self._load(p)
            m = self._rim_meta.get(p)
            return _unpack_rim(d, m) if m is not None else {k: v.copy() for k, v in d.items()}

        times = [s[0] for s in self.steps]
        idx_hi = int(np.searchsorted(times, sim_time_sec))
        idx_lo = max(0, idx_hi - 1)
        t0, p0 = self.steps[idx_lo]
        t1, p1 = self.steps[idx_hi]
        w = float(np.clip((sim_time_sec - t0) / max(t1 - t0, 1e-6), 0.0, 1.0))
        d0, d1 = self._load(p0), self._load(p1)
        m0, m1 = self._rim_meta.get(p0), self._rim_meta.get(p1)
        if m0 is not None and m1 is not None and m0[0] == m1[0] and m0[1] == m1[1] and np.array_equal(m0[2], m1[2]):
            blended = {
                k: ((1.0 - w) * d0[k] + w * d1[k]).astype(np.float32)
                for k in d0
                if k in d1
            }
            return _unpack_rim(blended, m0)
        f0 = _unpack_rim(d0, m0) if m0 is not None else d0
        f1 = _unpack_rim(d1, m1) if m1 is not None else d1
        return {
            k: ((1.0 - w) * f0[k] + w * f1[k]).astype(np.float32)
            for k in f0
            if k in f1
        }
