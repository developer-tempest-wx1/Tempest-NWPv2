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
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import xarray as xr

import wx1

from tempest_v2.projection import DomainSpec

logger = logging.getLogger("tempest_v2.lbc_manager")

LBC_3D_VARS = ("u", "v", "theta_p", "pi_p", "qv")
LBC_HYDROMETEORS = ("qc", "qr", "qi", "qs", "qg")


def _horizontal_regrid_3d_to_child(
    field_3d: np.ndarray,
    parent_spec: DomainSpec,
    child_spec: DomainSpec,
) -> np.ndarray:
    """Bilinearly regrid parent [nk, ny_p, nx_p] onto child [nk, ny_c, nx_c]."""
    nk = field_3d.shape[0]
    out = np.empty((nk, child_spec.ny, child_spec.nx), dtype=np.float32)

    if parent_spec.is_projected and parent_spec.projection is not None:
        tx, ty = parent_spec.projection.latlon_to_xy(child_spec.lat2d, child_spec.lon2d)
        pts = np.column_stack([ty.ravel(), tx.ravel()])
        py_axis = parent_spec.y_m
        px_axis = parent_spec.x_m
    else:
        lats = parent_spec.y_m
        lons = parent_spec.x_m
        lat_tgt = child_spec.lat2d
        lon_tgt = child_spec.lon2d
        if parent_spec.is_periodic_lon and lons[0] >= -1e-6 and np.nanmax(lons) > 180.0:
            lon_tgt = np.mod(lon_tgt, 360.0)
        else:
            lon_tgt = ((lon_tgt + 180.0) % 360.0) - 180.0
        if lats[0] > lats[-1]:
            lats = lats[::-1]
            field_3d = field_3d[:, ::-1, :]
        if parent_spec.is_periodic_lon and len(lons) >= 2:
            lons = np.append(lons, lons[0] + 360.0)
            field_3d = np.concatenate([field_3d, field_3d[:, :, :1]], axis=2)
            lon_tgt = np.where(lon_tgt < lons[0], lon_tgt + 360.0, lon_tgt)
        py_axis = lats
        px_axis = lons
        pts = np.column_stack([lat_tgt.ravel(), lon_tgt.ravel()])

    for k in range(nk):
        interp = RegularGridInterpolator(
            (py_axis, px_axis),
            field_3d[k].astype(np.float64),
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
        out[k] = interp(pts).reshape(child_spec.ny, child_spec.nx).astype(np.float32)
    return out


def _vertical_interp_columnwise(
    src_vals: np.ndarray,
    src_z: np.ndarray,
    dst_z: np.ndarray,
    clamp_non_negative: bool = False,
) -> np.ndarray:
    """Linearly interpolate 3D [nz_src, ny, nx] in height src_z onto dst_z [nz_dst, ny, nx] (row-batched float32)."""
    nz_src, ny, nx = src_vals.shape
    nz_dst = dst_z.shape[0]
    out = np.empty((nz_dst, ny, nx), dtype=np.float32)

    chunk_ny = 32
    for j0 in range(0, ny, chunk_ny):
        j1 = min(ny, j0 + chunk_ny)
        n_col = (j1 - j0) * nx
        sf = np.asarray(src_vals[:, j0:j1, :], dtype=np.float32).reshape(nz_src, n_col)
        zs = np.asarray(src_z[:, j0:j1, :], dtype=np.float32).reshape(nz_src, n_col)
        zd = np.asarray(dst_z[:, j0:j1, :], dtype=np.float32).reshape(nz_dst, n_col)

        k_hi = np.clip(np.sum(zs[:, None, :] < zd[None, :, :], axis=0), 1, max(1, nz_src - 1))
        k_lo = k_hi - 1
        c_idx = np.arange(n_col)[None, :]
        z_lo = zs[k_lo, c_idx]
        z_hi = zs[k_hi, c_idx]
        w = np.clip((zd - z_lo) / np.maximum(z_hi - z_lo, 1e-5), 0.0, 1.0).astype(np.float32)
        val = (1.0 - w) * sf[k_lo, c_idx] + w * sf[k_hi, c_idx]
        out[:, j0:j1, :] = val.reshape(nz_dst, j1 - j0, nx)

    if clamp_non_negative:
        np.maximum(out, 0.0, out=out)
    return out


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
) -> Path:
    """Regrid parent 3D state onto child LCC + Klemp terrain-following coordinates and store as WX-1 v0.1.8."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"lbc_{child_spec.name}_step_{step_index:04d}.wx1"

    u_src = np.asarray(parent_state["u"], dtype=np.float32)
    v_src = np.asarray(parent_state["v"], dtype=np.float32)
    if u_src.shape[2] == parent_spec.nx + 1:
        u_src = 0.5 * (u_src[:, :, :-1] + u_src[:, :, 1:])
    if v_src.shape[1] == parent_spec.ny + 1:
        v_src = 0.5 * (v_src[:, :-1, :] + v_src[:, 1:, :])

    if parent_spec.is_projected and parent_spec.projection is not None:
        u_geo, v_geo = parent_spec.projection.grid_to_geo_wind(
            u_src.astype(np.float64), v_src.astype(np.float64), parent_spec.lon2d[None, :, :]
        )
    else:
        u_geo, v_geo = u_src, v_src

    w_src = np.asarray(parent_state["w"], dtype=np.float32)
    if w_src.shape[0] == parent_spec.nz + 1:
        w_src = 0.5 * (w_src[:-1] + w_src[1:])

    hs_parent_on_child = _horizontal_regrid_3d_to_child(parent_hs_m[None, :, :], parent_spec, child_spec)[0]
    _, z_parent_col, _ = parent_spec.vertical.heights_3d(hs_parent_on_child)
    z_child_w, z_child_s, _ = child_spec.vertical.heights_3d(child_hs_m)

    u_geo_h = _horizontal_regrid_3d_to_child(u_geo.astype(np.float32), parent_spec, child_spec)
    v_geo_h = _horizontal_regrid_3d_to_child(v_geo.astype(np.float32), parent_spec, child_spec)
    if child_spec.is_projected and child_spec.projection is not None:
        u_ch_h, v_ch_h = child_spec.projection.geo_to_grid_wind(
            u_geo_h.astype(np.float64), v_geo_h.astype(np.float64), child_spec.lon2d[None, :, :]
        )
    else:
        u_ch_h, v_ch_h = u_geo_h, v_geo_h

    u_child = _vertical_interp_columnwise(np.asarray(u_ch_h, dtype=np.float32), z_parent_col, z_child_s)
    v_child = _vertical_interp_columnwise(np.asarray(v_ch_h, dtype=np.float32), z_parent_col, z_child_s)

    w_h = _horizontal_regrid_3d_to_child(w_src, parent_spec, child_spec)
    w_child = _vertical_interp_columnwise(w_h, z_parent_col, z_child_w)
    w_child[-1] = 0.0

    theta_h = _horizontal_regrid_3d_to_child(np.asarray(parent_state["theta_p"], dtype=np.float32), parent_spec, child_spec)
    pi_h = _horizontal_regrid_3d_to_child(np.asarray(parent_state["pi_p"], dtype=np.float32), parent_spec, child_spec)
    qv_h = _horizontal_regrid_3d_to_child(np.asarray(parent_state["qv"], dtype=np.float32), parent_spec, child_spec)

    theta_child = _vertical_interp_columnwise(theta_h, z_parent_col, z_child_s)
    pi_child = _vertical_interp_columnwise(pi_h, z_parent_col, z_child_s)
    qv_child = _vertical_interp_columnwise(qv_h, z_parent_col, z_child_s, clamp_non_negative=True)

    ps_p = np.asarray(parent_state.get("ps", np.full((parent_spec.ny, parent_spec.nx), 101_325.0)), dtype=np.float32)
    ps_child = _horizontal_regrid_3d_to_child(ps_p[None, :, :], parent_spec, child_spec)[0]

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
                qh = _horizontal_regrid_3d_to_child(np.asarray(parent_state[qname], dtype=np.float32), parent_spec, child_spec)
                data_vars[qname] = (("z", "y", "x"), _vertical_interp_columnwise(qh, z_parent_col, z_child_s, clamp_non_negative=True), {"units": "kg kg-1"})

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
    kz_chunk = max(1, min(int(child_spec.nz), (16 * 1024 * 1024) // max(1, child_spec.ny * child_spec.nx * 4)))
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

    def _load(self, path: Path) -> Dict[str, np.ndarray]:
        if path not in self._cache:
            if len(self._cache) >= 4:
                oldest = next(iter(self._cache))
                del self._cache[oldest]
            ds = wx1.open_dataset(path)
            try:
                self._cache[path] = {
                    name: np.asarray(ds[name].values, dtype=np.float32)
                    for name in ds.data_vars
                }
            finally:
                ds.close()
        return self._cache[path]

    def get_lbc_at_time(self, sim_time_sec: float) -> Dict[str, np.ndarray]:
        if len(self.steps) == 1 or sim_time_sec <= self.steps[0][0]:
            return {k: v.copy() for k, v in self._load(self.steps[0][1]).items()}
        if sim_time_sec >= self.steps[-1][0]:
            return {k: v.copy() for k, v in self._load(self.steps[-1][1]).items()}

        times = [s[0] for s in self.steps]
        idx_hi = int(np.searchsorted(times, sim_time_sec))
        idx_lo = max(0, idx_hi - 1)
        t0, p0 = self.steps[idx_lo]
        t1, p1 = self.steps[idx_hi]
        w = float(np.clip((sim_time_sec - t0) / max(t1 - t0, 1e-6), 0.0, 1.0))
        d0, d1 = self._load(p0), self._load(p1)
        return {
            k: ((1.0 - w) * d0[k] + w * d1[k]).astype(np.float32)
            for k in d0
            if k in d1
        }
