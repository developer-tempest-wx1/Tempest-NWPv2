"""Operational 2-Nest D2->D3 GPU Forecast Runner & WX-1 Output Pipeline for Tempest-NWP v2.

Strictly 100% deterministic classical atmospheric & surface physics (0% AI / 0% ML).
Features:
- Two-way / one-way nesting D2 Europe (7.5 km) -> WX-1 LBC -> D3 Benelux+Germany (1.5 km).
- Balanced 3D-Var + IAU warm-up with live KNMI 10-min AWS & DWD POI observations.
- Lossless WX-1 v0.1.8 (`zstd_shuffle`) hourly / `--ecmwf` forecast output schedule & restarts.
- Substepped physics cadence (`--physics-every`) for fast high-resolution GPU runs.
"""

from __future__ import annotations

import datetime
import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import xarray as xr
import wx1

from tempest_v2.cuda_bridge import NonHydrostaticGPUSolver
from tempest_v2.ic_builder import (
    aurora_wx1_to_ic,
    build_domain_static_fields,
    ecmwf_grib_to_domain_state,
    load_tempest_restart_wx1,
    save_tempest_restart_wx1,
    stagger_uv,
    unstagger_uv,
)
from tempest_v2.lbc_manager import LBCInterpolator, export_child_lbc_step
from tempest_v2.projection import DomainSpec, get_domain_spec
from tempest_v2.verification import (
    compute_mslp_pa,
    load_station_obs_wx1,
    run_domain_3dvar,
    verify_against_stations,
)

logger = logging.getLogger("tempest_v2.runner")

RUN_LAYOUT_SUBDIRS = (
    "config",
    "obs",
    "da",
    "d2",
    "d3",
    "restarts",
    "ensemble",
    "verification",
    "logs",
)


def create_run_layout(
    runs_root: Path | str,
    init_time_utc: datetime.datetime,
    run_tag: Optional[str] = None,
) -> Dict[str, Path]:
    """Create isolated per-cycle run directory layout `runs_root/YYYYMMDD_HHZ[/...]`."""
    dt_utc = (
        init_time_utc.astimezone(datetime.timezone.utc)
        if init_time_utc.tzinfo
        else init_time_utc.replace(tzinfo=datetime.timezone.utc)
    )
    suffix = f"_{run_tag.strip()}" if run_tag and run_tag.strip() else ""
    run_dir = Path(runs_root) / f"{dt_utc:%Y%m%d_%H}Z{suffix}"
    return _ensure_run_layout_dirs(run_dir)


def _ensure_run_layout_dirs(run_dir: Path) -> Dict[str, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    layout: Dict[str, Path] = {"run_dir": run_dir}
    for sub in RUN_LAYOUT_SUBDIRS:
        p = run_dir / sub
        p.mkdir(parents=True, exist_ok=True)
        layout[sub] = p
    return layout


def forecast_output_schedule_hours(
    max_hours: float,
    *,
    ecmwf_schedule: bool = False,
    hourly: bool = True,
) -> List[float]:
    """Compute output lead hours.

    `--ecmwf` strictly selects the output cadence (3-hourly through T+144, then 6-hourly through T+360);
    it never imports ECMWF model physics or downstream forecast fields.
    """
    m = max(0.0, float(max_hours))
    if ecmwf_schedule:
        hours: List[float] = [float(h) for h in range(0, 145, 3) if h <= m + 1e-6]
        hours.extend(float(h) for h in range(150, 361, 6) if h <= m + 1e-6)
        if not hours or abs(hours[-1] - m) > 1e-6:
            if m > 0.0 and (not hours or hours[-1] < m):
                hours.append(m)
        return hours
    if hourly:
        hours = [float(h) for h in range(0, int(math.floor(m + 1e-6)) + 1)]
        if m > hours[-1] + 1e-6:
            hours.append(m)
        return hours
    return [0.0, m] if m > 0.0 else [0.0]


def _lbc_dict_to_staggered(
    lbc_raw: Dict[str, np.ndarray],
    target_shape_like: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    u_raw = np.asarray(lbc_raw["u"], dtype=np.float32)
    v_raw = np.asarray(lbc_raw["v"], dtype=np.float32)
    if u_raw.shape[2] == target_shape_like["u"].shape[2] - 1:
        u_st, v_st = stagger_uv(u_raw, v_raw)
    else:
        u_st, v_st = u_raw, v_raw
    return {
        "u": u_st.astype(np.float32),
        "v": v_st.astype(np.float32),
        "w": np.asarray(lbc_raw["w"], dtype=np.float32),
        "theta_p": np.asarray(lbc_raw["theta_p"], dtype=np.float32),
        "pi_p": np.asarray(lbc_raw["pi_p"], dtype=np.float32),
    }


def write_forecast_wx1(
    out_path: Path | str,
    valid_time_utc: datetime.datetime,
    lead_hours: float,
    spec: DomainSpec,
    hs_m: np.ndarray,
    state_3d: Dict[str, np.ndarray],
    surf_state: Dict[str, np.ndarray],
    *,
    theta_sfc_ref: float = 288.15,
    p_sfc_ref: float = 101_325.0,
    n_ref_hz: float = 0.01,
) -> Path:
    """Write synoptic 2D + 4-layer soil forecast product in WX-1 v0.1.8 (`zstd_shuffle`)."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    v_dt = (
        valid_time_utc.astimezone(datetime.timezone.utc)
        if valid_time_utc.tzinfo
        else valid_time_utc.replace(tzinfo=datetime.timezone.utc)
    )
    t_coord = np.array([np.datetime64(int(v_dt.timestamp()), "s")], dtype="datetime64[s]")

    mslp = compute_mslp_pa(
        spec, hs_m, state_3d, theta_sfc_ref=theta_sfc_ref, p_sfc_ref=p_sfc_ref, n_ref_hz=n_ref_hz
    )

    u10_r = surf_state.get("u10m")
    v10_r = surf_state.get("v10m")
    if u10_r is not None and v10_r is not None and float(np.max(np.abs(u10_r)) + np.max(np.abs(v10_r))) > 1e-3:
        u_grid, v_grid = u10_r.astype(np.float64), v10_r.astype(np.float64)
    else:
        uc, vc = unstagger_uv(state_3d["u"], state_3d["v"])
        u_grid, v_grid = uc[0].astype(np.float64), vc[0].astype(np.float64)

    if spec.is_projected and spec.projection is not None:
        u10_geo, v10_geo = spec.projection.grid_to_geo_wind(u_grid, v_grid, spec.lon2d)
    else:
        u10_geo, v10_geo = u_grid, v_grid

    def _var2d(name: str, fallback: float = 0.0) -> np.ndarray:
        arr = surf_state.get(name)
        if arr is None:
            return np.full((spec.ny, spec.nx), fallback, dtype=np.float32)
        return np.asarray(arr, dtype=np.float32)

    data_vars: Dict[str, Any] = {
        "t2m": (("y", "x"), _var2d("t2m", 285.0), {"units": "K", "standard_name": "air_temperature"}),
        "q2m": (("y", "x"), _var2d("q2m", 0.005), {"units": "kg kg-1", "standard_name": "specific_humidity"}),
        "u10m_geo": (("y", "x"), u10_geo.astype(np.float32), {"units": "m s-1", "standard_name": "eastward_wind"}),
        "v10m_geo": (("y", "x"), v10_geo.astype(np.float32), {"units": "m s-1", "standard_name": "northward_wind"}),
        "mslp": (("y", "x"), mslp.astype(np.float32), {"units": "Pa", "standard_name": "air_pressure_at_mean_sea_level"}),
        "rain_acc": (("y", "x"), _var2d("rain_acc"), {"units": "kg m-2"}),
        "snow_acc": (("y", "x"), _var2d("snow_acc"), {"units": "kg m-2"}),
        "graupel_acc": (("y", "x"), _var2d("graupel_acc"), {"units": "kg m-2"}),
        "sw_down": (("y", "x"), _var2d("sw_down"), {"units": "W m-2"}),
        "lw_down": (("y", "x"), _var2d("lw_down"), {"units": "W m-2"}),
        "shf": (("y", "x"), _var2d("shf"), {"units": "W m-2"}),
        "lhf": (("y", "x"), _var2d("lhf"), {"units": "W m-2"}),
        "ustar": (("y", "x"), _var2d("ustar"), {"units": "m s-1"}),
        "pblh": (("y", "x"), _var2d("pblh"), {"units": "m"}),
        "t_skin": (("y", "x"), _var2d("t_skin", 285.0), {"units": "K"}),
        "snow_swe": (("y", "x"), _var2d("snow_swe"), {"units": "kg m-2"}),
        "dt_cool": (("y", "x"), _var2d("dt_cool"), {"units": "K"}),
        "dt_warm": (("y", "x"), _var2d("dt_warm"), {"units": "K"}),
        "lake_hice": (("y", "x"), _var2d("lake_hice"), {"units": "m"}),
        "lat": (("y", "x"), spec.lat2d.astype(np.float32), {"units": "degrees_north"}),
        "lon": (("y", "x"), spec.lon2d.astype(np.float32), {"units": "degrees_east"}),
    }
    if "soil_t" in surf_state:
        data_vars["soil_t"] = (("soil_layer", "y", "x"), np.asarray(surf_state["soil_t"], dtype=np.float32), {"units": "K"})
    if "soil_m" in surf_state:
        data_vars["soil_m"] = (("soil_layer", "y", "x"), np.asarray(surf_state["soil_m"], dtype=np.float32), {"units": "m3 m-3"})

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": ("time", t_coord),
            "soil_layer": ("soil_layer", np.array([0, 1, 2, 3], dtype=np.int32)),
            "y": ("y", spec.y_m.astype(np.float64), {"units": "m"}),
            "x": ("x", spec.x_m.astype(np.float64), {"units": "m"}),
        },
        attrs={
            "model": "Tempest-NWP v2 (0% AI/ML)",
            "domain_name": spec.name,
            "lead_hours": f"{float(lead_hours):.3f}",
            "valid_time_utc": v_dt.isoformat(),
        },
    )
    wx1.write_dataset(p, ds, dataset_id=f"fc_{spec.name}_{lead_hours:06.2f}h", codec="zstd_shuffle", workers=4)
    return p


def run_domain_forecast(
    spec: DomainSpec,
    hs_m: np.ndarray,
    surface_static: Dict[str, np.ndarray],
    state_3d: Dict[str, np.ndarray],
    surf_state: Dict[str, np.ndarray],
    init_time_utc: datetime.datetime,
    *,
    forecast_hours: float,
    dt_s: float = 4.0,
    physics_every: int = 1,
    lbc_interpolator: Optional[LBCInterpolator] = None,
    lbc_export_child: Optional[Tuple[DomainSpec, np.ndarray, Path, float]] = None,
    iau_forcing: Optional[Dict[str, np.ndarray]] = None,
    iau_duration_s: float = 0.0,
    output_dir: Optional[Path | str] = None,
    output_hours: Optional[Sequence[float]] = None,
    save_restart_path: Optional[Path | str] = None,
    save_cycle_restart_hours: Sequence[float] = (),
    cycle_restart_dir: Optional[Path | str] = None,
    enable_physics: bool = True,
    rad_interval_s: float = 900.0,
    lbc_update_interval_s: float = 300.0,
    cloud_num_land: float = 200.0e6,
    cloud_num_sea: float = 80.0e6,
    static_overrides: Optional[Dict[str, np.ndarray]] = None,
    theta_sfc_ref: float = 288.15,
    p_sfc_ref: float = 101_325.0,
    n_ref_hz: float = 0.01,
) -> Dict[str, Any]:
    """Integrate one nest on GPU with Davies LBCs, IAU spinup, WX-1 outputs & restart."""
    init_dt = (
        init_time_utc.astimezone(datetime.timezone.utc)
        if init_time_utc.tzinfo
        else init_time_utc.replace(tzinfo=datetime.timezone.utc)
    )
    total_sec = max(0.0, float(forecast_hours) * 3600.0)
    total_steps = int(round(total_sec / float(dt_s)))
    doy = float(init_dt.timetuple().tm_yday)
    utc_s = float(init_dt.hour * 3600 + init_dt.minute * 60 + init_dt.second)

    solver = NonHydrostaticGPUSolver(
        spec.vertical,
        dx_m=spec.dx_m,
        dy_m=spec.dy_m,
        hs_m=hs_m,
        msf=spec.map_factor,
        f_cor=spec.coriolis_f,
        has_moist=True,
        has_physics=enable_physics,
        bc_x=2,
        bc_y=2,
        use_coriolis=True,
        has_iau=(iau_forcing is not None),
        sponge_depth_m=min(6000.0, 0.22 * spec.vertical.z_top_m),
        sponge_tau_max_s=0.01,
        lbc_rim_width=max(2, min(8, spec.nx // 4)),
        lbc_tau_max_s=0.04,
        theta_sfc_k=theta_sfc_ref,
        p_sfc_pa=p_sfc_ref,
        n_ref_s=n_ref_hz,
    )

    out_paths: List[Path] = []
    lbc_paths: List[Path] = []
    cycle_restart_paths: List[Path] = []
    t0_wall = time.perf_counter()

    try:
        solver.upload_state(state_3d)
        eff_static = dict(surface_static)
        if static_overrides:
            eff_static.update(static_overrides)
        solver.set_surface_static(eff_static)
        solver.upload_surface(surf_state)
        solver.set_physics_config(
            enable_microphysics=1 if enable_physics else 0,
            enable_pbl=1 if enable_physics else 0,
            enable_radiation=1 if enable_physics else 0,
            enable_surface=1 if enable_physics else 0,
            rad_interval_s=float(rad_interval_s),
            cloud_num_land=float(cloud_num_land),
            cloud_num_sea=float(cloud_num_sea),
            julian_day=doy,
            utc_seconds=utc_s,
        )
        if lbc_interpolator is not None:
            init_lbc = _lbc_dict_to_staggered(lbc_interpolator.get_lbc_at_time(0.0), state_3d)
        else:
            init_lbc = {k: state_3d[k].copy() for k in ("u", "v", "w", "theta_p", "pi_p")}
        solver.set_lbc_target(init_lbc)

        iau_active = iau_forcing is not None and iau_duration_s > 0.0
        if iau_active:
            solver.set_iau_forcing(iau_forcing)  # type: ignore[arg-type]

        # Build sorted breakpoint times (outputs, LBC exports, cycle restarts, IAU turn-off, LBC refreshes)
        breaks = {0, total_steps}
        out_dir = Path(output_dir) if output_dir is not None else None
        out_step_map: Dict[int, float] = {}
        if out_dir is not None and output_hours:
            for hr in output_hours:
                s_idx = int(round(float(hr) * 3600.0 / float(dt_s)))
                if 0 <= s_idx <= total_steps:
                    breaks.add(s_idx)
                    out_step_map[s_idx] = float(hr)

        cyc_dir = Path(cycle_restart_dir) if cycle_restart_dir is not None else None
        cycle_restart_step_map: Dict[int, float] = {}
        if cyc_dir is not None and save_cycle_restart_hours:
            cyc_dir.mkdir(parents=True, exist_ok=True)
            for hr in save_cycle_restart_hours:
                s_idx = int(round(float(hr) * 3600.0 / float(dt_s)))
                if 0 < s_idx <= total_steps:
                    breaks.add(s_idx)
                    cycle_restart_step_map[s_idx] = float(hr)

        child_step_map: Dict[int, int] = {}
        if lbc_export_child is not None:
            _cs, _chs, lbc_dir_c, lbc_dt_c = lbc_export_child
            lbc_stride = max(1, int(round(float(lbc_dt_c) / float(dt_s))))
            for c_i, s_idx in enumerate(range(0, total_steps + 1, lbc_stride)):
                breaks.add(s_idx)
                child_step_map[s_idx] = c_i
            if total_steps not in child_step_map and total_steps > 0:
                breaks.add(total_steps)
                child_step_map[total_steps] = len(child_step_map)

        iau_off_step = int(round(float(iau_duration_s) / float(dt_s))) if iau_active else -1
        if 0 < iau_off_step < total_steps:
            breaks.add(iau_off_step)

        if lbc_interpolator is not None and total_steps > 0:
            lbc_stride_in = max(1, int(round(float(lbc_update_interval_s) / float(dt_s))))
            for s_idx in range(0, total_steps + 1, lbc_stride_in):
                breaks.add(s_idx)

        ordered_breaks = sorted(breaks)
        cur_state = state_3d
        cur_surf = surf_state

        def _handle_step_events(step_idx: int) -> None:
            nonlocal cur_state, cur_surf, iau_active
            sim_sec = step_idx * float(dt_s)
            v_dt = init_dt + datetime.timedelta(seconds=sim_sec)

            if step_idx > 0 and (
                step_idx in out_step_map
                or step_idx in child_step_map
                or step_idx in cycle_restart_step_map
                or step_idx == total_steps
            ):
                cur_state = solver.download_state()
                cur_surf = solver.download_surface()

            if step_idx in child_step_map and lbc_export_child is not None:
                c_spec, c_hs, lbc_dir_c, _ = lbc_export_child
                lp = export_child_lbc_step(
                    lbc_dir_c,
                    child_step_map[step_idx],
                    sim_sec,
                    v_dt,
                    spec,
                    cur_state,
                    hs_m,
                    c_spec,
                    c_hs,
                    include_hydrometeors=False,
                )
                lbc_paths.append(lp)

            if step_idx in out_step_map and out_dir is not None:
                hr = out_step_map[step_idx]
                tag = f"{int(round(hr)):03d}" if abs(hr - round(hr)) < 1e-4 else f"{hr:06.2f}"
                fp = out_dir / f"forecast_{spec.name}_T+{tag}h.wx1"
                write_forecast_wx1(
                    fp,
                    v_dt,
                    hr,
                    spec,
                    hs_m,
                    cur_state,
                    cur_surf,
                    theta_sfc_ref=theta_sfc_ref,
                    p_sfc_ref=p_sfc_ref,
                    n_ref_hz=n_ref_hz,
                )
                out_paths.append(fp)

            if step_idx in cycle_restart_step_map and cyc_dir is not None:
                hr = cycle_restart_step_map[step_idx]
                tag = f"{int(round(hr)):03d}" if abs(hr - round(hr)) < 1e-4 else f"{hr:06.2f}"
                rp = save_tempest_restart_wx1(
                    cyc_dir / f"{spec.name}_cycle_f{tag}_{v_dt:%Y%m%d_%H%M}Z.wx1",
                    v_dt,
                    spec,
                    hs_m,
                    cur_state,
                    cur_surf,
                    sim_time_sec=sim_sec,
                    theta_sfc_ref=theta_sfc_ref,
                    p_sfc_ref=p_sfc_ref,
                    n_ref_hz=n_ref_hz,
                )
                cycle_restart_paths.append(rp)

            if iau_active and step_idx >= iau_off_step and step_idx < total_steps:
                zero_iau = {k: np.zeros_like(v) for k, v in iau_forcing.items()}  # type: ignore[union-attr]
                solver.set_iau_forcing(zero_iau)
                iau_active = False

        _handle_step_events(0)

        for idx in range(len(ordered_breaks) - 1):
            s0, s1 = ordered_breaks[idx], ordered_breaks[idx + 1]
            n_seg = s1 - s0
            if n_seg <= 0:
                continue
            if lbc_interpolator is not None:
                t_mid = 0.5 * (s0 + s1) * float(dt_s)
                solver.set_lbc_target(_lbc_dict_to_staggered(lbc_interpolator.get_lbc_at_time(t_mid), state_3d))

            if enable_physics:
                solver.step_full(
                    float(dt_s),
                    n_steps=n_seg,
                    physics_every=physics_every,
                    flush_physics=(s1 == total_steps),
                )
            else:
                solver.step_rk3(float(dt_s), n_steps=n_seg)
            _handle_step_events(s1)

        if total_steps == 0:
            cur_state = solver.download_state()
            cur_surf = solver.download_surface()

        end_dt = init_dt + datetime.timedelta(seconds=total_steps * float(dt_s))
        restart_out: Optional[Path] = None
        if save_restart_path is not None:
            restart_out = save_tempest_restart_wx1(
                save_restart_path,
                end_dt,
                spec,
                hs_m,
                cur_state,
                cur_surf,
                sim_time_sec=total_steps * float(dt_s),
                theta_sfc_ref=theta_sfc_ref,
                p_sfc_ref=p_sfc_ref,
                n_ref_hz=n_ref_hz,
            )

        elapsed_s = time.perf_counter() - t0_wall
        return {
            "state_3d": cur_state,
            "surf_state": cur_surf,
            "valid_time_utc": end_dt,
            "elapsed_s": elapsed_s,
            "num_tiles": solver.num_tiles,
            "estimated_tile_vram_bytes": solver.estimated_tile_vram_bytes,
            "output_paths": out_paths,
            "lbc_paths": lbc_paths,
            "restart_path": restart_out,
            "cycle_restart_paths": cycle_restart_paths,
        }
    finally:
        solver.close()


def run_two_nest_pipeline(
    *,
    forecast_hours: float = 6.0,
    nz: int = 65,
    ic_source: str = "ecmwf",
    sfc_grib: Optional[Path | str] = None,
    pl_grib: Optional[Path | str] = None,
    aurora_wx1_path: Optional[Path | str] = None,
    restart_path: Optional[Path | str] = None,
    d2_restart_path: Optional[Path | str] = None,
    obs_dir: Optional[Path | str] = None,
    outdir: Path | str = "data/forecasts",
    runs_root: Optional[Path | str] = None,
    run_dir: Optional[Path | str] = None,
    run_tag: Optional[str] = None,
    cycle_restart_hours: Sequence[float] = (6.0,),
    ensemble_members: int = 0,
    ensemble_lead_hours: Optional[float] = None,
    ensemble_seed: Optional[int] = None,
    ensemble_domain: str = "d3",
    d2_spec: Optional[DomainSpec] = None,
    d3_spec: Optional[DomainSpec] = None,
    ecmwf_schedule: bool = False,
    enable_da: bool = True,
    dt_d2_s: float = 20.0,
    dt_d3_s: float = 4.0,
    physics_every: int = 5,
    lbc_interval_s: float = 1800.0,
) -> Dict[str, Any]:
    """Run operational D2 (7.5 km Europe) -> WX-1 LBC -> D3 (1.5 km Benelux+DE) pipeline."""
    t0_pipe = time.perf_counter()
    d2_spec = d2_spec if d2_spec is not None else get_domain_spec("d2", nz=nz)
    d3_spec = d3_spec if d3_spec is not None else get_domain_spec("d3", nz=nz)
    hs_d2, ss_d2 = build_domain_static_fields(d2_spec)
    hs_d3, ss_d3 = build_domain_static_fields(d3_spec)

    if ic_source == "restart" and restart_path is not None:
        st_d3, sf_d3, meta = load_tempest_restart_wx1(restart_path)
        init_dt = datetime.datetime.fromisoformat(meta["valid_time_utc"].replace("Z", "+00:00"))
        if d2_restart_path is not None:
            st_d2, sf_d2, _ = load_tempest_restart_wx1(d2_restart_path)
        elif sfc_grib is not None and pl_grib is not None:
            st_d2, sf_d2, _ = ecmwf_grib_to_domain_state(sfc_grib, pl_grib, d2_spec, hs_d2, surface_static=ss_d2)
        else:
            st_d2 = {k: np.zeros((nz if k != "w" else nz + 1, d2_spec.ny, d2_spec.nx + (1 if k == "u" else 0)), dtype=np.float32)
                     if k != "v" else np.zeros((nz, d2_spec.ny + 1, d2_spec.nx), dtype=np.float32)
                     for k in ("u", "v", "w", "theta_p", "pi_p", "qv", "qc", "qr", "qi", "qs", "qg")}
            sf_d2 = {k: np.zeros((d2_spec.ny, d2_spec.nx), dtype=np.float32)
                     for k in NonHydrostaticGPUSolver.SURFACE_2D_FIELDS}
            sf_d2["soil_t"] = np.full((4, d2_spec.ny, d2_spec.nx), 285.0, dtype=np.float32)
            sf_d2["soil_m"] = np.full((4, d2_spec.ny, d2_spec.nx), 0.28, dtype=np.float32)
    elif ic_source == "aurora" and aurora_wx1_path is not None:
        init_dt = datetime.datetime.now(datetime.timezone.utc).replace(minute=0, second=0, microsecond=0)
        st_d2, sf_d2 = aurora_wx1_to_ic(aurora_wx1_path, d2_spec, hs_d2)
        st_d3, sf_d3 = aurora_wx1_to_ic(aurora_wx1_path, d3_spec, hs_d3)
    elif sfc_grib is not None and pl_grib is not None:
        st_d2, sf_d2, init_dt = ecmwf_grib_to_domain_state(sfc_grib, pl_grib, d2_spec, hs_d2, surface_static=ss_d2)
        st_d3, sf_d3, _ = ecmwf_grib_to_domain_state(sfc_grib, pl_grib, d3_spec, hs_d3, surface_static=ss_d3)
    else:
        raise ValueError("Provide valid ECMWF GRIB2 paths, Aurora WX-1 path, or restart WX-1 path.")

    layout: Optional[Dict[str, Path]] = None
    if run_dir is not None:
        layout = _ensure_run_layout_dirs(Path(run_dir))
    elif runs_root is not None:
        layout = create_run_layout(runs_root, init_dt, run_tag=run_tag)

    if layout is not None:
        out_root = layout["run_dir"]
        lbc_dir = layout["d2"] / "lbc_d3"
        d2_out_dir = layout["d2"]
        d3_out_dir = layout["d3"]
        d2_restart_out = layout["restarts"] / f"restart_{d2_spec.name}.wx1"
        d3_restart_out = layout["restarts"] / f"restart_{d3_spec.name}.wx1"
        cyc_restart_dir: Optional[Path] = layout["restarts"]
        manifest_path: Optional[Path] = out_root / "manifest.json"
    else:
        out_root = Path(outdir)
        out_root.mkdir(parents=True, exist_ok=True)
        lbc_dir = out_root / "lbc_d3"
        d2_out_dir = None
        d3_out_dir = out_root / "d3_wx1"
        d2_restart_out = None
        d3_restart_out = out_root / f"restart_{d3_spec.name}.wx1"
        cyc_restart_dir = None
        manifest_path = None

    sched = forecast_output_schedule_hours(forecast_hours, ecmwf_schedule=ecmwf_schedule, hourly=True)

    # 1. Outer nest D2 -> export child WX-1 LBCs
    res_d2 = run_domain_forecast(
        d2_spec,
        hs_d2,
        ss_d2,
        st_d2,
        sf_d2,
        init_dt,
        forecast_hours=forecast_hours,
        dt_s=dt_d2_s,
        physics_every=max(1, physics_every // 2),
        lbc_export_child=(d3_spec, hs_d3, lbc_dir, lbc_interval_s),
        output_dir=d2_out_dir,
        output_hours=sched if d2_out_dir is not None else None,
        save_restart_path=d2_restart_out,
        save_cycle_restart_hours=cycle_restart_hours if cyc_restart_dir is not None else (),
        cycle_restart_dir=cyc_restart_dir,
    )

    # 2. Balanced 3D-Var + verification on D3 if observations exist
    da_metrics: Dict[str, Any] = {}
    iau_d3 = None
    if enable_da and obs_dir is not None and Path(obs_dir).exists():
        obs_list, stn_rows = load_station_obs_wx1(obs_dir, d3_spec, valid_time_utc=init_dt)
        if obs_list:
            da_metrics["verify_bg"] = verify_against_stations(d3_spec, st_d3, sf_d3, hs_d3, stn_rows)
            st_d3, sf_d3, iau_d3, var_res = run_domain_3dvar(
                d3_spec, st_d3, sf_d3, hs_d3, ss_d3["lsm"], obs_list, obs_dir=obs_dir
            )
            da_metrics["var3d"] = {
                "j0": var_res.initial_cost_j0,
                "jf": var_res.final_cost_j,
                "iterations": var_res.iterations,
                "converged": var_res.converged,
            }
            da_metrics["verify_ana"] = verify_against_stations(d3_spec, st_d3, sf_d3, hs_d3, stn_rows)

    # 3. Inner 1.5 km nest D3 (Control Forecast CF) with Davies LBCs & IAU
    lbc_interp = LBCInterpolator(lbc_dir, d3_spec.name)
    res_d3 = run_domain_forecast(
        d3_spec,
        hs_d3,
        ss_d3,
        st_d3,
        sf_d3,
        init_dt,
        forecast_hours=forecast_hours,
        dt_s=dt_d3_s,
        physics_every=physics_every,
        lbc_interpolator=lbc_interp,
        iau_forcing=iau_d3,
        iau_duration_s=900.0 if iau_d3 is not None else 0.0,
        output_dir=d3_out_dir,
        output_hours=sched,
        save_restart_path=d3_restart_out,
        save_cycle_restart_hours=cycle_restart_hours if cyc_restart_dir is not None else (),
        cycle_restart_dir=cyc_restart_dir,
    )

    if enable_da and obs_dir is not None and da_metrics.get("verify_bg"):
        _, stn_rows = load_station_obs_wx1(obs_dir, d3_spec, valid_time_utc=init_dt)
        da_metrics["verify_end"] = verify_against_stations(d3_spec, res_d3["state_3d"], res_d3["surf_state"], hs_d3, stn_rows)

    all_cyc_restarts = [str(p) for p in (res_d2.get("cycle_restart_paths", []) + res_d3.get("cycle_restart_paths", []))]
    ens_summary: Optional[Dict[str, Any]] = None
    _ = (ensemble_members, ensemble_lead_hours, ensemble_seed, ensemble_domain)

    if layout is not None and manifest_path is not None:
        manifest = {
            "init_time_utc": init_dt.isoformat(),
            "ic_source": ic_source,
            "warm_start": bool(ic_source == "restart"),
            "forecast_hours": float(forecast_hours),
            "nz": int(nz),
            "d2": {
                "domain": d2_spec.name,
                "elapsed_s": float(res_d2["elapsed_s"]),
                "restart_path": str(res_d2["restart_path"]) if res_d2.get("restart_path") else None,
                "cycle_restart_paths": [str(p) for p in res_d2.get("cycle_restart_paths", [])],
                "output_paths": [str(p) for p in res_d2.get("output_paths", [])],
            },
            "d3": {
                "domain": d3_spec.name,
                "elapsed_s": float(res_d3["elapsed_s"]),
                "restart_path": str(res_d3["restart_path"]) if res_d3.get("restart_path") else None,
                "cycle_restart_paths": [str(p) for p in res_d3.get("cycle_restart_paths", [])],
                "output_paths": [str(p) for p in res_d3.get("output_paths", [])],
            },
            "da": da_metrics,
            "ensemble": ens_summary,
            "cycle_restart_paths": all_cyc_restarts,
            "wall_time_s": float(time.perf_counter() - t0_pipe),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))

    return {
        "d2": res_d2,
        "d3": res_d3,
        "da": da_metrics,
        "run_dir": out_root,
        "layout": layout,
        "manifest_path": manifest_path,
        "cycle_restart_paths": res_d2.get("cycle_restart_paths", []) + res_d3.get("cycle_restart_paths", []),
        "ensemble": ens_summary,
    }
