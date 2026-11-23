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
import functools
import gc
import json
import logging
import math
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import xarray as xr
import wx1

try:
    from tqdm import tqdm as _tqdm
except ImportError:  # pragma: no cover - optional terminal progress bar
    _tqdm = None

from tempest_v2.cuda_bridge import NonHydrostaticGPUSolver, cuda_vram_info
from tempest_v2.diagnostics import (
    STANDARD_PLEV_PA,
    compute_pressure_level_fields,
    compute_synoptic_convective_2d,
)
from tempest_v2.ic_builder import (
    aurora_wx1_to_ic,
    build_domain_static_fields,
    clear_cfgrib_cache,
    ecmwf_grib_to_domain_state,
    load_tempest_restart_wx1,
    save_tempest_restart_wx1,
    stagger_uv,
    unstagger_uv,
)
from tempest_v2.ecmwf_lbc import ecmwf_grib_to_lbc_wx1, prepare_ecmwf_lbc_series
from tempest_v2.ingest_era5 import (
    download_era5_land_soil_cds,
    era5_wx1_to_domain_state,
    prepare_era5_lbc_series,
)
from tempest_v2.lbc_manager import LBCInterpolator, export_child_lbc_step
from tempest_v2.ocean_si3_bridge import NemoSi3GPUSolver
from tempest_v2.projection import (
    DomainSpec,
    LAM_DOMAIN_DEFAULT_DT_S,
    LAM_DOMAIN_DEFAULT_PHYSICS_EVERY,
    get_domain_spec,
)
from tempest_v2.verification import (
    compute_mslp_pa,
    load_station_obs_wx1,
    run_domain_3dvar,
    verify_against_stations,
)

logger = logging.getLogger("tempest_v2.runner")


@functools.lru_cache(maxsize=1)
def _get_git_commit() -> str:
    """Return short git commit hash of this Tempest repository (or 'unknown')."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            timeout=3.0,
        )
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    except Exception:
        pass
    return "unknown"

RUN_LAYOUT_SUBDIRS = (
    "config",
    "obs",
    "da",
    "d1",
    "d2",
    "d3",
    "ocean",
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


def _sample_mem_gpu_stats() -> Dict[str, Any]:
    """Sample process RSS/Swap, live np.ndarray count, and CUDA VRAM/thermal telemetry."""
    rss_mb = 0.0
    swap_mb = 0.0
    try:
        with open("/proc/self/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    rss_mb = float(line.split()[1]) / 1024.0
                elif line.startswith("VmSwap:"):
                    swap_mb = float(line.split()[1]) / 1024.0
    except OSError:
        pass
    vram_used_mib = 0
    vram_total_mib = 0
    try:
        free_b, total_b = cuda_vram_info()
        vram_used_mib = int((total_b - free_b) // (1024 * 1024))
        vram_total_mib = int(total_b // (1024 * 1024))
    except Exception:
        pass
    np_arrays = sum(1 for obj in gc.get_objects() if isinstance(obj, np.ndarray))
    gpu_temp = float("nan")
    gpu_clk = 0
    gpu_pwr = float("nan")
    gpu_thr = "n/a"
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu,clocks.sm,power.draw,clocks_throttle_reasons.active",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2.0,
        ).strip()
        if out:
            parts = [p.strip() for p in out.splitlines()[0].split(",")]
            if len(parts) >= 4:
                gpu_temp = float(parts[0])
                gpu_clk = int(float(parts[1]))
                gpu_pwr = float(parts[2])
                gpu_thr = parts[3]
    except Exception:
        pass
    return {
        "rss_mb": rss_mb,
        "swap_mb": swap_mb,
        "vram_used_mib": vram_used_mib,
        "vram_total_mib": vram_total_mib,
        "np_arrays": np_arrays,
        "gpu_temp_c": gpu_temp,
        "gpu_clk_mhz": gpu_clk,
        "gpu_pwr_w": gpu_pwr,
        "gpu_thr": gpu_thr,
    }


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
    out = {
        "u": u_st.astype(np.float32),
        "v": v_st.astype(np.float32),
        "w": np.asarray(lbc_raw["w"], dtype=np.float32),
        "theta_p": np.asarray(lbc_raw["theta_p"], dtype=np.float32),
        "pi_p": np.asarray(lbc_raw["pi_p"], dtype=np.float32),
    }
    if "qv" in lbc_raw:
        out["qv"] = np.asarray(lbc_raw["qv"], dtype=np.float32)
    return out


def extract_ocean_surface_diag(oce_state: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Extract centered 2D surface ocean & SI3 sea-ice diagnostics from a GPU NEMO+SI3 state."""
    u0 = np.asarray(oce_state["u_oce"][0], dtype=np.float32)
    v0 = np.asarray(oce_state["v_oce"][0], dtype=np.float32)
    u_c = 0.5 * (u0 + np.roll(u0, 1, axis=1))
    v_c = 0.5 * (v0 + np.pad(v0[:-1, :], ((1, 0), (0, 0)), mode="edge"))
    ui = np.asarray(oce_state.get("u_ice", np.zeros_like(u0)), dtype=np.float32)
    vi = np.asarray(oce_state.get("v_ice", np.zeros_like(v0)), dtype=np.float32)
    ui_c = 0.5 * (ui + np.roll(ui, 1, axis=1))
    vi_c = 0.5 * (vi + np.pad(vi[:-1, :], ((1, 0), (0, 0)), mode="edge"))
    return {
        "sst": np.asarray(oce_state["temp_k"][0], dtype=np.float32),
        "sss": np.asarray(oce_state["sal_psu"][0], dtype=np.float32),
        "ssh": np.asarray(oce_state["ssh"], dtype=np.float32),
        "mld": np.asarray(oce_state["mld"], dtype=np.float32),
        "u_oce_sfc": u_c.astype(np.float32),
        "v_oce_sfc": v_c.astype(np.float32),
        "sea_ice_frac": np.asarray(oce_state["ice_frac"], dtype=np.float32),
        "sea_ice_thick": np.asarray(oce_state["ice_thick"], dtype=np.float32),
        "sea_ice_tsfc": np.asarray(oce_state["ice_tsfc"], dtype=np.float32),
        "sea_ice_albedo": np.asarray(oce_state["ice_albedo"], dtype=np.float32),
        "u_ice": ui_c.astype(np.float32),
        "v_ice": vi_c.astype(np.float32),
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
    include_3d_diagnostics: bool = True,
    ocean_diag: Optional[Dict[str, np.ndarray]] = None,
    prev_precip_total: Optional[np.ndarray] = None,
    interval_s: float = 0.0,
    surface_static: Optional[Dict[str, np.ndarray]] = None,
    provenance_attrs: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write synoptic 2D, soil, optional 3D isobaric & NEMO+SI3 forecast product in WX-1 (`zstd_shuffle`)."""
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
    hs_arr = np.asarray(hs_m, dtype=np.float64)
    t2m_for_ps = np.maximum(np.asarray(surf_state.get("t2m", np.full((spec.ny, spec.nx), 285.0)), dtype=np.float64), 180.0)
    ps_pa = (mslp.astype(np.float64) * np.exp(-9.80665 * hs_arr / (287.058 * t2m_for_ps))).astype(np.float32)

    u10_r = surf_state.get("u10m")
    v10_r = surf_state.get("v10m")
    if u10_r is not None and v10_r is not None and float(np.max(np.abs(u10_r)) + np.max(np.abs(v10_r))) > 1e-3:
        u_grid, v_grid = u10_r.astype(np.float64), v10_r.astype(np.float64)
    else:
        uc, vc = unstagger_uv(state_3d["u"][:1], state_3d["v"][:1])
        z0_arr: Any = (
            np.clip(np.asarray(surface_static["z0m_base"], dtype=np.float64), 1e-4, 2.0)
            if (surface_static is not None and "z0m_base" in surface_static)
            else 0.15
        )
        z_0_agl = max(float(spec.vertical.zeta_s[0]), 10.5)
        fac_10m = np.clip(np.log(10.0 / z0_arr) / np.maximum(np.log(z_0_agl / z0_arr), 0.5), 0.25, 1.0)
        u_grid = fac_10m * uc[0].astype(np.float64)
        v_grid = fac_10m * vc[0].astype(np.float64)

    if spec.is_projected and spec.projection is not None:
        u10_geo, v10_geo = spec.projection.grid_to_geo_wind(u_grid, v_grid, spec.lon2d)
    else:
        u10_geo, v10_geo = u_grid, v_grid

    def _var2d(name: str, fallback: float = 0.0) -> np.ndarray:
        arr = surf_state.get(name)
        if arr is None:
            return np.full((spec.ny, spec.nx), fallback, dtype=np.float32)
        return np.asarray(arr, dtype=np.float32)

    lake_tml_out = (
        np.where(np.asarray(surface_static["lake_mask"]) >= 0.5, _var2d("t_skin", 283.0), _var2d("lake_tml", 283.0)).astype(np.float32)
        if (surface_static is not None and "lake_mask" in surface_static)
        else _var2d("lake_tml", 283.0)
    )

    data_vars: Dict[str, Any] = {
        "t2m": (("y", "x"), _var2d("t2m", 285.0), {"units": "K", "standard_name": "air_temperature"}),
        "q2m": (("y", "x"), _var2d("q2m", 0.005), {"units": "kg kg-1", "standard_name": "specific_humidity"}),
        "u10m_geo": (("y", "x"), u10_geo.astype(np.float32), {"units": "m s-1", "standard_name": "eastward_wind"}),
        "v10m_geo": (("y", "x"), v10_geo.astype(np.float32), {"units": "m s-1", "standard_name": "northward_wind"}),
        "ps": (("y", "x"), ps_pa, {"units": "Pa", "standard_name": "surface_air_pressure"}),
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
        "snow_temp": (("y", "x"), _var2d("snow_temp", 273.15), {"units": "K"}),
        "snow_rho": (("y", "x"), _var2d("snow_rho", 150.0), {"units": "kg m-3"}),
        "snow_albedo": (("y", "x"), _var2d("snow_albedo", 0.75), {"units": "1"}),
        "canopy_water": (("y", "x"), _var2d("canopy_water"), {"units": "kg m-2"}),
        "dt_cool": (("y", "x"), _var2d("dt_cool"), {"units": "K"}),
        "dt_warm": (("y", "x"), _var2d("dt_warm"), {"units": "K"}),
        "sst_bulk": (("y", "x"), _var2d("sst_bulk", 285.0), {"units": "K"}),
        "lake_tml": (("y", "x"), lake_tml_out, {"units": "K"}),
        "lake_tbot": (("y", "x"), _var2d("lake_tbot", 277.15), {"units": "K"}),
        "lake_hmld": (("y", "x"), _var2d("lake_hmld", 4.0), {"units": "m"}),
        "lake_hice": (("y", "x"), _var2d("lake_hice"), {"units": "m"}),
        "sea_ice_frac": (("y", "x"), _var2d("sea_ice_frac"), {"units": "1"}),
        "sea_ice_thick": (("y", "x"), _var2d("sea_ice_thick"), {"units": "m"}),
        "sea_ice_tsfc": (("y", "x"), _var2d("sea_ice_tsfc", 271.35), {"units": "K"}),
        "sea_ice_albedo": (("y", "x"), _var2d("sea_ice_albedo", 0.066), {"units": "1"}),
        "lat": (("y", "x"), spec.lat2d.astype(np.float32), {"units": "degrees_north"}),
        "lon": (("y", "x"), spec.lon2d.astype(np.float32), {"units": "degrees_east"}),
    }
    if "soil_t" in surf_state:
        data_vars["soil_t"] = (("soil_layer", "y", "x"), np.asarray(surf_state["soil_t"], dtype=np.float32), {"units": "K"})
    if "soil_m" in surf_state:
        data_vars["soil_m"] = (("soil_layer", "y", "x"), np.asarray(surf_state["soil_m"], dtype=np.float32), {"units": "m3 m-3"})

    coords: Dict[str, Any] = {
        "time": ("time", t_coord),
        "soil_layer": ("soil_layer", np.array([0, 1, 2, 3], dtype=np.int32)),
        "y": ("y", spec.y_m.astype(np.float64), {"units": "m"}),
        "x": ("x", spec.x_m.astype(np.float64), {"units": "m"}),
    }

    if include_3d_diagnostics:
        pl_fields = compute_pressure_level_fields(
            spec,
            hs_m,
            state_3d,
            plev_pa=STANDARD_PLEV_PA,
            theta_sfc_ref=theta_sfc_ref,
            p_sfc_ref=p_sfc_ref,
            n_ref_hz=n_ref_hz,
        )
        pl_units = {
            "t_pl": "K",
            "gh_pl": "m",
            "z_pl": "m2 s-2",
            "theta_pl": "K",
            "q_pl": "kg kg-1",
            "rh_pl": "%",
            "u_pl": "m s-1",
            "v_pl": "m s-1",
            "w_pl": "m s-1",
            "omega_pl": "Pa s-1",
            "vo_pl": "s-1",
            "div_pl": "s-1",
            "qc_pl": "kg kg-1",
            "qr_pl": "kg kg-1",
            "qi_pl": "kg kg-1",
            "qs_pl": "kg kg-1",
            "qg_pl": "kg kg-1",
        }
        coords["plev"] = ("plev", STANDARD_PLEV_PA, {"units": "Pa", "standard_name": "air_pressure"})
        for k, arr in pl_fields.items():
            data_vars[k] = (("plev", "y", "x"), arr, {"units": pl_units[k]})

        syn_fields = compute_synoptic_convective_2d(
            spec,
            hs_m,
            state_3d,
            surf_state,
            theta_sfc_ref=theta_sfc_ref,
            p_sfc_ref=p_sfc_ref,
            n_ref_hz=n_ref_hz,
            prev_precip_total=prev_precip_total,
            interval_s=interval_s,
        )
        syn_units = {
            "td2m": "K",
            "rh2m": "%",
            "gust10m": "m s-1",
            "tcc": "1",
            "lcc": "1",
            "mcc": "1",
            "hcc": "1",
            "tcwv": "kg m-2",
            "lwp": "kg m-2",
            "iwp": "kg m-2",
            "cape_sb": "J kg-1",
            "cin_sb": "J kg-1",
            "cape_mu": "J kg-1",
            "shear_0_6km": "m s-1",
            "srh_0_3km": "m2 s-2",
            "freezing_level_m": "m",
            "vis_m": "m",
            "precip_rate": "kg m-2 s-1",
        }
        for k, arr in syn_fields.items():
            data_vars[k] = (("y", "x"), arr, {"units": syn_units[k]})

    if ocean_diag is not None:
        od = extract_ocean_surface_diag(ocean_diag) if "temp_k" in ocean_diag else ocean_diag
        oce_units = {
            "sst": "K",
            "sss": "g kg-1",
            "ssh": "m",
            "mld": "m",
            "u_oce_sfc": "m s-1",
            "v_oce_sfc": "m s-1",
            "sea_ice_frac": "1",
            "sea_ice_thick": "m",
            "sea_ice_tsfc": "K",
            "sea_ice_albedo": "1",
            "u_ice": "m s-1",
            "v_ice": "m s-1",
        }
        for k, unit in oce_units.items():
            if k in od:
                data_vars[k] = (("y", "x"), np.asarray(od[k], dtype=np.float32), {"units": unit})

        if "temp_k" in ocean_diag:
            nz_o = int(np.asarray(ocean_diag["temp_k"]).shape[0])
            coords["z_oce"] = ("z_oce", np.arange(nz_o, dtype=np.int32))
            oce_3d_map = (
                ("oce_temp", "temp_k", "K"),
                ("oce_sal", "sal_psu", "g kg-1"),
                ("oce_u", "u_oce", "m s-1"),
                ("oce_v", "v_oce", "m s-1"),
                ("oce_tke", "tke", "m2 s-2"),
                ("oce_avm", "avm", "m2 s-1"),
                ("oce_avt", "avt", "m2 s-1"),
            )
            for out_k, src_k, unit in oce_3d_map:
                if src_k in ocean_diag:
                    data_vars[out_k] = (("z_oce", "y", "x"), np.asarray(ocean_diag[src_k], dtype=np.float32), {"units": unit})
            if "a_i" in ocean_diag:
                n_cat = int(np.asarray(ocean_diag["a_i"]).shape[0])
                coords["ice_cat"] = ("ice_cat", np.arange(n_cat, dtype=np.int32))
                si3_cat_map = (
                    ("si3_a_i", "a_i", "1"),
                    ("si3_v_i", "v_i", "m"),
                    ("si3_v_s", "v_s", "m"),
                    ("si3_t_su", "t_su", "K"),
                    ("si3_a_ip", "a_ip", "1"),
                    ("si3_v_ip", "v_ip", "m"),
                )
                for out_k, src_k, unit in si3_cat_map:
                    if src_k in ocean_diag:
                        data_vars[out_k] = (("ice_cat", "y", "x"), np.asarray(ocean_diag[src_k], dtype=np.float32), {"units": unit})

    ds_attrs: Dict[str, str] = {
        "model": "Tempest-NWP v2 (0% AI/ML)",
        "domain_name": spec.name,
        "lead_hours": f"{float(lead_hours):.3f}",
        "valid_time_utc": v_dt.isoformat(),
        "git_commit": _get_git_commit(),
        "nz": str(int(spec.nz)),
    }
    if provenance_attrs:
        for k_attr, v_attr in provenance_attrs.items():
            if v_attr is not None:
                ds_attrs[str(k_attr)] = str(v_attr)

    ds = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs=ds_attrs,
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
    enable_ocean_si3: bool = False,
    ocean_coupling_interval_s: float = 1800.0,
    ocean_nz: int = 16,
    osisaf_wx1_path: Optional[Path | str] = None,
    obs_dir: Optional[Path | str] = None,
    ocean_restart_in: Optional[Path | str] = None,
    ocean_restart_out: Optional[Path | str] = None,
    include_3d_diagnostics: bool = True,
    show_progress: bool = True,
    log_steps: bool = True,
    mem_profile_every: int = 0,
    pbl_last: bool = False,
    cy46_physics: bool = False,
) -> Dict[str, Any]:
    """Integrate one nest on GPU with optional NEMO+SI3 coupling, Davies LBCs, IAU, WX-1 outputs & restart."""
    init_dt = (
        init_time_utc.astimezone(datetime.timezone.utc)
        if init_time_utc.tzinfo
        else init_time_utc.replace(tzinfo=datetime.timezone.utc)
    )
    total_sec = max(0.0, float(forecast_hours) * 3600.0)
    total_steps = int(round(total_sec / float(dt_s)))
    doy = float(init_dt.timetuple().tm_yday)
    utc_s = float(init_dt.hour * 3600 + init_dt.minute * 60 + init_dt.second)

    eff_static = dict(surface_static)
    if static_overrides:
        eff_static.update(static_overrides)

    # CRITICAL VRAM ORDER: allocate NEMO+SI3 before NonHydrostaticGPUSolver so auto-tiling sees post-ocean free VRAM.
    ocean_solver: Optional[NemoSi3GPUSolver] = None
    if enable_ocean_si3:
        ocean_solver = NemoSi3GPUSolver(
            spec,
            eff_static,
            nz=int(ocean_nz),
            ah_lap=max(50.0, min(1200.0, 0.015 * spec.dx_m)),
        )

    use_spherical = bool(spec.is_spherical)
    bc_x_val = 1 if spec.is_periodic_lon else 2
    bc_y_val = 0 if (use_spherical or spec.is_periodic_lon) else 2
    has_parent_lbc = (lbc_interpolator is not None) or ((not use_spherical) and (not spec.is_periodic_lon))
    eff_lbc_rim = max(2, min(14, spec.nx // 4)) if has_parent_lbc else 0
    eff_lbc_tau = min(0.015, 0.25 / max(float(dt_s), 1.0)) if has_parent_lbc else 0.0
    prov_attrs = {
        "git_commit": _get_git_commit(),
        "nz": str(int(spec.nz)),
        "dt_s": f"{float(dt_s):.3f}",
        "physics_every": str(int(physics_every)),
        "cy46_physics": "true" if bool(cy46_physics) else "false",
        "pbl_last": "true" if bool(pbl_last) else "false",
        "enable_ocean_si3": "true" if bool(enable_ocean_si3) else "false",
        "lbc_rim_width": str(int(eff_lbc_rim)),
        "lbc_tau_max_s": f"{float(eff_lbc_tau):.5f}",
    }

    solver = NonHydrostaticGPUSolver(
        spec.vertical,
        dx_m=spec.dx_m,
        dy_m=spec.dy_m,
        hs_m=hs_m,
        msf=spec.map_factor,
        f_cor=spec.coriolis_f,
        has_moist=True,
        has_physics=enable_physics,
        bc_x=bc_x_val,
        bc_y=bc_y_val,
        use_coriolis=True,
        has_iau=(iau_forcing is not None),
        is_spherical=use_spherical,
        polar_filter_lat_deg=60.0 if use_spherical else 0.0,
        beta_offcenter=0.55,
        div_damp_coeff=0.04,
        hyperdiff_coeff=0.005,
        sponge_depth_m=min(6000.0, 0.22 * spec.vertical.z_top_m),
        sponge_tau_max_s=0.01,
        lbc_rim_width=eff_lbc_rim,
        lbc_tau_max_s=eff_lbc_tau,
        theta_sfc_k=theta_sfc_ref,
        p_sfc_pa=p_sfc_ref,
        n_ref_s=n_ref_hz,
    )

    out_paths: List[Path] = []
    lbc_paths: List[Path] = []
    cycle_restart_paths: List[Path] = []
    t0_wall = time.perf_counter()
    pbar = None

    try:
        cur_state = state_3d
        cur_surf = dict(surf_state)
        cur_oce_state: Optional[Dict[str, np.ndarray]] = None

        if ocean_solver is not None:
            t_oce_init = time.perf_counter()
            loaded_oce = False
            if ocean_restart_in is not None and Path(ocean_restart_in).is_file():
                logger.info("[%s] Loading coupled NEMO+SI3 restart: %s", spec.name, ocean_restart_in)
                cur_oce_state, _ = NemoSi3GPUSolver.load_restart_wx1(ocean_restart_in)
                loaded_oce = "temp_k" in cur_oce_state
                if not loaded_oce:
                    logger.warning("[%s] Ocean restart %s missing 'temp_k'; initializing ocean+SI3 cleanly.", spec.name, ocean_restart_in)
            if not loaded_oce:
                logger.info("[%s] Initializing coupled NEMO 5.0.2 + SI3 (%d z-levels)...", spec.name, int(ocean_nz))
                cur_oce_state = ocean_solver.init_from_atmosphere_and_osisaf(
                    cur_surf, osisaf_wx1_path=osisaf_wx1_path, obs_dir=obs_dir
                )
            ocean_solver.upload_state(cur_oce_state)
            cur_surf = ocean_solver.couple_to_surface_state(cur_oce_state, cur_surf)
            logger.info("[%s] Ocean+SI3 ready in %.2fs", spec.name, time.perf_counter() - t_oce_init)

        use_zero_copy = (spec.nz * spec.ny * spec.nx >= 20_000_000) and len(solver._tiles) > 1
        solver.upload_state(cur_state, copy=not use_zero_copy)
        solver.set_surface_static(eff_static)
        solver.upload_surface(cur_surf)
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
            pbl_last=1 if pbl_last else 0,
            cy46_physics=1 if cy46_physics else 0,
        )
        if has_parent_lbc:
            if lbc_interpolator is not None:
                init_lbc = _lbc_dict_to_staggered(lbc_interpolator.get_lbc_at_time(0.0), state_3d)
            else:
                init_lbc = {k: state_3d[k].copy() for k in ("u", "v", "w", "theta_p", "pi_p", "qv") if k in state_3d}
            solver.set_lbc_target(init_lbc)

        iau_active = iau_forcing is not None and iau_duration_s > 0.0
        if iau_active:
            solver.set_iau_forcing(iau_forcing)  # type: ignore[arg-type]

        # Build sorted breakpoint times (outputs, LBC exports, cycle restarts, ocean coupling, IAU, LBC refreshes)
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

        cpl_step_set: set[int] = set()
        if ocean_solver is not None and total_steps > 0:
            cpl_stride = max(1, int(round(max(float(dt_s), float(ocean_coupling_interval_s)) / float(dt_s))))
            for s_idx in range(0, total_steps + 1, cpl_stride):
                breaks.add(s_idx)
                cpl_step_set.add(s_idx)

        iau_off_step = int(round(float(iau_duration_s) / float(dt_s))) if iau_active else -1
        if 0 < iau_off_step < total_steps:
            breaks.add(iau_off_step)

        if lbc_interpolator is not None and total_steps > 0:
            lbc_stride_in = max(1, int(round(float(lbc_update_interval_s) / float(dt_s))))
            for s_idx in range(0, total_steps + 1, lbc_stride_in):
                breaks.add(s_idx)

        ordered_breaks = sorted(breaks)
        prev_oce_step = 0
        prev_oce_rain = np.asarray(cur_surf.get("rain_acc", np.zeros((spec.ny, spec.nx))), dtype=np.float32).copy()
        prev_oce_snow = (
            np.asarray(cur_surf.get("snow_acc", np.zeros((spec.ny, spec.nx))), dtype=np.float32)
            + np.asarray(cur_surf.get("graupel_acc", np.zeros((spec.ny, spec.nx))), dtype=np.float32)
        ).copy()
        prev_out_step = 0
        prev_out_precip = prev_oce_rain.astype(np.float64) + prev_oce_snow.astype(np.float64)

        logger.info(
            "[%s] Starting forecast | grid=%dx%dx%d | tiles=%d (shared=%s) | dt=%.1fs | steps=%d (%.2fh) | physics=%s (every=%d) | ocean_si3=%s | outputs=%d",
            spec.name,
            spec.nz,
            spec.ny,
            spec.nx,
            solver.num_tiles,
            bool(getattr(solver, "_shared_tile_handle", None) is not None or getattr(solver, "_tile_handles", None)),
            float(dt_s),
            total_steps,
            float(forecast_hours),
            "on" if enable_physics else "off",
            int(physics_every),
            "on" if ocean_solver is not None else "off",
            len(out_step_map),
        )
        pbar = (
            _tqdm(total=total_steps, desc=f"[{spec.name}]", unit="step", dynamic_ncols=True, mininterval=0.2)
            if (show_progress and _tqdm is not None and total_steps > 0)
            else None
        )
        cur_step = [0]
        step_t0 = [time.perf_counter()]

        def _on_tile(t_idx: int, n_tiles: int) -> None:
            if pbar is not None:
                next_h = ((cur_step[0] + 1) * float(dt_s)) / 3600.0
                pbar.set_postfix_str(
                    f"T+{next_h:05.2f}h/{float(forecast_hours):.1f}h | tile {t_idx}/{n_tiles}",
                    refresh=True,
                )

        def _on_step(_local_i: int, ran_phys: bool) -> None:
            step_elapsed = time.perf_counter() - step_t0[0]
            step_t0[0] = time.perf_counter()
            cur_step[0] += 1
            s_cur = cur_step[0]
            lead_h = (s_cur * float(dt_s)) / 3600.0
            pct = 100.0 * s_cur / max(1, total_steps)
            v_dt_step = init_dt + datetime.timedelta(seconds=s_cur * float(dt_s))
            mode_tag = "dyn+phys" if ran_phys else "dyn"
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix_str(
                    f"T+{lead_h:05.2f}h/{float(forecast_hours):.1f}h | {mode_tag} | {step_elapsed:.2f}s/step",
                    refresh=False,
                )
            if log_steps:
                logger.info(
                    "[%s] Step %d/%d (%.1f%%) | T+%05.2fh/%05.2fh | valid=%s | %s | %.2fs",
                    spec.name,
                    s_cur,
                    total_steps,
                    pct,
                    lead_h,
                    float(forecast_hours),
                    v_dt_step.strftime("%Y-%m-%d %H:%MZ"),
                    mode_tag,
                    step_elapsed,
                )
            if mem_profile_every > 0 and (s_cur == 1 or s_cur % mem_profile_every == 0 or s_cur == total_steps):
                st_m = _sample_mem_gpu_stats()
                logger.info(
                    "[%s] [MEM] step=%d/%d | rss=%.1fMB | swap=%.1fMB | vram=%d/%dMiB | np_arrays=%d | gpu=%.1fC %dMHz %.1fW thr=%s | dt_step=%.2fs",
                    spec.name,
                    s_cur,
                    total_steps,
                    st_m["rss_mb"],
                    st_m["swap_mb"],
                    st_m["vram_used_mib"],
                    st_m["vram_total_mib"],
                    st_m["np_arrays"],
                    st_m["gpu_temp_c"],
                    st_m["gpu_clk_mhz"],
                    st_m["gpu_pwr_w"],
                    st_m["gpu_thr"],
                    step_elapsed,
                )

        def _handle_step_events(step_idx: int) -> None:
            nonlocal cur_state, cur_surf, cur_oce_state, iau_active
            nonlocal prev_oce_step, prev_oce_rain, prev_oce_snow, prev_out_step, prev_out_precip
            sim_sec = step_idx * float(dt_s)
            v_dt = init_dt + datetime.timedelta(seconds=sim_sec)

            if step_idx > 0 and (
                step_idx in out_step_map
                or step_idx in child_step_map
                or step_idx in cycle_restart_step_map
                or step_idx in cpl_step_set
                or step_idx == total_steps
            ):
                cur_state = solver.download_state(copy=not use_zero_copy)
                cur_surf = solver.download_surface()

            if (
                ocean_solver is not None
                and step_idx > prev_oce_step
                and (step_idx in cpl_step_set or step_idx in out_step_map or step_idx == total_steps)
            ):
                dt_cpl = (step_idx - prev_oce_step) * float(dt_s)
                t_cpl0 = time.perf_counter()
                logger.info(
                    "[%s] Coupling NEMO+SI3 at T+%05.2fh (dt_cpl=%.0fs)...",
                    spec.name,
                    sim_sec / 3600.0,
                    dt_cpl,
                )
                mslp_pa = compute_mslp_pa(
                    spec, hs_m, cur_state, theta_sfc_ref=theta_sfc_ref, p_sfc_ref=p_sfc_ref, n_ref_hz=n_ref_hz
                )
                forcing = ocean_solver.build_forcing_from_atmosphere(
                    spec,
                    cur_state,
                    cur_surf,
                    mslp_pa=mslp_pa,
                    prev_rain=prev_oce_rain,
                    prev_snow=prev_oce_snow,
                    dt_interval_s=dt_cpl,
                )
                ocean_solver.step(forcing, dt_cpl)
                cur_oce_state = ocean_solver.download_state()
                cur_surf = ocean_solver.couple_to_surface_state(cur_oce_state, cur_surf)
                if step_idx < total_steps:
                    solver.upload_surface(cur_surf)
                prev_oce_rain = np.asarray(cur_surf.get("rain_acc", prev_oce_rain), dtype=np.float32).copy()
                prev_oce_snow = (
                    np.asarray(cur_surf.get("snow_acc", np.zeros_like(prev_oce_snow)), dtype=np.float32)
                    + np.asarray(cur_surf.get("graupel_acc", np.zeros_like(prev_oce_snow)), dtype=np.float32)
                ).copy()
                prev_oce_step = step_idx
                logger.info("[%s] Ocean+SI3 coupled in %.2fs", spec.name, time.perf_counter() - t_cpl0)

            if step_idx in child_step_map and lbc_export_child is not None:
                c_spec, c_hs, lbc_dir_c, _ = lbc_export_child
                t_lbc0 = time.perf_counter()
                logger.info(
                    "[%s] Exporting LBC step %d for %s at T+%05.2fh...",
                    spec.name,
                    child_step_map[step_idx],
                    c_spec.name,
                    sim_sec / 3600.0,
                )
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
                logger.info("[%s] Saved LBC %s (%.2fs)", spec.name, lp.name, time.perf_counter() - t_lbc0)

            if step_idx in out_step_map and out_dir is not None:
                hr = out_step_map[step_idx]
                tag = f"{int(round(hr)):03d}" if abs(hr - round(hr)) < 1e-4 else f"{hr:06.2f}"
                fp = out_dir / f"forecast_{spec.name}_T+{tag}h.wx1"
                t_wx0 = time.perf_counter()
                logger.info("[%s] Writing WX-1 forecast %s (T+%sh)...", spec.name, fp.name, tag)
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
                    include_3d_diagnostics=include_3d_diagnostics,
                    ocean_diag=cur_oce_state,
                    prev_precip_total=prev_out_precip if step_idx > prev_out_step else None,
                    interval_s=max(0.0, (step_idx - prev_out_step) * float(dt_s)),
                    surface_static=eff_static,
                    provenance_attrs=prov_attrs,
                )
                out_paths.append(fp)
                logger.info("[%s] Wrote %s in %.2fs", spec.name, fp.name, time.perf_counter() - t_wx0)
                prev_out_precip = (
                    np.asarray(cur_surf.get("rain_acc", np.zeros((spec.ny, spec.nx))), dtype=np.float64)
                    + np.asarray(cur_surf.get("snow_acc", np.zeros((spec.ny, spec.nx))), dtype=np.float64)
                    + np.asarray(cur_surf.get("graupel_acc", np.zeros((spec.ny, spec.nx))), dtype=np.float64)
                )
                prev_out_step = step_idx

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
                if ocean_solver is not None and cur_oce_state is not None:
                    orp = ocean_solver.save_restart_wx1(
                        cyc_dir / f"{spec.name}_ocean_cycle_f{tag}_{v_dt:%Y%m%d_%H%M}Z.wx1",
                        v_dt,
                        cur_oce_state,
                        sim_time_sec=sim_sec,
                    )
                    logger.info("[%s] Saved cycle restarts: %s & %s", spec.name, rp.name, orp.name)
                else:
                    logger.info("[%s] Saved cycle restart: %s", spec.name, rp.name)

            if iau_active and step_idx >= iau_off_step and step_idx < total_steps:
                solver.clear_iau_forcing()
                iau_forcing = None
                iau_active = False
                gc.collect()
                logger.info("[%s] IAU warm-up completed at step %d (T+%05.2fh); IAU buffers freed", spec.name, step_idx, sim_sec / 3600.0)

        _handle_step_events(0)

        for idx in range(len(ordered_breaks) - 1):
            s0, s1 = ordered_breaks[idx], ordered_breaks[idx + 1]
            n_seg = s1 - s0
            if n_seg <= 0:
                continue
            if lbc_interpolator is not None:
                t_mid = 0.5 * (s0 + s1) * float(dt_s)
                solver.set_lbc_target(_lbc_dict_to_staggered(lbc_interpolator.get_lbc_at_time(t_mid), state_3d))

            step_t0[0] = time.perf_counter()
            if enable_physics:
                solver.step_full(
                    float(dt_s),
                    n_steps=n_seg,
                    physics_every=physics_every,
                    flush_physics=(s1 == total_steps),
                    on_step=_on_step,
                    on_tile=_on_tile,
                )
            else:
                solver.step_rk3(float(dt_s), n_steps=n_seg, on_step=_on_step, on_tile=_on_tile)
            _handle_step_events(s1)

        if total_steps == 0:
            cur_state = solver.download_state(copy=not use_zero_copy)
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
            logger.info("[%s] Saved final restart: %s", spec.name, restart_out)

        ocean_restart_path: Optional[Path] = None
        if ocean_restart_out is not None and ocean_solver is not None and cur_oce_state is not None:
            ocean_restart_path = ocean_solver.save_restart_wx1(
                ocean_restart_out,
                end_dt,
                cur_oce_state,
                sim_time_sec=total_steps * float(dt_s),
            )
            logger.info("[%s] Saved final ocean restart: %s", spec.name, ocean_restart_path)

        elapsed_s = time.perf_counter() - t0_wall
        logger.info(
            "[%s] Forecast finished: %d steps (%.2fh) in %.2fs | outputs=%d | lbcs=%d",
            spec.name,
            total_steps,
            float(forecast_hours),
            elapsed_s,
            len(out_paths),
            len(lbc_paths),
        )
        return {
            "state_3d": cur_state,
            "surf_state": cur_surf,
            "ocean_state": cur_oce_state,
            "valid_time_utc": end_dt,
            "elapsed_s": elapsed_s,
            "num_tiles": solver.num_tiles,
            "estimated_tile_vram_bytes": solver.estimated_tile_vram_bytes,
            "output_paths": out_paths,
            "lbc_paths": lbc_paths,
            "restart_path": restart_out,
            "ocean_restart_path": ocean_restart_path,
            "cycle_restart_paths": cycle_restart_paths,
        }
    finally:
        if pbar is not None:
            pbar.close()
        solver.close()
        if ocean_solver is not None:
            ocean_solver.close()
        gc.collect()


def _resolve_static(
    spec: DomainSpec,
    override: Optional[Tuple[np.ndarray, Dict[str, np.ndarray]]],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    if override is not None:
        return override
    return build_domain_static_fields(spec)


def _build_zero_state_for_spec(spec: DomainSpec) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    nz, ny, nx = spec.nz, spec.ny, spec.nx
    state_3d = {
        k: np.zeros(
            (nz + 1 if k == "w" else nz, ny + (1 if k == "v" else 0), nx + (1 if k == "u" else 0)),
            dtype=np.float32,
        )
        for k in ("u", "v", "w", "theta_p", "pi_p", "qv", "qc", "qr", "qi", "qs", "qg")
    }
    surf_state = {k: np.zeros((ny, nx), dtype=np.float32) for k in NonHydrostaticGPUSolver.SURFACE_2D_FIELDS}
    surf_state["soil_t"] = np.full((4, ny, nx), 285.0, dtype=np.float32)
    surf_state["soil_m"] = np.full((4, ny, nx), 0.28, dtype=np.float32)
    surf_state["sst_bulk"] = np.full((ny, nx), 285.0, dtype=np.float32)
    surf_state["sea_ice_tsfc"] = np.full((ny, nx), 271.35, dtype=np.float32)
    surf_state["sea_ice_albedo"] = np.full((ny, nx), 0.066, dtype=np.float32)
    return state_3d, surf_state


def _load_domain_ic(
    spec: DomainSpec,
    hs_m: np.ndarray,
    sstatic: Dict[str, np.ndarray],
    *,
    ic_source: str,
    restart_path: Optional[Path | str] = None,
    sfc_grib: Optional[Path | str] = None,
    pl_grib: Optional[Path | str] = None,
    aurora_wx1_path: Optional[Path | str] = None,
    allow_zero_restart_fallback: bool = False,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Optional[datetime.datetime]]:
    if ic_source == "restart":
        if restart_path is not None:
            st, sf, meta = load_tempest_restart_wx1(restart_path)
            dt = datetime.datetime.fromisoformat(meta["valid_time_utc"].replace("Z", "+00:00"))
            return st, sf, dt
        if sfc_grib is not None and pl_grib is not None:
            return ecmwf_grib_to_domain_state(sfc_grib, pl_grib, spec, hs_m, surface_static=sstatic)
        if allow_zero_restart_fallback:
            st, sf = _build_zero_state_for_spec(spec)
            return st, sf, None
    if ic_source == "aurora" and aurora_wx1_path is not None:
        st, sf = aurora_wx1_to_ic(aurora_wx1_path, spec, hs_m)
        dt = datetime.datetime.now(datetime.timezone.utc).replace(minute=0, second=0, microsecond=0)
        return st, sf, dt
    if sfc_grib is not None and pl_grib is not None:
        return ecmwf_grib_to_domain_state(sfc_grib, pl_grib, spec, hs_m, surface_static=sstatic)
    raise ValueError("Provide valid ECMWF GRIB2 paths, Aurora WX-1 path, or restart WX-1 path.")


def _run_d3_da(
    d3_spec: DomainSpec,
    st_d3: Dict[str, np.ndarray],
    sf_d3: Dict[str, np.ndarray],
    hs_d3: np.ndarray,
    ss_d3: Dict[str, np.ndarray],
    init_dt: datetime.datetime,
    obs_dir: Optional[Path | str],
    enable_da: bool,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Optional[Dict[str, np.ndarray]], Dict[str, Any]]:
    da_metrics: Dict[str, Any] = {}
    iau_d3 = None
    if enable_da and obs_dir is not None and Path(obs_dir).exists():
        logger.info("[%s] Loading WX-1 DA observations from %s...", d3_spec.name, obs_dir)
        obs_list, stn_rows = load_station_obs_wx1(obs_dir, d3_spec, valid_time_utc=init_dt)
        logger.info(
            "[%s] Loaded %d DA observations across %d surface stations",
            d3_spec.name,
            len(obs_list),
            len(stn_rows),
        )
        if obs_list:
            st_bg = st_d3
            da_metrics["verify_bg"] = verify_against_stations(d3_spec, st_bg, sf_d3, hs_d3, stn_rows)
            use_iau = not (d3_spec.is_spherical and d3_spec.nz * d3_spec.ny * d3_spec.nx >= 50_000_000)
            ana_st, sf_d3, iau_d3, var_res = run_domain_3dvar(
                d3_spec, st_bg, sf_d3, hs_d3, ss_d3["lsm"], obs_list, obs_dir=obs_dir, compute_iau=use_iau
            )
            da_metrics["var3d"] = {
                "j0": var_res.initial_cost_j0,
                "jf": var_res.final_cost_j,
                "iterations": var_res.iterations,
                "converged": var_res.converged,
            }
            da_metrics["verify_ana"] = verify_against_stations(d3_spec, ana_st, sf_d3, hs_d3, stn_rows)
            if iau_d3 is not None:
                # Bloom et al. (1996) IAU: start prognostic momentum/thermo from background so the increment is added once over the IAU window.
                init_st = dict(ana_st)
                for k_iau in ("u", "v", "theta_p", "pi_p"):
                    init_st[k_iau] = st_bg[k_iau]
                st_d3 = init_st
            else:
                st_d3 = ana_st
        else:
            logger.info("[%s] No in-domain DA observations found; skipping 3D-Var.", d3_spec.name)
    return st_d3, sf_d3, iau_d3, da_metrics


def _domain_manifest_entry(spec: DomainSpec, res: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "domain": spec.name,
        "elapsed_s": float(res["elapsed_s"]),
        "restart_path": str(res["restart_path"]) if res.get("restart_path") else None,
        "ocean_restart_path": str(res["ocean_restart_path"]) if res.get("ocean_restart_path") else None,
        "cycle_restart_paths": [str(p) for p in res.get("cycle_restart_paths", [])],
        "output_paths": [str(p) for p in res.get("output_paths", [])],
    }


def _extract_cf_map(paths: Sequence[Path]) -> Dict[float, Path]:
    cf_map: Dict[float, Path] = {}
    for p in paths:
        name = Path(p).name
        if "_T+" in name and name.endswith("h.wx1"):
            try:
                cf_map[float(name.split("_T+")[1][:-5])] = Path(p)
            except ValueError:
                pass
    return cf_map


def _run_pipeline_ensemble(
    targets: Dict[str, Tuple[DomainSpec, np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], float, int, Optional[LBCInterpolator], Optional[Dict[str, np.ndarray]], Dict[str, Any]]],
    *,
    ensemble_domain: str,
    ensemble_members: int,
    ensemble_lead_hours: Optional[float],
    ensemble_seed: Optional[int],
    forecast_hours: float,
    sched: List[float],
    init_dt: datetime.datetime,
    ens_dir: Path,
    show_progress: bool = True,
) -> Optional[Dict[str, Any]]:
    if int(ensemble_members) <= 0:
        return None
    from tempest_v2.ensemble import EnsembleConfig, run_ensemble_forecast

    key = ensemble_domain.strip().lower()
    if key not in targets:
        key = "d3"
    spec_e, hs_e, ss_e, st_e, sf_e, dt_e, phys_e, lbc_e, iau_e, res_e = targets[key]
    ens_hours = float(ensemble_lead_hours) if ensemble_lead_hours is not None else float(forecast_hours)
    ens_sched = sorted({float(hr) for hr in sched if float(hr) <= ens_hours + 1e-6} | {ens_hours})
    cf_map = _extract_cf_map(res_e.get("output_paths", []))
    ens_cfg = EnsembleConfig(
        n_members=int(ensemble_members),
        seed=int(ensemble_seed) if ensemble_seed is not None else int(init_dt.strftime("%Y%m%d%H")),
    )
    logger.info(
        "=== Ensemble Phase [%s]: %d members (CF + %d perturbed) for %.2fh ===",
        spec_e.name,
        int(ensemble_members),
        max(0, int(ensemble_members) - 1),
        ens_hours,
    )
    return run_ensemble_forecast(
        spec_e,
        hs_e,
        ss_e,
        st_e,
        sf_e,
        init_dt,
        forecast_hours=ens_hours,
        ens_dir=ens_dir,
        cfg=ens_cfg,
        dt_s=dt_e,
        physics_every=phys_e,
        lbc_interpolator=lbc_e,
        base_iau_forcing=iau_e,
        base_iau_duration_s=900.0 if iau_e is not None else 0.0,
        output_hours=ens_sched,
        existing_cf_outputs=cf_map if cf_map else None,
        show_progress=show_progress,
        log_steps=False,
    )


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
    dt_d3_s: float = 6.0,
    physics_every: int = 5,
    lbc_interval_s: float = 1800.0,
    enable_ocean_si3: bool = False,
    include_3d_diagnostics: bool = True,
    ocean_nz: int = 16,
    ocean_coupling_interval_s: float = 1800.0,
    osisaf_wx1_path: Optional[Path | str] = None,
    d2_ocean_restart_in: Optional[Path | str] = None,
    d3_ocean_restart_in: Optional[Path | str] = None,
    d2_static: Optional[Tuple[np.ndarray, Dict[str, np.ndarray]]] = None,
    d3_static: Optional[Tuple[np.ndarray, Dict[str, np.ndarray]]] = None,
    show_progress: bool = True,
    log_steps: bool = True,
    mem_profile_every: int = 0,
    pbl_last: bool = False,
    cy46_physics: bool = False,
) -> Dict[str, Any]:
    """Run operational D2 (7.5 km Europe) -> WX-1 LBC -> D3 (2.5 km Benelux+DE default) pipeline."""
    t0_pipe = time.perf_counter()
    logger.info("=== Tempest-NWP v2 Pipeline [D2 -> D3] | nz=%d | hours=%.2fh ===", int(nz), float(forecast_hours))
    d2_spec = d2_spec if d2_spec is not None else get_domain_spec("d2", nz=nz)
    d3_spec = d3_spec if d3_spec is not None else get_domain_spec("d3", nz=nz)
    hs_d2, ss_d2 = _resolve_static(d2_spec, d2_static)
    hs_d3, ss_d3 = _resolve_static(d3_spec, d3_static)

    # Sequential domain lifecycle: load D2 first and defer D3 until D2 finishes so two large 3D states never coexist in RAM.
    logger.info("=== Phase 1/2: Europe D2 (%dx%dx%d) IC -> DA -> Forecast ===", d2_spec.nz, d2_spec.ny, d2_spec.nx)
    t_ic2 = time.perf_counter()
    st_d2, sf_d2, init_dt = _load_domain_ic(
        d2_spec,
        hs_d2,
        ss_d2,
        ic_source=ic_source,
        restart_path=d2_restart_path,
        sfc_grib=sfc_grib,
        pl_grib=pl_grib,
        aurora_wx1_path=aurora_wx1_path,
        allow_zero_restart_fallback=True,
    )
    if init_dt is None and restart_path is not None:
        with wx1.open_dataset(restart_path) as rds:
            init_dt = datetime.datetime.fromisoformat(str(rds.attrs["valid_time_utc"]).replace("Z", "+00:00"))
    assert init_dt is not None
    logger.info("[d2] Initial conditions ready in %.2fs (init=%s)", time.perf_counter() - t_ic2, init_dt.isoformat())

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
        d2_ocean_out = (layout["ocean"] / f"restart_ocean_{d2_spec.name}.wx1") if enable_ocean_si3 else None
        d3_ocean_out = (layout["ocean"] / f"restart_ocean_{d3_spec.name}.wx1") if enable_ocean_si3 else None
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
        d2_ocean_out = None
        d3_ocean_out = (out_root / f"restart_ocean_{d3_spec.name}.wx1") if enable_ocean_si3 else None
        cyc_restart_dir = None
        manifest_path = None

    sched = forecast_output_schedule_hours(forecast_hours, ecmwf_schedule=ecmwf_schedule, hourly=True)
    ens_dom_key = ensemble_domain.strip().lower()

    st_d2, sf_d2, iau_d2, da_d2 = _run_d3_da(
        d2_spec, st_d2, sf_d2, hs_d2, ss_d2, init_dt, obs_dir, enable_da
    )
    gc.collect()

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
        iau_forcing=iau_d2,
        iau_duration_s=900.0 if iau_d2 is not None else 0.0,
        output_dir=d2_out_dir,
        output_hours=sched if d2_out_dir is not None else None,
        save_restart_path=d2_restart_out,
        save_cycle_restart_hours=cycle_restart_hours if cyc_restart_dir is not None else (),
        cycle_restart_dir=cyc_restart_dir,
        enable_ocean_si3=enable_ocean_si3,
        ocean_coupling_interval_s=ocean_coupling_interval_s,
        ocean_nz=ocean_nz,
        osisaf_wx1_path=osisaf_wx1_path,
        obs_dir=obs_dir,
        ocean_restart_in=d2_ocean_restart_in,
        ocean_restart_out=d2_ocean_out,
        include_3d_diagnostics=include_3d_diagnostics,
        show_progress=show_progress,
        log_steps=log_steps,
        mem_profile_every=mem_profile_every,
        pbl_last=pbl_last,
        cy46_physics=cy46_physics,
    )
    if not (int(ensemble_members) > 0 and ens_dom_key == "d2"):
        del iau_d2
        st_d2, sf_d2 = {}, {}
        iau_d2 = None
        if d2_spec.nz * d2_spec.ny * d2_spec.nx >= 10_000_000:
            res_d2["state_3d"] = {}
            res_d2["ocean_state"] = None
        gc.collect()
        logger.info("[d2] Freed host 3D state before loading D3.")

    logger.info("=== Phase 2/2: Benelux+DE D3 (%dx%dx%d) IC -> DA -> Forecast ===", d3_spec.nz, d3_spec.ny, d3_spec.nx)
    t_ic3 = time.perf_counter()
    st_d3, sf_d3, _ = _load_domain_ic(
        d3_spec,
        hs_d3,
        ss_d3,
        ic_source=ic_source,
        restart_path=restart_path,
        sfc_grib=sfc_grib,
        pl_grib=pl_grib,
        aurora_wx1_path=aurora_wx1_path,
    )
    clear_cfgrib_cache()
    gc.collect()
    logger.info("[d3] Initial conditions ready in %.2fs", time.perf_counter() - t_ic3)
    st_d3, sf_d3, iau_d3, da_metrics = _run_d3_da(
        d3_spec, st_d3, sf_d3, hs_d3, ss_d3, init_dt, obs_dir, enable_da
    )
    gc.collect()
    if da_d2:
        da_metrics["d2"] = da_d2

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
        enable_ocean_si3=enable_ocean_si3,
        ocean_coupling_interval_s=ocean_coupling_interval_s,
        ocean_nz=ocean_nz,
        osisaf_wx1_path=osisaf_wx1_path,
        obs_dir=obs_dir,
        ocean_restart_in=d3_ocean_restart_in,
        ocean_restart_out=d3_ocean_out,
        include_3d_diagnostics=include_3d_diagnostics,
        show_progress=show_progress,
        log_steps=log_steps,
        mem_profile_every=mem_profile_every,
        pbl_last=pbl_last,
        cy46_physics=cy46_physics,
    )

    if enable_da and obs_dir is not None and da_metrics.get("verify_bg"):
        _, stn_rows = load_station_obs_wx1(obs_dir, d3_spec, valid_time_utc=init_dt)
        da_metrics["verify_end"] = verify_against_stations(
            d3_spec, res_d3["state_3d"], res_d3["surf_state"], hs_d3, stn_rows
        )

    all_cyc_restarts = [str(p) for p in (res_d2.get("cycle_restart_paths", []) + res_d3.get("cycle_restart_paths", []))]
    ens_dir = layout["ensemble"] if layout is not None else (out_root / "ensemble")
    ens_summary = _run_pipeline_ensemble(
        {
            "d2": (d2_spec, hs_d2, ss_d2, st_d2, sf_d2, dt_d2_s, max(1, physics_every // 2), None, None, res_d2),
            "d3": (d3_spec, hs_d3, ss_d3, st_d3, sf_d3, dt_d3_s, physics_every, lbc_interp, iau_d3, res_d3),
        },
        ensemble_domain=ensemble_domain,
        ensemble_members=ensemble_members,
        ensemble_lead_hours=ensemble_lead_hours,
        ensemble_seed=ensemble_seed,
        forecast_hours=forecast_hours,
        sched=sched,
        init_dt=init_dt,
        ens_dir=ens_dir,
        show_progress=show_progress,
    )

    if layout is not None and manifest_path is not None:
        manifest = {
            "init_time_utc": init_dt.isoformat(),
            "ic_source": ic_source,
            "warm_start": bool(ic_source == "restart"),
            "forecast_hours": float(forecast_hours),
            "nz": int(nz),
            "include_3d_diagnostics": bool(include_3d_diagnostics),
            "ocean": {
                "enabled": bool(enable_ocean_si3),
                "nz": int(ocean_nz),
                "coupling_interval_s": float(ocean_coupling_interval_s),
                "d2_restart_path": str(res_d2["ocean_restart_path"]) if res_d2.get("ocean_restart_path") else None,
                "d3_restart_path": str(res_d3["ocean_restart_path"]) if res_d3.get("ocean_restart_path") else None,
            },
            "d2": _domain_manifest_entry(d2_spec, res_d2),
            "d3": _domain_manifest_entry(d3_spec, res_d3),
            "da": da_metrics,
            "ensemble": ens_summary,
            "cycle_restart_paths": all_cyc_restarts,
            "wall_time_s": float(time.perf_counter() - t0_pipe),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("=== Tempest-NWP v2 Pipeline [D2 -> D3] finished in %.2fs ===", float(time.perf_counter() - t0_pipe))

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


def run_three_nest_pipeline(
    *,
    forecast_hours: float = 24.0,
    d1_forecast_hours: Optional[float] = None,
    nz: int = 65,
    ic_source: str = "ecmwf",
    sfc_grib: Optional[Path | str] = None,
    pl_grib: Optional[Path | str] = None,
    aurora_wx1_path: Optional[Path | str] = None,
    restart_path: Optional[Path | str] = None,
    d2_restart_path: Optional[Path | str] = None,
    d1_restart_path: Optional[Path | str] = None,
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
    d1_spec: Optional[DomainSpec] = None,
    d2_spec: Optional[DomainSpec] = None,
    d3_spec: Optional[DomainSpec] = None,
    ecmwf_schedule: bool = False,
    enable_da: bool = True,
    dt_d1_s: float = 60.0,
    dt_d2_s: float = 20.0,
    dt_d3_s: float = 6.0,
    physics_every: int = 5,
    d1_lbc_interval_s: float = 3600.0,
    lbc_interval_s: float = 1800.0,
    enable_ocean_si3: bool = True,
    include_3d_diagnostics: bool = True,
    ocean_nz: int = 16,
    ocean_coupling_interval_s: float = 1800.0,
    osisaf_wx1_path: Optional[Path | str] = None,
    d1_ocean_restart_in: Optional[Path | str] = None,
    d2_ocean_restart_in: Optional[Path | str] = None,
    d3_ocean_restart_in: Optional[Path | str] = None,
    d1_static: Optional[Tuple[np.ndarray, Dict[str, np.ndarray]]] = None,
    d2_static: Optional[Tuple[np.ndarray, Dict[str, np.ndarray]]] = None,
    d3_static: Optional[Tuple[np.ndarray, Dict[str, np.ndarray]]] = None,
    show_progress: bool = True,
    log_steps: bool = True,
    mem_profile_every: int = 0,
    pbl_last: bool = False,
    cy46_physics: bool = False,
) -> Dict[str, Any]:
    """Run coupled global D1 (0.25 deg) -> regional D2 (7.5 km) -> D3 (2.5 km Benelux+DE default) pipeline with NEMO+SI3 & 3D diagnostics."""
    t0_pipe = time.perf_counter()
    d1_hrs = max(float(forecast_hours), float(d1_forecast_hours)) if d1_forecast_hours is not None else float(forecast_hours)
    logger.info(
        "=== Tempest-NWP v2 Pipeline [D1 -> D2 -> D3] | nz=%d | d1=%.2fh | d2/d3=%.2fh | da=%s | ocean_si3=%s ===",
        int(nz),
        d1_hrs,
        float(forecast_hours),
        enable_da,
        enable_ocean_si3,
    )
    t_prep = time.perf_counter()
    d1_spec = d1_spec if d1_spec is not None else get_domain_spec("d1", nz=nz)
    d2_spec = d2_spec if d2_spec is not None else get_domain_spec("d2", nz=nz)
    d3_spec = d3_spec if d3_spec is not None else get_domain_spec("d3", nz=nz)
    hs_d1, ss_d1 = _resolve_static(d1_spec, d1_static)
    hs_d2, ss_d2 = _resolve_static(d2_spec, d2_static)
    hs_d3, ss_d3 = _resolve_static(d3_spec, d3_static)
    logger.info("Static terrain/soil/land-use fields ready for D1/D2/D3 in %.2fs", time.perf_counter() - t_prep)

    # Load D1 first (sequential domain lifecycle so D1, D2, D3 137-layer states never coexist in RAM)
    logger.info("=== Phase 1/3: Global D1 (%dx%dx%d) IC -> DA -> NEMO+SI3 Forecast ===", d1_spec.nz, d1_spec.ny, d1_spec.nx)
    t_ic1 = time.perf_counter()
    st_d1, sf_d1, init_dt = _load_domain_ic(
        d1_spec,
        hs_d1,
        ss_d1,
        ic_source=ic_source,
        restart_path=d1_restart_path,
        sfc_grib=sfc_grib,
        pl_grib=pl_grib,
        aurora_wx1_path=aurora_wx1_path,
        allow_zero_restart_fallback=True,
    )
    if init_dt is None and restart_path is not None:
        with wx1.open_dataset(restart_path) as rds:
            init_dt = datetime.datetime.fromisoformat(str(rds.attrs["valid_time_utc"]).replace("Z", "+00:00"))
    assert init_dt is not None
    logger.info("[d1] Initial conditions ready in %.2fs (init=%s)", time.perf_counter() - t_ic1, init_dt.isoformat())

    if run_dir is not None:
        layout = _ensure_run_layout_dirs(Path(run_dir))
    elif runs_root is not None:
        layout = create_run_layout(runs_root, init_dt, run_tag=run_tag)
    else:
        layout = create_run_layout(Path(outdir), init_dt, run_tag=run_tag)

    out_root = layout["run_dir"]
    logger.info("Run output directory: %s", out_root)
    lbc_d2_dir = layout["d1"] / "lbc_d2"
    lbc_d3_dir = layout["d2"] / "lbc_d3"
    cyc_dir = layout["restarts"]
    d1_ocean_out = (layout["ocean"] / f"restart_ocean_{d1_spec.name}.wx1") if enable_ocean_si3 else None
    d2_ocean_out = (layout["ocean"] / f"restart_ocean_{d2_spec.name}.wx1") if enable_ocean_si3 else None
    d3_ocean_out = (layout["ocean"] / f"restart_ocean_{d3_spec.name}.wx1") if enable_ocean_si3 else None

    sched_d1 = forecast_output_schedule_hours(d1_hrs, ecmwf_schedule=ecmwf_schedule, hourly=True)
    sched = forecast_output_schedule_hours(forecast_hours, ecmwf_schedule=ecmwf_schedule, hourly=True)
    ens_dom_key = ensemble_domain.strip().lower()

    # 1. Global D1 -> balanced 3D-Var/4D-FGAT + coupled NEMO+SI3 -> LBCs for Europe D2
    st_d1, sf_d1, iau_d1, da_d1 = _run_d3_da(
        d1_spec, st_d1, sf_d1, hs_d1, ss_d1, init_dt, obs_dir, enable_da
    )
    res_d1 = run_domain_forecast(
        d1_spec,
        hs_d1,
        ss_d1,
        st_d1,
        sf_d1,
        init_dt,
        forecast_hours=d1_hrs,
        dt_s=dt_d1_s,
        physics_every=max(1, physics_every // 4),
        lbc_export_child=(d2_spec, hs_d2, lbc_d2_dir, d1_lbc_interval_s),
        iau_forcing=iau_d1,
        iau_duration_s=900.0 if iau_d1 is not None else 0.0,
        output_dir=layout["d1"],
        output_hours=sched_d1,
        save_restart_path=cyc_dir / f"restart_{d1_spec.name}.wx1",
        save_cycle_restart_hours=cycle_restart_hours,
        cycle_restart_dir=cyc_dir,
        enable_ocean_si3=enable_ocean_si3,
        ocean_coupling_interval_s=ocean_coupling_interval_s,
        ocean_nz=ocean_nz,
        osisaf_wx1_path=osisaf_wx1_path,
        obs_dir=obs_dir,
        ocean_restart_in=d1_ocean_restart_in,
        ocean_restart_out=d1_ocean_out,
        include_3d_diagnostics=include_3d_diagnostics,
        show_progress=show_progress,
        log_steps=log_steps,
        mem_profile_every=mem_profile_every,
        pbl_last=pbl_last,
        cy46_physics=cy46_physics,
    )
    del iau_d1
    if not (int(ensemble_members) > 0 and ens_dom_key == "d1"):
        st_d1, sf_d1 = {}, {}
        if d1_spec.nz * d1_spec.ny * d1_spec.nx >= 10_000_000:
            res_d1["state_3d"] = {}
            res_d1["ocean_state"] = None
        gc.collect()
        logger.info("[d1] Freed host 3D state before loading D2.")

    # 2. Regional Europe D2 driven by D1 LBCs + balanced 3D-Var/4D-FGAT -> LBCs for Benelux+DE D3
    logger.info("=== Phase 2/3: Europe D2 (%dx%dx%d) IC -> DA -> Forecast ===", d2_spec.nz, d2_spec.ny, d2_spec.nx)
    t_ic2 = time.perf_counter()
    st_d2, sf_d2, _ = _load_domain_ic(
        d2_spec,
        hs_d2,
        ss_d2,
        ic_source=ic_source,
        restart_path=d2_restart_path,
        sfc_grib=sfc_grib,
        pl_grib=pl_grib,
        aurora_wx1_path=aurora_wx1_path,
        allow_zero_restart_fallback=True,
    )
    logger.info("[d2] Initial conditions ready in %.2fs", time.perf_counter() - t_ic2)
    st_d2, sf_d2, iau_d2, da_d2 = _run_d3_da(
        d2_spec, st_d2, sf_d2, hs_d2, ss_d2, init_dt, obs_dir, enable_da
    )
    lbc_d2_interp = LBCInterpolator(lbc_d2_dir, d2_spec.name)
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
        lbc_interpolator=lbc_d2_interp,
        lbc_export_child=(d3_spec, hs_d3, lbc_d3_dir, lbc_interval_s),
        iau_forcing=iau_d2,
        iau_duration_s=900.0 if iau_d2 is not None else 0.0,
        output_dir=layout["d2"],
        output_hours=sched,
        save_restart_path=cyc_dir / f"restart_{d2_spec.name}.wx1",
        save_cycle_restart_hours=cycle_restart_hours,
        cycle_restart_dir=cyc_dir,
        enable_ocean_si3=enable_ocean_si3,
        ocean_coupling_interval_s=ocean_coupling_interval_s,
        ocean_nz=ocean_nz,
        osisaf_wx1_path=osisaf_wx1_path,
        obs_dir=obs_dir,
        ocean_restart_in=d2_ocean_restart_in,
        ocean_restart_out=d2_ocean_out,
        include_3d_diagnostics=include_3d_diagnostics,
        show_progress=show_progress,
        log_steps=log_steps,
        mem_profile_every=mem_profile_every,
        pbl_last=pbl_last,
        cy46_physics=cy46_physics,
    )
    del iau_d2
    if not (int(ensemble_members) > 0 and ens_dom_key == "d2"):
        st_d2, sf_d2 = {}, {}
        if d2_spec.nz * d2_spec.ny * d2_spec.nx >= 10_000_000:
            res_d2["state_3d"] = {}
            res_d2["ocean_state"] = None
        gc.collect()
        logger.info("[d2] Freed host 3D state before loading D3.")

    # 3. Balanced 3D-Var/4D-FGAT + Radar LHN + IAU on 1.5 km D3
    logger.info("=== Phase 3/3: Benelux+DE D3 (%dx%dx%d) IC -> DA -> Forecast ===", d3_spec.nz, d3_spec.ny, d3_spec.nx)
    t_ic3 = time.perf_counter()
    st_d3, sf_d3, _ = _load_domain_ic(
        d3_spec,
        hs_d3,
        ss_d3,
        ic_source=ic_source,
        restart_path=restart_path,
        sfc_grib=sfc_grib,
        pl_grib=pl_grib,
        aurora_wx1_path=aurora_wx1_path,
    )
    clear_cfgrib_cache()
    gc.collect()
    logger.info("[d3] Initial conditions ready in %.2fs", time.perf_counter() - t_ic3)

    st_d3, sf_d3, iau_d3, da_metrics = _run_d3_da(
        d3_spec, st_d3, sf_d3, hs_d3, ss_d3, init_dt, obs_dir, enable_da
    )
    if da_d1:
        da_metrics["d1"] = da_d1
    if da_d2:
        da_metrics["d2"] = da_d2

    # 4. Benelux+Germany 1.5 km D3 CF driven by D2 LBCs & coupled coastal ocean/SI3
    lbc_d3_interp = LBCInterpolator(lbc_d3_dir, d3_spec.name)
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
        lbc_interpolator=lbc_d3_interp,
        iau_forcing=iau_d3,
        iau_duration_s=900.0 if iau_d3 is not None else 0.0,
        output_dir=layout["d3"],
        output_hours=sched,
        save_restart_path=cyc_dir / f"restart_{d3_spec.name}.wx1",
        save_cycle_restart_hours=cycle_restart_hours,
        cycle_restart_dir=cyc_dir,
        enable_ocean_si3=enable_ocean_si3,
        ocean_coupling_interval_s=ocean_coupling_interval_s,
        ocean_nz=ocean_nz,
        osisaf_wx1_path=osisaf_wx1_path,
        obs_dir=obs_dir,
        ocean_restart_in=d3_ocean_restart_in,
        ocean_restart_out=d3_ocean_out,
        include_3d_diagnostics=include_3d_diagnostics,
        show_progress=show_progress,
        log_steps=log_steps,
        mem_profile_every=mem_profile_every,
        pbl_last=pbl_last,
        cy46_physics=cy46_physics,
    )

    if enable_da and obs_dir is not None and da_metrics.get("verify_bg"):
        _, stn_rows = load_station_obs_wx1(obs_dir, d3_spec, valid_time_utc=init_dt)
        da_metrics["verify_end"] = verify_against_stations(
            d3_spec, res_d3["state_3d"], res_d3["surf_state"], hs_d3, stn_rows
        )

    cyc_list = (
        res_d1.get("cycle_restart_paths", [])
        + res_d2.get("cycle_restart_paths", [])
        + res_d3.get("cycle_restart_paths", [])
    )
    ens_summary = _run_pipeline_ensemble(
        {
            "d1": (d1_spec, hs_d1, ss_d1, st_d1, sf_d1, dt_d1_s, max(1, physics_every // 4), None, None, res_d1),
            "d2": (d2_spec, hs_d2, ss_d2, st_d2, sf_d2, dt_d2_s, max(1, physics_every // 2), lbc_d2_interp, None, res_d2),
            "d3": (d3_spec, hs_d3, ss_d3, st_d3, sf_d3, dt_d3_s, physics_every, lbc_d3_interp, iau_d3, res_d3),
        },
        ensemble_domain=ensemble_domain,
        ensemble_members=ensemble_members,
        ensemble_lead_hours=ensemble_lead_hours,
        ensemble_seed=ensemble_seed,
        forecast_hours=forecast_hours,
        sched=sched,
        init_dt=init_dt,
        ens_dir=layout["ensemble"],
        show_progress=show_progress,
    )

    manifest_path = out_root / "manifest.json"
    wall_total = float(time.perf_counter() - t0_pipe)
    manifest = {
        "git_commit": _get_git_commit(),
        "init_time_utc": init_dt.isoformat(),
        "ic_source": ic_source,
        "warm_start": bool(ic_source == "restart"),
        "forecast_hours": float(forecast_hours),
        "d1_forecast_hours": float(d1_hrs),
        "nz": int(nz),
        "cy46_physics": bool(cy46_physics),
        "pbl_last": bool(pbl_last),
        "include_3d_diagnostics": bool(include_3d_diagnostics),
        "ocean": {
            "enabled": bool(enable_ocean_si3),
            "nz": int(ocean_nz),
            "coupling_interval_s": float(ocean_coupling_interval_s),
            "d1_restart_path": str(res_d1["ocean_restart_path"]) if res_d1.get("ocean_restart_path") else None,
            "d2_restart_path": str(res_d2["ocean_restart_path"]) if res_d2.get("ocean_restart_path") else None,
            "d3_restart_path": str(res_d3["ocean_restart_path"]) if res_d3.get("ocean_restart_path") else None,
        },
        "d1": _domain_manifest_entry(d1_spec, res_d1),
        "d2": _domain_manifest_entry(d2_spec, res_d2),
        "d3": _domain_manifest_entry(d3_spec, res_d3),
        "da": da_metrics,
        "ensemble": ens_summary,
        "cycle_restart_paths": [str(p) for p in cyc_list],
        "wall_time_s": wall_total,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("=== Tempest-NWP v2 Pipeline [D1 -> D2 -> D3] finished in %.2fs | manifest=%s ===", wall_total, manifest_path)

    return {
        "d1": res_d1,
        "d2": res_d2,
        "d3": res_d3,
        "da": da_metrics,
        "run_dir": out_root,
        "layout": layout,
        "manifest_path": manifest_path,
        "cycle_restart_paths": cyc_list,
        "ensemble": ens_summary,
    }


def _infer_grib_init_utc(sfc_grib: Path | str, pl_grib: Optional[Path | str] = None) -> Optional[datetime.datetime]:
    sfc_p = Path(sfc_grib)
    cand_meta = Path(str(sfc_p).replace("_sfc_", "_").replace(".grib2", "_metadata.json"))
    if cand_meta.is_file():
        try:
            meta = json.loads(cand_meta.read_text(encoding="utf-8"))
            v_iso = meta.get("valid_time_utc")
            if v_iso:
                dt = datetime.datetime.fromisoformat(str(v_iso).replace("Z", "+00:00"))
                return dt.replace(tzinfo=datetime.timezone.utc) if dt.tzinfo is None else dt.astimezone(datetime.timezone.utc)
        except Exception:
            pass
    try:
        from tempest_v2.ic_builder import _read_cfgrib_var_map

        vmap = _read_cfgrib_var_map(Path(pl_grib) if pl_grib is not None else sfc_p, use_cache=False)
        for da, _la, _lo in vmap.values():
            if "valid_time" in da.coords:
                ts = int(np.datetime64(da.coords["valid_time"].values, "s").astype(np.int64))
                return datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
    except Exception:
        pass
    return None


def _execute_standalone_lam_run(
    lam_spec: DomainSpec,
    hs_lam: np.ndarray,
    ss_lam: Dict[str, np.ndarray],
    st_lam: Dict[str, np.ndarray],
    sf_lam: Dict[str, np.ndarray],
    init_dt: datetime.datetime,
    layout: Dict[str, Path],
    eff_lbc_dir: Path,
    lbc_meta: Dict[str, Any],
    *,
    mode: str,
    ic_source: str,
    t0_pipe: float,
    forecast_hours: float,
    dt_eff: float,
    phys_eff: int,
    obs_dir: Optional[Path | str],
    cycle_restart_hours: Sequence[float],
    ensemble_members: int,
    ensemble_lead_hours: Optional[float],
    ensemble_seed: Optional[int],
    ecmwf_schedule: bool,
    enable_da: bool,
    enable_ocean_si3: bool,
    include_3d_diagnostics: bool,
    ocean_nz: int,
    ocean_coupling_interval_s: float,
    osisaf_wx1_path: Optional[Path | str],
    ocean_restart_in: Optional[Path | str],
    show_progress: bool,
    log_steps: bool,
    mem_profile_every: int,
    pbl_last: bool,
    cy46_physics: bool,
    extra_manifest: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    out_root = layout["run_dir"]
    lam_out_dir = layout["d2"] if lam_spec.dx_m >= 4500.0 else layout["d3"]
    cyc_dir = layout["restarts"]

    st_lam, sf_lam, iau_lam, da_metrics = _run_d3_da(
        lam_spec, st_lam, sf_lam, hs_lam, ss_lam, init_dt, obs_dir, enable_da
    )
    gc.collect()

    lbc_interp = LBCInterpolator(eff_lbc_dir, lam_spec.name)
    sched = forecast_output_schedule_hours(forecast_hours, ecmwf_schedule=ecmwf_schedule, hourly=True)
    lam_ocean_out = (layout["ocean"] / f"restart_ocean_{lam_spec.name}.wx1") if enable_ocean_si3 else None

    res_lam = run_domain_forecast(
        lam_spec,
        hs_lam,
        ss_lam,
        st_lam,
        sf_lam,
        init_dt,
        forecast_hours=forecast_hours,
        dt_s=dt_eff,
        physics_every=phys_eff,
        lbc_interpolator=lbc_interp,
        iau_forcing=iau_lam,
        iau_duration_s=900.0 if iau_lam is not None else 0.0,
        output_dir=lam_out_dir,
        output_hours=sched,
        save_restart_path=cyc_dir / f"restart_{lam_spec.name}.wx1",
        save_cycle_restart_hours=cycle_restart_hours,
        cycle_restart_dir=cyc_dir,
        enable_ocean_si3=enable_ocean_si3,
        ocean_coupling_interval_s=ocean_coupling_interval_s,
        ocean_nz=ocean_nz,
        osisaf_wx1_path=osisaf_wx1_path,
        obs_dir=obs_dir,
        ocean_restart_in=ocean_restart_in,
        ocean_restart_out=lam_ocean_out,
        include_3d_diagnostics=include_3d_diagnostics,
        show_progress=show_progress,
        log_steps=log_steps,
        mem_profile_every=mem_profile_every,
        pbl_last=pbl_last,
        cy46_physics=cy46_physics,
    )

    if enable_da and obs_dir is not None and da_metrics.get("verify_bg"):
        _, stn_rows = load_station_obs_wx1(obs_dir, lam_spec, valid_time_utc=init_dt)
        da_metrics["verify_end"] = verify_against_stations(
            lam_spec, res_lam["state_3d"], res_lam["surf_state"], hs_lam, stn_rows
        )

    ens_summary = _run_pipeline_ensemble(
        {"d3": (lam_spec, hs_lam, ss_lam, st_lam, sf_lam, dt_eff, phys_eff, lbc_interp, iau_lam, res_lam)},
        ensemble_domain="d3",
        ensemble_members=ensemble_members,
        ensemble_lead_hours=ensemble_lead_hours,
        ensemble_seed=ensemble_seed,
        forecast_hours=forecast_hours,
        sched=sched,
        init_dt=init_dt,
        ens_dir=layout["ensemble"],
        show_progress=show_progress,
    )

    manifest_path = out_root / "manifest.json"
    wall_total = float(time.perf_counter() - t0_pipe)
    manifest: Dict[str, Any] = {
        "mode": mode,
        "git_commit": _get_git_commit(),
        "domain": lam_spec.name,
        "init_time_utc": init_dt.isoformat(),
        "ic_source": ic_source,
        "warm_start": bool(ic_source == "restart"),
        "forecast_hours": float(forecast_hours),
        "nz": int(lam_spec.nz),
        "dt_s": float(dt_eff),
        "physics_every": int(phys_eff),
        "cy46_physics": bool(cy46_physics),
        "pbl_last": bool(pbl_last),
        "include_3d_diagnostics": bool(include_3d_diagnostics),
        "lbc": lbc_meta,
        "ocean": {
            "enabled": bool(enable_ocean_si3),
            "nz": int(ocean_nz),
            "coupling_interval_s": float(ocean_coupling_interval_s),
            "restart_path": str(res_lam["ocean_restart_path"]) if res_lam.get("ocean_restart_path") else None,
        },
        "lam": _domain_manifest_entry(lam_spec, res_lam),
        "da": da_metrics,
        "ensemble": ens_summary,
        "cycle_restart_paths": [str(p) for p in res_lam.get("cycle_restart_paths", [])],
        "wall_time_s": wall_total,
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("=== Tempest-NWP v2 Pipeline [%s: %s] finished in %.2fs ===", mode, lam_spec.name, wall_total)

    return {
        "lam": res_lam,
        "d3": res_lam,
        "lbc": lbc_meta,
        "da": da_metrics,
        "run_dir": out_root,
        "layout": layout,
        "manifest_path": manifest_path,
        "cycle_restart_paths": res_lam.get("cycle_restart_paths", []),
        "ensemble": ens_summary,
    }


def run_era5_hindcast_lam_pipeline(
    init_time_utc: datetime.datetime,
    *,
    lam_domain: str = "lam_bnl_ns_de_2p5km",
    lam_spec: Optional[DomainSpec] = None,
    forecast_hours: float = 24.0,
    nz: int = 65,
    era5_land_soil_path: Optional[Path | str] = None,
    download_era5_land: bool = False,
    obs_dir: Optional[Path | str] = None,
    outdir: Path | str = "data/forecasts",
    runs_root: Optional[Path | str] = None,
    run_dir: Optional[Path | str] = None,
    run_tag: Optional[str] = None,
    cycle_restart_hours: Sequence[float] = (6.0,),
    ensemble_members: int = 0,
    ensemble_lead_hours: Optional[float] = None,
    ensemble_seed: Optional[int] = None,
    ecmwf_schedule: bool = False,
    enable_da: bool = False,
    dt_s: Optional[float] = None,
    physics_every: Optional[int] = None,
    lbc_dir: Optional[Path | str] = None,
    era5_cache_dir: Optional[Path | str] = None,
    lbc_interval_hours: float = 1.0,
    enable_ocean_si3: bool = True,
    include_3d_diagnostics: bool = True,
    ocean_nz: int = 16,
    ocean_coupling_interval_s: float = 1800.0,
    osisaf_wx1_path: Optional[Path | str] = None,
    ocean_restart_in: Optional[Path | str] = None,
    lam_static: Optional[Tuple[np.ndarray, Dict[str, np.ndarray]]] = None,
    show_progress: bool = True,
    log_steps: bool = True,
    mem_profile_every: int = 0,
    pbl_last: bool = False,
    cy46_physics: bool = False,
) -> Dict[str, Any]:
    """Run deterministic ARCO-ERA5 (+ optional ERA5-Land) hindcast LAM (0% AI / 0% ML)."""
    t0_pipe = time.perf_counter()
    lam_spec = lam_spec if lam_spec is not None else get_domain_spec(lam_domain, nz=nz)
    dt_eff = float(dt_s) if dt_s is not None else float(LAM_DOMAIN_DEFAULT_DT_S.get(lam_spec.name, 6.0))
    phys_eff = int(physics_every) if physics_every is not None else int(LAM_DOMAIN_DEFAULT_PHYSICS_EVERY.get(lam_spec.name, 5))

    init_dt = (
        init_time_utc.astimezone(datetime.timezone.utc)
        if init_time_utc.tzinfo is not None
        else init_time_utc.replace(tzinfo=datetime.timezone.utc)
    )

    logger.info(
        "=== Tempest-NWP v2 Pipeline [ERA5 Hindcast LAM: %s] | init=%s | nz=%d | hours=%.2fh | dt=%.1fs | phys_every=%d ===",
        lam_spec.name,
        init_dt.isoformat(),
        int(lam_spec.nz),
        float(forecast_hours),
        dt_eff,
        phys_eff,
    )
    hs_lam, ss_lam = _resolve_static(lam_spec, lam_static)

    if run_dir is not None:
        layout = _ensure_run_layout_dirs(Path(run_dir))
    elif runs_root is not None:
        layout = create_run_layout(runs_root, init_dt, run_tag=run_tag)
    else:
        layout = create_run_layout(Path(outdir), init_dt, run_tag=run_tag)

    out_root = layout["run_dir"]
    eff_lbc_dir = Path(lbc_dir) if lbc_dir is not None else (out_root / f"lbc_{lam_spec.name}")
    eff_cache_dir = (
        Path(era5_cache_dir)
        if era5_cache_dir is not None
        else ((Path(runs_root) / "era5_cache") if runs_root is not None else (out_root / "era5_cache"))
    )

    prep = prepare_era5_lbc_series(
        lam_spec,
        hs_lam,
        init_dt,
        forecast_hours,
        eff_lbc_dir,
        lbc_interval_hours=lbc_interval_hours,
        era5_cache_dir=eff_cache_dir,
    )
    lbc_meta = {
        "source": "arco_era5",
        "resol": "0p25",
        "lbc_interval_hours": float(lbc_interval_hours),
        "valid_times_utc": prep["valid_times_utc"],
        "lbc_dir": str(eff_lbc_dir),
        "era5_cache_dir": str(eff_cache_dir),
        "lbc_files": [str(p) for p in prep["lbc_paths"]],
    }

    eff_soil_path = Path(era5_land_soil_path) if era5_land_soil_path is not None else None
    if eff_soil_path is None and download_era5_land:
        eff_soil_path = download_era5_land_soil_cds(init_dt, lam_spec, eff_cache_dir)

    st_lam, sf_lam, init_dt = era5_wx1_to_domain_state(
        prep["ic_wx1"],
        lam_spec,
        hs_lam,
        era5_land_soil_path=eff_soil_path,
        surface_static=ss_lam,
    )
    gc.collect()

    return _execute_standalone_lam_run(
        lam_spec,
        hs_lam,
        ss_lam,
        st_lam,
        sf_lam,
        init_dt,
        layout,
        eff_lbc_dir,
        lbc_meta,
        mode="lam_era5_hindcast",
        ic_source="era5",
        t0_pipe=t0_pipe,
        forecast_hours=forecast_hours,
        dt_eff=dt_eff,
        phys_eff=phys_eff,
        obs_dir=obs_dir,
        cycle_restart_hours=cycle_restart_hours,
        ensemble_members=ensemble_members,
        ensemble_lead_hours=ensemble_lead_hours,
        ensemble_seed=ensemble_seed,
        ecmwf_schedule=ecmwf_schedule,
        enable_da=enable_da,
        enable_ocean_si3=enable_ocean_si3,
        include_3d_diagnostics=include_3d_diagnostics,
        ocean_nz=ocean_nz,
        ocean_coupling_interval_s=ocean_coupling_interval_s,
        osisaf_wx1_path=osisaf_wx1_path,
        ocean_restart_in=ocean_restart_in,
        show_progress=show_progress,
        log_steps=log_steps,
        mem_profile_every=mem_profile_every,
        pbl_last=pbl_last,
        cy46_physics=cy46_physics,
        extra_manifest={"era5_land_soil": str(eff_soil_path) if eff_soil_path is not None else None},
    )


def run_ecmwf_lbc_lam_pipeline(
    *,
    lam_domain: str = "lam_bnl_ns_de_2p5km",
    lam_spec: Optional[DomainSpec] = None,
    forecast_hours: float = 48.0,
    nz: int = 65,
    init_time_utc: Optional[datetime.datetime] = None,
    ic_source: str = "ecmwf",
    sfc_grib: Optional[Path | str] = None,
    pl_grib: Optional[Path | str] = None,
    aurora_wx1_path: Optional[Path | str] = None,
    restart_path: Optional[Path | str] = None,
    obs_dir: Optional[Path | str] = None,
    outdir: Path | str = "data/forecasts",
    runs_root: Optional[Path | str] = None,
    run_dir: Optional[Path | str] = None,
    run_tag: Optional[str] = None,
    cycle_restart_hours: Sequence[float] = (6.0,),
    ensemble_members: int = 0,
    ensemble_lead_hours: Optional[float] = None,
    ensemble_seed: Optional[int] = None,
    ecmwf_schedule: bool = False,
    enable_da: bool = True,
    dt_s: Optional[float] = None,
    physics_every: Optional[int] = None,
    lbc_dir: Optional[Path | str] = None,
    lbc_cache_dir: Optional[Path | str] = None,
    lbc_resol: str = "auto",
    lbc_interval_hours: float = 3.0,
    download_lbc: bool = True,
    ecmwf_source: str = "azure",
    enable_ocean_si3: bool = True,
    include_3d_diagnostics: bool = True,
    ocean_nz: int = 16,
    ocean_coupling_interval_s: float = 1800.0,
    osisaf_wx1_path: Optional[Path | str] = None,
    ocean_restart_in: Optional[Path | str] = None,
    lam_static: Optional[Tuple[np.ndarray, Dict[str, np.ndarray]]] = None,
    show_progress: bool = True,
    log_steps: bool = True,
    mem_profile_every: int = 0,
    pbl_last: bool = False,
    cy46_physics: bool = False,
) -> Dict[str, Any]:
    """Run KNMI-style standalone LAM (100% Tempest physics/dynamics/DA/NEMO+SI3) with shifted ECMWF LBCs."""
    t0_pipe = time.perf_counter()
    lam_spec = lam_spec if lam_spec is not None else get_domain_spec(lam_domain, nz=nz)
    dt_eff = float(dt_s) if dt_s is not None else float(LAM_DOMAIN_DEFAULT_DT_S.get(lam_spec.name, 6.0))
    phys_eff = int(physics_every) if physics_every is not None else int(LAM_DOMAIN_DEFAULT_PHYSICS_EVERY.get(lam_spec.name, 5))

    logger.info(
        "=== Tempest-NWP v2 Pipeline [LAM + ECMWF LBC: %s] | nz=%d | hours=%.2fh | dt=%.1fs | phys_every=%d | ocean_si3=%s ===",
        lam_spec.name,
        int(lam_spec.nz),
        float(forecast_hours),
        dt_eff,
        phys_eff,
        enable_ocean_si3,
    )
    t_prep = time.perf_counter()
    hs_lam, ss_lam = _resolve_static(lam_spec, lam_static)
    logger.info("[%s] Static terrain/soil/bathymetry ready in %.2fs", lam_spec.name, time.perf_counter() - t_prep)

    if init_time_utc is not None:
        init_dt = (
            init_time_utc.astimezone(datetime.timezone.utc)
            if init_time_utc.tzinfo is not None
            else init_time_utc.replace(tzinfo=datetime.timezone.utc)
        )
    elif ic_source == "restart" and restart_path is not None:
        with wx1.open_dataset(restart_path) as rds:
            raw_v = str(rds.attrs["valid_time_utc"])
        parsed = datetime.datetime.fromisoformat(raw_v.replace("Z", "+00:00"))
        init_dt = parsed.replace(tzinfo=datetime.timezone.utc) if parsed.tzinfo is None else parsed.astimezone(datetime.timezone.utc)
    elif sfc_grib is not None:
        inferred = _infer_grib_init_utc(sfc_grib, pl_grib)
        if inferred is not None:
            init_dt = inferred
        else:
            ref_utc = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=35)
            init_dt = ref_utc.replace(hour=(ref_utc.hour // 3) * 3, minute=0, second=0, microsecond=0)
    else:
        ref_utc = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=35)
        init_dt = ref_utc.replace(hour=(ref_utc.hour // 3) * 3, minute=0, second=0, microsecond=0)

    if run_dir is not None:
        layout = _ensure_run_layout_dirs(Path(run_dir))
    elif runs_root is not None:
        layout = create_run_layout(runs_root, init_dt, run_tag=run_tag)
    else:
        layout = create_run_layout(Path(outdir), init_dt, run_tag=run_tag)

    out_root = layout["run_dir"]
    lam_out_dir = layout["d2"] if lam_spec.dx_m >= 4500.0 else layout["d3"]
    cyc_dir = layout["restarts"]
    eff_lbc_dir = Path(lbc_dir) if lbc_dir is not None else (out_root / f"lbc_{lam_spec.name}")
    eff_cache_dir = (
        Path(lbc_cache_dir)
        if lbc_cache_dir is not None
        else ((Path(runs_root) / "ecmwf_lbc_cache") if runs_root is not None else (out_root / "raw_ecmwf"))
    )

    lbc_meta: Dict[str, Any] = {"lbc_dir": str(eff_lbc_dir), "resol": lbc_resol}
    existing_lbc = sorted(eff_lbc_dir.glob(f"lbc_{lam_spec.name}_step_*.wx1")) if eff_lbc_dir.is_dir() else []
    need_ic_grib = (ic_source == "ecmwf") and (sfc_grib is None or pl_grib is None)

    if existing_lbc and not need_ic_grib:
        logger.info("[%s] Reusing %d prebuilt LBC files in %s", lam_spec.name, len(existing_lbc), eff_lbc_dir)
        lbc_meta["source"] = "prebuilt"
        lbc_meta["lbc_files"] = [str(p) for p in existing_lbc]
    elif download_lbc:
        prep = prepare_ecmwf_lbc_series(
            lam_spec,
            hs_lam,
            init_dt,
            forecast_hours,
            eff_lbc_dir,
            resol=lbc_resol,
            lbc_interval_hours=lbc_interval_hours,
            source=ecmwf_source,
            grib_cache_dir=eff_cache_dir,
            need_ic_grib=need_ic_grib,
        )
        plan = prep["plan"]
        if need_ic_grib and prep.get("ic_sfc_grib") is not None and prep.get("ic_pl_grib") is not None:
            sfc_grib = prep["ic_sfc_grib"]
            pl_grib = prep["ic_pl_grib"]
        lbc_meta = {
            "source": "ecmwf_opendata",
            "resol": plan.resol,
            "ecmwf_cycle_utc": plan.ecmwf_cycle_utc.isoformat(),
            "lead_offset_hours": float(plan.lead_offset_hours),
            "ecmwf_steps": list(plan.ecmwf_steps),
            "ic_step": int(plan.ic_step),
            "lbc_dir": str(eff_lbc_dir),
            "lbc_files": [str(p) for p in prep["lbc_paths"]],
        }
    elif sfc_grib is not None and pl_grib is not None:
        end_dt_lbc = init_dt + datetime.timedelta(hours=float(forecast_hours))
        p0 = ecmwf_grib_to_lbc_wx1(
            sfc_grib, pl_grib, lam_spec, hs_lam, eff_lbc_dir, 0, 0.0, init_dt, resol=lbc_resol
        )
        p1 = ecmwf_grib_to_lbc_wx1(
            sfc_grib,
            pl_grib,
            lam_spec,
            hs_lam,
            eff_lbc_dir,
            1,
            max(1.0, float(forecast_hours) * 3600.0),
            end_dt_lbc,
            resol=lbc_resol,
        )
        lbc_meta = {
            "source": "local_grib_static",
            "resol": lbc_resol,
            "lbc_dir": str(eff_lbc_dir),
            "lbc_files": [str(p0), str(p1)],
        }
    else:
        raise ValueError("Provide prebuilt --lbc-dir, local --sfc-grib/--pl-grib, or enable LBC downloading.")

    t_ic = time.perf_counter()
    st_lam, sf_lam, ic_valid_dt = _load_domain_ic(
        lam_spec,
        hs_lam,
        ss_lam,
        ic_source=ic_source,
        restart_path=restart_path,
        sfc_grib=sfc_grib,
        pl_grib=pl_grib,
        aurora_wx1_path=aurora_wx1_path,
    )
    if ic_valid_dt is not None and init_time_utc is None and ic_source == "restart":
        init_dt = ic_valid_dt
    clear_cfgrib_cache()
    gc.collect()
    logger.info("[%s] Initial conditions ready in %.2fs (init=%s)", lam_spec.name, time.perf_counter() - t_ic, init_dt.isoformat())

    return _execute_standalone_lam_run(
        lam_spec,
        hs_lam,
        ss_lam,
        st_lam,
        sf_lam,
        init_dt,
        layout,
        eff_lbc_dir,
        lbc_meta,
        mode="lam_ecmwf_lbc",
        ic_source=ic_source,
        t0_pipe=t0_pipe,
        forecast_hours=forecast_hours,
        dt_eff=dt_eff,
        phys_eff=phys_eff,
        obs_dir=obs_dir,
        cycle_restart_hours=cycle_restart_hours,
        ensemble_members=ensemble_members,
        ensemble_lead_hours=ensemble_lead_hours,
        ensemble_seed=ensemble_seed,
        ecmwf_schedule=ecmwf_schedule,
        enable_da=enable_da,
        enable_ocean_si3=enable_ocean_si3,
        include_3d_diagnostics=include_3d_diagnostics,
        ocean_nz=ocean_nz,
        ocean_coupling_interval_s=ocean_coupling_interval_s,
        osisaf_wx1_path=osisaf_wx1_path,
        ocean_restart_in=ocean_restart_in,
        show_progress=show_progress,
        log_steps=log_steps,
        mem_profile_every=mem_profile_every,
        pbl_last=pbl_last,
        cy46_physics=cy46_physics,
    )
