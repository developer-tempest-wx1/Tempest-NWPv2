"""Closed-column compressible diabatic/moisture conservation and IC hydrostatic tests."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import pytest
import wx1

from tempest_v2.cuda_bridge import NonHydrostaticGPUSolver, cuda_vram_info
from tempest_v2.diagnostics import compute_pressure_level_fields
from tempest_v2.ic_builder import (
    _regrid_latlon_field,
    _vertical_map_isobaric_to_klemp,
    build_domain_static_fields,
)
from tempest_v2.ingest_era5 import era5_wx1_to_domain_state
from tempest_v2.projection import (
    CP_D,
    GRAVITY_M_S2,
    KAPPA,
    P0_PA,
    R_D,
    DomainSpec,
    LCCProjection,
    VerticalGrid,
    get_domain_spec,
    _make_lcc_domain,
)
from tempest_v2.verification import compute_mslp_pa

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ERA5_20190725 = _REPO_ROOT / "runs" / "era5_cache" / "era5_0p25_20190725_00Z.wx1"
_R_V = 461.5
_EPS_V = (_R_V / R_D) - 1.0
_CV_D = CP_D - R_D


def _require_cuda() -> None:
    try:
        free_b, _ = cuda_vram_info()
    except Exception as exc:
        pytest.skip(f"CUDA unavailable for GPU conservation test: {exc}")
    if free_b < 256 * 1024 * 1024:
        pytest.skip(f"Insufficient free CUDA VRAM ({free_b} B)")


def _make_column_spec(nz: int = 65) -> DomainSpec:
    vgrid = VerticalGrid(nz=nz, z_top_m=22_000.0, stretch_power=1.3)
    return _make_lcc_domain(
        "test_col_8x8",
        ny=8,
        nx=8,
        dx_m=2500.0,
        projection=LCCProjection(),
        vertical=vgrid,
        center_lat_deg=52.1,
        center_lon_deg=5.18,
    )


def _init_hydrostatic_column_state(
    spec: DomainSpec,
    hs_m: np.ndarray,
    *,
    theta_sfc_k: float,
    qv_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Build exact horizontally uniform `(nz, ny, nx)` state matching solver ref."""
    z_w, z_s, _ = spec.vertical.heights_3d(hs_m)
    ref = VerticalGrid.reference_state(
        z_s, theta_sfc_k=288.15, p_sfc_pa=101_325.0, brunt_vaisala_hz=0.01
    )
    ref_init = VerticalGrid.reference_state(
        z_s, theta_sfc_k=theta_sfc_k, p_sfc_pa=101_325.0, brunt_vaisala_hz=0.01
    )
    qv_64 = (
        np.zeros_like(z_s, dtype=np.float64)
        if qv_fn is None
        else np.maximum(0.0, np.asarray(qv_fn(z_s), dtype=np.float64))
    )
    theta_tot = ref_init["theta_bar"]
    theta_v = theta_tot * (1.0 + _EPS_V * qv_64)

    # Match the discrete hydrostatic vertical stencil in nh_core.cu.
    dzeta = spec.vertical.dzeta
    pi_tot = np.empty_like(z_s, dtype=np.float64)
    pi_sfc = (101_325.0 / P0_PA) ** KAPPA
    pi_tot[0] = pi_sfc - (GRAVITY_M_S2 * (z_s[0] - hs_m)) / (CP_D * theta_v[0])
    for k in range(1, spec.nz):
        wt = dzeta[k - 1] / (dzeta[k - 1] + dzeta[k])
        tv_iface = (1.0 - wt) * theta_v[k - 1] + wt * theta_v[k]
        dz_mid = z_s[k] - z_s[k - 1]
        pi_tot[k] = pi_tot[k - 1] - (GRAVITY_M_S2 * dz_mid) / (CP_D * tv_iface)

    nz, ny, nx = spec.nz, spec.ny, spec.nx
    shape3 = (nz, ny, nx)
    state: Dict[str, np.ndarray] = {
        "u": np.zeros((nz, ny, nx + 1), dtype=np.float32),
        "v": np.zeros((nz, ny + 1, nx), dtype=np.float32),
        "w": np.zeros((nz + 1, ny, nx), dtype=np.float32),
        "theta_p": np.broadcast_to(theta_tot - ref["theta_bar"], shape3).astype(np.float32).copy(),
        "pi_p": np.broadcast_to(pi_tot - ref["pi_bar"], shape3).astype(np.float32).copy(),
        "qv": np.broadcast_to(qv_64, shape3).astype(np.float32).copy(),
        "qc": np.zeros(shape3, dtype=np.float32),
        "qr": np.zeros(shape3, dtype=np.float32),
        "qi": np.zeros(shape3, dtype=np.float32),
        "qs": np.zeros(shape3, dtype=np.float32),
        "qg": np.zeros(shape3, dtype=np.float32),
        "tke": np.full(shape3, 0.01, dtype=np.float32),
    }
    _ = z_w
    return ref, state


def _column_budget(
    spec: DomainSpec,
    hs_m: np.ndarray,
    ref: Dict[str, np.ndarray],
    state: Dict[str, np.ndarray],
    surf: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """Evaluate closed-column dry, vapour, hydrometeor mass and diagnostic pressure."""
    theta = ref["theta_bar"] + state["theta_p"].astype(np.float64)
    pi = np.maximum(1e-5, ref["pi_bar"] + state["pi_p"].astype(np.float64))
    qv = state["qv"].astype(np.float64)
    qhyd = sum(state[sp].astype(np.float64) for sp in ("qc", "qr", "qi", "qs", "qg"))
    theta_v = theta * (1.0 + _EPS_V * qv)
    rho_g = (P0_PA / (R_D * theta_v)) * np.power(pi, _CV_D / R_D)

    z_w, _, _ = spec.vertical.heights_3d(hs_m)
    dz = np.diff(z_w, axis=0)
    to_hpa = GRAVITY_M_S2 / 100.0
    dry_hpa = float(np.mean(np.sum(rho_g * (1.0 - qv) * dz, axis=0)) * to_hpa)
    vap_hpa = float(np.mean(np.sum(rho_g * qv * dz, axis=0)) * to_hpa)
    hyd_hpa = float(np.mean(np.sum(rho_g * qhyd * dz, axis=0)) * to_hpa)
    tot_hpa = dry_hpa + vap_hpa + hyd_hpa
    pr_acc_hpa = float(
        np.mean(
            surf["rain_acc"].astype(np.float64)
            + surf["snow_acc"].astype(np.float64)
            + surf["graupel_acc"].astype(np.float64)
        )
        * to_hpa
    )
    mslp_hpa = float(np.mean(compute_mslp_pa(spec, hs_m, state)) / 100.0)
    p_top_hpa = float(
        np.mean(P0_PA * np.power(ref["pi_bar"][-1] + state["pi_p"][-1].astype(np.float64), 1.0 / KAPPA))
        / 100.0
    )
    gh500_m = float(
        np.mean(compute_pressure_level_fields(spec, hs_m, state, plev_pa=[50_000.0])["gh_pl"][0])
    )
    return {
        "dry_hpa": dry_hpa,
        "vap_hpa": vap_hpa,
        "hyd_hpa": hyd_hpa,
        "tot_hpa": tot_hpa,
        "pr_acc_hpa": pr_acc_hpa,
        "mslp_hpa": mslp_hpa,
        "p_top_hpa": p_top_hpa,
        "mslp_minus_ptop_hpa": mslp_hpa - p_top_hpa,
        "gh500_m": gh500_m,
    }


def _run_closed_column(
    *,
    duration_s: float,
    dt_s: float = 12.0,
    physics_every: int = 3,
    theta_sfc_k: float = 288.15,
    qv_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    t_skin_k: float = 286.0,
    soil_t_k: float = 285.0,
    enable_microphysics: int = 1,
    enable_pbl: int = 1,
    enable_radiation: int = 1,
    enable_surface: int = 1,
    julian_day: float = 172.0,
    utc_seconds: float = 36_000.0,
) -> Dict[str, float]:
    _require_cuda()
    spec = _make_column_spec(nz=65)
    hs_m = np.zeros((spec.ny, spec.nx), dtype=np.float64)
    ref, state0 = _init_hydrostatic_column_state(
        spec, hs_m, theta_sfc_k=theta_sfc_k, qv_fn=qv_fn
    )
    solver = NonHydrostaticGPUSolver(
        spec.vertical,
        spec.dx_m,
        spec.dy_m,
        hs_m,
        msf=spec.map_factor,
        f_cor=spec.coriolis_f,
        has_moist=True,
        has_physics=True,
        bc_x=1,
        bc_y=1,
        use_coriolis=False,
    )
    try:
        sstatic = solver.default_surface_static(
            lat_deg=52.1, lon_deg=5.18, lsm=1.0, soil_type=9.0
        )
        surf0 = solver.empty_surface_state()
        surf0["t_skin"].fill(t_skin_k)
        surf0["soil_t"].fill(soil_t_k)
        solver.set_surface_static(sstatic)
        solver.upload_surface(surf0)
        solver.upload_state(state0)
        solver.set_physics_config(
            enable_microphysics=enable_microphysics,
            enable_pbl=enable_pbl,
            enable_radiation=enable_radiation,
            enable_surface=enable_surface,
            rad_interval_s=900.0,
            julian_day=julian_day,
            utc_seconds=utc_seconds,
            cy46_physics=0,
            pbl_last=0,
        )
        b0 = _column_budget(spec, hs_m, ref, state0, surf0)
        n_steps = max(1, int(round(duration_s / dt_s)))
        solver.step_full(dt_s, n_steps, physics_every=physics_every, flush_physics=True)
        state1 = solver.download_state()
        surf1 = solver.download_surface()
        for name, arr in state1.items():
            assert np.all(np.isfinite(arr)), f"Non-finite values in {name}"
        b1 = _column_budget(spec, hs_m, ref, state1, surf1)
    finally:
        solver.close()

    return {
        f"d_{k}": b1[k] - b0[k]
        for k in (
            "dry_hpa",
            "vap_hpa",
            "hyd_hpa",
            "tot_hpa",
            "pr_acc_hpa",
            "mslp_hpa",
            "p_top_hpa",
            "mslp_minus_ptop_hpa",
            "gh500_m",
        )
    }


def _discrete_hydrostatic_w_acc(
    spec: DomainSpec, hs_m: np.ndarray, state: Dict[str, np.ndarray]
) -> np.ndarray:
    """Compute discrete vertical buoyancy-pressure residual at cell interfaces [m/s^2]."""
    _, z_s, _ = spec.vertical.heights_3d(hs_m)
    ref = VerticalGrid.reference_state(
        z_s, theta_sfc_k=288.15, p_sfc_pa=101_325.0, brunt_vaisala_hz=0.01
    )
    theta = ref["theta_bar"] + state["theta_p"].astype(np.float64)
    pi = ref["pi_bar"] + state["pi_p"].astype(np.float64)
    qv = state["qv"].astype(np.float64)
    theta_v = theta * (1.0 + _EPS_V * qv)
    dzeta = spec.vertical.dzeta[:, None, None]
    wt = dzeta[:-1] / (dzeta[:-1] + dzeta[1:])
    tv_iface = (1.0 - wt) * theta_v[:-1] + wt * theta_v[1:]
    dz_mid = z_s[1:] - z_s[:-1]
    return -(CP_D * tv_iface * (pi[1:] - pi[:-1]) / dz_mid + GRAVITY_M_S2)


def test_closed_column_radiation_and_daytime_pbl_conservation() -> None:
    """Day/night radiation and daytime Rad+Sfc+PBL preserve dry mass and hydrostatic weight."""
    moist_profile = lambda z: 0.008 * np.exp(-z / 2500.0)

    rad_day = _run_closed_column(
        duration_s=7200.0,
        qv_fn=moist_profile,
        enable_microphysics=0,
        enable_pbl=0,
        enable_radiation=1,
        enable_surface=0,
        utc_seconds=36_000.0,
    )
    assert abs(rad_day["d_dry_hpa"]) < 0.05
    assert abs(rad_day["d_tot_hpa"]) < 0.05
    assert abs(rad_day["d_mslp_minus_ptop_hpa"] - rad_day["d_tot_hpa"]) < 0.15

    rad_night = _run_closed_column(
        duration_s=7200.0,
        qv_fn=moist_profile,
        enable_microphysics=0,
        enable_pbl=0,
        enable_radiation=1,
        enable_surface=0,
        utc_seconds=72_000.0,
    )
    assert abs(rad_night["d_dry_hpa"]) < 0.05
    assert abs(rad_night["d_tot_hpa"]) < 0.05
    assert abs(rad_night["d_mslp_minus_ptop_hpa"] - rad_night["d_tot_hpa"]) < 0.15

    pbl_day = _run_closed_column(
        duration_s=7200.0,
        qv_fn=moist_profile,
        t_skin_k=288.15,
        soil_t_k=288.15,
        enable_microphysics=0,
        enable_pbl=1,
        enable_radiation=1,
        enable_surface=1,
        utc_seconds=36_000.0,
    )
    assert abs(pbl_day["d_dry_hpa"]) < 0.10
    assert abs(pbl_day["d_mslp_minus_ptop_hpa"] - pbl_day["d_tot_hpa"]) < 0.25
    assert pbl_day["d_gh500_m"] > 1.0


def test_closed_column_warm_and_cold_extremes_conservation() -> None:
    """Warm +15 K and Cold -15 K daytime Rad+Sfc+PBL runs remain conservative and finite."""
    moist_warm = lambda z: 0.008 * np.exp(-z / 2500.0)
    moist_cold = lambda z: 0.0025 * np.exp(-z / 2500.0)

    warm = _run_closed_column(
        duration_s=3600.0,
        theta_sfc_k=303.15,
        t_skin_k=303.15,
        soil_t_k=303.15,
        qv_fn=moist_warm,
        enable_microphysics=0,
        enable_pbl=1,
        enable_radiation=1,
        enable_surface=1,
        utc_seconds=36_000.0,
    )
    assert abs(warm["d_dry_hpa"]) < 0.10
    assert abs(warm["d_mslp_minus_ptop_hpa"] - warm["d_tot_hpa"]) < 0.25

    cold = _run_closed_column(
        duration_s=3600.0,
        theta_sfc_k=273.15,
        t_skin_k=273.15,
        soil_t_k=273.15,
        qv_fn=moist_cold,
        enable_microphysics=0,
        enable_pbl=1,
        enable_radiation=1,
        enable_surface=1,
        utc_seconds=36_000.0,
    )
    assert abs(cold["d_dry_hpa"]) < 0.10
    assert abs(cold["d_mslp_minus_ptop_hpa"] - cold["d_tot_hpa"]) < 0.25


def test_closed_column_microphysics_supersaturation_mass_conservation() -> None:
    """Supersaturated microphysics preserves dry mass and total mass + surface fallout."""
    # Supersaturate the lower-to-mid troposphere (0-3.5 km) so strong rainout
    # occurs within 0.5 h without triggering a global +50 K rigid-lid shockwave.
    supersat = _run_closed_column(
        duration_s=1800.0,
        qv_fn=lambda z: np.where(z < 3500.0, 0.014 * np.exp(-z / 3000.0), 0.001),
        enable_microphysics=1,
        enable_pbl=0,
        enable_radiation=0,
        enable_surface=0,
        utc_seconds=36_000.0,
    )
    assert supersat["d_pr_acc_hpa"] > 0.05
    assert abs(supersat["d_dry_hpa"]) < 0.05
    assert abs(supersat["d_tot_hpa"] + supersat["d_pr_acc_hpa"]) < 0.10
    assert abs(supersat["d_mslp_minus_ptop_hpa"] - supersat["d_tot_hpa"]) < 0.25


def test_ic_vertical_map_unsaturated_inversion_and_terrain_hydrostatic_balance() -> None:
    """Vertical IC mapping matches analytical unsaturated inversion, ps_child, and gh500."""
    vgrid = VerticalGrid(nz=65, z_top_m=22_000.0, stretch_power=1.3)
    spec = _make_lcc_domain(
        "test_ic_terrain",
        ny=16,
        nx=16,
        dx_m=2500.0,
        projection=LCCProjection(),
        vertical=vgrid,
    )
    yy, xx = np.meshgrid(
        np.arange(spec.ny, dtype=np.float64),
        np.arange(spec.nx, dtype=np.float64),
        indexing="ij",
    )
    hs_m = 800.0 * np.exp(-(((xx - 7.5) ** 2 + (yy - 7.5) ** 2) / 10.0))

    # Build exact piecewise-linear-in-z virtual temperature sounding with a low-level inversion.
    z_fine = np.linspace(0.0, 26_000.0, 2601, dtype=np.float64)
    q_const = 0.0005  # Stays far below 0.995 * qs across the profile.
    t_fine = np.where(
        z_fine <= 800.0,
        280.0 + 0.008 * z_fine,
        np.where(
            z_fine <= 11_000.0,
            286.4 - 0.0065 * (z_fine - 800.0),
            220.1,
        ),
    )
    tv_fine = t_fine * (1.0 + _EPS_V * q_const)
    mslp_true_pa = 101_500.0
    ln_p_fine = np.empty_like(z_fine)
    ln_p_fine[0] = np.log(mslp_true_pa)
    dz_f = np.diff(z_fine)
    inv_tv_mid = 0.5 * (1.0 / tv_fine[:-1] + 1.0 / tv_fine[1:])
    ln_p_fine[1:] = ln_p_fine[0] - np.cumsum((GRAVITY_M_S2 / R_D) * dz_f * inv_tv_mid)

    p_levels_pa = np.array(
        [
            100_000.0,
            97_500.0,
            95_000.0,
            92_500.0,
            90_000.0,
            85_000.0,
            80_000.0,
            75_000.0,
            70_000.0,
            65_000.0,
            60_000.0,
            55_000.0,
            50_000.0,
            45_000.0,
            40_000.0,
            35_000.0,
            30_000.0,
            25_000.0,
            20_000.0,
            15_000.0,
            10_000.0,
            5_000.0,
            2_000.0,
        ],
        dtype=np.float64,
    )
    ln_p_desc = ln_p_fine[::-1]
    z_p = np.interp(np.log(p_levels_pa), ln_p_desc, z_fine[::-1])
    t_p = np.interp(z_p, z_fine, t_fine)
    gh500_true_m = float(z_p[np.argmin(np.abs(p_levels_pa - 50_000.0))])

    n_lev = len(p_levels_pa)
    z_h = np.broadcast_to(z_p[:, None, None], (n_lev, spec.ny, spec.nx)).copy()
    t_h = np.broadcast_to(t_p[:, None, None], (n_lev, spec.ny, spec.nx)).copy()
    q_h = np.full((n_lev, spec.ny, spec.nx), q_const, dtype=np.float64)
    u_h = np.zeros_like(z_h)
    v_h = np.zeros_like(z_h)
    ps_child = np.exp(np.interp(hs_m, z_fine, ln_p_fine))

    state = _vertical_map_isobaric_to_klemp(
        spec,
        hs_m,
        p_levels_pa,
        z_h,
        t_h,
        u_h,
        v_h,
        q_h,
        ps_child,
        theta_sfc_ref=288.15,
        p_sfc_ref=101_325.0,
        n_ref_hz=0.01,
    )

    mslp_diag_hpa = compute_mslp_pa(spec, hs_m, state) / 100.0
    lowland = hs_m < 50.0
    assert np.any(lowland)
    assert float(np.max(np.abs(mslp_diag_hpa[lowland] - mslp_true_pa / 100.0))) < 0.08

    gh500_diag = compute_pressure_level_fields(spec, hs_m, state, plev_pa=[50_000.0])["gh_pl"][0]
    assert float(np.max(np.abs(gh500_diag - gh500_true_m))) < 1.0

    w_acc = _discrete_hydrostatic_w_acc(spec, hs_m, state)
    assert float(np.max(np.abs(w_acc))) < 0.02


@pytest.mark.skipif(
    not _ERA5_20190725.exists(),
    reason="Cached 20190725 ERA5 WX-1 file not present in runs/era5_cache",
)
def test_real_era5_20190725_ic_hydrostatic_and_mslp_if_cached() -> None:
    """Real cached ERA5 2019-07-25 00Z IC reproduces sea/lowland MSLP, GH500, and w_acc."""
    spec = get_domain_spec("d3_nl_2p5km", nz=65)
    hs_m, static = build_domain_static_fields(spec)
    state, _, _ = era5_wx1_to_domain_state(
        _ERA5_20190725, spec, hs_m, surface_static=static
    )

    ds = wx1.open_dataset(_ERA5_20190725)
    lats = np.asarray(ds["latitude"].values, dtype=np.float64)
    lons = np.asarray(ds["longitude"].values, dtype=np.float64)
    era_mslp_hpa = _regrid_latlon_field(np.asarray(ds["msl"].values, dtype=np.float64) / 100.0, lats, lons, spec)
    era_gh500_m = _regrid_latlon_field(
        np.asarray(ds["z"].sel(level=500).values, dtype=np.float64) / GRAVITY_M_S2,
        lats,
        lons,
        spec,
    )

    mslp_hpa = compute_mslp_pa(spec, hs_m, state) / 100.0
    lowland = hs_m < 50.0
    mslp_bias = float(np.mean((mslp_hpa - era_mslp_hpa)[lowland]))
    assert abs(mslp_bias) < 0.15

    gh500_m = compute_pressure_level_fields(spec, hs_m, state, plev_pa=[50_000.0])["gh_pl"][0]
    gh500_bias = float(np.mean(gh500_m - era_gh500_m))
    assert abs(gh500_bias) < 1.5

    w_acc = _discrete_hydrostatic_w_acc(spec, hs_m, state)
    assert float(np.max(np.abs(w_acc))) < 0.02
