"""Tests for 3-band SW optics, Sundqvist subgrid clouds, and ecRad Tripleclouds overlap."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pytest

from tempest_v2.cuda_bridge import NonHydrostaticGPUSolver, cuda_vram_info
from tempest_v2.diagnostics import compute_synoptic_convective_2d
from tempest_v2.projection import (
    CP_D,
    GRAVITY_M_S2,
    KAPPA,
    P0_PA,
    R_D,
    DomainSpec,
    LCCProjection,
    VerticalGrid,
    _make_lcc_domain,
)

_EPS_MW = 0.62198
_R_V = 461.5
_EPS_V = (_R_V / R_D) - 1.0
_CV_D = CP_D - R_D


def _require_cuda() -> None:
    try:
        free_b, _ = cuda_vram_info()
    except Exception as exc:
        pytest.skip(f"CUDA unavailable: {exc}")
    if free_b < 256 * 1024 * 1024:
        pytest.skip(f"Insufficient CUDA VRAM ({free_b} B)")


def _make_rad_spec(*, ny: int = 4, nx: int = 4, nz: int = 40) -> DomainSpec:
    vgrid = VerticalGrid(nz=nz, z_top_m=22_000.0, stretch_power=1.35)
    return _make_lcc_domain(
        "test_rad_col",
        ny=ny,
        nx=nx,
        dx_m=2500.0,
        projection=LCCProjection(),
        vertical=vgrid,
        center_lat_deg=52.10,
        center_lon_deg=5.18,
    )


def _es_water_ifs(t_k: np.ndarray) -> np.ndarray:
    t = np.clip(t_k, 173.15, 373.15)
    return 611.21 * np.exp(17.502 * (t - 273.15) / (t - 32.19))


def _es_ice_ifs(t_k: np.ndarray) -> np.ndarray:
    t = np.clip(t_k, 150.0, 273.15)
    return 611.21 * np.exp(22.587 * (t - 273.15) / (t + 0.7))


def _qsat_from_es(es: np.ndarray, p_pa: np.ndarray) -> np.ndarray:
    p = np.maximum(p_pa, 100.0)
    ec = np.minimum(es, 0.90 * p)
    return (_EPS_MW * ec) / np.maximum(p - (1.0 - _EPS_MW) * ec, 100.0)


def _eval_single_step_rad(
    spec: DomainSpec,
    *,
    julian_day: float,
    utc_seconds: float,
    rad_interval_s: float = 0.0,
    lsm: float = 1.0,
    albedo_base: float = 0.20,
    t_sfc_k: float = 285.0,
    qv_profile: np.ndarray | None = None,
    qc_profile: np.ndarray | None = None,
    enable_microphysics: int = 1,
    cy46_physics: int = 0,
) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    _require_cuda()
    hs = np.zeros((spec.ny, spec.nx), dtype=np.float64)
    _, z_s, _ = spec.vertical.heights_3d(hs)
    shape3 = (spec.nz, spec.ny, spec.nx)

    solver = NonHydrostaticGPUSolver(
        spec.vertical,
        spec.dx_m,
        spec.dy_m,
        hs,
        msf=spec.map_factor,
        f_cor=spec.coriolis_f,
        has_moist=True,
        has_physics=True,
        theta_sfc_k=t_sfc_k,
        p_sfc_pa=101_325.0,
        n_ref_s=0.011,
    )
    try:
        sstatic = solver.default_surface_static(
            lat_deg=52.10,
            lon_deg=5.18,
            lsm=lsm,
            soil_type=9.0,
            albedo_base=albedo_base,
            z0m_base=0.05 if lsm >= 0.5 else 0.001,
            veg_frac=0.60 if lsm >= 0.5 else 0.0,
        )
        solver.set_surface_static(sstatic)
        st = solver.empty_state()
        if qv_profile is not None:
            st["qv"] = np.broadcast_to(qv_profile, shape3).astype(np.float32).copy()
        if qc_profile is not None:
            st["qc"] = np.broadcast_to(qc_profile, shape3).astype(np.float32).copy()
        solver.upload_state(st)

        surf = solver.empty_surface_state()
        surf["t_skin"].fill(t_sfc_k)
        surf["soil_t"].fill(t_sfc_k)
        surf["sst_bulk"].fill(t_sfc_k)
        solver.upload_surface(surf)

        solver.set_physics_config(
            enable_microphysics=enable_microphysics,
            enable_pbl=0,
            enable_radiation=1,
            enable_surface=0,
            rad_interval_s=rad_interval_s,
            julian_day=julian_day,
            utc_seconds=utc_seconds,
            co2_ppm=425.0,
            cy46_physics=cy46_physics,
        )
        solver.step_physics(dt_s=12.0, n_steps=1)
        return solver.download_state(), solver.download_surface()
    finally:
        solver.close()


def _column_dry_mass_hpa(spec: DomainSpec, hs_m: np.ndarray, ref: Dict[str, np.ndarray], state: Dict[str, np.ndarray]) -> float:
    theta = ref["theta_bar"] + state["theta_p"].astype(np.float64)
    pi = np.maximum(1e-5, ref["pi_bar"] + state["pi_p"].astype(np.float64))
    qv = state["qv"].astype(np.float64)
    theta_v = theta * (1.0 + _EPS_V * qv)
    rho_g = (P0_PA / (R_D * theta_v)) * np.power(pi, _CV_D / R_D)
    z_w, _, _ = spec.vertical.heights_3d(hs_m)
    dz = np.diff(z_w, axis=0)
    return float(np.mean(np.sum(rho_g * (1.0 - qv) * dz, axis=0)) * GRAVITY_M_S2 / 100.0)


def test_clearsky_sw_calibration_june_march_and_land_sea_aerosol() -> None:
    """Verify June 21 & March 19 clear-sky SW calibration plus maritime vs continental AOD."""
    spec = _make_rad_spec()
    hs = np.zeros((spec.ny, spec.nx), dtype=np.float64)
    z_w, z_s, _ = spec.vertical.heights_3d(hs)
    dz = np.diff(z_w, axis=0)

    # Spring clear profile normalized to TCWV = 7.5 kg/m^2 (matching 2025-03-19)
    ref_mar = VerticalGrid.reference_state(z_s, theta_sfc_k=282.0, p_sfc_pa=102_000.0, brunt_vaisala_hz=0.011)
    qv_mar = 0.004 * np.exp(-z_s / 2200.0)
    tcwv_mar = float(np.mean(np.sum(ref_mar["rho_bar"] * qv_mar * dz, axis=0)))
    qv_mar = qv_mar * (7.5 / max(tcwv_mar, 1e-6))

    _, sf_mar_land = _eval_single_step_rad(
        spec,
        julian_day=78.0,
        utc_seconds=12.0 * 3600.0 - 450.0,
        rad_interval_s=900.0,
        lsm=1.0,
        albedo_base=0.20,
        t_sfc_k=282.0,
        qv_profile=qv_mar,
        enable_microphysics=0,
    )
    sw_mar_land = float(sf_mar_land["sw_down"].mean())
    assert 600.0 <= sw_mar_land <= 665.0, f"March 19 noon clear-sky SW={sw_mar_land:.1f} W/m^2"

    # June 21 noon (DOY 172, 12Z) with ~20 kg/m^2 TCWV
    qv_jun = 0.008 * np.exp(-z_s / 2500.0)
    _, sf_jun_land = _eval_single_step_rad(
        spec,
        julian_day=172.0,
        utc_seconds=12.0 * 3600.0,
        lsm=1.0,
        albedo_base=0.20,
        t_sfc_k=292.0,
        qv_profile=qv_jun,
        enable_microphysics=0,
    )
    sw_jun_land = float(sf_jun_land["sw_down"].mean())
    assert 750.0 <= sw_jun_land <= 890.0, f"June 21 noon clear-sky SW={sw_jun_land:.1f} W/m^2"

    # Isolate pure aerosol attenuation difference (maritime AOD=0.06 vs continental AOD=0.10) at fixed albedo
    _, sf_aer_land = _eval_single_step_rad(
        spec,
        julian_day=172.0,
        utc_seconds=12.0 * 3600.0,
        lsm=1.0,
        albedo_base=0.15,
        t_sfc_k=290.0,
        qv_profile=qv_jun,
        enable_microphysics=0,
    )
    _, sf_aer_sea = _eval_single_step_rad(
        spec,
        julian_day=172.0,
        utc_seconds=12.0 * 3600.0,
        lsm=0.0,
        albedo_base=0.15,
        t_sfc_k=290.0,
        qv_profile=qv_jun,
        enable_microphysics=0,
    )
    sw_aer_land = float(sf_aer_land["sw_down"].mean())
    sw_aer_sea = float(sf_aer_sea["sw_down"].mean())
    assert sw_aer_sea > sw_aer_land + 5.0, (
        f"Expected maritime AOD=0.06 ({sw_aer_sea:.1f}) brighter than continental AOD=0.10 ({sw_aer_land:.1f})"
    )


def test_subgrid_rh_cloud_radiative_impact_and_threshold_continuity() -> None:
    """Verify Sundqvist subgrid RH cloud cools SW / warms LW at RH=0.94 with zero subgrid jump below uc."""
    spec = _make_rad_spec()
    hs = np.zeros((spec.ny, spec.nx), dtype=np.float64)
    _, z_s, _ = spec.vertical.heights_3d(hs)
    ref = VerticalGrid.reference_state(z_s, theta_sfc_k=283.0, p_sfc_pa=101_325.0, brunt_vaisala_hz=0.011)
    qs_w = _qsat_from_es(_es_water_ifs(ref["t_bar"]), ref["p_bar"])

    bl_mask = (z_s >= 300.0) & (z_s <= 1400.0)
    qv_50 = np.where(bl_mask, 0.50 * qs_w, 0.25 * qs_w)
    qv_78 = np.where(bl_mask, 0.78 * qs_w, 0.25 * qs_w)
    qv_94 = np.where(bl_mask, 0.94 * qs_w, 0.25 * qs_w)

    _, sf_50 = _eval_single_step_rad(
        spec, julian_day=100.0, utc_seconds=12.0 * 3600.0, t_sfc_k=283.0, qv_profile=qv_50, enable_microphysics=1
    )
    _, sf_78 = _eval_single_step_rad(
        spec, julian_day=100.0, utc_seconds=12.0 * 3600.0, t_sfc_k=283.0, qv_profile=qv_78, enable_microphysics=1
    )
    _, sf_94 = _eval_single_step_rad(
        spec, julian_day=100.0, utc_seconds=12.0 * 3600.0, t_sfc_k=283.0, qv_profile=qv_94, enable_microphysics=1
    )

    sw_50, lw_50 = float(sf_50["sw_down"].mean()), float(sf_50["lw_down"].mean())
    sw_78, lw_78 = float(sf_78["sw_down"].mean()), float(sf_78["lw_down"].mean())
    sw_94, lw_94 = float(sf_94["sw_down"].mean()), float(sf_94["lw_down"].mean())

    # Below uc=0.80, RH=0.78 stays clear-sky (only smooth H2O gas absorption difference)
    assert abs(sw_50 - sw_78) < 25.0
    assert abs(lw_78 - lw_50) < 12.0

    # At RH=0.94 with zero resolved qc/qi, subgrid stratus strongly attenuates SW and warms LW
    assert sw_94 < 0.75 * sw_50, f"Expected >25% subgrid cloud SW attenuation: sw_50={sw_50:.1f}, sw_94={sw_94:.1f}"
    assert lw_94 > lw_50 + 25.0, f"Expected >+25 W/m^2 subgrid cloud LW warming: lw_50={lw_50:.1f}, lw_94={lw_94:.1f}"


def test_tripleclouds_resolved_cloud_and_closed_column_conservation() -> None:
    """Verify resolved-cloud Tripleclouds attenuation and 2h closed-column mass conservation with microphysics+rad."""
    spec = _make_rad_spec(ny=4, nx=4, nz=40)
    hs = np.zeros((spec.ny, spec.nx), dtype=np.float64)
    _, z_s, _ = spec.vertical.heights_3d(hs)
    qv_base = 0.006 * np.exp(-z_s / 2500.0)
    qc_cloud = np.zeros_like(z_s)
    qc_cloud[4:10] = 1.5e-3

    _, sf_clr = _eval_single_step_rad(
        spec, julian_day=172.0, utc_seconds=12.0 * 3600.0, t_sfc_k=288.0, qv_profile=qv_base, enable_microphysics=1
    )
    st_cld, sf_cld = _eval_single_step_rad(
        spec,
        julian_day=172.0,
        utc_seconds=12.0 * 3600.0,
        t_sfc_k=288.0,
        qv_profile=qv_base,
        qc_profile=qc_cloud,
        enable_microphysics=1,
    )
    sw_clr, lw_clr = float(sf_clr["sw_down"].mean()), float(sf_clr["lw_down"].mean())
    sw_cld, lw_cld = float(sf_cld["sw_down"].mean()), float(sf_cld["lw_down"].mean())
    assert np.isfinite(sw_cld) and np.isfinite(lw_cld)
    assert sw_cld < 0.70 * sw_clr
    assert lw_cld > lw_clr + 20.0
    assert float(st_cld["theta_p"][4:10].mean()) != 0.0

    # 2h coupled closed-column conservation with active subgrid RH clouds + Tripleclouds radiation
    _require_cuda()
    col_spec = _make_rad_spec(ny=8, nx=8, nz=65)
    hs_col = np.zeros((col_spec.ny, col_spec.nx), dtype=np.float64)
    _, z_col, _ = col_spec.vertical.heights_3d(hs_col)
    ref_col = VerticalGrid.reference_state(
        z_col, theta_sfc_k=288.15, p_sfc_pa=101_325.0, brunt_vaisala_hz=0.01
    )
    qs_col = _qsat_from_es(_es_water_ifs(ref_col["t_bar"]), ref_col["p_bar"])
    # Peak RH ~ 0.90 around 1.5-4 km to keep subgrid clouds active without heavy saturation shock
    rh_tgt = np.where((z_col >= 800.0) & (z_col <= 5000.0), 0.90, 0.55)
    qv_col = np.minimum(rh_tgt * qs_col, 0.012 * np.exp(-z_col / 2500.0))

    solver = NonHydrostaticGPUSolver(
        col_spec.vertical,
        col_spec.dx_m,
        col_spec.dy_m,
        hs_col,
        msf=col_spec.map_factor,
        f_cor=col_spec.coriolis_f,
        has_moist=True,
        has_physics=True,
        bc_x=1,
        bc_y=1,
        use_coriolis=False,
    )
    try:
        solver.set_surface_static(solver.default_surface_static(lat_deg=52.1, lon_deg=5.18, lsm=1.0, soil_type=9.0))
        sf0 = solver.empty_surface_state()
        sf0["t_skin"].fill(288.15)
        sf0["soil_t"].fill(288.15)
        solver.upload_surface(sf0)
        st0 = solver.empty_state()
        st0["qv"] = np.broadcast_to(qv_col, (col_spec.nz, col_spec.ny, col_spec.nx)).astype(np.float32).copy()
        solver.upload_state(st0)
        solver.set_physics_config(
            enable_microphysics=1,
            enable_pbl=1,
            enable_radiation=1,
            enable_surface=1,
            rad_interval_s=900.0,
            julian_day=172.0,
            utc_seconds=36_000.0,
            cy46_physics=1,
            pbl_last=1,
        )
        dry0 = _column_dry_mass_hpa(col_spec, hs_col, ref_col, st0)
        solver.step_full(12.0, 600, physics_every=3, flush_physics=True)  # 2h coupled run
        st1 = solver.download_state()
        for name, arr in st1.items():
            assert np.all(np.isfinite(arr)), f"Non-finite values in {name}"
        dry1 = _column_dry_mass_hpa(col_spec, hs_col, ref_col, st1)
        assert abs(dry1 - dry0) < 0.05, f"Closed-column dry mass drift={dry1 - dry0:+.4f} hPa"
    finally:
        solver.close()


def test_diagnostics_subgrid_mixed_phase_cloud_cover_and_exp_ran_overlap() -> None:
    """Verify TCC/LCC/MCC/HCC reflect subgrid BL stratus, cold ice cirrus, and Exp-Ran overlap bounds."""
    spec = _make_rad_spec(ny=4, nx=4, nz=65)
    hs = np.zeros((spec.ny, spec.nx), dtype=np.float64)
    z_w, z_s, _ = spec.vertical.heights_3d(hs)
    dz = np.diff(z_w, axis=0)
    ref = VerticalGrid.reference_state(z_s, theta_sfc_k=276.0, p_sfc_pa=101_325.0, brunt_vaisala_hz=0.01)
    shape3 = (spec.nz, spec.ny, spec.nx)

    def _run_diag(qv_1col: np.ndarray) -> Dict[str, np.ndarray]:
        st = {
            "u": np.zeros((spec.nz, spec.ny, spec.nx + 1), dtype=np.float32),
            "v": np.zeros((spec.nz, spec.ny + 1, spec.nx), dtype=np.float32),
            "w": np.zeros((spec.nz + 1, spec.ny, spec.nx), dtype=np.float32),
            "theta_p": np.broadcast_to(ref["theta_bar"] - VerticalGrid.reference_state(z_s)["theta_bar"], shape3).astype(np.float32).copy(),
            "pi_p": np.broadcast_to(ref["pi_bar"] - VerticalGrid.reference_state(z_s)["pi_bar"], shape3).astype(np.float32).copy(),
            "qv": np.broadcast_to(qv_1col, shape3).astype(np.float32).copy(),
            "qc": np.zeros(shape3, dtype=np.float32),
            "qr": np.zeros(shape3, dtype=np.float32),
            "qi": np.zeros(shape3, dtype=np.float32),
            "qs": np.zeros(shape3, dtype=np.float32),
            "qg": np.zeros(shape3, dtype=np.float32),
        }
        surf = {
            "t2m": np.full((spec.ny, spec.nx), 276.0, dtype=np.float32),
            "q2m": np.broadcast_to(qv_1col[0], (spec.ny, spec.nx)).astype(np.float32).copy(),
            "u10m": np.zeros((spec.ny, spec.nx), dtype=np.float32),
            "v10m": np.zeros((spec.ny, spec.nx), dtype=np.float32),
            "ustar": np.full((spec.ny, spec.nx), 0.2, dtype=np.float32),
            "rain_acc": np.zeros((spec.ny, spec.nx), dtype=np.float32),
            "snow_acc": np.zeros((spec.ny, spec.nx), dtype=np.float32),
            "graupel_acc": np.zeros((spec.ny, spec.nx), dtype=np.float32),
        }
        return compute_synoptic_convective_2d(
            spec, hs, st, surf, theta_sfc_ref=288.15, p_sfc_ref=101_325.0, n_ref_hz=0.01
        )

    t_col = ref["t_bar"]
    p_col = ref["p_bar"]
    qs_w = _qsat_from_es(_es_water_ifs(t_col), p_col)
    qs_i = _qsat_from_es(_es_ice_ifs(t_col), p_col)
    f_liq = np.clip((t_col - 250.16) / 23.0, 0.0, 1.0) ** 2
    qs_mix = _qsat_from_es(f_liq * _es_water_ifs(t_col) + (1.0 - f_liq) * _es_ice_ifs(t_col), p_col)

    # 1. Dry atmosphere (50% wrt mixed-phase saturation) -> all cloud fractions zero
    syn_dry = _run_diag(0.50 * qs_mix)
    assert float(syn_dry["tcc"].max()) == 0.0
    assert float(syn_dry["lcc"].max()) == 0.0
    assert float(syn_dry["mcc"].max()) == 0.0
    assert float(syn_dry["hcc"].max()) == 0.0

    # 2. Subgrid BL stratus (p >= 850 hPa, RH=0.95, qc=qi=0) -> high LCC & TCC
    qv_bl = np.where(p_col >= 85_000.0, 0.95 * qs_w, 0.30 * qs_mix)
    syn_bl = _run_diag(qv_bl)
    assert float(syn_bl["lcc"].mean()) > 0.65
    assert float(syn_bl["tcc"].mean()) > 0.65
    assert float(syn_bl["hcc"].max()) == 0.0

    # 3. Cold upper-tropospheric cirrus (p in 200..400 hPa, T < 245 K, RH_ice = 0.90)
    # Note liquid RH ~ 55% would miss this cloud without mixed-phase ice saturation!
    cirrus_mask = (p_col >= 20_000.0) & (p_col <= 40_000.0) & (t_col <= 245.0)
    qv_cirrus = np.where(cirrus_mask, 0.90 * qs_i, 0.30 * qs_mix)
    syn_cirrus = _run_diag(qv_cirrus)
    assert float(syn_cirrus["hcc"].mean()) > 0.40
    assert float(syn_cirrus["tcc"].mean()) > 0.40

    # 4. Contiguous mid-level cloud overlap obeys max(cf_k) <= tcc <= random_overlap(cf_k)
    mid_mask = (p_col >= 50_000.0) & (p_col <= 75_000.0)
    qv_mid = np.where(mid_mask, 0.88 * qs_mix, 0.25 * qs_mix)
    syn_mid = _run_diag(qv_mid)
    cf_mid_layers = np.where(mid_mask, 1.0 - np.sqrt(np.maximum(0.0, (1.0 - 0.88) / (1.0 - 0.75))), 0.0)
    cf_max = float(np.max(cf_mid_layers))
    cf_rand = float(1.0 - np.prod(1.0 - cf_mid_layers, axis=0).mean())
    tcc_mid = float(syn_mid["tcc"].mean())
    assert cf_max <= tcc_mid + 1e-5
    assert tcc_mid <= cf_rand + 1e-5
    _ = dz
