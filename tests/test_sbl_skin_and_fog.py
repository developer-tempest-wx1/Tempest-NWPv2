"""Unit and GPU physics tests for Stap 4: surface-anchored SBL ICs, snow/skin coupling, TR-302 2m, and fog visibility."""

from __future__ import annotations

import numpy as np
import pytest

from tempest_v2.cuda_bridge import NonHydrostaticGPUSolver, cuda_vram_info
from tempest_v2.diagnostics import compute_synoptic_convective_2d
from tempest_v2.ic_builder import _vertical_map_isobaric_to_klemp
from tempest_v2.projection import (
    CP_D,
    GRAVITY_M_S2,
    KAPPA,
    P0_PA,
    R_D,
    LCCProjection,
    VerticalGrid,
    _make_lcc_domain,
)
from tempest_v2.verification import compute_mslp_pa


def _require_cuda() -> None:
    try:
        free_b, _ = cuda_vram_info()
    except Exception as exc:
        pytest.skip(f"CUDA unavailable: {exc}")
    if free_b < 256 * 1024 * 1024:
        pytest.skip(f"Insufficient CUDA VRAM ({free_b} B)")


def _t_from_state(
    vgrid: VerticalGrid,
    state: dict[str, np.ndarray],
    *,
    theta_sfc_k: float,
    p_sfc_pa: float,
    n_ref_s: float,
) -> np.ndarray:
    """Reconstruct air temperature T(z) [K] from perturbation theta_p and pi_p."""
    z_s = vgrid.zeta_s[:, None, None]
    n2 = n_ref_s * n_ref_s
    pi_sfc = (p_sfc_pa / P0_PA) ** KAPPA
    g2_cp = (GRAVITY_M_S2 * GRAVITY_M_S2) / (CP_D * theta_sfc_k * max(n2, 1e-12))
    ef = np.exp((n2 / GRAVITY_M_S2) * z_s)
    thb = theta_sfc_k * ef
    pib = pi_sfc - g2_cp * (1.0 - 1.0 / ef)
    return (thb + state["theta_p"]) * (pib + state["pi_p"])


def _qsat_water_scalar(t_k: float, p_pa: float) -> float:
    es = 611.21 * float(np.exp(17.502 * (t_k - 273.15) / max(t_k - 32.19, 1.0)))
    return (0.62198 * es) / max(p_pa - 0.37802 * es, 100.0)


def test_ic_surface_anchored_nocturnal_inversion() -> None:
    """Verify surface-anchored IC preserves high-pressure nocturnal SBL temperature, moisture, wind, and MSLP."""
    ny, nx = 4, 4
    vgrid = VerticalGrid(nz=137, z_top_m=31_000.0, stretch_power=1.5, terrain_decay_power=2.5)
    proj = LCCProjection()
    spec = _make_lcc_domain("test_sbl_ic", ny=ny, nx=nx, dx_m=2500.0, projection=proj, vertical=vgrid)
    hs_m = np.zeros((ny, nx), dtype=np.float64)

    mslp_pa = 102_800.0
    t_env_sfc = 286.0
    gamma = 0.0065
    q_env = 0.0040
    p_levels_pa = np.array([1000, 975, 950, 925, 850, 700, 500, 300, 200, 100, 50], dtype=np.float64) * 100.0
    nlev = len(p_levels_pa)
    tv_env_sfc = t_env_sfc * (1.0 + 0.608 * q_env)
    exponent = (R_D * gamma) / GRAVITY_M_S2
    z_prof = (tv_env_sfc / gamma) * (1.0 - np.power(p_levels_pa / mslp_pa, exponent))
    t_prof = t_env_sfc - gamma * z_prof

    z_h = np.broadcast_to(z_prof[:, None, None], (nlev, ny, nx)).copy()
    t_h = np.broadcast_to(t_prof[:, None, None], (nlev, ny, nx)).copy()
    u_geo_h = np.full((nlev, ny, nx), 6.0, dtype=np.float64)
    v_geo_h = np.zeros((nlev, ny, nx), dtype=np.float64)
    q_h = np.full((nlev, ny, nx), q_env, dtype=np.float64)
    ps_child = np.full((ny, nx), mslp_pa, dtype=np.float64)

    t2m_h = np.full((ny, nx), 278.0, dtype=np.float64)
    q2m_h = np.full((ny, nx), 0.0053, dtype=np.float64)
    u10_geo_h = np.full((ny, nx), 1.5, dtype=np.float64)
    v10_geo_h = np.zeros((ny, nx), dtype=np.float64)

    state = _vertical_map_isobaric_to_klemp(
        spec,
        hs_m,
        p_levels_pa,
        z_h,
        t_h,
        u_geo_h,
        v_geo_h,
        q_h,
        ps_child,
        theta_sfc_ref=288.15,
        p_sfc_ref=101_325.0,
        n_ref_hz=0.01,
        t2m_h=t2m_h,
        q2m_h=q2m_h,
        u10_geo_h=u10_geo_h,
        v10_geo_h=v10_geo_h,
    )

    t_3d = _t_from_state(vgrid, state, theta_sfc_k=288.15, p_sfc_pa=101_325.0, n_ref_s=0.01)
    t_k0 = float(t_3d[0].mean())
    t_k1 = float(t_3d[1].mean())
    t_k3 = float(t_3d[3].mean())
    assert abs(t_k0 - 278.0) < 0.8, f"Expected T(k=0) anchored near t2m=278 K, got {t_k0:.2f} K"
    assert t_k1 > t_k0 and t_k3 > t_k1, f"Inversion lost: T=[{t_k0:.2f}, {t_k1:.2f}, {t_k3:.2f}]"

    u_c = 0.5 * (state["u"][0, :, :-1] + state["u"][0, :, 1:])
    v_c = 0.5 * (state["v"][0, :-1, :] + state["v"][0, 1:, :])
    wspd0 = float(np.hypot(u_c, v_c).mean())
    assert abs(wspd0 - 1.5) < 0.6, f"Expected k=0 wind near 1.5 m/s, got {wspd0:.2f} m/s"

    qv0 = float(state["qv"][0].mean())
    assert abs(qv0 - 0.0053) < 5e-4, f"Expected qv(k=0) near 0.0053, got {qv0:.5f}"

    mslp_rec = compute_mslp_pa(spec, hs_m, state).astype(np.float64)
    err_hpa = float(np.max(np.abs(mslp_rec - mslp_pa))) / 100.0
    assert err_hpa < 0.5, f"Expected MSLP error < 0.5 hPa, got {err_hpa:.3f} hPa"


def test_gpu_snowpack_insulation_and_ice_saturation() -> None:
    """Verify snowpack thermal insulation and ice-saturation lower nocturnal Tsk and prevent spurious sublimation."""
    _require_cuda()
    vgrid = VerticalGrid(nz=48, z_top_m=20_000.0, stretch_power=1.5)
    ny, nx = 2, 4
    hs = np.zeros((ny, nx), dtype=np.float64)
    solver = NonHydrostaticGPUSolver(
        vgrid,
        dx_m=2500.0,
        dy_m=2500.0,
        hs_m=hs,
        has_moist=True,
        has_physics=True,
        theta_sfc_k=267.0,
        p_sfc_pa=101_500.0,
        n_ref_s=0.011,
    )
    try:
        solver.set_physics_config(
            enable_microphysics=0,
            enable_pbl=1,
            enable_radiation=1,
            enable_surface=1,
            rad_interval_s=300.0,
            julian_day=40.0,
            utc_seconds=0.0,
            cy46_physics=1,
        )
        sstatic = solver.default_surface_static(
            lat_deg=52.1, lon_deg=5.2, lsm=1.0, soil_type=9.0, albedo_base=0.18, z0m_base=0.25, veg_frac=0.70
        )
        solver.set_surface_static(sstatic)

        st = solver.empty_state()
        st["u"][:] = 3.5
        st["qv"][:] = 0.0019
        solver.upload_state(st)

        surf = solver.empty_surface_state()
        surf["t_skin"].fill(266.5)
        surf["soil_t"][:].fill(272.5)
        surf["soil_m"][:].fill(0.32)
        surf["snow_swe"][:, :2] = 12.0  # Left half: insulated snowpack
        surf["snow_swe"][:, 2:] = 0.0   # Right half: bare soil over 272.5 K topsoil
        solver.upload_surface(surf)

        solver.step_physics(dt_s=60.0, n_steps=30)
        sf = solver.download_surface()

        tsk_snow = float(sf["t_skin"][:, :2].mean())
        tsk_bare = float(sf["t_skin"][:, 2:].mean())
        shf_snow = float(sf["shf"][:, :2].mean())
        lhf_snow = float(sf["lhf"][:, :2].mean())

        assert tsk_snow < tsk_bare - 0.5, f"Expected snow insulation Tsk_snow ({tsk_snow:.2f}) < Tsk_bare ({tsk_bare:.2f}) - 0.5"
        assert shf_snow < -3.0, f"Expected cooling snow SHF < -3 W/m^2, got {shf_snow:.2f}"
        assert lhf_snow < 8.0, f"Expected ice-saturation capped snow LHF < 8 W/m^2, got {lhf_snow:.2f}"
    finally:
        solver.close()


def test_gpu_stable_lambda_sk_and_knmi_tr302() -> None:
    """Verify stable nocturnal skin cooling and KNMI TR-302 2m temperature interpolation bounds."""
    _require_cuda()
    vgrid = VerticalGrid(nz=48, z_top_m=20_000.0, stretch_power=1.5)
    ny, nx = 2, 4
    hs = np.zeros((ny, nx), dtype=np.float64)
    solver = NonHydrostaticGPUSolver(
        vgrid,
        dx_m=2500.0,
        dy_m=2500.0,
        hs_m=hs,
        has_moist=True,
        has_physics=True,
        theta_sfc_k=282.0,
        p_sfc_pa=102_000.0,
        n_ref_s=0.011,
    )
    try:
        solver.set_physics_config(
            enable_microphysics=0,
            enable_pbl=1,
            enable_radiation=1,
            enable_surface=1,
            rad_interval_s=300.0,
            julian_day=80.0,
            utc_seconds=0.0,
            cy46_physics=1,
        )
        sstatic = solver.default_surface_static(
            lat_deg=52.1, lon_deg=5.2, lsm=1.0, soil_type=9.0, albedo_base=0.18, z0m_base=0.25, veg_frac=0.70
        )
        solver.set_surface_static(sstatic)

        st = solver.empty_state()
        st["u"][:] = 2.0
        st["qv"][:] = 0.0040
        solver.upload_state(st)

        surf = solver.empty_surface_state()
        surf["t_skin"].fill(276.0)
        surf["soil_t"][:].fill(280.0)
        surf["soil_m"][:].fill(0.28)
        solver.upload_surface(surf)

        solver.step_physics(dt_s=60.0, n_steps=20)
        st_out = solver.download_state()
        sf = solver.download_surface()

        t_3d = _t_from_state(vgrid, st_out, theta_sfc_k=282.0, p_sfc_pa=102_000.0, n_ref_s=0.011)
        t_k0 = float(t_3d[0].mean())
        tsk = float(sf["t_skin"].mean())
        t2m = float(sf["t2m"].mean())
        ts0 = float(sf["soil_t"][0].mean())
        shf_v = float(sf["shf"].mean())

        assert np.all(np.isfinite(sf["t2m"])) and np.all(np.isfinite(sf["shf"]))
        assert tsk < ts0, f"Expected nocturnal skin cooling t_skin ({tsk:.2f}) < soil_t0 ({ts0:.2f})"
        assert shf_v < -2.0, f"Expected nocturnal cooling SHF < -2 W/m^2, got {shf_v:.2f}"
        assert tsk < t2m < t_k0, f"Expected Tsk ({tsk:.2f}) < T2m ({t2m:.2f}) < Tk0 ({t_k0:.2f})"
        w_2m_eff = (t2m - tsk) / max(t_k0 - tsk, 0.1)
        assert w_2m_eff >= 0.45, f"Expected KNMI TR-302 w_2m_eff >= 0.45, got {w_2m_eff:.3f}"
    finally:
        solver.close()


def test_visibility_mature_fog_lifted_k1_and_no_windy_false_alarms() -> None:
    """Verify visibility handles isothermal radiation fog, shallow k=1 lifted fog, and avoids windy/dry false alarms."""
    ny, nx = 4, 4
    vgrid = VerticalGrid(nz=137, z_top_m=31_000.0, stretch_power=1.5, terrain_decay_power=2.5)
    spec = _make_lcc_domain("vis_sbl_test", ny=ny, nx=nx, dx_m=2500.0, projection=LCCProjection(), vertical=vgrid)
    hs = np.zeros((ny, nx), dtype=np.float64)

    n2 = 0.01 * 0.01
    pi_sfc = (101_325.0 / P0_PA) ** KAPPA
    g2_cp = (GRAVITY_M_S2 * GRAVITY_M_S2) / (CP_D * 288.15 * n2)
    z_s = vgrid.zeta_s[:, None, None]
    ef = np.exp((n2 / GRAVITY_M_S2) * z_s)
    thb = 288.15 * ef
    pib = pi_sfc - g2_cp * (1.0 - 1.0 / ef)

    def _run_case(t_prof_k: np.ndarray, t2m_val: float, rh_pct: float, wspd_val: float, qc1_val: float = 0.0) -> float:
        thp = (t_prof_k[:, None, None] / pib) - thb
        st = {
            "u": np.full((vgrid.nz, ny, nx + 1), wspd_val, dtype=np.float32),
            "v": np.zeros((vgrid.nz, ny + 1, nx), dtype=np.float32),
            "theta_p": np.broadcast_to(thp, (vgrid.nz, ny, nx)).astype(np.float32).copy(),
            "pi_p": np.zeros((vgrid.nz, ny, nx), dtype=np.float32),
            "qv": np.full((vgrid.nz, ny, nx), 0.0045, dtype=np.float32),
            "qc": np.zeros((vgrid.nz, ny, nx), dtype=np.float32),
            "qr": np.zeros((vgrid.nz, ny, nx), dtype=np.float32),
            "qi": np.zeros((vgrid.nz, ny, nx), dtype=np.float32),
            "qs": np.zeros((vgrid.nz, ny, nx), dtype=np.float32),
            "qg": np.zeros((vgrid.nz, ny, nx), dtype=np.float32),
        }
        st["qc"][1] = qc1_val
        p_sfc = float(P0_PA * (pib[0, 0, 0] ** (1.0 / KAPPA)))
        q2m_val = (rh_pct / 100.0) * _qsat_water_scalar(t2m_val, p_sfc)
        sf = {
            "t2m": np.full((ny, nx), t2m_val, dtype=np.float32),
            "q2m": np.full((ny, nx), q2m_val, dtype=np.float32),
            "u10m": np.full((ny, nx), wspd_val, dtype=np.float32),
            "v10m": np.zeros((ny, nx), dtype=np.float32),
            "ustar": np.full((ny, nx), 0.05 * wspd_val, dtype=np.float32),
        }
        return float(np.mean(compute_synoptic_convective_2d(spec, hs, st, sf)["vis_m"]))

    # A: Mature isothermal radiation fog capped at k=3 (~128 m AGL) with T_k0 == t2m
    t_mature = np.full(vgrid.nz, 276.0, dtype=np.float64)
    t_mature[1:6] = 279.0
    vis_a = _run_case(t_mature, 276.0, 98.5, 1.2, qc1_val=0.0)
    assert vis_a < 1000.0, f"Expected mature capped fog vis < 1000 m, got {vis_a:.1f} m"

    # B: 1-level lifted fog (qc[0]=0, qc[1]=2.5e-4, z_s[1]=37 m) with 97% RH
    t_iso = np.full(vgrid.nz, 276.0, dtype=np.float64)
    vis_b = _run_case(t_iso, 276.0, 97.0, 2.0, qc1_val=2.5e-4)
    assert vis_b < 1000.0, f"Expected lifted k=1 fog vis < 1000 m, got {vis_b:.1f} m"

    # C: Windy humid air (no fog FA)
    vis_c = _run_case(t_iso, 276.0, 98.5, 6.0, qc1_val=0.0)
    assert vis_c > 2000.0, f"Expected windy humid air vis > 2000 m, got {vis_c:.1f} m"

    # D: Dry clear air
    t_dry = np.full(vgrid.nz, 285.0, dtype=np.float64)
    vis_d = _run_case(t_dry, 285.0, 55.0, 4.0, qc1_val=0.0)
    assert vis_d > 20_000.0, f"Expected clear dry air vis > 20000 m, got {vis_d:.1f} m"
