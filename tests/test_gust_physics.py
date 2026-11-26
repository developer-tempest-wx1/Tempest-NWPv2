"""Deterministic classical physics unit tests for Stap 5: SPPGUST buoyancy & active-convection wind gusts."""

from __future__ import annotations

import numpy as np

from tempest_v2.diagnostics import compute_synoptic_convective_2d
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

_LV0 = 2.501e6
_THETA_REF = 288.15
_P_REF = 101_325.0
_N_REF = 0.01


def _build_column_inputs(
    t_prof_k: np.ndarray,
    qv_prof: np.ndarray,
    u_prof_ms: np.ndarray,
    *,
    t2m_val: float,
    q2m_val: float,
    u10_val: float,
    v10_val: float = 0.0,
    ustar_val: float = 0.25,
    shf_val: float = 0.0,
    lhf_val: float = 0.0,
    rain_acc_val: float = 0.0,
    qr_low_val: float = 0.0,
    qg_low_val: float = 0.0,
):
    ny, nx = 4, 4
    vgrid = VerticalGrid(nz=137, z_top_m=31_000.0, stretch_power=1.5, terrain_decay_power=2.5)
    spec = _make_lcc_domain("gust_phys_test", ny=ny, nx=nx, dx_m=2500.0, projection=LCCProjection(), vertical=vgrid)
    hs_m = np.zeros((ny, nx), dtype=np.float64)

    n2 = _N_REF * _N_REF
    pi_sfc = (_P_REF / P0_PA) ** KAPPA
    g2_cp = (GRAVITY_M_S2 * GRAVITY_M_S2) / (CP_D * _THETA_REF * n2)
    z_s = vgrid.zeta_s
    ef = np.exp((n2 / GRAVITY_M_S2) * z_s)
    thb = _THETA_REF * ef
    pib = pi_sfc - g2_cp * (1.0 - 1.0 / ef)

    thp = (t_prof_k / pib) - thb
    state_3d = {
        "u": np.broadcast_to(u_prof_ms[:, None, None], (vgrid.nz, ny, nx + 1)).astype(np.float32).copy(),
        "v": np.zeros((vgrid.nz, ny + 1, nx), dtype=np.float32),
        "theta_p": np.broadcast_to(thp[:, None, None], (vgrid.nz, ny, nx)).astype(np.float32).copy(),
        "pi_p": np.zeros((vgrid.nz, ny, nx), dtype=np.float32),
        "qv": np.broadcast_to(qv_prof[:, None, None], (vgrid.nz, ny, nx)).astype(np.float32).copy(),
        "qc": np.zeros((vgrid.nz, ny, nx), dtype=np.float32),
        "qr": np.zeros((vgrid.nz, ny, nx), dtype=np.float32),
        "qi": np.zeros((vgrid.nz, ny, nx), dtype=np.float32),
        "qs": np.zeros((vgrid.nz, ny, nx), dtype=np.float32),
        "qg": np.zeros((vgrid.nz, ny, nx), dtype=np.float32),
    }
    low_mask = (z_s >= 300.0) & (z_s <= 2000.0)
    if qr_low_val > 0.0:
        state_3d["qr"][low_mask, :, :] = qr_low_val
    if qg_low_val > 0.0:
        state_3d["qg"][low_mask, :, :] = qg_low_val

    surf_state = {
        "t2m": np.full((ny, nx), t2m_val, dtype=np.float32),
        "q2m": np.full((ny, nx), q2m_val, dtype=np.float32),
        "u10m": np.full((ny, nx), u10_val, dtype=np.float32),
        "v10m": np.full((ny, nx), v10_val, dtype=np.float32),
        "ustar": np.full((ny, nx), ustar_val, dtype=np.float32),
        "shf": np.full((ny, nx), shf_val, dtype=np.float32),
        "lhf": np.full((ny, nx), lhf_val, dtype=np.float32),
        "rain_acc": np.full((ny, nx), rain_acc_val, dtype=np.float32),
        "snow_acc": np.zeros((ny, nx), dtype=np.float32),
        "graupel_acc": np.zeros((ny, nx), dtype=np.float32),
    }

    p0 = P0_PA * (pib[0] ** (1.0 / KAPPA))
    tv0 = t_prof_k[0] * (1.0 + 0.608 * qv_prof[0])
    rho0 = p0 / (R_D * tv0)
    return spec, hs_m, vgrid, state_3d, surf_state, float(rho0)


def _stable_dry_profile(z_s: np.ndarray, t_sfc: float = 278.0, qv_val: float = 0.002):
    t_prof = np.maximum(t_sfc - 0.0035 * z_s, 210.0)
    qv_prof = np.full_like(z_s, qv_val)
    u_prof = np.full_like(z_s, 10.0)
    return t_prof, qv_prof, u_prof


def _heatwave_high_cape_profile(z_s: np.ndarray):
    # Hot moist boundary layer + steep mid-level lapse rate -> CAPE_sb > 1500 J/kg
    t_prof = np.maximum(303.15 - 0.0082 * z_s, 212.0)
    qv_prof = np.where(z_s <= 800.0, 0.0165, 0.0165 * np.exp(-np.maximum(z_s - 800.0, 0.0) / 2200.0))
    u_prof = np.interp(z_s, [0.0, 500.0, 1500.0, 31_000.0], [5.0, 10.0, 20.0, 20.0])
    return t_prof, qv_prof, u_prof


def _sppgust_expected(wspd10: float, ustar: float, t2m: float, rho0: float, shf: float, lhf: float) -> float:
    buoy_flux = (GRAVITY_M_S2 / max(t2m, 180.0)) * (
        shf / (rho0 * CP_D) + 0.608 * t2m * lhf / (rho0 * _LV0)
    )
    u_eff = np.cbrt(ustar**3 + (0.5 / 12.0) * 0.4 * 1000.0 * max(buoy_flux, 0.0))
    return float(wspd10 + 7.2 * u_eff)


def test_gust_neutral_stable_matches_sppgust_exact() -> None:
    """In neutral/stable non-convective conditions, SPPGUST reduces identically to wspd10 + 7.2 * ustar."""
    vgrid = VerticalGrid(nz=137, z_top_m=31_000.0, stretch_power=1.5, terrain_decay_power=2.5)
    t_prof, qv_prof, u_prof = _stable_dry_profile(vgrid.zeta_s)

    for shf, lhf in [(0.0, 0.0), (-40.0, -10.0)]:
        spec, hs_m, _, st, sf, _ = _build_column_inputs(
            t_prof, qv_prof, u_prof, t2m_val=278.0, q2m_val=0.002, u10_val=10.0, ustar_val=0.40, shf_val=shf, lhf_val=lhf
        )
        out = compute_synoptic_convective_2d(spec, hs_m, st, sf)
        assert abs(float(out["cape_sb"].mean())) < 1e-6
        assert abs(float(out["precip_rate"].mean())) < 1e-9
        expected = 10.0 + 7.2 * 0.40
        assert abs(float(out["gust10m"].mean()) - expected) < 1e-3


def test_gust_unstable_daytime_buoyancy_enhancement() -> None:
    """Unstable daytime buoyancy flux increases gust10m per SPPGUST without zero-ustar singularity."""
    vgrid = VerticalGrid(nz=137, z_top_m=31_000.0, stretch_power=1.5, terrain_decay_power=2.5)
    t_prof, qv_prof, u_prof = _stable_dry_profile(vgrid.zeta_s, t_sfc=295.0, qv_val=0.001)

    spec_u, hs_u, _, st_u, sf_u, rho0 = _build_column_inputs(
        t_prof, qv_prof, u_prof, t2m_val=295.0, q2m_val=0.001, u10_val=4.0, ustar_val=0.25, shf_val=250.0, lhf_val=200.0
    )
    out_u = compute_synoptic_convective_2d(spec_u, hs_u, st_u, sf_u)

    spec_s, hs_s, _, st_s, sf_s, _ = _build_column_inputs(
        t_prof, qv_prof, u_prof, t2m_val=295.0, q2m_val=0.001, u10_val=4.0, ustar_val=0.25, shf_val=-30.0, lhf_val=0.0
    )
    out_s = compute_synoptic_convective_2d(spec_s, hs_s, st_s, sf_s)

    gust_u = float(out_u["gust10m"].mean())
    gust_s = float(out_s["gust10m"].mean())
    assert abs(float(out_u["cape_sb"].mean())) < 1e-6
    assert gust_u - gust_s > 1.0

    expected_u = _sppgust_expected(4.0, 0.25, 295.0, rho0, 250.0, 200.0)
    assert abs(gust_u - expected_u) < 5e-3

    # Very small ustar = 0.02 m/s must stay finite, non-singular, and match algebraic SPPGUST
    spec_low, hs_low, _, st_low, sf_low, rho0_low = _build_column_inputs(
        t_prof, qv_prof, u_prof, t2m_val=295.0, q2m_val=0.001, u10_val=4.0, ustar_val=0.02, shf_val=250.0, lhf_val=200.0
    )
    out_low = compute_synoptic_convective_2d(spec_low, hs_low, st_low, sf_low)
    gust_low = float(out_low["gust10m"].mean())
    expected_low = _sppgust_expected(4.0, 0.02, 295.0, rho0_low, 250.0, 200.0)
    assert np.isfinite(gust_low) and gust_low > 4.0 + 1.0
    assert abs(gust_low - expected_low) < 5e-3


def test_gust_no_clear_sky_cape_phantom_gust() -> None:
    """High clear-sky CAPE with zero precip/hydrometeors must NOT add phantom thunderstorm gusts."""
    vgrid = VerticalGrid(nz=137, z_top_m=31_000.0, stretch_power=1.5, terrain_decay_power=2.5)
    t_prof, qv_prof, u_prof = _heatwave_high_cape_profile(vgrid.zeta_s)

    spec, hs_m, _, st, sf, rho0 = _build_column_inputs(
        t_prof,
        qv_prof,
        u_prof,
        t2m_val=303.15,
        q2m_val=0.0165,
        u10_val=5.0,
        ustar_val=0.28,
        shf_val=180.0,
        lhf_val=80.0,
        rain_acc_val=0.0,
    )
    out = compute_synoptic_convective_2d(spec, hs_m, st, sf)
    assert float(out["cape_sb"].mean()) > 1500.0

    expected_sppgust = _sppgust_expected(5.0, 0.28, 303.15, rho0, 180.0, 80.0)
    assert abs(float(out["gust10m"].mean()) - expected_sppgust) < 1e-3


def test_gust_active_convective_downdraft_and_shear_transport() -> None:
    """Active precipitating convection triggers Bechtold-Bidlot gusts; stratiform rain with zero CAPE adds none."""
    vgrid = VerticalGrid(nz=137, z_top_m=31_000.0, stretch_power=1.5, terrain_decay_power=2.5)
    t_prof, qv_prof, u_prof = _heatwave_high_cape_profile(vgrid.zeta_s)

    spec_dry, hs_dry, _, st_dry, sf_dry, _ = _build_column_inputs(
        t_prof, qv_prof, u_prof, t2m_val=303.15, q2m_val=0.0165, u10_val=5.0, ustar_val=0.28, shf_val=180.0, lhf_val=80.0
    )
    gust_dry = float(compute_synoptic_convective_2d(spec_dry, hs_dry, st_dry, sf_dry)["gust10m"].mean())

    # Active convection via surface rain rate (4.0 mm over 3600 s)
    spec_pr, hs_pr, _, st_pr, sf_pr, _ = _build_column_inputs(
        t_prof,
        qv_prof,
        u_prof,
        t2m_val=303.15,
        q2m_val=0.0165,
        u10_val=5.0,
        ustar_val=0.28,
        shf_val=180.0,
        lhf_val=80.0,
        rain_acc_val=4.0,
    )
    out_pr = compute_synoptic_convective_2d(
        spec_pr,
        hs_pr,
        st_pr,
        sf_pr,
        prev_precip_total=np.zeros((4, 4), dtype=np.float64),
        interval_s=3600.0,
    )
    gust_pr = float(out_pr["gust10m"].mean())
    assert gust_pr - gust_dry >= 4.0

    # Active convection via low-level convective hydrometeors (qr + qg = 8e-4 kg/kg below 2.5 km)
    spec_qm, hs_qm, _, st_qm, sf_qm, _ = _build_column_inputs(
        t_prof,
        qv_prof,
        u_prof,
        t2m_val=303.15,
        q2m_val=0.0165,
        u10_val=5.0,
        ustar_val=0.28,
        shf_val=180.0,
        lhf_val=80.0,
        qr_low_val=5.0e-4,
        qg_low_val=3.0e-4,
    )
    gust_qm = float(compute_synoptic_convective_2d(spec_qm, hs_qm, st_qm, sf_qm)["gust10m"].mean())
    assert gust_qm - gust_dry >= 4.0

    # Stratiform rain (CAPE_sb == 0, CAPE_mu == 0) must add 0.0 m/s convective gust
    t_strat, qv_strat, _ = _stable_dry_profile(vgrid.zeta_s)
    spec_sn, hs_sn, _, st_sn, sf_sn, _ = _build_column_inputs(
        t_strat, qv_strat, u_prof, t2m_val=278.0, q2m_val=0.002, u10_val=5.0, ustar_val=0.28, rain_acc_val=0.0
    )
    gust_strat_dry = float(compute_synoptic_convective_2d(spec_sn, hs_sn, st_sn, sf_sn)["gust10m"].mean())

    spec_sr, hs_sr, _, st_sr, sf_sr, _ = _build_column_inputs(
        t_strat, qv_strat, u_prof, t2m_val=278.0, q2m_val=0.002, u10_val=5.0, ustar_val=0.28, rain_acc_val=4.0
    )
    out_sr = compute_synoptic_convective_2d(
        spec_sr,
        hs_sr,
        st_sr,
        sf_sr,
        prev_precip_total=np.zeros((4, 4), dtype=np.float64),
        interval_s=3600.0,
    )
    assert abs(float(out_sr["cape_sb"].mean())) < 1e-6
    assert abs(float(out_sr["cape_mu"].mean())) < 1e-6
    assert abs(float(out_sr["gust10m"].mean()) - gust_strat_dry) < 1e-4


def test_gust_cold_start_zero_ustar_fallback() -> None:
    """Uninitialized T+00 (ustar=0.0) falls back to neutral WMO roughness 0.0688 * wspd10."""
    vgrid = VerticalGrid(nz=137, z_top_m=31_000.0, stretch_power=1.5, terrain_decay_power=2.5)
    t_prof, qv_prof, u_prof = _stable_dry_profile(vgrid.zeta_s)
    spec, hs_m, _, st, sf, _ = _build_column_inputs(
        t_prof, qv_prof, u_prof, t2m_val=278.0, q2m_val=0.002, u10_val=10.0, ustar_val=0.0, shf_val=0.0, lhf_val=0.0
    )
    out = compute_synoptic_convective_2d(spec, hs_m, st, sf)
    expected = 10.0 + 7.2 * 0.0688 * 10.0
    assert abs(float(out["gust10m"].mean()) - expected) < 1e-3
