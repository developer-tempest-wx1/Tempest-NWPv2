"""Unit tests for IC downward hydrostatic extrapolation, q2m initialization, and moisture Davies LBC."""

from __future__ import annotations

import numpy as np

from tempest_v2.ic_builder import _q_from_dewpoint_pa, _vertical_map_isobaric_to_klemp
from tempest_v2.projection import (
    GRAVITY_M_S2,
    P0_PA,
    R_D,
    LCCProjection,
    VerticalGrid,
    _make_lcc_domain,
)
from tempest_v2.verification import compute_mslp_pa


def test_ic_high_pressure_no_1000hpa_clamp() -> None:
    """Verify _vertical_map_isobaric_to_klemp extrapolates hydrostatically below z_1000 instead of clamping at 1000 hPa."""
    ny, nx = 6, 8
    vgrid = VerticalGrid(nz=24, z_top_m=20_000.0, stretch_power=1.3)
    proj = LCCProjection()
    spec = _make_lcc_domain("test_hp_ic", ny=ny, nx=nx, dx_m=7500.0, projection=proj, vertical=vgrid)
    hs_m = np.zeros((ny, nx), dtype=np.float64)

    # Synthetic hydrostatic high-pressure atmosphere (MSLP = 1028 hPa, T_sfc = 282 K, gamma = 0.0065 K/m)
    mslp_true_pa = 102_800.0
    t_sfc_k = 282.0
    q_const = 0.006
    gamma = 0.0065
    p_levels_pa = np.array([1000, 925, 850, 700, 500, 300, 200, 100, 50], dtype=np.float64) * 100.0
    nlev = len(p_levels_pa)
    tv_sfc_k = t_sfc_k * (1.0 + 0.608 * q_const)
    # Exact hydrostatic height z(p) for constant virtual lapse rate gamma:
    # p / p_sfc = (1 - gamma * z / Tv_sfc)^(g / (R_d * gamma))
    exponent = (R_D * gamma) / GRAVITY_M_S2
    z_prof = (tv_sfc_k / gamma) * (1.0 - np.power(p_levels_pa / mslp_true_pa, exponent))
    t_prof = t_sfc_k - gamma * z_prof

    z_h = np.broadcast_to(z_prof[:, None, None], (nlev, ny, nx)).copy()
    t_h = np.broadcast_to(t_prof[:, None, None], (nlev, ny, nx)).copy()
    u_geo_h = np.full((nlev, ny, nx), 5.0, dtype=np.float64)
    v_geo_h = np.zeros((nlev, ny, nx), dtype=np.float64)
    q_h = np.full((nlev, ny, nx), q_const, dtype=np.float64)
    ps_child = np.full((ny, nx), mslp_true_pa, dtype=np.float64)

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
    )

    mslp_rec_pa = compute_mslp_pa(spec, hs_m, state).astype(np.float64)
    err_hpa = np.abs(mslp_rec_pa - mslp_true_pa) / 100.0
    assert float(np.max(err_hpa)) < 0.5, f"MSLP IC error too large (1000 hPa clamp regression?): max={np.max(err_hpa):.2f} hPa"


def test_q_from_dewpoint_pa_roundtrip_and_supersat_clamp() -> None:
    """Verify _q_from_dewpoint_pa inverts back to exact Td2m and clamps Td <= T."""
    t2m = np.array([[275.0, 285.0, 295.0]], dtype=np.float64)
    d2m = np.array([[272.0, 285.5, 288.0]], dtype=np.float64)  # Middle cell tests Td > T clamping
    ps = np.array([[102_500.0, 101_325.0, 99_500.0]], dtype=np.float64)

    q2m = _q_from_dewpoint_pa(d2m, ps, t_k=t2m)
    assert np.all(np.isfinite(q2m))
    assert np.all(q2m >= 1e-7) and np.all(q2m <= 0.04)

    e_pa = (q2m * ps) / (0.62198 + 0.37802 * q2m)
    ln_e = np.log(e_pa / 611.2)
    td_rec = 273.15 + (243.5 * ln_e) / (17.67 - ln_e)
    td_expected = np.minimum(d2m, t2m)
    assert float(np.max(np.abs(td_rec - td_expected))) < 0.05
