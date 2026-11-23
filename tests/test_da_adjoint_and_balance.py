"""Step 4 verification: exact DA adjoint, tangent-linear FD, balance, SPD filter, 3D-Var & LHN tests."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from tempest_v2.da import (
    BalanceOperator,
    RecursiveSpatialFilter,
    Var3DConfig,
    VarObservation,
    VarObservationOperator,
    VarStateVec,
    compute_iau_tendencies,
    compute_lhn_tendencies,
    solve_3dvar_pcg,
)
from tempest_v2.projection import VerticalGrid, get_domain_spec


def _mini_spec(nz: int = 24, ny: int = 25, nx: int = 25):
    base = get_domain_spec("d2_eu_lcc", nz=nz)
    j0 = base.ny // 2 - ny // 2
    i0 = base.nx // 2 - nx // 2
    return replace(
        base,
        ny=ny,
        nx=nx,
        x_m=base.x_m[i0 : i0 + nx],
        y_m=base.y_m[j0 : j0 + ny],
        lat2d=base.lat2d[j0 : j0 + ny, i0 : i0 + nx],
        lon2d=base.lon2d[j0 : j0 + ny, i0 : i0 + nx],
        map_factor=base.map_factor[j0 : j0 + ny, i0 : i0 + nx],
        coriolis_f=base.coriolis_f[j0 : j0 + ny, i0 : i0 + nx],
    )


def _sample_bg(spec, hs):
    rng = np.random.default_rng(1234)
    nz, ny, nx = spec.nz, spec.ny, spec.nx
    _, z_s, _ = spec.vertical.heights_3d(hs)
    ref = VerticalGrid.reference_state(z_s)
    return {
        "ps": (ref["p_bar"][0] + 150.0 * rng.standard_normal((ny, nx))).astype(np.float64),
        "theta_p": (0.8 * rng.standard_normal((nz, ny, nx))).astype(np.float64),
        "pi_p": (1e-4 * rng.standard_normal((nz, ny, nx))).astype(np.float64),
        "u": (5.0 + 1.5 * rng.standard_normal((nz, ny, nx))).astype(np.float64),
        "v": (2.0 + 1.5 * rng.standard_normal((nz, ny, nx))).astype(np.float64),
        "qv": np.clip(4e-3 * np.exp(-z_s / 3000.0) + 1e-4 * rng.random((nz, ny, nx)), 1e-6, 0.03),
        "sst": (283.0 + 0.5 * rng.standard_normal((ny, nx))).astype(np.float64),
        "soil_t": (284.0 + 1.0 * rng.standard_normal((ny, nx))).astype(np.float64),
        "soil_m": np.clip(0.28 + 0.03 * rng.standard_normal((ny, nx)), 0.05, 0.45),
    }


def test_recursive_filter_is_exact_spd_symmetric():
    rng = np.random.default_rng(7)
    flt = RecursiveSpatialFilter(dx_m=7500.0, dy_m=7500.0, dz_m=1000.0, length_scale_h_m=45_000.0, length_scale_v_m=2500.0, num_passes=2)

    u2 = rng.standard_normal((19, 23))
    v2 = rng.standard_normal((19, 23))
    lhs2 = float(np.sum(flt.apply_2d(u2) * v2))
    rhs2 = float(np.sum(u2 * flt.apply_2d(v2)))
    assert abs(lhs2 - rhs2) / max(1.0, abs(lhs2)) < 1e-12
    assert float(np.sum(flt.apply_2d(u2) * u2)) >= 0.0

    u3 = rng.standard_normal((12, 17, 21))
    v3 = rng.standard_normal((12, 17, 21))
    lhs3 = float(np.sum(flt.apply_3d(u3) * v3))
    rhs3 = float(np.sum(u3 * flt.apply_3d(v3)))
    assert abs(lhs3 - rhs3) / max(1.0, abs(lhs3)) < 1e-12
    assert float(np.sum(flt.apply_3d(u3) * u3)) >= 0.0


def test_balance_operator_exact_adjoint_and_geostrophic_sign():
    spec = _mini_spec(nz=20, ny=18, nx=18)
    hs = 300.0 * np.exp(-((spec.lat2d - 51.0) ** 2 + (spec.lon2d - 8.5) ** 2) / 4.0)
    bal = BalanceOperator(spec, hs)

    rng = np.random.default_rng(99)
    shape3 = (spec.nz, spec.ny, spec.nx)
    shape2 = (spec.ny, spec.nx)
    ctrl = {
        "ps": rng.standard_normal(shape2),
        "theta_p": rng.standard_normal(shape3),
        "pi_p": rng.standard_normal(shape3),
        "u": rng.standard_normal(shape3),
        "v": rng.standard_normal(shape3),
        "qv": rng.standard_normal(shape3),
    }
    phys_w = {
        "ps": rng.standard_normal(shape2),
        "theta_p": rng.standard_normal(shape3),
        "pi_p": rng.standard_normal(shape3),
        "u": rng.standard_normal(shape3),
        "v": rng.standard_normal(shape3),
        "qv": rng.standard_normal(shape3),
    }

    k_ctrl = bal.forward(ctrl)
    kt_phys = bal.adjoint(phys_w, template_keys=ctrl.keys())
    lhs = sum(float(np.sum(k_ctrl[k] * phys_w[k])) for k in ctrl)
    rhs = sum(float(np.sum(ctrl[k] * kt_phys[k])) for k in ctrl)
    assert abs(lhs - rhs) / max(1.0, abs(lhs)) < 1e-12


def test_observation_operators_exact_adjoint_and_tangent_linear_fd():
    spec = _mini_spec(nz=24, ny=25, nx=25)
    hs = np.zeros((spec.ny, spec.nx), dtype=np.float64)
    bg = _sample_bg(spec, hs)
    hop = VarObservationOperator(spec, hs, bg)

    # Create observations covering ALL supported variable types away from boundary edges
    lat_c = float(spec.lat2d[12, 12]) + 0.04
    lon_c = float(spec.lon2d[12, 12]) + 0.06
    all_types = [
        ("OBS_PS", "ps", None),
        ("OBS_T2", "t2m", None),
        ("OBS_QV2", "qv2m", None),
        ("OBS_TH", "theta_p", 50_000.0),
        ("OBS_U", "u", 70_000.0),
        ("OBS_V", "v", 70_000.0),
        ("OBS_UG", "u_geo", 85_000.0),
        ("OBS_VG", "v_geo", 85_000.0),
        ("OBS_QV", "qv", 70_000.0),
        ("OBS_ZTD", "ztd", None),
        ("OBS_SST", "sst", None),
        ("OBS_ST", "soil_t", None),
        ("OBS_SM", "soil_m", None),
    ]
    obs_list = [
        VarObservation(
            station_id=sid,
            var_type=vt,
            lat=lat_c + 0.02 * idx,
            lon=lon_c - 0.03 * idx,
            alt_m=35.0,
            pressure_pa=plev,
            value=1.0,
            error_variance=1.0,
        )
        for idx, (sid, vt, plev) in enumerate(all_types)
    ]
    prepared = hop.prepare_observations(obs_list)
    assert all(o.valid for o in prepared)

    # 1. Dot-product adjoint test for EVERY individual obs type AND full stack
    rng = np.random.default_rng(42)
    dx = VarStateVec({k: rng.standard_normal(v.shape) for k, v in bg.items()})
    dy = rng.standard_normal(len(prepared))

    h_dx = hop.evaluate_linear_h(dx, prepared)
    ht_dy = hop.apply_adjoint_ht(dy, prepared, bg)
    lhs = float(np.dot(h_dx, dy))
    rhs = dx.dot(ht_dy)
    assert abs(lhs - rhs) / max(1.0, abs(lhs)) < 1e-12

    for single_obs in prepared:
        h1 = hop.evaluate_linear_h(dx, [single_obs])
        dy1 = np.array([rng.standard_normal()])
        ht1 = hop.apply_adjoint_ht(dy1, [single_obs], bg)
        l1 = float(np.dot(h1, dy1))
        r1 = dx.dot(ht1)
        assert abs(l1 - r1) / max(1.0, abs(l1)) < 1e-12, f"Adjoint mismatch on {single_obs.var_type}"

    # 2. Central finite-difference tangent linear check on nonlinear ZTD, T2m, and Ps
    nl_obs = [o for o in prepared if o.var_type in ("ztd", "t2m", "ps")]
    eps = 1e-4
    scales = {"ps": 10.0, "qv": 1e-4, "pi_p": 1e-4}
    dx_unit = VarStateVec({k: scales.get(k, 1e-2) * dx.fields[k] for k in bg})
    hp = hop.evaluate_h({k: bg[k] + eps * dx_unit.fields[k] for k in bg}, nl_obs)
    hm = hop.evaluate_h({k: bg[k] - eps * dx_unit.fields[k] for k in bg}, nl_obs)
    fd_diff = (hp - hm) / (2.0 * eps)
    tl_diff = hop.evaluate_linear_h(dx_unit, nl_obs)
    np.testing.assert_allclose(fd_diff, tl_diff, rtol=5e-5, atol=1e-9)


def test_balanced_single_obs_3dvar_and_lhn_radar():
    spec = _mini_spec(nz=24, ny=25, nx=25)
    hs = np.zeros((spec.ny, spec.nx), dtype=np.float64)
    _, z_s, _ = spec.vertical.heights_3d(hs)
    ref = VerticalGrid.reference_state(z_s)
    bg = {
        "ps": ref["p_bar"][0].copy(),
        "theta_p": np.zeros((spec.nz, spec.ny, spec.nx), dtype=np.float64),
        "pi_p": np.zeros((spec.nz, spec.ny, spec.nx), dtype=np.float64),
        "u": np.zeros((spec.nz, spec.ny, spec.nx), dtype=np.float64),
        "v": np.zeros((spec.nz, spec.ny, spec.nx), dtype=np.float64),
        "qv": np.full((spec.nz, spec.ny, spec.nx), 5e-3, dtype=np.float64),
    }

    bal = BalanceOperator(spec, hs, alpha_geo=1.0)
    hop = VarObservationOperator(spec, hs, bg)
    flt = RecursiveSpatialFilter(dx_m=spec.dx_m, dy_m=spec.dy_m, dz_m=2000.0, length_scale_h_m=30_000.0, num_passes=2)
    cfg = Var3DConfig(max_iterations=25, tolerance=1e-6, sigma_ps=150.0, use_balance=True)

    # Single negative surface pressure innovation (-400 Pa) at domain center -> cyclonic wind balance
    jc, ic = 12, 12
    obs = [
        VarObservation(
            station_id="LOW_CENTER",
            var_type="ps",
            lat=float(spec.lat2d[jc, ic]),
            lon=float(spec.lon2d[jc, ic]),
            alt_m=0.0,
            value=float(bg["ps"][jc, ic]) - 400.0,
            error_variance=50.0 ** 2,
        )
    ]
    res = solve_3dvar_pcg(bg, obs, hop, flt, cfg, balance_op=bal)
    assert res.converged
    assert res.final_cost_j < 0.25 * res.initial_cost_j0
    assert res.analysis_increments["ps"][jc, ic] < -50.0
    # Cyclonic balance in Northern Hemisphere around a low center:
    # South of center (jc-2, ic): westerly (u > 0); North of center (jc+2, ic): easterly (u < 0)
    # East of center (jc, ic+2): southerly (v > 0); West of center (jc, ic-2): northerly (v < 0)
    assert res.analysis_increments["u"][3, jc - 2, ic] > 0.05
    assert res.analysis_increments["u"][3, jc + 2, ic] < -0.05
    assert res.analysis_increments["v"][3, jc, ic + 2] > 0.05
    assert res.analysis_increments["v"][3, jc, ic - 2] < -0.05

    # IAU check
    iau = compute_iau_tendencies(res.analysis_increments, tau_sec=1800.0)
    np.testing.assert_allclose(iau["ps"], res.analysis_increments["ps"] / 1800.0, rtol=1e-5)

    # LHN radar check
    p_obs = np.full((spec.ny, spec.nx), 10.0 / 3600.0, dtype=np.float32)  # 10 mm/h heavy rain
    p_mod = np.zeros((spec.ny, spec.nx), dtype=np.float32)
    dth_dt, dqv_dt = compute_lhn_tendencies(
        p_obs, p_mod, z_s, hs, ref["rho_bar"], bg["theta_p"], ref["pi_bar"], bg["qv"]
    )
    assert float(dth_dt.max()) > 1e-4  # positive convective latent heating spinup
    assert float(dqv_dt.min()) >= -1e-9  # moisture strictly non-negative forcing
    assert np.all(bg["qv"] + 60.0 * dqv_dt >= 0.0)


def test_land_surface_oi_soil_temp_and_moisture_update() -> None:
    """Verify Mahfouf (1991) / Cy50r1 2D-OI updates t2m, q2m, t_skin, soil_t[0..1], and snow-free soil_m[0..2]."""
    from tempest_v2.verification import apply_land_surface_oi_from_increments

    # 3 grid cells: [0]=snow-free land, [1]=snow-covered land, [2]=ocean
    lsm = np.array([[1.0, 1.0, 0.0]], dtype=np.float32)
    surf = {
        "t2m": np.array([[280.0, 272.0, 283.0]], dtype=np.float32),
        "q2m": np.array([[0.0050, 0.0030, 0.0070]], dtype=np.float32),
        "t_skin": np.array([[279.0, 272.5, 284.0]], dtype=np.float32),
        "soil_t": np.broadcast_to(np.array([[[280.0, 272.0, 284.0]]], dtype=np.float32), (4, 1, 3)).copy(),
        "soil_m": np.full((4, 1, 3), 0.25, dtype=np.float32),
        "snow_swe": np.array([[0.0, 10.0, 0.0]], dtype=np.float32),
    }
    t0_bg = np.array([[282.0, 273.0, 283.5]], dtype=np.float64)  # 2 K warmer at k=0 (nocturnal inversion preserved!)
    qv0_bg = np.array([[0.0050, 0.0030, 0.0070]], dtype=np.float64)
    p0_bg = np.full((1, 3), 101_325.0, dtype=np.float64)
    # Model was too warm (-2.0 K increment) and too dry (+0.0005 kg/kg) -> soil should cool & moisten
    dt0 = np.array([[-2.0, +2.0, -2.0]], dtype=np.float64)
    dqv0 = np.array([[+0.0005, 0.0, +0.0005]], dtype=np.float64)

    out = apply_land_surface_oi_from_increments(surf, dt0, dqv0, t0_bg, qv0_bg, p0_bg, lsm)
    # Inversion preserved: t2m shifted incrementally from 280 -> 278, NOT overwritten with t0_ana (280)
    assert out["t2m"][0, 0] == pytest.approx(278.0, abs=1e-4)
    assert out["q2m"][0, 0] == pytest.approx(0.0055, abs=1e-6)
    # Snow-free land: t_skin & soil_t[0..1] cooled, deeper soil_t[2..3] untouched, soil_m[0..2] moistened
    assert out["t_skin"][0, 0] == pytest.approx(279.0 - 1.7, abs=1e-4)
    assert out["soil_t"][0, 0, 0] == pytest.approx(280.0 - 1.3, abs=1e-4)
    assert out["soil_t"][1, 0, 0] == pytest.approx(280.0 - 0.5, abs=1e-4)
    assert out["soil_t"][2, 0, 0] == pytest.approx(280.0, abs=1e-4)
    assert out["soil_m"][0, 0, 0] > 0.25 and out["soil_m"][1, 0, 0] > 0.25 and out["soil_m"][2, 0, 0] > 0.25
    assert out["soil_m"][3, 0, 0] == pytest.approx(0.25, abs=1e-6)
    # Snow-covered land: t_skin capped at freezing 273.15 K, soil_m untouched
    assert out["t_skin"][0, 1] <= 273.15 + 1e-4
    assert out["soil_m"][0, 0, 1] == pytest.approx(0.25, abs=1e-6)
    # Ocean: t_skin, soil_t, soil_m completely untouched
    assert out["t_skin"][0, 2] == pytest.approx(284.0, abs=1e-4)
    assert out["soil_t"][0, 0, 2] == pytest.approx(284.0, abs=1e-4)
    assert out["soil_m"][0, 0, 2] == pytest.approx(0.25, abs=1e-6)
