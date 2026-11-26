"""Deterministic GPU physics & verification tests for Stap 6: microphysics, CAPE, LPI, and lightning."""

from __future__ import annotations

import numpy as np
import pytest

from tempest_v2.cuda_bridge import NonHydrostaticGPUSolver, cuda_vram_info
from tempest_v2.diagnostics import (
    _bolton_theta_e_and_lcl,
    _moist_adiabat_t_from_theta_e,
    _qsat_water,
    compute_synoptic_convective_2d,
)
from tempest_v2.knmi_verify import _mod_thunder_indicator
from tempest_v2.projection import LCCProjection, VerticalGrid, _make_lcc_domain


def _require_cuda() -> None:
    try:
        free_b, _ = cuda_vram_info()
    except Exception as exc:
        pytest.skip(f"CUDA unavailable: {exc}")
    if free_b < 256 * 1024 * 1024:
        pytest.skip(f"Insufficient CUDA VRAM ({free_b} B)")


def _qsat_water_np(t_k: np.ndarray, p_pa: np.ndarray) -> np.ndarray:
    es = 611.21 * np.exp(17.502 * (t_k - 273.15) / np.maximum(t_k - 32.19, 1.0))
    return (0.62198 * es) / np.maximum(p_pa - 0.37802 * es, 100.0)


def _qsat_ice_np(t_k: np.ndarray, p_pa: np.ndarray) -> np.ndarray:
    es = 611.21 * np.exp(22.587 * (t_k - 273.15) / np.maximum(t_k + 0.7, 1.0))
    return (0.62198 * es) / np.maximum(p_pa - 0.37802 * es, 100.0)


def test_cy46_warm_rain_autoconversion_and_fog_protection() -> None:
    """Verify cy46_physics=1 produces warm rain above qc0, conserves water, and shields thin radiation fog."""
    _require_cuda()
    nz, ny, nx = 32, 4, 4
    vgrid = VerticalGrid(nz=nz, z_top_m=12_000.0, stretch_power=1.2)
    hs = np.zeros((ny, nx), dtype=np.float64)
    solver = NonHydrostaticGPUSolver(
        vgrid, dx_m=2500.0, dy_m=2500.0, hs_m=hs, has_moist=True, has_physics=True
    )
    try:
        _, z_s, _ = vgrid.heights_3d(hs)
        ref = VerticalGrid.reference_state(z_s, theta_sfc_k=288.15, p_sfc_pa=101_325.0, brunt_vaisala_hz=0.01)
        t_ref = ref["theta_bar"] * ref["pi_bar"]
        p_ref = ref["p_bar"]
        qsw = _qsat_water_np(t_ref, p_ref)

        # Case A: Warm cloud qc = 0.8 g/kg aloft (k=4..10) -> realistic collision-coalescence rain + conservation
        st_warm = solver.empty_state()
        st_warm["qv"][:] = (0.98 * qsw).astype(np.float32)
        st_warm["qv"][4:11] = qsw[4:11].astype(np.float32)
        st_warm["qc"][4:11] = 8.0e-4
        m_before = solver.tracer_mass(
            st_warm["qv"].astype(np.float64)
            + st_warm["qc"]
            + st_warm["qr"]
            + st_warm["qi"]
            + st_warm["qs"]
            + st_warm["qg"]
        )
        solver.upload_state(st_warm)
        solver.set_surface_static(solver.default_surface_static(lsm=1.0))
        solver.upload_surface(solver.empty_surface_state())
        solver.set_physics_config(
            enable_microphysics=1, enable_pbl=0, enable_radiation=0, enable_surface=0, cy46_physics=1
        )
        solver.step_physics(dt_s=30.0, n_steps=60)
        out_w = solver.download_state()
        sf_w = solver.download_surface()
        assert float(sf_w["rain_acc"].mean()) > 0.1, f"Expected warm rain > 0.1 mm, got {sf_w['rain_acc'].mean()}"
        m_after = solver.tracer_mass(
            out_w["qv"].astype(np.float64) + out_w["qc"] + out_w["qr"] + out_w["qi"] + out_w["qs"] + out_w["qg"]
        )
        m_prec = float(np.sum((sf_w["rain_acc"] + sf_w["snow_acc"] + sf_w["graupel_acc"]) * (2500.0 * 2500.0)))
        rel_err = abs((m_after + m_prec) - m_before) / max(m_before, 1e-12)
        assert rel_err < 1.0e-4, f"Water mass conservation error: rel_err={rel_err:.3e}"

        # Case B: Thin continental radiation fog qc = 0.15 g/kg at surface -> shielded from autoconversion
        st_fog = solver.empty_state()
        st_fog["qv"][:4] = qsw[:4].astype(np.float32)
        st_fog["qc"][:3] = 1.5e-4
        solver.upload_state(st_fog)
        solver.upload_surface(solver.empty_surface_state())
        solver.step_physics(dt_s=30.0, n_steps=20)
        out_f = solver.download_state()
        assert float(out_f["qr"].max()) < 1.0e-6, f"Thin radiation fog autoconverted to qr={out_f['qr'].max()}"
    finally:
        solver.close()


def test_cy46_snow_to_graupel_riming_and_stratiform_safety() -> None:
    """Mixed-phase convective layer forms graupel via riming; stratiform snow with low qc forms none."""
    _require_cuda()
    nz, ny, nx = 32, 4, 4
    vgrid = VerticalGrid(nz=nz, z_top_m=12_000.0, stretch_power=1.2)
    hs = np.zeros((ny, nx), dtype=np.float64)
    solver = NonHydrostaticGPUSolver(
        vgrid, dx_m=2500.0, dy_m=2500.0, hs_m=hs, has_moist=True, has_physics=True
    )
    try:
        _, z_s, _ = vgrid.heights_3d(hs)
        ref = VerticalGrid.reference_state(z_s, theta_sfc_k=288.15, p_sfc_pa=101_325.0, brunt_vaisala_hz=0.01)
        thb, pib, p_ref = ref["theta_bar"], ref["pi_bar"], ref["p_bar"]
        layer = slice(10, 19)
        t_cold = 261.15  # -12 C mixed-phase riming layer
        qsw_cold = _qsat_water_np(np.full_like(p_ref[layer], t_cold), p_ref[layer])

        def _run_rime(qc_val: float):
            st = solver.empty_state()
            st["theta_p"][layer] = ((t_cold / pib[layer]) - thb[layer]).astype(np.float32)
            st["qv"][layer] = qsw_cold.astype(np.float32)
            st["qs"][layer] = 5.0e-4
            st["qc"][layer] = qc_val
            solver.upload_state(st)
            solver.set_surface_static(solver.default_surface_static(lsm=1.0))
            solver.upload_surface(solver.empty_surface_state())
            solver.set_physics_config(
                enable_microphysics=1, enable_pbl=0, enable_radiation=0, enable_surface=0, cy46_physics=1
            )
            solver.step_physics(dt_s=20.0, n_steps=30)
            return solver.download_state(), solver.download_surface()

        out_conv, _ = _run_rime(1.0e-3)
        assert float(out_conv["qg"].max()) > 1.0e-5

        out_strat, sf_strat = _run_rime(5.0e-5)
        assert float(out_strat["qg"].max()) == 0.0
        assert float(sf_strat["graupel_acc"].max()) == 0.0

        # Fix 1C: -2.5 C (dtemp = 2.5 K) preserves freezing rain even with snow present;
        # -6.0 C (dtemp = 6.0 K) triggers supercooled rain+snow contact freezing into graupel.
        fz_sl = slice(0, 19)

        def _run_contact(t_k: float, qs_val: float):
            qsw_ct = _qsat_water_np(np.full_like(p_ref[fz_sl], t_k), p_ref[fz_sl])
            st = solver.empty_state()
            st["theta_p"][fz_sl] = ((t_k / pib[fz_sl]) - thb[fz_sl]).astype(np.float32)
            st["qv"][fz_sl] = qsw_ct.astype(np.float32)
            st["qr"][layer] = 3.0e-4
            st["qs"][layer] = qs_val
            solver.upload_state(st)
            solver.set_surface_static(solver.default_surface_static(lsm=1.0))
            solver.upload_surface(solver.empty_surface_state())
            solver.set_physics_config(
                enable_microphysics=1, enable_pbl=0, enable_radiation=0, enable_surface=0, cy46_physics=1
            )
            solver.step_physics(dt_s=20.0, n_steps=15)
            return solver.download_state()

        out_fzra = _run_contact(270.65, 3.0e-4)
        assert float(out_fzra["qg"].max()) == 0.0

        out_ct = _run_contact(267.15, 3.0e-4)
        assert float(out_ct["qg"].max()) > 2.0e-5
    finally:
        solver.close()


def test_cy46_wet_bulb_snow_melting_vs_saturated_melting() -> None:
    """At +1 C, subsaturated air (Tw < 0 C) preserves falling snow while saturated air (Tw = +1 C) melts to rain."""
    _require_cuda()
    nz, ny, nx = 32, 4, 4
    vgrid = VerticalGrid(nz=nz, z_top_m=10_000.0, stretch_power=1.1)
    hs = np.zeros((ny, nx), dtype=np.float64)
    solver = NonHydrostaticGPUSolver(
        vgrid, dx_m=2500.0, dy_m=2500.0, hs_m=hs, has_moist=True, has_physics=True
    )
    try:
        _, z_s, _ = vgrid.heights_3d(hs)
        ref = VerticalGrid.reference_state(z_s, theta_sfc_k=288.15, p_sfc_pa=101_325.0, brunt_vaisala_hz=0.01)
        thb, pib, p_ref = ref["theta_bar"], ref["pi_bar"], ref["p_bar"]
        t_plus1 = 274.15  # +1 C -> Tw = -0.8 C at RH = 72%, Tw = +1 C at RH = 100%
        qsw_p1 = _qsat_water_np(np.full_like(p_ref, t_plus1), p_ref)

        def _run_melt(rh_frac: float):
            st = solver.empty_state()
            st["theta_p"][:] = ((t_plus1 / pib) - thb).astype(np.float32)
            st["qv"][:] = (rh_frac * qsw_p1).astype(np.float32)
            st["qs"][1:10] = 8.0e-4
            solver.upload_state(st)
            solver.set_surface_static(solver.default_surface_static(lsm=1.0))
            solver.upload_surface(solver.empty_surface_state())
            solver.set_physics_config(
                enable_microphysics=1, enable_pbl=0, enable_radiation=0, enable_surface=0, cy46_physics=1
            )
            solver.step_physics(dt_s=10.0, n_steps=60)
            return solver.download_surface()

        sf_sub = _run_melt(0.72)
        sf_sat = _run_melt(1.00)
        assert float(sf_sub["snow_acc"].mean()) > float(sf_sub["rain_acc"].mean())
        assert float(sf_sub["rain_acc"].mean()) < 1.0e-4
        assert float(sf_sat["rain_acc"].mean()) > float(sf_sub["rain_acc"].mean()) + 0.05
    finally:
        solver.close()


def test_prognostic_lightning_acc_convective_vs_stratiform() -> None:
    """Convective updraft in mixed-phase charging zone accumulates lightning_acc; stratiform ascent yields 0."""
    _require_cuda()
    nz, ny, nx = 32, 4, 4
    vgrid = VerticalGrid(nz=nz, z_top_m=12_000.0, stretch_power=1.2)
    hs = np.zeros((ny, nx), dtype=np.float64)
    solver = NonHydrostaticGPUSolver(
        vgrid, dx_m=2500.0, dy_m=2500.0, hs_m=hs, has_moist=True, has_physics=True
    )
    try:
        _, z_s, _ = vgrid.heights_3d(hs)
        ref = VerticalGrid.reference_state(z_s, theta_sfc_k=288.15, p_sfc_pa=101_325.0, brunt_vaisala_hz=0.01)
        thb, pib, p_ref = ref["theta_bar"], ref["pi_bar"], ref["p_bar"]
        chg = slice(10, 17)
        t_m15 = 258.15  # -15 C charging zone
        qsi = _qsat_ice_np(np.full_like(p_ref[chg], t_m15), p_ref[chg])

        # Convective charging updraft
        st_c = solver.empty_state()
        st_c["theta_p"][chg] = ((t_m15 / pib[chg]) - thb[chg]).astype(np.float32)
        st_c["qv"][chg] = qsi.astype(np.float32)
        st_c["w"][:] = 3.0
        st_c["qg"][chg] = 1.0e-3
        st_c["qi"][chg] = 2.0e-4
        st_c["qc"][chg] = 5.0e-4
        solver.upload_state(st_c)
        solver.set_surface_static(solver.default_surface_static(lsm=1.0))
        solver.upload_surface(solver.empty_surface_state())
        solver.set_physics_config(
            enable_microphysics=1, enable_pbl=0, enable_radiation=0, enable_surface=0, cy46_physics=1
        )
        solver.step_physics(dt_s=30.0, n_steps=60)
        sf_c = solver.download_surface()
        assert float(sf_c["lightning_acc"].mean()) > 0.05

        # Weak stratiform snow ascent (w = 0.2 m/s, qg = 0) -> strictly 0 lightning_acc
        st_s = solver.empty_state()
        st_s["theta_p"][chg] = ((t_m15 / pib[chg]) - thb[chg]).astype(np.float32)
        st_s["qv"][chg] = qsi.astype(np.float32)
        st_s["w"][:] = 0.2
        st_s["qs"][chg] = 1.0e-3
        solver.upload_state(st_s)
        solver.upload_surface(solver.empty_surface_state())
        solver.step_physics(dt_s=30.0, n_steps=60)
        sf_s = solver.download_surface()
        assert float(sf_s["lightning_acc"].max()) == 0.0
    finally:
        solver.close()


def test_moist_adiabat_exact_inversion_and_colocated_thunder_verify() -> None:
    """Verify exact moist adiabat inversion, GWP/LPI diagnostics, and co-located thunder indicator."""
    te_p, _ = _bolton_theta_e_and_lcl(np.array([303.15]), np.array([0.015]), np.array([101_300.0]))
    t_900 = float(_moist_adiabat_t_from_theta_e(te_p, np.array([90_000.0]))[0])
    assert abs(t_900 - 292.21) < 0.05, f"Expected ~292.21 K at 900 hPa, got {t_900:.2f} K"

    for te_v in (260.0, 290.0, 320.0, 350.0, 385.0):
        for p_hpa in (1000, 850, 700, 500, 300, 200):
            p = np.array([p_hpa * 100.0])
            te = np.array([te_v])
            t_m = _moist_adiabat_t_from_theta_e(te, p)
            if 141.0 < float(t_m[0]) < 329.0:
                te_back, _ = _bolton_theta_e_and_lcl(t_m, _qsat_water(t_m, p), p)
                assert abs(float(te_back[0]) - te_v) < 0.35

    # GWP & LPI diagnostics with and without w
    ny, nx = 4, 4
    vgrid = VerticalGrid(nz=40, z_top_m=15_000.0, stretch_power=1.2)
    spec = _make_lcc_domain("lpi_test", ny=ny, nx=nx, dx_m=2500.0, projection=LCCProjection(), vertical=vgrid)
    hs = np.zeros((ny, nx), dtype=np.float64)
    _, z_s, _ = vgrid.heights_3d(hs)
    ref = VerticalGrid.reference_state(z_s, theta_sfc_k=288.15, p_sfc_pa=101_325.0, brunt_vaisala_hz=0.01)
    st = {
        "u": np.full((vgrid.nz, ny, nx + 1), 6.0, dtype=np.float32),
        "v": np.zeros((vgrid.nz, ny + 1, nx), dtype=np.float32),
        "w": np.full((vgrid.nz + 1, ny, nx), 4.0, dtype=np.float32),
        "theta_p": ((263.15 / ref["pi_bar"]) - ref["theta_bar"]).astype(np.float32),
        "pi_p": np.zeros((vgrid.nz, ny, nx), dtype=np.float32),
        "qv": np.full((vgrid.nz, ny, nx), 0.003, dtype=np.float32),
        "qc": np.full((vgrid.nz, ny, nx), 4.0e-4, dtype=np.float32),
        "qr": np.zeros((vgrid.nz, ny, nx), dtype=np.float32),
        "qi": np.full((vgrid.nz, ny, nx), 2.0e-4, dtype=np.float32),
        "qs": np.full((vgrid.nz, ny, nx), 4.0e-4, dtype=np.float32),
        "qg": np.full((vgrid.nz, ny, nx), 8.0e-4, dtype=np.float32),
    }
    sf = {"t2m": np.full((ny, nx), 288.0, dtype=np.float32), "q2m": np.full((ny, nx), 0.008, dtype=np.float32)}
    out_w = compute_synoptic_convective_2d(spec, hs, st, sf)
    assert float(out_w["gwp"].mean()) > 0.1 and float(out_w["lpi"].mean()) > 0.1
    st_now = {k: v for k, v in st.items() if k != "w"}
    out_now = compute_synoptic_convective_2d(spec, hs, st_now, sf)
    assert float(out_now["lpi"].max()) == 0.0

    # Prognostic lightning_step & co-located 25 km fallback logic
    assert _mod_thunder_indicator({"mod_lightning_step": 0.03, "mod_graupel_step_mm": 0.0}, neighborhood_25km=False) is True
    assert _mod_thunder_indicator({"mod_lightning_step": 0.005, "mod_graupel_step_mm": 0.0}, neighborhood_25km=False) is False
    assert (
        _mod_thunder_indicator(
            {"mod_lightning_step": 0.0, "mod_graupel_step_mm": 0.19, "mod_cape_mu": 15.0, "mod_precip_step_mm": 3.7},
            neighborhood_25km=False,
        )
        is False
    )
    assert (
        _mod_thunder_indicator(
            {
                "mod_nb25_graup_mm": 0.0,
                "mod_nb25_cape_mu": 2000.0,
                "mod_nb25_iwp": 1.0,
                "mod_nb25_precip_mm": 3.0,
                "mod_nb25_thunder_cell": 0.0,
            },
            neighborhood_25km=True,
        )
        is False
    )
