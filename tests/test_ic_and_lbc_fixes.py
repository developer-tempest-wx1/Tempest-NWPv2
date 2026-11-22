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


def test_moisture_davies_lbc_and_open_tracer_flux() -> None:
    """Verify bc=2 open tracer fluxes + d_lbc_qv Davies relaxation prevent western dry rim & eastern pileup."""
    from tempest_v2.cuda_bridge import NonHydrostaticGPUSolver

    nz, ny, nx = 12, 16, 48
    dx = dy = 2000.0
    vgrid = VerticalGrid(nz=nz, z_top_m=15_000.0, stretch_power=1.2)
    hs = np.zeros((ny, nx), dtype=np.float64)

    solver = NonHydrostaticGPUSolver(
        vgrid,
        dx,
        dy,
        hs,
        has_moist=True,
        bc_x=2,
        bc_y=2,
        use_coriolis=False,
        lbc_rim_width=8,
        lbc_tau_max_s=0.04,
    )
    try:
        st = solver.empty_state()
        st["u"][:] = 20.0
        st["qv"][:] = 0.010
        lbc = {k: st[k].copy() for k in ("u", "v", "w", "theta_p", "pi_p", "qv")}
        solver.upload_state(st)
        solver.set_lbc_target(lbc)
        solver.step_rk3(dt_s=10.0, n_steps=40)
        out = solver.download_state()
    finally:
        solver.close()

    qv_end = out["qv"]
    assert np.all(np.isfinite(qv_end))
    assert float(np.min(qv_end)) >= 0.0
    west_min = float(np.min(qv_end[:, 2:-2, 0:10]))
    east_max = float(np.max(qv_end[:, 2:-2, -3:]))
    assert west_min > 0.0095, f"Western rim moisture collapsed: min={west_min:.6f}"
    assert east_max < 0.0110, f"Eastern outflow tracer pileup: max={east_max:.6f}"


def test_multi_tile_matches_untiled_with_qv_lbc() -> None:
    """Verify 2x2 tiled solver matches 1-tile untiled bit-for-bit when lbc includes qv."""
    from tempest_v2.cuda_bridge import DEFAULT_MAX_VRAM_BYTES, NonHydrostaticGPUSolver, estimate_vram_bytes

    nz, ny, nx = 12, 56, 56
    dx = dy = 1500.0
    vgrid = VerticalGrid(nz=nz, z_top_m=15_000.0, stretch_power=1.2)
    y = (np.arange(ny, dtype=np.float64) - 0.5 * (ny - 1)) * dy
    x = (np.arange(nx, dtype=np.float64) - 0.5 * (nx - 1)) * dx
    yy, xx = np.meshgrid(y, x, indexing="ij")
    hs = 200.0 * np.exp(-(xx**2 + yy**2) / (15_000.0**2))
    vram_2x2 = estimate_vram_bytes(nz, ny // 2 + 16, nx // 2 + 16, moist=True, has_lbc=True, has_iau=False, has_physics=False)

    def _run_case(max_vram: int):
        solver = NonHydrostaticGPUSolver(
            vgrid,
            dx,
            dy,
            hs,
            has_moist=True,
            has_physics=False,
            bc_x=2,
            bc_y=2,
            use_coriolis=False,
            lbc_rim_width=6,
            lbc_tau_max_s=0.04,
            max_vram_bytes=max_vram,
            halo=16,
        )
        try:
            st = solver.empty_state()
            st["u"][:] = 12.0
            st["v"][:] = -3.0
            st["qv"][:] = 0.009
            lbc = {k: st[k].copy() for k in ("u", "v", "w", "theta_p", "pi_p", "qv")}
            lbc["qv"][:, :, :6] = 0.012
            solver.upload_state(st)
            solver.set_lbc_target(lbc)
            solver.step_rk3(dt_s=6.0, n_steps=4)
            return solver.num_tiles, solver.download_state()
        finally:
            solver.close()

    n_untiled, out_untiled = _run_case(DEFAULT_MAX_VRAM_BYTES)
    n_tiled, out_tiled = _run_case(vram_2x2)
    assert n_untiled == 1
    assert n_tiled == 4
    for key in out_untiled:
        diff = float(np.max(np.abs(out_tiled[key] - out_untiled[key])))
        assert diff < 1e-6, f"Tiled 3D field '{key}' diverged with qv LBC: max_abs_diff={diff:.3e}"


def test_d3_default_2p5km_1tile_vram_fit() -> None:
    """Verify default 'd3' domain is 2.5 km (432x432) fitting 1x1 GPU tile at nz=137 while keeping d3_1p5km."""
    from tempest_v2.cuda_bridge import DEFAULT_MAX_VRAM_BYTES, plan_horizontal_tiling
    from tempest_v2.projection import get_domain_spec

    d3 = get_domain_spec("d3", nz=137)
    assert d3.name == "d3_benelux_de_2p5km"
    assert (d3.nz, d3.ny, d3.nx) == (137, 432, 432)
    assert d3.dx_m == 2500.0
    assert d3.dy_m == 2500.0

    nt_y, nt_x, est = plan_horizontal_tiling(
        137,
        432,
        432,
        moist=True,
        has_lbc=True,
        has_iau=True,
        has_physics=True,
        max_vram_bytes=DEFAULT_MAX_VRAM_BYTES,
    )
    assert (nt_y, nt_x) == (1, 1), f"Expected 1x1 tile for D3 2.5km x137, got {nt_y}x{nt_x} ({est/1048576:.0f} MiB)"

    d3_1p5 = get_domain_spec("d3_1p5km", nz=137)
    assert d3_1p5.name == "d3_benelux_de_1p5km"
    assert (d3_1p5.nz, d3_1p5.ny, d3_1p5.nx) == (137, 720, 720)
    assert d3_1p5.dx_m == 1500.0


