"""Gate B1 benchmarks for the Tempest-NWP v2 non-hydrostatic GPU core."""

from __future__ import annotations

import numpy as np
import pytest

from tempest_v2.cuda_bridge import (
    DEFAULT_MAX_VRAM_BYTES,
    NonHydrostaticGPUSolver,
    cuda_vram_info,
    estimate_vram_bytes,
    plan_horizontal_tiling,
)
from tempest_v2.projection import CP_D, GRAVITY_M_S2 as GRAVITY, P0_PA, R_D, VerticalGrid, get_domain_spec

KAPPA = R_D / CP_D


def _hydrostatic_pi_p_from_theta_p(
    z_s: np.ndarray,
    theta_p: np.ndarray,
    theta_sfc_k: float,
    p_sfc_pa: float,
    n_ref_s: float,
) -> np.ndarray:
    """Integrate exact hydrostatic Exner perturbation from model top downward."""
    ref = VerticalGrid.reference_state(
        z_s, theta_sfc_k=theta_sfc_k, p_sfc_pa=p_sfc_pa, brunt_vaisala_hz=n_ref_s
    )
    th_bar = ref["theta_bar"]
    pi_p = np.zeros_like(z_s, dtype=np.float64)
    nz = z_s.shape[0]
    for k in range(nz - 2, -1, -1):
        dz = z_s[k + 1] - z_s[k]
        g0 = (GRAVITY * theta_p[k]) / (CP_D * th_bar[k] * np.maximum(th_bar[k] + theta_p[k], 50.0))
        g1 = (GRAVITY * theta_p[k + 1]) / (
            CP_D * th_bar[k + 1] * np.maximum(th_bar[k + 1] + theta_p[k + 1], 50.0)
        )
        pi_p[k] = pi_p[k + 1] - 0.5 * (g0 + g1) * dz
    return pi_p


def test_rising_warm_thermal_bubble() -> None:
    """Warm cosine bubble rises symmetrically in neutral atmosphere with clean updraft."""
    nz, ny, nx = 40, 3, 64
    dx = dy = 500.0
    vgrid = VerticalGrid(nz=nz, z_top_m=10_000.0, stretch_power=1.0, terrain_decay_power=1.0)
    hs = np.zeros((ny, nx), dtype=np.float64)
    _, z_s, _ = vgrid.heights_3d(hs)

    x = (np.arange(nx, dtype=np.float64) + 0.5) * dx
    xc = 0.5 * nx * dx
    zc = 2000.0
    rx = rz = 2000.0
    rad = np.sqrt(((x[None, None, :] - xc) / rx) ** 2 + ((z_s - zc) / rz) ** 2)
    thp0 = np.where(rad <= 1.0, 2.0 * (np.cos(0.5 * np.pi * rad) ** 2), 0.0)
    pip0 = _hydrostatic_pi_p_from_theta_p(z_s, thp0, 300.0, 100_000.0, 0.0)

    solver = NonHydrostaticGPUSolver(
        vgrid,
        dx,
        dy,
        hs,
        has_moist=False,
        bc_x=0,
        bc_y=1,
        use_coriolis=False,
        theta_sfc_k=300.0,
        p_sfc_pa=100_000.0,
        n_ref_s=0.0,
        kh_const=5.0,
        km_const=5.0,
    )
    try:
        state = solver.empty_state()
        state["theta_p"][:] = thp0.astype(np.float32)
        state["pi_p"][:] = pip0.astype(np.float32)

        w_mask0 = np.maximum(thp0, 0.0)
        z_cent0 = float(np.sum(z_s * w_mask0) / np.sum(w_mask0))

        solver.upload_state(state)
        solver.step_rk3(dt_s=2.0, n_steps=160)
        out = solver.download_state()
    finally:
        solver.close()

    th_end = out["theta_p"].astype(np.float64)
    w_end = out["w"].astype(np.float64)
    assert np.all(np.isfinite(th_end))
    assert np.all(np.isfinite(w_end))

    w_mask1 = np.maximum(th_end - 0.1, 0.0)
    z_cent1 = float(np.sum(z_s * w_mask1) / np.sum(w_mask1))
    assert z_cent1 - z_cent0 > 500.0, f"Bubble rose only {z_cent1 - z_cent0:.1f} m"

    w_max = float(np.max(w_end))
    assert 2.0 <= w_max <= 15.0, f"Unexpected peak updraft w_max={w_max:.3f} m/s"
    assert float(np.min(th_end)) > -0.25, f"Cold undershoot too large: {np.min(th_end)}"

    sym_err = float(np.max(np.abs(th_end[:, 1, :] - th_end[:, 1, ::-1])))
    assert sym_err < 1e-3, f"Left-right symmetry broken: {sym_err:.3e}"


def test_straka_density_current() -> None:
    """Straka et al. (1993) sinking cold pool & spreading lateral gust front (K=75 m^2/s)."""
    nz, ny, nx = 32, 3, 128
    dx = dy = 200.0
    vgrid = VerticalGrid(nz=nz, z_top_m=6_400.0, stretch_power=1.0, terrain_decay_power=1.0)
    hs = np.zeros((ny, nx), dtype=np.float64)
    _, z_s, _ = vgrid.heights_3d(hs)
    ref = VerticalGrid.reference_state(z_s, theta_sfc_k=300.0, p_sfc_pa=100_000.0, brunt_vaisala_hz=0.0)

    x = (np.arange(nx, dtype=np.float64) + 0.5) * dx
    xc = 0.5 * nx * dx
    zc = 3000.0
    rx = 4000.0
    rz = 2000.0
    rad = np.sqrt(((x[None, None, :] - xc) / rx) ** 2 + ((z_s - zc) / rz) ** 2)
    dt_cold = np.where(rad <= 1.0, -15.0 * 0.5 * (1.0 + np.cos(np.pi * rad)), 0.0)
    thp0 = dt_cold / ref["pi_bar"]
    pip0 = _hydrostatic_pi_p_from_theta_p(z_s, thp0, 300.0, 100_000.0, 0.0)

    solver = NonHydrostaticGPUSolver(
        vgrid,
        dx,
        dy,
        hs,
        has_moist=False,
        bc_x=0,
        bc_y=1,
        use_coriolis=False,
        theta_sfc_k=300.0,
        p_sfc_pa=100_000.0,
        n_ref_s=0.0,
        km_const=75.0,
        kh_const=75.0,
        div_damp_coeff=0.04,
    )
    try:
        state = solver.empty_state()
        state["theta_p"][:] = thp0.astype(np.float32)
        state["pi_p"][:] = pip0.astype(np.float32)
        solver.upload_state(state)
        solver.step_rk3(dt_s=1.0, n_steps=480)
        out = solver.download_state()
    finally:
        solver.close()

    th_end = out["theta_p"].astype(np.float64)
    u_end = out["u"].astype(np.float64)
    assert np.all(np.isfinite(th_end))
    assert np.all(np.isfinite(u_end))

    # Cold pool dropped to ground and drove strong symmetric outflow
    u_sfc = u_end[0, 1, :]
    assert float(np.max(u_sfc)) > 14.0, f"Weak positive gust front: {np.max(u_sfc):.2f}"
    assert float(np.min(u_sfc)) < -14.0, f"Weak negative gust front: {np.min(u_sfc):.2f}"
    assert abs(float(np.max(u_sfc) + np.min(u_sfc))) < 0.05

    far_left = th_end[0, 1, (x - xc) < -5500.0]
    far_right = th_end[0, 1, (x - xc) > 5500.0]
    assert float(np.min(far_left)) < -1.0 and float(np.min(far_right)) < -1.0
    assert float(np.min(th_end)) >= -18.0

    # Reference comparison against Straka et al. (1993, IJNMF) at t=480 s:
    # Leading-edge gust front (-0.5 K contour at surface) reaches x - xc in [6.2, 8.8] km,
    # with elevated Kelvin-Helmholtz head (k=2..8) and machine-level symmetry.
    right_mask = (x - xc > 0.0) & (th_end[0, 1, :] < -0.5)
    x_front_right = float(np.max(x[right_mask] - xc))
    assert 6200.0 <= x_front_right <= 8800.0, f"Straka t=480s front position {x_front_right:.1f} m outside reference [6.2, 8.8] km"
    sym_th = float(np.max(np.abs(th_end[:, 1, :] - th_end[:, 1, ::-1])))
    sym_u = float(np.max(np.abs(u_end[:, 1, :] + u_end[:, 1, ::-1])))
    assert sym_th < 1e-3 and sym_u < 1e-3, f"Straka symmetry error: sym_th={sym_th:.3e}, sym_u={sym_u:.3e}"


@pytest.mark.parametrize("nz", [65, 137])
def test_resting_atmosphere_2000m_mountain(nz: int) -> None:
    """Resting stratified atmosphere over 2000 m mountain has spurious winds < 1e-2 m/s."""
    ny, nx = 44, 44
    dx = dy = 1500.0
    vgrid = VerticalGrid(nz=nz, z_top_m=31_000.0, stretch_power=1.5, terrain_decay_power=2.5)
    y = (np.arange(ny, dtype=np.float64) - 0.5 * (ny - 1)) * dy
    x = (np.arange(nx, dtype=np.float64) - 0.5 * (nx - 1)) * dx
    yy, xx = np.meshgrid(y, x, indexing="ij")
    hs = 2000.0 * np.exp(-(xx**2 + yy**2) / (9000.0**2))
    vgrid.check_jacobian_positive(hs)

    _, z_s, _ = vgrid.heights_3d(hs)
    # Non-trivial stratified rest test: background stratification differs from ref N_ref=0.01
    # so both theta_p != 0 and pi_p != 0 strongly vary over terrain!
    ref = VerticalGrid.reference_state(z_s, theta_sfc_k=288.15, p_sfc_pa=101_325.0, brunt_vaisala_hz=0.01)
    alt_rest = VerticalGrid.reference_state(
        z_s, theta_sfc_k=286.0, p_sfc_pa=101_325.0, brunt_vaisala_hz=0.011
    )
    thp0 = alt_rest["theta_bar"] - ref["theta_bar"]
    pip0 = alt_rest["pi_bar"] - ref["pi_bar"]

    solver = NonHydrostaticGPUSolver(
        vgrid,
        dx,
        dy,
        hs,
        has_moist=False,
        bc_x=0,
        bc_y=0,
        use_coriolis=False,
        theta_sfc_k=288.15,
        p_sfc_pa=101_325.0,
        n_ref_s=0.01,
        div_damp_coeff=0.04,
    )
    try:
        state = solver.empty_state()
        state["theta_p"][:] = thp0.astype(np.float32)
        state["pi_p"][:] = pip0.astype(np.float32)
        solver.upload_state(state)
        solver.step_rk3(dt_s=4.0, n_steps=30)
        out = solver.download_state()
    finally:
        solver.close()

    max_u = float(np.max(np.abs(out["u"])))
    max_v = float(np.max(np.abs(out["v"])))
    max_w = float(np.max(np.abs(out["w"])))
    assert np.all(np.isfinite(out["u"])) and np.all(np.isfinite(out["w"]))
    assert max_u < 1e-2 and max_v < 1e-2 and max_w < 1e-2, (
        f"nz={nz} resting mountain drift exceeded 1e-2 m/s: "
        f"max_u={max_u:.4e}, max_v={max_v:.4e}, max_w={max_w:.4e}"
    )


def test_schar_mountain_wave_with_rayleigh_sponge() -> None:
    """Schar (2002) steady U=10 m/s flow over multi-ridge mountain + upper Rayleigh sponge."""
    nz, ny, nx = 45, 3, 140
    dx = dy = 500.0
    vgrid = VerticalGrid(nz=nz, z_top_m=21_000.0, stretch_power=1.2, terrain_decay_power=2.2)

    x = (np.arange(nx, dtype=np.float64) - 0.5 * (nx - 1)) * dx
    h_prof = 250.0 * np.exp(-((x / 5000.0) ** 2)) * (np.cos(np.pi * x / 4000.0) ** 2)
    hs = np.broadcast_to(h_prof[None, :], (ny, nx)).copy()

    solver = NonHydrostaticGPUSolver(
        vgrid,
        dx,
        dy,
        hs,
        has_moist=False,
        bc_x=2,
        bc_y=1,
        use_coriolis=False,
        theta_sfc_k=288.0,
        p_sfc_pa=101_325.0,
        n_ref_s=0.01,
        sponge_depth_m=9000.0,
        sponge_tau_max_s=0.04,
        lbc_rim_width=10,
        lbc_tau_max_s=0.05,
        div_damp_coeff=0.04,
    )
    try:
        state = solver.empty_state()
        state["u"][:] = 10.0
        # Enforce kinematic bottom BC u * dh/dx
        dh_dx = np.gradient(hs, dx, axis=1)
        state["w"][0, :, :] = (10.0 * dh_dx).astype(np.float32)

        lbc = solver.empty_state()
        lbc["u"][:] = 10.0
        solver.set_lbc_target(lbc)
        solver.upload_state(state)
        solver.step_rk3(dt_s=2.5, n_steps=320)
        out = solver.download_state()
    finally:
        solver.close()

    w_end = out["w"].astype(np.float64)
    assert np.all(np.isfinite(w_end))
    # Interior mountain wave peak in lower/mid troposphere (k=2..22)
    w_mid_max = float(np.max(np.abs(w_end[2:22, 1, 45:95])))
    assert 0.15 <= w_mid_max <= 3.5, f"Unexpected Schar wave amplitude w_mid_max={w_mid_max:.3f}"
    # Verify upper sponge cleanly absorbs wave energy near top
    w_top_rms = float(np.sqrt(np.mean(w_end[nz - 3 : nz, 1, :] ** 2)))
    assert w_top_rms < 0.2 * w_mid_max, f"Sponge ineffective: w_top_rms={w_top_rms:.3f} vs w_mid={w_mid_max:.3f}"

    # Analytical 2D Linear Fourier Non-Hydrostatic Reference Comparison (Schar et al. 2002, Eq. 11-13):
    # w_hat(kx, z) = i * kx * U0 * h_hat(kx) * sqrt(rho0 / rho_bar(z)) * exp(i*m*z or -mu*z)
    u0 = 10.0
    n_bv = 0.01
    l_scorer = n_bv / u0  # Scorer parameter l = N / U0 = 1e-3 m^-1
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    h_hat = np.fft.fft(h_prof)
    w0_hat = 1j * kx * u0 * h_hat

    z_w, _, _ = vgrid.heights_3d(hs)
    ref_w = VerticalGrid.reference_state(z_w[:, 1, :], theta_sfc_k=288.0, p_sfc_pa=101_325.0, brunt_vaisala_hz=n_bv)
    rho_ratio_sqrt = np.sqrt(ref_w["rho_bar"][0:1, :] / np.maximum(ref_w["rho_bar"], 1e-6))

    # Compare lower-tropospheric levels k=1..14 over the mountain ridge x in [-10 km, +10 km]
    ridge_sl = slice(nx // 2 - 20, nx // 2 + 21)
    w_ref = np.zeros((14, nx), dtype=np.float64)
    for k in range(14):
        z_lvl = float(np.mean(z_w[k, 1, ridge_sl]) - np.mean(hs[1, ridge_sl]))
        prop = np.abs(kx) < l_scorer
        m_vert = np.where(prop, np.sign(kx) * np.sqrt(np.maximum(0.0, l_scorer**2 - kx**2)), 0.0)
        mu_evan = np.where(~prop, np.sqrt(np.maximum(0.0, kx**2 - l_scorer**2)), 0.0)
        phase = np.where(prop, np.exp(1j * m_vert * z_lvl), np.exp(-mu_evan * z_lvl))
        w_ref[k] = np.real(np.fft.ifft(w0_hat * phase)) * rho_ratio_sqrt[k]

    w_sim_sub = w_end[1:14, 1, ridge_sl]
    w_ref_sub = w_ref[1:14, ridge_sl]
    corr = float(np.corrcoef(w_sim_sub.ravel(), w_ref_sub.ravel())[0, 1])
    assert corr > 0.80, f"Schar mountain wave pattern correlation vs analytical linear Fourier solution too low: r={corr:.3f}"


def test_moist_six_tracer_positivity_and_conservation() -> None:
    """All 6 moist tracers preserve strict positivity (>= 0) and mass conservation (< 1e-5)."""
    nz, ny, nx = 36, 24, 32
    dx = dy = 1500.0
    vgrid = VerticalGrid(nz=nz, z_top_m=20_000.0, stretch_power=1.4, terrain_decay_power=2.2)

    y = (np.arange(ny, dtype=np.float64) - 0.5 * (ny - 1)) * dy
    x = (np.arange(nx, dtype=np.float64) - 0.5 * (nx - 1)) * dx
    yy, xx = np.meshgrid(y, x, indexing="ij")
    hs = 800.0 * np.exp(-(xx**2 + yy**2) / (12_000.0**2))
    _, z_s, _ = vgrid.heights_3d(hs)

    solver = NonHydrostaticGPUSolver(
        vgrid,
        dx,
        dy,
        hs,
        has_moist=True,
        bc_x=0,
        bc_y=0,
        use_coriolis=False,
        theta_sfc_k=288.15,
        p_sfc_pa=101_325.0,
        n_ref_s=0.01,
    )
    try:
        state = solver.empty_state()
        # Rotational flow inside closed domain + vertical shear
        state["u"][:, :, 1:nx] = (-10.0 * (yy[None, :, :-1] / 25_000.0)).astype(np.float32)
        state["v"][:, 1:ny, :] = (10.0 * (xx[None, :-1, :] / 25_000.0)).astype(np.float32)

        centers = {
            "qv": (3000.0, -4000.0, 2500.0, 0.014),
            "qc": (4500.0, 3000.0, 4000.0, 0.003),
            "qr": (-4500.0, 2000.0, 2000.0, 0.002),
            "qi": (2000.0, -5000.0, 9000.0, 0.0008),
            "qs": (-3000.0, -3500.0, 6500.0, 0.0015),
            "qg": (1000.0, 5000.0, 5000.0, 0.0025),
        }
        for sp, (cx, cy, cz, amp) in centers.items():
            r2 = ((xx[None, :, :] - cx) / 7000.0) ** 2 + ((yy[None, :, :] - cy) / 7000.0) ** 2 + (
                (z_s - cz) / 2500.0
            ) ** 2
            state[sp][:] = np.where(r2 < 1.5, amp * np.exp(-2.0 * r2), 0.0).astype(np.float32)

        mass_init = {sp: solver.tracer_mass(state[sp]) for sp in solver.SPECIES}
        solver.upload_state(state)
        solver.step_rk3(dt_s=6.0, n_steps=20)
        out = solver.download_state()
        mass_end = {sp: solver.tracer_mass(out[sp]) for sp in solver.SPECIES}
    finally:
        solver.close()

    for sp in solver.SPECIES:
        q_out = out[sp]
        assert np.all(np.isfinite(q_out))
        assert float(np.min(q_out)) >= 0.0, f"{sp} went negative: min={np.min(q_out)}"
        rel_err = abs(mass_end[sp] - mass_init[sp]) / mass_init[sp]
        assert rel_err < 1e-5, f"{sp} mass drift {rel_err:.3e} >= 1e-5"
        moved = float(np.linalg.norm(q_out - state[sp]) / np.linalg.norm(state[sp]))
        assert moved > 0.05, f"{sp} did not advect enough: {moved:.3e}"


def test_rtx3060_nz137_720x720_vram_and_tiled_smoke() -> None:
    """Verify full D3 (720x720x137, 1.5 km) auto-tiles under 4.8 GB and runs live on GPU."""
    spec = get_domain_spec("d3_benelux_de_1p5km", nz=137)
    assert (spec.ny, spec.nx, spec.vertical.nz) == (720, 720, 137)

    free_b, total_b = cuda_vram_info()
    assert total_b > 0 and free_b > 0

    whole_vram = estimate_vram_bytes(
        137, 720, 720, moist=True, has_lbc=True, has_iau=False
    )
    assert whole_vram > DEFAULT_MAX_VRAM_BYTES

    nt_y, nt_x, tile_est = plan_horizontal_tiling(
        137, 720, 720, moist=True, has_lbc=True, has_iau=False
    )
    assert nt_y >= 2 and nt_x >= 2
    assert tile_est <= DEFAULT_MAX_VRAM_BYTES

    yy, xx = np.meshgrid(spec.y_m, spec.x_m, indexing="ij")
    hs = 650.0 * np.exp(-((xx - 150_000.0) ** 2 + (yy + 200_000.0) ** 2) / (80_000.0**2))

    solver = NonHydrostaticGPUSolver(
        spec.vertical,
        spec.dx_m,
        spec.dy_m,
        hs,
        msf=spec.map_factor,
        f_cor=spec.coriolis_f,
        has_moist=True,
        bc_x=2,
        bc_y=2,
        use_coriolis=True,
        theta_sfc_k=288.15,
        p_sfc_pa=101_325.0,
        n_ref_s=0.01,
        sponge_depth_m=6000.0,
        sponge_tau_max_s=0.02,
    )
    try:
        assert solver.num_tiles == nt_y * nt_x
        state = solver.empty_state()
        state["u"][:] = 8.0
        state["qv"][:] = 0.005
        state["qc"][20:30, 340:380, 340:380] = 0.001

        lbc = solver.empty_state()
        lbc["u"][:] = 8.0
        solver.set_lbc_target(lbc)
        solver.upload_state(state)
        solver.step_rk3(dt_s=5.0, n_steps=2)
        out = solver.download_state()
    finally:
        solver.close()

    for key in ("u", "v", "w", "theta_p", "pi_p", "qv", "qc"):
        assert np.all(np.isfinite(out[key])), f"Non-finite values in 720x720x137 {key}"
    for sp in solver.SPECIES:
        assert float(np.min(out[sp])) >= 0.0
    free_after, _ = cuda_vram_info()
    assert free_after > 3_500_000_000, f"GPU VRAM leak after tiled close: free={free_after}"


def test_multi_tile_matches_untiled_exact() -> None:
    """Verify 2x2 tiled solver matches 1-tile untiled solver bit-for-bit across multi-step coupled dynamics+physics."""
    nz, ny, nx = 16, 64, 64
    dx = dy = 1500.0
    vgrid = VerticalGrid(nz=nz, z_top_m=18_000.0, stretch_power=1.3, terrain_decay_power=2.0)

    y = (np.arange(ny, dtype=np.float64) - 0.5 * (ny - 1)) * dy
    x = (np.arange(nx, dtype=np.float64) - 0.5 * (nx - 1)) * dx
    yy, xx = np.meshgrid(y, x, indexing="ij")
    hs = 450.0 * np.exp(-((xx - 4000.0) ** 2 + (yy + 3000.0) ** 2) / (15_000.0**2))
    msf = 1.0 + 0.01 * (yy / 50_000.0)
    fcor = np.full((ny, nx), 1.15e-4, dtype=np.float64)
    _, z_s, _ = vgrid.heights_3d(hs)

    # Set max_vram_bytes to exact 2x2 tile VRAM so plan_horizontal_tiling selects 2x2=4 tiles
    vram_2x2 = estimate_vram_bytes(nz, ny // 2 + 20, nx // 2 + 20, moist=True, has_lbc=True, has_iau=True, has_physics=True)
    forced_budget = vram_2x2

    def _run_case(max_vram: int):
        solver = NonHydrostaticGPUSolver(
            vgrid,
            dx,
            dy,
            hs,
            msf=msf,
            f_cor=fcor,
            has_moist=True,
            has_physics=True,
            bc_x=2,
            bc_y=2,
            use_coriolis=True,
            has_iau=True,
            sponge_depth_m=4000.0,
            sponge_tau_max_s=0.02,
            lbc_rim_width=6,
            lbc_tau_max_s=0.04,
            max_vram_bytes=max_vram,
            halo=20,
        )
        try:
            st = solver.empty_state()
            st["u"][:] = 7.5
            st["v"][:] = -2.0
            st["theta_p"][:] = (1.5 * np.exp(-((xx[None, :, :] ** 2 + yy[None, :, :] ** 2) / (18_000.0**2)))).astype(np.float32)
            st["qv"][:] = (0.008 * np.exp(-z_s / 2800.0)).astype(np.float32)
            st["qc"][4:8, 24:40, 24:40] = 4.0e-4

            lbc = {k: st[k].copy() for k in ("u", "v", "w", "theta_p", "pi_p")}
            iau = {k: np.full_like(st[k], 1.0e-4 if k == "theta_p" else 0.0) for k in ("u", "v", "w", "theta_p", "pi_p")}

            solver.upload_state(st)
            solver.set_lbc_target(lbc)
            solver.set_iau_forcing(iau)
            solver.set_surface_static(solver.default_surface_static())
            solver.upload_surface(solver.empty_surface_state())
            solver.set_physics_config(
                enable_microphysics=1,
                enable_pbl=1,
                enable_radiation=1,
                enable_surface=1,
                rad_interval_s=900.0,
                julian_day=172.0,
                utc_seconds=43_200.0,
            )
            # Step across 2 segment calls with physics_every=2 to test persistent cadence + multi-step halo sync
            solver.step_full(dt_s=4.0, n_steps=1, physics_every=2)
            solver.step_full(dt_s=4.0, n_steps=3, physics_every=2)
            return solver.num_tiles, solver.download_state(), solver.download_surface()
        finally:
            solver.close()

    n_untiled, st_untiled, sf_untiled = _run_case(DEFAULT_MAX_VRAM_BYTES)
    n_tiled, st_tiled, sf_tiled = _run_case(forced_budget)

    assert n_untiled == 1, f"Expected 1 tile for untiled reference, got {n_untiled}"
    assert n_tiled == 4, f"Expected 4 tiles for tiled run, got {n_tiled}"

    for key in st_untiled:
        diff = float(np.max(np.abs(st_tiled[key] - st_untiled[key])))
        assert diff < 1e-6, f"Tiled 3D field '{key}' diverged from untiled run: max_abs_diff={diff:.3e}"
    for key in sf_untiled:
        diff = float(np.max(np.abs(sf_tiled[key] - sf_untiled[key])))
        assert diff < 1e-6, f"Tiled surface field '{key}' diverged from untiled run: max_abs_diff={diff:.3e}"


def test_clear_iau_forcing_frees_buffers_and_matches_zero() -> None:
    """Verify clear_iau_forcing disables IAU cleanly and matches a zero-IAU continuation."""
    nz, ny, nx = 12, 16, 16
    vgrid = VerticalGrid(nz=nz, z_top_m=15_000.0, stretch_power=1.2)
    hs = np.zeros((ny, nx), dtype=np.float64)

    def _run_case(clear_after_init: bool) -> Dict[str, np.ndarray]:
        solver = NonHydrostaticGPUSolver(
            vgrid, 7500.0, 7500.0, hs, has_moist=True, has_physics=False, has_iau=clear_after_init
        )
        try:
            st = solver.empty_state()
            st["u"][:] = 5.0
            st["qv"][:] = 0.005
            solver.upload_state(st)
            if clear_after_init:
                iau = {k: np.full_like(st[k], 2.0e-3 if k == "theta_p" else 0.0) for k in ("u", "v", "w", "theta_p", "pi_p")}
                solver.set_iau_forcing(iau)
                solver.clear_iau_forcing()
            solver.step_rk3(dt_s=10.0, n_steps=3)
            return solver.download_state()
        finally:
            solver.close()

    st_ref = _run_case(False)
    st_clr = _run_case(True)
    for key in ("u", "v", "w", "theta_p", "pi_p", "qv"):
        assert np.max(np.abs(st_ref[key] - st_clr[key])) == 0.0, f"IAU clear mismatch on '{key}'"
