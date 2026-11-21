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
