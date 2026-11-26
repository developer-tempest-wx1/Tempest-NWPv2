"""Verification tests for Tempest-NWP mixed-precision CUDA kernels (FP64 state/accumulators + FP32 shadows)."""

from __future__ import annotations

import ctypes
import numpy as np

from tempest_v2.cuda_bridge import NonHydrostaticGPUSolver
from tempest_v2.ocean_si3_bridge import NemoSi3GPUSolver, freezing_temp_c
from tempest_v2.projection import DomainSpec, VerticalGrid


def _qsat_water_np(t_k: np.ndarray, p_pa: np.ndarray) -> np.ndarray:
    tc = np.clip(t_k - 273.15, -80.0, 60.0)
    es = 611.2 * np.exp(17.67 * tc / (tc + 243.5))
    return 0.62198 * es / np.maximum(p_pa - 0.37802 * es, 100.0)


def _make_ocean_spec(ny: int = 12, nx: int = 12, dx_m: float = 25_000.0, lat0: float = 65.0) -> DomainSpec:
    vgrid = VerticalGrid(nz=10, z_top_m=15_000.0, stretch_power=1.2)
    x_m = np.arange(nx, dtype=np.float64) * dx_m
    y_m = np.arange(ny, dtype=np.float64) * dx_m
    lat2d = np.full((ny, nx), lat0, dtype=np.float64)
    lon2d = np.broadcast_to(np.linspace(-2.0, 2.0, nx, dtype=np.float64), (ny, nx)).copy()
    return DomainSpec(
        name="mixed_prec_oce_test",
        ny=ny,
        nx=nx,
        dx_m=dx_m,
        dy_m=dx_m,
        is_projected=True,
        is_periodic_lon=False,
        projection=None,
        x_m=x_m,
        y_m=y_m,
        lat2d=lat2d,
        lon2d=lon2d,
        map_factor=np.ones((ny, nx), dtype=np.float64),
        coriolis_f=np.full((ny, nx), 2.0 * 7.2921159e-5 * np.sin(np.radians(lat0)), dtype=np.float64),
        vertical=vgrid,
        is_spherical=False,
    )


def test_soil_layer4_fp64_accumulation() -> None:
    """Deep soil layer 4 (dz=1.89 m) accumulates small dT (~1.6e-6 K/step) in FP64 without FP32 stagnation."""
    nz, ny, nx = 16, 16, 16
    vg = VerticalGrid(nz=nz, z_top_m=12_000.0, stretch_power=1.2)
    hs = np.zeros((ny, nx), dtype=np.float64)
    solver = NonHydrostaticGPUSolver(
        vg,
        dx_m=2500.0,
        dy_m=2500.0,
        hs_m=hs,
        msf=np.ones((ny, nx)),
        f_cor=np.full((ny, nx), 1e-4),
        has_moist=True,
        has_physics=True,
        use_coriolis=False,
        bc_x=2,
        bc_y=2,
    )
    try:
        solver.upload_state(solver.empty_state())
        sstatic = solver.default_surface_static(lsm=1.0)
        sstatic["soil_type"][:] = 9.0
        solver.set_surface_static(sstatic)
        solver.set_physics_config(
            enable_microphysics=0,
            enable_pbl=0,
            enable_radiation=1,
            enable_surface=1,
            julian_day=172.0,
            utc_seconds=43_200.0,
        )
        sf0 = solver.empty_surface_state()
        sf0["soil_t"][2][:] = 285.2
        sf0["soil_t"][3][:] = 285.0
        solver.upload_surface(sf0)
        solver.step_physics(dt_s=36.0, n_steps=300)
        sf1 = solver.download_surface()
        assert sf1["soil_t"].dtype == np.float64
        assert sf1["soil_m"].dtype == np.float64
        assert np.all(np.isfinite(sf1["soil_t"]))
        assert np.all(np.isfinite(sf1["soil_m"]))
        d_layer4 = float(np.mean(sf1["soil_t"][3]) - 285.0)
        assert d_layer4 > 2.0e-4, f"Deep soil layer 4 stagnated or under-accumulated: delta={d_layer4:.3e} K"
    finally:
        solver.close()


def test_precip_accumulator_fp64_on_large_offset() -> None:
    """Light precipitation falling onto 100 mm offset (< FP32 half-ULP/step) accumulates & conserves mass in FP64."""
    nz, ny, nx = 20, 8, 8
    dx = 2000.0
    vg = VerticalGrid(nz=nz, z_top_m=10_000.0, stretch_power=1.2)
    hs = np.zeros((ny, nx), dtype=np.float64)
    solver = NonHydrostaticGPUSolver(
        vg,
        dx_m=dx,
        dy_m=dx,
        hs_m=hs,
        msf=np.ones((ny, nx)),
        f_cor=np.zeros((ny, nx)),
        has_moist=True,
        has_physics=True,
        use_coriolis=False,
        bc_x=0,
        bc_y=0,
    )
    try:
        _, z_s, _ = vg.heights_3d(hs)
        ref = VerticalGrid.reference_state(
            z_s,
            theta_sfc_k=solver.theta_sfc_k,
            p_sfc_pa=solver.p_sfc_pa,
            brunt_vaisala_hz=solver.n_ref_s,
        )
        t_bar = ref["theta_bar"] * ref["pi_bar"]
        st0 = solver.empty_state()
        qsat0 = _qsat_water_np(t_bar, ref["p_bar"])
        st0["qv"][:10] = (1.02 * qsat0[:10]).astype(np.float32)
        st0["qv"][10:] = (0.95 * qsat0[10:]).astype(np.float32)
        st0["qr"][:8] = 6.0e-7  # per 0.5s step << 3.81e-6 mm FP32 half-ULP at 100.0 mm
        solver.upload_state(st0)

        sstatic = solver.default_surface_static(lsm=1.0)
        solver.set_surface_static(sstatic)
        solver.set_physics_config(
            enable_radiation=0,
            enable_surface=0,
            enable_pbl=0,
            enable_microphysics=1,
            cy46_physics=1,
        )
        sf0 = solver.empty_surface_state()
        offset_mm = 100.0
        sf0["rain_acc"][:] = offset_mm
        solver.upload_surface(sf0)

        q0 = (
            st0["qv"].astype(np.float64)
            + st0["qc"]
            + st0["qr"]
            + st0["qi"]
            + st0["qs"]
            + st0["qg"]
        )
        m_air_0 = solver.tracer_mass(q0)
        m_acc_0 = float(np.sum(sf0["rain_acc"].astype(np.float64) * (dx * dx)))

        solver.step_physics(dt_s=0.5, n_steps=400)

        st1 = solver.download_state()
        sf1 = solver.download_surface()
        assert sf1["rain_acc"].dtype == np.float64
        assert sf1["snow_acc"].dtype == np.float64
        assert sf1["graupel_acc"].dtype == np.float64
        gain_mm = float(np.mean(sf1["rain_acc"] - offset_mm))
        assert gain_mm > 5.0e-5, f"Expected cumulative FP64 rain_acc > 5e-5 mm over 100.0 offset, got {gain_mm:.3e}"

        q1 = (
            st1["qv"].astype(np.float64)
            + st1["qc"]
            + st1["qr"]
            + st1["qi"]
            + st1["qs"]
            + st1["qg"]
        )
        m_air_1 = solver.tracer_mass(q1)
        m_acc_1 = float(
            np.sum(
                (sf1["rain_acc"].astype(np.float64) + sf1["snow_acc"] + sf1["graupel_acc"])
                * (dx * dx)
            )
        )
        rel_err = abs((m_air_1 + m_acc_1) - (m_air_0 + m_acc_0)) / max(m_air_0, 1.0)
        assert rel_err < 1.0e-5, f"Precipitation mass conservation drift too large: rel_err={rel_err:.3e}"
    finally:
        solver.close()


def test_ocean_si3_fp64_tracers_and_salt_conservation() -> None:
    """NEMO ocean/SI3 keeps FP64 tracers, responds smoothly to weak flux, and conserves salt to < 1e-5."""
    spec = _make_ocean_spec(ny=12, nx=12, dx_m=25_000.0, lat0=76.0)
    sstatic = {
        "lat_deg": spec.lat2d.astype(np.float32),
        "lon_deg": spec.lon2d.astype(np.float32),
        "lsm": np.zeros((spec.ny, spec.nx), dtype=np.float32),
        "lake_mask": np.zeros((spec.ny, spec.nx), dtype=np.float32),
        "bathy_depth_m": np.full((spec.ny, spec.nx), 400.0, dtype=np.float32),
    }

    # Subtest A: weak shortwave heating (tiny ~3e-6 K per 60s step < FP32 ULP at 285 K)
    forcing_warm = {
        "u10m": np.zeros((spec.ny, spec.nx), dtype=np.float32),
        "v10m": np.zeros((spec.ny, spec.nx), dtype=np.float32),
        "t2m": np.full((spec.ny, spec.nx), 285.0, dtype=np.float32),
        "q2m": np.full((spec.ny, spec.nx), 0.0085, dtype=np.float32),
        "mslp": np.full((spec.ny, spec.nx), 101_325.0, dtype=np.float32),
        "sw_down": np.full((spec.ny, spec.nx), 20.0, dtype=np.float32),
        "lw_down": np.full((spec.ny, spec.nx), 372.0, dtype=np.float32),
        "precip_liq": np.zeros((spec.ny, spec.nx), dtype=np.float32),
        "precip_ice": np.zeros((spec.ny, spec.nx), dtype=np.float32),
        "cld_fra": np.full((spec.ny, spec.nx), 0.5, dtype=np.float32),
    }
    with NemoSi3GPUSolver(spec, sstatic, nz=8, enable_dynamics=False) as oce:
        st0 = oce.empty_state()
        for k in range(8):
            st0["temp_k"][k] = 285.0
            st0["sal_psu"][k] = 35.0
        oce.upload_state(st0)
        for _ in range(120):
            oce.step(forcing_warm, dt_s=60.0)
        st_warm = oce.download_state()
        assert st_warm["temp_k"].dtype == np.float64
        assert st_warm["sal_psu"].dtype == np.float64
        assert st_warm["ssh"].dtype == np.float64
        assert np.all(np.isfinite(st_warm["temp_k"]))
        assert np.all(np.isfinite(st_warm["sal_psu"]))
        assert np.all(np.isfinite(st_warm["ssh"]))
        assert abs(float(np.mean(st_warm["temp_k"][0]) - 285.0)) > 5.0e-5

    # Subtest B: brine rejection + total ocean+ice salt conservation over 16 steps
    forcing_cold = {
        "u10m": np.full((spec.ny, spec.nx), 10.0, dtype=np.float32),
        "v10m": np.zeros((spec.ny, spec.nx), dtype=np.float32),
        "t2m": np.full((spec.ny, spec.nx), 243.15, dtype=np.float32),
        "q2m": np.full((spec.ny, spec.nx), 0.0003, dtype=np.float32),
        "mslp": np.full((spec.ny, spec.nx), 101_325.0, dtype=np.float32),
        "sw_down": np.zeros((spec.ny, spec.nx), dtype=np.float32),
        "lw_down": np.full((spec.ny, spec.nx), 165.0, dtype=np.float32),
        "precip_liq": np.zeros((spec.ny, spec.nx), dtype=np.float32),
        "precip_ice": np.zeros((spec.ny, spec.nx), dtype=np.float32),
        "cld_fra": np.full((spec.ny, spec.nx), 0.2, dtype=np.float32),
    }
    with NemoSi3GPUSolver(spec, sstatic, nz=8, enable_dynamics=False) as oce:
        st0 = oce.empty_state()
        for k in range(8):
            st0["temp_k"][k] = 273.15 + freezing_temp_c(34.0, 2.5) + 0.005
            st0["sal_psu"][k] = 34.0
        salt0 = float(np.sum(oce.total_salt_kg_m2(st0)))
        oce.upload_state(st0)
        for _ in range(16):
            oce.step(forcing_cold, dt_s=900.0)
        st1 = oce.download_state()
        salt1 = float(np.sum(oce.total_salt_kg_m2(st1)))
    rel_err = abs(salt1 - salt0) / max(salt0, 1.0)
    assert rel_err < 1.0e-5, f"FP64 ocean+ice salt conservation error rel_err={rel_err:.3e}"


def test_rebind_tile_grid_syncs_fp32_shadows() -> None:
    """nh_cuda_rebind_tile_grid refreshes FP32 shadow arrays so dynamics match a freshly constructed grid."""
    nz, ny, nx = 16, 32, 32
    vg = VerticalGrid(nz=nz, z_top_m=15_000.0, stretch_power=1.2)
    yy, xx = np.mgrid[0:ny, 0:nx]
    hs_flat = np.zeros((ny, nx), dtype=np.float64)
    msf_flat = np.ones((ny, nx), dtype=np.float64)
    fcor_flat = np.full((ny, nx), 1.0e-4, dtype=np.float64)

    hs_sloped = np.ascontiguousarray(
        600.0 * np.exp(-(((yy - 16.0) / 5.0) ** 2 + ((xx - 16.0) / 5.0) ** 2)), dtype=np.float64
    )
    msf_sloped = np.ascontiguousarray(1.0 + 0.08 * np.sin((yy / ny) * np.pi), dtype=np.float64)
    fcor_sloped = np.ascontiguousarray(1.1e-4 + 1.0e-5 * np.cos((yy / ny) * np.pi), dtype=np.float64)

    s_rebound = NonHydrostaticGPUSolver(
        vg,
        dx_m=2500.0,
        dy_m=2500.0,
        hs_m=hs_flat,
        msf=msf_flat,
        f_cor=fcor_flat,
        has_moist=False,
        has_physics=False,
        use_coriolis=True,
        bc_x=2,
        bc_y=2,
    )
    s_fresh = NonHydrostaticGPUSolver(
        vg,
        dx_m=2500.0,
        dy_m=2500.0,
        hs_m=hs_sloped,
        msf=msf_sloped,
        f_cor=fcor_sloped,
        has_moist=False,
        has_physics=False,
        use_coriolis=True,
        bc_x=2,
        bc_y=2,
    )
    try:
        assert s_rebound._handle is not None and s_fresh._handle is not None
        s_rebound.hs_m[:] = hs_sloped
        s_rebound.msf[:] = msf_sloped
        s_rebound.f_cor[:] = fcor_sloped
        cfg = s_rebound._make_cfg(s_rebound._tiles[0])
        s_rebound._raise_if_err(
            s_rebound.lib.nh_cuda_rebind_tile_grid(
                s_rebound._handle,
                ctypes.byref(cfg),
                hs_sloped.ctypes.data,
                msf_sloped.ctypes.data,
                fcor_sloped.ctypes.data,
            ),
            "rebind_tile_grid mixed-prec shadow test",
        )

        _, z_s, _ = vg.heights_3d(hs_sloped)
        st0 = s_fresh.empty_state()
        st0["u"][:] = (10.0 + 0.0005 * z_s[:, :, :1]).astype(np.float32)
        st0["v"][:] = 3.0

        s_rebound.upload_state(st0)
        s_fresh.upload_state(st0)
        s_rebound.step_rk3(dt_s=12.0, n_steps=5)
        s_fresh.step_rk3(dt_s=12.0, n_steps=5)
        out_r = s_rebound.download_state()
        out_f = s_fresh.download_state()

        assert float(np.max(np.abs(out_f["w"]))) > 0.05, "Expected mountain-wave vertical motion"
        for field in ("u", "v", "w", "theta_p", "pi_p"):
            diff = float(np.max(np.abs(out_r[field].astype(np.float64) - out_f[field].astype(np.float64))))
            assert diff < 1.0e-6, f"rebind_tile_grid {field} mismatch vs fresh solver: {diff:.3e}"
    finally:
        s_rebound.close()
        s_fresh.close()


def test_microphysics_pass2_water_conservation_tight() -> None:
    """Cached Pass 1 -> Pass 2 pure-FP32 fallout keeps strict FP64 water mass conservation (< 1e-5)."""
    nz, ny, nx = 24, 8, 8
    dx = 2000.0
    vg = VerticalGrid(nz=nz, z_top_m=14_000.0, stretch_power=1.25)
    hs = np.zeros((ny, nx), dtype=np.float64)
    solver = NonHydrostaticGPUSolver(
        vg,
        dx_m=dx,
        dy_m=dx,
        hs_m=hs,
        msf=np.ones((ny, nx)),
        f_cor=np.zeros((ny, nx)),
        has_moist=True,
        has_physics=True,
        use_coriolis=False,
        bc_x=0,
        bc_y=0,
    )
    try:
        st0 = solver.empty_state()
        st0["qv"][:] = 4.0e-3
        st0["qc"][8:16] = 1.2e-3
        st0["qr"][4:12] = 2.5e-3
        st0["qi"][14:21] = 5.0e-4
        st0["qs"][12:20] = 1.5e-3
        st0["qg"][10:18] = 1.0e-3
        q0 = (
            st0["qv"].astype(np.float64)
            + st0["qc"]
            + st0["qr"]
            + st0["qi"]
            + st0["qs"]
            + st0["qg"]
        )
        m_before = solver.tracer_mass(q0)

        solver.upload_state(st0)
        solver.set_surface_static(solver.default_surface_static(lsm=1.0))
        solver.upload_surface(solver.empty_surface_state())
        solver.set_physics_config(
            enable_radiation=0,
            enable_surface=0,
            enable_pbl=0,
            enable_microphysics=1,
            cy46_physics=1,
        )
        solver.step_physics(dt_s=10.0, n_steps=60)

        st1 = solver.download_state()
        sf1 = solver.download_surface()
        q1 = (
            st1["qv"].astype(np.float64)
            + st1["qc"]
            + st1["qr"]
            + st1["qi"]
            + st1["qs"]
            + st1["qg"]
        )
        m_after = solver.tracer_mass(q1)
        m_precip = float(
            np.sum(
                (sf1["rain_acc"].astype(np.float64) + sf1["snow_acc"] + sf1["graupel_acc"])
                * (dx * dx)
            )
        )
        rel_err = abs((m_after + m_precip) - m_before) / max(m_before, 1.0)
        assert m_precip > 0.0
        assert rel_err < 1.0e-5, f"Tight mixed-phase water conservation rel_err={rel_err:.3e}"
    finally:
        solver.close()
