"""Gate B2 validation suite for the Phase B2 CUDA physics suite."""

from __future__ import annotations

import time
import numpy as np
import pytest

from tempest_v2.cuda_bridge import DEFAULT_MAX_VRAM_BYTES, NonHydrostaticGPUSolver
from tempest_v2.projection import VerticalGrid, get_domain_spec


def _qsat_water_np(t_k: np.ndarray, p_pa: np.ndarray) -> np.ndarray:
    t = np.clip(t_k, 173.15, 373.15)
    es = 611.21 * np.exp(17.502 * (t - 273.15) / (t - 32.19))
    ec = np.minimum(es, 0.90 * p_pa)
    return 0.622 * ec / np.maximum(p_pa - 0.378 * ec, 100.0)


def _qsat_ice_np(t_k: np.ndarray, p_pa: np.ndarray) -> np.ndarray:
    t = np.clip(t_k, 150.0, 273.15)
    es = 611.21 * np.exp(22.587 * (t - 273.15) / (t + 0.7))
    ec = np.minimum(es, 0.90 * p_pa)
    return 0.622 * ec / np.maximum(p_pa - 0.378 * ec, 100.0)


def test_microphysics_water_conservation_slw_and_fallout() -> None:
    """Verify SLW preservation (-10C), homogeneous freeze (-43C), positivity, and water conservation."""
    nz, ny, nx = 48, 4, 4
    vgrid = VerticalGrid(nz=nz, z_top_m=15_000.0, stretch_power=1.2)
    hs = np.zeros((ny, nx), dtype=np.float64)
    theta_sfc = 288.15
    p_sfc = 101_325.0
    n_ref = 0.01

    solver = NonHydrostaticGPUSolver(
        vgrid,
        dx_m=1500.0,
        dy_m=1500.0,
        hs_m=hs,
        has_moist=True,
        has_physics=True,
        theta_sfc_k=theta_sfc,
        p_sfc_pa=p_sfc,
        n_ref_s=n_ref,
    )
    try:
        # Subtest A: SLW at -10C (263.15 K) and homogeneous glaciation below -37C (230.15 K)
        solver.set_physics_config(
            enable_microphysics=1,
            enable_pbl=0,
            enable_radiation=0,
            enable_surface=0,
        )
        _, z_s, _ = vgrid.heights_3d(hs)
        ref = VerticalGrid.reference_state(z_s, theta_sfc_k=theta_sfc, p_sfc_pa=p_sfc, brunt_vaisala_hz=n_ref)
        pi_ref = ref["pi_bar"]
        th_ref = ref["theta_bar"]
        p_ref = ref["p_bar"]

        state = solver.empty_state()
        # Inject warm/supercooled (-10C) layer k=8..14 and cold (-43C) layer k=28..32
        slw_sl = slice(8, 15)
        ice_sl = slice(28, 33)
        th_slw = 263.15 / pi_ref[slw_sl]
        th_ice = 230.15 / pi_ref[ice_sl]
        state["theta_p"][slw_sl] = (th_slw - th_ref[slw_sl]).astype(np.float32)
        state["theta_p"][ice_sl] = (th_ice - th_ref[ice_sl]).astype(np.float32)

        qsw_slw = _qsat_water_np(np.full_like(p_ref[slw_sl], 263.15), p_ref[slw_sl])
        qsi_ice = _qsat_ice_np(np.full_like(p_ref[ice_sl], 230.15), p_ref[ice_sl])
        state["qv"][slw_sl] = (qsw_slw + 1.2e-3).astype(np.float32)
        state["qc"][ice_sl] = 5.0e-4
        state["qv"][ice_sl] = (qsi_ice + 3.0e-4).astype(np.float32)

        solver.upload_state(state)
        solver.set_surface_static(solver.default_surface_static())
        solver.upload_surface(solver.empty_surface_state())
        solver.step_physics(dt_s=10.0, n_steps=3)

        out_a = solver.download_state()
        qc_slw = float(out_a["qc"][slw_sl].sum())
        qi_slw = float(out_a["qi"][slw_sl].sum())
        assert out_a["qc"][slw_sl].max() > 1.0e-4, f"Expected SLW survival, got max qc={out_a['qc'][slw_sl].max()}"
        assert qc_slw / (qc_slw + qi_slw + 1e-12) > 0.70, f"SLW glaciated too fast: qc={qc_slw}, qi={qi_slw}"

        assert out_a["qc"][ice_sl].max() < 1.0e-7, f"Expected homogeneous freezing at -43C, got qc={out_a['qc'][ice_sl].max()}"
        assert out_a["qi"][ice_sl].mean() > 1.0e-4, f"Expected qi formation at -43C, got qi={out_a['qi'][ice_sl].mean()}"

        # Subtest B: Rain + snow fallout + strict mass conservation over 10 min (60 * 10s)
        state_b = solver.empty_state()
        state_b["qv"][:] = 2.0e-3
        state_b["qr"][6:18] = 2.0e-3
        state_b["qs"][20:30] = 1.0e-3
        q_sum_0 = (
            state_b["qv"].astype(np.float64)
            + state_b["qc"]
            + state_b["qr"]
            + state_b["qi"]
            + state_b["qs"]
            + state_b["qg"]
        )
        m_before = solver.tracer_mass(q_sum_0)

        solver.upload_state(state_b)
        solver.upload_surface(solver.empty_surface_state())
        solver.step_physics(dt_s=10.0, n_steps=60)

        out_b = solver.download_state()
        surf_b = solver.download_surface()
        for sp in solver.SPECIES:
            assert np.all(np.isfinite(out_b[sp]))
            assert float(out_b[sp].min()) >= 0.0, f"Negative {sp} found"

        rain_mm = float(surf_b["rain_acc"].mean())
        assert rain_mm > 0.5, f"Expected substantial surface rain fallout > 0.5 mm, got {rain_mm:.4f} mm"

        q_sum_1 = (
            out_b["qv"].astype(np.float64)
            + out_b["qc"]
            + out_b["qr"]
            + out_b["qi"]
            + out_b["qs"]
            + out_b["qg"]
        )
        m_air_after = solver.tracer_mass(q_sum_1)
        m_precip_sfc = float(
            np.sum((surf_b["rain_acc"] + surf_b["snow_acc"] + surf_b["graupel_acc"]) * (1500.0 * 1500.0))
        )
        rel_err = abs((m_air_after + m_precip_sfc) - m_before) / m_before
        assert rel_err < 1.0e-4, f"Water mass conservation error too large: rel_err={rel_err:.3e}"
    finally:
        solver.close()


def test_diurnal_cycle_radiation_lsm_pbl_debilt() -> None:
    """Verify June 21 De Bilt 24h diurnal cycle, surface energy closure, PBL growth & cloud impact."""
    nz, ny, nx = 40, 4, 4
    vgrid = VerticalGrid(nz=nz, z_top_m=18_000.0, stretch_power=1.4)
    hs = np.zeros((ny, nx), dtype=np.float64)
    solver = NonHydrostaticGPUSolver(
        vgrid,
        dx_m=1500.0,
        dy_m=1500.0,
        hs_m=hs,
        has_moist=True,
        has_physics=True,
        theta_sfc_k=288.0,
        p_sfc_pa=101_325.0,
        n_ref_s=0.011,
    )
    try:
        solver.set_physics_config(
            enable_microphysics=0,
            enable_pbl=1,
            enable_radiation=1,
            enable_surface=1,
            rad_interval_s=900.0,
            julian_day=172.0,
            utc_seconds=0.0,
            co2_ppm=425.0,
        )
        sstatic = solver.default_surface_static(
            lat_deg=52.10,
            lon_deg=5.18,
            lsm=1.0,
            soil_type=9.0,
            albedo_base=0.20,
            z0m_base=0.05,
            veg_frac=0.60,
        )
        solver.set_surface_static(sstatic)

        _, z_s, _ = vgrid.heights_3d(hs)
        state = solver.empty_state()
        state["u"][:] = 4.0
        state["qv"][:] = (0.008 * np.exp(-z_s / 2500.0)).astype(np.float32)
        solver.upload_state(state)

        surf = solver.empty_surface_state()
        surf["t_skin"].fill(285.0)
        surf["soil_t"][0].fill(286.0)
        surf["soil_t"][1].fill(286.5)
        surf["soil_t"][2].fill(286.5)
        surf["soil_t"][3].fill(285.5)
        surf["soil_m"][:].fill(0.26)
        solver.upload_surface(surf)

        t2m_series = []
        night_snap = None
        noon_snap = None
        noon_state = None
        noon_surf = None

        # 24 hours at dt = 300 s (288 steps)
        for step in range(288):
            solver.step_physics(dt_s=300.0, n_steps=1)
            sf = solver.download_surface()
            t2m_series.append(float(sf["t2m"].mean()))
            if step == 12:  # ~01:00 UTC (night)
                night_snap = sf
            if step == 143:  # 12:00 UTC (noon; check SEB on 60s sub-step right after for negligible soil dT lag)
                solver.set_physics_config(
                    enable_microphysics=0,
                    enable_pbl=1,
                    enable_radiation=1,
                    enable_surface=1,
                    rad_interval_s=0.0,
                    julian_day=172.0,
                    utc_seconds=43_200.0,
                    co2_ppm=425.0,
                )
                surf_before_noon = solver.download_surface()
                soil_t0_pre = float(surf_before_noon["soil_t"][0].mean())
                solver.step_physics(dt_s=30.0, n_steps=1)
                noon_snap = solver.download_surface()
                noon_snap["_soil_t0_pre"] = np.array([soil_t0_pre], dtype=np.float64)
                noon_state = solver.download_state()
                noon_surf = noon_snap
                # Restore 900s radiation interval and align clock
                solver.set_physics_config(
                    enable_microphysics=0,
                    enable_pbl=1,
                    enable_radiation=1,
                    enable_surface=1,
                    rad_interval_s=900.0,
                    julian_day=172.0,
                    utc_seconds=43_200.0 + 300.0,
                    co2_ppm=425.0,
                )

        assert night_snap is not None and noon_snap is not None
        assert float(night_snap["sw_down"].mean()) == 0.0
        assert float(night_snap["shf"].mean()) < 0.0
        assert float(night_snap["pblh"].mean()) < 500.0

        sw_noon = float(noon_snap["sw_down"].mean())
        lw_noon = float(noon_snap["lw_down"].mean())
        shf_noon = float(noon_snap["shf"].mean())
        lhf_noon = float(noon_snap["lhf"].mean())
        pblh_noon = float(noon_snap["pblh"].mean())
        pblh_night = float(night_snap["pblh"].mean())
        tsk_noon = float(noon_snap["t_skin"].mean())
        tsk_night = float(night_snap["t_skin"].mean())

        assert 600.0 <= sw_noon <= 1000.0, f"Unexpected De Bilt noon SW={sw_noon:.1f} W/m^2"
        assert shf_noon > 80.0, f"Expected noon SHF > 80 W/m^2, got {shf_noon:.1f}"
        assert pblh_noon > 700.0 and pblh_noon > pblh_night + 300.0, (
            f"PBL growth insufficient: night={pblh_night:.1f} m, noon={pblh_noon:.1f} m"
        )
        assert tsk_noon > tsk_night + 5.0

        diurnal_amp = max(t2m_series) - min(t2m_series)
        assert 3.0 <= diurnal_amp <= 22.0, f"Diurnal T2m range out of bounds: {diurnal_amp:.2f} K"

        # Surface energy balance closure at noon: |R_net - (H + LE + G)| < 2 W/m^2
        emis = 0.96
        alb = 0.20
        lambda_sk = 12.0  # 15*(1 - 0.6) + 10*0.6
        r_net = (1.0 - alb) * sw_noon + emis * lw_noon - emis * 5.670374e-8 * (tsk_noon**4)
        g_flux = lambda_sk * (tsk_noon - float(noon_snap["_soil_t0_pre"][0]))
        seb_res = abs(r_net - (shf_noon + lhf_noon + g_flux))
        assert seb_res < 2.0, f"Surface energy balance residual={seb_res:.3f} W/m^2 (Rnet={r_net:.1f}, H={shf_noon:.1f}, LE={lhf_noon:.1f}, G={g_flux:.1f})"

        # Cloud radiative impact comparison at noon
        assert noon_state is not None and noon_surf is not None
        cloudy_state = {k: v.copy() for k, v in noon_state.items()}
        cloudy_state["qc"][4:9] = 1.5e-3
        solver.upload_state(cloudy_state)
        solver.upload_surface(noon_surf)
        solver.set_physics_config(
            enable_microphysics=0,
            enable_pbl=0,
            enable_radiation=1,
            enable_surface=0,
            rad_interval_s=0.0,
            julian_day=172.0,
            utc_seconds=43_200.0,
            co2_ppm=425.0,
        )
        solver.step_physics(dt_s=60.0, n_steps=1)
        cloudy_surf = solver.download_surface()
        sw_cloudy = float(cloudy_surf["sw_down"].mean())
        lw_cloudy = float(cloudy_surf["lw_down"].mean())
        assert sw_cloudy < 0.70 * sw_noon, f"Expected >30% SW cloud attenuation: clear={sw_noon:.1f}, cloudy={sw_cloudy:.1f}"
        assert lw_cloudy > lw_noon + 20.0, f"Expected >20 W/m^2 LW cloud warming: clear={lw_noon:.1f}, cloudy={lw_cloudy:.1f}"
    finally:
        solver.close()


def test_nemo_north_sea_coare_skin_and_flake_ijsselmeer() -> None:
    """Verify COARE 3.6 cool-skin/warm-layer, shallow-water z0, and FLake open water & freezing."""
    nz, ny, nx = 24, 2, 6
    vgrid = VerticalGrid(nz=nz, z_top_m=12_000.0, stretch_power=1.3)
    hs = np.zeros((ny, nx), dtype=np.float64)
    solver = NonHydrostaticGPUSolver(
        vgrid,
        dx_m=1500.0,
        dy_m=1500.0,
        hs_m=hs,
        has_moist=True,
        has_physics=True,
        theta_sfc_k=285.15,
        p_sfc_pa=101_325.0,
        n_ref_s=0.01,
    )
    try:
        sstatic = solver.default_surface_static(lat_deg=52.7, lon_deg=5.2, lsm=0.0, albedo_base=0.06)
        sstatic["lake_mask"][:, 4:6] = 1.0
        sstatic["bathy_depth_m"][:, 0:2] = 45.0  # Deep North Sea
        sstatic["bathy_depth_m"][:, 2:4] = 6.0   # Shallow coastal shelf
        sstatic["bathy_depth_m"][:, 4:6] = 4.5   # Lake IJsselmeer
        solver.set_surface_static(sstatic)

        # 1. Nighttime cooling: cool air over warmer sea -> dt_cool < 0
        state_night = solver.empty_state()
        state_night["u"][:] = 5.0
        state_night["theta_p"][:] = -5.0  # ~280 K air over 285 K sea
        state_night["qv"][:] = 0.0045
        solver.upload_state(state_night)

        surf = solver.empty_surface_state()
        surf["t_skin"].fill(285.15)
        surf["sst_bulk"].fill(285.15)
        surf["lake_tbot"].fill(283.15)
        surf["lake_hmld"].fill(2.5)
        surf["lake_ctheta"].fill(0.65)
        solver.upload_surface(surf)

        solver.set_physics_config(
            enable_microphysics=0,
            enable_pbl=1,
            enable_radiation=1,
            enable_surface=1,
            rad_interval_s=300.0,
            julian_day=172.0,
            utc_seconds=0.0,
        )
        solver.step_physics(dt_s=300.0, n_steps=6)
        sf_n = solver.download_surface()
        assert float(sf_n["dt_cool"][:, 0:4].mean()) < -0.02, f"Expected cool skin, got {sf_n['dt_cool'][:, 0:4]}"
        assert float(sf_n["t_skin"][:, 0:2].mean()) < float(sf_n["sst_bulk"][:, 0:2].mean())

        # Strong wind coastal wave shoaling test
        state_wind = solver.empty_state()
        state_wind["u"][:] = 15.0
        state_wind["qv"][:] = 0.006
        solver.upload_state(state_wind)
        solver.step_physics(dt_s=60.0, n_steps=1)
        sf_w = solver.download_surface()
        assert float(sf_w["ustar"][:, 2:4].mean()) >= float(sf_w["ustar"][:, 0:2].mean())

        # 2. Calm sunny afternoon: warm layer dt_warm > 0 + open FLake bounds
        state_warm = solver.empty_state()
        state_warm["u"][:] = 1.2
        state_warm["theta_p"][:] = 6.0  # ~291 K warm afternoon air
        state_warm["qv"][:] = 0.008
        solver.upload_state(state_warm)
        surf2 = solver.empty_surface_state()
        surf2["t_skin"].fill(285.15)
        surf2["sst_bulk"].fill(285.15)
        surf2["lake_tbot"].fill(283.15)
        surf2["lake_hmld"].fill(2.5)
        surf2["lake_ctheta"].fill(0.65)
        solver.upload_surface(surf2)
        solver.set_physics_config(
            enable_microphysics=0,
            enable_pbl=1,
            enable_radiation=1,
            enable_surface=1,
            rad_interval_s=300.0,
            julian_day=172.0,
            utc_seconds=46_800.0,
        )
        solver.step_physics(dt_s=300.0, n_steps=12)
        sf_warm = solver.download_surface()
        assert float(sf_warm["dt_warm"][:, 0:2].mean()) > 0.05, f"Expected warm layer >0.05 K, got {sf_warm['dt_warm'][:, 0:2]}"
        assert float(sf_warm["t_skin"][:, 0:2].mean()) > float(sf_warm["sst_bulk"][:, 0:2].mean())

        hmld = sf_warm["lake_hmld"][:, 4:6]
        cth = sf_warm["lake_ctheta"][:, 4:6]
        assert np.all((hmld >= 0.01) & (hmld <= 4.5)), f"FLake hmld out of bounds: {hmld}"
        assert np.all((cth >= 0.50) & (cth <= 0.80)), f"FLake ctheta out of bounds: {cth}"
        assert np.all(sf_warm["lake_hice"][:, 4:6] == 0.0)

        # 3. Cold winter lake freeze check -> freshwater ice grows in (0, 3.0] m
        state_ice = solver.empty_state()
        state_ice["u"][:] = 3.0
        state_ice["theta_p"][:] = -22.0  # ~263 K severe frost air
        state_ice["qv"][:] = 0.001
        solver.upload_state(state_ice)
        surf_ice = solver.empty_surface_state()
        surf_ice["t_skin"].fill(273.15)
        surf_ice["sst_bulk"].fill(275.15)
        surf_ice["lake_tbot"].fill(276.0)
        surf_ice["lake_hmld"].fill(4.5)
        surf_ice["lake_ctheta"].fill(0.65)
        solver.upload_surface(surf_ice)
        solver.set_physics_config(
            enable_microphysics=0,
            enable_pbl=0,
            enable_radiation=1,
            enable_surface=1,
            rad_interval_s=600.0,
            julian_day=15.0,
            utc_seconds=0.0,
        )
        solver.step_physics(dt_s=600.0, n_steps=24)
        sf_ice = solver.download_surface()
        hice = sf_ice["lake_hice"][:, 4:6]
        assert np.all((hice > 0.0) & (hice <= 3.0)), f"Expected FLake ice growth in (0, 3], got {hice}"
    finally:
        solver.close()


def test_coupled_moist_convective_bubble_precipitation() -> None:
    """Verify full coupled step_full on a buoyant moist bubble produces updraft, cloud, and rain."""
    nz, ny, nx = 40, 16, 16
    vgrid = VerticalGrid(nz=nz, z_top_m=16_000.0, stretch_power=1.2)
    hs = np.zeros((ny, nx), dtype=np.float64)
    theta_sfc = 296.0
    p_sfc = 101_325.0
    n_ref = 0.0085

    solver = NonHydrostaticGPUSolver(
        vgrid,
        dx_m=1500.0,
        dy_m=1500.0,
        hs_m=hs,
        has_moist=True,
        has_physics=True,
        theta_sfc_k=theta_sfc,
        p_sfc_pa=p_sfc,
        n_ref_s=n_ref,
        div_damp_coeff=0.05,
        sponge_depth_m=3000.0,
        sponge_tau_max_s=0.01,
    )
    try:
        _, z_s, _ = vgrid.heights_3d(hs)
        ref = VerticalGrid.reference_state(z_s, theta_sfc_k=theta_sfc, p_sfc_pa=p_sfc, brunt_vaisala_hz=n_ref)
        t_ref = ref["theta_bar"] * ref["pi_bar"]
        qs_ref = _qsat_water_np(t_ref, ref["p_bar"])

        yy = (np.arange(ny, dtype=np.float64) - 0.5 * (ny - 1)) * 1500.0
        xx = (np.arange(nx, dtype=np.float64) - 0.5 * (nx - 1)) * 1500.0
        y2, x2 = np.meshgrid(yy, xx, indexing="ij")
        r2 = (x2 * x2 + y2 * y2)[None, :, :]
        bub = np.exp(-r2 / (4000.0**2) - ((z_s - 1500.0) / 1000.0) ** 2)

        state = solver.empty_state()
        state["theta_p"][:] = (3.5 * bub).astype(np.float32)
        rh_base = np.where(z_s < 4000.0, 0.90, 0.45)
        state["qv"][:] = np.minimum(0.025, qs_ref * (rh_base + 0.14 * bub)).astype(np.float32)

        solver.upload_state(state)
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

        max_w_peak = 0.0
        max_qc_peak = 0.0
        for _ in range(25):
            solver.step_full(dt_s=4.0, n_steps=10)
            st_mid = solver.download_state()
            max_w_peak = max(max_w_peak, float(st_mid["w"].max()))
            max_qc_peak = max(max_qc_peak, float(st_mid["qc"].max()))

        out = solver.download_state()
        surf = solver.download_surface()
        for k, arr in out.items():
            assert np.all(np.isfinite(arr)), f"Non-finite in 3D {k}"
        for k, arr in surf.items():
            assert np.all(np.isfinite(arr)), f"Non-finite in surface {k}"

        total_precip = surf["rain_acc"] + surf["snow_acc"] + surf["graupel_acc"]
        assert max_w_peak > 1.0, f"Convective updraft too weak: max_w={max_w_peak:.3f} m/s"
        assert max_qc_peak > 3.0e-4, f"Insufficient cloud formation: max_qc={max_qc_peak:.3e} kg/kg"
        assert float(total_precip.max()) > 0.0, "Expected non-zero surface precipitation from convective bubble"
    finally:
        solver.close()


def test_rtx3060_nz137_720x720_full_physics_tiled_smoke() -> None:
    """Verify full-size operational D3 (720x720x137) coupled dynamics+physics fits 2x2 tiled VRAM."""
    spec = get_domain_spec("d3", nz=137, z_top_m=31_000.0, stretch_power=1.5)
    ny, nx, nz = spec.ny, spec.nx, spec.nz

    # Mild Ardennes terrain in SE, flat NL, North Sea NW corner, Lake IJsselmeer patch
    yy = np.linspace(0.0, 1.0, ny, dtype=np.float64)[:, None]
    xx = np.linspace(0.0, 1.0, nx, dtype=np.float64)[None, :]
    hs = 420.0 * np.exp(-((xx - 0.65) ** 2 + (yy - 0.25) ** 2) / 0.04)

    solver = NonHydrostaticGPUSolver(
        spec.vertical,
        dx_m=spec.dx_m,
        dy_m=spec.dy_m,
        hs_m=hs,
        msf=spec.map_factor,
        f_cor=spec.coriolis_f,
        has_moist=True,
        has_physics=True,
        bc_x=2,
        bc_y=2,
        use_coriolis=True,
        sponge_depth_m=6000.0,
        sponge_tau_max_s=0.01,
    )
    try:
        assert solver.num_tiles == 4, f"Expected 2x2=4 tiles for 720x720x137, got {solver.num_tiles}"
        assert solver.estimated_tile_vram_bytes <= DEFAULT_MAX_VRAM_BYTES

        state = solver.empty_state()
        state["u"][:] = 6.0
        state["v"][:] = 2.0
        _, z_s, _ = spec.vertical.heights_3d(hs)
        state["qv"][:] = (0.007 * np.exp(-z_s / 2500.0)).astype(np.float32)

        lbc = {k: state[k].copy() for k in ("u", "v", "w", "theta_p", "pi_p")}
        solver.upload_state(state)
        solver.set_lbc_target(lbc)

        sstatic = solver.default_surface_static()
        sstatic["lat_deg"] = spec.lat2d.astype(np.float32)
        sstatic["lon_deg"] = spec.lon2d.astype(np.float32)
        sea_mask = (spec.lat2d > 52.8) & (spec.lon2d < 5.0)
        lake_mask = (spec.lat2d >= 52.45) & (spec.lat2d <= 52.80) & (spec.lon2d >= 5.05) & (spec.lon2d <= 5.65)
        sstatic["lsm"][sea_mask | lake_mask] = 0.0
        sstatic["lake_mask"][lake_mask] = 1.0
        sstatic["bathy_depth_m"][sea_mask] = 32.0
        sstatic["bathy_depth_m"][lake_mask] = 4.5
        solver.set_surface_static(sstatic)
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

        t0 = time.perf_counter()
        solver.step_full(dt_s=4.0, n_steps=1)
        elapsed_s = time.perf_counter() - t0

        out = solver.download_state()
        surf = solver.download_surface()
        assert np.all(np.isfinite(out["theta_p"]))
        assert np.all(np.isfinite(out["u"]))
        assert float(out["qv"].min()) >= 0.0
        assert np.all(np.isfinite(surf["t2m"]))
        assert float(surf["sw_down"].mean()) > 100.0
        assert float(surf["pblh"].mean()) > 0.0
        t2_min, t2_max = float(surf["t2m"].min()), float(surf["t2m"].max())
        assert 220.0 <= t2_min and t2_max <= 330.0, f"Unexpected T2m bounds [{t2_min}, {t2_max}]"
        print(
            f"[D3 720x720x137 4-tile full step_full] elapsed={elapsed_s:.2f}s "
            f"est_tile_vram={solver.estimated_tile_vram_bytes / (1024**3):.2f} GiB"
        )
    finally:
        solver.close()


def test_pbl_last_order_stability_and_default_unchanged() -> None:
    """Verify opt-in IFS-50r1 PBL-last ordering runs cleanly and leaves default pbl_last=0 unchanged."""
    nz, ny, nx = 24, 4, 4
    vgrid = VerticalGrid(nz=nz, z_top_m=12_000.0, stretch_power=1.2)
    hs = np.zeros((ny, nx), dtype=np.float64)
    solver = NonHydrostaticGPUSolver(
        vgrid,
        dx_m=7500.0,
        dy_m=7500.0,
        hs_m=hs,
        has_moist=True,
        has_physics=True,
    )
    try:
        _, z_s, _ = vgrid.heights_3d(hs)
        ref = VerticalGrid.reference_state(z_s)
        base_state = solver.empty_state()
        base_state["u"][:] = 6.0
        base_state["theta_p"][0:5] = -2.0
        p_0 = ref["p_bar"][:6]
        t_0 = (ref["theta_bar"][:6] - 2.0) * ref["pi_bar"][:6]
        base_state["qv"][:6] = (_qsat_water_np(t_0, p_0) + 1.0e-3).astype(np.float32)

        sstatic = solver.default_surface_static()
        base_surf = solver.empty_surface_state()

        outs = []
        for kw in ({}, {"pbl_last": 0}, {"pbl_last": 1}):
            solver.upload_state(base_state)
            solver.set_surface_static(sstatic)
            solver.upload_surface(base_surf)
            solver.set_physics_config(
                enable_microphysics=1,
                enable_pbl=1,
                enable_radiation=0,
                enable_surface=1,
                **kw,
            )
            solver.step_physics(dt_s=40.0, n_steps=10)
            outs.append(solver.download_state())

        for sp in solver.SPECIES:
            np.testing.assert_array_equal(outs[0][sp], outs[1][sp])
            assert np.all(np.isfinite(outs[2][sp]))
            assert float(outs[2][sp].min()) >= 0.0
        np.testing.assert_array_equal(outs[0]["theta_p"], outs[1]["theta_p"])
        assert np.max(np.abs(outs[2]["qv"] - outs[0]["qv"])) > 1e-7
    finally:
        solver.close()
