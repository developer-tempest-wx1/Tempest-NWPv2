"""Phase B5 GPU Test Suite: Global 0.25 deg Spherical D1 + cuFFT Polar Filter + Periodic Strip Tiling (0% AI / 0% ML)."""

from __future__ import annotations

import datetime
import math
from pathlib import Path
from typing import Dict

import numpy as np

from tempest_v2.cuda_bridge import NonHydrostaticGPUSolver, cuda_vram_info, estimate_vram_bytes
from tempest_v2.ocean_si3_bridge import (
    NemoSi3GPUSolver,
    estimate_ocean_si3_vram_bytes,
    freezing_temp_c,
    teos10_rho,
)
from tempest_v2.projection import EARTH_RADIUS_M, OMEGA_RAD_S, DomainSpec, VerticalGrid, get_domain_spec


def _make_coarse_spherical_domain(
    ny: int = 73,
    nx: int = 96,
    nz: int = 14,
    z_top_m: float = 20_000.0,
) -> DomainSpec:
    dlat = 180.0 / float(ny - 1)
    dlon = 360.0 / float(nx)
    lats = np.linspace(-90.0, 90.0, ny, dtype=np.float64)
    lons = np.arange(nx, dtype=np.float64) * dlon
    lon2d, lat2d = np.meshgrid(lons, lats)
    dx_m = EARTH_RADIUS_M * math.radians(dlon)
    dy_m = EARTH_RADIUS_M * math.radians(dlat)
    cos_lat = np.maximum(np.cos(np.radians(lat2d)), math.cos(math.radians(89.5)))
    return DomainSpec(
        name="d1_coarse_test",
        ny=ny,
        nx=nx,
        dx_m=dx_m,
        dy_m=dy_m,
        is_projected=False,
        is_periodic_lon=True,
        projection=None,
        x_m=lons,
        y_m=lats,
        lat2d=lat2d,
        lon2d=lon2d,
        map_factor=1.0 / cos_lat,
        coriolis_f=2.0 * OMEGA_RAD_S * np.sin(np.radians(lat2d)),
        vertical=VerticalGrid(nz=nz, z_top_m=z_top_m, stretch_power=1.2),
        is_spherical=True,
    )


def _global_mountain(spec: DomainSpec, amp_m: float = 1800.0) -> np.ndarray:
    dlat = (spec.lat2d - 32.0) / 15.0
    dlon = (((spec.lon2d - 85.0 + 180.0) % 360.0) - 180.0) / 22.0
    return amp_m * np.exp(-(dlat * dlat + dlon * dlon))


def test_d1_spherical_resting_atmosphere_zero_drift() -> None:
    spec = _make_coarse_spherical_domain(ny=49, nx=64, nz=14)
    hs = _global_mountain(spec, amp_m=1500.0)
    solver = NonHydrostaticGPUSolver(
        spec.vertical,
        spec.dx_m,
        spec.dy_m,
        hs,
        msf=spec.map_factor,
        f_cor=spec.coriolis_f,
        has_moist=False,
        bc_x=1,
        bc_y=0,
        use_coriolis=True,
        is_spherical=True,
        polar_filter_lat_deg=60.0,
    )
    try:
        st = solver.empty_state()
        solver.upload_state(st)
        solver.step_rk3(dt_s=60.0, n_steps=10)
        out = solver.download_state()
    finally:
        solver.close()

    assert float(np.max(np.abs(out["u"]))) < 1e-6
    assert float(np.max(np.abs(out["v"]))) < 1e-6
    assert float(np.max(np.abs(out["w"]))) < 1e-6


def test_d1_spherical_balanced_jet_polar_filter_and_moisture_conservation() -> None:
    """Mid-latitude + polar-extending Rossby jet remains stable across +-89.75 deg at dt=60s and conserves spherical mass."""
    spec = _make_coarse_spherical_domain(ny=73, nx=96, nz=14)
    hs = _global_mountain(spec, amp_m=1200.0)
    solver = NonHydrostaticGPUSolver(
        spec.vertical,
        spec.dx_m,
        spec.dy_m,
        hs,
        msf=spec.map_factor,
        f_cor=spec.coriolis_f,
        has_moist=True,
        bc_x=1,
        bc_y=0,
        use_coriolis=True,
        is_spherical=True,
        polar_filter_lat_deg=60.0,
        sponge_depth_m=4000.0,
        sponge_tau_max_s=0.01,
    )
    try:
        st = solver.empty_state()
        lat_rad = np.radians(spec.lat2d)
        # Smooth zonal jet peaking at mid-latitudes and decaying toward poles without singular shear
        u_prof = 18.0 * np.cos(lat_rad) * (np.sin(2.0 * lat_rad) ** 2 + 0.15)
        u_wave = 2.0 * np.cos(lat_rad) * np.sin(np.radians(spec.lon2d) * 2.0)
        u_cell = np.broadcast_to((u_prof + u_wave)[None, :, :], (spec.nz, spec.ny, spec.nx)).copy()
        st["u"][:, :, :-1] = u_cell
        st["u"][:, :, -1] = u_cell[:, :, 0]

        # Moist plume reaching near the Arctic/Antarctic circles and across Greenwich 0E seam
        st["qv"][:] = 0.004 + 0.004 * np.cos(lat_rad)[None, :, :]
        st["qv"][:, -10:-2, :] += 0.002 * (1.0 + np.cos(np.radians(spec.lon2d)))[None, -10:-2, :]
        st["qc"][2:6, 5:15, :12] = 0.0008
        st["qc"][2:6, -15:-5, -12:] = 0.0008

        m_qv_0 = solver.tracer_mass(st["qv"])
        m_qc_0 = solver.tracer_mass(st["qc"])

        solver.upload_state(st)
        solver.step_rk3(dt_s=60.0, n_steps=12)
        out = solver.download_state()

        m_qv_1 = solver.tracer_mass(out["qv"])
        m_qc_1 = solver.tracer_mass(out["qc"])
    finally:
        solver.close()

    for k in ("u", "v", "w", "theta_p", "pi_p", "qv", "qc"):
        assert np.all(np.isfinite(out[k])), f"Non-finite values in spherical D1 field {k}"
    assert np.allclose(out["u"][:, :, 0], out["u"][:, :, -1], atol=1e-6)
    for sp in solver.SPECIES:
        assert float(np.min(out[sp])) >= 0.0
    assert abs(m_qv_1 - m_qv_0) / m_qv_0 < 1e-4
    assert abs(m_qc_1 - m_qc_0) / m_qc_0 < 1e-4
    assert float(np.max(np.abs(out["w"]))) < 5.0


def test_d1_periodic_latitude_strip_tiling_matches_untiled() -> None:
    """Periodic-x latitude-strip tiling (nt_x=1, nt_y>=2) must match untiled spherical solve cleanly."""
    spec = _make_coarse_spherical_domain(ny=161, nx=96, nz=24)
    hs = _global_mountain(spec, amp_m=800.0)
    common: Dict[str, object] = dict(
        msf=spec.map_factor,
        f_cor=spec.coriolis_f,
        has_moist=True,
        bc_x=1,
        bc_y=0,
        use_coriolis=True,
        is_spherical=True,
        polar_filter_lat_deg=60.0,
        halo=20,
    )

    s_untiled = NonHydrostaticGPUSolver(
        spec.vertical, spec.dx_m, spec.dy_m, hs, max_vram_bytes=2 * 1024**3, **common  # type: ignore[arg-type]
    )
    est_1 = estimate_vram_bytes(24, 161, 96, moist=True)
    est_2 = estimate_vram_bytes(24, (161 + 1) // 2 + 20, 96, moist=True)
    cap_bytes = (est_1 + est_2) // 2
    s_tiled = NonHydrostaticGPUSolver(
        spec.vertical, spec.dx_m, spec.dy_m, hs, max_vram_bytes=cap_bytes, **common  # type: ignore[arg-type]
    )
    try:
        assert s_untiled.num_tiles == 1
        assert s_tiled.nt_x == 1 and s_tiled.nt_y >= 2

        st = s_untiled.empty_state()
        lat_rad = np.radians(spec.lat2d)
        lon_rad = np.radians(spec.lon2d)
        u_c = np.broadcast_to((12.0 * np.cos(lat_rad))[None, :, :], (spec.nz, spec.ny, spec.nx)).copy()
        st["u"][:, :, :-1] = u_c
        st["u"][:, :, -1] = u_c[:, :, 0]
        st["v"][:, 1:-1, :] = 2.0 * np.sin(2.0 * lon_rad)[None, 1:, :]
        st["theta_p"][:] = (0.8 * np.cos(lat_rad) * np.sin(lon_rad))[None, :, :]
        st["qv"][:] = 0.006 + 0.002 * np.cos(lat_rad)[None, :, :]

        s_untiled.upload_state(st)
        s_tiled.upload_state(st)
        s_untiled.step_rk3(dt_s=45.0, n_steps=1)
        s_tiled.step_rk3(dt_s=45.0, n_steps=1)
        out_u = s_untiled.download_state()
        out_t = s_tiled.download_state()
    finally:
        s_untiled.close()
        s_tiled.close()

    for key in ("u", "v", "w", "theta_p", "pi_p", "qv"):
        max_diff = float(np.max(np.abs(out_u[key] - out_t[key])))
        assert max_diff < 1e-5, f"Periodic strip tiling mismatch in {key}: {max_diff:.3e}"


def test_rtx3060_d1_0p25_global_nz65_vram_and_tiled_smoke() -> None:
    """Verify improved D1 0.25 deg (721x1440x65) fits RTX 3060 via periodic latitude-strip tiling + cuFFT polar filter."""
    spec = get_domain_spec("d1_global_0p25", nz=65)
    assert spec.name == "d1_global_0p25"
    assert (spec.ny, spec.nx, spec.nz) == (721, 1440, 65)
    assert spec.is_spherical and spec.is_periodic_lon

    hs = _global_mountain(spec, amp_m=1500.0)
    solver = NonHydrostaticGPUSolver(
        spec.vertical,
        spec.dx_m,
        spec.dy_m,
        hs,
        msf=spec.map_factor,
        f_cor=spec.coriolis_f,
        has_moist=True,
        bc_x=1,
        bc_y=0,
        use_coriolis=True,
        is_spherical=True,
        polar_filter_lat_deg=60.0,
        sponge_depth_m=6000.0,
        sponge_tau_max_s=0.015,
    )
    try:
        assert solver.nt_x == 1 and solver.nt_y >= 2
        st = solver.empty_state()
        lat_rad = np.radians(spec.lat2d)
        u_c = np.broadcast_to((14.0 * np.cos(lat_rad))[None, :, :], (spec.nz, spec.ny, spec.nx)).copy()
        st["u"][:, :, :-1] = u_c
        st["u"][:, :, -1] = u_c[:, :, 0]
        st["qv"][:] = 0.005 + 0.004 * np.cos(lat_rad)[None, :, :]
        st["qc"][10:15, 500:540, 20:60] = 0.0005

        solver.upload_state(st)
        solver.step_rk3(dt_s=60.0, n_steps=1)
        out = solver.download_state()
    finally:
        solver.close()

    for key in ("u", "v", "w", "theta_p", "pi_p", "qv", "qc"):
        assert np.all(np.isfinite(out[key])), f"Non-finite values in D1 0.25 deg {key}"
    for sp in solver.SPECIES:
        assert float(np.min(out[sp])) >= 0.0
    free_after, _ = cuda_vram_info()
    assert free_after > 3_500_000_000, f"GPU VRAM leak after D1 0.25 deg close: free={free_after}"


def _make_regional_ocean_spec(ny: int = 24, nx: int = 24, dx_m: float = 20_000.0, lat0: float = 72.0) -> DomainSpec:
    x_m = np.arange(nx, dtype=np.float64) * dx_m
    y_m = np.arange(ny, dtype=np.float64) * dx_m
    yy, xx = np.meshgrid(y_m, x_m, indexing="ij")
    lat2d = lat0 + (yy - 0.5 * y_m[-1]) / 111_000.0
    lon2d = 5.0 + (xx - 0.5 * x_m[-1]) / (111_000.0 * math.cos(math.radians(lat0)))
    return DomainSpec(
        name="ocean_test_reg",
        ny=ny,
        nx=nx,
        dx_m=dx_m,
        dy_m=dx_m,
        is_projected=False,
        is_periodic_lon=False,
        projection=None,
        x_m=x_m,
        y_m=y_m,
        lat2d=lat2d,
        lon2d=lon2d,
        map_factor=np.ones((ny, nx), dtype=np.float64),
        coriolis_f=2.0 * OMEGA_RAD_S * np.sin(np.radians(lat2d)),
        vertical=VerticalGrid(nz=14, z_top_m=15_000.0, stretch_power=1.2),
        is_spherical=False,
    )


def test_nemo_teos10_eos_and_freezing_point_accuracy() -> None:
    """Verify exact Roquet (2015) TEOS-10 density checkvalue and freezing point."""
    rho_check = teos10_rho(3.0, 35.5, 3000.0)
    assert abs(rho_check - 1028.21993233072) < 1e-8
    rho_sfc = teos10_rho(10.0, 35.16504, 0.0)
    assert 1026.7 <= rho_sfc <= 1027.2
    tf0 = freezing_temp_c(35.0, 0.0)
    tf500 = freezing_temp_c(35.0, 500.0)
    assert -1.96 < tf0 < -1.86
    assert abs((tf500 - tf0) + 0.3765) < 1e-3
    assert estimate_ocean_si3_vram_bytes(16, 721, 1440, enable_si3=True) < 1_300_000_000


def test_nemo_barotropic_inverse_barometer_and_land_mask_isolation() -> None:
    """Split-explicit SSH responds to inverse-barometer cyclone while land & lake masks stay strictly zero."""
    spec = _make_regional_ocean_spec(ny=24, nx=24, dx_m=25_000.0, lat0=54.0)
    lsm = np.zeros((spec.ny, spec.nx), dtype=np.float32)
    lsm[0, :] = 1.0
    lsm[-1, :] = 1.0
    lsm[:, 0] = 1.0
    lsm[:, -1] = 1.0
    lake = np.zeros((spec.ny, spec.nx), dtype=np.float32)
    lake[3:5, 3:5] = 1.0
    bathy = np.where((lsm < 0.5) & (lake < 0.5), 120.0, 4.5).astype(np.float32)
    bathy[lsm >= 0.5] = 0.0
    sstatic = {
        "lat_deg": spec.lat2d.astype(np.float32),
        "lon_deg": spec.lon2d.astype(np.float32),
        "lsm": lsm,
        "lake_mask": lake,
        "bathy_depth_m": bathy,
    }

    yy, xx = np.mgrid[0:spec.ny, 0:spec.nx]
    mslp = 101_325.0 - 4000.0 * np.exp(-(((yy - 12.0) ** 2 + (xx - 12.0) ** 2) / 16.0))
    forcing = {
        "u10m": np.zeros((spec.ny, spec.nx), dtype=np.float32),
        "v10m": np.zeros((spec.ny, spec.nx), dtype=np.float32),
        "t2m": np.full((spec.ny, spec.nx), 283.0, dtype=np.float32),
        "q2m": np.full((spec.ny, spec.nx), 0.006, dtype=np.float32),
        "mslp": mslp.astype(np.float32),
        "sw_down": np.zeros((spec.ny, spec.nx), dtype=np.float32),
        "lw_down": np.full((spec.ny, spec.nx), 320.0, dtype=np.float32),
        "precip_liq": np.zeros((spec.ny, spec.nx), dtype=np.float32),
        "precip_ice": np.zeros((spec.ny, spec.nx), dtype=np.float32),
        "cld_fra": np.full((spec.ny, spec.nx), 0.5, dtype=np.float32),
    }

    with NemoSi3GPUSolver(spec, sstatic, nz=8) as solver:
        st0 = solver.init_from_atmosphere_and_osisaf({"sst_bulk": np.full((spec.ny, spec.nx), 283.0, dtype=np.float32)})
        solver.upload_state(st0)
        for _ in range(15):
            solver.step(forcing, dt_s=120.0)
        st1 = solver.download_state()

    dry = (lsm >= 0.5) | (lake >= 0.5)
    assert float(st1["ssh"][12, 12]) > 0.10, f"Expected IB sea-level rise at storm center, got {st1['ssh'][12, 12]}"
    assert float(np.max(np.abs(st1["ssh"][dry]))) == 0.0
    assert float(np.max(np.abs(st1["temp_k"][:, dry]))) == 0.0
    assert float(np.max(np.abs(st1["sal_psu"][:, dry]))) == 0.0
    assert float(np.max(np.abs(st1["ice_frac"][dry]))) == 0.0


def test_si3_polar_frazil_growth_brine_rejection_salt_conservation() -> None:
    """Cold Arctic cooling freezes frazil ice, raises SSS via brine rejection, and conserves ocean+ice salt."""
    spec = _make_regional_ocean_spec(ny=12, nx=12, dx_m=25_000.0, lat0=78.0)
    sstatic = {
        "lat_deg": spec.lat2d.astype(np.float32),
        "lon_deg": spec.lon2d.astype(np.float32),
        "lsm": np.zeros((spec.ny, spec.nx), dtype=np.float32),
        "lake_mask": np.zeros((spec.ny, spec.nx), dtype=np.float32),
        "bathy_depth_m": np.full((spec.ny, spec.nx), 500.0, dtype=np.float32),
    }
    forcing = {
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

    with NemoSi3GPUSolver(spec, sstatic, nz=8, enable_dynamics=False) as solver:
        st0 = solver.empty_state()
        for k in range(8):
            st0["temp_k"][k] = 273.15 + freezing_temp_c(34.0, 2.5) + 0.005
            st0["sal_psu"][k] = 34.0
        salt0 = float(np.sum(solver.total_salt_kg_m2(st0)))
        solver.upload_state(st0)
        for _ in range(16):
            solver.step(forcing, dt_s=900.0)
        st1 = solver.download_state()
        salt1 = float(np.sum(solver.total_salt_kg_m2(st1)))

    rel_err = abs(salt1 - salt0) / max(salt0, 1.0)
    assert float(np.mean(st1["ice_frac"])) > 0.05
    assert float(np.mean(st1["ice_thick"])) > 0.05
    assert float(np.mean(st1["ice_albedo"])) > 0.25
    assert float(np.mean(st1["sal_psu"][0])) > 34.0, "Brine rejection must increase surface salinity"
    assert rel_err < 1e-5, f"Total ocean+ice salt drift rel_err={rel_err:.3e}"


def test_si3_mevp_wind_drift_and_thorndike_ridging_cap() -> None:
    """mEVP rheology drifts pack ice downwind and Thorndike ridging strictly caps concentration <= 0.997."""
    spec = _make_regional_ocean_spec(ny=16, nx=20, dx_m=20_000.0, lat0=76.0)
    lsm = np.zeros((spec.ny, spec.nx), dtype=np.float32)
    lsm[:, -2:] = 1.0
    lsm[0, :] = 1.0
    lsm[-1, :] = 1.0
    bathy = np.where(lsm < 0.5, 250.0, 0.0).astype(np.float32)
    sstatic = {
        "lat_deg": spec.lat2d.astype(np.float32),
        "lon_deg": spec.lon2d.astype(np.float32),
        "lsm": lsm,
        "lake_mask": np.zeros((spec.ny, spec.nx), dtype=np.float32),
        "bathy_depth_m": bathy,
    }
    forcing = {
        "u10m": np.full((spec.ny, spec.nx), 16.0, dtype=np.float32),
        "v10m": np.zeros((spec.ny, spec.nx), dtype=np.float32),
        "t2m": np.full((spec.ny, spec.nx), 263.15, dtype=np.float32),
        "q2m": np.full((spec.ny, spec.nx), 0.001, dtype=np.float32),
        "mslp": np.full((spec.ny, spec.nx), 101_325.0, dtype=np.float32),
        "sw_down": np.full((spec.ny, spec.nx), 40.0, dtype=np.float32),
        "lw_down": np.full((spec.ny, spec.nx), 210.0, dtype=np.float32),
        "precip_liq": np.zeros((spec.ny, spec.nx), dtype=np.float32),
        "precip_ice": np.zeros((spec.ny, spec.nx), dtype=np.float32),
        "cld_fra": np.full((spec.ny, spec.nx), 0.5, dtype=np.float32),
    }

    with NemoSi3GPUSolver(spec, sstatic, nz=8, n_evp_substeps=40) as solver:
        surf_in = {
            "sst_bulk": np.full((spec.ny, spec.nx), 271.35, dtype=np.float32),
            "sea_ice_frac": np.where(lsm < 0.5, 0.96, 0.0).astype(np.float32),
            "sea_ice_thick": np.where(lsm < 0.5, 0.85, 0.0).astype(np.float32),
            "sea_ice_tsfc": np.full((spec.ny, spec.nx), 263.15, dtype=np.float32),
        }
        st0 = solver.init_from_atmosphere_and_osisaf(surf_in)
        solver.upload_state(st0)
        for _ in range(4):
            solver.step(forcing, dt_s=600.0)
        st1 = solver.download_state()

    wet = lsm < 0.5
    assert float(np.max(st1["u_ice"][wet])) > 0.03
    assert float(np.min(st1["a_i"])) >= 0.0 and float(np.min(st1["v_i"])) >= 0.0
    assert float(np.max(np.sum(st1["a_i"], axis=0))) <= 0.9971


def test_coupled_sea_ice_mosaic_radiation_and_heat_flux() -> None:
    """Two-way SI3 mosaic reduces open-water heat loss and cools t_skin relative to ice-free ocean."""
    vgrid = VerticalGrid(nz=16, z_top_m=15_000.0, stretch_power=1.2)
    ny, nx = 8, 8
    hs = np.zeros((ny, nx), dtype=np.float64)
    msf = np.ones((ny, nx), dtype=np.float64)
    fcor = np.full((ny, nx), 1.38e-4, dtype=np.float64)
    sstatic = {
        "lat_deg": np.full((ny, nx), 74.0, dtype=np.float32),
        "lon_deg": np.full((ny, nx), 10.0, dtype=np.float32),
        "lsm": np.zeros((ny, nx), dtype=np.float32),
        "lake_mask": np.zeros((ny, nx), dtype=np.float32),
        "soil_type": np.zeros((ny, nx), dtype=np.float32),
        "bathy_depth_m": np.full((ny, nx), 150.0, dtype=np.float32),
        "z0m_base": np.full((ny, nx), 1e-4, dtype=np.float32),
        "albedo_base": np.full((ny, nx), 0.06, dtype=np.float32),
        "veg_frac": np.zeros((ny, nx), dtype=np.float32),
    }

    def _run_case(sic: float) -> Dict[str, np.ndarray]:
        solver = NonHydrostaticGPUSolver(
            vgrid, 10_000.0, 10_000.0, hs, msf=msf, f_cor=fcor, has_moist=True,
            bc_x=0, bc_y=0, use_coriolis=True, max_vram_bytes=2 * 1024**3,
        )
        try:
            solver.set_surface_static(sstatic)
            solver.set_physics_config(rad_interval_s=0.0, julian_day=172.0, utc_seconds=43200.0)
            st = solver.empty_state()
            st["u"][:] = 8.0
            st["qv"][:] = 0.0012
            st["theta_p"][:] = -30.0 # ~258 K cold Arctic air mass
            surf = {
                "t_skin": np.full((ny, nx), 271.5, dtype=np.float32),
                "soil_t": np.full((4, ny, nx), 271.5, dtype=np.float32),
                "soil_m": np.full((4, ny, nx), 0.25, dtype=np.float32),
                "sst_bulk": np.full((ny, nx), 271.5, dtype=np.float32),
                "sea_ice_frac": np.full((ny, nx), sic, dtype=np.float32),
                "sea_ice_thick": np.full((ny, nx), 1.4 if sic > 0 else 0.0, dtype=np.float32),
                "sea_ice_tsfc": np.full((ny, nx), 260.0, dtype=np.float32),
                "sea_ice_albedo": np.full((ny, nx), 0.78, dtype=np.float32),
            }
            solver.upload_state(st)
            solver.upload_surface(surf)
            for _ in range(3):
                solver.step_physics(dt_s=30.0)
            return solver.download_surface()
        finally:
            solver.close()

    surf_open = _run_case(0.0)
    surf_ice = _run_case(0.85)
    assert float(np.mean(surf_open["shf"])) > float(np.mean(surf_ice["shf"])) + 10.0
    assert float(np.mean(surf_open["t_skin"])) > float(np.mean(surf_ice["t_skin"])) + 5.0


def test_ocean_si3_wx1_restart_bit_exact_roundtrip(tmp_path: Path) -> None:
    """NEMO+SI3 WX-1 lossless zstd_shuffle checkpoint reads back bit-exact and resumes identically."""
    spec = _make_regional_ocean_spec(ny=10, nx=10, dx_m=20_000.0, lat0=72.0)
    sstatic = {
        "lat_deg": spec.lat2d.astype(np.float32),
        "lon_deg": spec.lon2d.astype(np.float32),
        "lsm": np.zeros((spec.ny, spec.nx), dtype=np.float32),
        "lake_mask": np.zeros((spec.ny, spec.nx), dtype=np.float32),
        "bathy_depth_m": np.full((spec.ny, spec.nx), 200.0, dtype=np.float32),
    }
    forcing = {
        "u10m": np.full((spec.ny, spec.nx), 6.0, dtype=np.float32),
        "v10m": np.full((spec.ny, spec.nx), -3.0, dtype=np.float32),
        "t2m": np.full((spec.ny, spec.nx), 268.0, dtype=np.float32),
        "q2m": np.full((spec.ny, spec.nx), 0.002, dtype=np.float32),
        "mslp": np.full((spec.ny, spec.nx), 100_800.0, dtype=np.float32),
        "sw_down": np.full((spec.ny, spec.nx), 80.0, dtype=np.float32),
        "lw_down": np.full((spec.ny, spec.nx), 230.0, dtype=np.float32),
        "precip_liq": np.zeros((spec.ny, spec.nx), dtype=np.float32),
        "precip_ice": np.full((spec.ny, spec.nx), 1e-5, dtype=np.float32),
        "cld_fra": np.full((spec.ny, spec.nx), 0.6, dtype=np.float32),
    }
    rst_p = tmp_path / "ocean_si3_restart.wx1"
    valid_dt = datetime.datetime(2026, 11, 21, 6, 0, tzinfo=datetime.timezone.utc)

    with NemoSi3GPUSolver(spec, sstatic, nz=8) as s1:
        st0 = s1.init_from_atmosphere_and_osisaf({
            "sst_bulk": np.full((spec.ny, spec.nx), 271.5, dtype=np.float32),
            "sea_ice_frac": np.full((spec.ny, spec.nx), 0.70, dtype=np.float32),
            "sea_ice_thick": np.full((spec.ny, spec.nx), 1.10, dtype=np.float32),
            "sea_ice_tsfc": np.full((spec.ny, spec.nx), 266.0, dtype=np.float32),
        })
        s1.upload_state(st0)
        s1.step(forcing, dt_s=300.0)
        st_mid = s1.download_state()
        s1.save_restart_wx1(rst_p, valid_dt, st_mid, sim_time_sec=300.0)
        s1.step(forcing, dt_s=300.0)
        st_cont = s1.download_state()

    st_loaded, meta = NemoSi3GPUSolver.load_restart_wx1(rst_p)
    assert meta["kind"] == "tempest_v2_ocean_si3_restart"
    for k, arr in st_mid.items():
        np.testing.assert_array_equal(st_loaded[k], arr)

    with NemoSi3GPUSolver(spec, sstatic, nz=8) as s2:
        s2.upload_state(st_loaded)
        s2.step(forcing, dt_s=300.0)
        st_resumed = s2.download_state()

    for k in ("temp_k", "sal_psu", "ssh", "a_i", "v_i", "ice_frac", "ice_thick"):
        np.testing.assert_array_equal(st_resumed[k], st_cont[k])


def test_diagnostics_pressure_levels_convective_and_wx1_roundtrip(tmp_path: Path) -> None:
    """Verify 3D pressure-level interpolation, CAPE/CIN, visibility, and lossless WX-1 forecast write/read."""
    import wx1
    from tempest_v2.diagnostics import (
        STANDARD_PLEV_PA,
        compute_pressure_level_fields,
        compute_synoptic_convective_2d,
    )
    from tempest_v2.runner import write_forecast_wx1

    spec = _make_regional_ocean_spec(ny=8, nx=8, dx_m=5000.0, lat0=52.1)
    # Use high top (22 km) and 32 levels for accurate pressure-level sounding test
    spec = DomainSpec(
        name=spec.name, ny=8, nx=8, dx_m=5000.0, dy_m=5000.0, is_projected=False,
        is_periodic_lon=False, projection=None, x_m=spec.x_m, y_m=spec.y_m,
        lat2d=spec.lat2d, lon2d=spec.lon2d, map_factor=spec.map_factor,
        coriolis_f=spec.coriolis_f,
        vertical=VerticalGrid(nz=32, z_top_m=22_000.0, stretch_power=1.2),
        is_spherical=False,
    )
    hs = np.zeros((spec.ny, spec.nx), dtype=np.float64)
    z_w, z_s, _ = spec.vertical.heights_3d(hs)
    ref = VerticalGrid.reference_state(z_s, theta_sfc_k=288.15, p_sfc_pa=101_325.0, brunt_vaisala_hz=0.01)

    # Analytical ISA column (T = 288.15 - 0.0065*z, hydrostatic pressure)
    gamma = 0.0065
    t_isa = np.maximum(216.65, 288.15 - gamma * z_s)
    p_isa = 101_325.0 * np.power(t_isa / 288.15, 9.80665 / (287.058 * gamma))
    below_tp = (288.15 - gamma * z_s) >= 216.65
    z_tp = (288.15 - 216.65) / gamma
    p_tp = 101_325.0 * ((216.65 / 288.15) ** (9.80665 / (287.058 * gamma)))
    p_isa = np.where(below_tp, p_isa, p_tp * np.exp(-9.80665 * (z_s - z_tp) / (287.058 * 216.65)))
    pi_isa = np.power(p_isa / 100_000.0, 287.058 / 1004.64)
    th_isa = t_isa / pi_isa

    st_isa = {
        "u": np.full((32, 8, 9), 12.0, dtype=np.float32),
        "v": np.full((32, 9, 8), -4.0, dtype=np.float32),
        "w": np.zeros((33, 8, 8), dtype=np.float32),
        "theta_p": (th_isa - ref["theta_bar"]).astype(np.float32),
        "pi_p": (pi_isa - ref["pi_bar"]).astype(np.float32),
        "qv": np.full((32, 8, 8), 0.006, dtype=np.float32),
        "qc": np.zeros((32, 8, 8), dtype=np.float32),
        "qr": np.zeros((32, 8, 8), dtype=np.float32),
        "qi": np.zeros((32, 8, 8), dtype=np.float32),
        "qs": np.zeros((32, 8, 8), dtype=np.float32),
        "qg": np.zeros((32, 8, 8), dtype=np.float32),
    }
    pl = compute_pressure_level_fields(spec, hs, st_isa)
    # At 500 hPa (index 5 in 13-level 1000..50 hPa): exact ISA gh ~ 5574 m, T ~ 251.9 K
    idx_500 = list(STANDARD_PLEV_PA).index(50_000.0)
    t500_exact = 288.15 * ((50_000.0 / 101_325.0) ** (287.058 * gamma / 9.80665))
    z500_exact = (288.15 - t500_exact) / gamma
    assert abs(float(np.mean(pl["t_pl"][idx_500])) - t500_exact) < 0.8
    assert abs(float(np.mean(pl["gh_pl"][idx_500])) - z500_exact) < 35.0
    assert abs(float(np.mean(pl["z_pl"][idx_500])) - 9.80665 * z500_exact) < 350.0
    assert abs(float(np.mean(pl["u_pl"][idx_500])) - 12.0) < 1e-4

    # Sounding A: stable ISA -> zero CAPE; Sounding B: warm moist surface (303 K, qv=0.018) -> CAPE > 1500 J/kg
    surf_base = {
        "t2m": np.full((8, 8), 288.0, dtype=np.float32),
        "q2m": np.full((8, 8), 0.007, dtype=np.float32),
        "u10m": np.full((8, 8), 8.0, dtype=np.float32),
        "v10m": np.full((8, 8), -2.0, dtype=np.float32),
        "ustar": np.full((8, 8), 0.35, dtype=np.float32),
        "rain_acc": np.full((8, 8), 1.8, dtype=np.float32),
        "snow_acc": np.zeros((8, 8), dtype=np.float32),
        "graupel_acc": np.zeros((8, 8), dtype=np.float32),
    }
    syn_stable = compute_synoptic_convective_2d(spec, hs, st_isa, surf_base)
    assert float(np.max(syn_stable["cape_sb"])) == 0.0
    assert float(np.min(syn_stable["vis_m"])) > 10_000.0

    st_unstab = {k: v.copy() for k, v in st_isa.items()}
    # Warm moist PBL below ~1500 m + midlevel cold anomaly + dense surface fog in right half
    st_unstab["theta_p"][:3] += 15.0
    st_unstab["theta_p"][8:18] -= 6.0
    st_unstab["qv"][:3] = 0.018
    st_unstab["qc"][0, :, 4:] = 0.0005  # fog liquid water -> low vis_m
    syn_unstab = compute_synoptic_convective_2d(
        spec, hs, st_unstab, surf_base, prev_precip_total=np.zeros((8, 8), dtype=np.float32), interval_s=3600.0
    )
    assert float(np.mean(syn_unstab["cape_sb"])) > 1000.0
    assert float(np.mean(syn_unstab["cape_mu"])) >= float(np.mean(syn_unstab["cape_sb"])) - 1e-3
    assert np.all(syn_unstab["td2m"] <= surf_base["t2m"] + 1e-4)
    assert np.all((syn_unstab["rh2m"] >= 0.0) & (syn_unstab["rh2m"] <= 100.0))
    assert np.all((syn_unstab["tcc"] >= 0.0) & (syn_unstab["tcc"] <= 1.0))
    assert float(np.mean(syn_unstab["vis_m"][:, 4:])) < 500.0
    assert abs(float(np.mean(syn_unstab["precip_rate"])) - (1.8 / 3600.0)) < 1e-6

    # Write forecast WX-1 with 3D pressure levels + NEMO/SI3 diagnostics and verify round-trip
    oce_diag = {
        "sst": np.full((8, 8), 283.5, dtype=np.float32),
        "sss": np.full((8, 8), 34.2, dtype=np.float32),
        "ssh": np.full((8, 8), 0.12, dtype=np.float32),
        "mld": np.full((8, 8), 38.0, dtype=np.float32),
        "u_oce_sfc": np.full((8, 8), 0.25, dtype=np.float32),
        "v_oce_sfc": np.full((8, 8), -0.10, dtype=np.float32),
        "sea_ice_frac": np.full((8, 8), 0.45, dtype=np.float32),
        "sea_ice_thick": np.full((8, 8), 0.85, dtype=np.float32),
        "sea_ice_tsfc": np.full((8, 8), 268.0, dtype=np.float32),
        "sea_ice_albedo": np.full((8, 8), 0.68, dtype=np.float32),
        "u_ice": np.full((8, 8), 0.08, dtype=np.float32),
        "v_ice": np.full((8, 8), 0.02, dtype=np.float32),
    }
    fc_path = tmp_path / "fc_diag_test.wx1"
    write_forecast_wx1(
        fc_path,
        datetime.datetime(2026, 11, 21, 12, 0, tzinfo=datetime.timezone.utc),
        3.0,
        spec,
        hs,
        st_unstab,
        surf_base,
        include_3d_diagnostics=True,
        ocean_diag=oce_diag,
        prev_precip_total=np.zeros((8, 8), dtype=np.float32),
        interval_s=3600.0,
    )
    with wx1.open_dataset(fc_path) as ds:
        assert "plev" in ds.coords
        assert len(ds.coords["plev"]) == 13
        np.testing.assert_allclose(ds.coords["plev"].values, STANDARD_PLEV_PA)
        for vname in (
            "t_pl",
            "gh_pl",
            "z_pl",
            "theta_pl",
            "q_pl",
            "rh_pl",
            "u_pl",
            "v_pl",
            "w_pl",
            "omega_pl",
            "vo_pl",
            "div_pl",
            "qc_pl",
            "qr_pl",
            "qi_pl",
            "qs_pl",
            "qg_pl",
        ):
            assert vname in ds.data_vars
            assert ds[vname].shape == (13, 8, 8)
            assert np.all(np.isfinite(ds[vname].values))
        for vname in ("ps", "mslp", "td2m", "rh2m", "gust10m", "tcc", "cape_sb", "shear_0_6km", "vis_m", "sst", "sss", "ssh", "sea_ice_frac"):
            assert vname in ds.data_vars
            assert np.all(np.isfinite(ds[vname].values))


def test_three_nest_d1_to_d2_to_d3_coupled_ocean_si3_and_diagnostics(tmp_path: Path) -> None:
    """End-to-end GPU test of D1 (spherical+cuFFT) -> D2 -> D3 coupled with NEMO+SI3 and 3D/2D diagnostics."""
    import json
    import xarray as xr
    import wx1
    from tempest_v2.projection import LCCProjection, _make_lcc_domain
    from tempest_v2.runner import run_three_nest_pipeline

    d1_spec = _make_coarse_spherical_domain(ny=25, nx=32, nz=16, z_top_m=22_000.0)
    proj = LCCProjection(lat1_deg=48.0, lat2_deg=54.0, lat0_deg=51.0, lon0_deg=5.0)
    d2_spec = _make_lcc_domain(
        "d2_mini_3nest",
        ny=12,
        nx=12,
        dx_m=15_000.0,
        projection=proj,
        vertical=VerticalGrid(nz=16, z_top_m=22_000.0, stretch_power=1.2),
        center_lat_deg=52.0,
        center_lon_deg=4.5,
    )
    d3_spec = _make_lcc_domain(
        "d3_mini_3nest",
        ny=12,
        nx=12,
        dx_m=5_000.0,
        projection=proj,
        vertical=VerticalGrid(nz=16, z_top_m=22_000.0, stretch_power=1.2),
        center_lat_deg=52.0,
        center_lon_deg=4.5,
    )

    def _mk_static(spec: DomainSpec, global_band: bool = False):
        hs = np.zeros((spec.ny, spec.nx), dtype=np.float64)
        lsm = np.zeros((spec.ny, spec.nx), dtype=np.float32)
        if global_band:
            # Equatorial landmass around Americas/Africa + wet Arctic & North Atlantic
            land_mask = (np.abs(spec.lat2d) < 35.0) & (spec.lon2d > 40.0) & (spec.lon2d < 140.0)
            lsm[land_mask] = 1.0
        else:
            lsm[:, spec.nx // 2 :] = 1.0  # left half sea, right half land
        bathy = np.where(lsm < 0.5, 220.0, 0.0).astype(np.float32)
        sstatic = {
            "lat_deg": spec.lat2d.astype(np.float32),
            "lon_deg": spec.lon2d.astype(np.float32),
            "lsm": lsm,
            "lake_mask": np.zeros((spec.ny, spec.nx), dtype=np.float32),
            "soil_type": np.where(lsm >= 0.5, 9.0, 0.0).astype(np.float32),
            "bathy_depth_m": bathy,
            "z0m_base": np.where(lsm >= 0.5, 0.05, 0.001).astype(np.float32),
            "albedo_base": np.where(lsm >= 0.5, 0.20, 0.07).astype(np.float32),
            "veg_frac": np.where(lsm >= 0.5, 0.60, 0.0).astype(np.float32),
        }
        return hs, sstatic

    # Synthetic global Aurora WX-1 initial condition + OSI-SAF polar sea ice WX-1
    levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
    lats_a = np.linspace(-90.0, 90.0, 19, dtype=np.float32)
    lons_a = np.arange(0.0, 360.0, 15.0, dtype=np.float32)
    lat2_a, _lon2_a = np.meshgrid(lats_a, lons_a, indexing="ij")
    t2m_a = np.where(np.abs(lat2_a) > 65.0, 263.0, 284.0).astype(np.float32)
    data_vars = {
        "msl": (("time", "lat", "lon"), np.full((1, len(lats_a), len(lons_a)), 101_325.0, dtype=np.float32)),
        "2t": (("time", "lat", "lon"), t2m_a[None, :, :]),
    }
    for lev in levels:
        t_lev = np.maximum(215.0, t2m_a * ((lev / 1000.0) ** 0.2)).astype(np.float32)
        data_vars[f"t_{lev}"] = (("time", "lat", "lon"), t_lev[None, :, :])
        data_vars[f"u_{lev}"] = (("time", "lat", "lon"), np.full((1, len(lats_a), len(lons_a)), 8.0, dtype=np.float32))
        data_vars[f"v_{lev}"] = (("time", "lat", "lon"), np.full((1, len(lats_a), len(lons_a)), 1.5, dtype=np.float32))
        data_vars[f"q_{lev}"] = (("time", "lat", "lon"), np.full((1, len(lats_a), len(lons_a)), 0.004 * (lev / 1000.0), dtype=np.float32))
    aurora_path = tmp_path / "aurora_global_test.wx1"
    wx1.write_dataset(
        aurora_path,
        xr.Dataset(
            data_vars=data_vars,
            coords={
                "time": ("time", np.array(["2026-11-21T00:00:00"], dtype="datetime64[s]")),
                "lat": ("lat", lats_a),
                "lon": ("lon", lons_a),
            },
        ),
        dataset_id="aurora-global-test",
    )

    arctic_lats = np.repeat(np.array([72.0, 78.0, 84.0], dtype=np.float64), 8)
    arctic_lons = np.tile(np.linspace(0.0, 315.0, 8, dtype=np.float64), 3)
    obs_dir = tmp_path / "obs_live_cycle"
    for sub in ("sfc", "upperair", "satellite", "marine"):
        (obs_dir / sub).mkdir(parents=True, exist_ok=True)

    # 1. Multi-file OSI-SAF sea-ice product in marine/ (exact format from ingest_osi_saf.save_osi_saf_wx1)
    from tempest_v2.ingest_osi_saf import save_osi_saf_wx1

    osisaf_path = save_osi_saf_wx1(
        {
            "lat": arctic_lats.astype(np.float32),
            "lon": arctic_lons.astype(np.float32),
            "ice_conc": np.full(len(arctic_lats), 0.90, dtype=np.float32),
            "ice_thick": np.full(len(arctic_lats), 1.40, dtype=np.float32),
            "ice_uncertainty": np.full(len(arctic_lats), 0.03, dtype=np.float32),
        },
        obs_dir / "marine",
    )

    # 2. Multi-file Metop ASCAT ocean surface winds in satellite/ (no main eumetsat_ascat.wx1 file!)
    wx1.write_array(
        obs_dir / "satellite" / "eumetsat_ascat_u_wind.wx1",
        np.array([9.5, 8.8], dtype=np.float32),
        dataset_id="ascat_u",
        variable_path="u_wind",
        dimensions=["n"],
    )
    wx1.write_array(
        obs_dir / "satellite" / "eumetsat_ascat_v_wind.wx1",
        np.array([2.0, 1.5], dtype=np.float32),
        dataset_id="ascat_v",
        variable_path="v_wind",
        dimensions=["n"],
    )
    wx1.write_array(
        obs_dir / "satellite" / "eumetsat_ascat_latitude.wx1",
        np.array([52.0, 55.0], dtype=np.float32),
        dataset_id="ascat_lat",
        variable_path="latitude",
        dimensions=["n"],
    )
    wx1.write_array(
        obs_dir / "satellite" / "eumetsat_ascat_longitude.wx1",
        np.array([4.3, 15.0], dtype=np.float32),
        dataset_id="ascat_lon",
        variable_path="longitude",
        dimensions=["n"],
    )

    # 3. 3D Argo float CTD profile in marine/
    argo_z = np.array([5.0, 25.0, 100.0, 300.0], dtype=np.float64)
    wx1.write_array(obs_dir / "marine" / "argo_ocean_profiles.wx1", np.array([52.0], dtype=np.float64), dataset_id="a_lat", variable_path="latitude", dimensions=["f"])
    wx1.write_array(obs_dir / "marine" / "argo_ocean_profiles_lon.wx1", np.array([4.2], dtype=np.float64), dataset_id="a_lon", variable_path="longitude", dimensions=["f"])
    wx1.write_array(obs_dir / "marine" / "argo_ocean_depths.wx1", argo_z, dataset_id="a_z", variable_path="depth", dimensions=["z"])
    wx1.write_array(obs_dir / "marine" / "argo_ocean_profiles_temp.wx1", np.array([[285.5, 284.0, 281.0, 278.5]], dtype=np.float64), dataset_id="a_t", variable_path="temperature", dimensions=["f", "z"])
    wx1.write_array(obs_dir / "marine" / "argo_ocean_profiles_salinity.wx1", np.array([[34.6, 34.8, 35.0, 35.1]], dtype=np.float64), dataset_id="a_s", variable_path="salinity", dimensions=["f", "z"])

    # 4. WMO WIS2 surface report in sfc/ with non-zero time offset to exercise 4D-FGAT
    now_ts = datetime.datetime.now(datetime.timezone.utc).replace(minute=0, second=0, microsecond=0).timestamp()
    for suf, arr in (
        ("", np.array([101_200.0])),
        ("_lat", np.array([52.05])),
        ("_lon", np.array([4.55])),
        ("_temp", np.array([285.2])),
        ("_u_wind", np.array([8.2])),
        ("_v_wind", np.array([1.6])),
        ("_elev", np.array([2.0])),
        ("_time", np.array([now_ts - 1200.0])),
    ):
        var_p = suf[1:] if suf else "pressure"
        wx1.write_array(obs_dir / "sfc" / f"wis2_obs{suf}.wx1", arr, dataset_id=f"w{suf}", variable_path=var_p, dimensions=["s"])

    # 5. Radar QPE on D3 for Latent Heat Nudging (LHN)
    wx1.write_dataset(
        obs_dir / "radar_obs.wx1",
        xr.Dataset({"rain_rate_mm_h": (("y", "x"), np.full((d3_spec.ny, d3_spec.nx), 1.5, dtype=np.float32))}),
        dataset_id="radar-d3-test",
    )

    summary = run_three_nest_pipeline(
        forecast_hours=0.1,
        d1_forecast_hours=0.1,
        nz=16,
        ic_source="aurora",
        aurora_wx1_path=aurora_path,
        obs_dir=obs_dir,
        runs_root=tmp_path / "runs",
        cycle_restart_hours=(0.1,),
        d1_spec=d1_spec,
        d2_spec=d2_spec,
        d3_spec=d3_spec,
        d1_static=_mk_static(d1_spec, global_band=True),
        d2_static=_mk_static(d2_spec),
        d3_static=_mk_static(d3_spec),
        dt_d1_s=60.0,
        dt_d2_s=20.0,
        dt_d3_s=10.0,
        physics_every=1,
        d1_lbc_interval_s=180.0,
        lbc_interval_s=180.0,
        enable_da=True,
        enable_ocean_si3=True,
        include_3d_diagnostics=True,
        ocean_nz=8,
        ocean_coupling_interval_s=180.0,
        osisaf_wx1_path=osisaf_path,
    )

    run_dir = Path(summary["run_dir"])
    for sub in ("d1", "d2", "d3", "ocean", "restarts"):
        assert (run_dir / sub).is_dir()

    assert "var3d" in summary["da"]
    assert "d1" in summary["da"] and "var3d" in summary["da"]["d1"]
    assert "d2" in summary["da"] and "var3d" in summary["da"]["d2"]

    manifest = json.loads(Path(summary["manifest_path"]).read_text())
    assert manifest["ocean"]["enabled"] is True
    assert manifest["include_3d_diagnostics"] is True
    for dom in ("d1", "d2", "d3"):
        assert manifest[dom]["restart_path"] is not None and Path(manifest[dom]["restart_path"]).is_file()
        assert manifest[dom]["ocean_restart_path"] is not None and Path(manifest[dom]["ocean_restart_path"]).is_file()
        fc_paths = [Path(p) for p in summary[dom]["output_paths"]]
        assert len(fc_paths) >= 2
        with wx1.open_dataset(fc_paths[-1]) as ds:
            assert len(ds.coords["plev"]) == 13
            for vn in (
                "t_pl",
                "gh_pl",
                "z_pl",
                "theta_pl",
                "q_pl",
                "rh_pl",
                "u_pl",
                "v_pl",
                "w_pl",
                "omega_pl",
                "vo_pl",
                "div_pl",
                "qc_pl",
                "qr_pl",
                "qi_pl",
                "qs_pl",
                "qg_pl",
                "ps",
                "mslp",
                "cape_sb",
                "td2m",
                "gust10m",
                "vis_m",
                "sst",
                "sss",
                "ssh",
                "mld",
                "sea_ice_frac",
                "sea_ice_thick",
                "oce_temp",
                "oce_sal",
                "oce_u",
                "oce_v",
                "oce_tke",
                "oce_avm",
                "oce_avt",
                "si3_a_i",
                "si3_v_i",
            ):
                assert vn in ds.data_vars, f"Missing {vn} in {dom} forecast WX-1"
                assert np.all(np.isfinite(ds[vn].values))

    oce_st, oce_meta = NemoSi3GPUSolver.load_restart_wx1(summary["d1"]["ocean_restart_path"])
    assert oce_meta["kind"] == "tempest_v2_ocean_si3_restart"
    assert float(np.max(oce_st["ice_frac"])) > 0.50
    assert np.all(np.isfinite(oce_st["temp_k"]))


def test_d1_coarse_0p4_and_0p5_specs_and_0p4_single_tile_coupled_ocean_smoke() -> None:
    """Coarse D1 grids (0.4 deg / 0.5 deg) stay distinct in restart validator and 0.4 deg + ocean+SI3 fits 1 tile on 6 GB VRAM."""
    from tempest_v2.restart_validator import DOMAIN_GEOMETRIES, normalize_domain_name

    s04 = get_domain_spec("d1_global_0p4", nz=65)
    s05 = get_domain_spec("d1_global_0p5", nz=65)
    assert s04.name == "d1_global_0p4" and (s04.nz, s04.ny, s04.nx) == (65, 451, 900)
    assert s05.name == "d1_global_0p5" and (s05.nz, s05.ny, s05.nx) == (65, 361, 720)
    assert s04.is_spherical and s04.is_periodic_lon
    assert s05.is_spherical and s05.is_periodic_lon

    assert normalize_domain_name("d1_global_0p25") == "d1_global_0p25"
    assert normalize_domain_name("d1_global_0p4") == "d1_0p4"
    assert normalize_domain_name("d1_global_0p5") == "d1_global_0p5"
    assert DOMAIN_GEOMETRIES["d1_global_0p4"] == (451, 900)
    assert DOMAIN_GEOMETRIES["d1_global_0p5"] == (361, 720)

    hs = np.zeros((s04.ny, s04.nx), dtype=np.float64)
    sstatic = {
        "lat_deg": s04.lat2d.astype(np.float32),
        "lon_deg": s04.lon2d.astype(np.float32),
        "lsm": np.zeros((s04.ny, s04.nx), dtype=np.float32),
        "lake_mask": np.zeros((s04.ny, s04.nx), dtype=np.float32),
        "soil_type": np.zeros((s04.ny, s04.nx), dtype=np.float32),
        "bathy_depth_m": np.full((s04.ny, s04.nx), 1500.0, dtype=np.float32),
        "z0m_base": np.full((s04.ny, s04.nx), 1e-4, dtype=np.float32),
        "albedo_base": np.full((s04.ny, s04.nx), 0.06, dtype=np.float32),
        "veg_frac": np.zeros((s04.ny, s04.nx), dtype=np.float32),
    }

    ocean = NemoSi3GPUSolver(s04, sstatic, nz=8, n_bt_substeps=6, n_evp_substeps=20)
    solver = None
    try:
        solver = NonHydrostaticGPUSolver(
            s04.vertical,
            s04.dx_m,
            s04.dy_m,
            hs,
            msf=s04.map_factor,
            f_cor=s04.coriolis_f,
            has_moist=True,
            has_physics=True,
            bc_x=1,
            bc_y=0,
            use_coriolis=True,
            is_spherical=True,
            polar_filter_lat_deg=60.0,
        )
        assert solver.num_tiles == 1, f"Expected single-tile d1_global_0p4@nz=65, got {solver.num_tiles}"

        st = solver.empty_state()
        sf = solver.empty_surface_state()
        sf["sst_bulk"][:] = 285.0
        oce_st = ocean.init_from_atmosphere_and_osisaf(sf)
        ocean.upload_state(oce_st)
        sf = ocean.couple_to_surface_state(oce_st, sf)

        solver.upload_state(st, copy=False)
        solver.set_surface_static(sstatic)
        solver.upload_surface(sf)
        solver.set_physics_config(julian_day=172.0, utc_seconds=43200.0)
        solver.step_full(dt_s=90.0, n_steps=1, physics_every=1)
        out_atmo = solver.download_state(copy=False)
        out_surf = solver.download_surface()
        forcing = ocean.build_forcing_from_atmosphere(s04, out_atmo, out_surf, dt_interval_s=90.0)
        ocean.step(forcing, 90.0)
        oce_out = ocean.download_state()

        assert np.all(np.isfinite(out_atmo["u"])) and np.all(np.isfinite(out_atmo["qv"]))
        assert np.all(np.isfinite(oce_out["temp_k"]))
    finally:
        if solver is not None:
            solver.close()
        ocean.close()

    free_after, _ = cuda_vram_info()
    assert free_after > 3_500_000_000, f"VRAM leak after d1_global_0p4 coupled close: free={free_after}"

