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
    """Verify default 'd3' domain is compact NL+buffer 2.5 km (216x216, 1-tile <1.5 GiB) while keeping benelux_2p5km & d3_1p5km."""
    from tempest_v2.cuda_bridge import DEFAULT_MAX_VRAM_BYTES, plan_horizontal_tiling
    from tempest_v2.projection import get_domain_spec

    d3 = get_domain_spec("d3", nz=137)
    assert d3.name == "d3_nl_2p5km"
    assert (d3.nz, d3.ny, d3.nx) == (137, 216, 216)
    assert d3.dx_m == 2500.0
    assert d3.dy_m == 2500.0
    assert float(d3.lat2d.min()) < 50.0 and float(d3.lat2d.max()) > 54.0
    assert float(d3.lon2d.min()) < 2.0 and float(d3.lon2d.max()) > 8.5

    nt_y, nt_x, est = plan_horizontal_tiling(
        137,
        216,
        216,
        moist=True,
        has_lbc=True,
        has_iau=True,
        has_physics=True,
        max_vram_bytes=DEFAULT_MAX_VRAM_BYTES,
    )
    assert (nt_y, nt_x) == (1, 1), f"Expected 1x1 tile for d3_nl_2p5km x137, got {nt_y}x{nt_x} ({est/1048576:.0f} MiB)"
    assert est < int(1.5 * 1024**3), f"Expected < 1.5 GiB VRAM for 216x216x137, got {est/1048576:.0f} MiB"

    d3_ben = get_domain_spec("d3_benelux_2p5km", nz=137)
    assert d3_ben.name == "d3_benelux_de_2p5km"
    assert (d3_ben.nz, d3_ben.ny, d3_ben.nx) == (137, 432, 432)
    assert d3_ben.dx_m == 2500.0

    d3_1p5 = get_domain_spec("d3_1p5km", nz=137)
    assert d3_1p5.name == "d3_benelux_de_1p5km"
    assert (d3_1p5.nz, d3_1p5.ny, d3_1p5.nx) == (137, 720, 720)
    assert d3_1p5.dx_m == 1500.0


def test_lbc_hydrometeor_decay_and_sat_cap_with_physics() -> None:
    """Verify Davies hydrometeor relaxation damps rim qi/qs while active physics LBC qv avoids supersaturation runaway."""
    from tempest_v2.cuda_bridge import NonHydrostaticGPUSolver

    nz, ny, nx = 24, 16, 48
    vgrid = VerticalGrid(nz=nz, z_top_m=15_000.0, stretch_power=1.2)
    hs = np.zeros((ny, nx), dtype=np.float64)
    solver = NonHydrostaticGPUSolver(
        vgrid,
        dx_m=2500.0,
        dy_m=2500.0,
        hs_m=hs,
        has_moist=True,
        has_physics=True,
        bc_x=2,
        bc_y=2,
        use_coriolis=False,
        lbc_rim_width=10,
        lbc_tau_max_s=0.02,
    )
    try:
        solver.set_surface_static(solver.default_surface_static())
        solver.set_physics_config(enable_microphysics=1, enable_pbl=0, enable_radiation=0, enable_surface=0)
        st = solver.empty_state()
        st["u"][:] = 5.0
        st["qv"][:] = 0.002
        st["qi"][:] = 1.0e-3
        st["qs"][:] = 1.0e-3
        lbc = {k: st[k].copy() for k in ("u", "v", "w", "theta_p", "pi_p", "qv")}
        lbc["qv"][-6:, :, :] = 0.015  # Extreme supersaturation in upper-air LBC
        solver.upload_state(st)
        solver.upload_surface(solver.empty_surface_state())
        solver.set_lbc_target(lbc)
        solver.step_rk3(dt_s=10.0, n_steps=30)
        out = solver.download_state()

        rim_qi = float(out["qi"][10, 6:10, 0:2].mean())
        int_qi = float(out["qi"][10, 6:10, nx // 2].mean())
        assert rim_qi < 0.25 * int_qi, f"Expected Davies rim hydrometeor decay: rim={rim_qi:.3e} vs int={int_qi:.3e}"
        assert float(out["qv"][-4:, :, 0].max()) < 0.005, "Expected saturation-capped LBC qv nudging at cold upper levels"
    finally:
        solver.close()


def test_coastal_skt_water_masking_and_t0_10m_wind_reduction(tmp_path, monkeypatch) -> None:
    """Verify land-masked skt prevents coastal sea freezing, vertical qv caps at saturation, and T+0 10m wind uses log law."""
    from pathlib import Path
    import datetime
    import xarray as xr
    import wx1
    from tempest_v2 import ic_builder
    from tempest_v2.runner import write_forecast_wx1

    ny, nx = 8, 12
    vgrid = VerticalGrid(nz=16, z_top_m=15_000.0, stretch_power=1.2)
    spec = _make_lcc_domain("test_coast_sst", ny=ny, nx=nx, dx_m=2500.0, projection=LCCProjection(), vertical=vgrid)
    hs_m = np.zeros((ny, nx), dtype=np.float64)

    lats = np.linspace(float(spec.lat2d.min()) - 0.5, float(spec.lat2d.max()) + 0.5, 16)
    lons = np.linspace(float(spec.lon2d.min()) - 0.5, float(spec.lon2d.max()) + 0.5, 20)
    lon2, lat2 = np.meshgrid(lons, lats)
    mid_lon = float(np.mean(lons))
    lsm_ecmwf = np.where(lon2 >= mid_lon, 1.0, 0.0)
    skt_ecmwf = np.where(lsm_ecmwf >= 0.5, 267.0, 282.5)  # Freezing land next to +9.4C sea!
    p_hpa = np.array([1000, 850, 700, 500, 300, 100], dtype=np.float64)
    z_pl = np.array([110.0, 1450.0, 3000.0, 5570.0, 9160.0, 16_200.0])
    t_pl = 282.0 - 0.0065 * z_pl

    def _da2(arr):
        return xr.DataArray(arr.astype(np.float64), dims=("latitude", "longitude"), coords={"latitude": lats, "longitude": lons})

    def _da3(prof1d):
        arr = np.broadcast_to(prof1d[:, None, None], (len(p_hpa), len(lats), len(lons))).copy()
        return xr.DataArray(arr, dims=("isobaricInhPa", "latitude", "longitude"), coords={"isobaricInhPa": p_hpa, "latitude": lats, "longitude": lons})

    pl_vars = {
        "t": (_da3(t_pl), lats, lons),
        "u": (_da3(np.full(len(p_hpa), 10.0)), lats, lons),
        "v": (_da3(np.zeros(len(p_hpa))), lats, lons),
        "q": (_da3(np.full(len(p_hpa), 0.020)), lats, lons),  # Supersaturated 20 g/kg input -> must be capped!
        "gh": (_da3(z_pl), lats, lons),
    }
    sfc_vars = {
        "msl": (_da2(np.full_like(lat2, 101_325.0)), lats, lons),
        "t2m": (_da2(np.full_like(lat2, 276.0)), lats, lons),
        "d2m": (_da2(np.full_like(lat2, 275.0)), lats, lons),
        "skt": (_da2(skt_ecmwf), lats, lons),
        "lsm": (_da2(lsm_ecmwf), lats, lons),
        "sd": (_da2(np.zeros_like(lat2)), lats, lons),
    }

    sfc_p = tmp_path / "dummy_sfc.grib2"
    pl_p = tmp_path / "dummy_pl.grib2"
    sfc_p.write_bytes(b"grib")
    pl_p.write_bytes(b"grib")
    monkeypatch.setattr(ic_builder, "_read_cfgrib_var_map", lambda p: sfc_vars if "sfc" in str(p) else pl_vars)

    sstatic = {
        "lsm": np.where(spec.lon2d >= mid_lon + 0.05, 1.0, 0.0).astype(np.float32),  # Coastal water overlaps ECMWF land edge!
        "lake_mask": np.where((spec.lon2d >= mid_lon + 0.05) & (spec.lat2d > float(np.mean(lats))), 1.0, 0.0).astype(np.float32),
        "z0m_base": np.where(spec.lon2d >= mid_lon + 0.05, 0.15, 1e-4).astype(np.float32),
        "bathy_depth_m": np.full((ny, nx), 20.0, dtype=np.float32),
    }
    state3d, surf, _ = ic_builder.ecmwf_grib_to_domain_state(sfc_p, pl_p, spec, hs_m, surface_static=sstatic)

    is_ocean = (sstatic["lsm"] < 0.5) & (sstatic["lake_mask"] < 0.5)
    assert float(surf["sst_bulk"][is_ocean].min()) > 280.0, f"Land skt contaminated coastal sst_bulk: {surf['sst_bulk'][is_ocean].min()}"
    assert float(surf["t_skin"][is_ocean].min()) > 280.0
    assert float(state3d["qv"][0].max()) < 0.008, "Vertical qv interpolation failed to cap supersaturation"

    # Also verify T+0 10m wind uses logarithmic reduction and dynamic lake_tml uses t_skin
    out_wx1 = Path(tmp_path) / "t0_test.wx1"
    surf["t_skin"][:] = 278.5
    write_forecast_wx1(
        out_wx1,
        datetime.datetime(2026, 11, 22, 18, 0, tzinfo=datetime.timezone.utc),
        0.0,
        spec,
        hs_m,
        state3d,
        surf,
        include_3d_diagnostics=False,
        surface_static=sstatic,
    )
    with wx1.open_dataset(out_wx1) as ds:
        w10_land = float(np.hypot(ds["u10m_geo"].values, ds["v10m_geo"].values)[sstatic["lsm"] >= 0.5].mean())
        assert w10_land < 8.5, f"Expected T+0 log-law 10m wind reduction < 8.5 m/s for 10 m/s k=0 wind, got {w10_land:.2f}"
        assert abs(float(ds["lake_tml"].values[sstatic["lake_mask"] >= 0.5].mean()) - 278.5) < 1e-3


def test_ecmwf_t0_10m_wind_direct_and_wx1_provenance(tmp_path, monkeypatch) -> None:
    """Verify ECMWF u10/v10 populate rotated T+0 grid 10m wind and write_forecast_wx1 records provenance attrs."""
    import datetime
    import xarray as xr
    import wx1
    from tempest_v2 import ic_builder
    from tempest_v2.runner import write_forecast_wx1

    ny, nx = 8, 12
    vgrid = VerticalGrid(nz=16, z_top_m=15_000.0, stretch_power=1.2)
    spec = _make_lcc_domain("test_t0_u10", ny=ny, nx=nx, dx_m=2500.0, projection=LCCProjection(), vertical=vgrid)
    hs_m = np.zeros((ny, nx), dtype=np.float64)

    lats = np.linspace(float(spec.lat2d.min()) - 0.5, float(spec.lat2d.max()) + 0.5, 14)
    lons = np.linspace(float(spec.lon2d.min()) - 0.5, float(spec.lon2d.max()) + 0.5, 18)
    lon2, lat2 = np.meshgrid(lons, lats)
    p_hpa = np.array([1000, 850, 700, 500, 300, 100], dtype=np.float64)
    z_pl = np.array([110.0, 1450.0, 3000.0, 5570.0, 9160.0, 16_200.0])
    t_pl = 282.0 - 0.0065 * z_pl

    def _da2(val):
        return xr.DataArray(np.full_like(lat2, val, dtype=np.float64), dims=("latitude", "longitude"), coords={"latitude": lats, "longitude": lons})

    def _da3(prof1d):
        arr = np.broadcast_to(prof1d[:, None, None], (len(p_hpa), len(lats), len(lons))).copy()
        return xr.DataArray(arr, dims=("isobaricInhPa", "latitude", "longitude"), coords={"isobaricInhPa": p_hpa, "latitude": lats, "longitude": lons})

    pl_vars = {
        "t": (_da3(t_pl), lats, lons),
        "u": (_da3(np.full(len(p_hpa), 12.0)), lats, lons),
        "v": (_da3(np.full(len(p_hpa), 4.0)), lats, lons),
        "q": (_da3(np.full(len(p_hpa), 0.004)), lats, lons),
        "gh": (_da3(z_pl), lats, lons),
    }
    u10_target_geo = -3.4
    v10_target_geo = 2.1
    sfc_vars = {
        "msl": (_da2(101_500.0), lats, lons),
        "t2m": (_da2(278.0), lats, lons),
        "d2m": (_da2(276.0), lats, lons),
        "u10": (_da2(u10_target_geo), lats, lons),
        "v10": (_da2(v10_target_geo), lats, lons),
        "skt": (_da2(278.0), lats, lons),
        "lsm": (_da2(1.0), lats, lons),
        "sd": (_da2(0.0), lats, lons),
    }

    sfc_p = tmp_path / "dummy_sfc.grib2"
    pl_p = tmp_path / "dummy_pl.grib2"
    sfc_p.write_bytes(b"grib")
    pl_p.write_bytes(b"grib")
    monkeypatch.setattr(ic_builder, "_read_cfgrib_var_map", lambda p: sfc_vars if "sfc" in str(p) else pl_vars)

    state3d, surf, _ = ic_builder.ecmwf_grib_to_domain_state(sfc_p, pl_p, spec, hs_m)
    assert float(np.max(np.abs(surf["u10m"]))) > 0.5

    out_wx1 = tmp_path / "t0_direct_u10_prov.wx1"
    write_forecast_wx1(
        out_wx1,
        datetime.datetime(2026, 11, 22, 18, 0, tzinfo=datetime.timezone.utc),
        0.0,
        spec,
        hs_m,
        state3d,
        surf,
        include_3d_diagnostics=False,
        provenance_attrs={
            "cy46_physics": "true",
            "pbl_last": "true",
            "dt_s": "6.000",
            "lbc_rim_width": "14",
        },
    )
    with wx1.open_dataset(out_wx1) as ds:
        assert float(np.max(np.abs(ds["u10m_geo"].values - u10_target_geo))) < 0.05
        assert float(np.max(np.abs(ds["v10m_geo"].values - v10_target_geo))) < 0.05
        assert ds.attrs.get("git_commit") not in (None, "", "unknown")
        assert ds.attrs.get("nz") == "16"
        assert ds.attrs.get("cy46_physics") == "true"
        assert ds.attrs.get("pbl_last") == "true"
        assert ds.attrs.get("dt_s") == "6.000"
        assert ds.attrs.get("lbc_rim_width") == "14"



