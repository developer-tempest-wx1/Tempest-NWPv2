"""Gate B3 end-to-end tests: bit-exact WX-1 restart, D2->D3 2-nest + 3D-Var/IAU + WX-1 output, Notebook v4."""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import numpy as np
import wx1

from tempest_v2.cuda_bridge import NonHydrostaticGPUSolver
from tempest_v2.ic_builder import (
    load_tempest_restart_wx1,
    save_tempest_restart_wx1,
    stagger_uv,
)
from tempest_v2.projection import LCCProjection, VerticalGrid, _make_lcc_domain
from tempest_v2.runner import forecast_output_schedule_hours, run_domain_forecast
from tempest_v2.lbc_manager import LBCInterpolator
from tempest_v2.verification import (
    compute_mslp_pa,
    load_station_obs_wx1,
    run_domain_3dvar,
    verify_against_stations,
)


def _mini_spec(name: str, dx_m: float, nx: int = 16, ny: int = 16, nz: int = 24):
    vgrid = VerticalGrid(nz=nz, z_top_m=18_000.0, stretch_power=1.3)
    proj = LCCProjection(lat1_deg=48.0, lat2_deg=54.0, lat0_deg=52.1, lon0_deg=5.18)
    return _make_lcc_domain(
        name,
        ny=ny,
        nx=nx,
        dx_m=dx_m,
        projection=proj,
        vertical=vgrid,
        center_lat_deg=52.1,
        center_lon_deg=5.18,
    )


def test_wx1_restart_bit_exact_roundtrip(tmp_path: Path) -> None:
    """Verify bit-exact save/load of 3D+2D+soil WX-1 restart and bit-identical continuation."""
    spec = _mini_spec("restart_test", dx_m=3000.0, nx=12, ny=12, nz=16)
    hs_m = np.full((spec.ny, spec.nx), 42.5, dtype=np.float64)
    sstatic = {
        "lat_deg": spec.lat2d.astype(np.float32),
        "lon_deg": spec.lon2d.astype(np.float32),
        "lsm": np.ones((spec.ny, spec.nx), dtype=np.float32),
        "lake_mask": np.zeros((spec.ny, spec.nx), dtype=np.float32),
        "soil_type": np.full((spec.ny, spec.nx), 9.0, dtype=np.float32),
        "bathy_depth_m": np.zeros((spec.ny, spec.nx), dtype=np.float32),
        "z0m_base": np.full((spec.ny, spec.nx), 0.05, dtype=np.float32),
        "albedo_base": np.full((spec.ny, spec.nx), 0.20, dtype=np.float32),
        "veg_frac": np.full((spec.ny, spec.nx), 0.60, dtype=np.float32),
    }

    init_dt = datetime.datetime(2026, 6, 21, 12, 0, tzinfo=datetime.timezone.utc)
    solver_a = NonHydrostaticGPUSolver(
        spec.vertical, spec.dx_m, spec.dy_m, hs_m, msf=spec.map_factor, f_cor=spec.coriolis_f, has_moist=True, has_physics=True
    )
    try:
        st0 = solver_a.empty_state()
        st0["u"][:] = 6.5
        st0["v"][:] = -2.5
        st0["theta_p"][:] = 1.25
        st0["qv"][:] = 0.007
        st0["qc"][3:6] = 2.0e-4
        sf0 = solver_a.empty_surface_state()
        sf0["snow_swe"][0, 0] = 3.25
        solver_a.upload_state(st0)
        solver_a.set_surface_static(sstatic)
        solver_a.upload_surface(sf0)
        solver_a.set_physics_config(julian_day=172.0, utc_seconds=43_200.0, rad_interval_s=0.0)
        solver_a.step_full(dt_s=4.0, n_steps=2, physics_every=1)
        st_mid = solver_a.download_state()
        sf_mid = solver_a.download_surface()

        chk_path = tmp_path / "checkpoint_test.wx1"
        save_tempest_restart_wx1(chk_path, init_dt, spec, hs_m, st_mid, sf_mid, sim_time_sec=8.0)
        st_loaded, sf_loaded, meta = load_tempest_restart_wx1(chk_path)

        assert np.array_equal(meta["hs_m"], hs_m)
        for k, arr in st_mid.items():
            assert np.array_equal(st_loaded[k], arr), f"3D field '{k}' mismatch on WX-1 reload"
        for k, arr in sf_mid.items():
            assert np.array_equal(sf_loaded[k], arr), f"Surface field '{k}' mismatch on WX-1 reload"

        # Continue Solver A in-memory for 2 more steps and Solver B from reloaded WX-1
        solver_a.step_full(dt_s=4.0, n_steps=2, physics_every=1)
        st_a_end = solver_a.download_state()
        sf_a_end = solver_a.download_surface()

        solver_b = NonHydrostaticGPUSolver(
            spec.vertical, spec.dx_m, spec.dy_m, meta["hs_m"], msf=spec.map_factor, f_cor=spec.coriolis_f, has_moist=True, has_physics=True
        )
        try:
            solver_b.upload_state(st_loaded)
            solver_b.set_surface_static(sstatic)
            solver_b.upload_surface(sf_loaded)
            # Match advanced clock (started 43200 + 2*4s)
            solver_b.set_physics_config(julian_day=172.0, utc_seconds=43_208.0, rad_interval_s=0.0)
            solver_b.step_full(dt_s=4.0, n_steps=2, physics_every=1)
            st_b_end = solver_b.download_state()
            sf_b_end = solver_b.download_surface()

            for k in ("u", "v", "w", "theta_p", "pi_p", "qv", "qc", "qr"):
                max_diff = float(np.max(np.abs(st_a_end[k] - st_b_end[k])))
                assert max_diff == 0.0, f"Continuation drift on '{k}': max_diff={max_diff}"
            for k in ("t_skin", "soil_t", "soil_m", "t2m", "pblh"):
                max_diff = float(np.max(np.abs(sf_a_end[k] - sf_b_end[k])))
                assert max_diff == 0.0, f"Surface continuation drift on '{k}': max_diff={max_diff}"
        finally:
            solver_b.close()
    finally:
        solver_a.close()


def test_two_nest_d2_to_d3_lbc_da_and_wx1_output(tmp_path: Path) -> None:
    """Verify D2->D3 nesting via WX-1 LBCs, KNMI/DWD 3D-Var+IAU O-A improvement, and WX-1 outputs."""
    assert forecast_output_schedule_hours(360, ecmwf_schedule=True) == (
        list(range(0, 145, 3)) + list(range(150, 361, 6))
    )

    d2_spec = _mini_spec("d2_mini", dx_m=7500.0, nx=16, ny=16, nz=24)
    d3_spec = _mini_spec("d3_mini", dx_m=1500.0, nx=16, ny=16, nz=24)
    hs_d2 = np.zeros((d2_spec.ny, d2_spec.nx), dtype=np.float64)
    hs_d3 = np.zeros((d3_spec.ny, d3_spec.nx), dtype=np.float64)

    def _mk_sstatic(spec):
        return {
            "lat_deg": spec.lat2d.astype(np.float32),
            "lon_deg": spec.lon2d.astype(np.float32),
            "lsm": np.ones((spec.ny, spec.nx), dtype=np.float32),
            "lake_mask": np.zeros((spec.ny, spec.nx), dtype=np.float32),
            "soil_type": np.full((spec.ny, spec.nx), 9.0, dtype=np.float32),
            "bathy_depth_m": np.zeros((spec.ny, spec.nx), dtype=np.float32),
            "z0m_base": np.full((spec.ny, spec.nx), 0.05, dtype=np.float32),
            "albedo_base": np.full((spec.ny, spec.nx), 0.20, dtype=np.float32),
            "veg_frac": np.full((spec.ny, spec.nx), 0.60, dtype=np.float32),
        }

    def _mk_state(spec):
        nz, ny, nx = spec.nz, spec.ny, spec.nx
        u_c = np.full((nz, ny, nx), 6.0, dtype=np.float32)
        v_c = np.full((nz, ny, nx), 2.0, dtype=np.float32)
        u_st, v_st = stagger_uv(u_c, v_c)
        return {
            "u": u_st,
            "v": v_st,
            "w": np.zeros((nz + 1, ny, nx), dtype=np.float32),
            "theta_p": np.zeros((nz, ny, nx), dtype=np.float32),
            "pi_p": np.zeros((nz, ny, nx), dtype=np.float32),
            "qv": np.full((nz, ny, nx), 0.006, dtype=np.float32),
            "qc": np.zeros((nz, ny, nx), dtype=np.float32),
            "qr": np.zeros((nz, ny, nx), dtype=np.float32),
            "qi": np.zeros((nz, ny, nx), dtype=np.float32),
            "qs": np.zeros((nz, ny, nx), dtype=np.float32),
            "qg": np.zeros((nz, ny, nx), dtype=np.float32),
        }

    def _mk_surf(spec):
        return {
            "t_skin": np.full((spec.ny, spec.nx), 288.0, dtype=np.float32),
            "soil_t": np.full((4, spec.ny, spec.nx), 287.0, dtype=np.float32),
            "soil_m": np.full((4, spec.ny, spec.nx), 0.26, dtype=np.float32),
            "snow_swe": np.zeros((spec.ny, spec.nx), dtype=np.float32),
            "sst_bulk": np.full((spec.ny, spec.nx), 286.0, dtype=np.float32),
            "dt_cool": np.zeros((spec.ny, spec.nx), dtype=np.float32),
            "dt_warm": np.zeros((spec.ny, spec.nx), dtype=np.float32),
            "lake_tbot": np.full((spec.ny, spec.nx), 283.0, dtype=np.float32),
            "lake_hmld": np.full((spec.ny, spec.nx), 2.5, dtype=np.float32),
            "lake_ctheta": np.full((spec.ny, spec.nx), 0.65, dtype=np.float32),
            "lake_hice": np.zeros((spec.ny, spec.nx), dtype=np.float32),
            "rain_acc": np.zeros((spec.ny, spec.nx), dtype=np.float32),
            "snow_acc": np.zeros((spec.ny, spec.nx), dtype=np.float32),
            "graupel_acc": np.zeros((spec.ny, spec.nx), dtype=np.float32),
            "sw_down": np.zeros((spec.ny, spec.nx), dtype=np.float32),
            "lw_down": np.zeros((spec.ny, spec.nx), dtype=np.float32),
            "shf": np.zeros((spec.ny, spec.nx), dtype=np.float32),
            "lhf": np.zeros((spec.ny, spec.nx), dtype=np.float32),
            "ustar": np.zeros((spec.ny, spec.nx), dtype=np.float32),
            "pblh": np.zeros((spec.ny, spec.nx), dtype=np.float32),
            "t2m": np.full((spec.ny, spec.nx), 288.0, dtype=np.float32),
            "q2m": np.full((spec.ny, spec.nx), 0.006, dtype=np.float32),
            "u10m": np.full((spec.ny, spec.nx), 6.0, dtype=np.float32),
            "v10m": np.full((spec.ny, spec.nx), 2.0, dtype=np.float32),
        }

    init_dt = datetime.datetime(2026, 6, 21, 12, 0, tzinfo=datetime.timezone.utc)
    lbc_dir = tmp_path / "lbc_d3"

    # 1. Outer D2 integration exporting WX-1 LBCs
    run_domain_forecast(
        d2_spec,
        hs_d2,
        _mk_sstatic(d2_spec),
        _mk_state(d2_spec),
        _mk_surf(d2_spec),
        init_dt,
        forecast_hours=0.05,  # 180 s
        dt_s=10.0,
        physics_every=2,
        lbc_export_child=(d3_spec, hs_d3, lbc_dir, 60.0),
    )
    lbc_files = sorted(lbc_dir.glob(f"lbc_{d3_spec.name}_step_*.wx1"))
    assert len(lbc_files) >= 3

    # 2. Write synthetic KNMI + DWD station WX-1 containers matching ingest schema & test 3D-Var O-B -> O-A
    st_d3 = _mk_state(d3_spec)
    sf_d3 = _mk_surf(d3_spec)
    mslp_bg = compute_mslp_pa(d3_spec, hs_d3, st_d3)

    obs_dir = tmp_path / "obs"
    obs_dir.mkdir()
    lat_pts = d3_spec.lat2d[6:10, 6:10].ravel()
    lon_pts = d3_spec.lon2d[6:10, 6:10].ravel()
    mslp_obs = (mslp_bg[6:10, 6:10].ravel() - 350.0).astype(np.float32)
    t_obs = np.full_like(mslp_obs, 290.5)  # +2.5 K warmer than 288 K bg
    u_obs = np.full_like(mslp_obs, 8.5)
    v_obs = np.full_like(mslp_obs, 3.0)
    elev_obs = np.zeros_like(mslp_obs)
    time_obs = np.full(len(lat_pts), init_dt.timestamp(), dtype=np.float64)

    for prefix, slc in (("knmi_insitu_obs", slice(0, 8)), ("dwd_poi_obs", slice(8, 16))):
        wx1.write_array(obs_dir / f"{prefix}.wx1", mslp_obs[slc], dataset_id=prefix, variable_path="pressure", dimensions=["station_id"], codec="zstd_shuffle")
        for nm, arr in (
            ("lat", lat_pts[slc].astype(np.float32)),
            ("lon", lon_pts[slc].astype(np.float32)),
            ("elev", elev_obs[slc]),
            ("temp", t_obs[slc]),
            ("u_wind", u_obs[slc]),
            ("v_wind", v_obs[slc]),
            ("time", time_obs[slc]),
        ):
            wx1.write_array(obs_dir / f"{prefix}_{nm}.wx1", arr, dataset_id=f"{prefix}_{nm}", variable_path=nm, dimensions=["station_id"], codec="zstd_shuffle")

    obs_list, stn_rows = load_station_obs_wx1(obs_dir, d3_spec, valid_time_utc=init_dt)
    assert len(stn_rows) == 16
    assert len(obs_list) == 64  # 4 vars per station

    v_bg = verify_against_stations(d3_spec, st_d3, sf_d3, hs_d3, stn_rows)
    ana_st, ana_sf, iau_f, vres = run_domain_3dvar(d3_spec, st_d3, sf_d3, hs_d3, np.ones((16, 16), dtype=np.float32), obs_list)
    assert vres.final_cost_j < vres.initial_cost_j0
    v_ana = verify_against_stations(d3_spec, ana_st, ana_sf, hs_d3, stn_rows)
    assert v_ana["all"]["t2m_rmse_k"] < v_bg["all"]["t2m_rmse_k"]
    assert v_ana["all"]["mslp_rmse_hpa"] < v_bg["all"]["mslp_rmse_hpa"]

    # 3. Inner 1.5 km D3 forecast with WX-1 LBCs, IAU, substepped physics & WX-1 output
    out_d3 = tmp_path / "d3_out"
    res_d3 = run_domain_forecast(
        d3_spec,
        hs_d3,
        _mk_sstatic(d3_spec),
        ana_st,
        ana_sf,
        init_dt,
        forecast_hours=0.05,
        dt_s=4.0,
        physics_every=3,
        lbc_interpolator=LBCInterpolator(lbc_dir, d3_spec.name),
        iau_forcing=iau_f,
        iau_duration_s=90.0,
        output_dir=out_d3,
        output_hours=[0.0, 0.05],
        save_restart_path=tmp_path / "restart_d3.wx1",
    )
    assert len(res_d3["output_paths"]) == 2
    assert (tmp_path / "restart_d3.wx1").is_file()

    with wx1.open_dataset(res_d3["output_paths"][-1]) as fds:
        assert "t2m" in fds and "mslp" in fds and "rain_acc" in fds
        assert fds["t2m"].attrs.get("units") == "K"
        assert np.issubdtype(fds["time"].dtype, np.datetime64)
        t2_vals = np.asarray(fds["t2m"].values, dtype=np.float64)
        assert np.all(np.isfinite(t2_vals))
        assert 230.0 <= float(t2_vals.min()) and float(t2_vals.max()) <= 325.0


def test_colab_notebook_v4_valid_and_fixed() -> None:
    """Validate bundled Colab Aurora 1.5 notebook v4 structure and memory/WX-1 fixes."""
    nb_path = Path(__file__).resolve().parents[1] / "notebooks" / "aurora15_colab_fixed_v4.ipynb"
    assert nb_path.is_file(), f"Missing notebook: {nb_path}"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    cells = nb.get("cells", [])
    assert len(cells) >= 8
    src = "\n".join("".join(c.get("source", [])) for c in cells)
    assert "tempest-wx1" in src
    assert "mem_get_info" in src
    assert "datetime64" in src
    assert "E2STUDIO_VOCAB" in src and '"units"' in src


def test_multisource_da_radiosonde_ztd_marine_lhn_and_fgat(tmp_path: Path) -> None:
    """Verify multi-source WX-1 assimilation (radiosonde, SondeHub, E-GVAP ZTD, AMV, marine, OSI-SAF, LHN, 4D-FGAT)."""
    from dataclasses import replace
    from tempest_v2.projection import get_domain_spec

    d3_base = get_domain_spec("d3", nz=12, z_top_m=18_000.0)
    cy, cx = d3_base.ny // 2, d3_base.nx // 2
    sy, sx = slice(cy - 8, cy + 8), slice(cx - 8, cx + 8)
    spec = replace(
        d3_base,
        ny=16,
        nx=16,
        x_m=d3_base.x_m[sx].copy(),
        y_m=d3_base.y_m[sy].copy(),
        lat2d=d3_base.lat2d[sy, sx].copy(),
        lon2d=d3_base.lon2d[sy, sx].copy(),
        map_factor=d3_base.map_factor[sy, sx].copy(),
        coriolis_f=d3_base.coriolis_f[sy, sx].copy(),
    )
    hs = np.zeros((16, 16), dtype=np.float64)
    lsm = np.ones((16, 16), dtype=np.float32)
    lsm[:6, :6] = 0.0  # Marine corner for buoy SST & OSI-SAF sea-ice

    init_dt = datetime.datetime(2026, 11, 21, 12, 0, tzinfo=datetime.timezone.utc)
    t_epoch = init_dt.timestamp()
    lat_c = float(spec.lat2d[8, 8])
    lon_c = float(spec.lon2d[8, 8])
    lat_sea = float(spec.lat2d[4, 4])
    lon_sea = float(spec.lon2d[4, 4])

    obs_root = tmp_path / "obs_run"
    for sub in ("sfc", "upperair", "satellite", "marine"):
        (obs_root / sub).mkdir(parents=True)

    # 1. WIS2 surface station (with +1800s offset for 4D-FGAT)
    wx1.write_array(obs_root / "sfc" / "wis2_obs.wx1", np.array([101100.0], dtype=np.float32), dataset_id="wis2", variable_path="pressure", dimensions=["station"], codec="zstd_shuffle")
    for k, v in (("lat", [lat_c]), ("lon", [lon_c]), ("temp", [289.5]), ("u_wind", [7.0]), ("v_wind", [2.5]), ("elev", [5.0]), ("time", [t_epoch + 1800.0])):
        wx1.write_array(obs_root / "sfc" / f"wis2_obs_{k}.wx1", np.array(v, dtype=np.float64 if k == "time" else np.float32), dataset_id=f"wis2_{k}", variable_path=k, dimensions=["station"], codec="zstd_shuffle")

    # 2. Upper-air radiosonde & SondeHub sounding levels (700 hPa & 500 hPa)
    wx1.write_array(obs_root / "upperair" / "radiosonde_obs.wx1", np.array([70000.0, 50000.0], dtype=np.float32), dataset_id="rs", variable_path="pressure", dimensions=["level_id"], codec="zstd_shuffle")
    for k, v in (("lat", [lat_c, lat_c]), ("lon", [lon_c, lon_c]), ("temp", [268.0, 251.0]), ("u_wind", [14.0, 22.0]), ("v_wind", [4.0, 6.0]), ("time", [t_epoch, t_epoch])):
        wx1.write_array(obs_root / "upperair" / f"radiosonde_obs_{k}.wx1", np.array(v, dtype=np.float64 if k == "time" else np.float32), dataset_id=f"rs_{k}", variable_path=k, dimensions=["level_id"], codec="zstd_shuffle")

    wx1.write_array(obs_root / "upperair" / "sondehub_soundings.wx1", np.array([85000.0], dtype=np.float32), dataset_id="sh", variable_path="pressure", dimensions=["level_id"], codec="zstd_shuffle")
    for k, v in (("lat", [lat_c]), ("lon", [lon_c]), ("temp", [278.0]), ("rh", [75.0]), ("u_wind", [10.0]), ("v_wind", [3.0]), ("time", [t_epoch - 900.0])):
        wx1.write_array(obs_root / "upperair" / f"sondehub_soundings_{k}.wx1", np.array(v, dtype=np.float64 if k == "time" else np.float32), dataset_id=f"sh_{k}", variable_path=k, dimensions=["level_id"], codec="zstd_shuffle")

    # 3. E-GVAP GNSS ZTD
    wx1.write_array(obs_root / "upperair" / "egvap_gnss_ztd.wx1", np.array([2.36], dtype=np.float32), dataset_id="ztd", variable_path="ztd", dimensions=["n_stations"], codec="zstd_shuffle")
    for k, v in (("lat", [lat_c]), ("lon", [lon_c]), ("elev", [10.0]), ("sigma", [0.010]), ("time", [t_epoch])):
        wx1.write_array(obs_root / "upperair" / f"egvap_gnss_ztd_{k}.wx1", np.array(v, dtype=np.float32), dataset_id=f"ztd_{k}", variable_path=k, dimensions=["n_stations"], codec="zstd_shuffle")

    # 4. EUMETSAT AMV wind + raw-only ATMS (must NOT be assimilated)
    wx1.write_array(obs_root / "satellite" / "eumetsat.wx1", np.array([40000.0], dtype=np.float32), dataset_id="amv", variable_path="pressure", dimensions=["n_obs"], codec="zstd_shuffle")
    for k, v in (("lat", [lat_c]), ("lon", [lon_c]), ("u_wind", [25.0]), ("v_wind", [5.0]), ("time", [t_epoch])):
        wx1.write_array(obs_root / "satellite" / f"eumetsat_{k}.wx1", np.array(v, dtype=np.float64 if k == "time" else np.float32), dataset_id=f"amv_{k}", variable_path=k, dimensions=["n_obs"], codec="zstd_shuffle")
    wx1.write_array(obs_root / "satellite" / "satellite_atms_microwave_radiances.wx1", np.array([245.0], dtype=np.float32), dataset_id="atms_raw", variable_path="tb", dimensions=["n_obs"], codec="zstd_shuffle")

    # 5. Marine buoy SST & OSI-SAF sea-ice
    wx1.write_array(obs_root / "marine" / "marine_buoy_obs.wx1", np.array([285.2], dtype=np.float32), dataset_id="buoy", variable_path="sst", dimensions=["station_id"], codec="zstd_shuffle")
    for k, v in (("lat", [lat_sea]), ("lon", [lon_sea]), ("pressure", [101200.0]), ("temp", [284.8]), ("u_wind", [8.0]), ("v_wind", [1.0]), ("time", [t_epoch])):
        wx1.write_array(obs_root / "marine" / f"marine_buoy_obs_{k}.wx1", np.array(v, dtype=np.float64 if k == "time" else np.float32), dataset_id=f"buoy_{k}", variable_path=k, dimensions=["station_id"], codec="zstd_shuffle")

    lat_ice = float(spec.lat2d[3, 3])
    lon_ice = float(spec.lon2d[3, 3])
    wx1.write_array(obs_root / "marine" / "osi_saf_sea_ice.wx1", np.array([0.85], dtype=np.float32), dataset_id="osi", variable_path="ice_conc", dimensions=["n_points"], codec="zstd_shuffle")
    for k, v in (("lat", [lat_ice]), ("lon", [lon_ice]), ("ice_thick", [0.6])):
        wx1.write_array(obs_root / "marine" / f"osi_saf_sea_ice_{k}.wx1", np.array(v, dtype=np.float32), dataset_id=f"osi_{k}", variable_path=k, dimensions=["n_points"], codec="zstd_shuffle")

    # 6. Radar LHN container
    radar_rr = np.zeros((16, 16), dtype=np.float32)
    radar_rr[7:10, 7:10] = 6.0  # 6 mm/h radar precipitation cell
    wx1.write_array(obs_root / "radar_obs.wx1", radar_rr, dataset_id="radar", variable_path="rain_rate_mm_h", dimensions=["y", "x"], codec="zstd_shuffle")

    obs_list, _ = load_station_obs_wx1(obs_root, spec, valid_time_utc=init_dt)
    var_types = {o.var_type for o in obs_list}
    assert {"t2m", "ps", "u_geo", "v_geo", "theta_p", "qv", "ztd", "sst"}.issubset(var_types)
    assert any(abs(o.time_offset_sec - 1800.0) < 1.0 for o in obs_list)
    assert all("atms" not in o.station_id for o in obs_list)

    st = {
        "u": np.full((12, 16, 17), 5.0, dtype=np.float32),
        "v": np.full((12, 17, 16), 1.0, dtype=np.float32),
        "w": np.zeros((13, 16, 16), dtype=np.float32),
        "theta_p": np.zeros((12, 16, 16), dtype=np.float32),
        "pi_p": np.zeros((12, 16, 16), dtype=np.float32),
        "qv": np.full((12, 16, 16), 0.004, dtype=np.float32),
        "qc": np.zeros((12, 16, 16), dtype=np.float32),
        "qr": np.zeros((12, 16, 16), dtype=np.float32),
        "qi": np.zeros((12, 16, 16), dtype=np.float32),
        "qs": np.zeros((12, 16, 16), dtype=np.float32),
        "qg": np.zeros((12, 16, 16), dtype=np.float32),
    }
    sf = {
        "t_skin": np.full((16, 16), 287.0, dtype=np.float32),
        "t2m": np.full((16, 16), 287.0, dtype=np.float32),
        "sst_bulk": np.full((16, 16), 284.0, dtype=np.float32),
    }

    # Build 2-snapshot 4D-FGAT background trajectory at t=-900s and t=+1800s
    u_c = np.full((12, 16, 16), 5.0, dtype=np.float32)
    v_c = np.full((12, 16, 16), 1.0, dtype=np.float32)
    bg_snap = {
        "u": u_c,
        "v": v_c,
        "theta_p": st["theta_p"].copy(),
        "pi_p": st["pi_p"].copy(),
        "qv": st["qv"].copy(),
        "ps": np.full((16, 16), 101325.0, dtype=np.float32),
        "sst": sf["sst_bulk"].copy(),
    }
    traj = {-900.0: bg_snap, 0.0: bg_snap, 1800.0: bg_snap}

    ana_st, ana_sf, iau_f, vres = run_domain_3dvar(
        spec, st, sf, hs, lsm, obs_list, trajectory=traj, obs_dir=obs_root
    )
    assert vres.final_cost_j < vres.initial_cost_j0
    # OSI-SAF sea-ice cell at (3, 3) clamped to freezing SST 271.35 K
    assert abs(float(ana_sf["sst_bulk"][3, 3]) - 271.35) < 1e-3
    # Radar LHN positive latent heating & moisture scaling applied in radar cell [7:10, 7:10]
    assert float(np.max(iau_f["theta_p"][:, 7:10, 7:10])) > float(np.max(iau_f["theta_p"][:, 0:3, 0:3]))
    assert float(np.mean(ana_st["qv"][1:3, 7:10, 7:10])) > float(np.mean(ana_st["qv"][1:3, 0:3, 0:3]))
