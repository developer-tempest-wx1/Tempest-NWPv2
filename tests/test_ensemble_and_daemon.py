"""Comprehensive GPU & unit test suite for Phase B4 (0% AI / 0% ML):
1. Isolated per-cycle run directory layout & native v2 WX-1 restart validator.
2. Antithetic balanced + SPP + stochastic tendency perturbations (zero mean, deterministic).
3. Full 51-member GPU ensemble (member 0 = Control Forecast `cf`, `pf001..pf050`) -> WX-1 stats + CRPS + Brier.
4. Automated continuous operational cycling daemon (warm-start discovery, flock lockfile, `latest`, pruning).
5. KNMI NetCDF real coordinates + TITAN spatial buddy QC + DWD RADOLAN RW radar decoder/reprojector.
"""

from __future__ import annotations

import datetime
import struct
from pathlib import Path
from typing import Any, Dict

import netCDF4 as nc
import numpy as np
import pytest
import wx1

from tempest_v2.cuda_bridge import NonHydrostaticGPUSolver
from tempest_v2.daemon import (
    DaemonConfig,
    acquire_daemon_lock,
    latest_ready_cycle_time_utc,
    next_cycle_time_utc,
    prune_old_runs,
    run_daemon_loop,
)
from tempest_v2.ensemble import (
    EnsembleConfig,
    generate_balanced_member_perturbation,
    run_ensemble_forecast,
)
from tempest_v2.ic_builder import save_tempest_restart_wx1
from tempest_v2.ingest_knmi_observations import decode_knmi_netcdf_to_wx1
from tempest_v2.ingest_radar_qpe import DWD_RADOLAN_PROJ4, _reproject_radar_to_domain, parse_dwd_radolan_rw_bytes
from tempest_v2.projection import DomainSpec, LCCProjection, VerticalGrid
from tempest_v2.restart_validator import find_latest_valid_cycle_restart, validate_v2_restart
from tempest_v2.runner import create_run_layout, run_domain_forecast
from tempest_v2.verification import (
    compute_brier_score,
    compute_fair_crps,
    load_station_obs_wx1,
    verify_ensemble_against_stations,
)


def _make_small_domain(
    name: str = "d3_test_small",
    ny: int = 32,
    nx: int = 32,
    nz: int = 16,
    dx_m: float = 15_000.0,
) -> tuple[DomainSpec, np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    proj = LCCProjection(lat1_deg=48.0, lat2_deg=54.0, lat0_deg=52.0, lon0_deg=5.2)
    cx, cy = proj.latlon_to_xy(52.0, 5.2)
    x_m = float(cx) + (np.arange(nx, dtype=np.float64) - 0.5 * (nx - 1)) * dx_m
    y_m = float(cy) + (np.arange(ny, dtype=np.float64) - 0.5 * (ny - 1)) * dx_m
    xx, yy = np.meshgrid(x_m, y_m)
    lat2d, lon2d = proj.xy_to_latlon(xx, yy)
    vgrid = VerticalGrid(nz=nz, z_top_m=14_000.0, stretch_power=1.3)
    spec = DomainSpec(
        name=name,
        ny=ny,
        nx=nx,
        dx_m=dx_m,
        dy_m=dx_m,
        is_projected=True,
        is_periodic_lon=False,
        projection=proj,
        x_m=x_m,
        y_m=y_m,
        lat2d=lat2d,
        lon2d=lon2d,
        map_factor=proj.map_scale_factor(lat2d),
        coriolis_f=proj.coriolis_f(lat2d),
        vertical=vgrid,
    )
    hs = 120.0 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * (4.0 * dx_m) ** 2))
    sstatic = {
        "lat_deg": spec.lat2d.astype(np.float32),
        "lon_deg": spec.lon2d.astype(np.float32),
        "lsm": np.where(np.arange(nx)[None, :] > 4, 1.0, 0.0).astype(np.float32),
        "lake_mask": np.zeros((ny, nx), dtype=np.float32),
        "soil_type": np.full((ny, nx), 9.0, dtype=np.float32),
        "bathy_depth_m": np.where(np.arange(nx)[None, :] > 4, 0.0, 35.0).astype(np.float32),
        "z0m_base": np.full((ny, nx), 0.08, dtype=np.float32),
        "albedo_base": np.full((ny, nx), 0.20, dtype=np.float32),
        "veg_frac": np.full((ny, nx), 0.65, dtype=np.float32),
    }

    state_3d = {
        "u": np.full((nz, ny, nx + 1), 6.0, dtype=np.float32),
        "v": np.full((nz, ny + 1, nx), 2.5, dtype=np.float32),
        "w": np.zeros((nz + 1, ny, nx), dtype=np.float32),
        "theta_p": np.zeros((nz, ny, nx), dtype=np.float32),
        "pi_p": np.zeros((nz, ny, nx), dtype=np.float32),
        "qv": np.full((nz, ny, nx), 0.0065, dtype=np.float32),
        "qc": np.full((nz, ny, nx), 5.0e-5, dtype=np.float32),
        "qr": np.zeros((nz, ny, nx), dtype=np.float32),
        "qi": np.zeros((nz, ny, nx), dtype=np.float32),
        "qs": np.zeros((nz, ny, nx), dtype=np.float32),
        "qg": np.zeros((nz, ny, nx), dtype=np.float32),
    }
    surf_state = {
        "t_skin": np.full((ny, nx), 283.0, dtype=np.float32),
        "soil_t": np.full((4, ny, nx), 284.0, dtype=np.float32),
        "soil_m": np.full((4, ny, nx), 0.28, dtype=np.float32),
        "snow_swe": np.zeros((ny, nx), dtype=np.float32),
        "sst_bulk": np.full((ny, nx), 285.0, dtype=np.float32),
        "dt_cool": np.zeros((ny, nx), dtype=np.float32),
        "dt_warm": np.zeros((ny, nx), dtype=np.float32),
        "lake_tbot": np.full((ny, nx), 281.0, dtype=np.float32),
        "lake_hmld": np.full((ny, nx), 2.5, dtype=np.float32),
        "lake_ctheta": np.full((ny, nx), 0.65, dtype=np.float32),
        "lake_hice": np.zeros((ny, nx), dtype=np.float32),
        "rain_acc": np.zeros((ny, nx), dtype=np.float32),
        "snow_acc": np.zeros((ny, nx), dtype=np.float32),
        "graupel_acc": np.zeros((ny, nx), dtype=np.float32),
        "sw_down": np.zeros((ny, nx), dtype=np.float32),
        "lw_down": np.full((ny, nx), 310.0, dtype=np.float32),
        "shf": np.zeros((ny, nx), dtype=np.float32),
        "lhf": np.zeros((ny, nx), dtype=np.float32),
        "ustar": np.full((ny, nx), 0.25, dtype=np.float32),
        "pblh": np.full((ny, nx), 450.0, dtype=np.float32),
        "t2m": np.full((ny, nx), 283.5, dtype=np.float32),
        "q2m": np.full((ny, nx), 0.0065, dtype=np.float32),
        "u10m": np.full((ny, nx), 6.0, dtype=np.float32),
        "v10m": np.full((ny, nx), 2.5, dtype=np.float32),
    }
    return spec, hs, sstatic, state_3d, surf_state


def test_per_cycle_run_layout_and_v2_restart_validator(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    init_00 = datetime.datetime(2026, 11, 21, 0, 0, tzinfo=datetime.timezone.utc)
    init_06 = datetime.datetime(2026, 11, 21, 6, 0, tzinfo=datetime.timezone.utc)

    layout0 = create_run_layout(runs_root, init_00)
    layout1 = create_run_layout(runs_root, init_06)
    assert layout0["run_dir"].name == "20261121_00Z"
    assert layout1["run_dir"].name == "20261121_06Z"
    for key in ("config", "obs", "da", "d2", "d3", "restarts", "ensemble", "verification", "logs"):
        assert layout0[key].is_dir()

    spec, hs, _ss, st, sf = _make_small_domain(name="d3_benelux_de_1p5km", ny=16, nx=16, nz=12)
    rst_good = save_tempest_restart_wx1(
        layout0["restarts"] / "d3_cycle_f006_20261121_0600Z.wx1",
        init_06,
        spec,
        hs,
        st,
        sf,
        sim_time_sec=21_600.0,
    )
    ok, msg = validate_v2_restart(
        rst_good,
        expected_domain_name="d3_benelux_de_1p5km",
        expected_shape=(12, 16, 16),
        max_age_hours=6.0,
        reference_utc=init_06,
    )
    assert ok, msg

    # Wrong domain, wrong shape, expired age must fail cleanly
    ok_dom, _ = validate_v2_restart(rst_good, expected_domain_name="d2_eu_lcc")
    assert not ok_dom
    ok_shp, _ = validate_v2_restart(rst_good, expected_shape=(12, 32, 32))
    assert not ok_shp
    ok_age, _ = validate_v2_restart(
        rst_good,
        max_age_hours=2.0,
        reference_utc=init_06 + datetime.timedelta(hours=5),
    )
    assert not ok_age

    # NaN corruption and negative moisture must fail validation
    bad_st = {k: v.copy() for k, v in st.items()}
    bad_st["theta_p"][0, 0, 0] = np.nan
    rst_nan = save_tempest_restart_wx1(layout1["restarts"] / "bad_nan.wx1", init_06, spec, hs, bad_st, sf)
    ok_nan, _ = validate_v2_restart(rst_nan)
    assert not ok_nan

    neg_st = {k: v.copy() for k, v in st.items()}
    neg_st["qv"][0, 1, 1] = -0.05
    rst_neg = save_tempest_restart_wx1(layout1["restarts"] / "bad_neg.wx1", init_06, spec, hs, neg_st, sf)
    ok_neg, _ = validate_v2_restart(rst_neg)
    assert not ok_neg

    # Discovery must pick the sole valid restart targeting init_06 even alongside corrupted ones
    found = find_latest_valid_cycle_restart(
        runs_root,
        "d3_benelux_de_1p5km",
        init_06,
        expected_shape=(12, 16, 16),
        exact_valid_match=True,
    )
    assert found == rst_good


def test_antithetic_balanced_perturbations_zero_mean_and_deterministic() -> None:
    spec, hs, ss, st, sf = _make_small_domain(ny=24, nx=24, nz=14)
    cfg = EnsembleConfig(n_members=5, seed=424242, antithetic=True)

    p0 = generate_balanced_member_perturbation(0, spec, hs, st, sf, ss, cfg)
    assert p0["is_control"] is True
    assert p0["member_name"] == "cf"
    assert p0["stoch_iau"] is None
    for k in st:
        assert np.array_equal(p0["state_3d"][k], st[k])
    for k in sf:
        assert np.array_equal(p0["surf_state"][k], sf[k])

    p1 = generate_balanced_member_perturbation(1, spec, hs, st, sf, ss, cfg)
    p2 = generate_balanced_member_perturbation(2, spec, hs, st, sf, ss, cfg)
    p1_repeat = generate_balanced_member_perturbation(1, spec, hs, st, sf, ss, cfg)

    assert p1["member_name"] == "pf001" and p2["member_name"] == "pf002"
    assert p1["pair_id"] == p2["pair_id"] == 0
    assert p1["sign"] == 1.0 and p2["sign"] == -1.0

    # Strict seed determinism
    for k in ("u", "v", "theta_p", "pi_p", "qv"):
        assert np.array_equal(p1["state_3d"][k], p1_repeat["state_3d"][k])

    # Non-trivial physically balanced perturbations (hydrostatic + geostrophic coupling active)
    assert float(np.std(p1["delta_raw"]["pi_p"])) > 1e-6
    assert float(np.std(p1["delta_raw"]["u"])) > 0.05

    # Antithetic symmetry: pf001 + pf002 raw perturbations cancel to machine precision
    for k in ("u", "v", "theta_p", "pi_p", "soil_m", "sst_bulk", "stoch_theta_p"):
        max_sum = float(np.max(np.abs(p1["delta_raw"][k] + p2["delta_raw"][k])))
        assert max_sum < 1e-6, f"Antithetic cancellation failed for {k}: {max_sum}"


def test_ensemble_51_members_gpu_cf_and_wx1_stats_and_crps(tmp_path: Path) -> None:
    spec, hs, ss, st, sf = _make_small_domain(ny=32, nx=32, nz=16, dx_m=15_000.0)
    init_dt = datetime.datetime(2026, 11, 21, 12, 0, tzinfo=datetime.timezone.utc)
    ens_dir = tmp_path / "ensemble"

    cfg = EnsembleConfig(n_members=51, seed=20261121, antithetic=True, save_member_forecasts=True)
    summary = run_ensemble_forecast(
        spec,
        hs,
        ss,
        st,
        sf,
        init_dt,
        forecast_hours=0.25,
        ens_dir=ens_dir,
        cfg=cfg,
        dt_s=15.0,
        physics_every=2,
        output_hours=(0.25,),
        enable_physics=True,
        rad_interval_s=900.0,
    )

    assert summary["n_members"] == 51
    assert summary["control_member"] == "cf"
    assert (ens_dir / "cf" / f"forecast_{spec.name}_T+000.25h.wx1").is_file()
    assert (ens_dir / "pf001" / f"forecast_{spec.name}_T+000.25h.wx1").is_file()
    assert (ens_dir / "pf050" / f"forecast_{spec.name}_T+000.25h.wx1").is_file()

    stats_path = Path(summary["stats_paths"][0])
    assert stats_path.is_file()
    with wx1.open_dataset(ens_dir / "cf" / f"forecast_{spec.name}_T+000.25h.wx1") as dcf:
        t2m_cf_raw = np.asarray(dcf["t2m"].values, dtype=np.float32)

    with wx1.open_dataset(stats_path) as dstats:
        assert int(dstats.attrs["n_members"]) == 51
        assert dstats.attrs["control_member"] == "cf"
        t2m_cf = np.asarray(dstats["t2m_cf"].values, dtype=np.float32)
        t2m_spread = np.asarray(dstats["t2m_spread"].values, dtype=np.float32)
        t2m_p10 = np.asarray(dstats["t2m_p10"].values, dtype=np.float32)
        t2m_p50 = np.asarray(dstats["t2m_p50"].values, dtype=np.float32)
        t2m_p90 = np.asarray(dstats["t2m_p90"].values, dtype=np.float32)
        prob_frost = np.asarray(dstats["prob_frost_t2m_lt_0c"].values, dtype=np.float32)
        prob_p05 = np.asarray(dstats["prob_precip_ge_0p5mm"].values, dtype=np.float32)
        prob_w10 = np.asarray(dstats["prob_wind10m_ge_10ms"].values, dtype=np.float32)

        assert np.array_equal(t2m_cf, t2m_cf_raw)
        assert float(np.mean(t2m_spread)) > 1e-3
        assert np.all(t2m_p10 <= t2m_p50 + 1e-5)
        assert np.all(t2m_p50 <= t2m_p90 + 1e-5)
        assert np.all((prob_frost >= 0.0) & (prob_frost <= 1.0))
        assert np.all((prob_p05 >= 0.0) & (prob_p05 <= 1.0))
        assert np.all((prob_w10 >= 0.0) & (prob_w10 <= 1.0))

    # Verify fair CRPS, Brier score & verify_ensemble_against_stations
    assert compute_fair_crps(np.full((10, 5), 280.0), np.full(5, 282.0)) == pytest.approx(2.0)
    assert compute_brier_score(np.array([0.8, 0.2]), np.array([1.0, 0.0])) == pytest.approx(0.04)

    station_rows = []
    for jj, ii in ((8, 8), (12, 16), (20, 12), (24, 24)):
        station_rows.append(
            {
                "x_m": float(spec.x_m[ii]),
                "y_m": float(spec.y_m[jj]),
                "elev_m": float(hs[jj, ii]),
                "provider": "knmi",
                "t2m_k": float(t2m_cf_raw[jj, ii]) + 0.25,
                "mslp_pa": 101_325.0,
                "u10_ms": 6.0,
                "v10_ms": 2.5,
            }
        )
    v_ens = verify_ensemble_against_stations(ens_dir, 0.25, spec, station_rows, hs_m=hs)
    assert v_ens["n_members"] == 51
    assert v_ens["n_stations"] == 4
    assert np.isfinite(v_ens["t2m_crps_k"]) and v_ens["t2m_crps_k"] >= 0.0
    assert np.isfinite(v_ens["t2m_spread_k"]) and v_ens["t2m_spread_k"] > 0.0
    assert np.isfinite(v_ens["t2m_spread_skill_ratio"]) and v_ens["t2m_spread_skill_ratio"] > 0.0
    assert 0.0 <= v_ens["brier_frost"] <= 1.0


def test_daemon_warm_start_cycling_and_retention(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"

    # 1. Schedule helper sanity check
    now = datetime.datetime(2026, 11, 21, 6, 40, tzinfo=datetime.timezone.utc)
    ready_dt, ready_at = latest_ready_cycle_time_utc(now, (0, 6, 12, 18), 35)
    nxt_dt, nxt_at = next_cycle_time_utc(now, (0, 6, 12, 18), 35)
    assert ready_dt == datetime.datetime(2026, 11, 21, 6, 0, tzinfo=datetime.timezone.utc)
    assert ready_at == datetime.datetime(2026, 11, 21, 6, 35, tzinfo=datetime.timezone.utc)
    assert nxt_dt == datetime.datetime(2026, 11, 21, 12, 0, tzinfo=datetime.timezone.utc)
    assert nxt_at == datetime.datetime(2026, 11, 21, 12, 35, tzinfo=datetime.timezone.utc)

    # 2. Advisory lockfile must block concurrent daemon instances cleanly
    lock_file = runs_root / ".daemon.lock"
    with acquire_daemon_lock(lock_file):
        with pytest.raises(RuntimeError, match="already holds lock"):
            with acquire_daemon_lock(lock_file):
                pass

    # Create 2 legacy old run folders to verify pruning preserves newest keep_runs
    for old_name in ("20261119_00Z", "20261119_06Z"):
        d = runs_root / old_name
        d.mkdir(parents=True)
        (d / "dummy.txt").write_text("old")

    d2_spec, hs_d2, ss_d2, st_d2, sf_d2 = _make_small_domain(name="d2_eu_lcc", ny=20, nx=20, nz=12, dx_m=20_000.0)
    d3_spec, hs_d3, ss_d3, st_d3, sf_d3 = _make_small_domain(
        name="d3_benelux_de_1p5km", ny=20, nx=20, nz=12, dx_m=20_000.0
    )

    def _gpu_test_pipeline(**kwargs: Any) -> Dict[str, Any]:
        import json

        run_dir = Path(kwargs["run_dir"])
        restarts_dir = run_dir / "restarts"
        d3_dir = run_dir / "d3"
        ic_source = kwargs["ic_source"]
        restart_path = kwargs.get("restart_path")
        f_hours = float(kwargs["forecast_hours"])

        if ic_source == "restart" and restart_path is not None:
            st_in, sf_in, meta = __import__("tempest_v2.ic_builder", fromlist=["load_tempest_restart_wx1"]).load_tempest_restart_wx1(restart_path)
            init_dt = datetime.datetime.fromisoformat(meta["valid_time_utc"].replace("Z", "+00:00"))
        else:
            st_in, sf_in = st_d3, sf_d3
            init_dt = datetime.datetime(2026, 11, 21, 0, 0, tzinfo=datetime.timezone.utc)

        res_d3 = run_domain_forecast(
            d3_spec,
            hs_d3,
            ss_d3,
            st_in,
            sf_in,
            init_dt,
            forecast_hours=f_hours,
            dt_s=kwargs["dt_d3_s"],
            physics_every=2,
            output_dir=d3_dir,
            output_hours=[f_hours],
            save_restart_path=restarts_dir / f"restart_{d3_spec.name}.wx1",
            save_cycle_restart_hours=kwargs["cycle_restart_hours"],
            cycle_restart_dir=restarts_dir,
            enable_physics=False,
        )
        # Also save a valid D2 cycle restart when cold/warm
        cyc_valid = init_dt + datetime.timedelta(hours=float(kwargs["cycle_restart_hours"][0]))
        save_tempest_restart_wx1(
            restarts_dir / f"{d2_spec.name}_cycle_f001_{cyc_valid:%Y%m%d_%H%M}Z.wx1",
            cyc_valid,
            d2_spec,
            hs_d2,
            st_d2,
            sf_d2,
        )
        manifest = {"init_time_utc": init_dt.isoformat(), "warm_start": ic_source == "restart"}
        (run_dir / "manifest.json").write_text(json.dumps(manifest))
        return {"d3": res_d3, "da": {}}

    cfg = DaemonConfig(
        runs_root=runs_root,
        cycle_hours_utc=(0, 1, 2),
        keep_runs=2,
        d2_lead_hours=1.0,
        d3_lead_hours=1.0,
        nz=12,
        dt_d2_s=20.0,
        dt_d3_s=20.0,
        cycle_restart_hours=(1.0,),
        enable_da=False,
        enable_radar=False,
        enable_notifier=False,
    )
    t0 = datetime.datetime(2026, 11, 21, 0, 0, tzinfo=datetime.timezone.utc)
    t1 = datetime.datetime(2026, 11, 21, 1, 0, tzinfo=datetime.timezone.utc)
    results = run_daemon_loop(
        cfg,
        max_cycles=2,
        explicit_cycle_times=[t0, t1],
        d2_spec=d2_spec,
        d3_spec=d3_spec,
        pipeline_fn=_gpu_test_pipeline,
        ic_fn=lambda _cdir, _dt: (Path("unused_sfc.grib2"), Path("unused_pl.grib2")),
    )

    assert len(results) == 2
    assert results[0]["ic_source"] == "ecmwf" and results[0]["warm_started"] is False
    assert results[1]["ic_source"] == "restart" and results[1]["warm_started"] is True
    assert results[1]["restart_path"] is not None
    assert "20261121_00Z" in results[1]["restart_path"]
    assert (runs_root / "latest").is_symlink()
    assert (runs_root / "latest").resolve() == (runs_root / "20261121_01Z").resolve()

    # Old legacy folders pruned cleanly while keeping both recent 20261121_* cycles
    assert not (runs_root / "20261119_00Z").exists()
    assert not (runs_root / "20261119_06Z").exists()
    assert (runs_root / "20261121_00Z").is_dir()
    assert (runs_root / "20261121_01Z").is_dir()

    # Explicit standalone prune test
    (runs_root / "20261120_18Z").mkdir()
    pruned = prune_old_runs(runs_root, keep_runs=2)
    assert any(p.name == "20261120_18Z" for p in pruned)


def test_knmi_netcdf_real_coords_and_radar_qpe_decoder(tmp_path: Path) -> None:
    # 1. Synthetic KNMI 10-min NetCDF using official variable names & 1950 epoch
    nc_path = tmp_path / "knmi_sample.nc"
    obs_dir = tmp_path / "obs"
    obs_dir.mkdir()
    valid_dt = datetime.datetime(2026, 11, 21, 18, 0, tzinfo=datetime.timezone.utc)
    epoch_1950 = datetime.datetime(1950, 1, 1, tzinfo=datetime.timezone.utc)

    with nc.Dataset(nc_path, "w") as ds:
        ds.createDimension("station", 5)
        ds.createDimension("time", 1)
        t_var = ds.createVariable("time", "f8", ("time",))
        t_var[:] = [(valid_dt - epoch_1950).total_seconds()]

        st_var = ds.createVariable("station", str, ("station",))
        st_var[:] = np.array(["06260", "06348", "06240", "06275", "78900"], dtype=object)

        # De Bilt, Cabauw, Schiphol, Deelen, plus unknown BES station (78900) without coords
        ds.createVariable("lat", "f4", ("station",))[:] = [52.101, 51.970, 52.316, 52.056, np.nan]
        ds.createVariable("lon", "f4", ("station",))[:] = [5.176, 4.926, 4.790, 5.873, np.nan]
        ds.createVariable("height", "f4", ("station",))[:] = [1.9, -0.7, -3.3, 48.2, np.nan]
        # Inject warm spike outlier (+18C vs ~6C neighbors) on Deelen 06275 to trigger TITAN QC
        ds.createVariable("ta", "f4", ("station",))[:] = [279.15, 279.05, 279.35, 297.15, 299.15]
        ds.createVariable("pp", "f4", ("station",))[:] = [1012.0, 1012.1, 1011.9, 1012.2, 1012.0]
        ds.createVariable("ff", "f4", ("station",))[:] = [5.0, 5.5, 6.0, 4.5, 4.0]
        ds.createVariable("dd", "f4", ("station",))[:] = [230.0, 225.0, 235.0, 220.0, 90.0]

    count = decode_knmi_netcdf_to_wx1(nc_path, obs_dir)
    assert count == 4  # unknown BES 78900 skipped cleanly

    with wx1.open_dataset(obs_dir / "knmi_insitu_obs_lat.wx1") as ds_lat:
        lats = np.asarray(ds_lat["lat"].values, dtype=np.float64)
        assert np.isclose(lats[0], 52.101, atol=1e-3)

    # Load into D3-like domain covering NL and confirm TITAN QC removes the +18K outlier on Deelen
    spec, _hs, _ss, _st, _sf = _make_small_domain(ny=32, nx=32, nz=12, dx_m=15_000.0)
    obs_no_qc, _ = load_station_obs_wx1(obs_dir, spec, valid_time_utc=valid_dt, enable_titan_qc=False)
    obs_qc, _ = load_station_obs_wx1(obs_dir, spec, valid_time_utc=valid_dt, enable_titan_qc=True)
    t_no_qc = [o.value for o in obs_no_qc if o.var_type == "t2m"]
    t_with_qc = [o.value for o in obs_qc if o.var_type == "t2m"]
    assert len(t_no_qc) == 4
    assert max(t_no_qc) > 296.0
    assert len(t_with_qc) == 3
    assert max(t_with_qc) < 282.0

    # 2. Synthetic DWD RADOLAN RW binary product (900x900 @ 1 km polar stereographic)
    hdr = b"RW21185010000BY 1620100VS 3GP 900x 900PR E-01INT  60U0\x03"
    arr = np.zeros((900, 900), dtype="<u2")
    arr[500:540, 250:290] = 65  # 6.5 mm/h with PR E-01 & INT 60
    arr[0:5, 0:5] = 0x2000  # missing data flag bit 14
    raw_bytes = hdr + arr.tobytes()

    rain_src, meta = parse_dwd_radolan_rw_bytes(raw_bytes)
    assert rain_src.shape == (900, 900)
    assert np.isnan(rain_src[0, 0])
    assert rain_src[520, 270] == pytest.approx(6.5)

    reproj = _reproject_radar_to_domain(
        rain_src,
        DWD_RADOLAN_PROJ4,
        -523.4622,
        -4658.6447,
        1.0,
        1.0,
        spec,
        y_ascending=True,
        geo_crs="+proj=longlat +a=6370040 +b=6370040 +no_defs",
    )
    assert reproj.shape == (spec.ny, spec.nx)
    assert np.nanmax(reproj) >= 0.0


def test_daemon_config_passthrough_and_nz137_stratospheric_restart(tmp_path: Path) -> None:
    """Verify daemon forwards ecmwf/pbl_last/cy46_physics/d1_domain flags and validate_v2_restart permits nz=137 stratospheric theta_p."""
    from tempest_v2.daemon import run_single_cycle

    init_00 = datetime.datetime(2026, 11, 22, 0, 0, tzinfo=datetime.timezone.utc)
    spec, hs, _ss, st, sf = _make_small_domain(name="d1_global_0p4", ny=12, nx=12, nz=137)
    st_strat = {k: v.copy() for k, v in st.items()}
    st_strat["theta_p"][-1, :, :] = 580.0  # Real 10 hPa stratospheric departure vs bar (~395 K) at z_top=31 km
    rst_ok = save_tempest_restart_wx1(tmp_path / "d1_strat_ok.wx1", init_00, spec, hs, st_strat, sf)
    ok_137, msg_137 = validate_v2_restart(rst_ok, expected_domain_name="d1_global_0p4", expected_shape=(137, 12, 12))
    assert ok_137, msg_137

    st_bad = {k: v.copy() for k, v in st.items()}
    st_bad["theta_p"][-1, 0, 0] = 1200.0
    rst_bad = save_tempest_restart_wx1(tmp_path / "d1_strat_bad.wx1", init_00, spec, hs, st_bad, sf)
    ok_bad, _ = validate_v2_restart(rst_bad, expected_domain_name="d1_global_0p4", expected_shape=(137, 12, 12))
    assert not ok_bad

    captured: Dict[str, Any] = {}

    def _capture_pipeline(**kwargs: Any) -> Dict[str, Any]:
        captured.update(kwargs)
        return {"da": {}}

    cfg = DaemonConfig(
        runs_root=tmp_path / "runs",
        enable_d1=True,
        d1_domain="d1_global_0p4",
        ecmwf_schedule=True,
        pbl_last=True,
        cy46_physics=True,
        enable_da=False,
        enable_radar=False,
        enable_notifier=False,
    )
    run_single_cycle(
        cfg,
        init_00,
        pipeline_fn=_capture_pipeline,
        ic_fn=lambda _cdir, _dt: (tmp_path / "sfc.grib2", tmp_path / "pl.grib2"),
    )
    assert captured["ecmwf_schedule"] is True
    assert captured["pbl_last"] is True
    assert captured["cy46_physics"] is True


def test_daemon_cold_start_ecmwf_kwargs_and_latest_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Daemon passes run_date/run_time (never target_time) and falls back to latest step-0 when cycle isn't live yet."""
    import tempest_v2.ingest_ecmwf as ingest_ecmwf
    from tempest_v2.daemon import run_single_cycle

    calls: list[dict[str, Any]] = []

    def _fake_download(**kwargs: Any) -> tuple[Path, Path, str]:
        calls.append(dict(kwargs))
        assert "target_time" not in kwargs
        if kwargs.get("run_date") is not None:
            raise RuntimeError("404 Not Found")
        out_d = Path(kwargs["output_dir"])
        out_d.mkdir(parents=True, exist_ok=True)
        sfc = out_d / "sfc_latest.grib2"
        pl = out_d / "pl_latest.grib2"
        sfc.write_bytes(b"grib")
        pl.write_bytes(b"grib")
        return sfc, pl, "0p25"

    monkeypatch.setattr(ingest_ecmwf, "download_ecmwf_cold_start", _fake_download)

    captured: Dict[str, Any] = {}

    def _capture_pipeline(**kwargs: Any) -> Dict[str, Any]:
        captured.update(kwargs)
        return {"da": {}}

    cfg = DaemonConfig(
        runs_root=tmp_path / "runs",
        enable_da=False,
        enable_radar=False,
        enable_notifier=False,
    )
    init_18 = datetime.datetime(2026, 11, 22, 18, 0, tzinfo=datetime.timezone.utc)
    res = run_single_cycle(cfg, init_18, pipeline_fn=_capture_pipeline)

    assert len(calls) == 2
    assert calls[0]["run_date"] == "2026-11-22" and calls[0]["run_time"] == 18
    assert calls[1].get("run_date") is None and calls[1].get("run_time") is None
    assert Path(captured["sfc_grib"]).name == "sfc_latest.grib2"
    assert Path(captured["pl_grib"]).name == "pl_latest.grib2"
    assert res["ic_source"] == "ecmwf"


