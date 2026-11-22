"""Tests for KNMI-style ECMWF LBC LAM mode, NW Benelux-NorthSea 2.5km, and Europe 5km domains."""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.interpolate import RegularGridInterpolator
import wx1

from tempest_v2.daemon import DaemonConfig, run_single_cycle
from tempest_v2.ecmwf_lbc import (
    ecmwf_grib_to_lbc_wx1,
    plan_ecmwf_lbc_steps,
    prepare_ecmwf_lbc_series,
    resolve_ecmwf_lbc_plan,
)
from tempest_v2.ic_builder import _regrid_latlon_field, save_tempest_restart_wx1
from tempest_v2.lbc_manager import LBCInterpolator, export_child_lbc_step
from tempest_v2.projection import (
    LAM_DOMAIN_DEFAULT_DT_S,
    LCCProjection,
    VerticalGrid,
    _make_lcc_domain,
    get_domain_spec,
)
from tempest_v2.restart_validator import (
    DOMAIN_GEOMETRIES,
    normalize_domain_name,
    validate_v2_restart,
)
from tempest_v2.runner import run_ecmwf_lbc_lam_pipeline


def _utc(y: int, m: int, d: int, h: int, minute: int = 0) -> datetime.datetime:
    return datetime.datetime(y, m, d, h, minute, tzinfo=datetime.timezone.utc)


def test_new_lam_domain_specs_and_bboxes() -> None:
    bnl = get_domain_spec("lam_bnl_ns_de_2p5km", nz=65)
    assert bnl.name == "lam_bnl_ns_de_2p5km"
    assert (bnl.nz, bnl.ny, bnl.nx) == (65, 432, 432)
    assert bnl.dx_m == pytest.approx(2500.0)
    assert float(bnl.lat2d.min()) == pytest.approx(47.71, abs=0.25)
    assert float(bnl.lat2d.max()) > 57.4
    assert float(bnl.lon2d.min()) < -4.2
    assert float(bnl.lon2d.max()) > 13.2
    assert get_domain_spec("lam_bnl", nz=65).name == "lam_bnl_ns_de_2p5km"
    assert get_domain_spec("d3_nw", nz=137).name == "lam_bnl_ns_de_2p5km"

    eu = get_domain_spec("lam_eu_5km", nz=65)
    assert eu.name == "lam_eu_5km"
    assert (eu.nz, eu.ny, eu.nx) == (65, 480, 560)
    assert eu.dx_m == pytest.approx(5000.0)
    assert float(eu.lat2d.min()) < 40.0
    assert float(eu.lat2d.max()) > 62.0
    assert float(eu.lon2d.min()) < -19.0
    assert float(eu.lon2d.max()) > 31.0
    assert get_domain_spec("lam_eu", nz=65).name == "lam_eu_5km"
    assert get_domain_spec("europe_5km", nz=65).name == "lam_eu_5km"
    assert LAM_DOMAIN_DEFAULT_DT_S["lam_bnl_ns_de_2p5km"] == pytest.approx(6.0)
    assert LAM_DOMAIN_DEFAULT_DT_S["lam_eu_5km"] == pytest.approx(15.0)


def test_restart_validator_distinguishes_lam_domains() -> None:
    assert normalize_domain_name("lam_bnl_ns_de_2p5km") == "lam_bnl_ns_de_2p5km"
    assert normalize_domain_name("lam_bnl") == "lam_bnl_ns_de_2p5km"
    assert normalize_domain_name("d3_nw_2p5km") == "lam_bnl_ns_de_2p5km"
    assert normalize_domain_name("lam_eu_5km") == "lam_eu_5km"
    assert normalize_domain_name("europe_5km") == "lam_eu_5km"
    assert normalize_domain_name("d3_nl_2p5km") == "d3_nl_2p5km"
    assert normalize_domain_name("d2_eu_lcc") == "d2_eu_lcc"
    assert normalize_domain_name("d1_global_0p4") == "d1_0p4"
    assert normalize_domain_name("d1_global_0p25") == "d1_global_0p25"
    assert DOMAIN_GEOMETRIES["lam_bnl_ns_de_2p5km"] == (432, 432)
    assert DOMAIN_GEOMETRIES["lam_eu_5km"] == (480, 560)


def test_ecmwf_lbc_scheduler_shifted_cycles_and_fallbacks() -> None:
    # 1. Standard KNMI shift: 18Z LAM driven by 06Z ECMWF 0p1 (offset=12h, 48h -> steps 12..60)
    plan = resolve_ecmwf_lbc_plan(
        _utc(2026, 11, 22, 18),
        48.0,
        resol="auto",
        lbc_interval_hours=3.0,
        latest_fn=lambda res, _st: _utc(2026, 11, 22, 6 if res == "0p1" else 12),
    )
    assert plan.resol == "0p1"
    assert plan.ecmwf_cycle_utc == _utc(2026, 11, 22, 6)
    assert plan.lead_offset_hours == pytest.approx(12.0)
    assert plan.ic_step == 12
    assert plan.ecmwf_steps == tuple(range(12, 61, 3))
    assert plan.sim_times_sec[0] == pytest.approx(0.0)
    assert plan.sim_times_sec[-1] == pytest.approx(48.0 * 3600.0)

    # 2. Off-3h LAM target (19Z, latest=06Z, 6h) brackets t=0 cleanly: steps 12..21
    plan_19 = plan_ecmwf_lbc_steps(_utc(2026, 11, 22, 6), _utc(2026, 11, 22, 19), 6.0)
    assert plan_19.ecmwf_steps == (12, 15, 18, 21)
    assert plan_19.sim_times_sec == pytest.approx((-3600.0, 7200.0, 18000.0, 28800.0))

    # 3. 06Z/18Z 144h cap exceeded (target=18Z, hours=138 -> needs step 150) steps back to 00Z
    plan_long = resolve_ecmwf_lbc_plan(
        _utc(2026, 11, 22, 18),
        138.0,
        resol="0p1",
        latest_fn=lambda _res, _st: _utc(2026, 11, 22, 6),
    )
    assert plan_long.ecmwf_cycle_utc == _utc(2026, 11, 22, 0)
    assert plan_long.lead_offset_hours == pytest.approx(18.0)
    assert plan_long.ecmwf_steps[-1] == 156

    # 4. Mid-upload cycle (step=0 at 12Z, step=60 still at 06Z) steps back to complete 06Z cycle
    plan_mid = resolve_ecmwf_lbc_plan(
        _utc(2026, 11, 22, 18),
        48.0,
        resol="0p1",
        latest_fn=lambda _res, step: _utc(2026, 11, 22, 12 if step == 0 else 6),
    )
    assert plan_mid.ecmwf_cycle_utc == _utc(2026, 11, 22, 6)

    # 5. Stale/failing 0p1 falls back cleanly to 0p25 under resol="auto"
    def _stale_0p1(res: str, _step: int) -> datetime.datetime:
        if res == "0p1":
            raise RuntimeError("0p1 temporarily unreachable")
        return _utc(2026, 11, 22, 12)

    plan_fb = resolve_ecmwf_lbc_plan(_utc(2026, 11, 22, 18), 24.0, resol="auto", latest_fn=_stale_0p1)
    assert plan_fb.resol == "0p25"
    assert plan_fb.ecmwf_cycle_utc == _utc(2026, 11, 22, 12)
    assert plan_fb.lead_offset_hours == pytest.approx(6.0)


def test_cropped_regrid_matches_uncropped_exact() -> None:
    lats_desc = np.linspace(90.0, -90.0, 361, dtype=np.float64)
    lons_360 = np.arange(720, dtype=np.float64) * 0.5
    lon2d_360, lat2d_desc = np.meshgrid(lons_360, lats_desc)
    field_360 = (
        280.0
        + 12.0 * np.cos(np.radians(lat2d_desc))
        + 4.0 * np.sin(np.radians(2.0 * lon2d_360))
    ).astype(np.float32)

    # Compute exact reference via uncropped periodic-unwrapped RegularGridInterpolator
    lats_asc = lats_desc[::-1]
    lons_w = ((lons_360 + 180.0) % 360.0) - 180.0
    order = np.argsort(lons_w)
    lons_180 = np.append(lons_w[order], lons_w[order][0] + 360.0)
    f_ref = field_360[::-1, order].astype(np.float64)
    f_ref = np.concatenate([f_ref, f_ref[:, :1]], axis=1)
    itp_ref = RegularGridInterpolator((lats_asc, lons_180), f_ref, method="linear", bounds_error=False, fill_value=None)

    for dom_name in ("lam_bnl_ns_de_2p5km", "lam_eu_5km"):
        spec = get_domain_spec(dom_name, nz=16)
        tgt_lon = ((spec.lon2d + 180.0) % 360.0) - 180.0
        pts = np.column_stack([spec.lat2d.ravel(), tgt_lon.ravel()])
        ref_2d = itp_ref(pts).reshape(spec.ny, spec.nx)

        # 1) Global 0..360 descending lat
        got_360 = _regrid_latlon_field(field_360, lats_desc, lons_360, spec)
        np.testing.assert_allclose(got_360, ref_2d, atol=1e-5, rtol=1e-5)

        # 2) Global -180..179.5 ascending lat, 3D stack
        f_3d = np.stack([f_ref[:, :-1], 2.0 * f_ref[:, :-1]], axis=0).astype(np.float32)
        got_3d = _regrid_latlon_field(f_3d, lats_asc, lons_w[order], spec)
        np.testing.assert_allclose(got_3d[0], ref_2d, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(got_3d[1], 2.0 * ref_2d, atol=2e-5, rtol=1e-5)


def test_ecmwf_lbc_lam_pipeline_and_daemon_warm_start_gpu(tmp_path: Path) -> None:
    proj = LCCProjection(lat1_deg=50.0, lat2_deg=55.5, lat0_deg=52.8, lon0_deg=4.5)
    vgrid = VerticalGrid(nz=16, z_top_m=18_000.0)
    lam_spec = _make_lcc_domain(
        "lam_bnl_ns_de_2p5km",
        ny=24,
        nx=24,
        dx_m=10_000.0,
        projection=proj,
        vertical=vgrid,
        center_lat_deg=52.8,
        center_lon_deg=4.5,
    )
    nz, ny, nx = lam_spec.nz, lam_spec.ny, lam_spec.nx
    hs_m = np.zeros((ny, nx), dtype=np.float64)
    lsm = np.ones((ny, nx), dtype=np.float32)
    lsm[:, : nx // 2] = 0.0
    bathy = np.where(lsm < 0.5, 45.0, 0.0).astype(np.float32)
    sstatic = {
        "lat_deg": lam_spec.lat2d.astype(np.float32),
        "lon_deg": lam_spec.lon2d.astype(np.float32),
        "lsm": lsm,
        "lake_mask": np.zeros((ny, nx), dtype=np.float32),
        "soil_type": np.where(lsm > 0.5, 9.0, 0.0).astype(np.float32),
        "bathy_depth_m": bathy,
        "z0m_base": np.where(lsm > 0.5, 0.1, 1e-4).astype(np.float32),
        "albedo_base": np.where(lsm > 0.5, 0.2, 0.06).astype(np.float32),
        "veg_frac": np.where(lsm > 0.5, 0.5, 0.0).astype(np.float32),
    }

    st0 = {
        "u": np.full((nz, ny, nx + 1), 8.0, dtype=np.float32),
        "v": np.full((nz, ny + 1, nx), 2.0, dtype=np.float32),
        "w": np.zeros((nz + 1, ny, nx), dtype=np.float32),
        "theta_p": np.zeros((nz, ny, nx), dtype=np.float32),
        "pi_p": np.zeros((nz, ny, nx), dtype=np.float32),
        "qv": np.full((nz, ny, nx), 0.005, dtype=np.float32),
        "qc": np.zeros((nz, ny, nx), dtype=np.float32),
        "qr": np.zeros((nz, ny, nx), dtype=np.float32),
        "qi": np.zeros((nz, ny, nx), dtype=np.float32),
        "qs": np.zeros((nz, ny, nx), dtype=np.float32),
        "qg": np.zeros((nz, ny, nx), dtype=np.float32),
    }
    sf0 = {
        "soil_t": np.full((4, ny, nx), 283.0, dtype=np.float32),
        "soil_m": np.full((4, ny, nx), 0.30, dtype=np.float32),
        "t_skin": np.full((ny, nx), 283.0, dtype=np.float32),
        "t2m": np.full((ny, nx), 283.0, dtype=np.float32),
        "q2m": np.full((ny, nx), 0.006, dtype=np.float32),
        "sst_bulk": np.full((ny, nx), 284.0, dtype=np.float32),
        "lake_tml": np.full((ny, nx), 282.0, dtype=np.float32),
        "lake_tbot": np.full((ny, nx), 278.0, dtype=np.float32),
        "lake_hmld": np.full((ny, nx), 3.0, dtype=np.float32),
        "sea_ice_tsfc": np.full((ny, nx), 271.35, dtype=np.float32),
        "sea_ice_albedo": np.full((ny, nx), 0.066, dtype=np.float32),
    }

    init_t0 = _utc(2026, 11, 22, 6)
    rst0 = save_tempest_restart_wx1(tmp_path / "init_t0.wx1", init_t0, lam_spec, hs_m, st0, sf0)

    lbc_dir = tmp_path / "prebuilt_lbc"
    export_child_lbc_step(lbc_dir, 0, 0.0, init_t0, lam_spec, st0, hs_m, lam_spec, hs_m, rim_width=6)
    st1 = dict(st0)
    st1["u"] = np.full((nz, ny, nx + 1), 10.0, dtype=np.float32)
    export_child_lbc_step(
        lbc_dir, 1, 3600.0, init_t0 + datetime.timedelta(hours=1), lam_spec, st1, hs_m, lam_spec, hs_m, rim_width=6
    )

    runs_root = tmp_path / "runs"
    # Cycle 1: standalone LAM + Davies LBC + coupled NEMO+SI3 writing T+1.0h cycle restart (07:00Z)
    res1 = run_ecmwf_lbc_lam_pipeline(
        lam_spec=lam_spec,
        forecast_hours=1.0,
        nz=nz,
        init_time_utc=init_t0,
        ic_source="restart",
        restart_path=rst0,
        runs_root=runs_root,
        cycle_restart_hours=(1.0,),
        enable_da=False,
        dt_s=120.0,
        physics_every=2,
        lbc_dir=lbc_dir,
        download_lbc=False,
        enable_ocean_si3=True,
        include_3d_diagnostics=False,
        lam_static=(hs_m, sstatic),
        show_progress=False,
        log_steps=False,
    )
    assert len(res1["cycle_restart_paths"]) == 1
    cyc_rst = Path(res1["cycle_restart_paths"][0])
    ok_rst, msg_rst = validate_v2_restart(
        cyc_rst, expected_domain_name="lam_bnl_ns_de_2p5km", expected_shape=(nz, ny, nx)
    )
    assert ok_rst, msg_rst

    # Cycle 2 via daemon.run_single_cycle with lam_ecmwf_lbc=True -> warm-starts from Cycle 1's 07:00Z restart!
    cfg = DaemonConfig(
        runs_root=runs_root,
        lam_ecmwf_lbc=True,
        lam_domain="lam_bnl_ns_de_2p5km",
        lam_lead_hours=0.2,
        dt_lam_s=60.0,
        nz=nz,
        physics_every=2,
        enable_ocean_si3=True,
        include_3d_diagnostics=False,
        enable_da=False,
        enable_notifier=False,
    )
    # Pre-seed run2's prebuilt LBCs so zero network is touched
    t1 = init_t0 + datetime.timedelta(hours=1)
    run2_lbc_dir = runs_root / f"{t1:%Y%m%d_%H}Z" / f"lbc_{lam_spec.name}"
    export_child_lbc_step(run2_lbc_dir, 0, 0.0, t1, lam_spec, st0, hs_m, lam_spec, hs_m, rim_width=6)
    export_child_lbc_step(run2_lbc_dir, 1, 3600.0, t1 + datetime.timedelta(hours=1), lam_spec, st1, hs_m, lam_spec, hs_m, rim_width=6)

    # Monkeypatch-free: prebuilt run2_lbc_dir is automatically reused by run_ecmwf_lbc_lam_pipeline
    # Wait: _resolve_static would hit default static dir; use small test with cached static if available or run directly.
    import tempest_v2.runner as runner_mod

    orig_resolve = runner_mod._resolve_static
    runner_mod._resolve_static = lambda _sp, _ov=None: (hs_m, sstatic)
    try:
        cyc2 = run_single_cycle(cfg, t1, d3_spec=lam_spec)
    finally:
        runner_mod._resolve_static = orig_resolve

    assert cyc2["warm_started"] is True
    assert Path(cyc2["restart_path"]).resolve() in {
        cyc_rst.resolve(),
        (cyc_rst.parent / f"restart_{lam_spec.name}.wx1").resolve(),
    }
    assert cyc2["pipeline"]["lam"]["ocean_restart_path"] is not None
    manifest2 = json.loads(Path(cyc2["pipeline"]["manifest_path"]).read_text())
    assert manifest2["ocean"]["enabled"] is True
    assert np.all(np.isfinite(cyc2["pipeline"]["lam"]["state_3d"]["theta_p"]))

    # Offline real-GRIB2 check against existing sample ECMWF GRIB2 files on disk
    grib_dir = next(
        (
            d
            for d in (Path("data/live_verify/raw_ecmwf"), Path("runs/raw_ecmwf"))
            if (d / "ecmwf_ifs_0p25_sfc_step000.grib2").is_file()
            and (d / "ecmwf_ifs_0p25_pl_step000.grib2").is_file()
        ),
        None,
    )
    if grib_dir is not None:
        sfc_sample = grib_dir / "ecmwf_ifs_0p25_sfc_step000.grib2"
        pl_sample = grib_dir / "ecmwf_ifs_0p25_pl_step000.grib2"
        real_spec = get_domain_spec("lam_bnl_ns_de_2p5km", nz=65)
        real_hs = np.zeros((real_spec.ny, real_spec.nx), dtype=np.float64)
        lbc_wx1_dir = tmp_path / "real_grib_lbc"
        out_p = ecmwf_grib_to_lbc_wx1(
            sfc_sample,
            pl_sample,
            real_spec,
            real_hs,
            lbc_wx1_dir,
            0,
            0.0,
            init_t0,
            resol="0p25",
            ecmwf_cycle_utc=init_t0,
            ecmwf_step_hours=0,
            rim_width=16,
        )
        assert out_p.is_file()
        with wx1.open_dataset(out_p) as ds:
            assert ds.attrs.get("lbc_storage") == "rim_packed"
            assert int(ds.attrs["child_ny"]) == 432
            assert int(ds.attrs["child_nx"]) == 432
        interp = LBCInterpolator(lbc_wx1_dir, real_spec.name)
        unpacked = interp.get_lbc_at_time(0.0)
        assert unpacked["u"].shape == (65, 432, 432)
        assert unpacked["qv"].shape == (65, 432, 432)
        assert np.all(np.isfinite(unpacked["theta_p"]))


def test_prepare_ecmwf_lbc_series_ic_vs_lbc_vars_and_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import tempest_v2.ecmwf_lbc as elbc
    from tempest_v2.ingest_ecmwf import LBC_SFC_VARS, STANDARD_SFC_VARS

    spec = get_domain_spec("d3", nz=16)
    hs = np.zeros((spec.ny, spec.nx), dtype=np.float64)
    cyc_dt = _utc(2026, 11, 22, 6)
    tgt_dt = _utc(2026, 11, 22, 12)
    calls: list[tuple[int, tuple[str, ...], bool]] = []

    def _fake_dl(*, step: int, resol: str, output_dir: Path, sfc_vars, include_soil: bool, **_kw):
        calls.append((int(step), tuple(sfc_vars), bool(include_soil)))
        sfc_p = Path(output_dir) / f"sfc_{resol}_{step}.grib2"
        pl_p = Path(output_dir) / f"pl_{resol}_{step}.grib2"
        sfc_p.write_bytes(b"GRIB_SFC")
        pl_p.write_bytes(b"GRIB_PL")
        return sfc_p, pl_p

    def _fake_conv(sfc_p, pl_p, cspec, _hs, out_dir, step_idx, sim_s, v_dt, **_kw):
        out_p = Path(out_dir) / f"lbc_{cspec.name}_step_{step_idx:04d}.wx1"
        out_p.parent.mkdir(parents=True, exist_ok=True)
        # Write tiny valid WX-1 header matching _cached_lbc_wx1_matches metadata
        import xarray as xr

        ds = xr.Dataset(
            data_vars={"ps": (("y", "x"), np.full((cspec.ny, cspec.nx), 101325.0, dtype=np.float32))},
            coords={
                "z": ("z", np.arange(cspec.nz, dtype=np.float32)),
                "y": ("y", cspec.y_m.astype(np.float64)),
                "x": ("x", cspec.x_m.astype(np.float64)),
            },
            attrs={
                "child_domain": cspec.name,
                "child_ny": str(cspec.ny),
                "child_nx": str(cspec.nx),
                "sim_time_sec": f"{sim_s:.3f}",
                "valid_time_utc": v_dt.isoformat(),
            },
        )
        wx1.write_dataset(out_p, ds, dataset_id=f"test_{step_idx}", codec="zstd_shuffle")
        return out_p

    monkeypatch.setattr(elbc, "ecmwf_grib_to_lbc_wx1", _fake_conv)

    lbc_dir = tmp_path / "lbc"
    cache_dir = tmp_path / "cache"
    res = prepare_ecmwf_lbc_series(
        spec,
        hs,
        tgt_dt,
        6.0,
        lbc_dir,
        resol="0p1",
        grib_cache_dir=cache_dir,
        need_ic_grib=True,
        latest_fn=lambda _r, _s: cyc_dt,
        download_fn=_fake_dl,
    )
    assert [c[0] for c in calls] == [6, 9, 12]
    assert calls[0][1] == tuple(STANDARD_SFC_VARS) and calls[0][2] is True
    assert calls[1][1] == tuple(LBC_SFC_VARS) and calls[1][2] is False
    assert res["ic_sfc_grib"] is not None and res["ic_pl_grib"] is not None

    # Second call reusing existing LBC WX-1s without IC downloads should skip network completely
    calls.clear()
    prepare_ecmwf_lbc_series(
        spec,
        hs,
        tgt_dt,
        6.0,
        lbc_dir,
        resol="0p1",
        grib_cache_dir=cache_dir,
        need_ic_grib=False,
        latest_fn=lambda _r, _s: cyc_dt,
        download_fn=_fake_dl,
    )
    assert calls == []
