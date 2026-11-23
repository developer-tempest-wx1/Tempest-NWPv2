"""Tests for automated KNMI 10-minute verification metrics, offline .wx1 sampling, and map plotting."""

from __future__ import annotations

import datetime
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pytest
import xarray as xr
import wx1

from tempest_v2.knmi_verify import (
    _bilinear_sample_latlon_2d,
    _knmi_vv_code_to_meters,
    _normalize_knmi_station_id,
    compute_continuous_stats,
    compute_fog_contingency,
    compute_fractions_skill_score,
    compute_skill_score_vs_ifs,
    compute_wind_dir_stats_deg,
    load_knmi_stations,
    sample_ifs_grib_at_stations,
    verify_run_against_knmi,
)
from tempest_v2.projection import get_domain_spec


def test_continuous_wind_and_fog_metrics() -> None:
    st = compute_continuous_stats([10.0, 12.0, 14.0], [9.0, 12.0, 15.0])
    assert st["n"] == 3
    assert st["bias"] == pytest.approx(0.0)
    assert st["mae"] == pytest.approx(2.0 / 3.0)
    assert st["rmse"] == pytest.approx(np.sqrt(2.0 / 3.0))
    assert st["corr"] == pytest.approx(1.0)

    # Circular wrap-around test: 5 deg vs 355 deg is +10 deg, NOT -350 deg
    wd = compute_wind_dir_stats_deg([5.0, 180.0, 90.0], [355.0, 190.0, 0.0], [4.0, 3.0, 0.5], min_ws_ms=1.5)
    assert wd["n"] == 2  # third station below min_ws_ms excluded
    assert wd["bias_deg"] == pytest.approx(0.0)
    assert wd["mae_deg"] == pytest.approx(10.0)
    assert wd["rmse_deg"] == pytest.approx(10.0)

    # Default min_ws_ms = 3.0 excludes light-wind (< 3 m/s) stations
    wd_def = compute_wind_dir_stats_deg([5.0, 180.0, 90.0], [355.0, 190.0, 0.0], [4.0, 2.5, 0.5])
    assert wd_def["n"] == 1
    assert wd_def["min_ws_ms"] == pytest.approx(3.0)
    assert wd_def["mae_deg"] == pytest.approx(10.0)

    fg = compute_fog_contingency([500.0, 1500.0, 400.0, 2000.0], [300.0, 800.0, 1800.0, 3000.0], threshold_m=1000.0)
    assert fg["hits"] == 1
    assert fg["misses"] == 1
    assert fg["false_alarms"] == 1
    assert fg["correct_negatives"] == 1
    assert fg["pod"] == pytest.approx(0.5)
    assert fg["far"] == pytest.approx(0.5)
    assert fg["csi"] == pytest.approx(1.0 / 3.0)


def _write_synthetic_knmi_nc(
    path: Path,
    valid_dt: datetime.datetime,
    ta_c: np.ndarray,
    pp_hpa: np.ndarray,
    ff_ms: np.ndarray,
    dd_deg: np.ndarray,
    rh_pct: np.ndarray,
    vis_m: np.ndarray,
    qg_wm2: np.ndarray | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    epoch = datetime.datetime(1950, 1, 1, tzinfo=datetime.timezone.utc)
    sec_1950 = (valid_dt - epoch).total_seconds()
    stns = [
        ("06260", "DE BILT AWS", 52.100, 5.183, 1.9),
        ("06240", "SCHIPHOL AWS", 52.316, 4.790, -3.3),
        ("06290", "TWENTHE AWS", 52.750, 6.897, 16.0),
    ]
    n = len(stns)
    with nc.Dataset(path, "w") as ds:
        ds.createDimension("station", n)
        ds.createDimension("time", 1)
        v_id = ds.createVariable("station", "S5", ("station",))
        v_nm = ds.createVariable("stationname", "S20", ("station",))
        v_lat = ds.createVariable("lat", "f8", ("station",))
        v_lon = ds.createVariable("lon", "f8", ("station",))
        v_h = ds.createVariable("height", "f8", ("station",))
        v_t = ds.createVariable("time", "f8", ("time",))
        v_ta = ds.createVariable("ta", "f8", ("station", "time"))
        v_td = ds.createVariable("td", "f8", ("station", "time"))
        v_rh = ds.createVariable("rh", "f8", ("station", "time"))
        v_pp = ds.createVariable("pp", "f8", ("station", "time"))
        v_ff = ds.createVariable("ff", "f8", ("station", "time"))
        v_dd = ds.createVariable("dd", "f8", ("station", "time"))
        v_zm = ds.createVariable("zm", "f8", ("station", "time"))
        v_rg = ds.createVariable("rg", "f8", ("station", "time"))
        v_qg = ds.createVariable("qg", "f8", ("station", "time"))

        v_id[:] = np.array([s[0].encode() for s in stns], dtype="S5")
        v_nm[:] = np.array([s[1].ljust(20)[:20].encode() for s in stns], dtype="S20")
        v_lat[:] = [s[2] for s in stns]
        v_lon[:] = [s[3] for s in stns]
        v_h[:] = [s[4] for s in stns]
        v_t[:] = [sec_1950]
        v_ta[:, 0] = ta_c
        v_td[:, 0] = ta_c - 1.0
        v_rh[:, 0] = rh_pct
        v_pp[:, 0] = pp_hpa
        v_ff[:, 0] = ff_ms
        v_dd[:, 0] = dd_deg
        v_zm[:, 0] = vis_m
        v_rg[:, 0] = [0.0, 0.2, 0.0]
        v_qg[:, 0] = qg_wm2 if qg_wm2 is not None else [250.0, 260.0, 240.0]
    return path


def _write_synthetic_wx1(
    path: Path,
    spec,
    valid_dt: datetime.datetime,
    lead_h: float,
    t2m_c: float,
    mslp_hpa: float,
    *,
    tcc_frac: float = 0.5,
) -> None:
    ny, nx = spec.ny, spec.nx
    shape = (ny, nx)
    rain = np.zeros(shape, dtype=np.float32)
    rain[0, 0] = 20.0 * float(lead_h)  # simulated rim precip
    rain[ny // 2, nx // 2] = 0.5 * float(lead_h)
    pr_si = np.zeros(shape, dtype=np.float32)
    if lead_h > 0:
        pr_si[0, 0] = 10.0 / 3600.0
        pr_si[ny // 2, nx // 2] = 0.5 / 3600.0

    ds = xr.Dataset(
        data_vars={
            "t2m": (("y", "x"), np.full(shape, 273.15 + t2m_c, dtype=np.float32)),
            "td2m": (("y", "x"), np.full(shape, 273.15 + t2m_c - 1.0, dtype=np.float32)),
            "rh2m": (("y", "x"), np.full(shape, 92.0, dtype=np.float32)),
            "mslp": (("y", "x"), np.full(shape, mslp_hpa * 100.0, dtype=np.float32)),
            "u10m_geo": (("y", "x"), np.full(shape, 5.0, dtype=np.float32)),
            "v10m_geo": (("y", "x"), np.full(shape, 0.0, dtype=np.float32)),  # 270 deg, 5.0 m/s
            "gust10m": (("y", "x"), np.full(shape, 8.0, dtype=np.float32)),
            "vis_m": (("y", "x"), np.full(shape, 600.0 if lead_h > 0 else 5000.0, dtype=np.float32)),
            "sw_down": (("y", "x"), np.full(shape, 250.0, dtype=np.float32)),
            "tcc": (("y", "x"), np.full(shape, float(tcc_frac), dtype=np.float32)),
            "precip_rate": (("y", "x"), pr_si),
            "rain_acc": (("y", "x"), rain),
            "snow_acc": (("y", "x"), np.zeros(shape, dtype=np.float32)),
            "graupel_acc": (("y", "x"), np.zeros(shape, dtype=np.float32)),
            "lat": (("y", "x"), spec.lat2d.astype(np.float32)),
            "lon": (("y", "x"), spec.lon2d.astype(np.float32)),
        },
        coords={
            "y": ("y", spec.y_m.astype(np.float64)),
            "x": ("x", spec.x_m.astype(np.float64)),
        },
        attrs={
            "domain_name": spec.name,
            "git_commit": "deadbeef",
            "lead_hours": f"{lead_h:.3f}",
            "valid_time_utc": valid_dt.isoformat(),
        },
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    wx1.write_dataset(path, ds, dataset_id=f"test_{int(lead_h)}", codec="zstd_shuffle")


def test_verify_run_offline_synthetic_wx1_and_knmi_nc(tmp_path: Path) -> None:
    spec = get_domain_spec("lam_bnl_ns_de_2p5km", nz=16)
    t0 = datetime.datetime(2026, 11, 22, 18, 0, tzinfo=datetime.timezone.utc)
    t1 = datetime.datetime(2026, 11, 22, 19, 0, tzinfo=datetime.timezone.utc)

    run_dir = tmp_path / "run"
    _write_synthetic_wx1(run_dir / "d3" / f"forecast_{spec.name}_T+000h.wx1", spec, t0, 0.0, t2m_c=4.0, mslp_hpa=1012.0)
    _write_synthetic_wx1(run_dir / "d3" / f"forecast_{spec.name}_T+001h.wx1", spec, t1, 1.0, t2m_c=3.0, mslp_hpa=1014.0)

    cache_dir = tmp_path / "knmi_cache"
    _write_synthetic_knmi_nc(
        cache_dir / f"KMDS__OPER_P___10M_OBS_L2_{t0:%Y%m%d%H%M}.nc",
        t0,
        ta_c=np.array([5.0, 5.0, 5.0]),
        pp_hpa=np.array([1012.0, 1012.0, 1012.0]),
        ff_ms=np.array([4.0, 4.0, 4.0]),
        dd_deg=np.array([270.0, 270.0, 270.0]),
        rh_pct=np.array([90.0, 90.0, 90.0]),
        vis_m=np.array([5000.0, 5000.0, 5000.0]),
    )
    _write_synthetic_knmi_nc(
        cache_dir / f"KMDS__OPER_P___10M_OBS_L2_{t1:%Y%m%d%H%M}.nc",
        t1,
        ta_c=np.array([3.0, 3.0, 3.0]),
        pp_hpa=np.array([1014.0, 1014.0, 1014.0]),
        ff_ms=np.array([5.0, 5.0, 5.0]),
        dd_deg=np.array([270.0, 270.0, 270.0]),
        rh_pct=np.array([92.0, 92.0, 92.0]),
        vis_m=np.array([500.0, 2000.0, 400.0]),
    )

    out_dir = tmp_path / "verify_out"
    res = verify_run_against_knmi(
        run_dir,
        outdir=out_dir,
        knmi_cache_dir=cache_dir,
        independent_after_hour=1.0,
        make_maps=True,
    )

    assert res["verified_hours"] == 2
    assert res["hourly"][0]["land"]["t2m"]["bias"] == pytest.approx(-1.0, abs=1e-4)
    assert res["hourly"][0]["land"]["ws10"]["bias"] == pytest.approx(1.0, abs=1e-4)
    assert res["hourly"][0]["land"]["wdir"]["mae_deg"] == pytest.approx(0.0, abs=1e-3)
    assert res["hourly"][1]["land"]["t2m"]["rmse"] == pytest.approx(0.0, abs=1e-4)
    assert res["hourly"][1]["land"]["fog"]["hits"] == 2
    assert res["hourly"][1]["land"]["fog"]["false_alarms"] == 1
    assert res["independent_land"]["t2m"]["n"] == 3
    assert res["overall_land"]["n_unique_stations"] == 3
    assert res["overall_land"]["n_stations"] == 3
    assert res["overall_land"]["n_station_hours"] == 6
    assert res["overall_land"]["sw_down"]["n"] == 6
    assert res["overall_land"]["sw_down"]["bias"] == pytest.approx(0.0, abs=1e-3)
    assert res["run_metadata"]["git_commit"] == "deadbeef"
    assert Path(res["artifacts"]["summary_json"]).is_file()
    assert Path(res["artifacts"]["hourly_csv"]).is_file()
    assert Path(res["artifacts"]["station_csv"]).is_file()
    assert Path(res["artifacts"]["timeseries_png"]).stat().st_size > 10_000
    assert len(res["artifacts"]["map_pngs"]) == 2
    assert all(Path(p).stat().st_size > 10_000 for p in res["artifacts"]["map_pngs"])


def test_ifs_latlon_bilinear_sampling_and_comparison(tmp_path: Path, monkeypatch) -> None:
    """Verify IFS lat/lon bilinear sampling, Skill Score vs IFS, partial GRIB2 handling, and timeseries overlay."""
    from tempest_v2 import ic_builder

    # Test descending lat & 0..360 lon wrapping
    lats_desc = np.array([54.0, 53.0, 52.0, 51.0], dtype=np.float64)
    lons_360 = np.array([3.0, 5.0, 7.0], dtype=np.float64)
    grid = lats_desc[:, None] * 10.0 + lons_360[None, :]
    vals = _bilinear_sample_latlon_2d(grid, lats_desc, lons_360, np.array([52.5, 50.0]), np.array([5.5, 5.5]))
    assert vals[0] == pytest.approx(52.5 * 10.0 + 5.5, abs=1e-6)
    assert np.isnan(vals[1])

    spec = get_domain_spec("lam_bnl_ns_de_2p5km", nz=16)
    t0 = datetime.datetime(2026, 11, 22, 18, 0, tzinfo=datetime.timezone.utc)
    t3 = datetime.datetime(2026, 11, 22, 21, 0, tzinfo=datetime.timezone.utc)

    run_dir = tmp_path / "run"
    _write_synthetic_wx1(run_dir / "d3" / f"forecast_{spec.name}_T+000h.wx1", spec, t0, 0.0, t2m_c=4.0, mslp_hpa=1012.0)
    _write_synthetic_wx1(run_dir / "d3" / f"forecast_{spec.name}_T+003h.wx1", spec, t3, 3.0, t2m_c=3.2, mslp_hpa=1014.0)

    cache_dir = tmp_path / "knmi_cache"
    _write_synthetic_knmi_nc(
        cache_dir / f"KMDS__OPER_P___10M_OBS_L2_{t0:%Y%m%d%H%M}.nc",
        t0,
        ta_c=np.array([4.0, 4.0, 4.0]),
        pp_hpa=np.array([1012.0, 1012.0, 1012.0]),
        ff_ms=np.array([5.0, 5.0, 5.0]),
        dd_deg=np.array([270.0, 270.0, 270.0]),
        rh_pct=np.array([92.0, 92.0, 92.0]),
        vis_m=np.array([5000.0, 5000.0, 5000.0]),
    )
    _write_synthetic_knmi_nc(
        cache_dir / f"KMDS__OPER_P___10M_OBS_L2_{t3:%Y%m%d%H%M}.nc",
        t3,
        ta_c=np.array([3.0, 3.0, 3.0]),  # Tempest bias=+0.2 K, IFS will be 1.5 K warmer -> Tempest wins!
        pp_hpa=np.array([1014.0, 1014.0, 1014.0]),
        ff_ms=np.array([5.0, 5.0, 5.0]),
        dd_deg=np.array([270.0, 270.0, 270.0]),
        rh_pct=np.array([92.0, 92.0, 92.0]),
        vis_m=np.array([600.0, 2000.0, 600.0]),
    )

    ifs_dir = tmp_path / "ifs_cache"
    ifs_dir.mkdir()
    ifs_step12 = ifs_dir / "ecmwf_ifs_0p1_20261122_06Z_sfc_step012.grib2"
    ifs_step15 = ifs_dir / "ecmwf_ifs_0p1_20261122_06Z_sfc_step015.grib2"
    ifs_step12.write_bytes(b"grib")
    ifs_step15.write_bytes(b"grib")

    ifs_lats = np.linspace(54.5, 49.5, 11)
    ifs_lons = np.linspace(2.0, 8.0, 13)

    def _da(val: float):
        arr = np.full((len(ifs_lats), len(ifs_lons)), val, dtype=np.float64)
        return xr.DataArray(arr, dims=("latitude", "longitude"), coords={"latitude": ifs_lats, "longitude": ifs_lons})

    fake_maps = {
        str(ifs_step12): {
            "t2m": (_da(273.15 + 5.0), ifs_lats, ifs_lons),
            "d2m": (_da(273.15 + 3.0), ifs_lats, ifs_lons),
            "msl": (_da(101_300.0), ifs_lats, ifs_lons),
            "u10": (_da(7.0), ifs_lats, ifs_lons),
            "v10": (_da(0.0), ifs_lats, ifs_lons),
        },
        # Step 15 simulates partial LBC GRIB2 with only t2m + msl
        str(ifs_step15): {
            "t2m": (_da(273.15 + 4.5), ifs_lats, ifs_lons),
            "msl": (_da(101_500.0), ifs_lats, ifs_lons),
        },
    }
    monkeypatch.setattr(ic_builder, "_read_cfgrib_var_map", lambda p, use_cache=False: fake_maps[str(p)])

    # Direct unit check of sample_ifs_grib_at_stations
    rows_ifs = sample_ifs_grib_at_stations(ifs_step12, [{"lat": 52.1, "lon": 5.18, "station_id": "06260", "is_offshore": False}])
    assert rows_ifs[0]["mod_t2m_c"] == pytest.approx(5.0, abs=1e-4)
    assert rows_ifs[0]["mod_ws10_ms"] == pytest.approx(7.0, abs=1e-4)

    ss = compute_skill_score_vs_ifs({"t2m": {"rmse": 0.2, "bias": 0.2, "n": 3}}, {"t2m": {"rmse": 1.5, "bias": 1.5, "n": 3}})
    assert ss["t2m"]["skill_score"] == pytest.approx(1.0 - 0.2 / 1.5, abs=1e-4)

    out_dir = tmp_path / "verify_ifs"
    res = verify_run_against_knmi(
        run_dir,
        outdir=out_dir,
        knmi_cache_dir=cache_dir,
        independent_after_hour=2.0,
        make_maps=False,
        ifs_grib_dir=ifs_dir,
    )
    icmp = res["ifs_comparison"]
    assert icmp is not None
    assert icmp["matched_hours"] == 2
    assert icmp["matched_leads"] == [0.0, 3.0]
    assert icmp["independent_land"]["tempest"]["t2m"]["rmse"] == pytest.approx(0.2, abs=1e-4)
    assert icmp["independent_land"]["ifs"]["t2m"]["rmse"] == pytest.approx(1.5, abs=1e-4)
    assert icmp["independent_land"]["skill"]["t2m"]["skill_score"] > 0.85
    # Partial LBC step at T+3 lacks wind/Td, so ws10 is evaluated strictly on shared T+0 hour (n=3)
    assert icmp["overall_land"]["tempest"]["ws10"]["n"] == 3
    assert icmp["overall_land"]["ifs"]["ws10"]["n"] == 3
    csv_txt = Path(res["artifacts"]["hourly_csv"]).read_text(encoding="utf-8")
    assert "ifs_t2m_rmse_k" in csv_txt and "t2m_skill_vs_ifs" in csv_txt
    assert Path(res["artifacts"]["timeseries_png"]).stat().st_size > 10_000


def test_fractions_skill_score_roberts_lean_and_radar_pipeline(tmp_path: Path) -> None:
    """Verify Roberts & Lean (2008) FSS math, NaN radar-mask handling, clim NetCDF4 decoder, and verify_run wiring."""
    from tempest_v2.ingest_radar_qpe import KNMI_RADAR_PROJ4_DEFAULT, decode_knmi_clim_radar_01h_netcdf

    # 1. Analytical FSS properties on 40x40 @ 2.5 km grid
    obs = np.zeros((40, 40), dtype=np.float64)
    mod_same = np.zeros((40, 40), dtype=np.float64)
    mod_shift = np.zeros((40, 40), dtype=np.float64)
    obs[18:22, 18:22] = 3.0
    mod_same[18:22, 18:22] = 3.0
    mod_shift[18:22, 22:26] = 3.0  # Displaced 4 cells (10 km) east
    obs[:, :4] = np.nan  # Outside radar coverage

    fss_id = compute_fractions_skill_score(mod_same, obs, dx_km=2.5, thresholds_mm=(1.0,), scales_km=(2.5, 10.0, 25.0))
    assert fss_id["n_valid"] == 40 * 36
    assert fss_id["thresholds"]["1mm"]["scales"]["2.5km"]["fss"] == pytest.approx(1.0, abs=1e-9)

    fss_sh = compute_fractions_skill_score(mod_shift, obs, dx_km=2.5, thresholds_mm=(1.0,), scales_km=(2.5, 10.0, 25.0, 50.0))
    sc = fss_sh["thresholds"]["1mm"]["scales"]
    assert sc["2.5km"]["fss"] == pytest.approx(0.0, abs=1e-9)
    assert sc["2.5km"]["fss"] < sc["10km"]["fss"] < sc["25km"]["fss"] < sc["50km"]["fss"]
    assert sc["50km"]["fss"] > fss_sh["thresholds"]["1mm"]["fss_uniform"]

    # 2. Synthetic climatological rad_nl25_rac_mfbs_01h_netcdf4 NetCDF4 decoder check
    spec = get_domain_spec("d3_nl_2p5km", nz=16)
    rad_nc = tmp_path / "clim_01h.nc"
    with nc.Dataset(rad_nc, "w") as ds:
        ds.createDimension("y", 20)
        ds.createDimension("x", 20)
        vx = ds.createVariable("x", "f8", ("x",))
        vy = ds.createVariable("y", "f8", ("y",))
        vp = ds.createVariable("precipitation", "f4", ("y", "x"))
        vcrs = ds.createVariable("projection", "i4")
        vcrs.proj4_params = KNMI_RADAR_PROJ4_DEFAULT
        # Stereographic coordinates around De Bilt (~x=350 km, y=-3950 km)
        vx[:] = np.linspace(200.0, 580.0, 20)
        vy[:] = np.linspace(-3750.0, -4130.0, 20)
        vp[:, :] = 2.5
    reproj = decode_knmi_clim_radar_01h_netcdf(rad_nc, spec)
    assert reproj.shape == (spec.ny, spec.nx)
    assert np.any(np.isfinite(reproj))
    assert float(np.nanmean(reproj)) == pytest.approx(2.5, abs=0.1)

    # 3. End-to-end verify_run_against_knmi(..., verify_radar_fss=True)
    t0 = datetime.datetime(2026, 11, 22, 18, 0, tzinfo=datetime.timezone.utc)
    t1 = datetime.datetime(2026, 11, 22, 19, 0, tzinfo=datetime.timezone.utc)
    run_dir = tmp_path / "run_rad"
    _write_synthetic_wx1(run_dir / "d3" / f"forecast_{spec.name}_T+000h.wx1", spec, t0, 0.0, t2m_c=4.0, mslp_hpa=1012.0)
    _write_synthetic_wx1(run_dir / "d3" / f"forecast_{spec.name}_T+001h.wx1", spec, t1, 1.0, t2m_c=3.0, mslp_hpa=1014.0)
    cache_dir = tmp_path / "knmi_cache_rad"
    for dt_x in (t0, t1):
        _write_synthetic_knmi_nc(
            cache_dir / f"KMDS__OPER_P___10M_OBS_L2_{dt_x:%Y%m%d%H%M}.nc",
            dt_x,
            ta_c=np.array([4.0, 4.0, 4.0]),
            pp_hpa=np.array([1012.0, 1012.0, 1012.0]),
            ff_ms=np.array([5.0, 5.0, 5.0]),
            dd_deg=np.array([270.0, 270.0, 270.0]),
            rh_pct=np.array([92.0, 92.0, 92.0]),
            vis_m=np.array([5000.0, 5000.0, 5000.0]),
        )

    def _fake_radar(_dt, rspec):
        arr = np.zeros((rspec.ny, rspec.nx), dtype=np.float32)
        arr[rspec.ny // 2, rspec.nx // 2] = 0.5
        return arr

    res = verify_run_against_knmi(
        run_dir,
        outdir=tmp_path / "out_rad",
        knmi_cache_dir=cache_dir,
        independent_after_hour=1.0,
        make_maps=False,
        verify_radar_fss=True,
        radar_hourly_fetcher=_fake_radar,
    )
    assert res["radar_fss"] is not None
    assert res["radar_fss"]["overall"]["n_hours"] == 1
    assert res["radar_fss"]["overall"]["thresholds"]["0.1mm"]["scales"]["10km"]["fss"] > 0.5


def _write_synthetic_knmi_hourly_validated_nc(path: Path, valid_dt: datetime.datetime, *, fog_step: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    epoch = datetime.datetime(1950, 1, 1, tzinfo=datetime.timezone.utc)
    sec_1950 = (valid_dt - epoch).total_seconds()
    stns = [
        ("0-20000-0-06260", "DE BILT", 52.100, 5.183, 1.9),
        ("0-20000-0-06203", "P11-B", 52.360, 3.342, 41.8),
        ("0-20000-0-06290", "TWENTHE", 52.750, 6.897, 16.0),
        ("0-20000-0-06280", "EELDE", 53.125, 6.585, 5.2),
    ]
    n = len(stns)
    with nc.Dataset(path, "w") as ds:
        ds.createDimension("station", n)
        ds.createDimension("time", 1)
        v_id = ds.createVariable("station", "S15", ("station",))
        v_nm = ds.createVariable("stationname", "S15", ("station",))
        v_lat = ds.createVariable("lat", "f8", ("station",))
        v_lon = ds.createVariable("lon", "f8", ("station",))
        v_h = ds.createVariable("height", "f8", ("station",))
        v_t = ds.createVariable("time", "f8", ("time",))

        def _v2(name: str, vals: list[float]) -> None:
            v = ds.createVariable(name, "f8", ("station", "time"))
            v[:, 0] = vals

        v_id[:] = np.array([s[0].encode() for s in stns], dtype="S15")
        v_nm[:] = np.array([s[1].ljust(15)[:15].encode() for s in stns], dtype="S15")
        v_lat[:] = [s[2] for s in stns]
        v_lon[:] = [s[3] for s in stns]
        v_h[:] = [s[4] for s in stns]
        v_t[:] = [sec_1950]

        _v2("T", [2.0, 6.0, 2.0, 1.5])
        _v2("TD", [1.5, 4.0, 1.0, 1.0])
        _v2("U", [96.0, 82.0, 90.0, 95.0])
        _v2("P", [1016.0, 1015.0, 1016.0, 1016.0])
        _v2("FF", [5.0, np.nan, 4.0, 5.0])
        _v2("FH", [4.0, 9.0, 4.0, 5.0])
        _v2("FX", [7.0, 12.0, 6.0, 7.0])
        _v2("DD", [270.0, 990.0, 0.0, 270.0])  # 990=variable, 0=calm -> both must become NaN
        _v2("RH", [1.4 if fog_step else 0.0, 0.0, 0.2, 0.4])
        _v2("Q", [108.0, 72.0, 108.0, 108.0])  # 108 J/cm^2 == 300.0 W/m^2
        _v2("N", [4.0, 8.0, 9.0, 2.0])  # 9=obscured -> NaN
        _v2("VV", [9.0 if fog_step else 50.0, 80.0, 10.0, np.nan])  # 9=950m, 10=1050m, Eelde NaN->W1
        _v2("W1", [1.0 if fog_step else 0.0, 0.0, 0.0, 1.0 if fog_step else 0.0])
    return path


def test_knmi_hourly_validated_loader_tcc_and_auto_source(tmp_path: Path) -> None:
    assert _normalize_knmi_station_id("0-20000-0-06260") == "06260"
    assert _normalize_knmi_station_id("06260") == "06260"
    assert _knmi_vv_code_to_meters(9.0) == pytest.approx(950.0)
    assert _knmi_vv_code_to_meters(10.0) == pytest.approx(1050.0)
    assert _knmi_vv_code_to_meters(50.0) == pytest.approx(5500.0)
    assert _knmi_vv_code_to_meters(83.0) == pytest.approx(47500.0)
    assert np.isnan(_knmi_vv_code_to_meters(52.0))

    spec = get_domain_spec("lam_bnl_ns_de_2p5km", nz=16)
    t0 = datetime.datetime(2024, 1, 15, 12, 0, tzinfo=datetime.timezone.utc)
    t1 = datetime.datetime(2024, 1, 15, 13, 0, tzinfo=datetime.timezone.utc)

    cache_dir = tmp_path / "knmi_cache"
    nc0 = _write_synthetic_knmi_hourly_validated_nc(
        cache_dir / f"hourly-observations-validated-{t0:%Y%m%d-%H}.nc", t0, fog_step=False
    )
    nc1 = _write_synthetic_knmi_hourly_validated_nc(
        cache_dir / f"hourly-observations-validated-{t1:%Y%m%d-%H}.nc", t1, fog_step=True
    )

    stns = {r["station_id"]: r for r in load_knmi_stations(nc1)}
    assert set(stns.keys()) == {"06260", "06203", "06290", "06280"}
    assert stns["06203"]["is_offshore"] is True
    assert stns["06203"]["ws10_ms"] == pytest.approx(9.0)  # FF=NaN fell back to FH=9.0
    assert np.isnan(stns["06203"]["wd10_deg"])  # DD=990 variable rejected
    assert np.isnan(stns["06290"]["wd10_deg"])  # DD=0 calm rejected
    assert stns["06260"]["sw_down_wm2"] == pytest.approx(300.0)
    assert stns["06260"]["tcc"] == pytest.approx(0.5)
    assert np.isnan(stns["06290"]["tcc"])  # N=9 obscured rejected
    assert stns["06260"]["vis_m"] == pytest.approx(950.0)
    assert stns["06290"]["vis_m"] == pytest.approx(1050.0)
    assert stns["06280"]["vis_m"] == pytest.approx(500.0)  # VV=NaN fell back to W1=1
    assert stns["06260"]["precip_1h_mm"] == pytest.approx(1.4)

    run_dir = tmp_path / "run_hv"
    _write_synthetic_wx1(run_dir / "d3" / f"forecast_{spec.name}_T+000h.wx1", spec, t0, 0.0, t2m_c=2.0, mslp_hpa=1016.0, tcc_frac=0.5)
    _write_synthetic_wx1(run_dir / "d3" / f"forecast_{spec.name}_T+001h.wx1", spec, t1, 1.0, t2m_c=2.0, mslp_hpa=1016.0, tcc_frac=0.5)

    res = verify_run_against_knmi(
        run_dir,
        outdir=tmp_path / "out_hv",
        knmi_cache_dir=cache_dir,
        obs_source="auto",
        independent_after_hour=1.0,
        make_maps=False,
    )
    assert res["verified_hours"] == 2
    assert res["obs_sources_used"] == ["hourly-validated"]
    # T+1h: model tcc=0.5 vs obs [De Bilt 0.5, Twenthe NaN, Eelde 0.25] -> n=2, mean mod-obs = +0.125
    assert res["hourly"][1]["land"]["tcc"]["n"] == 2
    assert res["hourly"][1]["land"]["tcc"]["bias"] == pytest.approx(0.125, abs=1e-4)
    # T+1h fog (<1000m): mod_vis=600m at all 3 land stations; obs fog at De Bilt (950m) & Eelde (500m via W1) -> H=2, FA=1
    assert res["hourly"][1]["land"]["fog"]["hits"] == 2
    assert res["hourly"][1]["land"]["fog"]["false_alarms"] == 1
    assert res["hourly"][1]["land"]["precip"]["obs_mean"] == pytest.approx((1.4 + 0.2 + 0.4) / 3.0, abs=1e-4)
    csv_txt = Path(res["artifacts"]["hourly_csv"]).read_text(encoding="utf-8")
    assert "land_tcc_bias" in csv_txt and "land_tcc_rmse" in csv_txt

