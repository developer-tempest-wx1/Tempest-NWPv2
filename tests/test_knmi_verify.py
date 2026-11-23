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
    compute_continuous_stats,
    compute_fog_contingency,
    compute_wind_dir_stats_deg,
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
    return path


def _write_synthetic_wx1(path: Path, spec, valid_dt: datetime.datetime, lead_h: float, t2m_c: float, mslp_hpa: float) -> None:
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
    assert Path(res["artifacts"]["summary_json"]).is_file()
    assert Path(res["artifacts"]["hourly_csv"]).is_file()
    assert Path(res["artifacts"]["station_csv"]).is_file()
    assert Path(res["artifacts"]["timeseries_png"]).stat().st_size > 10_000
    assert len(res["artifacts"]["map_pngs"]) == 2
    assert all(Path(p).stat().st_size > 10_000 for p in res["artifacts"]["map_pngs"])
