"""Tests for deterministic ARCO-ERA5 & ERA5-Land hindcast ingestion, WX-1 caching, and LBC construction."""

from __future__ import annotations

import datetime
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import wx1

from tempest_v2.ingest_era5 import (
    download_era5_land_soil_cds,
    era5_wx1_to_domain_state,
    era5_wx1_to_lbc_wx1,
    fetch_era5_snapshot_wx1,
    prepare_era5_lbc_series,
)
from tempest_v2.lbc_manager import LBCInterpolator
from tempest_v2.projection import GRAVITY_M_S2, get_domain_spec


def _small_bnl_spec():
    base = get_domain_spec("lam_bnl_ns_de_2p5km", nz=16)
    cy, cx = base.ny // 2, base.nx // 2
    sy, sx = slice(cy - 12, cy + 12), slice(cx - 12, cx + 12)
    return replace(
        base,
        ny=24,
        nx=24,
        x_m=base.x_m[sx].copy(),
        y_m=base.y_m[sy].copy(),
        lat2d=base.lat2d[sy, sx].copy(),
        lon2d=base.lon2d[sy, sx].copy(),
        map_factor=base.map_factor[sy, sx].copy(),
        coriolis_f=base.coriolis_f[sy, sx].copy(),
    )


def _build_synthetic_arco_era5_ds(
    times_utc: list[datetime.datetime],
    *,
    nan_last_time: bool = False,
) -> xr.Dataset:
    """Create a synthetic ARCO-ERA5 Dataset (descending lat, 0..360 lon crossing 0E, 10 pressure levels)."""
    lats = np.linspace(65.0, 42.0, 93, dtype=np.float64)
    # 0..360 convention spanning prime meridian: 345..359.75 and 0..20
    lons_360 = np.concatenate(
        [
            np.arange(0.0, 22.0, 0.25, dtype=np.float64),
            np.arange(344.0, 360.0, 0.25, dtype=np.float64),
        ]
    )
    lons_360.sort()
    levels = np.array([50, 100, 200, 300, 500, 700, 850, 925, 975, 1000], dtype=np.int32)
    t_coords = np.array([np.datetime64(dt.replace(tzinfo=None), "s") for dt in times_utc], dtype="datetime64[ns]")

    nt, nz, ny, nx = len(times_utc), len(levels), len(lats), len(lons_360)
    t_3d = np.zeros((nt, nz, ny, nx), dtype=np.float32)
    z_3d = np.zeros((nt, nz, ny, nx), dtype=np.float32)
    q_3d = np.zeros((nt, nz, ny, nx), dtype=np.float32)
    u_3d = np.zeros((nt, nz, ny, nx), dtype=np.float32)
    v_3d = np.zeros((nt, nz, ny, nx), dtype=np.float32)

    for ti in range(nt):
        for k, p_hpa in enumerate(levels):
            p_rat = float(p_hpa) / 1000.0
            t_lvl = max(210.0, 276.0 - 60.0 * (1.0 - p_rat**0.22) + 0.4 * ti)
            z_m = max(80.0, -7200.0 * np.log(p_rat) + 120.0)
            t_3d[ti, k] = t_lvl
            z_3d[ti, k] = GRAVITY_M_S2 * z_m
            q_3d[ti, k] = 0.004 * (p_rat**2.5)
            u_3d[ti, k] = 6.0 + 10.0 * (1.0 - p_rat) + 0.5 * ti
            v_3d[ti, k] = 2.0 + 0.2 * ti

    lon_180 = np.where(lons_360 >= 180.0, lons_360 - 360.0, lons_360)
    lsm_2d = np.where((lats[:, None] >= 50.0) & (lats[:, None] <= 53.5) & (lon_180[None, :] >= 3.0) & (lon_180[None, :] <= 8.0), 1.0, 0.0).astype(np.float32)

    def _tile2d(val_fn):
        arr = np.empty((nt, ny, nx), dtype=np.float32)
        for ti in range(nt):
            arr[ti] = val_fn(ti)
        return arr

    t2m = _tile2d(lambda ti: 274.0 + 0.5 * ti + 0.0 * lsm_2d)
    d2m = _tile2d(lambda ti: 273.0 + 0.4 * ti + 0.0 * lsm_2d)
    msl = _tile2d(lambda ti: 102_500.0 + 100.0 * ti + 0.0 * lsm_2d)
    sp = _tile2d(lambda ti: 102_300.0 + 100.0 * ti + 0.0 * lsm_2d)
    u10 = _tile2d(lambda ti: 3.5 + 0.3 * ti + 0.0 * lsm_2d)
    v10 = _tile2d(lambda ti: 1.2 + 0.1 * ti + 0.0 * lsm_2d)
    skt = _tile2d(lambda ti: np.where(lsm_2d > 0.5, 273.5 + 0.5 * ti, 283.0))
    sst = _tile2d(lambda ti: np.where(lsm_2d > 0.5, np.nan, 283.0))
    siconc = _tile2d(lambda _ti: np.zeros((ny, nx), dtype=np.float32))
    sd = _tile2d(lambda _ti: np.full((ny, nx), 0.002, dtype=np.float32))  # 2 mm SWE

    if nan_last_time:
        t_3d[-1] = np.nan
        t2m[-1] = np.nan

    data_vars = {
        "geopotential": (("time", "level", "latitude", "longitude"), z_3d),
        "temperature": (("time", "level", "latitude", "longitude"), t_3d),
        "specific_humidity": (("time", "level", "latitude", "longitude"), q_3d),
        "u_component_of_wind": (("time", "level", "latitude", "longitude"), u_3d),
        "v_component_of_wind": (("time", "level", "latitude", "longitude"), v_3d),
        "mean_sea_level_pressure": (("time", "latitude", "longitude"), msl),
        "surface_pressure": (("time", "latitude", "longitude"), sp),
        "2m_temperature": (("time", "latitude", "longitude"), t2m),
        "2m_dewpoint_temperature": (("time", "latitude", "longitude"), d2m),
        "10m_u_component_of_wind": (("time", "latitude", "longitude"), u10),
        "10m_v_component_of_wind": (("time", "latitude", "longitude"), v10),
        "skin_temperature": (("time", "latitude", "longitude"), skt),
        "sea_surface_temperature": (("time", "latitude", "longitude"), sst),
        "sea_ice_cover": (("time", "latitude", "longitude"), siconc),
        "snow_depth": (("time", "latitude", "longitude"), sd),
        "land_sea_mask": (("time", "latitude", "longitude"), np.broadcast_to(lsm_2d[None], (nt, ny, nx)).copy()),
        "surface_solar_radiation_downwards": (("time", "latitude", "longitude"), np.zeros((nt, ny, nx), dtype=np.float32)),
        "soil_temperature_level_1": (("time", "latitude", "longitude"), _tile2d(lambda _ti: np.full((ny, nx), 275.0, dtype=np.float32))),
        "soil_temperature_level_2": (("time", "latitude", "longitude"), _tile2d(lambda _ti: np.full((ny, nx), 276.0, dtype=np.float32))),
        "soil_temperature_level_3": (("time", "latitude", "longitude"), _tile2d(lambda _ti: np.full((ny, nx), 279.0, dtype=np.float32))),
        "soil_temperature_level_4": (("time", "latitude", "longitude"), _tile2d(lambda _ti: np.full((ny, nx), 283.0, dtype=np.float32))),
        "volumetric_soil_water_layer_1": (("time", "latitude", "longitude"), _tile2d(lambda _ti: np.full((ny, nx), 0.34, dtype=np.float32))),
        "volumetric_soil_water_layer_2": (("time", "latitude", "longitude"), _tile2d(lambda _ti: np.full((ny, nx), 0.35, dtype=np.float32))),
        "volumetric_soil_water_layer_3": (("time", "latitude", "longitude"), _tile2d(lambda _ti: np.full((ny, nx), 0.36, dtype=np.float32))),
        "volumetric_soil_water_layer_4": (("time", "latitude", "longitude"), _tile2d(lambda _ti: np.full((ny, nx), 0.37, dtype=np.float32))),
    }
    return xr.Dataset(
        data_vars=data_vars,
        coords={"time": t_coords, "level": levels, "latitude": lats, "longitude": lons_360},
    )


def test_era5_snapshot_cache_domain_state_and_land_override(tmp_path: Path) -> None:
    spec = _small_bnl_spec()
    hs_m = np.zeros((spec.ny, spec.nx), dtype=np.float64)
    lsm_static = np.ones((spec.ny, spec.nx), dtype=np.float32)
    lsm_static[:8, :8] = 0.0  # NW North Sea ocean corner
    sstatic = {"lsm": lsm_static, "lake_mask": np.zeros_like(lsm_static)}

    t0 = datetime.datetime(2024, 1, 15, 0, 0, tzinfo=datetime.timezone.utc)
    t1 = datetime.datetime(2024, 1, 15, 1, 0, tzinfo=datetime.timezone.utc)
    t_future = datetime.datetime(2024, 1, 15, 2, 0, tzinfo=datetime.timezone.utc)
    ds_syn = _build_synthetic_arco_era5_ds([t0, t1, t_future], nan_last_time=True)

    cache_dir = tmp_path / "era5_cache"
    wx1_p = fetch_era5_snapshot_wx1(t0, spec, cache_dir, era5_ds=ds_syn)
    assert wx1_p.is_file()
    mtime0 = wx1_p.stat().st_mtime_ns

    # Second fetch reuses WX-1 cache without needing remote dataset
    wx1_p2 = fetch_era5_snapshot_wx1(t0, spec, cache_dir, era5_ds=None)
    assert wx1_p2 == wx1_p
    assert wx1_p2.stat().st_mtime_ns == mtime0

    # Unpopulated (all-NaN) ERA5 timestamp raises clean ValueError
    with pytest.raises(ValueError, match="unpopulated"):
        fetch_era5_snapshot_wx1(t_future, spec, cache_dir, era5_ds=ds_syn)

    # Build synthetic 9 km ERA5-Land WX-1 soil file (warm soil_t[0]=279 K vs 275 K ERA5, ocean NaN)
    el_lats = np.linspace(55.0, 49.0, 61, dtype=np.float64)
    el_lons = np.linspace(-1.0, 10.0, 111, dtype=np.float64)
    el_lsm_ok = np.broadcast_to(el_lons[None, :] >= 2.0, (len(el_lats), len(el_lons)))
    el_ds = xr.Dataset(
        data_vars={
            "skt": (("latitude", "longitude"), np.where(el_lsm_ok > 0.5, 278.5, np.nan).astype(np.float32)),
            "stl1": (("latitude", "longitude"), np.where(el_lsm_ok > 0.5, 279.0, np.nan).astype(np.float32)),
            "stl2": (("latitude", "longitude"), np.full((61, 111), 278.0, dtype=np.float32)),
            "stl3": (("latitude", "longitude"), np.full((61, 111), 280.0, dtype=np.float32)),
            "stl4": (("latitude", "longitude"), np.full((61, 111), 282.0, dtype=np.float32)),
            "swvl1": (("latitude", "longitude"), np.full((61, 111), 0.42, dtype=np.float32)),
            "sd": (("latitude", "longitude"), np.full((61, 111), 0.005, dtype=np.float32)),
        },
        coords={"latitude": el_lats, "longitude": el_lons},
    )
    el_path = tmp_path / "era5_land_0p09.wx1"
    wx1.write_dataset(el_path, el_ds, dataset_id="era5_land_test", codec="zstd_shuffle")

    st_base, sf_base, vdt_base = era5_wx1_to_domain_state(wx1_p, spec, hs_m, surface_static=sstatic)
    assert vdt_base == t0
    assert float(np.mean(sf_base["soil_t"][0, 12:, 12:])) == pytest.approx(275.0, abs=0.2)
    assert float(np.mean(sf_base["snow_swe"])) == pytest.approx(2.0, abs=0.1)

    st_el, sf_el, _ = era5_wx1_to_domain_state(
        wx1_p, spec, hs_m, era5_land_soil_path=el_path, surface_static=sstatic
    )
    # Land overridden by 9 km ERA5-Land (279 K, 0.42 m3/m3, 5 mm SWE), ocean SST preserved (283 K)
    assert float(np.mean(sf_el["soil_t"][0, 12:, 12:])) == pytest.approx(279.0, abs=0.2)
    assert float(np.mean(sf_el["soil_m"][0, 12:, 12:])) == pytest.approx(0.42, abs=0.02)
    assert float(np.mean(sf_el["t_skin"][12:, 12:])) == pytest.approx(278.5, abs=0.2)
    assert float(np.mean(sf_el["t_skin"][:6, :6])) == pytest.approx(283.0, abs=0.2)
    assert float(np.mean(sf_el["snow_swe"][12:, 12:])) == pytest.approx(5.0, abs=0.2)
    assert all(np.all(np.isfinite(arr)) for arr in st_el.values())


def test_prepare_era5_lbc_series_and_interpolator(tmp_path: Path) -> None:
    spec = _small_bnl_spec()
    hs_m = np.zeros((spec.ny, spec.nx), dtype=np.float64)
    t0 = datetime.datetime(2024, 1, 15, 0, 0, tzinfo=datetime.timezone.utc)
    t1 = datetime.datetime(2024, 1, 15, 1, 0, tzinfo=datetime.timezone.utc)
    t2 = datetime.datetime(2024, 1, 15, 2, 0, tzinfo=datetime.timezone.utc)
    ds_syn = _build_synthetic_arco_era5_ds([t0, t1, t2])

    lbc_dir = tmp_path / "lbc_era5"
    cache_dir = tmp_path / "era5_cache"
    prep = prepare_era5_lbc_series(
        spec,
        hs_m,
        t0,
        forecast_hours=2.0,
        lbc_dir=lbc_dir,
        lbc_interval_hours=1.0,
        era5_cache_dir=cache_dir,
        rim_width=4,
        era5_ds=ds_syn,
    )
    assert len(prep["lbc_paths"]) == 3
    assert all(Path(p).is_file() for p in prep["lbc_paths"])

    interp = LBCInterpolator(lbc_dir, spec.name)
    lbc_mid = interp.get_lbc_at_time(1800.0)
    assert "u" in lbc_mid and "v" in lbc_mid and "theta_p" in lbc_mid
    assert lbc_mid["theta_p"].shape == (spec.nz, spec.ny, spec.nx)
    assert np.all(np.isfinite(lbc_mid["theta_p"]))

    # Also test full Benelux+North Sea spec crossing 0E prime meridian (-3.5..10E)
    full_spec = get_domain_spec("lam_bnl_ns_de_2p5km", nz=16)
    wx1_wrap = fetch_era5_snapshot_wx1(t0, full_spec, tmp_path / "wrap_cache", era5_ds=ds_syn)
    with wx1.open_dataset(wx1_wrap) as wds:
        w_lons = np.asarray(wds["longitude"].values, dtype=np.float64)
        assert float(w_lons.min()) < 0.0 and float(w_lons.max()) > 5.0
        assert np.all(np.diff(w_lons) > 0.0)


def test_download_era5_land_soil_cds_requires_valid_client(tmp_path: Path, monkeypatch) -> None:
    import cdsapi

    class _FailingClient:
        def __init__(self):
            raise Exception("Missing ~/.cdsapirc")

    monkeypatch.setattr(cdsapi, "Client", _FailingClient)
    spec = _small_bnl_spec()
    with pytest.raises(RuntimeError, match="cdsapirc"):
        download_era5_land_soil_cds(
            datetime.datetime(2024, 1, 15, 0, 0, tzinfo=datetime.timezone.utc),
            spec,
            tmp_path / "cds_cache",
        )


def test_prepare_era5_lbc_series_streaming_and_cached_skip(tmp_path: Path) -> None:
    spec = _small_bnl_spec()
    hs_m = np.zeros((spec.ny, spec.nx), dtype=np.float64)
    t0 = datetime.datetime(2024, 1, 15, 0, 0, tzinfo=datetime.timezone.utc)
    t1 = datetime.datetime(2024, 1, 15, 1, 0, tzinfo=datetime.timezone.utc)
    t2 = datetime.datetime(2024, 1, 15, 2, 0, tzinfo=datetime.timezone.utc)
    ds_syn = _build_synthetic_arco_era5_ds([t0, t1, t2])

    lbc_dir = tmp_path / "lbc_stream"
    cache_dir = tmp_path / "era5_cache_stream"
    prep = prepare_era5_lbc_series(
        spec,
        hs_m,
        t0,
        forecast_hours=2.0,
        lbc_dir=lbc_dir,
        lbc_interval_hours=1.0,
        era5_cache_dir=cache_dir,
        rim_width=4,
        era5_ds=ds_syn,
        stream_lbc=True,
    )
    assert prep["streamed_lbc"] is True
    assert len(prep["lbc_paths"]) == 3
    assert len(prep["era5_wx1_paths"]) == 1  # only IC written to era5_cache
    assert list(cache_dir.glob("*.wx1")) == [prep["ic_wx1"]]

    interp = LBCInterpolator(lbc_dir, spec.name)
    lbc_mid = interp.get_lbc_at_time(3600.0)
    assert np.all(np.isfinite(lbc_mid["theta_p"]))

    # Re-running with era5_ds=None reuses cached IC + cached LBC WX-1s with zero remote access
    prep_cached = prepare_era5_lbc_series(
        spec,
        hs_m,
        t0,
        forecast_hours=2.0,
        lbc_dir=lbc_dir,
        lbc_interval_hours=1.0,
        era5_cache_dir=cache_dir,
        rim_width=4,
        era5_ds=None,
        stream_lbc=True,
    )
    assert prep_cached["lbc_paths"] == prep["lbc_paths"]


def test_select_era5_cases_ranking(tmp_path: Path) -> None:
    import gzip
    import sys

    scripts_dir = str(Path(__file__).resolve().parents[1] / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from select_era5_cases import rank_knmi_cases

    csv_gz = tmp_path / "knmi_uur_2025.csv.gz"
    rows = ["STN,YYYYMMDD,HH,FH,FX,T,RH,S,O,M"]
    for hh in range(1, 25):
        # 20250110: widespread radiation fog (M=1), light wind (FH=20 -> 2.0 m/s), dry
        rows.append(f"260,20250110,{hh},20.0,30.0,15.0,0.0,0.0,0.0,1.0")
        # 20250215: winter storm (FX=320 -> 32.0 m/s, FH=180 -> 18.0 m/s), wet
        rows.append(f"260,20250215,{hh},180.0,320.0,80.0,12.0,0.0,1.0,0.0")
        # 20250620: clear spring/summer high diurnal range (8 C .. 26 C = 18 C), completely dry
        t_val = 80.0 + (180.0 if 10 <= hh <= 16 else 0.0)
        rows.append(f"260,20250620,{hh},40.0,70.0,{t_val:.1f},0.0,0.0,0.0,0.0")
    with gzip.open(csv_gz, "wt", encoding="utf-8") as fh:
        fh.write("\n".join(rows) + "\n")

    ranked = rank_knmi_cases(tmp_path, regime="all", top=2, max_date="2026-11-12")
    assert ranked["radiation_fog"][0]["date"] == "2025-01-10"
    assert ranked["storm"][0]["date"] == "2025-02-15"
    assert ranked["clear_diurnal"][0]["date"] == "2025-06-20"
    assert ranked["rain"][0]["date"] == "2025-02-15"

