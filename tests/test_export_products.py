"""Tests for WX-1 post-processing exporter (geopy resolution, CSV, Cartopy maps, MetPy Skew-T & GIF animations)."""

from __future__ import annotations

import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import List

import numpy as np
import pandas as pd
from PIL import Image
import xarray as xr
import wx1

from tempest_v2.export_products import (
    discover_wx1_forecasts,
    export_wx1_products,
    resolve_locations,
)


def _write_synthetic_wx1(
    path: Path,
    lead_hr: float,
    *,
    global_0_360: bool = False,
) -> Path:
    if global_0_360:
        lats_1d = np.linspace(45.0, 58.0, 27, dtype=np.float32)
        lons_1d = np.arange(0.0, 360.0, 2.5, dtype=np.float32)
    else:
        lats_1d = np.linspace(49.5, 54.0, 24, dtype=np.float32)
        lons_1d = np.linspace(2.5, 8.5, 28, dtype=np.float32)

    lon2d, lat2d = np.meshgrid(lons_1d, lats_1d)
    ny, nx = lat2d.shape
    plev_hpa = np.array([1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50], dtype=np.float32)
    plev_pa = plev_hpa * 100.0
    nlev = len(plev_pa)

    v_dt = datetime.datetime(2026, 11, 22, 0, 0, tzinfo=datetime.timezone.utc) + datetime.timedelta(hours=float(lead_hr))
    t_coord = np.array([np.datetime64(int(v_dt.timestamp()), "s")], dtype="datetime64[s]")

    # Realistic temperature gradient & baroclinic 3D sounding profile
    t2m = (283.15 + 1.5 * np.sin(lead_hr * 0.4) - 0.3 * (lat2d - 51.5) + 0.2 * np.cos(np.radians(lon2d))).astype(np.float32)
    td2m = (t2m - 3.5).astype(np.float32)
    mslp = (101325.0 + 400.0 * np.sin(np.radians(lon2d) * 2.0 + lead_hr * 0.2)).astype(np.float32)
    u10 = (6.0 + 2.0 * np.cos(np.radians(lat2d) * 3.0)).astype(np.float32)
    v10 = (3.5 + 1.5 * np.sin(np.radians(lon2d) * 2.0)).astype(np.float32)
    gust = (np.hypot(u10, v10) * 1.45).astype(np.float32)
    rain_acc = np.full((ny, nx), 1.5 * lead_hr, dtype=np.float32)
    precip_rate = np.full((ny, nx), 1.5 / 3600.0 if lead_hr > 0 else 0.0, dtype=np.float32)
    cape = np.full((ny, nx), 320.0 + 40.0 * lead_hr, dtype=np.float32)
    shear = np.full((ny, nx), 16.0 + 0.5 * lead_hr, dtype=np.float32)

    t_pl = np.empty((nlev, ny, nx), dtype=np.float32)
    gh_pl = np.empty((nlev, ny, nx), dtype=np.float32)
    q_pl = np.empty((nlev, ny, nx), dtype=np.float32)
    rh_pl = np.empty((nlev, ny, nx), dtype=np.float32)
    u_pl = np.empty((nlev, ny, nx), dtype=np.float32)
    v_pl = np.empty((nlev, ny, nx), dtype=np.float32)

    for k, ph in enumerate(plev_hpa):
        # Standard hydrostatic temperature decay with altitude and moist boundary layer
        t_k = t2m - 48.0 * (1.0 - (ph / 1000.0) ** 0.286) * 2.2
        t_pl[k] = np.maximum(t_k, 208.0).astype(np.float32)
        gh_pl[k] = (8000.0 * np.log(1013.25 / ph)).astype(np.float32)
        rh_val = float(np.clip(82.0 - 0.05 * (1000.0 - ph), 25.0, 90.0))
        rh_pl[k] = np.full((ny, nx), rh_val, dtype=np.float32)
        es = 611.2 * np.exp(17.67 * (t_pl[k] - 273.15) / (t_pl[k] - 29.65))
        e_v = 0.01 * rh_val * es
        q_pl[k] = np.clip(0.622 * e_v / (ph * 100.0), 1e-6, 0.03).astype(np.float32)
        u_pl[k] = (u10 + 25.0 * (1.0 - ph / 1000.0)).astype(np.float32)
        v_pl[k] = (v10 + 10.0 * np.sin((1.0 - ph / 1000.0) * np.pi)).astype(np.float32)

    ds = xr.Dataset(
        data_vars={
            "t2m": (("y", "x"), t2m),
            "td2m": (("y", "x"), td2m),
            "rh2m": (("y", "x"), np.full((ny, nx), 78.0, dtype=np.float32)),
            "q2m": (("y", "x"), q_pl[0]),
            "u10m_geo": (("y", "x"), u10),
            "v10m_geo": (("y", "x"), v10),
            "gust10m": (("y", "x"), gust),
            "mslp": (("y", "x"), mslp),
            "ps": (("y", "x"), mslp),
            "rain_acc": (("y", "x"), rain_acc),
            "snow_acc": (("y", "x"), np.zeros((ny, nx), dtype=np.float32)),
            "graupel_acc": (("y", "x"), np.zeros((ny, nx), dtype=np.float32)),
            "precip_rate": (("y", "x"), precip_rate),
            "tcc": (("y", "x"), np.full((ny, nx), 0.65, dtype=np.float32)),
            "cape_sb": (("y", "x"), cape),
            "cin_sb": (("y", "x"), np.full((ny, nx), 15.0, dtype=np.float32)),
            "shear_0_6km": (("y", "x"), shear),
            "freezing_level_m": (("y", "x"), np.full((ny, nx), 1850.0, dtype=np.float32)),
            "lat": (("y", "x"), lat2d),
            "lon": (("y", "x"), lon2d),
            "t_pl": (("plev", "y", "x"), t_pl),
            "gh_pl": (("plev", "y", "x"), gh_pl),
            "q_pl": (("plev", "y", "x"), q_pl),
            "rh_pl": (("plev", "y", "x"), rh_pl),
            "u_pl": (("plev", "y", "x"), u_pl),
            "v_pl": (("plev", "y", "x"), v_pl),
        },
        coords={
            "time": ("time", t_coord),
            "plev": ("plev", plev_pa),
            "y": ("y", np.arange(ny, dtype=np.float64)),
            "x": ("x", np.arange(nx, dtype=np.float64)),
        },
        attrs={
            "domain_name": "d1_global_0p25" if global_0_360 else "d3_benelux_de_1p5km",
            "lead_hours": f"{lead_hr:.3f}",
            "valid_time_utc": v_dt.isoformat(),
        },
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    wx1.write_dataset(path, ds, dataset_id=f"test_{int(lead_hr)}", codec="zstd_shuffle", workers=2)
    return path


def test_resolve_locations_geopy_cache_coords_and_builtin_fallback(tmp_path: Path) -> None:
    calls: List[str] = []

    class _MockGeocoder:
        def geocode(self, q: str):
            calls.append(q)
            if "BredaCustom" in q:
                return SimpleNamespace(latitude=51.5866, longitude=4.7760, address="Breda, Noord-Brabant, NL")
            return None

    cache_file = tmp_path / "geo_cache.json"
    pts = resolve_locations(
        ["BredaCustom", "De Bilt", "Noordzee:52.5,3.5"],
        use_geopy=True,
        cache_path=cache_file,
        geocoder=_MockGeocoder(),
    )
    assert len(pts) == 3
    assert pts[0].source == "geopy" and abs(pts[0].lat - 51.5866) < 1e-4
    assert pts[1].source == "builtin_fallback" and abs(pts[1].lat - 52.10) < 1e-3
    assert pts[2].source == "coordinates" and abs(pts[2].lon - 3.5) < 1e-4
    assert cache_file.is_file()

    # Second call for BredaCustom should hit JSON cache instead of geocoder
    calls.clear()
    pts_cached = resolve_locations(
        ["BredaCustom"],
        use_geopy=True,
        cache_path=cache_file,
        geocoder=_MockGeocoder(),
    )
    assert calls == []
    assert pts_cached[0].source == "cache"


def test_export_wx1_products_csv_maps_meteogram_skewt_and_gifs(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "20261122_00Z"
    d3_dir = run_dir / "d3"
    for hr in (0.0, 3.0, 6.0):
        _write_synthetic_wx1(d3_dir / f"forecast_d3_benelux_de_1p5km_T+{int(hr):03d}h.wx1", hr)

    # Empty older/newer folder should be ignored when discovering newest valid run
    (tmp_path / "runs" / "20261122_12Z" / "d3").mkdir(parents=True, exist_ok=True)
    files, dom, _ = discover_wx1_forecasts(tmp_path / "runs", domain="auto")
    assert dom == "d3" and len(files) == 3

    out_dir = tmp_path / "exports"
    summary = export_wx1_products(
        run_dir,
        ["De Bilt", "Tilburg:51.5555,5.0913"],
        domain="d3",
        outdir=out_dir,
        use_geopy=False,
        map_types=["t2m_mslp", "precip_wind", "cape_shear", "z500_t850"],
        dpi=90,
        fps=3.0,
        show_progress=False,
    )

    assert len(summary["csv_files"]) == 5  # 2 ts + 2 sounding + all_locations
    ts_df = pd.read_csv(out_dir / "csv" / "tilburg_timeseries.csv")
    assert len(ts_df) == 3
    assert np.allclose(ts_df["t2m_c"], ts_df["t2m_k"] - 273.15, atol=1e-4)
    assert list(ts_df["precip_step_mm"]) == [0.0, 4.5, 4.5]

    snd_df = pd.read_csv(out_dir / "csv" / "de_bilt_sounding_profiles.csv")
    assert len(snd_df) == 3 * 13
    assert np.all(snd_df["td_c"] <= snd_df["t_c"])

    # Meteograms + 4 map types (3 frames each) + Skew-T (3 frames per point) + looping GIFs
    assert len(summary["meteogram_files"]) == 2
    for mtype in ("t2m_mslp", "precip_wind", "cape_shear", "z500_t850"):
        assert len(summary["map_files"][mtype]) == 3
        gif_p = out_dir / "animations" / f"{mtype}.gif"
        assert gif_p.is_file()
        with Image.open(gif_p) as im:
            assert im.n_frames == 3

    for slug in ("de_bilt", "tilburg"):
        assert len(summary["skewt_files"][slug]) == 3
        sk_gif = out_dir / "animations" / f"skewt_{slug}.gif"
        assert sk_gif.is_file()
        with Image.open(sk_gif) as im:
            assert im.n_frames == 3


def test_export_wx1_global_0_360_prime_meridian_handling(tmp_path: Path) -> None:
    d1_dir = tmp_path / "run_d1" / "d1"
    for hr in (0.0, 6.0):
        _write_synthetic_wx1(d1_dir / f"forecast_d1_global_0p25_T+{int(hr):03d}h.wx1", hr, global_0_360=True)

    out_dir = tmp_path / "exports_d1"
    summary = export_wx1_products(
        d1_dir,
        ["London", "De Bilt"],
        domain="d1",
        outdir=out_dir,
        use_geopy=False,
        map_types=["t2m_mslp"],
        dpi=90,
        show_progress=False,
    )
    df_lon = pd.read_csv(out_dir / "csv" / "london_timeseries.csv")
    assert np.all(np.isfinite(df_lon["t2m_c"]))
    assert float(df_lon["grid_dist_km"].iloc[0]) < 200.0
    assert (out_dir / "animations" / "t2m_mslp.gif").is_file()
    assert len(summary["skewt_files"]["london"]) == 2
