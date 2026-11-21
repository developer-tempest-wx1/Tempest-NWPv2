"""Verification tests for Step 3 (Phase B0): LCC projection, Klemp vertical grid,
offline Copernicus DEM 30m/90m + WBM, HWSD v2.2, GEBCO bathymetry, LBC & Aurora adapter.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import wx1

from tempest_v2.aurora_adapter import aurora_wx1_to_domain_state
from tempest_v2.bathymetry import prepare_bathymetry_on_2d_grid
from tempest_v2.copernicus_dem import sample_copernicus_on_2d_grid, write_copernicus_static_wx1
from tempest_v2.hwsd22_soil import sample_hwsd22_on_grid
from tempest_v2.lbc_manager import LBCInterpolator, export_child_lbc_step
from tempest_v2.projection import (
    CP_D,
    GRAVITY_M_S2,
    LCCProjection,
    VerticalGrid,
    get_domain_spec,
)


def test_lcc_projection_inverse_and_wind_rotation():
    proj = LCCProjection()
    lats = np.array([[46.0, 51.0, 56.0], [48.0, 52.1, 54.0]], dtype=np.float64)
    lons = np.array([[0.5, 8.5, 16.5], [4.9, 5.18, 13.4]], dtype=np.float64)

    x_m, y_m = proj.latlon_to_xy(lats, lons)
    lat_rt, lon_rt = proj.xy_to_latlon(x_m, y_m)
    x_rt, y_rt = proj.latlon_to_xy(lat_rt, lon_rt)

    np.testing.assert_allclose(lat_rt, lats, atol=1e-10)
    np.testing.assert_allclose(lon_rt, lons, atol=1e-10)
    assert np.max(np.hypot(x_rt - x_m, y_rt - y_m)) < 1e-6

    # Map scale factor must equal 1.000000 on standard parallels (48 N, 54 N)
    np.testing.assert_allclose(proj.map_scale_factor(np.array([48.0, 54.0])), [1.0, 1.0], atol=1e-12)

    # Check tangent direction along parallel against wind rotation
    eps = 1e-5
    x1, y1 = proj.latlon_to_xy(52.0, 15.0 - eps)
    x2, y2 = proj.latlon_to_xy(52.0, 15.0 + eps)
    tx, ty = x2 - x1, y2 - y1
    norm = np.hypot(tx, ty)
    ug, vg = proj.geo_to_grid_wind(np.array([1.0]), np.array([0.0]), np.array([15.0]))
    np.testing.assert_allclose([ug[0], vg[0]], [tx / norm, ty / norm], atol=1e-7)

    u0 = np.array([[12.3, -4.5], [0.0, 18.7]], dtype=np.float64)
    v0 = np.array([[-3.1, 8.9], [5.4, -7.2]], dtype=np.float64)
    u_g, v_g = proj.geo_to_grid_wind(u0, v0, lons[:2, :2])
    u_back, v_back = proj.grid_to_geo_wind(u_g, v_g, lons[:2, :2])
    np.testing.assert_allclose(u_back, u0, atol=1e-12)
    np.testing.assert_allclose(v_back, v0, atol=1e-12)


def test_domain_presets_and_vertical_grid_137():
    d3 = get_domain_spec("d3_benelux_de_1p5km", nz=137)
    assert (d3.nz, d3.ny, d3.nx) == (137, 720, 720)
    assert float(d3.lat2d.min()) <= 46.5
    assert float(d3.lat2d.max()) >= 55.5
    assert float(d3.lon2d.min()) <= 1.2
    assert float(d3.lon2d.max()) >= 15.8

    # Jacobian strictly positive even at Mont Blanc 4810 m
    hs_extreme = np.array([[0.0, 2000.0, 4810.0]])
    g_min = d3.vertical.check_jacobian_positive(hs_extreme)
    assert g_min > 0.5
    assert 10.0 <= float(d3.vertical.dzeta[0]) <= 25.0

    # Analytic hydrostatic check d(pi_bar)/dz == -g / (cp * theta_bar)
    z = np.array([0.0, 1000.0, 10000.0, 30000.0])
    dz = 1e-1
    ref_p = VerticalGrid.reference_state(z + dz)
    ref_m = VerticalGrid.reference_state(z - dz)
    ref_0 = VerticalGrid.reference_state(z)
    dpi_dz_fd = (ref_p["pi_bar"] - ref_m["pi_bar"]) / (2.0 * dz)
    dpi_dz_exact = -GRAVITY_M_S2 / (CP_D * ref_0["theta_bar"])
    np.testing.assert_allclose(dpi_dz_fd, dpi_dz_exact, rtol=1e-8)


def test_offline_copernicus_dem_polders_lakes_mountains_and_wx1_write(tmp_path):
    dem_root = Path("/mnt/d/AI/weer-ai/Tempest/Tempest-NWPv2/data/static/copernicus_dem_raw")
    if not dem_root.exists():
        pytest.skip("Local Copernicus DEM directory not mounted")

    # Small patch covering Zuidplaspolder (52.00N, 4.63E), Bodensee Lake (47.60N, 9.50E, WBM==2),
    # North Sea (52.5N, 3.5E), Alpine/GLO-90 fallback edge (46.5N, 15.5E), Amstelmeer (52.77N, 5.06E, WBM==2), Zugspitze (47.42N, 10.98E)
    lat2d = np.array([
        [52.00, 47.60, 52.50],
        [46.50, 52.77, 47.42],
    ], dtype=np.float64)
    lon2d = np.array([
        [4.63, 9.50, 3.50],
        [15.50, 5.06, 10.98],
    ], dtype=np.float64)

    fields = sample_copernicus_on_2d_grid(
        lat2d, lon2d, dx_m=1500.0, dy_m=1500.0, local_root=dem_root, offline=True
    )
    # Dutch polder negative elevation preserved
    assert fields["orography_m"][0, 0] < -1.0
    assert fields["lsm_frac"][0, 0] > 0.8
    # Bodensee & Amstelmeer detected as inland lakes (WBM == 2)
    assert fields["lake_frac"][0, 1] > 0.7
    assert fields["lake_frac"][1, 1] > 0.01
    # Open North Sea cell strictly 0.0 m
    assert fields["orography_m"][0, 2] == 0.0
    assert fields["lsm_frac"][0, 2] == 0.0
    # Alps/Zugspitze high terrain + GLO-90 fallback tile populated
    assert fields["orography_m"][1, 2] > 1000.0
    assert fields["orography_m"][1, 0] > 50.0

    out_wx1 = tmp_path / "copernicus_patch.wx1"
    write_copernicus_static_wx1(
        out_wx1,
        fields,
        x_coords=np.arange(3, dtype=np.float64) * 1500.0,
        y_coords=np.arange(2, dtype=np.float64) * 1500.0,
        lat2d=lat2d,
        lon2d=lon2d,
        domain_name="test_patch",
    )
    with wx1.open_dataset(out_wx1) as ds:
        assert "orography_m" in ds.data_vars
        assert "lake_frac" in ds.data_vars
        assert "lat2d" in ds.coords
        np.testing.assert_array_equal(ds["orography_m"].values, fields["orography_m"])


def test_offline_hwsd22_and_gebco_bathymetry_2d_patch(tmp_path):
    hwsd_dir = Path("/mnt/d/AI/weer-ai/Tempest/Tempest-NWPv2/data/static/hwsd_v22/extracted/db/HWSD2.2.mdb")
    gebco_nc = Path("/mnt/d/AI/weer-ai/Tempest/Tempest-NWPv2/data/static/bathymetry_raw/GEBCO_2026/GEBCO_2026.nc")
    if not hwsd_dir.is_file() or not gebco_nc.is_file():
        pytest.skip("Local HWSD v2.2 or GEBCO_2026 not present")

    lat2d = np.array([[52.15, 52.5], [53.2, 53.6]], dtype=np.float64)
    lon2d = np.array([[5.40, 3.4], [5.8, 3.0]], dtype=np.float64)

    soil_wx1 = sample_hwsd22_on_grid(lat2d, lon2d, "test_soil_patch", output_path=tmp_path / "soil_type_patch.wx1")
    with wx1.open_file(soil_wx1) as f:
        stype = f.read_array("/soil_type")
        assert stype.shape == (2, 2)
        assert int(stype[0, 0]) > 0  # Veluwe / Amersfoort agricultural land soil class 1..13
        assert int(stype[0, 1]) == 0  # North Sea water/no-data class 0

    bathy_wx1 = prepare_bathymetry_on_2d_grid(
        gebco_nc, "test_bathy_patch", lat2d, lon2d, vertical_datum="msl", output_path=tmp_path / "bathy_patch.wx1"
    )
    with wx1.open_dataset(bathy_wx1) as ds:
        depth = np.asarray(ds["bathymetry_depth"].values)
        assert depth[0, 1] > 10.0  # North Sea depth positive downward [m]
        assert depth[0, 0] == 0.0  # Land cell depth == 0.0


def test_aurora_adapter_and_lbc_interpolation_roundtrip(tmp_path):
    # Create tiny synthetic Aurora .wx1 file over Europe [42..60 N, -5..22 E]
    levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
    lats = np.linspace(42.0, 60.0, 10, dtype=np.float32)
    lons = np.linspace(-5.0, 22.0, 12, dtype=np.float32)
    ny_a, nx_a = len(lats), len(lons)

    data_vars = {
        "msl": (("time", "lat", "lon"), np.full((1, ny_a, nx_a), 101_325.0, dtype=np.float32)),
        "2t": (("time", "lat", "lon"), np.full((1, ny_a, nx_a), 285.0, dtype=np.float32)),
    }
    for lev in levels:
        t_val = 285.0 * ((lev / 1000.0) ** 0.2)
        data_vars[f"t_{lev}"] = (("time", "lat", "lon"), np.full((1, ny_a, nx_a), t_val, dtype=np.float32))
        data_vars[f"u_{lev}"] = (("time", "lat", "lon"), np.full((1, ny_a, nx_a), 10.0, dtype=np.float32))
        data_vars[f"v_{lev}"] = (("time", "lat", "lon"), np.full((1, ny_a, nx_a), 0.0, dtype=np.float32))
        data_vars[f"q_{lev}"] = (("time", "lat", "lon"), np.full((1, ny_a, nx_a), 5e-3 * (lev / 1000.0), dtype=np.float32))

    aurora_path = tmp_path / "aurora_test.wx1"
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": ("time", np.array(["2026-11-21T00:00:00"], dtype="datetime64[s]")),
            "lat": ("lat", lats),
            "lon": ("lon", lons),
        },
    )
    wx1.write_dataset(aurora_path, ds, dataset_id="aurora-synthetic")

    # Test mapping onto a small cropped child + parent LCC pair
    d2_small = get_domain_spec("d2_eu_lcc", nz=65)
    hs_d2 = np.zeros((d2_small.ny, d2_small.nx), dtype=np.float32)
    st_d2 = aurora_wx1_to_domain_state(str(aurora_path), d2_small, hs_d2)
    assert st_d2["theta_p"].shape == (65, d2_small.ny, d2_small.nx)
    assert np.all(np.isfinite(st_d2["theta_p"]))
    assert float(st_d2["qv"].min()) >= 0.0

    # Test LBC export & interpolation
    d3_small = get_domain_spec("d3_benelux_de_1p5km", nz=65)
    # Create tiny slice by mutating shape only in a local test spec
    from dataclasses import replace
    d3_mini = replace(
        d3_small,
        ny=16, nx=16,
        x_m=d3_small.x_m[:16], y_m=d3_small.y_m[:16],
        lat2d=d3_small.lat2d[:16, :16], lon2d=d3_small.lon2d[:16, :16],
        map_factor=d3_small.map_factor[:16, :16], coriolis_f=d3_small.coriolis_f[:16, :16],
    )
    hs_mini = np.zeros((16, 16), dtype=np.float32)
    t0 = datetime.datetime(2026, 11, 21, 0, 0, tzinfo=datetime.timezone.utc)
    lbc_dir = tmp_path / "lbc"
    export_child_lbc_step(lbc_dir, 0, 0.0, t0, d2_small, st_d2, hs_d2, d3_mini, hs_mini)
    st_d2_t1 = {k: (v + 1.0 if k == "u" else v) for k, v in st_d2.items()}
    export_child_lbc_step(lbc_dir, 1, 3600.0, t0 + datetime.timedelta(hours=1), d2_small, st_d2_t1, hs_d2, d3_mini, hs_mini)

    itp = LBCInterpolator(lbc_dir, d3_mini.name)
    mid = itp.get_lbc_at_time(1800.0)
    low = itp.get_lbc_at_time(0.0)
    np.testing.assert_allclose(mid["u"], low["u"] + 0.5, atol=1e-5)
