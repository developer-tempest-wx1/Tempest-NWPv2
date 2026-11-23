"""Unit tests for portable static geophysical downloader, ERDDAP bathymetry, and CUDA build config."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import wx1
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[1]
PY_ROOT = REPO_ROOT / "python"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import download_static_data as dl  # noqa: E402
from tempest_v2.bathymetry import prepare_bathymetry_on_2d_grid  # noqa: E402
from tempest_v2.hwsd22_soil import resolve_hwsd_root  # noqa: E402


def test_copernicus_tile_urls_and_bbox():
    dem_u, wbm_u, dem_f, wbm_f = dl.copernicus_tile_urls(52, 5, "glo30")
    assert dem_f == "Copernicus_DSM_COG_10_N52_00_E005_00_DEM.tif"
    assert wbm_f == "Copernicus_DSM_COG_10_N52_00_E005_00_WBM.tif"
    assert dem_u.startswith("https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com/")
    assert "/AUXFILES/Copernicus_DSM_COG_10_N52_00_E005_00_WBM.tif" in wbm_u

    dem_s, wbm_s, dem_sf, _ = dl.copernicus_tile_urls(-4, -73, "glo90")
    assert dem_sf == "Copernicus_DSM_COG_30_S04_00_W073_00_DEM.tif"
    assert "copernicus-dem-90m.s3.eu-central-1.amazonaws.com" in dem_s
    assert "/AUXFILES/" in wbm_s

    tiles = dl.tiles_for_bbox(50.8, 53.2, 3.1, 7.2)
    assert (50, 3) in tiles
    assert (53, 7) in tiles
    assert len(tiles) == 4 * 5


def test_domain_bbox_and_bathymetry_urls():
    lat0, lat1, lon0, lon1 = dl.bbox_from_domain("d3_nl_2p5km")
    assert 48.0 < lat0 < 51.0
    assert 53.0 < lat1 < 56.5
    assert 0.0 < lon0 < 5.0
    assert 7.0 < lon1 < 12.0

    erddap_u = dl.gebco_erddap_url(lat0, lat1, lon0, lon1)
    assert erddap_u.startswith("https://coastwatch.pfeg.noaa.gov/erddap/griddap/GEBCO_2020.nc?")
    assert "elevation" in erddap_u

    wcs_u = dl.emodnet_wcs_url(51.0, 52.0, 3.0, 4.0, width=512, height=512)
    assert wcs_u.startswith("https://ows.emodnet-bathymetry.eu/wcs?")
    assert "coverage=emodnet%3Amean" in wcs_u
    assert "bbox=3.0000%2C51.0000%2C4.0000%2C52.0000" in wcs_u


def test_prepare_bathymetry_erddap_latitude_longitude_descending(tmp_path):
    # ERDDAP subsets use (latitude, longitude) and sometimes descending latitude order
    lats = np.array([54.0, 53.0, 52.0, 51.0], dtype=np.float64)
    lons = np.array([2.0, 4.0, 6.0], dtype=np.float64)
    # Sea on west (lon=2, depth 30m -> elev=-30), land on east (lon=6, elev=+25)
    elev = np.array(
        [
            [-30.0, -10.0, 25.0],
            [-30.0, -10.0, 25.0],
            [-30.0, -10.0, 25.0],
            [-30.0, -10.0, 25.0],
        ],
        dtype=np.float64,
    )
    nc_path = tmp_path / "erddap_subset.nc"
    xr.Dataset(
        {"elevation": (("latitude", "longitude"), elev)},
        coords={"latitude": lats, "longitude": lons},
    ).to_netcdf(nc_path)

    lat2d = np.array([[52.5, 52.5]], dtype=np.float64)
    lon2d = np.array([[2.5, 5.5]], dtype=np.float64)
    out_wx1 = tmp_path / "bathy_erddap.wx1"
    prepare_bathymetry_on_2d_grid(nc_path, "erddap_test", lat2d, lon2d, vertical_datum="msl", output_path=out_wx1)

    with wx1.open_dataset(out_wx1) as ds:
        depth = np.asarray(ds["bathymetry_depth"].values)
        wet = np.asarray(ds["wet_mask"].values)
        assert depth.shape == (1, 2)
        assert depth[0, 0] > 20.0
        assert int(wet[0, 0]) == 1
        assert depth[0, 1] == 0.0
        assert int(wet[0, 1]) == 0


def test_env_var_portability_and_cli_dry_run(tmp_path, monkeypatch):
    monkeypatch.setenv("TEMPEST_HWSD_V22_DIR", str(tmp_path / "custom_hwsd"))
    assert resolve_hwsd_root() == tmp_path / "custom_hwsd"

    proc = subprocess.run(
        [sys.executable, str(SCRIPTS_ROOT / "download_static_data.py"), "--all", "--domain", "d3_nl_2p5km", "--dry-run"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Copernicus DEM" in proc.stdout
    assert "Bathymetry" in proc.stdout
    assert "FAO HWSD v2.2" in proc.stdout


def test_build_script_and_cuda_rtx50_support():
    build_rs = (REPO_ROOT / "crates" / "tempest-cuda" / "build.rs").read_text(encoding="utf-8")
    assert "sm_" in build_rs
    assert "120" in build_rs
    assert "nvidia-smi" in build_rs
    assert "CUDA Toolkit >= 12.8" in build_rs

    build_sh = REPO_ROOT / "scripts" / "build.sh"
    assert build_sh.is_file()
    assert os.access(build_sh, os.X_OK)
    proc = subprocess.run(["bash", str(build_sh), "--help"], capture_output=True, text=True, check=True)
    assert "--arch" in proc.stdout
    assert "sm_120" in proc.stdout
