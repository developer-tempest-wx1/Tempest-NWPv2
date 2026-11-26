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


# ---------------------------------------------------------------------------
# Public-release downloader, GEBCO 2026 picker & Apache-2.0 metadata tests
# ---------------------------------------------------------------------------


def test_copernicus_layer_presets_and_asset_urls() -> None:
    import sys
    from pathlib import Path

    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import download_static_data as dl

    assert dl.parse_copernicus_layers("minimal") == ("DEM", "WBM")
    assert dl.parse_copernicus_layers("standard") == ("DEM", "WBM", "EDM", "FLM", "HEM")
    assert dl.parse_copernicus_layers("all") == dl.COPERNICUS_LAYER_NAMES

    assets = dl.copernicus_tile_asset_urls(52, 5, "glo30", layers="all")
    by_rel = {rel: url for _, url, rel in assets}
    assert "DEM/Copernicus_DSM_COG_10_N52_00_E005_00_DEM.tif" in by_rel
    assert by_rel["WBM/Copernicus_DSM_COG_10_N52_00_E005_00_WBM.tif"].endswith("/AUXFILES/Copernicus_DSM_COG_10_N52_00_E005_00_WBM.tif")
    assert by_rel["EDM/Copernicus_DSM_COG_10_N52_00_E005_00_EDM.tif"].endswith("/AUXFILES/Copernicus_DSM_COG_10_N52_00_E005_00_EDM.tif")
    assert by_rel["FLM/Copernicus_DSM_COG_10_N52_00_E005_00_FLM.tif"].endswith("/AUXFILES/Copernicus_DSM_COG_10_N52_00_E005_00_FLM.tif")
    assert by_rel["HEM/Copernicus_DSM_COG_10_N52_00_E005_00_HEM.tif"].endswith("/AUXFILES/Copernicus_DSM_COG_10_N52_00_E005_00_HEM.tif")
    assert by_rel["ACM/Copernicus_DSM_10_N52_00_E005_00_ACM.kml"].endswith("/AUXFILES/Copernicus_DSM_10_N52_00_E005_00_ACM.kml")
    assert by_rel["XML/Copernicus_DSM_10_N52_00_E005_00.xml"].endswith("/Copernicus_DSM_10_N52_00_E005_00.xml")
    assert by_rel["PREVIEW/Copernicus_DSM_10_N52_00_E005_00_DEM_QL.tif"].endswith("/PREVIEW/Copernicus_DSM_10_N52_00_E005_00_DEM_QL.tif")
    assert by_rel["INFO/eula_F.pdf"].endswith("/INFO/eula_F.pdf")


def test_gebco_2026_urls_safe_zip_and_picker(tmp_path) -> None:
    import sys
    import zipfile
    from pathlib import Path
    import pytest

    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import download_static_data as dl
    from tempest_v2.ic_builder import _pick_gebco_elevation_nc

    assert dl.GEBCO_CEDA_2026_ZIP_URL.endswith("/ice_surface_elevation/netcdf/GEBCO_2026.zip")
    assert dl.GEBCO_CEDA_2026_SUB_ICE_ZIP_URL.endswith("/sub_ice_topography_bathymetry/netcdf/GEBCO_2026_sub_ice.zip")
    assert dl.GEBCO_CEDA_2026_TID_ZIP_URL.endswith("/type_identifier_grid/netcdf/gebco_2026_tid.zip")

    ok_zip = tmp_path / "ok.zip"
    with zipfile.ZipFile(ok_zip, "w") as zf:
        zf.writestr("nested/GEBCO_2026.nc", b"CDF-test-payload")
        zf.writestr("nested/ignore.txt", b"ignored")
    extracted = dl._safe_extract_zip(ok_zip, tmp_path / "out", suffixes=(".nc",))
    assert extracted == [(tmp_path / "out" / "GEBCO_2026.nc").resolve()]
    assert (tmp_path / "out" / "GEBCO_2026.nc").read_bytes() == b"CDF-test-payload"

    bad_zip = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad_zip, "w") as zf:
        zf.writestr("../escape.nc", b"evil")
    with pytest.raises(RuntimeError, match="Unsafe ZIP member path"):
        dl._safe_extract_zip(bad_zip, tmp_path / "out2", suffixes=(".nc",))

    tid_only = tmp_path / "tid_only"
    (tid_only / "gebco_2026_tid").mkdir(parents=True)
    (tid_only / "gebco_2026_tid" / "GEBCO_2026_TID.nc").write_bytes(b"tid")
    assert _pick_gebco_elevation_nc(tid_only) is None

    full_dir = tmp_path / "gebco_full"
    (full_dir / "gebco_2026_tid").mkdir(parents=True)
    (full_dir / "GEBCO_2026_sub_ice").mkdir(parents=True)
    (full_dir / "GEBCO_2026").mkdir(parents=True)
    (full_dir / "gebco_2026_tid" / "GEBCO_2026_TID.nc").write_bytes(b"tid")
    (full_dir / "GEBCO_2026_sub_ice" / "GEBCO_2026_sub_ice.nc").write_bytes(b"sub")
    (full_dir / "GEBCO_2026" / "GEBCO_2026.nc").write_bytes(b"ice")
    assert _pick_gebco_elevation_nc(full_dir) == full_dir / "GEBCO_2026" / "GEBCO_2026.nc"


def test_worldcover_geojson_fallback_and_urls(tmp_path) -> None:
    import json
    import sys
    from pathlib import Path

    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import download_static_data as dl

    url21, fn21 = dl.worldcover_tile_url("N51E003", year=2021)
    assert fn21 == "ESA_WorldCover_10m_2021_v200_N51E003_Map.tif"
    assert url21.endswith(f"/v200/2021/map/{fn21}")
    url20, fn20 = dl.worldcover_tile_url("N51E003", year=2020)
    assert url20.endswith(f"/v100/2020/map/{fn20}")

    fb = dl.worldcover_tiles_for_bbox_fallback(50.5, 53.7, 3.2, 7.3)
    assert "N48E003" in fb
    assert "N51E003" in fb
    assert "N51E006" in fb

    grid_path = tmp_path / "esa_worldcover_grid.geojson"
    grid_path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"ll_tile": "N51E003"},
                        "geometry": {"type": "Polygon", "coordinates": [[[3.0, 51.0], [6.0, 51.0], [6.0, 54.0], [3.0, 54.0], [3.0, 51.0]]]},
                    },
                    {
                        "type": "Feature",
                        "properties": {"ll_tile": "S30W060"},
                        "geometry": {"type": "Polygon", "coordinates": [[[-60.0, -30.0], [-57.0, -30.0], [-57.0, -27.0], [-60.0, -27.0], [-60.0, -30.0]]]},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    selected = dl.worldcover_tiles_from_geojson(51.5, 52.5, 4.0, 5.5, grid_cache=grid_path, allow_network=False)
    assert selected == ["N51E003"]


def test_soilgrids_jobs_wcs_url_and_xml_rejection(tmp_path) -> None:
    import sys
    from pathlib import Path
    import pytest

    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import download_static_data as dl

    assert len(dl.soilgrids_jobs("full", include_wrb=True)) == 55
    assert len(dl.soilgrids_jobs("tempest", include_wrb=True)) == 14
    assert len(dl.soilgrids_jobs("surface", include_wrb=True)) == 10
    assert len(dl.soilgrids_jobs("full", include_wrb=False)) == 54

    url_small = dl.soilgrids_wcs_url(51.0, 53.0, 4.0, 6.0, map_name="clay", coverage_id="clay_0-5cm_Q0.5")
    assert "map=/map/clay.map" in url_small
    assert "COVERAGEID=clay_0-5cm_Q0.5" in url_small
    assert "SCALESIZE=" not in url_small

    url_large = dl.soilgrids_wcs_url(35.0, 72.0, -15.0, 35.0, map_name="clay", coverage_id="clay_0-5cm_Q0.5", max_dim=4096)
    assert "SCALESIZE=X(" in url_large

    xml_src = tmp_path / "ogc_err.xml"
    xml_src.write_bytes(b'  <?xml version="1.0"?><ServiceExceptionReport>' + b"x" * 160 + b"</ServiceExceptionReport>")
    with pytest.raises(RuntimeError, match="XML/HTML"):
        dl._download_file(xml_src.as_uri(), tmp_path / "soil.tif", reject_xml=True, resume=False)


def test_fabdem_and_ahn_helpers() -> None:
    import sys
    from pathlib import Path
    import pytest

    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import download_static_data as dl

    u1, f1 = dl.fabdem_tile_url(52, 5)
    assert f1 == "N52E005_FABDEM_V1-2.tif"
    assert "/N50E000-N60E010_FABDEM_V1-2/N52E005_FABDEM_V1-2.tif" in u1

    u2, f2 = dl.fabdem_tile_url(-1, -161)
    assert f2 == "S01W161_FABDEM_V1-2.tif"
    assert "/S10W170-N00W160_FABDEM_V1-2/S01W161_FABDEM_V1-2.tif" in u2

    ahn_u = dl.ahn_wcs_url(51.5, 52.5, 4.5, 5.8, product="dtm_05m", resolution_m=5.0)
    assert "COVERAGEID=dtm_05m" in ahn_u
    assert "SCALESIZE=x(" in ahn_u

    with pytest.raises(ValueError, match="Netherlands"):
        dl.ahn_wcs_url(35.0, 40.0, -5.0, 0.0)


def test_cli_dry_run_new_flags(capsys) -> None:
    import sys
    from pathlib import Path

    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import download_static_data as dl

    dl.main([
        "--all",
        "--copernicus-layers",
        "all",
        "--bathy-source",
        "gebco-2026-full",
        "--soilgrids-mode",
        "full",
        "--domain",
        "d3_nl_2p5km",
        "--dry-run",
    ])
    out = capsys.readouterr().out
    assert "GEBCO_2026.zip" in out
    assert "GEBCO_2026_sub_ice.zip" in out
    assert "gebco_2026_tid.zip" in out
    assert "layers=DEM,WBM,EDM,FLM,HEM,ACM,XML,PREVIEW,INFO" in out
    assert "'layers': 55" in out
    assert "ESA WorldCover 2021" in out


def test_apache_license_and_public_metadata_consistency() -> None:
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    lic = (root / "LICENSE").read_text(encoding="utf-8")
    assert "Apache License" in lic
    assert "Version 2.0, January 2004" in lic

    cargo = tomllib.loads((root / "Cargo.toml").read_text(encoding="utf-8"))
    pyproj = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert cargo["workspace"]["package"]["license"] == "Apache-2.0"
    assert pyproj["project"]["license"]["text"] == "Apache-2.0"

    readme = (root / "README.md").read_text(encoding="utf-8")
    assert "/home/j_ord" not in readme
    assert "/mnt/d/" not in readme
    assert "Private-to-Public" not in readme
    assert "Apache License, Version 2.0" in readme
    assert "https://github.com/developer-tempest-wx1/Tempest-NWPv2" in readme
