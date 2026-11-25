"""Tests for Class 14 Peat physics, ECMWF SMI soil-moisture remapping, WorldCover 10 m, SoilGrids 250 m, and Copernicus SSO pre-filtering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import wx1

from tempest_v2.cuda_bridge import NonHydrostaticGPUSolver
from tempest_v2.ic_builder import build_domain_static_fields
from tempest_v2.ingest_era5 import era5_wx1_to_domain_state
from tempest_v2.land_static import (
    WORLDCOVER_CLASS_PHYSICS,
    resolve_soilgrids_root,
    resolve_worldcover_root,
    sample_worldcover_on_grid,
)
from tempest_v2.projection import VerticalGrid, get_domain_spec
from tempest_v2.soil_model import (
    ECMWF_HTESSEL_SOIL_HYDRO,
    SOIL_PARAM_TABLE,
    NoahLandSurfaceState,
    remap_ecmwf_swvl_via_smi,
)


def test_soil_table_class14_peat_and_smi_remapping() -> None:
    assert SOIL_PARAM_TABLE.shape == (15, 8)
    np.testing.assert_allclose(
        SOIL_PARAM_TABLE[14],
        [0.766, 0.663, 0.267, 6.10, 0.103, 2.00e-6, 0.05, 0.55],
        rtol=1e-6,
    )

    lsm = NoahLandSurfaceState(1, 2, soil_type=np.array([[14, 13]], dtype=np.uint8), lsm=np.ones((1, 2)))
    assert float(lsm.theta_sat[0, 0]) == pytest.approx(0.766, abs=1e-5)
    assert float(lsm.theta_fc[0, 0]) == pytest.approx(0.663, abs=1e-5)
    assert float(lsm.soil_moisture[0, 0, 0]) > 0.55

    # Columns: [0:Sand(13), 1:Loam(9), 2:Clay(1), 3:Peat(14), 4:Water(0)]
    st_dst = np.array([[13, 9, 1, 14, 0]], dtype=np.float32)

    # Plant-available SMI preservation from HTESSEL Type 2 Medium (wp=0.151, fc=0.347)
    w_src = 0.249  # exact midpoint -> SMI = 0.50
    swvl_mid = np.full((4, 1, 5), w_src, dtype=np.float32)
    mapped_mid = remap_ecmwf_swvl_via_smi(swvl_mid, st_dst)
    assert mapped_mid.shape == (4, 1, 5)
    for col, cls_id in enumerate((13, 9, 1, 14)):
        wp_d = float(SOIL_PARAM_TABLE[cls_id, 2])
        fc_d = float(SOIL_PARAM_TABLE[cls_id, 1])
        expected = 0.5 * (wp_d + fc_d)
        assert float(mapped_mid[0, 0, col]) == pytest.approx(expected, abs=1e-4)

    # Peat column receiving wet ERA5 Organic Type 6 (swvl=0.70 > 0.52) stays > 0.65 (not truncated at 0.50!)
    swvl_peat = np.full((4, 1, 5), 0.70, dtype=np.float32)
    mapped_peat = remap_ecmwf_swvl_via_smi(swvl_peat, st_dst)
    assert float(mapped_peat[0, 0, 3]) > 0.65
    assert float(mapped_peat[0, 0, 0]) < float(SOIL_PARAM_TABLE[13, 0]) + 1e-6

    # Check explicit src_soil_type & 2D shape support
    mapped_2d = remap_ecmwf_swvl_via_smi(
        np.array([[0.347, 0.347]], dtype=np.float32),
        np.array([[13, 14]], dtype=np.float32),
        src_soil_type=np.array([[2, 2]], dtype=np.float32),
    )
    assert mapped_2d.shape == (1, 2)
    assert float(mapped_2d[0, 0]) == pytest.approx(float(SOIL_PARAM_TABLE[13, 1]), abs=1e-5)
    assert float(mapped_2d[0, 1]) == pytest.approx(float(SOIL_PARAM_TABLE[14, 1]), abs=1e-5)
    assert float(ECMWF_HTESSEL_SOIL_HYDRO[6, 2]) == pytest.approx(0.766, abs=1e-5)


def test_worldcover_blending_height_and_no_double_counting(tmp_path: Path) -> None:
    import rasterio
    from rasterio.transform import from_origin

    # Physical blending-height formula (z_b = 50 m): log-drag average lies strictly between geometric (0.173) and arithmetic (0.515) mean
    z_b = 50.0
    cd_grass = 1.0 / (np.log(z_b / WORLDCOVER_CLASS_PHYSICS[30][0]) ** 2)
    cd_forest = 1.0 / (np.log(z_b / WORLDCOVER_CLASS_PHYSICS[10][0]) ** 2)
    z0_mix = z_b * np.exp(-1.0 / np.sqrt(0.5 * (cd_grass + cd_forest)))
    assert 0.173 < z0_mix < 0.515

    # Smooth TOFD orographic drag: sigma <= 15 m adds 0, continuous at 15.01 m, increases at 200 m
    def _tot_z0(z0_veg: float, sigma: float) -> float:
        inv_v = 1.0 / (np.log(z_b / z0_veg) ** 2)
        sig_e = max(0.0, sigma - 15.0)
        inv_t = inv_v + 1.5 * ((sig_e / 1000.0) ** 2)
        return float(np.clip(z_b * np.exp(-1.0 / np.sqrt(inv_t)), 0.002, 1.5))

    assert _tot_z0(0.03, 0.0) == pytest.approx(0.03, abs=1e-6)
    assert _tot_z0(0.03, 15.0) == pytest.approx(0.03, abs=1e-6)
    assert abs(_tot_z0(0.03, 15.01) - _tot_z0(0.03, 15.0)) < 1e-6
    assert _tot_z0(0.03, 200.0) > _tot_z0(0.03, 15.0)

    # Synthetic 100x100 WorldCover GeoTIFF test: left half forest (10), right half grassland (30)
    wc_dir = tmp_path / "wc"
    wc_dir.mkdir()
    tile_path = wc_dir / "ESA_WorldCover_10m_2021_v200_N51E003_Map.tif"
    arr = np.full((100, 100), 30, dtype=np.uint8)
    arr[:, :50] = 10
    res_deg = 0.0005  # 50 m pixels -> spans lon [5.00, 5.05], lat [52.00, 52.05]
    tr = from_origin(5.00, 52.05, res_deg, res_deg)
    with rasterio.open(
        tile_path, "w", driver="GTiff", height=100, width=100, count=1, dtype="uint8", crs="EPSG:4326", transform=tr
    ) as dst:
        dst.write(arr, 1)

    spec = get_domain_spec("d3_nl_2p5km", nz=16)
    res = sample_worldcover_on_grid(spec, local_root=wc_dir, z_blend_m=50.0)
    assert res["wc_valid"].shape == (spec.ny, spec.nx)
    assert float(res["tree_frac"].max()) > 0.3
    assert float(res["dsm_height_offset_m"].max()) > 2.0


def test_real_d3_nl_static_fields_worldcover_soilgrids_copernicus() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    dem_dir = repo_root / "data/static/copernicus_dem_raw"
    if not dem_dir.exists() or resolve_worldcover_root() is None or resolve_soilgrids_root() is None:
        pytest.skip("Local Copernicus / ESA WorldCover / SoilGrids datasets not present")

    spec = get_domain_spec("d3_nl_2p5km", nz=16)
    hs_m, sstatic = build_domain_static_fields(spec, force=False)

    is_land = sstatic["lsm"] >= 0.5
    assert int(is_land.sum()) > 10_000

    with wx1.open_dataset(repo_root / "data/static/landcover_worldcover_d3_nl_2p5km.wx1") as wds:
        wc_valid = np.asarray(wds["wc_valid"].values) >= 0.5
        assert np.all(wc_valid[is_land])

    with wx1.open_dataset(repo_root / "data/static/soil_type_d3_nl_2p5km_soilgrids250m.wx1") as sds:
        sg_valid = np.asarray(sds["sg_valid"].values) >= 0.5
        assert np.all(sg_valid[is_land])

    with wx1.open_dataset(repo_root / "data/static/static_copernicus_d3_nl_2p5km.wx1") as cds:
        assert "sso_prefilter_m" in cds.attrs
        oro_std = np.asarray(cds["orography_std_m"].values, dtype=np.float32)

    stype_land = np.rint(sstatic["soil_type"][is_land]).astype(np.int32)
    peat_pct = 100.0 * float((stype_land == 14).sum()) / float(stype_land.size)
    assert 1.5 <= peat_pct <= 20.0, f"Unexpected peat class 14 share on d3_nl_2p5km: {peat_pct:.2f}%"

    # Check physical values at known Dutch landmarks
    def _nearest_ij(lat_t: float, lon_t: float) -> tuple[int, int]:
        dist2 = (spec.lat2d - lat_t) ** 2 + (spec.lon2d - lon_t) ** 2
        j, i = np.unravel_index(int(np.argmin(dist2)), spec.lat2d.shape)
        return int(j), int(i)

    j_zp, i_zp = _nearest_ij(52.00, 4.63)   # Zuidplaspolder
    j_vel, i_vel = _nearest_ij(52.18, 5.82) # Veluwe forest
    j_nh, i_nh = _nearest_ij(52.68, 4.95)   # Noord-Holland grassland/polder
    j_sea, i_sea = _nearest_ij(53.00, 3.00) # North Sea

    assert float(hs_m[j_zp, i_zp]) < -1.0
    assert float(oro_std[j_zp, i_zp]) < 3.0
    assert float(sstatic["z0m_base"][j_vel, i_vel]) > 0.35
    assert float(sstatic["z0m_base"][j_nh, i_nh]) < 0.12
    assert float(sstatic["z0m_base"][j_sea, i_sea]) == pytest.approx(1.0e-4, abs=1e-6)

    # If cached Eunice ERA5 snapshot is present, verify SMI remapping preserves peat vs sand moisture
    eunice_era5 = repo_root / "runs/era5_cache/era5_0p25_20220218_00Z.wx1"
    if eunice_era5.is_file():
        _, surf, _ = era5_wx1_to_domain_state(eunice_era5, spec, hs_m, surface_static=sstatic)
        peat_mask = is_land & (np.rint(sstatic["soil_type"]) == 14)
        sand_mask = is_land & (np.rint(sstatic["soil_type"]) == 13)
        assert float(np.median(surf["soil_m"][0][peat_mask])) > 0.55
        assert float(np.median(surf["soil_m"][0][sand_mask])) < 0.33
        assert float(np.percentile(surf["soil_m"][0][is_land], 1)) > 0.08
        assert float(np.min(surf["soil_m"][0][is_land])) >= 0.02


def test_cuda_class14_peat_soil_stepping() -> None:
    nz, ny, nx = 24, 4, 4
    vgrid = VerticalGrid(nz=nz, z_top_m=15_000.0, stretch_power=1.3)
    hs = np.zeros((ny, nx), dtype=np.float64)
    solver = NonHydrostaticGPUSolver(
        vgrid,
        dx_m=2500.0,
        dy_m=2500.0,
        hs_m=hs,
        has_moist=True,
        has_physics=True,
    )
    try:
        solver.set_physics_config(
            enable_microphysics=0,
            enable_pbl=1,
            enable_radiation=1,
            enable_surface=1,
            rad_interval_s=300.0,
            julian_day=172.0,
            utc_seconds=43_200.0,
            cy46_physics=1,
        )
        sstatic = solver.default_surface_static(
            lat_deg=52.10,
            lon_deg=4.85,
            lsm=1.0,
            soil_type=14.0,
            albedo_base=0.18,
            z0m_base=0.04,
            veg_frac=0.80,
        )
        solver.set_surface_static(sstatic)

        state = solver.empty_state()
        state["u"][:] = 5.0
        state["qv"][:] = 0.008
        solver.upload_state(state)

        surf = solver.empty_surface_state()
        surf["t_skin"].fill(289.0)
        surf["soil_t"][:].fill(287.0)
        surf["soil_m"][:].fill(0.68)
        solver.upload_surface(surf)

        solver.step_physics(dt_s=60.0, n_steps=20)
        st_out = solver.download_state()
        sf_out = solver.download_surface()

        assert all(np.all(np.isfinite(arr)) for arr in st_out.values())
        assert all(np.all(np.isfinite(arr)) for arr in sf_out.values())
        assert float(sf_out["soil_m"].mean()) > 0.55
    finally:
        solver.close()
