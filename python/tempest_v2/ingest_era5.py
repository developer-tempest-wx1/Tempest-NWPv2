"""Deterministic ARCO-ERA5 & ERA5-Land Hindcast Ingestor and WX-1 Builder for Tempest-NWP v2.

Strictly 100% Classical Atmospheric & Land-Surface Preprocessing (0% AI / 0% ML), SI Units Only:
- Slices public Google Cloud ARCO-ERA5 (`full_37-1h-0p25deg-chunk-1.zarr-v3`, anonymous access)
  over the target domain bounding box with clean `0..360` prime-meridian longitude wrapping.
- Optionally queries Copernicus CDS (`reanalysis-era5-land`, ~9 km) via `cdsapi` for high-resolution
  land skin/soil temperature, volumetric soil moisture, and snow water equivalent.
- Caches lossless `zstd_shuffle` WX-1 regional snapshots, constructs balanced C-grid initial states,
  and writes rim-packed LBC `.wx1` series directly consumable by `LBCInterpolator`.
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import xarray as xr

import wx1

from tempest_v2.ecmwf_lbc import _cached_lbc_wx1_matches, _extract_rim_packed_lbc
from tempest_v2.ic_builder import (
    _empty_surface_state_for_spec,
    _fill_nan_nearest,
    _q_from_dewpoint_pa,
    _read_cfgrib_var_map,
    _regrid_latlon_field,
    _vertical_map_isobaric_to_klemp,
)
from tempest_v2.projection import GRAVITY_M_S2, R_D, DomainSpec

logger = logging.getLogger("tempest_v2.ingest_era5")

ARCO_ERA5_ZARR_URL = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

_ARCO_3D_VAR_MAP: Tuple[Tuple[str, str, str], ...] = (
    ("geopotential", "z", "m**2 s**-2"),
    ("temperature", "t", "K"),
    ("specific_humidity", "q", "kg kg**-1"),
    ("u_component_of_wind", "u", "m s**-1"),
    ("v_component_of_wind", "v", "m s**-1"),
)

_ARCO_2D_BASE_MAP: Tuple[Tuple[str, str, str], ...] = (
    ("mean_sea_level_pressure", "msl", "Pa"),
    ("surface_pressure", "sp", "Pa"),
    ("2m_temperature", "t2m", "K"),
    ("2m_dewpoint_temperature", "d2m", "K"),
    ("10m_u_component_of_wind", "u10", "m s**-1"),
    ("10m_v_component_of_wind", "v10", "m s**-1"),
    ("skin_temperature", "skt", "K"),
    ("sea_surface_temperature", "sst", "K"),
    ("sea_ice_cover", "siconc", "1"),
    ("snow_depth", "sd", "m"),
    ("land_sea_mask", "lsm", "1"),
    ("surface_solar_radiation_downwards", "ssrd", "J m**-2"),
)

_ARCO_2D_SOIL_MAP: Tuple[Tuple[str, str, str], ...] = (
    ("soil_temperature_level_1", "stl1", "K"),
    ("soil_temperature_level_2", "stl2", "K"),
    ("soil_temperature_level_3", "stl3", "K"),
    ("soil_temperature_level_4", "stl4", "K"),
    ("volumetric_soil_water_layer_1", "swvl1", "m**3 m**-3"),
    ("volumetric_soil_water_layer_2", "swvl2", "m**3 m**-3"),
    ("volumetric_soil_water_layer_3", "swvl3", "m**3 m**-3"),
    ("volumetric_soil_water_layer_4", "swvl4", "m**3 m**-3"),
)


def _to_utc(dt: datetime.datetime) -> datetime.datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=datetime.timezone.utc)
    return dt.astimezone(datetime.timezone.utc)


def _compute_domain_bbox(domain_spec: DomainSpec, pad_deg: float) -> Tuple[float, float, float, float]:
    """Return `(lat_min, lat_max, lon_min, lon_max)` in `[-180, 180)` with padding."""
    lat2d = np.asarray(domain_spec.lat2d, dtype=np.float64)
    lon_180 = ((np.asarray(domain_spec.lon2d, dtype=np.float64) + 180.0) % 360.0) - 180.0
    lat_min = max(-90.0, float(np.min(lat2d)) - float(pad_deg))
    lat_max = min(90.0, float(np.max(lat2d)) + float(pad_deg))
    if domain_spec.is_periodic_lon or (float(np.max(lon_180)) - float(np.min(lon_180))) > 300.0:
        return lat_min, lat_max, -180.0, 180.0
    lon_min = max(-180.0, float(np.min(lon_180)) - float(pad_deg))
    lon_max = min(180.0, float(np.max(lon_180)) + float(pad_deg))
    return lat_min, lat_max, lon_min, lon_max


def open_arco_era5_dataset(zarr_url: str = ARCO_ERA5_ZARR_URL) -> xr.Dataset:
    """Open public Google Cloud ARCO-ERA5 1-hourly 0.25-deg Zarr store anonymously."""
    logger.info("Opening public ARCO-ERA5 Zarr store (%s)...", zarr_url)
    return xr.open_zarr(zarr_url, chunks=None, storage_options={"token": "anon"})


def _extract_lon_wrapped_subset(
    ds_t: xr.Dataset,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> xr.Dataset:
    """Slice `(latitude, longitude)` sub-domain handling descending latitudes and `0..360` prime-meridian wrap."""
    lats_all = np.asarray(ds_t["latitude"].values, dtype=np.float64)
    lat_sel = lats_all[(lats_all >= lat_min - 1e-6) & (lats_all <= lat_max + 1e-6)]
    if lat_sel.size < 2:
        raise ValueError(f"ERA5 dataset contains <2 latitude points inside [{lat_min:.2f}, {lat_max:.2f}]")
    # Standardize descending latitude order [north -> south]
    lat_desc = np.sort(lat_sel)[::-1]
    ds_lat = ds_t.sel(latitude=lat_desc)

    lons_all = np.asarray(ds_lat["longitude"].values, dtype=np.float64)
    uses_360 = float(np.min(lons_all)) >= -1e-6 and float(np.max(lons_all)) > 180.0
    is_global_lon = (lon_max - lon_min) >= 359.0

    if is_global_lon:
        if uses_360:
            lons_180 = ((lons_all + 180.0) % 360.0) - 180.0
            order = np.argsort(lons_180)
            out = ds_lat.isel(longitude=order)
            return out.assign_coords(longitude=lons_180[order])
        return ds_lat.sel(longitude=np.sort(lons_all))

    if uses_360:
        if lon_min < 0.0 and lon_max >= 0.0:
            west = lons_all[lons_all >= (360.0 + lon_min - 1e-6)]
            east = lons_all[lons_all <= (lon_max + 1e-6)]
            parts = []
            if west.size > 0:
                ds_w = ds_lat.sel(longitude=np.sort(west))
                ds_w = ds_w.assign_coords(longitude=np.sort(west) - 360.0)
                parts.append(ds_w)
            if east.size > 0:
                parts.append(ds_lat.sel(longitude=np.sort(east)))
            if len(parts) == 2:
                return xr.concat(parts, dim="longitude")
            if len(parts) == 1:
                return parts[0]
            raise ValueError(f"No ERA5 longitudes matched [{lon_min:.2f}, {lon_max:.2f}]")
        if lon_max < 0.0:
            sel_360 = np.sort(lons_all[(lons_all >= 360.0 + lon_min - 1e-6) & (lons_all <= 360.0 + lon_max + 1e-6)])
            return ds_lat.sel(longitude=sel_360).assign_coords(longitude=sel_360 - 360.0)
        sel_pos = np.sort(lons_all[(lons_all >= lon_min - 1e-6) & (lons_all <= lon_max + 1e-6)])
        return ds_lat.sel(longitude=sel_pos)

    # Input already in [-180, 180]
    sel_180 = np.sort(lons_all[(lons_all >= lon_min - 1e-6) & (lons_all <= lon_max + 1e-6)])
    if sel_180.size < 2:
        raise ValueError(f"ERA5 dataset contains <2 longitude points inside [{lon_min:.2f}, {lon_max:.2f}]")
    return ds_lat.sel(longitude=sel_180)


def _slice_era5_bbox(
    ds: xr.Dataset,
    domain_spec: DomainSpec,
    valid_dt: datetime.datetime,
    *,
    pad_deg: float = 2.5,
    include_soil: bool = True,
) -> xr.Dataset:
    """Extract a validated, compact regional ERA5 `xr.Dataset` at `valid_dt` with short standard names."""
    v_dt = _to_utc(valid_dt)
    t_target = np.datetime64(v_dt.replace(tzinfo=None), "s")

    var_pairs: List[Tuple[str, str, str]] = list(_ARCO_3D_VAR_MAP) + list(_ARCO_2D_BASE_MAP)
    if include_soil:
        var_pairs.extend(_ARCO_2D_SOIL_MAP)

    # Support datasets already keyed by short names (e.g. unit fixtures) or ARCO long names
    need_vars: List[Tuple[str, str, str]] = []
    for long_name, short_name, units in var_pairs:
        if long_name in ds.data_vars:
            need_vars.append((long_name, short_name, units))
        elif short_name in ds.data_vars:
            need_vars.append((short_name, short_name, units))

    ds_vars_only = ds[[src for src, _, _ in need_vars]]
    ds_t = ds_vars_only.sel(time=t_target, method="nearest")
    sel_ts_ns = np.datetime64(ds_t["time"].values, "ns")
    sel_sec = int(sel_ts_ns.astype("datetime64[s]").astype(np.int64))
    if abs(sel_sec - int(v_dt.timestamp())) > 1800:
        sel_iso = datetime.datetime.fromtimestamp(sel_sec, tz=datetime.timezone.utc).isoformat()
        raise ValueError(
            f"Requested ERA5 valid time {v_dt.isoformat()} does not match nearest available timestamp {sel_iso}"
        )

    lat_min, lat_max, lon_min, lon_max = _compute_domain_bbox(domain_spec, pad_deg=pad_deg)
    ds_sub = _extract_lon_wrapped_subset(ds_t, lat_min, lat_max, lon_min, lon_max)

    # Sort pressure levels descending [surface/1000 hPa -> TOA/1 hPa] cleanly
    levels = np.asarray(ds_sub["level"].values, dtype=np.float64)
    lev_desc = np.sort(levels)[::-1]
    ds_sub = ds_sub.sel(level=lev_desc)

    ds_loaded = ds_sub.load()
    data_vars: Dict[str, Any] = {}
    for src_name, short_name, units in need_vars:
        arr = np.asarray(ds_loaded[src_name].values, dtype=np.float32)
        dims = tuple(str(d) for d in ds_loaded[src_name].dims if d != "time")
        while arr.ndim > len(dims):
            arr = arr[0]
        data_vars[short_name] = (dims, arr, {"units": units})

    for req_2d in ("t2m", "sp"):
        if req_2d not in data_vars and req_2d == "sp" and "msl" in data_vars:
            continue
        if req_2d not in data_vars:
            raise ValueError(f"ERA5 snapshot missing required surface variable '{req_2d}'")
    for req_3d in ("t", "u", "v", "q", "z"):
        if req_3d not in data_vars:
            raise ValueError(f"ERA5 snapshot missing required pressure-level variable '{req_3d}'")

    # ARCO-ERA5 time coordinates run to 2050 with NaNs beyond the latest ingested archive date
    t2m_vals = data_vars["t2m"][1]
    t3d_vals = data_vars["t"][1]
    if not np.any(np.isfinite(t2m_vals)) or not np.any(np.isfinite(t3d_vals)):
        raise ValueError(
            f"ERA5 data at {v_dt.isoformat()} is unpopulated (all NaN); choose an archived ERA5 timestamp."
        )

    coords = {
        "level": ("level", np.asarray(ds_loaded["level"].values, dtype=np.float32), {"units": "hPa"}),
        "latitude": ("latitude", np.asarray(ds_loaded["latitude"].values, dtype=np.float64), {"units": "degrees_north"}),
        "longitude": ("longitude", np.asarray(ds_loaded["longitude"].values, dtype=np.float64), {"units": "degrees_east"}),
    }
    attrs = {
        "source": "ARCO-ERA5",
        "valid_time_utc": v_dt.isoformat(),
        "resol": "0p25",
        "lat_min": f"{float(coords['latitude'][1].min()):.3f}",
        "lat_max": f"{float(coords['latitude'][1].max()):.3f}",
        "lon_min": f"{float(coords['longitude'][1].min()):.3f}",
        "lon_max": f"{float(coords['longitude'][1].max()):.3f}",
    }
    return xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)


def _era5_wx1_cache_valid(
    path: Path,
    domain_spec: DomainSpec,
    valid_dt: datetime.datetime,
    *,
    require_soil: bool,
) -> bool:
    if not path.is_file() or path.stat().st_size < 1024:
        return False
    try:
        with wx1.open_dataset(path) as ds:
            if ds.attrs.get("valid_time_utc") != _to_utc(valid_dt).isoformat():
                return False
            for k in ("t", "u", "v", "q", "z", "t2m", "msl"):
                if k not in ds:
                    return False
            if require_soil and ("stl1" not in ds or "swvl1" not in ds):
                return False
            lats = np.asarray(ds["latitude"].values, dtype=np.float64)
            lons = np.asarray(ds["longitude"].values, dtype=np.float64)
            lat_min_d = float(np.min(domain_spec.lat2d))
            lat_max_d = float(np.max(domain_spec.lat2d))
            if float(lats.min()) > lat_min_d + 0.3 or float(lats.max()) < lat_max_d - 0.3:
                return False
            lon_180 = ((np.asarray(domain_spec.lon2d, dtype=np.float64) + 180.0) % 360.0) - 180.0
            if not domain_spec.is_periodic_lon and (float(lon_180.max()) - float(lon_180.min())) < 300.0:
                if float(lons.min()) > float(lon_180.min()) + 0.3 or float(lons.max()) < float(lon_180.max()) - 0.3:
                    return False
        return True
    except Exception:
        return False


def fetch_era5_snapshot_wx1(
    valid_time_utc: datetime.datetime,
    domain_spec: DomainSpec,
    cache_dir: Path | str,
    *,
    zarr_url: str = ARCO_ERA5_ZARR_URL,
    pad_deg: float = 2.5,
    include_soil: bool = True,
    era5_ds: Optional[xr.Dataset] = None,
) -> Path:
    """Download one ARCO-ERA5 regional snapshot (if not cached) and write lossless `zstd_shuffle` WX-1."""
    v_dt = _to_utc(valid_time_utc)
    cdir = Path(cache_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    out_path = cdir / f"era5_0p25_{v_dt:%Y%m%d_%H}Z.wx1"
    if _era5_wx1_cache_valid(out_path, domain_spec, v_dt, require_soil=include_soil):
        logger.info("Reusing cached ERA5 snapshot %s", out_path.name)
        return out_path

    ds_src = era5_ds if era5_ds is not None else open_arco_era5_dataset(zarr_url=zarr_url)
    ds_reg = _slice_era5_bbox(ds_src, domain_spec, v_dt, pad_deg=pad_deg, include_soil=include_soil)
    chunks_map: Dict[str, List[int]] = {}
    for vn, da in ds_reg.data_vars.items():
        if da.ndim == 3:
            chunks_map[str(vn)] = [int(da.shape[0]), int(da.shape[1]), int(da.shape[2])]
        elif da.ndim == 2:
            chunks_map[str(vn)] = [int(da.shape[0]), int(da.shape[1])]

    wx1.write_dataset(
        out_path,
        ds_reg,
        dataset_id=f"era5_0p25_{v_dt:%Y%m%d_%H}Z",
        codec="zstd_shuffle",
        chunk_shapes=chunks_map,
        workers=4,
    )
    logger.info("Saved ERA5 WX-1 snapshot %s (%d levels, %dx%d)", out_path.name, int(ds_reg.sizes["level"]), int(ds_reg.sizes["latitude"]), int(ds_reg.sizes["longitude"]))
    return out_path


def download_era5_land_soil_cds(
    valid_time_utc: datetime.datetime,
    domain_spec: DomainSpec,
    cache_dir: Path | str,
    *,
    pad_deg: float = 2.0,
) -> Path:
    """Download 9 km ERA5-Land skin/soil/snow fields from Copernicus CDS into a cached WX-1 file."""
    v_dt = _to_utc(valid_time_utc)
    cdir = Path(cache_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    out_path = cdir / f"era5_land_0p09_{v_dt:%Y%m%d_%H}Z.wx1"
    if out_path.is_file() and out_path.stat().st_size > 1024:
        return out_path

    try:
        import cdsapi
    except ImportError as exc:
        raise RuntimeError("`cdsapi` package is required for 9 km ERA5-Land soil download.") from exc

    lat_min, lat_max, lon_min, lon_max = _compute_domain_bbox(domain_spec, pad_deg=pad_deg)
    raw_nc = cdir / f"_raw_era5_land_{v_dt:%Y%m%d_%H}Z.nc"
    req = {
        "variable": [
            "skin_temperature",
            "soil_temperature_level_1",
            "soil_temperature_level_2",
            "soil_temperature_level_3",
            "soil_temperature_level_4",
            "volumetric_soil_water_layer_1",
            "volumetric_soil_water_layer_2",
            "volumetric_soil_water_layer_3",
            "volumetric_soil_water_layer_4",
            "snow_depth",
        ],
        "year": f"{v_dt.year:04d}",
        "month": f"{v_dt.month:02d}",
        "day": f"{v_dt.day:02d}",
        "time": f"{v_dt.hour:02d}:00",
        "area": [lat_max, lon_min, lat_min, lon_max],
        "data_format": "netcdf",
    }
    try:
        client = cdsapi.Client()
        client.retrieve("reanalysis-era5-land", req, str(raw_nc))
    except Exception as exc:
        raise RuntimeError(
            f"Copernicus CDS ERA5-Land download failed ({exc}). "
            "Configure ~/.cdsapirc (url + key for https://cds.climate.copernicus.eu/api) "
            "or run without --era5-land-soil to use ARCO-ERA5 soil layers."
        ) from exc

    try:
        with xr.open_dataset(raw_nc) as ds_raw:
            t_dim = next((c for c in ("valid_time", "time") if c in ds_raw.dims or c in ds_raw.coords), None)
            if t_dim is not None and t_dim in ds_raw.dims:
                ds_raw = ds_raw.isel({t_dim: 0})
            lat_name = next(c for c in ("latitude", "lat") if c in ds_raw.coords)
            lon_name = next(c for c in ("longitude", "lon") if c in ds_raw.coords)
            data_vars = {
                str(vn): (
                    ("latitude", "longitude"),
                    np.asarray(ds_raw[vn].values, dtype=np.float32),
                    {k: str(v) for k, v in ds_raw[vn].attrs.items() if k in ("units", "long_name")},
                )
                for vn in ds_raw.data_vars
                if ds_raw[vn].ndim == 2
            }
            ds_wx = xr.Dataset(
                data_vars=data_vars,
                coords={
                    "latitude": ("latitude", np.asarray(ds_raw[lat_name].values, dtype=np.float64)),
                    "longitude": ("longitude", np.asarray(ds_raw[lon_name].values, dtype=np.float64)),
                },
                attrs={"source": "CDS ERA5-Land 0.09deg", "valid_time_utc": v_dt.isoformat()},
            )
            wx1.write_dataset(out_path, ds_wx, dataset_id=f"era5_land_{v_dt:%Y%m%d_%H}Z", codec="zstd_shuffle")
    finally:
        if raw_nc.exists():
            raw_nc.unlink()
    return out_path


def _era5_wx1_to_thermo_grid(
    era5_wx1_path: Path | str,
    spec: DomainSpec,
    hs_m: np.ndarray,
    *,
    theta_sfc_ref: float = 288.15,
    p_sfc_ref: float = 101_325.0,
    n_ref_hz: float = 0.01,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any], datetime.datetime]:
    """Decode ERA5 `.wx1` snapshot onto target `spec` 3D C-grid state plus regridded surface dict."""
    with wx1.open_dataset(Path(era5_wx1_path)) as ds:
        raw_v = str(ds.attrs.get("valid_time_utc", "")).strip()
        valid_dt = (
            _to_utc(datetime.datetime.fromisoformat(raw_v.replace("Z", "+00:00")))
            if raw_v
            else datetime.datetime.now(datetime.timezone.utc).replace(minute=0, second=0, microsecond=0)
        )
        p_levels_pa = np.asarray(ds["level"].values, dtype=np.float64) * 100.0
        lats = np.asarray(ds["latitude"].values, dtype=np.float64)
        lons = np.asarray(ds["longitude"].values, dtype=np.float64)

        def _arr3(name: str) -> np.ndarray:
            arr = np.asarray(ds[name].values, dtype=np.float64)
            while arr.ndim > 3:
                arr = arr[0]
            return arr

        t_h = _regrid_latlon_field(_arr3("t"), lats, lons, spec, fill_nan_value=260.0)
        u_geo_h = _regrid_latlon_field(_arr3("u"), lats, lons, spec, fill_nan_value=0.0)
        v_geo_h = _regrid_latlon_field(_arr3("v"), lats, lons, spec, fill_nan_value=0.0)
        q_h = np.clip(_regrid_latlon_field(_arr3("q"), lats, lons, spec, fill_nan_value=1e-4), 1e-7, 0.04)
        z_h = _regrid_latlon_field(_arr3("z") / GRAVITY_M_S2, lats, lons, spec, fill_nan_value=0.0)

        def _arr2(name: str) -> Optional[np.ndarray]:
            if name not in ds:
                return None
            arr = np.asarray(ds[name].values, dtype=np.float64)
            while arr.ndim > 2:
                arr = arr[0]
            return arr

        t2m_raw = _arr2("t2m")
        msl_raw = _arr2("msl")
        sp_raw = _arr2("sp")
        t2m_h = _regrid_latlon_field(t2m_raw, lats, lons, spec, fill_nan_value=285.0) if t2m_raw is not None else np.full((spec.ny, spec.nx), 285.0, dtype=np.float64)
        msl_h = (
            _regrid_latlon_field(msl_raw, lats, lons, spec, fill_nan_value=101_325.0)
            if msl_raw is not None
            else _regrid_latlon_field(sp_raw, lats, lons, spec, fill_nan_value=101_325.0)
        )

        hs = np.asarray(hs_m, dtype=np.float64)
        ps_child = msl_h * np.exp(-(GRAVITY_M_S2 * hs) / (R_D * np.maximum(t2m_h, 200.0)))

        state_3d = _vertical_map_isobaric_to_klemp(
            spec,
            hs,
            p_levels_pa,
            z_h,
            t_h,
            u_geo_h,
            v_geo_h,
            q_h,
            ps_child,
            theta_sfc_ref=theta_sfc_ref,
            p_sfc_ref=p_sfc_ref,
            n_ref_hz=n_ref_hz,
        )

        sfc_raw_bundle: Dict[str, Any] = {
            "lats": lats,
            "lons": lons,
            "t2m_h": t2m_h,
            "msl_h": msl_h,
            "ps_child": ps_child,
        }
        for k in ("d2m", "u10", "v10", "skt", "sst", "siconc", "sd", "lsm", "stl1", "stl2", "stl3", "stl4", "swvl1", "swvl2", "swvl3", "swvl4"):
            v2 = _arr2(k)
            if v2 is not None:
                sfc_raw_bundle[k] = v2

    return state_3d, sfc_raw_bundle, valid_dt


def _load_era5_land_soil_maps(path: Path | str) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Read 2D `(skt, stl1..4, swvl1..4, sd)` arrays from an ERA5-Land `.wx1`, `.nc`, or `.grib2` file."""
    p = Path(path)
    out: Dict[str, np.ndarray] = {}
    if p.suffix.lower() in (".grib", ".grib2", ".grb"):
        vmap = _read_cfgrib_var_map(p, use_cache=False)
        any_key = next(iter(vmap))
        _, lats, lons = vmap[any_key]
        for k, (da, _, _) in vmap.items():
            arr = np.asarray(da.values, dtype=np.float64)
            while arr.ndim > 2:
                arr = arr[0]
            out[str(k)] = arr
        return out, np.asarray(lats, dtype=np.float64), np.asarray(lons, dtype=np.float64)

    ds = wx1.open_dataset(p) if p.suffix.lower() == ".wx1" else xr.open_dataset(p)
    try:
        lat_k = next(c for c in ("latitude", "lat") if c in ds.coords)
        lon_k = next(c for c in ("longitude", "lon") if c in ds.coords)
        lats = np.asarray(ds[lat_k].values, dtype=np.float64)
        lons = np.asarray(ds[lon_k].values, dtype=np.float64)
        for vn in ds.data_vars:
            arr = np.asarray(ds[vn].values, dtype=np.float64)
            while arr.ndim > 2:
                arr = arr[0]
            out[str(vn)] = arr
    finally:
        ds.close()
    return out, lats, lons


def era5_wx1_to_domain_state(
    era5_wx1_path: Path | str,
    spec: DomainSpec,
    hs_m: np.ndarray,
    *,
    era5_land_soil_path: Optional[Path | str] = None,
    surface_static: Optional[Dict[str, np.ndarray]] = None,
    theta_sfc_ref: float = 288.15,
    p_sfc_ref: float = 101_325.0,
    n_ref_hz: float = 0.01,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], datetime.datetime]:
    """Build balanced C-grid 3D + 4-layer surface IC from an ARCO-ERA5 WX-1 snapshot (+ optional ERA5-Land)."""
    state_3d, sb, valid_dt = _era5_wx1_to_thermo_grid(
        era5_wx1_path,
        spec,
        hs_m,
        theta_sfc_ref=theta_sfc_ref,
        p_sfc_ref=p_sfc_ref,
        n_ref_hz=n_ref_hz,
    )
    lats, lons = sb["lats"], sb["lons"]
    t2m_h = sb["t2m_h"]
    ps_child = sb["ps_child"]

    surf = _empty_surface_state_for_spec(spec)
    surf["t2m"] = np.clip(t2m_h, 210.0, 330.0).astype(np.float32)

    if "d2m" in sb:
        d2m_h = _regrid_latlon_field(sb["d2m"], lats, lons, spec, fill_nan_value=float(np.mean(t2m_h)))
        surf["q2m"] = _q_from_dewpoint_pa(d2m_h, ps_child, t_k=t2m_h).astype(np.float32)
    else:
        surf["q2m"] = np.clip(state_3d["qv"][0], 1e-7, 0.04).astype(np.float32)

    if "u10" in sb and "v10" in sb:
        u10_geo = _regrid_latlon_field(sb["u10"], lats, lons, spec, fill_nan_value=0.0)
        v10_geo = _regrid_latlon_field(sb["v10"], lats, lons, spec, fill_nan_value=0.0)
        if spec.is_projected and spec.projection is not None:
            u10_grid, v10_grid = spec.projection.geo_to_grid_wind(u10_geo, v10_geo, spec.lon2d)
        else:
            u10_grid, v10_grid = u10_geo, v10_geo
        surf["u10m"] = np.clip(u10_grid, -100.0, 100.0).astype(np.float32)
        surf["v10m"] = np.clip(v10_grid, -100.0, 100.0).astype(np.float32)

    t2m_mean = float(np.nanmean(t2m_h))
    skt_h = (
        _regrid_latlon_field(sb["skt"], lats, lons, spec, fill_nan_value=t2m_mean)
        if "skt" in sb
        else t2m_h
    )
    surf["t_skin"] = np.clip(skt_h, 210.0, 330.0).astype(np.float32)

    water_fb = max(t2m_mean, 278.0)
    if "sst" in sb and np.any(np.isfinite(sb["sst"])):
        sst_raw = _fill_nan_nearest(sb["sst"], water_fb)
        sst_water_h = _regrid_latlon_field(sst_raw, lats, lons, spec, fill_nan_value=water_fb)
    elif "skt" in sb and "lsm" in sb:
        skt_water = np.where(sb["lsm"] < 0.5, sb["skt"], np.nan)
        sst_water_h = _regrid_latlon_field(_fill_nan_nearest(skt_water, water_fb), lats, lons, spec, fill_nan_value=water_fb)
    else:
        sst_water_h = skt_h
    surf["sst_bulk"] = np.clip(sst_water_h, 271.15, 310.0).astype(np.float32)

    if surface_static is not None and "lsm" in surface_static:
        lsm_child = np.asarray(surface_static["lsm"], dtype=np.float64)
        is_ocean = (lsm_child < 0.5) & (np.asarray(surface_static.get("lake_mask", 0.0)) < 0.5)
        surf["t_skin"] = np.where(is_ocean, surf["sst_bulk"], surf["t_skin"]).astype(np.float32)
    elif "lsm" in sb:
        lsm_child = _regrid_latlon_field(sb["lsm"], lats, lons, spec, fill_nan_value=0.5)
    else:
        lsm_child = np.ones((spec.ny, spec.nx), dtype=np.float64)
    is_land = lsm_child >= 0.5

    if "siconc" in sb:
        sic_raw = np.where(np.isfinite(sb["siconc"]), sb["siconc"], 0.0)
        surf["sea_ice_frac"] = np.clip(_regrid_latlon_field(sic_raw, lats, lons, spec, fill_nan_value=0.0), 0.0, 1.0).astype(np.float32)

    if "sd" in sb:
        # ERA5 snow depth is meters of water equivalent -> kg/m^2 = mm
        sd_mm = np.clip(np.where(np.isfinite(sb["sd"]), sb["sd"], 0.0), 0.0, 2.0) * 1000.0
        surf["snow_swe"] = np.clip(_regrid_latlon_field(sd_mm, lats, lons, spec, fill_nan_value=0.0), 0.0, 2000.0).astype(np.float32)

    for m in range(4):
        tk = f"stl{m + 1}"
        mk = f"swvl{m + 1}"
        if tk in sb:
            arr_t = np.where((sb[tk] < 150.0) | (sb[tk] > 360.0), np.nan, sb[tk])
            surf["soil_t"][m] = np.clip(_regrid_latlon_field(arr_t, lats, lons, spec, fill_nan_value=t2m_mean), 210.0, 340.0).astype(np.float32)
        else:
            surf["soil_t"][m] = surf["t_skin"]
        if mk in sb:
            arr_m = np.where((sb[mk] < 0.0) | (sb[mk] > 0.9), np.nan, sb[mk])
            surf["soil_m"][m] = np.clip(_regrid_latlon_field(arr_m, lats, lons, spec, fill_nan_value=0.28), 0.03, 0.50).astype(np.float32)

    if era5_land_soil_path is not None:
        lmap, l_lats, l_lons = _load_era5_land_soil_maps(era5_land_soil_path)

        def _regrid_land(aliases: Tuple[str, ...], fallback: float, lo: float, hi: float) -> Optional[np.ndarray]:
            for ak in aliases:
                if ak in lmap:
                    return np.clip(_regrid_latlon_field(lmap[ak], l_lats, l_lons, spec, fill_nan_value=fallback), lo, hi)
            return None

        skt_el = _regrid_land(("skt", "skin_temperature"), t2m_mean, 210.0, 340.0)
        if skt_el is not None:
            surf["t_skin"] = np.where(is_land, skt_el, surf["t_skin"]).astype(np.float32)
        for m in range(4):
            idx = m + 1
            st_el = _regrid_land((f"stl{idx}", f"soil_temperature_level_{idx}"), t2m_mean, 210.0, 340.0)
            if st_el is not None:
                surf["soil_t"][m] = np.where(is_land, st_el, surf["soil_t"][m]).astype(np.float32)
            sm_el = _regrid_land((f"swvl{idx}", f"volumetric_soil_water_layer_{idx}"), 0.28, 0.03, 0.50)
            if sm_el is not None:
                surf["soil_m"][m] = np.where(is_land, sm_el, surf["soil_m"][m]).astype(np.float32)
        sd_el = _regrid_land(("sd", "snow_depth"), 0.0, 0.0, 2.0)
        if sd_el is not None:
            surf["snow_swe"] = np.where(is_land, sd_el * 1000.0, surf["snow_swe"]).astype(np.float32)

    surf["lake_tbot"] = np.clip(skt_h - 1.5, 273.15, 305.0).astype(np.float32)
    if surface_static is not None and "bathy_depth_m" in surface_static:
        d = np.asarray(surface_static["bathy_depth_m"], dtype=np.float32)
        surf["lake_hmld"] = np.clip(0.6 * np.where(d > 0.5, d, 4.5), 0.2, 20.0).astype(np.float32)

    return state_3d, surf, valid_dt


def era5_wx1_to_lbc_wx1(
    era5_wx1_path: Path | str,
    child_spec: DomainSpec,
    child_hs_m: np.ndarray,
    output_dir: Path | str,
    step_index: int,
    sim_time_sec: float,
    valid_time_utc: datetime.datetime,
    *,
    rim_width: int = 16,
    theta_sfc_ref: float = 288.15,
    p_sfc_ref: float = 101_325.0,
    n_ref_hz: float = 0.01,
) -> Path:
    """Convert one ARCO-ERA5 WX-1 regional snapshot into a rim-packed Tempest WX-1 LBC file."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"lbc_{child_spec.name}_step_{int(step_index):04d}.wx1"

    state_3d, sb, _ = _era5_wx1_to_thermo_grid(
        era5_wx1_path,
        child_spec,
        child_hs_m,
        theta_sfc_ref=theta_sfc_ref,
        p_sfc_ref=p_sfc_ref,
        n_ref_hz=n_ref_hz,
    )
    data_vars, extra_coords, rim_info = _extract_rim_packed_lbc(child_spec, state_3d, sb["ps_child"], rim_width)

    v_dt = _to_utc(valid_time_utc)
    t_coord = np.array([np.datetime64(int(v_dt.timestamp()), "s")], dtype="datetime64[s]")
    coords: Dict[str, Any] = {
        "time": ("time", t_coord),
        "z": ("z", child_spec.vertical.zeta_s.astype(np.float32), {"units": "m"}),
        "z_w": ("z_w", child_spec.vertical.zeta_w.astype(np.float32), {"units": "m"}),
    }
    for ck, cv in extra_coords.items():
        coords[ck] = (ck, cv)

    attrs: Dict[str, str] = {
        "parent_domain": "era5_0p25",
        "child_domain": child_spec.name,
        "step_index": str(int(step_index)),
        "sim_time_sec": f"{float(sim_time_sec):.3f}",
        "valid_time_utc": v_dt.isoformat(),
        "era5_resol": "0p25",
        "child_nz": str(int(child_spec.nz)),
        "child_ny": str(int(child_spec.ny)),
        "child_nx": str(int(child_spec.nx)),
    }
    if rim_info["use_rim"]:
        attrs["lbc_storage"] = "rim_packed"
        attrs["lbc_rim_width"] = str(int(rim_info["rim_width"]))
        chunks_map = {
            str(vn): [min(int(v[1].shape[0]), 64), min(int(v[1].shape[1]), 65536)]
            for vn, v in data_vars.items()
            if v[1].ndim == 2
        }
    else:
        kz_chunk = max(1, min(int(child_spec.nz), (16 * 1024 * 1024) // max(1, child_spec.ny * child_spec.nx * 4)))
        chunks_map = {
            str(vn): [min(int(v[1].shape[0]), kz_chunk), int(v[1].shape[1]), int(v[1].shape[2])]
            for vn, v in data_vars.items()
            if v[1].ndim == 3
        }

    ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
    wx1.write_dataset(
        out_path,
        ds,
        dataset_id=f"lbc_{child_spec.name}_{int(step_index):04d}",
        codec="zstd_shuffle",
        chunk_shapes=chunks_map,
        workers=4,
    )
    return out_path


def prepare_era5_lbc_series(
    child_spec: DomainSpec,
    child_hs_m: np.ndarray,
    init_utc: datetime.datetime,
    forecast_hours: float,
    lbc_dir: Path | str,
    *,
    lbc_interval_hours: float = 1.0,
    era5_cache_dir: Optional[Path | str] = None,
    zarr_url: str = ARCO_ERA5_ZARR_URL,
    rim_width: int = 16,
    era5_ds: Optional[xr.Dataset] = None,
) -> Dict[str, Any]:
    """Fetch/cache ARCO-ERA5 regional snapshots across `[init_utc, +forecast_hours]` and build LBC WX-1s."""
    init_dt = _to_utc(init_utc)
    if float(lbc_interval_hours) <= 0.0:
        raise ValueError(f"lbc_interval_hours must be positive, got {lbc_interval_hours}")

    f_hrs = max(0.0, float(forecast_hours))
    step_hrs: List[float] = [0.0]
    cur = float(lbc_interval_hours)
    while cur < f_hrs - 1e-6:
        step_hrs.append(cur)
        cur += float(lbc_interval_hours)
    if f_hrs > 1e-6:
        step_hrs.append(f_hrs)

    out_lbc_dir = Path(lbc_dir)
    out_lbc_dir.mkdir(parents=True, exist_ok=True)
    eff_cache = Path(era5_cache_dir) if era5_cache_dir is not None else (out_lbc_dir.parent / "era5_cache")
    eff_cache.mkdir(parents=True, exist_ok=True)

    opened_ds = era5_ds
    lbc_paths: List[Path] = []
    era5_paths: List[Path] = []
    valid_times_utc: List[str] = []

    for idx, lead_h in enumerate(step_hrs):
        v_dt = init_dt + datetime.timedelta(hours=float(lead_h))
        sim_sec = float(lead_h) * 3600.0
        include_soil = idx == 0
        era5_p = eff_cache / f"era5_0p25_{v_dt:%Y%m%d_%H}Z.wx1"
        lbc_p = out_lbc_dir / f"lbc_{child_spec.name}_step_{idx:04d}.wx1"

        if not _era5_wx1_cache_valid(era5_p, child_spec, v_dt, require_soil=include_soil):
            if opened_ds is None:
                opened_ds = open_arco_era5_dataset(zarr_url=zarr_url)
            era5_p = fetch_era5_snapshot_wx1(
                v_dt,
                child_spec,
                eff_cache,
                zarr_url=zarr_url,
                include_soil=include_soil,
                era5_ds=opened_ds,
            )

        if not _cached_lbc_wx1_matches(lbc_p, child_spec, v_dt, sim_sec):
            lbc_p = era5_wx1_to_lbc_wx1(
                era5_p,
                child_spec,
                child_hs_m,
                out_lbc_dir,
                idx,
                sim_sec,
                v_dt,
                rim_width=rim_width,
            )

        era5_paths.append(era5_p)
        lbc_paths.append(lbc_p)
        valid_times_utc.append(v_dt.isoformat())

    return {
        "ic_wx1": era5_paths[0],
        "lbc_paths": lbc_paths,
        "era5_wx1_paths": era5_paths,
        "valid_times_utc": valid_times_utc,
        "lbc_interval_hours": float(lbc_interval_hours),
    }
