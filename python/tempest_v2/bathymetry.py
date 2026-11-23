"""Bathymetry preparation and strict vertical-datum handling for Tempest-NWP.

Inputs are user-downloaded GEBCO/EMODnet GeoTIFF rasters. Copernicus DEM is a
land-terrain product and is deliberately rejected as an ocean bathymetry source.
Output depths are positive downward, metres, with land represented by 0 and a
separate wet mask. EMODnet LAT/ALAT data are never silently mixed with MSL data:
a caller must supply a spatial MSL-minus-chart-datum correction raster.

This module prepares static bathymetry; it does not turn the current ocean
prototype into a validated NEMO/SI3 system.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import wx1

logger = logging.getLogger("tempest_v2.bathymetry")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BATHY_ROOT = PROJECT_ROOT / "data" / "static" / "bathymetry"


def domain_coordinates(domain: str, d3_resolution_m: Optional[int] = None):
    name = domain.lower()
    if "d1" in name or "global" in name:
        return (np.linspace(90.0, -90.0, 451, dtype=np.float64),
                np.arange(900, dtype=np.float64) * 0.4)
    if "d2" in name or "europe" in name:
        return (np.linspace(72.0, 30.0, 421, dtype=np.float64),
                np.linspace(-25.0, 45.0, 701, dtype=np.float64))
    if "d3" in name or "benelux" in name:
        from tempest_v2.static_fields import get_benelux_grid_coords
        return get_benelux_grid_coords(resolution_m=d3_resolution_m)
    raise ValueError(f"Unsupported bathymetry domain: {domain}")


def _target_transform(lats: np.ndarray, lons_sorted: np.ndarray):
    from affine import Affine
    if len(lats) < 2 or len(lons_sorted) < 2:
        raise ValueError("Target grid needs at least 2 latitude and longitude coordinates")
    dy = float(np.median(np.diff(lats)))
    dx = float(np.median(np.diff(lons_sorted)))
    if not np.allclose(np.diff(lats), dy, rtol=0, atol=1e-7):
        raise ValueError("Target latitude coordinates are not regularly spaced")
    if not np.allclose(np.diff(lons_sorted), dx, rtol=0, atol=1e-7):
        raise ValueError("Target longitude coordinates are not regularly spaced")
    # Raster row 0 and column 0 correspond to the first target cell centre.
    return Affine(dx, 0.0, float(lons_sorted[0]) - dx / 2.0,
                  0.0, dy, float(lats[0]) - dy / 2.0)


def prepare_bathymetry(
    source_path: str | Path,
    domain: str,
    source_name: str,
    source_values: str,
    vertical_datum: str,
    vertical_offset_raster: str | Path | None = None,
    d3_resolution_m: Optional[int] = None,
    output_path: str | Path | None = None,
) -> Path:
    """Reproject source GeoTIFF to a Tempest domain grid and write WX-1 fields.

    `source_values` must be `elevation_m` (negative below datum) or
    `positive_depth_m`. For chart-datum input, the correction raster must hold
    MSL-minus-chart-datum metres on the same source grid; it is reprojected
    using the same target grid before conversion to depth.
    """
    source_path = Path(source_path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if source_path.suffix.lower() not in {".tif", ".tiff", ".vrt"}:
        raise ValueError("Bathymetry importer currently accepts GeoTIFF/VRT only; convert NetCDF with GDAL first")
    if "copernicus" in source_path.name.lower() or ".ocean" in source_path.name.lower():
        raise ValueError("Copernicus DEM and tiny .ocean sidecar/sentinel files are not bathymetry sources")
    if source_values not in {"elevation_m", "positive_depth_m"}:
        raise ValueError("source_values must be elevation_m or positive_depth_m")
    datum = vertical_datum.lower()
    if datum not in {"msl", "lat", "alat", "unknown"}:
        raise ValueError("vertical_datum must be msl, lat, alat, or unknown")
    if datum in {"lat", "alat"} and vertical_offset_raster is None:
        raise ValueError("EMODnet LAT/ALAT heights must not be mixed into MSL; provide a spatial MSL-minus-LAT/ALAT correction GeoTIFF")

    try:
        import rasterio
        from rasterio.warp import reproject, Resampling
    except Exception as exc:
        raise RuntimeError("Bathymetry preparation requires rasterio/GDAL. Install rasterio in the active environment.") from exc

    lats, lons = domain_coordinates(domain, d3_resolution_m)
    # EPSG:4326 longitudes must be monotonic. Keep an inverse map to restore
    # the source model's 0..360 or -180..180 coordinate convention afterward.
    lon_wrapped = ((np.asarray(lons, dtype=np.float64) + 180.0) % 360.0) - 180.0
    order = np.argsort(lon_wrapped)
    lon_sorted = lon_wrapped[order]
    if len(lon_sorted) > 1 and np.any(np.diff(lon_sorted) <= 0):
        raise ValueError("Target longitude grid duplicates a meridian after wrapping")
    dst_transform = _target_transform(np.asarray(lats, dtype=np.float64), lon_sorted)
    dst_shape = (len(lats), len(lon_sorted))
    raw = np.full(dst_shape, np.nan, dtype=np.float32)

    with rasterio.open(source_path) as src:
        if src.crs is None:
            raise ValueError("Source bathymetry GeoTIFF has no CRS; refusing to assume geographic coordinates")
        if src.count < 1:
            raise ValueError("Source bathymetry has no raster band")
        reproject(
            source=rasterio.band(src, 1), destination=raw,
            src_transform=src.transform, src_crs=src.crs,
            src_nodata=src.nodata, dst_transform=dst_transform,
            dst_crs="EPSG:4326", dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
        source_tags = src.tags()
        source_bounds = tuple(float(v) for v in src.bounds)
        source_crs = str(src.crs)
        source_nodata = src.nodata

    if vertical_offset_raster is not None:
        offset_path = Path(vertical_offset_raster).expanduser().resolve()
        if not offset_path.is_file():
            raise FileNotFoundError(offset_path)
        offset = np.full(dst_shape, np.nan, dtype=np.float32)
        with rasterio.open(offset_path) as off:
            if off.crs is None:
                raise ValueError("Vertical offset raster has no CRS")
            reproject(
                source=rasterio.band(off, 1), destination=offset,
                src_transform=off.transform, src_crs=off.crs,
                src_nodata=off.nodata, dst_transform=dst_transform,
                dst_crs="EPSG:4326", dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
        if not np.isfinite(offset).all():
            raise ValueError("Vertical datum correction is missing over part of the target grid; no silent fill is allowed")
        # A chart-datum elevation z_chart is converted to MSL elevation by
        # z_MSL = z_chart + (MSL - chart datum).
        if source_values == "elevation_m":
            raw = raw + offset
        else:
            # positive chart-depth = -z_chart; positive MSL-depth = -(z_chart + offset)
            raw = raw - offset

    if source_values == "elevation_m":
        depth = np.maximum(-raw, 0.0)
    else:
        depth = np.maximum(raw, 0.0)
    valid = np.isfinite(raw)
    wet = valid & (depth > 0.0)
    depth = np.where(valid, depth, 0.0).astype(np.float32)
    wet = wet.astype(np.uint8)
    # Restore original longitude order used by the model.
    inverse = np.argsort(order)
    depth = depth[:, inverse]
    wet = wet[:, inverse]

    if output_path is None:
        tag = "d3_benelux" if "d3" in domain.lower() or "benelux" in domain.lower() else ("d2_europe" if "d2" in domain.lower() or "europe" in domain.lower() else "d1_global")
        out_dir = BATHY_ROOT / tag
        out_path = out_dir / "bathymetry_depth.wx1"
    else:
        out_path = Path(output_path)
        out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    wx1.write_dataset(
        out_path,
        {"bathymetry_depth": depth, "wet_mask": wet},
        dataset_id=f"bathymetry_{domain}",
        dimensions=["latitude", "longitude"],
        chunk_shapes={"bathymetry_depth": [min(128, depth.shape[0]), min(256, depth.shape[1])],
                      "wet_mask": [min(128, depth.shape[0]), min(256, depth.shape[1])]},
        attributes={"domain": domain, "source": source_name, "source_file": str(source_path),
                    "source_crs": source_crs, "source_bounds": source_bounds,
                    "source_nodata": source_nodata, "source_tags": source_tags,
                    "source_values": source_values, "vertical_datum": datum,
                    "vertical_offset_raster": str(vertical_offset_raster) if vertical_offset_raster else None,
                    "depth_positive": "down", "units": "m", "land_depth_value": 0.0,
                    "wet_mask_definition": "valid source cells with positive water depth",
                    "grid_shape": list(depth.shape), "grid_lat_first": float(lats[0]),
                    "grid_lat_last": float(lats[-1]), "grid_lon_first": float(lons[0]),
                    "grid_lon_last": float(lons[-1]),
                    "note": "Static bathymetry preparation only; vertical datum and source quality vary by source."},
        codec="zstd",
    )
    manifest = {
        "domain": domain, "source": source_name, "source_file": str(source_path),
        "source_crs": source_crs, "vertical_datum": datum,
        "vertical_offset_raster": str(vertical_offset_raster) if vertical_offset_raster else None,
        "source_values": source_values, "grid_shape": list(depth.shape),
        "valid_cells": int(valid.sum()), "wet_cells": int(wet.sum()),
        "max_depth_m": float(depth.max()), "min_wet_depth_m": float(depth[wet > 0].min()) if wet.any() else None,
        "prepared_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "quality_note": "Not a full ocean model validation; inspect source provenance and vertical datum before operational use.",
    }
    (out_dir / "bathymetry_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Prepared %s bathymetry: shape=%s max=%.1fm wet=%d", domain, depth.shape, depth.max(), wet.sum())
    return out_path


def load_bathymetry(path: str | Path, expected_shape: tuple[int, int] | None = None):
    """Load a prepared bathymetry WX-1 and return (positive-down depth, wet mask, attrs)."""
    p = Path(path)
    if not p.is_file():
        return None
    ds = wx1.open_dataset(p)

    # Extract bathymetry depth array
    depth_var = None
    for k in ["bathymetry_depth", "/bathymetry_depth"]:
        if k in ds.variables:
            depth_var = ds.variables[k]
            break
    if depth_var is None and len(ds.variables) > 0:
        depth_var = list(ds.variables.values())[0]

    if depth_var is None:
        raise ValueError(f"{p} must contain bathymetry_depth variable")

    depth = np.asarray(depth_var.values, dtype=np.float32)

    # Extract wet mask array from same dataset or companion bathymetry_wet_mask.wx1
    wet_var = None
    for k in ["wet_mask", "/wet_mask"]:
        if k in ds.variables:
            wet_var = ds.variables[k]
            break

    if wet_var is not None:
        wet = np.asarray(wet_var.values, dtype=np.uint8)
    else:
        mask_path = p.parent / "bathymetry_wet_mask.wx1"
        if mask_path.is_file():
            ds_m = wx1.open_dataset(mask_path)
            wet = np.asarray(list(ds_m.variables.values())[0].values, dtype=np.uint8)
        else:
            wet = (depth > 0.0).astype(np.uint8)

    if depth.ndim != 2 or wet.shape != depth.shape:
        raise ValueError("Bathymetry depth and wet mask must be matching 2D arrays")
    if expected_shape is not None and depth.shape != tuple(expected_shape):
        raise ValueError(f"Bathymetry grid mismatch: expected {expected_shape}, got {depth.shape}")
    if not np.isfinite(depth).all() or np.any(depth < 0):
        raise ValueError("Bathymetry contains non-finite or negative depths")
    return depth, wet.astype(bool), getattr(ds, "attrs", {})


def prepare_bathymetry_on_2d_grid(
    source_path: str | Path,
    domain_name: str,
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    *,
    vertical_datum: str = "msl",
    output_path: str | Path | None = None,
) -> Path:
    """Regrid GEBCO NetCDF (elevation positive-up, MSL) onto a 2D (lat, lon) grid and store in WX-1."""
    import xarray as xr
    from scipy.interpolate import RegularGridInterpolator

    datum = vertical_datum.lower()
    if datum != "msl":
        raise ValueError(f"prepare_bathymetry_on_2d_grid requires MSL datum, got {vertical_datum!r}")
    src_p = Path(source_path).expanduser().resolve()
    if not src_p.is_file():
        raise FileNotFoundError(src_p)
    if src_p.suffix.lower() != ".nc":
        raise ValueError("2D curvilinear bathymetry helper currently expects GEBCO NetCDF (.nc); use prepare_bathymetry for GeoTIFFs")

    lat_arr = np.asarray(lat2d, dtype=np.float64)
    lon_arr = ((np.asarray(lon2d, dtype=np.float64) + 180.0) % 360.0) - 180.0
    with xr.open_dataset(src_p) as ds:
        lat_coord = "lat" if "lat" in ds.coords or "lat" in ds.dims else "latitude"
        lon_coord = "lon" if "lon" in ds.coords or "lon" in ds.dims else "longitude"
        elev_var = "elevation" if "elevation" in ds.data_vars else next(iter(ds.data_vars))
        lat_head = np.asarray(ds.coords[lat_coord].isel({lat_coord: slice(0, 2)}).values, dtype=np.float64)
        lon_head = np.asarray(ds.coords[lon_coord].isel({lon_coord: slice(0, 2)}).values, dtype=np.float64)
        lat_step = abs(float(lat_head[1] - lat_head[0])) if len(lat_head) > 1 else 0.25
        lon_step = abs(float(lon_head[1] - lon_head[0])) if len(lon_head) > 1 else 0.25
        pad_lat = max(0.1, 2.0 * lat_step)
        pad_lon = max(0.1, 2.0 * lon_step)
        la0, la1 = float(lat_arr.min()) - pad_lat, float(lat_arr.max()) + pad_lat
        lo0, lo1 = float(lon_arr.min()) - pad_lon, float(lon_arr.max()) + pad_lon
        lat_desc = len(lat_head) > 1 and lat_head[0] > lat_head[1]
        lon_desc = len(lon_head) > 1 and lon_head[0] > lon_head[1]
        lat_slice = slice(la1, la0) if lat_desc else slice(la0, la1)
        lon_slice = slice(lo1, lo0) if lon_desc else slice(lo0, lo1)
        sub = ds[elev_var].sel({lat_coord: lat_slice, lon_coord: lon_slice})
        if sub.sizes.get(lat_coord, 0) > 4096 or sub.sizes.get(lon_coord, 0) > 4096:
            sy = max(1, int(sub.sizes[lat_coord]) // 2048)
            sx = max(1, int(sub.sizes[lon_coord]) // 2048)
            sub = sub.isel({lat_coord: slice(None, None, sy), lon_coord: slice(None, None, sx)})
        sub_lats = np.asarray(sub.coords[lat_coord].values, dtype=np.float64)
        sub_lons = np.asarray(sub.coords[lon_coord].values, dtype=np.float64)
        elev = np.asarray(sub.values, dtype=np.float64)
        if len(sub_lats) > 1 and sub_lats[0] > sub_lats[-1]:
            sub_lats = sub_lats[::-1]
            elev = elev[::-1, :]
        if len(sub_lons) > 1 and sub_lons[0] > sub_lons[-1]:
            sub_lons = sub_lons[::-1]
            elev = elev[:, ::-1]

    interp = RegularGridInterpolator((sub_lats, sub_lons), elev, method="linear", bounds_error=False, fill_value=0.0)
    sampled_elev = interp(np.column_stack([lat_arr.ravel(), lon_arr.ravel()])).reshape(lat_arr.shape)
    depth = np.maximum(-sampled_elev, 0.0).astype(np.float32)
    wet = (depth > 0.0).astype(np.uint8)

    out_path = Path(output_path) if output_path is not None else (BATHY_ROOT / domain_name / "bathymetry_depth.wx1")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_ds = xr.Dataset(
        data_vars={
            "bathymetry_depth": (("y", "x"), depth, {"units": "m", "positive": "down"}),
            "wet_mask": (("y", "x"), wet, {"units": "1"}),
        },
        coords={
            "lat2d": (("y", "x"), lat_arr, {"units": "degrees_north"}),
            "lon2d": (("y", "x"), lon_arr, {"units": "degrees_east"}),
        },
        attrs={
            "domain": domain_name,
            "source": str(src_p.name),
            "vertical_datum": datum,
        },
    )
    wx1.write_dataset(out_path, out_ds, dataset_id=f"bathymetry_{domain_name}", codec="zstd_shuffle", workers=4)
    return out_path
