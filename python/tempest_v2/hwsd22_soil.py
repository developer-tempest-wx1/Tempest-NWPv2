"""HWSD v2.2 soil-data preparation for Tempest-NWP.

This reader deliberately requires the *real* v2.2 database.  It never interprets
HWSD 1.x table/column names as v2.2.  Microsoft Access MDB extraction is done
through the open-source mdbtools command-line utilities (mdb-tables/mdb-export).

The 1 km raster stores HWSD2_SMU identifiers; HWSD2_LAYERS stores properties
for component/depth rows.  Component SHARE and TOPDEP/BOTDEP overlaps are
area/depth weighted into Tempest's four Noah layers: 0-10, 10-40, 40-100,
100-200 cm.  Properties are only emitted when present in the source table.
"""
from __future__ import annotations

import csv
import io
import json
import logging
import os
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import wx1

logger = logging.getLogger("tempest_v2.hwsd22")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATIC_ROOT = PROJECT_ROOT / "data" / "static"
SOURCE_ROOT = STATIC_ROOT / "hwsd_v22"
DOWNLOAD_ROOT = SOURCE_ROOT / "downloads"
EXTRACT_ROOT = SOURCE_ROOT / "extracted"
RASTER_SHAPE = (21600, 43200)
RASTER_DTYPE = "<u2"
RASTER_NODATA = 65535
RASTER_STEP = 1.0 / 120.0
RASTER_UL_LON = -180.0 + RASTER_STEP / 2.0
RASTER_UL_LAT = 90.0 - RASTER_STEP / 2.0
TARGET_BOUNDS_M = np.array([0.0, 0.10, 0.40, 1.00, 2.00], dtype=np.float64)
PROPERTY_FIELDS = {
    "sand_pct": ("SAND", "percent"),
    "silt_pct": ("SILT", "percent"),
    "clay_pct": ("CLAY", "percent"),
    "bulk_density": ("BULK", "g cm-3_or_source_units"),
    "reference_bulk_density": ("REF_BULK", "g cm-3_or_source_units"),
    "organic_carbon": ("ORG_CARBON", "g kg-1_or_source_units"),
    "ph_water": ("PH_WATER", "pH"),
    "total_nitrogen": ("TOTAL_N", "source_units"),
    "cec_soil": ("CEC_SOIL", "cmolc kg-1_or_source_units"),
    "cec_clay": ("CEC_CLAY", "cmolc kg-1_or_source_units"),
    "cec_effective": ("CEC_EFF", "cmolc kg-1_or_source_units"),
    "base_saturation": ("BSAT", "percent_or_source_units"),
    "aluminium_saturation": ("ALUM_SAT", "percent_or_source_units"),
    "total_carbonate_equivalent": ("TCARBON_EQ", "source_units"),
    "gypsum": ("GYPSUM", "source_units"),
    "electrical_conductivity": ("ELEC_COND", "source_units"),
}


def _run(command: list[str]) -> str:
    try:
        proc = subprocess.run(command, check=True, capture_output=True, text=True, encoding="utf-8", errors="replace")
    except FileNotFoundError as exc:
        raise RuntimeError(
            "HWSD v2.2 Access database needs mdbtools (`mdb-tables` and `mdb-export`). "
            "Install the OS package mdbtools; do not substitute the HWSD v1 parser."
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()[-2000:]
        raise RuntimeError(f"mdbtools command failed: {' '.join(command)}: {detail}") from exc
    return proc.stdout


def _find_one(root: Path, suffix: str) -> Path:
    matches = list(root.rglob(suffix))
    if not matches:
        raise FileNotFoundError(f"Could not find {suffix} under {root}")
    if len(matches) > 1:
        # Prefer the canonical v2.2 names, while refusing ambiguity otherwise.
        matches.sort(key=lambda p: ("HWSD2.2.mdb" not in p.name, len(p.parts), str(p)))
    return matches[0]


def _extract_archives(force: bool = False) -> tuple[Path, Path]:
    EXTRACT_ROOT.mkdir(parents=True, exist_ok=True)
    db_root, raster_root = EXTRACT_ROOT / "db", EXTRACT_ROOT / "raster"
    db_file = db_root / "HWSD2.2.mdb"
    bil_file = raster_root / "HWSD2.bil"
    if not force and db_file.is_file() and bil_file.is_file():
        return db_file, bil_file

    db_zip = DOWNLOAD_ROOT / "hwsd2.2_db.zip"
    if not db_zip.is_file():
        db_zip = SOURCE_ROOT / "hwsd2.2_db.zip"
    raster_zip = DOWNLOAD_ROOT / "hwsd2.2_raster.zip"
    if not raster_zip.is_file():
        raster_zip = SOURCE_ROOT / "hwsd2.2_raster.zip"
    if not db_zip.is_file() or not raster_zip.is_file():
        raise FileNotFoundError(
            f"HWSD v2.2 archives missing. Expected {db_zip} and {raster_zip}; "
            "the v2.2 source archives are not interchangeable with HWSD v1.x."
        )
    if force or not db_file.exists():
        db_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(db_zip) as zf:
            bad = zf.testzip()
            if bad:
                raise RuntimeError(f"HWSD v2.2 database ZIP CRC error at {bad}")
            for member in zf.namelist():
                p = Path(member)
                if p.is_absolute() or ".." in p.parts:
                    raise RuntimeError(f"Unsafe path in HWSD v2.2 database ZIP: {member}")
            zf.extractall(db_root)
    if force or not bil_file.exists():
        raster_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(raster_zip) as zf:
            bad = zf.testzip()
            if bad:
                raise RuntimeError(f"HWSD v2.2 raster ZIP CRC error at {bad}")
            for member in zf.namelist():
                p = Path(member)
                if p.is_absolute() or ".." in p.parts:
                    raise RuntimeError(f"Unsafe path in HWSD v2.2 raster ZIP: {member}")
            zf.extractall(raster_root)
    return _find_one(db_root, "HWSD2.2.mdb"), _find_one(raster_root, "HWSD2.bil")


def _read_table(db_path: Path, table_name: str) -> list[dict[str, str]]:
    csv_text = _run(["mdb-export", str(db_path), table_name])
    reader = csv.DictReader(io.StringIO(csv_text))
    if not reader.fieldnames:
        raise RuntimeError(f"MDB table {table_name} has no header or is empty")
    return [{(k or "").strip().upper(): (v or "").strip() for k, v in row.items()} for row in reader]


def _float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value in (None, "", "NULL", "null"):
        return float("nan")
    try:
        return float(value.replace(",", "."))
    except (AttributeError, ValueError):
        return float("nan")


def _texture_class(sand: float, silt: float, clay: float) -> int:
    if not np.isfinite([sand, silt, clay]).all() or sand + silt + clay <= 0:
        return 9
    # Match the existing Tempest USDA-class convention (1..13).
    if clay >= 40:
        return 8 if sand >= 45 else (2 if silt >= 40 else 1)
    if clay >= 27:
        return 10 if sand >= 45 else (4 if silt >= 28 else 5)
    if sand >= 70:
        return 11 if clay >= 15 else (13 if sand >= 85 else 12)
    if silt >= 50:
        return 6 if silt >= 80 and clay < 12 else 7
    return 11 if sand >= 52 else 9


def _make_smu_profiles(rows: list[dict[str, str]]) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, str]]:
    """Depth/area-weight source component rows into 4 Tempest layers by SMU ID."""
    if not rows:
        raise RuntimeError("HWSD2_LAYERS has no records")
    all_fields = set().union(*(r.keys() for r in rows))
    id_field = next((k for k in ("HWSD2_SMU_ID", "SMU_ID", "MU_GLOBAL") if k in all_fields), None)
    if not id_field:
        raise RuntimeError(f"HWSD2_LAYERS has no known v2.2 soil mapping-unit key; fields={sorted(all_fields)}")
    if not {"TOPDEP", "BOTDEP"}.issubset(all_fields):
        raise RuntimeError("HWSD2_LAYERS lacks TOPDEP/BOTDEP; refusing to infer the v2.2 layer schema")
    numeric_fields = {out: src for out, (src, _unit) in PROPERTY_FIELDS.items() if src in all_fields}
    if not {"SAND", "SILT", "CLAY"}.issubset(all_fields):
        raise RuntimeError("HWSD2_LAYERS lacks SAND/SILT/CLAY; cannot derive a defensible USDA texture map")

    parsed: dict[int, list[tuple[float, float, float, dict[str, float]]]] = {}
    depth_max = 0.0
    for row in rows:
        try:
            smu = int(float(row.get(id_field, "")))
        except (ValueError, TypeError):
            continue
        top, bottom = _float(row, "TOPDEP"), _float(row, "BOTDEP")
        if not np.isfinite([top, bottom]).all() or bottom <= top:
            continue
        # HWSD depth fields are conventionally cm. Keep the source schema's
        # actual range as a sanity guard; convert only if clearly metre-based.
        if bottom <= 2.5:
            top *= 100.0
            bottom *= 100.0
        share = _float(row, "SHARE")
        if not np.isfinite(share) or share <= 0:
            share = 1.0
        vals = {out: _float(row, src) for out, src in numeric_fields.items()}
        parsed.setdefault(smu, []).append((top / 100.0, bottom / 100.0, share, vals))
        depth_max = max(depth_max, bottom / 100.0)

    if not parsed:
        raise RuntimeError("No valid HWSD v2.2 layer rows with SMU IDs and depth intervals were found")
    ids = np.array(sorted(parsed), dtype=np.int32)
    profiles = {name: np.full((4, len(ids)), np.nan, dtype=np.float32) for name in numeric_fields}
    for smu_idx, smu in enumerate(ids):
        entries = parsed[int(smu)]
        for layer in range(4):
            a, b = TARGET_BOUNDS_M[layer], TARGET_BOUNDS_M[layer + 1]
            for name in numeric_fields:
                numerator = 0.0
                denominator = 0.0
                for top, bottom, share, vals in entries:
                    overlap = max(0.0, min(b, bottom) - max(a, top))
                    val = vals.get(name, float("nan"))
                    if overlap > 0 and np.isfinite(val):
                        weight = share * overlap
                        numerator += val * weight
                        denominator += weight
                if denominator > 0:
                    profiles[name][layer, smu_idx] = numerator / denominator
    # Keep the actual DB key, not the old v1 global MU ID.
    return ids, profiles, {"smu_id_field": id_field, "source_depth_max_m": depth_max,
                            "source_fields": sorted(all_fields),
                            "available_properties": {k: PROPERTY_FIELDS[k][1] for k in numeric_fields}}


def _sample_raster_ids(bil_path: Path, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    expected = int(np.prod(RASTER_SHAPE) * np.dtype(RASTER_DTYPE).itemsize)
    actual = bil_path.stat().st_size
    if actual != expected:
        raise RuntimeError(f"HWSD2.bil byte length mismatch: expected {expected:,}, got {actual:,}")
    raster = np.memmap(bil_path, dtype=RASTER_DTYPE, mode="r", shape=RASTER_SHAPE)
    lat_arr = np.asarray(lats, dtype=np.float64)
    lon_arr = ((np.asarray(lons, dtype=np.float64) + 180.0) % 360.0) - 180.0
    row = np.clip(np.rint((RASTER_UL_LAT - lat_arr) / RASTER_STEP).astype(np.int64), 0, RASTER_SHAPE[0] - 1)
    col = np.clip(np.rint((lon_arr - RASTER_UL_LON) / RASTER_STEP).astype(np.int64), 0, RASTER_SHAPE[1] - 1)
    if lat_arr.ndim == 2 and lon_arr.ndim == 2:
        return np.asarray(raster[row, col], dtype=np.uint16)
    return np.asarray(raster[np.ix_(row, col)], dtype=np.uint16)


def sample_hwsd22_on_grid(
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    domain_name: str,
    output_path: Optional[Path | str] = None,
) -> Path:
    """Sample real HWSD v2.2 USDA soil class and 4-layer soil properties onto an arbitrary 2D (lat, lon) grid."""
    db_path, bil_path = _extract_archives(force=False)
    table_names = [x.strip() for x in _run(["mdb-tables", "-1", str(db_path)]).splitlines() if x.strip()]
    table_lookup = {x.upper(): x for x in table_names}
    table_name = table_lookup.get("HWSD2_LAYERS")
    if not table_name:
        raise RuntimeError(f"HWSD2_LAYERS not found in v2.2 database; tables include {table_names[:40]}")
    rows = _read_table(db_path, table_name)
    smu_ids, profiles, _db_meta = _make_smu_profiles(rows)
    ids_map = _sample_raster_ids(bil_path, lat2d, lon2d)
    out = Path(output_path) if output_path is not None else (STATIC_ROOT / f"soil_type_{domain_name}_hwsd22.wx1")
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_domain(out, ids_map, smu_ids, profiles, domain_name, (1, ids_map.shape[0], ids_map.shape[1]))
    return out


def _write_domain(out_path: Path, ids_map: np.ndarray, smu_ids: np.ndarray,
                  profiles: dict[str, np.ndarray], domain: str,
                  layers: tuple[int, int, int]) -> None:
    ny, nx = ids_map.shape
    lookup = {int(v): i for i, v in enumerate(smu_ids)}
    max_id = int(max(lookup)) if lookup else 0
    # A compact dense lookup is much faster than Python iteration for the map.
    dense = np.full(max_id + 1, -1, dtype=np.int32)
    for smu, idx in lookup.items():
        dense[smu] = idx
    good = (ids_map != RASTER_NODATA) & (ids_map < len(dense))
    mapped = np.full(ids_map.shape, -1, dtype=np.int32)
    mapped[good] = dense[ids_map[good]]

    layer_values: dict[str, np.ndarray] = {}
    for name, table in profiles.items():
        out = np.full((4, ny, nx), np.nan, dtype=np.float32)
        valid = mapped >= 0
        for k in range(4):
            out[k, valid] = table[k, mapped[valid]]
        layer_values[name] = out

    sand = layer_values.get("sand_pct")
    silt = layer_values.get("silt_pct")
    clay = layer_values.get("clay_pct")
    if sand is None or silt is None or clay is None:
        raise RuntimeError("HWSD v2.2 lacks texture fractions required for the soil class product")
    soil_type = np.zeros((ny, nx), dtype=np.uint8)
    valid_tex = (mapped >= 0) & np.isfinite(sand[0]) & np.isfinite(silt[0]) & np.isfinite(clay[0])
    ss, si, cc = sand[0], silt[0], clay[0]
    # Vectorized USDA texture classification; unclassified/no-data remains 0.
    m = valid_tex & (ss + si + cc > 0)
    soil_type[m & (cc >= 40) & (ss >= 45)] = 8
    soil_type[m & (cc >= 40) & (ss < 45) & (si >= 40)] = 2
    soil_type[m & (cc >= 40) & (ss < 45) & (si < 40)] = 1
    soil_type[m & (cc >= 27) & (cc < 40) & (ss >= 45)] = 10
    soil_type[m & (cc >= 27) & (cc < 40) & (ss < 45) & (si >= 28)] = 4
    soil_type[m & (cc >= 27) & (cc < 40) & (ss < 45) & (si < 28)] = 5
    soil_type[m & (cc < 27) & (ss >= 70) & (cc >= 15)] = 11
    soil_type[m & (cc < 15) & (ss >= 85)] = 13
    soil_type[m & (cc < 15) & (ss >= 70) & (ss < 85)] = 12
    soil_type[m & (cc < 27) & (ss < 70) & (si >= 50) & (si >= 80) & (cc < 12)] = 6
    soil_type[m & (cc < 27) & (ss < 70) & (si >= 50) & ~((si >= 80) & (cc < 12))] = 7
    soil_type[m & (cc < 27) & (ss < 70) & (si < 50) & (ss >= 52)] = 11
    soil_type[m & (cc < 27) & (ss < 70) & (si < 50) & (ss < 52)] = 9


    attrs = {
        "source": "FAO/IIASA Harmonized World Soil Database v2.2",
        "source_db_table": "HWSD2_LAYERS",
        "source_raster": "HWSD2.bil (HWSD2_SMU identifiers)",
        "domain": domain,
        "target_depth_bounds_m": TARGET_BOUNDS_M.tolist(),
        "missing_or_nodata_soil_class": 0,
        "soil_class_convention": "Tempest USDA 13 texture classes; 0 water/no-data",
    }
    wx1.write_dataset(
        out_path,
        {"soil_type": soil_type,
         "sand_pct": np.nan_to_num(sand[0], nan=-9999.0),
         "silt_pct": np.nan_to_num(silt[0], nan=-9999.0),
         "clay_pct": np.nan_to_num(clay[0], nan=-9999.0)},
        dataset_id=f"hwsd22_soil_type_{domain}",
        dimensions=["latitude", "longitude"],
        chunk_shapes={"soil_type": [ny, nx], "sand_pct": [ny, nx], "silt_pct": [ny, nx], "clay_pct": [ny, nx]},
        attributes=attrs,
        codec="zstd",
    )
    profile_path = out_path.with_name(out_path.stem.replace("soil_type_", "soil_profile_") + ".wx1")
    profile_write = {k: np.nan_to_num(v, nan=-9999.0).astype(np.float32) for k, v in layer_values.items()}
    if profile_write:
        wx1.write_dataset(
            profile_path, profile_write,
            dataset_id=f"hwsd22_soil_profile_{domain}",
            dimensions=["soil_layer", "latitude", "longitude"],
            chunk_shapes={k: [1, ny, nx] for k in profile_write},
            attributes={**attrs, "soil_layer_bounds_m": TARGET_BOUNDS_M.tolist(),
                        "missing_value": -9999.0, "profile_note": "source-weighted vertical overlap; -9999 means no source value"},
            codec="zstd",
        )


def generate_hwsd22_static_containers(force: bool = False) -> dict[str, Path]:
    """Create D1/D2/D3 HWSD v2.2 soil products from the bundled official archives."""
    db_path, bil_path = _extract_archives(force=force)
    table_names = [x.strip() for x in _run(["mdb-tables", "-1", str(db_path)]).splitlines() if x.strip()]
    table_lookup = {x.upper(): x for x in table_names}
    table_name = table_lookup.get("HWSD2_LAYERS")
    if not table_name:
        raise RuntimeError(f"HWSD2_LAYERS not found in v2.2 database; tables include {table_names[:40]}")
    rows = _read_table(db_path, table_name)
    smu_ids, profiles, db_meta = _make_smu_profiles(rows)

    from tempest_v2.static_fields import get_benelux_grid_coords, benelux_grid_tag
    d3_tag = benelux_grid_tag()
    domains = {
        "d1_global_0p4": (np.linspace(90.0, -90.0, 451, dtype=np.float32),
                          np.linspace(0.0, 360.0 - 360.0 / 900.0, 900, dtype=np.float32)),
        "d2_europe_0p1": (np.linspace(72.0, 30.0, 421, dtype=np.float32),
                           np.linspace(-25.0, 45.0, 701, dtype=np.float32)),
        f"d3_benelux_{d3_tag}": get_benelux_grid_coords(),
    }
    results: dict[str, Path] = {}
    metadata = {**db_meta, "version": "2.2", "archive_files": [
        "downloads/hwsd2.2_db.zip", "downloads/hwsd2.2_raster.zip"],
        "raster_shape": list(RASTER_SHAPE), "raster_dtype": RASTER_DTYPE,
        "raster_nodata": RASTER_NODATA, "target_layer_bounds_m": TARGET_BOUNDS_M.tolist(),
        "created_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()}
    for domain, (lats, lons) in domains.items():
        out = STATIC_ROOT / f"soil_type_{domain}_hwsd22.wx1"
        if force or not out.exists():
            ids = _sample_raster_ids(bil_path, lats, lons)
            _write_domain(out, ids, smu_ids, profiles, domain, ids.shape)
        results[domain] = out
    SOURCE_ROOT.mkdir(parents=True, exist_ok=True)
    (SOURCE_ROOT / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Created HWSD v2.2 static soil products for %s", ", ".join(results))
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for key, path in generate_hwsd22_static_containers(force=True).items():
        print(f"{key}: {path}")
