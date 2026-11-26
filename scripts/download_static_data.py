"""Deterministic Static Geophysical Data Downloader for Tempest-NWP v2.

Downloads official open-access static datasets into the exact layout expected by
`tempest_v2.copernicus_dem`, `tempest_v2.bathymetry`, `tempest_v2.hwsd22_soil`,
`tempest_v2.land_static`, and `tempest_v2.ic_builder` (no credentials required
for Copernicus DEM, EMODnet, GEBCO ERDDAP/CEDA, ESA WorldCover, ISRIC SoilGrids
250 m WCS, FAO HWSD v2.2, FABDEM v1.2, or PDOK AHN):

  1. Copernicus DEM GLO-30 (30 m) & GLO-90 (90 m) + auxiliary masks (`WBM`, `EDM`,
     `FLM`, `HEM`, `ACM`, `XML`, `PREVIEW`, `INFO`) from public AWS S3.
  2. Bathymetry:
     - `gebco-erddap` (default): regional MSL NetCDF subset via NOAA/PFEL ERDDAP.
     - `gebco-2026-global`: global `GEBCO_2026.nc` (~7.0 GiB, MSL) via CEDA/BODC.
     - `gebco-2026-zip`: global `GEBCO_2026.zip` (~4.0 GiB, extracted to `.nc`) via CEDA.
     - `gebco-2026-full`: full GEBCO 2026 suite (`GEBCO_2026`, `GEBCO_2026_sub_ice`,
       and `gebco_2026_tid` ZIPs + extracted `.nc` files) via CEDA/BODC.
     - `gebco-2025-zip`: global `gebco_2025.zip` via CEDA/BODC.
     - `emodnet-wcs`: high-res European GeoTIFF subset via EMODnet WCS (LAT datum).
  3. ESA WorldCover 10 m (`2021` v200 / `2020` v100) from public AWS S3.
  4. ISRIC SoilGrids 250 m v2.0 WCS (`tempest`, `surface`, or `full` 55 layers + WRB).
  5. FAO / IIASA HWSD v2.2 (`hwsd2.2_db.zip` + `hwsd2.2_raster.zip`) + `mdbtools` check.
  6. Optional opt-in datasets: FABDEM v1.2 (`--fabdem`) & PDOK AHN (`--ahn`).
  7. Optional `--prepare-wx1` pass converting raw sources into lossless `zstd_shuffle` `.wx1`.

Strictly deterministic data acquisition (0% AI / 0% ML).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from typing import Dict, Iterable, List, Optional, Set, Tuple

try:
    from tqdm import tqdm as _tqdm
except ImportError:  # pragma: no cover - optional terminal progress bar
    _tqdm = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PY_ROOT = PROJECT_ROOT / "python"
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))

from tempest_v2.projection import get_domain_spec  # noqa: E402

COPERNICUS_URLS = {
    "glo30": ("https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com", "10", "dem30_europe"),
    "glo90": ("https://copernicus-dem-90m.s3.eu-central-1.amazonaws.com", "30", "dem90_world"),
}
COPERNICUS_LAYER_NAMES: Tuple[str, ...] = (
    "DEM",
    "WBM",
    "EDM",
    "FLM",
    "HEM",
    "ACM",
    "XML",
    "PREVIEW",
    "INFO",
)
COPERNICUS_LAYER_PRESETS: Dict[str, Tuple[str, ...]] = {
    "minimal": ("DEM", "WBM"),
    "standard": ("DEM", "WBM", "EDM", "FLM", "HEM"),
    "all": COPERNICUS_LAYER_NAMES,
}
COPERNICUS_PREVIEW_SUFFIXES: Tuple[str, ...] = (
    "DEM_QL.tif",
    "DEM_ABS_QL.tif",
    "EDM_QL.tif",
    "FLM_QL.tif",
    "HEM_QL.tif",
    "WBM_QL.tif",
    "QL.kml",
    "SRC.kml",
)

GEBCO_CEDA_2026_NC_URL = "https://dap.ceda.ac.uk/bodc/gebco/global/gebco_2026/ice_surface_elevation/netcdf/GEBCO_2026.nc"
GEBCO_CEDA_2026_ZIP_URL = "https://dap.ceda.ac.uk/bodc/gebco/global/gebco_2026/ice_surface_elevation/netcdf/GEBCO_2026.zip"
GEBCO_CEDA_2026_SUB_ICE_ZIP_URL = "https://dap.ceda.ac.uk/bodc/gebco/global/gebco_2026/sub_ice_topography_bathymetry/netcdf/GEBCO_2026_sub_ice.zip"
GEBCO_CEDA_2026_TID_ZIP_URL = "https://dap.ceda.ac.uk/bodc/gebco/global/gebco_2026/type_identifier_grid/netcdf/gebco_2026_tid.zip"
GEBCO_CEDA_2025_ZIP_URL = "https://dap.ceda.ac.uk/bodc/gebco/global/gebco_2025/ice_surface_elevation/netcdf/gebco_2025.zip"
GEBCO_ERDDAP_BASE = "https://coastwatch.pfeg.noaa.gov/erddap/griddap"
EMODNET_WCS_BASE = "https://ows.emodnet-bathymetry.eu/wcs"

WORLDCOVER_BASE_URL = "https://esa-worldcover.s3.eu-central-1.amazonaws.com"
WORLDCOVER_GRID_URL = f"{WORLDCOVER_BASE_URL}/esa_worldcover_grid.geojson"
WORLDCOVER_VERSIONS: Dict[int, str] = {2020: "v100", 2021: "v200"}

SOILGRIDS_WCS_BASE = "https://maps.isric.org/mapserv"
SOILGRIDS_PROPERTIES: Tuple[str, ...] = (
    "clay",
    "sand",
    "silt",
    "soc",
    "bdod",
    "phh2o",
    "wv0010",
    "wv0033",
    "wv1500",
)
SOILGRIDS_DEPTHS: Tuple[str, ...] = (
    "0-5cm",
    "5-15cm",
    "15-30cm",
    "30-60cm",
    "60-100cm",
    "100-200cm",
)
TEMPEST_SOILGRIDS_DEPTHS: Dict[str, Tuple[str, ...]] = {
    "clay": ("0-5cm", "5-15cm", "15-30cm"),
    "sand": ("0-5cm", "5-15cm", "15-30cm"),
    "silt": ("0-5cm", "5-15cm", "15-30cm"),
    "soc": ("0-5cm", "15-30cm"),
    "bdod": ("0-5cm", "15-30cm"),
}

FABDEM_BASE_URL = "https://huggingface.co/datasets/links-ads/fabdem-v12/resolve/main/tiles"
AHN_WCS_BASE = "https://service.pdok.nl/rws/ahn/wcs/v1_0"
NL_BOUNDS: Tuple[float, float, float, float] = (50.6, 53.7, 3.2, 7.3)

HWSD_V22_FILES: Tuple[Dict[str, str], ...] = (
    {
        "filename": "hwsd2.2_db.zip",
        "url": "https://data.apps.fao.org/catalog/dataset/fedae0d8-677c-47a7-9049-6cba560e6c0a/resource/2a815078-f166-4fd6-af0c-733235c15daf/download/hwsd2.2_db.zip",
        "sha256": "593d281efd5e9491e4ff7548b5a9af00d3d0a6d70efa8eab255c4c0f861f6305",
        "required_suffix": ".mdb",
    },
    {
        "filename": "hwsd2.2_raster.zip",
        "url": "https://data.apps.fao.org/catalog/dataset/fedae0d8-677c-47a7-9049-6cba560e6c0a/resource/3d3ad0cb-c840-4eea-8ae1-fada78c5eebb/download/hwsd2.2_raster.zip",
        "sha256": "55612130174f228f48d0691a2e01ab92150617dfd0d6605e65e3619eb5c123d8",
        "required_suffix": ".bil",
    },
)

USER_AGENT = "Tempest-NWP-StaticDownloader/2.0"


def _copernicus_tile_stems(lat: int, lon: int, resolution: str = "glo30") -> Tuple[str, str, str, str]:
    """Return `(base_url, cog_stem, raw_stem, folder)` for integer SW tile `(lat, lon)`."""
    res_key = resolution.lower().strip()
    if res_key not in COPERNICUS_URLS:
        raise ValueError(f"Unsupported Copernicus DEM resolution {resolution!r}; expected 'glo30' or 'glo90'")
    base_url, tag, _ = COPERNICUS_URLS[res_key]
    ns = "N" if int(lat) >= 0 else "S"
    ew = "E" if int(lon) >= 0 else "W"
    coord = f"{ns}{abs(int(lat)):02d}_00_{ew}{abs(int(lon)):03d}_00"
    cog_stem = f"Copernicus_DSM_COG_{tag}_{coord}"
    raw_stem = f"Copernicus_DSM_{tag}_{coord}"
    folder = f"{cog_stem}_DEM"
    return base_url, cog_stem, raw_stem, folder


def parse_copernicus_layers(spec: str | Iterable[str]) -> Tuple[str, ...]:
    """Normalize a preset name (`minimal`, `standard`, `all`) or comma-separated layer list."""
    if not isinstance(spec, str):
        tokens = [str(item).strip().upper() for item in spec if str(item).strip()]
    else:
        key = spec.strip().lower()
        if key in COPERNICUS_LAYER_PRESETS:
            return COPERNICUS_LAYER_PRESETS[key]
        tokens = [tok.strip().upper() for tok in spec.split(",") if tok.strip()]
    if not tokens:
        raise ValueError("Copernicus layer specification cannot be empty")
    out: List[str] = []
    seen: Set[str] = set()
    for tok in tokens:
        if tok not in COPERNICUS_LAYER_NAMES:
            raise ValueError(f"Unknown Copernicus layer {tok!r}; expected one of {COPERNICUS_LAYER_NAMES}")
        if tok not in seen:
            seen.add(tok)
            out.append(tok)
    if "DEM" not in seen:
        out.insert(0, "DEM")
    return tuple(out)


def copernicus_tile_urls(lat: int, lon: int, resolution: str = "glo30") -> Tuple[str, str, str, str]:
    """Return `(dem_url, wbm_url, dem_filename, wbm_filename)` for integer SW tile `(lat, lon)`."""
    base_url, cog_stem, _, folder = _copernicus_tile_stems(lat, lon, resolution)
    dem_file = f"{cog_stem}_DEM.tif"
    wbm_file = f"{cog_stem}_WBM.tif"
    dem_url = f"{base_url}/{folder}/{dem_file}"
    wbm_url = f"{base_url}/{folder}/AUXFILES/{wbm_file}"
    return dem_url, wbm_url, dem_file, wbm_file


def copernicus_tile_asset_urls(
    lat: int,
    lon: int,
    resolution: str = "glo30",
    layers: str | Iterable[str] = COPERNICUS_LAYER_PRESETS["standard"],
) -> List[Tuple[str, str, str]]:
    """Return `[(layer_key, url, relative_path), ...]` for requested Copernicus tile assets."""
    selected = parse_copernicus_layers(layers)
    base_url, cog_stem, raw_stem, folder = _copernicus_tile_stems(lat, lon, resolution)
    assets: List[Tuple[str, str, str]] = []
    for layer in selected:
        if layer == "DEM":
            fname = f"{cog_stem}_DEM.tif"
            assets.append(("DEM", f"{base_url}/{folder}/{fname}", f"DEM/{fname}"))
        elif layer in ("WBM", "EDM", "FLM", "HEM"):
            fname = f"{cog_stem}_{layer}.tif"
            assets.append((layer, f"{base_url}/{folder}/AUXFILES/{fname}", f"{layer}/{fname}"))
        elif layer == "ACM":
            fname = f"{raw_stem}_ACM.kml"
            assets.append(("ACM", f"{base_url}/{folder}/AUXFILES/{fname}", f"ACM/{fname}"))
        elif layer == "XML":
            fname = f"{raw_stem}.xml"
            assets.append(("XML", f"{base_url}/{folder}/{fname}", f"XML/{fname}"))
        elif layer == "PREVIEW":
            for suffix in COPERNICUS_PREVIEW_SUFFIXES:
                fname = f"{raw_stem}_{suffix}"
                assets.append(("PREVIEW", f"{base_url}/{folder}/PREVIEW/{fname}", f"PREVIEW/{fname}"))
        elif layer == "INFO":
            assets.append(("INFO", f"{base_url}/{folder}/INFO/eula_F.pdf", "INFO/eula_F.pdf"))
    return assets


def tiles_for_bbox(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> List[Tuple[int, int]]:
    """Return sorted integer SW-corner `(lat, lon)` Copernicus tile indices covering bbox."""
    if lat_min > lat_max:
        raise ValueError(f"lat_min ({lat_min}) must be <= lat_max ({lat_max})")
    if lon_min > lon_max:
        raise ValueError(f"lon_min ({lon_min}) must be <= lon_max ({lon_max})")
    la0 = max(-90, math.floor(float(lat_min)))
    la1 = min(89, math.floor(float(lat_max)))
    lo0 = max(-180, math.floor(float(lon_min)))
    lo1 = min(179, math.floor(float(lon_max)))
    return [(lat, lon) for lat in range(la0, la1 + 1) for lon in range(lo0, lo1 + 1)]


def bbox_from_domain(domain_name: str, pad_deg: float = 0.5) -> Tuple[float, float, float, float]:
    """Compute `(lat_min, lat_max, lon_min, lon_max)` in [-180, 180] covering `domain_name`."""
    spec = get_domain_spec(domain_name, nz=65)
    lat = spec.lat2d
    lon = ((spec.lon2d + 180.0) % 360.0) - 180.0
    lat_min = max(-90.0, float(lat.min()) - float(pad_deg))
    lat_max = min(90.0, float(lat.max()) + float(pad_deg))
    if not spec.is_projected and spec.is_periodic_lon:
        lon_min, lon_max = -180.0, 179.999
    else:
        lon_min = max(-180.0, float(lon.min()) - float(pad_deg))
        lon_max = min(179.999, float(lon.max()) + float(pad_deg))
    return lat_min, lat_max, lon_min, lon_max


def parse_bbox_arg(bbox_str: str) -> Tuple[float, float, float, float]:
    parts = [float(x.strip()) for x in bbox_str.split(",") if x.strip()]
    if len(parts) != 4:
        raise ValueError(f"--bbox expects 'lat_min,lat_max,lon_min,lon_max', got {bbox_str!r}")
    lat_min, lat_max, lon_min, lon_max = parts
    if lat_min > lat_max or lon_min > lon_max:
        raise ValueError(f"Invalid bbox bounds: {parts}")
    return lat_min, lat_max, lon_min, lon_max


def gebco_erddap_url(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    *,
    dataset_id: str = "GEBCO_2020",
    stride: int = 1,
) -> str:
    """Build NOAA/PFEL ERDDAP regional NetCDF subset URL (vertical datum = MSL)."""
    query = urllib.parse.quote(
        f"elevation[({lat_min:.4f}):{int(stride)}:({lat_max:.4f})]"
        f"[({lon_min:.4f}):{int(stride)}:({lon_max:.4f})]",
        safe="():",
    )
    return f"{GEBCO_ERDDAP_BASE}/{dataset_id}.nc?{query}"


def emodnet_wcs_url(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    *,
    width: int = 2048,
    height: int = 2048,
    coverage: str = "emodnet:mean",
) -> str:
    """Build EMODnet Bathymetry WCS 1.0.0 GeoTIFF request (vertical datum = LAT)."""
    params = urllib.parse.urlencode(
        {
            "service": "WCS",
            "version": "1.0.0",
            "request": "GetCoverage",
            "coverage": coverage,
            "crs": "EPSG:4326",
            "bbox": f"{lon_min:.4f},{lat_min:.4f},{lon_max:.4f},{lat_max:.4f}",
            "width": str(int(width)),
            "height": str(int(height)),
            "format": "GeoTIFF",
        }
    )
    return f"{EMODNET_WCS_BASE}?{params}"


def _download_file(
    url: str,
    dest: Path,
    *,
    allow_404: bool = False,
    reject_xml: bool = False,
    resume: bool = True,
    timeout_s: float = 120.0,
    min_bytes: int = 128,
    retries: int = 3,
) -> str:
    """Atomically download `url` to `dest`. Returns `'cached'`, `'downloaded'`, or `'not_found'`."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size >= min_bytes:
        return "cached"

    def _validate_part(part_path: Path) -> None:
        if not part_path.is_file() or part_path.stat().st_size < min_bytes:
            raise RuntimeError(f"Downloaded file {dest} from {url} is unexpectedly small")
        if reject_xml:
            with open(part_path, "rb") as fh:
                head = fh.read(256)
            if head.lstrip().startswith(b"<"):
                part_path.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Server returned XML/HTML instead of binary data for {dest.name} from {url}: {head[:160]!r}"
                )

    part = dest.with_suffix(dest.suffix + ".part")
    attempts = max(1, int(retries))
    for attempt in range(1, attempts + 1):
        existing = part.stat().st_size if (resume and part.is_file()) else 0
        headers = {"User-Agent": USER_AGENT}
        if existing > 0:
            headers["Range"] = f"bytes={existing}-"

        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                status = getattr(resp, "status", 200)
                if existing > 0 and status == 206:
                    mode = "ab"
                    offset = existing
                else:
                    mode = "wb"
                    offset = 0

                total_hdr = resp.headers.get("Content-Length")
                total_bytes = (offset + int(total_hdr)) if total_hdr and total_hdr.isdigit() else None
                pbar = None
                if _tqdm is not None and total_bytes and total_bytes > 5 * 1024 * 1024:
                    pbar = _tqdm(
                        total=total_bytes,
                        initial=offset,
                        unit="B",
                        unit_scale=True,
                        desc=dest.name,
                        leave=False,
                    )
                try:
                    with open(part, mode) as fh:
                        while True:
                            chunk = resp.read(256 * 1024)
                            if not chunk:
                                break
                            fh.write(chunk)
                            if pbar is not None:
                                pbar.update(len(chunk))
                finally:
                    if pbar is not None:
                        pbar.close()
            break
        except urllib.error.HTTPError as exc:
            if allow_404 and exc.code in (403, 404):
                part.unlink(missing_ok=True)
                return "not_found"
            if exc.code == 416 and part.is_file() and part.stat().st_size >= min_bytes:
                _validate_part(part)
                part.replace(dest)
                return "downloaded"
            if exc.code >= 500 and attempt < attempts:
                if not resume:
                    part.unlink(missing_ok=True)
                time.sleep(float(attempt))
                continue
            raise
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt >= attempts:
                raise
            if not resume:
                part.unlink(missing_ok=True)
            time.sleep(float(attempt))

    _validate_part(part)
    part.replace(dest)
    return "downloaded"


def _safe_extract_zip(
    zip_path: Path,
    out_dir: Path,
    *,
    suffixes: Tuple[str, ...] = (".nc", ".pdf"),
) -> List[Path]:
    """Safely extract files matching `suffixes` from `zip_path` into `out_dir` without path traversal."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_resolved = out_dir.resolve()
    extracted: List[Path] = []
    suffixes_l = tuple(s.lower() for s in suffixes)
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            member_path = Path(info.filename)
            if member_path.is_absolute() or ".." in member_path.parts or not member_path.name:
                raise RuntimeError(f"Unsafe ZIP member path in {zip_path}: {info.filename!r}")
            if suffixes_l and not member_path.name.lower().endswith(suffixes_l):
                continue
            target = (out_dir / member_path.name).resolve()
            try:
                target.relative_to(out_resolved)
            except ValueError as exc:
                raise RuntimeError(f"ZIP extraction escapes target dir: {target}") from exc
            if target.is_file() and target.stat().st_size == info.file_size and info.file_size > 0:
                extracted.append(target)
                continue
            part = target.with_suffix(target.suffix + ".part")
            with zf.open(info, "r") as src, open(part, "wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            part.replace(target)
            extracted.append(target)
    return extracted


def fetch_copernicus_tile_index(resolution: str = "glo30", timeout_s: float = 30.0) -> Optional[Set[str]]:
    """Fetch public `tileList.txt` folder index for `glo30`/`glo90` (`None` if offline)."""
    res_key = resolution.lower().strip()
    if res_key not in COPERNICUS_URLS:
        raise ValueError(f"Unsupported Copernicus DEM resolution {resolution!r}")
    base_url, _, _ = COPERNICUS_URLS[res_key]
    url = f"{base_url}/tileList.txt"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            text = resp.read().decode("utf-8", errors="ignore")
    except (urllib.error.URLError, TimeoutError, OSError):
        return None
    folders = {line.strip().rstrip("/").split("/")[0] for line in text.splitlines() if line.strip()}
    return folders or None


def download_copernicus_dem(
    bbox: Tuple[float, float, float, float],
    dem_root: Path,
    *,
    include_glo30: bool = True,
    include_glo90: bool = False,
    layers: str | Iterable[str] = COPERNICUS_LAYER_PRESETS["standard"],
    use_tile_index: bool = True,
    workers: int = 4,
    dry_run: bool = False,
) -> Dict[str, int]:
    """Download Copernicus DEM & auxiliary layers covering `bbox` into `dem_root/{dem30_europe,dem90_world}`."""
    tiles = tiles_for_bbox(*bbox)
    selected_layers = parse_copernicus_layers(layers)
    resolutions: List[str] = []
    if include_glo30:
        resolutions.append("glo30")
    if include_glo90:
        resolutions.append("glo90")

    stats = {
        "tiles_checked": len(tiles) * len(resolutions),
        "downloaded": 0,
        "cached": 0,
        "ocean_404": 0,
        "layers": len(selected_layers),
    }
    if dry_run:
        return stats

    tile_layers = tuple(lyr for lyr in selected_layers if lyr != "INFO")
    want_info = "INFO" in selected_layers
    indices: Dict[str, Optional[Set[str]]] = {}
    if use_tile_index:
        for res_key in resolutions:
            indices[res_key] = fetch_copernicus_tile_index(res_key)

    def _fetch_one(res_key: str, lat: int, lon: int) -> Tuple[int, int, int, Optional[str]]:
        _, _, subdir = COPERNICUS_URLS[res_key]
        _, _, _, folder = _copernicus_tile_stems(lat, lon, res_key)
        idx = indices.get(res_key)
        if idx is not None and folder not in idx:
            return 0, 0, 1, None

        assets = copernicus_tile_asset_urls(lat, lon, res_key, tile_layers)
        _, dem_url, dem_rel = assets[0]
        st_dem = _download_file(dem_url, dem_root / subdir / dem_rel, allow_404=True, resume=False)
        if st_dem == "not_found":
            return 0, 0, 1, None

        dl = int(st_dem == "downloaded")
        ch = int(st_dem == "cached")
        for _, url, rel_path in assets[1:]:
            st = _download_file(url, dem_root / subdir / rel_path, allow_404=True, resume=False)
            dl += int(st == "downloaded")
            ch += int(st == "cached")
        return dl, ch, 0, folder

    first_valid_folder: Dict[str, str] = {}
    jobs = [(res_key, lat, lon) for res_key in resolutions for (lat, lon) in tiles]
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futs = {pool.submit(_fetch_one, rk, la, lo): rk for (rk, la, lo) in jobs}
        iterator = _tqdm(as_completed(futs), total=len(futs), desc="Copernicus DEM", unit="tile") if _tqdm else as_completed(futs)
        for fut in iterator:
            rk = futs[fut]
            dl, ch, oc, folder = fut.result()
            stats["downloaded"] += dl
            stats["cached"] += ch
            stats["ocean_404"] += oc
            if folder and rk not in first_valid_folder:
                first_valid_folder[rk] = folder

    if want_info:
        for res_key, folder in first_valid_folder.items():
            base_url, _, subdir = COPERNICUS_URLS[res_key]
            st = _download_file(
                f"{base_url}/{folder}/INFO/eula_F.pdf",
                dem_root / subdir / "INFO" / "eula_F.pdf",
                allow_404=True,
                resume=False,
            )
            stats["downloaded"] += int(st == "downloaded")
            stats["cached"] += int(st == "cached")
    return stats


def download_bathymetry(
    bbox: Tuple[float, float, float, float],
    bathy_root: Path,
    *,
    source: str = "gebco-erddap",
    label: str = "custom",
    dry_run: bool = False,
) -> Path:
    """Download bathymetry source matching `tempest_v2.bathymetry` and `ic_builder`."""
    bathy_root.mkdir(parents=True, exist_ok=True)
    tag = label.lower().replace("-", "_")

    if source == "gebco-erddap":
        url = gebco_erddap_url(*bbox)
        out = bathy_root / f"GEBCO_regional_{tag}.nc"
        if not dry_run:
            _download_file(url, out, resume=False, timeout_s=300.0)
        return out

    if source == "gebco-2026-global":
        out = bathy_root / "GEBCO_2026" / "GEBCO_2026.nc"
        if not dry_run:
            _download_file(GEBCO_CEDA_2026_NC_URL, out, resume=True, timeout_s=600.0)
        return out

    if source == "gebco-2026-zip":
        out_dir = bathy_root / "GEBCO_2026"
        zip_path = out_dir / "GEBCO_2026.zip"
        if not dry_run:
            _download_file(GEBCO_CEDA_2026_ZIP_URL, zip_path, resume=True, timeout_s=600.0, min_bytes=1_000_000)
            _safe_extract_zip(zip_path, out_dir, suffixes=(".nc", ".pdf"))
        return out_dir / "GEBCO_2026.nc"

    if source == "gebco-2026-full":
        archives = (
            (GEBCO_CEDA_2026_ZIP_URL, bathy_root / "GEBCO_2026" / "GEBCO_2026.zip", bathy_root / "GEBCO_2026"),
            (
                GEBCO_CEDA_2026_SUB_ICE_ZIP_URL,
                bathy_root / "GEBCO_2026_sub_ice" / "GEBCO_2026_sub_ice.zip",
                bathy_root / "GEBCO_2026_sub_ice",
            ),
            (
                GEBCO_CEDA_2026_TID_ZIP_URL,
                bathy_root / "gebco_2026_tid" / "gebco_2026_tid.zip",
                bathy_root / "gebco_2026_tid",
            ),
        )
        if not dry_run:
            for url, zip_path, out_dir in archives:
                _download_file(url, zip_path, resume=True, timeout_s=600.0, min_bytes=1_000_000)
                _safe_extract_zip(zip_path, out_dir, suffixes=(".nc", ".pdf"))
        return bathy_root / "GEBCO_2026" / "GEBCO_2026.nc"

    if source == "gebco-2025-zip":
        zip_path = bathy_root / "gebco_2025.zip"
        out_dir = bathy_root / "GEBCO_2025"
        if not dry_run:
            _download_file(GEBCO_CEDA_2025_ZIP_URL, zip_path, resume=True, timeout_s=600.0)
            _safe_extract_zip(zip_path, out_dir, suffixes=(".nc",))
        return out_dir / "GEBCO_2025.nc"

    if source == "emodnet-wcs":
        lat_min, lat_max, lon_min, lon_max = bbox
        dlat = max(0.25, lat_max - lat_min)
        dlon = max(0.25, lon_max - lon_min)
        # Keep single-request dimensions bounded cleanly for WCS server stability
        height = max(256, min(4096, int(round(dlat * 240))))
        width = max(256, min(4096, int(round(dlon * 240 * math.cos(math.radians(0.5 * (lat_min + lat_max)))))))
        url = emodnet_wcs_url(lat_min, lat_max, lon_min, lon_max, width=width, height=height)
        out = bathy_root / f"emodnet_lat_{tag}.tif"
        if not dry_run:
            _download_file(url, out, reject_xml=True, resume=False, timeout_s=300.0)
            sidecar = out.with_suffix(".json")
            sidecar.write_text(
                json.dumps(
                    {
                        "source": "EMODnet Bathymetry Digital Terrain Model (emodnet:mean)",
                        "wcs_url": url,
                        "vertical_datum": "lat",
                        "source_values": "elevation_m",
                        "bbox_lat_lon": [lat_min, lat_max, lon_min, lon_max],
                        "downloaded_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "note": "EMODnet heights are referenced to LAT; convert to MSL before mixing with MSL datasets.",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        return out

    raise ValueError(f"Unsupported --bathy-source {source!r}")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            blk = fh.read(1024 * 1024)
            if not blk:
                break
            h.update(blk)
    return h.hexdigest()


def download_hwsd_v22(hwsd_root: Path, *, dry_run: bool = False) -> Dict[str, str]:
    """Download FAO/IIASA HWSD v2.2 ZIP archives into `hwsd_root/downloads` and verify ZIP CRCs."""
    dl_dir = hwsd_root / "downloads"
    results: Dict[str, str] = {}
    for entry in HWSD_V22_FILES:
        dest = dl_dir / entry["filename"]
        if dry_run:
            results[entry["filename"]] = f"dry-run -> {dest}"
            continue
        state = _download_file(entry["url"], dest, resume=True, timeout_s=300.0, min_bytes=1_000_000)
        with zipfile.ZipFile(dest) as zf:
            bad = zf.testzip()
            if bad:
                raise RuntimeError(f"ZIP CRC error in {dest} at {bad}")
            members = [m.lower() for m in zf.namelist()]
            if not any(m.endswith(entry["required_suffix"]) for m in members):
                raise RuntimeError(f"{dest} missing expected {entry['required_suffix']} payload")
        digest = _sha256_file(dest)
        sha_ok = digest == entry["sha256"]
        results[entry["filename"]] = f"{state} (zip CRC ok, sha256_match={sha_ok})"

    if not dry_run:
        has_mdb = shutil.which("mdb-tables") is not None and shutil.which("mdb-export") is not None
        if not has_mdb:
            print(
                "[WARNING] `mdb-tables`/`mdb-export` not found on PATH. "
                "Install OS package `mdbtools` (e.g. `sudo apt install -y mdbtools`) before preparing HWSD v2.2.",
                file=sys.stderr,
            )
    return results


def worldcover_tile_url(tile_id: str, year: int = 2021) -> Tuple[str, str]:
    """Return `(url, filename)` for an ESA WorldCover 10 m GeoTIFF tile."""
    if year not in WORLDCOVER_VERSIONS:
        raise ValueError(f"Unsupported WorldCover year {year}; expected one of {tuple(WORLDCOVER_VERSIONS)}")
    version = WORLDCOVER_VERSIONS[year]
    fname = f"ESA_WorldCover_10m_{year}_{version}_{tile_id}_Map.tif"
    return f"{WORLDCOVER_BASE_URL}/{version}/{year}/map/{fname}", fname


def worldcover_tiles_for_bbox_fallback(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> List[str]:
    """Return deterministic 3-degree SW-corner WorldCover tile IDs covering `bbox`."""
    if lat_min > lat_max or lon_min > lon_max:
        raise ValueError("Invalid bbox bounds for WorldCover")
    la0 = max(-90, 3 * math.floor(float(lat_min) / 3.0))
    la1 = min(87, 3 * math.floor(float(lat_max) / 3.0))
    lo0 = max(-180, 3 * math.floor(float(lon_min) / 3.0))
    lo1 = min(177, 3 * math.floor(float(lon_max) / 3.0))
    tiles: List[str] = []
    for la in range(la0, la1 + 1, 3):
        for lo in range(lo0, lo1 + 1, 3):
            ns = "N" if la >= 0 else "S"
            ew = "E" if lo >= 0 else "W"
            tiles.append(f"{ns}{abs(la):02d}{ew}{abs(lo):03d}")
    return tiles


def _iter_geojson_coords(coords: object) -> Iterable[Tuple[float, float]]:
    if isinstance(coords, (list, tuple)):
        if len(coords) >= 2 and isinstance(coords[0], (int, float)) and isinstance(coords[1], (int, float)):
            yield float(coords[0]), float(coords[1])
        else:
            for item in coords:
                yield from _iter_geojson_coords(item)


def worldcover_tiles_from_geojson(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    *,
    grid_cache: Optional[Path] = None,
    allow_network: bool = True,
    timeout_s: float = 30.0,
) -> List[str]:
    """Resolve WorldCover land tile IDs from official GeoJSON grid (with offline 3-deg fallback)."""
    gj: Optional[dict] = None
    if grid_cache is not None and grid_cache.is_file() and grid_cache.stat().st_size > 1024:
        try:
            gj = json.loads(grid_cache.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            gj = None

    if gj is None and allow_network:
        try:
            if grid_cache is not None:
                _download_file(WORLDCOVER_GRID_URL, grid_cache, resume=False, timeout_s=timeout_s, min_bytes=1024)
                gj = json.loads(grid_cache.read_text(encoding="utf-8"))
            else:
                req = urllib.request.Request(WORLDCOVER_GRID_URL, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                    gj = json.loads(resp.read().decode("utf-8"))
        except Exception:
            gj = None

    if gj is None:
        return worldcover_tiles_for_bbox_fallback(lat_min, lat_max, lon_min, lon_max)

    matched: List[str] = []
    for feat in gj.get("features", []):
        props = feat.get("properties") or {}
        tile_id = props.get("ll_tile")
        geom = feat.get("geometry") or {}
        pts = list(_iter_geojson_coords(geom.get("coordinates")))
        if not tile_id or not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        if max(xs) >= lon_min and min(xs) <= lon_max and max(ys) >= lat_min and min(ys) <= lat_max:
            matched.append(str(tile_id))
    return sorted(set(matched)) if matched else worldcover_tiles_for_bbox_fallback(lat_min, lat_max, lon_min, lon_max)


def download_worldcover(
    bbox: Tuple[float, float, float, float],
    worldcover_root: Path,
    *,
    year: int = 2021,
    workers: int = 4,
    dry_run: bool = False,
) -> Dict[str, int]:
    """Download ESA WorldCover 10 m tiles covering `bbox` into `worldcover_root`."""
    grid_cache = worldcover_root / "esa_worldcover_grid.geojson"
    tiles = worldcover_tiles_from_geojson(
        *bbox,
        grid_cache=grid_cache,
        allow_network=not dry_run,
    )
    stats = {"tiles": len(tiles), "downloaded": 0, "cached": 0, "not_found": 0}
    if dry_run:
        return stats

    worldcover_root.mkdir(parents=True, exist_ok=True)

    def _fetch_wc(tile_id: str) -> str:
        url, fname = worldcover_tile_url(tile_id, year=year)
        return _download_file(url, worldcover_root / fname, allow_404=True, resume=True, timeout_s=300.0)

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futs = [pool.submit(_fetch_wc, t) for t in tiles]
        iterator = _tqdm(as_completed(futs), total=len(futs), desc=f"WorldCover {year}", unit="tile") if _tqdm else as_completed(futs)
        for fut in iterator:
            st = fut.result()
            stats["downloaded"] += int(st == "downloaded")
            stats["cached"] += int(st == "cached")
            stats["not_found"] += int(st == "not_found")
    return stats


def soilgrids_jobs(mode: str = "tempest", include_wrb: bool = True) -> List[Tuple[str, str, str]]:
    """Return `[(map_name, coverage_id, filename), ...]` for `tempest` (14), `surface` (10), or `full` (55)."""
    mode_l = mode.lower().strip()
    if mode_l not in ("tempest", "surface", "full"):
        raise ValueError(f"Unsupported SoilGrids mode {mode!r}; expected 'tempest', 'surface', or 'full'")

    jobs: List[Tuple[str, str, str]] = []
    if mode_l == "full":
        for prop in SOILGRIDS_PROPERTIES:
            for depth in SOILGRIDS_DEPTHS:
                jobs.append((prop, f"{prop}_{depth}_Q0.5", f"SoilGrids_{prop}_{depth}_Q0p5.tif"))
    elif mode_l == "surface":
        for prop in SOILGRIDS_PROPERTIES:
            jobs.append((prop, f"{prop}_0-5cm_Q0.5", f"SoilGrids_{prop}_0-5cm_Q0p5.tif"))
    else:
        for prop, depths in TEMPEST_SOILGRIDS_DEPTHS.items():
            for depth in depths:
                jobs.append((prop, f"{prop}_{depth}_Q0.5", f"SoilGrids_{prop}_{depth}_Q0p5.tif"))

    if include_wrb:
        jobs.append(("wrb", "MostProbable", "SoilGrids_WRB_MostProbable.tif"))
    return jobs


def soilgrids_wcs_url(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    *,
    map_name: str,
    coverage_id: str,
    max_dim: int = 8192,
) -> str:
    """Build ISRIC SoilGrids 250 m WCS 2.0.1 GeoTIFF request in EPSG:4326."""
    params = (
        f"map=/map/{urllib.parse.quote(map_name)}.map"
        f"&SERVICE=WCS&VERSION=2.0.1&REQUEST=GetCoverage"
        f"&COVERAGEID={urllib.parse.quote(coverage_id)}"
        f"&FORMAT=GEOTIFF_INT16"
        f"&SUBSET=X({lon_min:.4f},{lon_max:.4f})"
        f"&SUBSET=Y({lat_min:.4f},{lat_max:.4f})"
        f"&SUBSETTINGCRS=http://www.opengis.net/def/crs/EPSG/0/4326"
        f"&OUTPUTCRS=http://www.opengis.net/def/crs/EPSG/0/4326"
    )
    est_w = max(1, int(round((lon_max - lon_min) / 0.00225)))
    est_h = max(1, int(round((lat_max - lat_min) / 0.00225)))
    if est_w > max_dim or est_h > max_dim:
        scale = min(max_dim / est_w, max_dim / est_h)
        w = max(64, int(round(est_w * scale)))
        h = max(64, int(round(est_h * scale)))
        params += f"&SCALESIZE=X({w}),Y({h})"
    return f"{SOILGRIDS_WCS_BASE}?{params}"


def download_soilgrids(
    bbox: Tuple[float, float, float, float],
    soilgrids_root: Path,
    *,
    mode: str = "tempest",
    include_wrb: bool = True,
    delay_s: float = 0.35,
    dry_run: bool = False,
) -> Dict[str, int]:
    """Download ISRIC SoilGrids 250 m WCS layers for `bbox` into `soilgrids_root`."""
    jobs = soilgrids_jobs(mode=mode, include_wrb=include_wrb)
    stats = {"layers": len(jobs), "downloaded": 0, "cached": 0}
    if dry_run:
        return stats

    soilgrids_root.mkdir(parents=True, exist_ok=True)
    iterator = _tqdm(jobs, desc=f"SoilGrids ({mode})", unit="layer") if _tqdm else jobs
    for map_name, cov_id, fname in iterator:
        url = soilgrids_wcs_url(*bbox, map_name=map_name, coverage_id=cov_id)
        st = _download_file(
            url,
            soilgrids_root / fname,
            reject_xml=True,
            resume=False,
            timeout_s=300.0,
            min_bytes=256,
        )
        stats["downloaded"] += int(st == "downloaded")
        stats["cached"] += int(st == "cached")
        if st == "downloaded" and delay_s > 0.0:
            time.sleep(delay_s)
    return stats


def fabdem_group_name(lat: int, lon: int) -> str:
    """Return 10-degree FABDEM v1.2 HuggingFace directory name containing tile `(lat, lon)`."""
    la0 = 10 * math.floor(int(lat) / 10)
    la1 = la0 + 10
    lo0 = 10 * math.floor(int(lon) / 10)
    lo1 = lo0 + 10

    def _fmt_lat(val: int) -> str:
        return f"{'N' if val >= 0 else 'S'}{abs(val):02d}"

    def _fmt_lon(val: int) -> str:
        return f"{'E' if val >= 0 else 'W'}{abs(val):03d}"

    return f"{_fmt_lat(la0)}{_fmt_lon(lo0)}-{_fmt_lat(la1)}{_fmt_lon(lo1)}_FABDEM_V1-2"


def fabdem_tile_url(lat: int, lon: int) -> Tuple[str, str]:
    """Return `(url, filename)` for a 1-degree FABDEM v1.2 bare-earth DEM tile."""
    ns = "N" if int(lat) >= 0 else "S"
    ew = "E" if int(lon) >= 0 else "W"
    fname = f"{ns}{abs(int(lat)):02d}{ew}{abs(int(lon)):03d}_FABDEM_V1-2.tif"
    group = fabdem_group_name(lat, lon)
    return f"{FABDEM_BASE_URL}/{group}/{fname}", fname


def download_fabdem(
    bbox: Tuple[float, float, float, float],
    fabdem_root: Path,
    *,
    workers: int = 4,
    dry_run: bool = False,
) -> Dict[str, int]:
    """Download optional FABDEM v1.2 bare-earth DEM tiles covering `bbox`."""
    tiles = tiles_for_bbox(*bbox)
    stats = {"tiles_checked": len(tiles), "downloaded": 0, "cached": 0, "ocean_404": 0}
    if dry_run:
        return stats

    fabdem_root.mkdir(parents=True, exist_ok=True)

    def _fetch_fab(lat: int, lon: int) -> str:
        url, fname = fabdem_tile_url(lat, lon)
        return _download_file(url, fabdem_root / fname, allow_404=True, resume=True, timeout_s=300.0)

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futs = [pool.submit(_fetch_fab, la, lo) for (la, lo) in tiles]
        iterator = _tqdm(as_completed(futs), total=len(futs), desc="FABDEM v1.2", unit="tile") if _tqdm else as_completed(futs)
        for fut in iterator:
            st = fut.result()
            stats["downloaded"] += int(st == "downloaded")
            stats["cached"] += int(st == "cached")
            stats["ocean_404"] += int(st == "not_found")
    return stats


def _clip_nl_bbox(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> Tuple[float, float, float, float]:
    la0 = max(float(lat_min), NL_BOUNDS[0])
    la1 = min(float(lat_max), NL_BOUNDS[1])
    lo0 = max(float(lon_min), NL_BOUNDS[2])
    lo1 = min(float(lon_max), NL_BOUNDS[3])
    if la0 >= la1 or lo0 >= lo1:
        raise ValueError(
            f"Bounding box ({lat_min}, {lat_max}, {lon_min}, {lon_max}) does not overlap Netherlands AHN bounds {NL_BOUNDS}"
        )
    return la0, la1, lo0, lo1


def ahn_wcs_url(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    *,
    product: str = "dtm_05m",
    resolution_m: float = 5.0,
    max_dim: int = 4096,
) -> str:
    """Build PDOK AHN WCS 2.0.1 GeoTIFF URL clipped to the Netherlands in EPSG:4326."""
    if product not in ("dtm_05m", "dsm_05m"):
        raise ValueError(f"Unsupported AHN product {product!r}; expected 'dtm_05m' or 'dsm_05m'")
    if resolution_m <= 0.0:
        raise ValueError(f"AHN resolution_m must be > 0, got {resolution_m}")
    la0, la1, lo0, lo1 = _clip_nl_bbox(lat_min, lat_max, lon_min, lon_max)
    mid_lat_rad = math.radians(0.5 * (la0 + la1))
    meters_y = (la1 - la0) * 111_320.0
    meters_x = (lo1 - lo0) * 111_320.0 * max(0.1, math.cos(mid_lat_rad))
    height = max(64, min(max_dim, int(round(meters_y / resolution_m))))
    width = max(64, min(max_dim, int(round(meters_x / resolution_m))))
    return (
        f"{AHN_WCS_BASE}?SERVICE=WCS&VERSION=2.0.1&REQUEST=GetCoverage"
        f"&COVERAGEID={urllib.parse.quote(product)}"
        f"&FORMAT=image/tiff"
        f"&SUBSET=x({lo0:.4f},{lo1:.4f})"
        f"&SUBSET=y({la0:.4f},{la1:.4f})"
        f"&SUBSETTINGCRS=http://www.opengis.net/def/crs/EPSG/0/4326"
        f"&OUTPUTCRS=http://www.opengis.net/def/crs/EPSG/0/4326"
        f"&SCALESIZE=x({width}),y({height})"
    )


def download_ahn(
    bbox: Tuple[float, float, float, float],
    ahn_root: Path,
    *,
    product: str = "dtm_05m",
    resolution_m: float = 5.0,
    label: str = "nl",
    dry_run: bool = False,
) -> Path:
    """Download optional PDOK AHN WCS GeoTIFF clipped to the Netherlands."""
    url = ahn_wcs_url(*bbox, product=product, resolution_m=resolution_m)
    tag = label.lower().replace("-", "_")
    out = ahn_root / f"ahn_{product}_{tag}.tif"
    if not dry_run:
        _download_file(url, out, reject_xml=True, resume=False, timeout_s=300.0, min_bytes=256)
    return out


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Download static geophysical data for Tempest-NWP v2.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download Copernicus DEM GLO-30, Bathymetry, HWSD v2.2, ESA WorldCover, and SoilGrids",
    )
    parser.add_argument("--copernicus-dem", action="store_true", help="Download Copernicus DEM GLO-30 + auxiliary masks")
    parser.add_argument(
        "--copernicus-layers",
        type=str,
        default="standard",
        help="Preset (minimal, standard, all) or comma-separated layers: DEM,WBM,EDM,FLM,HEM,ACM,XML,PREVIEW,INFO",
    )
    parser.add_argument("--glo90", action="store_true", help="Also download Copernicus DEM GLO-90 fallback tiles")
    parser.add_argument("--bathymetry", action="store_true", help="Download ocean/coastal bathymetry")
    parser.add_argument(
        "--bathy-source",
        choices=(
            "gebco-erddap",
            "gebco-2026-global",
            "gebco-2026-zip",
            "gebco-2026-full",
            "gebco-2025-zip",
            "emodnet-wcs",
        ),
        default="gebco-erddap",
        help="Bathymetry source (default: gebco-erddap regional MSL subset; use gebco-2026-full for full global 2026 archives)",
    )
    parser.add_argument("--hwsd", action="store_true", help="Download FAO/IIASA HWSD v2.2 soil database & raster")
    parser.add_argument("--worldcover", action="store_true", help="Download ESA WorldCover 10 m land-cover GeoTIFFs")
    parser.add_argument("--worldcover-year", type=int, choices=(2020, 2021), default=2021, help="WorldCover vintage")
    parser.add_argument("--soilgrids", action="store_true", help="Download ISRIC SoilGrids 250 m WCS soil layers")
    parser.add_argument(
        "--soilgrids-mode",
        choices=("tempest", "surface", "full"),
        default="tempest",
        help="SoilGrids layer set: tempest (14 layers used by Tempest), surface (10 layers), or full (55 layers)",
    )
    parser.add_argument("--no-wrb", action="store_true", help="Skip SoilGrids WRB MostProbable layer")
    parser.add_argument("--fabdem", action="store_true", help="Optional opt-in: download FABDEM v1.2 bare-earth DEM")
    parser.add_argument("--ahn", action="store_true", help="Optional opt-in: download PDOK AHN DTM (NL only)")
    parser.add_argument("--ahn-product", choices=("dtm_05m", "dsm_05m"), default="dtm_05m")
    parser.add_argument("--ahn-resolution", type=float, default=5.0, help="Target AHN pixel spacing in meters (default: 5)")
    parser.add_argument(
        "--domain",
        type=str,
        default="lam_bnl_ns_de_2p5km",
        help="Target Tempest domain for bbox slicing and --prepare-wx1 (default: lam_bnl_ns_de_2p5km)",
    )
    parser.add_argument(
        "--bbox",
        type=str,
        default=None,
        help="Explicit 'lat_min,lat_max,lon_min,lon_max' bounding box overriding --domain bounds",
    )
    parser.add_argument("--dem-dir", type=Path, default=None, help="Output dir for Copernicus DEM tiles")
    parser.add_argument("--bathy-dir", type=Path, default=None, help="Output dir for raw bathymetry")
    parser.add_argument("--hwsd-dir", type=Path, default=None, help="Output dir for FAO HWSD v2.2")
    parser.add_argument("--worldcover-dir", type=Path, default=None, help="Output dir for ESA WorldCover")
    parser.add_argument("--soilgrids-dir", type=Path, default=None, help="Output dir for ISRIC SoilGrids 250 m")
    parser.add_argument("--fabdem-dir", type=Path, default=None, help="Output dir for optional FABDEM v1.2")
    parser.add_argument("--ahn-dir", type=Path, default=None, help="Output dir for optional PDOK AHN")
    parser.add_argument("--workers", type=int, default=4, help="Parallel tile download workers (default: 4)")
    parser.add_argument("--prepare-wx1", action="store_true", help="Compile static WX-1 containers for --domain")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved URLs/counts without downloading")
    args = parser.parse_args(argv)

    want_dem = args.all or args.copernicus_dem or args.glo90
    want_bathy = args.all or args.bathymetry
    want_hwsd = args.all or args.hwsd
    want_wc = args.all or args.worldcover
    want_sg = args.all or args.soilgrids
    want_fab = args.fabdem
    want_ahn = args.ahn
    if not (want_dem or want_bathy or want_hwsd or want_wc or want_sg or want_fab or want_ahn or args.prepare_wx1):
        parser.print_help()
        return

    static_root = PROJECT_ROOT / "data" / "static"
    dem_root = Path(args.dem_dir or os.environ.get("TEMPEST_COPERNICUS_DEM_DIR") or (static_root / "copernicus_dem_raw"))
    bathy_root = Path(args.bathy_dir or os.environ.get("TEMPEST_BATHY_DIR") or (static_root / "bathymetry_raw"))
    hwsd_root = Path(args.hwsd_dir or os.environ.get("TEMPEST_HWSD_V22_DIR") or (static_root / "hwsd_v22"))
    wc_default = "esa_worldcover_2021" if args.worldcover_year == 2021 else "esa_worldcover_2020"
    wc_root = Path(args.worldcover_dir or os.environ.get("TEMPEST_WORLDCOVER_DIR") or (static_root / wc_default))
    sg_root = Path(args.soilgrids_dir or os.environ.get("TEMPEST_SOILGRIDS_DIR") or (static_root / "soilgrids_250m"))
    fab_root = Path(args.fabdem_dir or (static_root / "fabdem_v12"))
    ahn_root = Path(args.ahn_dir or (static_root / "pdok_ahn"))

    bbox = parse_bbox_arg(args.bbox) if args.bbox else bbox_from_domain(args.domain)
    print(f"Target domain={args.domain} | bbox=(lat {bbox[0]:.2f}..{bbox[1]:.2f}, lon {bbox[2]:.2f}..{bbox[3]:.2f})")

    if want_dem:
        cop_layers = parse_copernicus_layers(args.copernicus_layers)
        res_stats = download_copernicus_dem(
            bbox,
            dem_root,
            include_glo30=args.copernicus_dem or args.all or not args.glo90,
            include_glo90=args.glo90,
            layers=cop_layers,
            workers=args.workers,
            dry_run=args.dry_run,
        )
        print(f"Copernicus DEM ({dem_root}, layers={','.join(cop_layers)}): {res_stats}")

    if want_bathy:
        if args.dry_run:
            preview_map = {
                "gebco-erddap": gebco_erddap_url(*bbox),
                "emodnet-wcs": emodnet_wcs_url(*bbox),
                "gebco-2026-global": GEBCO_CEDA_2026_NC_URL,
                "gebco-2026-zip": GEBCO_CEDA_2026_ZIP_URL,
                "gebco-2026-full": (
                    f"{GEBCO_CEDA_2026_ZIP_URL} + {GEBCO_CEDA_2026_SUB_ICE_ZIP_URL} + {GEBCO_CEDA_2026_TID_ZIP_URL}"
                ),
                "gebco-2025-zip": GEBCO_CEDA_2025_ZIP_URL,
            }
            print(f"Bathymetry ({args.bathy_source}): dry-run URL -> {preview_map[args.bathy_source]}")
        else:
            bathy_out = download_bathymetry(bbox, bathy_root, source=args.bathy_source, label=args.domain)
            print(f"Bathymetry ({args.bathy_source}): saved -> {bathy_out}")

    if want_hwsd:
        hwsd_stats = download_hwsd_v22(hwsd_root, dry_run=args.dry_run)
        print(f"FAO HWSD v2.2 ({hwsd_root}): {hwsd_stats}")

    if want_wc:
        wc_stats = download_worldcover(
            bbox,
            wc_root,
            year=args.worldcover_year,
            workers=args.workers,
            dry_run=args.dry_run,
        )
        print(f"ESA WorldCover {args.worldcover_year} ({wc_root}): {wc_stats}")

    if want_sg:
        sg_stats = download_soilgrids(
            bbox,
            sg_root,
            mode=args.soilgrids_mode,
            include_wrb=not args.no_wrb,
            dry_run=args.dry_run,
        )
        print(f"ISRIC SoilGrids 250m ({sg_root}, mode={args.soilgrids_mode}): {sg_stats}")

    if want_fab:
        fab_stats = download_fabdem(bbox, fab_root, workers=args.workers, dry_run=args.dry_run)
        print(f"FABDEM v1.2 ({fab_root}): {fab_stats}")

    if want_ahn:
        if args.dry_run:
            print(f"PDOK AHN ({args.ahn_product}): dry-run URL -> {ahn_wcs_url(*bbox, product=args.ahn_product, resolution_m=args.ahn_resolution)}")
        else:
            ahn_out = download_ahn(
                bbox,
                ahn_root,
                product=args.ahn_product,
                resolution_m=args.ahn_resolution,
                label=args.domain,
            )
            print(f"PDOK AHN ({args.ahn_product}): saved -> {ahn_out}")

    if args.prepare_wx1 and not args.dry_run:
        os.environ["TEMPEST_COPERNICUS_DEM_DIR"] = str(dem_root)
        os.environ["TEMPEST_BATHY_DIR"] = str(bathy_root)
        os.environ["TEMPEST_HWSD_V22_DIR"] = str(hwsd_root)
        os.environ["TEMPEST_WORLDCOVER_DIR"] = str(wc_root)
        os.environ["TEMPEST_SOILGRIDS_DIR"] = str(sg_root)
        from tempest_v2.ic_builder import build_domain_static_fields

        spec = get_domain_spec(args.domain)
        hs_m, surf = build_domain_static_fields(
            spec,
            dem_root=dem_root,
            gebco_root=bathy_root,
            worldcover_root=wc_root if wc_root.exists() else None,
            soilgrids_root=sg_root if sg_root.exists() else None,
            force=True,
            allow_synthetic_fallback=False,
        )
        print(
            f"Prepared static WX-1 for {spec.name}: "
            f"max_oro={float(hs_m.max()):.1f}m, max_bathy={float(surf['bathy_depth_m'].max()):.1f}m"
        )


if __name__ == "__main__":
    main()
