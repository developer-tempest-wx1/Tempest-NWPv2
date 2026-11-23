"""Deterministic Static Geophysical Data Downloader for Tempest-NWP v2.

Downloads official open-access static datasets into the exact layout expected by
`tempest_v2.copernicus_dem`, `tempest_v2.bathymetry`, `tempest_v2.hwsd22_soil`,
and `tempest_v2.ic_builder` (no credentials required for Copernicus DEM, EMODnet,
GEBCO ERDDAP/CEDA, or FAO HWSD v2.2):

  1. Copernicus DEM GLO-30 (30 m) & GLO-90 (90 m) + Water Body Mask (`WBM.tif`)
     from public anonymous AWS S3 (`copernicus-dem-30m` / `copernicus-dem-90m`).
  2. Bathymetry:
     - `gebco-erddap` (default): regional MSL NetCDF subset via NOAA/PFEL ERDDAP (`GEBCO_2020`).
     - `gebco-2026-global`: full global `GEBCO_2026.nc` (~7.0 GiB, MSL) via CEDA/BODC.
     - `gebco-2025-zip`: full global `gebco_2025.zip` (~4.0 GiB, MSL) via CEDA/BODC.
     - `emodnet-wcs`: high-res European GeoTIFF subset via EMODnet Bathymetry WCS (LAT datum!).
  3. FAO / IIASA HWSD v2.2 (`hwsd2.2_db.zip` + `hwsd2.2_raster.zip`) + `mdbtools` check.
  4. Optional `--prepare-wx1` pass converting raw sources into lossless `zstd_shuffle` `.wx1`.

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
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from typing import Dict, List, Optional, Tuple

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

GEBCO_CEDA_2026_NC_URL = "https://dap.ceda.ac.uk/bodc/gebco/global/gebco_2026/ice_surface_elevation/netcdf/GEBCO_2026.nc"
GEBCO_CEDA_2025_ZIP_URL = "https://dap.ceda.ac.uk/bodc/gebco/global/gebco_2025/ice_surface_elevation/netcdf/gebco_2025.zip"
GEBCO_ERDDAP_BASE = "https://coastwatch.pfeg.noaa.gov/erddap/griddap"
EMODNET_WCS_BASE = "https://ows.emodnet-bathymetry.eu/wcs"

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


def copernicus_tile_urls(lat: int, lon: int, resolution: str = "glo30") -> Tuple[str, str, str, str]:
    """Return `(dem_url, wbm_url, dem_filename, wbm_filename)` for integer SW tile `(lat, lon)`."""
    res_key = resolution.lower().strip()
    if res_key not in COPERNICUS_URLS:
        raise ValueError(f"Unsupported Copernicus DEM resolution {resolution!r}; expected 'glo30' or 'glo90'")
    base_url, tag, _ = COPERNICUS_URLS[res_key]
    ns = "N" if int(lat) >= 0 else "S"
    ew = "E" if int(lon) >= 0 else "W"
    stem = f"Copernicus_DSM_COG_{tag}_{ns}{abs(int(lat)):02d}_00_{ew}{abs(int(lon)):03d}_00"
    dem_file = f"{stem}_DEM.tif"
    wbm_file = f"{stem}_WBM.tif"
    folder = f"{stem}_DEM"
    dem_url = f"{base_url}/{folder}/{dem_file}"
    wbm_url = f"{base_url}/{folder}/AUXFILES/{wbm_file}"
    return dem_url, wbm_url, dem_file, wbm_file


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
    resume: bool = True,
    timeout_s: float = 120.0,
    min_bytes: int = 128,
) -> str:
    """Atomically download `url` to `dest`. Returns `'cached'`, `'downloaded'`, or `'not_found'`."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size >= min_bytes:
        return "cached"

    part = dest.with_suffix(dest.suffix + ".part")
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
    except urllib.error.HTTPError as exc:
        if allow_404 and exc.code in (403, 404):
            if part.is_file():
                part.unlink()
            return "not_found"
        if exc.code == 416 and part.is_file() and part.stat().st_size >= min_bytes:
            part.replace(dest)
            return "downloaded"
        raise

    if not part.is_file() or part.stat().st_size < min_bytes:
        raise RuntimeError(f"Downloaded file {dest} from {url} is unexpectedly small")
    part.replace(dest)
    return "downloaded"


def download_copernicus_dem(
    bbox: Tuple[float, float, float, float],
    dem_root: Path,
    *,
    include_glo30: bool = True,
    include_glo90: bool = False,
    workers: int = 4,
    dry_run: bool = False,
) -> Dict[str, int]:
    """Download Copernicus DEM & WBM tiles covering `bbox` into `dem_root/{dem30_europe,dem90_world}`."""
    tiles = tiles_for_bbox(*bbox)
    resolutions: List[str] = []
    if include_glo30:
        resolutions.append("glo30")
    if include_glo90:
        resolutions.append("glo90")

    stats = {"tiles_checked": len(tiles) * len(resolutions), "downloaded": 0, "cached": 0, "ocean_404": 0}
    if dry_run:
        return stats

    def _fetch_one(res_key: str, lat: int, lon: int) -> Tuple[int, int, int]:
        _, _, subdir = COPERNICUS_URLS[res_key]
        dem_url, wbm_url, dem_name, wbm_name = copernicus_tile_urls(lat, lon, res_key)
        dem_path = dem_root / subdir / "DEM" / dem_name
        wbm_path = dem_root / subdir / "WBM" / wbm_name
        st_dem = _download_file(dem_url, dem_path, allow_404=True, resume=False)
        if st_dem == "not_found":
            return 0, 0, 1
        st_wbm = _download_file(wbm_url, wbm_path, allow_404=True, resume=False)
        dl = int(st_dem == "downloaded") + int(st_wbm == "downloaded")
        ch = int(st_dem == "cached") + int(st_wbm == "cached")
        return dl, ch, 0

    jobs = [(res_key, lat, lon) for res_key in resolutions for (lat, lon) in tiles]
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futs = [pool.submit(_fetch_one, rk, la, lo) for (rk, la, lo) in jobs]
        iterator = _tqdm(as_completed(futs), total=len(futs), desc="Copernicus DEM", unit="tile") if _tqdm else as_completed(futs)
        for fut in iterator:
            dl, ch, oc = fut.result()
            stats["downloaded"] += dl
            stats["cached"] += ch
            stats["ocean_404"] += oc
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

    if source == "gebco-2025-zip":
        zip_path = bathy_root / "gebco_2025.zip"
        out_dir = bathy_root / "GEBCO_2025"
        if not dry_run:
            _download_file(GEBCO_CEDA_2025_ZIP_URL, zip_path, resume=True, timeout_s=600.0)
            out_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path) as zf:
                for member in zf.namelist():
                    p = Path(member)
                    if not p.is_absolute() and ".." not in p.parts and member.lower().endswith(".nc"):
                        zf.extract(member, out_dir)
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
            _download_file(url, out, resume=False, timeout_s=300.0)
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


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Download static geophysical data for Tempest-NWP v2.")
    parser.add_argument("--all", action="store_true", help="Download Copernicus DEM GLO-30, Bathymetry, and HWSD v2.2")
    parser.add_argument("--copernicus-dem", action="store_true", help="Download Copernicus DEM GLO-30 + WBM tiles")
    parser.add_argument("--glo90", action="store_true", help="Also download Copernicus DEM GLO-90 fallback tiles")
    parser.add_argument("--bathymetry", action="store_true", help="Download ocean/coastal bathymetry")
    parser.add_argument(
        "--bathy-source",
        choices=("gebco-erddap", "gebco-2026-global", "gebco-2025-zip", "emodnet-wcs"),
        default="gebco-erddap",
        help="Bathymetry source (default: gebco-erddap fast MSL NetCDF subset)",
    )
    parser.add_argument("--hwsd", action="store_true", help="Download FAO/IIASA HWSD v2.2 soil database & raster")
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
    parser.add_argument("--workers", type=int, default=4, help="Parallel tile download workers (default: 4)")
    parser.add_argument("--prepare-wx1", action="store_true", help="Compile static WX-1 containers for --domain")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved URLs/counts without downloading")
    args = parser.parse_args(argv)

    want_dem = args.all or args.copernicus_dem or args.glo90
    want_bathy = args.all or args.bathymetry
    want_hwsd = args.all or args.hwsd
    if not (want_dem or want_bathy or want_hwsd or args.prepare_wx1):
        parser.print_help()
        return

    static_root = PROJECT_ROOT / "data" / "static"
    dem_root = Path(args.dem_dir or os.environ.get("TEMPEST_COPERNICUS_DEM_DIR") or (static_root / "copernicus_dem_raw"))
    bathy_root = Path(args.bathy_dir or os.environ.get("TEMPEST_BATHY_DIR") or (static_root / "bathymetry_raw"))
    hwsd_root = Path(args.hwsd_dir or os.environ.get("TEMPEST_HWSD_V22_DIR") or (static_root / "hwsd_v22"))

    bbox = parse_bbox_arg(args.bbox) if args.bbox else bbox_from_domain(args.domain)
    print(f"Target domain={args.domain} | bbox=(lat {bbox[0]:.2f}..{bbox[1]:.2f}, lon {bbox[2]:.2f}..{bbox[3]:.2f})")

    if want_dem:
        res_stats = download_copernicus_dem(
            bbox,
            dem_root,
            include_glo30=args.copernicus_dem or args.all or not args.glo90,
            include_glo90=args.glo90,
            workers=args.workers,
            dry_run=args.dry_run,
        )
        print(f"Copernicus DEM ({dem_root}): {res_stats}")

    if want_bathy:
        if args.dry_run:
            preview = gebco_erddap_url(*bbox) if args.bathy_source == "gebco-erddap" else (
                emodnet_wcs_url(*bbox) if args.bathy_source == "emodnet-wcs" else (
                    GEBCO_CEDA_2026_NC_URL if args.bathy_source == "gebco-2026-global" else GEBCO_CEDA_2025_ZIP_URL
                )
            )
            print(f"Bathymetry ({args.bathy_source}): dry-run URL -> {preview}")
        else:
            bathy_out = download_bathymetry(bbox, bathy_root, source=args.bathy_source, label=args.domain)
            print(f"Bathymetry ({args.bathy_source}): saved -> {bathy_out}")

    if want_hwsd:
        hwsd_stats = download_hwsd_v22(hwsd_root, dry_run=args.dry_run)
        print(f"FAO HWSD v2.2 ({hwsd_root}): {hwsd_stats}")

    if args.prepare_wx1 and not args.dry_run:
        os.environ["TEMPEST_COPERNICUS_DEM_DIR"] = str(dem_root)
        os.environ["TEMPEST_BATHY_DIR"] = str(bathy_root)
        os.environ["TEMPEST_HWSD_V22_DIR"] = str(hwsd_root)
        from tempest_v2.ic_builder import build_domain_static_fields

        spec = get_domain_spec(args.domain)
        hs_m, surf = build_domain_static_fields(
            spec,
            dem_root=dem_root,
            gebco_root=bathy_root,
            force=True,
            allow_synthetic_fallback=False,
        )
        print(
            f"Prepared static WX-1 for {spec.name}: "
            f"max_oro={float(hs_m.max()):.1f}m, max_bathy={float(surf['bathy_depth_m'].max()):.1f}m"
        )


if __name__ == "__main__":
    main()
