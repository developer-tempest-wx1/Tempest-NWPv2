"""Post-Processing & Visualization Exporter for Tempest-NWP v2 WX-1 Forecast Outputs.

Strictly 100% deterministic classical meteorology & standard thermodynamics (0% AI / 0% ML).
Exports for user-chosen locations (resolved via geopy/Nominatim, local cache, built-in Benelux/EU
gazetteer, or explicit lat,lon coordinates):
  - Surface synoptic/convective/ocean CSV time series (`<slug>_timeseries.csv`, `all_locations_timeseries.csv`)
  - 3D isobaric vertical sounding CSV tables (`<slug>_sounding_profiles.csv`)
  - 4-panel location meteograms (`<slug>_meteogram.png`)
  - Cartopy geographic weather maps (`t2m_mslp`, `precip_wind`, `cape_shear`, `z500_t850`, `sst_ice`)
  - MetPy Skew-T Log-P + Hodograph sounding diagrams (`skewt_<slug>_T+<hr>h.png`)
  - Optimized Pillow looping GIF animations (and optional MP4 when `ffmpeg` is available)
"""

from __future__ import annotations

import datetime
import json
import logging
import math
import re
import shutil
import subprocess
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import metpy.calc as mpcalc
from metpy.plots import Hodograph, SkewT
from metpy.units import units
import wx1

try:
    from geopy.geocoders import Nominatim
except ImportError:  # pragma: no cover - geopy optional fallback
    Nominatim = None  # type: ignore[assignment]

try:
    from geopy.extra.rate_limiter import RateLimiter
except ImportError:  # pragma: no cover - rate limiter optional
    RateLimiter = None  # type: ignore[assignment]

try:
    from tqdm import tqdm as _tqdm
except ImportError:  # pragma: no cover
    _tqdm = None

logger = logging.getLogger("tempest_v2.export_products")

VALID_MAP_TYPES: Tuple[str, ...] = (
    "t2m_mslp",
    "precip_wind",
    "cape_shear",
    "z500_t850",
    "sst_ice",
)

# Verified WGS-84 coordinates for offline fallback & fast Benelux/Europe selection
BUILTIN_LOCATIONS: Dict[str, Tuple[float, float]] = {
    "de bilt": (52.1000, 5.1800),
    "tilburg": (51.5555, 5.0913),
    "amsterdam": (52.3676, 4.9041),
    "rotterdam": (51.9244, 4.4777),
    "utrecht": (52.0907, 5.1214),
    "eindhoven": (51.4416, 5.4697),
    "groningen": (53.2194, 6.5665),
    "maastricht": (50.8514, 5.6910),
    "den haag": (52.0705, 4.3007),
    "arnhem": (51.9851, 5.8987),
    "zwolle": (52.5168, 6.0830),
    "vlissingen": (51.4426, 3.5736),
    "den helder": (52.9563, 4.7608),
    "leeuwarden": (53.2012, 5.7999),
    "enschede": (52.2215, 6.8937),
    "breda": (51.5719, 4.7683),
    "nijmegen": (51.8126, 5.8372),
    "brussel": (50.8503, 4.3517),
    "brussels": (50.8503, 4.3517),
    "antwerpen": (51.2194, 4.4025),
    "antwerp": (51.2194, 4.4025),
    "gent": (51.0543, 3.7174),
    "luik": (50.6326, 5.5797),
    "oostende": (51.2154, 2.9286),
    "luxemburg": (49.6116, 6.1319),
    "luxembourg": (49.6116, 6.1319),
    "berlijn": (52.5200, 13.4050),
    "berlin": (52.5200, 13.4050),
    "keulen": (50.9375, 6.9603),
    "cologne": (50.9375, 6.9603),
    "dusseldorf": (51.2277, 6.7735),
    "hamburg": (53.5511, 9.9937),
    "munchen": (48.1351, 11.5820),
    "munich": (48.1351, 11.5820),
    "frankfurt": (50.1109, 8.6821),
    "parijs": (48.8566, 2.3522),
    "paris": (48.8566, 2.3522),
    "londen": (51.5074, -0.1278),
    "london": (51.5074, -0.1278),
}

_EPS_MW = 0.62198
_EARTH_RADIUS_KM = 6371.0
_DEFAULT_CACHE_PATH = Path.home() / ".cache" / "tempest_v2" / "geocode_cache.json"
_COORD_PAIR_RE = re.compile(r"^\s*(?:(?P<label>[^:]+):)?\s*(?P<lat>[-+]?\d+(?:\.\d+)?)\s*[,;]\s*(?P<lon>[-+]?\d+(?:\.\d+)?)\s*$")
_LEAD_HR_RE = re.compile(r"_T\+([-+]?\d+(?:\.\d+)?)h\.wx1$")


@dataclass(frozen=True)
class LocationPoint:
    """Resolved geographic point for extraction and plotting."""

    name: str
    slug: str
    lat: float
    lon: float
    source: str = "manual"


def _slugify(text: str) -> str:
    norm = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", norm.strip().lower()).strip("_")
    return slug or "location"


def _lead_tag(hr: float) -> str:
    if abs(float(hr) - round(float(hr))) < 1e-4:
        return f"{int(round(float(hr))):03d}"
    return f"{float(hr):06.2f}"


def _wrap_lon_180(lon_deg: float) -> float:
    return ((float(lon_deg) + 180.0) % 360.0) - 180.0


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(_wrap_lon_180(lon2 - lon1))
    a = math.sin(dphi * 0.5) ** 2 + math.cos(phi1) * math.cos(phi2) * (math.sin(dl * 0.5) ** 2)
    return 2.0 * _EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(max(0.0, a))))


def _load_geocode_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    try:
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def _save_geocode_cache(path: Path, cache: Dict[str, Dict[str, Any]]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def parse_location_spec(raw: str) -> Optional[Tuple[str, float, float]]:
    """Parse `'Name:lat,lon'` or `'lat,lon'` into `(label, lat, lon)`, or None if plain place name."""
    m = _COORD_PAIR_RE.match(raw)
    if not m:
        return None
    lat = float(m.group("lat"))
    lon = float(m.group("lon"))
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 360.0):
        raise ValueError(f"Coordinates out of range in '{raw}': lat={lat}, lon={lon}")
    lon_norm = _wrap_lon_180(lon)
    raw_lbl = m.group("label")
    label = raw_lbl.strip() if raw_lbl and raw_lbl.strip() else f"{lat:.3f}N_{lon_norm:.3f}E"
    return label, lat, lon_norm


def resolve_locations(
    location_queries: Sequence[str],
    *,
    use_geopy: bool = True,
    cache_path: Optional[Path | str] = None,
    geocoder: Optional[Any] = None,
) -> List[LocationPoint]:
    """Resolve user-supplied place names (via geopy Nominatim + cache + fallback) or `Naam:lat,lon`."""
    if not location_queries:
        raise ValueError("Provide at least one location name or coordinate pair.")

    cpath = Path(cache_path) if cache_path is not None else _DEFAULT_CACHE_PATH
    cache = _load_geocode_cache(cpath)
    cache_dirty = False

    geocode_fn: Optional[Any] = None
    if use_geopy:
        if geocoder is not None:
            geocode_fn = geocoder.geocode if hasattr(geocoder, "geocode") else geocoder
        elif Nominatim is not None:
            nom = Nominatim(user_agent="tempest-nwp-v2-wx1-exporter", timeout=10)
            if RateLimiter is not None:
                geocode_fn = RateLimiter(nom.geocode, min_delay_seconds=1.0, max_retries=2, swallow_exceptions=False)
            else:
                geocode_fn = nom.geocode

    resolved: List[LocationPoint] = []
    used_slugs: Dict[str, int] = {}

    for raw in location_queries:
        q = str(raw).strip()
        if not q:
            continue
        parsed = parse_location_spec(q)
        if parsed is not None:
            label, lat, lon = parsed
            source = "coordinates"
        else:
            key = q.lower().strip()
            short_key = key.split(",", 1)[0].strip()
            entry = cache.get(key)
            if entry is not None and "lat" in entry and "lon" in entry:
                label = q.split(",", 1)[0].strip() or str(entry.get("display_name", q))
                lat = float(entry["lat"])
                lon = _wrap_lon_180(float(entry["lon"]))
                source = "cache"
            else:
                geo_hit = None
                if geocode_fn is not None:
                    try:
                        geo_hit = geocode_fn(q)
                    except Exception as exc:
                        logger.warning("geopy lookup failed for '%s': %s", q, exc)
                if geo_hit is not None:
                    lat = float(geo_hit.latitude)
                    lon = _wrap_lon_180(float(geo_hit.longitude))
                    label = q.split(",", 1)[0].strip() or q
                    source = "geopy"
                    cache[key] = {
                        "lat": lat,
                        "lon": lon,
                        "display_name": str(getattr(geo_hit, "address", q)),
                    }
                    cache_dirty = True
                elif key in BUILTIN_LOCATIONS or short_key in BUILTIN_LOCATIONS:
                    b_key = key if key in BUILTIN_LOCATIONS else short_key
                    lat, lon = BUILTIN_LOCATIONS[b_key]
                    label = q.split(",", 1)[0].strip() or q
                    source = "builtin_fallback"
                else:
                    raise ValueError(
                        f"Locatie '{q}' kon niet worden gevonden via geopy of ingebouwde plaatsnamen. "
                        f"Controleer de spelling of geef coördinaten op als '{q}:lat,lon'."
                    )

        base_slug = _slugify(label)
        cnt = used_slugs.get(base_slug, 0) + 1
        used_slugs[base_slug] = cnt
        slug = base_slug if cnt == 1 else f"{base_slug}_{cnt}"
        resolved.append(LocationPoint(name=label, slug=slug, lat=lat, lon=lon, source=source))

    if cache_dirty:
        _save_geocode_cache(cpath, cache)

    if not resolved:
        raise ValueError("No valid locations were resolved.")
    return resolved


def _extract_lead_hours(path: Path) -> float:
    m = _LEAD_HR_RE.search(path.name)
    if m:
        return float(m.group(1))
    with wx1.open_dataset(path) as ds:
        if "lead_hours" in ds.attrs:
            return float(ds.attrs["lead_hours"])
    return 0.0


def _detect_domain_from_path(path: Path) -> str:
    stem = path.name.lower()
    if "_d1" in stem or path.parent.name.lower() == "d1":
        return "d1"
    if "_d2" in stem or path.parent.name.lower() == "d2":
        return "d2"
    if "_d3" in stem or path.parent.name.lower() == "d3":
        return "d3"
    return "domain"


def discover_wx1_forecasts(
    input_path: Path | str,
    domain: str = "auto",
) -> Tuple[List[Path], str, Path]:
    """Locate sorted `.wx1` forecast files from a file, domain dir, run dir, or `runs/` root."""
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"Input path does not exist: {p}")

    dom_req = str(domain).strip().lower()
    if p.is_file():
        if p.suffix.lower() != ".wx1":
            raise ValueError(f"Expected a .wx1 file, got: {p}")
        det_dom = dom_req if dom_req in ("d1", "d2", "d3") else _detect_domain_from_path(p)
        return [p], det_dom, p.parent / "exports"

    direct_files = sorted(p.glob("forecast_*_T+*h.wx1"), key=_extract_lead_hours)
    if direct_files and dom_req in ("auto", _detect_domain_from_path(direct_files[0]), p.name.lower()):
        det_dom = dom_req if dom_req in ("d1", "d2", "d3") else _detect_domain_from_path(direct_files[0])
        base_out = p.parent / "exports" / det_dom if p.name.lower() in ("d1", "d2", "d3") else p / "exports"
        return direct_files, det_dom, base_out

    def _check_run_dir(run_d: Path) -> Optional[Tuple[List[Path], str, Path]]:
        order = [dom_req] if dom_req in ("d1", "d2", "d3") else ["d3", "d2", "d1"]
        for d_name in order:
            sub = run_d / d_name
            if sub.is_dir():
                cands = sorted(sub.glob("forecast_*_T+*h.wx1"), key=_extract_lead_hours)
                if cands:
                    return cands, d_name, run_d / "exports" / d_name
        return None

    direct_run = _check_run_dir(p)
    if direct_run is not None:
        return direct_run

    # Scan child run folders newest-first (ignoring raw_ecmwf / hidden dirs)
    child_dirs = sorted(
        [d for d in p.iterdir() if d.is_dir() and not d.name.startswith(".") and d.name != "raw_ecmwf"],
        key=lambda d: d.name,
        reverse=True,
    )
    for cd in child_dirs:
        hit = _check_run_dir(cd)
        if hit is not None:
            return hit

    raise FileNotFoundError(
        f"No forecast_*_T+*h.wx1 files found under '{p}' (domain={domain})."
    )


def _find_point_stencil(
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    pt_lat: float,
    pt_lon: float,
) -> Dict[str, Any]:
    """Compute exact 2x2 bilinear interpolation stencil & weights for regular or curvilinear grids."""
    ny, nx = lat2d.shape
    lat_1d_col = lat2d[:, 0]
    lon_1d_row = lon2d[0, :]
    is_separable = bool(
        ny >= 2
        and nx >= 2
        and np.allclose(lat2d, lat_1d_col[:, None], atol=1e-4)
        and np.allclose(lon2d, lon_1d_row[None, :], atol=1e-4)
    )

    if is_separable:
        lat_asc = bool(lat_1d_col[-1] > lat_1d_col[0])
        lat_vec = lat_1d_col if lat_asc else lat_1d_col[::-1]
        is_0_360 = float(np.nanmax(lon_1d_row)) > 180.0 and float(np.nanmin(lon_1d_row)) >= -1e-3
        q_lon = (pt_lon % 360.0) if is_0_360 else _wrap_lon_180(pt_lon)

        idx_y = int(np.searchsorted(lat_vec, pt_lat, side="right")) - 1
        if idx_y < 0:
            y0, y1, wy = 0, 0, 0.0
        elif idx_y >= ny - 1:
            y0, y1, wy = ny - 1, ny - 1, 0.0
        else:
            y0, y1 = idx_y, idx_y + 1
            wy = float(np.clip((pt_lat - lat_vec[y0]) / max(lat_vec[y1] - lat_vec[y0], 1e-9), 0.0, 1.0))
        if not lat_asc:
            y0, y1 = ny - 1 - y0, ny - 1 - y1

        lon_vec = lon_1d_row
        dlon_grid = float(np.median(np.abs(np.diff(lon_vec)))) if nx >= 2 else 1.0
        is_periodic = is_0_360 and (lon_vec[-1] - lon_vec[0] + dlon_grid) >= 359.0
        if q_lon < float(lon_vec[0]):
            if is_periodic:
                x0, x1 = nx - 1, 0
                span = (lon_vec[0] + 360.0) - lon_vec[-1]
                wx = float(np.clip((q_lon + 360.0 - lon_vec[-1]) / max(span, 1e-9), 0.0, 1.0))
            else:
                x0, x1, wx = 0, 0, 0.0
        elif q_lon >= float(lon_vec[-1]):
            if is_periodic:
                x0, x1 = nx - 1, 0
                span = (lon_vec[0] + 360.0) - lon_vec[-1]
                wx = float(np.clip((q_lon - lon_vec[-1]) / max(span, 1e-9), 0.0, 1.0))
            else:
                x0, x1, wx = nx - 1, nx - 1, 0.0
        else:
            x0 = int(np.searchsorted(lon_vec, q_lon, side="right")) - 1
            x0 = max(0, min(nx - 2, x0))
            x1 = x0 + 1
            wx = float(np.clip((q_lon - lon_vec[x0]) / max(lon_vec[x1] - lon_vec[x0], 1e-9), 0.0, 1.0))

        g_lat = (1.0 - wy) * float(lat2d[y0, x0]) + wy * float(lat2d[y1, x0])
        l0 = float(lon2d[y0, x0])
        l1 = float(lon2d[y0, x1])
        if is_periodic and x1 == 0 and l1 < l0:
            l1 += 360.0
        g_lon = _wrap_lon_180((1.0 - wx) * l0 + wx * l1)
        return {
            "j0": y0,
            "j1": y1,
            "i0": x0,
            "i1": x1,
            "wy": wy,
            "wx": wx,
            "grid_lat": g_lat,
            "grid_lon": g_lon,
            "dist_km": _haversine_km(pt_lat, pt_lon, g_lat, g_lon),
        }

    # General curvilinear 2D LCC grid
    cos_lat = math.cos(math.radians(max(-85.0, min(85.0, pt_lat))))
    dlon_arr = ((lon2d - _wrap_lon_180(pt_lon) + 180.0) % 360.0) - 180.0
    d2 = (lat2d - pt_lat) ** 2 + (dlon_arr * cos_lat) ** 2
    jc, ic = np.unravel_index(int(np.argmin(d2)), d2.shape)
    jc_i, ic_i = int(jc), int(ic)

    # Pick local 2x2 cell containing target and solve local linear Jacobian for bilinear weights
    j0 = max(0, min(ny - 2, jc_i if (ny >= 2 and (pt_lat - lat2d[jc_i, ic_i]) * (lat2d[min(ny - 1, jc_i + 1), ic_i] - lat2d[jc_i, ic_i]) >= 0) else jc_i - 1))
    i0 = max(0, min(nx - 2, ic_i if (nx >= 2 and (-dlon_arr[jc_i, ic_i]) * (((lon2d[jc_i, min(nx - 1, ic_i + 1)] - lon2d[jc_i, ic_i] + 180.0) % 360.0) - 180.0) >= 0) else ic_i - 1))
    j1 = min(ny - 1, j0 + 1)
    i1 = min(nx - 1, i0 + 1)

    wy, wx = 0.0, 0.0
    if j1 > j0 and i1 > i0:
        dlat_dy = float(lat2d[j1, i0] - lat2d[j0, i0])
        dlat_dx = float(lat2d[j0, i1] - lat2d[j0, i0])
        dlon_dy = _wrap_lon_180(float(lon2d[j1, i0]) - float(lon2d[j0, i0]))
        dlon_dx = _wrap_lon_180(float(lon2d[j0, i1]) - float(lon2d[j0, i0]))
        det = dlat_dy * dlon_dx - dlat_dx * dlon_dy
        if abs(det) > 1e-12:
            rhs_lat = pt_lat - float(lat2d[j0, i0])
            rhs_lon = _wrap_lon_180(pt_lon - float(lon2d[j0, i0]))
            wy_raw = (rhs_lat * dlon_dx - dlat_dx * rhs_lon) / det
            wx_raw = (dlat_dy * rhs_lon - rhs_lat * dlon_dy) / det
            if -0.25 <= wy_raw <= 1.25 and -0.25 <= wx_raw <= 1.25:
                wy = float(np.clip(wy_raw, 0.0, 1.0))
                wx = float(np.clip(wx_raw, 0.0, 1.0))
            else:
                wy = 0.0 if jc_i == j0 else 1.0
                wx = 0.0 if ic_i == i0 else 1.0
        else:
            wy = 0.0 if jc_i == j0 else 1.0
            wx = 0.0 if ic_i == i0 else 1.0

    g_lat = (
        (1.0 - wy) * (1.0 - wx) * float(lat2d[j0, i0])
        + (1.0 - wy) * wx * float(lat2d[j0, i1])
        + wy * (1.0 - wx) * float(lat2d[j1, i0])
        + wy * wx * float(lat2d[j1, i1])
    )
    l_ref = float(lon2d[j0, i0])
    g_lon = _wrap_lon_180(
        l_ref
        + (1.0 - wy) * wx * _wrap_lon_180(float(lon2d[j0, i1]) - l_ref)
        + wy * (1.0 - wx) * _wrap_lon_180(float(lon2d[j1, i0]) - l_ref)
        + wy * wx * _wrap_lon_180(float(lon2d[j1, i1]) - l_ref)
    )
    return {
        "j0": j0,
        "j1": j1,
        "i0": i0,
        "i1": i1,
        "wy": wy,
        "wx": wx,
        "grid_lat": g_lat,
        "grid_lon": g_lon,
        "dist_km": _haversine_km(pt_lat, pt_lon, g_lat, g_lon),
    }


def _sample_field(da: Any, st: Dict[str, Any]) -> np.ndarray:
    """Sample 2D `(y, x)` or 3D `(plev, y, x)` field cleanly via precomputed stencil."""
    j0, j1, i0, i1 = st["j0"], st["j1"], st["i0"], st["i1"]
    wy, wx = st["wy"], st["wx"]
    arr = da.values
    if arr.ndim == 2:
        v00 = float(arr[j0, i0])
        v01 = float(arr[j0, i1])
        v10 = float(arr[j1, i0])
        v11 = float(arr[j1, i1])
        return np.array((1.0 - wy) * ((1.0 - wx) * v00 + wx * v01) + wy * ((1.0 - wx) * v10 + wx * v11), dtype=np.float64)
    v00 = np.asarray(arr[:, j0, i0], dtype=np.float64)
    v01 = np.asarray(arr[:, j0, i1], dtype=np.float64)
    v10 = np.asarray(arr[:, j1, i0], dtype=np.float64)
    v11 = np.asarray(arr[:, j1, i1], dtype=np.float64)
    return (1.0 - wy) * ((1.0 - wx) * v00 + wx * v01) + wy * ((1.0 - wx) * v10 + wx * v11)


def _wind_dir_deg(u_ms: np.ndarray | float, v_ms: np.ndarray | float) -> np.ndarray | float:
    spd = np.hypot(u_ms, v_ms)
    wdir = (270.0 - np.degrees(np.arctan2(v_ms, u_ms))) % 360.0
    if isinstance(spd, np.ndarray):
        return np.where(spd > 1e-6, wdir, 0.0)
    return float(wdir) if float(spd) > 1e-6 else 0.0


def _dewpoint_c_from_tq_or_rh(
    t_c: np.ndarray,
    p_pa: np.ndarray,
    qv: Optional[np.ndarray],
    rh_pct: Optional[np.ndarray],
) -> np.ndarray:
    es_pa = 611.2 * np.exp((17.67 * np.clip(t_c, -90.0, 60.0)) / (np.clip(t_c, -90.0, 60.0) + 243.5))
    if qv is not None:
        q_c = np.clip(qv, 1e-7, 0.05)
        e_pa = (q_c * p_pa) / (_EPS_MW + (1.0 - _EPS_MW) * q_c)
    elif rh_pct is not None:
        e_pa = np.clip(rh_pct, 0.5, 100.0) * 0.01 * es_pa
    else:
        e_pa = 0.25 * es_pa
    e_pa = np.clip(e_pa, 0.5, np.maximum(es_pa * 0.997, 0.5))
    ln_e = np.log(e_pa / 611.2)
    td_c = (243.5 * ln_e) / np.maximum(17.67 - ln_e, 1e-3)
    return np.clip(td_c, t_c - 75.0, t_c - 0.05)


def extract_all_locations_data(
    wx1_files: Sequence[Path],
    locations: Sequence[LocationPoint],
    domain: str,
    *,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Open each WX-1 file once and extract full synoptic rows and vertical soundings for all locations."""
    sfc_rows_by_slug: Dict[str, List[Dict[str, Any]]] = {loc.slug: [] for loc in locations}
    sounding_rows_by_slug: Dict[str, List[Dict[str, Any]]] = {loc.slug: [] for loc in locations}
    stencils: Optional[Dict[str, Dict[str, Any]]] = None
    prev_precip: Dict[str, float] = {}

    file_iter = (
        _tqdm(wx1_files, desc="[export] Extracting points", unit="file", dynamic_ncols=True)
        if (show_progress and _tqdm is not None and len(wx1_files) > 1)
        else wx1_files
    )

    for fp in file_iter:
        with wx1.open_dataset(fp) as ds:
            lat2d = np.asarray(ds["lat"].values, dtype=np.float64)
            lon2d = np.asarray(ds["lon"].values, dtype=np.float64)
            if stencils is None:
                stencils = {}
                for loc in locations:
                    st = _find_point_stencil(lat2d, lon2d, loc.lat, loc.lon)
                    stencils[loc.slug] = st
                    if st["dist_km"] > 50.0:
                        logger.warning(
                            "Location '%s' (%.3f, %.3f) is %.1f km from nearest %s grid cell (%.3f, %.3f) — clamped to domain edge.",
                            loc.name,
                            loc.lat,
                            loc.lon,
                            st["dist_km"],
                            domain,
                            st["grid_lat"],
                            st["grid_lon"],
                        )

            lead_hr = float(ds.attrs.get("lead_hours", _extract_lead_hours(fp)))
            valid_str = str(ds.attrs.get("valid_time_utc", ""))
            has_plev = "plev" in ds.coords and "t_pl" in ds.data_vars
            plev_pa = np.asarray(ds["plev"].values, dtype=np.float64) if has_plev else np.array([], dtype=np.float64)
            plev_hpa = plev_pa * 0.01

            def _get2d(name: str, st: Dict[str, Any], default: float = float("nan")) -> float:
                if name not in ds.data_vars:
                    return default
                return float(_sample_field(ds[name], st))

            def _get3d(name: str, st: Dict[str, Any]) -> Optional[np.ndarray]:
                if name not in ds.data_vars:
                    return None
                return _sample_field(ds[name], st)

            for loc in locations:
                st = stencils[loc.slug]
                t2m_k = _get2d("t2m", st)
                td2m_k = _get2d("td2m", st)
                u10 = _get2d("u10m_geo", st, 0.0)
                v10 = _get2d("v10m_geo", st, 0.0)
                wspd10 = float(math.hypot(u10, v10))
                gust10 = _get2d("gust10m", st)
                mslp_pa = _get2d("mslp", st)
                ps_pa = _get2d("ps", st)
                rain = _get2d("rain_acc", st, 0.0)
                snow = _get2d("snow_acc", st, 0.0)
                graupel = _get2d("graupel_acc", st, 0.0)
                p_tot = (0.0 if math.isnan(rain) else rain) + (0.0 if math.isnan(snow) else snow) + (
                    0.0 if math.isnan(graupel) else graupel
                )
                p_step = max(0.0, p_tot - prev_precip[loc.slug]) if loc.slug in prev_precip else 0.0
                prev_precip[loc.slug] = p_tot
                p_rate_si = _get2d("precip_rate", st, 0.0)

                row: Dict[str, Any] = {
                    "lead_hours": round(lead_hr, 4),
                    "valid_time_utc": valid_str,
                    "location": loc.name,
                    "lat": round(loc.lat, 5),
                    "lon": round(loc.lon, 5),
                    "grid_lat": round(st["grid_lat"], 5),
                    "grid_lon": round(st["grid_lon"], 5),
                    "grid_dist_km": round(st["dist_km"], 2),
                    "domain": domain,
                    "t2m_k": t2m_k,
                    "t2m_c": t2m_k - 273.15 if np.isfinite(t2m_k) else float("nan"),
                    "td2m_k": td2m_k,
                    "td2m_c": td2m_k - 273.15 if np.isfinite(td2m_k) else float("nan"),
                    "rh2m_pct": _get2d("rh2m", st),
                    "q2m_kg_kg": _get2d("q2m", st),
                    "t_skin_c": _get2d("t_skin", st) - 273.15 if "t_skin" in ds.data_vars else float("nan"),
                    "u10m_ms": u10,
                    "v10m_ms": v10,
                    "wspd10m_ms": wspd10,
                    "wspd10m_kmh": wspd10 * 3.6,
                    "wspd10m_kt": wspd10 * 1.943844,
                    "wdir10m_deg": _wind_dir_deg(u10, v10),
                    "gust10m_ms": gust10,
                    "gust10m_kmh": gust10 * 3.6 if np.isfinite(gust10) else float("nan"),
                    "mslp_pa": mslp_pa,
                    "mslp_hpa": mslp_pa * 0.01 if np.isfinite(mslp_pa) else float("nan"),
                    "ps_pa": ps_pa,
                    "ps_hpa": ps_pa * 0.01 if np.isfinite(ps_pa) else float("nan"),
                    "rain_acc_mm": rain,
                    "snow_acc_mm": snow,
                    "graupel_acc_mm": graupel,
                    "precip_total_mm": p_tot,
                    "precip_step_mm": p_step,
                    "precip_rate_mm_h": p_rate_si * 3600.0 if np.isfinite(p_rate_si) else float("nan"),
                    "tcc_frac": _get2d("tcc", st),
                    "lcc_frac": _get2d("lcc", st),
                    "mcc_frac": _get2d("mcc", st),
                    "hcc_frac": _get2d("hcc", st),
                    "tcwv_kg_m2": _get2d("tcwv", st),
                    "lwp_kg_m2": _get2d("lwp", st),
                    "iwp_kg_m2": _get2d("iwp", st),
                    "cape_sb_j_kg": _get2d("cape_sb", st),
                    "cin_sb_j_kg": _get2d("cin_sb", st),
                    "cape_mu_j_kg": _get2d("cape_mu", st),
                    "shear_0_6km_ms": _get2d("shear_0_6km", st),
                    "srh_0_3km_m2_s2": _get2d("srh_0_3km", st),
                    "freezing_level_m": _get2d("freezing_level_m", st),
                    "vis_m": _get2d("vis_m", st),
                    "pblh_m": _get2d("pblh", st),
                    "sw_down_w_m2": _get2d("sw_down", st),
                    "lw_down_w_m2": _get2d("lw_down", st),
                    "shf_w_m2": _get2d("shf", st),
                    "lhf_w_m2": _get2d("lhf", st),
                    "snow_swe_kg_m2": _get2d("snow_swe", st),
                }

                sst_k = _get2d("sst", st) if "sst" in ds.data_vars else _get2d("sst_bulk", st)
                row["sst_c"] = (sst_k - 273.15) if (np.isfinite(sst_k) and sst_k > 200.0) else float("nan")
                row["sea_ice_frac"] = _get2d("sea_ice_frac", st)
                row["sea_ice_thick_m"] = _get2d("sea_ice_thick", st)
                row["mld_m"] = _get2d("mld", st)
                row["ssh_m"] = _get2d("ssh", st)

                if has_plev:
                    t_pl = _get3d("t_pl", st)
                    gh_pl = _get3d("gh_pl", st)
                    theta_pl = _get3d("theta_pl", st)
                    q_pl = _get3d("q_pl", st)
                    rh_pl = _get3d("rh_pl", st)
                    u_pl = _get3d("u_pl", st)
                    v_pl = _get3d("v_pl", st)
                    w_pl = _get3d("w_pl", st)
                    omega_pl = _get3d("omega_pl", st)

                    assert t_pl is not None
                    t_pl_c = t_pl - 273.15
                    td_pl_c = _dewpoint_c_from_tq_or_rh(t_pl_c, plev_pa, q_pl, rh_pl)

                    def _at_hpa(arr: Optional[np.ndarray], hpa_tgt: float, offset: float = 0.0) -> float:
                        if arr is None:
                            return float("nan")
                        hits = np.where(np.isclose(plev_hpa, hpa_tgt, atol=1.0))[0]
                        if len(hits) == 0:
                            return float("nan")
                        return float(arr[int(hits[0])]) + offset

                    row["t850_c"] = _at_hpa(t_pl, 850.0, -273.15)
                    row["t500_c"] = _at_hpa(t_pl, 500.0, -273.15)
                    row["gh500_m"] = _at_hpa(gh_pl, 500.0)
                    row["rh700_pct"] = _at_hpa(rh_pl, 700.0)
                    u300 = _at_hpa(u_pl, 300.0)
                    v300 = _at_hpa(v_pl, 300.0)
                    row["wspd300_ms"] = float(math.hypot(u300, v300)) if np.isfinite(u300) and np.isfinite(v300) else float("nan")

                    order = np.argsort(plev_hpa)[::-1]
                    for ik in order:
                        u_k = float(u_pl[ik]) if u_pl is not None else float("nan")
                        v_k = float(v_pl[ik]) if v_pl is not None else float("nan")
                        spd_k = float(math.hypot(u_k, v_k)) if np.isfinite(u_k) and np.isfinite(v_k) else float("nan")
                        sounding_rows_by_slug[loc.slug].append(
                            {
                                "lead_hours": round(lead_hr, 4),
                                "valid_time_utc": valid_str,
                                "location": loc.name,
                                "plev_hpa": float(plev_hpa[ik]),
                                "gh_m": float(gh_pl[ik]) if gh_pl is not None else float("nan"),
                                "t_c": float(t_pl_c[ik]),
                                "td_c": float(td_pl_c[ik]),
                                "theta_k": float(theta_pl[ik]) if theta_pl is not None else float("nan"),
                                "q_kg_kg": float(q_pl[ik]) if q_pl is not None else float("nan"),
                                "rh_pct": float(rh_pl[ik]) if rh_pl is not None else float("nan"),
                                "u_ms": u_k,
                                "v_ms": v_k,
                                "wspd_ms": spd_k,
                                "wdir_deg": _wind_dir_deg(u_k, v_k) if np.isfinite(spd_k) else float("nan"),
                                "w_ms": float(w_pl[ik]) if w_pl is not None else float("nan"),
                                "omega_pa_s": float(omega_pl[ik]) if omega_pl is not None else float("nan"),
                            }
                        )

                sfc_rows_by_slug[loc.slug].append(row)

    return {
        "surface_df_by_slug": {slug: pd.DataFrame(rows) for slug, rows in sfc_rows_by_slug.items()},
        "sounding_df_by_slug": {slug: pd.DataFrame(rows) for slug, rows in sounding_rows_by_slug.items() if rows},
    }


def export_location_csvs(
    extracted: Dict[str, Any],
    outdir: Path | str,
) -> List[Path]:
    """Write per-location CSV time series, combined all-location CSV, and sounding profile CSVs."""
    csv_dir = Path(outdir) / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    all_frames: List[pd.DataFrame] = []

    for slug, df in extracted["surface_df_by_slug"].items():
        if df.empty:
            continue
        p = csv_dir / f"{slug}_timeseries.csv"
        df.to_csv(p, index=False)
        written.append(p)
        all_frames.append(df)

    for slug, sdf in extracted["sounding_df_by_slug"].items():
        if sdf.empty:
            continue
        sp = csv_dir / f"{slug}_sounding_profiles.csv"
        sdf.to_csv(sp, index=False)
        written.append(sp)

    if all_frames:
        all_p = csv_dir / "all_locations_timeseries.csv"
        pd.concat(all_frames, ignore_index=True).to_csv(all_p, index=False)
        written.append(all_p)

    return written


def render_meteograms(
    extracted: Dict[str, Any],
    locations: Sequence[LocationPoint],
    outdir: Path | str,
    *,
    dpi: int = 150,
) -> List[Path]:
    """Render 4-panel meteograms per location when at least 2 lead times are present."""
    met_dir = Path(outdir) / "meteograms"
    out_paths: List[Path] = []

    for loc in locations:
        df = extracted["surface_df_by_slug"].get(loc.slug)
        if df is None or len(df) < 2:
            continue
        met_dir.mkdir(parents=True, exist_ok=True)

        times = pd.to_datetime(df["valid_time_utc"], utc=True, errors="coerce")
        use_dt = bool(times.notna().all())
        x_vals = times if use_dt else df["lead_hours"].to_numpy()

        fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
        try:
            # Panel 1: T2m & Td2m
            ax0 = axes[0]
            ax0.plot(x_vals, df["t2m_c"], "o-", color="#d62728", linewidth=1.8, markersize=3.5, label="T2m (°C)")
            if df["td2m_c"].notna().any():
                ax0.plot(x_vals, df["td2m_c"], "s--", color="#2ca02c", linewidth=1.5, markersize=3.0, label="Td2m (°C)")
            ax0.axhline(0.0, color="#666666", linewidth=0.7, linestyle=":")
            ax0.set_ylabel("Temp (°C)")
            ax0.grid(True, alpha=0.3)
            ax0.legend(loc="upper right", fontsize=8)
            ax0.set_title(
                f"Tempest-NWP v2 Meteogram — {loc.name} ({loc.lat:.2f}°N, {loc.lon:.2f}°E | grid dist {df['grid_dist_km'].iloc[0]:.1f} km)",
                fontsize=11,
                fontweight="bold",
            )

            # Panel 2: Wind & Gusts
            ax1 = axes[1]
            ax1.plot(x_vals, df["wspd10m_kmh"], "o-", color="#1f77b4", linewidth=1.8, markersize=3.5, label="Wind 10m (km/h)")
            if df["gust10m_kmh"].notna().any():
                ax1.plot(x_vals, df["gust10m_kmh"], "^:", color="#ff7f0e", linewidth=1.4, markersize=3.0, label="Windstoot (km/h)")
            ax1.set_ylabel("Wind (km/h)")
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc="upper right", fontsize=8)

            # Panel 3: Precip step + MSLP
            ax2 = axes[2]
            bar_w = datetime.timedelta(hours=0.7) if use_dt else 0.7
            ax2.bar(x_vals, df["precip_step_mm"], width=bar_w, color="#4c9be8", alpha=0.7, label="Neerslag stap (mm)")
            ax2.set_ylabel("Neerslag (mm)")
            ax2.grid(True, alpha=0.3)
            ax2_r = ax2.twinx()
            if df["mslp_hpa"].notna().any():
                ax2_r.plot(x_vals, df["mslp_hpa"], "k-", linewidth=1.6, label="MSLP (hPa)")
                ax2_r.set_ylabel("MSLP (hPa)")
            h2a, l2a = ax2.get_legend_handles_labels()
            h2b, l2b = ax2_r.get_legend_handles_labels()
            ax2.legend(h2a + h2b, l2a + l2b, loc="upper right", fontsize=8)

            # Panel 4: TCC + CAPE
            ax3 = axes[3]
            if df["tcc_frac"].notna().any():
                ax3.fill_between(x_vals, 0.0, df["tcc_frac"] * 100.0, color="#8c96a0", alpha=0.45, label="Bewolking (%)")
            ax3.set_ylim(0, 105)
            ax3.set_ylabel("Bewolking (%)")
            ax3.grid(True, alpha=0.3)
            ax3_r = ax3.twinx()
            if df["cape_sb_j_kg"].notna().any():
                ax3_r.plot(x_vals, df["cape_sb_j_kg"], "r--", linewidth=1.4, label="SBCAPE (J/kg)")
                ax3_r.set_ylabel("CAPE (J/kg)")
            h3a, l3a = ax3.get_legend_handles_labels()
            h3b, l3b = ax3_r.get_legend_handles_labels()
            ax3.legend(h3a + h3b, l3a + l3b, loc="upper right", fontsize=8)

            if use_dt:
                ax3.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m\n%HZ"))
                ax3.set_xlabel("Valid Time (UTC)")
            else:
                ax3.set_xlabel("Lead Time (h)")

            fig.tight_layout()
            out_p = met_dir / f"{loc.slug}_meteogram.png"
            fig.savefig(out_p, dpi=dpi)
            out_paths.append(out_p)
        finally:
            plt.close(fig)

    if len(out_paths) == 0:
        logger.info("Skipping meteograms (need >= 2 forecast lead times).")
    return out_paths


def _prepare_2d_for_plot(lat2d: np.ndarray, lon2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Optional[int], bool]:
    """Normalize `[0, 359.75]` global grids to `[-180, 179.75]` via horizontal roll so prime meridian never splits."""
    lon_min = float(np.nanmin(lon2d))
    lon_max = float(np.nanmax(lon2d))
    ny, nx = lon2d.shape
    is_separable = bool(
        ny >= 2
        and nx >= 2
        and np.allclose(lat2d, lat2d[:, :1], atol=1e-4)
        and np.allclose(lon2d, lon2d[:1, :], atol=1e-4)
    )
    if is_separable and lon_max > 180.0 and lon_min >= -1e-3:
        lon_w = _wrap_lon_180_arr(lon2d[0, :])
        shift = int(np.argmin(lon_w))
        lon_rolled = np.roll(lon_w, -shift)
        lon2d_out = np.broadcast_to(lon_rolled[None, :], lon2d.shape).copy()
        return lat2d, lon2d_out, -shift, True
    return lat2d, _wrap_lon_180_arr(lon2d), None, is_separable


def _wrap_lon_180_arr(arr: np.ndarray) -> np.ndarray:
    return ((arr + 180.0) % 360.0) - 180.0


def _roll_field_if_needed(arr: np.ndarray, roll_shift: Optional[int]) -> np.ndarray:
    if roll_shift is None or roll_shift == 0:
        return arr
    return np.roll(arr, roll_shift, axis=-1)


def _compute_map_extent(
    lat2d_p: np.ndarray,
    lon2d_p: np.ndarray,
    locations: Sequence[LocationPoint],
    zoom_deg: Optional[float],
) -> Tuple[float, float, float, float]:
    d_lon_min, d_lon_max = float(np.nanmin(lon2d_p)), float(np.nanmax(lon2d_p))
    d_lat_min, d_lat_max = float(np.nanmin(lat2d_p)), float(np.nanmax(lat2d_p))
    is_global = (d_lon_max - d_lon_min) > 180.0

    if zoom_deg is not None and float(zoom_deg) <= 0.0:
        return (d_lon_min, d_lon_max, d_lat_min, d_lat_max)

    eff_pad = float(zoom_deg) if (zoom_deg is not None and float(zoom_deg) > 0.0) else (12.0 if is_global else 0.0)
    if eff_pad <= 0.0 or not locations:
        return (d_lon_min, d_lon_max, d_lat_min, d_lat_max)

    lats = [loc.lat for loc in locations]
    lons = [loc.lon for loc in locations]
    lon0 = max(d_lon_min, min(lons) - eff_pad)
    lon1 = min(d_lon_max, max(lons) + eff_pad)
    lat0 = max(d_lat_min, min(lats) - 0.7 * eff_pad)
    lat1 = min(d_lat_max, max(lats) + 0.7 * eff_pad)
    if lon1 - lon0 < 4.0:
        c = 0.5 * (lon0 + lon1)
        lon0, lon1 = max(d_lon_min, c - 4.0), min(d_lon_max, c + 4.0)
    if lat1 - lat0 < 3.0:
        c = 0.5 * (lat0 + lat1)
        lat0, lat1 = max(d_lat_min, c - 3.0), min(d_lat_max, c + 3.0)
    return (lon0, lon1, lat0, lat1)


def _crop_separable_for_extent(
    lat2d_p: np.ndarray,
    lon2d_p: np.ndarray,
    extent: Tuple[float, float, float, float],
    is_separable: bool,
) -> Tuple[slice, slice]:
    ny, nx = lat2d_p.shape
    if not is_separable:
        return slice(0, ny), slice(0, nx)
    lon0, lon1, lat0, lat1 = extent
    lat_col = lat2d_p[:, 0]
    lon_row = lon2d_p[0, :]
    ymask = (lat_col >= lat0 - 1.5) & (lat_col <= lat1 + 1.5)
    xmask = (lon_row >= lon0 - 1.5) & (lon_row <= lon1 + 1.5)
    if not np.any(ymask) or not np.any(xmask):
        return slice(0, ny), slice(0, nx)
    yj = np.where(ymask)[0]
    xi = np.where(xmask)[0]
    return slice(int(yj[0]), int(yj[-1]) + 1), slice(int(xi[0]), int(xi[-1]) + 1)


def _safe_levels(vmin: float, vmax: float, step: float, max_levels: int = 32) -> np.ndarray:
    if not (np.isfinite(vmin) and np.isfinite(vmax)) or vmax <= vmin + 1e-6:
        c = 0.0 if not np.isfinite(vmin) else vmin
        return np.linspace(c - 1.0, c + 1.0, 9)
    n_est = max(2, int((vmax - vmin) / max(step, 1e-6)))
    if n_est > max_levels:
        return np.linspace(vmin, vmax, max_levels)
    return np.linspace(vmin, vmax, max(7, n_est + 1))


def _annotate_locations_on_map(
    ax: Any,
    locations: Sequence[LocationPoint],
    extent: Tuple[float, float, float, float],
) -> None:
    lon0, lon1, lat0, lat1 = extent
    for loc in locations:
        if lon0 <= loc.lon <= lon1 and lat0 <= loc.lat <= lat1:
            ax.plot(
                loc.lon,
                loc.lat,
                marker="*",
                markersize=9,
                color="#111111",
                markeredgecolor="white",
                markeredgewidth=0.9,
                transform=ccrs.PlateCarree(),
                zorder=10,
            )
            ax.text(
                loc.lon + 0.015 * max(1.0, lon1 - lon0),
                loc.lat + 0.012 * max(1.0, lat1 - lat0),
                loc.name,
                fontsize=8,
                fontweight="bold",
                color="#111111",
                transform=ccrs.PlateCarree(),
                zorder=11,
                path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
            )


def render_maps(
    wx1_files: Sequence[Path],
    locations: Sequence[LocationPoint],
    domain: str,
    outdir: Path | str,
    *,
    map_types: Sequence[str] = ("t2m_mslp", "precip_wind", "cape_shear", "z500_t850"),
    zoom_deg: Optional[float] = None,
    dpi: int = 150,
    show_progress: bool = True,
) -> Dict[str, List[Path]]:
    """Render Cartopy weather map PNGs for each selected map type and forecast lead time."""
    active_maps = [m for m in map_types if m in VALID_MAP_TYPES]
    if not active_maps or not wx1_files:
        return {}

    maps_root = Path(outdir) / "maps"
    frames_by_type: Dict[str, List[Path]] = {m: [] for m in active_maps}

    with wx1.open_dataset(wx1_files[0]) as ds0:
        lat2d_raw = np.asarray(ds0["lat"].values, dtype=np.float64)
        lon2d_raw = np.asarray(ds0["lon"].values, dtype=np.float64)

    lat2d_p, lon2d_p, roll_shift, is_sep = _prepare_2d_for_plot(lat2d_raw, lon2d_raw)
    extent = _compute_map_extent(lat2d_p, lon2d_p, locations, zoom_deg)
    ys, xs = _crop_separable_for_extent(lat2d_p, lon2d_p, extent, is_sep)
    lat_sub = lat2d_p[ys, xs]
    lon_sub = lon2d_p[ys, xs]
    b_step = max(1, max(lat_sub.shape) // 22)

    prev_acc_sub: Optional[np.ndarray] = None
    file_iter = (
        _tqdm(wx1_files, desc="[export] Rendering maps", unit="file", dynamic_ncols=True)
        if (show_progress and _tqdm is not None and len(wx1_files) > 1)
        else wx1_files
    )

    for fp in file_iter:
        with wx1.open_dataset(fp) as ds:
            lead_hr = float(ds.attrs.get("lead_hours", _extract_lead_hours(fp)))
            valid_str = str(ds.attrs.get("valid_time_utc", ""))
            tag = _lead_tag(lead_hr)
            plev_hpa = (np.asarray(ds["plev"].values, dtype=np.float64) * 0.01) if "plev" in ds.coords else None

            def _read_sub2d(name: str) -> Optional[np.ndarray]:
                if name not in ds.data_vars:
                    return None
                arr = _roll_field_if_needed(np.asarray(ds[name].values, dtype=np.float64), roll_shift)
                return arr[ys, xs]

            def _read_sub_plev(name: str, hpa_tgt: float) -> Optional[np.ndarray]:
                if name not in ds.data_vars or plev_hpa is None:
                    return None
                hits = np.where(np.isclose(plev_hpa, hpa_tgt, atol=1.0))[0]
                if len(hits) == 0:
                    return None
                arr = _roll_field_if_needed(np.asarray(ds[name][int(hits[0])].values, dtype=np.float64), roll_shift)
                return arr[ys, xs]

            cur_acc_sub = np.zeros_like(lat_sub)
            for acc_k in ("rain_acc", "snow_acc", "graupel_acc"):
                a = _read_sub2d(acc_k)
                if a is not None:
                    cur_acc_sub = cur_acc_sub + np.nan_to_num(a, nan=0.0)
            step_acc_sub = np.maximum(0.0, cur_acc_sub - prev_acc_sub) if prev_acc_sub is not None else cur_acc_sub
            prev_acc_sub = cur_acc_sub

            for mtype in active_maps:
                out_dir = maps_root / mtype
                out_dir.mkdir(parents=True, exist_ok=True)
                out_png = out_dir / f"{mtype}_T+{tag}h.png"

                fig = plt.figure(figsize=(10.5, 7.2))
                try:
                    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
                    ax.set_extent(extent, crs=ccrs.PlateCarree())
                    ax.coastlines(resolution="10m" if (extent[1] - extent[0]) < 80.0 else "50m", linewidth=0.75, zorder=5)
                    ax.add_feature(cfeature.BORDERS, linewidth=0.55, edgecolor="#333333", zorder=5)
                    gl = ax.gridlines(draw_labels=True, linewidth=0.35, color="#888888", alpha=0.5, linestyle="--")
                    gl.top_labels = False
                    gl.right_labels = False

                    plotted = False
                    title_field = mtype
                    if mtype == "t2m_mslp":
                        t2m = _read_sub2d("t2m")
                        mslp = _read_sub2d("mslp")
                        u10 = _read_sub2d("u10m_geo")
                        v10 = _read_sub2d("v10m_geo")
                        if t2m is not None:
                            t2m_c = t2m - 273.15
                            lvls = _safe_levels(float(np.nanpercentile(t2m_c, 1)), float(np.nanpercentile(t2m_c, 99)), 2.0)
                            cf = ax.contourf(lon_sub, lat_sub, t2m_c, levels=lvls, cmap="RdYlBu_r", extend="both", transform=ccrs.PlateCarree(), zorder=1)
                            cb = fig.colorbar(cf, ax=ax, shrink=0.82, pad=0.03)
                            cb.set_label("2m Temperatuur (°C)")
                            plotted = True
                        if mslp is not None:
                            mslp_hpa = mslp * 0.01
                            pmin = math.floor(float(np.nanmin(mslp_hpa)) / 4.0) * 4.0
                            pmax = math.ceil(float(np.nanmax(mslp_hpa)) / 4.0) * 4.0
                            p_lvls = np.arange(pmin, max(pmin + 4.0, pmax) + 2.0, 4.0)
                            if len(p_lvls) >= 2:
                                cs = ax.contour(lon_sub, lat_sub, mslp_hpa, levels=p_lvls, colors="k", linewidths=0.85, transform=ccrs.PlateCarree(), zorder=4)
                                ax.clabel(cs, fmt="%d", fontsize=7)
                            plotted = True
                        if u10 is not None and v10 is not None:
                            ax.barbs(
                                lon_sub[::b_step, ::b_step],
                                lat_sub[::b_step, ::b_step],
                                u10[::b_step, ::b_step] * 1.943844,
                                v10[::b_step, ::b_step] * 1.943844,
                                length=5,
                                linewidth=0.55,
                                color="#222222",
                                transform=ccrs.PlateCarree(),
                                zorder=6,
                            )
                        title_field = "T2m (°C) + MSLP (hPa) + 10m Wind (kt)"

                    elif mtype == "precip_wind":
                        pr_rate = _read_sub2d("precip_rate")
                        pr_plot = step_acc_sub if float(np.nanmax(step_acc_sub)) > 1e-4 else (
                            pr_rate * 3600.0 if pr_rate is not None else step_acc_sub
                        )
                        lvls = _safe_levels(0.1, max(5.0, float(np.nanmax(pr_plot))), 1.0)
                        cf = ax.contourf(lon_sub, lat_sub, np.where(pr_plot >= 0.05, pr_plot, np.nan), levels=lvls, cmap="YlGnBu", extend="max", transform=ccrs.PlateCarree(), zorder=2)
                        cb = fig.colorbar(cf, ax=ax, shrink=0.82, pad=0.03)
                        cb.set_label("Neerslag (mm)")
                        u10 = _read_sub2d("u10m_geo")
                        v10 = _read_sub2d("v10m_geo")
                        if u10 is not None and v10 is not None:
                            wspd = np.hypot(u10, v10) * 3.6
                            ax.contour(lon_sub, lat_sub, wspd, levels=[30, 50, 75, 100], colors=["#555555", "#d95f02", "#e7298a", "#980043"], linewidths=0.8, transform=ccrs.PlateCarree(), zorder=4)
                            ax.barbs(
                                lon_sub[::b_step, ::b_step],
                                lat_sub[::b_step, ::b_step],
                                u10[::b_step, ::b_step] * 1.943844,
                                v10[::b_step, ::b_step] * 1.943844,
                                length=5,
                                linewidth=0.55,
                                color="#222222",
                                transform=ccrs.PlateCarree(),
                                zorder=6,
                            )
                        plotted = True
                        title_field = "Neerslag (mm) + 10m Wind (kt & km/h)"

                    elif mtype == "cape_shear":
                        cape = _read_sub2d("cape_sb")
                        shear = _read_sub2d("shear_0_6km")
                        if cape is not None:
                            lvls = _safe_levels(50.0, max(500.0, float(np.nanmax(cape))), 100.0)
                            cf = ax.contourf(lon_sub, lat_sub, np.where(cape >= 25.0, cape, np.nan), levels=lvls, cmap="YlOrRd", extend="max", transform=ccrs.PlateCarree(), zorder=2)
                            cb = fig.colorbar(cf, ax=ax, shrink=0.82, pad=0.03)
                            cb.set_label("SBCAPE (J kg⁻¹)")
                            plotted = True
                        if shear is not None and float(np.nanmax(shear)) > 5.0:
                            cs = ax.contour(lon_sub, lat_sub, shear, levels=[10, 15, 20, 25, 30], colors="#4a1486", linewidths=0.9, transform=ccrs.PlateCarree(), zorder=4)
                            ax.clabel(cs, fmt="%d m/s", fontsize=7)
                            plotted = True
                        title_field = "SBCAPE (J/kg) + 0-6 km Schering (m/s)"

                    elif mtype == "z500_t850":
                        t850 = _read_sub_plev("t_pl", 850.0)
                        gh500 = _read_sub_plev("gh_pl", 500.0)
                        u500 = _read_sub_plev("u_pl", 500.0)
                        v500 = _read_sub_plev("v_pl", 500.0)
                        if t850 is not None:
                            t850_c = t850 - 273.15
                            lvls = _safe_levels(float(np.nanpercentile(t850_c, 1)), float(np.nanpercentile(t850_c, 99)), 2.0)
                            cf = ax.contourf(lon_sub, lat_sub, t850_c, levels=lvls, cmap="coolwarm", extend="both", transform=ccrs.PlateCarree(), zorder=1)
                            cb = fig.colorbar(cf, ax=ax, shrink=0.82, pad=0.03)
                            cb.set_label("850 hPa Temperatuur (°C)")
                            plotted = True
                        if gh500 is not None:
                            gh_dam = gh500 * 0.1
                            gmin = math.floor(float(np.nanmin(gh_dam)) / 4.0) * 4.0
                            gmax = math.ceil(float(np.nanmax(gh_dam)) / 4.0) * 4.0
                            g_lvls = np.arange(gmin, max(gmin + 4.0, gmax) + 2.0, 4.0)
                            cs = ax.contour(lon_sub, lat_sub, gh_dam, levels=g_lvls, colors="k", linewidths=0.9, transform=ccrs.PlateCarree(), zorder=4)
                            ax.clabel(cs, fmt="%d", fontsize=7)
                            plotted = True
                        if u500 is not None and v500 is not None:
                            ax.barbs(
                                lon_sub[::b_step, ::b_step],
                                lat_sub[::b_step, ::b_step],
                                u500[::b_step, ::b_step] * 1.943844,
                                v500[::b_step, ::b_step] * 1.943844,
                                length=5,
                                linewidth=0.55,
                                color="#222222",
                                transform=ccrs.PlateCarree(),
                                zorder=6,
                            )
                        title_field = "500 hPa Geopotentiaal (dam) + 850 hPa Temp (°C)"

                    elif mtype == "sst_ice":
                        sst = _read_sub2d("sst")
                        if sst is None:
                            sst = _read_sub2d("sst_bulk")
                        sice = _read_sub2d("sea_ice_frac")
                        if sst is not None:
                            sst_c = np.where(sst > 200.0, sst - 273.15, np.nan)
                            valid_sst = sst_c[np.isfinite(sst_c)]
                            vmax_sst = float(np.nanmax(valid_sst)) if valid_sst.size else 25.0
                            lvls = _safe_levels(-2.0, max(25.0, vmax_sst), 1.5)
                            cf = ax.contourf(lon_sub, lat_sub, sst_c, levels=lvls, cmap="turbo", extend="both", transform=ccrs.PlateCarree(), zorder=1)
                            cb = fig.colorbar(cf, ax=ax, shrink=0.82, pad=0.03)
                            cb.set_label("Zeewatertemperatuur (°C)")
                            plotted = True
                        if sice is not None and float(np.nanmax(sice)) > 0.15:
                            ax.contourf(lon_sub, lat_sub, np.where(sice >= 0.15, sice, np.nan), levels=[0.15, 0.5, 0.85, 1.0], cmap="cool", alpha=0.65, transform=ccrs.PlateCarree(), zorder=3)
                            plotted = True
                        title_field = "NEMO SST (°C) + SI³ Zee-ijsfractie"

                    if not plotted:
                        plt.close(fig)
                        continue

                    _annotate_locations_on_map(ax, locations, extent)
                    ax.set_title(
                        f"Tempest-NWP v2 [{domain.upper()}] — {title_field}\nLead T+{tag}h | Valid: {valid_str}",
                        fontsize=10.5,
                        fontweight="bold",
                    )
                    fig.tight_layout()
                    fig.savefig(out_png, dpi=dpi)
                    frames_by_type[mtype].append(out_png)
                finally:
                    plt.close(fig)

    return {k: v for k, v in frames_by_type.items() if v}


def render_skewt_soundings(
    extracted: Dict[str, Any],
    locations: Sequence[LocationPoint],
    domain: str,
    outdir: Path | str,
    *,
    skewt_hours: Optional[Sequence[float]] = None,
    dpi: int = 150,
    show_progress: bool = True,
) -> Dict[str, List[Path]]:
    """Render MetPy Skew-T Log-P + Hodograph soundings per chosen location and forecast hour."""
    skew_root = Path(outdir) / "skewt"
    out_by_slug: Dict[str, List[Path]] = {}

    for loc in locations:
        sdf = extracted["sounding_df_by_slug"].get(loc.slug)
        ts_df = extracted["surface_df_by_slug"].get(loc.slug)
        if sdf is None or sdf.empty:
            continue

        all_hours = sorted({float(h) for h in sdf["lead_hours"].unique()})
        if skewt_hours is not None:
            sel_hours = [h for h in all_hours if any(abs(h - req) <= 0.05 for req in skewt_hours)]
        else:
            sel_hours = all_hours

        if not sel_hours:
            continue

        loc_dir = skew_root / loc.slug
        loc_dir.mkdir(parents=True, exist_ok=True)
        out_by_slug[loc.slug] = []

        hr_iter = (
            _tqdm(sel_hours, desc=f"[export] Skew-T {loc.name}", unit="hr", dynamic_ncols=True)
            if (show_progress and _tqdm is not None and len(sel_hours) > 1)
            else sel_hours
        )

        for hr in hr_iter:
            sub = sdf[sdf["lead_hours"] == hr].sort_values("plev_hpa", ascending=False)
            if len(sub) < 4:
                continue

            p_q = sub["plev_hpa"].to_numpy(dtype=np.float64) * units.hPa
            t_q = sub["t_c"].to_numpy(dtype=np.float64) * units.degC
            td_arr = np.minimum(sub["td_c"].to_numpy(dtype=np.float64), sub["t_c"].to_numpy(dtype=np.float64) - 0.05)
            td_q = td_arr * units.degC
            u_kt = sub["u_ms"].to_numpy(dtype=np.float64) * 1.943844 * units.knots
            v_kt = sub["v_ms"].to_numpy(dtype=np.float64) * 1.943844 * units.knots
            valid_str = str(sub["valid_time_utc"].iloc[0])
            tag = _lead_tag(hr)

            row_ts = ts_df[ts_df["lead_hours"] == hr].iloc[0] if (ts_df is not None and not ts_df[ts_df["lead_hours"] == hr].empty) else None

            fig = plt.figure(figsize=(10.0, 9.2))
            try:
                skew = SkewT(fig, rotation=45, rect=(0.08, 0.08, 0.72, 0.82))
                skew.plot(p_q, t_q, "#d62728", linewidth=2.2, label="Temperatuur (°C)")
                skew.plot(p_q, td_q, "#2ca02c", linewidth=2.0, label="Dauwpunt (°C)")
                skew.plot_barbs(p_q, u_kt, v_kt, xloc=1.02)
                skew.plot_dry_adiabats(alpha=0.35, linewidth=0.7)
                skew.plot_moist_adiabats(alpha=0.35, linewidth=0.7)
                skew.plot_mixing_lines(alpha=0.30, linewidth=0.7)
                skew.ax.axvline(0.0 * units.degC, color="#1f77b4", linestyle="--", linewidth=1.1, alpha=0.8)
                skew.ax.set_ylim(1020, 50)
                skew.ax.set_xlim(-50, 35)

                cape_val, cin_val = float("nan"), float("nan")
                try:
                    lcl_p, lcl_t = mpcalc.lcl(p_q[0], t_q[0], td_q[0])
                    skew.plot(lcl_p, lcl_t, "ko", markerfacecolor="yellow", markersize=5, label="LCL")
                    prof = mpcalc.parcel_profile(p_q, t_q[0], td_q[0]).to("degC")
                    skew.plot(p_q, prof, "k--", linewidth=1.4, label="Parcel-profiel")
                    skew.shade_cape(p_q, t_q, prof, alpha=0.22)
                    skew.shade_cin(p_q, t_q, prof, td_q, alpha=0.18)
                    c_out, ci_out = mpcalc.cape_cin(p_q, t_q, td_q, prof)
                    cape_val = float(c_out.to("J/kg").magnitude)
                    cin_val = float(ci_out.to("J/kg").magnitude)
                except Exception:
                    pass

                # Hodograph inset on top-right
                ax_hodo = fig.add_axes([0.67, 0.62, 0.28, 0.28])
                u_ms_arr = sub["u_ms"].to_numpy(dtype=np.float64)
                v_ms_arr = sub["v_ms"].to_numpy(dtype=np.float64)
                p_hpa_arr = sub["plev_hpa"].to_numpy(dtype=np.float64)
                u_kt_arr = u_ms_arr * 1.943844
                v_kt_arr = v_ms_arr * 1.943844
                max_kt = float(np.nanmax(np.hypot(u_kt_arr, v_kt_arr))) if len(u_kt_arr) else 60.0
                hodo_range = max(60.0, float(math.ceil(max_kt / 20.0) * 20.0)) if np.isfinite(max_kt) else 60.0
                h = Hodograph(ax_hodo, component_range=hodo_range)
                h.add_grid(increment=max(20.0, hodo_range / 4.0))
                h.plot_colormapped(u_kt_arr, v_kt_arr, p_hpa_arr, cmap="viridis_r", linewidth=1.8)
                ax_hodo.set_title("Hodograaf (kt)", fontsize=8.5, fontweight="bold")
                ax_hodo.tick_params(labelsize=7)

                # Classical sounding telemetry box bottom-right
                info_lines = [
                    f"Locatie: {loc.name}",
                    f"Coords: {loc.lat:.3f}°N, {loc.lon:.3f}°E",
                    f"Domein: {domain.upper()} | T+{tag}h",
                ]
                if row_ts is not None:
                    if np.isfinite(row_ts.get("t2m_c", float("nan"))):
                        info_lines.append(f"T2m / Td2m: {row_ts['t2m_c']:.1f} / {row_ts.get('td2m_c', float('nan')):.1f} °C")
                    if np.isfinite(row_ts.get("mslp_hpa", float("nan"))):
                        info_lines.append(f"MSLP: {row_ts['mslp_hpa']:.1f} hPa")
                    if np.isfinite(row_ts.get("cape_sb_j_kg", float("nan"))):
                        info_lines.append(f"Model SBCAPE: {row_ts['cape_sb_j_kg']:.0f} J/kg")
                    if np.isfinite(row_ts.get("shear_0_6km_ms", float("nan"))):
                        info_lines.append(f"0-6 km Schering: {row_ts['shear_0_6km_ms']:.1f} m/s")
                    if np.isfinite(row_ts.get("freezing_level_m", float("nan"))):
                        info_lines.append(f"Vorstgrens: {row_ts['freezing_level_m']:.0f} m")
                if np.isfinite(cape_val):
                    info_lines.append(f"Profiel CAPE/CIN: {cape_val:.0f} / {cin_val:.0f}")

                fig.text(
                    0.67,
                    0.56,
                    "\n".join(info_lines),
                    fontsize=8.2,
                    va="top",
                    family="monospace",
                    bbox=dict(boxstyle="round,pad=0.45", facecolor="#f7f7f7", edgecolor="#cccccc"),
                )

                skew.ax.legend(loc="upper left", fontsize=8)
                skew.ax.set_xlabel("Temperatuur (°C)")
                skew.ax.set_ylabel("Druk (hPa)")
                fig.suptitle(
                    f"Tempest-NWP v2 Skew-T Log-P — {loc.name} | T+{tag}h | Valid: {valid_str}",
                    fontsize=12,
                    fontweight="bold",
                    y=0.97,
                )

                out_p = loc_dir / f"skewt_{loc.slug}_T+{tag}h.png"
                fig.savefig(out_p, dpi=dpi)
                out_by_slug[loc.slug].append(out_p)
            finally:
                plt.close(fig)

    return {k: v for k, v in out_by_slug.items() if v}


def build_gif_animation(
    frame_paths: Sequence[Path],
    out_gif: Path | str,
    *,
    fps: float = 2.5,
    also_mp4: bool = False,
) -> Tuple[Optional[Path], Optional[Path]]:
    """Combine ordered PNG frames (`len >= 2`) into a Pillow looping GIF (and optional MP4 via ffmpeg)."""
    valid_frames = [Path(f) for f in frame_paths if Path(f).is_file()]
    if len(valid_frames) < 2:
        return None, None

    gif_p = Path(out_gif)
    gif_p.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = int(round(1000.0 / max(0.2, float(fps))))

    imgs: List[Image.Image] = []
    try:
        for fp in valid_frames:
            with Image.open(fp) as im:
                imgs.append(im.convert("RGB"))
        imgs[0].save(
            gif_p,
            save_all=True,
            append_images=imgs[1:],
            duration=duration_ms,
            loop=0,
            optimize=True,
        )
    finally:
        for im in imgs:
            im.close()

    mp4_p: Optional[Path] = None
    if also_mp4:
        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin is not None:
            cand_mp4 = gif_p.with_suffix(".mp4")
            list_txt = gif_p.with_suffix(".frames.txt")
            try:
                sec_per_frame = 1.0 / max(0.2, float(fps))
                lines = []
                for fp in valid_frames:
                    lines.append(f"file '{fp.resolve()}'")
                    lines.append(f"duration {sec_per_frame:.4f}")
                lines.append(f"file '{valid_frames[-1].resolve()}'")
                list_txt.write_text("\n".join(lines), encoding="utf-8")
                cmd = [
                    ffmpeg_bin,
                    "-y",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(list_txt),
                    "-vf",
                    "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    str(cand_mp4),
                ]
                proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
                if proc.returncode == 0 and cand_mp4.is_file():
                    mp4_p = cand_mp4
                else:
                    logger.warning("ffmpeg MP4 encode failed for %s: %s", gif_p.name, proc.stderr[-300:])
            finally:
                list_txt.unlink(missing_ok=True)
        else:
            logger.info("ffmpeg not found on PATH; skipping optional MP4 for %s", gif_p.name)

    return gif_p, mp4_p


def build_all_animations(
    map_frames: Dict[str, List[Path]],
    skewt_frames: Dict[str, List[Path]],
    outdir: Path | str,
    *,
    fps: float = 2.5,
    also_mp4: bool = False,
) -> List[Path]:
    """Build looping GIF (and optional MP4) animations for every map series and Skew-T series with >=2 frames."""
    anim_dir = Path(outdir) / "animations"
    created: List[Path] = []

    for mtype, frames in map_frames.items():
        gif_p, mp4_p = build_gif_animation(frames, anim_dir / f"{mtype}.gif", fps=fps, also_mp4=also_mp4)
        if gif_p is not None:
            created.append(gif_p)
        if mp4_p is not None:
            created.append(mp4_p)

    for slug, frames in skewt_frames.items():
        gif_p, mp4_p = build_gif_animation(frames, anim_dir / f"skewt_{slug}.gif", fps=fps, also_mp4=also_mp4)
        if gif_p is not None:
            created.append(gif_p)
        if mp4_p is not None:
            created.append(mp4_p)

    return created


def export_wx1_products(
    input_path: Path | str,
    location_queries: Sequence[str],
    *,
    domain: str = "auto",
    outdir: Optional[Path | str] = None,
    use_geopy: bool = True,
    geocoder: Optional[Any] = None,
    cache_path: Optional[Path | str] = None,
    map_types: Sequence[str] = ("t2m_mslp", "precip_wind", "cape_shear", "z500_t850"),
    zoom_deg: Optional[float] = None,
    skewt_hours: Optional[Sequence[float]] = None,
    fps: float = 2.5,
    dpi: int = 150,
    also_mp4: bool = False,
    enable_csv: bool = True,
    enable_meteograms: bool = True,
    enable_maps: bool = True,
    enable_skewt: bool = True,
    enable_animations: bool = True,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """End-to-end WX-1 forecast exporter: geopy resolution, CSVs, meteograms, maps, Skew-Ts & animations."""
    wx1_files, det_dom, default_out = discover_wx1_forecasts(input_path, domain=domain)
    out_root = Path(outdir) if outdir is not None else default_out
    out_root.mkdir(parents=True, exist_ok=True)

    locations = resolve_locations(
        location_queries,
        use_geopy=use_geopy,
        cache_path=cache_path,
        geocoder=geocoder,
    )
    logger.info(
        "[export] Domain=%s | files=%d (T+%.2fh .. T+%.2fh) | locations=%s | outdir=%s",
        det_dom,
        len(wx1_files),
        _extract_lead_hours(wx1_files[0]),
        _extract_lead_hours(wx1_files[-1]),
        ", ".join(f"{l.name} ({l.lat:.3f},{l.lon:.3f} [{l.source}])" for l in locations),
        out_root,
    )

    extracted = extract_all_locations_data(wx1_files, locations, det_dom, show_progress=show_progress)

    csv_files = export_location_csvs(extracted, out_root) if enable_csv else []
    meteogram_files = render_meteograms(extracted, locations, out_root, dpi=dpi) if enable_meteograms else []
    map_frames = (
        render_maps(
            wx1_files,
            locations,
            det_dom,
            out_root,
            map_types=map_types,
            zoom_deg=zoom_deg,
            dpi=dpi,
            show_progress=show_progress,
        )
        if enable_maps
        else {}
    )
    skewt_frames = (
        render_skewt_soundings(
            extracted,
            locations,
            det_dom,
            out_root,
            skewt_hours=skewt_hours,
            dpi=dpi,
            show_progress=show_progress,
        )
        if enable_skewt
        else {}
    )
    anim_files = (
        build_all_animations(map_frames, skewt_frames, out_root, fps=fps, also_mp4=also_mp4)
        if enable_animations
        else []
    )

    return {
        "domain": det_dom,
        "outdir": str(out_root),
        "n_wx1_files": len(wx1_files),
        "locations": [
            {"name": l.name, "slug": l.slug, "lat": l.lat, "lon": l.lon, "source": l.source} for l in locations
        ],
        "csv_files": [str(p) for p in csv_files],
        "meteogram_files": [str(p) for p in meteogram_files],
        "map_files": {k: [str(p) for p in v] for k, v in map_frames.items()},
        "skewt_files": {k: [str(p) for p in v] for k, v in skewt_frames.items()},
        "animation_files": [str(p) for p in anim_files],
    }
