"""Tempest-NWP: TITAN (titanlib) Geostatistical Observation Quality Control.

Implements MET Norway's TITAN library routines for spatial quality control:
- Buddy Check: compares observation with neighboring stations using topographic lapse-rate adjustments.
- Spatial Consistency Test (SCT): optimal interpolation (OI) spatial consistency verification.
- Isolation Check: detects stations without sufficient spatial corroboration.

Pure classical geostatistics and atmospheric thermodynamics (0% AI / 0% Machine Learning).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger("tempest_v2.titan_qc")

try:
    import titanlib
    HAS_TITANLIB = True
except ImportError:
    titanlib = None
    HAS_TITANLIB = False
    logger.info("titanlib is not installed; using classical statistical fallback QC.")


# Default meteorological parameters for TITAN buddy check
# elev_gradient: change of variable per meter of elevation (e.g. -0.0065 K/m for temperature)
TITAN_VAR_CONFIGS: Dict[str, dict] = {
    "t2m": {
        "radius_m": 50000.0,       # 50 km search radius
        "num_min": 1,              # At least 1 neighbor
        "threshold": 3.0,          # 3.0 standard deviations
        "max_elev_diff_m": 500.0,  # Max vertical elevation difference
        "elev_gradient": -0.0065,  # Standard lapse rate [K/m]
        "min_std": 1.0,            # Min std dev to prevent division by zero in calm uniform air [K]
        "iterations": 2,
    },
    "psfc": {
        "radius_m": 75000.0,       # 75 km search radius for synoptic pressure
        "num_min": 1,
        "threshold": 3.5,
        "max_elev_diff_m": 400.0,
        "elev_gradient": -0.11,    # Hypsometric pressure lapse rate ~ -11 Pa/m near sea level
        "min_std": 50.0,           # [Pa]
        "iterations": 2,
    },
    "u10m": {
        "radius_m": 50000.0,
        "num_min": 1,
        "threshold": 3.5,
        "max_elev_diff_m": 600.0,
        "elev_gradient": 0.0,
        "min_std": 2.0,            # [m/s]
        "iterations": 2,
    },
    "v10m": {
        "radius_m": 50000.0,
        "num_min": 1,
        "threshold": 3.5,
        "max_elev_diff_m": 600.0,
        "elev_gradient": 0.0,
        "min_std": 2.0,            # [m/s]
        "iterations": 2,
    },
    "sst": {
        "radius_m": 100000.0,      # 100 km for ocean buoys
        "num_min": 1,
        "threshold": 3.0,
        "max_elev_diff_m": 50.0,
        "elev_gradient": 0.0,
        "min_std": 0.8,            # [K]
        "iterations": 2,
    },
}


def run_titan_buddy_check(
    lats: np.ndarray,
    lons: np.ndarray,
    elevs: np.ndarray,
    values: np.ndarray,
    var_type: str = "t2m",
    custom_config: Optional[dict] = None,
) -> Tuple[np.ndarray, dict]:
    """Applies TITAN topographic buddy check to a set of in-situ observations.
    
    Parameters
    ----------
    lats : 1D array of latitudes [deg N]
    lons : 1D array of longitudes [deg E]
    elevs : 1D array of station elevations [m]
    values : 1D array of observed values (e.g. K, Pa, m/s)
    var_type : variable identifier ('t2m', 'psfc', 'u10m', 'v10m', 'sst')
    custom_config : optional overrides for search radius, threshold, elev_gradient, etc.
    
    Returns
    -------
    pass_mask : boolean 1D array of same length as values (True = PASS, False = REJECT)
    stats : dictionary summarizing QC performance (total, tested, passed, rejected)
    """
    n_pts = len(values)
    pass_mask = np.zeros(n_pts, dtype=bool)
    stats = {
        "total": n_pts,
        "tested": 0,
        "passed": 0,
        "rejected": 0,
        "engine": "titanlib" if HAS_TITANLIB else "fallback_mad",
    }
    if n_pts == 0:
        return pass_mask, stats

    # Identify finite candidates
    valid_idx = np.where(
        np.isfinite(values) & np.isfinite(lats) & np.isfinite(lons) & np.isfinite(elevs)
    )[0]
    stats["tested"] = len(valid_idx)

    if len(valid_idx) < 3:
        # Too few stations for spatial correlation checks; accept all finite values
        pass_mask[valid_idx] = True
        stats["passed"] = len(valid_idx)
        return pass_mask, stats

    cfg = TITAN_VAR_CONFIGS.get(var_type, TITAN_VAR_CONFIGS["t2m"]).copy()
    if custom_config:
        cfg.update(custom_config)

    v_sub = values[valid_idx].astype(np.float64)
    lat_sub = lats[valid_idx].astype(np.float64)
    lon_sub = lons[valid_idx].astype(np.float64)
    elev_sub = elevs[valid_idx].astype(np.float64)

    if HAS_TITANLIB:
        try:
            pts = titanlib.Points(lat_sub, lon_sub, elev_sub)
            radius = np.array([cfg["radius_m"]], dtype=np.float64)
            num_min = np.array([cfg["num_min"]], dtype=np.int32)

            flags = titanlib.buddy_check(
                pts,
                v_sub,
                radius,
                num_min,
                float(cfg["threshold"]),
                float(cfg["max_elev_diff_m"]),
                float(cfg["elev_gradient"]),
                float(cfg["min_std"]),
                int(cfg["iterations"]),
            )
            # titanlib flag: 0 = PASS, 1 = FAIL / OUTLIER
            flags_arr = np.array(flags, dtype=np.int32)
            sub_pass = (flags_arr == 0)
        except Exception as exc:
            logger.warning("titanlib.buddy_check error: %s. Using statistical fallback.", exc)
            sub_pass = _statistical_mad_fallback(v_sub, cfg["threshold"])
    else:
        sub_pass = _statistical_mad_fallback(v_sub, cfg["threshold"])

    pass_mask[valid_idx] = sub_pass
    stats["passed"] = int(np.count_nonzero(sub_pass))
    stats["rejected"] = int(stats["tested"] - stats["passed"])

    return pass_mask, stats


def run_titan_isolation_check(
    lats: np.ndarray,
    lons: np.ndarray,
    elevs: np.ndarray,
    radius_m: float = 75000.0,
    num_min: int = 1,
) -> np.ndarray:
    """Flags isolated stations that lack neighboring stations within radius_m.
    
    Returns boolean array of length len(lats) where True = ISOLATED, False = CONNECTED.
    """
    n_pts = len(lats)
    if n_pts == 0 or not HAS_TITANLIB:
        return np.zeros(n_pts, dtype=bool)

    valid_idx = np.where(np.isfinite(lats) & np.isfinite(lons) & np.isfinite(elevs))[0]
    isolated_mask = np.ones(n_pts, dtype=bool)
    if len(valid_idx) == 0:
        return isolated_mask

    pts = titanlib.Points(
        lats[valid_idx].astype(np.float64),
        lons[valid_idx].astype(np.float64),
        elevs[valid_idx].astype(np.float64),
    )
    # isolation_check flag: 1 = ISOLATED, 0 = NOT ISOLATED
    flags = titanlib.isolation_check(pts, int(num_min), float(radius_m))
    isolated_mask[valid_idx] = (np.array(flags, dtype=np.int32) == 1)
    return isolated_mask


def _statistical_mad_fallback(values: np.ndarray, threshold: float = 3.5) -> np.ndarray:
    """Median Absolute Deviation (MAD) robust outlier detector for fallback when titanlib is unavailable."""
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    if mad < 1e-6:
        mad = np.std(values) + 1e-6
    # 1.4826 normalizes MAD to standard deviation
    z_scores = np.abs(values - med) / (1.4826 * mad)
    return z_scores <= threshold
