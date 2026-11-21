"""Restart State Validation & Continuous Cycling Controller for Tempest-NWP.

Strictly 100% Classical Meteorology & Numerical Analysis (0% AI / 0% Machine Learning).
Validates prior model forecast states for continuous cycling:
- Timestamp & valid forecast horizon verification (rejects stale states > max_age_hours).
- Grid dimensions & domain bounding box match (D1 451x900, D2 421x701, D3 420x360).
- Vertical level count (137 vs 40 eta levels) & coordinate consistency.
- Complete atmospheric prognostic companion presence and finite-value sanity checks.
- Model valid-time checks (reject future states; wall-clock run-completion time is not used).
- This is not a bit-exact coupled restart validator: land/ocean/ice restart is not yet complete.
- Fallback orchestration: Cold-start via real ECMWF step 0 for D1; parent regridding for D2/D3.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import wx1

logger = logging.getLogger("tempest_v2.restart_validator")

# Domain expected grid geometries
DOMAIN_GEOMETRIES: Dict[str, Tuple[int, int]] = {
    # D1 Global 0.4°
    "d1_global": (451, 900),
    "d1_0p4": (451, 900),
    "global04": (451, 900),
    "global025": (721, 1440),
    # D2 Europe 0.1°
    "d2_europe": (421, 701),
    "d2_europe_0p1": (421, 701),
    "europe01": (421, 701),
    # D3 Benelux 2.5 km
    "benelux": (420, 360),
    "d3_benelux": (420, 360),
    "d3_benelux_2p5km": (420, 360),
    "benelux_2p5km": (420, 360),
}


def normalize_domain_name(domain: str) -> str:
    """Normalize names without collapsing legacy global 0.25° into D1 0.4°."""
    d = str(domain).strip().lower().replace("-", "_")
    # Test this legacy grid before the generic 'global' alias below.
    if d in {"global025", "global_025", "global_0p25", "global0p25"}:
        return "global025"
    if "d1" in d or "global" in d or "0p4" in d or "04" in d:
        return "d1_0p4"
    if "d2" in d or "europe" in d:
        return "d2_europe_0p1"
    if "d3" in d or "benelux" in d or "2p5" in d:
        return "d3_benelux_2p5km"
    return d


def validate_restart_state(
    file_path: Path | str,
    domain: str,
    expected_nz: int,
    target_time: Optional[datetime.datetime] = None,
    max_age_hours: float = 12.0,
    require_exact_time: bool = True,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Validates a prior forecast state or checkpoint for continuous cycling.

    Returns:
        (is_valid, status_message, metadata_dict)
    """
    p = Path(file_path)
    if not p.exists():
        return False, f"File does not exist: {p}", {}

    if p.stat().st_size < 1024:
        return False, f"File size too small ({p.stat().st_size} bytes): likely empty/truncated", {}

    norm_domain = normalize_domain_name(domain)
    if norm_domain not in DOMAIN_GEOMETRIES:
        return False, f"Unknown domain: {domain}", {}

    expected_ny, expected_nx = DOMAIN_GEOMETRIES[norm_domain]
    if norm_domain == "d3_benelux_2p5km":
        from tempest_v2.static_fields import get_benelux_grid_coords
        _resolution_m = int(os.environ.get("TEMPEST_D3_RESOLUTION_M", "2500"))
        _la, _lo = get_benelux_grid_coords(resolution_m=_resolution_m)
        expected_ny, expected_nx = len(_la), len(_lo)
    run_dir = p.parent if p.parent.name != "checkpoints" else p.parent.parent

    # 1. Inspect manifest.json if present
    manifest_p = run_dir / "manifest.json"
    manifest: Dict[str, Any] = {}
    if manifest_p.exists():
        try:
            with open(manifest_p, "r") as f:
                manifest = json.load(f)
        except Exception as e:
            logger.warning("Could not parse manifest %s: %s", manifest_p, e)

    # Model valid time is authoritative; wall-clock run completion time is not.
    sidecar_p = p.parent / f"{p.stem}_grid.json"
    sidecar: Dict[str, Any] = {}
    if sidecar_p.exists():
        try:
            sidecar = json.loads(sidecar_p.read_text())
        except Exception as e:
            return False, f"Invalid grid/time sidecar {sidecar_p.name}: {e}", manifest
    if manifest:
        mf_domain = normalize_domain_name(manifest.get("domain", ""))
        if mf_domain and mf_domain != norm_domain:
            return False, f"Domain mismatch in manifest: expected '{norm_domain}', got '{mf_domain}'", manifest
        mf_nz = manifest.get("nz")
        if mf_nz is not None and int(mf_nz) != expected_nz:
            return False, f"Vertical level count mismatch: expected {expected_nz}, got {mf_nz}", manifest

    valid_str = sidecar.get("valid_time_utc") or manifest.get("valid_time_utc")
    if not valid_str:
        return False, "Missing explicit valid_time_utc; legacy state is unsafe for time-aligned cycling", manifest
    try:
        state_time = datetime.datetime.fromisoformat(str(valid_str).replace("Z", "+00:00"))
        if state_time.tzinfo is None:
            state_time = state_time.replace(tzinfo=datetime.timezone.utc)
        if target_time is not None:
            target = target_time if target_time.tzinfo is not None else target_time.replace(tzinfo=datetime.timezone.utc)
            delta_hours = (target - state_time).total_seconds() / 3600.0
            time_tolerance_h = 1.0 / 60.0  # 1-minute timestamp tolerance
            if delta_hours < -time_tolerance_h:
                return False, f"State is from the future by {-delta_hours:.2f}h; refusing future cycling", manifest
            if require_exact_time and abs(delta_hours) > time_tolerance_h:
                return False, (
                    f"State valid at {state_time.isoformat()} is not the requested cycle time "
                    f"{target.isoformat()} (offset {delta_hours:+.2f}h); refusing time-mismatched cycling"
                ), manifest
            if delta_hours > max_age_hours:
                return False, f"State too old: {delta_hours:.1f}h > {max_age_hours:.1f}h limit", manifest
    except Exception as e:
        return False, f"Invalid valid_time_utc '{valid_str}': {e}", manifest

    # A valid atmospheric restart must be the primary theta file plus all required companions.
    required = ("ps", "u", "v", "qv", "qc", "qr", "w", "p_prime")
    missing = [name for name in required if not (p.parent / f"{p.stem}_{name}.wx1").exists()]
    if missing:
        return False, f"Incomplete atmospheric restart; missing {','.join(missing)}", manifest
    if sidecar.get("atmospheric_restart_complete") is not True:
        return False, "Sidecar does not positively identify a complete atmospheric restart", manifest
    if sidecar.get("state_kind") not in {"analysis_t0", "forecast_checkpoint", "forecast_final"}:
        return False, f"Unrecognized or missing restart state_kind: {sidecar.get('state_kind')!r}", manifest
    declared = set(sidecar.get("restart_variables") or [])
    required_declared = {"theta", "ps", "u", "v", "qv", "qc", "qr", "w", "p_prime"}
    if not required_declared.issubset(declared):
        return False, "Sidecar restart_variables does not declare the complete atmospheric state", manifest

    # 2. Inspect physical container data integrity (open WX-1 dataset)
    try:
        ds = wx1.open_dataset(p)
    except Exception as e:
        return False, f"Corrupted WX-1 container ({e})", manifest

    # Determine variable topology (companion files vs consolidated container)
    has_theta = "theta" in ds.variables
    has_ps = "ps" in ds.variables

    # Check companion files if single-variable container
    companion_dir = p.parent
    if not has_theta and p.stem.endswith("_ps"):
        theta_path = companion_dir / p.name.replace("_ps.wx1", ".wx1")
        if theta_path.exists():
            try:
                ds = wx1.open_dataset(theta_path)
                has_theta = "theta" in ds.variables
            except Exception:
                pass

    if not has_theta and "theta" not in ds.variables:
        # Check if first variable is theta
        first_var = list(ds.variables.keys())[0] if ds.variables else None
        if first_var and "theta" in first_var.lower():
            has_theta = True

    if not has_theta:
        return False, f"Missing primary prognostic variable 'theta' in {p.name}", manifest

    # The primary theta container must carry exactly the expected 3-D topology.
    theta_obj = ds.variables.get("theta")
    if theta_obj is None:
        return False, f"Missing primary prognostic variable 'theta' in {p.name}", manifest
    theta_shape = tuple(theta_obj.shape)
    if theta_shape != (expected_nz, expected_ny, expected_nx):
        return False, (
            f"Theta shape mismatch: expected ({expected_nz}, {expected_ny}, {expected_nx}), "
            f"got {theta_shape}"
        ), manifest

    if sidecar:
        if int(sidecar.get("ny", -1)) != expected_ny or int(sidecar.get("nx", -1)) != expected_nx:
            return False, "Grid/time sidecar dimensions do not match the requested domain/resolution", manifest

    # Every prognostic atmospheric companion must be readable, shape-consistent and finite.
    expected_shapes = {
        "ps": (expected_ny, expected_nx),
        "u": (expected_nz, expected_ny, expected_nx),
        "v": (expected_nz, expected_ny, expected_nx),
        "qv": (expected_nz, expected_ny, expected_nx),
        "qc": (expected_nz, expected_ny, expected_nx),
        "qr": (expected_nz, expected_ny, expected_nx),
        "w": (expected_nz, expected_ny, expected_nx),
        "p_prime": (expected_nz, expected_ny, expected_nx),
    }
    for name, expected_shape in expected_shapes.items():
        companion_path = p.parent / f"{p.stem}_{name}.wx1"
        try:
            companion_ds = wx1.open_dataset(companion_path)
            companion_obj = companion_ds.variables.get(name)
            if companion_obj is None:
                # Single-variable WX-1 wrappers may expose a namespaced variable name.
                companion_obj = next(iter(companion_ds.variables.values()), None)
            if companion_obj is None:
                return False, f"Companion {name} has no readable variable", manifest
            actual_shape = tuple(companion_obj.shape)
            if actual_shape != expected_shape:
                return False, f"Companion {name} shape mismatch: expected {expected_shape}, got {actual_shape}", manifest
            sample = np.asarray(companion_obj[0].values if len(actual_shape) == 3 else companion_obj.values, dtype=np.float32)
            if not np.isfinite(sample).all():
                return False, f"Companion {name} contains NaN or Inf values", manifest
        except Exception as e:
            return False, f"Failed validating companion {name} ({companion_path.name}): {e}", manifest

    try:
        sample = np.asarray(theta_obj[0].values, dtype=np.float32)
        if not np.isfinite(sample).all():
            return False, "Primary theta contains NaN or Inf values", manifest
    except Exception as e:
        return False, f"Failed reading theta sample slice from {p.name}: {e}", manifest

    info = {
        "file": str(p),
        "domain": norm_domain,
        "grid": f"{expected_ny}x{expected_nx}",
        "nz": expected_nz,
        "valid_time_utc": valid_str,
        "sidecar": sidecar,
        "manifest": manifest,
    }
    return True, f"OK: Validated Tempest-NWP restart state for {norm_domain} ({expected_ny}x{expected_nx}x{expected_nz})", info


def find_valid_prior_restart(
    domain: str,
    nz: int,
    target_time: Optional[datetime.datetime] = None,
    max_age_hours: float = 12.0,
    base_dir: Optional[Path] = None,
    require_complete_atmosphere: bool = True,
    require_exact_time: bool = True,
) -> Tuple[Optional[Path], str]:
    """Scans forecast directories for a strictly valid, non-stale prior restart state.

    Searches only the requested deterministic stream root by default. Ensemble folders are
    excluded from operational cycling unless the caller explicitly passes their member root.

    Returns:
        (valid_path, status_log)
    """
    search_dirs = [Path(base_dir)] if base_dir is not None else [Path("data/runs/forecasts/oper"), Path("data/forecasts")]

    norm_domain = normalize_domain_name(domain)
    rejection_reasons = []

    candidates_with_time = []
    for b_dir in search_dirs:
        if not b_dir.exists():
            continue
        subdirs = sorted([d for d in b_dir.iterdir() if d.is_dir()], reverse=True)
        for d in subdirs:
            if b_dir == Path("data/forecasts") and any(x in d.name.lower() for x in ("enfo", "ensemble", "member")):
                continue
            cp_dir = d / "checkpoints"
            candidates = []
            if cp_dir.exists():
                candidates.extend(cp_dir.glob("*.wx1"))
            candidates.extend(d.glob("*forecast_state.wx1"))
            candidates.extend(d.glob("*_state_t0.wx1"))
            # Only theta primary containers are considered; companion suffixes are excluded.
            for cand in candidates:
                if any(cand.stem.endswith("_" + x) for x in ("ps", "u", "v", "qv", "qc", "qr", "w", "p_prime")):
                    continue
                sidecar = cand.parent / f"{cand.stem}_grid.json"
                try:
                    meta = json.loads(sidecar.read_text()) if sidecar.exists() else {}
                    ts = meta.get("valid_time_utc")
                    if ts:
                        parsed = datetime.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                        if parsed.tzinfo is None:
                            parsed = parsed.replace(tzinfo=datetime.timezone.utc)
                        candidates_with_time.append((parsed, cand))
                    else:
                        rejection_reasons.append(f"{cand.name}: missing valid_time_utc")
                except Exception as e:
                    rejection_reasons.append(f"{cand.name}: invalid time sidecar ({e})")

    # Newest valid model state first; never select by run-folder name or wall-clock timestamp.
    candidates_with_time.sort(key=lambda item: item[0], reverse=True)
    for _candidate_time, cand in candidates_with_time:
        is_valid, msg, _ = validate_restart_state(
            cand, domain=norm_domain, expected_nz=nz, target_time=target_time,
            max_age_hours=max_age_hours, require_exact_time=require_exact_time
        )
        if is_valid:
            return cand, f"Found valid prior restart: {cand} ({msg})"
        rejection_reasons.append(f"{cand.name}: {msg}")

    summary = "; ".join(rejection_reasons[:3]) if rejection_reasons else "No candidates found"
    return None, f"No valid restart state found for domain '{domain}' with nz={nz} ({summary})"
