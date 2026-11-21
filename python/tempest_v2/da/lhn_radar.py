"""Classical Latent Heat Nudging (LHN) for 1 km KNMI QPE / DWD RADOLAN Radar Rates.

Implements deterministic Stephan et al. (2008) / DWD ICON-D2 style latent heat nudging (0% AI / 0% ML):
- Compares radar-observed surface precipitation rate P_obs [kg/(m^2 s)] with model rate P_mod.
- Computes bounded column latent heating & moistening tendencies weighted toward the
  active condensation/cloud layer, spinning up convective updrafts smoothly without negative moisture.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

LV0 = 2.501e6
CP_D = 1004.5


def compute_lhn_tendencies(
    precip_obs_kg_m2_s: np.ndarray,
    precip_mod_kg_m2_s: np.ndarray,
    z_s_m: np.ndarray,
    hs_m: np.ndarray,
    rho_bar: np.ndarray,
    theta_p: np.ndarray,
    pi_bar: np.ndarray,
    qv: np.ndarray,
    qc: Optional[np.ndarray] = None,
    *,
    tau_lhn_sec: float = 600.0,
    p_ref_kg_m2_s: float = 0.1 / 3600.0,
    max_dtheta_per_hour_k: float = 8.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute physical (dtheta_dt [K/s], dqv_dt [kg/kg/s]) LHN tendencies.

    Positive wherever radar precipitation exceeds model precipitation, negative/evaporative
    (bounded so qv stays >= 0) where model overpredicts rain relative to radar.
    """
    p_obs = np.maximum(0.0, np.asarray(precip_obs_kg_m2_s, dtype=np.float64))
    p_mod = np.maximum(0.0, np.asarray(precip_mod_kg_m2_s, dtype=np.float64))
    dp = p_obs - p_mod

    z_agl = np.maximum(0.0, np.asarray(z_s_m, dtype=np.float64) - np.asarray(hs_m, dtype=np.float64)[None, :, :])
    nz, ny, nx = z_agl.shape
    dz = np.zeros_like(z_agl)
    dz[1:-1] = 0.5 * (z_agl[2:] - z_agl[:-2])
    dz[0] = z_agl[1] - z_agl[0]
    dz[-1] = z_agl[-1] - z_agl[-2]

    # Normalized mid-tropospheric latent heating profile (peaking around 2..7 km AGL)
    clim_profile = np.exp(-0.5 * ((z_agl - 4000.0) / 2200.0) ** 2) * (z_agl > 300.0) * (z_agl < 13_000.0)
    if qc is not None:
        qc_arr = np.maximum(0.0, np.asarray(qc, dtype=np.float64))
        has_cloud = np.sum(qc_arr, axis=0, keepdims=True) > 1e-6
        w_raw = np.where(has_cloud, qc_arr + 0.05 * clim_profile * np.max(qc_arr, axis=0, keepdims=True), clim_profile)
    else:
        w_raw = clim_profile

    col_mass_weight = np.maximum(np.sum(np.asarray(rho_bar, dtype=np.float64) * w_raw * dz, axis=0, keepdims=True), 1e-6)
    w_norm = w_raw / col_mass_weight  # integral(rho * w_norm * dz) == 1

    # Bounded Stephan et al. scaling factor alpha in [-0.8, +2.0]
    alpha_scale = np.clip((p_obs + p_ref_kg_m2_s) / (p_mod + p_ref_kg_m2_s) - 1.0, -0.8, 2.0)
    active = (p_obs > 1e-7) | (p_mod > 1e-7)
    dp_eff = np.where(active, dp, 0.0)

    # Energy consistency: column integral (rho * cp * pi * dtheta/dt * dz) == Lv * (P_obs - P_mod)
    pi_arr = np.maximum(0.1, np.asarray(pi_bar, dtype=np.float64))
    dtheta_dt = (LV0 / (CP_D * pi_arr)) * w_norm * dp_eff[None, :, :]

    max_rate_k_s = max_dtheta_per_hour_k / 3600.0
    dtheta_dt = np.clip(dtheta_dt, -max_rate_k_s, max_rate_k_s)

    # Moisture nudging toward moist spinup (positive where P_obs > P_mod), bounded so qv >= 0
    dqv_dt = 0.5 * w_norm * dp_eff[None, :, :]
    max_dry_rate = -0.9 * np.maximum(0.0, np.asarray(qv, dtype=np.float64)) / max(1.0, float(tau_lhn_sec))
    dqv_dt = np.maximum(dqv_dt, max_dry_rate)

    return dtheta_dt.astype(np.float32), dqv_dt.astype(np.float32)
