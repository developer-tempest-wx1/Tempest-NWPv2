"""Tempest-NWP: WMO Standard Isobaric (hPa) Vertical Interpolator.

Performs hydrostatic integration and log-pressure vertical interpolation from
model hybrid/sigma coordinates to standard WMO mandatory isobaric levels:
[1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10] hPa.

Computes:
- Geopotential Height (Z [gpm]) via hydrostatic hypsometric column integration
- Temperature (T [K])
- Zonal Wind (u [m/s])
- Meridional Wind (v [m/s])
- Specific Humidity (q [kg/kg])
- Vertical Velocity (omega [Pa/s])
- Relative Humidity (RH [%])

Strictly 100% classical physics (0% AI / 0% Machine Learning).
Conforms to WMO-No. 306 & CF-1.8 metadata standards.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Sequence

import numpy as np
import wx1

logger = logging.getLogger("tempest_v2.vertical_interpolator")

# Physical constants (SI units)
R_D = 287.058        # Gas constant for dry air [J/(kg K)]
G = 9.80665          # Standard gravitational acceleration [m/s^2]
CP = 1004.64         # Specific heat of dry air at constant pressure [J/(kg K)]
KAPPA = R_D / CP     # 0.285714
P0 = 100000.0        # Reference pressure [Pa]
GAMMA_STD = 0.0065   # Standard atmospheric lapse rate [K/m]

# WMO standard mandatory isobaric surfaces [hPa] (Troposphere + Tropopause + Stratosphere down to 1 hPa)
# plus intermediate & boundary-layer diagnostic surfaces (950, 900, 800, 600, 350 hPa)
WMO_MANDATORY_LEVELS_HPA = np.array([
    1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10, 7, 5, 3, 2, 1
], dtype=np.float32)

EXTRA_OPERATIONAL_LEVELS_HPA = np.array([
    950, 900, 800, 600, 350
], dtype=np.float32)

WMO_ISOBARIC_LEVELS_HPA = np.array([
    1000, 950, 925, 900, 850, 800, 700, 600, 500, 400, 350, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10, 7, 5, 3, 2, 1
], dtype=np.float32)

WMO_ISOBARIC_LEVELS_PA = WMO_ISOBARIC_LEVELS_HPA * 100.0


def compute_3d_pressure(
    ps: np.ndarray,
    nz: int,
    p_top: float = 1000.0,
    a_coeffs: Optional[np.ndarray] = None,
    b_coeffs: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Computes 3D full-level pressure field [nz, ny, nx] in Pascals.
    
    If hybrid coefficients (A_k, B_k) are not supplied, uses standard
    terrain-following sigma-pressure distribution:
        p(k) = p_top + sigma(k) * (ps - p_top)
    """
    ny, nx = ps.shape
    if a_coeffs is not None and b_coeffs is not None:
        p3d = np.zeros((nz, ny, nx), dtype=np.float32)
        for k in range(nz):
            p3d[k] = a_coeffs[k] * P0 + b_coeffs[k] * ps
        return p3d

    # Standard quadratic sigma distribution concentrated towards surface & tropopause
    sigma = np.linspace(1.0, 0.0, nz, endpoint=False) - (0.5 / nz)
    p3d = np.zeros((nz, ny, nx), dtype=np.float32)
    for k in range(nz):
        sig = float(sigma[k])
        p3d[k] = p_top + sig * (ps - p_top)
    return p3d


def compute_hydrostatic_geopotential(
    p3d: np.ndarray,
    theta: np.ndarray,
    qv: np.ndarray,
    zs: np.ndarray,
) -> np.ndarray:
    """Computes hydrostatic geopotential height Z [nz, ny, nx] in geopotential meters (gpm).
    
    Uses hypsometric integration from surface upwards:
        Phi(z) = g * zs + int_{p}^{ps} R_d * T_v * d(ln p)
        Z(z) = Phi(z) / g
    where T_v = T * (1 + 0.608 * qv) is virtual temperature.
    """
    nz, ny, nx = p3d.shape
    # Absolute temperature T = theta * (p / p0)^kappa
    t3d = theta * ((p3d / P0) ** KAPPA)
    # Virtual temperature T_v
    tv3d = t3d * (1.0 + 0.608 * np.maximum(qv, 0.0))

    # Geopotential in m^2/s^2
    phi = np.zeros((nz, ny, nx), dtype=np.float32)
    # Surface geopotential
    phi_s = (G * zs).astype(np.float32)

    # Log pressure
    ln_p = np.log(np.maximum(p3d, 1.0))

    # Integrate layer by layer upwards from lowest level (k=0 is surface-most level)
    # Layer 0 from surface to midpoint of level 0
    # Approximate surface pressure from level 0
    phi[0] = phi_s + R_D * tv3d[0] * 0.5 * np.maximum(0.0, np.log(101325.0) - ln_p[0])

    for k in range(1, nz):
        # Trapezoidal rule in ln(p): dPhi = - R_d * 0.5 * (Tv_{k} + Tv_{k-1}) * (ln p_k - ln p_{k-1})
        # Note: as k increases, pressure decreases, so ln_p[k] < ln_p[k-1]
        d_ln_p = ln_p[k - 1] - ln_p[k]
        tv_mean = 0.5 * (tv3d[k] + tv3d[k - 1])
        phi[k] = phi[k - 1] + R_D * tv_mean * d_ln_p

    # Convert to geopotential meters [gpm]
    z_gpm = phi / G
    return z_gpm, t3d


def interpolate_column_to_isobaric(
    p_profile: np.ndarray,      # [nz] Pa, descending from surface to top
    field_profile: np.ndarray,  # [nz]
    target_p_pa: np.ndarray,    # [n_target] Pa
    extrapolate_mode: str = "linear",
    surface_val: Optional[float] = None,
) -> np.ndarray:
    """Interpolates a 1D vertical column to target pressure levels in ln(p)."""
    ln_p_model = np.log(np.maximum(p_profile, 1.0))
    ln_p_target = np.log(np.maximum(target_p_pa, 1.0))

    # Model goes from high pressure (surface) to low pressure (top).
    # np.interp expects strictly increasing x-coordinates, so sort ascending in ln(p)
    idx_sort = np.argsort(ln_p_model)
    x_sort = ln_p_model[idx_sort]
    y_sort = field_profile[idx_sort]

    # Standard interp
    left_val = y_sort[0] if surface_val is None else surface_val
    right_val = y_sort[-1]
    return np.interp(ln_p_target, x_sort, y_sort, left=left_val, right=right_val).astype(np.float32)


def interpolate_model_to_wmo_isobaric(
    ps: np.ndarray,             # [ny, nx] Pa
    zs: np.ndarray,             # [ny, nx] m
    theta: np.ndarray,          # [nz, ny, nx] K
    u: np.ndarray,              # [nz, ny, nx] m/s
    v: np.ndarray,              # [nz, ny, nx] m/s
    qv: np.ndarray,             # [nz, ny, nx] kg/kg
    w: Optional[np.ndarray] = None,  # [nz, ny, nx] m/s
    isobaric_levels_hpa: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Interpolates full 3D atmospheric model state onto standard WMO isobaric pressure levels.
    
    Parameters
    ----------
    ps : Surface pressure in Pa [ny, nx]
    zs : Surface geopotential height / topography in meters [ny, nx]
    theta : Potential temperature in K [nz, ny, nx]
    u : Zonal wind in m/s [nz, ny, nx]
    v : Meridional wind in m/s [nz, ny, nx]
    qv : Specific humidity in kg/kg [nz, ny, nx]
    w : Vertical velocity in m/s [nz, ny, nx]
    isobaric_levels_hpa : Array of isobaric surfaces in hPa (defaults to 17 WMO standard levels)

    Returns
    -------
    dict of arrays, each shaped [n_levels, ny, nx]:
        "geopotential_height" : Z [gpm]
        "temperature"         : T [K]
        "u_wind"              : u [m/s]
        "v_wind"              : v [m/s]
        "specific_humidity"   : qv [kg/kg]
        "vertical_velocity"   : omega [Pa/s]
    """
    if isobaric_levels_hpa is None:
        isobaric_levels_hpa = WMO_ISOBARIC_LEVELS_HPA

    target_pa = isobaric_levels_hpa * 100.0
    n_levels = len(target_pa)
    nz, ny, nx = theta.shape

    # 1. Compute 3D model pressure
    p3d = compute_3d_pressure(ps, nz)

    # 2. Compute hydrostatic geopotential height and absolute temperature
    z3d, t3d = compute_hydrostatic_geopotential(p3d, theta, qv, zs)

    # 3. Vectorized log-pressure vertical interpolation
    out_z = np.zeros((n_levels, ny, nx), dtype=np.float32)
    out_t = np.zeros((n_levels, ny, nx), dtype=np.float32)
    out_u = np.zeros((n_levels, ny, nx), dtype=np.float32)
    out_v = np.zeros((n_levels, ny, nx), dtype=np.float32)
    out_q = np.zeros((n_levels, ny, nx), dtype=np.float32)
    out_rh = np.zeros((n_levels, ny, nx), dtype=np.float32)
    out_ws = np.zeros((n_levels, ny, nx), dtype=np.float32)
    out_w = np.zeros((n_levels, ny, nx), dtype=np.float32)

    ln_target_p = np.log(target_pa)

    # Loop over target levels (n_levels is typically 17, very fast vectorization over horizontal grid)
    for ilev, p_tgt in enumerate(target_pa):
        ln_pt = ln_target_p[ilev]

        # Subterranean mask: where target pressure is higher than local surface pressure
        sub_mask = p_tgt > ps
        atm_mask = ~sub_mask

        # For points within the atmospheric column (ps >= p_tgt >= p_top)
        # Find bracketing indices k where p3d[k] >= p_tgt >= p3d[k+1]
        # Pressure decreases with level index k: p3d[0] > p3d[1] > ... > p3d[nz-1]
        # Using vectorized search across levels
        ln_p3d = np.log(np.maximum(p3d, 1.0))

        # Vectorized bracket locator
        # Find the highest level index k where p3d[k] >= p_tgt
        bracket_k = np.zeros((ny, nx), dtype=np.int32)
        for k in range(nz - 1):
            valid_bracket = (p3d[k] >= p_tgt) & (p3d[k + 1] <= p_tgt)
            bracket_k[valid_bracket] = k

        # Handle top extrapolation (p_tgt < p3d[nz-1])
        top_mask = p_tgt < p3d[nz - 1]
        bracket_k[top_mask] = nz - 2

        # Extract bracketing values
        # k0 = bracket_k, k1 = bracket_k + 1
        k0 = bracket_k
        k1 = np.minimum(bracket_k + 1, nz - 1)

        y_idx, x_idx = np.indices((ny, nx))

        ln_p0 = ln_p3d[k0, y_idx, x_idx]
        ln_p1 = ln_p3d[k1, y_idx, x_idx]
        d_ln_p = np.maximum(ln_p0 - ln_p1, 1e-6)

        # Interpolation weight in ln(p): w = (ln(p0) - ln(p_tgt)) / (ln(p0) - ln(p1))
        weight = np.clip((ln_p0 - ln_pt) / d_ln_p, 0.0, 1.0)

        # Standard linear interpolation in ln(p) for atmospheric points
        z_interp = (1.0 - weight) * z3d[k0, y_idx, x_idx] + weight * z3d[k1, y_idx, x_idx]
        t_interp = (1.0 - weight) * t3d[k0, y_idx, x_idx] + weight * t3d[k1, y_idx, x_idx]
        u_interp = (1.0 - weight) * u[k0, y_idx, x_idx] + weight * u[k1, y_idx, x_idx]
        v_interp = (1.0 - weight) * v[k0, y_idx, x_idx] + weight * v[k1, y_idx, x_idx]
        q_interp = np.maximum(0.0, (1.0 - weight) * qv[k0, y_idx, x_idx] + weight * qv[k1, y_idx, x_idx])

        if w is not None:
            # Omega approximation: omega = - rho * g * w
            rho = p_tgt / (R_D * np.maximum(t_interp, 180.0))
            w_m_s = (1.0 - weight) * w[k0, y_idx, x_idx] + weight * w[k1, y_idx, x_idx]
            omega_interp = -rho * G * w_m_s
        else:
            omega_interp = np.zeros((ny, nx), dtype=np.float32)

        # Subterranean reduction (WMO standard extrapolation for isobaric surfaces below terrain)
        if np.any(sub_mask):
            ts_sfc = t3d[0]
            # Standard WMO lapse rate temperature downward extrapolation:
            # T(p) = Ts * (p / ps)^(R_d * Gamma / g)
            exp_factor = (R_D * GAMMA_STD) / G
            p_ratio = p_tgt / np.maximum(ps, 50000.0)
            t_sub = ts_sfc * (p_ratio ** exp_factor)
            
            # Hydrostatic geopotential height downward extrapolation:
            # Z(p) = zs - (Ts / Gamma) * ((p / ps)^(R_d * Gamma / g) - 1.0)
            z_sub = zs - (ts_sfc / GAMMA_STD) * ((p_ratio ** exp_factor) - 1.0)

            # Assign subterranean points
            t_interp[sub_mask] = t_sub[sub_mask]
            z_interp[sub_mask] = z_sub[sub_mask]
            # Surface wind smoothly damped underground
            u_interp[sub_mask] = u[0][sub_mask] * 0.8
            v_interp[sub_mask] = v[0][sub_mask] * 0.8
            q_interp[sub_mask] = qv[0][sub_mask]
            omega_interp[sub_mask] = 0.0

        out_z[ilev] = z_interp
        out_t[ilev] = t_interp
        out_u[ilev] = u_interp
        out_v[ilev] = v_interp
        out_q[ilev] = q_interp
        out_w[ilev] = omega_interp

        # Relative humidity and horizontal wind speed on isobaric surface
        t_safe = np.maximum(t_interp, 150.0)
        e_sat = 611.2 * np.exp(np.where(t_safe >= 273.15,
                                        17.67 * (t_safe - 273.15) / (t_safe - 29.65),
                                        21.875 * (t_safe - 273.15) / (t_safe - 7.66)))
        q_safe = np.maximum(q_interp, 1e-7)
        e_vap = (p_tgt * q_safe) / (0.622 + 0.378 * q_safe)
        out_rh[ilev] = np.clip(100.0 * (e_vap / np.maximum(e_sat, 1e-3)), 0.0, 100.0).astype(np.float32)
        out_ws[ilev] = np.sqrt(u_interp**2 + v_interp**2).astype(np.float32)

    return {
        "geopotential_height": out_z,
        "temperature": out_t,
        "u_wind": out_u,
        "v_wind": out_v,
        "specific_humidity": out_q,
        "relative_humidity": out_rh,
        "wind_speed": out_ws,
        "vertical_velocity": out_w,
    }


def save_isobaric_pl_to_wx1(
    out_dir: Path,
    pl_data: Dict[str, np.ndarray],
    isobaric_levels_hpa: np.ndarray = WMO_ISOBARIC_LEVELS_HPA,
    run_timestamp: Optional[str] = None,
) -> None:
    """Saves isobaric pressure level products into WX-1 containers in the pl/ directory.
    
    Files created:
        gh_pl.wx1 : Geopotential Height [n_levels, ny, nx] in gpm
        t_pl.wx1  : Temperature [n_levels, ny, nx] in K
        u_pl.wx1  : Zonal wind [n_levels, ny, nx] in m/s
        v_pl.wx1  : Meridional wind [n_levels, ny, nx] in m/s
        q_pl.wx1  : Specific humidity [n_levels, ny, nx] in kg/kg
        w_pl.wx1  : Vertical velocity [n_levels, ny, nx] in Pa/s
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    levels_str = ",".join(str(int(lev)) for lev in isobaric_levels_hpa)

    var_metadata = {
        "geopotential_height": ("gh_pl.wx1", "geopotential_height", "gpm", "Geopotential Height on WMO isobaric levels"),
        "temperature": ("t_pl.wx1", "air_temperature", "K", "Air Temperature on WMO isobaric levels"),
        "u_wind": ("u_pl.wx1", "eastward_wind", "m/s", "Zonal eastward wind component on WMO isobaric levels"),
        "v_wind": ("v_pl.wx1", "northward_wind", "m/s", "Meridional northward wind component on WMO isobaric levels"),
        "specific_humidity": ("q_pl.wx1", "specific_humidity", "kg/kg", "Specific humidity on WMO isobaric levels"),
        "vertical_velocity": ("w_pl.wx1", "vertical_velocity", "Pa/s", "Vertical velocity (omega) on WMO isobaric levels"),
    }

    for key, (filename, std_name, units, long_name) in var_metadata.items():
        if key not in pl_data:
            continue
        arr = pl_data[key].astype(np.float32)
        filepath = out_dir / filename
        c_shape = [1, arr.shape[1], arr.shape[2]]
        wx1.write_array(
            filepath,
            arr,
            dataset_id=f"wmo_pl_{key}",
            variable_path=key,
            dimensions=["isobaric_level", "latitude", "longitude"],
            chunk_shape=c_shape,
            profile="output",
            codec="zstd_shuffle",
            attributes={
                "standard_name": std_name,
                "long_name": long_name,
                "units": units,
                "isobaric_levels_hpa": levels_str,
                "wmo_conventions": "WMO-No. 306 Manual on Codes",
                "init_time_zulu": run_timestamp or "UNKNOWN",
            },
        )
        logger.info("Saved WMO isobaric level field %s -> %s", key, filepath)


def save_isobaric_pl_per_level_to_wx1(
    out_dir: Path,
    pl_data: Dict[str, np.ndarray],
    isobaric_levels_hpa: np.ndarray = WMO_ISOBARIC_LEVELS_HPA,
    run_timestamp: Optional[str] = None,
    file_prefix: Optional[str] = None,
) -> None:
    """Saves isobaric pressure level products PER ISOBARIC LEVEL into WX-1 containers in pl/.
    
    Creates:
        pl/pl_1000hPa.wx1 or pl/{file_prefix}_1000hPa.wx1 : All variables (z, t, u, v, q, w) for 1000 hPa
        pl/pl_850hPa.wx1  or pl/{file_prefix}_850hPa.wx1  : All variables for 850 hPa
        pl/pl_500hPa.wx1  or pl/{file_prefix}_500hPa.wx1  : All variables for 500 hPa
        ...
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    var_aliases = {
        "geopotential_height": "z",
        "temperature": "t",
        "u_wind": "u",
        "v_wind": "v",
        "specific_humidity": "q",
        "relative_humidity": "r",
        "wind_speed": "ws",
        "vertical_velocity": "w",
    }
    has_write_dataset = hasattr(wx1, "write_dataset")

    for ilev, lev_hpa in enumerate(isobaric_levels_hpa):
        lev_int = int(round(float(lev_hpa)))
        canonical_name = f"pl_{lev_int}hPa.wx1"
        ecmwf_name = f"{file_prefix}_{lev_int}hPa.wx1" if file_prefix else canonical_name
        filepath = out_dir / ecmwf_name
        
        lev_vars = {}
        for key, short_name in var_aliases.items():
            if key in pl_data:
                lev_vars[short_name] = pl_data[key][ilev].astype(np.float32)

        if not lev_vars:
            continue

        ny, nx = next(iter(lev_vars.values())).shape
        targets = [filepath]
        if file_prefix and (out_dir / canonical_name) != filepath:
            targets.append(out_dir / canonical_name)

        for tgt in targets:
            if has_write_dataset:
                wx1.write_dataset(
                    tgt,
                    lev_vars,
                    dataset_id=f"wmo_pl_{lev_int}hpa",
                    dimensions=["latitude", "longitude"],
                    chunk_shapes={name: [ny, nx] for name in lev_vars},
                    attributes={
                        "isobaric_level_hpa": lev_int,
                        "wmo_conventions": "WMO-No. 306 Manual on Codes",
                        "init_time_zulu": run_timestamp or "UNKNOWN",
                    },
                    codec="zstd",
                    workers=1,
                    durability="buffered",
                )
            else:
                for vname, varr in lev_vars.items():
                    wx1.write_array(
                        tgt,
                        varr,
                        dataset_id=f"wmo_pl_{lev_int}hpa_{vname}",
                        variable_path=vname,
                        dimensions=["latitude", "longitude"],
                        profile="output",
                        codec="zstd_shuffle",
                    )
        logger.info("Saved WMO isobaric per-level container -> %s (%d variables)", filepath, len(lev_vars))


def interpolate_wind_to_heights(
    u: np.ndarray,
    v: np.ndarray,
    ps: np.ndarray,
    theta: np.ndarray,
    qv: np.ndarray,
    zs: np.ndarray,
    target_heights: Sequence[float] = (10.0, 50.0, 100.0, 200.0),
    z0_m: float = 0.05,
) -> Dict[str, np.ndarray]:
    """Interpolates 3D atmospheric wind field (u, v) to standardized heights above ground level (AGL).
    
    Standard meteorological and wind turbine hub heights:
      - 10m  : WMO standard anemometer height (u10m, v10m, wind10m)
      - 50m  : Small/medium turbine & boundary layer low-level jet (u50m, v50m, wind50m)
      - 100m : Modern onshore wind turbine hub height (u100m, v100m, wind100m)
      - 200m : Modern offshore wind turbine hub / upper boundary layer (u200m, v200m, wind200m)
    
    Uses hydrostatic height calculation z_agl(k) = z3d(k) - zs, with logarithmic surface layer profile
    for heights below the first full model layer, and linear-in-height interpolation between model layers.
    """
    nz, ny, nx = u.shape
    p3d = compute_3d_pressure(ps, nz)
    z3d, _ = compute_hydrostatic_geopotential(p3d, theta, qv, zs)
    z_agl = np.maximum(z3d - zs[np.newaxis, :, :], 0.1)

    out_winds: Dict[str, np.ndarray] = {}
    for h in target_heights:
        h_val = float(h)
        h_int = int(round(h_val))
        
        u_h = np.empty((ny, nx), dtype=np.float32)
        v_h = np.empty((ny, nx), dtype=np.float32)

        z0_layer = z_agl[0]
        below_mask = h_val < z0_layer

        if np.any(below_mask):
            log_scale = np.log(max(h_val, 0.1) / z0_m) / np.maximum(np.log(z0_layer[below_mask] / z0_m), 0.1)
            u_h[below_mask] = (u[0][below_mask] * log_scale).astype(np.float32)
            v_h[below_mask] = (v[0][below_mask] * log_scale).astype(np.float32)

        above_mask = ~below_mask
        if np.any(above_mask):
            k_bracket = np.zeros((ny, nx), dtype=np.int32)
            for k in range(nz - 1):
                cond = (z_agl[k] <= h_val) & (z_agl[k + 1] > h_val)
                k_bracket[cond & above_mask] = k
            k_bracket[z_agl[nz - 1] <= h_val] = nz - 2

            y_i, x_i = np.indices((ny, nx))
            k0 = k_bracket[above_mask]
            y0 = y_i[above_mask]
            x0 = x_i[above_mask]
            k1 = k0 + 1

            z0_pts = z_agl[k0, y0, x0]
            z1_pts = z_agl[k1, y0, x0]
            dz = np.maximum(z1_pts - z0_pts, 1e-3)
            w1 = np.clip((h_val - z0_pts) / dz, 0.0, 1.0)
            w0 = 1.0 - w1

            u_h[above_mask] = (w0 * u[k0, y0, x0] + w1 * u[k1, y0, x0]).astype(np.float32)
            v_h[above_mask] = (w0 * v[k0, y0, x0] + w1 * v[k1, y0, x0]).astype(np.float32)

        ws_h = np.sqrt(u_h**2 + v_h**2).astype(np.float32)
        out_winds[f"u{h_int}m"] = u_h
        out_winds[f"v{h_int}m"] = v_h
        out_winds[f"wind{h_int}m"] = ws_h

    return out_winds

