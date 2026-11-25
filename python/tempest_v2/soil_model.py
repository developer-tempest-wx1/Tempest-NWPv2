"""
Noah 4-Layer Land Surface Model (LSM) with FAO HWSD Soil Texture Physics.

Couples atmospheric boundary layer fluxes to a 4-layer soil column:
- Layers: 10 cm, 30 cm, 60 cm, 100 cm (total 2.0 m depth).
- Texture parameters: USDA 13 classes (Clapp & Hornberger 1978, Cosby et al. 1984).
- Soil Moisture: 1D vertical Richard's equation with hydraulic conductivity and matric potential.
- Soil Temperature: 1D thermal diffusion with Johansen thermal conductivity and heat capacity.
- Soil Freezing/Thawing: Latent heat of fusion (3.34e5 J/kg) buffering frost penetration.
- Ground Heat Flux G: Energy balance closure R_net = H + LE + G.

Strictly 100% Deterministic Atmospheric & Soil Physics (0% AI / 0% ML).
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger("tempest.soil_model")

NUM_SOIL_LAYERS = 4
SOIL_LAYER_DZ = np.array([0.10, 0.30, 0.60, 1.00], dtype=np.float32)  # [m]
SOIL_LAYER_Z = np.array([0.05, 0.25, 0.70, 1.50], dtype=np.float32)   # [m] Layer centers

# Physical constants
RHO_WATER = 1000.0       # Density of liquid water [kg/m3]
RHO_ICE = 917.0          # Density of ice [kg/m3]
CP_WATER = 4184.0        # Specific heat of water [J/(kg K)]
CP_ICE = 2090.0          # Specific heat of ice [J/(kg K)]
LATENT_HEAT_FUSION = 3.34e5 # Latent heat of freezing/melting [J/kg]
T_FREEZE = 273.15        # Freezing point of water [K]

# USDA 13-Class Mineral + Organic Peat (Class 14) Physical Parameters
# (Noah LSM / Cosby et al. 1984 / Clapp & Hornberger 1978; Class 14: Balsamo et al. 2009 HTESSEL + Letts et al. 2000)
# Indices:
# 0: Water/Ocean (neutral dummy parameters)
# 1: Clay (heavy), 2: Silty clay, 3: Clay (light), 4: Silty clay loam, 5: Clay loam,
# 6: Silt, 7: Silt loam, 8: Sandy clay, 9: Loam, 10: Sandy clay loam, 11: Sandy loam,
# 12: Loamy sand, 13: Sand, 14: Organic / Peat
#
# Columns: [theta_sat, theta_fc, theta_wilt, b_param, psi_sat [m], ksat [m/s], kappa_dry [W/mK], kappa_sat [W/mK]]
SOIL_PARAM_TABLE = np.array([
    # theta_sat, theta_fc, theta_wilt, b,    psi_s,  ksat,      k_dry, k_sat
    [0.000,      0.000,    0.000,      0.00, 0.000,  0.00e-6,   0.00,  0.60],  # 0: Water/Ocean
    [0.482,      0.395,    0.286,     11.40, 0.405,  1.28e-6,   0.20,  1.25],  # 1: Clay (heavy)
    [0.470,      0.380,    0.270,     10.40, 0.350,  1.50e-6,   0.21,  1.30],  # 2: Silty clay
    [0.476,      0.367,    0.257,      8.52, 0.356,  2.17e-6,   0.22,  1.35],  # 3: Clay (light)
    [0.477,      0.350,    0.225,      7.75, 0.320,  2.80e-6,   0.23,  1.40],  # 4: Silty clay loam
    [0.465,      0.325,    0.198,      6.77, 0.299,  3.50e-6,   0.24,  1.45],  # 5: Clay loam
    [0.460,      0.315,    0.180,      5.30, 0.280,  4.20e-6,   0.24,  1.50],  # 6: Silt
    [0.485,      0.300,    0.150,      5.30, 0.260,  5.23e-6,   0.25,  1.55],  # 7: Silt loam
    [0.430,      0.310,    0.210,      6.00, 0.200,  4.80e-6,   0.25,  1.60],  # 8: Sandy clay
    [0.451,      0.283,    0.119,      5.39, 0.478,  3.38e-6,   0.25,  1.58],  # 9: Loam
    [0.420,      0.255,    0.135,      4.90, 0.290,  5.20e-6,   0.26,  1.70],  # 10: Sandy clay loam
    [0.435,      0.215,    0.084,      4.74, 0.141,  1.41e-5,   0.27,  1.80],  # 11: Sandy loam
    [0.410,      0.174,    0.068,      4.26, 0.090,  4.79e-5,   0.28,  2.00],  # 12: Loamy sand
    [0.395,      0.135,    0.045,      4.05, 0.121,  1.76e-4,   0.30,  2.20],  # 13: Sand
    [0.766,      0.663,    0.267,      6.10, 0.103,  2.00e-6,   0.05,  0.55],  # 14: Organic / Peat
], dtype=np.float32)

# ECMWF IFS / ERA5 HTESSEL 7-class soil hydrology [theta_wilt, theta_fc, theta_sat]
# (sussoil_mod.F90 & susdp_deriv_ctl_mod.F90; class 0 = water, 7 = tropical organic fallback)
ECMWF_HTESSEL_SOIL_HYDRO = np.array([
    [0.000, 0.000, 0.000],  # 0: Water
    [0.059, 0.244, 0.403],  # 1: Coarse
    [0.151, 0.347, 0.439],  # 2: Medium (ECMWF default reference)
    [0.133, 0.383, 0.430],  # 3: Medium fine
    [0.279, 0.448, 0.520],  # 4: Fine
    [0.335, 0.541, 0.614],  # 5: Very fine
    [0.267, 0.663, 0.766],  # 6: Organic / Peat
    [0.151, 0.347, 0.439],  # 7: Tropical organic / medium fallback
], dtype=np.float64)


class NoahLandSurfaceState:
    """State of the 4-layer soil column across a 2D horizontal domain (ny, nx)."""

    def __init__(self, ny: int, nx: int, soil_type: Optional[np.ndarray] = None, lsm: Optional[np.ndarray] = None):
        self.ny = ny
        self.nx = nx

        # Soil classification: default to Loam (9) over land, 0 over sea
        if soil_type is not None:
            self.soil_type = np.array(soil_type, dtype=np.uint8, copy=True)
        else:
            self.soil_type = np.full((ny, nx), 9, dtype=np.uint8)
            if lsm is not None:
                self.soil_type[lsm < 0.5] = 0

        self.lsm = np.array(lsm, dtype=np.float32, copy=True) if lsm is not None else np.where(self.soil_type > 0, 1.0, 0.0).astype(np.float32)

        # Ensure any land cell (lsm >= 0.5) has a valid physical soil texture class (1..14).
        # Water/unclassified cells (0) on land are assigned regional default Loam (class 9).
        land_cells = self.lsm >= 0.5
        st_clean = np.where(land_cells & (self.soil_type <= 0), 9, self.soil_type)
        self.soil_type[:] = st_clean

        # 4-layer soil temperatures [K]
        self.soil_temp = np.zeros((NUM_SOIL_LAYERS, ny, nx), dtype=np.float32)
        # 4-layer volumetric soil moisture [m3/m3]
        self.soil_moisture = np.zeros((NUM_SOIL_LAYERS, ny, nx), dtype=np.float32)
        # 4-layer volumetric ice fraction [m3/m3]
        self.soil_ice = np.zeros((NUM_SOIL_LAYERS, ny, nx), dtype=np.float32)

        # Diagnostics
        self.skin_temp = np.full((ny, nx), 288.15, dtype=np.float32)
        self.ground_heat_flux = np.zeros((ny, nx), dtype=np.float32)
        self.surface_runoff = np.zeros((ny, nx), dtype=np.float32)

        # Broadcast texture parameters to (ny, nx)
        st_clamped = np.clip(self.soil_type, 0, 14)
        self.theta_sat = SOIL_PARAM_TABLE[st_clamped, 0]
        self.theta_fc = SOIL_PARAM_TABLE[st_clamped, 1]
        self.theta_wilt = SOIL_PARAM_TABLE[st_clamped, 2]
        self.b_param = SOIL_PARAM_TABLE[st_clamped, 3]
        self.psi_sat = SOIL_PARAM_TABLE[st_clamped, 4]
        self.ksat = SOIL_PARAM_TABLE[st_clamped, 5]
        self.kappa_dry = SOIL_PARAM_TABLE[st_clamped, 6]
        self.kappa_sat = SOIL_PARAM_TABLE[st_clamped, 7]

        # Initialize realistic default soil moisture (near field capacity)
        for l in range(NUM_SOIL_LAYERS):
            self.soil_temp[l] = 288.15 - l * 0.5
            self.soil_moisture[l] = np.clip(self.theta_fc * 0.95, 0.05, self.theta_sat * 0.95)



# ECMWF IFS / ERA5 soil layers (cm): 0-7, 7-28, 28-100, 100-289. Tempest/Noah layers: 0-10, 10-40, 40-100, 100-200.
ECMWF_SOIL_BOUNDS_M = np.array([0.0, 0.07, 0.28, 1.00, 2.89], dtype=np.float64)
TEMPEST_SOIL_BOUNDS_M = np.concatenate([[0.0], np.cumsum(SOIL_LAYER_DZ.astype(np.float64))])


def remap_ecmwf_soil_temperature(sot: np.ndarray) -> np.ndarray:
    """Soil temperature [K]: linear interpolation in depth between ECMWF layer MIDPOINTS, evaluated at
    the Tempest layer midpoints (constant extrapolation outside the ECMWF midpoint range)."""
    src_z = 0.5 * (ECMWF_SOIL_BOUNDS_M[:-1] + ECMWF_SOIL_BOUNDS_M[1:])
    dst_z = 0.5 * (TEMPEST_SOIL_BOUNDS_M[:-1] + TEMPEST_SOIL_BOUNDS_M[1:])
    out = np.empty((len(dst_z),) + sot.shape[1:], dtype=np.float32)
    for l, z in enumerate(dst_z):
        j = int(np.clip(np.searchsorted(src_z, z), 1, len(src_z) - 1))
        w = float(np.clip((z - src_z[j - 1]) / (src_z[j] - src_z[j - 1]), 0.0, 1.0))
        out[l] = (1.0 - w) * sot[j - 1] + w * sot[j]
    return out


def remap_ecmwf_soil_moisture(vsw: np.ndarray) -> np.ndarray:
    """Volumetric soil water [m3/m3]: thickness-weighted overlap average (conserves the water column)."""
    out = np.zeros((len(SOIL_LAYER_DZ),) + vsw.shape[1:], dtype=np.float32)
    for l in range(len(SOIL_LAYER_DZ)):
        a, b = TEMPEST_SOIL_BOUNDS_M[l], TEMPEST_SOIL_BOUNDS_M[l + 1]
        for k in range(4):
            ov = max(0.0, min(b, ECMWF_SOIL_BOUNDS_M[k + 1]) - max(a, ECMWF_SOIL_BOUNDS_M[k]))
            if ov > 0.0:
                out[l] += (ov / (b - a)) * vsw[k]
    return out


def remap_ecmwf_swvl_via_smi(
    swvl: np.ndarray,
    target_soil_type: np.ndarray,
    src_soil_type: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Remap ECMWF HTESSEL volumetric soil water [m3/m3] to Tempest soil classes via Soil Moisture Index (SMI).

    Uses continuous 3-segment piecewise-linear scaling across `(0..theta_wilt, theta_wilt..theta_fc, theta_fc..theta_sat)`
    following Koster et al. (2009) so plant-available moisture fraction is preserved across mineral & peat transitions.
    """
    arr = np.asarray(swvl, dtype=np.float64)
    is_2d = arr.ndim == 2
    w_in = arr[None, :, :] if is_2d else arr
    if w_in.ndim != 3:
        raise ValueError(f"Expected 2D or 3D swvl array, got shape {arr.shape}")

    st_dst = np.clip(np.rint(np.asarray(target_soil_type, dtype=np.float64)).astype(np.int32), 0, 14)
    if w_in.shape[1:] != st_dst.shape:
        raise ValueError(f"swvl spatial shape {w_in.shape[1:]} does not match target_soil_type {st_dst.shape}")

    if src_soil_type is not None:
        st_src_raw = np.rint(np.asarray(src_soil_type, dtype=np.float64)).astype(np.int32)
        st_src = np.where((st_src_raw >= 1) & (st_src_raw <= 7), st_src_raw, 2)
    else:
        with np.errstate(all="ignore"):
            col_max = np.nanmax(w_in, axis=0)
        # All HTESSEL mineral soils 1..4 have theta_sat <= 0.520; > 0.52 unambiguously indicates HTESSEL organic class 6.
        st_src = np.where(np.isfinite(col_max) & (col_max > 0.52), 6, 2)

    wp_s = ECMWF_HTESSEL_SOIL_HYDRO[st_src, 0]
    fc_s = ECMWF_HTESSEL_SOIL_HYDRO[st_src, 1]
    sat_s = ECMWF_HTESSEL_SOIL_HYDRO[st_src, 2]

    sat_d = SOIL_PARAM_TABLE[st_dst, 0].astype(np.float64)
    fc_d = SOIL_PARAM_TABLE[st_dst, 1].astype(np.float64)
    wp_d = SOIL_PARAM_TABLE[st_dst, 2].astype(np.float64)
    land = (st_dst >= 1) & (sat_d > 0.05)

    out = np.empty_like(w_in, dtype=np.float32)
    inv_fc_range_s = 1.0 / np.maximum(fc_s - wp_s, 1e-6)
    inv_sat_range_s = 1.0 / np.maximum(sat_s - fc_s, 1e-6)
    inv_wp_s = 1.0 / np.maximum(wp_s, 1e-6)

    for l in range(w_in.shape[0]):
        w = np.where(np.isfinite(w_in[l]), w_in[l], 0.28)
        w_below = np.clip(w * inv_wp_s, 0.0, 1.0) * wp_d
        smi = (w - wp_s) * inv_fc_range_s
        w_mid = wp_d + smi * (fc_d - wp_d)
        f_wet = np.clip((w - fc_s) * inv_sat_range_s, 0.0, 1.0)
        w_abv = fc_d + f_wet * (sat_d - fc_d)
        w_dst = np.where(w < wp_s, w_below, np.where(w <= fc_s, w_mid, w_abv))
        out[l] = np.where(land, np.clip(w_dst, 0.02, sat_d), np.clip(w, 0.02, 0.50)).astype(np.float32)

    return out[0] if is_2d else out


def initialize_soil_from_ecmwf(
    state: NoahLandSurfaceState,
    sot_layers: Optional[np.ndarray] = None,
    vsw_layers: Optional[np.ndarray] = None,
    skt: Optional[np.ndarray] = None,
) -> None:
    """Initializes Noah soil state from real ECMWF 4-layer soil analysis."""
    if sot_layers is not None and sot_layers.shape == (NUM_SOIL_LAYERS, state.ny, state.nx):
        # ECMWF layers (0-7/7-28/28-100/100-289 cm) != Tempest layers (0-10/10-40/40-100/100-200 cm): remap.
        state.soil_temp[:] = remap_ecmwf_soil_temperature(sot_layers.astype(np.float32))
        logger.info("Initialized Noah soil temperatures from ECMWF SOT (depth-remapped to Tempest layers).")
    if vsw_layers is not None and vsw_layers.shape == (NUM_SOIL_LAYERS, state.ny, state.nx):
        state.soil_moisture[:] = np.clip(remap_ecmwf_soil_moisture(vsw_layers.astype(np.float32)), 0.02, state.theta_sat)
        logger.info("Initialized Noah soil moisture from ECMWF VSW (thickness-weighted remap to Tempest layers).")
    if skt is not None and skt.shape == (state.ny, state.nx):
        state.skin_temp[:] = skt.astype(np.float32)


def step_noah_lsm(
    state: NoahLandSurfaceState,
    r_net: np.ndarray,
    h_flux: np.ndarray,
    le_flux: np.ndarray,
    rain_rate: np.ndarray,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Steps the Noah 4-layer LSM forward by dt seconds.
    
    Arguments:
        state: NoahLandSurfaceState instance.
        r_net: Downward net radiation [W/m2].
        h_flux: Upward sensible heat flux [W/m2].
        le_flux: Upward latent heat flux [W/m2].
        rain_rate: Liquid precipitation reaching the surface [kg/(m2 s)].
        dt: Integration timestep [s].
        
    Returns:
        (skin_temp, ground_heat_flux): Updated surface skin temperature [K] and ground heat flux G [W/m2].
    """
    land_mask = (state.lsm >= 0.5)
    if not np.any(land_mask):
        return state.skin_temp, state.ground_heat_flux

    dz = SOIL_LAYER_DZ
    c_mineral = 2.0e6  # Volumetric heat capacity of dry soil minerals [J/(m3 K)]

    # -------------------------------------------------------------
    # 1. Soil Thermal Conductivity & Heat Capacity (Johansen 1975)
    # -------------------------------------------------------------
    # Volumetric heat capacity of each layer: C = (1 - theta_sat)*C_mineral + theta*C_water + theta_ice*C_ice
    cap = np.zeros_like(state.soil_temp)
    k_soil = np.zeros_like(state.soil_temp)

    for l in range(NUM_SOIL_LAYERS):
        th = state.soil_moisture[l]
        th_s = np.maximum(state.theta_sat, 0.01)
        sr = np.clip(th / th_s, 0.01, 1.0) # Saturation degree

        # Kersten number Ke
        ke = np.clip(np.log10(sr) + 1.0, 0.0, 1.0)
        k_soil[l] = state.kappa_dry + ke * (state.kappa_sat - state.kappa_dry)

        cap[l] = (1.0 - th_s) * c_mineral + th * (RHO_WATER * CP_WATER) + state.soil_ice[l] * (RHO_ICE * CP_ICE)

    # -------------------------------------------------------------
    # 2. Surface Energy Balance Closure: G = R_net - H - LE
    # -------------------------------------------------------------
    seb_residual = r_net - h_flux - le_flux

    # Top boundary ground heat flux: G = kappa_1 * (T_skin - T_soil,1) / (0.5 * dz0)
    kappa_0 = k_soil[0]
    dz0 = dz[0]
    g_conductance = kappa_0 / (0.5 * dz0)

    # Solve for balanced ground heat flux G
    g_flux = np.where(land_mask, seb_residual, 0.0).astype(np.float32)
    state.ground_heat_flux[:] = g_flux

    # Update skin temperature: delta_T = G / conductance
    state.skin_temp[land_mask] = state.soil_temp[0][land_mask] + np.clip(g_flux[land_mask] / np.maximum(g_conductance[land_mask], 5.0), -15.0, 20.0)

    # -------------------------------------------------------------
    # 3. 1D Thermal Diffusion across the 4 Soil Layers
    # -------------------------------------------------------------
    # Layer 0 (Topsoil, 0-10 cm)
    flux_01 = 0.5 * (k_soil[0] + k_soil[1]) * (state.soil_temp[0] - state.soil_temp[1]) / (0.5 * (dz[0] + dz[1]))
    dt_soil0 = (dt / (cap[0] * dz[0])) * (g_flux - flux_01)
    state.soil_temp[0][land_mask] += dt_soil0[land_mask]

    # Layer 1 (10-40 cm)
    flux_12 = 0.5 * (k_soil[1] + k_soil[2]) * (state.soil_temp[1] - state.soil_temp[2]) / (0.5 * (dz[1] + dz[2]))
    dt_soil1 = (dt / (cap[1] * dz[1])) * (flux_01 - flux_12)
    state.soil_temp[1][land_mask] += dt_soil1[land_mask]

    # Layer 2 (40-100 cm)
    flux_23 = 0.5 * (k_soil[2] + k_soil[3]) * (state.soil_temp[2] - state.soil_temp[3]) / (0.5 * (dz[2] + dz[3]))
    dt_soil2 = (dt / (cap[2] * dz[2])) * (flux_12 - flux_23)
    state.soil_temp[2][land_mask] += dt_soil2[land_mask]

    # Layer 3 (100-200 cm, deep thermal boundary with geothermal damping)
    dt_soil3 = (dt / (cap[3] * dz[3])) * (flux_23)
    state.soil_temp[3][land_mask] += dt_soil3[land_mask]

    # -------------------------------------------------------------
    # 4. Phase Change: Freezing & Melting Latent Heat Buffer
    # -------------------------------------------------------------
    for l in range(NUM_SOIL_LAYERS):
        freezing = land_mask & (state.soil_temp[l] < T_FREEZE) & (state.soil_moisture[l] > state.theta_wilt)
        if np.any(freezing):
            # Energy needed to freeze water: dQ = cap * (T_FREEZE - T_soil)
            excess_cold = (T_FREEZE - state.soil_temp[l][freezing])
            ice_to_form = np.minimum(state.soil_moisture[l][freezing] - state.theta_wilt[freezing],
                                     (cap[l][freezing] * excess_cold) / (RHO_WATER * LATENT_HEAT_FUSION))
            state.soil_ice[l][freezing] += ice_to_form
            state.soil_moisture[l][freezing] -= ice_to_form
            # Temperature buffered back towards freezing point
            state.soil_temp[l][freezing] = np.maximum(state.soil_temp[l][freezing], T_FREEZE - 0.5)

        melting = land_mask & (state.soil_temp[l] > T_FREEZE) & (state.soil_ice[l] > 0.0)
        if np.any(melting):
            excess_warm = (state.soil_temp[l][melting] - T_FREEZE)
            ice_to_melt = np.minimum(state.soil_ice[l][melting],
                                     (cap[l][melting] * excess_warm) / (RHO_WATER * LATENT_HEAT_FUSION))
            state.soil_ice[l][melting] -= ice_to_melt
            state.soil_moisture[l][melting] += ice_to_melt
            state.soil_temp[l][melting] = np.minimum(state.soil_temp[l][melting], T_FREEZE + 0.5)

    # -------------------------------------------------------------
    # 5. Soil Moisture Dynamics (Richard's Equation & Infiltration)
    # -------------------------------------------------------------
    # Surface infiltration rate: P_infil = min(rain_rate, ksat)
    p_infil = np.minimum(rain_rate, state.ksat * RHO_WATER)
    state.surface_runoff[land_mask] = np.maximum(0.0, rain_rate[land_mask] - p_infil[land_mask])

    # Infiltration added to Layer 0
    d_theta_infil = (p_infil * dt) / (RHO_WATER * dz[0])
    state.soil_moisture[0][land_mask] += d_theta_infil[land_mask]

    # Gravitational drainage and capillary flux between layers
    # Flux w_flux = K(theta) * [1 + d(psi)/dz]
    for l in range(NUM_SOIL_LAYERS - 1):
        th_l = state.soil_moisture[l]
        th_next = state.soil_moisture[l + 1]
        sat_l = np.maximum(state.theta_sat, 0.01)

        # Average hydraulic conductivity K = Ksat * (theta / theta_sat)^(2b + 3)
        sr_avg = np.clip(0.5 * (th_l + th_next) / sat_l, 0.05, 1.0)
        b_exp = 2.0 * state.b_param + 3.0
        k_flow = state.ksat * (sr_avg ** b_exp)

        # Moisture gradient
        d_th_dz = (th_l - th_next) / (0.5 * (dz[l] + dz[l + 1]))
        water_flux = k_flow * (1.0 + d_th_dz * 0.5) # [m/s]

        # Transfer moisture
        d_th_dt_l = (water_flux * dt) / dz[l]
        d_th_dt_next = (water_flux * dt) / dz[l + 1]

        state.soil_moisture[l][land_mask] -= d_th_dt_l[land_mask]
        state.soil_moisture[l + 1][land_mask] += d_th_dt_next[land_mask]

    # Deep gravitational percolation out of layer 3
    deep_drain = state.ksat * ((np.clip(state.soil_moisture[3] / np.maximum(state.theta_sat, 0.01), 0.05, 1.0)) ** (2.0 * state.b_param + 3.0))
    state.soil_moisture[3][land_mask] -= (deep_drain[land_mask] * dt) / dz[3]

    # Clamp volumetric soil moisture between residual and porosity
    for l in range(NUM_SOIL_LAYERS):
        state.soil_moisture[l] = np.clip(state.soil_moisture[l], 0.02, state.theta_sat)

    return state.skin_temp, state.ground_heat_flux
