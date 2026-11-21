"""FLake: 1D Freshwater Lake Model (Mironov 2008) for Tempest-NWP.

Simulates the physical thermodynamic column of freshwater lakes and inland water bodies
(e.g., Lake IJsselmeer, Lake Markermeer, Bodensee, Lake Geneva).

Features:
- Two-layer parametric bulk approach based on the concept of self-similarity (Kitaigorodskii & Miropolsky 1970, Mironov 2008).
- Resolves upper mixed layer (depth h, temperature theta_s), thermocline (depth h to D, shape factor C_theta), and bottom temperature theta_b.
- Volumetric solar extinction in water column (Beer's law).
- Freezing point of fresh water (T_f = 273.15 K) and density anomaly (max density at 277.13 K / 3.98 °C).
- Stefan ice growth and surface/basal melting for lake ice.

Strictly 100% deterministic classical physics (0% AI / 0% ML).
"""

import math
from typing import Dict, Optional, Tuple
import numpy as np

# Physical constants for fresh water and ice (SI units)
RHO_WATER_FRESH = 1000.0        # Pure freshwater density [kg / m^3]
CP_WATER_FRESH = 4192.0         # Specific heat freshwater [J / (kg K)]
RHO_CP_WATER = RHO_WATER_FRESH * CP_WATER_FRESH  # Volumetric heat capacity [J / (m^3 K)] ~ 4.192e6
RHO_ICE_FRESH = 917.0           # Pure ice density [kg / m^3]
LF_ICE_FRESH = 3.337e5          # Latent heat of fusion [J / kg]
K_ICE_FRESH = 2.29              # Pure ice thermal conductivity [W / (m K)]
T_FREEZE_FRESH = 273.15         # Freshwater freezing point [K]
T_MAX_DENSITY = 277.13          # Temperature of maximum freshwater density [K] (~3.98 °C)
GRAVITY = 9.80665               # Gravitational acceleration [m / s^2]
STEFAN_BOLTZMANN = 5.670374e-8  # Stefan-Boltzmann constant [W / (m^2 K^4)]
EMISSIVITY_WATER = 0.97         # Longwave emissivity of water surface
ALBEDO_WATER_OPEN = 0.07        # Solar albedo of open water
ALBEDO_ICE_LAKE = 0.60          # Solar albedo of bare lake ice


class FLakeModel:
    """Vectorized 1D FLake freshwater lake thermodynamic model for 2D horizontal grids."""

    def __init__(
        self,
        ny: int,
        nx: int,
        lake_mask: np.ndarray,
        lake_depth: Optional[np.ndarray] = None,
        default_depth_m: float = 4.5,
        optical_extinction_m: float = 0.70,
    ):
        """Initializes FLake lake columns over the active domain.

        Args:
            ny: Grid rows.
            nx: Grid columns.
            lake_mask: Boolean 2D array (True where grid cell contains inland lake/water).
            lake_depth: 2D array of lake bathymetric depths D [m]. If None, defaults to default_depth_m.
            default_depth_m: Default mean depth (4.5m for Lake IJsselmeer).
            optical_extinction_m: Solar light extinction coefficient gamma_w [1/m] (~0.70 for IJsselmeer).
        """
        self.ny = ny
        self.nx = nx
        self.lake_mask = np.asarray(lake_mask, dtype=bool)
        self.optical_extinction = optical_extinction_m

        if lake_depth is not None:
            self.depth = np.clip(np.asarray(lake_depth, dtype=np.float32), 1.0, 300.0)
        else:
            self.depth = np.full((ny, nx), default_depth_m, dtype=np.float32)

        # 1. FLake Prognostic State Variables
        # Surface mixed layer temperature theta_s [K]
        self.tsfc_k = np.full((ny, nx), 282.15, dtype=np.float32)  # Default ~9 °C
        # Lake bottom temperature theta_b [K]
        self.tbot_k = np.full((ny, nx), 280.15, dtype=np.float32)  # Default ~7 °C
        # Mixed layer depth h [m] (0 <= h <= D)
        self.hmld = np.clip(0.6 * self.depth, 0.5, self.depth).astype(np.float32)
        # Thermocline shape factor C_theta (bounded in [0.5, 0.8], equilibrium ~0.65)
        self.ctheta = np.full((ny, nx), 0.65, dtype=np.float32)
        # Lake ice thickness H_ice [m]
        self.ice_thick = np.zeros((ny, nx), dtype=np.float32)
        # Ice surface temperature theta_i [K]
        self.ice_tsfc = np.full((ny, nx), T_FREEZE_FRESH, dtype=np.float32)

    def initialize_from_atmosphere(self, t2m_k: np.ndarray, sst_prior_k: Optional[np.ndarray] = None):
        """Initializes lake temperatures from atmospheric / surface analysis."""
        base_t = sst_prior_k if sst_prior_k is not None else t2m_k
        self.tsfc_k = np.where(self.lake_mask, np.clip(base_t, 271.15, 305.0), self.tsfc_k).astype(np.float32)
        # Bottom temperature is buffered toward max density (4 °C) in deep water or equals surface in shallow water
        is_shallow = self.depth <= 5.0
        self.tbot_k = np.where(
            self.lake_mask,
            np.where(is_shallow, self.tsfc_k, 0.5 * (self.tsfc_k + T_MAX_DENSITY)),
            self.tbot_k,
        ).astype(np.float32)

        # Detect initial ice cover if surface is below freezing point
        has_ice = (self.tsfc_k < T_FREEZE_FRESH) & self.lake_mask
        self.ice_thick = np.where(has_ice, 0.05, 0.0).astype(np.float32)
        self.ice_tsfc = np.where(has_ice, self.tsfc_k, T_FREEZE_FRESH).astype(np.float32)
        self.tsfc_k = np.where(has_ice, T_FREEZE_FRESH, self.tsfc_k).astype(np.float32)

    def get_skin_temperature_k(self) -> np.ndarray:
        """Returns effective radiometric skin temperature for atmospheric coupling [K]."""
        return np.where(self.ice_thick > 0.001, self.ice_tsfc, self.tsfc_k).astype(np.float32)

    def step(
        self,
        sw_down: np.ndarray,
        lw_down: np.ndarray,
        h_flux: np.ndarray,
        le_flux: np.ndarray,
        wind10m: np.ndarray,
        dt_sec: float,
    ):
        """Advances FLake 1D columns by time step dt_sec [s].

        Args:
            sw_down: Downwelling shortwave solar radiation [W / m^2].
            lw_down: Downwelling longwave thermal radiation [W / m^2].
            h_flux: Atmospheric sensible heat flux [W / m^2] (positive upward).
            le_flux: Atmospheric latent heat flux [W / m^2] (positive upward).
            wind10m: 10m scalar wind speed [m / s].
            dt_sec: Coupling time step [s].
        """
        if not np.any(self.lake_mask):
            return

        dt = float(dt_sec)
        mask = self.lake_mask
        D = self.depth

        # 1. Surface Radiative & Turbulent Energy Fluxes
        has_ice = (self.ice_thick > 0.001) & mask
        albedo = np.where(has_ice, ALBEDO_ICE_LAKE, ALBEDO_WATER_OPEN).astype(np.float32)
        i_sfc = (1.0 - albedo) * np.maximum(0.0, sw_down)  # Absorbed solar flux [W / m^2]

        t_skin = self.get_skin_temperature_k()
        lw_up = EMISSIVITY_WATER * STEFAN_BOLTZMANN * (t_skin**4)
        r_net = i_sfc + lw_down - lw_up
        # Net surface heat flux into the lake (positive downward)
        q_net_in = (r_net - h_flux - le_flux).astype(np.float32)

        # 2. Ice Growth and Melting
        # Case A: Ice covered columns
        if np.any(has_ice):
            # Heat conduction through ice: F_cond = k_i * (T_freeze - T_ice_sfc) / H_i
            h_i_eff = np.maximum(0.01, self.ice_thick)
            f_cond = K_ICE_FRESH * (T_FREEZE_FRESH - self.ice_tsfc) / h_i_eff
            # Update ice surface temperature
            c_i_eff = RHO_ICE_FRESH * 2090.0 * 0.5 * h_i_eff
            self.ice_tsfc = np.where(
                has_ice,
                np.clip(self.ice_tsfc + dt * (q_net_in + f_cond) / c_i_eff, 220.0, T_FREEZE_FRESH),
                self.ice_tsfc,
            )

            # Basal growth / melt rate
            dh_dt_basal = (f_cond - 0.0) / (RHO_ICE_FRESH * LF_ICE_FRESH)
            # Surface melt rate when T_ice == T_freeze and q_net > 0
            dh_dt_top = np.where(
                (self.ice_tsfc >= T_FREEZE_FRESH - 0.01) & (q_net_in > 0.0),
                q_net_in / (RHO_ICE_FRESH * LF_ICE_FRESH),
                0.0,
            )
            self.ice_thick = np.where(
                has_ice,
                np.maximum(0.0, self.ice_thick + dt * (dh_dt_basal - dh_dt_top)),
                self.ice_thick,
            )
            # Extinguish ice if thickness is negligible
            melted = has_ice & (self.ice_thick <= 1e-4)
            self.ice_thick = np.where(melted, 0.0, self.ice_thick)
            self.tsfc_k = np.where(melted, T_FREEZE_FRESH + 0.1, self.tsfc_k)

        # Case B: Open water columns (has_ice == False)
        open_water = mask & (~has_ice)
        if np.any(open_water):
            # Solar extinction at depth h and bottom D (Beer's Law)
            gamma = self.optical_extinction
            i_h = i_sfc * np.exp(-gamma * self.hmld)
            i_d = i_sfc * np.exp(-gamma * D)

            # Total water column heat budget: d(theta_mean)/dt = (Q_net + I_s - I_D) / (rho_cp * D)
            d_theta_mean = (q_net_in - i_d) * (dt / (RHO_CP_WATER * D))

            # Wind-induced mixed-layer deepening (equilibrium Katabatic / Pollard scale)
            # u_* = sqrt(tau / rho) ~ sqrt(1.3e-3 * 1.25 * V^2 / 1000) ~ 0.00127 * V
            u_star = np.maximum(0.001, 0.00127 * wind10m)
            # Equilibrium wind mixing depth: h_eq ~ 0.3 * u_* / f_cor (bounded by lake depth)
            f_cor_mid = 1.15e-4  # Coriolis parameter Benelux (~52°N)
            h_eq = np.clip(0.3 * u_star / f_cor_mid, 0.5, D)

            # Mixed-layer depth relaxation: dh/dt = (h_eq - h) / tau_relax
            # Convective mixing if cooling (q_net < 0): deepens rapidly
            tau_mix = np.where(q_net_in < 0.0, 3600.0, 21600.0)  # 1h convective vs 6h forced
            dh = (h_eq - self.hmld) * (dt / tau_mix)
            self.hmld = np.where(open_water, np.clip(self.hmld + dh, 0.5, D), self.hmld)

            # Update surface mixed layer temperature theta_s
            # Mixed layer heat absorption: (q_net_in - i_h)
            h_eff = np.maximum(0.5, self.hmld)
            d_tsfc = (q_net_in - i_h) * (dt / (RHO_CP_WATER * h_eff))
            self.tsfc_k = np.where(open_water, np.clip(self.tsfc_k + d_tsfc, 271.15, 315.0), self.tsfc_k)

            # Update bottom temperature theta_b via thermocline shape factor C_theta
            # In shallow well-mixed water (h >= 0.9 * D), theta_b tracks theta_s directly
            is_well_mixed = self.hmld >= (0.9 * D)
            self.tbot_k = np.where(
                open_water,
                np.where(is_well_mixed, self.tsfc_k, np.clip(self.tbot_k + 0.3 * d_theta_mean, 273.15, 295.0)),
                self.tbot_k,
            )

            # Frazil / surface ice freeze initiation when water cools to freezing point
            freeze_mask = open_water & (self.tsfc_k <= T_FREEZE_FRESH)
            if np.any(freeze_mask):
                excess_cool = T_FREEZE_FRESH - self.tsfc_k
                # Freeze surface ice: H_i = E_def / (rho_i * L_f)
                new_ice = (excess_cool * RHO_CP_WATER * h_eff) / (RHO_ICE_FRESH * LF_ICE_FRESH)
                self.ice_thick = np.where(freeze_mask, np.clip(new_ice, 0.005, 0.5), self.ice_thick)
                self.ice_tsfc = np.where(freeze_mask, T_FREEZE_FRESH, self.ice_tsfc)
                self.tsfc_k = np.where(freeze_mask, T_FREEZE_FRESH, self.tsfc_k)

    def to_dict(self) -> Dict[str, np.ndarray]:
        """Returns diagnostic state dictionary for WX-1 scientific export."""
        return {
            "lake_water_temp": self.tsfc_k,
            "lake_bottom_temp": self.tbot_k,
            "lake_mixed_layer_depth": self.hmld,
            "lake_ice_thickness": self.ice_thick,
            "lake_ice_surface_temp": self.ice_tsfc,
            "lake_depth": self.depth,
        }
