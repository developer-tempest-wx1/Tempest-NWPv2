//! Physical, meteorological, and thermodynamic constants.
//!
//! Strictly adheres to SI units and standard WMO / IUGG values.
//! 100% deterministic classical atmospheric physics.

/// Specific gas constant for dry air $R_d$ [J / (kg K)].
pub const R_D: f64 = 287.058;

/// Specific gas constant for water vapor $R_v$ [J / (kg K)].
pub const R_V: f64 = 461.500;

/// Specific heat of dry air at constant pressure $c_p$ [J / (kg K)].
pub const CP_D: f64 = 1004.5;

/// Specific heat of dry air at constant volume $c_v$ [J / (kg K)].
pub const CV_D: f64 = 717.5;

/// Specific heat of liquid water $c_l$ [J / (kg K)].
pub const C_L: f64 = 4186.0;

/// Specific heat of ice $c_i$ [J / (kg K)].
pub const C_I: f64 = 2106.0;

/// Ratio of gas constants $\kappa = R_d / c_p \approx 0.285714$.
pub const KAPPA: f64 = R_D / CP_D;

/// Molecular weight ratio of water vapor to dry air $\epsilon = R_d / R_v \approx 0.622$.
pub const EPSILON: f64 = R_D / R_V;

/// Factor used in virtual temperature calculation: $(1/\epsilon - 1) \approx 0.6078$.
pub const VIRTUAL_FACTOR: f64 = (1.0 / EPSILON) - 1.0;

/// Standard acceleration due to gravity at sea level $g$ [m / s²].
pub const GRAVITY: f64 = 9.80665;

/// Standard Earth angular velocity $\Omega$ [rad / s].
pub const EARTH_ANGULAR_VELOCITY: f64 = 7.292115e-5;

/// Mean Earth radius $a$ [m].
pub const EARTH_RADIUS: f64 = 6371229.0;

/// Standard reference sea-level surface pressure $P_0$ [Pa].
pub const P0: f64 = 100000.0;

/// Freezing point of water at 1 atm $T_0$ [K].
pub const T0: f64 = 273.15;

/// Latent heat of vaporization at $0^\circ\text{C}$ $L_v$ [J / kg].
pub const LV0: f64 = 2.501e6;

/// Latent heat of sublimation at $0^\circ\text{C}$ $L_s$ [J / kg].
pub const LS0: f64 = 2.834e6;

/// Latent heat of fusion $L_f$ [J / kg].
pub const LF0: f64 = 3.337e5;

/// Von Kármán turbulence constant $\kappa_v$ [dimensionless].
pub const VON_KARMAN: f64 = 0.40;

/// Stefan-Boltzmann constant $\sigma_{SB}$ [W / (m² K⁴)].
pub const STEFAN_BOLTZMANN: f64 = 5.670374419e-8;
