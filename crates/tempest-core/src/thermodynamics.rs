//! Classical atmospheric thermodynamics functions.
//!
//! All calculations are strictly physical and analytical, based on
//! Gibbs free energy formulations, Clausius-Clapeyron relation, and
//! ideal gas laws. Zero heuristics or ML approximations.

use crate::constants::*;

/// Computes the Exner function $\Pi = (p / P_0)^\kappa$.
#[inline]
pub fn exner(pressure_pa: f64) -> f64 {
    (pressure_pa / P0).powf(KAPPA)
}

/// Computes dry air density $\rho = p / (R_d T_v)$ [kg / m³].
#[inline]
pub fn density(pressure_pa: f64, virtual_temp_k: f64) -> f64 {
    pressure_pa / (R_D * virtual_temp_k)
}

/// Computes potential temperature $\theta = T / \Pi = T (P_0 / p)^\kappa$ [K].
#[inline]
pub fn potential_temperature(temp_k: f64, pressure_pa: f64) -> f64 {
    temp_k / exner(pressure_pa)
}

/// Computes temperature from potential temperature and Exner function $T = \theta \cdot \Pi$ [K].
#[inline]
pub fn temperature_from_theta(theta_k: f64, exner_val: f64) -> f64 {
    theta_k * exner_val
}

/// Computes virtual temperature $T_v = T (1 + 0.6078 q_v - q_{condensed})$ [K].
#[inline]
pub fn virtual_temperature(temp_k: f64, qv_kg_kg: f64, q_condensed_kg_kg: f64) -> f64 {
    temp_k * (1.0 + VIRTUAL_FACTOR * qv_kg_kg - q_condensed_kg_kg)
}

/// Computes virtual potential temperature $\theta_v = \theta (1 + 0.6078 q_v - q_{condensed})$ [K].
#[inline]
pub fn virtual_potential_temperature(theta_k: f64, qv_kg_kg: f64, q_condensed_kg_kg: f64) -> f64 {
    theta_k * (1.0 + VIRTUAL_FACTOR * qv_kg_kg - q_condensed_kg_kg)
}

/// Computes saturation vapor pressure over liquid water using the Tetens formula [Pa].
///
/// Valid for temperatures between $-40^\circ\text{C}$ and $+50^\circ\text{C}$.
#[inline]
pub fn saturation_vapor_pressure_liquid(temp_k: f64) -> f64 {
    let t_c = temp_k - T0;
    610.78 * (17.27 * t_c / (t_c + 237.3)).exp()
}

/// Computes saturation vapor pressure over ice using Tetens formula [Pa].
#[inline]
pub fn saturation_vapor_pressure_ice(temp_k: f64) -> f64 {
    let t_c = temp_k - T0;
    610.78 * (21.875 * t_c / (t_c + 265.5)).exp()
}

/// Computes saturation mixing ratio $q_{vsat} = \frac{\epsilon e_{sat}}{p - (1 - \epsilon) e_{sat}}$ [kg / kg].
#[inline]
pub fn saturation_mixing_ratio(pressure_pa: f64, e_sat_pa: f64) -> f64 {
    let denom = pressure_pa - (1.0 - EPSILON) * e_sat_pa;
    if denom <= 0.0 {
        0.0
    } else {
        (EPSILON * e_sat_pa) / denom
    }
}

/// Computes vapor pressure from water vapor mixing ratio $e = \frac{p \cdot q_v}{\epsilon + (1 - \epsilon) q_v}$ [Pa].
#[inline]
pub fn vapor_pressure(pressure_pa: f64, qv_kg_kg: f64) -> f64 {
    (pressure_pa * qv_kg_kg) / (EPSILON + (1.0 - EPSILON) * qv_kg_kg)
}

/// Computes dew point temperature $T_d$ from vapor pressure using inverted Tetens formula [K].
#[inline]
pub fn dewpoint(e_pa: f64) -> f64 {
    let safe_e = e_pa.max(1e-4);
    let log_ratio = (safe_e / 610.78).ln();
    let t_c = (237.3 * log_ratio) / (17.27 - log_ratio);
    t_c + T0
}

/// Computes equivalent potential temperature $\theta_e$ [K] using Bolton (1980) formulation.
pub fn equivalent_potential_temperature(temp_k: f64, pressure_pa: f64, qv_kg_kg: f64) -> f64 {
    let e = vapor_pressure(pressure_pa, qv_kg_kg);
    let t_dew = dewpoint(e);
    // Temperature at Lifting Condensation Level (LCL)
    let t_lcl = 1.0 / (1.0 / (t_dew - 56.0) + (temp_k / t_dew).ln() / 800.0) + 56.0;
    let theta_l = temp_k * (P0 / (pressure_pa - e)).powf(0.2854) * (temp_k / t_lcl).powf(0.28 * qv_kg_kg);
    theta_l * ((3036.0 / t_lcl - 1.78) * qv_kg_kg * (1.0 + 0.448 * qv_kg_kg)).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exner_standard_sea_level() {
        let pi = exner(P0);
        assert!((pi - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_potential_temperature() {
        let t = 288.15;
        let p = 100000.0;
        let theta = potential_temperature(t, p);
        assert!((theta - 288.15).abs() < 1e-10);

        // At 500 hPa (50000 Pa), potential temperature should be higher than T
        let theta_500 = potential_temperature(250.0, 50000.0);
        assert!(theta_500 > 250.0);
    }

    #[test]
    fn test_saturation_vapor_pressure() {
        // At 0 deg C (273.15 K), saturation vapor pressure is ~610.78 Pa
        let es = saturation_vapor_pressure_liquid(273.15);
        assert!((es - 610.78).abs() < 1.0);
    }

    #[test]
    fn test_dewpoint_cycle() {
        let t = 293.15; // 20 C
        let es = saturation_vapor_pressure_liquid(t);
        let td = dewpoint(es);
        assert!((td - t).abs() < 0.05);
    }
}
