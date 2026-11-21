//! Geodesy and spherical coordinate transformations on Earth.
//!
//! Provides Coriolis parameter, metric terms, map scale factors, and distances.

use crate::constants::{EARTH_ANGULAR_VELOCITY, EARTH_RADIUS};
use std::f64::consts::PI;

/// Converts degrees to radians.
#[inline]
pub fn deg_to_rad(deg: f64) -> f64 {
    deg * (PI / 180.0)
}

/// Converts radians to degrees.
#[inline]
pub fn rad_to_deg(rad: f64) -> f64 {
    rad * (180.0 / PI)
}

/// Computes the Coriolis parameter $f = 2 \Omega \sin(\phi)$ [s⁻¹].
#[inline]
pub fn coriolis_parameter(lat_rad: f64) -> f64 {
    2.0 * EARTH_ANGULAR_VELOCITY * lat_rad.sin()
}

/// Computes the horizontal grid spacing $\Delta x$ along a parallel at latitude $\phi$:
/// $\Delta x = a \cos(\phi) \Delta \lambda$ [m].
#[inline]
pub fn dx_at_latitude(lat_rad: f64, dlon_rad: f64) -> f64 {
    EARTH_RADIUS * lat_rad.cos() * dlon_rad
}

/// Computes the meridional grid spacing $\Delta y$ along a meridian:
/// $\Delta y = a \Delta \phi$ [m].
#[inline]
pub fn dy_along_meridian(dlat_rad: f64) -> f64 {
    EARTH_RADIUS * dlat_rad
}

/// Great circle distance between two points on the sphere using the Haversine formula [m].
pub fn haversine_distance(lat1_rad: f64, lon1_rad: f64, lat2_rad: f64, lon2_rad: f64) -> f64 {
    let dlat = lat2_rad - lat1_rad;
    let dlon = lon2_rad - lon1_rad;
    let a = (dlat / 2.0).sin().powi(2)
        + lat1_rad.cos() * lat2_rad.cos() * (dlon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());
    EARTH_RADIUS * c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coriolis_at_equator_and_pole() {
        assert_eq!(coriolis_parameter(0.0), 0.0);
        let f_pole = coriolis_parameter(PI / 2.0);
        assert!((f_pole - 2.0 * EARTH_ANGULAR_VELOCITY).abs() < 1e-15);
    }

    #[test]
    fn test_great_circle_distance() {
        // Quarter meridian from equator to north pole: 0.5 * PI * R
        let d = haversine_distance(0.0, 0.0, PI / 2.0, 0.0);
        let expected = 0.5 * PI * EARTH_RADIUS;
        assert!((d - expected).abs() < 1e-6);
    }
}
