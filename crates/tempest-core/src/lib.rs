#![forbid(unsafe_code)]

//! Tempest-Core: Foundational atmospheric physics, constants, and thermodynamics.
//!
//! Strictly 100% classical physics, conservation laws, and thermodynamics. Zero AI/ML.

pub mod constants;
pub mod geodesy;
pub mod thermodynamics;

pub use constants::*;
pub use geodesy::*;
pub use thermodynamics::*;
