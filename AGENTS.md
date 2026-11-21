# AGENTS.md — Tempest-NWP Project Guidelines & Rules

## 1. Absolute Project Directive: 0% AI / 0% Machine Learning
**CRITICAL RULE: This project is strictly a classical, physics-based Numerical Weather Prediction (NWP) model.**
* Under NO circumstances may neural networks, deep learning models (GraphCast, ClimaX, Pangu-Weather, FourCastNet, GenCast), ML surrogate models, or data-driven approximations be introduced into the model physics or dynamical core.
* Every single line of code must be rooted in deterministic atmospheric physics, thermodynamics, classical fluid dynamics (Euler / Navier-Stokes equations), and established numerical analysis.
* All implemented equations should be formulated consistently with physical conservation laws. Each production kernel must still be audited for conservation and positivity; do not describe the full model as conservative until verified. Intended invariants include:
  1. Conservation of mass (continuity equation).
  2. Conservation of momentum (Navier-Stokes on a rotating sphere with Coriolis acceleration).
  3. Conservation of energy (First Law of Thermodynamics, potential temperature $\theta$, latent heat exchange).
  4. Conservation of moisture tracers (positive-definite advection of water vapor $q_v$, cloud liquid $q_c$, rain $q_r$, cloud ice $q_i$, snow $q_s$, graupel $q_g$).

---

## 2. Storage Backbone: WX-1 (`../WX-1`)
* All static datasets (topography, land-use, roughness, soil types), boundary conditions (LBC), initial conditions (IC), model checkpoints, and forecast output MUST use the **WX-1 scientific container format** (`.wx1`).
* WX-1 is located at `D:\AI\weer-ai\Tempest\WX-1` (`/mnt/d/AI/weer-ai/Tempest/WX-1`).
* **Bit-Exact IEEE 754 Determinism (target, not current guarantee)**: Lossless storage is required for prognostic fields. Do not claim full restart bit-exactness unless it is demonstrated across the atmospheric, land, ocean and sea-ice state on the target hardware.
* Utilize optimized chunk topologies:
  * Horizontal synoptic forecast maps: `[1, nlat, nlon]` (instant global horizontal slicing).
  * Vertical atmospheric columns / soundings: `[time, nlevels, 1, 1]` (instant local atmospheric column retrieval).
* Codecs: Default to `zstd_shuffle` for lossless atmospheric fields. Lossy compression is prohibited in prognostic checkpointing.

---

## 3. Operational Initialisation and Observation Ingestion
* Tempest-NWP is an independent numerical weather prediction model. Do not describe ECMWF or GFS as its dynamics/physics engine.
* The current cold-start path may use an ECMWF IFS step-0 analysis as a background/initialisation field when no valid Tempest restart exists. It does not automatically select a T+3/T+6 forecast lead, and it does not feed later ECMWF/GFS forecast fields into every model timestep.
* The `--ecmwf` option selects the forecast output schedule (3-hourly through T+144, then 6-hourly through T+360); it does not select ECMWF model physics or forecast data as the Tempest model.
* Operational DA observation inputs are gathered from the configured station, radiosonde, satellite, WIS2 and marine providers, QC-filtered, converted to WX-1 and passed to the active local analysis increments. Raw-only feeds must not be counted as assimilated.
* Preserve source provenance and valid times. All gridded model state, static fields, observation containers and forecast products use the WX-1 scientific container format where the current reader/writer supports it.

---

## 4. Git Workflow Ruleset
1. **Status First**: Run a clean git status check before making any changes.
2. **Atomic Commits**: Make small, logical commits with clear descriptive commit messages detailing the physical or numerical change.
3. **Verification Before Commit**: Always verify builds, unit tests, and numerical benchmarks before committing.
4. **No Binary Bloat**: Never commit large GRIB2, NetCDF, or WX-1 data files directly into git; store binaries and test datasets in external data paths or gitignore.

---

## 5. Ponytail Ruleset (Anti-Bloat & Minimal Complexity)
You are a senior meteorological software engineer who abhors code bloat. Always deliver the smallest possible, mathematically sound implementation. Before writing code, adhere to these 7 steps:
1. **YAGNI**: Is this strictly required for atmospheric physics or numerical stability?
2. **Re-use**: Can existing vectorized array operations or standard mathematical libraries solve this?
3. **Standard Library**: Does the standard library already provide this algorithm?
4. **Native Platform**: Does the OS/compiler (e.g. SIMD AVX2/AVX-512, openMP, BLAS) provide hardware acceleration?
5. **Installed Dependency**: Does an already installed verified crate/package cover this?
6. **One Expression**: Prefer clean, readable, mathematically transparent expressions over multi-layer wrapper boilerplate.
7. **Minimum Implementation**: Implement only the required physical parameterization or numerical solver.

---

## 6. Caveman Ruleset (Explicit Over Implicit)
Write code as if unwarranted abstraction is dangerous to numerical stability and reproducibility:
1. **No Clever Tricks**: No obscure bit-hacks or hidden side effects. Write explicit, clearly documented mathematical steps that mirror the physics equations in textbooks.
2. **Inline over Nested Indirection**: Do not hide simple tensor arithmetic behind 5 layers of abstract classes or traits. Keep the numerical kernel transparent.
3. **Beaten Path**: Use time-tested numerical schemes (e.g., Arakawa C-grid, Runge-Kutta 3rd order, WENO-5 advection, Monin-Obukhov surface flux).
4. **Explicit Over Implicit**: Variable names must carry clear meteorological meaning (e.g., `u_wind`, `v_wind`, `theta_pot_temp`, `geopotential_phi`, `exner_pi`, `pressure_p`).
5. **Strict SI Units**: All internal model calculations must strictly use standard SI units:
   * Temperature: Kelvin ($K$)
   * Pressure: Pascals ($Pa$)
   * Velocity: Meters per second ($m/s$)
   * Specific humidity / mixing ratios: Kilograms per kilogram ($kg/kg$)
   * Geopotential: $m^2/s^2$
   * Precipitation rates: $kg/(m^2 \cdot s)$ or $mm/s$
