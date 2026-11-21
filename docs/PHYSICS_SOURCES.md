# Tempest-NWP v2 Physics Sources, Equations & Numerical Guards

**Strict Rule:** 100% deterministic classical atmospheric, land, lake, and ocean physics (**0% AI / 0% ML**), SI units (`K`, `Pa`, `m/s`, `kg/kg`, `W/m^2`), and strict clean-room licensing hygiene. All CUDA kernels in `crates/tempest-cuda/src/cuda/` derive strictly from **Public Domain / BSD-3-Clause / Apache-2.0** reference codebases and peer-reviewed published equations. No GPL-3 or CeCILL source code is copied.

---

## 1. Inspected Reference Repositories (`/mnt/d/AI/weer-ai/Tempest/_ref_repos/`)

| Component / Agency | Repository | Commit (`HEAD`) | License | Inspected Physics Modules |
| :--- | :--- | :--- | :--- | :--- |
| **NCAR WRF** | `wrf-model/WRF` | `06d4240` | Public Domain | `phys/module_mp_wsm6.F`, `module_bl_ysu.F`, `module_sf_sfclayrev.F`, `module_sf_noahlsm.F` |
| **NCAR MMM CCPP** | `NCAR/MMM-physics` | `d524bc9` | **BSD-3-Clause** | `mp_wsm6.F90`, `mp_wsm6_effectRad.F90`, `bl_ysu.F90`, `sf_sfclayrev.F90` |
| **ECMWF ecRad** | `ecmwf-ifs/ecrad` | `c061bb6` | **Apache-2.0** | `radiation/{radiation_delta_eddington.h, radiation_two_stream.F90, radiation_liquid_optics_socrates.F90, radiation_ice_optics_fu.F90, radiation_adding_ica_sw.F90, radiation_adding_ica_lw.F90}` |
| **ECMWF CLOUDSC** | `ecmwf-ifs/dwarf-p-cloudsc` | `f7ba9f8` | **Apache-2.0** | `src/cloudsc_cuda/cloudsc/cloudsc_c_k_caching.cu` |
| **ECMWF ecLand** | `ecmwf-ifs/ecLand` | `066c0c8` | **Apache-2.0** | `src/surf/module/{voskin_mod.F90, srfts_mod.F90, sussoil_mod.F90, susveg_mod.F90}` |
| **DWD ICON** | `icon-model` | `7de8533` | **BSD-3-Clause** | `ragnarok/aes_microphysics/{constants.hpp, graupel.hpp, graupel.ipp, physics.hpp, transition.hpp}`, `src/physics_granules/microphys_1mom/gscp_{data,graupel}.f90`, `src/lnd_phy_schemes/sfc_flake{_data,}.f90` |
| **KNMI / ACCORD PHYEX** | `UMR-CNRM/PHYEX` | `346fde1` | CeCILL-C *(formulas only)* | `src/phyex/micro/{modd_icet_param.F90, compute_frac_ice.func.h, ice4_nucleation.func.h}`, `src/phyex/turb/{shallow_mf.F90, mode_compute_updraft.F90}` |
| **NEMO 5.0 Ocean** | `nemo` | `d6c9fd7` | CeCILL *(published Fairall/Edson/Takaya papers + Apache ecLand VOSKIN)* | `src/OCE/SBC/{sbcblk_algo_coare3p6.F90, sbcblk_skin_coare.F90}`, `src/OCE/ZDF/zdftke.F90` |
| **KNMI / WUR MicroHH** | `microhh/microhh` | `736a2f5` | GPL-3 *(GPU layout study only; zero code copied)* | `src/microphys_nsw6.cu` |

---

## 2. Adopted Equations, Constants & Stability Guards by Pillar

### Pillar 1 — 6-Class Mixed-Phase Microphysics + SLW Preservation + Fallout (`nh_microphysics_kernel`)

*Sources:* `MMM-physics/mp_wsm6.F90`, `icon-model/ragnarok/aes_microphysics/{constants.hpp, physics.hpp, transition.hpp, graupel.ipp}`, `dwarf-p-cloudsc/cloudsc_c_k_caching.cu`, and published ICE-T / HARMONIE nucleation threshold literature.

1. **Supercooled Liquid Water (SLW) Preservation:**
   - Instantaneous saturation adjustment condenses excess vapor to cloud liquid `qc` for $T > T_{\text{hom}} = 236.15\text{ K}$ (`tfrz_hom = tmelt - 37 K` in `icon-model`), avoiding premature glaciation between $0^\circ\text{C}$ and $-37^\circ\text{C}$ (critical for North Sea/Benelux winter stratus and icing).
   - Above $273.15\text{ K}$, `qi` melts instantaneously to `qc`; below $236.15\text{ K}$, `qc` freezes homogeneously to `qi`.
   - Evaporation cleanup removes sub-threshold micro-trace amounts ($< 10^{-12}\text{ kg/kg}$).
2. **Microphysical Transition Matrix `sx2x[6][6]`:**
   - *Warm-rain (`qc -> qr`):* Seifert & Beheng (2001) autoconversion + accretion from `icon-model/transition.hpp` with maritime/continental droplet number concentration (`cloud_num_land = 200e6 m^-3`, `cloud_num_sea = 80e6 m^-3`).
   - *Ice Nucleation (`qv -> qi`):* Cooper number curve $N_i = \min(2.5\times 10^5, \, 5 \exp(0.304(T_0 - T))) / \rho$ (`physics.hpp`), active for $(T < T_0 - 6\text{ K} \wedge q_c > q_{\min})$ or $(T < T_0 - 25\text{ K} \wedge q_v > q_{\text{sat},i})$ with initial crystal mass $m_{0,i} = 10^{-12}\text{ kg}$.
   - *Ice/Snow/Graupel Deposition & Sublimation (WBF):* `mp_wsm6.F90` / `physics.hpp` diffusional growth $F_{\text{dep}}(T, q_{\text{sat},i})$ capped by ice supersaturation $\Delta q_i / \Delta t$.
   - *Riming, Aggregation & Freezing:* Field et al. snow $N_{0s}(T, \rho q_s)$ + sticking efficiency `ice_sticking(T)`, `cloud_to_snow`, `cloud_to_graupel`, `rain_to_graupel`, and `snow_to_rain`/`graupel_to_rain` melting at $T > 273.15\text{ K}$.
3. **Positivity & Mass/Enthalpy Conservation Guard (`compute_sink`, `graupel.ipp`):**
   - For each species $s \in \{q_v, q_c, q_r, q_i, q_s, q_g\}$, total outgoing rate $S_s = \sum_{d} R_{s \to d}$ is scaled proportionally whenever $S_s \Delta t > q_s$: $R_{s \to d} \leftarrow R_{s \to d} \cdot (q_s / (S_s \Delta t))$.
   - Guarantees $q_s \ge 0$ strictly everywhere and exact local water mass conservation before precipitation fallout.
4. **Conservative Subcycled Fallout (`precip`):**
   - WSM6 / COSMO Marshall-Palmer mass-weighted fall speeds (`pvtr, pvts(n0s(T)), pvtg, v_i`) scaled by $(\rho_0/\rho)^{0.5}$.
   - Column-local subcycling ensures Courant number $v_t \delta t / \Delta z \le 0.70$ at `nz=137` ($\Delta z_0 \approx 19\text{ m}$), with exact finite-volume flux telescoping in `float64` accumulating surface `rain_acc`, `snow_acc`, and `graupel_acc` $[\text{kg}/\text{m}^2]$.

---

### Pillar 2 — Nonlocal YSU PBL + Shallow EDMFm Mass-Flux (`nh_pbl_*_kernel`)

*Sources:* `MMM-physics/bl_ysu.F90`, plus published Siebesma et al. (2007) / Pergaud et al. (2009) shallow EDMF updraft equations.

1. **PBL Top & Scaling (`bl_ysu.F90`):**
   - Bulk Richardson thresholds: $Ri_{b,\text{cr}} = 0.0$ (unstable), $0.25$ (stable land), and Rossby-dependent $\min(0.16 (10^{-7} Ro)^{-0.18}, 0.3)$ over sea.
   - Surface convective velocity $w_* = \left((g/\theta_v) \overline{w'\theta'_v}_0 h_{\text{pbl}}\right)^{1/3}$, mixed velocity scale $w_s = (u_*^3 + 8 \kappa w_*^3 \cdot 0.5)^{1/3}$ clamped to $[u_*/5, \, 16 u_*]$.
   - Unstable surface thermal excess & nonlocal counter-gradient $\gamma_\theta, \gamma_q$.
2. **Nonlocal K-Profile, Top Entrainment & Free-Atmosphere Louis Mixing:**
   - Within PBL: $K_m(z) = w_s \kappa z (1 - z/h)^2$, $K_h = K_m / Pr$, clamped to $[K_{\min,m}=0.1, 1000.0]\text{ m}^2/\text{s}$ (`xkzminh = 0.01`).
   - Explicit PBL-top entrainment buoyancy flux: $\overline{w'\theta'_v}_h = -0.15 (\theta_{v0}/g) w_m^3 / h_{\text{pbl}}$, $w_m^3 = w_*^3 + 5 u_*^3$, exponentially decaying into entrainment zone.
   - Above PBL: moist Richardson mixing $K_m, K_h = \ell^2 S \, f(Ri)$.
3. **Shallow Thermal Updraft Mass-Flux Coupling (`EDMFm`):**
   - Over convective columns ($w_* > 0$), integrate an entraining steady plume $w_u^2(z), \theta_{l,u}(z), q_{t,u}(z)$ from surface up to dry/moist zero-buoyancy top (capped at `4000 m`), yielding mass flux $M_u = a_u \rho w_u$ ($a_u = 0.1$) entering the scalar implicit solver alongside $-K \partial_\zeta$.
4. **Column-Implicit Tridiagonal Solver:**
   - Reuses global `d_tri_c`, `d_tri_d` in `double` for $\theta', q_v, q_c, q_i$ and staggered $u, v$ columns (exact mass conservation + unconditional vertical stability at `nz=137`).

---

### Pillar 3 — Cloud-Interactive Two-Stream SW/LW Radiation (`nh_radiation_ecrad_kernel`)

*Source:* `ecmwf-ifs/ecrad/radiation/` (**Apache-2.0**) + `MMM-physics/mp_wsm6_effectRad.F90`.

1. **Solar Geometry & Cloud Effective Radii:**
   - Exact astronomical declination $\delta(J_d)$, equation of time, hour angle from `lon_deg` + `utc_seconds`, yielding $\mu_0 = \max(\cos\theta_z, 0)$.
   - Effective radii from liquid/ice density (`mp_wsm6_effectRad.F90` / `ecrad`), clamped to SOCRATES `[1.2e-6, 50e-6] m` and Fu `[5e-6, 100e-6] m`.
2. **Delta-Eddington + Meador-Weaver SW (`radiation_delta_eddington.h`, `radiation_two_stream.F90`):**
   - Delta-Eddington scaling: $f = g^2$, $\tau' = \tau(1 - \omega f)$, $\omega' = \omega(1 - f)/(1 - \omega f)$, $g' = g/(1+g)$.
   - Zdunkowski PIFM two-stream $\gamma_1, \gamma_2, \gamma_3, \gamma_4 = 1 - \gamma_3$, $k = \sqrt{\max((\gamma_1 - \gamma_2)(\gamma_1 + \gamma_2), 10^{-12})}$.
   - Near-resonance guard on $\mu_0$ when $|1 - k\mu_0| < 10^{-6}$, energy clamping $R_{\text{diff}}, T_{\text{diff}}, R_{\text{dir}}, T_{\text{dir,diff}} \in [0, 1]$ with $R + T \le 1$.
3. **Longwave Meador-Weaver + Linear-in-$\tau$ Planck + Adding Method (`radiation_two_stream.F90`, `adding_ica_*`):**
   - `LwDiffusivity = 1.66`, $\gamma_1 = D(1 - 0.5\omega(1+g))$, $\gamma_2 = 0.5 D \omega(1-g)$.
   - Thin-layer Taylor branch when $\tau \le 10^{-3}$ avoids `0/0` in Stackhouse & Stephens linear Planck source terms.
   - Multi-band Lacis & Hansen (1974) / Shonk & Hogan (2008) Adding Method using `d_tri_c`/`d_tri_d` (2 SW bands UV/Vis+NIR; 3 LW bands Window+$\text{H}_2\text{O}$+$\text{CO}_2$ calibrated to standard clear-sky & cloudy profiles) yields `sw_down`, `lw_down`, and heating rate `dtheta/dt`.

---

### Pillar 4 — Surface Layer, Land LSM, North Sea COARE 3.6 + Skin & FLake (`nh_surface_and_lsm_kernel`)

*Sources:* `MMM-physics/sf_sfclayrev.F90`, `ecLand/src/surf/module/{voskin_mod.F90, srfts_mod.F90, sussoil_mod.F90, susveg_mod.F90}`, `WRF/phys/module_sf_noahlsm.F`, `icon-model/src/lnd_phy_schemes/sfc_flake{_data,}.f90`, and published Fairall et al. (2003, 2019) / Edson et al. (2013) COARE 3.6.

1. **Surface Layer (`sf_sfclayrev.F90`):**
   - Jiménez et al. (2012) integrated stability functions $\psi_m(\zeta), \psi_h(\zeta)$ across unstable, weakly stable, and very stable regimes, solved via bounded secant solver (`zolri`) with Beljaars gustiness $w_{\text{conv}}$.
   - 2 m temperature/humidity and 10 m wind diagnosed using M-O profile logs and roughness `z0m`, `z0h`.
2. **Land Surface (`ecLand` + `NOAA Noah` 4-layer HTESSEL grid):**
   - 4 soil layers `(0.07, 0.21, 0.72, 1.89) m` matching ECMWF IFS / `tempest_v2.ingest_ecmwf`.
   - `ecLand` skin conductivity $\Lambda_{\text{sk}}$ (`10 W/(m^2 K)` vegetated, `15` bare, `58` snow), USDA 13-class FAO HWSD v2.2 thermal/hydraulic parameters (`soil_model.py`), Jarvis stomatal resistance, Peters-Lidard Kersten thermal conductivity, freeze/thaw latent buffering, and 2-iteration skin energy balance closure $R_{\text{net}} = H + LE + G$.
3. **North Sea Ocean (`COARE 3.6` + `ecLand VOSKIN`):**
   - $0.98$ sea-salt vapor saturation factor; Charnock $\alpha_{\text{Ch}}(U_{10n}) = \max(\min(0.0017 U_{10n} - 0.005, \, 0.028), \, 0.0)$ + smooth viscous term $0.11\nu_a/u_*$ + scalar $z_{0t} = z_{0q} = \min(1.6\times 10^{-4}, \, 5.8\times 10^{-5} Re_r^{-0.72})$ + convective gustiness ($\beta_0 = 1.2, z_i = 600\text{ m}$).
   - Shallow North Sea bathymetry roughness boost from GEBCO 2026 depth (`depth_dependent_z0`, clamped `[10, 100] m`, capped `0.1 m`).
   - Fairall (1996) / `voskin_mod` cool-skin $\Delta T_{\text{cool}}$ + Takaya/Zeng prognostic warm-layer $\Delta T_{\text{warm}}$ (`CHVER='C'`, warm layer scale depth clamped for shallow shelf water) -> $T_{\text{sk}} = \text{SST}_{\text{bulk}} + \Delta T_{\text{warm}} + \Delta T_{\text{cool}}$.
4. **Freshwater Lakes (`ICON BSD-3` `sfc_flake_data`):**
   - 1D two-layer FLake column for IJsselmeer, Markermeer, Bodensee (`lake_mask >= 0.5`), strictly enforcing `h_ML in [0.01, D]`, `C_T in [0.5, 0.8]`, $T_{\max\rho} = 277.13\text{ K}$, $T_f = 273.15\text{ K}$, `H_Ice_max = 3.0 m`, Mironov & Ritter white/blue ice albedo `c_albice_MR = 95.6`, and Stefan ice thickness growth/melt.
