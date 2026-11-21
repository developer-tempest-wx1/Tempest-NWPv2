# NEMO 5.0.2 + SI³ Port Design for Tempest-NWP v2 GPU (`nh_ocean_si3.cu`)

**Reference source cloned (read-only, gitignored):**
- `external_refs/nemo_5.0.2` — official `https://forge.nemo-ocean.eu/nemo/nemo.git` tag `5.0.2` (commit `dcc7fb8`)
- Cross-checked against Tempest-NWP v1 CPU reference: `../Tempest-NWP/crates/tempest-ocean/src/`

**Absolute constraint:** 100% deterministic classical ocean fluid mechanics, sea-ice rheology, and thermodynamics (0% AI / 0% ML), strict SI units (`K`, `Pa`, `m`, `s`, `m/s`, `kg/m³`, `W/m²`, `kg/(m²·s)`, `PSU`), lossless WX-1 (`zstd_shuffle`) checkpoints and outputs.

---

## 1. Equations & Constants Carried from NEMO 5.0.2 `src/OCE` + `src/ICE`

| Subsystem | NEMO 5.0.2 Reference Module(s) | Exact Formulations & Namelist Defaults in `nh_ocean_si3.cu` |
|---|---|---|
| **Equation of State & Freezing** | `src/OCE/TRA/eosbn2.F90` | Full Roquet et al. (2015) 75-term TEOS-10 polynomial $\rho(\Theta, S_A, z)$ (`EOS000..EOS013`, `rdeltaS=32`, `r1_S0=0.875/35.16504`, `r1_T0=1/40`, `r1_Z0=1e-4`), UNESCO/Millero seawater freezing point $T_f(S) = (-0.0575 + 1.710523\times 10^{-3}\sqrt{S} - 2.154996\times 10^{-4} S) S$ ($\text{°C}$), Brunt-Väisälä $N^2$. Reference $\rho_0 = 1026.0\,\text{kg m}^{-3}$ (or `1025.0` where specified in bulk drag), $c_{p,o} = 3991.867957\,\text{J kg}^{-1}\text{K}^{-1}$. |
| **Split-Explicit Barotropic Mode + IB** | `src/OCE/DYN/dynspg_ts.F90`, `sbcapr.F90` | Staggered Arakawa C-grid forward-backward barotropic sub-cycling (`n_bt_substeps`) for $(\eta_{\text{ssh}}, u_b, v_b)$ forced by Inverse Barometer pressure anomaly $\eta_{\text{ib}} = -(p_{\text{mslp}} - 101325)/(\rho_0 g)$ and depth-mean wind/ice stress, strictly masked by `tmask` (`0` on land/lakes). |
| **3D Baroclinic Momentum & Vertical Mixing** | `dynhpg.F90`, `dynvor.F90`, `dynzdf.F90` | Hydrostatic pressure gradient from TEOS-10 density anomaly $\text{rhd}$, Coriolis + spherical metric, explicit horizontal biharmonic/Laplacian viscosity, and column-implicit Thomas vertical diffusion with quadratic bottom drag ($C_d = 2.5\times 10^{-3}$). |
| **Prognostic TKE + EVD + MLD** | `src/OCE/ZDF/zdftke.F90`, `zdfevd.F90`, `zdfmxl.F90` | Gaspar et al. (1990) / Blanke & Delecluse (1993) 1.5-order TKE closure (`rn_ediff=0.1`, `rn_ediss=0.7`, `rn_ebb=67.83`, `rn_emin=1e-6`, `rn_emin0=1e-4`, `rn_lc=0.15` Axell Langmuir term), Enhanced Vertical Diffusion ($K = 10\,\text{m}^2\text{s}^{-1}$ when $N^2 < 0$), mixed layer depth $\Delta\rho = 0.03\,\text{kg m}^{-3}$. |
| **Tracers & Penetrative Solar Flux** | `src/OCE/TRA/traadv*.F90`, `traqsr.F90`, `trazdf.F90` | Conservative positive-definite 3D upwind/TVD advection for $(T, S)$, Type-I 2-band solar penetration (`rn_abs=0.58`, $\zeta_1=0.35\,\text{m}$, $\zeta_2=23.0\,\text{m}$), implicit Thomas vertical diffusion. |
| **SI³ 5-Category ITD & Constants** | `src/ICE/ice.F90`, `par_ice.F90`, `iceitd.F90`, `phycst.F90` | $jpl=5$ thickness categories (`[0, 0.45, 1.10, 2.10, 4.00, 100.0]` m), $n_{\text{lay},i}=4$ BL99 layers; $\rho_i=917\,\text{kg m}^{-3}$, $\rho_s=330\,\text{kg m}^{-3}$, $k_i^0=2.034396\,\text{W m}^{-1}\text{K}^{-1}$, $c_{p,i}=2096.7\,\text{J kg}^{-1}\text{K}^{-1}$, $L_{f,i}=3.333601\times 10^5\,\text{J kg}^{-1}$, $\mu=0.054\,\text{K PSU}^{-1}$. |
| **SI³ mEVP Rheology, Advection & Ridging** | `src/ICE/icedyn_rhg_evp.F90`, `icedyn_adv.F90`, `icedyn_rdgrft.F90` | Bouillon (2013) modified EVP sub-cycled rheology (`n_evp_substeps=60`, `rn_pstar=2e4 N m-2`, `rn_crhg=20`, `rn_ecc=2`, `rn_Cd_io=5e-3`), conservative upwind/TVD advection, Thorndike ridging capping $\sum a_c \le 1$ (`rn_gstar=0.15`, `rn_fsnwrdg=0.5`) + ITD category remapping. |
| **SI³ BL99 Enthalpy, Growth/Melt, Frazil, Ponds & Albedo** | `icethd_zdf_bl99.F90`, `icethd_dh.F90`, `icethd_do.F90`, `icethd_pnd.F90`, `icealb.F90` | 4-layer implicit Thomas heat conduction with salinity-dependent $k_i(T,S)$ and $c_i(T,S)$, surface snow/ice melt, basal freezing/melt, Archimedes snow-ice flooding, open-water frazil accretion (`rn_hinew=0.10 m`) with brine rejection salt conservation, level melt-pond evolution, spectral albedo (`rn_alb_sdry=0.85`, `rn_alb_smlt=0.75`, `rn_alb_idry=0.64`, `rn_alb_imlt=0.53`, `rn_alb_dpnd=0.18`, `rn_alb_lpnd=0.30`, `rn_alb_hpiv=1.0 m`). |
| **Two-Way Atmosphere–Ice–Ocean Flux Update** | `src/ICE/iceupdate.F90` | Area-weighted ocean surface stress $(1 - A)\tau_{\text{atm}} + A\tau_{\text{io}}$, transmitted solar + non-solar fluxes, brine/freshwater salt balance, and fractional sea-ice mosaic coupling into Tempest `nh_physics.cu` (ecRad albedo & M-O/COARE surface fluxes). |

---

## 2. Intentionally Omitted Components

- **`src/TOP/` (PISCES biogeochemistry, CFC/AGE/C14 passive tracers):** outside operational weather & sea-state forecasting scope.
- **ORCA Tripolar North-Fold (`lbcnfd.F90`, `jperio=4/6`):** Tempest v2 operates on native regular spherical lat-lon (`D1`, periodic east-west with cuFFT polar filter) and Lambert Conformal Conic regional nests (`D2`/`D3`), eliminating tripolar fold bookkeeping.
- **Fortran XIOS XML server (`src/OCE/IOM/`, `xios-2.6.1.0`):** replaced directly by Tempest's native lossless WX-1 (`zstd_shuffle`) container writer and restart checkpoints.
- **`src/ABL/`, `src/SAS/`, `src/SWE/`:** standalone 1D/simplified test drivers unneeded because Tempest runs a full 3D non-hydrostatic GPU atmosphere.
