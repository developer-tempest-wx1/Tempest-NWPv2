#include "nh_internal.cuh"

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

using namespace tempest_cuda_internal;

namespace {

// WMO / NOAA / USDA-HWSD 14-row soil lookup table matching tempest_v2/soil_model.py:
// Columns: [0:theta_sat, 1:theta_fc, 2:theta_wilt, 3:b, 4:psi_sat(m), 5:ksat(m/s), 6:kappa_dry(W/mK), 7:kappa_sat(W/mK)]
__constant__ double c_soil_table[14][8] = {
    {0.000, 0.000, 0.000,  0.00, 0.000, 0.00e-6, 0.00, 0.60}, // 0: Water
    {0.482, 0.395, 0.286, 11.40, 0.405, 1.28e-6, 0.20, 1.25}, // 1: Clay (heavy)
    {0.470, 0.380, 0.270, 10.40, 0.350, 1.50e-6, 0.21, 1.30}, // 2: Silty clay
    {0.476, 0.367, 0.257,  8.52, 0.356, 2.17e-6, 0.22, 1.35}, // 3: Clay (light)
    {0.477, 0.350, 0.225,  7.75, 0.320, 2.80e-6, 0.23, 1.40}, // 4: Silty clay loam
    {0.465, 0.325, 0.198,  6.77, 0.299, 3.50e-6, 0.24, 1.45}, // 5: Clay loam
    {0.460, 0.315, 0.180,  5.30, 0.280, 4.20e-6, 0.24, 1.50}, // 6: Silt
    {0.485, 0.300, 0.150,  5.30, 0.260, 5.23e-6, 0.25, 1.55}, // 7: Silt loam
    {0.430, 0.310, 0.210,  6.00, 0.200, 4.80e-6, 0.25, 1.60}, // 8: Sandy clay
    {0.451, 0.283, 0.119,  5.39, 0.478, 3.38e-6, 0.25, 1.58}, // 9: Loam
    {0.420, 0.255, 0.135,  4.90, 0.290, 5.20e-6, 0.26, 1.70}, // 10: Sandy clay loam
    {0.435, 0.215, 0.084,  4.74, 0.141, 1.41e-5, 0.27, 1.80}, // 11: Sandy loam
    {0.410, 0.174, 0.068,  4.26, 0.090, 4.79e-5, 0.28, 2.00}, // 12: Loamy sand
    {0.395, 0.135, 0.045,  4.05, 0.121, 1.76e-4, 0.30, 2.20}  // 13: Sand
};

// ECMWF HTESSEL 4-layer soil thicknesses [m]
__constant__ double c_soil_dz[4] = {0.07, 0.21, 0.72, 1.89};

// ---------------------------------------------------------------------------
// Shared Device Thermodynamics & Column Geometry Helpers
// ---------------------------------------------------------------------------

__device__ inline double es_water(double t_k) {
    const double t = fmin(fmax(t_k, 173.15), 373.15);
    return 611.21 * exp(17.502 * (t - T_FREEZE) / (t - 32.19));
}

__device__ inline double es_ice(double t_k) {
    const double t = fmin(fmax(t_k, 150.0), T_FREEZE);
    return 611.21 * exp(22.587 * (t - T_FREEZE) / (t + 0.7));
}

__device__ inline double qsat_from_es(double es, double p_pa) {
    const double p = fmax(p_pa, 100.0);
    const double ec = fmin(es, 0.90 * p);
    return EPS_MW * ec / fmax(p - (1.0 - EPS_MW) * ec, 100.0);
}

__device__ inline double qsat_water(double t_k, double p_pa) {
    return qsat_from_es(es_water(t_k), p_pa);
}

__device__ inline double qsat_ice(double t_k, double p_pa) {
    return qsat_from_es(es_ice(t_k), p_pa);
}

__device__ inline double dqsat_water_dt(double t_k, double p_pa) {
    const double t = fmin(fmax(t_k, 173.15), 373.15);
    const double es = fmin(es_water(t), 0.90 * fmax(p_pa, 100.0));
    const double qs = qsat_from_es(es, p_pa);
    const double dlne = 4217.28192 / ((t - 32.19) * (t - 32.19));
    return qs * p_pa / fmax(p_pa - (1.0 - EPS_MW) * es, 100.0) * dlne;
}

__device__ inline double dqsat_ice_dt(double t_k, double p_pa) {
    const double t = fmin(fmax(t_k, 150.0), T_FREEZE);
    const double es = fmin(es_ice(t), 0.90 * fmax(p_pa, 100.0));
    const double qs = qsat_from_es(es, p_pa);
    const double dlne = 6185.45015 / ((t + 0.7) * (t + 0.7));
    return qs * p_pa / fmax(p_pa - (1.0 - EPS_MW) * es, 100.0) * dlne;
}

__device__ inline void eval_cell_thermo(
    const NhGridConfig& cfg,
    const DeviceStaticGrid& g,
    int32_t k, int32_t col, double hs,
    const float* thp, const float* pip,
    const float* qv, const float* qc, const float* qr,
    const float* qi, const float* qs, const float* qg,
    double& z_s, double& dz_k,
    double& thb, double& pib,
    double& th_tot, double& pi_tot,
    double& t_k, double& p_pa, double& rho
) {
    const size_t n2 = static_cast<size_t>(cfg.ny) * cfg.nx;
    const size_t idx = static_cast<size_t>(k) * n2 + col;
    z_s = g.zeta_s[k] + g.b_s[k] * hs;
    dz_k = fmax(1.0, (1.0 + g.db_s[k] * hs) * g.dzeta[k]);
    double dthdz, rhob, cs2b;
    eval_ref_state(z_s, cfg.theta_sfc_k, cfg.p_sfc_pa, cfg.n_ref_s, thb, pib, dthdz, rhob, cs2b);
    th_tot = fmax(50.0, thb + static_cast<double>(thp[idx]));
    pi_tot = fmax(1e-4, pib + static_cast<double>(pip[idx]));
    t_k = fmax(140.0, th_tot * pi_tot);
    p_pa = P0_PA * pow(pi_tot, 1.0 / KAPPA);
    double qv_v = 0.0, qhyd = 0.0;
    if (cfg.has_moist) {
        if (qv) qv_v = fmax(0.0, static_cast<double>(qv[idx]));
        if (qc) qhyd += fmax(0.0, static_cast<double>(qc[idx]));
        if (qr) qhyd += fmax(0.0, static_cast<double>(qr[idx]));
        if (qi) qhyd += fmax(0.0, static_cast<double>(qi[idx]));
        if (qs) qhyd += fmax(0.0, static_cast<double>(qs[idx]));
        if (qg) qhyd += fmax(0.0, static_cast<double>(qg[idx]));
    }
    const double tv = t_k * fmax(0.5, 1.0 + EPS_V * qv_v - qhyd);
    rho = fmax(1e-5, p_pa / (R_D * tv));
}

// ===========================================================================
// PILLAR 3 (Radiation): ECMWF ecRad Delta-Eddington + Meador-Weaver + Adding
// ===========================================================================

__device__ inline double planck_frac_above_nu(double nu_cm, double t_k) {
    // Widger & Woodall (1976) normalized Planck integral fraction for wavenumber > nu_cm [cm^-1]
    // c2 = h*c/k_B = 1.438775 cm*K
    const double x = 1.438775 * nu_cm / fmax(t_k, 140.0);
    if (x <= 1e-4) return 1.0;
    if (x >= 45.0) return 0.0;
    double sum = 0.0;
    const double x2 = x * x, x3 = x2 * x;
    for (int n = 1; n <= 6; ++n) {
        const double dn = static_cast<double>(n);
        const double inv_n = 1.0 / dn;
        sum += exp(-dn * x) * inv_n * (x3 + 3.0 * x2 * inv_n + 6.0 * x * inv_n * inv_n + 6.0 * inv_n * inv_n * inv_n);
    }
    // 15 / pi^4 = 0.15398973382026508
    return fmin(1.0, fmax(0.0, 0.15398973382026508 * sum));
}

__device__ inline double lw_band_planck_flux(int band, double t_k) {
    // Partition sigma * T^4 across 3 non-overlapping energy fractions summing to 1.0:
    //   Band 0: Far-IR (<580 cm^-1) + High-IR (>1250 cm^-1) H2O rotational / vib-rot
    //   Band 1: CO2 15um + H2O wing (580..800 cm^-1)
    //   Band 2: Atmospheric window (800..1250 cm^-1)
    const double b_tot = STEFAN_BOLTZMANN * pow(fmax(t_k, 140.0), 4.0);
    const double p580  = planck_frac_above_nu(580.0, t_k);
    const double p800  = planck_frac_above_nu(800.0, t_k);
    const double p1250 = planck_frac_above_nu(1250.0, t_k);
    const double f1 = fmax(0.0, p580 - p800);
    const double f2 = fmax(0.0, p800 - p1250);
    const double f0 = fmax(0.0, 1.0 - f1 - f2);
    if (band == 0) return b_tot * f0;
    if (band == 1) return b_tot * f1;
    return b_tot * f2;
}

__device__ inline void sw_layer_optics(
    int band, double p_pa, double t_k, double dz, double rho,
    double qv, double qc, double qr, double qi, double qs,
    double n_drop_m3, double co2_ppm,
    double mu0, double& r_diff, double& t_diff, double& r_dir, double& t_dir_diff, double& t_dir_dir
) {
    const double dp = rho * GRAVITY * dz;
    const double dm_air = dp / GRAVITY;
    const double dm_wv = fmax(0.0, qv) * dm_air;
    const double p_scale = pow(fmin(fmax(p_pa / P0_PA, 1e-4), 1.2), 0.85) * sqrt(273.15 / fmax(t_k, 150.0));
    const double dm_wv_p = dm_wv * p_scale;

    double od_gas = 0.0, od_sca_ray = 0.0, od_cld = 0.0, ssa_cld = 0.999, g_cld = 0.85;
    if (band == 0) {
        // UV/Vis (~0.2 - 0.7 um): Rayleigh + stratospheric O3 peak (~30 hPa)
        od_sca_ray = 0.145 * (dp / 101325.0);
        const double ln_p = log(fmax(p_pa, 50.0) / 3000.0);
        const double o3_prof = exp(-0.5 * ln_p * ln_p / (1.1 * 1.1));
        od_gas = 0.025 * (dp / 101325.0) + 0.032 * o3_prof * (dp / 8000.0);
    } else {
        // NIR (>0.7 um): H2O + CO2 + O2 absorption + weak Rayleigh
        od_sca_ray = 0.012 * (dp / 101325.0);
        od_gas = 0.0085 * dm_wv_p + 0.00018 * (co2_ppm / 425.0) * dm_air * p_scale;
    }

    const double lwc = rho * fmax(0.0, qc + 0.15 * qr);
    const double iwc = rho * fmax(0.0, qi + 0.35 * qs);
    double tau_tot = od_gas + od_sca_ray;
    double sca_tot = od_sca_ray;
    double g_sca_tot = 0.0;

    if (lwc > 1e-8) {
        // MMM mp_wsm6_effectRad / Martin (1994) droplet effective radius [m]
        const double re_l = fmin(50.0e-6, fmax(3.0e-6,
            cbrt(3.0 * lwc / (4.0 * PI_CONST * 1000.0 * 0.8 * fmax(n_drop_m3, 2e7))) * 1.07));
        const double lwp = lwc * dz;
        const double tau_l = 1.5 * lwp / (1000.0 * re_l) * (1.0 + 1.1e-6 / re_l);
        const double ssa_l = (band == 0) ? 0.9999 : fmin(0.998, fmax(0.955, 0.993 - 120.0 * re_l));
        const double g_l = (band == 0) ? 0.855 : 0.820;
        od_cld += tau_l;
        ssa_cld = ssa_l;
        g_cld = g_l;
        tau_tot += tau_l;
        sca_tot += tau_l * ssa_l;
        g_sca_tot += tau_l * ssa_l * g_l;
    }
    if (iwc > 1e-8) {
        // Fu (1996) / ecRad ice effective radius [um]
        const double re_um = fmin(100.0, fmax(8.0, 28.0 + 18.0 * pow(iwc * 1000.0, 0.25)));
        const double iwp = iwc * dz;
        const double tau_i = iwp * fmax(0.002, -0.0003 + 2.52 / re_um) * 1000.0;
        const double ssa_i = (band == 0) ? 0.9995 : fmin(0.995, fmax(0.94, 0.990 - 0.0004 * re_um));
        const double g_i = fmin(0.88, fmax(0.75, 0.77 + 0.0005 * re_um));
        tau_tot += tau_i;
        sca_tot += tau_i * ssa_i;
        g_sca_tot += tau_i * ssa_i * g_i;
    }
    (void)od_cld; (void)ssa_cld; (void)g_cld;

    if (tau_tot < 1e-9) {
        r_diff = 0.0; t_diff = 1.0; r_dir = 0.0; t_dir_diff = 0.0; t_dir_dir = 1.0;
        return;
    }

    const double ssa = fmin(0.999999, sca_tot / tau_tot);
    const double asym = (sca_tot > 1e-12) ? fmin(0.95, fmax(0.0, g_sca_tot / sca_tot)) : 0.0;

    // Delta-Eddington scaling (ecrad/radiation/radiation_delta_eddington.h)
    const double f = asym * asym;
    const double denom = fmax(1.0 - ssa * f, 1e-12);
    const double od = tau_tot * denom;
    const double ssa_s = fmin(0.999999, ssa * (1.0 - f) / denom);
    const double g_s = asym / (1.0 + asym);

    // Zdunkowski PIFM two-stream coefficients (ecrad/radiation_two_stream.F90)
    const double g1 = (8.0 - ssa_s * (5.0 + 3.0 * g_s)) * 0.25;
    const double g2 = 3.0 * ssa_s * (1.0 - g_s) * 0.25;
    const double g3 = (2.0 - 3.0 * g_s * mu0) * 0.25;
    const double g4 = 1.0 - g3;
    const double a1 = g1 * g4 + g2 * g3;
    const double a2 = g1 * g3 + g2 * g4;
    const double k_exp = sqrt(fmax((g1 - g2) * (g1 + g2), 1e-12));

    double mu0_loc = mu0;
    if (fabs(1.0 - k_exp * mu0_loc) < 1e-6) {
        mu0_loc *= (1.0 - 1e-5);
    }
    const double od_over_mu0 = fmin(50.0, fmax(0.0, od / mu0_loc));
    const double exp0 = exp(-od_over_mu0);
    t_dir_dir = exp0;

    const double k_mu0 = k_exp * mu0_loc;
    const double k_g3 = k_exp * g3;
    const double k_g4 = k_exp * g4;
    const double exp1 = exp(-fmin(50.0, k_exp * od));
    const double exp2 = exp1 * exp1;
    const double k2e = 2.0 * k_exp * exp1;
    const double rtf = 1.0 / fmax(k_exp + g1 + (k_exp - g1) * exp2, 1e-15);

    r_diff = fmin(1.0, fmax(0.0, g2 * (1.0 - exp2) * rtf));
    t_diff = fmin(1.0 - r_diff, fmax(0.0, k2e * rtf));

    const double rtf_dir = mu0_loc * ssa_s * rtf / (1.0 - k_mu0 * k_mu0);
    r_dir = rtf_dir * ((1.0 - k_mu0) * (a2 + k_g3) - (1.0 + k_mu0) * (a2 - k_g3) * exp2 - k2e * (g3 - a2 * mu0_loc) * exp0);
    t_dir_diff = rtf_dir * (k2e * (g4 + a1 * mu0_loc) - exp0 * ((1.0 + k_mu0) * (a1 + k_g4) - (1.0 - k_mu0) * (a1 - k_g4) * exp2));
    r_dir = fmin(mu0_loc, fmax(0.0, r_dir));
    t_dir_diff = fmin(mu0_loc - r_dir, fmax(0.0, t_dir_diff));
}

__device__ inline void lw_layer_optics(
    int band, double p_pa, double t_k, double t_bot, double t_top, double dz, double rho,
    double qv, double qc, double qr, double qi, double qs,
    double n_drop_m3, double co2_ppm,
    double& ref, double& trans, double& src_up, double& src_dn
) {
    const double dp = rho * GRAVITY * dz;
    const double dm_air = dp / GRAVITY;
    const double dm_wv = fmax(0.0, qv) * dm_air;
    const double p_scale = pow(fmin(fmax(p_pa / P0_PA, 1e-4), 1.2), 0.80) * sqrt(273.15 / fmax(t_k, 150.0));
    const double dm_wv_p = dm_wv * p_scale;
    const double e_vap = fmax(0.0, p_pa * qv / (EPS_MW + (1.0 - EPS_MW) * qv));

    double od_abs = 0.0, od_sca = 0.0, g_sca = 0.0;
    if (band == 0) {
        // H2O rotational / high-IR absorption
        od_abs = 0.42 * dm_wv_p + 0.015 * dm_wv * (e_vap / 1000.0 + 0.25);
    } else if (band == 1) {
        // CO2 15 um + H2O wing (column clear-sky optical depth ~3-6, scaling with CO2)
        od_abs = 0.00085 * (co2_ppm / 425.0) * dm_air * pow(fmax(p_pa / P0_PA, 1e-4), 0.70)
               + 0.035 * dm_wv_p;
    } else {
        // Atmospheric window 8-12 um + water vapor continuum (clear-sky tau ~0.18-0.30)
        od_abs = 0.0065 * dm_wv * (e_vap / 1000.0 + 0.25)
               + 0.000015 * (co2_ppm / 425.0) * dm_air * p_scale;
    }

    const double lwc = rho * fmax(0.0, qc + 0.25 * qr);
    const double iwc = rho * fmax(0.0, qi + 0.40 * qs);
    if (lwc > 1e-8) {
        const double lwp = lwc * dz;
        const double re_um = fmin(40.0, fmax(4.0,
            1e6 * cbrt(3.0 * lwc / (4.0 * PI_CONST * 1000.0 * 0.8 * fmax(n_drop_m3, 2e7))) * 1.07));
        // SOCRATES LW mass extinction (~95..140 m^2/kg) + moderate forward scattering
        const double od_l = lwp * (90.0 + 280.0 / re_um);
        const double ssa_l = (band == 2) ? 0.30 : 0.18;
        od_abs += od_l * (1.0 - ssa_l);
        od_sca += od_l * ssa_l;
        g_sca += od_l * ssa_l * 0.72;
    }
    if (iwc > 1e-8) {
        const double iwp = iwc * dz;
        const double re_um = fmin(100.0, fmax(8.0, 28.0 + 18.0 * pow(iwc * 1000.0, 0.25)));
        const double od_i = iwp * (10.0 + 1850.0 / re_um);
        const double ssa_i = 0.22;
        od_abs += od_i * (1.0 - ssa_i);
        od_sca += od_i * ssa_i;
        g_sca += od_i * ssa_i * 0.76;
    }

    const double od = od_abs + od_sca;
    const double ssa = (od > 1e-12) ? fmin(0.99, od_sca / od) : 0.0;
    const double asym = (od_sca > 1e-12) ? fmin(0.95, g_sca / od_sca) : 0.0;

    // Elsasser diffusivity D = 1.66 (ecrad/radiation_two_stream.F90)
    constexpr double D_LW = 1.66;
    const double g1 = D_LW * (1.0 - 0.5 * ssa * (1.0 + asym));
    const double g2 = D_LW * 0.5 * ssa * (1.0 - asym);
    const double k_exp = sqrt(fmax((g1 - g2) * (g1 + g2), 1e-12));

    const double p_top = lw_band_planck_flux(band, t_top);
    const double p_bot = lw_band_planck_flux(band, t_bot);

    if (od > 1.0e-3) {
        const double exp1 = exp(-fmin(50.0, k_exp * od));
        const double exp2 = exp1 * exp1;
        const double rtf = 1.0 / fmax(k_exp + g1 + (k_exp - g1) * exp2, 1e-15);
        ref = g2 * (1.0 - exp2) * rtf;
        trans = 2.0 * k_exp * exp1 * rtf;
        // Stackhouse & Stephens (JAS 1991) Eqs 5 & 12 (linear-in-tau Planck)
        const double coeff = (p_bot - p_top) / fmax(od * (g1 + g2), 1e-12);
        const double c_ut = coeff + p_top;
        const double c_ub = coeff + p_bot;
        const double c_dt = -coeff + p_top;
        const double c_db = -coeff + p_bot;
        src_up = fmax(0.0, c_ut - ref * c_dt - trans * c_ub);
        src_dn = fmax(0.0, c_db - ref * c_ub - trans * c_dt);
    } else {
        ref = g2 * od;
        trans = (1.0 - k_exp * od) / (1.0 + od * (g1 - k_exp));
        trans = fmin(1.0 - ref, fmax(0.0, trans));
        const double s_iso = fmax(0.0, 1.0 - ref - trans) * 0.5 * (p_top + p_bot);
        src_up = s_iso;
        src_dn = s_iso;
    }
}

__global__ void nh_radiation_ecrad_kernel(
    NhGridConfig cfg,
    NhPhysicsConfig pcfg,
    DeviceStaticGrid g,
    const float* thp, const float* pip,
    const float* qv, const float* qc, const float* qr, const float* qi, const float* qs, const float* qg,
    const float* lat_deg, const float* lon_deg, const float* lsm,
    const float* albedo_base, const float* snow_swe, const float* lake_mask, const float* lake_hice,
    const float* t_skin,
    float* rad_dth_dt, float* sw_down_out, float* lw_down_out,
    float* scratch_f, double* tri_c, double* tri_d
) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= cfg.ny || i >= cfg.nx) return;

    const int32_t nz = cfg.nz;
    const size_t n2 = static_cast<size_t>(cfg.ny) * cfg.nx;
    const size_t col = static_cast<size_t>(j) * cfg.nx + i;
    const size_t str3 = n2;

    const double hs = g.hs[col];
    const double lat_v = lat_deg ? static_cast<double>(lat_deg[col]) : 52.1;
    const double lon_v = lon_deg ? static_cast<double>(lon_deg[col]) : 5.18;
    const double lsm_v = lsm ? static_cast<double>(lsm[col]) : 1.0;
    const double snow_v = snow_swe ? static_cast<double>(snow_swe[col]) : 0.0;
    const double lake_v = lake_mask ? static_cast<double>(lake_mask[col]) : 0.0;
    const double hice_v = lake_hice ? static_cast<double>(lake_hice[col]) : 0.0;
    const double alb0 = albedo_base ? static_cast<double>(albedo_base[col]) : ((lsm_v >= 0.5) ? 0.20 : 0.06);
    const double n_drop = (lsm_v >= 0.5) ? pcfg.cloud_num_land : pcfg.cloud_num_sea;

    double alb_sfc = alb0;
    if (lsm_v >= 0.5 && snow_v > 0.0) {
        const double sf = fmin(1.0, snow_v / 15.0);
        alb_sfc = (1.0 - sf) * alb0 + sf * 0.72;
    } else if (lsm_v < 0.5 && lake_v >= 0.5 && hice_v > 1e-4) {
        alb_sfc = 0.55;
    }
    const double emis_sfc = (lsm_v >= 0.5) ? ((snow_v > 1.0) ? 0.985 : 0.96) : 0.97;
    const double t_sk = t_skin ? fmin(fmax(static_cast<double>(t_skin[col]), 180.0), 340.0) : 285.0;

    for (int32_t k = 0; k < nz; ++k) {
        rad_dth_dt[static_cast<size_t>(k) * str3 + col] = 0.0f;
    }

    // Astronomical geometry (Spencer 1971)
    const double gamma = 2.0 * PI_CONST * (pcfg.julian_day - 1.0) / 365.0;
    const double decl = 0.006918 - 0.399912 * cos(gamma) + 0.070257 * sin(gamma)
                      - 0.006758 * cos(2.0 * gamma) + 0.000907 * sin(2.0 * gamma)
                      - 0.002697 * cos(3.0 * gamma) + 0.001480 * sin(3.0 * gamma);
    const double eqt = 0.000075 + 0.001868 * cos(gamma) - 0.032077 * sin(gamma)
                     - 0.014615 * cos(2.0 * gamma) - 0.040849 * sin(2.0 * gamma);
    const double hr_ang = (pcfg.utc_seconds / 86400.0 - 0.5) * 2.0 * PI_CONST + lon_v * (PI_CONST / 180.0) + eqt;
    const double lat_r = lat_v * (PI_CONST / 180.0);
    const double mu0 = sin(lat_r) * sin(decl) + cos(lat_r) * cos(decl) * cos(hr_ang);
    const double s0_toa = 1361.0 * (1.000110 + 0.034221 * cos(gamma) + 0.001280 * sin(gamma)
                                  + 0.000719 * cos(2.0 * gamma) + 0.000077 * sin(2.0 * gamma));

    double sw_dn_sfc = 0.0;
    if (mu0 > 0.01) {
        const double sw_frac[2] = {0.46, 0.54};
        for (int b = 0; b < 2; ++b) {
            const double s0_b = s0_toa * sw_frac[b];
            // Top-down direct beam sweep (store direct flux entering layer k top in scratch_f[k])
            double f_dir = s0_b;
            for (int32_t k = nz - 1; k >= 0; --k) {
                const size_t idx = static_cast<size_t>(k) * str3 + col;
                double z_s, dz_k, thb, pib, th_t, pi_t, t_k, p_k, rho;
                eval_cell_thermo(cfg, g, k, col, hs, thp, pip, qv, qc, qr, qi, qs, qg,
                                 z_s, dz_k, thb, pib, th_t, pi_t, t_k, p_k, rho);
                const double qv_v = (cfg.has_moist && qv) ? fmax(0.0, static_cast<double>(qv[idx])) : 0.0;
                const double qc_v = (cfg.has_moist && qc) ? fmax(0.0, static_cast<double>(qc[idx])) : 0.0;
                const double qr_v = (cfg.has_moist && qr) ? fmax(0.0, static_cast<double>(qr[idx])) : 0.0;
                const double qi_v = (cfg.has_moist && qi) ? fmax(0.0, static_cast<double>(qi[idx])) : 0.0;
                const double qs_v = (cfg.has_moist && qs) ? fmax(0.0, static_cast<double>(qs[idx])) : 0.0;

                scratch_f[idx] = static_cast<float>(f_dir);
                double r_d, t_d, r_dir, t_dd, t_dir;
                sw_layer_optics(b, p_k, t_k, dz_k, rho, qv_v, qc_v, qr_v, qi_v, qs_v,
                                n_drop, pcfg.co2_ppm, mu0, r_d, t_d, r_dir, t_dd, t_dir);
                f_dir *= t_dir;
            }
            const double f_dir_sfc = f_dir;

            // Bottom-up Adding pass (k = 0..nz-1)
            tri_c[col] = alb_sfc;
            tri_d[col] = alb_sfc * f_dir_sfc * mu0;
            for (int32_t k = 0; k < nz; ++k) {
                const size_t idx = static_cast<size_t>(k) * str3 + col;
                double z_s, dz_k, thb, pib, th_t, pi_t, t_k, p_k, rho;
                eval_cell_thermo(cfg, g, k, col, hs, thp, pip, qv, qc, qr, qi, qs, qg,
                                 z_s, dz_k, thb, pib, th_t, pi_t, t_k, p_k, rho);
                const double qv_v = (cfg.has_moist && qv) ? fmax(0.0, static_cast<double>(qv[idx])) : 0.0;
                const double qc_v = (cfg.has_moist && qc) ? fmax(0.0, static_cast<double>(qc[idx])) : 0.0;
                const double qr_v = (cfg.has_moist && qr) ? fmax(0.0, static_cast<double>(qr[idx])) : 0.0;
                const double qi_v = (cfg.has_moist && qi) ? fmax(0.0, static_cast<double>(qi[idx])) : 0.0;
                const double qs_v = (cfg.has_moist && qs) ? fmax(0.0, static_cast<double>(qs[idx])) : 0.0;

                double r_d, t_d, r_dir, t_dd, t_dir;
                sw_layer_optics(b, p_k, t_k, dz_k, rho, qv_v, qc_v, qr_v, qi_v, qs_v,
                                n_drop, pcfg.co2_ppm, mu0, r_d, t_d, r_dir, t_dd, t_dir);
                const double f_dir_in = static_cast<double>(scratch_f[idx]);
                const double r_b = tri_c[static_cast<size_t>(k) * str3 + col];
                const double s_b = tri_d[static_cast<size_t>(k) * str3 + col];
                const double inv = 1.0 / fmax(1.0 - r_b * r_d, 1e-12);
                tri_c[static_cast<size_t>(k + 1) * str3 + col] = r_d + t_d * t_d * r_b * inv;
                tri_d[static_cast<size_t>(k + 1) * str3 + col] =
                    r_dir * f_dir_in + t_d * (s_b + r_b * t_dd * f_dir_in) * inv;
            }

            // Top-down diffuse sweep & radiative heating accumulation
            double f_dn_diff = 0.0;
            double f_up_top = tri_d[static_cast<size_t>(nz) * str3 + col];
            double f_net_above = s0_b * mu0 - f_up_top;

            for (int32_t k = nz - 1; k >= 0; --k) {
                const size_t idx = static_cast<size_t>(k) * str3 + col;
                double z_s, dz_k, thb, pib, th_t, pi_t, t_k, p_k, rho;
                eval_cell_thermo(cfg, g, k, col, hs, thp, pip, qv, qc, qr, qi, qs, qg,
                                 z_s, dz_k, thb, pib, th_t, pi_t, t_k, p_k, rho);
                const double qv_v = (cfg.has_moist && qv) ? fmax(0.0, static_cast<double>(qv[idx])) : 0.0;
                const double qc_v = (cfg.has_moist && qc) ? fmax(0.0, static_cast<double>(qc[idx])) : 0.0;
                const double qr_v = (cfg.has_moist && qr) ? fmax(0.0, static_cast<double>(qr[idx])) : 0.0;
                const double qi_v = (cfg.has_moist && qi) ? fmax(0.0, static_cast<double>(qi[idx])) : 0.0;
                const double qs_v = (cfg.has_moist && qs) ? fmax(0.0, static_cast<double>(qs[idx])) : 0.0;

                double r_d, t_d, r_dir, t_dd, t_dir;
                sw_layer_optics(b, p_k, t_k, dz_k, rho, qv_v, qc_v, qr_v, qi_v, qs_v,
                                n_drop, pcfg.co2_ppm, mu0, r_d, t_d, r_dir, t_dd, t_dir);
                const double f_dir_in = static_cast<double>(scratch_f[idx]);
                const double r_b = tri_c[idx];
                const double s_b = tri_d[idx];
                const double inv = 1.0 / fmax(1.0 - r_b * r_d, 1e-12);
                f_dn_diff = (t_d * f_dn_diff + r_d * s_b + t_dd * f_dir_in) * inv;
                const double f_up_below = r_b * f_dn_diff + s_b;
                const double f_dir_below = f_dir_in * t_dir;
                const double f_net_below = f_dn_diff + f_dir_below * mu0 - f_up_below;
                const double dth_dt = (f_net_above - f_net_below) / (rho * CP_D * dz_k * pi_t);
                rad_dth_dt[idx] = static_cast<float>(static_cast<double>(rad_dth_dt[idx]) + dth_dt);
                f_net_above = f_net_below;
            }
            sw_dn_sfc += f_dn_diff + f_dir_sfc * mu0;
        }
    }

    // Longwave 3-band Linear-Planck Adding Method
    double lw_dn_sfc = 0.0;
    for (int b = 0; b < 3; ++b) {
        tri_c[col] = 1.0 - emis_sfc;
        tri_d[col] = emis_sfc * lw_band_planck_flux(b, t_sk);

        double t_prev = t_sk;
        for (int32_t k = 0; k < nz; ++k) {
            const size_t idx = static_cast<size_t>(k) * str3 + col;
            double z_s, dz_k, thb, pib, th_t, pi_t, t_k, p_k, rho;
            eval_cell_thermo(cfg, g, k, col, hs, thp, pip, qv, qc, qr, qi, qs, qg,
                             z_s, dz_k, thb, pib, th_t, pi_t, t_k, p_k, rho);
            double t_nxt = t_k;
            if (k + 1 < nz) {
                double z_n, dz_n, thbn, pibn, thn, pin, tn, pn, rhon;
                eval_cell_thermo(cfg, g, k + 1, col, hs, thp, pip, qv, qc, qr, qi, qs, qg,
                                 z_n, dz_n, thbn, pibn, thn, pin, tn, pn, rhon);
                t_nxt = tn;
            }
            const double t_bot_if = (k == 0) ? t_k : 0.5 * (t_prev + t_k);
            const double t_top_if = (k + 1 == nz) ? t_k : 0.5 * (t_k + t_nxt);
            t_prev = t_k;

            const double qv_v = (cfg.has_moist && qv) ? fmax(0.0, static_cast<double>(qv[idx])) : 0.0;
            const double qc_v = (cfg.has_moist && qc) ? fmax(0.0, static_cast<double>(qc[idx])) : 0.0;
            const double qr_v = (cfg.has_moist && qr) ? fmax(0.0, static_cast<double>(qr[idx])) : 0.0;
            const double qi_v = (cfg.has_moist && qi) ? fmax(0.0, static_cast<double>(qi[idx])) : 0.0;
            const double qs_v = (cfg.has_moist && qs) ? fmax(0.0, static_cast<double>(qs[idx])) : 0.0;

            double r_d, t_d, s_up_k, s_dn_k;
            lw_layer_optics(b, p_k, t_k, t_bot_if, t_top_if, dz_k, rho,
                            qv_v, qc_v, qr_v, qi_v, qs_v, n_drop, pcfg.co2_ppm,
                            r_d, t_d, s_up_k, s_dn_k);

            const double r_b = tri_c[idx];
            const double s_b = tri_d[idx];
            const double inv = 1.0 / fmax(1.0 - r_b * r_d, 1e-12);
            tri_c[static_cast<size_t>(k + 1) * str3 + col] = r_d + t_d * t_d * r_b * inv;
            tri_d[static_cast<size_t>(k + 1) * str3 + col] = s_up_k + t_d * (s_b + r_b * s_dn_k) * inv;
        }

        double f_dn = 0.0; // TOA downward LW = 0
        double f_up_top = tri_d[static_cast<size_t>(nz) * str3 + col];
        double f_net_above = f_dn - f_up_top;

        for (int32_t k = nz - 1; k >= 0; --k) {
            const size_t idx = static_cast<size_t>(k) * str3 + col;
            double z_s, dz_k, thb, pib, th_t, pi_t, t_k, p_k, rho;
            eval_cell_thermo(cfg, g, k, col, hs, thp, pip, qv, qc, qr, qi, qs, qg,
                             z_s, dz_k, thb, pib, th_t, pi_t, t_k, p_k, rho);
            double t_bot_if = t_k, t_top_if = t_k;
            if (k > 0) {
                double z_p, dz_p, thbp, pibp, thp_v, pip_v, tp, pp, rhop;
                eval_cell_thermo(cfg, g, k - 1, col, hs, thp, pip, qv, qc, qr, qi, qs, qg,
                                 z_p, dz_p, thbp, pibp, thp_v, pip_v, tp, pp, rhop);
                t_bot_if = 0.5 * (tp + t_k);
            }
            if (k + 1 < nz) {
                double z_n, dz_n, thbn, pibn, thn, pin, tn, pn, rhon;
                eval_cell_thermo(cfg, g, k + 1, col, hs, thp, pip, qv, qc, qr, qi, qs, qg,
                                 z_n, dz_n, thbn, pibn, thn, pin, tn, pn, rhon);
                t_top_if = 0.5 * (t_k + tn);
            }
            const double qv_v = (cfg.has_moist && qv) ? fmax(0.0, static_cast<double>(qv[idx])) : 0.0;
            const double qc_v = (cfg.has_moist && qc) ? fmax(0.0, static_cast<double>(qc[idx])) : 0.0;
            const double qr_v = (cfg.has_moist && qr) ? fmax(0.0, static_cast<double>(qr[idx])) : 0.0;
            const double qi_v = (cfg.has_moist && qi) ? fmax(0.0, static_cast<double>(qi[idx])) : 0.0;
            const double qs_v = (cfg.has_moist && qs) ? fmax(0.0, static_cast<double>(qs[idx])) : 0.0;

            double r_d, t_d, s_up_k, s_dn_k;
            lw_layer_optics(b, p_k, t_k, t_bot_if, t_top_if, dz_k, rho,
                            qv_v, qc_v, qr_v, qi_v, qs_v, n_drop, pcfg.co2_ppm,
                            r_d, t_d, s_up_k, s_dn_k);

            const double r_b = tri_c[idx];
            const double s_b = tri_d[idx];
            const double inv = 1.0 / fmax(1.0 - r_b * r_d, 1e-12);
            f_dn = (t_d * f_dn + r_d * s_b + s_dn_k) * inv;
            const double f_up_below = r_b * f_dn + s_b;
            const double f_net_below = f_dn - f_up_below;
            const double dth_dt = (f_net_above - f_net_below) / (rho * CP_D * dz_k * pi_t);
            rad_dth_dt[idx] = static_cast<float>(static_cast<double>(rad_dth_dt[idx]) + dth_dt);
            f_net_above = f_net_below;
        }
        lw_dn_sfc += f_dn;
    }

    if (sw_down_out) sw_down_out[col] = static_cast<float>(sw_dn_sfc);
    if (lw_down_out) lw_down_out[col] = static_cast<float>(lw_dn_sfc);
}

__global__ void nh_apply_rad_kernel(
    NhGridConfig cfg, double dt_s, const float* rad_dth_dt, float* thp
) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= cfg.ny || i >= cfg.nx) return;
    const size_t str = static_cast<size_t>(cfg.ny) * cfg.nx;
    const size_t col = static_cast<size_t>(j) * cfg.nx + i;
    for (int32_t k = 0; k < cfg.nz; ++k) {
        const size_t idx = static_cast<size_t>(k) * str + col;
        thp[idx] = static_cast<float>(static_cast<double>(thp[idx]) + dt_s * static_cast<double>(rad_dth_dt[idx]));
    }
}

// ===========================================================================
// PILLAR 4 (Surface Layer, Land LSM, Ocean COARE 3.6 + VOSKIN, FLake)
// ===========================================================================

__device__ inline double psim_jimenez(double zol) {
    // Jimenez et al. (2012) / MMM sf_sfclayrev.F90 (BSD-3-Clause)
    if (zol <= 0.0) {
        const double z = fmax(zol, -15.0);
        const double x = pow(1.0 - 16.0 * z, 0.25);
        const double psim_k = 2.0 * log(0.5 * (1.0 + x)) + log(0.5 * (1.0 + x * x))
                            - 2.0 * atan(x) + 0.5 * PI_CONST;
        const double y = cbrt(1.0 - 10.15 * z);
        const double psim_c = 1.5 * log((y * y + y + 1.0) / 3.0)
                            - sqrt(3.0) * atan((2.0 * y + 1.0) / sqrt(3.0)) + PI_CONST / sqrt(3.0);
        const double f = (z * z) / (1.0 + z * z);
        return (1.0 - f) * psim_k + f * psim_c;
    } else {
        const double z = fmin(zol, 10.0);
        return -6.1 * log(z + pow(1.0 + pow(z, 2.5), 0.4));
    }
}

__device__ inline double psih_jimenez(double zol) {
    if (zol <= 0.0) {
        const double z = fmax(zol, -15.0);
        const double x = pow(1.0 - 16.0 * z, 0.25);
        const double psih_k = 2.0 * log(0.5 * (1.0 + x * x));
        const double y = cbrt(1.0 - 34.15 * z);
        const double psih_c = 1.5 * log((y * y + y + 1.0) / 3.0)
                            - sqrt(3.0) * atan((2.0 * y + 1.0) / sqrt(3.0)) + PI_CONST / sqrt(3.0);
        const double f = (z * z) / (1.0 + z * z);
        return (1.0 - f) * psih_k + f * psih_c;
    } else {
        const double z = fmin(zol, 10.0);
        return -5.3 * log(z + pow(1.0 + pow(z, 1.1), 1.0 / 1.1));
    }
}

__device__ inline void solve_mo_sfclayrev(
    double za, double u_spd, double thv_a, double thv_s,
    double z0m, double z0h, double& ustar, double& zol_out, double& cm_out, double& ch_out
) {
    const double dthv = thv_a - thv_s;
    const double wspd = fmax(u_spd, 0.3);
    const double rib = GRAVITY * dthv * za / (fmax(0.5 * (thv_a + thv_s), 150.0) * wspd * wspd);
    const double ln_zm = log(fmax(za / fmax(z0m, 1e-5), 1.5));
    const double ln_zh = log(fmax(za / fmax(z0h, 1e-6), 1.5));

    double zol = fmin(fmax(rib * ln_zm, -10.0), 5.0);
    for (int it = 0; it < 10; ++it) {
        const double denom_m = fmax(ln_zm - psim_jimenez(zol) + psim_jimenez(zol * z0m / za), 0.5);
        const double denom_h = fmax(ln_zh - psih_jimenez(zol) + psih_jimenez(zol * z0h / za), 0.5);
        const double zol_new = rib * (denom_m * denom_m) / denom_h;
        zol = fmin(fmax(0.5 * zol + 0.5 * zol_new, -12.0), 8.0);
    }
    const double denom_m = fmax(ln_zm - psim_jimenez(zol) + psim_jimenez(zol * z0m / za), 0.5);
    const double denom_h = fmax(ln_zh - psih_jimenez(zol) + psih_jimenez(zol * z0h / za), 0.5);
    ustar = fmax(0.02, KARMAN * wspd / denom_m);
    cm_out = (KARMAN / denom_m) * (KARMAN / denom_m);
    ch_out = (KARMAN / denom_m) * (KARMAN / denom_h);
    zol_out = zol;
}

__device__ inline void solve_coare36(
    double za, double u_spd, double t_a, double th_a, double qv_a, double p_a,
    double t_s, double th_s, double p_s, double bathy_depth_m,
    double& ustar, double& z0m, double& z0h, double& zol_out, double& cm_out, double& ch_out,
    double& q_sea_sat, double& wspd_out
) {
    // COARE 3.6 bulk algorithm (Fairall et al. 2003/2019; Edson et al. 2013; 0.98 sea-salt factor)
    q_sea_sat = 0.98 * qsat_water(t_s, p_s);
    const double thv_a = th_a * (1.0 + 0.608 * qv_a);
    const double thv_s = th_s * (1.0 + 0.608 * q_sea_sat);
    const double nu_a = 1.48e-5; // air kinematic viscosity

    double ug = 0.25;
    z0m = 1e-4;
    z0h = 1e-4;
    ustar = fmax(0.05, 0.035 * fmax(u_spd, 0.5));
    double zol = 0.0, cm = 1e-3, ch = 1e-3;

    for (int it = 0; it < 8; ++it) {
        const double wspd = sqrt(u_spd * u_spd + ug * ug);
        solve_mo_sfclayrev(za, wspd, thv_a, thv_s, z0m, z0h, ustar, zol, cm, ch);
        const double ln_10 = log(10.0 / fmax(z0m, 1e-6));
        const double u10n = ustar / KARMAN * ln_10;
        const double charn = fmin(0.028, fmax(0.0, 0.0017 * u10n - 0.005));
        double z0_wave = 0.0;
        if (bathy_depth_m > 0.5 && bathy_depth_m < 100.0) {
            // Shallow North Sea wave shoaling enhancement clamped cleanly
            const double ratio = pow(fmax(bathy_depth_m, 2.0) / 30.0, -0.8);
            z0_wave = fmin(0.05, 0.0008 * ratio * (u10n / 15.0) * (u10n / 15.0));
        }
        z0m = fmin(0.10, fmax(1e-5, charn * ustar * ustar / GRAVITY + 0.11 * nu_a / ustar + z0_wave));
        const double re_r = fmax(0.1, ustar * z0m / nu_a);
        z0h = fmin(1.6e-4, fmax(1e-6, 5.8e-5 * pow(re_r, -0.72)));

        // Convective gustiness Beta0 * w*, zi = 600 m (theta* = C_H / sqrt(C_M) * dtheta)
        const double inv_scm = 1.0 / sqrt(fmax(cm, 1e-6));
        const double qstar_est = (ch * inv_scm) * (qv_a - q_sea_sat);
        const double tstar_est = (ch * inv_scm) * (th_a - th_s);
        const double bf = -(GRAVITY / fmax(thv_a, 150.0)) * ustar * (tstar_est + 0.608 * th_a * qstar_est);
        ug = (bf > 0.0) ? fmax(0.25, 1.2 * cbrt(bf * 600.0)) : 0.20;
    }
    const double wspd = sqrt(u_spd * u_spd + ug * ug);
    solve_mo_sfclayrev(za, wspd, thv_a, thv_s, z0m, z0h, ustar, zol, cm, ch);
    zol_out = zol; cm_out = cm; ch_out = ch; wspd_out = wspd;
    (void)t_a; (void)p_a;
}

__device__ inline void solve_tridiag_4(
    const double a[4], const double b[4], const double c[4], const double d[4], double x[4]
) {
    double cp[4], dp[4];
    cp[0] = c[0] / b[0];
    dp[0] = d[0] / b[0];
    for (int m = 1; m < 4; ++m) {
        const double denom = b[m] - a[m] * cp[m - 1];
        cp[m] = (m < 3) ? (c[m] / denom) : 0.0;
        dp[m] = (d[m] - a[m] * dp[m - 1]) / denom;
    }
    x[3] = dp[3];
    for (int m = 2; m >= 0; --m) {
        x[m] = dp[m] - cp[m] * x[m + 1];
    }
}

__global__ void nh_surface_and_lsm_kernel(
    NhGridConfig cfg,
    NhPhysicsConfig pcfg,
    DeviceStaticGrid g,
    double dt_s,
    const float* u, const float* v, const float* thp, const float* pip,
    const float* qv, const float* qc, const float* qr, const float* qi, const float* qs, const float* qg,
    const float* lsm, const float* lake_mask, const float* soil_type, const float* bathy_depth_m,
    const float* z0m_base, const float* albedo_base, const float* veg_frac,
    const float* sw_down, const float* lw_down,
    const float* precip_liq_rate, const float* precip_ice_rate,
    float* t_skin, float* soil_t, float* soil_m, float* snow_swe,
    float* sst_bulk, float* dt_cool, float* dt_warm,
    float* lake_tbot, float* lake_hmld, float* lake_ctheta, float* lake_hice,
    float* shf, float* lhf, float* ustar, float* t2m, float* q2m, float* u10m, float* v10m,
    float* sfc_drag_coeff
) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= cfg.ny || i >= cfg.nx) return;

    const size_t n2 = static_cast<size_t>(cfg.ny) * cfg.nx;
    const size_t col = static_cast<size_t>(j) * cfg.nx + i;
    const double hs = g.hs[col];

    double z_0, dz_0, thb0, pib0, th_a, pi_a, t_a, p_a, rho_a;
    eval_cell_thermo(cfg, g, 0, col, hs, thp, pip, qv, qc, qr, qi, qs, qg,
                     z_0, dz_0, thb0, pib0, th_a, pi_a, t_a, p_a, rho_a);
    const double za = fmax(2.0, z_0 - hs);
    const double qv_a = (cfg.has_moist && qv) ? fmax(1e-6, static_cast<double>(qv[col])) : 1e-6;
    const double tv_a = t_a * (1.0 + EPS_V * qv_a);
    const double p_s = p_a * exp(GRAVITY * za / (R_D * tv_a));
    const double pi_s = pow(p_s / P0_PA, KAPPA);
    const double t_a_at_sfc = th_a * pi_s;

    const double u_c = 0.5 * (read_u(u, 0, j, i, cfg.nz, cfg.ny, cfg.nx, cfg.bc_x, cfg.bc_y)
                            + read_u(u, 0, j, i + 1, cfg.nz, cfg.ny, cfg.nx, cfg.bc_x, cfg.bc_y));
    const double v_c = 0.5 * (read_v(v, 0, j, i, cfg.nz, cfg.ny, cfg.nx, cfg.bc_x, cfg.bc_y)
                            + read_v(v, 0, j + 1, i, cfg.nz, cfg.ny, cfg.nx, cfg.bc_x, cfg.bc_y));
    const double u_spd = sqrt(u_c * u_c + v_c * v_c);
    const double wspd_eff = fmax(0.5, u_spd);

    const double sw_in = sw_down ? fmax(0.0, static_cast<double>(sw_down[col])) : 0.0;
    const double lw_in = lw_down ? fmax(50.0, static_cast<double>(lw_down[col])) : 300.0;
    const double p_liq = precip_liq_rate ? fmax(0.0, static_cast<double>(precip_liq_rate[col])) : 0.0;
    const double p_ice = precip_ice_rate ? fmax(0.0, static_cast<double>(precip_ice_rate[col])) : 0.0;

    const double lsm_v = lsm ? static_cast<double>(lsm[col]) : 1.0;
    const double lake_v = lake_mask ? static_cast<double>(lake_mask[col]) : 0.0;
    const double z0_base_v = z0m_base ? fmax(1e-4, static_cast<double>(z0m_base[col])) : ((lsm_v >= 0.5) ? 0.05 : 1e-4);
    const double alb_base_v = albedo_base ? fmin(0.85, fmax(0.04, static_cast<double>(albedo_base[col]))) : ((lsm_v >= 0.5) ? 0.20 : 0.06);
    const double veg_v = veg_frac ? fmin(1.0, fmax(0.0, static_cast<double>(veg_frac[col]))) : 0.60;
    const double depth_v = bathy_depth_m ? fmax(1.0, static_cast<double>(bathy_depth_m[col])) : ((lake_v >= 0.5) ? 4.5 : 40.0);

    double t_sk_new = t_a_at_sfc;
    double q_sk_eff = qv_a;
    double ustar_v = 0.15, zol_v = 0.0, cm_v = 1e-3, ch_v = 1e-3, z0m_v = z0_base_v, z0h_v = z0_base_v * 0.1;
    double h_flux = 0.0, le_flux = 0.0, wspd_drag = wspd_eff;

    if (lsm_v >= 0.5) {
        // -------------------------------------------------------------------
        // REGIME 1: LAND (sfclayrev + Newton Skin Balance + 4-Layer HTESSEL/Noah)
        // -------------------------------------------------------------------
        int stype = soil_type ? static_cast<int>(llround(static_cast<double>(soil_type[col]))) : 9;
        if (stype < 1) stype = 9;
        if (stype > 13) stype = 13;

        const double th_sat  = c_soil_table[stype][0];
        const double th_fc   = c_soil_table[stype][1];
        const double th_wilt = c_soil_table[stype][2];
        const double b_hyd   = c_soil_table[stype][3];
        const double psi_sat = c_soil_table[stype][4];
        const double ksat_h  = c_soil_table[stype][5];
        const double k_dry   = c_soil_table[stype][6];
        const double k_sat   = c_soil_table[stype][7];

        double ts_soil[4], ms_soil[4];
        for (int m = 0; m < 4; ++m) {
            const size_t sidx = static_cast<size_t>(m) * n2 + col;
            ts_soil[m] = soil_t ? fmin(340.0, fmax(210.0, static_cast<double>(soil_t[sidx]))) : 285.0;
            ms_soil[m] = soil_m ? fmin(th_sat, fmax(0.02, static_cast<double>(soil_m[sidx]))) : th_fc;
        }
        double swe = snow_swe ? fmax(0.0, static_cast<double>(snow_swe[col]) + p_ice * dt_s) : 0.0;
        const double snow_cov = fmin(1.0, swe / 15.0);
        const double alb = (1.0 - snow_cov) * alb_base_v + snow_cov * 0.72;
        const double emis = (1.0 - snow_cov) * 0.96 + snow_cov * 0.985;
        z0m_v = fmax(1e-3, (1.0 - snow_cov) * z0_base_v + snow_cov * 0.002);
        z0h_v = z0m_v * 0.1;

        const double r_abs = (1.0 - alb) * sw_in + emis * lw_in;
        const double lambda_sk = (1.0 - snow_cov) * (15.0 * (1.0 - veg_v) + 10.0 * veg_v) + snow_cov * 20.0;
        // Soil moisture availability + Jarvis stomatal parameters (ecLand susveg_mod)
        const double f_sm = fmin(1.0, fmax(0.05, (ms_soil[0] - th_wilt) / fmax(th_fc - th_wilt, 0.05)));
        const double f_rad = (0.55 * sw_in * 2.0 / 100.0 + 0.02) / (0.55 * sw_in * 2.0 / 100.0 + 1.0);
        const double r_canopy = 110.0 / fmax(f_sm * f_rad, 0.04);
        double beta_ev = snow_cov + (1.0 - snow_cov) * f_sm * (1.0 - 0.5 * veg_v);

        t_sk_new = t_skin ? fmin(340.0, fmax(210.0, static_cast<double>(t_skin[col]))) : ts_soil[0];
        if (t_sk_new <= 0.0) t_sk_new = ts_soil[0];

        // 2-outer coupled (M-O <-> Newton skin balance) closure
        for (int outer = 0; outer < 2; ++outer) {
            const double th_sk = t_sk_new / pi_s;
            const double q_sat_sk = qsat_water(t_sk_new, p_s);
            const double q_sk_est = qv_a + beta_ev * (q_sat_sk - qv_a);
            const double thv_a = th_a * (1.0 + EPS_V * qv_a);
            const double thv_s = th_sk * (1.0 + EPS_V * q_sk_est);
            solve_mo_sfclayrev(za, wspd_eff, thv_a, thv_s, z0m_v, z0h_v, ustar_v, zol_v, cm_v, ch_v);

            const double g_exch = ch_v * wspd_eff; // exact C_H * U [m/s]
            const double beta_veg = 1.0 / (1.0 + g_exch * r_canopy);
            beta_ev = snow_cov * 1.0 + (1.0 - snow_cov) * (veg_v * beta_veg + (1.0 - veg_v) * f_sm);
            const double flhc = rho_a * CP_D * g_exch / pi_s;
            const double flqc = rho_a * ((swe > 0.1) ? LS0 : LV0) * g_exch * beta_ev;

            for (int it = 0; it < 6; ++it) {
                const double qs_t = qsat_water(t_sk_new, p_s);
                const double dqs_t = dqsat_water_dt(t_sk_new, p_s);
                const double rad_out = emis * STEFAN_BOLTZMANN * pow(t_sk_new, 4.0);
                const double f_val = r_abs - rad_out - flhc * (t_sk_new - t_a_at_sfc) - flqc * (qs_t - qv_a) - lambda_sk * (t_sk_new - ts_soil[0]);
                const double df = -4.0 * emis * STEFAN_BOLTZMANN * pow(t_sk_new, 3.0) - flhc - flqc * dqs_t - lambda_sk;
                const double dt_step = fmin(15.0, fmax(-15.0, -f_val / fmin(df, -1.0)));
                t_sk_new = fmin(340.0, fmax(200.0, t_sk_new + dt_step));
                if (fabs(dt_step) < 1e-4) break;
            }
            if (swe > 0.05 && t_sk_new > T_FREEZE) {
                t_sk_new = T_FREEZE;
            }
            h_flux = flhc * (t_sk_new - t_a_at_sfc);
            const double qs_fin = qsat_water(t_sk_new, p_s);
            le_flux = flqc * (qs_fin - qv_a);
            q_sk_eff = qv_a + beta_ev * (qs_fin - qv_a);
        }

        double g0_flux = lambda_sk * (t_sk_new - ts_soil[0]);
        double melt_mass = 0.0;
        if (swe > 0.05 && t_sk_new >= T_FREEZE) {
            const double e_avail = r_abs - emis * STEFAN_BOLTZMANN * pow(T_FREEZE, 4.0) - h_flux - le_flux - g0_flux;
            if (e_avail > 0.0) {
                const double dm = fmin(swe, e_avail * dt_s / LF0);
                swe -= dm;
                melt_mass = dm;
            }
        }

        // 4-layer Implicit Soil Temperature Diffusion (Peters-Lidard Kersten thermal conductivity)
        double kappa_s[4], cs_s[4];
        for (int m = 0; m < 4; ++m) {
            const double sr = fmin(1.0, fmax(0.05, ms_soil[m] / th_sat));
            const double ke = (ts_soil[m] < T_FREEZE) ? sr : fmax(0.0, log10(sr) + 1.0);
            kappa_s[m] = (k_sat - k_dry) * ke + k_dry;
            const double frozen_c = (fabs(ts_soil[m] - T_FREEZE) < 1.5) ? 8.0e6 : 0.0; // smooth freeze-thaw heat buffering
            cs_s[m] = (1.0 - th_sat) * 2.1e6 + ms_soil[m] * 4.18e6 + frozen_c;
        }
        double a_t[4] = {}, b_t[4] = {}, c_t[4] = {}, d_t[4] = {};
        for (int m = 0; m < 4; ++m) {
            const double cap = cs_s[m] * c_soil_dz[m] / dt_s;
            double c_up = 0.0, c_dn = 0.0;
            if (m > 0) {
                const double k_if = 2.0 * kappa_s[m - 1] * kappa_s[m] / fmax(kappa_s[m - 1] + kappa_s[m], 0.05);
                c_up = k_if / (0.5 * (c_soil_dz[m - 1] + c_soil_dz[m]));
            }
            if (m < 3) {
                const double k_if = 2.0 * kappa_s[m] * kappa_s[m + 1] / fmax(kappa_s[m] + kappa_s[m + 1], 0.05);
                c_dn = k_if / (0.5 * (c_soil_dz[m] + c_soil_dz[m + 1]));
            }
            a_t[m] = -c_up;
            c_t[m] = -c_dn;
            b_t[m] = cap + c_up + c_dn;
            d_t[m] = cap * ts_soil[m];
        }
        d_t[0] += g0_flux;
        solve_tridiag_4(a_t, b_t, c_t, d_t, ts_soil);

        // 4-layer Soil Moisture Richards Equation (Campbella hydraulic diffusivity + drainage)
        const double evap_ms = fmax(0.0, le_flux / (LV0 * 1000.0)) * (1.0 - snow_cov);
        const double infil_ms = p_liq / 1000.0 + melt_mass / (1000.0 * dt_s) - evap_ms;
        double hyd_d[4], hyd_k[4];
        for (int m = 0; m < 4; ++m) {
            const double sr = fmin(1.0, fmax(0.05, ms_soil[m] / th_sat));
            hyd_k[m] = ksat_h * pow(sr, 2.0 * b_hyd + 3.0);
            hyd_d[m] = (b_hyd * ksat_h * psi_sat / th_sat) * pow(sr, b_hyd + 2.0);
        }
        double a_m[4] = {}, b_m[4] = {}, c_m[4] = {}, d_m[4] = {};
        for (int m = 0; m < 4; ++m) {
            const double cap_m = c_soil_dz[m] / dt_s;
            const double g_up = (m > 0) ? 0.5 * (hyd_d[m - 1] + hyd_d[m]) / (0.5 * (c_soil_dz[m - 1] + c_soil_dz[m])) : 0.0;
            const double g_dn = (m < 3) ? 0.5 * (hyd_d[m] + hyd_d[m + 1]) / (0.5 * (c_soil_dz[m] + c_soil_dz[m + 1])) : 0.0;
            const double k_up = (m > 0) ? 0.5 * (hyd_k[m - 1] + hyd_k[m]) : 0.0;
            const double k_dn = (m < 3) ? 0.5 * (hyd_k[m] + hyd_k[m + 1]) : hyd_k[3];
            a_m[m] = -g_up;
            c_m[m] = -g_dn;
            b_m[m] = cap_m + g_up + g_dn;
            d_m[m] = cap_m * ms_soil[m] + (k_up - k_dn);
        }
        d_m[0] += infil_ms;
        solve_tridiag_4(a_m, b_m, c_m, d_m, ms_soil);

        for (int m = 0; m < 4; ++m) {
            const size_t sidx = static_cast<size_t>(m) * n2 + col;
            if (soil_t) soil_t[sidx] = static_cast<float>(fmin(340.0, fmax(210.0, ts_soil[m])));
            if (soil_m) soil_m[sidx] = static_cast<float>(fmin(th_sat, fmax(0.02, ms_soil[m])));
        }
        if (snow_swe) snow_swe[col] = static_cast<float>(swe);
    } else if (lake_v >= 0.5) {
        // -------------------------------------------------------------------
        // REGIME 2: FRESHWATER INLAND LAKE (FLake Mironov 2008 / ICON BSD-3)
        // -------------------------------------------------------------------
        constexpr double RHO_W = 1000.0, CP_W = 4192.0, RHO_CP_W = RHO_W * CP_W;
        constexpr double K_ICE = 2.29, RHO_ICE = 917.0, C_ALB_MR = 95.6, T_MAX_DENS = 277.13;

        double t_w = t_skin ? fmin(315.0, fmax(250.0, static_cast<double>(t_skin[col]))) : 283.0;
        double t_b = lake_tbot ? fmin(315.0, fmax(273.15, static_cast<double>(lake_tbot[col]))) : 280.0;
        double h_ml = lake_hmld ? fmin(depth_v, fmax(0.01, static_cast<double>(lake_hmld[col]))) : fmin(depth_v, 2.5);
        double c_th = lake_ctheta ? fmin(0.80, fmax(0.50, static_cast<double>(lake_ctheta[col]))) : 0.65;
        double h_ice = lake_hice ? fmin(3.0, fmax(0.0, static_cast<double>(lake_hice[col]))) : 0.0;

        const bool ice_cov = (h_ice > 1e-4);
        const double alb = ice_cov ? fmin(0.65, fmax(0.15, 0.60 - 0.45 * exp(-C_ALB_MR * fmax(0.0, T_FREEZE - t_w) / T_FREEZE))) : 0.07;
        const double emis = ice_cov ? 0.985 : 0.97;
        z0m_v = ice_cov ? 1e-3 : fmax(1e-4, 0.014 * 0.04 / GRAVITY + 1.5e-5);
        z0h_v = z0m_v * 0.1;

        for (int it = 0; it < 2; ++it) {
            const double th_sk = t_w / pi_s;
            const double qs_sk = ice_cov ? qsat_ice(t_w, p_s) : qsat_water(t_w, p_s);
            const double thv_a = th_a * (1.0 + EPS_V * qv_a);
            const double thv_s = th_sk * (1.0 + EPS_V * qs_sk);
            solve_mo_sfclayrev(za, wspd_eff, thv_a, thv_s, z0m_v, z0h_v, ustar_v, zol_v, cm_v, ch_v);
            if (!ice_cov) z0m_v = fmin(0.02, fmax(1e-4, 0.014 * ustar_v * ustar_v / GRAVITY + 1.5e-5 / ustar_v));
            const double g_exch = ch_v * wspd_eff;
            h_flux = rho_a * CP_D * g_exch * (th_sk - th_a);
            le_flux = rho_a * (ice_cov ? LS0 : LV0) * g_exch * (qs_sk - qv_a);
            q_sk_eff = qs_sk;
        }
        const double q_net = (1.0 - alb) * sw_in + emis * lw_in - emis * STEFAN_BOLTZMANN * pow(t_w, 4.0) - h_flux - le_flux;

        if (h_ice > 1e-4 || (t_w <= T_FREEZE && q_net < 0.0)) {
            if (h_ice < 1e-4) h_ice = 0.005;
            // Conductive flux from basal freezing interface (T_FREEZE) to ice surface
            const double t_new_ice = t_w + (q_net - K_ICE * (t_w - T_FREEZE) / fmax(h_ice, 0.01)) * dt_s / (RHO_ICE * 2100.0 * fmin(h_ice, 0.1));
            if (t_new_ice > T_FREEZE) {
                t_w = T_FREEZE;
                const double e_melt = fmax(0.0, q_net) * dt_s;
                h_ice = fmax(0.0, h_ice - e_melt / (RHO_ICE * LF0));
            } else {
                t_w = fmax(230.0, t_new_ice);
                const double q_cond = K_ICE * (T_FREEZE - t_w) / fmax(h_ice, 0.01);
                h_ice = fmin(3.0, fmax(0.0, h_ice + q_cond * dt_s / (RHO_ICE * LF0)));
            }
            if (h_ice <= 1e-4) { h_ice = 0.0; t_w = T_FREEZE; }
            t_sk_new = t_w;
        } else {
            // Open freshwater two-layer FLake update
            const double c_eff = RHO_CP_W * fmin(depth_v, fmax(0.5, h_ml + (1.0 - c_th) * (depth_v - h_ml)));
            const double alpha_t = 1.6509e-5 * (t_w - T_MAX_DENS);
            const double buoy_s = GRAVITY * alpha_t * q_net / RHO_CP_W;
            const double u_star_w = sqrt(rho_a / RHO_W) * ustar_v;
            if (buoy_s < 0.0 || u_star_w > 0.015) {
                const double w_conv = (buoy_s < 0.0) ? cbrt(-buoy_s * h_ml) : 0.0;
                const double dh = 0.25 * sqrt(u_star_w * u_star_w + w_conv * w_conv);
                h_ml = fmin(depth_v, h_ml + dh * dt_s * 0.2);
            } else {
                const double l_ob = fmax(0.2, pow(u_star_w, 3.0) / fmax(KARMAN * buoy_s, 1e-12));
                h_ml = fmin(depth_v, fmax(0.2, h_ml + (fmin(depth_v, l_ob) - h_ml) * (dt_s / 3600.0)));
            }
            t_w = fmin(315.0, fmax(T_FREEZE, t_w + q_net * dt_s / c_eff));
            if (h_ml >= 0.98 * depth_v) {
                t_b = t_w;
            } else {
                t_b += (t_w - t_b) * (dt_s / (14.0 * 86400.0));
            }
            c_th = fmin(0.80, fmax(0.50, c_th + (0.65 - c_th) * (dt_s / 7200.0)));
            t_sk_new = t_w;
        }

        if (lake_tbot) lake_tbot[col] = static_cast<float>(t_b);
        if (lake_hmld) lake_hmld[col] = static_cast<float>(h_ml);
        if (lake_ctheta) lake_ctheta[col] = static_cast<float>(c_th);
        if (lake_hice) lake_hice[col] = static_cast<float>(h_ice);
    } else {
        // -------------------------------------------------------------------
        // REGIME 3: NORTH SEA / OCEAN (COARE 3.6 + GEBCO z0 + ecLand VOSKIN)
        // -------------------------------------------------------------------
        const double sst_b = sst_bulk ? fmin(312.0, fmax(271.15, static_cast<double>(sst_bulk[col])))
                                      : (t_skin ? fmin(312.0, fmax(271.15, static_cast<double>(t_skin[col]))) : 284.0);
        double d_cool = dt_cool ? fmin(0.3, fmax(-2.0, static_cast<double>(dt_cool[col]))) : 0.0;
        double d_warm = dt_warm ? fmin(5.0, fmax(0.0, static_cast<double>(dt_warm[col]))) : 0.0;
        double t_sk = fmin(315.0, fmax(271.15, sst_b + d_cool + d_warm));
        const double th_sk = t_sk / pi_s;

        double q_sea_sat = 0.0, wspd_coare = wspd_eff;
        solve_coare36(za, u_spd, t_a, th_a, qv_a, p_a, t_sk, th_sk, p_s, depth_v,
                      ustar_v, z0m_v, z0h_v, zol_v, cm_v, ch_v, q_sea_sat, wspd_coare);

        wspd_drag = fmax(wspd_coare, 0.3);
        const double g_exch = ch_v * wspd_drag;
        h_flux = rho_a * CP_D * g_exch * (th_sk - th_a);
        le_flux = rho_a * LV0 * g_exch * (q_sea_sat - qv_a);
        q_sk_eff = q_sea_sat;

        // Fairall (1996) / ecLand VOSKIN Cool Skin & Takaya/Zeng Warm Layer
        constexpr double RHO_W = 1025.0, CP_W = 3985.0, K_W = 0.60, NU_W = 1.1e-6;
        constexpr double EMIS_O = 0.97, ALB_O = 0.06;
        const double sw_net = (1.0 - ALB_O) * sw_in;
        const double lw_net = EMIS_O * lw_in - EMIS_O * STEFAN_BOLTZMANN * pow(t_sk, 4.0);
        // Non-solar cooling leaving water surface (positive = surface loses heat -> cool skin < 0)
        const double q_ns_cool = -(lw_net - h_flux - le_flux);
        const double q_cool_eff = q_ns_cool - 0.12 * sw_net;
        const double us_w = fmax(1e-4, sqrt(rho_a / RHO_W) * ustar_v);
        const double a_cs = 16.0 * GRAVITY * 2.1e-4 * RHO_W * CP_W * pow(NU_W, 3.0) * fmax(0.0, q_cool_eff)
                          / (pow(us_w, 4.0) * K_W * K_W + 1e-18);
        const double lambda_s = 6.0 / pow(1.0 + pow(a_cs, 0.75), 1.0 / 3.0);
        const double delta_cool = fmin(0.01, fmax(2e-4, lambda_s * NU_W / us_w));
        d_cool = fmin(0.25, fmax(-1.5, -q_cool_eff * delta_cool / K_W));

        // Takaya/Zeng warm layer over slab depth (shallow shelf coastal waters warm slightly faster)
        const double d_warm_depth = fmin(3.0, fmax(1.0, 0.5 * depth_v));
        const double sw_abs_w = 0.65 * sw_net;
        const double q_warm_net = sw_abs_w - q_ns_cool;
        const double mix_diss = 2.0 * KARMAN * us_w / d_warm_depth; // exponential wind mixing decay
        d_warm = fmax(0.0, (d_warm + (q_warm_net / (RHO_W * CP_W * d_warm_depth)) * dt_s) / (1.0 + mix_diss * dt_s));
        d_warm = fmin(4.5, d_warm);

        t_sk_new = fmin(315.0, fmax(271.15, sst_b + d_cool + d_warm));
        if (dt_cool) dt_cool[col] = static_cast<float>(d_cool);
        if (dt_warm) dt_warm[col] = static_cast<float>(d_warm);
    }

    // Store common fluxes, implicit PBL bottom drag, and M-O 2m/10m screen diagnostics
    if (t_skin) t_skin[col] = static_cast<float>(t_sk_new);
    if (shf) shf[col] = static_cast<float>(h_flux);
    if (lhf) lhf[col] = static_cast<float>(le_flux);
    if (ustar) ustar[col] = static_cast<float>(ustar_v);
    if (sfc_drag_coeff) sfc_drag_coeff[col] = static_cast<float>(rho_a * cm_v * wspd_drag);

    // M-O profile interpolation at 2m (T, q) and 10m (u, v)
    const double z_2 = fmin(za, 2.0);
    const double z_10 = fmin(za, 10.0);
    const double fh_num = fmax(0.1, log(z_2 / fmax(z0h_v, 1e-6)) - psih_jimenez(zol_v * (z_2 / za)) + psih_jimenez(zol_v * z0h_v / za));
    const double fh_den = fmax(0.5, log(za / fmax(z0h_v, 1e-6)) - psih_jimenez(zol_v) + psih_jimenez(zol_v * z0h_v / za));
    const double w_2m = fmin(1.0, fmax(0.0, fh_num / fh_den));

    const double fm_num = fmax(0.1, log(z_10 / fmax(z0m_v, 1e-5)) - psim_jimenez(zol_v * (z_10 / za)) + psim_jimenez(zol_v * z0m_v / za));
    const double fm_den = fmax(0.5, log(za / fmax(z0m_v, 1e-5)) - psim_jimenez(zol_v) + psim_jimenez(zol_v * z0m_v / za));
    const double w_10m = fmin(1.0, fmax(0.0, fm_num / fm_den));

    const double th_sk = t_sk_new / pi_s;
    const double th_2m = th_sk + w_2m * (th_a - th_sk);
    const double p_2m = p_s * exp(-GRAVITY * z_2 / (R_D * tv_a));
    if (t2m) t2m[col] = static_cast<float>(th_2m * pow(p_2m / P0_PA, KAPPA));
    if (q2m) q2m[col] = static_cast<float>(fmax(1e-7, q_sk_eff + w_2m * (qv_a - q_sk_eff)));
    if (u10m) u10m[col] = static_cast<float>(w_10m * u_c);
    if (v10m) v10m[col] = static_cast<float>(w_10m * v_c);
}

// ===========================================================================
// PILLAR 2 (YSU Nonlocal PBL + Shallow EDMFm + Implicit Double Thomas Solver)
// ===========================================================================

__global__ void nh_pbl_scalar_kernel(
    NhGridConfig cfg,
    DeviceStaticGrid g,
    double dt_s,
    const float* u, const float* v, const float* pip,
    const float* lsm, const float* shf, const float* lhf, const float* ustar,
    float* thp, float* qv, float* qc, float* qi, float* pblh,
    float* mu_scratch, float* kh_scratch, double* tri_c, double* tri_d
) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= cfg.ny || i >= cfg.nx) return;

    const int32_t nz = cfg.nz;
    const size_t n2 = static_cast<size_t>(cfg.ny) * cfg.nx;
    const size_t col = static_cast<size_t>(j) * cfg.nx + i;
    const double hs = g.hs[col];
    const double lsm_v = lsm ? static_cast<double>(lsm[col]) : 1.0;
    const double f_cor = fabs(g.fcor[col]);
    const double ustar_v = ustar ? fmax(0.05, static_cast<double>(ustar[col])) : 0.15;
    const double shf_v = shf ? static_cast<double>(shf[col]) : 0.0;
    const double lhf_v = lhf ? static_cast<double>(lhf[col]) : 0.0;

    // Surface thermodynamic level k=0
    double z_0, dz_0, thb0, pib0, th0, pi0, t0, p0, rho0;
    eval_cell_thermo(cfg, g, 0, col, hs, thp, pip, qv, nullptr, nullptr, nullptr, nullptr, nullptr,
                     z_0, dz_0, thb0, pib0, th0, pi0, t0, p0, rho0);
    const double qv0 = (cfg.has_moist && qv) ? fmax(0.0, static_cast<double>(qv[col])) : 0.0;
    const double thv0_raw = th0 * (1.0 + EPS_V * qv0);
    const double wth0 = shf_v / (rho0 * CP_D);
    const double wq0 = lhf_v / (rho0 * LV0);
    const double wtv0 = wth0 + 0.608 * t0 * wq0;

    // 1. Diagnose YSU PBL top hpbl using Bulk Richardson number (MMM bl_ysu.F90)
    const double brcr = (wtv0 > 0.0) ? 0.0
        : ((lsm_v < 0.5) ? fmin(0.30, fmax(0.10, 0.16 * pow(1e-7 * ustar_v / fmax(f_cor * 1.5e-5, 1e-12), -0.18)))
                         : 0.25);

    double hpbl = fmax(dz_0, 40.0);
    // Initial w* estimate using 600 m reference to get thermal excess
    const double wstar_init = (wtv0 > 0.0) ? cbrt((GRAVITY / fmax(thv0_raw, 150.0)) * wtv0 * 600.0) : 0.0;
    const double wscale_init = fmax(ustar_v, cbrt(pow(ustar_v, 3.0) + 4.0 * KARMAN * pow(wstar_init, 3.0)));
    const double dth_excess = (wtv0 > 0.0) ? fmin(3.0, 6.8 * wtv0 / wscale_init) : 0.0;
    const double thv_ref = thv0_raw + dth_excess;

    double rib_prev = -1.0, z_prev = z_0 - hs;
    for (int32_t k = 1; k < nz; ++k) {
        const size_t idx = static_cast<size_t>(k) * n2 + col;
        double zk, dzk, thbk, pibk, thk, pik, tk, pk, rhok;
        eval_cell_thermo(cfg, g, k, col, hs, thp, pip, qv, nullptr, nullptr, nullptr, nullptr, nullptr,
                         zk, dzk, thbk, pibk, thk, pik, tk, pk, rhok);
        const double qvk = (cfg.has_moist && qv) ? fmax(0.0, static_cast<double>(qv[idx])) : 0.0;
        const double thvk = thk * (1.0 + EPS_V * qvk);
        const double z_agl = fmax(1.0, zk - hs);
        const double uc = 0.5 * (read_u(u, k, j, i, nz, cfg.ny, cfg.nx, cfg.bc_x, cfg.bc_y)
                               + read_u(u, k, j, i + 1, nz, cfg.ny, cfg.nx, cfg.bc_x, cfg.bc_y));
        const double vc = 0.5 * (read_v(v, k, j, i, nz, cfg.ny, cfg.nx, cfg.bc_x, cfg.bc_y)
                               + read_v(v, k, j + 1, i, nz, cfg.ny, cfg.nx, cfg.bc_x, cfg.bc_y));
        const double rib = GRAVITY * z_agl * (thvk - thv_ref) / (thv_ref * fmax(uc * uc + vc * vc, 0.5));
        if (rib >= brcr) {
            const double frac = (rib > rib_prev) ? fmin(1.0, fmax(0.0, (brcr - rib_prev) / (rib - rib_prev))) : 0.5;
            hpbl = z_prev + frac * (z_agl - z_prev);
            break;
        }
        rib_prev = rib;
        z_prev = z_agl;
        if (z_agl > 4500.0) { hpbl = 4500.0; break; }
    }
    hpbl = (wtv0 > 0.0) ? fmin(4500.0, fmax(dz_0, hpbl)) : fmin(1500.0, fmax(dz_0, hpbl));
    if (pblh) pblh[col] = static_cast<float>(hpbl);

    const double wstar = (wtv0 > 0.0) ? cbrt((GRAVITY / fmax(thv0_raw, 150.0)) * wtv0 * hpbl) : 0.0;
    const double wscale = fmin(16.0 * ustar_v, fmax(0.2 * ustar_v, cbrt(pow(ustar_v, 3.0) + 4.0 * KARMAN * pow(wstar, 3.0))));
    const double obukhov_l = (fabs(wtv0) > 1e-8) ? (-thv0_raw * pow(ustar_v, 3.0) / (KARMAN * GRAVITY * wtv0)) : 1e6;

    // 2. Compute YSU K_h[k] -> kh_scratch[k] and Shallow EDMF Updraft Mass Flux M_u[k] -> mu_scratch[k] at interfaces k+1
    constexpr double AU_EDMF = 0.10;
    double w_u2 = (wtv0 > 0.0) ? fmax(0.25, 1.5 * wstar * wstar + ustar_v * ustar_v) : 0.0;
    const double qc0 = (cfg.has_moist && qc) ? fmax(0.0, static_cast<double>(qc[col])) : 0.0;
    double thl_u = th0 - (LV0 / (CP_D * pi0)) * qc0 + ((wtv0 > 0.0) ? 1.5 * wth0 / wscale : 0.0);
    double qt_u = (qv0 + qc0) + ((wtv0 > 0.0) ? 1.5 * fmax(0.0, wq0) / wscale : 0.0);
    bool updraft_alive = (wtv0 > 0.05 * ustar_v * 0.01 || wtv0 > 1e-4);

    for (int32_t k = 0; k < nz - 1; ++k) {
        const size_t idx_k = static_cast<size_t>(k) * n2 + col;
        const size_t idx_kp = static_cast<size_t>(k + 1) * n2 + col;
        double zk, dzk, thbk, pibk, thk, pik, tk, pk, rhok;
        double zkp, dzkp, thbkp, pibkp, thkp, pikp, tkp, pkp, rhokp;
        eval_cell_thermo(cfg, g, k, col, hs, thp, pip, qv, qc, nullptr, qi, nullptr, nullptr,
                         zk, dzk, thbk, pibk, thk, pik, tk, pk, rhok);
        eval_cell_thermo(cfg, g, k + 1, col, hs, thp, pip, qv, qc, nullptr, qi, nullptr, nullptr,
                         zkp, dzkp, thbkp, pibkp, thkp, pikp, tkp, pkp, rhokp);

        const double z_if = (g.zeta_w[k + 1] + g.b_w[k + 1] * hs) - hs;
        const double dz_mid = fmax(1.0, zkp - zk);
        const double rho_if = 0.5 * (rhok + rhokp);

        const double qvk = (cfg.has_moist && qv) ? fmax(0.0, static_cast<double>(qv[idx_k])) : 0.0;
        const double qvkp = (cfg.has_moist && qv) ? fmax(0.0, static_cast<double>(qv[idx_kp])) : 0.0;
        const double thvk = thk * (1.0 + EPS_V * qvk);
        const double thvkp = thkp * (1.0 + EPS_V * qvkp);

        // Interface K_h (YSU within PBL, Louis local K above)
        double kh_val = 0.01;
        if (z_if < hpbl) {
            const double s = z_if / hpbl;
            const double km = KARMAN * wscale * z_if * (1.0 - s) * (1.0 - s);
            const double pr = fmin(2.5, fmax(0.65, 1.0 + 2.1 * ((z_0 - hs) / fmax(fabs(obukhov_l), 5.0))));
            kh_val = fmin(1000.0, fmax(0.01, km / pr));
        } else {
            const double uk = 0.5 * (read_u(u, k, j, i, nz, cfg.ny, cfg.nx, cfg.bc_x, cfg.bc_y)
                                   + read_u(u, k, j, i + 1, nz, cfg.ny, cfg.nx, cfg.bc_x, cfg.bc_y));
            const double ukp = 0.5 * (read_u(u, k + 1, j, i, nz, cfg.ny, cfg.nx, cfg.bc_x, cfg.bc_y)
                                    + read_u(u, k + 1, j, i + 1, nz, cfg.ny, cfg.nx, cfg.bc_x, cfg.bc_y));
            const double vk = 0.5 * (read_v(v, k, j, i, nz, cfg.ny, cfg.nx, cfg.bc_x, cfg.bc_y)
                                   + read_v(v, k, j + 1, i, nz, cfg.ny, cfg.nx, cfg.bc_x, cfg.bc_y));
            const double vkp = 0.5 * (read_v(v, k + 1, j, i, nz, cfg.ny, cfg.nx, cfg.bc_x, cfg.bc_y)
                                    + read_v(v, k + 1, j + 1, i, nz, cfg.ny, cfg.nx, cfg.bc_x, cfg.bc_y));
            const double dudz = (ukp - uk) / dz_mid;
            const double dvdz = (vkp - vk) / dz_mid;
            const double s2 = dudz * dudz + dvdz * dvdz + 1e-8;
            const double n2 = (GRAVITY / (0.5 * (thvk + thvkp))) * ((thvkp - thvk) / dz_mid);
            const double ri_loc = n2 / s2;
            const double ell = (KARMAN * z_if) / (1.0 + KARMAN * z_if / 30.0);
            double fh = 1.0;
            if (ri_loc >= 0.0) {
                fh = 1.0 / ((1.0 + 5.0 * ri_loc) * (1.0 + 5.0 * ri_loc));
            } else {
                fh = 1.0 - (15.0 * ri_loc) / (1.0 + 75.0 * ell * ell * sqrt(-ri_loc) / (dz_mid * dz_mid));
            }
            kh_val = fmin(50.0, fmax(0.01, ell * ell * sqrt(s2) * fh));
        }
        // Enhance mixing weakly right across PBL top for YSU entrainment zone
        if (wtv0 > 0.0 && fabs(z_if - hpbl) < 0.15 * hpbl) {
            const double wm3 = pow(wstar, 3.0) + 5.0 * pow(ustar_v, 3.0);
            const double k_ent = 0.15 * dz_mid * cbrt(wm3) * 0.25;
            kh_val = fmin(100.0, kh_val + k_ent);
        }
        kh_scratch[idx_k] = static_cast<float>(kh_val);

        // Shallow EDMF thermal plume integration (Siebesma et al. 2007 / Pergaud et al. 2009)
        double mu_val = 0.0;
        if (updraft_alive && z_if < 4000.0) {
            const double qck = (cfg.has_moist && qc) ? fmax(0.0, static_cast<double>(qc[idx_k])) : 0.0;
            const double thl_env = thk - (LV0 / (CP_D * pik)) * qck;
            const double qt_env = qvk + qck;
            const double eps_u = 1.8 / fmax(z_if + 100.0, 100.0);
            thl_u = (thl_u + eps_u * dz_mid * thl_env) / (1.0 + eps_u * dz_mid);
            qt_u = (qt_u + eps_u * dz_mid * qt_env) / (1.0 + eps_u * dz_mid);

            const double pi_if = 0.5 * (pik + pikp);
            const double p_if = 0.5 * (pk + pkp);
            double t_u = thl_u * pi_if;
            double qc_u = fmax(0.0, qt_u - qsat_water(t_u, p_if));
            t_u = fmin(350.0, fmax(160.0, (thl_u + (LV0 / (CP_D * pi_if)) * qc_u) * pi_if));
            const double qv_u = fmax(0.0, qt_u - qc_u);
            const double thv_u = (t_u / pi_if) * (1.0 + EPS_V * qv_u - qc_u);
            const double thv_env = 0.5 * (thvk + thvkp);
            const double buoy = GRAVITY * (thv_u - thv_env) / fmax(thv_env, 150.0);
            w_u2 = (w_u2 + 2.0 * dz_mid * buoy) / (1.0 + 2.0 * 1.0 * eps_u * dz_mid);
            if (w_u2 > 0.01) {
                mu_val = AU_EDMF * rho_if * sqrt(w_u2);
            } else {
                updraft_alive = false;
            }
        }
        mu_scratch[idx_k] = static_cast<float>(mu_val);
    }

    // 3. Implicit double Thomas vertical solver for [thp] (with surface SHF + YSU nonlocal countergradient)
    // and [qv] (with surface LHF + YSU countergradient) and [qc, qi]
    const double gamma_th = (wtv0 > 0.0) ? fmin(5.0e-3, 6.8 * wth0 / fmax(wscale * hpbl, 1.0)) : 0.0;
    const double gamma_qv = (wtv0 > 0.0 && lhf_v > 0.0) ? fmin(2.0e-6, 6.8 * wq0 / fmax(wscale * hpbl, 1.0)) : 0.0;

    // Solve species: 0 = thp, 1 = qv, 2 = qc, 3 = qi
    for (int sp = 0; sp < 4; ++sp) {
        if (sp > 0 && (!cfg.has_moist || !qv)) break;
        float* fptr = (sp == 0) ? thp : ((sp == 1) ? qv : ((sp == 2) ? qc : qi));
        if (!fptr) continue;

        double c_prev = 0.0;
        for (int32_t k = 0; k < nz; ++k) {
            const size_t idx = static_cast<size_t>(k) * n2 + col;
            double zk, dzk, thbk, pibk, thk, pik, tk, pk, rhok;
            eval_cell_thermo(cfg, g, k, col, hs, thp, pip, qv, nullptr, nullptr, nullptr, nullptr, nullptr,
                             zk, dzk, thbk, pibk, thk, pik, tk, pk, rhok);
            const double mass_k = rhok * dzk; // [kg/m^2]

            double a_diag = 0.0, c_diag = 0.0;
            double explicit_src = 0.0;

            if (k > 0) {
                const size_t idx_km = static_cast<size_t>(k - 1) * n2 + col;
                double zkm, dzkm, thbkm, pibkm, thkm, pikm, tkm, pkm, rhokm;
                eval_cell_thermo(cfg, g, k - 1, col, hs, thp, pip, qv, nullptr, nullptr, nullptr, nullptr, nullptr,
                                 zkm, dzkm, thbkm, pibkm, thkm, pikm, tkm, pkm, rhokm);
                const double dz_dn = fmax(1.0, zk - zkm);
                const double rho_dn = 0.5 * (rhokm + rhok);
                const double kh_dn = static_cast<double>(kh_scratch[idx_km]);
                const double mu_dn = static_cast<double>(mu_scratch[idx_km]);
                const double g_dn = rho_dn * kh_dn / dz_dn;
                a_diag = -g_dn;

                const double z_if_dn = (g.zeta_w[k] + g.b_w[k] * hs) - hs;
                if (z_if_dn < hpbl) {
                    const double gg = (sp == 0) ? gamma_th : ((sp == 1) ? gamma_qv : 0.0);
                    explicit_src += rho_dn * kh_dn * gg;
                }
                if (sp == 0) {
                    explicit_src -= rho_dn * kh_dn * (thbk - thbkm) / dz_dn;
                }
                if (mu_dn > 0.0 && (sp == 0 || sp == 1)) {
                    const double env0 = static_cast<double>(fptr[col]) + ((sp == 0) ? thb0 : 0.0);
                    const double envk = static_cast<double>(fptr[idx]) + ((sp == 0) ? thbk : 0.0);
                    explicit_src += mu_dn * 0.5 * (env0 - envk);
                }
            } else {
                // Surface flux at z_w(0) entering k=0 cell
                if (sp == 0) {
                    explicit_src += shf_v / (CP_D * pi0);
                } else if (sp == 1) {
                    explicit_src += lhf_v / LV0;
                }
            }

            if (k + 1 < nz) {
                double zkp, dzkp, thbkp, pibkp, thkp, pikp, tkp, pkp, rhokp;
                eval_cell_thermo(cfg, g, k + 1, col, hs, thp, pip, qv, nullptr, nullptr, nullptr, nullptr, nullptr,
                                 zkp, dzkp, thbkp, pibkp, thkp, pikp, tkp, pkp, rhokp);
                const double dz_up = fmax(1.0, zkp - zk);
                const double rho_up = 0.5 * (rhok + rhokp);
                const double kh_up = static_cast<double>(kh_scratch[idx]);
                const double mu_up = static_cast<double>(mu_scratch[idx]);
                const double g_up = rho_up * kh_up / dz_up;
                c_diag = -g_up;

                const double z_if_up = (g.zeta_w[k + 1] + g.b_w[k + 1] * hs) - hs;
                if (z_if_up < hpbl) {
                    const double gg = (sp == 0) ? gamma_th : ((sp == 1) ? gamma_qv : 0.0);
                    explicit_src -= rho_up * kh_up * gg;
                }
                if (sp == 0) {
                    explicit_src += rho_up * kh_up * (thbkp - thbk) / dz_up;
                }
                if (mu_up > 0.0 && (sp == 0 || sp == 1)) {
                    const double env0 = static_cast<double>(fptr[col]) + ((sp == 0) ? thb0 : 0.0);
                    const double envkp = static_cast<double>(fptr[static_cast<size_t>(k + 1) * n2 + col]) + ((sp == 0) ? thbkp : 0.0);
                    explicit_src -= mu_up * 0.5 * (env0 - envkp);
                }
            }

            const double b_diag = mass_k / dt_s - a_diag - c_diag;
            const double rhs = (mass_k / dt_s) * static_cast<double>(fptr[idx]) + explicit_src;
            const double denom = b_diag - a_diag * c_prev;
            c_prev = (k + 1 < nz) ? (c_diag / denom) : 0.0;
            const double d_prev = (k > 0) ? tri_d[static_cast<size_t>(k - 1) * n2 + col] : 0.0;
            tri_c[idx] = c_prev;
            tri_d[idx] = (rhs - a_diag * d_prev) / denom;
        }

        double x_nxt = tri_d[static_cast<size_t>(nz - 1) * n2 + col];
        fptr[static_cast<size_t>(nz - 1) * n2 + col] = static_cast<float>((sp > 0) ? fmax(0.0, x_nxt) : x_nxt);
        for (int32_t k = nz - 2; k >= 0; --k) {
            const size_t idx = static_cast<size_t>(k) * n2 + col;
            x_nxt = tri_d[idx] - tri_c[idx] * x_nxt;
            fptr[idx] = static_cast<float>((sp > 0) ? fmax(0.0, x_nxt) : x_nxt);
        }
    }
}

__global__ void nh_pbl_momentum_u_kernel(
    NhGridConfig cfg,
    DeviceStaticGrid g,
    double dt_s,
    int32_t pass_edge,
    const float* pblh, const float* ustar, const float* sfc_drag_coeff,
    float* u, double* tri_c, double* tri_d
) {
    const int32_t i = pass_edge ? cfg.nx : (blockIdx.x * blockDim.x + threadIdx.x);
    const int32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= cfg.ny || i > cfg.nx) return;
    if (!pass_edge && i >= cfg.nx) return;
    if (cfg.bc_x == 0 && (i == 0 || i == cfg.nx)) return;

    const int32_t nz = cfg.nz;
    const int32_t nx_u = cfg.nx + 1;
    const size_t n2 = static_cast<size_t>(cfg.ny) * cfg.nx;
    const size_t tri_col = static_cast<size_t>(j) * cfg.nx + (pass_edge ? 0 : i);

    const int32_t i_l = (cfg.bc_x == 1) ? wrap_idx(i - 1, cfg.nx) : ((i - 1 < 0) ? 0 : i - 1);
    const int32_t i_r = (cfg.bc_x == 1) ? wrap_idx(i, cfg.nx) : ((i >= cfg.nx) ? cfg.nx - 1 : i);
    const size_t col_l = static_cast<size_t>(j) * cfg.nx + i_l;
    const size_t col_r = static_cast<size_t>(j) * cfg.nx + i_r;

    const double hs = 0.5 * (g.hs[col_l] + g.hs[col_r]);
    const double hpbl = pblh ? 0.5 * (static_cast<double>(pblh[col_l]) + static_cast<double>(pblh[col_r])) : 500.0;
    const double us = ustar ? fmax(0.05, 0.5 * (static_cast<double>(ustar[col_l]) + static_cast<double>(ustar[col_r]))) : 0.15;
    const double c_drag = sfc_drag_coeff ? 0.5 * (static_cast<double>(sfc_drag_coeff[col_l]) + static_cast<double>(sfc_drag_coeff[col_r])) : 0.0;

    double c_prev = 0.0;
    for (int32_t k = 0; k < nz; ++k) {
        const size_t u_idx = static_cast<size_t>(k) * cfg.ny * nx_u + static_cast<size_t>(j) * nx_u + i;
        const size_t t_idx = static_cast<size_t>(k) * n2 + tri_col;
        const double z_k = g.zeta_s[k] + g.b_s[k] * hs;
        const double dz_k = fmax(1.0, (1.0 + g.db_s[k] * hs) * g.dzeta[k]);
        double thb, pib, dth, rhob, cs2;
        eval_ref_state(z_k, cfg.theta_sfc_k, cfg.p_sfc_pa, cfg.n_ref_s, thb, pib, dth, rhob, cs2);
        const double mass_k = rhob * dz_k;

        double a_d = 0.0, c_d = 0.0, sfc_impl = 0.0;
        if (k == 0) {
            sfc_impl = c_drag; // bottom stress rho * C_d * |U| * u_0^{n+1}
        } else {
            const double z_km = g.zeta_s[k - 1] + g.b_s[k - 1] * hs;
            const double z_if = (g.zeta_w[k] + g.b_w[k] * hs) - hs;
            double thbm, pibm, dthm, rhobm, cs2m;
            eval_ref_state(z_km, cfg.theta_sfc_k, cfg.p_sfc_pa, cfg.n_ref_s, thbm, pibm, dthm, rhobm, cs2m);
            const double km = (z_if < hpbl)
                ? fmax(0.1, fmin(1000.0, KARMAN * us * 2.0 * z_if * (1.0 - z_if / hpbl) * (1.0 - z_if / hpbl)))
                : 0.2;
            a_d = -(0.5 * (rhobm + rhob)) * km / fmax(1.0, z_k - z_km);
        }
        if (k + 1 < nz) {
            const double z_kp = g.zeta_s[k + 1] + g.b_s[k + 1] * hs;
            const double z_if = (g.zeta_w[k + 1] + g.b_w[k + 1] * hs) - hs;
            double thbp, pibp, dthp, rhobp, cs2p;
            eval_ref_state(z_kp, cfg.theta_sfc_k, cfg.p_sfc_pa, cfg.n_ref_s, thbp, pibp, dthp, rhobp, cs2p);
            const double km = (z_if < hpbl)
                ? fmax(0.1, fmin(1000.0, KARMAN * us * 2.0 * z_if * (1.0 - z_if / hpbl) * (1.0 - z_if / hpbl)))
                : 0.2;
            c_d = -(0.5 * (rhob + rhobp)) * km / fmax(1.0, z_kp - z_k);
        }
        const double b_d = mass_k / dt_s + sfc_impl - a_d - c_d;
        const double rhs = (mass_k / dt_s) * static_cast<double>(u[u_idx]);
        const double denom = b_d - a_d * c_prev;
        c_prev = (k + 1 < nz) ? (c_d / denom) : 0.0;
        const double d_prev = (k > 0) ? tri_d[static_cast<size_t>(k - 1) * n2 + tri_col] : 0.0;
        tri_c[t_idx] = c_prev;
        tri_d[t_idx] = (rhs - a_d * d_prev) / denom;
    }

    double x_nxt = tri_d[static_cast<size_t>(nz - 1) * n2 + tri_col];
    u[static_cast<size_t>(nz - 1) * cfg.ny * nx_u + static_cast<size_t>(j) * nx_u + i] = static_cast<float>(x_nxt);
    for (int32_t k = nz - 2; k >= 0; --k) {
        const size_t t_idx = static_cast<size_t>(k) * n2 + tri_col;
        x_nxt = tri_d[t_idx] - tri_c[t_idx] * x_nxt;
        u[static_cast<size_t>(k) * cfg.ny * nx_u + static_cast<size_t>(j) * nx_u + i] = static_cast<float>(x_nxt);
    }
}

__global__ void nh_pbl_momentum_v_kernel(
    NhGridConfig cfg,
    DeviceStaticGrid g,
    double dt_s,
    int32_t pass_edge,
    const float* pblh, const float* ustar, const float* sfc_drag_coeff,
    float* v, double* tri_c, double* tri_d
) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t j = pass_edge ? cfg.ny : (blockIdx.y * blockDim.y + threadIdx.y);
    if (i >= cfg.nx || j > cfg.ny) return;
    if (!pass_edge && j >= cfg.ny) return;
    if (cfg.bc_y == 0 && (j == 0 || j == cfg.ny)) return;

    const int32_t nz = cfg.nz;
    const int32_t ny_v = cfg.ny + 1;
    const size_t n2 = static_cast<size_t>(cfg.ny) * cfg.nx;
    const size_t tri_col = static_cast<size_t>(pass_edge ? 0 : j) * cfg.nx + i;

    const int32_t j_s = (cfg.bc_y == 1) ? wrap_idx(j - 1, cfg.ny) : ((j - 1 < 0) ? 0 : j - 1);
    const int32_t j_n = (cfg.bc_y == 1) ? wrap_idx(j, cfg.ny) : ((j >= cfg.ny) ? cfg.ny - 1 : j);
    const size_t col_s = static_cast<size_t>(j_s) * cfg.nx + i;
    const size_t col_n = static_cast<size_t>(j_n) * cfg.nx + i;

    const double hs = 0.5 * (g.hs[col_s] + g.hs[col_n]);
    const double hpbl = pblh ? 0.5 * (static_cast<double>(pblh[col_s]) + static_cast<double>(pblh[col_n])) : 500.0;
    const double us = ustar ? fmax(0.05, 0.5 * (static_cast<double>(ustar[col_s]) + static_cast<double>(ustar[col_n]))) : 0.15;
    const double c_drag = sfc_drag_coeff ? 0.5 * (static_cast<double>(sfc_drag_coeff[col_s]) + static_cast<double>(sfc_drag_coeff[col_n])) : 0.0;

    double c_prev = 0.0;
    for (int32_t k = 0; k < nz; ++k) {
        const size_t v_idx = static_cast<size_t>(k) * ny_v * cfg.nx + static_cast<size_t>(j) * cfg.nx + i;
        const size_t t_idx = static_cast<size_t>(k) * n2 + tri_col;
        const double z_k = g.zeta_s[k] + g.b_s[k] * hs;
        const double dz_k = fmax(1.0, (1.0 + g.db_s[k] * hs) * g.dzeta[k]);
        double thb, pib, dth, rhob, cs2;
        eval_ref_state(z_k, cfg.theta_sfc_k, cfg.p_sfc_pa, cfg.n_ref_s, thb, pib, dth, rhob, cs2);
        const double mass_k = rhob * dz_k;

        double a_d = 0.0, c_d = 0.0, sfc_impl = 0.0;
        if (k == 0) {
            sfc_impl = c_drag;
        } else {
            const double z_km = g.zeta_s[k - 1] + g.b_s[k - 1] * hs;
            const double z_if = (g.zeta_w[k] + g.b_w[k] * hs) - hs;
            double thbm, pibm, dthm, rhobm, cs2m;
            eval_ref_state(z_km, cfg.theta_sfc_k, cfg.p_sfc_pa, cfg.n_ref_s, thbm, pibm, dthm, rhobm, cs2m);
            const double km = (z_if < hpbl)
                ? fmax(0.1, fmin(1000.0, KARMAN * us * 2.0 * z_if * (1.0 - z_if / hpbl) * (1.0 - z_if / hpbl)))
                : 0.2;
            a_d = -(0.5 * (rhobm + rhob)) * km / fmax(1.0, z_k - z_km);
        }
        if (k + 1 < nz) {
            const double z_kp = g.zeta_s[k + 1] + g.b_s[k + 1] * hs;
            const double z_if = (g.zeta_w[k + 1] + g.b_w[k + 1] * hs) - hs;
            double thbp, pibp, dthp, rhobp, cs2p;
            eval_ref_state(z_kp, cfg.theta_sfc_k, cfg.p_sfc_pa, cfg.n_ref_s, thbp, pibp, dthp, rhobp, cs2p);
            const double km = (z_if < hpbl)
                ? fmax(0.1, fmin(1000.0, KARMAN * us * 2.0 * z_if * (1.0 - z_if / hpbl) * (1.0 - z_if / hpbl)))
                : 0.2;
            c_d = -(0.5 * (rhob + rhobp)) * km / fmax(1.0, z_kp - z_k);
        }
        const double b_d = mass_k / dt_s + sfc_impl - a_d - c_d;
        const double rhs = (mass_k / dt_s) * static_cast<double>(v[v_idx]);
        const double denom = b_d - a_d * c_prev;
        c_prev = (k + 1 < nz) ? (c_d / denom) : 0.0;
        const double d_prev = (k > 0) ? tri_d[static_cast<size_t>(k - 1) * n2 + tri_col] : 0.0;
        tri_c[t_idx] = c_prev;
        tri_d[t_idx] = (rhs - a_d * d_prev) / denom;
    }

    double x_nxt = tri_d[static_cast<size_t>(nz - 1) * n2 + tri_col];
    v[static_cast<size_t>(nz - 1) * ny_v * cfg.nx + static_cast<size_t>(j) * cfg.nx + i] = static_cast<float>(x_nxt);
    for (int32_t k = nz - 2; k >= 0; --k) {
        const size_t t_idx = static_cast<size_t>(k) * n2 + tri_col;
        x_nxt = tri_d[t_idx] - tri_c[t_idx] * x_nxt;
        v[static_cast<size_t>(k) * ny_v * cfg.nx + static_cast<size_t>(j) * cfg.nx + i] = static_cast<float>(x_nxt);
    }
}

// ===========================================================================
// PILLAR 1 (6-Class Mixed-Phase Microphysics + SLW + ICON compute_sink + Fallout)
// ===========================================================================

__device__ inline double mp_fall_speed(int sp_idx, double rho, double t_k, double q) {
    // sp_idx: 2=rain, 3=ice, 4=snow, 5=graupel (WSM6 Gamma-weighted terminal velocities)
    if (q <= 1e-12) return 0.0;
    const double rho_corr = sqrt(1.225 / fmax(rho, 0.05));
    const double rq = fmax(0.0, rho * q);
    if (sp_idx == 2) {
        // WSM6 rain Marshall-Palmer (N0r = 8e6, pvtr = 841.9 * Gamma(4.8) / 6 = 2502.8)
        const double lam_r = pow(PI_CONST * 1000.0 * 8.0e6 / rq, 0.25);
        return fmin(12.0, 2502.8 * rho_corr * pow(lam_r, -0.8));
    }
    if (sp_idx == 3) {
        // Ice crystal (ICON Heymsfield & Donner)
        return fmin(1.2, 0.32 * rho_corr * pow(fmax(q, 1e-9), 0.16));
    }
    if (sp_idx == 4) {
        // WSM6 snow (N0s(T), pvts = 11.72 * Gamma(4.41) / 6 = 23.59)
        const double n0s = fmin(2.0e8, 2.0e6 * exp(0.12 * fmax(0.0, T_FREEZE - t_k)));
        const double lam_s = pow(PI_CONST * 100.0 * n0s / rq, 0.25);
        return fmin(3.0, 23.59 * rho_corr * pow(lam_s, -0.41));
    }
    // WSM6 graupel (N0g = 4e6, rho_g = 500, pvtg = 330 * Gamma(4.8) / 6 = 981.1)
    const double lam_g = pow(PI_CONST * 500.0 * 4.0e6 / rq, 0.25);
    return fmin(16.0, 981.1 * rho_corr * pow(lam_g, -0.8));
}

__global__ void nh_microphysics_kernel(
    NhGridConfig cfg,
    NhPhysicsConfig pcfg,
    DeviceStaticGrid g,
    double dt_s,
    const float* lsm,
    float* thp, const float* pip,
    float* qv, float* qc, float* qr, float* qi, float* qs, float* qg,
    float* rain_acc, float* snow_acc, float* graupel_acc,
    float* precip_liq_rate, float* precip_ice_rate,
    float* dz_scratch, double* tri_mass, double* tri_mair
) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= cfg.ny || i >= cfg.nx) return;

    const int32_t nz = cfg.nz;
    const size_t n2 = static_cast<size_t>(cfg.ny) * cfg.nx;
    const size_t col = static_cast<size_t>(j) * cfg.nx + i;
    const double hs = g.hs[col];
    const double lsm_v = lsm ? static_cast<double>(lsm[col]) : 1.0;
    const double n_drop = (lsm_v >= 0.5) ? pcfg.cloud_num_land : pcfg.cloud_num_sea;
    constexpr double T_HOM = 236.15; // -37C homogeneous freezing threshold (SLW preserved for T > 236.15 K!)

    // Phase-pair latent heat table [src][dst] in J/kg: 0=v, 1=c, 2=r, 3=i, 4=s, 5=g
    const double lat_heat[6][6] = {
        { 0.0,  LV0,  LV0,  LS0,  LS0,  LS0},
        {-LV0,  0.0,  0.0,  LF0,  LF0,  LF0},
        {-LV0,  0.0,  0.0,  LF0,  LF0,  LF0},
        {-LS0, -LF0, -LF0,  0.0,  0.0,  0.0},
        {-LS0, -LF0, -LF0,  0.0,  0.0,  0.0},
        {-LS0, -LF0, -LF0,  0.0,  0.0,  0.0}
    };

    // Pass 1: In-situ Thermodynamics + Saturation Adjustment + Transitions (Exact Water & Energy)
    for (int32_t k = 0; k < nz; ++k) {
        const size_t idx = static_cast<size_t>(k) * n2 + col;
        double z_k, dz_k, thb, pib, th_k, pi_k, t_k, p_k, rho;
        eval_cell_thermo(cfg, g, k, col, hs, thp, pip, qv, qc, qr, qi, qs, qg,
                         z_k, dz_k, thb, pib, th_k, pi_k, t_k, p_k, rho);
        double thb_r, pib_r, dth_r, rhob_r, cs2_r;
        eval_ref_state(z_k, cfg.theta_sfc_k, cfg.p_sfc_pa, cfg.n_ref_s, thb_r, pib_r, dth_r, rhob_r, cs2_r);
        dz_scratch[idx] = static_cast<float>(dz_k);
        tri_mair[idx] = rhob_r * dz_k; // Matches dynamical FCT & tracer_mass() reference weight!

        double q[6] = {
            fmax(0.0, static_cast<double>(qv[idx])),
            fmax(0.0, static_cast<double>(qc[idx])),
            fmax(0.0, static_cast<double>(qr[idx])),
            fmax(0.0, static_cast<double>(qi[idx])),
            fmax(0.0, static_cast<double>(qs[idx])),
            fmax(0.0, static_cast<double>(qg[idx]))
        };

        // 1a. Fast melting (>0C) & homogeneous freezing (<-37C); SLW preserved for -37C..0C!
        if (t_k > T_FREEZE && q[3] > 0.0) {
            const double dm = fmin(q[3], CP_D * (t_k - T_FREEZE) / LF0);
            q[3] -= dm; q[1] += dm; t_k -= dm * LF0 / CP_D;
        }
        if (t_k <= T_HOM && q[1] > 0.0) {
            q[3] += q[1]; t_k += q[1] * LF0 / CP_D; q[1] = 0.0;
        }

        // 1b. Fast condensation/evaporation saturation adjustment (2-iter Newton)
        const bool liq_phase = (t_k > T_HOM);
        const double l_sat = liq_phase ? LV0 : LS0;
        for (int it = 0; it < 2; ++it) {
            const double qs_val = liq_phase ? qsat_water(t_k, p_k) : qsat_ice(t_k, p_k);
            const double dqs_dt = liq_phase ? dqsat_water_dt(t_k, p_k) : dqsat_ice_dt(t_k, p_k);
            const double dq = (q[0] - qs_val) / (1.0 + (l_sat / CP_D) * dqs_dt);
            if (dq > 0.0) {
                const double c = fmin(dq, q[0]);
                q[0] -= c;
                if (liq_phase) q[1] += c; else q[3] += c;
                t_k += (l_sat / CP_D) * c;
            } else if (dq < 0.0) {
                double avail = liq_phase ? q[1] : q[3];
                const double e = fmin(-dq, avail);
                q[0] += e;
                if (liq_phase) q[1] -= e; else q[3] -= e;
                t_k -= (l_sat / CP_D) * e;
            }
        }

        // 1c. Slow microphysical transfer matrix sx2x[6][6] (ICON graupel / Seifert-Beheng / WSM6)
        double sx2x[6][6] = {};
        const double rho_corr = sqrt(1.225 / fmax(rho, 0.05));
        const double k_diff = 1.41e-5 * (273.15 / t_k);

        // Warm-rain: WSM6 Liu-Daum (2004) + Seifert-Beheng (2001) autoconversion & accretion c -> r
        if (q[1] > 1e-7) {
            const double l_c = rho * q[1];
            const double l_r = rho * q[2];
            const double nc = fmax(n_drop, 2e7);
            const double x_c = l_c / nc;
            const double tau = fmin(0.99, l_r / fmax(l_c + l_r, 1e-12));
            const double tau68 = pow(tau, 0.68);
            const double phi_au = 600.0 * tau68 * pow(1.0 - tau68, 3.0);
            const double au_sb = 6.80769e18 * l_c * l_c * x_c * x_c
                               * (1.0 + phi_au / pow(1.0 - tau + 1e-3, 2.0)) * (1.225 / rho);
            sx2x[1][2] += au_sb / rho;

            // Liu & Daum (2004) thresholded autoconversion when droplets reach critical r_c0 ~ 9.0 um
            const double qc0 = (4.0 / 3.0) * PI_CONST * pow(9.0e-6, 3.0) * 1000.0 * nc / rho;
            if (q[1] > qc0) {
                const double qck1 = (0.104 * GRAVITY * 0.55 * pow(rho, 4.0 / 3.0)) / (1.71e-5 * cbrt(nc * 1000.0));
                sx2x[1][2] += qck1 * pow(q[1], 7.0 / 3.0);
            }

            if (q[2] > 1e-8) {
                const double phi_ac = pow(tau / (tau + 5.0e-4), 4.0);
                sx2x[1][2] += (5.25 * l_c * l_r * phi_ac * rho_corr) / rho;
            }
        }

        // Rain evaporation below saturation (r -> v)
        if (q[2] > 1e-9) {
            const double qsw = qsat_water(t_k, p_k);
            if (q[0] < qsw) {
                const double esw = es_water(t_k);
                const double ab = (LV0 * LV0) / (0.024 * R_V * t_k * t_k) + (R_V * t_k) / (k_diff * (P0_PA / p_k) * esw);
                const double l_r = rho * q[2];
                const double vent = 0.78 * pow(l_r, 0.50) + 0.31 * sqrt(rho_corr) * pow(l_r, 0.65);
                sx2x[2][0] += fmax(0.0, 2.0e3 * (1.0 - q[0] / qsw) * vent / (rho * fmax(ab, 1e4)));
            }
        }

        if (t_k < T_FREEZE) {
            const double qsi = qsat_ice(t_k, p_k);
            const double dtemp = T_FREEZE - t_k;

            // Cooper heterogeneous nucleation (requires supercooling >= 6C with cloud or >= 25C vapor)
            if ((t_k < T_FREEZE - 6.0 && q[1] > 1e-6) || (t_k < T_FREEZE - 25.0 && q[0] > qsi)) {
                const double ni = fmin(2.5e5, 5.0 * exp(0.304 * dtemp)) / rho;
                const double qi_tgt = ni * 1.0e-12;
                if (qi_tgt > q[3]) {
                    sx2x[0][3] += (qi_tgt - q[3]) / dt_s;
                }
            }
            // Ice crystal diffusional deposition/sublimation (v <-> i)
            if (q[3] > 1e-11) {
                const double f_dep = 1.0e-2 * (q[0] - qsi) / fmax(qsi, 1e-6) * pow(q[3], 0.33);
                if (q[0] > qsi) sx2x[0][3] += fmax(0.0, fmin(f_dep, (q[0] - qsi) / dt_s));
                else sx2x[3][0] += fmax(0.0, fmin(-f_dep, (qsi - q[0]) / dt_s));
            }
            // Ice aggregation to snow (i -> s)
            if (q[3] > 1e-6) {
                const double eff_stick = exp(0.025 * (t_k - T_FREEZE));
                sx2x[3][4] += 1.0e-3 * eff_stick * rho * q[3];
            }
            // Snow riming (c -> s) & deposition (v <-> s)
            if (q[4] > 1e-8) {
                if (q[1] > 1e-6) sx2x[1][4] += 2.2 * rho * q[4] * q[1] * rho_corr;
                const double f_sdep = 4.0e-3 * (q[0] - qsi) / fmax(qsi, 1e-6) * pow(rho * q[4], 0.6);
                if (q[0] > qsi) sx2x[0][4] += fmax(0.0, f_sdep);
                else sx2x[4][0] += fmax(0.0, -f_sdep);
            }
            // Graupel riming (c -> g) & rain freezing / collection -> graupel
            if (q[2] > 1e-7 && dtemp > 4.0) {
                // Bigg heterogeneous drop freezing
                sx2x[2][5] += 100.0 * (exp(0.66 * dtemp) - 1.0) * (rho * q[2] * q[2]) / 1000.0;
                if (q[4] > 1e-6 || q[3] > 1e-6) {
                    sx2x[2][5] += 4.0 * rho * q[2] * (q[4] + q[3]);
                }
            }
            if (q[5] > 1e-8 && q[1] > 1e-6) {
                sx2x[1][5] += 5.0 * rho * q[5] * q[1] * rho_corr;
            }
            // Snow riming -> graupel conversion when heavy cloud water present
            if (q[4] > 1e-5 && q[1] > 2e-4) {
                sx2x[4][5] += 1.5e-2 * q[4] * (rho * q[1]);
            }
        } else {
            // Melting above 0C (s -> r, g -> r)
            const double dtemp = t_k - T_FREEZE;
            if (q[4] > 1e-9) {
                sx2x[4][2] += 1.2e-3 * dtemp * pow(rho * q[4], 0.65) / rho;
            }
            if (q[5] > 1e-9) {
                sx2x[5][2] += 8.0e-4 * dtemp * pow(rho * q[5], 0.65) / rho;
            }
        }

        // ICON BSD-3 compute_sink proportional sink limiter (exact mass conservation + q >= 0)
        for (int s = 0; s < 6; ++s) {
            double total_sink = 0.0;
            for (int d = 0; d < 6; ++d) {
                if (d != s) total_sink += sx2x[s][d] * dt_s;
            }
            if (total_sink > q[s] && total_sink > 0.0) {
                const double scale = q[s] / total_sink;
                for (int d = 0; d < 6; ++d) {
                    if (d != s) sx2x[s][d] *= scale;
                }
            }
        }

        double dq[6] = {};
        double dh = 0.0;
        for (int s = 0; s < 6; ++s) {
            for (int d = 0; d < 6; ++d) {
                if (s == d) continue;
                const double flux = sx2x[s][d] * dt_s;
                dq[s] -= flux;
                dq[d] += flux;
                dh += flux * lat_heat[s][d];
            }
        }
        for (int s = 0; s < 6; ++s) {
            q[s] = fmax(0.0, q[s] + dq[s]);
            if (q[s] < 1e-13) {
                if (s != 0) q[0] += q[s]; // reclaim negligible float trace into vapor to preserve sum
                q[s] = 0.0;
            }
        }
        t_k = fmax(140.0, t_k + dh / CP_D);

        thp[idx] = static_cast<float>(t_k / pi_k - thb);
        qv[idx] = static_cast<float>(q[0]);
        qc[idx] = static_cast<float>(q[1]);
        qr[idx] = static_cast<float>(q[2]);
        qi[idx] = static_cast<float>(q[3]);
        qs[idx] = static_cast<float>(q[4]);
        qg[idx] = static_cast<float>(q[5]);
    }

    // Pass 2: Subcycled Conservative Float64 Column Fallout (qr, qi, qs, qg)
    float* fall_ptrs[4] = {qr, qi, qs, qg};
    const int fall_ids[4] = {2, 3, 4, 5};
    double acc_col[4] = {0.0, 0.0, 0.0, 0.0};

    for (int f = 0; f < 4; ++f) {
        float* qf = fall_ptrs[f];
        const int sp_id = fall_ids[f];
        double max_cfl = 0.0;
        bool any_mass = false;

        for (int32_t k = 0; k < nz; ++k) {
            const size_t idx = static_cast<size_t>(k) * n2 + col;
            const double m_air = tri_mair[idx];
            const double dz_k = static_cast<double>(dz_scratch[idx]);
            const double q_k = fmax(0.0, static_cast<double>(qf[idx]));
            tri_mass[idx] = m_air * q_k;
            if (q_k > 1e-12) {
                any_mass = true;
                const double z_k = g.zeta_s[k] + g.b_s[k] * hs;
                double thb, pib, dth, rhob, cs2;
                eval_ref_state(z_k, cfg.theta_sfc_k, cfg.p_sfc_pa, cfg.n_ref_s, thb, pib, dth, rhob, cs2);
                const double pi_k = fmax(1e-4, pib + static_cast<double>(pip[idx]));
                const double t_k = (thb + static_cast<double>(thp[idx])) * pi_k;
                const double p_k = P0_PA * pow(pi_k, 1.0 / KAPPA);
                const double rho_act = fmax(1e-5, p_k / (R_D * t_k));
                const double vt = mp_fall_speed(sp_id, rho_act, t_k, q_k);
                max_cfl = fmax(max_cfl, vt * dt_s / dz_k);
            }
        }
        if (!any_mass) continue;

        const int n_sub = (max_cfl > 0.65) ? min(512, static_cast<int>(ceil(max_cfl / 0.65))) : 1;
        const double dt_sub = dt_s / static_cast<double>(n_sub);

        for (int sub = 0; sub < n_sub; ++sub) {
            double flux_above = 0.0;
            for (int32_t k = nz - 1; k >= 0; --k) {
                const size_t idx = static_cast<size_t>(k) * n2 + col;
                const double m_air = tri_mair[idx];
                const double dz_k = static_cast<double>(dz_scratch[idx]);
                const double mk = tri_mass[idx];
                double flux_out = 0.0;
                if (mk > 1e-18) {
                    const double q_k = mk / m_air;
                    const double z_k = g.zeta_s[k] + g.b_s[k] * hs;
                    double thb, pib, dth, rhob, cs2;
                    eval_ref_state(z_k, cfg.theta_sfc_k, cfg.p_sfc_pa, cfg.n_ref_s, thb, pib, dth, rhob, cs2);
                    const double pi_k = fmax(1e-4, pib + static_cast<double>(pip[idx]));
                    const double t_k = (thb + static_cast<double>(thp[idx])) * pi_k;
                    const double p_k = P0_PA * pow(pi_k, 1.0 / KAPPA);
                    const double rho_act = fmax(1e-5, p_k / (R_D * t_k));
                    const double vt = mp_fall_speed(sp_id, rho_act, t_k, q_k);
                    const double cfl = fmin(0.95, vt * dt_sub / dz_k);
                    flux_out = mk * cfl;
                }
                tri_mass[idx] = mk + flux_above - flux_out;
                flux_above = flux_out;
            }
            acc_col[f] += flux_above;
        }

        for (int32_t k = 0; k < nz; ++k) {
            const size_t idx = static_cast<size_t>(k) * n2 + col;
            qf[idx] = static_cast<float>(fmax(0.0, tri_mass[idx] / tri_mair[idx]));
        }
    }

    if (rain_acc) rain_acc[col] = static_cast<float>(static_cast<double>(rain_acc[col]) + acc_col[0]);
    if (snow_acc) snow_acc[col] = static_cast<float>(static_cast<double>(snow_acc[col]) + acc_col[1] + acc_col[2]);
    if (graupel_acc) graupel_acc[col] = static_cast<float>(static_cast<double>(graupel_acc[col]) + acc_col[3]);
    if (precip_liq_rate) precip_liq_rate[col] = static_cast<float>(acc_col[0] / dt_s);
    if (precip_ice_rate) precip_ice_rate[col] = static_cast<float>((acc_col[1] + acc_col[2] + acc_col[3]) / dt_s);
}

// ---------------------------------------------------------------------------
// Lazy Buffer Management & Helpers
// ---------------------------------------------------------------------------

inline DeviceStaticGrid static_view_p(const NhCudaHandle* h) {
    return DeviceStaticGrid{
        h->d_zeta_w, h->d_zeta_s, h->d_dzeta,
        h->d_b_w, h->d_b_s, h->d_db_w, h->d_db_s,
        h->d_hs, h->d_msf, h->d_fcor
    };
}

__global__ void nh_init_default_surface_kernel(
    size_t n2,
    float* lat_deg, float* lon_deg, float* lsm, float* lake_mask, float* soil_type,
    float* bathy_depth_m, float* z0m_base, float* albedo_base, float* veg_frac,
    float* t_skin, float* soil_t, float* soil_m, float* sst_bulk,
    float* lake_tbot, float* lake_hmld, float* lake_ctheta
) {
    const size_t col = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (col >= n2) return;
    lat_deg[col] = 52.1f;
    lon_deg[col] = 5.18f;
    lsm[col] = 1.0f;
    lake_mask[col] = 0.0f;
    soil_type[col] = 9.0f;
    bathy_depth_m[col] = 4.5f;
    z0m_base[col] = 0.05f;
    albedo_base[col] = 0.20f;
    veg_frac[col] = 0.60f;
    t_skin[col] = 286.0f;
    for (int m = 0; m < 4; ++m) {
        soil_t[static_cast<size_t>(m) * n2 + col] = 285.0f;
        soil_m[static_cast<size_t>(m) * n2 + col] = 0.28f;
    }
    sst_bulk[col] = 284.0f;
    lake_tbot[col] = 281.0f;
    lake_hmld[col] = 2.5f;
    lake_ctheta[col] = 0.65f;
}

int32_t ensure_physics_buffers(NhCudaHandle* h) {
    if (!h) {
        nh_internal_set_error("Null NhCudaHandle in ensure_physics_buffers");
        return -1;
    }
    if (h->has_physics_alloc) return 0;

    const size_t n2 = static_cast<size_t>(h->cfg.ny) * static_cast<size_t>(h->cfg.nx);
    auto alloc_z = [&](float** p, size_t count) -> cudaError_t {
        cudaError_t e = cudaMalloc(p, count * sizeof(float));
        if (e != cudaSuccess) return e;
        return cudaMemset(*p, 0, count * sizeof(float));
    };

    NH_CUDA_CHECK(alloc_z(&h->d_rad_dth_dt, h->n_s));
    NH_CUDA_CHECK(alloc_z(&h->d_lat_deg, n2));
    NH_CUDA_CHECK(alloc_z(&h->d_lon_deg, n2));
    NH_CUDA_CHECK(alloc_z(&h->d_lsm, n2));
    NH_CUDA_CHECK(alloc_z(&h->d_lake_mask, n2));
    NH_CUDA_CHECK(alloc_z(&h->d_soil_type, n2));
    NH_CUDA_CHECK(alloc_z(&h->d_bathy_depth_m, n2));
    NH_CUDA_CHECK(alloc_z(&h->d_z0m_base, n2));
    NH_CUDA_CHECK(alloc_z(&h->d_albedo_base, n2));
    NH_CUDA_CHECK(alloc_z(&h->d_veg_frac, n2));

    NH_CUDA_CHECK(alloc_z(&h->d_t_skin, n2));
    NH_CUDA_CHECK(alloc_z(&h->d_soil_t, 4 * n2));
    NH_CUDA_CHECK(alloc_z(&h->d_soil_m, 4 * n2));
    NH_CUDA_CHECK(alloc_z(&h->d_snow_swe, n2));
    NH_CUDA_CHECK(alloc_z(&h->d_sst_bulk, n2));
    NH_CUDA_CHECK(alloc_z(&h->d_dt_cool, n2));
    NH_CUDA_CHECK(alloc_z(&h->d_dt_warm, n2));
    NH_CUDA_CHECK(alloc_z(&h->d_lake_tbot, n2));
    NH_CUDA_CHECK(alloc_z(&h->d_lake_hmld, n2));
    NH_CUDA_CHECK(alloc_z(&h->d_lake_ctheta, n2));
    NH_CUDA_CHECK(alloc_z(&h->d_lake_hice, n2));
    NH_CUDA_CHECK(alloc_z(&h->d_rain_acc, n2));
    NH_CUDA_CHECK(alloc_z(&h->d_snow_acc, n2));
    NH_CUDA_CHECK(alloc_z(&h->d_graupel_acc, n2));
    NH_CUDA_CHECK(alloc_z(&h->d_sw_down, n2));
    NH_CUDA_CHECK(alloc_z(&h->d_lw_down, n2));
    NH_CUDA_CHECK(alloc_z(&h->d_shf, n2));
    NH_CUDA_CHECK(alloc_z(&h->d_lhf, n2));
    NH_CUDA_CHECK(alloc_z(&h->d_ustar, n2));
    NH_CUDA_CHECK(alloc_z(&h->d_pblh, n2));
    NH_CUDA_CHECK(alloc_z(&h->d_t2m, n2));
    NH_CUDA_CHECK(alloc_z(&h->d_q2m, n2));
    NH_CUDA_CHECK(alloc_z(&h->d_u10m, n2));
    NH_CUDA_CHECK(alloc_z(&h->d_v10m, n2));
    NH_CUDA_CHECK(alloc_z(&h->d_precip_liq_rate, n2));
    NH_CUDA_CHECK(alloc_z(&h->d_precip_ice_rate, n2));
    NH_CUDA_CHECK(alloc_z(&h->d_sfc_drag_coeff, n2));

    const uint32_t blk = 256;
    const uint32_t grd = static_cast<uint32_t>((n2 + blk - 1) / blk);
    nh_init_default_surface_kernel<<<grd, blk>>>(
        n2, h->d_lat_deg, h->d_lon_deg, h->d_lsm, h->d_lake_mask, h->d_soil_type,
        h->d_bathy_depth_m, h->d_z0m_base, h->d_albedo_base, h->d_veg_frac,
        h->d_t_skin, h->d_soil_t, h->d_soil_m, h->d_sst_bulk,
        h->d_lake_tbot, h->d_lake_hmld, h->d_lake_ctheta
    );
    NH_CUDA_CHECK(cudaGetLastError());

    if (h->pcfg.cloud_num_land <= 0.0) h->pcfg.cloud_num_land = 200.0e6;
    if (h->pcfg.cloud_num_sea <= 0.0) h->pcfg.cloud_num_sea = 80.0e6;
    if (h->pcfg.co2_ppm <= 0.0) h->pcfg.co2_ppm = 425.0;
    if (h->pcfg.julian_day <= 0.0) h->pcfg.julian_day = 172.0;

    h->has_physics_alloc = 1;
    h->rad_initialized = 0;
    h->rad_timer_s = 0.0;
    return 0;
}

} // namespace

void nh_physics_free_buffers(NhCudaHandle* h) {
    if (!h) return;
    auto cf = [](float*& p) { if (p) { cudaFree(p); p = nullptr; } };
    cf(h->d_rad_dth_dt);
    cf(h->d_lat_deg); cf(h->d_lon_deg); cf(h->d_lsm); cf(h->d_lake_mask);
    cf(h->d_soil_type); cf(h->d_bathy_depth_m); cf(h->d_z0m_base);
    cf(h->d_albedo_base); cf(h->d_veg_frac);
    cf(h->d_t_skin); cf(h->d_soil_t); cf(h->d_soil_m); cf(h->d_snow_swe);
    cf(h->d_sst_bulk); cf(h->d_dt_cool); cf(h->d_dt_warm);
    cf(h->d_lake_tbot); cf(h->d_lake_hmld); cf(h->d_lake_ctheta); cf(h->d_lake_hice);
    cf(h->d_rain_acc); cf(h->d_snow_acc); cf(h->d_graupel_acc);
    cf(h->d_sw_down); cf(h->d_lw_down); cf(h->d_shf); cf(h->d_lhf);
    cf(h->d_ustar); cf(h->d_pblh); cf(h->d_t2m); cf(h->d_q2m); cf(h->d_u10m); cf(h->d_v10m);
    cf(h->d_precip_liq_rate); cf(h->d_precip_ice_rate); cf(h->d_sfc_drag_coeff);
    h->has_physics_alloc = 0;
    h->rad_initialized = 0;
    h->rad_timer_s = 0.0;
}

extern "C" {

int32_t nh_cuda_set_physics_config(NhCudaHandle* h, const NhPhysicsConfig* pcfg) {
    if (!h || !pcfg) {
        nh_internal_set_error("Null pointer in nh_cuda_set_physics_config");
        return -1;
    }
    h->pcfg = *pcfg;
    if (h->pcfg.cloud_num_land <= 0.0) h->pcfg.cloud_num_land = 200.0e6;
    if (h->pcfg.cloud_num_sea <= 0.0) h->pcfg.cloud_num_sea = 80.0e6;
    if (h->pcfg.co2_ppm <= 0.0) h->pcfg.co2_ppm = 425.0;
    if (h->pcfg.julian_day <= 0.0) h->pcfg.julian_day = 172.0;
    h->rad_initialized = 0;
    h->rad_timer_s = 0.0;
    return ensure_physics_buffers(h);
}

int32_t nh_cuda_set_surface_static(NhCudaHandle* h, const NhSurfaceStaticPtrs* sstatic) {
    if (!h || !sstatic) {
        nh_internal_set_error("Null pointer in nh_cuda_set_surface_static");
        return -1;
    }
    if (ensure_physics_buffers(h) != 0) return -1;
    const size_t n2 = static_cast<size_t>(h->cfg.ny) * static_cast<size_t>(h->cfg.nx);
    auto cp = [&](float* dst, const float* src) -> int32_t {
        if (!src) return 0;
        NH_CUDA_CHECK(cudaMemcpy(dst, src, n2 * sizeof(float), cudaMemcpyHostToDevice));
        return 0;
    };
    if (cp(h->d_lat_deg, sstatic->lat_deg) != 0) return -1;
    if (cp(h->d_lon_deg, sstatic->lon_deg) != 0) return -1;
    if (cp(h->d_lsm, sstatic->lsm) != 0) return -1;
    if (cp(h->d_lake_mask, sstatic->lake_mask) != 0) return -1;
    if (cp(h->d_soil_type, sstatic->soil_type) != 0) return -1;
    if (cp(h->d_bathy_depth_m, sstatic->bathy_depth_m) != 0) return -1;
    if (cp(h->d_z0m_base, sstatic->z0m_base) != 0) return -1;
    if (cp(h->d_albedo_base, sstatic->albedo_base) != 0) return -1;
    if (cp(h->d_veg_frac, sstatic->veg_frac) != 0) return -1;
    h->rad_initialized = 0;
    return 0;
}

int32_t nh_cuda_upload_surface(NhCudaHandle* h, const NhConstSurfaceStatePtrs* src) {
    if (!h || !src) {
        nh_internal_set_error("Null pointer in nh_cuda_upload_surface");
        return -1;
    }
    if (ensure_physics_buffers(h) != 0) return -1;
    const size_t n2 = static_cast<size_t>(h->cfg.ny) * static_cast<size_t>(h->cfg.nx);
    auto cp = [&](float* dst, const float* s, size_t cnt) -> int32_t {
        if (!s) return 0;
        NH_CUDA_CHECK(cudaMemcpy(dst, s, cnt * sizeof(float), cudaMemcpyHostToDevice));
        return 0;
    };
    if (cp(h->d_t_skin, src->t_skin, n2) != 0) return -1;
    if (cp(h->d_soil_t, src->soil_t, 4 * n2) != 0) return -1;
    if (cp(h->d_soil_m, src->soil_m, 4 * n2) != 0) return -1;
    if (cp(h->d_snow_swe, src->snow_swe, n2) != 0) return -1;
    if (cp(h->d_sst_bulk, src->sst_bulk, n2) != 0) return -1;
    if (cp(h->d_dt_cool, src->dt_cool, n2) != 0) return -1;
    if (cp(h->d_dt_warm, src->dt_warm, n2) != 0) return -1;
    if (cp(h->d_lake_tbot, src->lake_tbot, n2) != 0) return -1;
    if (cp(h->d_lake_hmld, src->lake_hmld, n2) != 0) return -1;
    if (cp(h->d_lake_ctheta, src->lake_ctheta, n2) != 0) return -1;
    if (cp(h->d_lake_hice, src->lake_hice, n2) != 0) return -1;
    if (cp(h->d_rain_acc, src->rain_acc, n2) != 0) return -1;
    if (cp(h->d_snow_acc, src->snow_acc, n2) != 0) return -1;
    if (cp(h->d_graupel_acc, src->graupel_acc, n2) != 0) return -1;
    if (cp(h->d_sw_down, src->sw_down, n2) != 0) return -1;
    if (cp(h->d_lw_down, src->lw_down, n2) != 0) return -1;
    if (cp(h->d_shf, src->shf, n2) != 0) return -1;
    if (cp(h->d_lhf, src->lhf, n2) != 0) return -1;
    if (cp(h->d_ustar, src->ustar, n2) != 0) return -1;
    if (cp(h->d_pblh, src->pblh, n2) != 0) return -1;
    if (cp(h->d_t2m, src->t2m, n2) != 0) return -1;
    if (cp(h->d_q2m, src->q2m, n2) != 0) return -1;
    if (cp(h->d_u10m, src->u10m, n2) != 0) return -1;
    if (cp(h->d_v10m, src->v10m, n2) != 0) return -1;
    return 0;
}

int32_t nh_cuda_download_surface(NhCudaHandle* h, const NhSurfaceStatePtrs* dst) {
    if (!h || !dst) {
        nh_internal_set_error("Null pointer in nh_cuda_download_surface");
        return -1;
    }
    if (!h->has_physics_alloc) {
        nh_internal_set_error("Physics buffers not allocated before nh_cuda_download_surface");
        return -1;
    }
    const size_t n2 = static_cast<size_t>(h->cfg.ny) * static_cast<size_t>(h->cfg.nx);
    auto cp = [&](float* d, const float* src, size_t cnt) -> int32_t {
        if (!d) return 0;
        NH_CUDA_CHECK(cudaMemcpy(d, src, cnt * sizeof(float), cudaMemcpyDeviceToHost));
        return 0;
    };
    if (cp(dst->t_skin, h->d_t_skin, n2) != 0) return -1;
    if (cp(dst->soil_t, h->d_soil_t, 4 * n2) != 0) return -1;
    if (cp(dst->soil_m, h->d_soil_m, 4 * n2) != 0) return -1;
    if (cp(dst->snow_swe, h->d_snow_swe, n2) != 0) return -1;
    if (cp(dst->sst_bulk, h->d_sst_bulk, n2) != 0) return -1;
    if (cp(dst->dt_cool, h->d_dt_cool, n2) != 0) return -1;
    if (cp(dst->dt_warm, h->d_dt_warm, n2) != 0) return -1;
    if (cp(dst->lake_tbot, h->d_lake_tbot, n2) != 0) return -1;
    if (cp(dst->lake_hmld, h->d_lake_hmld, n2) != 0) return -1;
    if (cp(dst->lake_ctheta, h->d_lake_ctheta, n2) != 0) return -1;
    if (cp(dst->lake_hice, h->d_lake_hice, n2) != 0) return -1;
    if (cp(dst->rain_acc, h->d_rain_acc, n2) != 0) return -1;
    if (cp(dst->snow_acc, h->d_snow_acc, n2) != 0) return -1;
    if (cp(dst->graupel_acc, h->d_graupel_acc, n2) != 0) return -1;
    if (cp(dst->sw_down, h->d_sw_down, n2) != 0) return -1;
    if (cp(dst->lw_down, h->d_lw_down, n2) != 0) return -1;
    if (cp(dst->shf, h->d_shf, n2) != 0) return -1;
    if (cp(dst->lhf, h->d_lhf, n2) != 0) return -1;
    if (cp(dst->ustar, h->d_ustar, n2) != 0) return -1;
    if (cp(dst->pblh, h->d_pblh, n2) != 0) return -1;
    if (cp(dst->t2m, h->d_t2m, n2) != 0) return -1;
    if (cp(dst->q2m, h->d_q2m, n2) != 0) return -1;
    if (cp(dst->u10m, h->d_u10m, n2) != 0) return -1;
    if (cp(dst->v10m, h->d_v10m, n2) != 0) return -1;
    return 0;
}

int32_t nh_cuda_step_physics(NhCudaHandle* h, double dt_s) {
    if (!h) {
        nh_internal_set_error("Null NhCudaHandle in nh_cuda_step_physics");
        return -1;
    }
    if (!(dt_s > 0.0)) {
        nh_internal_set_error("dt_s must be positive in nh_cuda_step_physics");
        return -1;
    }
    if (ensure_physics_buffers(h) != 0) return -1;

    const DeviceStaticGrid g = static_view_p(h);
    const size_t n2 = static_cast<size_t>(h->cfg.ny) * static_cast<size_t>(h->cfg.nx);
    const dim3 block(16, 16);
    const dim3 grid((h->cfg.nx + 15) / 16, (h->cfg.ny + 15) / 16);

    // 1. Cloud-interactive two-stream SW/LW Radiation (ecRad)
    if (h->pcfg.enable_radiation) {
        if (!h->rad_initialized || h->pcfg.rad_interval_s <= 0.0 ||
            (h->rad_timer_s + 1e-9 >= h->pcfg.rad_interval_s)) {
            nh_radiation_ecrad_kernel<<<grid, block>>>(
                h->cfg, h->pcfg, g,
                h->d_thp, h->d_pip, h->d_qv, h->d_qc, h->d_qr, h->d_qi, h->d_qs, h->d_qg,
                h->d_lat_deg, h->d_lon_deg, h->d_lsm, h->d_albedo_base, h->d_snow_swe,
                h->d_lake_mask, h->d_lake_hice, h->d_t_skin,
                h->d_rad_dth_dt, h->d_sw_down, h->d_lw_down,
                h->d_div, h->d_tri_c, h->d_tri_d
            );
            NH_CUDA_CHECK(cudaGetLastError());
            h->rad_initialized = 1;
            h->rad_timer_s = 0.0;
        }
        h->rad_timer_s += dt_s;
        nh_apply_rad_kernel<<<grid, block>>>(h->cfg, dt_s, h->d_rad_dth_dt, h->d_thp);
        NH_CUDA_CHECK(cudaGetLastError());
    }

    // 2. Surface Layer, Land LSM, Ocean COARE 3.6 + Skin, FLake
    if (h->pcfg.enable_surface) {
        nh_surface_and_lsm_kernel<<<grid, block>>>(
            h->cfg, h->pcfg, g, dt_s,
            h->d_u, h->d_v, h->d_thp, h->d_pip,
            h->d_qv, h->d_qc, h->d_qr, h->d_qi, h->d_qs, h->d_qg,
            h->d_lsm, h->d_lake_mask, h->d_soil_type, h->d_bathy_depth_m,
            h->d_z0m_base, h->d_albedo_base, h->d_veg_frac,
            h->d_sw_down, h->d_lw_down, h->d_precip_liq_rate, h->d_precip_ice_rate,
            h->d_t_skin, h->d_soil_t, h->d_soil_m, h->d_snow_swe,
            h->d_sst_bulk, h->d_dt_cool, h->d_dt_warm,
            h->d_lake_tbot, h->d_lake_hmld, h->d_lake_ctheta, h->d_lake_hice,
            h->d_shf, h->d_lhf, h->d_ustar, h->d_t2m, h->d_q2m, h->d_u10m, h->d_v10m,
            h->d_sfc_drag_coeff
        );
        NH_CUDA_CHECK(cudaGetLastError());
    } else {
        NH_CUDA_CHECK(cudaMemset(h->d_shf, 0, n2 * sizeof(float)));
        NH_CUDA_CHECK(cudaMemset(h->d_lhf, 0, n2 * sizeof(float)));
        NH_CUDA_CHECK(cudaMemset(h->d_sfc_drag_coeff, 0, n2 * sizeof(float)));
    }

    // 3. YSU Nonlocal PBL + Shallow EDMFm Mass-Flux + Implicit Vertical Diffusion
    if (h->pcfg.enable_pbl) {
        nh_pbl_scalar_kernel<<<grid, block>>>(
            h->cfg, g, dt_s,
            h->d_u, h->d_v, h->d_pip,
            h->d_lsm, h->d_shf, h->d_lhf, h->d_ustar,
            h->d_thp, h->d_qv, h->d_qc, h->d_qi, h->d_pblh,
            h->d_div, h->d_q_tmp, h->d_tri_c, h->d_tri_d
        );
        NH_CUDA_CHECK(cudaGetLastError());

        nh_pbl_momentum_u_kernel<<<grid, block>>>(
            h->cfg, g, dt_s, 0, h->d_pblh, h->d_ustar, h->d_sfc_drag_coeff,
            h->d_u, h->d_tri_c, h->d_tri_d
        );
        nh_pbl_momentum_u_kernel<<<dim3(1, (h->cfg.ny + 15) / 16), dim3(1, 16)>>>(
            h->cfg, g, dt_s, 1, h->d_pblh, h->d_ustar, h->d_sfc_drag_coeff,
            h->d_u, h->d_tri_c, h->d_tri_d
        );
        nh_pbl_momentum_v_kernel<<<grid, block>>>(
            h->cfg, g, dt_s, 0, h->d_pblh, h->d_ustar, h->d_sfc_drag_coeff,
            h->d_v, h->d_tri_c, h->d_tri_d
        );
        nh_pbl_momentum_v_kernel<<<dim3((h->cfg.nx + 15) / 16, 1), dim3(16, 1)>>>(
            h->cfg, g, dt_s, 1, h->d_pblh, h->d_ustar, h->d_sfc_drag_coeff,
            h->d_v, h->d_tri_c, h->d_tri_d
        );
        NH_CUDA_CHECK(cudaGetLastError());
    }

    // 4. 6-Class Mixed-Phase Microphysics + SLW Preservation + Subcycled Fallout
    if (h->pcfg.enable_microphysics && h->cfg.has_moist) {
        nh_microphysics_kernel<<<grid, block>>>(
            h->cfg, h->pcfg, g, dt_s, h->d_lsm,
            h->d_thp, h->d_pip, h->d_qv, h->d_qc, h->d_qr, h->d_qi, h->d_qs, h->d_qg,
            h->d_rain_acc, h->d_snow_acc, h->d_graupel_acc,
            h->d_precip_liq_rate, h->d_precip_ice_rate,
            h->d_div, h->d_tri_c, h->d_tri_d
        );
        NH_CUDA_CHECK(cudaGetLastError());
    }

    // Advance solar UTC clock continuously across diurnal physics steps
    h->pcfg.utc_seconds += dt_s;
    while (h->pcfg.utc_seconds >= 86400.0) {
        h->pcfg.utc_seconds -= 86400.0;
        h->pcfg.julian_day += 1.0;
        if (h->pcfg.julian_day > 366.0) h->pcfg.julian_day -= 365.0;
    }

    NH_CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
}

} // extern "C"
