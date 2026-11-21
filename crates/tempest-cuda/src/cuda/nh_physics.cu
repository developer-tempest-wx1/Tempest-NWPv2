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

__device__ inline float es_water(float t_k) {
    const float t = fminf(fmaxf(t_k, 173.15f), 373.15f);
    return 611.21f * expf(17.502f * (t - T_FREEZE_F) / (t - 32.19f));
}

__device__ inline float es_ice(float t_k) {
    const float t = fminf(fmaxf(t_k, 150.0f), T_FREEZE_F);
    return 611.21f * expf(22.587f * (t - T_FREEZE_F) / (t + 0.7f));
}

__device__ inline float qsat_from_es(float es, float p_pa) {
    const float p = fmaxf(p_pa, 100.0f);
    const float ec = fminf(es, 0.90f * p);
    return EPS_MW_F * ec / fmaxf(p - (1.0f - EPS_MW_F) * ec, 100.0f);
}

__device__ inline float qsat_water(float t_k, float p_pa) {
    return qsat_from_es(es_water(t_k), p_pa);
}

__device__ inline float qsat_ice(float t_k, float p_pa) {
    return qsat_from_es(es_ice(t_k), p_pa);
}

__device__ inline float dqsat_water_dt(float t_k, float p_pa) {
    const float t = fminf(fmaxf(t_k, 173.15f), 373.15f);
    const float es = fminf(es_water(t), 0.90f * fmaxf(p_pa, 100.0f));
    const float qs = qsat_from_es(es, p_pa);
    const float dlne = 4217.28192f / ((t - 32.19f) * (t - 32.19f));
    return qs * p_pa / fmaxf(p_pa - (1.0f - EPS_MW_F) * es, 100.0f) * dlne;
}

__device__ inline float dqsat_ice_dt(float t_k, float p_pa) {
    const float t = fminf(fmaxf(t_k, 150.0f), T_FREEZE_F);
    const float es = fminf(es_ice(t), 0.90f * fmaxf(p_pa, 100.0f));
    const float qs = qsat_from_es(es, p_pa);
    const float dlne = 6185.45015f / ((t + 0.7f) * (t + 0.7f));
    return qs * p_pa / fmaxf(p_pa - (1.0f - EPS_MW_F) * es, 100.0f) * dlne;
}

__device__ inline void eval_cell_thermo(
    const NhGridConfig& cfg,
    const DeviceStaticGrid& g,
    int32_t k, int32_t col, float hs, float pi_sfc_f,
    const float* thp, const float* pip,
    const float* qv, const float* qc, const float* qr,
    const float* qi, const float* qs, const float* qg,
    float& z_s, float& dz_k,
    float& thb, float& pib,
    float& th_tot, float& pi_tot,
    float& t_k, float& p_pa, float& rho
) {
    const size_t n2 = static_cast<size_t>(cfg.ny) * cfg.nx;
    const size_t idx = static_cast<size_t>(k) * n2 + col;
    z_s = static_cast<float>(g.zeta_s[k]) + static_cast<float>(g.b_s[k]) * hs;
    dz_k = fmaxf(1.0f, (1.0f + static_cast<float>(g.db_s[k]) * hs) * static_cast<float>(g.dzeta[k]));
    eval_theta_pi_bar_f32(
        z_s, static_cast<float>(cfg.theta_sfc_k), pi_sfc_f, static_cast<float>(cfg.n_ref_s), thb, pib
    );
    th_tot = fmaxf(50.0f, thb + thp[idx]);
    pi_tot = fmaxf(1e-4f, pib + pip[idx]);
    t_k = fmaxf(140.0f, th_tot * pi_tot);
    p_pa = P0_PA_F * powf(pi_tot, 1.0f / KAPPA_F);
    float qv_v = 0.0f, qhyd = 0.0f;
    if (cfg.has_moist) {
        if (qv) qv_v = fmaxf(0.0f, qv[idx]);
        if (qc) qhyd += fmaxf(0.0f, qc[idx]);
        if (qr) qhyd += fmaxf(0.0f, qr[idx]);
        if (qi) qhyd += fmaxf(0.0f, qi[idx]);
        if (qs) qhyd += fmaxf(0.0f, qs[idx]);
        if (qg) qhyd += fmaxf(0.0f, qg[idx]);
    }
    const float tv = t_k * fmaxf(0.5f, 1.0f + EPS_V_F * qv_v - qhyd);
    rho = fmaxf(1e-5f, p_pa / (R_D_F * tv));
}

// ===========================================================================
// PILLAR 3 (Radiation): ECMWF ecRad Delta-Eddington + Meador-Weaver + Adding
// ===========================================================================

__device__ inline float planck_frac_above_nu(float nu_cm, float t_k) {
    // Widger & Woodall (1976) normalized Planck integral fraction for wavenumber > nu_cm [cm^-1]
    // c2 = h*c/k_B = 1.438775 cm*K
    const float x = 1.438775f * nu_cm / fmaxf(t_k, 140.0f);
    if (x <= 1e-4f) return 1.0f;
    if (x >= 45.0f) return 0.0f;
    float sum = 0.0f;
    const float x2 = x * x, x3 = x2 * x;
    for (int n = 1; n <= 6; ++n) {
        const float dn = static_cast<float>(n);
        const float inv_n = 1.0f / dn;
        sum += expf(-dn * x) * inv_n * (x3 + 3.0f * x2 * inv_n + 6.0f * x * inv_n * inv_n + 6.0f * inv_n * inv_n * inv_n);
    }
    // 15 / pi^4 = 0.15398973382026508
    return fminf(1.0f, fmaxf(0.0f, 0.15398973f * sum));
}

__device__ inline float lw_band_planck_flux(int band, float t_k) {
    const float tc = fmaxf(t_k, 140.0f);
    const float tc2 = tc * tc;
    const float b_tot = STEFAN_BOLTZMANN_F * tc2 * tc2;
    const float p580  = planck_frac_above_nu(580.0f, t_k);
    const float p800  = planck_frac_above_nu(800.0f, t_k);
    const float p1250 = planck_frac_above_nu(1250.0f, t_k);
    const float f1 = fmaxf(0.0f, p580 - p800);
    const float f2 = fmaxf(0.0f, p800 - p1250);
    const float f0 = fmaxf(0.0f, 1.0f - f1 - f2);
    if (band == 0) return b_tot * f0;
    if (band == 1) return b_tot * f1;
    return b_tot * f2;
}

__device__ inline void sw_layer_optics(
    int band, float p_pa, float t_k, float dz, float rho,
    float qv, float qc, float qr, float qi, float qs,
    float n_drop_m3, float co2_ppm,
    float mu0, float& r_diff, float& t_diff, float& r_dir, float& t_dir_diff, float& t_dir_dir
) {
    const float dp = rho * GRAVITY_F * dz;
    const float dm_air = dp / GRAVITY_F;
    const float dm_wv = fmaxf(0.0f, qv) * dm_air;
    const float p_scale = powf(fminf(fmaxf(p_pa / P0_PA_F, 1e-4f), 1.2f), 0.85f) * sqrtf(273.15f / fmaxf(t_k, 150.0f));
    const float dm_wv_p = dm_wv * p_scale;

    float od_gas = 0.0f, od_sca_ray = 0.0f, od_cld = 0.0f, ssa_cld = 0.999f, g_cld = 0.85f;
    if (band == 0) {
        od_sca_ray = 0.145f * (dp / 101325.0f);
        const float ln_p = logf(fmaxf(p_pa, 50.0f) / 3000.0f);
        const float o3_prof = expf(-0.5f * ln_p * ln_p / (1.1f * 1.1f));
        od_gas = 0.025f * (dp / 101325.0f) + 0.032f * o3_prof * (dp / 8000.0f);
    } else {
        od_sca_ray = 0.012f * (dp / 101325.0f);
        od_gas = 0.0085f * dm_wv_p + 0.00018f * (co2_ppm / 425.0f) * dm_air * p_scale;
    }

    const float lwc = rho * fmaxf(0.0f, qc + 0.15f * qr);
    const float iwc = rho * fmaxf(0.0f, qi + 0.35f * qs);
    float tau_tot = od_gas + od_sca_ray;
    float sca_tot = od_sca_ray;
    float g_sca_tot = 0.0f;

    if (lwc > 1e-8f) {
        const float re_l = fminf(50.0e-6f, fmaxf(3.0e-6f,
            cbrtf(3.0f * lwc / (4.0f * PI_CONST_F * 1000.0f * 0.8f * fmaxf(n_drop_m3, 2e7f))) * 1.07f));
        const float lwp = lwc * dz;
        const float tau_l = 1.5f * lwp / (1000.0f * re_l) * (1.0f + 1.1e-6f / re_l);
        const float ssa_l = (band == 0) ? 0.9999f : fminf(0.998f, fmaxf(0.955f, 0.993f - 120.0f * re_l));
        const float g_l = (band == 0) ? 0.855f : 0.820f;
        od_cld += tau_l;
        ssa_cld = ssa_l;
        g_cld = g_l;
        tau_tot += tau_l;
        sca_tot += tau_l * ssa_l;
        g_sca_tot += tau_l * ssa_l * g_l;
    }
    if (iwc > 1e-8f) {
        const float re_um = fminf(100.0f, fmaxf(8.0f, 28.0f + 18.0f * powf(iwc * 1000.0f, 0.25f)));
        const float iwp = iwc * dz;
        const float tau_i = iwp * fmaxf(0.002f, -0.0003f + 2.52f / re_um) * 1000.0f;
        const float ssa_i = (band == 0) ? 0.9995f : fminf(0.995f, fmaxf(0.94f, 0.990f - 0.0004f * re_um));
        const float g_i = fminf(0.88f, fmaxf(0.75f, 0.77f + 0.0005f * re_um));
        tau_tot += tau_i;
        sca_tot += tau_i * ssa_i;
        g_sca_tot += tau_i * ssa_i * g_i;
    }
    (void)od_cld; (void)ssa_cld; (void)g_cld;

    if (tau_tot < 1e-9f) {
        r_diff = 0.0f; t_diff = 1.0f; r_dir = 0.0f; t_dir_diff = 0.0f; t_dir_dir = 1.0f;
        return;
    }

    const float ssa = fminf(0.99999f, sca_tot / tau_tot);
    const float asym = (sca_tot > 1e-12f) ? fminf(0.95f, fmaxf(0.0f, g_sca_tot / sca_tot)) : 0.0f;

    // Delta-Eddington scaling (ecrad/radiation/radiation_delta_eddington.h)
    const float f = asym * asym;
    const float denom = fmaxf(1.0f - ssa * f, 1e-12f);
    const float od = tau_tot * denom;
    const float ssa_s = fminf(0.99999f, ssa * (1.0f - f) / denom);
    const float g_s = asym / (1.0f + asym);

    // Zdunkowski PIFM two-stream coefficients (ecrad/radiation_two_stream.F90)
    const float g1 = (8.0f - ssa_s * (5.0f + 3.0f * g_s)) * 0.25f;
    const float g2 = 3.0f * ssa_s * (1.0f - g_s) * 0.25f;
    const float g3 = (2.0f - 3.0f * g_s * mu0) * 0.25f;
    const float g4 = 1.0f - g3;
    const float a1 = g1 * g4 + g2 * g3;
    const float a2 = g1 * g3 + g2 * g4;
    const float k_exp = sqrtf(fmaxf((g1 - g2) * (g1 + g2), 1e-12f));

    float mu0_loc = mu0;
    if (fabsf(1.0f - k_exp * mu0_loc) < 1e-4f) {
        mu0_loc *= (1.0f - 1e-3f);
    }
    const float od_over_mu0 = fminf(50.0f, fmaxf(0.0f, od / mu0_loc));
    const float exp0 = expf(-od_over_mu0);
    t_dir_dir = exp0;

    const float k_mu0 = k_exp * mu0_loc;
    const float k_g3 = k_exp * g3;
    const float k_g4 = k_exp * g4;
    const float exp1 = expf(-fminf(50.0f, k_exp * od));
    const float exp2 = exp1 * exp1;
    const float k2e = 2.0f * k_exp * exp1;
    const float rtf = 1.0f / fmaxf(k_exp + g1 + (k_exp - g1) * exp2, 1e-12f);

    r_diff = fminf(1.0f, fmaxf(0.0f, g2 * (1.0f - exp2) * rtf));
    t_diff = fminf(1.0f - r_diff, fmaxf(0.0f, k2e * rtf));

    const float rtf_dir = mu0_loc * ssa_s * rtf / (1.0f - k_mu0 * k_mu0);
    r_dir = rtf_dir * ((1.0f - k_mu0) * (a2 + k_g3) - (1.0f + k_mu0) * (a2 - k_g3) * exp2 - k2e * (g3 - a2 * mu0_loc) * exp0);
    t_dir_diff = rtf_dir * (k2e * (g4 + a1 * mu0_loc) - exp0 * ((1.0f + k_mu0) * (a1 + k_g4) - (1.0f - k_mu0) * (a1 - k_g4) * exp2));
    r_dir = fminf(mu0_loc, fmaxf(0.0f, r_dir));
    t_dir_diff = fminf(mu0_loc - r_dir, fmaxf(0.0f, t_dir_diff));
}

__device__ inline void lw_layer_optics(
    int band, float p_pa, float t_k, float t_bot, float t_top, float dz, float rho,
    float qv, float qc, float qr, float qi, float qs,
    float n_drop_m3, float co2_ppm,
    float& ref, float& trans, float& src_up, float& src_dn
) {
    const float dp = rho * GRAVITY_F * dz;
    const float dm_air = dp / GRAVITY_F;
    const float dm_wv = fmaxf(0.0f, qv) * dm_air;
    const float p_scale = powf(fminf(fmaxf(p_pa / P0_PA_F, 1e-4f), 1.2f), 0.80f) * sqrtf(273.15f / fmaxf(t_k, 150.0f));
    const float dm_wv_p = dm_wv * p_scale;
    const float e_vap = fmaxf(0.0f, p_pa * qv / (EPS_MW_F + (1.0f - EPS_MW_F) * qv));

    float od_abs = 0.0f, od_sca = 0.0f, g_sca = 0.0f;
    if (band == 0) {
        od_abs = 0.42f * dm_wv_p + 0.015f * dm_wv * (e_vap / 1000.0f + 0.25f);
    } else if (band == 1) {
        od_abs = 0.00085f * (co2_ppm / 425.0f) * dm_air * powf(fmaxf(p_pa / P0_PA_F, 1e-4f), 0.70f)
               + 0.035f * dm_wv_p;
    } else {
        od_abs = 0.0065f * dm_wv * (e_vap / 1000.0f + 0.25f)
               + 0.000015f * (co2_ppm / 425.0f) * dm_air * p_scale;
    }

    const float lwc = rho * fmaxf(0.0f, qc + 0.25f * qr);
    const float iwc = rho * fmaxf(0.0f, qi + 0.40f * qs);
    if (lwc > 1e-8f) {
        const float lwp = lwc * dz;
        const float re_um = fminf(40.0f, fmaxf(4.0f,
            1e6f * cbrtf(3.0f * lwc / (4.0f * PI_CONST_F * 1000.0f * 0.8f * fmaxf(n_drop_m3, 2e7f))) * 1.07f));
        const float od_l = lwp * (90.0f + 280.0f / re_um);
        const float ssa_l = (band == 2) ? 0.30f : 0.18f;
        od_abs += od_l * (1.0f - ssa_l);
        od_sca += od_l * ssa_l;
        g_sca += od_l * ssa_l * 0.72f;
    }
    if (iwc > 1e-8f) {
        const float iwp = iwc * dz;
        const float re_um = fminf(100.0f, fmaxf(8.0f, 28.0f + 18.0f * powf(iwc * 1000.0f, 0.25f)));
        const float od_i = iwp * (10.0f + 1850.0f / re_um);
        const float ssa_i = 0.22f;
        od_abs += od_i * (1.0f - ssa_i);
        od_sca += od_i * ssa_i;
        g_sca += od_i * ssa_i * 0.76f;
    }

    const float od = od_abs + od_sca;
    const float ssa = (od > 1e-12f) ? fminf(0.99f, od_sca / od) : 0.0f;
    const float asym = (od_sca > 1e-12f) ? fminf(0.95f, g_sca / od_sca) : 0.0f;

    // Elsasser diffusivity D = 1.66 (ecrad/radiation_two_stream.F90)
    constexpr float D_LW = 1.66f;
    const float g1 = D_LW * (1.0f - 0.5f * ssa * (1.0f + asym));
    const float g2 = D_LW * 0.5f * ssa * (1.0f - asym);
    const float k_exp = sqrtf(fmaxf((g1 - g2) * (g1 + g2), 1e-12f));

    const float p_top = lw_band_planck_flux(band, t_top);
    const float p_bot = lw_band_planck_flux(band, t_bot);

    if (od > 1.0e-3f) {
        const float exp1 = expf(-fminf(50.0f, k_exp * od));
        const float exp2 = exp1 * exp1;
        const float rtf = 1.0f / fmaxf(k_exp + g1 + (k_exp - g1) * exp2, 1e-12f);
        ref = g2 * (1.0f - exp2) * rtf;
        trans = 2.0f * k_exp * exp1 * rtf;
        const float coeff = (p_bot - p_top) / fmaxf(od * (g1 + g2), 1e-12f);
        const float c_ut = coeff + p_top;
        const float c_ub = coeff + p_bot;
        const float c_dt = -coeff + p_top;
        const float c_db = -coeff + p_bot;
        src_up = fmaxf(0.0f, c_ut - ref * c_dt - trans * c_ub);
        src_dn = fmaxf(0.0f, c_db - ref * c_ub - trans * c_dt);
    } else {
        ref = g2 * od;
        trans = (1.0f - k_exp * od) / (1.0f + od * (g1 - k_exp));
        trans = fminf(1.0f - ref, fmaxf(0.0f, trans));
        const float s_iso = fmaxf(0.0f, 1.0f - ref - trans) * 0.5f * (p_top + p_bot);
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
    const float* sea_ice_frac, const float* sea_ice_tsfc, const float* sea_ice_albedo,
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

    const float pi_sfc_f = powf(static_cast<float>(cfg.p_sfc_pa) / P0_PA_F, KAPPA_F);
    const float hs = static_cast<float>(g.hs[col]);
    const float lat_v = lat_deg ? lat_deg[col] : 52.1f;
    const float lon_v = lon_deg ? lon_deg[col] : 5.18f;
    const float lsm_v = lsm ? lsm[col] : 1.0f;
    const float snow_v = snow_swe ? snow_swe[col] : 0.0f;
    const float lake_v = lake_mask ? lake_mask[col] : 0.0f;
    const float hice_v = lake_hice ? lake_hice[col] : 0.0f;
    const float sic_v = (lsm_v < 0.5f && lake_v < 0.5f && sea_ice_frac)
                      ? fminf(1.0f, fmaxf(0.0f, sea_ice_frac[col])) : 0.0f;
    const float alb0 = albedo_base ? albedo_base[col] : ((lsm_v >= 0.5f) ? 0.20f : 0.06f);
    const float n_drop = static_cast<float>((lsm_v >= 0.5f) ? pcfg.cloud_num_land : pcfg.cloud_num_sea);
    const float co2_ppm = static_cast<float>(pcfg.co2_ppm);

    float alb_sfc = alb0;
    if (lsm_v >= 0.5f && snow_v > 0.0f) {
        const float sf = fminf(1.0f, snow_v / 15.0f);
        alb_sfc = (1.0f - sf) * alb0 + sf * 0.72f;
    } else if (lsm_v < 0.5f && lake_v >= 0.5f && hice_v > 1e-4f) {
        alb_sfc = 0.55f;
    } else if (sic_v > 1e-4f) {
        const float alb_i = sea_ice_albedo ? fminf(0.90f, fmaxf(0.15f, sea_ice_albedo[col])) : 0.64f;
        alb_sfc = (1.0f - sic_v) * alb0 + sic_v * alb_i;
    }
    float emis_sfc = (lsm_v >= 0.5f) ? ((snow_v > 1.0f) ? 0.985f : 0.96f) : 0.97f;
    const float t_sk = t_skin ? fminf(fmaxf(t_skin[col], 180.0f), 340.0f) : 285.0f;
    float t_rad_sk = t_sk;
    if (sic_v > 1e-4f) {
        emis_sfc = (1.0f - sic_v) * 0.97f + sic_v * 0.985f;
        const float t_ice = sea_ice_tsfc ? fminf(273.15f, fmaxf(180.0f, sea_ice_tsfc[col])) : 268.0f;
        const float t_o4 = t_sk * t_sk * t_sk * t_sk;
        const float t_i4 = t_ice * t_ice * t_ice * t_ice;
        t_rad_sk = powf((1.0f - sic_v) * t_o4 + sic_v * t_i4, 0.25f);
    }

    for (int32_t k = 0; k < nz; ++k) {
        rad_dth_dt[static_cast<size_t>(k) * str3 + col] = 0.0f;
    }

    // Astronomical geometry (Spencer 1971)
    const float gamma = 2.0f * PI_CONST_F * static_cast<float>(pcfg.julian_day - 1.0) / 365.0f;
    const float decl = 0.006918f - 0.399912f * cosf(gamma) + 0.070257f * sinf(gamma)
                     - 0.006758f * cosf(2.0f * gamma) + 0.000907f * sinf(2.0f * gamma)
                     - 0.002697f * cosf(3.0f * gamma) + 0.001480f * sinf(3.0f * gamma);
    const float eqt = 0.000075f + 0.001868f * cosf(gamma) - 0.032077f * sinf(gamma)
                    - 0.014615f * cosf(2.0f * gamma) - 0.040849f * sinf(2.0f * gamma);
    const float hr_ang = (static_cast<float>(pcfg.utc_seconds) / 86400.0f - 0.5f) * 2.0f * PI_CONST_F
                       + lon_v * (PI_CONST_F / 180.0f) + eqt;
    const float lat_r = lat_v * (PI_CONST_F / 180.0f);
    const float mu0 = sinf(lat_r) * sinf(decl) + cosf(lat_r) * cosf(decl) * cosf(hr_ang);
    const float s0_toa = 1361.0f * (1.000110f + 0.034221f * cosf(gamma) + 0.001280f * sinf(gamma)
                                  + 0.000719f * cosf(2.0f * gamma) + 0.000077f * sinf(2.0f * gamma));

    float sw_dn_sfc = 0.0f;
    if (mu0 > 0.01f) {
        const float sw_frac[2] = {0.46f, 0.54f};
        for (int b = 0; b < 2; ++b) {
            const float s0_b = s0_toa * sw_frac[b];
            float f_dir = s0_b;
            for (int32_t k = nz - 1; k >= 0; --k) {
                const size_t idx = static_cast<size_t>(k) * str3 + col;
                float z_s, dz_k, thb, pib, th_t, pi_t, t_k, p_k, rho;
                eval_cell_thermo(cfg, g, k, col, hs, pi_sfc_f, thp, pip, qv, qc, qr, qi, qs, qg,
                                 z_s, dz_k, thb, pib, th_t, pi_t, t_k, p_k, rho);
                const float qv_v = (cfg.has_moist && qv) ? fmaxf(0.0f, qv[idx]) : 0.0f;
                const float qc_v = (cfg.has_moist && qc) ? fmaxf(0.0f, qc[idx]) : 0.0f;
                const float qr_v = (cfg.has_moist && qr) ? fmaxf(0.0f, qr[idx]) : 0.0f;
                const float qi_v = (cfg.has_moist && qi) ? fmaxf(0.0f, qi[idx]) : 0.0f;
                const float qs_v = (cfg.has_moist && qs) ? fmaxf(0.0f, qs[idx]) : 0.0f;

                scratch_f[idx] = f_dir;
                float r_d, t_d, r_dir, t_dd, t_dir;
                sw_layer_optics(b, p_k, t_k, dz_k, rho, qv_v, qc_v, qr_v, qi_v, qs_v,
                                n_drop, co2_ppm, mu0, r_d, t_d, r_dir, t_dd, t_dir);
                f_dir *= t_dir;
            }
            const float f_dir_sfc = f_dir;

            // Bottom-up Adding pass (k = 0..nz-1)
            tri_c[col] = static_cast<double>(alb_sfc);
            tri_d[col] = static_cast<double>(alb_sfc * f_dir_sfc * mu0);
            for (int32_t k = 0; k < nz; ++k) {
                const size_t idx = static_cast<size_t>(k) * str3 + col;
                float z_s, dz_k, thb, pib, th_t, pi_t, t_k, p_k, rho;
                eval_cell_thermo(cfg, g, k, col, hs, pi_sfc_f, thp, pip, qv, qc, qr, qi, qs, qg,
                                 z_s, dz_k, thb, pib, th_t, pi_t, t_k, p_k, rho);
                const float qv_v = (cfg.has_moist && qv) ? fmaxf(0.0f, qv[idx]) : 0.0f;
                const float qc_v = (cfg.has_moist && qc) ? fmaxf(0.0f, qc[idx]) : 0.0f;
                const float qr_v = (cfg.has_moist && qr) ? fmaxf(0.0f, qr[idx]) : 0.0f;
                const float qi_v = (cfg.has_moist && qi) ? fmaxf(0.0f, qi[idx]) : 0.0f;
                const float qs_v = (cfg.has_moist && qs) ? fmaxf(0.0f, qs[idx]) : 0.0f;

                float r_d, t_d, r_dir, t_dd, t_dir;
                sw_layer_optics(b, p_k, t_k, dz_k, rho, qv_v, qc_v, qr_v, qi_v, qs_v,
                                n_drop, co2_ppm, mu0, r_d, t_d, r_dir, t_dd, t_dir);
                const float f_dir_in = scratch_f[idx];
                const float r_b = static_cast<float>(tri_c[static_cast<size_t>(k) * str3 + col]);
                const float s_b = static_cast<float>(tri_d[static_cast<size_t>(k) * str3 + col]);
                const float inv = 1.0f / fmaxf(1.0f - r_b * r_d, 1e-7f);
                tri_c[static_cast<size_t>(k + 1) * str3 + col] = static_cast<double>(r_d + t_d * t_d * r_b * inv);
                tri_d[static_cast<size_t>(k + 1) * str3 + col] =
                    static_cast<double>(r_dir * f_dir_in + t_d * (s_b + r_b * t_dd * f_dir_in) * inv);
            }

            // Top-down diffuse sweep & radiative heating accumulation
            float f_dn_diff = 0.0f;
            float f_up_top = static_cast<float>(tri_d[static_cast<size_t>(nz) * str3 + col]);
            float f_net_above = s0_b * mu0 - f_up_top;

            for (int32_t k = nz - 1; k >= 0; --k) {
                const size_t idx = static_cast<size_t>(k) * str3 + col;
                float z_s, dz_k, thb, pib, th_t, pi_t, t_k, p_k, rho;
                eval_cell_thermo(cfg, g, k, col, hs, pi_sfc_f, thp, pip, qv, qc, qr, qi, qs, qg,
                                 z_s, dz_k, thb, pib, th_t, pi_t, t_k, p_k, rho);
                const float qv_v = (cfg.has_moist && qv) ? fmaxf(0.0f, qv[idx]) : 0.0f;
                const float qc_v = (cfg.has_moist && qc) ? fmaxf(0.0f, qc[idx]) : 0.0f;
                const float qr_v = (cfg.has_moist && qr) ? fmaxf(0.0f, qr[idx]) : 0.0f;
                const float qi_v = (cfg.has_moist && qi) ? fmaxf(0.0f, qi[idx]) : 0.0f;
                const float qs_v = (cfg.has_moist && qs) ? fmaxf(0.0f, qs[idx]) : 0.0f;

                float r_d, t_d, r_dir, t_dd, t_dir;
                sw_layer_optics(b, p_k, t_k, dz_k, rho, qv_v, qc_v, qr_v, qi_v, qs_v,
                                n_drop, co2_ppm, mu0, r_d, t_d, r_dir, t_dd, t_dir);
                const float f_dir_in = scratch_f[idx];
                const float r_b = static_cast<float>(tri_c[idx]);
                const float s_b = static_cast<float>(tri_d[idx]);
                const float inv = 1.0f / fmaxf(1.0f - r_b * r_d, 1e-7f);
                f_dn_diff = (t_d * f_dn_diff + r_d * s_b + t_dd * f_dir_in) * inv;
                const float f_up_below = r_b * f_dn_diff + s_b;
                const float f_dir_below = f_dir_in * t_dir;
                const float f_net_below = f_dn_diff + f_dir_below * mu0 - f_up_below;
                const float dth_dt = (f_net_above - f_net_below) / (rho * CP_D_F * dz_k * pi_t);
                rad_dth_dt[idx] += dth_dt;
                f_net_above = f_net_below;
            }
            sw_dn_sfc += f_dn_diff + f_dir_sfc * mu0;
        }
    }

    // Longwave 3-band Linear-Planck Adding Method (sliding column T cache avoids redundant thermo evals)
    float lw_dn_sfc = 0.0f;
    for (int b = 0; b < 3; ++b) {
        tri_c[col] = static_cast<double>(1.0f - emis_sfc);
        tri_d[col] = static_cast<double>(emis_sfc * lw_band_planck_flux(b, t_rad_sk));

        float t_prev = t_rad_sk;
        float z_c, dz_c, thb_c, pib_c, th_c, pi_c, t_c, p_c, rho_c;
        eval_cell_thermo(cfg, g, 0, col, hs, pi_sfc_f, thp, pip, qv, qc, qr, qi, qs, qg,
                         z_c, dz_c, thb_c, pib_c, th_c, pi_c, t_c, p_c, rho_c);

        for (int32_t k = 0; k < nz; ++k) {
            const size_t idx = static_cast<size_t>(k) * str3 + col;
            float z_n, dz_n, thbn, pibn, thn, pin, t_nxt = t_c, pn, rhon;
            if (k + 1 < nz) {
                eval_cell_thermo(cfg, g, k + 1, col, hs, pi_sfc_f, thp, pip, qv, qc, qr, qi, qs, qg,
                                 z_n, dz_n, thbn, pibn, thn, pin, t_nxt, pn, rhon);
            }
            const float t_bot_if = (k == 0) ? t_c : 0.5f * (t_prev + t_c);
            const float t_top_if = (k + 1 == nz) ? t_c : 0.5f * (t_c + t_nxt);
            t_prev = t_c;

            const float qv_v = (cfg.has_moist && qv) ? fmaxf(0.0f, qv[idx]) : 0.0f;
            const float qc_v = (cfg.has_moist && qc) ? fmaxf(0.0f, qc[idx]) : 0.0f;
            const float qr_v = (cfg.has_moist && qr) ? fmaxf(0.0f, qr[idx]) : 0.0f;
            const float qi_v = (cfg.has_moist && qi) ? fmaxf(0.0f, qi[idx]) : 0.0f;
            const float qs_v = (cfg.has_moist && qs) ? fmaxf(0.0f, qs[idx]) : 0.0f;

            float r_d, t_d, s_up_k, s_dn_k;
            lw_layer_optics(b, p_c, t_c, t_bot_if, t_top_if, dz_c, rho_c,
                            qv_v, qc_v, qr_v, qi_v, qs_v, n_drop, co2_ppm,
                            r_d, t_d, s_up_k, s_dn_k);

            const float r_b = static_cast<float>(tri_c[idx]);
            const float s_b = static_cast<float>(tri_d[idx]);
            const float inv = 1.0f / fmaxf(1.0f - r_b * r_d, 1e-7f);
            tri_c[static_cast<size_t>(k + 1) * str3 + col] = static_cast<double>(r_d + t_d * t_d * r_b * inv);
            tri_d[static_cast<size_t>(k + 1) * str3 + col] = static_cast<double>(s_up_k + t_d * (s_b + r_b * s_dn_k) * inv);

            if (k + 1 < nz) {
                z_c = z_n; dz_c = dz_n; thb_c = thbn; pib_c = pibn;
                th_c = thn; pi_c = pin; t_c = t_nxt; p_c = pn; rho_c = rhon;
            }
        }

        float f_dn = 0.0f; // TOA downward LW = 0
        float f_up_top = static_cast<float>(tri_d[static_cast<size_t>(nz) * str3 + col]);
        float f_net_above = f_dn - f_up_top;

        // Pass 2 downward: t_c starts at nz-1 (already held in registers!), t_above holds k+1, compute k-1 once per k
        float t_above = t_c;
        for (int32_t k = nz - 1; k >= 0; --k) {
            const size_t idx = static_cast<size_t>(k) * str3 + col;
            float z_p, dz_p, thbp, pibp, thp_v, pip_v, t_below = t_c, pp, rhop;
            if (k > 0) {
                eval_cell_thermo(cfg, g, k - 1, col, hs, pi_sfc_f, thp, pip, qv, qc, qr, qi, qs, qg,
                                 z_p, dz_p, thbp, pibp, thp_v, pip_v, t_below, pp, rhop);
            }
            const float t_bot_if = (k > 0) ? 0.5f * (t_below + t_c) : t_c;
            const float t_top_if = (k + 1 < nz) ? 0.5f * (t_c + t_above) : t_c;

            const float qv_v = (cfg.has_moist && qv) ? fmaxf(0.0f, qv[idx]) : 0.0f;
            const float qc_v = (cfg.has_moist && qc) ? fmaxf(0.0f, qc[idx]) : 0.0f;
            const float qr_v = (cfg.has_moist && qr) ? fmaxf(0.0f, qr[idx]) : 0.0f;
            const float qi_v = (cfg.has_moist && qi) ? fmaxf(0.0f, qi[idx]) : 0.0f;
            const float qs_v = (cfg.has_moist && qs) ? fmaxf(0.0f, qs[idx]) : 0.0f;

            float r_d, t_d, s_up_k, s_dn_k;
            lw_layer_optics(b, p_c, t_c, t_bot_if, t_top_if, dz_c, rho_c,
                            qv_v, qc_v, qr_v, qi_v, qs_v, n_drop, co2_ppm,
                            r_d, t_d, s_up_k, s_dn_k);

            const float r_b = static_cast<float>(tri_c[idx]);
            const float s_b = static_cast<float>(tri_d[idx]);
            const float inv = 1.0f / fmaxf(1.0f - r_b * r_d, 1e-7f);
            f_dn = (t_d * f_dn + r_d * s_b + s_dn_k) * inv;
            const float f_up_below = r_b * f_dn + s_b;
            const float f_net_below = f_dn - f_up_below;
            const float dth_dt = (f_net_above - f_net_below) / (rho_c * CP_D_F * dz_c * pi_c);
            rad_dth_dt[idx] += dth_dt;
            f_net_above = f_net_below;

            t_above = t_c;
            if (k > 0) {
                z_c = z_p; dz_c = dz_p; thb_c = thbp; pib_c = pibp;
                th_c = thp_v; pi_c = pip_v; t_c = t_below; p_c = pp; rho_c = rhop;
            }
        }
        lw_dn_sfc += f_dn;
    }

    if (sw_down_out) sw_down_out[col] = sw_dn_sfc;
    if (lw_down_out) lw_down_out[col] = lw_dn_sfc;
}

__global__ void nh_apply_rad_kernel(
    NhGridConfig cfg, double dt_s, const float* rad_dth_dt, float* thp
) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= cfg.ny || i >= cfg.nx) return;
    const size_t str = static_cast<size_t>(cfg.ny) * cfg.nx;
    const size_t col = static_cast<size_t>(j) * cfg.nx + i;
    const float dt_f = static_cast<float>(dt_s);
    for (int32_t k = 0; k < cfg.nz; ++k) {
        const size_t idx = static_cast<size_t>(k) * str + col;
        thp[idx] += dt_f * rad_dth_dt[idx];
    }
}

// ===========================================================================
// PILLAR 4 (Surface Layer, Land LSM, Ocean COARE 3.6 + VOSKIN, FLake)
// ===========================================================================

__device__ inline float psim_jimenez(float zol) {
    // Jimenez et al. (2012) / MMM sf_sfclayrev.F90 (BSD-3-Clause)
    if (zol <= 0.0f) {
        const float z = fmaxf(zol, -15.0f);
        const float x = powf(1.0f - 16.0f * z, 0.25f);
        const float psim_k = 2.0f * logf(0.5f * (1.0f + x)) + logf(0.5f * (1.0f + x * x))
                           - 2.0f * atanf(x) + 0.5f * PI_CONST_F;
        const float y = cbrtf(1.0f - 10.15f * z);
        const float s3 = 1.7320508f;
        const float psim_c = 1.5f * logf((y * y + y + 1.0f) / 3.0f)
                           - s3 * atanf((2.0f * y + 1.0f) / s3) + PI_CONST_F / s3;
        const float f = (z * z) / (1.0f + z * z);
        return (1.0f - f) * psim_k + f * psim_c;
    } else {
        const float z = fminf(zol, 10.0f);
        return -6.1f * logf(z + powf(1.0f + powf(z, 2.5f), 0.4f));
    }
}

__device__ inline float psih_jimenez(float zol) {
    if (zol <= 0.0f) {
        const float z = fmaxf(zol, -15.0f);
        const float x = powf(1.0f - 16.0f * z, 0.25f);
        const float psih_k = 2.0f * logf(0.5f * (1.0f + x * x));
        const float y = cbrtf(1.0f - 34.15f * z);
        const float s3 = 1.7320508f;
        const float psih_c = 1.5f * logf((y * y + y + 1.0f) / 3.0f)
                           - s3 * atanf((2.0f * y + 1.0f) / s3) + PI_CONST_F / s3;
        const float f = (z * z) / (1.0f + z * z);
        return (1.0f - f) * psih_k + f * psih_c;
    } else {
        const float z = fminf(zol, 10.0f);
        return -5.3f * logf(z + powf(1.0f + powf(z, 1.1f), 1.0f / 1.1f));
    }
}

__device__ inline void solve_mo_sfclayrev(
    float za, float u_spd, float thv_a, float thv_s,
    float z0m, float z0h, float& ustar, float& zol_out, float& cm_out, float& ch_out
) {
    const float dthv = thv_a - thv_s;
    const float wspd = fmaxf(u_spd, 0.3f);
    const float rib = GRAVITY_F * dthv * za / (fmaxf(0.5f * (thv_a + thv_s), 150.0f) * wspd * wspd);
    const float ln_zm = logf(fmaxf(za / fmaxf(z0m, 1e-5f), 1.5f));
    const float ln_zh = logf(fmaxf(za / fmaxf(z0h, 1e-6f), 1.5f));

    float zol = fminf(fmaxf(rib * ln_zm, -10.0f), 5.0f);
    for (int it = 0; it < 10; ++it) {
        const float denom_m = fmaxf(ln_zm - psim_jimenez(zol) + psim_jimenez(zol * z0m / za), 0.5f);
        const float denom_h = fmaxf(ln_zh - psih_jimenez(zol) + psih_jimenez(zol * z0h / za), 0.5f);
        const float zol_new = rib * (denom_m * denom_m) / denom_h;
        zol = fminf(fmaxf(0.5f * zol + 0.5f * zol_new, -12.0f), 8.0f);
    }
    const float denom_m = fmaxf(ln_zm - psim_jimenez(zol) + psim_jimenez(zol * z0m / za), 0.5f);
    const float denom_h = fmaxf(ln_zh - psih_jimenez(zol) + psih_jimenez(zol * z0h / za), 0.5f);
    ustar = fmaxf(0.02f, KARMAN_F * wspd / denom_m);
    cm_out = (KARMAN_F / denom_m) * (KARMAN_F / denom_m);
    ch_out = (KARMAN_F / denom_m) * (KARMAN_F / denom_h);
    zol_out = zol;
}

__device__ inline void solve_coare36(
    float za, float u_spd, float t_a, float th_a, float qv_a, float p_a,
    float t_s, float th_s, float p_s, float bathy_depth_m,
    float& ustar, float& z0m, float& z0h, float& zol_out, float& cm_out, float& ch_out,
    float& q_sea_sat, float& wspd_out
) {
    // COARE 3.6 bulk algorithm (Fairall et al. 2003/2019; Edson et al. 2013; 0.98 sea-salt factor)
    q_sea_sat = 0.98f * qsat_water(t_s, p_s);
    const float thv_a = th_a * (1.0f + 0.608f * qv_a);
    const float thv_s = th_s * (1.0f + 0.608f * q_sea_sat);
    const float nu_a = 1.48e-5f; // air kinematic viscosity

    float ug = 0.25f;
    z0m = 1e-4f;
    z0h = 1e-4f;
    ustar = fmaxf(0.05f, 0.035f * fmaxf(u_spd, 0.5f));
    float zol = 0.0f, cm = 1e-3f, ch = 1e-3f;

    for (int it = 0; it < 8; ++it) {
        const float wspd = sqrtf(u_spd * u_spd + ug * ug);
        solve_mo_sfclayrev(za, wspd, thv_a, thv_s, z0m, z0h, ustar, zol, cm, ch);
        const float ln_10 = logf(10.0f / fmaxf(z0m, 1e-6f));
        const float u10n = ustar / KARMAN_F * ln_10;
        const float charn = fminf(0.028f, fmaxf(0.0f, 0.0017f * u10n - 0.005f));
        float z0_wave = 0.0f;
        if (bathy_depth_m > 0.5f && bathy_depth_m < 100.0f) {
            const float ratio = powf(fmaxf(bathy_depth_m, 2.0f) / 30.0f, -0.8f);
            z0_wave = fminf(0.05f, 0.0008f * ratio * (u10n / 15.0f) * (u10n / 15.0f));
        }
        z0m = fminf(0.10f, fmaxf(1e-5f, charn * ustar * ustar / GRAVITY_F + 0.11f * nu_a / ustar + z0_wave));
        const float re_r = fmaxf(0.1f, ustar * z0m / nu_a);
        z0h = fminf(1.6e-4f, fmaxf(1e-6f, 5.8e-5f * powf(re_r, -0.72f)));

        const float inv_scm = 1.0f / sqrtf(fmaxf(cm, 1e-6f));
        const float qstar_est = (ch * inv_scm) * (qv_a - q_sea_sat);
        const float tstar_est = (ch * inv_scm) * (th_a - th_s);
        const float bf = -(GRAVITY_F / fmaxf(thv_a, 150.0f)) * ustar * (tstar_est + 0.608f * th_a * qstar_est);
        ug = (bf > 0.0f) ? fmaxf(0.25f, 1.2f * cbrtf(bf * 600.0f)) : 0.20f;
    }
    const float wspd = sqrtf(u_spd * u_spd + ug * ug);
    solve_mo_sfclayrev(za, wspd, thv_a, thv_s, z0m, z0h, ustar, zol, cm, ch);
    zol_out = zol; cm_out = cm; ch_out = ch; wspd_out = wspd;
    (void)t_a; (void)p_a;
}

__device__ inline void solve_tridiag_4(
    const float a[4], const float b[4], const float c[4], const float d[4], float x[4]
) {
    float cp[4], dp[4];
    const float b0 = fmaxf(b[0], 1e-20f);
    cp[0] = c[0] / b0;
    dp[0] = d[0] / b0;
    for (int m = 1; m < 4; ++m) {
        const float denom = fmaxf(b[m] - a[m] * cp[m - 1], 1e-20f);
        cp[m] = (m < 3) ? (c[m] / denom) : 0.0f;
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
    const float* sea_ice_frac, const float* sea_ice_thick, const float* sea_ice_tsfc,
    float* shf, float* lhf, float* ustar, float* t2m, float* q2m, float* u10m, float* v10m,
    float* sfc_drag_coeff
) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= cfg.ny || i >= cfg.nx) return;

    const size_t n2 = static_cast<size_t>(cfg.ny) * cfg.nx;
    const size_t col = static_cast<size_t>(j) * cfg.nx + i;
    const float dt_f = static_cast<float>(dt_s);
    const float pi_sfc_f = powf(static_cast<float>(cfg.p_sfc_pa) / P0_PA_F, KAPPA_F);
    const float hs = static_cast<float>(g.hs[col]);

    float z_0, dz_0, thb0, pib0, th_a, pi_a, t_a, p_a, rho_a;
    eval_cell_thermo(cfg, g, 0, col, hs, pi_sfc_f, thp, pip, qv, qc, qr, qi, qs, qg,
                     z_0, dz_0, thb0, pib0, th_a, pi_a, t_a, p_a, rho_a);
    const float za = fmaxf(2.0f, z_0 - hs);
    const float qv_a = (cfg.has_moist && qv) ? fmaxf(1e-6f, qv[col]) : 1e-6f;
    const float tv_a = t_a * (1.0f + EPS_V_F * qv_a);
    const float p_s = p_a * expf(GRAVITY_F * za / (R_D_F * tv_a));
    const float pi_s = powf(p_s / P0_PA_F, KAPPA_F);
    const float t_a_at_sfc = th_a * pi_s;

    const float u_c = 0.5f * (read_u_f(u, 0, j, i, cfg.nz, cfg.ny, cfg.nx, cfg.bc_x, cfg.bc_y)
                            + read_u_f(u, 0, j, i + 1, cfg.nz, cfg.ny, cfg.nx, cfg.bc_x, cfg.bc_y));
    const float v_c = 0.5f * (read_v_f(v, 0, j, i, cfg.nz, cfg.ny, cfg.nx, cfg.bc_x, cfg.bc_y)
                            + read_v_f(v, 0, j + 1, i, cfg.nz, cfg.ny, cfg.nx, cfg.bc_x, cfg.bc_y));
    const float u_spd = sqrtf(u_c * u_c + v_c * v_c);
    const float wspd_eff = fmaxf(0.5f, u_spd);

    const float sw_in = sw_down ? fmaxf(0.0f, sw_down[col]) : 0.0f;
    const float lw_in = lw_down ? fmaxf(50.0f, lw_down[col]) : 300.0f;
    const float p_liq = precip_liq_rate ? fmaxf(0.0f, precip_liq_rate[col]) : 0.0f;
    const float p_ice = precip_ice_rate ? fmaxf(0.0f, precip_ice_rate[col]) : 0.0f;

    const float lsm_v = lsm ? lsm[col] : 1.0f;
    const float lake_v = lake_mask ? lake_mask[col] : 0.0f;
    const float z0_base_v = z0m_base ? fmaxf(1e-4f, z0m_base[col]) : ((lsm_v >= 0.5f) ? 0.05f : 1e-4f);
    const float alb_base_v = albedo_base ? fminf(0.85f, fmaxf(0.04f, albedo_base[col])) : ((lsm_v >= 0.5f) ? 0.20f : 0.06f);
    const float veg_v = veg_frac ? fminf(1.0f, fmaxf(0.0f, veg_frac[col])) : 0.60f;
    const float depth_v = bathy_depth_m ? fmaxf(1.0f, bathy_depth_m[col]) : ((lake_v >= 0.5f) ? 4.5f : 40.0f);

    float t_sk_new = t_a_at_sfc;
    float q_sk_eff = qv_a;
    float ustar_v = 0.15f, zol_v = 0.0f, cm_v = 1e-3f, ch_v = 1e-3f, z0m_v = z0_base_v, z0h_v = z0_base_v * 0.1f;
    float h_flux = 0.0f, le_flux = 0.0f, wspd_drag = wspd_eff;

    if (lsm_v >= 0.5f) {
        int stype = soil_type ? static_cast<int>(lroundf(soil_type[col])) : 9;
        if (stype < 1) stype = 9;
        if (stype > 13) stype = 13;

        const float th_sat  = static_cast<float>(c_soil_table[stype][0]);
        const float th_fc   = static_cast<float>(c_soil_table[stype][1]);
        const float th_wilt = static_cast<float>(c_soil_table[stype][2]);
        const float b_hyd   = static_cast<float>(c_soil_table[stype][3]);
        const float psi_sat = static_cast<float>(c_soil_table[stype][4]);
        const float ksat_h  = static_cast<float>(c_soil_table[stype][5]);
        const float k_dry   = static_cast<float>(c_soil_table[stype][6]);
        const float k_sat   = static_cast<float>(c_soil_table[stype][7]);

        float ts_soil[4], ms_soil[4];
        for (int m = 0; m < 4; ++m) {
            const size_t sidx = static_cast<size_t>(m) * n2 + col;
            ts_soil[m] = soil_t ? fminf(340.0f, fmaxf(210.0f, soil_t[sidx])) : 285.0f;
            ms_soil[m] = soil_m ? fminf(th_sat, fmaxf(0.02f, soil_m[sidx])) : th_fc;
        }
        float swe = snow_swe ? fmaxf(0.0f, snow_swe[col] + p_ice * dt_f) : 0.0f;
        const float snow_cov = fminf(1.0f, swe / 15.0f);
        const float alb = (1.0f - snow_cov) * alb_base_v + snow_cov * 0.72f;
        const float emis = (1.0f - snow_cov) * 0.96f + snow_cov * 0.985f;
        z0m_v = fmaxf(1e-3f, (1.0f - snow_cov) * z0_base_v + snow_cov * 0.002f);
        z0h_v = z0m_v * 0.1f;

        const float r_abs = (1.0f - alb) * sw_in + emis * lw_in;
        const float lambda_sk = (1.0f - snow_cov) * (15.0f * (1.0f - veg_v) + 10.0f * veg_v) + snow_cov * 20.0f;
        const float f_sm = fminf(1.0f, fmaxf(0.05f, (ms_soil[0] - th_wilt) / fmaxf(th_fc - th_wilt, 0.05f)));
        const float f_rad = (0.55f * sw_in * 2.0f / 100.0f + 0.02f) / (0.55f * sw_in * 2.0f / 100.0f + 1.0f);
        const float r_canopy = 110.0f / fmaxf(f_sm * f_rad, 0.04f);
        float beta_ev = snow_cov + (1.0f - snow_cov) * f_sm * (1.0f - 0.5f * veg_v);

        t_sk_new = t_skin ? fminf(340.0f, fmaxf(210.0f, t_skin[col])) : ts_soil[0];
        if (t_sk_new <= 0.0f) t_sk_new = ts_soil[0];

        // 2-outer coupled (M-O <-> Newton skin balance) closure
        for (int outer = 0; outer < 2; ++outer) {
            const float th_sk = t_sk_new / pi_s;
            const float q_sat_sk = qsat_water(t_sk_new, p_s);
            const float q_sk_est = qv_a + beta_ev * (q_sat_sk - qv_a);
            const float thv_a = th_a * (1.0f + EPS_V_F * qv_a);
            const float thv_s = th_sk * (1.0f + EPS_V_F * q_sk_est);
            solve_mo_sfclayrev(za, wspd_eff, thv_a, thv_s, z0m_v, z0h_v, ustar_v, zol_v, cm_v, ch_v);

            const float g_exch = ch_v * wspd_eff;
            const float beta_veg = 1.0f / (1.0f + g_exch * r_canopy);
            beta_ev = snow_cov * 1.0f + (1.0f - snow_cov) * (veg_v * beta_veg + (1.0f - veg_v) * f_sm);
            const float flhc = rho_a * CP_D_F * g_exch / pi_s;
            const float flqc = rho_a * ((swe > 0.1f) ? LS0_F : LV0_F) * g_exch * beta_ev;

            for (int it = 0; it < 6; ++it) {
                const float qs_t = qsat_water(t_sk_new, p_s);
                const float dqs_t = dqsat_water_dt(t_sk_new, p_s);
                const float t2 = t_sk_new * t_sk_new;
                const float t3 = t2 * t_sk_new;
                const float rad_out = emis * STEFAN_BOLTZMANN_F * t3 * t_sk_new;
                const float f_val = r_abs - rad_out - flhc * (t_sk_new - t_a_at_sfc) - flqc * (qs_t - qv_a) - lambda_sk * (t_sk_new - ts_soil[0]);
                const float df = -4.0f * emis * STEFAN_BOLTZMANN_F * t3 - flhc - flqc * dqs_t - lambda_sk;
                const float dt_step = fminf(15.0f, fmaxf(-15.0f, -f_val / fminf(df, -1.0f)));
                t_sk_new = fminf(340.0f, fmaxf(200.0f, t_sk_new + dt_step));
                if (fabsf(dt_step) < 1e-4f) break;
            }
            if (swe > 0.05f && t_sk_new > T_FREEZE_F) {
                t_sk_new = T_FREEZE_F;
            }
            h_flux = flhc * (t_sk_new - t_a_at_sfc);
            const float qs_fin = qsat_water(t_sk_new, p_s);
            le_flux = flqc * (qs_fin - qv_a);
            q_sk_eff = qv_a + beta_ev * (qs_fin - qv_a);
        }

        const float g0_flux = lambda_sk * (t_sk_new - ts_soil[0]);
        float melt_mass = 0.0f;
        if (swe > 0.05f && t_sk_new >= T_FREEZE_F) {
            const float tf2 = T_FREEZE_F * T_FREEZE_F;
            const float e_avail = r_abs - emis * STEFAN_BOLTZMANN_F * tf2 * tf2 - h_flux - le_flux - g0_flux;
            if (e_avail > 0.0f) {
                const float dm = fminf(swe, e_avail * dt_f / LF0_F);
                swe -= dm;
                melt_mass = dm;
            }
        }

        float kappa_s[4], cs_s[4];
        for (int m = 0; m < 4; ++m) {
            const float sr = fminf(1.0f, fmaxf(0.05f, ms_soil[m] / th_sat));
            const float ke = (ts_soil[m] < T_FREEZE_F) ? sr : fmaxf(0.0f, log10f(sr) + 1.0f);
            kappa_s[m] = (k_sat - k_dry) * ke + k_dry;
            const float frozen_c = (fabsf(ts_soil[m] - T_FREEZE_F) < 1.5f) ? 8.0e6f : 0.0f;
            cs_s[m] = (1.0f - th_sat) * 2.1e6f + ms_soil[m] * 4.18e6f + frozen_c;
        }
        float a_t[4] = {}, b_t[4] = {}, c_t[4] = {}, d_t[4] = {};
        for (int m = 0; m < 4; ++m) {
            const float dz_m = static_cast<float>(c_soil_dz[m]);
            const float cap = cs_s[m] * dz_m / dt_f;
            float c_up = 0.0f, c_dn = 0.0f;
            if (m > 0) {
                const float k_if = 2.0f * kappa_s[m - 1] * kappa_s[m] / fmaxf(kappa_s[m - 1] + kappa_s[m], 0.05f);
                c_up = k_if / (0.5f * (static_cast<float>(c_soil_dz[m - 1]) + dz_m));
            }
            if (m < 3) {
                const float k_if = 2.0f * kappa_s[m] * kappa_s[m + 1] / fmaxf(kappa_s[m] + kappa_s[m + 1], 0.05f);
                c_dn = k_if / (0.5f * (dz_m + static_cast<float>(c_soil_dz[m + 1])));
            }
            a_t[m] = -c_up;
            c_t[m] = -c_dn;
            b_t[m] = cap + c_up + c_dn;
            d_t[m] = cap * ts_soil[m];
        }
        d_t[0] += g0_flux;
        solve_tridiag_4(a_t, b_t, c_t, d_t, ts_soil);

        const float evap_ms = fmaxf(0.0f, le_flux / (LV0_F * 1000.0f)) * (1.0f - snow_cov);
        const float infil_ms = p_liq / 1000.0f + melt_mass / (1000.0f * dt_f) - evap_ms;
        float hyd_d[4], hyd_k[4];
        for (int m = 0; m < 4; ++m) {
            const float sr = fminf(1.0f, fmaxf(0.05f, ms_soil[m] / th_sat));
            hyd_k[m] = ksat_h * powf(sr, 2.0f * b_hyd + 3.0f);
            hyd_d[m] = (b_hyd * ksat_h * psi_sat / th_sat) * powf(sr, b_hyd + 2.0f);
        }
        float a_m[4] = {}, b_m[4] = {}, c_m[4] = {}, d_m[4] = {};
        for (int m = 0; m < 4; ++m) {
            const float dz_m = static_cast<float>(c_soil_dz[m]);
            const float cap_m = dz_m / dt_f;
            const float g_up = (m > 0) ? 0.5f * (hyd_d[m - 1] + hyd_d[m]) / (0.5f * (static_cast<float>(c_soil_dz[m - 1]) + dz_m)) : 0.0f;
            const float g_dn = (m < 3) ? 0.5f * (hyd_d[m] + hyd_d[m + 1]) / (0.5f * (dz_m + static_cast<float>(c_soil_dz[m + 1]))) : 0.0f;
            const float k_up = (m > 0) ? 0.5f * (hyd_k[m - 1] + hyd_k[m]) : 0.0f;
            const float k_dn = (m < 3) ? 0.5f * (hyd_k[m] + hyd_k[m + 1]) : hyd_k[3];
            a_m[m] = -g_up;
            c_m[m] = -g_dn;
            b_m[m] = cap_m + g_up + g_dn;
            d_m[m] = cap_m * ms_soil[m] + (k_up - k_dn);
        }
        d_m[0] += infil_ms;
        solve_tridiag_4(a_m, b_m, c_m, d_m, ms_soil);

        for (int m = 0; m < 4; ++m) {
            const size_t sidx = static_cast<size_t>(m) * n2 + col;
            if (soil_t) soil_t[sidx] = fminf(340.0f, fmaxf(210.0f, ts_soil[m]));
            if (soil_m) soil_m[sidx] = fminf(th_sat, fmaxf(0.02f, ms_soil[m]));
        }
        if (snow_swe) snow_swe[col] = swe;
    } else if (lake_v >= 0.5f) {
        constexpr float RHO_W = 1000.0f, CP_W = 4192.0f, RHO_CP_W = RHO_W * CP_W;
        constexpr float K_ICE = 2.29f, RHO_ICE = 917.0f, C_ALB_MR = 95.6f, T_MAX_DENS = 277.13f;

        float t_w = t_skin ? fminf(315.0f, fmaxf(250.0f, t_skin[col])) : 283.0f;
        float t_b = lake_tbot ? fminf(315.0f, fmaxf(273.15f, lake_tbot[col])) : 280.0f;
        float h_ml = lake_hmld ? fminf(depth_v, fmaxf(0.01f, lake_hmld[col])) : fminf(depth_v, 2.5f);
        float c_th = lake_ctheta ? fminf(0.80f, fmaxf(0.50f, lake_ctheta[col])) : 0.65f;
        float h_ice = lake_hice ? fminf(3.0f, fmaxf(0.0f, lake_hice[col])) : 0.0f;

        const bool ice_cov = (h_ice > 1e-4f);
        const float alb = ice_cov ? fminf(0.65f, fmaxf(0.15f, 0.60f - 0.45f * expf(-C_ALB_MR * fmaxf(0.0f, T_FREEZE_F - t_w) / T_FREEZE_F))) : 0.07f;
        const float emis = ice_cov ? 0.985f : 0.97f;
        z0m_v = ice_cov ? 1e-3f : fmaxf(1e-4f, 0.014f * 0.04f / GRAVITY_F + 1.5e-5f);
        z0h_v = z0m_v * 0.1f;

        for (int it = 0; it < 2; ++it) {
            const float th_sk = t_w / pi_s;
            const float qs_sk = ice_cov ? qsat_ice(t_w, p_s) : qsat_water(t_w, p_s);
            const float thv_a = th_a * (1.0f + EPS_V_F * qv_a);
            const float thv_s = th_sk * (1.0f + EPS_V_F * qs_sk);
            solve_mo_sfclayrev(za, wspd_eff, thv_a, thv_s, z0m_v, z0h_v, ustar_v, zol_v, cm_v, ch_v);
            if (!ice_cov) z0m_v = fminf(0.02f, fmaxf(1e-4f, 0.014f * ustar_v * ustar_v / GRAVITY_F + 1.5e-5f / ustar_v));
            const float g_exch = ch_v * wspd_eff;
            h_flux = rho_a * CP_D_F * g_exch * (th_sk - th_a);
            le_flux = rho_a * (ice_cov ? LS0_F : LV0_F) * g_exch * (qs_sk - qv_a);
            q_sk_eff = qs_sk;
        }
        const float tw2 = t_w * t_w;
        const float q_net = (1.0f - alb) * sw_in + emis * lw_in - emis * STEFAN_BOLTZMANN_F * tw2 * tw2 - h_flux - le_flux;

        if (h_ice > 1e-4f || (t_w <= T_FREEZE_F && q_net < 0.0f)) {
            if (h_ice < 1e-4f) h_ice = 0.005f;
            const float t_new_ice = t_w + (q_net - K_ICE * (t_w - T_FREEZE_F) / fmaxf(h_ice, 0.01f)) * dt_f / (RHO_ICE * 2100.0f * fminf(h_ice, 0.1f));
            if (t_new_ice > T_FREEZE_F) {
                t_w = T_FREEZE_F;
                const float e_melt = fmaxf(0.0f, q_net) * dt_f;
                h_ice = fmaxf(0.0f, h_ice - e_melt / (RHO_ICE * LF0_F));
            } else {
                t_w = fmaxf(230.0f, t_new_ice);
                const float q_cond = K_ICE * (T_FREEZE_F - t_w) / fmaxf(h_ice, 0.01f);
                h_ice = fminf(3.0f, fmaxf(0.0f, h_ice + q_cond * dt_f / (RHO_ICE * LF0_F)));
            }
            if (h_ice <= 1e-4f) { h_ice = 0.0f; t_w = T_FREEZE_F; }
            t_sk_new = t_w;
        } else {
            const float c_eff = RHO_CP_W * fminf(depth_v, fmaxf(0.5f, h_ml + (1.0f - c_th) * (depth_v - h_ml)));
            const float alpha_t = 1.6509e-5f * (t_w - T_MAX_DENS);
            const float buoy_s = GRAVITY_F * alpha_t * q_net / RHO_CP_W;
            const float u_star_w = sqrtf(rho_a / RHO_W) * ustar_v;
            if (buoy_s < 0.0f || u_star_w > 0.015f) {
                const float w_conv = (buoy_s < 0.0f) ? cbrtf(-buoy_s * h_ml) : 0.0f;
                const float dh = 0.25f * sqrtf(u_star_w * u_star_w + w_conv * w_conv);
                h_ml = fminf(depth_v, h_ml + dh * dt_f * 0.2f);
            } else {
                const float us3 = u_star_w * u_star_w * u_star_w;
                const float l_ob = fmaxf(0.2f, us3 / fmaxf(KARMAN_F * buoy_s, 1e-12f));
                h_ml = fminf(depth_v, fmaxf(0.2f, h_ml + (fminf(depth_v, l_ob) - h_ml) * (dt_f / 3600.0f)));
            }
            t_w = fminf(315.0f, fmaxf(T_FREEZE_F, t_w + q_net * dt_f / c_eff));
            if (h_ml >= 0.98f * depth_v) {
                t_b = t_w;
            } else {
                t_b += (t_w - t_b) * (dt_f / (14.0f * 86400.0f));
            }
            c_th = fminf(0.80f, fmaxf(0.50f, c_th + (0.65f - c_th) * (dt_f / 7200.0f)));
            t_sk_new = t_w;
        }

        if (lake_tbot) lake_tbot[col] = t_b;
        if (lake_hmld) lake_hmld[col] = h_ml;
        if (lake_ctheta) lake_ctheta[col] = c_th;
        if (lake_hice) lake_hice[col] = h_ice;
    } else {
        const float sst_b = sst_bulk ? fminf(312.0f, fmaxf(271.15f, sst_bulk[col]))
                                     : (t_skin ? fminf(312.0f, fmaxf(271.15f, t_skin[col])) : 284.0f);
        float d_cool = dt_cool ? fminf(0.3f, fmaxf(-2.0f, dt_cool[col])) : 0.0f;
        float d_warm = dt_warm ? fminf(5.0f, fmaxf(0.0f, dt_warm[col])) : 0.0f;
        const float t_sk = fminf(315.0f, fmaxf(271.15f, sst_b + d_cool + d_warm));
        const float th_sk = t_sk / pi_s;

        float q_sea_sat = 0.0f, wspd_coare = wspd_eff;
        solve_coare36(za, u_spd, t_a, th_a, qv_a, p_a, t_sk, th_sk, p_s, depth_v,
                      ustar_v, z0m_v, z0h_v, zol_v, cm_v, ch_v, q_sea_sat, wspd_coare);

        wspd_drag = fmaxf(wspd_coare, 0.3f);
        const float g_exch = ch_v * wspd_drag;
        h_flux = rho_a * CP_D_F * g_exch * (th_sk - th_a);
        le_flux = rho_a * LV0_F * g_exch * (q_sea_sat - qv_a);
        q_sk_eff = q_sea_sat;

        constexpr float RHO_W = 1025.0f, CP_W = 3985.0f, K_W = 0.60f, NU_W = 1.1e-6f;
        constexpr float EMIS_O = 0.97f, ALB_O = 0.06f;
        const float sw_net = (1.0f - ALB_O) * sw_in;
        const float ts2 = t_sk * t_sk;
        const float lw_net = EMIS_O * lw_in - EMIS_O * STEFAN_BOLTZMANN_F * ts2 * ts2;
        const float q_ns_cool = -(lw_net - h_flux - le_flux);
        const float q_cool_eff = q_ns_cool - 0.12f * sw_net;
        const float us_w = fmaxf(1e-4f, sqrtf(rho_a / RHO_W) * ustar_v);
        const float nu3 = NU_W * NU_W * NU_W;
        const float us2 = us_w * us_w;
        const float a_cs = 16.0f * GRAVITY_F * 2.1e-4f * RHO_W * CP_W * nu3 * fmaxf(0.0f, q_cool_eff)
                         / (us2 * us2 * K_W * K_W + 1e-18f);
        const float lambda_s = 6.0f / cbrtf(1.0f + powf(a_cs, 0.75f));
        const float delta_cool = fminf(0.01f, fmaxf(2e-4f, lambda_s * NU_W / us_w));
        d_cool = fminf(0.25f, fmaxf(-1.5f, -q_cool_eff * delta_cool / K_W));

        const float d_warm_depth = fminf(3.0f, fmaxf(1.0f, 0.5f * depth_v));
        const float sw_abs_w = 0.65f * sw_net;
        const float q_warm_net = sw_abs_w - q_ns_cool;
        const float mix_diss = 2.0f * KARMAN_F * us_w / d_warm_depth;
        d_warm = fmaxf(0.0f, (d_warm + (q_warm_net / (RHO_W * CP_W * d_warm_depth)) * dt_f) / (1.0f + mix_diss * dt_f));
        d_warm = fminf(4.5f, d_warm);

        t_sk_new = fminf(315.0f, fmaxf(271.15f, sst_b + d_cool + d_warm));
        if (dt_cool) dt_cool[col] = d_cool;
        if (dt_warm) dt_warm[col] = d_warm;

        // Fractional SI3 sea-ice mosaic over ocean: blend COARE open-ocean fluxes with snow/ice M-O
        const float sic_v = sea_ice_frac ? fminf(1.0f, fmaxf(0.0f, sea_ice_frac[col])) : 0.0f;
        (void)sea_ice_thick;
        if (sic_v > 1e-4f) {
            const float t_ice = sea_ice_tsfc ? fminf(273.15f, fmaxf(190.0f, sea_ice_tsfc[col])) : 268.0f;
            const float th_ice = t_ice / pi_s;
            const float q_ice_sat = qsat_from_es(es_ice(t_ice), p_s);
            const float z0m_ice = 0.001f;
            const float z0h_ice = 0.0002f;
            float ustar_ice = 0.05f, zol_ice = 0.0f, cm_ice = 1e-3f, ch_ice = 1e-3f;
            solve_mo_sfclayrev(
                za, wspd_eff,
                th_a * (1.0f + 0.608f * qv_a),
                th_ice * (1.0f + 0.608f * q_ice_sat),
                z0m_ice, z0h_ice, ustar_ice, zol_ice, cm_ice, ch_ice
            );
            const float h_ice = rho_a * CP_D_F * ch_ice * wspd_eff * (th_ice - th_a);
            const float le_ice = rho_a * 2.834e6f * ch_ice * wspd_eff * (q_ice_sat - qv_a);
            const float open_f = 1.0f - sic_v;
            h_flux = open_f * h_flux + sic_v * h_ice;
            le_flux = open_f * le_flux + sic_v * le_ice;
            q_sk_eff = open_f * q_sk_eff + sic_v * q_ice_sat;
            t_sk_new = open_f * t_sk_new + sic_v * t_ice;
            ustar_v = open_f * ustar_v + sic_v * ustar_ice;
            cm_v = open_f * cm_v + sic_v * cm_ice;
            ch_v = open_f * ch_v + sic_v * ch_ice;
            z0m_v = open_f * z0m_v + sic_v * z0m_ice;
            z0h_v = open_f * z0h_v + sic_v * z0h_ice;
            zol_v = open_f * zol_v + sic_v * zol_ice;
            wspd_drag = open_f * wspd_drag + sic_v * wspd_eff;
        }
    }

    if (t_skin) t_skin[col] = t_sk_new;
    if (shf) shf[col] = h_flux;
    if (lhf) lhf[col] = le_flux;
    if (ustar) ustar[col] = ustar_v;
    if (sfc_drag_coeff) sfc_drag_coeff[col] = rho_a * cm_v * wspd_drag;

    // M-O profile interpolation at 2m (T, q) and 10m (u, v)
    const float z_2 = fminf(za, 2.0f);
    const float z_10 = fminf(za, 10.0f);
    const float fh_num = fmaxf(0.1f, logf(z_2 / fmaxf(z0h_v, 1e-6f)) - psih_jimenez(zol_v * (z_2 / za)) + psih_jimenez(zol_v * z0h_v / za));
    const float fh_den = fmaxf(0.5f, logf(za / fmaxf(z0h_v, 1e-6f)) - psih_jimenez(zol_v) + psih_jimenez(zol_v * z0h_v / za));
    const float w_2m = fminf(1.0f, fmaxf(0.0f, fh_num / fh_den));

    const float fm_num = fmaxf(0.1f, logf(z_10 / fmaxf(z0m_v, 1e-5f)) - psim_jimenez(zol_v * (z_10 / za)) + psim_jimenez(zol_v * z0m_v / za));
    const float fm_den = fmaxf(0.5f, logf(za / fmaxf(z0m_v, 1e-5f)) - psim_jimenez(zol_v) + psim_jimenez(zol_v * z0m_v / za));
    const float w_10m = fminf(1.0f, fmaxf(0.0f, fm_num / fm_den));

    const float th_sk = t_sk_new / pi_s;
    const float th_2m = th_sk + w_2m * (th_a - th_sk);
    const float p_2m = p_s * expf(-GRAVITY_F * z_2 / (R_D_F * tv_a));
    if (t2m) t2m[col] = th_2m * powf(p_2m / P0_PA_F, KAPPA_F);
    if (q2m) q2m[col] = fmaxf(1e-7f, q_sk_eff + w_2m * (qv_a - q_sk_eff));
    if (u10m) u10m[col] = w_10m * u_c;
    if (v10m) v10m[col] = w_10m * v_c;
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
    const float pi_sfc_f = powf(static_cast<float>(cfg.p_sfc_pa) / P0_PA_F, KAPPA_F);
    const float hs = static_cast<float>(g.hs[col]);
    const float lsm_v = lsm ? lsm[col] : 1.0f;
    const float f_cor = fabsf(static_cast<float>(g.fcor[col]));
    const float ustar_v = ustar ? fmaxf(0.05f, ustar[col]) : 0.15f;
    const float shf_v = shf ? shf[col] : 0.0f;
    const float lhf_v = lhf ? lhf[col] : 0.0f;

    // Surface thermodynamic level k=0
    float z_0, dz_0, thb0, pib0, th0, pi0, t0, p0, rho0;
    eval_cell_thermo(cfg, g, 0, col, hs, pi_sfc_f, thp, pip, qv, nullptr, nullptr, nullptr, nullptr, nullptr,
                     z_0, dz_0, thb0, pib0, th0, pi0, t0, p0, rho0);
    const float qv0 = (cfg.has_moist && qv) ? fmaxf(0.0f, qv[col]) : 0.0f;
    const float thv0_raw = th0 * (1.0f + EPS_V_F * qv0);
    const float wth0 = shf_v / (rho0 * CP_D_F);
    const float wq0 = lhf_v / (rho0 * LV0_F);
    const float wtv0 = wth0 + 0.608f * t0 * wq0;

    // 1. Diagnose YSU PBL top hpbl using Bulk Richardson number (MMM bl_ysu.F90)
    const float brcr = (wtv0 > 0.0f) ? 0.0f
        : ((lsm_v < 0.5f) ? fminf(0.30f, fmaxf(0.10f, 0.16f * powf(1e-7f * ustar_v / fmaxf(f_cor * 1.5e-5f, 1e-12f), -0.18f)))
                          : 0.25f);

    float hpbl = fmaxf(dz_0, 40.0f);
    const float wstar_init = (wtv0 > 0.0f) ? cbrtf((GRAVITY_F / fmaxf(thv0_raw, 150.0f)) * wtv0 * 600.0f) : 0.0f;
    const float us3_i = ustar_v * ustar_v * ustar_v;
    const float ws3_i = wstar_init * wstar_init * wstar_init;
    const float wscale_init = fmaxf(ustar_v, cbrtf(us3_i + 4.0f * KARMAN_F * ws3_i));
    const float dth_excess = (wtv0 > 0.0f) ? fminf(3.0f, 6.8f * wtv0 / wscale_init) : 0.0f;
    const float thv_ref = thv0_raw + dth_excess;

    float rib_prev = -1.0f, z_prev = z_0 - hs;
    for (int32_t k = 1; k < nz; ++k) {
        const size_t idx = static_cast<size_t>(k) * n2 + col;
        float zk, dzk, thbk, pibk, thk, pik, tk, pk, rhok;
        eval_cell_thermo(cfg, g, k, col, hs, pi_sfc_f, thp, pip, qv, nullptr, nullptr, nullptr, nullptr, nullptr,
                         zk, dzk, thbk, pibk, thk, pik, tk, pk, rhok);
        const float qvk = (cfg.has_moist && qv) ? fmaxf(0.0f, qv[idx]) : 0.0f;
        const float thvk = thk * (1.0f + EPS_V_F * qvk);
        const float z_agl = fmaxf(1.0f, zk - hs);
        const float uc = 0.5f * (read_u_f(u, k, j, i, nz, cfg.ny, cfg.nx, cfg.bc_x, cfg.bc_y)
                               + read_u_f(u, k, j, i + 1, nz, cfg.ny, cfg.nx, cfg.bc_x, cfg.bc_y));
        const float vc = 0.5f * (read_v_f(v, k, j, i, nz, cfg.ny, cfg.nx, cfg.bc_x, cfg.bc_y)
                               + read_v_f(v, k, j + 1, i, nz, cfg.ny, cfg.nx, cfg.bc_x, cfg.bc_y));
        const float rib = GRAVITY_F * z_agl * (thvk - thv_ref) / (thv_ref * fmaxf(uc * uc + vc * vc, 0.5f));
        if (rib >= brcr) {
            const float frac = (rib > rib_prev) ? fminf(1.0f, fmaxf(0.0f, (brcr - rib_prev) / (rib - rib_prev))) : 0.5f;
            hpbl = z_prev + frac * (z_agl - z_prev);
            break;
        }
        rib_prev = rib;
        z_prev = z_agl;
        if (z_agl > 4500.0f) { hpbl = 4500.0f; break; }
    }
    hpbl = (wtv0 > 0.0f) ? fminf(4500.0f, fmaxf(dz_0, hpbl)) : fminf(1500.0f, fmaxf(dz_0, hpbl));
    if (pblh) pblh[col] = hpbl;

    const float wstar = (wtv0 > 0.0f) ? cbrtf((GRAVITY_F / fmaxf(thv0_raw, 150.0f)) * wtv0 * hpbl) : 0.0f;
    const float wstar3 = wstar * wstar * wstar;
    const float wscale = fminf(16.0f * ustar_v, fmaxf(0.2f * ustar_v, cbrtf(us3_i + 4.0f * KARMAN_F * wstar3)));
    const float obukhov_l = (fabsf(wtv0) > 1e-8f) ? (-thv0_raw * us3_i / (KARMAN_F * GRAVITY_F * wtv0)) : 1e6f;

    // 2. Compute YSU K_h[k] -> kh_scratch[k] and Shallow EDMF Updraft Mass Flux M_u[k] -> mu_scratch[k]
    constexpr float AU_EDMF = 0.10f;
    float w_u2 = (wtv0 > 0.0f) ? fmaxf(0.25f, 1.5f * wstar * wstar + ustar_v * ustar_v) : 0.0f;
    const float qc0 = (cfg.has_moist && qc) ? fmaxf(0.0f, qc[col]) : 0.0f;
    float thl_u = th0 - (LV0_F / (CP_D_F * pi0)) * qc0 + ((wtv0 > 0.0f) ? 1.5f * wth0 / wscale : 0.0f);
    float qt_u = (qv0 + qc0) + ((wtv0 > 0.0f) ? 1.5f * fmaxf(0.0f, wq0) / wscale : 0.0f);
    bool updraft_alive = (wtv0 > 0.05f * ustar_v * 0.01f || wtv0 > 1e-4f);

    float zk = z_0, dzk = dz_0, thbk = thb0, pibk = pib0, thk = th0, pik = pi0, tk = t0, pk = p0, rhok = rho0;
    for (int32_t k = 0; k < nz - 1; ++k) {
        const size_t idx_k = static_cast<size_t>(k) * n2 + col;
        const size_t idx_kp = static_cast<size_t>(k + 1) * n2 + col;
        float zkp, dzkp, thbkp, pibkp, thkp, pikp, tkp, pkp, rhokp;
        eval_cell_thermo(cfg, g, k + 1, col, hs, pi_sfc_f, thp, pip, qv, qc, nullptr, qi, nullptr, nullptr,
                         zkp, dzkp, thbkp, pibkp, thkp, pikp, tkp, pkp, rhokp);

        const float z_if = (static_cast<float>(g.zeta_w[k + 1]) + static_cast<float>(g.b_w[k + 1]) * hs) - hs;
        const float dz_mid = fmaxf(1.0f, zkp - zk);
        const float rho_if = 0.5f * (rhok + rhokp);

        const float qvk = (cfg.has_moist && qv) ? fmaxf(0.0f, qv[idx_k]) : 0.0f;
        const float qvkp = (cfg.has_moist && qv) ? fmaxf(0.0f, qv[idx_kp]) : 0.0f;
        const float thvk = thk * (1.0f + EPS_V_F * qvk);
        const float thvkp = thkp * (1.0f + EPS_V_F * qvkp);

        float kh_val = 0.01f;
        if (z_if < hpbl) {
            const float s = z_if / hpbl;
            const float km = KARMAN_F * wscale * z_if * (1.0f - s) * (1.0f - s);
            const float pr = fminf(2.5f, fmaxf(0.65f, 1.0f + 2.1f * ((z_0 - hs) / fmaxf(fabsf(obukhov_l), 5.0f))));
            kh_val = fminf(1000.0f, fmaxf(0.01f, km / pr));
        } else {
            const float uk = 0.5f * (read_u_f(u, k, j, i, nz, cfg.ny, cfg.nx, cfg.bc_x, cfg.bc_y)
                                   + read_u_f(u, k, j, i + 1, nz, cfg.ny, cfg.nx, cfg.bc_x, cfg.bc_y));
            const float ukp = 0.5f * (read_u_f(u, k + 1, j, i, nz, cfg.ny, cfg.nx, cfg.bc_x, cfg.bc_y)
                                    + read_u_f(u, k + 1, j, i + 1, nz, cfg.ny, cfg.nx, cfg.bc_x, cfg.bc_y));
            const float vk = 0.5f * (read_v_f(v, k, j, i, nz, cfg.ny, cfg.nx, cfg.bc_x, cfg.bc_y)
                                   + read_v_f(v, k, j + 1, i, nz, cfg.ny, cfg.nx, cfg.bc_x, cfg.bc_y));
            const float vkp = 0.5f * (read_v_f(v, k + 1, j, i, nz, cfg.ny, cfg.nx, cfg.bc_x, cfg.bc_y)
                                    + read_v_f(v, k + 1, j + 1, i, nz, cfg.ny, cfg.nx, cfg.bc_x, cfg.bc_y));
            const float dudz = (ukp - uk) / dz_mid;
            const float dvdz = (vkp - vk) / dz_mid;
            const float s2 = dudz * dudz + dvdz * dvdz + 1e-8f;
            const float n2 = (GRAVITY_F / (0.5f * (thvk + thvkp))) * ((thvkp - thvk) / dz_mid);
            const float ri_loc = n2 / s2;
            const float ell = (KARMAN_F * z_if) / (1.0f + KARMAN_F * z_if / 30.0f);
            float fh = 1.0f;
            if (ri_loc >= 0.0f) {
                fh = 1.0f / ((1.0f + 5.0f * ri_loc) * (1.0f + 5.0f * ri_loc));
            } else {
                fh = 1.0f - (15.0f * ri_loc) / (1.0f + 75.0f * ell * ell * sqrtf(-ri_loc) / (dz_mid * dz_mid));
            }
            kh_val = fminf(50.0f, fmaxf(0.01f, ell * ell * sqrtf(s2) * fh));
        }
        if (wtv0 > 0.0f && fabsf(z_if - hpbl) < 0.15f * hpbl) {
            const float wm3 = wstar3 + 5.0f * us3_i;
            const float k_ent = 0.15f * dz_mid * cbrtf(wm3) * 0.25f;
            kh_val = fminf(100.0f, kh_val + k_ent);
        }
        kh_scratch[idx_k] = kh_val;

        float mu_val = 0.0f;
        if (updraft_alive && z_if < 4000.0f) {
            const float qck = (cfg.has_moist && qc) ? fmaxf(0.0f, qc[idx_k]) : 0.0f;
            const float thl_env = thk - (LV0_F / (CP_D_F * pik)) * qck;
            const float qt_env = qvk + qck;
            const float eps_u = 1.8f / fmaxf(z_if + 100.0f, 100.0f);
            thl_u = (thl_u + eps_u * dz_mid * thl_env) / (1.0f + eps_u * dz_mid);
            qt_u = (qt_u + eps_u * dz_mid * qt_env) / (1.0f + eps_u * dz_mid);

            const float pi_if = 0.5f * (pik + pikp);
            const float p_if = 0.5f * (pk + pkp);
            float t_u = thl_u * pi_if;
            const float qc_u = fmaxf(0.0f, qt_u - qsat_water(t_u, p_if));
            t_u = fminf(350.0f, fmaxf(160.0f, (thl_u + (LV0_F / (CP_D_F * pi_if)) * qc_u) * pi_if));
            const float qv_u = fmaxf(0.0f, qt_u - qc_u);
            const float thv_u = (t_u / pi_if) * (1.0f + EPS_V_F * qv_u - qc_u);
            const float thv_env = 0.5f * (thvk + thvkp);
            const float buoy = GRAVITY_F * (thv_u - thv_env) / fmaxf(thv_env, 150.0f);
            w_u2 = (w_u2 + 2.0f * dz_mid * buoy) / (1.0f + 2.0f * eps_u * dz_mid);
            if (w_u2 > 0.01f) {
                mu_val = AU_EDMF * rho_if * sqrtf(w_u2);
            } else {
                updraft_alive = false;
            }
        }
        mu_scratch[idx_k] = mu_val;

        zk = zkp; dzk = dzkp; thbk = thbkp; pibk = pibkp;
        thk = thkp; pik = pikp; tk = tkp; pk = pkp; rhok = rhokp;
    }

    // 3. Implicit double Thomas vertical solver for [thp, qv, qc, qi] with sliding column thermo
    const float gamma_th = (wtv0 > 0.0f) ? fminf(5.0e-3f, 6.8f * wth0 / fmaxf(wscale * hpbl, 1.0f)) : 0.0f;
    const float gamma_qv = (wtv0 > 0.0f && lhf_v > 0.0f) ? fminf(2.0e-6f, 6.8f * wq0 / fmaxf(wscale * hpbl, 1.0f)) : 0.0f;

    for (int sp = 0; sp < 4; ++sp) {
        if (sp > 0 && (!cfg.has_moist || !qv)) break;
        float* fptr = (sp == 0) ? thp : ((sp == 1) ? qv : ((sp == 2) ? qc : qi));
        if (!fptr) continue;

        double c_prev = 0.0;
        float z_km = 0.0f, thb_km = 0.0f, rho_km = 0.0f;
        float z_c = z_0, dz_c = dz_0, thb_c = thb0, pib_c, th_c, pi_c, t_c, p_c, rho_c = rho0;
        eval_cell_thermo(cfg, g, 0, col, hs, pi_sfc_f, thp, pip, qv, nullptr, nullptr, nullptr, nullptr, nullptr,
                         z_c, dz_c, thb_c, pib_c, th_c, pi_c, t_c, p_c, rho_c);

        for (int32_t k = 0; k < nz; ++k) {
            const size_t idx = static_cast<size_t>(k) * n2 + col;
            float z_kp = z_c, dz_kp = dz_c, thb_kp = thb_c, pib_kp, th_kp, pi_kp, t_kp, p_kp, rho_kp = rho_c;
            if (k + 1 < nz) {
                eval_cell_thermo(cfg, g, k + 1, col, hs, pi_sfc_f, thp, pip, qv, nullptr, nullptr, nullptr, nullptr, nullptr,
                                 z_kp, dz_kp, thb_kp, pib_kp, th_kp, pi_kp, t_kp, p_kp, rho_kp);
            }

            const double mass_k = static_cast<double>(rho_c * dz_c);
            double a_diag = 0.0, c_diag = 0.0;
            double explicit_src = 0.0;

            if (k > 0) {
                const size_t idx_km = static_cast<size_t>(k - 1) * n2 + col;
                const float dz_dn = fmaxf(1.0f, z_c - z_km);
                const float rho_dn = 0.5f * (rho_km + rho_c);
                const float kh_dn = kh_scratch[idx_km];
                const float mu_dn = mu_scratch[idx_km];
                const double g_dn = static_cast<double>(rho_dn * kh_dn / dz_dn);
                a_diag = -g_dn;

                const float z_if_dn = (static_cast<float>(g.zeta_w[k]) + static_cast<float>(g.b_w[k]) * hs) - hs;
                if (z_if_dn < hpbl) {
                    const float gg = (sp == 0) ? gamma_th : ((sp == 1) ? gamma_qv : 0.0f);
                    explicit_src += static_cast<double>(rho_dn * kh_dn * gg);
                }
                if (sp == 0) {
                    explicit_src -= static_cast<double>(rho_dn * kh_dn * (thb_c - thb_km) / dz_dn);
                }
                if (mu_dn > 0.0f && (sp == 0 || sp == 1)) {
                    const double env0 = static_cast<double>(fptr[col]) + ((sp == 0) ? static_cast<double>(thb0) : 0.0);
                    const double envk = static_cast<double>(fptr[idx]) + ((sp == 0) ? static_cast<double>(thb_c) : 0.0);
                    explicit_src += static_cast<double>(mu_dn) * 0.5 * (env0 - envk);
                }
            } else {
                if (sp == 0) {
                    explicit_src += static_cast<double>(shf_v / (CP_D_F * pi0));
                } else if (sp == 1) {
                    explicit_src += static_cast<double>(lhf_v / LV0_F);
                }
            }

            if (k + 1 < nz) {
                const float dz_up = fmaxf(1.0f, z_kp - z_c);
                const float rho_up = 0.5f * (rho_c + rho_kp);
                const float kh_up = kh_scratch[idx];
                const float mu_up = mu_scratch[idx];
                const double g_up = static_cast<double>(rho_up * kh_up / dz_up);
                c_diag = -g_up;

                const float z_if_up = (static_cast<float>(g.zeta_w[k + 1]) + static_cast<float>(g.b_w[k + 1]) * hs) - hs;
                if (z_if_up < hpbl) {
                    const float gg = (sp == 0) ? gamma_th : ((sp == 1) ? gamma_qv : 0.0f);
                    explicit_src -= static_cast<double>(rho_up * kh_up * gg);
                }
                if (sp == 0) {
                    explicit_src += static_cast<double>(rho_up * kh_up * (thb_kp - thb_c) / dz_up);
                }
                if (mu_up > 0.0f && (sp == 0 || sp == 1)) {
                    const double env0 = static_cast<double>(fptr[col]) + ((sp == 0) ? static_cast<double>(thb0) : 0.0);
                    const double envkp = static_cast<double>(fptr[static_cast<size_t>(k + 1) * n2 + col])
                                       + ((sp == 0) ? static_cast<double>(thb_kp) : 0.0);
                    explicit_src -= static_cast<double>(mu_up) * 0.5 * (env0 - envkp);
                }
            }

            const double b_diag = mass_k / dt_s - a_diag - c_diag;
            const double rhs = (mass_k / dt_s) * static_cast<double>(fptr[idx]) + explicit_src;
            const double denom = b_diag - a_diag * c_prev;
            c_prev = (k + 1 < nz) ? (c_diag / denom) : 0.0;
            const double d_prev = (k > 0) ? tri_d[static_cast<size_t>(k - 1) * n2 + col] : 0.0;
            tri_c[idx] = c_prev;
            tri_d[idx] = (rhs - a_diag * d_prev) / denom;

            z_km = z_c; thb_km = thb_c; rho_km = rho_c;
            z_c = z_kp; dz_c = dz_kp; thb_c = thb_kp; rho_c = rho_kp;
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

    const float pi_sfc_f = powf(static_cast<float>(cfg.p_sfc_pa) / P0_PA_F, KAPPA_F);
    const float th0_f = static_cast<float>(cfg.theta_sfc_k);
    const float nref_f = static_cast<float>(cfg.n_ref_s);
    const float hs = 0.5f * (static_cast<float>(g.hs[col_l]) + static_cast<float>(g.hs[col_r]));
    const float hpbl = pblh ? 0.5f * (pblh[col_l] + pblh[col_r]) : 500.0f;
    const float us = ustar ? fmaxf(0.05f, 0.5f * (ustar[col_l] + ustar[col_r])) : 0.15f;
    const float c_drag = sfc_drag_coeff ? 0.5f * (sfc_drag_coeff[col_l] + sfc_drag_coeff[col_r]) : 0.0f;

    float z_km = 0.0f, rhob_km = 0.0f;
    float z_c = static_cast<float>(g.zeta_s[0]) + static_cast<float>(g.b_s[0]) * hs;
    float rhob_c = eval_rho_bar_f32(z_c, th0_f, pi_sfc_f, nref_f);

    double c_prev = 0.0;
    for (int32_t k = 0; k < nz; ++k) {
        const size_t u_idx = static_cast<size_t>(k) * cfg.ny * nx_u + static_cast<size_t>(j) * nx_u + i;
        const size_t t_idx = static_cast<size_t>(k) * n2 + tri_col;
        const float dz_k = fmaxf(1.0f, (1.0f + static_cast<float>(g.db_s[k]) * hs) * static_cast<float>(g.dzeta[k]));
        const double mass_k = static_cast<double>(rhob_c * dz_k);

        double a_d = 0.0, c_d = 0.0, sfc_impl = 0.0;
        if (k == 0) {
            sfc_impl = static_cast<double>(c_drag);
        } else {
            const float z_if = (static_cast<float>(g.zeta_w[k]) + static_cast<float>(g.b_w[k]) * hs) - hs;
            const float km = (z_if < hpbl)
                ? fmaxf(0.1f, fminf(1000.0f, KARMAN_F * us * 2.0f * z_if * (1.0f - z_if / hpbl) * (1.0f - z_if / hpbl)))
                : 0.2f;
            a_d = -static_cast<double>((0.5f * (rhob_km + rhob_c)) * km / fmaxf(1.0f, z_c - z_km));
        }

        float z_kp = z_c, rhob_kp = rhob_c;
        if (k + 1 < nz) {
            z_kp = static_cast<float>(g.zeta_s[k + 1]) + static_cast<float>(g.b_s[k + 1]) * hs;
            rhob_kp = eval_rho_bar_f32(z_kp, th0_f, pi_sfc_f, nref_f);
            const float z_if = (static_cast<float>(g.zeta_w[k + 1]) + static_cast<float>(g.b_w[k + 1]) * hs) - hs;
            const float km = (z_if < hpbl)
                ? fmaxf(0.1f, fminf(1000.0f, KARMAN_F * us * 2.0f * z_if * (1.0f - z_if / hpbl) * (1.0f - z_if / hpbl)))
                : 0.2f;
            c_d = -static_cast<double>((0.5f * (rhob_c + rhob_kp)) * km / fmaxf(1.0f, z_kp - z_c));
        }
        const double b_d = mass_k / dt_s + sfc_impl - a_d - c_d;
        const double rhs = (mass_k / dt_s) * static_cast<double>(u[u_idx]);
        const double denom = b_d - a_d * c_prev;
        c_prev = (k + 1 < nz) ? (c_d / denom) : 0.0;
        const double d_prev = (k > 0) ? tri_d[static_cast<size_t>(k - 1) * n2 + tri_col] : 0.0;
        tri_c[t_idx] = c_prev;
        tri_d[t_idx] = (rhs - a_d * d_prev) / denom;

        z_km = z_c; rhob_km = rhob_c;
        z_c = z_kp; rhob_c = rhob_kp;
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

    const float pi_sfc_f = powf(static_cast<float>(cfg.p_sfc_pa) / P0_PA_F, KAPPA_F);
    const float th0_f = static_cast<float>(cfg.theta_sfc_k);
    const float nref_f = static_cast<float>(cfg.n_ref_s);
    const float hs = 0.5f * (static_cast<float>(g.hs[col_s]) + static_cast<float>(g.hs[col_n]));
    const float hpbl = pblh ? 0.5f * (pblh[col_s] + pblh[col_n]) : 500.0f;
    const float us = ustar ? fmaxf(0.05f, 0.5f * (ustar[col_s] + ustar[col_n])) : 0.15f;
    const float c_drag = sfc_drag_coeff ? 0.5f * (sfc_drag_coeff[col_s] + sfc_drag_coeff[col_n]) : 0.0f;

    float z_km = 0.0f, rhob_km = 0.0f;
    float z_c = static_cast<float>(g.zeta_s[0]) + static_cast<float>(g.b_s[0]) * hs;
    float rhob_c = eval_rho_bar_f32(z_c, th0_f, pi_sfc_f, nref_f);

    double c_prev = 0.0;
    for (int32_t k = 0; k < nz; ++k) {
        const size_t v_idx = static_cast<size_t>(k) * ny_v * cfg.nx + static_cast<size_t>(j) * cfg.nx + i;
        const size_t t_idx = static_cast<size_t>(k) * n2 + tri_col;
        const float dz_k = fmaxf(1.0f, (1.0f + static_cast<float>(g.db_s[k]) * hs) * static_cast<float>(g.dzeta[k]));
        const double mass_k = static_cast<double>(rhob_c * dz_k);

        double a_d = 0.0, c_d = 0.0, sfc_impl = 0.0;
        if (k == 0) {
            sfc_impl = static_cast<double>(c_drag);
        } else {
            const float z_if = (static_cast<float>(g.zeta_w[k]) + static_cast<float>(g.b_w[k]) * hs) - hs;
            const float km = (z_if < hpbl)
                ? fmaxf(0.1f, fminf(1000.0f, KARMAN_F * us * 2.0f * z_if * (1.0f - z_if / hpbl) * (1.0f - z_if / hpbl)))
                : 0.2f;
            a_d = -static_cast<double>((0.5f * (rhob_km + rhob_c)) * km / fmaxf(1.0f, z_c - z_km));
        }

        float z_kp = z_c, rhob_kp = rhob_c;
        if (k + 1 < nz) {
            z_kp = static_cast<float>(g.zeta_s[k + 1]) + static_cast<float>(g.b_s[k + 1]) * hs;
            rhob_kp = eval_rho_bar_f32(z_kp, th0_f, pi_sfc_f, nref_f);
            const float z_if = (static_cast<float>(g.zeta_w[k + 1]) + static_cast<float>(g.b_w[k + 1]) * hs) - hs;
            const float km = (z_if < hpbl)
                ? fmaxf(0.1f, fminf(1000.0f, KARMAN_F * us * 2.0f * z_if * (1.0f - z_if / hpbl) * (1.0f - z_if / hpbl)))
                : 0.2f;
            c_d = -static_cast<double>((0.5f * (rhob_c + rhob_kp)) * km / fmaxf(1.0f, z_kp - z_c));
        }
        const double b_d = mass_k / dt_s + sfc_impl - a_d - c_d;
        const double rhs = (mass_k / dt_s) * static_cast<double>(v[v_idx]);
        const double denom = b_d - a_d * c_prev;
        c_prev = (k + 1 < nz) ? (c_d / denom) : 0.0;
        const double d_prev = (k > 0) ? tri_d[static_cast<size_t>(k - 1) * n2 + tri_col] : 0.0;
        tri_c[t_idx] = c_prev;
        tri_d[t_idx] = (rhs - a_d * d_prev) / denom;

        z_km = z_c; rhob_km = rhob_c;
        z_c = z_kp; rhob_c = rhob_kp;
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

__device__ inline float mp_fall_speed(int sp_idx, float rho, float t_k, float q) {
    // sp_idx: 2=rain, 3=ice, 4=snow, 5=graupel (WSM6 Gamma-weighted terminal velocities)
    if (q <= 1e-12f) return 0.0f;
    const float rho_corr = sqrtf(1.225f / fmaxf(rho, 0.05f));
    const float rq = rho * q;
    if (rq <= 1e-18f) return 0.0f;
    if (sp_idx == 2) {
        const float lam_r = powf(PI_CONST_F * 1000.0f * 8.0e6f / rq, 0.25f);
        return fminf(12.0f, 2502.8f * rho_corr * powf(lam_r, -0.8f));
    }
    if (sp_idx == 3) {
        return fminf(1.2f, 0.32f * rho_corr * powf(fmaxf(q, 1e-9f), 0.16f));
    }
    if (sp_idx == 4) {
        const float n0s = fminf(2.0e8f, 2.0e6f * expf(0.12f * fmaxf(0.0f, T_FREEZE_F - t_k)));
        const float lam_s = powf(PI_CONST_F * 100.0f * n0s / rq, 0.25f);
        return fminf(3.0f, 23.59f * rho_corr * powf(lam_s, -0.41f));
    }
    const float lam_g = powf(PI_CONST_F * 500.0f * 4.0e6f / rq, 0.25f);
    return fminf(16.0f, 981.1f * rho_corr * powf(lam_g, -0.8f));
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
    const float dt_f = static_cast<float>(dt_s);
    const float pi_sfc_f = powf(static_cast<float>(cfg.p_sfc_pa) / P0_PA_F, KAPPA_F);
    const float th0_f = static_cast<float>(cfg.theta_sfc_k);
    const float nref_f = static_cast<float>(cfg.n_ref_s);
    const float hs = static_cast<float>(g.hs[col]);
    const float lsm_v = lsm ? lsm[col] : 1.0f;
    const float n_drop = static_cast<float>((lsm_v >= 0.5f) ? pcfg.cloud_num_land : pcfg.cloud_num_sea);
    constexpr float T_HOM = 236.15f; // -37C homogeneous freezing threshold (SLW preserved for T > 236.15 K!)

    // Phase-pair latent heat table [src][dst] in J/kg: 0=v, 1=c, 2=r, 3=i, 4=s, 5=g
    const float lat_heat[6][6] = {
        { 0.0f,  LV0_F,  LV0_F,  LS0_F,  LS0_F,  LS0_F},
        {-LV0_F,  0.0f,   0.0f,  LF0_F,  LF0_F,  LF0_F},
        {-LV0_F,  0.0f,   0.0f,  LF0_F,  LF0_F,  LF0_F},
        {-LS0_F, -LF0_F, -LF0_F,  0.0f,   0.0f,   0.0f},
        {-LS0_F, -LF0_F, -LF0_F,  0.0f,   0.0f,   0.0f},
        {-LS0_F, -LF0_F, -LF0_F,  0.0f,   0.0f,   0.0f}
    };

    // Pass 1: In-situ Thermodynamics + Saturation Adjustment + Transitions
    for (int32_t k = 0; k < nz; ++k) {
        const size_t idx = static_cast<size_t>(k) * n2 + col;
        float z_k, dz_k, thb, pib, th_k, pi_k, t_k, p_k, rho;
        eval_cell_thermo(cfg, g, k, col, hs, pi_sfc_f, thp, pip, qv, qc, qr, qi, qs, qg,
                         z_k, dz_k, thb, pib, th_k, pi_k, t_k, p_k, rho);
        const float rhob_r = eval_rho_bar_f32(z_k, th0_f, pi_sfc_f, nref_f);
        dz_scratch[idx] = dz_k;
        tri_mair[idx] = static_cast<double>(rhob_r) * static_cast<double>(dz_k);

        float q[6] = {
            fmaxf(0.0f, qv[idx]),
            fmaxf(0.0f, qc[idx]),
            fmaxf(0.0f, qr[idx]),
            fmaxf(0.0f, qi[idx]),
            fmaxf(0.0f, qs[idx]),
            fmaxf(0.0f, qg[idx])
        };

        // 1a. Fast melting (>0C) & homogeneous freezing (<-37C); SLW preserved for -37C..0C!
        if (t_k > T_FREEZE_F && q[3] > 0.0f) {
            const float dm = fminf(q[3], CP_D_F * (t_k - T_FREEZE_F) / LF0_F);
            q[3] -= dm; q[1] += dm; t_k -= dm * LF0_F / CP_D_F;
        }
        if (t_k <= T_HOM && q[1] > 0.0f) {
            q[3] += q[1]; t_k += q[1] * LF0_F / CP_D_F; q[1] = 0.0f;
        }

        // 1b. Fast condensation/evaporation saturation adjustment (2-iter Newton)
        const bool liq_phase = (t_k > T_HOM);
        const float l_sat = liq_phase ? LV0_F : LS0_F;
        for (int it = 0; it < 2; ++it) {
            const float qs_val = liq_phase ? qsat_water(t_k, p_k) : qsat_ice(t_k, p_k);
            const float dqs_dt = liq_phase ? dqsat_water_dt(t_k, p_k) : dqsat_ice_dt(t_k, p_k);
            const float dq = (q[0] - qs_val) / (1.0f + (l_sat / CP_D_F) * dqs_dt);
            if (dq > 0.0f) {
                const float c = fminf(dq, q[0]);
                q[0] -= c;
                if (liq_phase) q[1] += c; else q[3] += c;
                t_k += (l_sat / CP_D_F) * c;
            } else if (dq < 0.0f) {
                const float avail = liq_phase ? q[1] : q[3];
                const float e = fminf(-dq, avail);
                q[0] += e;
                if (liq_phase) q[1] -= e; else q[3] -= e;
                t_k -= (l_sat / CP_D_F) * e;
            }
        }

        // 1c. Slow microphysical transfer matrix sx2x[6][6]
        float sx2x[6][6] = {};
        const float rho_corr = sqrtf(1.225f / fmaxf(rho, 0.05f));
        const float k_diff = 1.41e-5f * (273.15f / t_k);

        if (q[1] > 1e-7f) {
            const float l_c = rho * q[1];
            const float l_r = rho * q[2];
            const float nc = fmaxf(n_drop, 2e7f);
            const float x_c = l_c / nc;
            const float tau = fminf(0.99f, l_r / fmaxf(l_c + l_r, 1e-12f));
            const float tau68 = powf(tau, 0.68f);
            const float omt = 1.0f - tau68;
            const float phi_au = 600.0f * tau68 * omt * omt * omt;
            const float om_tau = 1.0f - tau + 1e-3f;
            const float lx_scale = (6.80769e9f * l_c) * (x_c * 1e9f);
            const float au_sb = (lx_scale * (l_c * x_c))
                              * (1.0f + phi_au / (om_tau * om_tau)) * (1.225f / rho);
            sx2x[1][2] += au_sb / rho;

            constexpr float RC0_CUBE = 9.0e-6f * 9.0e-6f * 9.0e-6f;
            const float qc0 = (4.0f / 3.0f) * PI_CONST_F * RC0_CUBE * 1000.0f * nc / rho;
            if (q[1] > qc0) {
                const float qck1 = (0.104f * GRAVITY_F * 0.55f * powf(rho, 4.0f / 3.0f)) / (1.71e-5f * cbrtf(nc * 1000.0f));
                sx2x[1][2] += qck1 * powf(q[1], 7.0f / 3.0f);
            }

            if (q[2] > 1e-8f) {
                const float r_ac = tau / (tau + 5.0e-4f);
                const float r_ac2 = r_ac * r_ac;
                const float phi_ac = r_ac2 * r_ac2;
                sx2x[1][2] += (5.25f * l_c * l_r * phi_ac * rho_corr) / rho;
            }
        }

        if (q[2] > 1e-9f) {
            const float qsw = qsat_water(t_k, p_k);
            if (q[0] < qsw) {
                const float esw = es_water(t_k);
                const float ab = (LV0_F * LV0_F) / (0.024f * R_V_F * t_k * t_k) + (R_V_F * t_k) / (k_diff * (P0_PA_F / p_k) * esw);
                const float l_r = rho * q[2];
                const float vent = 0.78f * sqrtf(l_r) + 0.31f * sqrtf(rho_corr) * powf(l_r, 0.65f);
                sx2x[2][0] += fmaxf(0.0f, 2.0e3f * (1.0f - q[0] / qsw) * vent / (rho * fmaxf(ab, 1e4f)));
            }
        }

        if (t_k < T_FREEZE_F) {
            const float qsi = qsat_ice(t_k, p_k);
            const float dtemp = T_FREEZE_F - t_k;

            if ((t_k < T_FREEZE_F - 6.0f && q[1] > 1e-6f) || (t_k < T_FREEZE_F - 25.0f && q[0] > qsi)) {
                const float ni = fminf(2.5e5f, 5.0f * expf(0.304f * dtemp)) / rho;
                const float qi_tgt = ni * 1.0e-12f;
                if (qi_tgt > q[3]) {
                    sx2x[0][3] += (qi_tgt - q[3]) / dt_f;
                }
            }
            if (q[3] > 1e-11f) {
                const float f_dep = 1.0e-2f * (q[0] - qsi) / fmaxf(qsi, 1e-6f) * powf(q[3], 0.33f);
                if (q[0] > qsi) sx2x[0][3] += fmaxf(0.0f, fminf(f_dep, (q[0] - qsi) / dt_f));
                else sx2x[3][0] += fmaxf(0.0f, fminf(-f_dep, (qsi - q[0]) / dt_f));
            }
            if (q[3] > 1e-6f) {
                const float eff_stick = expf(0.025f * (t_k - T_FREEZE_F));
                sx2x[3][4] += 1.0e-3f * eff_stick * rho * q[3];
            }
            if (q[4] > 1e-8f) {
                if (q[1] > 1e-6f) sx2x[1][4] += 2.2f * rho * q[4] * q[1] * rho_corr;
                const float f_sdep = 4.0e-3f * (q[0] - qsi) / fmaxf(qsi, 1e-6f) * powf(rho * q[4], 0.6f);
                if (q[0] > qsi) sx2x[0][4] += fmaxf(0.0f, f_sdep);
                else sx2x[4][0] += fmaxf(0.0f, -f_sdep);
            }
            if (q[2] > 1e-7f && dtemp > 4.0f) {
                sx2x[2][5] += 0.1f * (expf(0.66f * dtemp) - 1.0f) * (rho * q[2] * q[2]);
                if (q[4] > 1e-6f || q[3] > 1e-6f) {
                    sx2x[2][5] += 4.0f * rho * q[2] * (q[4] + q[3]);
                }
            }
            if (q[5] > 1e-8f && q[1] > 1e-6f) {
                sx2x[1][5] += 5.0f * rho * q[5] * q[1] * rho_corr;
            }
            if (q[4] > 1e-5f && q[1] > 2e-4f) {
                sx2x[4][5] += 1.5e-2f * q[4] * (rho * q[1]);
            }
        } else {
            const float dtemp = t_k - T_FREEZE_F;
            if (q[4] > 1e-9f) {
                sx2x[4][2] += 1.2e-3f * dtemp * powf(rho * q[4], 0.65f) / rho;
            }
            if (q[5] > 1e-9f) {
                sx2x[5][2] += 8.0e-4f * dtemp * powf(rho * q[5], 0.65f) / rho;
            }
        }

        // ICON BSD-3 compute_sink proportional sink limiter (exact mass conservation + q >= 0)
        for (int s = 0; s < 6; ++s) {
            float total_sink = 0.0f;
            for (int d = 0; d < 6; ++d) {
                if (d != s) total_sink += sx2x[s][d] * dt_f;
            }
            if (total_sink > q[s] && total_sink > 0.0f) {
                const float scale = q[s] / total_sink;
                for (int d = 0; d < 6; ++d) {
                    if (d != s) sx2x[s][d] *= scale;
                }
            }
        }

        float dq[6] = {};
        float dh = 0.0f;
        for (int s = 0; s < 6; ++s) {
            for (int d = 0; d < 6; ++d) {
                if (s == d) continue;
                const float flux = sx2x[s][d] * dt_f;
                dq[s] -= flux;
                dq[d] += flux;
                dh += flux * lat_heat[s][d];
            }
        }
        for (int s = 0; s < 6; ++s) {
            q[s] = fmaxf(0.0f, q[s] + dq[s]);
            if (q[s] < 1e-13f) {
                if (s != 0) q[0] += q[s];
                q[s] = 0.0f;
            }
        }
        t_k = fmaxf(140.0f, t_k + dh / CP_D_F);

        thp[idx] = t_k / pi_k - thb;
        qv[idx] = q[0];
        qc[idx] = q[1];
        qr[idx] = q[2];
        qi[idx] = q[3];
        qs[idx] = q[4];
        qg[idx] = q[5];
    }

    // Pass 2: Subcycled Conservative Float64 Budget Column Fallout (qr, qi, qs, qg)
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
                const float z_k = static_cast<float>(g.zeta_s[k]) + static_cast<float>(g.b_s[k]) * hs;
                float thb = 0.0f, pib = 0.0f;
                eval_theta_pi_bar_f32(z_k, th0_f, pi_sfc_f, nref_f, thb, pib);
                const float pi_k = fmaxf(1e-4f, pib + pip[idx]);
                const float t_k = (thb + thp[idx]) * pi_k;
                const float p_k = P0_PA_F * powf(pi_k, 1.0f / KAPPA_F);
                const float rho_act = fmaxf(1e-5f, p_k / (R_D_F * t_k));
                const float vt = mp_fall_speed(sp_id, rho_act, t_k, static_cast<float>(q_k));
                max_cfl = fmax(max_cfl, static_cast<double>(vt) * dt_s / dz_k);
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
                    const float z_k = static_cast<float>(g.zeta_s[k]) + static_cast<float>(g.b_s[k]) * hs;
                    float thb = 0.0f, pib = 0.0f;
                    eval_theta_pi_bar_f32(z_k, th0_f, pi_sfc_f, nref_f, thb, pib);
                    const float pi_k = fmaxf(1e-4f, pib + pip[idx]);
                    const float t_k = (thb + thp[idx]) * pi_k;
                    const float p_k = P0_PA_F * powf(pi_k, 1.0f / KAPPA_F);
                    const float rho_act = fmaxf(1e-5f, p_k / (R_D_F * t_k));
                    const float vt = mp_fall_speed(sp_id, rho_act, t_k, static_cast<float>(q_k));
                    const double cfl = fmin(0.95, static_cast<double>(vt) * dt_sub / dz_k);
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
    float* lake_tbot, float* lake_hmld, float* lake_ctheta,
    float* sea_ice_frac, float* sea_ice_thick, float* sea_ice_tsfc, float* sea_ice_albedo
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
    if (sea_ice_frac) sea_ice_frac[col] = 0.0f;
    if (sea_ice_thick) sea_ice_thick[col] = 0.0f;
    if (sea_ice_tsfc) sea_ice_tsfc[col] = 271.35f;
    if (sea_ice_albedo) sea_ice_albedo[col] = 0.64f;
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
    NH_CUDA_CHECK(alloc_z(&h->d_sea_ice_frac, n2));
    NH_CUDA_CHECK(alloc_z(&h->d_sea_ice_thick, n2));
    NH_CUDA_CHECK(alloc_z(&h->d_sea_ice_tsfc, n2));
    NH_CUDA_CHECK(alloc_z(&h->d_sea_ice_albedo, n2));
    NH_CUDA_CHECK(alloc_z(&h->d_precip_liq_rate, n2));
    NH_CUDA_CHECK(alloc_z(&h->d_precip_ice_rate, n2));
    NH_CUDA_CHECK(alloc_z(&h->d_sfc_drag_coeff, n2));

    const uint32_t blk = 256;
    const uint32_t grd = static_cast<uint32_t>((n2 + blk - 1) / blk);
    nh_init_default_surface_kernel<<<grd, blk>>>(
        n2, h->d_lat_deg, h->d_lon_deg, h->d_lsm, h->d_lake_mask, h->d_soil_type,
        h->d_bathy_depth_m, h->d_z0m_base, h->d_albedo_base, h->d_veg_frac,
        h->d_t_skin, h->d_soil_t, h->d_soil_m, h->d_sst_bulk,
        h->d_lake_tbot, h->d_lake_hmld, h->d_lake_ctheta,
        h->d_sea_ice_frac, h->d_sea_ice_thick, h->d_sea_ice_tsfc, h->d_sea_ice_albedo
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
    cf(h->d_sea_ice_frac); cf(h->d_sea_ice_thick); cf(h->d_sea_ice_tsfc); cf(h->d_sea_ice_albedo);
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
    if (src->sea_ice_frac) {
        if (cp(h->d_sea_ice_frac, src->sea_ice_frac, n2) != 0) return -1;
    } else {
        NH_CUDA_CHECK(cudaMemset(h->d_sea_ice_frac, 0, n2 * sizeof(float)));
    }
    if (src->sea_ice_thick) {
        if (cp(h->d_sea_ice_thick, src->sea_ice_thick, n2) != 0) return -1;
    } else {
        NH_CUDA_CHECK(cudaMemset(h->d_sea_ice_thick, 0, n2 * sizeof(float)));
    }
    if (cp(h->d_sea_ice_tsfc, src->sea_ice_tsfc, n2) != 0) return -1;
    if (cp(h->d_sea_ice_albedo, src->sea_ice_albedo, n2) != 0) return -1;
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
    if (cp(dst->sea_ice_frac, h->d_sea_ice_frac, n2) != 0) return -1;
    if (cp(dst->sea_ice_thick, h->d_sea_ice_thick, n2) != 0) return -1;
    if (cp(dst->sea_ice_tsfc, h->d_sea_ice_tsfc, n2) != 0) return -1;
    if (cp(dst->sea_ice_albedo, h->d_sea_ice_albedo, n2) != 0) return -1;
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
                h->d_sea_ice_frac, h->d_sea_ice_tsfc, h->d_sea_ice_albedo,
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

    // 2. Surface Layer, Land LSM, Ocean COARE 3.6 + Skin + SI3 Sea-Ice Mosaic, FLake
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
            h->d_sea_ice_frac, h->d_sea_ice_thick, h->d_sea_ice_tsfc,
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

int32_t nh_cuda_sync_physics_cache(
    NhCudaHandle* h,
    float* rad_dth_dt,
    float* precip_liq_rate,
    float* precip_ice_rate,
    int32_t* rad_init,
    double* rad_timer,
    int32_t to_device
) {
    if (!h) return -1;
    if (ensure_physics_buffers(h) != 0) return -1;
    const size_t n2 = static_cast<size_t>(h->cfg.ny) * static_cast<size_t>(h->cfg.nx);
    if (to_device) {
        if (rad_dth_dt) NH_CUDA_CHECK(cudaMemcpy(h->d_rad_dth_dt, rad_dth_dt, h->n_s * sizeof(float), cudaMemcpyHostToDevice));
        if (precip_liq_rate) NH_CUDA_CHECK(cudaMemcpy(h->d_precip_liq_rate, precip_liq_rate, n2 * sizeof(float), cudaMemcpyHostToDevice));
        if (precip_ice_rate) NH_CUDA_CHECK(cudaMemcpy(h->d_precip_ice_rate, precip_ice_rate, n2 * sizeof(float), cudaMemcpyHostToDevice));
        if (rad_init) h->rad_initialized = *rad_init;
        if (rad_timer) h->rad_timer_s = *rad_timer;
    } else {
        if (rad_dth_dt) NH_CUDA_CHECK(cudaMemcpy(rad_dth_dt, h->d_rad_dth_dt, h->n_s * sizeof(float), cudaMemcpyDeviceToHost));
        if (precip_liq_rate) NH_CUDA_CHECK(cudaMemcpy(precip_liq_rate, h->d_precip_liq_rate, n2 * sizeof(float), cudaMemcpyDeviceToHost));
        if (precip_ice_rate) NH_CUDA_CHECK(cudaMemcpy(precip_ice_rate, h->d_precip_ice_rate, n2 * sizeof(float), cudaMemcpyDeviceToHost));
        if (rad_init) *rad_init = h->rad_initialized;
        if (rad_timer) *rad_timer = h->rad_timer_s;
    }
    return 0;
}

} // extern "C"
