#include "nh_ocean_si3.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace {

constexpr double O_GRAVITY = 9.80665;
constexpr double O_OMEGA = 7.2921159e-5;
constexpr double O_STEFAN = 5.670374419e-8;
constexpr double O_LV = 2.501e6;
constexpr double O_LS = 2.834e6;
constexpr double O_RT0 = 273.15;

// NEMO 5.0.2 / SI3 phycst.F90 & par_ice.F90 constants
constexpr double O_RHO0 = 1026.0;
constexpr double O_CP0 = 3991.867957;
constexpr double SI3_RHOI = 917.0;
constexpr double SI3_RHOS = 330.0;
constexpr double SI3_RHO_AIR = 1.225;
constexpr double SI3_CPI = 2096.7;
constexpr double SI3_LFI = 3.333601e5;
constexpr double SI3_KI0 = 2.034396;
constexpr double SI3_KS = 0.31;
constexpr double SI3_RTMLT = 0.054;
constexpr double SI3_ALB_SDRY = 0.85;
constexpr double SI3_ALB_SMLT = 0.75;
constexpr double SI3_ALB_IDRY = 0.64;
constexpr double SI3_ALB_IMLT = 0.53;
constexpr double SI3_ALB_DPND = 0.18;
constexpr double SI3_ALB_LPND = 0.30;
constexpr double SI3_ALB_HPIV = 1.0;
constexpr double SI3_ALB_OCE = 0.066;

__constant__ double c_cat_lo[TEMPEST_SI3_NCAT] = {0.0, 0.45, 1.10, 2.10, 4.00};
__constant__ double c_cat_hi[TEMPEST_SI3_NCAT] = {0.45, 1.10, 2.10, 4.00, 100.0};
__constant__ double c_cat_rep[TEMPEST_SI3_NCAT] = {0.25, 0.75, 1.55, 3.00, 5.50};

thread_local char g_oce_last_err[512] = "";

void oce_set_err(const char* msg) {
    snprintf(g_oce_last_err, sizeof(g_oce_last_err), "%s", msg ? msg : "unknown error");
}

void oce_set_cuda_err(const char* file, int line, cudaError_t err) {
    snprintf(g_oce_last_err, sizeof(g_oce_last_err),
             "CUDA error at %s:%d: %s", file, line, cudaGetErrorString(err));
}

#define OCE_CHECK(expr)                                         \
    do {                                                        \
        cudaError_t _e = (expr);                                \
        if (_e != cudaSuccess) {                                \
            oce_set_cuda_err(__FILE__, __LINE__, _e);           \
            return -1;                                          \
        }                                                       \
    } while (0)

// Exact Roquet et al. (2015) 75-term TEOS-10 polynomial (NEMO 5.0.2 eosbn2.F90 lines 258-289, 1926-1982)
__device__ __host__ inline double eval_teos10_rho_hd(double ct_c, double sa_psu, double depth_m) {
    constexpr double rdeltaS = 32.0;
    constexpr double r1_S0 = 0.875 / 35.16504;
    constexpr double r1_T0 = 1.0 / 40.0;
    constexpr double r1_Z0 = 1.0e-4;

    constexpr double EOS000 = 8.0189615746e+02;
    constexpr double EOS100 = 8.6672408165e+02;
    constexpr double EOS200 = -1.7864682637e+03;
    constexpr double EOS300 = 2.0375295546e+03;
    constexpr double EOS400 = -1.2849161071e+03;
    constexpr double EOS500 = 4.3227585684e+02;
    constexpr double EOS600 = -6.0579916612e+01;
    constexpr double EOS010 = 2.6010145068e+01;
    constexpr double EOS110 = -6.5281885265e+01;
    constexpr double EOS210 = 8.1770425108e+01;
    constexpr double EOS310 = -5.6888046321e+01;
    constexpr double EOS410 = 1.7681814114e+01;
    constexpr double EOS510 = -1.9193502195e+00;
    constexpr double EOS020 = -3.7074170417e+01;
    constexpr double EOS120 = 6.1548258127e+01;
    constexpr double EOS220 = -6.0362551501e+01;
    constexpr double EOS320 = 2.9130021253e+01;
    constexpr double EOS420 = -5.4723692739e+00;
    constexpr double EOS030 = 2.1661789529e+01;
    constexpr double EOS130 = -3.3449108469e+01;
    constexpr double EOS230 = 1.9717078466e+01;
    constexpr double EOS330 = -3.1742946532e+00;
    constexpr double EOS040 = -8.3627885467e+00;
    constexpr double EOS140 = 1.1311538584e+01;
    constexpr double EOS240 = -5.3563304045e+00;
    constexpr double EOS050 = 5.4048723791e-01;
    constexpr double EOS150 = 4.8169980163e-01;
    constexpr double EOS060 = -1.9083568888e-01;
    constexpr double EOS001 = 1.9681925209e+01;
    constexpr double EOS101 = -4.2549998214e+01;
    constexpr double EOS201 = 5.0774768218e+01;
    constexpr double EOS301 = -3.0938076334e+01;
    constexpr double EOS401 = 6.6051753097e+00;
    constexpr double EOS011 = -1.3336301113e+01;
    constexpr double EOS111 = -4.4870114575e+00;
    constexpr double EOS211 = 5.0042598061e+00;
    constexpr double EOS311 = -6.5399043664e-01;
    constexpr double EOS021 = 6.7080479603e+00;
    constexpr double EOS121 = 3.5063081279e+00;
    constexpr double EOS221 = -1.8795372996e+00;
    constexpr double EOS031 = -2.4649669534e+00;
    constexpr double EOS131 = -5.5077101279e-01;
    constexpr double EOS041 = 5.5927935970e-01;
    constexpr double EOS002 = 2.0660924175e+00;
    constexpr double EOS102 = -4.9527603989e+00;
    constexpr double EOS202 = 2.5019633244e+00;
    constexpr double EOS012 = 2.0564311499e+00;
    constexpr double EOS112 = -2.1311365518e-01;
    constexpr double EOS022 = -1.2419983026e+00;
    constexpr double EOS003 = -2.3342758797e-02;
    constexpr double EOS103 = -1.8507636718e-02;
    constexpr double EOS013 = 3.7969820455e-01;

    const double zh = fmax(0.0, depth_m) * r1_Z0;
    const double zt = fmax(-10.0, fmin(45.0, ct_c)) * r1_T0;
    const double sa_clamped = fmax(0.0, fmin(60.0, sa_psu));
    const double zs = sqrt(fabs(sa_clamped + rdeltaS) * r1_S0);

    const double zn3 = EOS013 * zt + EOS103 * zs + EOS003;
    const double zn2 = (EOS022 * zt + EOS112 * zs + EOS012) * zt
                     + (EOS202 * zs + EOS102) * zs + EOS002;
    const double zn1 = (((EOS041 * zt + EOS131 * zs + EOS031) * zt
                       + (EOS221 * zs + EOS121) * zs + EOS021) * zt
                      + ((EOS311 * zs + EOS211) * zs + EOS111) * zs + EOS011) * zt
                     + (((EOS401 * zs + EOS301) * zs + EOS201) * zs + EOS101) * zs + EOS001;
    const double zn0 = (((((EOS060 * zt + EOS150 * zs + EOS050) * zt
                         + (EOS240 * zs + EOS140) * zs + EOS040) * zt
                        + ((EOS330 * zs + EOS230) * zs + EOS130) * zs + EOS030) * zt
                       + (((EOS420 * zs + EOS320) * zs + EOS220) * zs + EOS120) * zs + EOS020) * zt
                      + ((((EOS510 * zs + EOS410) * zs + EOS310) * zs + EOS210) * zs + EOS110) * zs + EOS010) * zt
                     + (((((EOS600 * zs + EOS500) * zs + EOS400) * zs + EOS300) * zs + EOS200) * zs + EOS100) * zs + EOS000;

    return ((zn3 * zh + zn2) * zh + zn1) * zh + zn0;
}

// Exact NEMO 5.0.2 np_teos10 freezing temperature [Celsius] (eosbn2.F90 lines 359-364)
__device__ __host__ inline double eval_teos10_fzp_hd(double sa_psu, double depth_m) {
    constexpr double r1_S0 = 1.0 / 35.16504;
    const double s = fmax(0.0, sa_psu);
    const double zs = sqrt(s * r1_S0);
    return (((((1.46873e-03 * zs - 9.64972e-03) * zs + 2.28348e-02) * zs
              - 3.12775e-02) * zs + 2.07679e-02) * zs - 5.87701e-02) * s
         - 7.53e-4 * fmax(0.0, depth_m);
}

// BL99 enthalpy <-> temperature conversions (NEMO 5.0.2 icevar.F90 lines 388-396 & icethd_do.F90 line 195)
__device__ inline double bl99_temp_c_from_enthalpy(double e_vol_jm3, double s_ice_psu) {
    const double tmlts = -SI3_RTMLT * fmax(0.0, s_ice_psu);
    const double b = (O_CP0 - SI3_CPI) * tmlts + e_vol_jm3 / SI3_RHOI - SI3_LFI;
    const double disc = fmax(0.0, b * b - 4.0 * SI3_CPI * SI3_LFI * tmlts);
    const double t_c = -0.5 * (b + sqrt(disc)) / SI3_CPI;
    return fmax(-100.0, fmin(t_c, tmlts));
}

__device__ inline double bl99_enthalpy_from_temp_c(double t_c, double s_ice_psu) {
    const double tmlts = -SI3_RTMLT * fmax(0.0, s_ice_psu);
    const double tc_clamped = fmin(t_c, tmlts - 1e-6);
    const double frac_melt = (tmlts < -1e-9) ? fmax(0.0, 1.0 - tmlts / fmin(tc_clamped, -1e-9)) : 1.0;
    return SI3_RHOI * (SI3_CPI * (tmlts - tc_clamped) + SI3_LFI * frac_melt - O_CP0 * tmlts);
}

__device__ inline int32_t wrap_i(int32_t i, int32_t nx, int32_t bc_x, int32_t& valid) {
    if (bc_x == 1) {
        int32_t m = i % nx;
        valid = 1;
        return (m < 0) ? (m + nx) : m;
    }
    if (i < 0) { valid = 0; return 0; }
    if (i >= nx) { valid = 0; return nx - 1; }
    valid = 1;
    return i;
}

__device__ inline int32_t clamp_j(int32_t j, int32_t ny, int32_t& valid) {
    if (j < 0) { valid = 0; return 0; }
    if (j >= ny) { valid = 0; return ny - 1; }
    valid = 1;
    return j;
}

__device__ inline void eval_dx_dy(
    NhOceanConfig cfg, const float* msf, size_t col, double& dx_eff, double& dy_eff
) {
    const double m = msf ? fmax(0.05, fmin(120.0, static_cast<double>(msf[col]))) : 1.0;
    dx_eff = cfg.dx_m / m;
    dy_eff = cfg.is_spherical ? cfg.dy_m : (cfg.dy_m / m);
}

} // namespace

struct NhOceanSi3Handle {
    NhOceanConfig cfg{};
    size_t n2 = 0;
    size_t n3 = 0;
    double min_dx_mid = 1000.0;

    double *d_dz = nullptr, *d_zt = nullptr, *d_zw = nullptr;
    float *d_msf = nullptr, *d_fcor = nullptr, *d_bathy = nullptr, *d_tmask = nullptr;
    int32_t *d_mbkt = nullptr;

    // 3D ocean prognostic & turbulence/density arrays [nz, ny, nx] (temp/sal in FP64 + FP64 scratch)
    double *d_temp = nullptr, *d_sal = nullptr, *d_tmp_tr = nullptr;
    float *d_u = nullptr, *d_v = nullptr, *d_w = nullptr;
    float *d_tke = nullptr, *d_avm = nullptr, *d_avt = nullptr, *d_rhd = nullptr;
    float *d_tmp1 = nullptr, *d_tmp2 = nullptr;

    // 2D barotropic & coupled flux fields [ny, nx] (ssh in FP64)
    double *d_ssh = nullptr;
    float *d_ub = nullptr, *d_vb = nullptr;
    float *d_ub_rhs = nullptr, *d_vb_rhs = nullptr, *d_eta_ib = nullptr, *d_mld = nullptr;
    float *d_taux_atm = nullptr, *d_tauy_atm = nullptr;
    float *d_taux_oce = nullptr, *d_tauy_oce = nullptr;
    float *d_qsr_oce = nullptr, *d_qns_oce = nullptr, *d_emp_oce = nullptr;
    float *d_qsr_tot = nullptr, *d_qns_tot = nullptr, *d_emp_tot = nullptr, *d_sfx_tot = nullptr;

    // Forcing buffers [ny, nx]
    float *d_f_u10 = nullptr, *d_f_v10 = nullptr, *d_f_t2m = nullptr, *d_f_q2m = nullptr, *d_f_mslp = nullptr;
    float *d_f_sw = nullptr, *d_f_lw = nullptr, *d_f_pliq = nullptr, *d_f_pice = nullptr, *d_f_cld = nullptr;

    // SI3 5-category sea-ice state
    float *d_a_i = nullptr, *d_v_i = nullptr, *d_v_s = nullptr, *d_sv_i = nullptr;
    float *d_e_i = nullptr, *d_t_su = nullptr, *d_t_s = nullptr, *d_a_ip = nullptr, *d_v_ip = nullptr;
    float *d_u_ice = nullptr, *d_v_ice = nullptr;
    float *d_s1_ice = nullptr, *d_s2_ice = nullptr, *d_s12_ice = nullptr;
    float *d_strength = nullptr, *d_delta_ice = nullptr, *d_div_ice = nullptr;
    float *d_ice_frac = nullptr, *d_ice_thick = nullptr, *d_ice_tsfc = nullptr, *d_ice_albedo = nullptr;
};

namespace {

__global__ void oce_init_masks_kernel(
    int32_t nz, size_t n2, const double* z_w, const float* bathy, const float* ocean_mask,
    float* tmask, int32_t* mbkt, float* ice_frac, float* ice_thick, float* ice_tsfc, float* ice_alb
) {
    const size_t col = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (col >= n2) return;
    const float b = bathy ? bathy[col] : 0.0f;
    const float m = ocean_mask ? ocean_mask[col] : ((b > 1.0f) ? 1.0f : 0.0f);
    const bool wet = (m >= 0.5f) && (b > 1.0f);
    tmask[col] = wet ? 1.0f : 0.0f;
    int32_t kbot = 0;
    if (wet) {
        kbot = 1;
        for (int32_t k = 1; k < nz; ++k) {
            if (static_cast<double>(b) > z_w[k]) {
                kbot = k + 1;
            }
        }
    }
    mbkt[col] = kbot;
    if (ice_frac) ice_frac[col] = 0.0f;
    if (ice_thick) ice_thick[col] = 0.0f;
    if (ice_tsfc) ice_tsfc[col] = static_cast<float>(O_RT0 - 1.8);
    if (ice_alb) ice_alb[col] = static_cast<float>(SI3_ALB_OCE);
}

// 1. Air-sea & air-ice surface bulk fluxes + Inverse Barometer potential
__global__ void oce_surface_forcing_kernel(
    NhOceanConfig cfg, const float* tmask,
    const float* u10m, const float* v10m, const float* t2m, const float* q2m, const float* mslp,
    const float* sw_dn, const float* lw_dn, const float* p_liq, const float* p_ice,
    const double* temp_k,
    float* eta_ib, float* taux_atm, float* tauy_atm, float* qsr_oce, float* qns_oce, float* emp_oce
) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= cfg.ny || i >= cfg.nx) return;
    const size_t col = static_cast<size_t>(j) * cfg.nx + i;

    if (tmask[col] < 0.5f) {
        eta_ib[col] = 0.0f;
        taux_atm[col] = 0.0f;
        tauy_atm[col] = 0.0f;
        qsr_oce[col] = 0.0f;
        qns_oce[col] = 0.0f;
        emp_oce[col] = 0.0f;
        return;
    }

    const double p_mslp = mslp ? fmax(85000.0, fmin(110000.0, static_cast<double>(mslp[col]))) : cfg.p_ref_pa;
    eta_ib[col] = static_cast<float>(-(p_mslp - cfg.p_ref_pa) / (cfg.rho0 * O_GRAVITY));

    const double u10 = u10m ? static_cast<double>(u10m[col]) : 0.0;
    const double v10 = v10m ? static_cast<double>(v10m[col]) : 0.0;
    const double wspd = fmax(0.5, sqrt(u10 * u10 + v10 * v10));
    const double ta = t2m ? fmax(190.0, fmin(330.0, static_cast<double>(t2m[col]))) : 285.0;
    const double qa = q2m ? fmax(1e-6, fmin(0.05, static_cast<double>(q2m[col]))) : 0.006;
    const double ts = fmax(268.0, fmin(315.0, temp_k[col]));

    // Large & Pond / COARE-like open-ocean bulk coefficients
    const double cd = (wspd < 11.0) ? 1.2e-3 : fmin(2.6e-3, (0.49 + 0.065 * wspd) * 1.0e-3);
    const double ch = (ts > ta) ? 1.25e-3 : 0.9e-3;
    const double ce = 1.2e-3;

    taux_atm[col] = static_cast<float>(SI3_RHO_AIR * cd * wspd * u10);
    tauy_atm[col] = static_cast<float>(SI3_RHO_AIR * cd * wspd * v10);

    const double sw = sw_dn ? fmax(0.0, static_cast<double>(sw_dn[col])) : 0.0;
    const double lw = lw_dn ? fmax(40.0, static_cast<double>(lw_dn[col])) : 300.0;
    const double pl = p_liq ? fmax(0.0, static_cast<double>(p_liq[col])) : 0.0;
    const double pi = p_ice ? fmax(0.0, static_cast<double>(p_ice[col])) : 0.0;

    // Buck saturation vapor pressure over seawater (2% vapor pressure depression)
    const double tc_s = ts - O_RT0;
    const double es = 0.98 * 611.21 * exp((18.678 - tc_s / 234.5) * (tc_s / (257.14 + tc_s)));
    const double qs = 0.622 * es / fmax(10000.0, p_mslp - 0.378 * es);

    const double shf_dn = SI3_RHO_AIR * 1004.64 * ch * wspd * (ta - ts); // >0 warms ocean
    const double evap = fmax(-5e-4, SI3_RHO_AIR * ce * wspd * (qs - qa)); // [kg m^-2 s^-1], >0 loss
    const double lhf_dn = -O_LV * evap;
    const double ts2 = ts * ts;
    const double lw_net = 0.97 * (lw - O_STEFAN * ts2 * ts2);
    const double snow_melt_heat = -pi * SI3_LFI; // latent cooling if snow falls into open ocean

    qsr_oce[col] = static_cast<float>((1.0 - SI3_ALB_OCE) * sw);
    qns_oce[col] = static_cast<float>(lw_net + shf_dn + lhf_dn + snow_melt_heat);
    emp_oce[col] = static_cast<float>(evap - pl - pi); // positive = net mass loss (evap - precip)
}

// 2. TEOS-10 Density + Brunt-Vaisala N^2 + Prognostic Gaspar TKE + EVD + MLD
__global__ void oce_eos_tke_kernel(
    NhOceanConfig cfg, double dt_s,
    const double* dz, const double* z_t, const double* z_w,
    const float* tmask, const int32_t* mbkt,
    const double* temp_k, const double* sal_psu,
    const float* u_oce, const float* v_oce,
    const float* taux_oce, const float* tauy_oce, const float* ice_frac,
    float* rhd, float* tke, float* avm, float* avt, float* mld
) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= cfg.ny || i >= cfg.nx) return;
    const size_t col = static_cast<size_t>(j) * cfg.nx + i;
    const size_t n2 = static_cast<size_t>(cfg.ny) * cfg.nx;

    const int32_t kbot = mbkt[col];
    if (tmask[col] < 0.5f || kbot <= 0) {
        mld[col] = 0.0f;
        for (int32_t k = 0; k < cfg.nz; ++k) {
            const size_t idx = static_cast<size_t>(k) * n2 + col;
            rhd[idx] = 0.0f;
            tke[idx] = 0.0f;
            avm[idx] = 0.0f;
            avt[idx] = 0.0f;
        }
        return;
    }

    int32_t valid;
    const int32_t im = wrap_i(i - 1, cfg.nx, cfg.bc_x, valid);
    const size_t col_w = static_cast<size_t>(j) * cfg.nx + im;
    const int32_t jm = clamp_j(j - 1, cfg.ny, valid);
    const size_t col_s = static_cast<size_t>(jm) * cfg.nx + i;

    double rho_0_surf = 1026.0;
    double mld_val = z_t[kbot - 1];
    bool found_mld = false;

    for (int32_t k = 0; k < cfg.nz; ++k) {
        const size_t idx = static_cast<size_t>(k) * n2 + col;
        if (k >= kbot) {
            rhd[idx] = 0.0f;
            tke[idx] = 0.0f;
            avm[idx] = 0.0f;
            avt[idx] = 0.0f;
            continue;
        }
        const double ct_c = temp_k[idx] - O_RT0;
        const double sa = sal_psu[idx];
        const double rho = eval_teos10_rho_hd(ct_c, sa, z_t[k]);
        rhd[idx] = static_cast<float>((rho - cfg.rho0) / cfg.rho0);

        const double rho_pot = eval_teos10_rho_hd(ct_c, sa, 0.0);
        if (k == 0) {
            rho_0_surf = rho_pot;
        } else if (!found_mld && (rho_pot - rho_0_surf) >= 0.03) {
            mld_val = z_t[k];
            found_mld = true;
        }
    }
    mld[col] = static_cast<float>(mld_val);

    // NEMO zdftke.F90 surface TKE BC with sea-ice attenuation
    constexpr double rn_ediff = 0.1;
    constexpr double rn_ediss = 0.7;
    constexpr double rn_ebb = 67.83;
    constexpr double rn_emin = 1.0e-6;
    constexpr double rn_emin0 = 1.0e-4;
    const double tx = taux_oce[col];
    const double ty = tauy_oce[col];
    const double tau_m = sqrt(tx * tx + ty * ty);
    const double ice_att = 1.0 - fmin(1.0, 1.0 * static_cast<double>(ice_frac[col]));
    const double tke_sfc = fmax(rn_emin, ice_att * fmax(rn_emin0, rn_ebb * tau_m / cfg.rho0));

    tke[col] = static_cast<float>(tke_sfc);
    avm[col] = static_cast<float>(cfg.avm_bg);
    avt[col] = static_cast<float>(cfg.avt_bg);

    for (int32_t k = 1; k < kbot; ++k) {
        const size_t idx_up = static_cast<size_t>(k - 1) * n2 + col;
        const size_t idx_dn = static_cast<size_t>(k) * n2 + col;
        const double dz_w = 0.5 * (dz[k - 1] + dz[k]);

        // Local adiabatic N^2 at interface z_w[k]
        const double rho_up = eval_teos10_rho_hd(temp_k[idx_up] - O_RT0, sal_psu[idx_up], z_w[k]);
        const double rho_dn = eval_teos10_rho_hd(temp_k[idx_dn] - O_RT0, sal_psu[idx_dn], z_w[k]);
        const double n2_bv = (O_GRAVITY / cfg.rho0) * (rho_dn - rho_up) / dz_w;

        // Vertical velocity shear squared
        const double u_up = 0.5 * (u_oce[idx_up] + u_oce[static_cast<size_t>(k - 1) * n2 + col_w]);
        const double u_dn = 0.5 * (u_oce[idx_dn] + u_oce[static_cast<size_t>(k) * n2 + col_w]);
        const double v_up = 0.5 * (v_oce[idx_up] + v_oce[static_cast<size_t>(k - 1) * n2 + col_s]);
        const double v_dn = 0.5 * (v_oce[idx_dn] + v_oce[static_cast<size_t>(k) * n2 + col_s]);
        const double du = (u_up - u_dn) / dz_w;
        const double dv = (v_up - v_dn) / dz_w;
        const double sh2 = du * du + dv * dv;

        const double e_old = fmax(rn_emin, static_cast<double>(tke[idx_dn]));
        const double avm_old = fmax(cfg.avm_bg, static_cast<double>(avm[idx_dn]));
        const double avt_old = fmax(cfg.avt_bg, static_cast<double>(avt[idx_dn]));

        // Mixing length bounded by Ozmidov/buoyancy scale & wall distance
        const double l_buoy = (n2_bv > 1e-10) ? sqrt(2.0 * e_old / n2_bv) : 100.0;
        const double l_wall = fmin(0.4 * z_w[k], fmax(0.5, 0.4 * (z_w[kbot] - z_w[k])));
        const double l_m = fmax(0.5, fmin(fmin(l_buoy, l_wall), 80.0));

        // Axell (2002) Langmuir turbulence production in mixed layer (rn_lc = 0.15)
        const double w_lc = 0.016 * sqrt(fmax(0.0, tau_m) / (SI3_RHO_AIR * 1.5e-3));
        const double h_lc = fmax(5.0, mld_val);
        const double lang_prod = (z_w[k] < h_lc && ice_att > 0.01)
            ? ice_att * pow(0.15 * w_lc * sin(3.14159265358979 * z_w[k] / h_lc), 3.0) / h_lc
            : 0.0;

        // Semi-implicit TKE update
        const double prod = avm_old * sh2 + lang_prod - avt_old * fmax(0.0, n2_bv);
        const double damp = rn_ediss * sqrt(e_old) / l_m;
        const double e_new = fmax(rn_emin, fmin(0.5, (e_old + dt_s * fmax(0.0, prod)) / (1.0 + dt_s * damp)));
        tke[idx_dn] = static_cast<float>(e_new);

        const double ri = fmax(0.0, n2_bv) / fmax(sh2, 1.0e-10);
        const double pr = (ri > 0.2) ? fmin(10.0, 5.0 * ri) : 1.0;
        const double k_m_tke = rn_ediff * l_m * sqrt(e_new);
        const double evd = (n2_bv < -1.0e-8) ? 10.0 : 0.0; // NEMO zdfevd convection

        avm[idx_dn] = static_cast<float>(fmin(10.0, fmax(cfg.avm_bg, k_m_tke + evd)));
        avt[idx_dn] = static_cast<float>(fmin(10.0, fmax(cfg.avt_bg, k_m_tke / pr + evd)));
    }
}

// 3. SI3 Sea-Ice mEVP Rheology Kernels (Bouillon 2013 / NEMO 5.0.2 icedyn_rhg_evp.F90)
__global__ void si3_mevp_init_kernel(
    NhOceanConfig cfg, double dt_evp,
    const float* tmask, const float* msf, const float* bathy,
    const float* a_i, const float* v_i,
    float* strength
) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= cfg.ny || i >= cfg.nx) return;
    const size_t n2 = static_cast<size_t>(cfg.ny) * cfg.nx;
    const size_t col = static_cast<size_t>(j) * cfg.nx + i;
    (void)dt_evp; (void)msf; (void)bathy;
    if (tmask[col] < 0.5f) {
        strength[col] = 0.0f;
        return;
    }
    double at = 0.0, vt = 0.0;
    for (int32_t c = 0; c < TEMPEST_SI3_NCAT; ++c) {
        at += fmax(0.0, static_cast<double>(a_i[static_cast<size_t>(c) * n2 + col]));
        vt += fmax(0.0, static_cast<double>(v_i[static_cast<size_t>(c) * n2 + col]));
    }
    at = fmin(1.0, at);
    strength[col] = static_cast<float>((at > 1e-4) ? (cfg.rn_pstar * vt * exp(-cfg.rn_crhg * (1.0 - at))) : 0.0);
}

__global__ void si3_mevp_stress_kernel(
    NhOceanConfig cfg, const float* tmask, const float* msf,
    const float* a_i, const float* strength, const float* u_ice, const float* v_ice,
    float* s1, float* s2, float* s12, float* delta_out, float* div_out
) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= cfg.ny || i >= cfg.nx) return;
    const size_t n2 = static_cast<size_t>(cfg.ny) * cfg.nx;
    const size_t col = static_cast<size_t>(j) * cfg.nx + i;

    if (tmask[col] < 0.5f) {
        s1[col] = 0.0f; s2[col] = 0.0f; s12[col] = 0.0f;
        if (delta_out) delta_out[col] = 0.0f;
        if (div_out) div_out[col] = 0.0f;
        return;
    }

    double at = 0.0;
    for (int32_t c = 0; c < TEMPEST_SI3_NCAT; ++c) {
        at += fmax(0.0, static_cast<double>(a_i[static_cast<size_t>(c) * n2 + col]));
    }
    if (at < 1.0e-5) {
        s1[col] = 0.0f; s2[col] = 0.0f; s12[col] = 0.0f;
        if (delta_out) delta_out[col] = 0.0f;
        if (div_out) div_out[col] = 0.0f;
        return;
    }

    double dx, dy;
    eval_dx_dy(cfg, msf, col, dx, dy);
    int32_t v_im, v_ip, v_jm, v_jp;
    const int32_t im_idx = wrap_i(i - 1, cfg.nx, cfg.bc_x, v_im);
    const int32_t ip = wrap_i(i + 1, cfg.nx, cfg.bc_x, v_ip);
    const int32_t jm = clamp_j(j - 1, cfg.ny, v_jm);
    const int32_t jp_idx = clamp_j(j + 1, cfg.ny, v_jp);

    const double ue = static_cast<double>(u_ice[col]);
    const double uw_e = v_im ? static_cast<double>(u_ice[static_cast<size_t>(j) * cfg.nx + im_idx]) : 0.0;
    const double vn = static_cast<double>(v_ice[col]);
    const double vs_e = v_jm ? static_cast<double>(v_ice[static_cast<size_t>(jm) * cfg.nx + i]) : 0.0;

    const double u_n = v_jp ? static_cast<double>(u_ice[static_cast<size_t>(jp_idx) * cfg.nx + i]) : ue;
    const double u_s = v_jm ? static_cast<double>(u_ice[static_cast<size_t>(jm) * cfg.nx + i]) : ue;
    const double v_e = v_ip ? static_cast<double>(v_ice[static_cast<size_t>(j) * cfg.nx + ip]) : vn;
    const double v_w = v_im ? static_cast<double>(v_ice[static_cast<size_t>(j) * cfg.nx + im_idx]) : vn;

    const double e11 = (ue - uw_e) / dx;
    const double e22 = (vn - vs_e) / dy;
    const double div = e11 + e22;
    const double tens = e11 - e22;
    const double shear = 0.5 * ((u_n - u_s) / (2.0 * dy) + (v_e - v_w) / (2.0 * dx));

    const double ecc2 = cfg.rn_ecc * cfg.rn_ecc;
    const double delta = sqrt(div * div + (tens * tens + shear * shear) / ecc2);
    const double p_str = 0.5 * static_cast<double>(strength[col]) + 0.125 * (
        static_cast<double>(strength[static_cast<size_t>(j) * cfg.nx + ip]) +
        static_cast<double>(strength[static_cast<size_t>(j) * cfg.nx + im_idx]) +
        static_cast<double>(strength[static_cast<size_t>(jp_idx) * cfg.nx + i]) +
        static_cast<double>(strength[static_cast<size_t>(jm) * cfg.nx + i])
    );
    const double p_d = p_str / (delta + 2.0e-9);

    // Bouillon mEVP alpha_evp = 500 -> subcycle elastic wave damping
    constexpr double alpha_evp = 500.0;
    const double inv_a1 = 1.0 / (alpha_evp + 1.0);
    s1[col] = static_cast<float>((static_cast<double>(s1[col]) * alpha_evp + p_d * (div - delta)) * inv_a1);
    s2[col] = static_cast<float>((static_cast<double>(s2[col]) * alpha_evp + p_d * (tens / ecc2)) * inv_a1);
    s12[col] = static_cast<float>((static_cast<double>(s12[col]) * alpha_evp + p_d * (shear * 0.5 / ecc2)) * inv_a1);

    if (delta_out) delta_out[col] = static_cast<float>(delta);
    if (div_out) div_out[col] = static_cast<float>(div);
}

__global__ void si3_mevp_vel_kernel(
    NhOceanConfig cfg, double dt_step,
    const float* tmask, const float* msf, const float* fcor,
    const float* a_i, const float* v_i, const float* v_s,
    const float* u_oce, const float* v_oce, const double* ssh,
    const float* taux_atm, const float* tauy_atm,
    const float* s1, const float* s2, const float* s12,
    const float* u_ice_in, const float* v_ice_in,
    float* u_ice_out, float* v_ice_out
) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= cfg.ny || i >= cfg.nx) return;
    const size_t n2 = static_cast<size_t>(cfg.ny) * cfg.nx;
    const size_t col = static_cast<size_t>(j) * cfg.nx + i;

    int32_t v_ip, v_im, v_jp, v_jm;
    const int32_t ip = wrap_i(i + 1, cfg.nx, cfg.bc_x, v_ip);
    const int32_t jp = clamp_j(j + 1, cfg.ny, v_jp);
    const int32_t jm = clamp_j(j - 1, cfg.ny, v_jm);
    const int32_t im = wrap_i(i - 1, cfg.nx, cfg.bc_x, v_im);
    (void)v_im; (void)v_jm;
    const size_t col_e = static_cast<size_t>(j) * cfg.nx + ip;
    const size_t col_n = static_cast<size_t>(jp) * cfg.nx + i;

    const bool umask = (tmask[col] >= 0.5f) && (v_ip || cfg.bc_x == 1) && (tmask[col_e] >= 0.5f);
    const bool vmask = (tmask[col] >= 0.5f) && (v_jp != 0) && (tmask[col_n] >= 0.5f);

    double dx, dy;
    eval_dx_dy(cfg, msf, col, dx, dy);
    // Stabilize polar convergence in explicit mEVP
    const double dx_safe = fmax(dx, 0.25 * cfg.dx_m);
    const double dy_safe = fmax(dy, 0.25 * cfg.dx_m);
    constexpr double beta_evp = 500.0;

    auto cell_mass_at = [&](size_t cidx, double& at_c, double& m_c) {
        at_c = 0.0;
        double vt_i = 0.0, vt_s = 0.0;
        for (int32_t k = 0; k < TEMPEST_SI3_NCAT; ++k) {
            at_c += fmax(0.0, static_cast<double>(a_i[static_cast<size_t>(k) * n2 + cidx]));
            vt_i += fmax(0.0, static_cast<double>(v_i[static_cast<size_t>(k) * n2 + cidx]));
            vt_s += fmax(0.0, static_cast<double>(v_s[static_cast<size_t>(k) * n2 + cidx]));
        }
        at_c = fmin(1.0, at_c);
        m_c = SI3_RHOI * vt_i + SI3_RHOS * vt_s;
    };

    double at0, m0, ate, me, atn, mn;
    cell_mass_at(col, at0, m0);
    cell_mass_at(col_e, ate, me);
    cell_mass_at(col_n, atn, mn);

    if (umask && (m0 + me) > 1.0 && (at0 + ate) > 1e-4) {
        const double au = 0.5 * (at0 + ate);
        const double mu = fmax(1.0, 0.5 * (m0 + me));
        const double m_dt = mu / dt_step;
        const double ui = static_cast<double>(u_ice_in[col]);
        const double vi_avg = 0.25 * (v_ice_in[col] + v_ice_in[col_e] +
                                      v_ice_in[static_cast<size_t>(jm) * cfg.nx + i] +
                                      v_ice_in[static_cast<size_t>(jm) * cfg.nx + ip]);
        const double uo = static_cast<double>(u_oce[col]);
        const double vo_avg = 0.25 * (v_oce[col] + v_oce[col_e] +
                                      v_oce[static_cast<size_t>(jm) * cfg.nx + i] +
                                      v_oce[static_cast<size_t>(jm) * cfg.nx + ip]);
        const double du = ui - uo, dv = vi_avg - vo_avg;
        const double drag_io = cfg.rho0 * cfg.rn_cd_io * au * sqrt(du * du + dv * dv);
        const double dsx = 0.5 * ((s1[col_e] + s2[col_e]) - (s1[col] + s2[col])) / dx_safe
                         + (s12[col_n] - s12[static_cast<size_t>(jm) * cfg.nx + i]) / (2.0 * dy_safe);
        const double tau_a = 0.5 * (taux_atm[col] + taux_atm[col_e]) * au;
        const double fc = 0.5 * (fcor[col] + fcor[col_e]);
        const double cor = mu * fc * vi_avg;
        const double spg = -mu * O_GRAVITY * (ssh[col_e] - ssh[col]) / dx_safe;
        const double rhs = dsx + tau_a + cor + spg + drag_io * uo;
        const double u_next = (m_dt * ((beta_evp + 1.0) * ui) + rhs) / (m_dt * (beta_evp + 1.0) + drag_io);
        u_ice_out[col] = static_cast<float>(fmax(-3.5, fmin(3.5, u_next)));
    } else {
        u_ice_out[col] = 0.0f;
    }

    if (vmask && (m0 + mn) > 1.0 && (at0 + atn) > 1e-4) {
        const double av = 0.5 * (at0 + atn);
        const double mv = fmax(1.0, 0.5 * (m0 + mn));
        const double m_dt = mv / dt_step;
        const double vi = static_cast<double>(v_ice_in[col]);
        const double ui_avg = 0.25 * (u_ice_in[col] + u_ice_in[col_n] +
                                      u_ice_in[static_cast<size_t>(j) * cfg.nx + im] +
                                      u_ice_in[static_cast<size_t>(jp) * cfg.nx + im]);
        const double vo = static_cast<double>(v_oce[col]);
        const double uo_avg = 0.25 * (u_oce[col] + u_oce[col_n] +
                                      u_oce[static_cast<size_t>(j) * cfg.nx + im] +
                                      u_oce[static_cast<size_t>(jp) * cfg.nx + im]);
        const double du = ui_avg - uo_avg, dv = vi - vo;
        const double drag_io = cfg.rho0 * cfg.rn_cd_io * av * sqrt(du * du + dv * dv);
        const double dsy = 0.5 * ((s1[col_n] - s2[col_n]) - (s1[col] - s2[col])) / dy_safe
                         + (s12[col_e] - s12[static_cast<size_t>(j) * cfg.nx + im]) / (2.0 * dx_safe);
        const double tau_a = 0.5 * (tauy_atm[col] + tauy_atm[col_n]) * av;
        const double fc = 0.5 * (fcor[col] + fcor[col_n]);
        const double cor = -mv * fc * ui_avg;
        const double spg = -mv * O_GRAVITY * (ssh[col_n] - ssh[col]) / dy_safe;
        const double rhs = dsy + tau_a + cor + spg + drag_io * vo;
        const double v_next = (m_dt * ((beta_evp + 1.0) * vi) + rhs) / (m_dt * (beta_evp + 1.0) + drag_io);
        v_ice_out[col] = static_cast<float>(fmax(-3.5, fmin(3.5, v_next)));
    } else {
        v_ice_out[col] = 0.0f;
    }
}

// 4. Conservative Positive-Definite Upwind Transport for Any Scalar Field on C-Grid
__global__ void advect_upwind_2d_layer_kernel(
    NhOceanConfig cfg, double dt_s, int32_t layer_idx,
    const float* tmask, const float* msf,
    const float* u_face, const float* v_face,
    const float* field_in, float* field_out
) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= cfg.ny || i >= cfg.nx) return;
    const size_t n2 = static_cast<size_t>(cfg.ny) * cfg.nx;
    const size_t col = static_cast<size_t>(j) * cfg.nx + i;
    const size_t off = static_cast<size_t>(layer_idx) * n2;

    if (tmask[col] < 0.5f) {
        field_out[col] = 0.0f;
        return;
    }

    int32_t v_im, v_ip, v_jm, v_jp;
    const int32_t im = wrap_i(i - 1, cfg.nx, cfg.bc_x, v_im);
    const int32_t ip = wrap_i(i + 1, cfg.nx, cfg.bc_x, v_ip);
    const int32_t jm = clamp_j(j - 1, cfg.ny, v_jm);
    const int32_t jp = clamp_j(j + 1, cfg.ny, v_jp);

    const size_t col_w = static_cast<size_t>(j) * cfg.nx + im;
    const size_t col_e = static_cast<size_t>(j) * cfg.nx + ip;
    const size_t col_s = static_cast<size_t>(jm) * cfg.nx + i;
    const size_t col_n = static_cast<size_t>(jp) * cfg.nx + i;

    double dx, dy;
    eval_dx_dy(cfg, msf, col, dx, dy);

    const double fc_c = static_cast<double>(field_in[off + col]);
    const double fc_w = (v_im && tmask[col_w] >= 0.5f) ? static_cast<double>(field_in[off + col_w]) : 0.0;
    const double fc_e = (v_ip && tmask[col_e] >= 0.5f) ? static_cast<double>(field_in[off + col_e]) : 0.0;
    const double fc_s = (v_jm && tmask[col_s] >= 0.5f) ? static_cast<double>(field_in[off + col_s]) : 0.0;
    const double fc_n = (v_jp && tmask[col_n] >= 0.5f) ? static_cast<double>(field_in[off + col_n]) : 0.0;

    // Velocity on East (col), West (col_w), North (col), South (col_s) faces masked at walls
    double ue = (v_ip && tmask[col_e] >= 0.5f) ? static_cast<double>(u_face[col]) : 0.0;
    double uw = (v_im && tmask[col_w] >= 0.5f) ? static_cast<double>(u_face[col_w]) : 0.0;
    double vn = (v_jp && tmask[col_n] >= 0.5f) ? static_cast<double>(v_face[col]) : 0.0;
    double vs = (v_jm && tmask[col_s] >= 0.5f) ? static_cast<double>(v_face[col_s]) : 0.0;

    // Metric-aware Courant limit <= 0.45 guarantees strict positivity & no polar instability
    const double u_cap = 0.22 * dx / fmax(dt_s, 1e-6);
    const double v_cap = 0.22 * dy / fmax(dt_s, 1e-6);
    ue = fmax(-u_cap, fmin(u_cap, ue));
    uw = fmax(-u_cap, fmin(u_cap, uw));
    vn = fmax(-v_cap, fmin(v_cap, vn));
    vs = fmax(-v_cap, fmin(v_cap, vs));

    // Geometric dy factor on spherical East/West faces matches area element dx * dy
    const double f_e = (ue >= 0.0) ? (ue * fc_c) : (ue * fc_e);
    const double f_w = (uw >= 0.0) ? (uw * fc_w) : (uw * fc_c);
    const double f_n = (vn >= 0.0) ? (vn * fc_c) : (vn * fc_n);
    const double f_s = (vs >= 0.0) ? (vs * fc_s) : (vs * fc_c);

    // Note on spherical grid: area = dx(j) * dy, N/S face lengths are dx(j+-1/2)
    double dx_n = dx, dx_s = dx, dy_unused = dy;
    if (cfg.is_spherical) {
        double msf_n = v_jp ? static_cast<double>(msf[col_n]) : static_cast<double>(msf[col]);
        double msf_s = v_jm ? static_cast<double>(msf[col_s]) : static_cast<double>(msf[col]);
        dx_n = 0.5 * (dx + cfg.dx_m / fmax(0.05, fmin(120.0, msf_n)));
        dx_s = 0.5 * (dx + cfg.dx_m / fmax(0.05, fmin(120.0, msf_s)));
        (void)dy_unused;
    }
    const double area = dx * dy;
    const double net_div = ((f_e - f_w) * dy + (f_n * dx_n - f_s * dx_s)) / area;
    const double val = fc_c - dt_s * net_div;
    field_out[col] = static_cast<float>(val);
}

// 5. Thorndike Mechanical Ridging + ITD Category Remapping (caps sum a_i <= rn_amax < 1.0)
__global__ void si3_ridge_and_itd_remap_kernel(
    NhOceanConfig cfg, double dt_s, const float* tmask, const float* div_ice, const float* delta_ice,
    float* a_i, float* v_i, float* v_s, float* sv_i, float* e_i,
    float* t_su, float* t_s, float* a_ip, float* v_ip, float* qns_tot, float* emp_tot
) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= cfg.ny || i >= cfg.nx) return;
    const size_t n2 = static_cast<size_t>(cfg.ny) * cfg.nx;
    const size_t col = static_cast<size_t>(j) * cfg.nx + i;

    if (tmask[col] < 0.5f) return;

    double a[TEMPEST_SI3_NCAT], vi[TEMPEST_SI3_NCAT], vs[TEMPEST_SI3_NCAT], svi[TEMPEST_SI3_NCAT];
    double ei[TEMPEST_SI3_NCAT][TEMPEST_SI3_NLAY], tsu[TEMPEST_SI3_NCAT], ts[TEMPEST_SI3_NCAT];
    double aip[TEMPEST_SI3_NCAT], vip[TEMPEST_SI3_NCAT];

    double at = 0.0;
    for (int32_t c = 0; c < TEMPEST_SI3_NCAT; ++c) {
        const size_t idx_c = static_cast<size_t>(c) * n2 + col;
        a[c] = fmax(0.0, static_cast<double>(a_i[idx_c]));
        vi[c] = fmax(0.0, static_cast<double>(v_i[idx_c]));
        vs[c] = fmax(0.0, static_cast<double>(v_s[idx_c]));
        svi[c] = fmax(0.0, static_cast<double>(sv_i[idx_c]));
        tsu[c] = fmax(180.0, fmin(O_RT0, static_cast<double>(t_su[idx_c])));
        ts[c] = fmax(180.0, fmin(O_RT0, static_cast<double>(t_s[idx_c])));
        aip[c] = fmax(0.0, static_cast<double>(a_ip[idx_c]));
        vip[c] = fmax(0.0, static_cast<double>(v_ip[idx_c]));
        if (vi[c] < 1e-8 || a[c] < 1e-7) {
            a[c] = 0.0; vi[c] = 0.0; vs[c] = 0.0; svi[c] = 0.0; aip[c] = 0.0; vip[c] = 0.0;
            for (int32_t l = 0; l < TEMPEST_SI3_NLAY; ++l) ei[c][l] = 0.0;
        } else {
            for (int32_t l = 0; l < TEMPEST_SI3_NLAY; ++l) {
                const size_t il = static_cast<size_t>(c * TEMPEST_SI3_NLAY + l) * n2 + col;
                ei[c][l] = fmax(0.0, static_cast<double>(e_i[il]));
            }
        }
        at += a[c];
    }

    // Shear + convergence Thorndike closing + hard cap on sum(a_i) <= rn_amax
    const double div_v = static_cast<double>(div_ice[col]);
    const double del_v = static_cast<double>(delta_ice[col]);
    const double closing_conv = fmax(0.0, -div_v + 0.5 * (del_v - fabs(div_v)));
    const double excess_area = fmax(0.0, at - cfg.rn_amax);
    double rdg_area = (at > cfg.rn_amax) ? (excess_area + 0.5 * dt_s * closing_conv * at)
                                         : fmax(0.0, (at + closing_conv * dt_s * 0.5 - cfg.rn_amax));
    // Also ridge during active convergence even when at <= rn_amax (Lipscomb / Thorndike)
    if (at > 0.05 && div_v < -1e-7) {
        rdg_area = fmax(rdg_area, fmin(0.35 * at, -div_v * dt_s * 0.5));
    }

    if (rdg_area > 1e-9 && at > 1e-6) {
        // Ridging transfers fraction of thinner cats (c=0..3) into thicker cats (min(c+1..4))
        // conserving total ice volume, salt (sv_i), and enthalpy (e_i), shedding 50% snow to ocean
        double rem = rdg_area;
        for (int32_t c = 0; c < TEMPEST_SI3_NCAT - 1 && rem > 1e-9; ++c) {
            if (a[c] <= 1e-7) continue;
            const double take_a = fmin(0.8 * a[c], rem * 1.35);
            const double frac = take_a / a[c];
            const double new_a = 0.5 * take_a; // ridging roughly doubles local thickness (h -> 2h)
            const int32_t dst = min(TEMPEST_SI3_NCAT - 1, c + 1);

            a[c] -= take_a;
            a[dst] += new_a;

            const double dvi = frac * vi[c]; vi[c] -= dvi; vi[dst] += dvi;
            const double dvs = frac * vs[c]; vs[c] -= dvs; vs[dst] += 0.5 * dvs; // rn_fsnwrdg = 0.5
            // Shed remaining 0.5*dvs snow into ocean freshwater/heat budget
            const double dms_shed = 0.5 * dvs * SI3_RHOS / dt_s;
            if (emp_tot) emp_tot[col] -= static_cast<float>(dms_shed);
            if (qns_tot) qns_tot[col] -= static_cast<float>(dms_shed * SI3_LFI);

            const double dsv = frac * svi[c]; svi[c] -= dsv; svi[dst] += dsv;
            for (int32_t l = 0; l < TEMPEST_SI3_NLAY; ++l) {
                const double dei = frac * ei[c][l]; ei[c][l] -= dei; ei[dst][l] += dei;
            }
            const double daip = frac * aip[c]; aip[c] -= daip; aip[dst] += daip;
            const double dvip = frac * vip[c]; vip[c] -= dvip; vip[dst] += dvip;

            rem -= (take_a - new_a);
        }
        // Final strict cap on sum(a) <= rn_amax preserving volumes/salt/enthalpy exactly
        at = 0.0;
        for (int32_t c = 0; c < TEMPEST_SI3_NCAT; ++c) at += a[c];
        if (at > cfg.rn_amax) {
            const double scale = cfg.rn_amax / at;
            for (int32_t c = 0; c < TEMPEST_SI3_NCAT; ++c) {
                a[c] *= scale;
                aip[c] = fmin(aip[c], a[c]);
            }
        }
    }

    // ITD Conservative Bin Remapping: move category c -> target bin if mean h drifts outside [lo, hi]
    double a2[TEMPEST_SI3_NCAT] = {0}, vi2[TEMPEST_SI3_NCAT] = {0}, vs2[TEMPEST_SI3_NCAT] = {0};
    double svi2[TEMPEST_SI3_NCAT] = {0}, ei2[TEMPEST_SI3_NCAT][TEMPEST_SI3_NLAY] = {{0}};
    double aip2[TEMPEST_SI3_NCAT] = {0}, vip2[TEMPEST_SI3_NCAT] = {0};
    double tsu_w[TEMPEST_SI3_NCAT] = {0}, ts_w[TEMPEST_SI3_NCAT] = {0};

    for (int32_t c = 0; c < TEMPEST_SI3_NCAT; ++c) {
        if (a[c] <= 1e-7 || vi[c] <= 1e-8) continue;
        const double hc = vi[c] / a[c];
        int32_t tgt = c;
        for (int32_t b = 0; b < TEMPEST_SI3_NCAT; ++b) {
            if (hc >= c_cat_lo[b] && (hc < c_cat_hi[b] || b == TEMPEST_SI3_NCAT - 1)) {
                tgt = b;
                break;
            }
        }
        a2[tgt] += a[c];
        vi2[tgt] += vi[c];
        vs2[tgt] += vs[c];
        svi2[tgt] += svi[c];
        aip2[tgt] += fmin(aip[c], a[c]);
        vip2[tgt] += vip[c];
        tsu_w[tgt] += a[c] * tsu[c];
        ts_w[tgt] += a[c] * ts[c];
        for (int32_t l = 0; l < TEMPEST_SI3_NLAY; ++l) ei2[tgt][l] += ei[c][l];
    }

    // Re-enforce sum(a) <= rn_amax cleanly after remapping
    double at2 = 0.0;
    for (int32_t c = 0; c < TEMPEST_SI3_NCAT; ++c) at2 += a2[c];
    const double cap_s = (at2 > cfg.rn_amax && at2 > 0.0) ? (cfg.rn_amax / at2) : 1.0;

    for (int32_t c = 0; c < TEMPEST_SI3_NCAT; ++c) {
        const size_t idx_c = static_cast<size_t>(c) * n2 + col;
        a_i[idx_c] = static_cast<float>(a2[c] * cap_s);
        v_i[idx_c] = static_cast<float>(vi2[c]);
        v_s[idx_c] = static_cast<float>(vs2[c]);
        sv_i[idx_c] = static_cast<float>(svi2[c]);
        a_ip[idx_c] = static_cast<float>(fmin(aip2[c] * cap_s, a2[c] * cap_s));
        v_ip[idx_c] = static_cast<float>(vip2[c]);
        t_su[idx_c] = static_cast<float>((a2[c] > 1e-7) ? (tsu_w[c] / a2[c]) : O_RT0);
        t_s[idx_c] = static_cast<float>((a2[c] > 1e-7) ? (ts_w[c] / a2[c]) : O_RT0);
        for (int32_t l = 0; l < TEMPEST_SI3_NLAY; ++l) {
            const size_t il = static_cast<size_t>(c * TEMPEST_SI3_NLAY + l) * n2 + col;
            e_i[il] = static_cast<float>(ei2[c][l]);
        }
    }
}

// 6. SI3 BL99 4-Layer Conduction + Growth/Melt + Flooding + Frazil + Melt Ponds + Spectral Albedo + iceupdate
__global__ void si3_thermo_bl99_frazil_kernel(
    NhOceanConfig cfg, double dt_s, const double* dz,
    const float* tmask,
    const float* u10m, const float* v10m, const float* t2m, const float* q2m, const float* mslp,
    const float* sw_dn, const float* lw_dn, const float* p_liq, const float* p_ice, const float* cld_fra,
    const float* u_oce, const float* v_oce, const float* u_ice, const float* v_ice,
    const float* qsr_oce, const float* qns_oce, const float* emp_oce,
    const float* taux_atm, const float* tauy_atm,
    double* temp_k, double* sal_psu, double* ssh,
    float* a_i, float* v_i, float* v_s, float* sv_i, float* e_i,
    float* t_su, float* t_s, float* a_ip, float* v_ip,
    float* qsr_tot, float* qns_tot, float* emp_tot, float* sfx_tot,
    float* taux_oce, float* tauy_oce,
    float* ice_frac, float* ice_thick, float* ice_tsfc, float* ice_albedo
) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= cfg.ny || i >= cfg.nx) return;
    const size_t n2 = static_cast<size_t>(cfg.ny) * cfg.nx;
    const size_t col = static_cast<size_t>(j) * cfg.nx + i;

    if (tmask[col] < 0.5f) {
        qsr_tot[col] = 0.0f; qns_tot[col] = 0.0f; emp_tot[col] = 0.0f; sfx_tot[col] = 0.0f;
        taux_oce[col] = 0.0f; tauy_oce[col] = 0.0f;
        ice_frac[col] = 0.0f; ice_thick[col] = 0.0f;
        ice_tsfc[col] = static_cast<float>(O_RT0 - 1.8);
        ice_albedo[col] = static_cast<float>(SI3_ALB_OCE);
        return;
    }

    double at_b = 0.0, sv_before = 0.0;
    for (int32_t c = 0; c < TEMPEST_SI3_NCAT; ++c) {
        const size_t ic = static_cast<size_t>(c) * n2 + col;
        at_b += fmax(0.0, static_cast<double>(a_i[ic]));
        sv_before += fmax(0.0, static_cast<double>(sv_i[ic]));
    }
    at_b = fmin(1.0, at_b);

    const double u10 = u10m ? static_cast<double>(u10m[col]) : 0.0;
    const double v10 = v10m ? static_cast<double>(v10m[col]) : 0.0;
    const double wspd = fmax(0.5, sqrt(u10 * u10 + v10 * v10));
    const double ta = t2m ? fmax(190.0, fmin(330.0, static_cast<double>(t2m[col]))) : 270.0;
    const double qa = q2m ? fmax(1e-6, fmin(0.05, static_cast<double>(q2m[col]))) : 0.002;
    const double p_mslp = mslp ? static_cast<double>(mslp[col]) : cfg.p_ref_pa;
    const double sw = sw_dn ? fmax(0.0, static_cast<double>(sw_dn[col])) : 0.0;
    const double lw = lw_dn ? fmax(40.0, static_cast<double>(lw_dn[col])) : 260.0;
    const double pl = p_liq ? fmax(0.0, static_cast<double>(p_liq[col])) : 0.0;
    const double pi_rate = p_ice ? fmax(0.0, static_cast<double>(p_ice[col])) : 0.0;
    const double cld = cld_fra ? fmax(0.0, fmin(1.0, static_cast<double>(cld_fra[col]))) : 0.5;

    const double s_oce_0 = fmax(0.0, static_cast<double>(sal_psu[col]));
    const double t_oce_0_c = static_cast<double>(temp_k[col]) - O_RT0;
    const double tf_c = eval_teos10_fzp_hd(s_oce_0, 0.5 * dz[0]);
    const double t_bo_k = O_RT0 + tf_c;

    const double ssh_old = static_cast<double>(ssh[col]);
    const double h0_old = fmax(0.5 * dz[0], dz[0] + ssh_old);
    const double salt_oce_before = h0_old * cfg.rho0 * s_oce_0;

    // Ice-ocean turbulent heat exchange velocity (McPhee / NEMO icesbc)
    const double u_rel = static_cast<double>(u_ice[col]) - static_cast<double>(u_oce[col]);
    const double v_rel = static_cast<double>(v_ice[col]) - static_cast<double>(v_oce[col]);
    const double u_star_io = fmax(0.002, sqrt(cfg.rn_cd_io) * sqrt(u_rel * u_rel + v_rel * v_rel));
    const double q_oce_basal = -cfg.rho0 * O_CP0 * 0.006 * u_star_io * fmax(0.0, t_oce_0_c - tf_c); // <0 melts ice

    double qtr_ice_bot_sum = 0.0;
    double q_to_oce_from_ice = 0.0; // heat absorbed/released by ocean from ice melt/conduction
    double wfx_ice_to_oce = 0.0;    // freshwater flux >0 into ocean [kg m^-2 s^-1]
    double dh_melt_top_tot = 0.0;

    if (cfg.enable_si3) {
        for (int32_t c = 0; c < TEMPEST_SI3_NCAT; ++c) {
            const size_t ic = static_cast<size_t>(c) * n2 + col;
            double ac = fmax(0.0, static_cast<double>(a_i[ic]));
            double vic = fmax(0.0, static_cast<double>(v_i[ic]));
            if (ac <= 1e-6 || vic <= 1e-7) {
                a_i[ic] = 0.0f; v_i[ic] = 0.0f; v_s[ic] = 0.0f; sv_i[ic] = 0.0f;
                a_ip[ic] = 0.0f; v_ip[ic] = 0.0f;
                t_su[ic] = static_cast<float>(t_bo_k);
                t_s[ic] = static_cast<float>(t_bo_k);
                for (int32_t l = 0; l < TEMPEST_SI3_NLAY; ++l) {
                    e_i[static_cast<size_t>(c * TEMPEST_SI3_NLAY + l) * n2 + col] = 0.0f;
                }
                continue;
            }

            double hi = fmax(0.01, vic / ac);
            double vsc = fmax(0.0, static_cast<double>(v_s[ic]));
            double hs = vsc / ac;
            double sc = fmax(0.0, fmin(s_oce_0, static_cast<double>(sv_i[ic]) / vic));
            double tsuc = fmax(180.0, fmin(O_RT0, static_cast<double>(t_su[ic])));
            double tsc = fmax(180.0, fmin(O_RT0, static_cast<double>(t_s[ic])));
            double ap_frac = fmin(0.8, fmax(0.0, static_cast<double>(a_ip[ic]) / ac));
            double hp = fmin(0.5 * hi, fmax(0.0, static_cast<double>(v_ip[ic]) / fmax(1e-6, static_cast<double>(a_ip[ic]))));

            // NEMO spectral albedo (icealb.F90 lines 135-186)
            const double snow_frac = hs / fmax(0.02, hs + 0.02);
            const double ap_eff = fmin(ap_frac, 1.0 - snow_frac);
            const double ai_bare = fmax(0.0, 1.0 - snow_frac - ap_eff);
            double alb_ice = (hs < 1e-6 && tsuc >= O_RT0 - 1e-3) ? SI3_ALB_IMLT : SI3_ALB_IDRY;
            if (hi > 0.05 && hi <= SI3_ALB_HPIV) {
                alb_ice += (0.18 - alb_ice) * (log(SI3_ALB_HPIV) - log(hi)) / (log(SI3_ALB_HPIV) - log(0.05));
            } else if (hi <= 0.05) {
                alb_ice = SI3_ALB_OCE + (0.18 - SI3_ALB_OCE) * (hi / 0.05);
            }
            const double alb_snw = (tsuc < O_RT0 - 1e-3)
                ? (SI3_ALB_SDRY - (SI3_ALB_SDRY - alb_ice) * exp(-hs / 0.02))
                : (SI3_ALB_SMLT - (SI3_ALB_SMLT - alb_ice) * exp(-hs / 0.03));
            const double alb_pnd = SI3_ALB_DPND - (SI3_ALB_DPND - alb_ice) * exp(-hp / 0.05);
            const double alb_os = snow_frac * alb_snw + ap_eff * alb_pnd + ai_bare * alb_ice;
            const double alb_cs = alb_os - (-0.1010 * alb_os * alb_os + 0.1933 * alb_os - 0.0148);
            const double alb_c = (1.0 - cld) * alb_cs + cld * alb_os;

            // Unpack 4 BL99 layer temperatures
            double ti_c[TEMPEST_SI3_NLAY], ei_vol[TEMPEST_SI3_NLAY];
            for (int32_t l = 0; l < TEMPEST_SI3_NLAY; ++l) {
                const size_t il = static_cast<size_t>(c * TEMPEST_SI3_NLAY + l) * n2 + col;
                ei_vol[l] = fmax(0.0, static_cast<double>(e_i[il])) * TEMPEST_SI3_NLAY / vic;
                ti_c[l] = bl99_temp_c_from_enthalpy(ei_vol[l], sc);
            }

            // Surface snowfall deposition on ice
            hs += (pi_rate * dt_s) / SI3_RHOS;

            // Atmospheric fluxes over ice surface & dQ/dT_su
            const double tmc_su = (hs > 1e-4) ? 0.0 : (-SI3_RTMLT * sc);
            const double cd_i = 1.4e-3;
            const double es_i = 611.15 * exp(22.452 * (tsuc - O_RT0) / (tsuc - 0.61));
            const double qs_i = 0.622 * es_i / fmax(10000.0, p_mslp - 0.378 * es_i);
            const double shf_i = SI3_RHO_AIR * 1004.64 * cd_i * wspd * (ta - tsuc);
            const double evap_i = fmax(0.0, SI3_RHO_AIR * cd_i * wspd * (qs_i - qa)); // [kg/m2/s]
            const double lhf_i = -O_LS * evap_i;
            const double d_shf_dt = -SI3_RHO_AIR * 1004.64 * cd_i * wspd;
            const double d_qs_dt = qs_i * 22.452 * 272.54 / ((tsuc - 0.61) * (tsuc - 0.61));
            const double d_lhf_dt = -O_LS * SI3_RHO_AIR * cd_i * wspd * d_qs_dt;
            const double t3 = tsuc * tsuc * tsuc;
            const double lw_net_i = 0.98 * (lw - O_STEFAN * t3 * tsuc);
            const double d_lw_dt = -4.0 * 0.98 * O_STEFAN * t3;

            const double qsr_c = (1.0 - alb_c) * sw;
            const double qtr_top = (hs < 1e-3) ? (0.18 * qsr_c) : 0.0;
            const double qsr_surf = qsr_c - qtr_top;
            const double qtr_bot = qtr_top * exp(-1.5 * hi);
            const double q_abs_int = qtr_top - qtr_bot;
            qtr_ice_bot_sum += ac * qtr_bot;

            const double qns_i0 = lw_net_i + shf_i + lhf_i;
            const double dqns_dt = d_lw_dt + d_shf_dt + d_lhf_dt;

            // BL99 5-node coupled implicit Thomas solver (node 0 = snow/top, 1..4 = 4 ice layers)
            const double dz_i = hi / TEMPEST_SI3_NLAY;
            const double cond_s_eff = (hs > 1e-3) ? (SI3_KS / hs) : 1.0e5;
            double k_layer[TEMPEST_SI3_NLAY];
            for (int32_t l = 0; l < TEMPEST_SI3_NLAY; ++l) {
                k_layer[l] = fmax(0.15, SI3_KI0 + 0.09 * sc / fmin(-0.05, ti_c[l]));
            }
            const double kappa_0 = (hs > 1e-3)
                ? (1.0 / (0.5 / cond_s_eff + 0.5 * dz_i / k_layer[0]))
                : (k_layer[0] / (0.5 * dz_i));

            // Update surface temperature balancing F_atm(T_su) + conduction
            const double t_top_c = (hs > 1e-3) ? (tsc - O_RT0) : ti_c[0];
            const double denom_su = fmax(1.0, kappa_0 - dqns_dt);
            const double tsu_new_c = fmax(-80.0,
                t_top_c + (qsr_surf + qns_i0 - kappa_0 * (tsuc - O_RT0 - t_top_c)) / denom_su);
            const bool melting_top = (tsu_new_c > tmc_su);
            const double tsuc_new = O_RT0 + fmin(tmc_su, tsu_new_c);
            const double f_cond_top = (hs > 1e-3 ? cond_s_eff : (k_layer[0] / (0.5 * dz_i))) * (t_top_c - (tsuc_new - O_RT0));
            const double q_net_surf = qsr_surf + qns_i0 + dqns_dt * (tsuc_new - tsuc) + f_cond_top;

            // Update snow temperature if present
            if (hs > 1e-3) {
                const double c_snw = SI3_RHOS * SI3_CPI * hs / dt_s;
                const double g_s_top = cond_s_eff;
                const double g_s_bot = kappa_0;
                tsc = O_RT0 + fmin(0.0, fmax(-80.0,
                    (c_snw * (tsc - O_RT0) + g_s_top * (tsuc_new - O_RT0) + g_s_bot * ti_c[0]) /
                    (c_snw + g_s_top + g_s_bot)));
            } else {
                tsc = tsuc_new;
            }

            // Thomas solve across 4 BL99 ice layers [0..3]
            double sub[4], diag[4], sup[4], rhs_l[4], cp_t[4], dp_t[4];
            for (int32_t l = 0; l < 4; ++l) {
                const double cp_eff = SI3_CPI + 17.997e6 * sc / pow(fmin(-0.2, ti_c[l]), 2.0);
                const double cap = SI3_RHOI * cp_eff * dz_i / dt_s;
                const double k_up = (l == 0) ? kappa_0
                                             : (0.5 * (k_layer[l - 1] + k_layer[l]) / dz_i);
                const double k_dn = (l == 3) ? (k_layer[3] / (0.5 * dz_i))
                                             : (0.5 * (k_layer[l] + k_layer[l + 1]) / dz_i);
                sub[l] = (l == 0) ? 0.0 : -k_up;
                sup[l] = (l == 3) ? 0.0 : -k_dn;
                diag[l] = cap + k_up + k_dn;
                rhs_l[l] = cap * ti_c[l] + (q_abs_int * 0.25);
                if (l == 0) rhs_l[l] += k_up * ((hs > 1e-3) ? (tsc - O_RT0) : (tsuc_new - O_RT0));
                if (l == 3) rhs_l[l] += k_dn * tf_c;
            }
            cp_t[0] = sup[0] / diag[0];
            dp_t[0] = rhs_l[0] / diag[0];
            for (int32_t l = 1; l < 4; ++l) {
                const double inv = 1.0 / (diag[l] - sub[l] * cp_t[l - 1]);
                cp_t[l] = sup[l] * inv;
                dp_t[l] = (rhs_l[l] - sub[l] * dp_t[l - 1]) * inv;
            }
            ti_c[3] = dp_t[3];
            for (int32_t l = 2; l >= 0; --l) ti_c[l] = dp_t[l] - cp_t[l] * ti_c[l + 1];
            const double tmlts_c = -SI3_RTMLT * sc;
            for (int32_t l = 0; l < 4; ++l) ti_c[l] = fmax(-85.0, fmin(tmlts_c - 1e-4, ti_c[l]));

            // Surface melt when melting_top
            double melt_vol_m = 0.0;
            if (melting_top && q_net_surf > 0.0) {
                double e_rem = q_net_surf * dt_s;
                if (hs > 0.0) {
                    const double e_snw = SI3_RHOS * SI3_LFI;
                    const double dhs = fmin(hs, e_rem / e_snw);
                    hs -= dhs;
                    e_rem -= dhs * e_snw;
                    wfx_ice_to_oce += ac * dhs * SI3_RHOS / dt_s;
                    melt_vol_m += dhs * (SI3_RHOS / O_RHO0);
                }
                if (e_rem > 0.0 && hi > 0.0) {
                    const double dhi_top = fmin(hi, e_rem / (SI3_RHOI * SI3_LFI));
                    hi -= dhi_top;
                    wfx_ice_to_oce += ac * dhi_top * SI3_RHOI / dt_s;
                    melt_vol_m += dhi_top * (SI3_RHOI / O_RHO0);
                }
            }
            dh_melt_top_tot += ac * melt_vol_m;

            // Basal heat balance: conductive flux from ice bottom (+ upward into ice) - ocean basal heat
            const double f_cond_bot = (k_layer[3] / (0.5 * dz_i)) * (ti_c[3] - tf_c); // <0 -> freezes at bottom
            const double q_bot_net = -f_cond_bot + (-q_oce_basal); // >0 melts bottom, <0 freezes bottom
            const double dhi_bot = -q_bot_net * dt_s / (SI3_RHOI * SI3_LFI);
            const double hi_old_post_top = hi;
            hi = fmax(0.0, hi + dhi_bot);
            if (dhi_bot > 0.0) {
                const double s_add = 0.25 * s_oce_0;
                sc = (hi_old_post_top * sc + dhi_bot * s_add) / fmax(1e-6, hi);
                wfx_ice_to_oce -= ac * dhi_bot * SI3_RHOI / dt_s;
            } else if (dhi_bot < 0.0) {
                const double melted = fmin(hi_old_post_top, -dhi_bot);
                wfx_ice_to_oce += ac * melted * SI3_RHOI / dt_s;
                q_to_oce_from_ice += ac * q_oce_basal;
            }

            // Archimedes snow-ice flooding when snow load exceeds freeboard: hs > (rho0 - rhoi)/rhos * hi
            const double hs_crit = ((cfg.rho0 - SI3_RHOI) / SI3_RHOS) * hi;
            if (hs > hs_crit && hi > 0.02) {
                const double dhi_sni = (SI3_RHOS * hs - (cfg.rho0 - SI3_RHOI) * hi) / (SI3_RHOI + SI3_RHOS - cfg.rho0);
                const double dh_flood = fmin(0.5 * hs, fmax(0.0, dhi_sni));
                hs -= dh_flood;
                hi += dh_flood;
                // Sea-water fills pores -> snow-ice salinity ~ s_oce * (1 - rhos/rhoi)
                const double s_flood = s_oce_0 * (1.0 - SI3_RHOS / SI3_RHOI);
                sc = ((hi - dh_flood) * sc + dh_flood * s_flood) / fmax(1e-6, hi);
            }

            // Melt-pond evolution (Lecomte/Holland pnd_LEV): trap 20% of top meltwater when melting, drain when freezing
            if (melt_vol_m > 0.0) {
                hp = fmin(0.5 * hi, hp + 0.20 * melt_vol_m);
                ap_frac = fmin(0.60, fmax(ap_frac, hp / 0.35));
            } else if (tsuc_new < O_RT0 - 0.5) {
                hp *= exp(-dt_s / 86400.0);
                ap_frac *= exp(-dt_s / 86400.0);
            }

            // Pack updated state conserving total salt & enthalpy
            const double v_i_new = ac * hi;
            const double v_s_new = ac * hs;
            if (v_i_new < 1e-6) {
                wfx_ice_to_oce += v_s_new * SI3_RHOS / dt_s;
                a_i[ic] = 0.0f; v_i[ic] = 0.0f; v_s[ic] = 0.0f; sv_i[ic] = 0.0f;
                a_ip[ic] = 0.0f; v_ip[ic] = 0.0f;
                t_su[ic] = static_cast<float>(t_bo_k);
                t_s[ic] = static_cast<float>(t_bo_k);
                for (int32_t l = 0; l < TEMPEST_SI3_NLAY; ++l) {
                    e_i[static_cast<size_t>(c * TEMPEST_SI3_NLAY + l) * n2 + col] = 0.0f;
                }
            } else {
                a_i[ic] = static_cast<float>(ac);
                v_i[ic] = static_cast<float>(v_i_new);
                v_s[ic] = static_cast<float>(v_s_new);
                sv_i[ic] = static_cast<float>(v_i_new * sc);
                a_ip[ic] = static_cast<float>(ac * ap_frac);
                v_ip[ic] = static_cast<float>(ac * ap_frac * hp);
                t_su[ic] = static_cast<float>(tsuc_new);
                t_s[ic] = static_cast<float>(tsc);
                for (int32_t l = 0; l < TEMPEST_SI3_NLAY; ++l) {
                    const size_t il = static_cast<size_t>(c * TEMPEST_SI3_NLAY + l) * n2 + col;
                    const double e_vol = bl99_enthalpy_from_temp_c(ti_c[l], sc);
                    e_i[il] = static_cast<float>((v_i_new / TEMPEST_SI3_NLAY) * e_vol);
                }
            }
        }
    }

    // Apply open-water surface cooling/heating over (1 - at_b) + ice-ocean heat + ridging snow shed + rain/melt
    const double open_frac = fmax(0.0, 1.0 - at_b);
    const double qsr_net_surf = open_frac * static_cast<double>(qsr_oce[col]) + qtr_ice_bot_sum;
    const double qns_net_surf = open_frac * static_cast<double>(qns_oce[col]) + q_to_oce_from_ice
                              + static_cast<double>(qns_tot[col]);
    const double ridg_emp = static_cast<double>(emp_tot[col]);
    // Note: qsr penetrates vertically via traqsr later; apply non-solar heat to layer 0 now to test frazil!
    const double d_t0 = (qns_net_surf * dt_s) / (cfg.rho0 * cfg.cp_o * h0_old);
    double t0_new_c = t_oce_0_c + d_t0;

    // Open-water frazil ice formation whenever surface supercools below freezing point T_f(S)
    double at_curr = 0.0;
    for (int32_t c = 0; c < TEMPEST_SI3_NCAT; ++c) {
        at_curr += fmax(0.0, static_cast<double>(a_i[static_cast<size_t>(c) * n2 + col]));
    }
    if (cfg.enable_si3 && t0_new_c < tf_c) {
        const double q_deficit = cfg.rho0 * cfg.cp_o * h0_old * (tf_c - t0_new_c); // [J m^-2]
        const double s_new = 0.25 * s_oce_0;
        const double e_new_vol = bl99_enthalpy_from_temp_c(tf_c - 0.1, s_new);
        const double v_new = q_deficit / fmax(1.0e7, e_new_vol);
        const double h_new = fmax(0.05, cfg.rn_hinew);
        const double a_new_raw = v_new / h_new;
        const double a_new = fmin(a_new_raw, fmax(0.0, cfg.rn_amax - at_curr));

        // Add frazil ice to category 0 (0..0.45 m) conservatively
        const size_t i0 = col;
        const double a0 = static_cast<double>(a_i[i0]);
        const double v0 = static_cast<double>(v_i[i0]);
        const double sv0 = static_cast<double>(sv_i[i0]);
        const double v0_new = v0 + v_new;
        a_i[i0] = static_cast<float>(fmin(cfg.rn_amax, a0 + a_new));
        v_i[i0] = static_cast<float>(v0_new);
        sv_i[i0] = static_cast<float>(sv0 + v_new * s_new);
        t_su[i0] = static_cast<float>(O_RT0 + fmin(tf_c, static_cast<double>(t2m ? t2m[col] : t_bo_k) - O_RT0));
        if (a0 < 1e-6) {
            t_s[i0] = static_cast<float>(t_bo_k);
        }
        for (int32_t l = 0; l < TEMPEST_SI3_NLAY; ++l) {
            const size_t il = static_cast<size_t>(l) * n2 + col;
            e_i[il] = static_cast<float>(static_cast<double>(e_i[il]) + (v_new / TEMPEST_SI3_NLAY) * e_new_vol);
        }
        wfx_ice_to_oce -= v_new * SI3_RHOI / dt_s;
        t0_new_c = tf_c;
    }

    // Exact ocean + sea-ice salt conservation & SSH mass update
    double sv_after = 0.0;
    for (int32_t c = 0; c < TEMPEST_SI3_NCAT; ++c) {
        sv_after += fmax(0.0, static_cast<double>(sv_i[static_cast<size_t>(c) * n2 + col]));
    }
    const double net_emp_oce = open_frac * static_cast<double>(emp_oce[col]) - wfx_ice_to_oce + ridg_emp; // >0 removes ocean water
    const double d_ssh_fw = -(net_emp_oce * dt_s) / cfg.rho0;
    const double ssh_new = fmax(-8.0, fmin(8.0, ssh_old + d_ssh_fw));
    const double h0_new = fmax(0.5 * dz[0], dz[0] + ssh_new);
    const double sal_new = fmax(0.0, (salt_oce_before - SI3_RHOI * (sv_after - sv_before)) / (h0_new * cfg.rho0));

    temp_k[col] = O_RT0 + fmax(tf_c, fmin(42.0, t0_new_c));
    sal_psu[col] = sal_new;
    ssh[col] = ssh_new;

    // Total fluxes to 3D ocean (taux, tauy, qsr, qns, emp)
    qsr_tot[col] = static_cast<float>(qsr_net_surf);
    qns_tot[col] = static_cast<float>(qns_net_surf);
    emp_tot[col] = static_cast<float>(net_emp_oce);
    sfx_tot[col] = static_cast<float>(SI3_RHOI * (sv_after - sv_before) / dt_s);

    // Momentum stress weighted between wind over open water & ice-ocean quadratic drag
    const double tau_io_x = cfg.rho0 * cfg.rn_cd_io * sqrt(u_rel * u_rel + v_rel * v_rel) * u_rel;
    const double tau_io_y = cfg.rho0 * cfg.rn_cd_io * sqrt(u_rel * u_rel + v_rel * v_rel) * v_rel;
    taux_oce[col] = static_cast<float>(open_frac * static_cast<double>(taux_atm[col]) + at_b * tau_io_x);
    tauy_oce[col] = static_cast<float>(open_frac * static_cast<double>(tauy_atm[col]) + at_b * tau_io_y);

    // Aggregate 2D SI3 diagnostics for atmosphere mosaic coupling & WX-1 output
    double at_f = 0.0, vt_f = 0.0, tsu_sum = 0.0, alb_sum = 0.0;
    for (int32_t c = 0; c < TEMPEST_SI3_NCAT; ++c) {
        const size_t ic = static_cast<size_t>(c) * n2 + col;
        const double ac = fmax(0.0, static_cast<double>(a_i[ic]));
        const double vc = fmax(0.0, static_cast<double>(v_i[ic]));
        if (ac > 1e-6 && vc > 1e-7) {
            const double hi = vc / ac;
            const double hs = fmax(0.0, static_cast<double>(v_s[ic])) / ac;
            const double tsuc = static_cast<double>(t_su[ic]);
            const double ap_f = fmin(0.8, fmax(0.0, static_cast<double>(a_ip[ic]) / ac));
            const double hp_f = (a_ip[ic] > 1e-6f) ? (v_ip[ic] / a_ip[ic]) : 0.0;
            const double sf = hs / fmax(0.02, hs + 0.02);
            const double ape = fmin(ap_f, 1.0 - sf);
            const double aib = fmax(0.0, 1.0 - sf - ape);
            double ai = (hs < 1e-6 && tsuc >= O_RT0 - 1e-3) ? SI3_ALB_IMLT : SI3_ALB_IDRY;
            if (hi > 0.05 && hi <= SI3_ALB_HPIV) {
                ai += (0.18 - ai) * (log(SI3_ALB_HPIV) - log(hi)) / (log(SI3_ALB_HPIV) - log(0.05));
            } else if (hi <= 0.05) {
                ai = SI3_ALB_OCE + (0.18 - SI3_ALB_OCE) * (hi / 0.05);
            }
            const double asnw = (tsuc < O_RT0 - 1e-3)
                ? (SI3_ALB_SDRY - (SI3_ALB_SDRY - ai) * exp(-hs / 0.02))
                : (SI3_ALB_SMLT - (SI3_ALB_SMLT - ai) * exp(-hs / 0.03));
            const double apnd = SI3_ALB_DPND - (SI3_ALB_DPND - ai) * exp(-hp_f / 0.05);
            const double aos = sf * asnw + ape * apnd + aib * ai;
            const double acs = aos - (-0.1010 * aos * aos + 0.1933 * aos - 0.0148);
            const double albc = (1.0 - cld) * acs + cld * aos;

            at_f += ac;
            vt_f += vc;
            tsu_sum += ac * tsuc;
            alb_sum += ac * albc;
        }
    }
    at_f = fmin(cfg.rn_amax, at_f);
    ice_frac[col] = static_cast<float>(at_f);
    ice_thick[col] = static_cast<float>((at_f > 1e-5) ? (vt_f / at_f) : 0.0);
    ice_tsfc[col] = static_cast<float>((at_f > 1e-5) ? (tsu_sum / at_f) : t_bo_k);
    ice_albedo[col] = static_cast<float>((at_f > 1e-5) ? (alb_sum / at_f) : SI3_ALB_OCE);
}

// 7. Split-Explicit Barotropic Mode + Inverse Barometer + Coriolis + Stress (NEMO dynspg_ts.F90)
__global__ void oce_barotropic_uv_kernel(
    NhOceanConfig cfg, double dt_bt,
    const float* tmask, const float* bathy, const float* msf, const float* fcor,
    const double* ssh, const float* eta_ib, const float* taux_oce, const float* tauy_oce,
    const float* ub_rhs, const float* vb_rhs,
    const float* ub_in, const float* vb_in,
    float* ub_out, float* vb_out
) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= cfg.ny || i >= cfg.nx) return;
    const size_t col = static_cast<size_t>(j) * cfg.nx + i;

    int32_t v_ip, v_im, v_jp, v_jm;
    const int32_t ip = wrap_i(i + 1, cfg.nx, cfg.bc_x, v_ip);
    const int32_t im = wrap_i(i - 1, cfg.nx, cfg.bc_x, v_im);
    const int32_t jp = clamp_j(j + 1, cfg.ny, v_jp);
    const int32_t jm = clamp_j(j - 1, cfg.ny, v_jm);
    (void)v_im; (void)v_jm;

    const size_t col_e = static_cast<size_t>(j) * cfg.nx + ip;
    const size_t col_n = static_cast<size_t>(jp) * cfg.nx + i;

    const bool umask = (tmask[col] >= 0.5f) && (v_ip || cfg.bc_x == 1) && (tmask[col_e] >= 0.5f);
    const bool vmask = (tmask[col] >= 0.5f) && (v_jp != 0) && (tmask[col_n] >= 0.5f);

    double dx, dy;
    eval_dx_dy(cfg, msf, col, dx, dy);

    if (umask) {
        const double hu = fmax(5.0, 0.5 * (static_cast<double>(bathy[col]) + static_cast<double>(bathy[col_e])));
        const double u_old = static_cast<double>(ub_in[col]);
        const double v_avg = 0.25 * (vb_in[col] + vb_in[col_e] +
                                     vb_in[static_cast<size_t>(jm) * cfg.nx + i] +
                                     vb_in[static_cast<size_t>(jm) * cfg.nx + ip]);
        const double eta_e = ssh[col_e] - static_cast<double>(eta_ib[col_e]);
        const double eta_c = ssh[col] - static_cast<double>(eta_ib[col]);
        const double grad_x = -O_GRAVITY * (eta_e - eta_c) / dx;
        const double fc = 0.5 * (fcor[col] + fcor[col_e]);
        const double cor = fc * v_avg;
        const double wind = 0.5 * (taux_oce[col] + taux_oce[col_e]) / (cfg.rho0 * hu);
        const double drag = cfg.cd_bot * sqrt(u_old * u_old + v_avg * v_avg) / hu;
        const double rhs = grad_x + cor + wind + static_cast<double>(ub_rhs ? ub_rhs[col] : 0.0f);
        // Unconditional Courant stability scaling toward the poles keeps dt_bt stable everywhere
        const double cfl = sqrt(O_GRAVITY * hu) * dt_bt / dx;
        const double stab = 1.0 + fmax(0.0, cfl * cfl - 0.18) * 1.5;
        const double u_new = (u_old + dt_bt * rhs) / (stab + dt_bt * drag);
        ub_out[col] = static_cast<float>(fmax(-5.0, fmin(5.0, u_new)));
    } else {
        ub_out[col] = 0.0f;
    }

    if (vmask) {
        const double hv = fmax(5.0, 0.5 * (static_cast<double>(bathy[col]) + static_cast<double>(bathy[col_n])));
        const double v_old = static_cast<double>(vb_in[col]);
        const double u_avg = 0.25 * (ub_in[col] + ub_in[col_n] +
                                     ub_in[static_cast<size_t>(j) * cfg.nx + im] +
                                     ub_in[static_cast<size_t>(jp) * cfg.nx + im]);
        const double eta_n = ssh[col_n] - static_cast<double>(eta_ib[col_n]);
        const double eta_c = ssh[col] - static_cast<double>(eta_ib[col]);
        const double grad_y = -O_GRAVITY * (eta_n - eta_c) / dy;
        const double fc = 0.5 * (fcor[col] + fcor[col_n]);
        const double cor = -fc * u_avg;
        const double wind = 0.5 * (tauy_oce[col] + tauy_oce[col_n]) / (cfg.rho0 * hv);
        const double drag = cfg.cd_bot * sqrt(u_avg * u_avg + v_old * v_old) / hv;
        const double rhs = grad_y + cor + wind + static_cast<double>(vb_rhs ? vb_rhs[col] : 0.0f);
        const double cfl = sqrt(O_GRAVITY * hv) * dt_bt / dy;
        const double stab = 1.0 + fmax(0.0, cfl * cfl - 0.18) * 1.5;
        const double v_new = (v_old + dt_bt * rhs) / (stab + dt_bt * drag);
        vb_out[col] = static_cast<float>(fmax(-5.0, fmin(5.0, v_new)));
    } else {
        vb_out[col] = 0.0f;
    }
}

__global__ void oce_barotropic_ssh_kernel(
    NhOceanConfig cfg, double dt_bt,
    const float* tmask, const float* bathy, const float* msf,
    const float* ub, const float* vb, double* ssh
) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= cfg.ny || i >= cfg.nx) return;
    const size_t col = static_cast<size_t>(j) * cfg.nx + i;

    if (tmask[col] < 0.5f) {
        ssh[col] = 0.0;
        return;
    }

    int32_t v_im, v_ip, v_jm, v_jp;
    const int32_t im = wrap_i(i - 1, cfg.nx, cfg.bc_x, v_im);
    const int32_t ip = wrap_i(i + 1, cfg.nx, cfg.bc_x, v_ip);
    const int32_t jm = clamp_j(j - 1, cfg.ny, v_jm);
    const int32_t jp = clamp_j(j + 1, cfg.ny, v_jp);
    const size_t col_w = static_cast<size_t>(j) * cfg.nx + im;
    const size_t col_e = static_cast<size_t>(j) * cfg.nx + ip;
    const size_t col_s = static_cast<size_t>(jm) * cfg.nx + i;
    const size_t col_n = static_cast<size_t>(jp) * cfg.nx + i;

    double dx, dy;
    eval_dx_dy(cfg, msf, col, dx, dy);
    const double hc = static_cast<double>(bathy[col]);
    const double he = (v_ip && tmask[col_e] >= 0.5f) ? 0.5 * (hc + static_cast<double>(bathy[col_e])) : 0.0;
    const double hw = (v_im && tmask[col_w] >= 0.5f) ? 0.5 * (hc + static_cast<double>(bathy[col_w])) : 0.0;
    const double hn = (v_jp && tmask[col_n] >= 0.5f) ? 0.5 * (hc + static_cast<double>(bathy[col_n])) : 0.0;
    const double hs = (v_jm && tmask[col_s] >= 0.5f) ? 0.5 * (hc + static_cast<double>(bathy[col_s])) : 0.0;

    const double div_u = (he * static_cast<double>(ub[col]) - hw * static_cast<double>(ub[col_w])) / dx
                       + (hn * static_cast<double>(vb[col]) - hs * static_cast<double>(vb[col_s])) / dy;
    const double ssh_new = fmax(-8.0, fmin(8.0, ssh[col] - dt_bt * div_u));
    ssh[col] = ssh_new;
}

// 8. 3D Hydrostatic Pressure Gradient + Coriolis + Biharmonic/Laplacian Visc + Implicit Thomas Vertical Mixing
__global__ void oce_baroclinic_3d_kernel(
    NhOceanConfig cfg, double dt_s,
    const double* dz, const float* tmask, const int32_t* mbkt, const float* msf, const float* fcor,
    const float* rhd, const float* avm, const double* ssh, const float* ub, const float* vb,
    const float* taux_oce, const float* tauy_oce,
    const float* u_oce_in, const float* v_oce_in,
    float* u_oce_out, float* v_oce_out
) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= cfg.ny || i >= cfg.nx) return;
    const size_t n2 = static_cast<size_t>(cfg.ny) * cfg.nx;
    const size_t col = static_cast<size_t>(j) * cfg.nx + i;

    int32_t v_ip, v_im, v_jp, v_jm;
    const int32_t ip = wrap_i(i + 1, cfg.nx, cfg.bc_x, v_ip);
    const int32_t jp = clamp_j(j + 1, cfg.ny, v_jp);
    const int32_t im = wrap_i(i - 1, cfg.nx, cfg.bc_x, v_im);
    const int32_t jm = clamp_j(j - 1, cfg.ny, v_jm);
    (void)v_im; (void)v_jm;
    const size_t col_e = static_cast<size_t>(j) * cfg.nx + ip;
    const size_t col_n = static_cast<size_t>(jp) * cfg.nx + i;
    const size_t col_w = static_cast<size_t>(j) * cfg.nx + im;
    const size_t col_s = static_cast<size_t>(jm) * cfg.nx + i;

    const int32_t ku_bot = (tmask[col] >= 0.5f && (v_ip || cfg.bc_x == 1) && tmask[col_e] >= 0.5f)
                         ? min(mbkt[col], mbkt[col_e]) : 0;
    const int32_t kv_bot = (tmask[col] >= 0.5f && v_jp && tmask[col_n] >= 0.5f)
                         ? min(mbkt[col], mbkt[col_n]) : 0;

    double dx, dy;
    eval_dx_dy(cfg, msf, col, dx, dy);
    const double dx_safe = fmax(dx, 0.25 * cfg.dx_m);
    const double dy_safe = fmax(dy, 0.25 * cfg.dx_m);

    auto solve_col_uv = [&](bool is_u, int32_t kbot_f, const float* vel_in, float* vel_out, const float* other,
                            const float* tau_s, size_t col_nb, double baro_target, double d_spc) {
        if (kbot_f <= 0) {
            for (int32_t k = 0; k < cfg.nz; ++k) vel_out[static_cast<size_t>(k) * n2 + col] = 0.0f;
            return;
        }
        double cp[TEMPEST_OCEAN_MAX_NZ], dp[TEMPEST_OCEAN_MAX_NZ];
        double h_cum = 0.0;
        double ph_c = 0.0, ph_nb = 0.0;
        const double fc = 0.5 * (fcor[col] + fcor[col_nb]);
        const double ah = fmin(cfg.ah_lap, 0.05 * d_spc * d_spc / fmax(dt_s, 1.0));

        for (int32_t k = 0; k < kbot_f; ++k) {
            const size_t idx = static_cast<size_t>(k) * n2 + col;
            const size_t idx_nb = static_cast<size_t>(k) * n2 + col_nb;
            ph_c += O_GRAVITY * static_cast<double>(rhd[idx]) * 0.5 * dz[k];
            ph_nb += O_GRAVITY * static_cast<double>(rhd[idx_nb]) * 0.5 * dz[k];

            const double grad_h = -(ph_nb - ph_c) / d_spc
                                - O_GRAVITY * (static_cast<double>(ssh[col_nb]) - static_cast<double>(ssh[col])) / d_spc;
            ph_c += O_GRAVITY * static_cast<double>(rhd[idx]) * 0.5 * dz[k];
            ph_nb += O_GRAVITY * static_cast<double>(rhd[idx_nb]) * 0.5 * dz[k];

            const double v_o = 0.25 * (
                other[idx] + other[idx_nb] +
                (is_u ? (other[static_cast<size_t>(k) * n2 + col_s] + other[static_cast<size_t>(k) * n2 + static_cast<size_t>(jm) * cfg.nx + ip])
                      : (other[static_cast<size_t>(k) * n2 + col_w] + other[static_cast<size_t>(k) * n2 + static_cast<size_t>(jp) * cfg.nx + im]))
            );
            const double cor = is_u ? (fc * v_o) : (-fc * v_o);

            const double u_c = static_cast<double>(vel_in[idx]);
            const double lap = ah * (
                ((vel_in[static_cast<size_t>(k) * n2 + col_e] - 2.0 * u_c + vel_in[static_cast<size_t>(k) * n2 + col_w]) / (dx_safe * dx_safe)) +
                ((vel_in[static_cast<size_t>(k) * n2 + col_n] - 2.0 * u_c + vel_in[static_cast<size_t>(k) * n2 + col_s]) / (dy_safe * dy_safe))
            );
            const double wind = (k == 0) ? (0.5 * (tau_s[col] + tau_s[col_nb]) / (cfg.rho0 * dz[0])) : 0.0;
            const double bdrag = (k == kbot_f - 1) ? (cfg.cd_bot * sqrt(u_c * u_c + v_o * v_o) / dz[k]) : 0.0;

            double a = 0.0, c = 0.0;
            if (k > 0) {
                const double av_up = 0.5 * (avm[idx] + avm[idx_nb]);
                a = -dt_s * av_up / (dz[k] * 0.5 * (dz[k - 1] + dz[k]));
            }
            if (k < kbot_f - 1) {
                const size_t idx_dn = static_cast<size_t>(k + 1) * n2 + col;
                const size_t idx_dn_nb = static_cast<size_t>(k + 1) * n2 + col_nb;
                const double av_dn = 0.5 * (avm[idx_dn] + avm[idx_dn_nb]);
                c = -dt_s * av_dn / (dz[k] * 0.5 * (dz[k] + dz[k + 1]));
            }
            const double b = 1.0 - a - c + dt_s * bdrag;
            const double rhs = u_c + dt_s * (grad_h + cor + lap + wind);
            if (k == 0) {
                cp[0] = c / b;
                dp[0] = rhs / b;
            } else {
                const double inv = 1.0 / (b - a * cp[k - 1]);
                cp[k] = c * inv;
                dp[k] = (rhs - a * dp[k - 1]) * inv;
            }
            h_cum += dz[k];
        }
        double u_col[TEMPEST_OCEAN_MAX_NZ];
        u_col[kbot_f - 1] = dp[kbot_f - 1];
        for (int32_t k = kbot_f - 2; k >= 0; --k) {
            u_col[k] = dp[k] - cp[k] * u_col[k + 1];
        }
        // Barotropic mode matching: align depth-mean (u_oce, v_oce) to sub-cycled (ub, vb)
        double mean_u = 0.0;
        for (int32_t k = 0; k < kbot_f; ++k) mean_u += u_col[k] * dz[k];
        mean_u /= fmax(1.0, h_cum);
        const double corr = baro_target - mean_u;
        for (int32_t k = 0; k < cfg.nz; ++k) {
            const size_t idx = static_cast<size_t>(k) * n2 + col;
            vel_out[idx] = (k < kbot_f) ? static_cast<float>(fmax(-5.0, fmin(5.0, u_col[k] + corr))) : 0.0f;
        }
    };

    solve_col_uv(true, ku_bot, u_oce_in, u_oce_out, v_oce_in, taux_oce, col_e, static_cast<double>(ub[col]), dx_safe);
    solve_col_uv(false, kv_bot, v_oce_in, v_oce_out, u_oce_in, tauy_oce, col_n, static_cast<double>(vb[col]), dy_safe);
}

// 9. 3D Non-divergent Continuity w_oce & 3D Tracer Update (Upwind + Type-I 2-band traqsr + Implicit Thomas avt)
__global__ void oce_continuity_w_kernel(
    NhOceanConfig cfg, const double* dz, const float* tmask, const int32_t* mbkt,
    const float* msf, const float* u_oce, const float* v_oce, float* w_oce
) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= cfg.ny || i >= cfg.nx) return;
    const size_t n2 = static_cast<size_t>(cfg.ny) * cfg.nx;
    const size_t col = static_cast<size_t>(j) * cfg.nx + i;

    const int32_t kbot = mbkt[col];
    if (tmask[col] < 0.5f || kbot <= 0) {
        for (int32_t k = 0; k < cfg.nz; ++k) w_oce[static_cast<size_t>(k) * n2 + col] = 0.0f;
        return;
    }

    int32_t v_im, v_jm;
    const int32_t im = wrap_i(i - 1, cfg.nx, cfg.bc_x, v_im);
    const int32_t jm = clamp_j(j - 1, cfg.ny, v_jm);
    const size_t col_w = static_cast<size_t>(j) * cfg.nx + im;
    const size_t col_s = static_cast<size_t>(jm) * cfg.nx + i;

    double dx, dy;
    eval_dx_dy(cfg, msf, col, dx, dy);

    double w_up = 0.0;
    for (int32_t k = cfg.nz - 1; k >= 0; --k) {
        const size_t idx = static_cast<size_t>(k) * n2 + col;
        if (k >= kbot) {
            w_oce[idx] = 0.0f;
            continue;
        }
        const double ue = static_cast<double>(u_oce[idx]);
        const double uw = v_im ? static_cast<double>(u_oce[static_cast<size_t>(k) * n2 + col_w]) : 0.0;
        const double vn = static_cast<double>(v_oce[idx]);
        const double vs = v_jm ? static_cast<double>(v_oce[static_cast<size_t>(k) * n2 + col_s]) : 0.0;
        const double div_h = (ue - uw) / dx + (vn - vs) / dy;
        w_up = w_up - div_h * dz[k];
        // w_oce[k] stores interface velocity at TOP of cell k (w_oce[0] = surface w)
        w_oce[idx] = static_cast<float>(fmax(-0.05, fmin(0.05, w_up)));
    }
    w_oce[col] = 0.0f; // Rigid/kinematic surface handled separately
}

__global__ void oce_tracer_adv_diff_step_kernel(
    NhOceanConfig cfg, double dt_s, int32_t is_temp,
    const double* dz, const double* z_w, const float* tmask, const int32_t* mbkt,
    const float* msf, const float* u_oce, const float* v_oce, const float* w_oce,
    const float* avt, const float* qsr_tot, const double* tracer_in, double* tracer_out
) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= cfg.ny || i >= cfg.nx) return;
    const size_t n2 = static_cast<size_t>(cfg.ny) * cfg.nx;
    const size_t col = static_cast<size_t>(j) * cfg.nx + i;
    const int32_t kbot = mbkt[col];

    if (tmask[col] < 0.5f || kbot <= 0) {
        for (int32_t k = 0; k < cfg.nz; ++k) tracer_out[static_cast<size_t>(k) * n2 + col] = 0.0;
        return;
    }

    int32_t v_im, v_ip, v_jm, v_jp;
    const int32_t im = wrap_i(i - 1, cfg.nx, cfg.bc_x, v_im);
    const int32_t ip = wrap_i(i + 1, cfg.nx, cfg.bc_x, v_ip);
    const int32_t jm = clamp_j(j - 1, cfg.ny, v_jm);
    const int32_t jp = clamp_j(j + 1, cfg.ny, v_jp);
    const size_t col_w = static_cast<size_t>(j) * cfg.nx + im;
    const size_t col_e = static_cast<size_t>(j) * cfg.nx + ip;
    const size_t col_s = static_cast<size_t>(jm) * cfg.nx + i;
    const size_t col_n = static_cast<size_t>(jp) * cfg.nx + i;

    double dx, dy;
    eval_dx_dy(cfg, msf, col, dx, dy);
    const double dx_safe = fmax(dx, 0.25 * cfg.dx_m);
    const double dy_safe = fmax(dy, 0.25 * cfg.dx_m);
    const double ah = fmin(0.2 * cfg.ah_lap, 0.04 * fmin(dx_safe * dx_safe, dy_safe * dy_safe) / fmax(dt_s, 1.0));
    const double u_cap = 0.20 * dx / fmax(dt_s, 1.0);
    const double v_cap = 0.20 * dy / fmax(dt_s, 1.0);

    // 2-Band Type-I Solar Penetration (NEMO traqsr.F90: rn_abs=0.58, z1=0.35 m, z2=23.0 m)
    const double qsr0 = is_temp ? fmax(0.0, static_cast<double>(qsr_tot[col])) : 0.0;
    double i_top = qsr0;

    double cp[TEMPEST_OCEAN_MAX_NZ], dp[TEMPEST_OCEAN_MAX_NZ];
    for (int32_t k = 0; k < kbot; ++k) {
        const size_t idx = static_cast<size_t>(k) * n2 + col;
        const double tc = static_cast<double>(tracer_in[idx]);

        const bool wet_e = (v_ip && k < mbkt[col_e]);
        const bool wet_w = (v_im && k < mbkt[col_w]);
        const bool wet_n = (v_jp && k < mbkt[col_n]);
        const bool wet_s = (v_jm && k < mbkt[col_s]);

        const double te = wet_e ? static_cast<double>(tracer_in[static_cast<size_t>(k) * n2 + col_e]) : tc;
        const double tw = wet_w ? static_cast<double>(tracer_in[static_cast<size_t>(k) * n2 + col_w]) : tc;
        const double tn = wet_n ? static_cast<double>(tracer_in[static_cast<size_t>(k) * n2 + col_n]) : tc;
        const double ts = wet_s ? static_cast<double>(tracer_in[static_cast<size_t>(k) * n2 + col_s]) : tc;

        double adv_tend = 0.0;
        if (cfg.enable_dynamics) {
            const double ue = wet_e ? fmax(-u_cap, fmin(u_cap, static_cast<double>(u_oce[idx]))) : 0.0;
            const double uw = wet_w ? fmax(-u_cap, fmin(u_cap, static_cast<double>(u_oce[static_cast<size_t>(k) * n2 + col_w]))) : 0.0;
            const double vn = wet_n ? fmax(-v_cap, fmin(v_cap, static_cast<double>(v_oce[idx]))) : 0.0;
            const double vs = wet_s ? fmax(-v_cap, fmin(v_cap, static_cast<double>(v_oce[static_cast<size_t>(k) * n2 + col_s]))) : 0.0;

            const double fe = (ue >= 0.0) ? ue * tc : ue * te;
            const double fw = (uw >= 0.0) ? uw * tw : uw * tc;
            const double fn = (vn >= 0.0) ? vn * tc : vn * tn;
            const double fs = (vs >= 0.0) ? vs * ts : vs * tc;

            const double w_top = (k > 0) ? static_cast<double>(w_oce[idx]) : 0.0;
            const double w_bot = (k < kbot - 1) ? static_cast<double>(w_oce[static_cast<size_t>(k + 1) * n2 + col]) : 0.0;
            const double f_top = (k > 0) ? ((w_top >= 0.0) ? w_top * tc : w_top * static_cast<double>(tracer_in[static_cast<size_t>(k - 1) * n2 + col])) : 0.0;
            const double f_bot = (k < kbot - 1) ? ((w_bot >= 0.0) ? w_bot * static_cast<double>(tracer_in[static_cast<size_t>(k + 1) * n2 + col]) : w_bot * tc) : 0.0;

            const double div_f = ((fe - fw) / dx + (fn - fs) / dy) + (f_top - f_bot) / dz[k];
            const double conv_comp = tc * (((ue - uw) / dx + (vn - vs) / dy) + (w_top - w_bot) / dz[k]);
            adv_tend = -(div_f - conv_comp);
        }

        const double lap = cfg.enable_dynamics
            ? ah * (((te - 2.0 * tc + tw) / (dx_safe * dx_safe)) + ((tn - 2.0 * tc + ts) / (dy_safe * dy_safe)))
            : 0.0;

        double qsr_tend = 0.0;
        if (is_temp && qsr0 > 0.0) {
            const double zb = z_w[k + 1];
            const double i_bot = (k == kbot - 1) ? 0.0 : qsr0 * (0.58 * exp(-zb / 0.35) + 0.42 * exp(-zb / 23.0));
            qsr_tend = (i_top - i_bot) / (cfg.rho0 * cfg.cp_o * dz[k]);
            i_top = i_bot;
        }

        double a = 0.0, c = 0.0;
        if (k > 0) a = -dt_s * static_cast<double>(avt[idx]) / (dz[k] * 0.5 * (dz[k - 1] + dz[k]));
        if (k < kbot - 1) c = -dt_s * static_cast<double>(avt[static_cast<size_t>(k + 1) * n2 + col]) / (dz[k] * 0.5 * (dz[k] + dz[k + 1]));
        const double b = 1.0 - a - c;
        const double rhs = tc + dt_s * (adv_tend + lap + qsr_tend);

        if (k == 0) {
            cp[0] = c / b;
            dp[0] = rhs / b;
        } else {
            const double inv = 1.0 / (b - a * cp[k - 1]);
            cp[k] = c * inv;
            dp[k] = (rhs - a * dp[k - 1]) * inv;
        }
    }

    double col_out[TEMPEST_OCEAN_MAX_NZ];
    col_out[kbot - 1] = dp[kbot - 1];
    for (int32_t k = kbot - 2; k >= 0; --k) col_out[k] = dp[k] - cp[k] * col_out[k + 1];
    for (int32_t k = 0; k < cfg.nz; ++k) {
        const size_t idx = static_cast<size_t>(k) * n2 + col;
        if (k < kbot) {
            if (is_temp) {
                tracer_out[idx] = fmax(O_RT0 - 2.6, fmin(318.0, col_out[k]));
            } else {
                tracer_out[idx] = fmax(0.0, fmin(48.0, col_out[k]));
            }
        } else {
            tracer_out[idx] = 0.0;
        }
    }
}

void nh_oce_free(NhOceanSi3Handle* h) {
    if (!h) return;
    auto fd = [](double*& p) { if (p) { cudaFree(p); p = nullptr; } };
    auto ff = [](float*& p) { if (p) { cudaFree(p); p = nullptr; } };
    auto fi = [](int32_t*& p) { if (p) { cudaFree(p); p = nullptr; } };

    fd(h->d_dz); fd(h->d_zt); fd(h->d_zw);
    ff(h->d_msf); ff(h->d_fcor); ff(h->d_bathy); ff(h->d_tmask); fi(h->d_mbkt);
    fd(h->d_temp); fd(h->d_sal); fd(h->d_tmp_tr);
    ff(h->d_u); ff(h->d_v); ff(h->d_w);
    ff(h->d_tke); ff(h->d_avm); ff(h->d_avt); ff(h->d_rhd); ff(h->d_tmp1); ff(h->d_tmp2);
    fd(h->d_ssh); ff(h->d_ub); ff(h->d_vb); ff(h->d_ub_rhs); ff(h->d_vb_rhs); ff(h->d_eta_ib); ff(h->d_mld);
    ff(h->d_taux_atm); ff(h->d_tauy_atm); ff(h->d_taux_oce); ff(h->d_tauy_oce);
    ff(h->d_qsr_oce); ff(h->d_qns_oce); ff(h->d_emp_oce);
    ff(h->d_qsr_tot); ff(h->d_qns_tot); ff(h->d_emp_tot); ff(h->d_sfx_tot);
    ff(h->d_f_u10); ff(h->d_f_v10); ff(h->d_f_t2m); ff(h->d_f_q2m); ff(h->d_f_mslp);
    ff(h->d_f_sw); ff(h->d_f_lw); ff(h->d_f_pliq); ff(h->d_f_pice); ff(h->d_f_cld);
    ff(h->d_a_i); ff(h->d_v_i); ff(h->d_v_s); ff(h->d_sv_i);
    ff(h->d_e_i); ff(h->d_t_su); ff(h->d_t_s); ff(h->d_a_ip); ff(h->d_v_ip);
    ff(h->d_u_ice); ff(h->d_v_ice); ff(h->d_s1_ice); ff(h->d_s2_ice); ff(h->d_s12_ice);
    ff(h->d_strength); ff(h->d_delta_ice); ff(h->d_div_ice);
    ff(h->d_ice_frac); ff(h->d_ice_thick); ff(h->d_ice_tsfc); ff(h->d_ice_albedo);
    delete h;
}

} // namespace

extern "C" {

uint64_t nh_ocean_si3_estimate_vram_bytes(int32_t nz, int32_t ny, int32_t nx, int32_t enable_si3) {
    const uint64_t n2 = static_cast<uint64_t>(ny) * static_cast<uint64_t>(nx);
    const uint64_t n3 = static_cast<uint64_t>(nz) * n2;
    const uint64_t bytes_3d = 9ULL * n3 * sizeof(float) + 3ULL * n3 * sizeof(double);
    const uint64_t bytes_2d = 27ULL * n2 * sizeof(float) + n2 * sizeof(double);
    const uint64_t bytes_si3 = enable_si3
        ? ((8ULL * TEMPEST_SI3_NCAT + TEMPEST_SI3_NCAT * TEMPEST_SI3_NLAY + 12ULL) * n2 * sizeof(float))
        : (4ULL * n2 * sizeof(float));
    return bytes_3d + bytes_2d + bytes_si3 + (16ULL * 1024ULL * 1024ULL);
}

NhOceanSi3Handle* nh_ocean_si3_create(const NhOceanConfig* cfg_in, const NhOceanStaticPtrs* statics) {
    if (!cfg_in || !statics || !statics->dz_m || !statics->msf || !statics->f_cor || !statics->bathy_m) {
        oce_set_err("Null argument to nh_ocean_si3_create");
        return nullptr;
    }
    if (cfg_in->nz <= 0 || cfg_in->nz > TEMPEST_OCEAN_MAX_NZ || cfg_in->ny <= 1 || cfg_in->nx <= 1) {
        oce_set_err("Invalid ocean grid dimensions in nh_ocean_si3_create");
        return nullptr;
    }

    NhOceanSi3Handle* h = new NhOceanSi3Handle();
    h->cfg = *cfg_in;
    if (h->cfg.rho0 <= 0.0) h->cfg.rho0 = O_RHO0;
    if (h->cfg.cp_o <= 0.0) h->cfg.cp_o = O_CP0;
    if (h->cfg.p_ref_pa <= 0.0) h->cfg.p_ref_pa = 101325.0;
    if (h->cfg.n_bt_substeps <= 0) h->cfg.n_bt_substeps = 12;
    if (h->cfg.n_evp_substeps <= 0) h->cfg.n_evp_substeps = 60;
    if (h->cfg.rn_pstar <= 0.0) h->cfg.rn_pstar = 2.0e4;
    if (h->cfg.rn_crhg <= 0.0) h->cfg.rn_crhg = 20.0;
    if (h->cfg.rn_ecc <= 0.0) h->cfg.rn_ecc = 2.0;
    if (h->cfg.rn_cd_io <= 0.0) h->cfg.rn_cd_io = 5.0e-3;
    if (h->cfg.rn_hinew <= 0.0) h->cfg.rn_hinew = 0.10;
    if (h->cfg.rn_amax <= 0.0 || h->cfg.rn_amax > 1.0) h->cfg.rn_amax = 0.997;

    h->n2 = static_cast<size_t>(h->cfg.ny) * static_cast<size_t>(h->cfg.nx);
    h->n3 = static_cast<size_t>(h->cfg.nz) * h->n2;

    double zt[TEMPEST_OCEAN_MAX_NZ], zw[TEMPEST_OCEAN_MAX_NZ + 1];
    zw[0] = 0.0;
    for (int32_t k = 0; k < h->cfg.nz; ++k) {
        zt[k] = zw[k] + 0.5 * statics->dz_m[k];
        zw[k + 1] = zw[k] + statics->dz_m[k];
    }

    auto alloc_f = [&](float** p, size_t cnt) -> bool {
        cudaError_t e = cudaMalloc(p, cnt * sizeof(float));
        if (e != cudaSuccess) { oce_set_cuda_err(__FILE__, __LINE__, e); return false; }
        e = cudaMemset(*p, 0, cnt * sizeof(float));
        if (e != cudaSuccess) { oce_set_cuda_err(__FILE__, __LINE__, e); return false; }
        return true;
    };
    auto alloc_d = [&](double** p, size_t cnt) -> bool {
        cudaError_t e = cudaMalloc(p, cnt * sizeof(double));
        if (e != cudaSuccess) { oce_set_cuda_err(__FILE__, __LINE__, e); return false; }
        e = cudaMemset(*p, 0, cnt * sizeof(double));
        if (e != cudaSuccess) { oce_set_cuda_err(__FILE__, __LINE__, e); return false; }
        return true;
    };

    if (cudaMalloc(&h->d_dz, h->cfg.nz * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&h->d_zt, h->cfg.nz * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&h->d_zw, (h->cfg.nz + 1) * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&h->d_mbkt, h->n2 * sizeof(int32_t)) != cudaSuccess) {
        oce_set_err("Failed allocating vertical/mask ocean buffers");
        nh_oce_free(h);
        return nullptr;
    }
    cudaMemcpy(h->d_dz, statics->dz_m, h->cfg.nz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(h->d_zt, zt, h->cfg.nz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(h->d_zw, zw, (h->cfg.nz + 1) * sizeof(double), cudaMemcpyHostToDevice);

    bool ok = alloc_f(&h->d_msf, h->n2) && alloc_f(&h->d_fcor, h->n2) &&
              alloc_f(&h->d_bathy, h->n2) && alloc_f(&h->d_tmask, h->n2) &&
              alloc_d(&h->d_temp, h->n3) && alloc_d(&h->d_sal, h->n3) && alloc_d(&h->d_tmp_tr, h->n3) &&
              alloc_f(&h->d_u, h->n3) && alloc_f(&h->d_v, h->n3) && alloc_f(&h->d_w, h->n3) &&
              alloc_f(&h->d_tke, h->n3) && alloc_f(&h->d_avm, h->n3) && alloc_f(&h->d_avt, h->n3) &&
              alloc_f(&h->d_rhd, h->n3) && alloc_f(&h->d_tmp1, h->n3) && alloc_f(&h->d_tmp2, h->n3) &&
              alloc_d(&h->d_ssh, h->n2) && alloc_f(&h->d_ub, h->n2) && alloc_f(&h->d_vb, h->n2) &&
              alloc_f(&h->d_ub_rhs, h->n2) && alloc_f(&h->d_vb_rhs, h->n2) &&
              alloc_f(&h->d_eta_ib, h->n2) && alloc_f(&h->d_mld, h->n2) &&
              alloc_f(&h->d_taux_atm, h->n2) && alloc_f(&h->d_tauy_atm, h->n2) &&
              alloc_f(&h->d_taux_oce, h->n2) && alloc_f(&h->d_tauy_oce, h->n2) &&
              alloc_f(&h->d_qsr_oce, h->n2) && alloc_f(&h->d_qns_oce, h->n2) && alloc_f(&h->d_emp_oce, h->n2) &&
              alloc_f(&h->d_qsr_tot, h->n2) && alloc_f(&h->d_qns_tot, h->n2) &&
              alloc_f(&h->d_emp_tot, h->n2) && alloc_f(&h->d_sfx_tot, h->n2) &&
              alloc_f(&h->d_f_u10, h->n2) && alloc_f(&h->d_f_v10, h->n2) &&
              alloc_f(&h->d_f_t2m, h->n2) && alloc_f(&h->d_f_q2m, h->n2) && alloc_f(&h->d_f_mslp, h->n2) &&
              alloc_f(&h->d_f_sw, h->n2) && alloc_f(&h->d_f_lw, h->n2) &&
              alloc_f(&h->d_f_pliq, h->n2) && alloc_f(&h->d_f_pice, h->n2) && alloc_f(&h->d_f_cld, h->n2) &&
              alloc_f(&h->d_ice_frac, h->n2) && alloc_f(&h->d_ice_thick, h->n2) &&
              alloc_f(&h->d_ice_tsfc, h->n2) && alloc_f(&h->d_ice_albedo, h->n2);

    if (ok && h->cfg.enable_si3) {
        const size_t n_cat = TEMPEST_SI3_NCAT * h->n2;
        const size_t n_e = TEMPEST_SI3_NCAT * TEMPEST_SI3_NLAY * h->n2;
        ok = alloc_f(&h->d_a_i, n_cat) && alloc_f(&h->d_v_i, n_cat) &&
             alloc_f(&h->d_v_s, n_cat) && alloc_f(&h->d_sv_i, n_cat) &&
             alloc_f(&h->d_e_i, n_e) && alloc_f(&h->d_t_su, n_cat) && alloc_f(&h->d_t_s, n_cat) &&
             alloc_f(&h->d_a_ip, n_cat) && alloc_f(&h->d_v_ip, n_cat) &&
             alloc_f(&h->d_u_ice, h->n2) && alloc_f(&h->d_v_ice, h->n2) &&
             alloc_f(&h->d_s1_ice, h->n2) && alloc_f(&h->d_s2_ice, h->n2) && alloc_f(&h->d_s12_ice, h->n2) &&
             alloc_f(&h->d_strength, h->n2) && alloc_f(&h->d_delta_ice, h->n2) && alloc_f(&h->d_div_ice, h->n2);
    }

    if (!ok) {
        nh_oce_free(h);
        return nullptr;
    }

    cudaMemcpy(h->d_msf, statics->msf, h->n2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(h->d_fcor, statics->f_cor, h->n2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(h->d_bathy, statics->bathy_m, h->n2 * sizeof(float), cudaMemcpyHostToDevice);
    if (statics->ocean_mask) {
        cudaMemcpy(h->d_tmask, statics->ocean_mask, h->n2 * sizeof(float), cudaMemcpyHostToDevice);
    }

    const uint32_t blk = 256;
    const uint32_t grd = static_cast<uint32_t>((h->n2 + blk - 1) / blk);
    oce_init_masks_kernel<<<grd, blk>>>(
        h->cfg.nz, h->n2, h->d_zw, h->d_bathy,
        statics->ocean_mask ? h->d_tmask : nullptr,
        h->d_tmask, h->d_mbkt,
        h->d_ice_frac, h->d_ice_thick, h->d_ice_tsfc, h->d_ice_albedo
    );
    if (cudaDeviceSynchronize() != cudaSuccess) {
        oce_set_cuda_err(__FILE__, __LINE__, cudaGetLastError());
        nh_oce_free(h);
        return nullptr;
    }
    h->min_dx_mid = fmax(250.0, 0.35 * fmin(h->cfg.dx_m, h->cfg.dy_m));
    return h;
}

void nh_ocean_si3_destroy(NhOceanSi3Handle* h) {
    nh_oce_free(h);
}

int32_t nh_ocean_si3_upload_state(NhOceanSi3Handle* h, const NhOceanConstStatePtrs* src) {
    if (!h || !src) { oce_set_err("Null ptr in nh_ocean_si3_upload_state"); return -1; }
    auto cp = [&](float* dst, const float* s, size_t cnt) -> int32_t {
        if (!dst || !s) return 0;
        OCE_CHECK(cudaMemcpy(dst, s, cnt * sizeof(float), cudaMemcpyHostToDevice));
        return 0;
    };
    auto cp_d = [&](double* dst, const double* s, size_t cnt) -> int32_t {
        if (!dst || !s) return 0;
        OCE_CHECK(cudaMemcpy(dst, s, cnt * sizeof(double), cudaMemcpyHostToDevice));
        return 0;
    };
    const size_t n_cat = TEMPEST_SI3_NCAT * h->n2;
    const size_t n_e = TEMPEST_SI3_NCAT * TEMPEST_SI3_NLAY * h->n2;
    if (cp_d(h->d_temp, src->temp_k, h->n3) || cp_d(h->d_sal, src->sal_psu, h->n3) ||
        cp(h->d_u, src->u_oce, h->n3) || cp(h->d_v, src->v_oce, h->n3) || cp(h->d_tke, src->tke, h->n3) ||
        cp(h->d_avm, src->avm, h->n3) || cp(h->d_avt, src->avt, h->n3) ||
        cp_d(h->d_ssh, src->ssh, h->n2) || cp(h->d_ub, src->ub, h->n2) ||
        cp(h->d_vb, src->vb, h->n2) || cp(h->d_mld, src->mld, h->n2) ||
        cp(h->d_a_i, src->a_i, n_cat) || cp(h->d_v_i, src->v_i, n_cat) ||
        cp(h->d_v_s, src->v_s, n_cat) || cp(h->d_sv_i, src->sv_i, n_cat) ||
        cp(h->d_e_i, src->e_i, n_e) || cp(h->d_t_su, src->t_su, n_cat) || cp(h->d_t_s, src->t_s, n_cat) ||
        cp(h->d_a_ip, src->a_ip, n_cat) || cp(h->d_v_ip, src->v_ip, n_cat) ||
        cp(h->d_u_ice, src->u_ice, h->n2) || cp(h->d_v_ice, src->v_ice, h->n2) ||
        cp(h->d_s1_ice, src->stress1, h->n2) || cp(h->d_s2_ice, src->stress2, h->n2) || cp(h->d_s12_ice, src->stress12, h->n2) ||
        cp(h->d_ice_frac, src->ice_frac, h->n2) || cp(h->d_ice_thick, src->ice_thick, h->n2) ||
        cp(h->d_ice_tsfc, src->ice_tsfc, h->n2) || cp(h->d_ice_albedo, src->ice_albedo, h->n2)) {
        return -1;
    }
    return 0;
}

int32_t nh_ocean_si3_download_state(NhOceanSi3Handle* h, const NhOceanStatePtrs* dst) {
    if (!h || !dst) { oce_set_err("Null ptr in nh_ocean_si3_download_state"); return -1; }
    auto cp = [&](float* d, const float* s, size_t cnt) -> int32_t {
        if (!d || !s) return 0;
        OCE_CHECK(cudaMemcpy(d, s, cnt * sizeof(float), cudaMemcpyDeviceToHost));
        return 0;
    };
    auto cp_d = [&](double* d, const double* s, size_t cnt) -> int32_t {
        if (!d || !s) return 0;
        OCE_CHECK(cudaMemcpy(d, s, cnt * sizeof(double), cudaMemcpyDeviceToHost));
        return 0;
    };
    const size_t n_cat = TEMPEST_SI3_NCAT * h->n2;
    const size_t n_e = TEMPEST_SI3_NCAT * TEMPEST_SI3_NLAY * h->n2;
    if (cp_d(dst->temp_k, h->d_temp, h->n3) || cp_d(dst->sal_psu, h->d_sal, h->n3) ||
        cp(dst->u_oce, h->d_u, h->n3) || cp(dst->v_oce, h->d_v, h->n3) || cp(dst->tke, h->d_tke, h->n3) ||
        cp(dst->avm, h->d_avm, h->n3) || cp(dst->avt, h->d_avt, h->n3) ||
        cp_d(dst->ssh, h->d_ssh, h->n2) || cp(dst->ub, h->d_ub, h->n2) ||
        cp(dst->vb, h->d_vb, h->n2) || cp(dst->mld, h->d_mld, h->n2) ||
        cp(dst->a_i, h->d_a_i, n_cat) || cp(dst->v_i, h->d_v_i, n_cat) ||
        cp(dst->v_s, h->d_v_s, n_cat) || cp(dst->sv_i, h->d_sv_i, n_cat) ||
        cp(dst->e_i, h->d_e_i, n_e) || cp(dst->t_su, h->d_t_su, n_cat) || cp(dst->t_s, h->d_t_s, n_cat) ||
        cp(dst->a_ip, h->d_a_ip, n_cat) || cp(dst->v_ip, h->d_v_ip, n_cat) ||
        cp(dst->u_ice, h->d_u_ice, h->n2) || cp(dst->v_ice, h->d_v_ice, h->n2) ||
        cp(dst->stress1, h->d_s1_ice, h->n2) || cp(dst->stress2, h->d_s2_ice, h->n2) || cp(dst->stress12, h->d_s12_ice, h->n2) ||
        cp(dst->ice_frac, h->d_ice_frac, h->n2) || cp(dst->ice_thick, h->d_ice_thick, h->n2) ||
        cp(dst->ice_tsfc, h->d_ice_tsfc, h->n2) || cp(dst->ice_albedo, h->d_ice_albedo, h->n2)) {
        return -1;
    }
    return 0;
}

int32_t nh_ocean_si3_step(NhOceanSi3Handle* h, const NhOceanForcingPtrs* f, double dt_s) {
    if (!h || !f) { oce_set_err("Null argument in nh_ocean_si3_step"); return -1; }
    if (!(dt_s > 0.0)) { oce_set_err("dt_s must be positive in nh_ocean_si3_step"); return -1; }

    auto up = [&](float* dst, const float* s) -> int32_t {
        if (!s) { OCE_CHECK(cudaMemset(dst, 0, h->n2 * sizeof(float))); return 0; }
        OCE_CHECK(cudaMemcpy(dst, s, h->n2 * sizeof(float), cudaMemcpyHostToDevice));
        return 0;
    };
    if (up(h->d_f_u10, f->u10m) || up(h->d_f_v10, f->v10m) ||
        up(h->d_f_t2m, f->t2m) || up(h->d_f_q2m, f->q2m) || up(h->d_f_mslp, f->mslp) ||
        up(h->d_f_sw, f->sw_down) || up(h->d_f_lw, f->lw_down) ||
        up(h->d_f_pliq, f->precip_liq) || up(h->d_f_pice, f->precip_ice) || up(h->d_f_cld, f->cld_fra)) {
        return -1;
    }

    const dim3 block(16, 16);
    const dim3 grid((h->cfg.nx + 15) / 16, (h->cfg.ny + 15) / 16);

    // 1. Surface open-ocean fluxes & inverse-barometer eta_ib
    oce_surface_forcing_kernel<<<grid, block>>>(
        h->cfg, h->d_tmask,
        h->d_f_u10, h->d_f_v10, h->d_f_t2m, h->d_f_q2m, h->d_f_mslp,
        h->d_f_sw, h->d_f_lw, h->d_f_pliq, h->d_f_pice, h->d_temp,
        h->d_eta_ib, h->d_taux_atm, h->d_tauy_atm, h->d_qsr_oce, h->d_qns_oce, h->d_emp_oce
    );
    OCE_CHECK(cudaGetLastError());

    OCE_CHECK(cudaMemset(h->d_qns_tot, 0, h->n2 * sizeof(float)));
    OCE_CHECK(cudaMemset(h->d_emp_tot, 0, h->n2 * sizeof(float)));

    // 2. SI3 dynamics: mEVP rheology, advection, Thorndike ridging + ITD remapping
    if (h->cfg.enable_si3) {
        if (h->cfg.enable_dynamics) {
            si3_mevp_init_kernel<<<grid, block>>>(
                h->cfg, dt_s, h->d_tmask, h->d_msf, h->d_bathy, h->d_a_i, h->d_v_i, h->d_strength
            );
            OCE_CHECK(cudaGetLastError());
            for (int32_t iter = 0; iter < h->cfg.n_evp_substeps; ++iter) {
                si3_mevp_stress_kernel<<<grid, block>>>(
                    h->cfg, h->d_tmask, h->d_msf, h->d_a_i, h->d_strength, h->d_u_ice, h->d_v_ice,
                    h->d_s1_ice, h->d_s2_ice, h->d_s12_ice, h->d_delta_ice, h->d_div_ice
                );
                si3_mevp_vel_kernel<<<grid, block>>>(
                    h->cfg, dt_s, h->d_tmask, h->d_msf, h->d_fcor,
                    h->d_a_i, h->d_v_i, h->d_v_s, h->d_u, h->d_v, h->d_ssh,
                    h->d_taux_atm, h->d_tauy_atm, h->d_s1_ice, h->d_s2_ice, h->d_s12_ice,
                    h->d_u_ice, h->d_v_ice, h->d_tmp1, h->d_tmp2
                );
                OCE_CHECK(cudaMemcpy(h->d_u_ice, h->d_tmp1, h->n2 * sizeof(float), cudaMemcpyDeviceToDevice));
                OCE_CHECK(cudaMemcpy(h->d_v_ice, h->d_tmp2, h->n2 * sizeof(float), cudaMemcpyDeviceToDevice));
            }
            OCE_CHECK(cudaGetLastError());

            auto advect_cat = [&](float* arr, int32_t n_layers) -> int32_t {
                for (int32_t l = 0; l < n_layers; ++l) {
                    advect_upwind_2d_layer_kernel<<<grid, block>>>(
                        h->cfg, dt_s, l, h->d_tmask, h->d_msf, h->d_u_ice, h->d_v_ice, arr, h->d_tmp1
                    );
                    OCE_CHECK(cudaMemcpy(arr + static_cast<size_t>(l) * h->n2, h->d_tmp1,
                                         h->n2 * sizeof(float), cudaMemcpyDeviceToDevice));
                }
                return 0;
            };
            if (advect_cat(h->d_a_i, TEMPEST_SI3_NCAT) || advect_cat(h->d_v_i, TEMPEST_SI3_NCAT) ||
                advect_cat(h->d_v_s, TEMPEST_SI3_NCAT) || advect_cat(h->d_sv_i, TEMPEST_SI3_NCAT) ||
                advect_cat(h->d_e_i, TEMPEST_SI3_NCAT * TEMPEST_SI3_NLAY) ||
                advect_cat(h->d_a_ip, TEMPEST_SI3_NCAT) || advect_cat(h->d_v_ip, TEMPEST_SI3_NCAT)) {
                return -1;
            }
        } else {
            OCE_CHECK(cudaMemset(h->d_div_ice, 0, h->n2 * sizeof(float)));
            OCE_CHECK(cudaMemset(h->d_delta_ice, 0, h->n2 * sizeof(float)));
        }

        si3_ridge_and_itd_remap_kernel<<<grid, block>>>(
            h->cfg, dt_s, h->d_tmask, h->d_div_ice, h->d_delta_ice,
            h->d_a_i, h->d_v_i, h->d_v_s, h->d_sv_i, h->d_e_i,
            h->d_t_su, h->d_t_s, h->d_a_ip, h->d_v_ip, h->d_qns_tot, h->d_emp_tot
        );
        OCE_CHECK(cudaGetLastError());
    }

    // 3. SI3 BL99 4-layer thermodynamics, frazil accretion, ponds, albedo, & iceupdate
    si3_thermo_bl99_frazil_kernel<<<grid, block>>>(
        h->cfg, dt_s, h->d_dz, h->d_tmask,
        h->d_f_u10, h->d_f_v10, h->d_f_t2m, h->d_f_q2m, h->d_f_mslp,
        h->d_f_sw, h->d_f_lw, h->d_f_pliq, h->d_f_pice, h->d_f_cld,
        h->d_u, h->d_v, h->d_u_ice ? h->d_u_ice : h->d_tmp1, h->d_v_ice ? h->d_v_ice : h->d_tmp1,
        h->d_qsr_oce, h->d_qns_oce, h->d_emp_oce, h->d_taux_atm, h->d_tauy_atm,
        h->d_temp, h->d_sal, h->d_ssh,
        h->d_a_i, h->d_v_i, h->d_v_s, h->d_sv_i, h->d_e_i,
        h->d_t_su, h->d_t_s, h->d_a_ip, h->d_v_ip,
        h->d_qsr_tot, h->d_qns_tot, h->d_emp_tot, h->d_sfx_tot,
        h->d_taux_oce, h->d_tauy_oce,
        h->d_ice_frac, h->d_ice_thick, h->d_ice_tsfc, h->d_ice_albedo
    );
    OCE_CHECK(cudaGetLastError());

    // 4. TEOS-10 density, N^2, Gaspar TKE + EVD, MLD
    oce_eos_tke_kernel<<<grid, block>>>(
        h->cfg, dt_s, h->d_dz, h->d_zt, h->d_zw, h->d_tmask, h->d_mbkt,
        h->d_temp, h->d_sal, h->d_u, h->d_v, h->d_taux_oce, h->d_tauy_oce, h->d_ice_frac,
        h->d_rhd, h->d_tke, h->d_avm, h->d_avt, h->d_mld
    );
    OCE_CHECK(cudaGetLastError());

    // 5. Split-Explicit Barotropic Mode + 3D Baroclinic Momentum + Continuity
    if (h->cfg.enable_dynamics) {
        const int32_t n_bt = max(h->cfg.n_bt_substeps,
                                 min(120, static_cast<int32_t>(ceil(dt_s * 180.0 / (0.4 * h->min_dx_mid)))));
        const double dt_bt = dt_s / static_cast<double>(n_bt);
        for (int32_t sb = 0; sb < n_bt; ++sb) {
            oce_barotropic_uv_kernel<<<grid, block>>>(
                h->cfg, dt_bt, h->d_tmask, h->d_bathy, h->d_msf, h->d_fcor,
                h->d_ssh, h->d_eta_ib, h->d_taux_oce, h->d_tauy_oce,
                nullptr, nullptr, h->d_ub, h->d_vb, h->d_tmp1, h->d_tmp2
            );
            OCE_CHECK(cudaMemcpy(h->d_ub, h->d_tmp1, h->n2 * sizeof(float), cudaMemcpyDeviceToDevice));
            OCE_CHECK(cudaMemcpy(h->d_vb, h->d_tmp2, h->n2 * sizeof(float), cudaMemcpyDeviceToDevice));
            oce_barotropic_ssh_kernel<<<grid, block>>>(
                h->cfg, dt_bt, h->d_tmask, h->d_bathy, h->d_msf,
                h->d_ub, h->d_vb, h->d_ssh
            );
        }
        OCE_CHECK(cudaGetLastError());

        oce_baroclinic_3d_kernel<<<grid, block>>>(
            h->cfg, dt_s, h->d_dz, h->d_tmask, h->d_mbkt, h->d_msf, h->d_fcor,
            h->d_rhd, h->d_avm, h->d_ssh, h->d_ub, h->d_vb, h->d_taux_oce, h->d_tauy_oce,
            h->d_u, h->d_v, h->d_tmp1, h->d_tmp2
        );
        OCE_CHECK(cudaMemcpy(h->d_u, h->d_tmp1, h->n3 * sizeof(float), cudaMemcpyDeviceToDevice));
        OCE_CHECK(cudaMemcpy(h->d_v, h->d_tmp2, h->n3 * sizeof(float), cudaMemcpyDeviceToDevice));
        OCE_CHECK(cudaGetLastError());

        oce_continuity_w_kernel<<<grid, block>>>(
            h->cfg, h->d_dz, h->d_tmask, h->d_mbkt, h->d_msf, h->d_u, h->d_v, h->d_w
        );
        OCE_CHECK(cudaGetLastError());
    }

    // 6. 3D Tracers (Conservative advection + 2-band traqsr + Implicit Thomas vertical mixing)
    oce_tracer_adv_diff_step_kernel<<<grid, block>>>(
        h->cfg, dt_s, 1, h->d_dz, h->d_zw, h->d_tmask, h->d_mbkt, h->d_msf,
        h->d_u, h->d_v, h->d_w, h->d_avt, h->d_qsr_tot, h->d_temp, h->d_tmp_tr
    );
    OCE_CHECK(cudaGetLastError());
    OCE_CHECK(cudaMemcpy(h->d_temp, h->d_tmp_tr, h->n3 * sizeof(double), cudaMemcpyDeviceToDevice));
    oce_tracer_adv_diff_step_kernel<<<grid, block>>>(
        h->cfg, dt_s, 0, h->d_dz, h->d_zw, h->d_tmask, h->d_mbkt, h->d_msf,
        h->d_u, h->d_v, h->d_w, h->d_avt, h->d_qsr_tot, h->d_sal, h->d_tmp_tr
    );
    OCE_CHECK(cudaGetLastError());
    OCE_CHECK(cudaMemcpy(h->d_sal, h->d_tmp_tr, h->n3 * sizeof(double), cudaMemcpyDeviceToDevice));
    OCE_CHECK(cudaDeviceSynchronize());
    return 0;
}

double nh_ocean_si3_teos10_rho(double ct_c, double sa_psu, double depth_m) {
    return eval_teos10_rho_hd(ct_c, sa_psu, depth_m);
}

double nh_ocean_si3_freezing_temp_c(double sa_psu, double depth_m) {
    return eval_teos10_fzp_hd(sa_psu, depth_m);
}

const char* nh_ocean_si3_last_error(void) {
    return g_oce_last_err;
}

} // extern "C"
