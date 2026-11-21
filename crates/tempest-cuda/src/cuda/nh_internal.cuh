#ifndef TEMPEST_NWP_V2_NH_INTERNAL_CUH
#define TEMPEST_NWP_V2_NH_INTERNAL_CUH

#include "nh_core.h"

#include <cuda_runtime.h>
#include <cufft.h>
#include <math.h>
#include <stdint.h>

namespace tempest_cuda_internal {

// Strict SI thermodynamic & planetary constants matching tempest-core / projection.py
constexpr double R_D = 287.058;
constexpr double R_V = 461.5;
constexpr double CP_D = 1004.5;
constexpr double CV_D = CP_D - R_D;
constexpr double KAPPA = R_D / CP_D;
constexpr double GAMMA_GAS = CP_D / CV_D;
constexpr double GRAVITY = 9.80665;
constexpr double P0_PA = 100000.0;
constexpr double EPS_V = (R_V / R_D) - 1.0;
constexpr double EPS_MW = R_D / R_V; // ~0.622

constexpr double LV0 = 2.501e6;
constexpr double LF0 = 3.337e5;
constexpr double LS0 = LV0 + LF0;
constexpr double T_FREEZE = 273.15;
constexpr double STEFAN_BOLTZMANN = 5.670374e-8;
constexpr double KARMAN = 0.4;
constexpr double PI_CONST = 3.14159265358979323846;
constexpr double EARTH_RADIUS = 6371229.0;
constexpr double OMEGA = 7.292115e-5;

struct DeviceStaticGrid {
    const double* zeta_w;
    const double* zeta_s;
    const double* dzeta;
    const double* b_w;
    const double* b_s;
    const double* db_w;
    const double* db_s;
    const double* hs;
    const double* msf;
    const double* fcor;
};

constexpr float R_D_F = 287.058f;
constexpr float R_V_F = 461.5f;
constexpr float CP_D_F = 1004.5f;
constexpr float CV_D_F = CP_D_F - R_D_F;
constexpr float KAPPA_F = R_D_F / CP_D_F;
constexpr float GAMMA_GAS_F = CP_D_F / CV_D_F;
constexpr float GRAVITY_F = 9.80665f;
constexpr float P0_PA_F = 100000.0f;
constexpr float EPS_V_F = (R_V_F / R_D_F) - 1.0f;
constexpr float EPS_MW_F = R_D_F / R_V_F;
constexpr float LV0_F = 2.501e6f;
constexpr float LF0_F = 3.337e5f;
constexpr float LS0_F = LV0_F + LF0_F;
constexpr float T_FREEZE_F = 273.15f;
constexpr float STEFAN_BOLTZMANN_F = 5.670374e-8f;
constexpr float KARMAN_F = 0.4f;
constexpr float PI_CONST_F = 3.14159265358979323846f;
constexpr float EARTH_RADIUS_F = 6371229.0f;
constexpr float OMEGA_F = 7.292115e-5f;

__device__ __host__ inline void eval_ref_state_pis(
    double z,
    double theta_sfc,
    double pi_sfc,
    double n_ref,
    double& theta_bar,
    double& pi_bar,
    double& dtheta_dz,
    double& rho_bar,
    double& cs2_bar
) {
    const double n2 = n_ref * n_ref;
    if (n2 > 1e-12) {
        const double exp_arg = (n2 * z) / GRAVITY;
        const double e_pos = exp(exp_arg);
        const double e_neg = exp(-exp_arg);
        theta_bar = theta_sfc * e_pos;
        pi_bar = pi_sfc - (GRAVITY * GRAVITY) / (CP_D * theta_sfc * n2) * (1.0 - e_neg);
        dtheta_dz = theta_bar * (n2 / GRAVITY);
    } else {
        theta_bar = theta_sfc;
        pi_bar = pi_sfc - (GRAVITY * z) / (CP_D * theta_sfc);
        dtheta_dz = 0.0;
    }
    if (pi_bar < 1e-4) pi_bar = 1e-4;
    const double p_bar = P0_PA * pow(pi_bar, 1.0 / KAPPA);
    const double t_bar = theta_bar * pi_bar;
    rho_bar = p_bar / (R_D * t_bar);
    cs2_bar = GAMMA_GAS * R_D * t_bar;
}

__device__ __host__ inline double eval_theta_bar(
    double z,
    double theta_sfc,
    double n_ref
) {
    const double n2 = n_ref * n_ref;
    if (n2 > 1e-12) {
        return theta_sfc * exp((n2 * z) / GRAVITY);
    }
    return theta_sfc;
}

__device__ __host__ inline void eval_theta_pi_bar_pis(
    double z,
    double theta_sfc,
    double pi_sfc,
    double n_ref,
    double& theta_bar,
    double& pi_bar
) {
    const double n2 = n_ref * n_ref;
    if (n2 > 1e-12) {
        const double e_pos = exp((n2 * z) / GRAVITY);
        const double e_neg = 1.0 / e_pos;
        theta_bar = theta_sfc * e_pos;
        pi_bar = pi_sfc - (GRAVITY * GRAVITY) / (CP_D * theta_sfc * n2) * (1.0 - e_neg);
    } else {
        theta_bar = theta_sfc;
        pi_bar = pi_sfc - (GRAVITY * z) / (CP_D * theta_sfc);
    }
    if (pi_bar < 1e-4) pi_bar = 1e-4;
}

__device__ __host__ inline void eval_theta_pi_bar_f32(
    float z,
    float theta_sfc,
    float pi_sfc,
    float n_ref,
    float& theta_bar,
    float& pi_bar
) {
    const float n2 = n_ref * n_ref;
    if (n2 > 1e-12f) {
        const float e_pos = expf((n2 * z) / GRAVITY_F);
        const float e_neg = 1.0f / e_pos;
        theta_bar = theta_sfc * e_pos;
        pi_bar = pi_sfc - (GRAVITY_F * GRAVITY_F) / (CP_D_F * theta_sfc * n2) * (1.0f - e_neg);
    } else {
        theta_bar = theta_sfc;
        pi_bar = pi_sfc - (GRAVITY_F * z) / (CP_D_F * theta_sfc);
    }
    if (pi_bar < 1e-4f) pi_bar = 1e-4f;
}

__device__ __host__ inline float eval_rho_bar_f32(
    float z,
    float theta_sfc,
    float pi_sfc,
    float n_ref
) {
    const float n2 = n_ref * n_ref;
    float theta_bar;
    float pi_bar;
    if (n2 > 1e-12f) {
        const float e_pos = expf((n2 * z) / GRAVITY_F);
        const float e_neg = 1.0f / e_pos;
        theta_bar = theta_sfc * e_pos;
        pi_bar = pi_sfc - (GRAVITY_F * GRAVITY_F) / (CP_D_F * theta_sfc * n2) * (1.0f - e_neg);
    } else {
        theta_bar = theta_sfc;
        pi_bar = pi_sfc - (GRAVITY_F * z) / (CP_D_F * theta_sfc);
    }
    if (pi_bar < 1e-4f) pi_bar = 1e-4f;
    const float p_bar = P0_PA_F * powf(pi_bar, 1.0f / KAPPA_F);
    const float t_bar = theta_bar * pi_bar;
    return p_bar / (R_D_F * t_bar);
}

__device__ __host__ inline void eval_ref_state(
    double z,
    double theta_sfc,
    double p_sfc,
    double n_ref,
    double& theta_bar,
    double& pi_bar,
    double& dtheta_dz,
    double& rho_bar,
    double& cs2_bar
) {
    const double pi_sfc = pow(p_sfc / P0_PA, KAPPA);
    const double n2 = n_ref * n_ref;
    if (n2 > 1e-12) {
        const double exp_arg = (n2 * z) / GRAVITY;
        const double e_pos = exp(exp_arg);
        const double e_neg = exp(-exp_arg);
        theta_bar = theta_sfc * e_pos;
        pi_bar = pi_sfc - (GRAVITY * GRAVITY) / (CP_D * theta_sfc * n2) * (1.0 - e_neg);
        dtheta_dz = theta_bar * (n2 / GRAVITY);
    } else {
        theta_bar = theta_sfc;
        pi_bar = pi_sfc - (GRAVITY * z) / (CP_D * theta_sfc);
        dtheta_dz = 0.0;
    }
    if (pi_bar < 1e-4) pi_bar = 1e-4;
    const double p_bar = P0_PA * pow(pi_bar, 1.0 / KAPPA);
    const double t_bar = theta_bar * pi_bar;
    rho_bar = p_bar / (R_D * t_bar);
    cs2_bar = GAMMA_GAS * R_D * t_bar;
}

__device__ inline int32_t wrap_idx(int32_t i, int32_t n) {
    int32_t m = i % n;
    return (m < 0) ? (m + n) : m;
}

__device__ inline int32_t idx_h(int32_t j, int32_t i, int32_t ny, int32_t nx, int32_t bc_x, int32_t bc_y) {
    const int32_t jj = (bc_y == 1) ? wrap_idx(j, ny) : ((j < 0) ? 0 : ((j >= ny) ? ny - 1 : j));
    const int32_t ii = (bc_x == 1) ? wrap_idx(i, nx) : ((i < 0) ? 0 : ((i >= nx) ? nx - 1 : i));
    return jj * nx + ii;
}

__device__ inline double read_2d(
    const double* arr, int32_t j, int32_t i,
    int32_t ny, int32_t nx, int32_t bc_x, int32_t bc_y
) {
    return arr[idx_h(j, i, ny, nx, bc_x, bc_y)];
}

__device__ inline double read_s(
    const float* arr, int32_t k, int32_t j, int32_t i,
    int32_t nz, int32_t ny, int32_t nx, int32_t bc_x, int32_t bc_y
) {
    const int32_t kk = (k < 0) ? 0 : ((k >= nz) ? nz - 1 : k);
    return static_cast<double>(arr[static_cast<size_t>(kk) * ny * nx + idx_h(j, i, ny, nx, bc_x, bc_y)]);
}

__device__ inline double read_u(
    const float* u, int32_t k, int32_t j, int32_t i,
    int32_t nz, int32_t ny, int32_t nx, int32_t bc_x, int32_t bc_y
) {
    const int32_t kk = (k < 0) ? 0 : ((k >= nz) ? nz - 1 : k);
    const int32_t jj = (bc_y == 1) ? wrap_idx(j, ny) : ((j < 0) ? 0 : ((j >= ny) ? ny - 1 : j));
    int32_t ii = i;
    if (bc_x == 1) {
        ii = wrap_idx(i, nx);
    } else {
        if (bc_x == 0 && (i <= 0 || i >= nx)) return 0.0;
        if (ii < 0) ii = 0;
        if (ii > nx) ii = nx;
    }
    return static_cast<double>(u[static_cast<size_t>(kk) * ny * (nx + 1) + jj * (nx + 1) + ii]);
}

__device__ inline double read_v(
    const float* v, int32_t k, int32_t j, int32_t i,
    int32_t nz, int32_t ny, int32_t nx, int32_t bc_x, int32_t bc_y
) {
    const int32_t kk = (k < 0) ? 0 : ((k >= nz) ? nz - 1 : k);
    const int32_t ii = (bc_x == 1) ? wrap_idx(i, nx) : ((i < 0) ? 0 : ((i >= nx) ? nx - 1 : i));
    int32_t jj = j;
    if (bc_y == 1) {
        jj = wrap_idx(j, ny);
    } else {
        if (bc_y == 0 && (j <= 0 || j >= ny)) return 0.0;
        if (jj < 0) jj = 0;
        if (jj > ny) jj = ny;
    }
    return static_cast<double>(v[static_cast<size_t>(kk) * (ny + 1) * nx + jj * nx + ii]);
}

__device__ inline double read_w(
    const float* w, int32_t k, int32_t j, int32_t i,
    int32_t nz, int32_t ny, int32_t nx, int32_t bc_x, int32_t bc_y
) {
    if (k <= 0) return static_cast<double>(w[idx_h(j, i, ny, nx, bc_x, bc_y)]);
    if (k >= nz) return 0.0;
    return static_cast<double>(w[static_cast<size_t>(k) * ny * nx + idx_h(j, i, ny, nx, bc_x, bc_y)]);
}

__device__ inline float read_2d_f(
    const double* arr, int32_t j, int32_t i,
    int32_t ny, int32_t nx, int32_t bc_x, int32_t bc_y
) {
    return static_cast<float>(arr[idx_h(j, i, ny, nx, bc_x, bc_y)]);
}

__device__ inline float read_s_f(
    const float* arr, int32_t k, int32_t j, int32_t i,
    int32_t nz, int32_t ny, int32_t nx, int32_t bc_x, int32_t bc_y
) {
    const int32_t kk = (k < 0) ? 0 : ((k >= nz) ? nz - 1 : k);
    return arr[static_cast<size_t>(kk) * ny * nx + idx_h(j, i, ny, nx, bc_x, bc_y)];
}

__device__ inline float read_u_f(
    const float* u, int32_t k, int32_t j, int32_t i,
    int32_t nz, int32_t ny, int32_t nx, int32_t bc_x, int32_t bc_y
) {
    const int32_t kk = (k < 0) ? 0 : ((k >= nz) ? nz - 1 : k);
    const int32_t jj = (bc_y == 1) ? wrap_idx(j, ny) : ((j < 0) ? 0 : ((j >= ny) ? ny - 1 : j));
    int32_t ii = i;
    if (bc_x == 1) {
        ii = wrap_idx(i, nx);
    } else {
        if (bc_x == 0 && (i <= 0 || i >= nx)) return 0.0f;
        if (ii < 0) ii = 0;
        if (ii > nx) ii = nx;
    }
    return u[static_cast<size_t>(kk) * ny * (nx + 1) + jj * (nx + 1) + ii];
}

__device__ inline float read_v_f(
    const float* v, int32_t k, int32_t j, int32_t i,
    int32_t nz, int32_t ny, int32_t nx, int32_t bc_x, int32_t bc_y
) {
    const int32_t kk = (k < 0) ? 0 : ((k >= nz) ? nz - 1 : k);
    const int32_t ii = (bc_x == 1) ? wrap_idx(i, nx) : ((i < 0) ? 0 : ((i >= nx) ? nx - 1 : i));
    int32_t jj = j;
    if (bc_y == 1) {
        jj = wrap_idx(j, ny);
    } else {
        if (bc_y == 0 && (j <= 0 || j >= ny)) return 0.0f;
        if (jj < 0) jj = 0;
        if (jj > ny) jj = ny;
    }
    return v[static_cast<size_t>(kk) * (ny + 1) * nx + jj * nx + ii];
}

__device__ inline float read_w_f(
    const float* w, int32_t k, int32_t j, int32_t i,
    int32_t nz, int32_t ny, int32_t nx, int32_t bc_x, int32_t bc_y
) {
    if (k <= 0) return w[idx_h(j, i, ny, nx, bc_x, bc_y)];
    if (k >= nz) return 0.0f;
    return w[static_cast<size_t>(k) * ny * nx + idx_h(j, i, ny, nx, bc_x, bc_y)];
}

} // namespace tempest_cuda_internal

struct NhCudaHandle {
    NhGridConfig cfg;
    size_t n_s;
    size_t n_u;
    size_t n_v;
    size_t n_w;

    double *d_zeta_w = nullptr, *d_zeta_s = nullptr, *d_dzeta = nullptr;
    double *d_b_w = nullptr, *d_b_s = nullptr, *d_db_w = nullptr, *d_db_s = nullptr;
    double *d_hs = nullptr, *d_msf = nullptr, *d_fcor = nullptr;

    float *d_u = nullptr, *d_v = nullptr, *d_w = nullptr, *d_thp = nullptr, *d_pip = nullptr;
    float *d_qv = nullptr, *d_qc = nullptr, *d_qr = nullptr, *d_qi = nullptr, *d_qs = nullptr, *d_qg = nullptr;

    float *d_u_n = nullptr, *d_v_n = nullptr, *d_w_n = nullptr, *d_thp_n = nullptr, *d_pip_n = nullptr;
    float *d_ru = nullptr, *d_rv = nullptr, *d_rw = nullptr, *d_rth = nullptr, *d_rpi = nullptr;

    float *d_div = nullptr, *d_q_tmp = nullptr;
    double *d_tri_c = nullptr, *d_tri_d = nullptr;
    uint32_t *d_max_u32 = nullptr;

    // Optional LBC & IAU buffers
    float *d_lbc_u = nullptr, *d_lbc_v = nullptr, *d_lbc_thp = nullptr, *d_lbc_pip = nullptr;
    float *d_iau_u = nullptr, *d_iau_v = nullptr, *d_iau_thp = nullptr, *d_iau_pip = nullptr;

    // Phase B2 GPU physics state & buffers (allocated lazily when physics enabled/uploaded)
    NhPhysicsConfig pcfg{};
    int32_t has_physics_alloc = 0;
    int32_t rad_initialized = 0;
    double rad_timer_s = 0.0;

    float *d_rad_dth_dt = nullptr; // [nz, ny, nx] cached SW+LW theta tendency [K/s]

    // 2D static surface descriptors [ny, nx]
    float *d_lat_deg = nullptr, *d_lon_deg = nullptr, *d_lsm = nullptr, *d_lake_mask = nullptr;
    float *d_soil_type = nullptr, *d_bathy_depth_m = nullptr, *d_z0m_base = nullptr;
    float *d_albedo_base = nullptr, *d_veg_frac = nullptr;

    // 2D / 4-layer prognostic & diagnostic surface state
    float *d_t_skin = nullptr, *d_soil_t = nullptr, *d_soil_m = nullptr, *d_snow_swe = nullptr;
    float *d_sst_bulk = nullptr, *d_dt_cool = nullptr, *d_dt_warm = nullptr;
    float *d_lake_tbot = nullptr, *d_lake_hmld = nullptr, *d_lake_ctheta = nullptr, *d_lake_hice = nullptr;
    float *d_rain_acc = nullptr, *d_snow_acc = nullptr, *d_graupel_acc = nullptr;
    float *d_sw_down = nullptr, *d_lw_down = nullptr, *d_shf = nullptr, *d_lhf = nullptr;
    float *d_ustar = nullptr, *d_pblh = nullptr, *d_t2m = nullptr, *d_q2m = nullptr, *d_u10m = nullptr, *d_v10m = nullptr;
    float *d_sea_ice_frac = nullptr, *d_sea_ice_thick = nullptr, *d_sea_ice_tsfc = nullptr, *d_sea_ice_albedo = nullptr;

    // Transient intra-step physics coupling buffers [ny, nx]
    float *d_precip_liq_rate = nullptr, *d_precip_ice_rate = nullptr;
    float *d_sfc_drag_coeff = nullptr; // rho * u*^2 / wspd for implicit PBL bottom momentum stress

    // GPU cuFFT 1D zonal polar filter state (allocated only when is_spherical == 1 && polar_filter_lat_deg > 0)
    cufftHandle fft_r2c_s = 0;
    cufftHandle fft_c2r_s = 0;
    cufftHandle fft_r2c_v = 0;
    cufftHandle fft_c2r_v = 0;
    int32_t fft_ready = 0;
    int32_t fft_n_rows_s = 0;
    int32_t fft_n_rows_v = 0;
    int32_t *d_polar_rows_s = nullptr;
    int32_t *d_polar_rows_v = nullptr;
    float *d_polar_cos_s = nullptr;
    float *d_polar_cos_v = nullptr;
    float *d_fft_real = nullptr;
    float *d_polar_save = nullptr;
    cufftComplex *d_fft_cplx = nullptr;
};

void nh_internal_set_error(const char* msg);
void nh_internal_set_cuda_error(const char* file, int line, cudaError_t err);
void nh_physics_free_buffers(NhCudaHandle* h);

#define NH_CUDA_CHECK(expr)                                                     \
    do {                                                                        \
        cudaError_t _err = (expr);                                              \
        if (_err != cudaSuccess) {                                              \
            nh_internal_set_cuda_error(__FILE__, __LINE__, _err);               \
            return -1;                                                          \
        }                                                                       \
    } while (0)

#endif /* TEMPEST_NWP_V2_NH_INTERNAL_CUH */
