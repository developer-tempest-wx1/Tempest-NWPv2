#ifndef TEMPEST_NWP_V2_NH_CORE_H
#define TEMPEST_NWP_V2_NH_CORE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Grid, reference atmosphere, boundary condition, and solver parameters
 * for the Arakawa C-grid split-explicit RK3 non-hydrostatic GPU core.
 *
 * All quantities use strict SI units (m, s, K, Pa, kg/kg).
 */
typedef struct NhGridConfig {
    int32_t nz;
    int32_t ny;
    int32_t nx;
    int32_t has_moist;      /* 0 = dry (5 fields), 1 = moist (6 species qv..qg) */
    int32_t bc_x;           /* 0 = rigid wall/zero-grad, 1 = periodic, 2 = Davies LBC */
    int32_t bc_y;           /* 0 = rigid wall/zero-grad, 1 = periodic, 2 = Davies LBC */
    int32_t lbc_edges;      /* Bitmask for Davies edges: 1=West, 2=East, 4=South, 8=North (15=all) */
    int32_t use_coriolis;   /* 0 = f=0 and m=1 curvature off, 1 = active LCC Coriolis/metric */
    int32_t adv_order;      /* 3 = UP3, 5 = WENO-5 horizontal/vertical */
    int32_t lbc_rim_width;  /* Davies boundary rim width in grid cells */
    int32_t has_iau;        /* 0 = off, 1 = apply incremental analysis update tendencies */

    double dx_m;
    double dy_m;
    double z_top_m;

    double theta_sfc_k;     /* Reference surface potential temperature [K] */
    double p_sfc_pa;        /* Reference surface pressure [Pa] */
    double n_ref_s;         /* Reference Brunt-Vaisala angular frequency [s^-1] */

    double beta_offcenter;  /* Vertical acoustic off-centering in [0.5, 0.7], default 0.55 */
    double div_damp_coeff;  /* Dimensionless 3D divergence damping factor (~0.03) */
    double km_const;        /* Explicit momentum diffusivity [m^2/s] */
    double kh_const;        /* Explicit scalar diffusivity [m^2/s] */
    double hyperdiff_coeff; /* Optional 4th-order horizontal hyperdiffusion (~0.0..0.02) */
    double sponge_depth_m;  /* Upper Klemp-Wilhelmson Rayleigh sponge depth [m] */
    double sponge_tau_max_s;/* Max Rayleigh damping rate at z_top [s^-1] */
    double lbc_tau_max_s;   /* Max Davies relaxation rate at outer rim [s^-1] */
} NhGridConfig;

/**
 * Host pointers for prognostic state arrays (float32).
 *
 * Expected shapes:
 *   u       : [nz,     ny,     nx + 1]
 *   v       : [nz,     ny + 1, nx    ]
 *   w       : [nz + 1, ny,     nx    ]
 *   theta_p : [nz,     ny,     nx    ]  (perturbation theta' = theta - theta_bar)
 *   pi_p    : [nz,     ny,     nx    ]  (perturbation Exner pi' = pi - pi_bar)
 *   qv..qg  : [nz,     ny,     nx    ]  (nullable when has_moist == 0)
 */
typedef struct NhStatePtrs {
    float* u;
    float* v;
    float* w;
    float* theta_p;
    float* pi_p;
    float* qv;
    float* qc;
    float* qr;
    float* qi;
    float* qs;
    float* qg;
} NhStatePtrs;

/**
 * Const host pointers for setting LBC target or IAU forcing buffers.
 */
typedef struct NhConstStatePtrs {
    const float* u;
    const float* v;
    const float* w;
    const float* theta_p;
    const float* pi_p;
    const float* qv;
    const float* qc;
    const float* qr;
    const float* qi;
    const float* qs;
    const float* qg;
} NhConstStatePtrs;

typedef struct NhCudaHandle NhCudaHandle;

/**
 * Query active CUDA device VRAM free/total bytes (returns 0 on success).
 */
int32_t nh_cuda_vram_info(uint64_t* free_bytes, uint64_t* total_bytes);

/**
 * Estimate device VRAM bytes required by a tile of dimensions (nz, ny, nx).
 */
uint64_t nh_cuda_estimate_vram_bytes(
    int32_t nz,
    int32_t ny,
    int32_t nx,
    int32_t has_moist,
    int32_t has_lbc,
    int32_t has_iau
);

/**
 * Create a persistent non-hydrostatic GPU solver context on the active CUDA device.
 * Returns NULL on error (check nh_cuda_last_error()).
 *
 * Vertical 1D arrays match VerticalGrid:
 *   zeta_w, B_w, dB_dzeta_w : [nz + 1]
 *   zeta_s, dzeta, B_s, dB_dzeta_s : [nz]
 * Surface 2D arrays:
 *   hs_m, msf, f_cor : [ny, nx]
 */
NhCudaHandle* nh_cuda_create(
    const NhGridConfig* cfg,
    const double* zeta_w,
    const double* zeta_s,
    const double* dzeta,
    const double* b_w,
    const double* b_s,
    const double* db_dzeta_w,
    const double* db_dzeta_s,
    const double* hs_m,
    const double* msf,
    const double* f_cor
);

void nh_cuda_destroy(NhCudaHandle* handle);

int32_t nh_cuda_upload_state(NhCudaHandle* handle, const NhConstStatePtrs* src);
int32_t nh_cuda_download_state(NhCudaHandle* handle, const NhStatePtrs* dst);

int32_t nh_cuda_set_lbc_target(NhCudaHandle* handle, const NhConstStatePtrs* target);
int32_t nh_cuda_set_iau_forcing(NhCudaHandle* handle, const NhConstStatePtrs* ddt_state);

/**
 * Advance prognostic state by dt_s using split-explicit Wicker-Skamarock RK3.
 * If n_substeps <= 0, substeps per RK3 stage are chosen automatically from
 * horizontal acoustic CFL <= 0.40.
 * Returns 0 on success, non-zero on CUDA/validation failure.
 */
int32_t nh_cuda_step_rk3(NhCudaHandle* handle, double dt_s, int32_t n_substeps);

/**
 * Return last human-readable CUDA error string on the current thread.
 */
const char* nh_cuda_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* TEMPEST_NWP_V2_NH_CORE_H */
