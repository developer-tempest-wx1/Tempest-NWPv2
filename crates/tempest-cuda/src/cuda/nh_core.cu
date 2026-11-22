#include "nh_internal.cuh"

#include <cuda_runtime.h>
#include <cufft.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

using namespace tempest_cuda_internal;

namespace {

thread_local char g_last_error[512] = "";

inline void set_error(const char* msg) {
    snprintf(g_last_error, sizeof(g_last_error), "%s", msg);
}

#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t _err = (expr);                                              \
        if (_err != cudaSuccess) {                                              \
            snprintf(g_last_error, sizeof(g_last_error),                        \
                     "CUDA error (%s:%d): %s", __FILE__, __LINE__,              \
                     cudaGetErrorString(_err));                                 \
            return -1;                                                          \
        }                                                                       \
    } while (0)

#define CUDA_CHECK_PTR(expr)                                                    \
    do {                                                                        \
        cudaError_t _err = (expr);                                              \
        if (_err != cudaSuccess) {                                              \
            snprintf(g_last_error, sizeof(g_last_error),                        \
                     "CUDA error (%s:%d): %s", __FILE__, __LINE__,              \
                     cudaGetErrorString(_err));                                 \
            nh_cuda_destroy(h);                                                 \
            return nullptr;                                                     \
        }                                                                       \
    } while (0)

__device__ inline float weno5_left_f(float qm2, float qm1, float q0, float qp1, float qp2) {
    constexpr float eps = 1e-12f;
    const float p0 = (1.0f / 3.0f) * qm2 - (7.0f / 6.0f) * qm1 + (11.0f / 6.0f) * q0;
    const float p1 = -(1.0f / 6.0f) * qm1 + (5.0f / 6.0f) * q0 + (1.0f / 3.0f) * qp1;
    const float p2 = (1.0f / 3.0f) * q0 + (5.0f / 6.0f) * qp1 - (1.0f / 6.0f) * qp2;

    const float s0a = qm2 - 2.0f * qm1 + q0;
    const float s0b = qm2 - 4.0f * qm1 + 3.0f * q0;
    const float b0 = (13.0f / 12.0f) * s0a * s0a + 0.25f * s0b * s0b;

    const float s1a = qm1 - 2.0f * q0 + qp1;
    const float s1b = qm1 - qp1;
    const float b1 = (13.0f / 12.0f) * s1a * s1a + 0.25f * s1b * s1b;

    const float s2a = q0 - 2.0f * qp1 + qp2;
    const float s2b = 3.0f * q0 - 4.0f * qp1 + qp2;
    const float b2 = (13.0f / 12.0f) * s2a * s2a + 0.25f * s2b * s2b;

    const float d0 = eps + b0;
    const float d1 = eps + b1;
    const float d2 = eps + b2;
    const float a0 = 0.1f / (d0 * d0);
    const float a1 = 0.6f / (d1 * d1);
    const float a2 = 0.3f / (d2 * d2);
    const float inv_s = 1.0f / (a0 + a1 + a2);
    return (a0 * p0 + a1 * p1 + a2 * p2) * inv_s;
}

__device__ inline float up3_left_f(float qm1, float q0, float qp1) {
    return (1.0f / 3.0f) * qp1 + (5.0f / 6.0f) * q0 - (1.0f / 6.0f) * qm1;
}

__device__ inline float koren_slope_f(float dl, float dr) {
    if (dl * dr <= 0.0f) return 0.0f;
    const float c = 0.5f * (dl + dr);
    const float a = 2.0f * dl;
    const float b = 2.0f * dr;
    if (c > 0.0f) return fminf(c, fminf(a, b));
    return fmaxf(c, fmaxf(a, b));
}

__device__ inline double koren_slope(double dl, double dr) {
    // Symmetric Koren/MC TVD slope limiter for non-uniform grids.
    if (dl * dr <= 0.0) return 0.0;
    const double c = 0.5 * (dl + dr);
    const double a = 2.0 * dl;
    const double b = 2.0 * dr;
    if (c > 0.0) return fmin(c, fmin(a, b));
    return fmax(c, fmax(a, b));
}

__device__ inline double moisture_load_scalar(
    int32_t has_moist, size_t idx,
    const float* qv, const float* qc, const float* qr,
    const float* qi, const float* qs, const float* qg,
    double& qv_out, double& qhyd_out
) {
    if (!has_moist) {
        qv_out = 0.0;
        qhyd_out = 0.0;
        return 1.0;
    }
    qv_out = static_cast<double>(qv[idx]);
    qhyd_out = static_cast<double>(qc[idx]) + static_cast<double>(qr[idx])
             + static_cast<double>(qi[idx]) + static_cast<double>(qs[idx])
             + static_cast<double>(qg[idx]);
    return 1.0 + EPS_V * qv_out - qhyd_out;
}

__device__ inline float moisture_load_scalar_f(
    int32_t has_moist, size_t idx,
    const float* qv, const float* qc, const float* qr,
    const float* qi, const float* qs, const float* qg,
    float& qv_out, float& qhyd_out
) {
    if (!has_moist) {
        qv_out = 0.0f;
        qhyd_out = 0.0f;
        return 1.0f;
    }
    qv_out = qv[idx];
    qhyd_out = qc[idx] + qr[idx] + qi[idx] + qs[idx] + qg[idx];
    return 1.0f + EPS_V_F * qv_out - qhyd_out;
}

__device__ inline double interp_pi_at_z_f(
    const NhGridConfig& cfg,
    const DeviceStaticGrid& g,
    const float* pi_p,
    const float* theta_p,
    const float* qv, const float* qc, const float* qr,
    const float* qi, const float* qs, const float* qg,
    int32_t k_hint,
    int32_t j_clamped,
    int32_t i_clamped,
    double hs_col,
    double z_target
) {
    const int32_t nz = cfg.nz;
    const int32_t ny = cfg.ny;
    const int32_t nx = cfg.nx;
    int32_t m = k_hint;
    if (m < 0) m = 0;
    if (m > nz - 2) m = nz - 2;

    auto z_col = [&](int32_t kk) -> double {
        return g.zeta_s[kk] + g.b_s[kk] * hs_col;
    };

    auto hydro_slope = [&](size_t s_idx, float z_lvl) -> double {
        float qvv, qhv;
        const float thb = eval_theta_bar_f32(z_lvl, static_cast<float>(cfg.theta_sfc_k), static_cast<float>(cfg.n_ref_s));
        const float vfac = moisture_load_scalar_f(cfg.has_moist, s_idx, qv, qc, qr, qi, qs, qg, qvv, qhv);
        const float th_tot = (thb + theta_p[s_idx]) * vfac;
        const float dth_v = theta_p[s_idx] * vfac + thb * (vfac - 1.0f);
        return static_cast<double>((GRAVITY_F * dth_v) / (CP_D_F * thb * fmaxf(th_tot, 50.0f)));
    };

    if (z_target <= z_col(0)) {
        const size_t idx0 = static_cast<size_t>(j_clamped) * nx + i_clamped;
        const size_t idx1 = static_cast<size_t>(ny) * nx + idx0;
        const double z0 = z_col(0);
        const double z1 = z_col(1);
        const double s0 = hydro_slope(idx0, static_cast<float>(z0));
        const double s1 = hydro_slope(idx1, static_cast<float>(z1));
        const double d2pi = fmax(-1.5e-9, fmin(1.5e-9, (s1 - s0) / fmax(z1 - z0, 1.0)));
        const double dz = z_target - z0;
        return static_cast<double>(pi_p[idx0]) + s0 * dz + 0.5 * d2pi * dz * dz;
    }
    const double z_top_s = z_col(nz - 1);
    if (z_target >= z_top_s) {
        const size_t idx_t = static_cast<size_t>(nz - 1) * ny * nx + static_cast<size_t>(j_clamped) * nx + i_clamped;
        const size_t idx_m = static_cast<size_t>(nz - 2) * ny * nx + static_cast<size_t>(j_clamped) * nx + i_clamped;
        const double z_m = z_col(nz - 2);
        const double s_t = hydro_slope(idx_t, static_cast<float>(z_top_s));
        const double s_m = hydro_slope(idx_m, static_cast<float>(z_m));
        const double d2pi = fmax(-1.5e-9, fmin(1.5e-9, (s_t - s_m) / fmax(z_top_s - z_m, 1.0)));
        const double dz = z_target - z_top_s;
        return static_cast<double>(pi_p[idx_t]) + s_t * dz + 0.5 * d2pi * dz * dz;
    }

    while (m > 0 && z_target < z_col(m)) --m;
    while (m < nz - 2 && z_target > z_col(m + 1)) ++m;

    const double z0 = z_col(m);
    const double z1 = z_col(m + 1);
    const size_t i0 = static_cast<size_t>(m) * ny * nx + static_cast<size_t>(j_clamped) * nx + i_clamped;
    const size_t i1 = static_cast<size_t>(m + 1) * ny * nx + static_cast<size_t>(j_clamped) * nx + i_clamped;
    const double w = (z_target - z0) / fmax(z1 - z0, 1e-6);
    return (1.0 - w) * static_cast<double>(pi_p[i0]) + w * static_cast<double>(pi_p[i1]);
}

__device__ inline float sponge_tau_f(float z, float z_top, float depth, float tau_max) {
    if (depth <= 0.0f || tau_max <= 0.0f) return 0.0f;
    const float z_start = z_top - depth;
    if (z <= z_start) return 0.0f;
    const float s = sinf(0.5f * PI_CONST_F * (z - z_start) / depth);
    return tau_max * s * s;
}

__device__ inline double interp_pi_at_z(
    const NhGridConfig& cfg,
    const DeviceStaticGrid& g,
    const float* pi_p,
    const float* theta_p,
    const float* qv, const float* qc, const float* qr,
    const float* qi, const float* qs, const float* qg,
    int32_t k_hint,
    int32_t j_clamped,
    int32_t i_clamped,
    double hs_col,
    double z_target
) {
    const int32_t nz = cfg.nz;
    const int32_t ny = cfg.ny;
    const int32_t nx = cfg.nx;
    int32_t m = k_hint;
    if (m < 0) m = 0;
    if (m > nz - 2) m = nz - 2;

    auto z_col = [&](int32_t kk) -> double {
        return g.zeta_s[kk] + g.b_s[kk] * hs_col;
    };

    auto hydro_slope = [&](size_t s_idx, double z_lvl) -> double {
        double qvv, qhv;
        const double thb = eval_theta_bar(z_lvl, cfg.theta_sfc_k, cfg.n_ref_s);
        const double vfac = moisture_load_scalar(cfg.has_moist, s_idx, qv, qc, qr, qi, qs, qg, qvv, qhv);
        const double th_tot = (thb + static_cast<double>(theta_p[s_idx])) * vfac;
        return (GRAVITY * (th_tot - thb)) / (CP_D * thb * fmax(th_tot, 50.0));
    };

    if (z_target <= z_col(0)) {
        const size_t idx0 = static_cast<size_t>(j_clamped) * nx + i_clamped;
        const double z0 = z_col(0);
        const double dpi0 = hydro_slope(idx0, z0);
        return static_cast<double>(pi_p[idx0]) + dpi0 * (z_target - z0);
    }
    const double z_top_s = z_col(nz - 1);
    if (z_target >= z_top_s) {
        const size_t idx_t = static_cast<size_t>(nz - 1) * ny * nx + static_cast<size_t>(j_clamped) * nx + i_clamped;
        const double dpi_t = hydro_slope(idx_t, z_top_s);
        return static_cast<double>(pi_p[idx_t]) + dpi_t * (z_target - z_top_s);
    }

    while (m > 0 && z_target < z_col(m)) --m;
    while (m < nz - 2 && z_target > z_col(m + 1)) ++m;

    const double z0 = z_col(m);
    const double z1 = z_col(m + 1);
    const size_t i0 = static_cast<size_t>(m) * ny * nx + static_cast<size_t>(j_clamped) * nx + i_clamped;
    const size_t i1 = static_cast<size_t>(m + 1) * ny * nx + static_cast<size_t>(j_clamped) * nx + i_clamped;
    const double w = (z_target - z0) / fmax(z1 - z0, 1e-6);
    return (1.0 - w) * static_cast<double>(pi_p[i0]) + w * static_cast<double>(pi_p[i1]);
}

__device__ inline float davies_weight_f(int32_t dist, int32_t rim_width, float tau_max) {
    if (rim_width <= 0 || dist >= rim_width) return 0.0f;
    const float f = static_cast<float>(rim_width - dist) / static_cast<float>(rim_width);
    return tau_max * f * f;
}

__device__ inline double sponge_tau(double z, double z_top, double depth, double tau_max) {
    if (depth <= 0.0 || tau_max <= 0.0) return 0.0;
    const double z_start = z_top - depth;
    if (z <= z_start) return 0.0;
    const double s = sin(0.5 * 3.14159265358979323846 * (z - z_start) / depth);
    return tau_max * s * s;
}

__device__ inline float horiz_face_val_x_f(
    const float* f, int32_t k, int32_t j, int32_t i_face, float dir_u,
    int32_t nz, int32_t ny, int32_t nx, int32_t bc_x, int32_t bc_y, int32_t adv_order
) {
    // Reconstruct scalar at x-face i_face - 1/2 (between i_face-1 and i_face).
    if (dir_u >= 0.0f) {
        if (adv_order >= 5) {
            return weno5_left_f(
                read_s_f(f, k, j, i_face - 3, nz, ny, nx, bc_x, bc_y),
                read_s_f(f, k, j, i_face - 2, nz, ny, nx, bc_x, bc_y),
                read_s_f(f, k, j, i_face - 1, nz, ny, nx, bc_x, bc_y),
                read_s_f(f, k, j, i_face,     nz, ny, nx, bc_x, bc_y),
                read_s_f(f, k, j, i_face + 1, nz, ny, nx, bc_x, bc_y)
            );
        }
        return up3_left_f(
            read_s_f(f, k, j, i_face - 2, nz, ny, nx, bc_x, bc_y),
            read_s_f(f, k, j, i_face - 1, nz, ny, nx, bc_x, bc_y),
            read_s_f(f, k, j, i_face,     nz, ny, nx, bc_x, bc_y)
        );
    } else {
        if (adv_order >= 5) {
            return weno5_left_f(
                read_s_f(f, k, j, i_face + 2, nz, ny, nx, bc_x, bc_y),
                read_s_f(f, k, j, i_face + 1, nz, ny, nx, bc_x, bc_y),
                read_s_f(f, k, j, i_face,     nz, ny, nx, bc_x, bc_y),
                read_s_f(f, k, j, i_face - 1, nz, ny, nx, bc_x, bc_y),
                read_s_f(f, k, j, i_face - 2, nz, ny, nx, bc_x, bc_y)
            );
        }
        return up3_left_f(
            read_s_f(f, k, j, i_face + 1, nz, ny, nx, bc_x, bc_y),
            read_s_f(f, k, j, i_face,     nz, ny, nx, bc_x, bc_y),
            read_s_f(f, k, j, i_face - 1, nz, ny, nx, bc_x, bc_y)
        );
    }
}

__device__ inline float horiz_face_val_y_f(
    const float* f, int32_t k, int32_t j_face, int32_t i, float dir_v,
    int32_t nz, int32_t ny, int32_t nx, int32_t bc_x, int32_t bc_y, int32_t adv_order
) {
    if (dir_v >= 0.0f) {
        if (adv_order >= 5) {
            return weno5_left_f(
                read_s_f(f, k, j_face - 3, i, nz, ny, nx, bc_x, bc_y),
                read_s_f(f, k, j_face - 2, i, nz, ny, nx, bc_x, bc_y),
                read_s_f(f, k, j_face - 1, i, nz, ny, nx, bc_x, bc_y),
                read_s_f(f, k, j_face,     i, nz, ny, nx, bc_x, bc_y),
                read_s_f(f, k, j_face + 1, i, nz, ny, nx, bc_x, bc_y)
            );
        }
        return up3_left_f(
            read_s_f(f, k, j_face - 2, i, nz, ny, nx, bc_x, bc_y),
            read_s_f(f, k, j_face - 1, i, nz, ny, nx, bc_x, bc_y),
            read_s_f(f, k, j_face,     i, nz, ny, nx, bc_x, bc_y)
        );
    } else {
        if (adv_order >= 5) {
            return weno5_left_f(
                read_s_f(f, k, j_face + 2, i, nz, ny, nx, bc_x, bc_y),
                read_s_f(f, k, j_face + 1, i, nz, ny, nx, bc_x, bc_y),
                read_s_f(f, k, j_face,     i, nz, ny, nx, bc_x, bc_y),
                read_s_f(f, k, j_face - 1, i, nz, ny, nx, bc_x, bc_y),
                read_s_f(f, k, j_face - 2, i, nz, ny, nx, bc_x, bc_y)
            );
        }
        return up3_left_f(
            read_s_f(f, k, j_face + 1, i, nz, ny, nx, bc_x, bc_y),
            read_s_f(f, k, j_face,     i, nz, ny, nx, bc_x, bc_y),
            read_s_f(f, k, j_face - 1, i, nz, ny, nx, bc_x, bc_y)
        );
    }
}

__device__ inline float vert_tvd_face_val_f(
    const float* f, int32_t k_face, int32_t j, int32_t i, float dir_w,
    const double* dzeta, int32_t nz, int32_t ny, int32_t nx, int32_t bc_x, int32_t bc_y
) {
    // Reconstruct scalar onto interface k_face (between k_face-1 and k_face).
    if (k_face <= 0) return read_s_f(f, 0, j, i, nz, ny, nx, bc_x, bc_y);
    if (k_face >= nz) return read_s_f(f, nz - 1, j, i, nz, ny, nx, bc_x, bc_y);
    if (dir_w >= 0.0f) {
        const int32_t km = k_face - 1;
        const float vc = read_s_f(f, km, j, i, nz, ny, nx, bc_x, bc_y);
        if (km == 0) return vc;
        const float vm = read_s_f(f, km - 1, j, i, nz, ny, nx, bc_x, bc_y);
        const float vp = read_s_f(f, km + 1, j, i, nz, ny, nx, bc_x, bc_y);
        const float dz_km = static_cast<float>(dzeta[km]);
        const float dz_prev = static_cast<float>(dzeta[km - 1]);
        const float dz_next = static_cast<float>(dzeta[km + 1]);
        const float dl = (vc - vm) / (0.5f * (dz_km + dz_prev));
        const float dr = (vp - vc) / (0.5f * (dz_next + dz_km));
        return vc + 0.5f * dz_km * koren_slope_f(dl, dr);
    } else {
        const int32_t kp = k_face;
        const float vc = read_s_f(f, kp, j, i, nz, ny, nx, bc_x, bc_y);
        if (kp == nz - 1) return vc;
        const float vm = read_s_f(f, kp - 1, j, i, nz, ny, nx, bc_x, bc_y);
        const float vp = read_s_f(f, kp + 1, j, i, nz, ny, nx, bc_x, bc_y);
        const float dz_kp = static_cast<float>(dzeta[kp]);
        const float dz_prev = static_cast<float>(dzeta[kp - 1]);
        const float dz_next = static_cast<float>(dzeta[kp + 1]);
        const float dl = (vc - vm) / (0.5f * (dz_kp + dz_prev));
        const float dr = (vp - vc) / (0.5f * (dz_next + dz_kp));
        return vc - 0.5f * dz_kp * koren_slope_f(dl, dr);
    }
}

__device__ inline void compute_contravariant_at_scalar_col_f(
    const NhGridConfig& cfg,
    const DeviceStaticGrid& g,
    const float* u,
    const float* v,
    const float* w,
    int32_t k_iface,
    int32_t j,
    int32_t i,
    float& gdotzeta_out
) {
    if (k_iface <= 0 || k_iface >= cfg.nz) {
        gdotzeta_out = 0.0f;
        return;
    }
    const int32_t nz = cfg.nz, ny = cfg.ny, nx = cfg.nx;
    const float msf = read_2d_f(g.msf, j, i, ny, nx, cfg.bc_x, cfg.bc_y);
    const float dx_f = static_cast<float>(cfg.dx_m);
    const float dy_f = static_cast<float>(cfg.dy_m);
    const float dh_dx = (read_2d_f(g.hs, j, i + 1, ny, nx, cfg.bc_x, cfg.bc_y) -
                         read_2d_f(g.hs, j, i - 1, ny, nx, cfg.bc_x, cfg.bc_y)) / (2.0f * dx_f);
    const float dh_dy = (read_2d_f(g.hs, j + 1, i, ny, nx, cfg.bc_x, cfg.bc_y) -
                         read_2d_f(g.hs, j - 1, i, ny, nx, cfg.bc_x, cfg.bc_y)) / (2.0f * dy_f);

    const float ds_k = static_cast<float>(g.dzeta[k_iface]);
    const float ds_km = static_cast<float>(g.dzeta[k_iface - 1]);
    const float wt = ds_km / (ds_k + ds_km);

    const float u_lo = 0.5f * (read_u_f(u, k_iface - 1, j, i, nz, ny, nx, cfg.bc_x, cfg.bc_y) +
                               read_u_f(u, k_iface - 1, j, i + 1, nz, ny, nx, cfg.bc_x, cfg.bc_y));
    const float u_hi = 0.5f * (read_u_f(u, k_iface, j, i, nz, ny, nx, cfg.bc_x, cfg.bc_y) +
                               read_u_f(u, k_iface, j, i + 1, nz, ny, nx, cfg.bc_x, cfg.bc_y));
    const float v_lo = 0.5f * (read_v_f(v, k_iface - 1, j, i, nz, ny, nx, cfg.bc_x, cfg.bc_y) +
                               read_v_f(v, k_iface - 1, j + 1, i, nz, ny, nx, cfg.bc_x, cfg.bc_y));
    const float v_hi = 0.5f * (read_v_f(v, k_iface, j, i, nz, ny, nx, cfg.bc_x, cfg.bc_y) +
                               read_v_f(v, k_iface, j + 1, i, nz, ny, nx, cfg.bc_x, cfg.bc_y));

    const float u_w = (1.0f - wt) * u_lo + wt * u_hi;
    const float v_w = (1.0f - wt) * v_lo + wt * v_hi;
    const float w_val = read_w_f(w, k_iface, j, i, nz, ny, nx, cfg.bc_x, cfg.bc_y);
    const float bw = static_cast<float>(g.b_w[k_iface]);
    const float my = cfg.is_spherical ? 1.0f : msf;
    gdotzeta_out = w_val - bw * (msf * u_w * dh_dx + my * v_w * dh_dy);
}

// ---------------------------------------------------------------------------
// Kernel 1: Slow tendencies (advection, Coriolis/LCC curvature, explicit diff,
// Davies relaxation, IAU)
// ---------------------------------------------------------------------------
__global__ void nh_slow_scalar_kernel(
    NhGridConfig cfg,
    DeviceStaticGrid g,
    float tau_stage,
    const float* u,
    const float* v,
    const float* w,
    const float* thp,
    const float* pip,
    const float* lbc_thp,
    const float* lbc_pip,
    const float* iau_thp,
    const float* iau_pip,
    float* rth,
    float* rpi
) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    const int32_t k = blockIdx.z;
    if (k >= cfg.nz || j >= cfg.ny || i >= cfg.nx) return;

    const int32_t nz = cfg.nz, ny = cfg.ny, nx = cfg.nx;
    const size_t idx = static_cast<size_t>(k) * ny * nx + static_cast<size_t>(j) * nx + i;

    const float dx_f = static_cast<float>(cfg.dx_m);
    const float dy_f = static_cast<float>(cfg.dy_m);
    const float dz_k = static_cast<float>(g.dzeta[k]);
    const float msf = static_cast<float>(g.msf[j * nx + i]);
    const float my = cfg.is_spherical ? 1.0f : msf;
    const float hs = static_cast<float>(g.hs[j * nx + i]);
    const float g_s = 1.0f + static_cast<float>(g.db_s[k]) * hs;

    const float uc = 0.5f * (read_u_f(u, k, j, i, nz, ny, nx, cfg.bc_x, cfg.bc_y) +
                             read_u_f(u, k, j, i + 1, nz, ny, nx, cfg.bc_x, cfg.bc_y));
    const float vc = 0.5f * (read_v_f(v, k, j, i, nz, ny, nx, cfg.bc_x, cfg.bc_y) +
                             read_v_f(v, k, j + 1, i, nz, ny, nx, cfg.bc_x, cfg.bc_y));
    float gz_lo = 0.0f, gz_hi = 0.0f;
    compute_contravariant_at_scalar_col_f(cfg, g, u, v, w, k, j, i, gz_lo);
    compute_contravariant_at_scalar_col_f(cfg, g, u, v, w, k + 1, j, i, gz_hi);
    const float zeta_dot = 0.5f * (gz_lo + gz_hi) / fmaxf(g_s, 1e-4f);
    // Wicker & Skamarock (2020) IEVA split: cap explicit vertical Courant number at 0.45
    const float zd_cap = 0.45f * dz_k / fmaxf(tau_stage, 1e-6f);
    const float zd_exp = fmaxf(-zd_cap, fminf(zd_cap, zeta_dot));

    // Advect theta'
    const float th_xp = horiz_face_val_x_f(thp, k, j, i + 1, uc, nz, ny, nx, cfg.bc_x, cfg.bc_y, cfg.adv_order);
    const float th_xm = horiz_face_val_x_f(thp, k, j, i,     uc, nz, ny, nx, cfg.bc_x, cfg.bc_y, cfg.adv_order);
    const float th_yp = horiz_face_val_y_f(thp, k, j + 1, i, vc, nz, ny, nx, cfg.bc_x, cfg.bc_y, cfg.adv_order);
    const float th_ym = horiz_face_val_y_f(thp, k, j,     i, vc, nz, ny, nx, cfg.bc_x, cfg.bc_y, cfg.adv_order);
    const float th_zp = vert_tvd_face_val_f(thp, k + 1, j, i, zd_exp, g.dzeta, nz, ny, nx, cfg.bc_x, cfg.bc_y);
    const float th_zm = vert_tvd_face_val_f(thp, k,     j, i, zd_exp, g.dzeta, nz, ny, nx, cfg.bc_x, cfg.bc_y);

    float tend_th = -(msf * uc * (th_xp - th_xm) / dx_f + my * vc * (th_yp - th_ym) / dy_f)
                    - zd_exp * (th_zp - th_zm) / dz_k;

    // Advect pi'
    const float pi_xp = horiz_face_val_x_f(pip, k, j, i + 1, uc, nz, ny, nx, cfg.bc_x, cfg.bc_y, cfg.adv_order);
    const float pi_xm = horiz_face_val_x_f(pip, k, j, i,     uc, nz, ny, nx, cfg.bc_x, cfg.bc_y, cfg.adv_order);
    const float pi_yp = horiz_face_val_y_f(pip, k, j + 1, i, vc, nz, ny, nx, cfg.bc_x, cfg.bc_y, cfg.adv_order);
    const float pi_ym = horiz_face_val_y_f(pip, k, j,     i, vc, nz, ny, nx, cfg.bc_x, cfg.bc_y, cfg.adv_order);
    const float pi_zp = vert_tvd_face_val_f(pip, k + 1, j, i, zd_exp, g.dzeta, nz, ny, nx, cfg.bc_x, cfg.bc_y);
    const float pi_zm = vert_tvd_face_val_f(pip, k,     j, i, zd_exp, g.dzeta, nz, ny, nx, cfg.bc_x, cfg.bc_y);

    float tend_pi = -(msf * uc * (pi_xp - pi_xm) / dx_f + my * vc * (pi_yp - pi_ym) / dy_f)
                    - zd_exp * (pi_zp - pi_zm) / dz_k;

    // Explicit 3D Laplacian diffusion (used cleanly for Straka K=75 m^2/s)
    const float th0 = thp[idx];
    const float pi0 = pip[idx];
    if (cfg.kh_const > 0.0) {
        const float th_e = read_s_f(thp, k, j, i + 1, nz, ny, nx, cfg.bc_x, cfg.bc_y);
        const float th_w = read_s_f(thp, k, j, i - 1, nz, ny, nx, cfg.bc_x, cfg.bc_y);
        const float th_n = read_s_f(thp, k, j + 1, i, nz, ny, nx, cfg.bc_x, cfg.bc_y);
        const float th_s = read_s_f(thp, k, j - 1, i, nz, ny, nx, cfg.bc_x, cfg.bc_y);
        const float lap_h = (msf * msf) * (th_e - 2.0f * th0 + th_w) / (dx_f * dx_f)
                          + (my  * my ) * (th_n - 2.0f * th0 + th_s) / (dy_f * dy_f);
        const float dz_c = g_s * dz_k;
        float lap_v = 0.0f;
        if (k < nz - 1) {
            const float dz_p = 0.5f * (dz_c + (1.0f + static_cast<float>(g.db_s[k + 1]) * hs) * static_cast<float>(g.dzeta[k + 1]));
            lap_v += (read_s_f(thp, k + 1, j, i, nz, ny, nx, cfg.bc_x, cfg.bc_y) - th0) / (dz_p * dz_c);
        }
        if (k > 0) {
            const float dz_m = 0.5f * (dz_c + (1.0f + static_cast<float>(g.db_s[k - 1]) * hs) * static_cast<float>(g.dzeta[k - 1]));
            lap_v -= (th0 - read_s_f(thp, k - 1, j, i, nz, ny, nx, cfg.bc_x, cfg.bc_y)) / (dz_m * dz_c);
        }
        tend_th += static_cast<float>(cfg.kh_const) * (lap_h + lap_v);
    }

    // Optional dimensionless 4th-order horizontal hyperdiffusion for 2*dx steep-orography noise control
    if (cfg.hyperdiff_coeff > 0.0) {
        const bool ok_x = (cfg.bc_x == 1) || (i >= 2 && i <= nx - 3);
        const bool ok_y = (cfg.bc_y == 1) || (j >= 2 && j <= ny - 3);
        float d4 = 0.0f;
        if (ok_x) {
            d4 += read_s_f(thp, k, j, i - 2, nz, ny, nx, cfg.bc_x, cfg.bc_y)
                - 4.0f * read_s_f(thp, k, j, i - 1, nz, ny, nx, cfg.bc_x, cfg.bc_y)
                + 6.0f * th0
                - 4.0f * read_s_f(thp, k, j, i + 1, nz, ny, nx, cfg.bc_x, cfg.bc_y)
                + read_s_f(thp, k, j, i + 2, nz, ny, nx, cfg.bc_x, cfg.bc_y);
        }
        if (ok_y) {
            d4 += read_s_f(thp, k, j - 2, i, nz, ny, nx, cfg.bc_x, cfg.bc_y)
                - 4.0f * read_s_f(thp, k, j - 1, i, nz, ny, nx, cfg.bc_x, cfg.bc_y)
                + 6.0f * th0
                - 4.0f * read_s_f(thp, k, j + 1, i, nz, ny, nx, cfg.bc_x, cfg.bc_y)
                + read_s_f(thp, k, j + 2, i, nz, ny, nx, cfg.bc_x, cfg.bc_y);
        }
        tend_th -= (static_cast<float>(cfg.hyperdiff_coeff) / fmaxf(tau_stage, 1e-6f)) * d4;
    }

    // Optional Davies lateral boundary relaxation
    if ((cfg.bc_x == 2 || cfg.bc_y == 2) && lbc_thp != nullptr && cfg.lbc_tau_max_s > 0.0) {
        const float tau_max = static_cast<float>(cfg.lbc_tau_max_s);
        float tau = 0.0f;
        if (cfg.bc_x == 2) {
            if (cfg.lbc_edges & 1) tau = fmaxf(tau, davies_weight_f(i, cfg.lbc_rim_width, tau_max));
            if (cfg.lbc_edges & 2) tau = fmaxf(tau, davies_weight_f(nx - 1 - i, cfg.lbc_rim_width, tau_max));
        }
        if (cfg.bc_y == 2) {
            if (cfg.lbc_edges & 4) tau = fmaxf(tau, davies_weight_f(j, cfg.lbc_rim_width, tau_max));
            if (cfg.lbc_edges & 8) tau = fmaxf(tau, davies_weight_f(ny - 1 - j, cfg.lbc_rim_width, tau_max));
        }
        if (tau > 0.0f) {
            tend_th -= tau * (th0 - lbc_thp[idx]);
            if (lbc_pip != nullptr) {
                tend_pi -= tau * (pi0 - lbc_pip[idx]);
            }
        }
    }

    if (cfg.has_iau && iau_thp != nullptr) tend_th += iau_thp[idx];
    if (cfg.has_iau && iau_pip != nullptr) tend_pi += iau_pip[idx];

    rth[idx] = tend_th;
    rpi[idx] = tend_pi;
}

__global__ void nh_slow_u_kernel(
    NhGridConfig cfg,
    DeviceStaticGrid g,
    float tau_stage,
    const float* u,
    const float* v,
    const float* w,
    const float* lbc_u,
    const float* iau_u,
    float* ru
) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    const int32_t k = blockIdx.z;
    if (k >= cfg.nz || j >= cfg.ny || i > cfg.nx) return;

    const int32_t nz = cfg.nz, ny = cfg.ny, nx = cfg.nx;
    const size_t idx = static_cast<size_t>(k) * ny * (nx + 1) + static_cast<size_t>(j) * (nx + 1) + i;

    if (cfg.bc_x == 0 && (i == 0 || i == nx)) {
        ru[idx] = 0.0f;
        return;
    }

    const int32_t il = (cfg.bc_x == 1) ? wrap_idx(i - 1, nx) : ((i <= 0) ? 0 : i - 1);
    const int32_t ir = (cfg.bc_x == 1) ? wrap_idx(i, nx) : ((i >= nx) ? nx - 1 : i);
    const float dx_f = static_cast<float>(cfg.dx_m);
    const float dy_f = static_cast<float>(cfg.dy_m);
    const float msf_l0 = static_cast<float>(g.msf[j * nx + il]);
    const float msf_r0 = static_cast<float>(g.msf[j * nx + ir]);
    const float msf = 0.5f * (msf_l0 + msf_r0);
    const float my = cfg.is_spherical ? 1.0f : msf;
    const float hs_u = 0.5f * (static_cast<float>(g.hs[j * nx + il]) + static_cast<float>(g.hs[j * nx + ir]));
    const float g_u = 1.0f + static_cast<float>(g.db_s[k]) * hs_u;

    const float uc = read_u_f(u, k, j, i, nz, ny, nx, cfg.bc_x, cfg.bc_y);
    const float vc = 0.25f * (
        read_v_f(v, k, j,     il, nz, ny, nx, cfg.bc_x, cfg.bc_y) +
        read_v_f(v, k, j + 1, il, nz, ny, nx, cfg.bc_x, cfg.bc_y) +
        read_v_f(v, k, j,     ir, nz, ny, nx, cfg.bc_x, cfg.bc_y) +
        read_v_f(v, k, j + 1, ir, nz, ny, nx, cfg.bc_x, cfg.bc_y)
    );
    float gz_ll = 0.0f, gz_lh = 0.0f, gz_rl = 0.0f, gz_rh = 0.0f;
    compute_contravariant_at_scalar_col_f(cfg, g, u, v, w, k,     j, il, gz_ll);
    compute_contravariant_at_scalar_col_f(cfg, g, u, v, w, k + 1, j, il, gz_lh);
    compute_contravariant_at_scalar_col_f(cfg, g, u, v, w, k,     j, ir, gz_rl);
    compute_contravariant_at_scalar_col_f(cfg, g, u, v, w, k + 1, j, ir, gz_rh);
    const float zeta_dot = 0.25f * (gz_ll + gz_lh + gz_rl + gz_rh) / fmaxf(g_u, 1e-4f);

    // Centered/UP3 momentum advection on u-points
    auto u_get = [&](int32_t kk, int32_t jj, int32_t ii) -> float {
        return read_u_f(u, kk, jj, ii, nz, ny, nx, cfg.bc_x, cfg.bc_y);
    };
    float dudx = 0.0f;
    if (uc >= 0.0f) {
        dudx = (up3_left_f(u_get(k, j, i - 1), uc, u_get(k, j, i + 1)) -
                up3_left_f(u_get(k, j, i - 2), u_get(k, j, i - 1), uc)) / dx_f;
    } else {
        dudx = (up3_left_f(u_get(k, j, i + 2), u_get(k, j, i + 1), uc) -
                up3_left_f(u_get(k, j, i + 1), uc, u_get(k, j, i - 1))) / dx_f;
    }
    float dudy = 0.0f;
    if (vc >= 0.0f) {
        dudy = (up3_left_f(u_get(k, j - 1, i), uc, u_get(k, j + 1, i)) -
                up3_left_f(u_get(k, j - 2, i), u_get(k, j - 1, i), uc)) / dy_f;
    } else {
        dudy = (up3_left_f(u_get(k, j + 2, i), u_get(k, j + 1, i), uc) -
                up3_left_f(u_get(k, j + 1, i), uc, u_get(k, j - 1, i))) / dy_f;
    }
    const float dz_k = static_cast<float>(g.dzeta[k]);
    const float zd_cap = 0.45f * dz_k / fmaxf(tau_stage, 1e-6f);
    const float zd_exp = fmaxf(-zd_cap, fminf(zd_cap, zeta_dot));
    float dudz = 0.0f;
    if (zd_exp >= 0.0f && k > 0) {
        dudz = (uc - u_get(k - 1, j, i)) / (0.5f * (dz_k + static_cast<float>(g.dzeta[k - 1])));
    } else if (zd_exp < 0.0f && k < nz - 1) {
        dudz = (u_get(k + 1, j, i) - uc) / (0.5f * (static_cast<float>(g.dzeta[k + 1]) + dz_k));
    }

    float tend = -(msf * uc * dudx + my * vc * dudy) - zd_exp * dudz;

    if (cfg.use_coriolis) {
        const float f_u = 0.5f * (static_cast<float>(g.fcor[j * nx + il]) + static_cast<float>(g.fcor[j * nx + ir]));
        if (cfg.is_spherical) {
            const float tan_phi = fminf(20.0f, fmaxf(-20.0f, (f_u * msf) / (2.0f * OMEGA_F)));
            tend += (f_u + (uc * tan_phi) / EARTH_RADIUS_F) * vc;
        } else {
            const float dm_dy = (read_2d_f(g.msf, j + 1, il, ny, nx, cfg.bc_x, cfg.bc_y) -
                                 read_2d_f(g.msf, j - 1, il, ny, nx, cfg.bc_x, cfg.bc_y) +
                                 read_2d_f(g.msf, j + 1, ir, ny, nx, cfg.bc_x, cfg.bc_y) -
                                 read_2d_f(g.msf, j - 1, ir, ny, nx, cfg.bc_x, cfg.bc_y)) / (4.0f * dy_f);
            const float dm_dx = (msf_r0 - msf_l0) / dx_f;
            tend += (f_u + uc * dm_dy - vc * dm_dx) * vc;
        }
    }

    if (cfg.km_const > 0.0) {
        const float lap_h = (msf * msf) * (u_get(k, j, i + 1) - 2.0f * uc + u_get(k, j, i - 1)) / (dx_f * dx_f)
                          + (my  * my ) * (u_get(k, j + 1, i) - 2.0f * uc + u_get(k, j - 1, i)) / (dy_f * dy_f);
        const float dz_c = g_u * dz_k;
        float lap_v = 0.0f;
        if (k < nz - 1) {
            const float dz_p = 0.5f * (dz_c + (1.0f + static_cast<float>(g.db_s[k + 1]) * hs_u) * static_cast<float>(g.dzeta[k + 1]));
            lap_v += (u_get(k + 1, j, i) - uc) / (dz_p * dz_c);
        }
        if (k > 0) {
            const float dz_m = 0.5f * (dz_c + (1.0f + static_cast<float>(g.db_s[k - 1]) * hs_u) * static_cast<float>(g.dzeta[k - 1]));
            lap_v -= (uc - u_get(k - 1, j, i)) / (dz_m * dz_c);
        }
        tend += static_cast<float>(cfg.km_const) * (lap_h + lap_v);
    }

    if (cfg.hyperdiff_coeff > 0.0) {
        const bool ok_x = (cfg.bc_x == 1) || (i >= 2 && i <= nx - 2);
        const bool ok_y = (cfg.bc_y == 1) || (j >= 2 && j <= ny - 3);
        float d4 = 0.0f;
        if (ok_x) {
            d4 += u_get(k, j, i - 2) - 4.0f * u_get(k, j, i - 1) + 6.0f * uc
                - 4.0f * u_get(k, j, i + 1) + u_get(k, j, i + 2);
        }
        if (ok_y) {
            d4 += u_get(k, j - 2, i) - 4.0f * u_get(k, j - 1, i) + 6.0f * uc
                - 4.0f * u_get(k, j + 1, i) + u_get(k, j + 2, i);
        }
        tend -= (static_cast<float>(cfg.hyperdiff_coeff) / fmaxf(tau_stage, 1e-6f)) * d4;
    }

    if ((cfg.bc_x == 2 || cfg.bc_y == 2) && lbc_u != nullptr && cfg.lbc_tau_max_s > 0.0) {
        const float tau_max = static_cast<float>(cfg.lbc_tau_max_s);
        float tau = 0.0f;
        if (cfg.bc_x == 2) {
            if (cfg.lbc_edges & 1) tau = fmaxf(tau, davies_weight_f(i, cfg.lbc_rim_width, tau_max));
            if (cfg.lbc_edges & 2) tau = fmaxf(tau, davies_weight_f(nx - i, cfg.lbc_rim_width, tau_max));
        }
        if (cfg.bc_y == 2) {
            if (cfg.lbc_edges & 4) tau = fmaxf(tau, davies_weight_f(j, cfg.lbc_rim_width, tau_max));
            if (cfg.lbc_edges & 8) tau = fmaxf(tau, davies_weight_f(ny - 1 - j, cfg.lbc_rim_width, tau_max));
        }
        if (tau > 0.0f) {
            tend -= tau * (uc - lbc_u[idx]);
        }
    }

    if (cfg.has_iau && iau_u != nullptr) tend += iau_u[idx];
    ru[idx] = tend;
}

__global__ void nh_slow_v_kernel(
    NhGridConfig cfg,
    DeviceStaticGrid g,
    float tau_stage,
    const float* u,
    const float* v,
    const float* w,
    const float* lbc_v,
    const float* iau_v,
    float* rv
) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    const int32_t k = blockIdx.z;
    if (k >= cfg.nz || j > cfg.ny || i >= cfg.nx) return;

    const int32_t nz = cfg.nz, ny = cfg.ny, nx = cfg.nx;
    const size_t idx = static_cast<size_t>(k) * (ny + 1) * nx + static_cast<size_t>(j) * nx + i;

    if (cfg.bc_y == 0 && (j == 0 || j == ny)) {
        rv[idx] = 0.0f;
        return;
    }

    const int32_t jl = (cfg.bc_y == 1) ? wrap_idx(j - 1, ny) : ((j <= 0) ? 0 : j - 1);
    const int32_t jr = (cfg.bc_y == 1) ? wrap_idx(j, ny) : ((j >= ny) ? ny - 1 : j);
    const float dx_f = static_cast<float>(cfg.dx_m);
    const float dy_f = static_cast<float>(cfg.dy_m);
    const float msf_l0 = static_cast<float>(g.msf[jl * nx + i]);
    const float msf_r0 = static_cast<float>(g.msf[jr * nx + i]);
    const float msf = 0.5f * (msf_l0 + msf_r0);
    const float my = cfg.is_spherical ? 1.0f : msf;
    const float hs_v = 0.5f * (static_cast<float>(g.hs[jl * nx + i]) + static_cast<float>(g.hs[jr * nx + i]));
    const float g_v = 1.0f + static_cast<float>(g.db_s[k]) * hs_v;

    const float vc = read_v_f(v, k, j, i, nz, ny, nx, cfg.bc_x, cfg.bc_y);
    const float uc = 0.25f * (
        read_u_f(u, k, jl, i,     nz, ny, nx, cfg.bc_x, cfg.bc_y) +
        read_u_f(u, k, jl, i + 1, nz, ny, nx, cfg.bc_x, cfg.bc_y) +
        read_u_f(u, k, jr, i,     nz, ny, nx, cfg.bc_x, cfg.bc_y) +
        read_u_f(u, k, jr, i + 1, nz, ny, nx, cfg.bc_x, cfg.bc_y)
    );
    float gz_ll = 0.0f, gz_lh = 0.0f, gz_rl = 0.0f, gz_rh = 0.0f;
    compute_contravariant_at_scalar_col_f(cfg, g, u, v, w, k,     jl, i, gz_ll);
    compute_contravariant_at_scalar_col_f(cfg, g, u, v, w, k + 1, jl, i, gz_lh);
    compute_contravariant_at_scalar_col_f(cfg, g, u, v, w, k,     jr, i, gz_rl);
    compute_contravariant_at_scalar_col_f(cfg, g, u, v, w, k + 1, jr, i, gz_rh);
    const float zeta_dot = 0.25f * (gz_ll + gz_lh + gz_rl + gz_rh) / fmaxf(g_v, 1e-4f);

    auto v_get = [&](int32_t kk, int32_t jj, int32_t ii) -> float {
        return read_v_f(v, kk, jj, ii, nz, ny, nx, cfg.bc_x, cfg.bc_y);
    };
    float dvdx = 0.0f;
    if (uc >= 0.0f) {
        dvdx = (up3_left_f(v_get(k, j, i - 1), vc, v_get(k, j, i + 1)) -
                up3_left_f(v_get(k, j, i - 2), v_get(k, j, i - 1), vc)) / dx_f;
    } else {
        dvdx = (up3_left_f(v_get(k, j, i + 2), v_get(k, j, i + 1), vc) -
                up3_left_f(v_get(k, j, i + 1), vc, v_get(k, j, i - 1))) / dx_f;
    }
    float dvdy = 0.0f;
    if (vc >= 0.0f) {
        dvdy = (up3_left_f(v_get(k, j - 1, i), vc, v_get(k, j + 1, i)) -
                up3_left_f(v_get(k, j - 2, i), v_get(k, j - 1, i), vc)) / dy_f;
    } else {
        dvdy = (up3_left_f(v_get(k, j + 2, i), v_get(k, j + 1, i), vc) -
                up3_left_f(v_get(k, j + 1, i), vc, v_get(k, j - 1, i))) / dy_f;
    }
    const float dz_k = static_cast<float>(g.dzeta[k]);
    const float zd_cap = 0.45f * dz_k / fmaxf(tau_stage, 1e-6f);
    const float zd_exp = fmaxf(-zd_cap, fminf(zd_cap, zeta_dot));
    float dvdz = 0.0f;
    if (zd_exp >= 0.0f && k > 0) {
        dvdz = (vc - v_get(k - 1, j, i)) / (0.5f * (dz_k + static_cast<float>(g.dzeta[k - 1])));
    } else if (zd_exp < 0.0f && k < nz - 1) {
        dvdz = (v_get(k + 1, j, i) - vc) / (0.5f * (static_cast<float>(g.dzeta[k + 1]) + dz_k));
    }

    float tend = -(msf * uc * dvdx + my * vc * dvdy) - zd_exp * dvdz;

    if (cfg.use_coriolis) {
        const float f_v = 0.5f * (static_cast<float>(g.fcor[jl * nx + i]) + static_cast<float>(g.fcor[jr * nx + i]));
        if (cfg.is_spherical) {
            const float tan_phi = fminf(20.0f, fmaxf(-20.0f, (f_v * msf) / (2.0f * OMEGA_F)));
            tend -= (f_v + (uc * tan_phi) / EARTH_RADIUS_F) * uc;
        } else {
            const float dm_dx = (read_2d_f(g.msf, jl, i + 1, ny, nx, cfg.bc_x, cfg.bc_y) -
                                 read_2d_f(g.msf, jl, i - 1, ny, nx, cfg.bc_x, cfg.bc_y) +
                                 read_2d_f(g.msf, jr, i + 1, ny, nx, cfg.bc_x, cfg.bc_y) -
                                 read_2d_f(g.msf, jr, i - 1, ny, nx, cfg.bc_x, cfg.bc_y)) / (4.0f * dx_f);
            const float dm_dy = (msf_r0 - msf_l0) / dy_f;
            tend -= (f_v + uc * dm_dy - vc * dm_dx) * uc;
        }
    }

    if (cfg.km_const > 0.0) {
        const float lap_h = (msf * msf) * (v_get(k, j, i + 1) - 2.0f * vc + v_get(k, j, i - 1)) / (dx_f * dx_f)
                          + (my  * my ) * (v_get(k, j + 1, i) - 2.0f * vc + v_get(k, j - 1, i)) / (dy_f * dy_f);
        const float dz_c = g_v * dz_k;
        float lap_v = 0.0f;
        if (k < nz - 1) {
            const float dz_p = 0.5f * (dz_c + (1.0f + static_cast<float>(g.db_s[k + 1]) * hs_v) * static_cast<float>(g.dzeta[k + 1]));
            lap_v += (v_get(k + 1, j, i) - vc) / (dz_p * dz_c);
        }
        if (k > 0) {
            const float dz_m = 0.5f * (dz_c + (1.0f + static_cast<float>(g.db_s[k - 1]) * hs_v) * static_cast<float>(g.dzeta[k - 1]));
            lap_v -= (vc - v_get(k - 1, j, i)) / (dz_m * dz_c);
        }
        tend += static_cast<float>(cfg.km_const) * (lap_h + lap_v);
    }

    if (cfg.hyperdiff_coeff > 0.0) {
        const bool ok_x = (cfg.bc_x == 1) || (i >= 2 && i <= nx - 3);
        const bool ok_y = (cfg.bc_y == 1) || (j >= 2 && j <= ny - 2);
        float d4 = 0.0f;
        if (ok_x) {
            d4 += v_get(k, j, i - 2) - 4.0f * v_get(k, j, i - 1) + 6.0f * vc
                - 4.0f * v_get(k, j, i + 1) + v_get(k, j, i + 2);
        }
        if (ok_y) {
            d4 += v_get(k, j - 2, i) - 4.0f * v_get(k, j - 1, i) + 6.0f * vc
                - 4.0f * v_get(k, j + 1, i) + v_get(k, j + 2, i);
        }
        tend -= (static_cast<float>(cfg.hyperdiff_coeff) / fmaxf(tau_stage, 1e-6f)) * d4;
    }

    if ((cfg.bc_x == 2 || cfg.bc_y == 2) && lbc_v != nullptr && cfg.lbc_tau_max_s > 0.0) {
        const float tau_max = static_cast<float>(cfg.lbc_tau_max_s);
        float tau = 0.0f;
        if (cfg.bc_x == 2) {
            if (cfg.lbc_edges & 1) tau = fmaxf(tau, davies_weight_f(i, cfg.lbc_rim_width, tau_max));
            if (cfg.lbc_edges & 2) tau = fmaxf(tau, davies_weight_f(nx - 1 - i, cfg.lbc_rim_width, tau_max));
        }
        if (cfg.bc_y == 2) {
            if (cfg.lbc_edges & 4) tau = fmaxf(tau, davies_weight_f(j, cfg.lbc_rim_width, tau_max));
            if (cfg.lbc_edges & 8) tau = fmaxf(tau, davies_weight_f(ny - j, cfg.lbc_rim_width, tau_max));
        }
        if (tau > 0.0f) {
            tend -= tau * (vc - lbc_v[idx]);
        }
    }

    if (cfg.has_iau && iau_v != nullptr) tend += iau_v[idx];
    rv[idx] = tend;
}

__global__ void nh_slow_w_kernel(
    NhGridConfig cfg,
    DeviceStaticGrid g,
    float tau_stage,
    const float* u,
    const float* v,
    const float* w,
    float* rw
) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    const int32_t k = blockIdx.z;
    if (k == 0 || k >= cfg.nz || j >= cfg.ny || i >= cfg.nx) {
        if (k <= cfg.nz && j < cfg.ny && i < cfg.nx) {
            rw[static_cast<size_t>(k) * cfg.ny * cfg.nx + j * cfg.nx + i] = 0.0f;
        }
        return;
    }

    const int32_t nz = cfg.nz, ny = cfg.ny, nx = cfg.nx;
    const size_t idx = static_cast<size_t>(k) * ny * nx + static_cast<size_t>(j) * nx + i;
    const float dx_f = static_cast<float>(cfg.dx_m);
    const float dy_f = static_cast<float>(cfg.dy_m);
    const float msf = static_cast<float>(g.msf[j * nx + i]);
    const float my = cfg.is_spherical ? 1.0f : msf;
    const float hs = static_cast<float>(g.hs[j * nx + i]);
    const float ds_k = static_cast<float>(g.dzeta[k]);
    const float ds_km = static_cast<float>(g.dzeta[k - 1]);
    const float wt = ds_km / (ds_k + ds_km);

    const float u_w = (1.0f - wt) * 0.5f * (read_u_f(u, k - 1, j, i, nz, ny, nx, cfg.bc_x, cfg.bc_y) +
                                            read_u_f(u, k - 1, j, i + 1, nz, ny, nx, cfg.bc_x, cfg.bc_y)) +
                      wt          * 0.5f * (read_u_f(u, k, j, i, nz, ny, nx, cfg.bc_x, cfg.bc_y) +
                                            read_u_f(u, k, j, i + 1, nz, ny, nx, cfg.bc_x, cfg.bc_y));
    const float v_w = (1.0f - wt) * 0.5f * (read_v_f(v, k - 1, j, i, nz, ny, nx, cfg.bc_x, cfg.bc_y) +
                                            read_v_f(v, k - 1, j + 1, i, nz, ny, nx, cfg.bc_x, cfg.bc_y)) +
                      wt          * 0.5f * (read_v_f(v, k, j, i, nz, ny, nx, cfg.bc_x, cfg.bc_y) +
                                            read_v_f(v, k, j + 1, i, nz, ny, nx, cfg.bc_x, cfg.bc_y));
    float gz_w = 0.0f;
    compute_contravariant_at_scalar_col_f(cfg, g, u, v, w, k, j, i, gz_w);
    const float g_w = 1.0f + static_cast<float>(g.db_w[k]) * hs;
    const float zeta_dot = gz_w / fmaxf(g_w, 1e-4f);

    auto w_get = [&](int32_t kk, int32_t jj, int32_t ii) -> float {
        return read_w_f(w, kk, jj, ii, nz, ny, nx, cfg.bc_x, cfg.bc_y);
    };
    const float wc = w_get(k, j, i);
    float dwdx = 0.0f;
    if (u_w >= 0.0f) {
        dwdx = (up3_left_f(w_get(k, j, i - 1), wc, w_get(k, j, i + 1)) -
                up3_left_f(w_get(k, j, i - 2), w_get(k, j, i - 1), wc)) / dx_f;
    } else {
        dwdx = (up3_left_f(w_get(k, j, i + 2), w_get(k, j, i + 1), wc) -
                up3_left_f(w_get(k, j, i + 1), wc, w_get(k, j, i - 1))) / dx_f;
    }
    float dwdy = 0.0f;
    if (v_w >= 0.0f) {
        dwdy = (up3_left_f(w_get(k, j - 1, i), wc, w_get(k, j + 1, i)) -
                up3_left_f(w_get(k, j - 2, i), w_get(k, j - 1, i), wc)) / dy_f;
    } else {
        dwdy = (up3_left_f(w_get(k, j + 2, i), w_get(k, j + 1, i), wc) -
                up3_left_f(w_get(k, j + 1, i), wc, w_get(k, j - 1, i))) / dy_f;
    }
    const float dz_w_min = fminf(ds_k, ds_km);
    const float zd_cap = 0.45f * dz_w_min / fmaxf(tau_stage, 1e-6f);
    const float zd_exp = fmaxf(-zd_cap, fminf(zd_cap, zeta_dot));
    float dwdz = 0.0f;
    if (zd_exp >= 0.0f) {
        dwdz = (wc - w_get(k - 1, j, i)) / ds_km;
    } else {
        dwdz = (w_get(k + 1, j, i) - wc) / ds_k;
    }

    float tend = -(msf * u_w * dwdx + my * v_w * dwdy) - zd_exp * dwdz;

    if (cfg.km_const > 0.0) {
        const float lap_h = (msf * msf) * (w_get(k, j, i + 1) - 2.0f * wc + w_get(k, j, i - 1)) / (dx_f * dx_f)
                          + (my  * my ) * (w_get(k, j + 1, i) - 2.0f * wc + w_get(k, j - 1, i)) / (dy_f * dy_f);
        const float dz_w = g_w * 0.5f * (ds_k + ds_km);
        const float dz_p = (1.0f + static_cast<float>(g.db_s[k]) * hs) * ds_k;
        const float dz_m = (1.0f + static_cast<float>(g.db_s[k - 1]) * hs) * ds_km;
        const float lap_v = ((w_get(k + 1, j, i) - wc) / dz_p - (wc - w_get(k - 1, j, i)) / dz_m) / dz_w;
        tend += static_cast<float>(cfg.km_const) * (lap_h + lap_v);
    }

    if (cfg.bc_x == 2 || cfg.bc_y == 2) {
        const float tau_max = static_cast<float>(cfg.lbc_tau_max_s);
        float tau = 0.0f;
        if (cfg.bc_x == 2) {
            if (cfg.lbc_edges & 1) tau = fmaxf(tau, davies_weight_f(i, cfg.lbc_rim_width, tau_max));
            if (cfg.lbc_edges & 2) tau = fmaxf(tau, davies_weight_f(nx - 1 - i, cfg.lbc_rim_width, tau_max));
        }
        if (cfg.bc_y == 2) {
            if (cfg.lbc_edges & 4) tau = fmaxf(tau, davies_weight_f(j, cfg.lbc_rim_width, tau_max));
            if (cfg.lbc_edges & 8) tau = fmaxf(tau, davies_weight_f(ny - 1 - j, cfg.lbc_rim_width, tau_max));
        }
        tend -= tau * wc;
    }

    rw[idx] = tend;
}

// ---------------------------------------------------------------------------
// IEVA (Wicker & Skamarock 2020): Column-Implicit Residual Vertical Advection
// Solves strictly diagonally-dominant tridiagonal system wherever |CFL_z| > 0.45
// ---------------------------------------------------------------------------
__global__ void nh_ieva_vert_advect_scalar_kernel(
    NhGridConfig cfg,
    DeviceStaticGrid g,
    float tau_stage,
    const float* u,
    const float* v,
    const float* w,
    float* thp,
    float* pip,
    double* tri_c,
    double* tri_d
) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= cfg.ny || i >= cfg.nx) return;

    const int32_t nz = cfg.nz, ny = cfg.ny, nx = cfg.nx;
    const size_t col_stride = static_cast<size_t>(ny) * nx;
    const size_t col = static_cast<size_t>(j) * nx + i;
    const float hs = static_cast<float>(g.hs[col]);
    const float tau_safe = fmaxf(tau_stage, 1e-6f);

    bool any_imp = false;
    float gz_lo = 0.0f;
    compute_contravariant_at_scalar_col_f(cfg, g, u, v, w, 0, j, i, gz_lo);
    for (int32_t k = 0; k < nz; ++k) {
        float gz_hi = 0.0f;
        compute_contravariant_at_scalar_col_f(cfg, g, u, v, w, k + 1, j, i, gz_hi);
        const float g_s = 1.0f + static_cast<float>(g.db_s[k]) * hs;
        const float zd = 0.5f * (gz_lo + gz_hi) / fmaxf(g_s, 1e-4f);
        gz_lo = gz_hi;
        const float dz_k = static_cast<float>(g.dzeta[k]);
        const float zd_cap = 0.45f * dz_k / tau_safe;
        if (fabsf(zd) > zd_cap) {
            any_imp = true;
            break;
        }
    }
    if (!any_imp) return;

    // Pass 1: Solve tridiagonal system for theta' and factorize tri_c once
    compute_contravariant_at_scalar_col_f(cfg, g, u, v, w, 0, j, i, gz_lo);
    for (int32_t k = 0; k < nz; ++k) {
        float gz_hi = 0.0f;
        compute_contravariant_at_scalar_col_f(cfg, g, u, v, w, k + 1, j, i, gz_hi);
        const float g_s = 1.0f + static_cast<float>(g.db_s[k]) * hs;
        const float zd = 0.5f * (gz_lo + gz_hi) / fmaxf(g_s, 1e-4f);
        gz_lo = gz_hi;
        const float dz_k = static_cast<float>(g.dzeta[k]);
        const float zd_cap = 0.45f * dz_k / tau_safe;
        const float zd_exp = fmaxf(-zd_cap, fminf(zd_cap, zd));
        const double zd_imp = static_cast<double>(zd - zd_exp);

        const size_t idx = static_cast<size_t>(k) * col_stride + col;
        const double A_k = (k > 0 && zd_imp > 0.0) ? (-tau_stage * zd_imp / g.dzeta[k]) : 0.0;
        const double C_k = (k < nz - 1 && zd_imp < 0.0) ? (tau_stage * zd_imp / g.dzeta[k]) : 0.0;
        const double B_k = 1.0 - A_k - C_k;
        const double rhs = static_cast<double>(thp[idx]);
        if (k == 0) {
            tri_c[idx] = C_k / B_k;
            tri_d[idx] = rhs / B_k;
        } else {
            const size_t idx_m = static_cast<size_t>(k - 1) * col_stride + col;
            const double denom = B_k - A_k * tri_c[idx_m];
            tri_c[idx] = (k < nz - 1) ? (C_k / denom) : 0.0;
            tri_d[idx] = (rhs - A_k * tri_d[idx_m]) / denom;
        }
    }
    double x_next = tri_d[static_cast<size_t>(nz - 1) * col_stride + col];
    thp[static_cast<size_t>(nz - 1) * col_stride + col] = static_cast<float>(x_next);
    for (int32_t k = nz - 2; k >= 0; --k) {
        const size_t idx = static_cast<size_t>(k) * col_stride + col;
        x_next = tri_d[idx] - tri_c[idx] * x_next;
        thp[idx] = static_cast<float>(x_next);
    }

    // Pass 2: Forward/backward Thomas for pi' using identical velocity field
    compute_contravariant_at_scalar_col_f(cfg, g, u, v, w, 0, j, i, gz_lo);
    for (int32_t k = 0; k < nz; ++k) {
        float gz_hi = 0.0f;
        compute_contravariant_at_scalar_col_f(cfg, g, u, v, w, k + 1, j, i, gz_hi);
        const float g_s = 1.0f + static_cast<float>(g.db_s[k]) * hs;
        const float zd = 0.5f * (gz_lo + gz_hi) / fmaxf(g_s, 1e-4f);
        gz_lo = gz_hi;
        const float dz_k = static_cast<float>(g.dzeta[k]);
        const float zd_cap = 0.45f * dz_k / tau_safe;
        const float zd_exp = fmaxf(-zd_cap, fminf(zd_cap, zd));
        const double zd_imp = static_cast<double>(zd - zd_exp);

        const size_t idx = static_cast<size_t>(k) * col_stride + col;
        const double A_k = (k > 0 && zd_imp > 0.0) ? (-tau_stage * zd_imp / g.dzeta[k]) : 0.0;
        const double C_k = (k < nz - 1 && zd_imp < 0.0) ? (tau_stage * zd_imp / g.dzeta[k]) : 0.0;
        const double B_k = 1.0 - A_k - C_k;
        const double rhs = static_cast<double>(pip[idx]);
        if (k == 0) {
            tri_c[idx] = C_k / B_k;
            tri_d[idx] = rhs / B_k;
        } else {
            const size_t idx_m = static_cast<size_t>(k - 1) * col_stride + col;
            const double denom = B_k - A_k * tri_c[idx_m];
            tri_c[idx] = (k < nz - 1) ? (C_k / denom) : 0.0;
            tri_d[idx] = (rhs - A_k * tri_d[idx_m]) / denom;
        }
    }
    x_next = tri_d[static_cast<size_t>(nz - 1) * col_stride + col];
    pip[static_cast<size_t>(nz - 1) * col_stride + col] = static_cast<float>(x_next);
    for (int32_t k = nz - 2; k >= 0; --k) {
        const size_t idx = static_cast<size_t>(k) * col_stride + col;
        x_next = tri_d[idx] - tri_c[idx] * x_next;
        pip[idx] = static_cast<float>(x_next);
    }
}

__global__ void nh_ieva_vert_advect_u_kernel(
    NhGridConfig cfg,
    DeviceStaticGrid g,
    float tau_stage,
    const float* v,
    const float* w,
    float* u,
    double* tri_c,
    double* tri_d
) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= cfg.ny || i > cfg.nx) return;
    if (cfg.bc_x == 0 && (i == 0 || i == cfg.nx)) return;

    const int32_t nz = cfg.nz, ny = cfg.ny, nx = cfg.nx;
    const size_t col_stride = static_cast<size_t>(ny) * (nx + 1);
    const size_t col = static_cast<size_t>(j) * (nx + 1) + i;
    const int32_t il = (cfg.bc_x == 1) ? wrap_idx(i - 1, nx) : ((i <= 0) ? 0 : i - 1);
    const int32_t ir = (cfg.bc_x == 1) ? wrap_idx(i, nx) : ((i >= nx) ? nx - 1 : i);
    const float hs_u = 0.5f * (static_cast<float>(g.hs[j * nx + il]) + static_cast<float>(g.hs[j * nx + ir]));
    const float tau_safe = fmaxf(tau_stage, 1e-6f);

    bool any_imp = false;
    float gzl_lo = 0.0f, gzr_lo = 0.0f;
    compute_contravariant_at_scalar_col_f(cfg, g, u, v, w, 0, j, il, gzl_lo);
    compute_contravariant_at_scalar_col_f(cfg, g, u, v, w, 0, j, ir, gzr_lo);
    for (int32_t k = 0; k < nz; ++k) {
        float gzl_hi = 0.0f, gzr_hi = 0.0f;
        compute_contravariant_at_scalar_col_f(cfg, g, u, v, w, k + 1, j, il, gzl_hi);
        compute_contravariant_at_scalar_col_f(cfg, g, u, v, w, k + 1, j, ir, gzr_hi);
        const float g_u = 1.0f + static_cast<float>(g.db_s[k]) * hs_u;
        const float zd = 0.25f * (gzl_lo + gzl_hi + gzr_lo + gzr_hi) / fmaxf(g_u, 1e-4f);
        gzl_lo = gzl_hi; gzr_lo = gzr_hi;
        const float dz_k = static_cast<float>(g.dzeta[k]);
        if (fabsf(zd) > 0.45f * dz_k / tau_safe) {
            any_imp = true;
            break;
        }
    }
    if (!any_imp) return;

    compute_contravariant_at_scalar_col_f(cfg, g, u, v, w, 0, j, il, gzl_lo);
    compute_contravariant_at_scalar_col_f(cfg, g, u, v, w, 0, j, ir, gzr_lo);
    for (int32_t k = 0; k < nz; ++k) {
        float gzl_hi = 0.0f, gzr_hi = 0.0f;
        compute_contravariant_at_scalar_col_f(cfg, g, u, v, w, k + 1, j, il, gzl_hi);
        compute_contravariant_at_scalar_col_f(cfg, g, u, v, w, k + 1, j, ir, gzr_hi);
        const float g_u = 1.0f + static_cast<float>(g.db_s[k]) * hs_u;
        const float zd = 0.25f * (gzl_lo + gzl_hi + gzr_lo + gzr_hi) / fmaxf(g_u, 1e-4f);
        gzl_lo = gzl_hi; gzr_lo = gzr_hi;
        const float dz_k = static_cast<float>(g.dzeta[k]);
        const float zd_cap = 0.45f * dz_k / tau_safe;
        const float zd_exp = fmaxf(-zd_cap, fminf(zd_cap, zd));
        const double zd_imp = static_cast<double>(zd - zd_exp);

        const size_t idx = static_cast<size_t>(k) * col_stride + col;
        const double A_k = (k > 0 && zd_imp > 0.0) ? (-tau_stage * zd_imp / (0.5 * (g.dzeta[k] + g.dzeta[k - 1]))) : 0.0;
        const double C_k = (k < nz - 1 && zd_imp < 0.0) ? (tau_stage * zd_imp / (0.5 * (g.dzeta[k + 1] + g.dzeta[k]))) : 0.0;
        const double B_k = 1.0 - A_k - C_k;
        const double rhs = static_cast<double>(u[idx]);
        if (k == 0) {
            tri_c[idx] = C_k / B_k;
            tri_d[idx] = rhs / B_k;
        } else {
            const size_t idx_m = static_cast<size_t>(k - 1) * col_stride + col;
            const double denom = B_k - A_k * tri_c[idx_m];
            tri_c[idx] = (k < nz - 1) ? (C_k / denom) : 0.0;
            tri_d[idx] = (rhs - A_k * tri_d[idx_m]) / denom;
        }
    }
    double x_next = tri_d[static_cast<size_t>(nz - 1) * col_stride + col];
    u[static_cast<size_t>(nz - 1) * col_stride + col] = static_cast<float>(x_next);
    for (int32_t k = nz - 2; k >= 0; --k) {
        const size_t idx = static_cast<size_t>(k) * col_stride + col;
        x_next = tri_d[idx] - tri_c[idx] * x_next;
        u[idx] = static_cast<float>(x_next);
    }
}

__global__ void nh_ieva_vert_advect_v_kernel(
    NhGridConfig cfg,
    DeviceStaticGrid g,
    float tau_stage,
    const float* u,
    const float* w,
    float* v,
    double* tri_c,
    double* tri_d
) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j > cfg.ny || i >= cfg.nx) return;
    if (cfg.bc_y == 0 && (j == 0 || j == cfg.ny)) return;

    const int32_t nz = cfg.nz, ny = cfg.ny, nx = cfg.nx;
    const size_t col_stride = static_cast<size_t>(ny + 1) * nx;
    const size_t col = static_cast<size_t>(j) * nx + i;
    const int32_t jl = (cfg.bc_y == 1) ? wrap_idx(j - 1, ny) : ((j <= 0) ? 0 : j - 1);
    const int32_t jr = (cfg.bc_y == 1) ? wrap_idx(j, ny) : ((j >= ny) ? ny - 1 : j);
    const float hs_v = 0.5f * (static_cast<float>(g.hs[jl * nx + i]) + static_cast<float>(g.hs[jr * nx + i]));
    const float tau_safe = fmaxf(tau_stage, 1e-6f);

    bool any_imp = false;
    float gzl_lo = 0.0f, gzr_lo = 0.0f;
    compute_contravariant_at_scalar_col_f(cfg, g, u, v, w, 0, jl, i, gzl_lo);
    compute_contravariant_at_scalar_col_f(cfg, g, u, v, w, 0, jr, i, gzr_lo);
    for (int32_t k = 0; k < nz; ++k) {
        float gzl_hi = 0.0f, gzr_hi = 0.0f;
        compute_contravariant_at_scalar_col_f(cfg, g, u, v, w, k + 1, jl, i, gzl_hi);
        compute_contravariant_at_scalar_col_f(cfg, g, u, v, w, k + 1, jr, i, gzr_hi);
        const float g_v = 1.0f + static_cast<float>(g.db_s[k]) * hs_v;
        const float zd = 0.25f * (gzl_lo + gzl_hi + gzr_lo + gzr_hi) / fmaxf(g_v, 1e-4f);
        gzl_lo = gzl_hi; gzr_lo = gzr_hi;
        const float dz_k = static_cast<float>(g.dzeta[k]);
        if (fabsf(zd) > 0.45f * dz_k / tau_safe) {
            any_imp = true;
            break;
        }
    }
    if (!any_imp) return;

    compute_contravariant_at_scalar_col_f(cfg, g, u, v, w, 0, jl, i, gzl_lo);
    compute_contravariant_at_scalar_col_f(cfg, g, u, v, w, 0, jr, i, gzr_lo);
    for (int32_t k = 0; k < nz; ++k) {
        float gzl_hi = 0.0f, gzr_hi = 0.0f;
        compute_contravariant_at_scalar_col_f(cfg, g, u, v, w, k + 1, jl, i, gzl_hi);
        compute_contravariant_at_scalar_col_f(cfg, g, u, v, w, k + 1, jr, i, gzr_hi);
        const float g_v = 1.0f + static_cast<float>(g.db_s[k]) * hs_v;
        const float zd = 0.25f * (gzl_lo + gzl_hi + gzr_lo + gzr_hi) / fmaxf(g_v, 1e-4f);
        gzl_lo = gzl_hi; gzr_lo = gzr_hi;
        const float dz_k = static_cast<float>(g.dzeta[k]);
        const float zd_cap = 0.45f * dz_k / tau_safe;
        const float zd_exp = fmaxf(-zd_cap, fminf(zd_cap, zd));
        const double zd_imp = static_cast<double>(zd - zd_exp);

        const size_t idx = static_cast<size_t>(k) * col_stride + col;
        const double A_k = (k > 0 && zd_imp > 0.0) ? (-tau_stage * zd_imp / (0.5 * (g.dzeta[k] + g.dzeta[k - 1]))) : 0.0;
        const double C_k = (k < nz - 1 && zd_imp < 0.0) ? (tau_stage * zd_imp / (0.5 * (g.dzeta[k + 1] + g.dzeta[k]))) : 0.0;
        const double B_k = 1.0 - A_k - C_k;
        const double rhs = static_cast<double>(v[idx]);
        if (k == 0) {
            tri_c[idx] = C_k / B_k;
            tri_d[idx] = rhs / B_k;
        } else {
            const size_t idx_m = static_cast<size_t>(k - 1) * col_stride + col;
            const double denom = B_k - A_k * tri_c[idx_m];
            tri_c[idx] = (k < nz - 1) ? (C_k / denom) : 0.0;
            tri_d[idx] = (rhs - A_k * tri_d[idx_m]) / denom;
        }
    }
    double x_next = tri_d[static_cast<size_t>(nz - 1) * col_stride + col];
    v[static_cast<size_t>(nz - 1) * col_stride + col] = static_cast<float>(x_next);
    for (int32_t k = nz - 2; k >= 0; --k) {
        const size_t idx = static_cast<size_t>(k) * col_stride + col;
        x_next = tri_d[idx] - tri_c[idx] * x_next;
        v[idx] = static_cast<float>(x_next);
    }
}

__global__ void nh_ieva_vert_advect_w_kernel(
    NhGridConfig cfg,
    DeviceStaticGrid g,
    float tau_stage,
    const float* u,
    const float* v,
    float* w,
    double* tri_c,
    double* tri_d
) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= cfg.ny || i >= cfg.nx) return;

    const int32_t nz = cfg.nz, ny = cfg.ny, nx = cfg.nx;
    if (nz < 3) return;
    const size_t col_stride = static_cast<size_t>(ny) * nx;
    const size_t col = static_cast<size_t>(j) * nx + i;
    const float hs = static_cast<float>(g.hs[col]);
    const float tau_safe = fmaxf(tau_stage, 1e-6f);

    bool any_imp = false;
    for (int32_t k = 1; k < nz; ++k) {
        float gz_w = 0.0f;
        compute_contravariant_at_scalar_col_f(cfg, g, u, v, w, k, j, i, gz_w);
        const float g_w = 1.0f + static_cast<float>(g.db_w[k]) * hs;
        const float zd = gz_w / fmaxf(g_w, 1e-4f);
        const float dz_min = fminf(static_cast<float>(g.dzeta[k]), static_cast<float>(g.dzeta[k - 1]));
        if (fabsf(zd) > 0.45f * dz_min / tau_safe) {
            any_imp = true;
            break;
        }
    }
    if (!any_imp) return;

    const double w0_bc = static_cast<double>(w[col]);
    for (int32_t k = 1; k < nz; ++k) {
        float gz_w = 0.0f;
        compute_contravariant_at_scalar_col_f(cfg, g, u, v, w, k, j, i, gz_w);
        const float g_w = 1.0f + static_cast<float>(g.db_w[k]) * hs;
        const float zd = gz_w / fmaxf(g_w, 1e-4f);
        const float dz_min = fminf(static_cast<float>(g.dzeta[k]), static_cast<float>(g.dzeta[k - 1]));
        const float zd_cap = 0.45f * dz_min / tau_safe;
        const float zd_exp = fmaxf(-zd_cap, fminf(zd_cap, zd));
        const double zd_imp = static_cast<double>(zd - zd_exp);

        const size_t idx = static_cast<size_t>(k) * col_stride + col;
        const double A_k = (zd_imp > 0.0) ? (-tau_stage * zd_imp / g.dzeta[k - 1]) : 0.0;
        const double C_k = (zd_imp < 0.0) ? (tau_stage * zd_imp / g.dzeta[k]) : 0.0;
        const double B_k = 1.0 - A_k - C_k;
        double rhs = static_cast<double>(w[idx]);
        if (k == 1) {
            rhs -= A_k * w0_bc;
            tri_c[idx] = C_k / B_k;
            tri_d[idx] = rhs / B_k;
        } else {
            const size_t idx_m = static_cast<size_t>(k - 1) * col_stride + col;
            const double denom = B_k - A_k * tri_c[idx_m];
            tri_c[idx] = (k < nz - 1) ? (C_k / denom) : 0.0;
            tri_d[idx] = (rhs - A_k * tri_d[idx_m]) / denom;
        }
    }
    double x_next = tri_d[static_cast<size_t>(nz - 1) * col_stride + col];
    w[static_cast<size_t>(nz - 1) * col_stride + col] = static_cast<float>(x_next);
    for (int32_t k = nz - 2; k >= 1; --k) {
        const size_t idx = static_cast<size_t>(k) * col_stride + col;
        x_next = tri_d[idx] - tri_c[idx] * x_next;
        w[idx] = static_cast<float>(x_next);
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: Horizontal Divergence (for symmetric acoustic divergence damping) + acoustic horiz step
// ---------------------------------------------------------------------------
__global__ void nh_compute_div_kernel(
    NhGridConfig cfg,
    DeviceStaticGrid g,
    const float* u,
    const float* v,
    const float* w,
    float* div3d
) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    const int32_t k = blockIdx.z;
    if (k >= cfg.nz || j >= cfg.ny || i >= cfg.nx) return;

    const int32_t nz = cfg.nz, ny = cfg.ny, nx = cfg.nx;
    const size_t idx = static_cast<size_t>(k) * ny * nx + static_cast<size_t>(j) * nx + i;
    const float dx_f = static_cast<float>(cfg.dx_m);
    const float dy_f = static_cast<float>(cfg.dy_m);
    const float db_s_k = static_cast<float>(g.db_s[k]);
    const float msf = static_cast<float>(g.msf[j * nx + i]);
    const float hs = static_cast<float>(g.hs[j * nx + i]);
    const float g_s = 1.0f + db_s_k * hs;

    const float u_r = read_u_f(u, k, j, i + 1, nz, ny, nx, cfg.bc_x, cfg.bc_y);
    const float u_l = read_u_f(u, k, j, i,     nz, ny, nx, cfg.bc_x, cfg.bc_y);
    const float v_n = read_v_f(v, k, j + 1, i, nz, ny, nx, cfg.bc_x, cfg.bc_y);
    const float v_s = read_v_f(v, k, j,     i, nz, ny, nx, cfg.bc_x, cfg.bc_y);

    const float msf_r = read_2d_f(g.msf, j, i + 1, ny, nx, cfg.bc_x, cfg.bc_y);
    const float msf_l = read_2d_f(g.msf, j, i - 1, ny, nx, cfg.bc_x, cfg.bc_y);
    const float msf_n = read_2d_f(g.msf, j + 1, i, ny, nx, cfg.bc_x, cfg.bc_y);
    const float msf_s = read_2d_f(g.msf, j - 1, i, ny, nx, cfg.bc_x, cfg.bc_y);

    const float hs_r = read_2d_f(g.hs, j, i + 1, ny, nx, cfg.bc_x, cfg.bc_y);
    const float hs_l = read_2d_f(g.hs, j, i - 1, ny, nx, cfg.bc_x, cfg.bc_y);
    const float hs_n = read_2d_f(g.hs, j + 1, i, ny, nx, cfg.bc_x, cfg.bc_y);
    const float hs_s = read_2d_f(g.hs, j - 1, i, ny, nx, cfg.bc_x, cfg.bc_y);

    const float gu_r = 1.0f + db_s_k * 0.5f * (hs + hs_r);
    const float gu_l = 1.0f + db_s_k * 0.5f * (hs + hs_l);
    const float gv_n = 1.0f + db_s_k * 0.5f * (hs + hs_n);
    const float gv_s = 1.0f + db_s_k * 0.5f * (hs + hs_s);

    const float mu_r = 0.5f * (msf + msf_r);
    const float mu_l = 0.5f * (msf + msf_l);
    const float mv_n = 0.5f * (msf + msf_n);
    const float mv_s = 0.5f * (msf + msf_s);

    const float hdiv = cfg.is_spherical
        ? msf * (
            (gu_r * u_r - gu_l * u_l) / dx_f +
            (gv_n * v_n / mv_n - gv_s * v_s / mv_s) / dy_f
        )
        : msf * msf * (
            (gu_r * u_r / mu_r - gu_l * u_l / mu_l) / dx_f +
            (gv_n * v_n / mv_n - gv_s * v_s / mv_s) / dy_f
        );

    // Strictly symmetric negative-semi-definite horizontal acoustic divergence damping:
    // Including vertical divergence vdiv = d(G*zeta_dot)/dzeta in div3d while applying grad(div3d)
    // only to horizontal (u, v) produces an asymmetric operator with positive anti-damping eigenvalue
    // +0.5 * nu_div * |kx * kz| on anisotropic NWP grids (dx >> dz) and explicit c_div*(|dh|/dz) blowup
    // over steep terrain. Vertical acoustic modes are damped by off-centered HEVI (beta > 0.5).
    (void)w;
    div3d[idx] = hdiv / fmaxf(g_s, 1e-4f);
}

__global__ void nh_acoustic_u_kernel(
    NhGridConfig cfg,
    DeviceStaticGrid g,
    float dts,
    const float* ru,
    const float* thp,
    const float* pip,
    const float* div3d,
    const float* qv, const float* qc, const float* qr,
    const float* qi, const float* qs, const float* qg,
    float* u
) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    const int32_t k = blockIdx.z;
    if (k >= cfg.nz || j >= cfg.ny || i > cfg.nx) return;

    const int32_t nz = cfg.nz, ny = cfg.ny, nx = cfg.nx;
    const size_t idx = static_cast<size_t>(k) * ny * (nx + 1) + static_cast<size_t>(j) * (nx + 1) + i;

    if (cfg.bc_x == 0 && (i == 0 || i == nx)) {
        u[idx] = 0.0f;
        return;
    }

    const int32_t il = (cfg.bc_x == 1) ? wrap_idx(i - 1, nx) : ((i <= 0) ? 0 : i - 1);
    const int32_t ir = (cfg.bc_x == 1) ? wrap_idx(i, nx) : ((i >= nx) ? nx - 1 : i);
    const size_t idxl = static_cast<size_t>(k) * ny * nx + static_cast<size_t>(j) * nx + il;
    const size_t idxr = static_cast<size_t>(k) * ny * nx + static_cast<size_t>(j) * nx + ir;

    const float dx_f = static_cast<float>(cfg.dx_m);
    const float dy_f = static_cast<float>(cfg.dy_m);
    const float hs_l = static_cast<float>(g.hs[j * nx + il]);
    const float hs_r = static_cast<float>(g.hs[j * nx + ir]);
    const float msf_u = 0.5f * (static_cast<float>(g.msf[j * nx + il]) + static_cast<float>(g.msf[j * nx + ir]));

    float dpidx_z = 0.0f;
    if (cfg.bc_x == 1 || (i > 0 && i < nx)) {
        if (hs_l == hs_r) {
            dpidx_z = (pip[idxr] - pip[idxl]) / dx_f;
        } else {
            const double hs_l_d = g.hs[j * nx + il];
            const double hs_r_d = g.hs[j * nx + ir];
            const double z_l = g.zeta_s[k] + g.b_s[k] * hs_l_d;
            const double z_r = g.zeta_s[k] + g.b_s[k] * hs_r_d;
            const double z_u_d = 0.5 * (z_l + z_r);
            const double pi_l = interp_pi_at_z_f(cfg, g, pip, thp, qv, qc, qr, qi, qs, qg, k, j, il, hs_l_d, z_u_d);
            const double pi_r = interp_pi_at_z_f(cfg, g, pip, thp, qv, qc, qr, qi, qs, qg, k, j, ir, hs_r_d, z_u_d);
            dpidx_z = static_cast<float>((pi_r - pi_l) / cfg.dx_m);
        }
    }

    const float z_u = static_cast<float>(g.zeta_s[k]) + static_cast<float>(g.b_s[k]) * (0.5f * (hs_l + hs_r));
    float qvl, qhl, qvr, qhr;
    const float thb = eval_theta_bar_f32(z_u, static_cast<float>(cfg.theta_sfc_k), static_cast<float>(cfg.n_ref_s));
    const float vf_l = moisture_load_scalar_f(cfg.has_moist, idxl, qv, qc, qr, qi, qs, qg, qvl, qhl);
    const float vf_r = moisture_load_scalar_f(cfg.has_moist, idxr, qv, qc, qr, qi, qs, qg, qvr, qhr);
    const float thv_u = 0.5f * ((thb + thp[idxl]) * vf_l + (thb + thp[idxr]) * vf_r);

    float ddiv_dx = 0.0f;
    if (cfg.div_damp_coeff > 0.0 && div3d != nullptr && (cfg.bc_x == 1 || (i > 0 && i < nx))) {
        const float dx_loc = cfg.is_spherical ? (dx_f / fmaxf(msf_u, 1e-6f)) : dx_f;
        const float dmin = fminf(dx_loc, dy_f);
        const float nu_div = static_cast<float>(cfg.div_damp_coeff) * (dmin * dmin) / fmaxf(dts, 1e-6f);
        ddiv_dx = nu_div * msf_u * (div3d[idxr] - div3d[idxl]) / dx_f;
    }

    u[idx] += dts * (ru[idx] - CP_D_F * thv_u * msf_u * dpidx_z + ddiv_dx);
}

__global__ void nh_acoustic_v_kernel(
    NhGridConfig cfg,
    DeviceStaticGrid g,
    float dts,
    const float* rv,
    const float* thp,
    const float* pip,
    const float* div3d,
    const float* qv, const float* qc, const float* qr,
    const float* qi, const float* qs, const float* qg,
    float* v
) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    const int32_t k = blockIdx.z;
    if (k >= cfg.nz || j > cfg.ny || i >= cfg.nx) return;

    const int32_t nz = cfg.nz, ny = cfg.ny, nx = cfg.nx;
    const size_t idx = static_cast<size_t>(k) * (ny + 1) * nx + static_cast<size_t>(j) * nx + i;

    if (cfg.bc_y == 0 && (j == 0 || j == ny)) {
        v[idx] = 0.0f;
        return;
    }

    const int32_t jl = (cfg.bc_y == 1) ? wrap_idx(j - 1, ny) : ((j <= 0) ? 0 : j - 1);
    const int32_t jr = (cfg.bc_y == 1) ? wrap_idx(j, ny) : ((j >= ny) ? ny - 1 : j);
    const size_t idxl = static_cast<size_t>(k) * ny * nx + static_cast<size_t>(jl) * nx + i;
    const size_t idxr = static_cast<size_t>(k) * ny * nx + static_cast<size_t>(jr) * nx + i;

    const float dx_f = static_cast<float>(cfg.dx_m);
    const float dy_f = static_cast<float>(cfg.dy_m);
    const float hs_l = static_cast<float>(g.hs[jl * nx + i]);
    const float hs_r = static_cast<float>(g.hs[jr * nx + i]);
    const float msf_v = 0.5f * (static_cast<float>(g.msf[jl * nx + i]) + static_cast<float>(g.msf[jr * nx + i]));
    const float my_v = cfg.is_spherical ? 1.0f : msf_v;

    float dpidy_z = 0.0f;
    if (cfg.bc_y == 1 || (j > 0 && j < ny)) {
        if (hs_l == hs_r) {
            dpidy_z = (pip[idxr] - pip[idxl]) / dy_f;
        } else {
            const double hs_l_d = g.hs[jl * nx + i];
            const double hs_r_d = g.hs[jr * nx + i];
            const double z_l = g.zeta_s[k] + g.b_s[k] * hs_l_d;
            const double z_r = g.zeta_s[k] + g.b_s[k] * hs_r_d;
            const double z_v_d = 0.5 * (z_l + z_r);
            const double pi_l = interp_pi_at_z_f(cfg, g, pip, thp, qv, qc, qr, qi, qs, qg, k, jl, i, hs_l_d, z_v_d);
            const double pi_r = interp_pi_at_z_f(cfg, g, pip, thp, qv, qc, qr, qi, qs, qg, k, jr, i, hs_r_d, z_v_d);
            dpidy_z = static_cast<float>((pi_r - pi_l) / cfg.dy_m);
        }
    }

    const float z_v = static_cast<float>(g.zeta_s[k]) + static_cast<float>(g.b_s[k]) * (0.5f * (hs_l + hs_r));
    float qvl, qhl, qvr, qhr;
    const float thb = eval_theta_bar_f32(z_v, static_cast<float>(cfg.theta_sfc_k), static_cast<float>(cfg.n_ref_s));
    const float vf_l = moisture_load_scalar_f(cfg.has_moist, idxl, qv, qc, qr, qi, qs, qg, qvl, qhl);
    const float vf_r = moisture_load_scalar_f(cfg.has_moist, idxr, qv, qc, qr, qi, qs, qg, qvr, qhr);
    const float thv_v = 0.5f * ((thb + thp[idxl]) * vf_l + (thb + thp[idxr]) * vf_r);

    float ddiv_dy = 0.0f;
    if (cfg.div_damp_coeff > 0.0 && div3d != nullptr && (cfg.bc_y == 1 || (j > 0 && j < ny))) {
        const float dx_loc = cfg.is_spherical ? (dx_f / fmaxf(msf_v, 1e-6f)) : dx_f;
        const float dmin = fminf(dx_loc, dy_f);
        const float nu_div = static_cast<float>(cfg.div_damp_coeff) * (dmin * dmin) / fmaxf(dts, 1e-6f);
        ddiv_dy = nu_div * my_v * (div3d[idxr] - div3d[idxl]) / dy_f;
    }

    v[idx] += dts * (rv[idx] - CP_D_F * thv_v * my_v * dpidy_z + ddiv_dy);
}

// ---------------------------------------------------------------------------
// Kernel 3: Column-implicit HEVI vertical (w, pi') Thomas solver + theta' update
// Uses global coalesced FP32 views of d_tri_c and d_tri_d (zero per-thread nz arrays!).
// ---------------------------------------------------------------------------
__global__ void nh_acoustic_vert_implicit_kernel(
    NhGridConfig cfg,
    DeviceStaticGrid g,
    float dts,
    float pi_sfc_f,
    const float* u,
    const float* v,
    const float* rw,
    const float* rth,
    const float* rpi,
    const float* qv, const float* qc, const float* qr,
    const float* qi, const float* qs, const float* qg,
    float* w,
    float* thp,
    float* pip,
    float* tri_c,
    float* tri_d
) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= cfg.ny || i >= cfg.nx) return;

    const int32_t nz = cfg.nz, ny = cfg.ny, nx = cfg.nx;
    const size_t col_stride = static_cast<size_t>(ny) * nx;
    const size_t h_idx = static_cast<size_t>(j) * nx + i;

    const float dx_f = static_cast<float>(cfg.dx_m);
    const float dy_f = static_cast<float>(cfg.dy_m);
    const float hs = static_cast<float>(g.hs[h_idx]);
    const float msf = static_cast<float>(g.msf[h_idx]);
    const float dh_dx = (read_2d_f(g.hs, j, i + 1, ny, nx, cfg.bc_x, cfg.bc_y) -
                         read_2d_f(g.hs, j, i - 1, ny, nx, cfg.bc_x, cfg.bc_y)) / (2.0f * dx_f);
    const float dh_dy = (read_2d_f(g.hs, j + 1, i, ny, nx, cfg.bc_x, cfg.bc_y) -
                         read_2d_f(g.hs, j - 1, i, ny, nx, cfg.bc_x, cfg.bc_y)) / (2.0f * dy_f);
    const float slope_h = msf * dh_dx;
    const float slope_y = (cfg.is_spherical ? 1.0f : msf) * dh_dy;

    const float msf_r = read_2d_f(g.msf, j, i + 1, ny, nx, cfg.bc_x, cfg.bc_y);
    const float msf_l = read_2d_f(g.msf, j, i - 1, ny, nx, cfg.bc_x, cfg.bc_y);
    const float msf_n = read_2d_f(g.msf, j + 1, i, ny, nx, cfg.bc_x, cfg.bc_y);
    const float msf_s = read_2d_f(g.msf, j - 1, i, ny, nx, cfg.bc_x, cfg.bc_y);
    const float mu_r = 0.5f * (msf + msf_r), mu_l = 0.5f * (msf + msf_l);
    const float mv_n = 0.5f * (msf + msf_n), mv_s = 0.5f * (msf + msf_s);

    const float hs_r = read_2d_f(g.hs, j, i + 1, ny, nx, cfg.bc_x, cfg.bc_y);
    const float hs_l = read_2d_f(g.hs, j, i - 1, ny, nx, cfg.bc_x, cfg.bc_y);
    const float hs_n = read_2d_f(g.hs, j + 1, i, ny, nx, cfg.bc_x, cfg.bc_y);
    const float hs_s = read_2d_f(g.hs, j - 1, i, ny, nx, cfg.bc_x, cfg.bc_y);
    const float hsu_r = 0.5f * (hs + hs_r), hsu_l = 0.5f * (hs + hs_l);
    const float hsv_n = 0.5f * (hs + hs_n), hsv_s = 0.5f * (hs + hs_s);

    const float beta = static_cast<float>(cfg.beta_offcenter);
    const float omb = 1.0f - beta;
    const float th_sfc_f = static_cast<float>(cfg.theta_sfc_k);
    const float n_ref_f = static_cast<float>(cfg.n_ref_s);

    auto omega_h_at = [&](int32_t k_iface) -> float {
        if (k_iface >= nz) return 0.0f;
        float uw = 0.0f, vw = 0.0f;
        if (k_iface == 0) {
            uw = 0.5f * (read_u_f(u, 0, j, i, nz, ny, nx, cfg.bc_x, cfg.bc_y) +
                         read_u_f(u, 0, j, i + 1, nz, ny, nx, cfg.bc_x, cfg.bc_y));
            vw = 0.5f * (read_v_f(v, 0, j, i, nz, ny, nx, cfg.bc_x, cfg.bc_y) +
                         read_v_f(v, 0, j + 1, i, nz, ny, nx, cfg.bc_x, cfg.bc_y));
        } else {
            const float ds_k = static_cast<float>(g.dzeta[k_iface]);
            const float ds_km = static_cast<float>(g.dzeta[k_iface - 1]);
            const float wt = ds_km / (ds_k + ds_km);
            uw = (1.0f - wt) * 0.5f * (read_u_f(u, k_iface - 1, j, i, nz, ny, nx, cfg.bc_x, cfg.bc_y) +
                                       read_u_f(u, k_iface - 1, j, i + 1, nz, ny, nx, cfg.bc_x, cfg.bc_y)) +
                 wt          * 0.5f * (read_u_f(u, k_iface, j, i, nz, ny, nx, cfg.bc_x, cfg.bc_y) +
                                       read_u_f(u, k_iface, j, i + 1, nz, ny, nx, cfg.bc_x, cfg.bc_y));
            vw = (1.0f - wt) * 0.5f * (read_v_f(v, k_iface - 1, j, i, nz, ny, nx, cfg.bc_x, cfg.bc_y) +
                                       read_v_f(v, k_iface - 1, j + 1, i, nz, ny, nx, cfg.bc_x, cfg.bc_y)) +
                 wt          * 0.5f * (read_v_f(v, k_iface, j, i, nz, ny, nx, cfg.bc_x, cfg.bc_y) +
                                       read_v_f(v, k_iface, j + 1, i, nz, ny, nx, cfg.bc_x, cfg.bc_y));
        }
        return -static_cast<float>(g.b_w[k_iface]) * (slope_h * uw + slope_y * vw);
    };

    const float w0_new = -omega_h_at(0);
    w[h_idx] = w0_new;
    w[static_cast<size_t>(nz) * col_stride + h_idx] = 0.0f;

    float om_lo = omega_h_at(0);
    float ap_prev = 0.0f, am_prev = 0.0f, pi_eff_prev = 0.0f, pi_hat_prev = 0.0f, thb_km = 0.0f;
    float wm_prev = w0_new;

    for (int32_t k = 0; k < nz; ++k) {
        const size_t idx = static_cast<size_t>(k) * col_stride + h_idx;
        const float dz_k = static_cast<float>(g.dzeta[k]);
        const float db_s_k = static_cast<float>(g.db_s[k]);
        const float z_s = static_cast<float>(g.zeta_s[k]) + static_cast<float>(g.b_s[k]) * hs;
        const float g_s = 1.0f + db_s_k * hs;
        const float dz_s = g_s * dz_k;

        float thb = 0.0f, pib = 0.0f;
        eval_theta_pi_bar_f32(z_s, th_sfc_f, pi_sfc_f, n_ref_f, thb, pib);
        const float c_ac = (R_D_F / CV_D_F) * pib;

        const float om_hi = omega_h_at(k + 1);
        const float gu_r = 1.0f + db_s_k * hsu_r, gu_l = 1.0f + db_s_k * hsu_l;
        const float gv_n = 1.0f + db_s_k * hsv_n, gv_s = 1.0f + db_s_k * hsv_s;
        const float u_r = read_u_f(u, k, j, i + 1, nz, ny, nx, cfg.bc_x, cfg.bc_y);
        const float u_l = read_u_f(u, k, j, i,     nz, ny, nx, cfg.bc_x, cfg.bc_y);
        const float v_n = read_v_f(v, k, j + 1, i, nz, ny, nx, cfg.bc_x, cfg.bc_y);
        const float v_s = read_v_f(v, k, j,     i, nz, ny, nx, cfg.bc_x, cfg.bc_y);

        const float hdiv_v = cfg.is_spherical
            ? msf * (
                (gu_r * u_r - gu_l * u_l) / dx_f +
                (gv_n * v_n / mv_n - gv_s * v_s / mv_s) / dy_f
            )
            : msf * msf * (
                (gu_r * u_r / mu_r - gu_l * u_l / mu_l) / dx_f +
                (gv_n * v_n / mv_n - gv_s * v_s / mv_s) / dy_f
            );
        const float d_h = (hdiv_v + (om_hi - om_lo) / dz_k) / fmaxf(g_s, 1e-4f);
        om_lo = om_hi;

        const float grav_term = GRAVITY_F / (2.0f * CP_D_F * thb);
        const float ap = c_ac / dz_s - grav_term;
        const float am = c_ac / dz_s + grav_term;
        const float wm_cur = wm_prev;
        const float wm_nxt = (k + 1 < nz) ? w[static_cast<size_t>(k + 1) * col_stride + h_idx] : 0.0f;
        wm_prev = wm_nxt;

        const float pi_m = pip[idx];
        const float pi_hat = pi_m + dts * (rpi[idx] - c_ac * d_h - omb * (ap * wm_nxt - am * wm_cur));
        const float pi_eff = omb * pi_m + beta * pi_hat;

        if (k > 0) {
            pip[static_cast<size_t>(k - 1) * col_stride + h_idx] = pi_hat_prev;

            const size_t w_idx = static_cast<size_t>(k) * col_stride + h_idx;
            const float z_w = static_cast<float>(g.zeta_w[k]) + static_cast<float>(g.b_w[k]) * hs;
            const float g_w = 1.0f + static_cast<float>(g.db_w[k]) * hs;
            const float dz_km = static_cast<float>(g.dzeta[k - 1]);
            const float dz_w = g_w * 0.5f * (dz_k + dz_km);

            const float thb_w = eval_theta_bar_f32(z_w, th_sfc_f, n_ref_f);

            const size_t idx_km = static_cast<size_t>(k - 1) * col_stride + h_idx;
            float qvm, qhm, qvp, qhp;
            const float vfm = moisture_load_scalar_f(cfg.has_moist, idx_km, qv, qc, qr, qi, qs, qg, qvm, qhm);
            const float vfp = moisture_load_scalar_f(cfg.has_moist, idx,    qv, qc, qr, qi, qs, qg, qvp, qhp);

            const float wt = dz_km / (dz_k + dz_km);
            const float dthv_km = thp[idx_km] * vfm + thb_km * (vfm - 1.0f);
            const float dthv_kp = thp[idx]    * vfp + thb    * (vfp - 1.0f);
            const float dthv_w = (1.0f - wt) * dthv_km + wt * dthv_kp;
            const float thv_w = thb_w + dthv_w;
            const float buoy = GRAVITY_F * dthv_w / thb_w;

            const float pw = dts * (CP_D_F * thv_w) / dz_w;
            const float tau_sp = sponge_tau_f(
                z_w, static_cast<float>(cfg.z_top_m), static_cast<float>(cfg.sponge_depth_m), static_cast<float>(cfg.sponge_tau_max_s)
            );

            const float pw_b2 = pw * dts * beta * beta;
            const float A_k = -pw_b2 * am_prev;
            const float B_k = 1.0f + dts * tau_sp + pw_b2 * (am + ap_prev);
            const float C_k = (k < nz - 1) ? (-pw_b2 * ap) : 0.0f;

            float rhs_k = w[w_idx] + dts * (rw[w_idx] + buoy) - pw * (pi_eff - pi_eff_prev);
            if (k == 1) rhs_k -= A_k * w0_new;

            if (k == 1) {
                tri_c[w_idx] = C_k / B_k;
                tri_d[w_idx] = rhs_k / B_k;
            } else {
                const size_t wm_idx = static_cast<size_t>(k - 1) * col_stride + h_idx;
                const float denom = B_k - A_k * tri_c[wm_idx];
                tri_c[w_idx] = C_k / denom;
                tri_d[w_idx] = (rhs_k - A_k * tri_d[wm_idx]) / denom;
            }
        }

        ap_prev = ap;
        am_prev = am;
        pi_eff_prev = pi_eff;
        pi_hat_prev = pi_hat;
        thb_km = thb;
    }
    pip[static_cast<size_t>(nz - 1) * col_stride + h_idx] = pi_hat_prev;

    // Pass 2: Backward substitution for w[nz-1..1], then update pi_p[nz-1..0] and theta_p[nz-1..0]
    float w_above = 0.0f; // w[nz] = 0
    float thb_w_hi = eval_theta_bar_f32(
        static_cast<float>(g.zeta_w[nz]) + static_cast<float>(g.b_w[nz]) * hs, th_sfc_f, n_ref_f
    );
    for (int32_t k = nz - 1; k >= 0; --k) {
        float w_below = w0_new;
        if (k > 0) {
            const size_t w_idx = static_cast<size_t>(k) * col_stride + h_idx;
            w_below = tri_d[w_idx] - ((k < nz - 1) ? tri_c[w_idx] * w_above : 0.0f);
            w[w_idx] = w_below;
        }

        const size_t idx = static_cast<size_t>(k) * col_stride + h_idx;
        const float z_s = static_cast<float>(g.zeta_s[k]) + static_cast<float>(g.b_s[k]) * hs;
        const float g_s = 1.0f + static_cast<float>(g.db_s[k]) * hs;
        const float dz_s = g_s * static_cast<float>(g.dzeta[k]);
        float thb = 0.0f, pib = 0.0f;
        eval_theta_pi_bar_f32(z_s, th_sfc_f, pi_sfc_f, n_ref_f, thb, pib);
        const float c_ac = (R_D_F / CV_D_F) * pib;
        const float grav_term = GRAVITY_F / (2.0f * CP_D_F * thb);
        const float ap = c_ac / dz_s - grav_term;
        const float am = c_ac / dz_s + grav_term;

        const float pi_hat = pip[idx];
        pip[idx] = pi_hat - dts * beta * (ap * w_above - am * w_below);

        // Symplectic Forward-Backward vertical theta_bar advection using updated w^{m+1}
        const float thb_w_lo = eval_theta_bar_f32(
            static_cast<float>(g.zeta_w[k]) + static_cast<float>(g.b_w[k]) * hs, th_sfc_f, n_ref_f
        );
        const float w_avg = 0.5f * (w_above + w_below);
        thp[idx] += dts * (rth[idx] - w_avg * (thb_w_hi - thb_w_lo) / dz_s);

        thb_w_hi = thb_w_lo;
        w_above = w_below;
    }
}

// ---------------------------------------------------------------------------
// Kernel 4: Positive-definite conservative 3D FCT Zalesak tracer advection (FP32)
// Conserves integral sum(rho_bar * G_s * dzeta / msf^2 * q) >= 0 everywhere.
// ---------------------------------------------------------------------------
__device__ inline float cell_mass_vol_f(
    const NhGridConfig& cfg, const DeviceStaticGrid& g, float pi_sfc_f,
    int32_t k, int32_t j, int32_t i
) {
    const float hs = static_cast<float>(g.hs[j * cfg.nx + i]);
    const float msf = static_cast<float>(g.msf[j * cfg.nx + i]);
    const float z_s = static_cast<float>(g.zeta_s[k]) + static_cast<float>(g.b_s[k]) * hs;
    const float g_s = 1.0f + static_cast<float>(g.db_s[k]) * hs;
    const float rhob = eval_rho_bar_f32(
        z_s, static_cast<float>(cfg.theta_sfc_k), pi_sfc_f, static_cast<float>(cfg.n_ref_s)
    );
    const float area_inv = cfg.is_spherical ? msf : (msf * msf);
    return rhob * g_s * static_cast<float>(g.dzeta[k]) / area_inv;
}

__device__ inline float face_flux_x_f(
    const NhGridConfig& cfg, const DeviceStaticGrid& g, float pi_sfc_f,
    const float* u, int32_t k, int32_t j, int32_t i_face
) {
    if (cfg.bc_x == 0 && (i_face <= 0 || i_face >= cfg.nx)) return 0.0f;
    const int32_t il = (cfg.bc_x == 1)
        ? wrap_idx(i_face - 1, cfg.nx)
        : ((i_face - 1 < 0) ? 0 : ((i_face - 1 >= cfg.nx) ? (cfg.nx - 1) : (i_face - 1)));
    const int32_t ir = (cfg.bc_x == 1)
        ? wrap_idx(i_face, cfg.nx)
        : ((i_face < 0) ? 0 : ((i_face >= cfg.nx) ? (cfg.nx - 1) : i_face));
    const float hs_u = 0.5f * (static_cast<float>(g.hs[j * cfg.nx + il]) + static_cast<float>(g.hs[j * cfg.nx + ir]));
    const float msf_u = 0.5f * (static_cast<float>(g.msf[j * cfg.nx + il]) + static_cast<float>(g.msf[j * cfg.nx + ir]));
    const float z_u = static_cast<float>(g.zeta_s[k]) + static_cast<float>(g.b_s[k]) * hs_u;
    const float g_u = 1.0f + static_cast<float>(g.db_s[k]) * hs_u;
    const float rhob = eval_rho_bar_f32(
        z_u, static_cast<float>(cfg.theta_sfc_k), pi_sfc_f, static_cast<float>(cfg.n_ref_s)
    );
    const float uval = read_u_f(u, k, j, i_face, cfg.nz, cfg.ny, cfg.nx, cfg.bc_x, cfg.bc_y);
    return cfg.is_spherical
        ? (rhob * g_u * static_cast<float>(g.dzeta[k]) * uval)
        : (rhob * g_u * static_cast<float>(g.dzeta[k]) * uval / msf_u);
}

__device__ inline float face_flux_y_f(
    const NhGridConfig& cfg, const DeviceStaticGrid& g, float pi_sfc_f,
    const float* v, int32_t k, int32_t j_face, int32_t i
) {
    if (cfg.bc_y == 0 && (j_face <= 0 || j_face >= cfg.ny)) return 0.0f;
    const int32_t jl = (cfg.bc_y == 1)
        ? wrap_idx(j_face - 1, cfg.ny)
        : ((j_face - 1 < 0) ? 0 : ((j_face - 1 >= cfg.ny) ? (cfg.ny - 1) : (j_face - 1)));
    const int32_t jr = (cfg.bc_y == 1)
        ? wrap_idx(j_face, cfg.ny)
        : ((j_face < 0) ? 0 : ((j_face >= cfg.ny) ? (cfg.ny - 1) : j_face));
    const float hs_v = 0.5f * (static_cast<float>(g.hs[jl * cfg.nx + i]) + static_cast<float>(g.hs[jr * cfg.nx + i]));
    const float msf_v = 0.5f * (static_cast<float>(g.msf[jl * cfg.nx + i]) + static_cast<float>(g.msf[jr * cfg.nx + i]));
    const float z_v = static_cast<float>(g.zeta_s[k]) + static_cast<float>(g.b_s[k]) * hs_v;
    const float g_v = 1.0f + static_cast<float>(g.db_s[k]) * hs_v;
    const float rhob = eval_rho_bar_f32(
        z_v, static_cast<float>(cfg.theta_sfc_k), pi_sfc_f, static_cast<float>(cfg.n_ref_s)
    );
    const float vval = read_v_f(v, k, j_face, i, cfg.nz, cfg.ny, cfg.nx, cfg.bc_x, cfg.bc_y);
    return rhob * g_v * static_cast<float>(g.dzeta[k]) * vval / msf_v;
}

__device__ inline float face_flux_z_f(
    const NhGridConfig& cfg, const DeviceStaticGrid& g, float pi_sfc_f,
    const float* u, const float* v, const float* w, int32_t k_face, int32_t j, int32_t i
) {
    if (k_face <= 0 || k_face >= cfg.nz) return 0.0f;
    const float hs = static_cast<float>(g.hs[j * cfg.nx + i]);
    const float msf = static_cast<float>(g.msf[j * cfg.nx + i]);
    const float z_w = static_cast<float>(g.zeta_w[k_face]) + static_cast<float>(g.b_w[k_face]) * hs;
    const float rhob = eval_rho_bar_f32(
        z_w, static_cast<float>(cfg.theta_sfc_k), pi_sfc_f, static_cast<float>(cfg.n_ref_s)
    );
    float gz = 0.0f;
    compute_contravariant_at_scalar_col_f(cfg, g, u, v, w, k_face, j, i, gz);
    const float area_inv = cfg.is_spherical ? msf : (msf * msf);
    return rhob * gz / area_inv;
}

__device__ inline void compute_fluxes_for_cell_f(
    const NhGridConfig& cfg, const DeviceStaticGrid& g, float pi_sfc_f,
    const float* q, const float* u, const float* v, const float* w,
    int32_t k, int32_t j, int32_t i,
    float& fx_l_low, float& fx_l_hi,
    float& fy_l_low, float& fy_l_hi,
    float& fz_l_low, float& fz_l_hi
) {
    const int32_t nz = cfg.nz, ny = cfg.ny, nx = cfg.nx;
    // Left x-face (i - 1/2)
    const float ux = face_flux_x_f(cfg, g, pi_sfc_f, u, k, j, i);
    const float q_xl_up = (ux >= 0.0f)
        ? read_s_f(q, k, j, i - 1, nz, ny, nx, cfg.bc_x, cfg.bc_y)
        : read_s_f(q, k, j, i,     nz, ny, nx, cfg.bc_x, cfg.bc_y);
    const float q_xl_ho = horiz_face_val_x_f(q, k, j, i, ux, nz, ny, nx, cfg.bc_x, cfg.bc_y, cfg.adv_order);
    fx_l_low = ux * q_xl_up;
    fx_l_hi  = ux * fmaxf(0.0f, q_xl_ho);

    // South y-face (j - 1/2)
    const float vy = face_flux_y_f(cfg, g, pi_sfc_f, v, k, j, i);
    const float q_yl_up = (vy >= 0.0f)
        ? read_s_f(q, k, j - 1, i, nz, ny, nx, cfg.bc_x, cfg.bc_y)
        : read_s_f(q, k, j,     i, nz, ny, nx, cfg.bc_x, cfg.bc_y);
    const float q_yl_ho = horiz_face_val_y_f(q, k, j, i, vy, nz, ny, nx, cfg.bc_x, cfg.bc_y, cfg.adv_order);
    fy_l_low = vy * q_yl_up;
    fy_l_hi  = vy * fmaxf(0.0f, q_yl_ho);

    // Bottom z-face (k - 1/2)
    const float wz = face_flux_z_f(cfg, g, pi_sfc_f, u, v, w, k, j, i);
    const float q_zl_up = (wz >= 0.0f)
        ? read_s_f(q, k - 1, j, i, nz, ny, nx, cfg.bc_x, cfg.bc_y)
        : read_s_f(q, k,     j, i, nz, ny, nx, cfg.bc_x, cfg.bc_y);
    const float q_zl_ho = vert_tvd_face_val_f(q, k, j, i, wz, g.dzeta, nz, ny, nx, cfg.bc_x, cfg.bc_y);
    fz_l_low = wz * q_zl_up;
    fz_l_hi  = wz * fmaxf(0.0f, q_zl_ho);
}

__global__ void nh_tracer_cfl_h_kernel(
    NhGridConfig cfg,
    DeviceStaticGrid g,
    float dt_s,
    float pi_sfc_f,
    const float* u,
    const float* v,
    uint32_t* max_cfl_u32
) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    const int32_t k = blockIdx.z;
    float cfl = 0.0f;
    if (k < cfg.nz && j < cfg.ny && i < cfg.nx) {
        const float dx_f = static_cast<float>(cfg.dx_m);
        const float dy_f = static_cast<float>(cfg.dy_m);
        const float vol = fmaxf(cell_mass_vol_f(cfg, g, pi_sfc_f, k, j, i), 1e-12f);
        const float fx_r = face_flux_x_f(cfg, g, pi_sfc_f, u, k, j, i + 1);
        const float fx_l = face_flux_x_f(cfg, g, pi_sfc_f, u, k, j, i);
        const float fy_n = face_flux_y_f(cfg, g, pi_sfc_f, v, k, j + 1, i);
        const float fy_s = face_flux_y_f(cfg, g, pi_sfc_f, v, k, j, i);
        const float out = (fmaxf(0.0f, fx_r) + fmaxf(0.0f, -fx_l)) / dx_f
                        + (fmaxf(0.0f, fy_n) + fmaxf(0.0f, -fy_s)) / dy_f;
        cfl = (dt_s * out) / vol;
    }
    const unsigned int lane = threadIdx.x & 31;
    for (int offset = 16; offset > 0; offset >>= 1) {
        cfl = fmaxf(cfl, __shfl_down_sync(0xffffffffu, cfl, offset));
    }
    if (lane == 0 && cfl > 0.0f) {
        atomicMax(max_cfl_u32, __float_as_uint(cfl));
    }
}

__global__ void nh_tracer_fct_pass1_kernel(
    NhGridConfig cfg,
    DeviceStaticGrid g,
    float dt_q,
    float pi_sfc_f,
    const float* u,
    const float* v,
    const float* q,
    float* q_low,
    float* r_out
) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    const int32_t k = blockIdx.z;
    if (k >= cfg.nz || j >= cfg.ny || i >= cfg.nx) return;

    const int32_t nz = cfg.nz, ny = cfg.ny, nx = cfg.nx;
    const size_t idx = static_cast<size_t>(k) * ny * nx + static_cast<size_t>(j) * nx + i;
    const float dx_f = static_cast<float>(cfg.dx_m);
    const float dy_f = static_cast<float>(cfg.dy_m);
    const float vol = fmaxf(cell_mass_vol_f(cfg, g, pi_sfc_f, k, j, i), 1e-12f);

    float fxl_low, fxl_hi, fyl_low, fyl_hi;
    {
        const float ux = face_flux_x_f(cfg, g, pi_sfc_f, u, k, j, i);
        const float q_up = (ux >= 0.0f) ? read_s_f(q, k, j, i - 1, nz, ny, nx, cfg.bc_x, cfg.bc_y) : read_s_f(q, k, j, i, nz, ny, nx, cfg.bc_x, cfg.bc_y);
        fxl_low = ux * q_up;
        if (cfg.bc_x != 1 && i == 0) {
            fxl_hi = fxl_low;
        } else {
            const float q_ho = horiz_face_val_x_f(q, k, j, i, ux, nz, ny, nx, cfg.bc_x, cfg.bc_y, cfg.adv_order);
            fxl_hi = ux * fmaxf(0.0f, q_ho);
        }
    }
    {
        const float vy = face_flux_y_f(cfg, g, pi_sfc_f, v, k, j, i);
        const float q_up = (vy >= 0.0f) ? read_s_f(q, k, j - 1, i, nz, ny, nx, cfg.bc_x, cfg.bc_y) : read_s_f(q, k, j, i, nz, ny, nx, cfg.bc_x, cfg.bc_y);
        fyl_low = vy * q_up;
        if (cfg.bc_y != 1 && j == 0) {
            fyl_hi = fyl_low;
        } else {
            const float q_ho = horiz_face_val_y_f(q, k, j, i, vy, nz, ny, nx, cfg.bc_x, cfg.bc_y, cfg.adv_order);
            fyl_hi = vy * fmaxf(0.0f, q_ho);
        }
    }

    float fxr_low, fxr_hi, fyr_low, fyr_hi;
    {
        const float ux = face_flux_x_f(cfg, g, pi_sfc_f, u, k, j, i + 1);
        const float q_up = (ux >= 0.0f) ? read_s_f(q, k, j, i, nz, ny, nx, cfg.bc_x, cfg.bc_y) : read_s_f(q, k, j, i + 1, nz, ny, nx, cfg.bc_x, cfg.bc_y);
        fxr_low = ux * q_up;
        if (cfg.bc_x != 1 && i == nx - 1) {
            fxr_hi = fxr_low;
        } else {
            const float q_ho = horiz_face_val_x_f(q, k, j, i + 1, ux, nz, ny, nx, cfg.bc_x, cfg.bc_y, cfg.adv_order);
            fxr_hi = ux * fmaxf(0.0f, q_ho);
        }
    }
    {
        const float vy = face_flux_y_f(cfg, g, pi_sfc_f, v, k, j + 1, i);
        const float q_up = (vy >= 0.0f) ? read_s_f(q, k, j, i, nz, ny, nx, cfg.bc_x, cfg.bc_y) : read_s_f(q, k, j + 1, i, nz, ny, nx, cfg.bc_x, cfg.bc_y);
        fyr_low = vy * q_up;
        if (cfg.bc_y != 1 && j == ny - 1) {
            fyr_hi = fyr_low;
        } else {
            const float q_ho = horiz_face_val_y_f(q, k, j + 1, i, vy, nz, ny, nx, cfg.bc_x, cfg.bc_y, cfg.adv_order);
            fyr_hi = vy * fmaxf(0.0f, q_ho);
        }
    }

    const float q0 = fmaxf(0.0f, q[idx]);
    const float div_low = (fxr_low - fxl_low) / dx_f + (fyr_low - fyl_low) / dy_f;
    const float qlow_val = fmaxf(0.0f, q0 - (dt_q / vol) * div_low);
    q_low[idx] = qlow_val;

    const float ax_r = fxr_hi - fxr_low;
    const float ax_l = fxl_hi - fxl_low;
    const float ay_n = fyr_hi - fyr_low;
    const float ay_s = fyl_hi - fyl_low;

    const float p_out = dt_q * (
        (fmaxf(0.0f, ax_r) + fmaxf(0.0f, -ax_l)) / dx_f +
        (fmaxf(0.0f, ay_n) + fmaxf(0.0f, -ay_s)) / dy_f
    );
    const float q_avail = vol * qlow_val;
    r_out[idx] = (p_out > 1e-24f) ? fminf(1.0f, 0.999999f * (q_avail / p_out)) : 1.0f;
}

__global__ void nh_tracer_fct_pass2_kernel(
    NhGridConfig cfg,
    DeviceStaticGrid g,
    float dt_q,
    float pi_sfc_f,
    const float* u,
    const float* v,
    const float* q_old,
    const float* q_low,
    const float* r_out,
    float* q_new
) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    const int32_t k = blockIdx.z;
    if (k >= cfg.nz || j >= cfg.ny || i >= cfg.nx) return;

    const int32_t nz = cfg.nz, ny = cfg.ny, nx = cfg.nx;
    const size_t idx = static_cast<size_t>(k) * ny * nx + static_cast<size_t>(j) * nx + i;
    const float dx_f = static_cast<float>(cfg.dx_m);
    const float dy_f = static_cast<float>(cfg.dy_m);
    const float vol = fmaxf(cell_mass_vol_f(cfg, g, pi_sfc_f, k, j, i), 1e-12f);

    auto limited_x = [&](int32_t iface) -> float {
        if (cfg.bc_x != 1 && (iface <= 0 || iface >= nx)) return 0.0f;
        const float ux = face_flux_x_f(cfg, g, pi_sfc_f, u, k, j, iface);
        const float q_up = (ux >= 0.0f)
            ? read_s_f(q_old, k, j, iface - 1, nz, ny, nx, cfg.bc_x, cfg.bc_y)
            : read_s_f(q_old, k, j, iface,     nz, ny, nx, cfg.bc_x, cfg.bc_y);
        const float q_ho = horiz_face_val_x_f(q_old, k, j, iface, ux, nz, ny, nx, cfg.bc_x, cfg.bc_y, cfg.adv_order);
        const float A = ux * (fmaxf(0.0f, q_ho) - q_up);
        const float r_donor = (A >= 0.0f)
            ? read_s_f(r_out, k, j, iface - 1, nz, ny, nx, cfg.bc_x, cfg.bc_y)
            : read_s_f(r_out, k, j, iface,     nz, ny, nx, cfg.bc_x, cfg.bc_y);
        return A * r_donor;
    };
    auto limited_y = [&](int32_t jface) -> float {
        if (cfg.bc_y != 1 && (jface <= 0 || jface >= ny)) return 0.0f;
        const float vy = face_flux_y_f(cfg, g, pi_sfc_f, v, k, jface, i);
        const float q_up = (vy >= 0.0f)
            ? read_s_f(q_old, k, jface - 1, i, nz, ny, nx, cfg.bc_x, cfg.bc_y)
            : read_s_f(q_old, k, jface,     i, nz, ny, nx, cfg.bc_x, cfg.bc_y);
        const float q_ho = horiz_face_val_y_f(q_old, k, jface, i, vy, nz, ny, nx, cfg.bc_x, cfg.bc_y, cfg.adv_order);
        const float A = vy * (fmaxf(0.0f, q_ho) - q_up);
        const float r_donor = (A >= 0.0f)
            ? read_s_f(r_out, k, jface - 1, i, nz, ny, nx, cfg.bc_x, cfg.bc_y)
            : read_s_f(r_out, k, jface,     i, nz, ny, nx, cfg.bc_x, cfg.bc_y);
        return A * r_donor;
    };

    const float ax_r = limited_x(i + 1);
    const float ax_l = limited_x(i);
    const float ay_n = limited_y(j + 1);
    const float ay_s = limited_y(j);

    const float div_anti = (ax_r - ax_l) / dx_f + (ay_n - ay_s) / dy_f;
    const float qn = q_low[idx] - (dt_q / vol) * div_anti;
    q_new[idx] = fmaxf(0.0f, qn);
}

__global__ void nh_tracer_vert_column_kernel(
    NhGridConfig cfg,
    DeviceStaticGrid g,
    float dt_s,
    float pi_sfc_f,
    const float* u,
    const float* v,
    const float* w,
    float* q,
    float* q_low_buf,
    float* r_out_buf,
    double* tri_c,
    double* tri_d
) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= cfg.ny || i >= cfg.nx) return;

    const int32_t nz = cfg.nz, ny = cfg.ny, nx = cfg.nx;
    const size_t col_stride = static_cast<size_t>(ny) * nx;
    const size_t col = static_cast<size_t>(j) * nx + i;

    float col_cfl_z = 0.0f;
    float fz_bot = 0.0f;
    for (int32_t k = 0; k < nz; ++k) {
        const size_t idx = static_cast<size_t>(k) * col_stride + col;
        const float fz_top = (k < nz - 1) ? face_flux_z_f(cfg, g, pi_sfc_f, u, v, w, k + 1, j, i) : 0.0f;
        tri_c[idx] = static_cast<double>(fz_top);
        const float vol = fmaxf(cell_mass_vol_f(cfg, g, pi_sfc_f, k, j, i), 1e-12f);
        const float cfl_k = dt_s * (fmaxf(0.0f, fz_top) + fmaxf(0.0f, -fz_bot)) / vol;
        col_cfl_z = fmaxf(col_cfl_z, cfl_k);
        fz_bot = fz_top;
    }
    if (col_cfl_z <= 1e-12f) return;

    if (col_cfl_z <= 0.65f) {
        // Exact 1D conservative positive-definite Koren-TVD Zalesak FCT on full dt_s
        float fzl_low = 0.0f, fzl_hi = 0.0f;
        for (int32_t k = 0; k < nz; ++k) {
            const size_t idx = static_cast<size_t>(k) * col_stride + col;
            const float fz_top = static_cast<float>(tri_c[idx]);
            float fzr_low = 0.0f, fzr_hi = 0.0f;
            if (k < nz - 1) {
                const float q_up = (fz_top >= 0.0f)
                    ? read_s_f(q, k, j, i, nz, ny, nx, cfg.bc_x, cfg.bc_y)
                    : read_s_f(q, k + 1, j, i, nz, ny, nx, cfg.bc_x, cfg.bc_y);
                const float q_ho = vert_tvd_face_val_f(q, k + 1, j, i, fz_top, g.dzeta, nz, ny, nx, cfg.bc_x, cfg.bc_y);
                fzr_low = fz_top * q_up;
                fzr_hi = fz_top * fmaxf(0.0f, q_ho);
            }
            const float vol = fmaxf(cell_mass_vol_f(cfg, g, pi_sfc_f, k, j, i), 1e-12f);
            const float q0 = fmaxf(0.0f, q[idx]);
            const float qlow_val = fmaxf(0.0f, q0 - (dt_s / vol) * (fzr_low - fzl_low));
            q_low_buf[idx] = qlow_val;
            const float az_t = fzr_hi - fzr_low;
            const float az_b = fzl_hi - fzl_low;
            const float p_out = dt_s * (fmaxf(0.0f, az_t) + fmaxf(0.0f, -az_b));
            r_out_buf[idx] = (p_out > 1e-24f) ? fminf(1.0f, 0.999999f * ((vol * qlow_val) / p_out)) : 1.0f;
            fzl_low = fzr_low;
            fzl_hi = fzr_hi;
        }
        float az_b = 0.0f;
        for (int32_t k = 0; k < nz; ++k) {
            const size_t idx = static_cast<size_t>(k) * col_stride + col;
            const float fz_top = static_cast<float>(tri_c[idx]);
            float az_t = 0.0f;
            if (k < nz - 1) {
                const float q_up = (fz_top >= 0.0f)
                    ? read_s_f(q, k, j, i, nz, ny, nx, cfg.bc_x, cfg.bc_y)
                    : read_s_f(q, k + 1, j, i, nz, ny, nx, cfg.bc_x, cfg.bc_y);
                const float q_ho = vert_tvd_face_val_f(q, k + 1, j, i, fz_top, g.dzeta, nz, ny, nx, cfg.bc_x, cfg.bc_y);
                const float A = fz_top * (fmaxf(0.0f, q_ho) - q_up);
                const float r_donor = (A >= 0.0f)
                    ? r_out_buf[idx]
                    : r_out_buf[static_cast<size_t>(k + 1) * col_stride + col];
                az_t = A * r_donor;
            }
            const float vol = fmaxf(cell_mass_vol_f(cfg, g, pi_sfc_f, k, j, i), 1e-12f);
            tri_d[idx] = static_cast<double>(fmaxf(0.0f, q_low_buf[idx] - (dt_s / vol) * (az_t - az_b)));
            az_b = az_t;
        }
        for (int32_t k = 0; k < nz; ++k) {
            const size_t idx = static_cast<size_t>(k) * col_stride + col;
            q[idx] = static_cast<float>(tri_d[idx]);
        }
        return;
    }

    // Exact conservative, strictly column-diagonally-dominant backward-Euler M-matrix Thomas solve (CFL_z > 0.65)
    double fz_bot_d = 0.0;
    for (int32_t k = 0; k < nz; ++k) {
        const size_t idx = static_cast<size_t>(k) * col_stride + col;
        const double fz_top_d = (k < nz - 1) ? tri_c[idx] : 0.0;
        const double vol = static_cast<double>(fmaxf(cell_mass_vol_f(cfg, g, pi_sfc_f, k, j, i), 1e-12f));
        const double A_k = (k > 0) ? (-static_cast<double>(dt_s) * fmax(0.0, fz_bot_d)) : 0.0;
        const double C_k = (k < nz - 1) ? (-static_cast<double>(dt_s) * fmax(0.0, -fz_top_d)) : 0.0;
        const double B_k = vol + static_cast<double>(dt_s) * (fmax(0.0, fz_top_d) + fmax(0.0, -fz_bot_d));
        const double rhs = vol * static_cast<double>(fmaxf(0.0f, q[idx]));
        if (k == 0) {
            tri_c[idx] = C_k / B_k;
            tri_d[idx] = rhs / B_k;
        } else {
            const size_t idx_m = static_cast<size_t>(k - 1) * col_stride + col;
            const double denom = B_k - A_k * tri_c[idx_m];
            tri_c[idx] = (k < nz - 1) ? (C_k / denom) : 0.0;
            tri_d[idx] = (rhs - A_k * tri_d[idx_m]) / denom;
        }
        fz_bot_d = fz_top_d;
    }
    double q_next = tri_d[static_cast<size_t>(nz - 1) * col_stride + col];
    q[static_cast<size_t>(nz - 1) * col_stride + col] = fmaxf(0.0f, static_cast<float>(q_next));
    for (int32_t k = nz - 2; k >= 0; --k) {
        const size_t idx = static_cast<size_t>(k) * col_stride + col;
        q_next = tri_d[idx] - tri_c[idx] * q_next;
        q[idx] = fmaxf(0.0f, static_cast<float>(q_next));
    }
}

__global__ void nh_sync_periodic_faces_kernel(
    NhGridConfig cfg,
    float* u,
    float* v
) {
    const int32_t idx_xy = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t k = blockIdx.z;
    if (k >= cfg.nz) return;
    if (cfg.bc_x == 1 && idx_xy < cfg.ny) {
        const size_t base = static_cast<size_t>(k) * cfg.ny * (cfg.nx + 1) + static_cast<size_t>(idx_xy) * (cfg.nx + 1);
        u[base + cfg.nx] = u[base];
    }
    if (cfg.bc_y == 1 && idx_xy < cfg.nx) {
        const size_t base = static_cast<size_t>(k) * (cfg.ny + 1) * cfg.nx + idx_xy;
        v[base + static_cast<size_t>(cfg.ny) * cfg.nx] = v[base];
    }
}

__global__ void nh_estimate_cs_max_kernel(
    NhGridConfig cfg,
    DeviceStaticGrid g,
    const float* thp,
    const float* pip,
    uint32_t* max_cfl_u32
) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    const int32_t k = blockIdx.z;
    float cs_msf = 0.0f;
    if (k < cfg.nz && j < cfg.ny && i < cfg.nx) {
        const size_t idx = static_cast<size_t>(k) * cfg.ny * cfg.nx + j * cfg.nx + i;
        const float z = static_cast<float>(g.zeta_s[k] + g.b_s[k] * g.hs[j * cfg.nx + i]);
        const float pi_sfc_f = powf(static_cast<float>(cfg.p_sfc_pa) / P0_PA_F, KAPPA_F);
        float thb = 0.0f, pib = 0.0f;
        eval_theta_pi_bar_f32(
            z, static_cast<float>(cfg.theta_sfc_k), pi_sfc_f, static_cast<float>(cfg.n_ref_s), thb, pib
        );
        const float t_tot = fmaxf(50.0f, (thb + thp[idx]) * fmaxf(1e-4f, pib + pip[idx]));
        const float cs = sqrtf(GAMMA_GAS_F * R_D_F * t_tot);
        float msf_eff = static_cast<float>(g.msf[j * cfg.nx + i]);
        if (cfg.is_spherical && cfg.polar_filter_lat_deg > 0.0 && cfg.polar_filter_lat_deg < 89.9) {
            const float cos_crit = cosf(static_cast<float>(cfg.polar_filter_lat_deg) * (PI_CONST_F / 180.0f));
            msf_eff = fminf(msf_eff, 1.0f / fmaxf(cos_crit, 1e-3f));
        }
        cs_msf = cs * msf_eff;
    }
    const unsigned int lane = threadIdx.x & 31;
    for (int offset = 16; offset > 0; offset >>= 1) {
        cs_msf = fmaxf(cs_msf, __shfl_down_sync(0xffffffffu, cs_msf, offset));
    }
    if (lane == 0 && cs_msf > 0.0f) {
        atomicMax(max_cfl_u32, __float_as_uint(cs_msf));
    }
}

__global__ void nh_lbc_relax_qv_kernel(
    NhGridConfig cfg,
    float dt_s,
    const float* lbc_qv,
    float* qv
) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    const int32_t k = blockIdx.z;
    if (k >= cfg.nz || j >= cfg.ny || i >= cfg.nx || lbc_qv == nullptr) return;
    if (cfg.lbc_tau_max_s <= 0.0 || cfg.lbc_rim_width <= 0) return;

    const float tau_max = static_cast<float>(cfg.lbc_tau_max_s);
    float tau = 0.0f;
    if (cfg.bc_x == 2) {
        if (cfg.lbc_edges & 1) tau = fmaxf(tau, davies_weight_f(i, cfg.lbc_rim_width, tau_max));
        if (cfg.lbc_edges & 2) tau = fmaxf(tau, davies_weight_f(cfg.nx - 1 - i, cfg.lbc_rim_width, tau_max));
    }
    if (cfg.bc_y == 2) {
        if (cfg.lbc_edges & 4) tau = fmaxf(tau, davies_weight_f(j, cfg.lbc_rim_width, tau_max));
        if (cfg.lbc_edges & 8) tau = fmaxf(tau, davies_weight_f(cfg.ny - 1 - j, cfg.lbc_rim_width, tau_max));
    }
    if (tau <= 0.0f) return;

    const size_t idx = static_cast<size_t>(k) * cfg.ny * cfg.nx + static_cast<size_t>(j) * cfg.nx + i;
    const float q_cur = qv[idx];
    const float q_tgt = fmaxf(0.0f, lbc_qv[idx]);
    const float alpha = 1.0f - expf(-tau * dt_s);
    qv[idx] = fmaxf(0.0f, q_cur + alpha * (q_tgt - q_cur));
}

inline int32_t advect_one_tracer(
    NhCudaHandle* h,
    float* q_ptr,
    float dt_full_s,
    int32_t n_sub_h,
    float dt_h,
    float pi_sfc_f
);

} // namespace

namespace {

inline DeviceStaticGrid static_view(const NhCudaHandle* h) {
    return DeviceStaticGrid{
        h->d_zeta_w, h->d_zeta_s, h->d_dzeta,
        h->d_b_w, h->d_b_s, h->d_db_w, h->d_db_s,
        h->d_hs, h->d_msf, h->d_fcor
    };
}

inline int32_t advect_one_tracer(
    NhCudaHandle* h,
    float* q_ptr,
    float dt_full_s,
    int32_t n_sub_h,
    float dt_h,
    float pi_sfc_f
) {
    if (!q_ptr || dt_full_s <= 0.0f || n_sub_h < 1) return 0;
    const int32_t nz = h->cfg.nz, ny = h->cfg.ny, nx = h->cfg.nx;
    const DeviceStaticGrid g = static_view(h);
    dim3 b3(32, 8, 1);
    dim3 g3((nx + 31) / 32, (ny + 7) / 8, nz);
    dim3 gcol((nx + 31) / 32, (ny + 7) / 8, 1);

    for (int32_t s = 0; s < n_sub_h; ++s) {
        nh_tracer_fct_pass1_kernel<<<g3, b3>>>(
            h->cfg, g, dt_h, pi_sfc_f, h->d_u, h->d_v, q_ptr, h->d_q_tmp, h->d_div
        );
        CUDA_CHECK(cudaGetLastError());
        nh_tracer_fct_pass2_kernel<<<g3, b3>>>(
            h->cfg, g, dt_h, pi_sfc_f, h->d_u, h->d_v, q_ptr, h->d_q_tmp, h->d_div, h->d_q_tmp
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(q_ptr, h->d_q_tmp, h->n_s * sizeof(float), cudaMemcpyDeviceToDevice));
    }

    nh_tracer_vert_column_kernel<<<gcol, b3>>>(
        h->cfg, g, dt_full_s, pi_sfc_f, h->d_u, h->d_v, h->d_w, q_ptr,
        h->d_q_tmp, h->d_div, h->d_tri_c, h->d_tri_d
    );
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

inline int32_t copy_state_to_n(NhCudaHandle* h) {
    CUDA_CHECK(cudaMemcpy(h->d_u_n, h->d_u, h->n_u * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(h->d_v_n, h->d_v, h->n_v * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(h->d_w_n, h->d_w, h->n_w * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(h->d_thp_n, h->d_thp, h->n_s * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(h->d_pip_n, h->d_pip, h->n_s * sizeof(float), cudaMemcpyDeviceToDevice));
    return 0;
}

inline int32_t reset_state_from_n(NhCudaHandle* h) {
    CUDA_CHECK(cudaMemcpy(h->d_u, h->d_u_n, h->n_u * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(h->d_v, h->d_v_n, h->n_v * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(h->d_w, h->d_w_n, h->n_w * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(h->d_thp, h->d_thp_n, h->n_s * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(h->d_pip, h->d_pip_n, h->n_s * sizeof(float), cudaMemcpyDeviceToDevice));
    return 0;
}

__global__ void nh_polar_pack_kernel(
    const float* src,
    float* dst,
    const int32_t* rows,
    int32_t n_rows,
    int32_t nz,
    int32_t ny,
    int32_t nx,
    int32_t stride_x
) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t r = blockIdx.y * blockDim.y + threadIdx.y;
    const int32_t k = blockIdx.z;
    if (k >= nz || r >= n_rows || i >= nx) return;
    const int32_t j = rows[r];
    dst[(static_cast<size_t>(k) * n_rows + r) * nx + i] =
        src[(static_cast<size_t>(k) * ny + j) * stride_x + i];
}

__global__ void nh_polar_diff_kernel(
    const float* src_after,
    const float* saved_before,
    float* dst_inc,
    const int32_t* rows,
    int32_t n_rows,
    int32_t nz,
    int32_t ny,
    int32_t nx,
    int32_t stride_x
) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t r = blockIdx.y * blockDim.y + threadIdx.y;
    const int32_t k = blockIdx.z;
    if (k >= nz || r >= n_rows || i >= nx) return;
    const int32_t j = rows[r];
    const size_t pidx = (static_cast<size_t>(k) * n_rows + r) * nx + i;
    dst_inc[pidx] = src_after[(static_cast<size_t>(k) * ny + j) * stride_x + i] - saved_before[pidx];
}

__global__ void nh_polar_spec_filter_kernel(
    cufftComplex* spec,
    const float* row_cos,
    int32_t n_rows,
    int32_t nz,
    int32_t nx,
    int32_t n_freq,
    float cos_crit
) {
    const int32_t m = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t r = blockIdx.y * blockDim.y + threadIdx.y;
    const int32_t k = blockIdx.z;
    if (k >= nz || r >= n_rows || m >= n_freq) return;

    float scale = 1.0f / static_cast<float>(nx);
    if (m >= 1) {
        const float c_row = fmaxf(row_cos[r], 0.002f);
        const float s_arg = sinf((PI_CONST_F * static_cast<float>(m)) / static_cast<float>(nx));
        float damp = fminf(1.0f, (c_row / fmaxf(cos_crit, 1e-3f)) / fmaxf(s_arg, 1e-6f));
        if (c_row < 0.03f) damp *= damp;
        scale *= damp;
    }
    const size_t cidx = (static_cast<size_t>(k) * n_rows + r) * n_freq + m;
    spec[cidx].x *= scale;
    spec[cidx].y *= scale;
}

__global__ void nh_polar_unpack_kernel(
    const float* filtered,
    const float* saved_before,
    float* dst,
    const int32_t* rows,
    int32_t n_rows,
    int32_t nz,
    int32_t ny,
    int32_t nx,
    int32_t stride_x,
    int32_t sync_periodic_u
) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t r = blockIdx.y * blockDim.y + threadIdx.y;
    const int32_t k = blockIdx.z;
    if (k >= nz || r >= n_rows || i >= nx) return;
    const int32_t j = rows[r];
    const size_t pidx = (static_cast<size_t>(k) * n_rows + r) * nx + i;
    const float val = filtered[pidx] + (saved_before ? saved_before[pidx] : 0.0f);
    const size_t base = (static_cast<size_t>(k) * ny + j) * stride_x;
    dst[base + i] = val;
    if (sync_periodic_u && i == 0) {
        dst[base + nx] = val;
    }
}

static void free_polar_fft(NhCudaHandle* h) {
    if (!h) return;
    if (h->fft_r2c_s) { cufftDestroy(h->fft_r2c_s); h->fft_r2c_s = 0; }
    if (h->fft_c2r_s) { cufftDestroy(h->fft_c2r_s); h->fft_c2r_s = 0; }
    if (h->fft_r2c_v) { cufftDestroy(h->fft_r2c_v); h->fft_r2c_v = 0; }
    if (h->fft_c2r_v) { cufftDestroy(h->fft_c2r_v); h->fft_c2r_v = 0; }
    auto cf = [](void* p) { if (p) cudaFree(p); };
    cf(h->d_polar_rows_s); h->d_polar_rows_s = nullptr;
    cf(h->d_polar_rows_v); h->d_polar_rows_v = nullptr;
    cf(h->d_polar_cos_s);  h->d_polar_cos_s = nullptr;
    cf(h->d_polar_cos_v);  h->d_polar_cos_v = nullptr;
    cf(h->d_fft_real);     h->d_fft_real = nullptr;
    cf(h->d_polar_save);   h->d_polar_save = nullptr;
    cf(h->d_fft_cplx);     h->d_fft_cplx = nullptr;
    h->fft_ready = 0;
    h->fft_n_rows_s = 0;
    h->fft_n_rows_v = 0;
}

static int32_t refresh_polar_fft(NhCudaHandle* h, const double* host_msf) {
    free_polar_fft(h);
    if (!h || !h->cfg.is_spherical || h->cfg.bc_x != 1 ||
        h->cfg.polar_filter_lat_deg <= 0.0 || h->cfg.polar_filter_lat_deg >= 89.9 ||
        h->cfg.nx < 8 || !host_msf) {
        return 0;
    }

    const double cos_crit = cos(h->cfg.polar_filter_lat_deg * (PI_CONST / 180.0));
    const int32_t nz = h->cfg.nz, ny = h->cfg.ny, nx = h->cfg.nx;
    std::vector<int32_t> rows_s, rows_v;
    std::vector<float> cos_s, cos_v;

    for (int32_t j = 0; j < ny; ++j) {
        const double c = 1.0 / fmax(host_msf[static_cast<size_t>(j) * nx], 1.0);
        if (c < cos_crit) {
            rows_s.push_back(j);
            cos_s.push_back(static_cast<float>(c));
        }
    }
    for (int32_t j = 1; j < ny; ++j) {
        const double cm = 1.0 / fmax(host_msf[static_cast<size_t>(j - 1) * nx], 1.0);
        const double cp = 1.0 / fmax(host_msf[static_cast<size_t>(j) * nx], 1.0);
        const double c = 0.5 * (cm + cp);
        if (c < cos_crit) {
            rows_v.push_back(j);
            cos_v.push_back(static_cast<float>(c));
        }
    }

    h->fft_n_rows_s = static_cast<int32_t>(rows_s.size());
    h->fft_n_rows_v = static_cast<int32_t>(rows_v.size());
    const int32_t max_rows = (h->fft_n_rows_s > h->fft_n_rows_v) ? h->fft_n_rows_s : h->fft_n_rows_v;
    if (max_rows <= 0) {
        h->fft_ready = 1;
        return 0;
    }

    const int32_t n_freq = nx / 2 + 1;
    const size_t real_elems = static_cast<size_t>(nz) * static_cast<size_t>(max_rows) * static_cast<size_t>(nx);
    const size_t cplx_elems = static_cast<size_t>(nz) * static_cast<size_t>(max_rows) * static_cast<size_t>(n_freq);
    CUDA_CHECK(cudaMalloc(&h->d_fft_real, real_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&h->d_polar_save, 3ULL * real_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&h->d_fft_cplx, cplx_elems * sizeof(cufftComplex)));

    if (h->fft_n_rows_s > 0) {
        CUDA_CHECK(cudaMalloc(&h->d_polar_rows_s, h->fft_n_rows_s * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&h->d_polar_cos_s, h->fft_n_rows_s * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(h->d_polar_rows_s, rows_s.data(), h->fft_n_rows_s * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(h->d_polar_cos_s, cos_s.data(), h->fft_n_rows_s * sizeof(float), cudaMemcpyHostToDevice));
        if (cufftPlan1d(&h->fft_r2c_s, nx, CUFFT_R2C, nz * h->fft_n_rows_s) != CUFFT_SUCCESS ||
            cufftPlan1d(&h->fft_c2r_s, nx, CUFFT_C2R, nz * h->fft_n_rows_s) != CUFFT_SUCCESS) {
            set_error("cufftPlan1d failed for scalar polar filter");
            free_polar_fft(h);
            return -1;
        }
    }
    if (h->fft_n_rows_v > 0) {
        CUDA_CHECK(cudaMalloc(&h->d_polar_rows_v, h->fft_n_rows_v * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&h->d_polar_cos_v, h->fft_n_rows_v * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(h->d_polar_rows_v, rows_v.data(), h->fft_n_rows_v * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(h->d_polar_cos_v, cos_v.data(), h->fft_n_rows_v * sizeof(float), cudaMemcpyHostToDevice));
        if (cufftPlan1d(&h->fft_r2c_v, nx, CUFFT_R2C, nz * h->fft_n_rows_v) != CUFFT_SUCCESS ||
            cufftPlan1d(&h->fft_c2r_v, nx, CUFFT_C2R, nz * h->fft_n_rows_v) != CUFFT_SUCCESS) {
            set_error("cufftPlan1d failed for meridional polar filter");
            free_polar_fft(h);
            return -1;
        }
    }

    h->fft_ready = 1;
    return 0;
}

static int32_t save_polar_snapshot(
    NhCudaHandle* h,
    const float* field,
    int32_t ny_field,
    int32_t stride_x,
    int32_t is_v,
    int32_t slot
) {
    if (!h || !h->fft_ready) return 0;
    const int32_t n_rows = is_v ? h->fft_n_rows_v : h->fft_n_rows_s;
    const int32_t* rows = is_v ? h->d_polar_rows_v : h->d_polar_rows_s;
    if (n_rows <= 0) return 0;

    const int32_t nz = h->cfg.nz, nx = h->cfg.nx;
    const int32_t max_rows = (h->fft_n_rows_s > h->fft_n_rows_v) ? h->fft_n_rows_s : h->fft_n_rows_v;
    const size_t slot_elems = static_cast<size_t>(nz) * max_rows * nx;
    float* dst = h->d_polar_save + static_cast<size_t>(slot) * slot_elems;
    dim3 b(32, 4, 1);
    dim3 g((nx + 31) / 32, (n_rows + 3) / 4, nz);
    nh_polar_pack_kernel<<<g, b>>>(field, dst, rows, n_rows, nz, ny_field, nx, stride_x);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

static int32_t filter_polar_field(
    NhCudaHandle* h,
    float* field,
    int32_t ny_field,
    int32_t stride_x,
    int32_t is_v,
    int32_t is_u,
    int32_t from_snapshot_slot
) {
    if (!h || !h->fft_ready) return 0;
    const int32_t n_rows = is_v ? h->fft_n_rows_v : h->fft_n_rows_s;
    const int32_t* rows = is_v ? h->d_polar_rows_v : h->d_polar_rows_s;
    const float* row_cos = is_v ? h->d_polar_cos_v : h->d_polar_cos_s;
    cufftHandle r2c = is_v ? h->fft_r2c_v : h->fft_r2c_s;
    cufftHandle c2r = is_v ? h->fft_c2r_v : h->fft_c2r_s;
    if (n_rows <= 0) return 0;

    const int32_t nz = h->cfg.nz, nx = h->cfg.nx, n_freq = nx / 2 + 1;
    const float cos_crit = cosf(static_cast<float>(h->cfg.polar_filter_lat_deg) * (PI_CONST_F / 180.0f));
    const int32_t max_rows = (h->fft_n_rows_s > h->fft_n_rows_v) ? h->fft_n_rows_s : h->fft_n_rows_v;
    const size_t slot_elems = static_cast<size_t>(nz) * max_rows * nx;
    const float* saved_ptr = (from_snapshot_slot >= 0) ? (h->d_polar_save + static_cast<size_t>(from_snapshot_slot) * slot_elems) : nullptr;

    dim3 b(32, 4, 1);
    dim3 gp((nx + 31) / 32, (n_rows + 3) / 4, nz);
    if (saved_ptr) {
        nh_polar_diff_kernel<<<gp, b>>>(field, saved_ptr, h->d_fft_real, rows, n_rows, nz, ny_field, nx, stride_x);
    } else {
        nh_polar_pack_kernel<<<gp, b>>>(field, h->d_fft_real, rows, n_rows, nz, ny_field, nx, stride_x);
    }
    CUDA_CHECK(cudaGetLastError());

    if (cufftExecR2C(r2c, reinterpret_cast<cufftReal*>(h->d_fft_real), h->d_fft_cplx) != CUFFT_SUCCESS) {
        set_error("cufftExecR2C failed in polar filter");
        return -1;
    }

    dim3 gs((n_freq + 31) / 32, (n_rows + 3) / 4, nz);
    nh_polar_spec_filter_kernel<<<gs, b>>>(h->d_fft_cplx, row_cos, n_rows, nz, nx, n_freq, cos_crit);
    CUDA_CHECK(cudaGetLastError());

    if (cufftExecC2R(c2r, h->d_fft_cplx, reinterpret_cast<cufftReal*>(h->d_fft_real)) != CUFFT_SUCCESS) {
        set_error("cufftExecC2R failed in polar filter");
        return -1;
    }

    nh_polar_unpack_kernel<<<gp, b>>>(h->d_fft_real, saved_ptr, field, rows, n_rows, nz, ny_field, nx, stride_x, is_u ? 1 : 0);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

} // namespace

void nh_internal_set_error(const char* msg) {
    set_error(msg);
}

void nh_internal_set_cuda_error(const char* file, int line, cudaError_t err) {
    snprintf(g_last_error, sizeof(g_last_error),
             "CUDA error (%s:%d): %s", file, line, cudaGetErrorString(err));
}

extern "C" {

const char* nh_cuda_last_error(void) {
    return g_last_error;
}

int32_t nh_cuda_vram_info(uint64_t* free_bytes, uint64_t* total_bytes) {
    size_t f = 0, t = 0;
    CUDA_CHECK(cudaMemGetInfo(&f, &t));
    if (free_bytes) *free_bytes = static_cast<uint64_t>(f);
    if (total_bytes) *total_bytes = static_cast<uint64_t>(t);
    return 0;
}

uint64_t nh_cuda_estimate_vram_bytes_ex(
    int32_t nz, int32_t ny, int32_t nx,
    int32_t has_moist, int32_t has_lbc, int32_t has_iau, int32_t has_physics
) {
    const uint64_t cells = static_cast<uint64_t>(nz) * static_cast<uint64_t>(ny) * static_cast<uint64_t>(nx);
    // Dry core: 5 state + 5 backup + 5 rhs + 2 scratch (div, q_tmp) = 17 float32 + 2 float64 tridiag (4 float eq) = 21 float32 eq
    uint64_t float32_arrays = 21;
    if (has_moist) float32_arrays += 6;
    if (has_lbc) float32_arrays += (has_moist ? 5 : 4);
    if (has_iau) float32_arrays += 4;
    if (has_physics) float32_arrays += 1; // d_rad_dth_dt
    const uint64_t field_bytes = cells * float32_arrays * 4ULL;
    uint64_t static_2d = static_cast<uint64_t>(ny) * static_cast<uint64_t>(nx) * 24ULL;
    if (has_physics) {
        // 9 static + 22 surface 2D + 8 soil (4*2) + 3 intra-step scratch = 42 float32 arrays per column
        static_2d += static_cast<uint64_t>(ny) * static_cast<uint64_t>(nx) * 42ULL * 4ULL;
    }
    const uint64_t overhead = 128ULL * 1024ULL * 1024ULL; // CUDA runtime context + margin
    return field_bytes + static_2d + overhead;
}

uint64_t nh_cuda_estimate_vram_bytes(
    int32_t nz, int32_t ny, int32_t nx,
    int32_t has_moist, int32_t has_lbc, int32_t has_iau
) {
    return nh_cuda_estimate_vram_bytes_ex(nz, ny, nx, has_moist, has_lbc, has_iau, 0);
}

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
) {
    if (!cfg || !zeta_w || !zeta_s || !dzeta || !b_w || !b_s || !db_dzeta_w || !db_dzeta_s || !hs_m || !msf || !f_cor) {
        set_error("Null pointer passed to nh_cuda_create");
        return nullptr;
    }
    if (cfg->nz < 4 || cfg->ny < 2 || cfg->nx < 2) {
        set_error("Invalid grid dimensions in nh_cuda_create");
        return nullptr;
    }

    NhCudaHandle* h = new NhCudaHandle();
    h->cfg = *cfg;
    const size_t nz = static_cast<size_t>(cfg->nz);
    const size_t ny = static_cast<size_t>(cfg->ny);
    const size_t nx = static_cast<size_t>(cfg->nx);
    h->n_s = nz * ny * nx;
    h->n_u = nz * ny * (nx + 1);
    h->n_v = nz * (ny + 1) * nx;
    h->n_w = (nz + 1) * ny * nx;
    const size_t n2 = ny * nx;

    CUDA_CHECK_PTR(cudaMalloc(&h->d_zeta_w, (nz + 1) * sizeof(double)));
    CUDA_CHECK_PTR(cudaMalloc(&h->d_zeta_s, nz * sizeof(double)));
    CUDA_CHECK_PTR(cudaMalloc(&h->d_dzeta, nz * sizeof(double)));
    CUDA_CHECK_PTR(cudaMalloc(&h->d_b_w, (nz + 1) * sizeof(double)));
    CUDA_CHECK_PTR(cudaMalloc(&h->d_b_s, nz * sizeof(double)));
    CUDA_CHECK_PTR(cudaMalloc(&h->d_db_w, (nz + 1) * sizeof(double)));
    CUDA_CHECK_PTR(cudaMalloc(&h->d_db_s, nz * sizeof(double)));
    CUDA_CHECK_PTR(cudaMalloc(&h->d_hs, n2 * sizeof(double)));
    CUDA_CHECK_PTR(cudaMalloc(&h->d_msf, n2 * sizeof(double)));
    CUDA_CHECK_PTR(cudaMalloc(&h->d_fcor, n2 * sizeof(double)));

    CUDA_CHECK_PTR(cudaMemcpy(h->d_zeta_w, zeta_w, (nz + 1) * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK_PTR(cudaMemcpy(h->d_zeta_s, zeta_s, nz * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK_PTR(cudaMemcpy(h->d_dzeta, dzeta, nz * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK_PTR(cudaMemcpy(h->d_b_w, b_w, (nz + 1) * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK_PTR(cudaMemcpy(h->d_b_s, b_s, nz * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK_PTR(cudaMemcpy(h->d_db_w, db_dzeta_w, (nz + 1) * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK_PTR(cudaMemcpy(h->d_db_s, db_dzeta_s, nz * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK_PTR(cudaMemcpy(h->d_hs, hs_m, n2 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK_PTR(cudaMemcpy(h->d_msf, msf, n2 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK_PTR(cudaMemcpy(h->d_fcor, f_cor, n2 * sizeof(double), cudaMemcpyHostToDevice));

    CUDA_CHECK_PTR(cudaMalloc(&h->d_u, h->n_u * sizeof(float)));
    CUDA_CHECK_PTR(cudaMalloc(&h->d_v, h->n_v * sizeof(float)));
    CUDA_CHECK_PTR(cudaMalloc(&h->d_w, h->n_w * sizeof(float)));
    CUDA_CHECK_PTR(cudaMalloc(&h->d_thp, h->n_s * sizeof(float)));
    CUDA_CHECK_PTR(cudaMalloc(&h->d_pip, h->n_s * sizeof(float)));

    CUDA_CHECK_PTR(cudaMalloc(&h->d_u_n, h->n_u * sizeof(float)));
    CUDA_CHECK_PTR(cudaMalloc(&h->d_v_n, h->n_v * sizeof(float)));
    CUDA_CHECK_PTR(cudaMalloc(&h->d_w_n, h->n_w * sizeof(float)));
    CUDA_CHECK_PTR(cudaMalloc(&h->d_thp_n, h->n_s * sizeof(float)));
    CUDA_CHECK_PTR(cudaMalloc(&h->d_pip_n, h->n_s * sizeof(float)));

    CUDA_CHECK_PTR(cudaMalloc(&h->d_ru, h->n_u * sizeof(float)));
    CUDA_CHECK_PTR(cudaMalloc(&h->d_rv, h->n_v * sizeof(float)));
    CUDA_CHECK_PTR(cudaMalloc(&h->d_rw, h->n_w * sizeof(float)));
    CUDA_CHECK_PTR(cudaMalloc(&h->d_rth, h->n_s * sizeof(float)));
    CUDA_CHECK_PTR(cudaMalloc(&h->d_rpi, h->n_s * sizeof(float)));

    CUDA_CHECK_PTR(cudaMalloc(&h->d_div, h->n_s * sizeof(float)));
    CUDA_CHECK_PTR(cudaMalloc(&h->d_q_tmp, h->n_s * sizeof(float)));
    const size_t n_tri = max(h->n_w, max(h->n_u, h->n_v));
    CUDA_CHECK_PTR(cudaMalloc(&h->d_tri_c, n_tri * sizeof(double)));
    CUDA_CHECK_PTR(cudaMalloc(&h->d_tri_d, n_tri * sizeof(double)));
    CUDA_CHECK_PTR(cudaMalloc(&h->d_max_u32, sizeof(uint32_t)));

    if (cfg->has_moist) {
        CUDA_CHECK_PTR(cudaMalloc(&h->d_qv, h->n_s * sizeof(float)));
        CUDA_CHECK_PTR(cudaMalloc(&h->d_qc, h->n_s * sizeof(float)));
        CUDA_CHECK_PTR(cudaMalloc(&h->d_qr, h->n_s * sizeof(float)));
        CUDA_CHECK_PTR(cudaMalloc(&h->d_qi, h->n_s * sizeof(float)));
        CUDA_CHECK_PTR(cudaMalloc(&h->d_qs, h->n_s * sizeof(float)));
        CUDA_CHECK_PTR(cudaMalloc(&h->d_qg, h->n_s * sizeof(float)));
    }
    if (cfg->bc_x == 2 || cfg->bc_y == 2) {
        CUDA_CHECK_PTR(cudaMalloc(&h->d_lbc_u, h->n_u * sizeof(float)));
        CUDA_CHECK_PTR(cudaMalloc(&h->d_lbc_v, h->n_v * sizeof(float)));
        CUDA_CHECK_PTR(cudaMalloc(&h->d_lbc_thp, h->n_s * sizeof(float)));
        CUDA_CHECK_PTR(cudaMalloc(&h->d_lbc_pip, h->n_s * sizeof(float)));
        CUDA_CHECK_PTR(cudaMemset(h->d_lbc_u, 0, h->n_u * sizeof(float)));
        CUDA_CHECK_PTR(cudaMemset(h->d_lbc_v, 0, h->n_v * sizeof(float)));
        CUDA_CHECK_PTR(cudaMemset(h->d_lbc_thp, 0, h->n_s * sizeof(float)));
        CUDA_CHECK_PTR(cudaMemset(h->d_lbc_pip, 0, h->n_s * sizeof(float)));
        if (cfg->has_moist) {
            CUDA_CHECK_PTR(cudaMalloc(&h->d_lbc_qv, h->n_s * sizeof(float)));
            CUDA_CHECK_PTR(cudaMemset(h->d_lbc_qv, 0, h->n_s * sizeof(float)));
        }
        h->has_lbc_qv = 0;
    }
    if (cfg->has_iau) {
        CUDA_CHECK_PTR(cudaMalloc(&h->d_iau_u, h->n_u * sizeof(float)));
        CUDA_CHECK_PTR(cudaMalloc(&h->d_iau_v, h->n_v * sizeof(float)));
        CUDA_CHECK_PTR(cudaMalloc(&h->d_iau_thp, h->n_s * sizeof(float)));
        CUDA_CHECK_PTR(cudaMalloc(&h->d_iau_pip, h->n_s * sizeof(float)));
        CUDA_CHECK_PTR(cudaMemset(h->d_iau_u, 0, h->n_u * sizeof(float)));
        CUDA_CHECK_PTR(cudaMemset(h->d_iau_v, 0, h->n_v * sizeof(float)));
        CUDA_CHECK_PTR(cudaMemset(h->d_iau_thp, 0, h->n_s * sizeof(float)));
        CUDA_CHECK_PTR(cudaMemset(h->d_iau_pip, 0, h->n_s * sizeof(float)));
    }
    if (refresh_polar_fft(h, msf) != 0) {
        nh_cuda_destroy(h);
        return nullptr;
    }

    return h;
}

void nh_cuda_destroy(NhCudaHandle* h) {
    if (!h) return;
    free_polar_fft(h);
    auto cf = [](void* p) { if (p) cudaFree(p); };
    cf(h->d_zeta_w); cf(h->d_zeta_s); cf(h->d_dzeta);
    cf(h->d_b_w); cf(h->d_b_s); cf(h->d_db_w); cf(h->d_db_s);
    cf(h->d_hs); cf(h->d_msf); cf(h->d_fcor);
    cf(h->d_u); cf(h->d_v); cf(h->d_w); cf(h->d_thp); cf(h->d_pip);
    cf(h->d_qv); cf(h->d_qc); cf(h->d_qr); cf(h->d_qi); cf(h->d_qs); cf(h->d_qg);
    cf(h->d_u_n); cf(h->d_v_n); cf(h->d_w_n); cf(h->d_thp_n); cf(h->d_pip_n);
    cf(h->d_ru); cf(h->d_rv); cf(h->d_rw); cf(h->d_rth); cf(h->d_rpi);
    cf(h->d_div); cf(h->d_q_tmp); cf(h->d_tri_c); cf(h->d_tri_d); cf(h->d_max_u32);
    cf(h->d_lbc_u); cf(h->d_lbc_v); cf(h->d_lbc_thp); cf(h->d_lbc_pip); cf(h->d_lbc_qv);
    cf(h->d_iau_u); cf(h->d_iau_v); cf(h->d_iau_thp); cf(h->d_iau_pip);
    nh_physics_free_buffers(h);
    delete h;
}

int32_t nh_cuda_upload_state(NhCudaHandle* h, const NhConstStatePtrs* src) {
    if (!h || !src || !src->u || !src->v || !src->w || !src->theta_p || !src->pi_p) {
        set_error("Null pointer in nh_cuda_upload_state");
        return -1;
    }
    CUDA_CHECK(cudaMemcpy(h->d_u, src->u, h->n_u * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(h->d_v, src->v, h->n_v * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(h->d_w, src->w, h->n_w * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(h->d_thp, src->theta_p, h->n_s * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(h->d_pip, src->pi_p, h->n_s * sizeof(float), cudaMemcpyHostToDevice));
    if (h->cfg.has_moist) {
        if (!src->qv || !src->qc || !src->qr || !src->qi || !src->qs || !src->qg) {
            set_error("Moist state requested but tracer pointer is null");
            return -1;
        }
        CUDA_CHECK(cudaMemcpy(h->d_qv, src->qv, h->n_s * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(h->d_qc, src->qc, h->n_s * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(h->d_qr, src->qr, h->n_s * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(h->d_qi, src->qi, h->n_s * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(h->d_qs, src->qs, h->n_s * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(h->d_qg, src->qg, h->n_s * sizeof(float), cudaMemcpyHostToDevice));
    }
    return 0;
}

int32_t nh_cuda_download_state(NhCudaHandle* h, const NhStatePtrs* dst) {
    if (!h || !dst || !dst->u || !dst->v || !dst->w || !dst->theta_p || !dst->pi_p) {
        set_error("Null pointer in nh_cuda_download_state");
        return -1;
    }
    CUDA_CHECK(cudaMemcpy(dst->u, h->d_u, h->n_u * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dst->v, h->d_v, h->n_v * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dst->w, h->d_w, h->n_w * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dst->theta_p, h->d_thp, h->n_s * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dst->pi_p, h->d_pip, h->n_s * sizeof(float), cudaMemcpyDeviceToHost));
    if (h->cfg.has_moist) {
        if (dst->qv) CUDA_CHECK(cudaMemcpy(dst->qv, h->d_qv, h->n_s * sizeof(float), cudaMemcpyDeviceToHost));
        if (dst->qc) CUDA_CHECK(cudaMemcpy(dst->qc, h->d_qc, h->n_s * sizeof(float), cudaMemcpyDeviceToHost));
        if (dst->qr) CUDA_CHECK(cudaMemcpy(dst->qr, h->d_qr, h->n_s * sizeof(float), cudaMemcpyDeviceToHost));
        if (dst->qi) CUDA_CHECK(cudaMemcpy(dst->qi, h->d_qi, h->n_s * sizeof(float), cudaMemcpyDeviceToHost));
        if (dst->qs) CUDA_CHECK(cudaMemcpy(dst->qs, h->d_qs, h->n_s * sizeof(float), cudaMemcpyDeviceToHost));
        if (dst->qg) CUDA_CHECK(cudaMemcpy(dst->qg, h->d_qg, h->n_s * sizeof(float), cudaMemcpyDeviceToHost));
    }
    return 0;
}

int32_t nh_cuda_set_lbc_target(NhCudaHandle* h, const NhConstStatePtrs* target) {
    if (!h || !target) return -1;
    if (!h->d_lbc_u) {
        set_error("LBC buffer not allocated (bc_x != 2 and bc_y != 2)");
        return -1;
    }
    if (target->u)       CUDA_CHECK(cudaMemcpy(h->d_lbc_u, target->u, h->n_u * sizeof(float), cudaMemcpyHostToDevice));
    if (target->v)       CUDA_CHECK(cudaMemcpy(h->d_lbc_v, target->v, h->n_v * sizeof(float), cudaMemcpyHostToDevice));
    if (target->theta_p) CUDA_CHECK(cudaMemcpy(h->d_lbc_thp, target->theta_p, h->n_s * sizeof(float), cudaMemcpyHostToDevice));
    if (target->pi_p)    CUDA_CHECK(cudaMemcpy(h->d_lbc_pip, target->pi_p, h->n_s * sizeof(float), cudaMemcpyHostToDevice));
    if (h->d_lbc_qv) {
        if (target->qv) {
            CUDA_CHECK(cudaMemcpy(h->d_lbc_qv, target->qv, h->n_s * sizeof(float), cudaMemcpyHostToDevice));
            h->has_lbc_qv = 1;
        } else {
            h->has_lbc_qv = 0;
        }
    }
    return 0;
}

int32_t nh_cuda_set_iau_forcing(NhCudaHandle* h, const NhConstStatePtrs* ddt_state) {
    if (!h || !ddt_state) return -1;
    if (!h->d_iau_u) {
        set_error("IAU buffer not allocated (has_iau == 0)");
        return -1;
    }
    if (ddt_state->u)       CUDA_CHECK(cudaMemcpy(h->d_iau_u, ddt_state->u, h->n_u * sizeof(float), cudaMemcpyHostToDevice));
    if (ddt_state->v)       CUDA_CHECK(cudaMemcpy(h->d_iau_v, ddt_state->v, h->n_v * sizeof(float), cudaMemcpyHostToDevice));
    if (ddt_state->theta_p) CUDA_CHECK(cudaMemcpy(h->d_iau_thp, ddt_state->theta_p, h->n_s * sizeof(float), cudaMemcpyHostToDevice));
    if (ddt_state->pi_p)    CUDA_CHECK(cudaMemcpy(h->d_iau_pip, ddt_state->pi_p, h->n_s * sizeof(float), cudaMemcpyHostToDevice));
    return 0;
}

int32_t nh_cuda_clear_iau_forcing(NhCudaHandle* h) {
    if (!h) return -1;
    auto cf = [](float*& p) {
        if (p) {
            cudaFree(p);
            p = nullptr;
        }
    };
    cf(h->d_iau_u);
    cf(h->d_iau_v);
    cf(h->d_iau_thp);
    cf(h->d_iau_pip);
    h->cfg.has_iau = 0;
    return 0;
}

int32_t nh_cuda_rebind_tile_grid(
    NhCudaHandle* h,
    const NhGridConfig* cfg,
    const double* hs_m,
    const double* msf,
    const double* f_cor
) {
    if (!h || !cfg || !hs_m || !msf || !f_cor) {
        set_error("Null pointer in nh_cuda_rebind_tile_grid");
        return -1;
    }
    if (cfg->nz != h->cfg.nz || cfg->ny != h->cfg.ny || cfg->nx != h->cfg.nx) {
        set_error("Dimension mismatch in nh_cuda_rebind_tile_grid");
        return -1;
    }
    h->cfg = *cfg;
    const size_t n2 = static_cast<size_t>(cfg->ny) * static_cast<size_t>(cfg->nx);
    CUDA_CHECK(cudaMemcpy(h->d_hs, hs_m, n2 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(h->d_msf, msf, n2 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(h->d_fcor, f_cor, n2 * sizeof(double), cudaMemcpyHostToDevice));
    if (refresh_polar_fft(h, msf) != 0) return -1;
    return 0;
}

int32_t nh_cuda_estimate_cs_u32(NhCudaHandle* h, uint32_t* out_cs_u32) {
    if (!h || !out_cs_u32) return -1;
    const int32_t nz = h->cfg.nz, ny = h->cfg.ny, nx = h->cfg.nx;
    const DeviceStaticGrid g = static_view(h);
    const dim3 block(32, 8, 1);
    const dim3 grid_s((nx + 31) / 32, (ny + 7) / 8, nz);
    CUDA_CHECK(cudaMemset(h->d_max_u32, 0, sizeof(uint32_t)));
    nh_estimate_cs_max_kernel<<<grid_s, block>>>(h->cfg, g, h->d_thp, h->d_pip, h->d_max_u32);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(out_cs_u32, h->d_max_u32, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    return 0;
}

int32_t nh_cuda_step_rk3(NhCudaHandle* h, double dt_s, int32_t n_substeps) {
    if (!h || dt_s <= 0.0) {
        set_error("Invalid handle or non-positive dt_s in nh_cuda_step_rk3");
        return -1;
    }

    const int32_t nz = h->cfg.nz, ny = h->cfg.ny, nx = h->cfg.nx;
    const DeviceStaticGrid g = static_view(h);
    const dim3 block(32, 8, 1);
    const dim3 grid_s((nx + 31) / 32, (ny + 7) / 8, nz);
    const dim3 grid_u((nx + 1 + 31) / 32, (ny + 7) / 8, nz);
    const dim3 grid_v((nx + 31) / 32, (ny + 1 + 7) / 8, nz);
    const dim3 grid_w((nx + 31) / 32, (ny + 7) / 8, nz + 1);
    const dim3 grid_col((nx + 31) / 32, (ny + 7) / 8, 1);
    const dim3 grid_per((max(nx, ny) + 255) / 256, 1, nz);

    double cs_msf_max = 360.0;
    if (n_substeps <= 0) {
        uint32_t cs_u32 = 0;
        if (n_substeps < 0) {
            cs_u32 = static_cast<uint32_t>(-n_substeps);
        } else {
            CUDA_CHECK(cudaMemset(h->d_max_u32, 0, sizeof(uint32_t)));
            nh_estimate_cs_max_kernel<<<grid_s, block>>>(h->cfg, g, h->d_thp, h->d_pip, h->d_max_u32);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(&cs_u32, h->d_max_u32, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        }
        float cs_val = 0.0f;
        memcpy(&cs_val, &cs_u32, sizeof(float));
        if (!isfinite(cs_val) || cs_val > 2500.0f) {
            snprintf(g_last_error, sizeof(g_last_error), "Acoustic speed cs*msf out of physical bounds: %.2f m/s", cs_val);
            return -1;
        }
        if (cs_val > 10.0f) cs_msf_max = static_cast<double>(cs_val);
    }

    if (copy_state_to_n(h) != 0) return -1;

    const double rk_dt[3] = {dt_s / 3.0, dt_s / 2.0, dt_s};
    const double dmin = fmin(h->cfg.dx_m, h->cfg.dy_m);
    const float pi_sfc_f = static_cast<float>(pow(h->cfg.p_sfc_pa / P0_PA, KAPPA));

    for (int stage = 0; stage < 3; ++stage) {
        const double tau_stage = rk_dt[stage];
        const float tau_stage_f = static_cast<float>(tau_stage);

        // 1. Evaluate slow tendencies from current stage state (U^n, U^*, U^{**}).
        nh_slow_scalar_kernel<<<grid_s, block>>>(
            h->cfg, g, tau_stage_f, h->d_u, h->d_v, h->d_w, h->d_thp, h->d_pip,
            h->d_lbc_thp, h->d_lbc_pip, h->d_iau_thp, h->d_iau_pip, h->d_rth, h->d_rpi
        );
        nh_slow_u_kernel<<<grid_u, block>>>(
            h->cfg, g, tau_stage_f, h->d_u, h->d_v, h->d_w, h->d_lbc_u, h->d_iau_u, h->d_ru
        );
        nh_slow_v_kernel<<<grid_v, block>>>(
            h->cfg, g, tau_stage_f, h->d_u, h->d_v, h->d_w, h->d_lbc_v, h->d_iau_v, h->d_rv
        );
        nh_slow_w_kernel<<<grid_w, block>>>(
            h->cfg, g, tau_stage_f, h->d_u, h->d_v, h->d_w, h->d_rw
        );
        CUDA_CHECK(cudaGetLastError());

        if (h->fft_ready) {
            if (filter_polar_field(h, h->d_rth, ny, nx, 0, 0, -1) != 0) return -1;
            if (filter_polar_field(h, h->d_rpi, ny, nx, 0, 0, -1) != 0) return -1;
            if (filter_polar_field(h, h->d_ru,  ny, nx + 1, 0, 1, -1) != 0) return -1;
            if (filter_polar_field(h, h->d_rv,  ny + 1, nx, 1, 0, -1) != 0) return -1;
            if (filter_polar_field(h, h->d_rw,  ny, nx, 0, 0, -1) != 0) return -1;
        }

        // 2. Always start acoustic integration for stage s from U^n (Wicker & Skamarock 2002).
        if (stage > 0) {
            if (reset_state_from_n(h) != 0) return -1;
        }

        int32_t ns = n_substeps;
        if (ns <= 0) {
            ns = static_cast<int32_t>(ceil((tau_stage * cs_msf_max) / (0.38 * dmin)));
            if (ns < 1) ns = 1;
            if (ns > 256) {
                snprintf(
                    g_last_error,
                    sizeof(g_last_error),
                    "Acoustic substep count ns=%d exceeded safety cap (256), cs_msf_max=%.2f",
                    ns,
                    cs_msf_max
                );
                return -1;
            }
        }
        const double dts = tau_stage / static_cast<double>(ns);
        const float dts_f = static_cast<float>(dts);

        for (int32_t m = 0; m < ns; ++m) {
            if (h->cfg.div_damp_coeff > 0.0) {
                nh_compute_div_kernel<<<grid_s, block>>>(h->cfg, g, h->d_u, h->d_v, h->d_w, h->d_div);
                CUDA_CHECK(cudaGetLastError());
            }
            if (h->fft_ready) {
                if (save_polar_snapshot(h, h->d_u, ny, nx + 1, 0, 0) != 0) return -1;
                if (save_polar_snapshot(h, h->d_v, ny + 1, nx, 1, 1) != 0) return -1;
            }
            nh_acoustic_u_kernel<<<grid_u, block>>>(
                h->cfg, g, dts_f, h->d_ru, h->d_thp, h->d_pip, h->d_div,
                h->d_qv, h->d_qc, h->d_qr, h->d_qi, h->d_qs, h->d_qg, h->d_u
            );
            nh_acoustic_v_kernel<<<grid_v, block>>>(
                h->cfg, g, dts_f, h->d_rv, h->d_thp, h->d_pip, h->d_div,
                h->d_qv, h->d_qc, h->d_qr, h->d_qi, h->d_qs, h->d_qg, h->d_v
            );
            CUDA_CHECK(cudaGetLastError());

            if (h->cfg.bc_x == 1 || h->cfg.bc_y == 1) {
                nh_sync_periodic_faces_kernel<<<grid_per, 256>>>(h->cfg, h->d_u, h->d_v);
                CUDA_CHECK(cudaGetLastError());
            }

            if (h->fft_ready) {
                if (filter_polar_field(h, h->d_u, ny, nx + 1, 0, 1, 0) != 0) return -1;
                if (filter_polar_field(h, h->d_v, ny + 1, nx, 1, 0, 1) != 0) return -1;
                if (save_polar_snapshot(h, h->d_w,   ny, nx, 0, 0) != 0) return -1;
                if (save_polar_snapshot(h, h->d_thp, ny, nx, 0, 1) != 0) return -1;
                if (save_polar_snapshot(h, h->d_pip, ny, nx, 0, 2) != 0) return -1;
            }

            nh_acoustic_vert_implicit_kernel<<<grid_col, block>>>(
                h->cfg, g, dts_f, pi_sfc_f, h->d_u, h->d_v, h->d_rw, h->d_rth, h->d_rpi,
                h->d_qv, h->d_qc, h->d_qr, h->d_qi, h->d_qs, h->d_qg,
                h->d_w, h->d_thp, h->d_pip,
                reinterpret_cast<float*>(h->d_tri_c), reinterpret_cast<float*>(h->d_tri_d)
            );
            CUDA_CHECK(cudaGetLastError());

            if (h->fft_ready) {
                if (filter_polar_field(h, h->d_w,   ny, nx, 0, 0, 0) != 0) return -1;
                if (filter_polar_field(h, h->d_thp, ny, nx, 0, 0, 1) != 0) return -1;
                if (filter_polar_field(h, h->d_pip, ny, nx, 0, 0, 2) != 0) return -1;
            }
        }

        // 2b. IEVA column-implicit residual vertical advection where |CFL_z| > 0.45
        const dim3 grid_col_u(((nx + 1) + 31) / 32, (ny + 7) / 8, 1);
        const dim3 grid_col_v((nx + 31) / 32, ((ny + 1) + 7) / 8, 1);
        nh_ieva_vert_advect_scalar_kernel<<<grid_col, block>>>(
            h->cfg, g, tau_stage_f, h->d_u, h->d_v, h->d_w, h->d_thp, h->d_pip, h->d_tri_c, h->d_tri_d
        );
        nh_ieva_vert_advect_u_kernel<<<grid_col_u, block>>>(
            h->cfg, g, tau_stage_f, h->d_v, h->d_w, h->d_u, h->d_tri_c, h->d_tri_d
        );
        nh_ieva_vert_advect_v_kernel<<<grid_col_v, block>>>(
            h->cfg, g, tau_stage_f, h->d_u, h->d_w, h->d_v, h->d_tri_c, h->d_tri_d
        );
        nh_ieva_vert_advect_w_kernel<<<grid_col, block>>>(
            h->cfg, g, tau_stage_f, h->d_u, h->d_v, h->d_w, h->d_tri_c, h->d_tri_d
        );
        CUDA_CHECK(cudaGetLastError());
        if (h->cfg.bc_x == 1 || h->cfg.bc_y == 1) {
            nh_sync_periodic_faces_kernel<<<grid_per, 256>>>(h->cfg, h->d_u, h->d_v);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    // 3. Conservative positive-definite 2D horizontal FCT + 1D vertical conservative column transport on full dt_s.
    if (h->cfg.has_moist) {
        const float pi_sfc_f = static_cast<float>(pow(h->cfg.p_sfc_pa / P0_PA, KAPPA));
        CUDA_CHECK(cudaMemset(h->d_max_u32, 0, sizeof(uint32_t)));
        dim3 b_cfl(32, 8, 1);
        dim3 g_cfl((h->cfg.nx + 31) / 32, (h->cfg.ny + 7) / 8, h->cfg.nz);
        nh_tracer_cfl_h_kernel<<<g_cfl, b_cfl>>>(
            h->cfg, g, static_cast<float>(dt_s), pi_sfc_f, h->d_u, h->d_v, h->d_max_u32
        );
        CUDA_CHECK(cudaGetLastError());

        uint32_t cfl_u32 = 0;
        CUDA_CHECK(cudaMemcpy(&cfl_u32, h->d_max_u32, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        float cfl_h_max = 0.0f;
        memcpy(&cfl_h_max, &cfl_u32, sizeof(float));

        if (!isfinite(cfl_h_max)) {
            set_error("Non-finite horizontal tracer CFL detected");
            return -1;
        }
        int32_t n_sub_h = static_cast<int32_t>(ceil(static_cast<double>(cfl_h_max) / 0.70));
        if (n_sub_h < 1) n_sub_h = 1;
        if (n_sub_h > 64) {
            snprintf(
                g_last_error,
                sizeof(g_last_error),
                "Horizontal tracer substep count n_sub_h=%d exceeded safety cap (64), cfl_h_max=%.3f",
                n_sub_h,
                cfl_h_max
            );
            return -1;
        }
        const float dt_full_f = static_cast<float>(dt_s);
        const float dt_h = static_cast<float>(dt_s / static_cast<double>(n_sub_h));

        if (advect_one_tracer(h, h->d_qv, dt_full_f, n_sub_h, dt_h, pi_sfc_f) != 0) return -1;
        if ((h->cfg.bc_x == 2 || h->cfg.bc_y == 2) && h->d_lbc_qv && h->has_lbc_qv && h->cfg.lbc_tau_max_s > 0.0) {
            nh_lbc_relax_qv_kernel<<<grid_s, block>>>(h->cfg, dt_full_f, h->d_lbc_qv, h->d_qv);
            CUDA_CHECK(cudaGetLastError());
        }
        if (advect_one_tracer(h, h->d_qc, dt_full_f, n_sub_h, dt_h, pi_sfc_f) != 0) return -1;
        if (advect_one_tracer(h, h->d_qr, dt_full_f, n_sub_h, dt_h, pi_sfc_f) != 0) return -1;
        if (advect_one_tracer(h, h->d_qi, dt_full_f, n_sub_h, dt_h, pi_sfc_f) != 0) return -1;
        if (advect_one_tracer(h, h->d_qs, dt_full_f, n_sub_h, dt_h, pi_sfc_f) != 0) return -1;
        if (advect_one_tracer(h, h->d_qg, dt_full_f, n_sub_h, dt_h, pi_sfc_f) != 0) return -1;
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
}

} // extern "C"
