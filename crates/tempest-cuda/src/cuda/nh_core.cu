#include "nh_internal.cuh"

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

__device__ inline double weno5_left(double qm2, double qm1, double q0, double qp1, double qp2) {
    constexpr double eps = 1e-18;
    const double p0 = (1.0 / 3.0) * qm2 - (7.0 / 6.0) * qm1 + (11.0 / 6.0) * q0;
    const double p1 = -(1.0 / 6.0) * qm1 + (5.0 / 6.0) * q0 + (1.0 / 3.0) * qp1;
    const double p2 = (1.0 / 3.0) * q0 + (5.0 / 6.0) * qp1 - (1.0 / 6.0) * qp2;

    const double s0a = qm2 - 2.0 * qm1 + q0;
    const double s0b = qm2 - 4.0 * qm1 + 3.0 * q0;
    const double b0 = (13.0 / 12.0) * s0a * s0a + 0.25 * s0b * s0b;

    const double s1a = qm1 - 2.0 * q0 + qp1;
    const double s1b = qm1 - qp1;
    const double b1 = (13.0 / 12.0) * s1a * s1a + 0.25 * s1b * s1b;

    const double s2a = q0 - 2.0 * qp1 + qp2;
    const double s2b = 3.0 * q0 - 4.0 * qp1 + qp2;
    const double b2 = (13.0 / 12.0) * s2a * s2a + 0.25 * s2b * s2b;

    const double a0 = 0.1 / ((eps + b0) * (eps + b0));
    const double a1 = 0.6 / ((eps + b1) * (eps + b1));
    const double a2 = 0.3 / ((eps + b2) * (eps + b2));
    const double inv_s = 1.0 / (a0 + a1 + a2);
    return (a0 * p0 + a1 * p1 + a2 * p2) * inv_s;
}

__device__ inline double up3_left(double qm1, double q0, double qp1) {
    return (1.0 / 3.0) * qp1 + (5.0 / 6.0) * q0 - (1.0 / 6.0) * qm1;
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
        double thb, pib, dthdz, rhob, cs2b, qvv, qhv;
        eval_ref_state(z_lvl, cfg.theta_sfc_k, cfg.p_sfc_pa, cfg.n_ref_s, thb, pib, dthdz, rhob, cs2b);
        const double vfac = moisture_load_scalar(cfg.has_moist, s_idx, qv, qc, qr, qi, qs, qg, qvv, qhv);
        const double th_tot = (thb + static_cast<double>(theta_p[s_idx])) * vfac;
        return (GRAVITY * (th_tot - thb)) / (CP_D * thb * fmax(th_tot, 50.0));
    };

    if (z_target <= z_col(0)) {
        const size_t idx0 = static_cast<size_t>(j_clamped) * nx + i_clamped;
        const size_t idx1 = static_cast<size_t>(ny) * nx + idx0;
        const double z0 = z_col(0);
        const double z1 = z_col(1);
        const double dpi0 = hydro_slope(idx0, z0);
        const double dpi1 = hydro_slope(idx1, z1);
        const double d2pi = (dpi1 - dpi0) / fmax(z1 - z0, 1e-6);
        const double dz = z_target - z0;
        return static_cast<double>(pi_p[idx0]) + (dpi0 + 0.5 * d2pi * dz) * dz;
    }
    const double z_top_s = z_col(nz - 1);
    if (z_target >= z_top_s) {
        const size_t idx_t = static_cast<size_t>(nz - 1) * ny * nx + static_cast<size_t>(j_clamped) * nx + i_clamped;
        const size_t idx_tm = static_cast<size_t>(nz - 2) * ny * nx + static_cast<size_t>(j_clamped) * nx + i_clamped;
        const double z_tm = z_col(nz - 2);
        const double dpi_t = hydro_slope(idx_t, z_top_s);
        const double dpi_tm = hydro_slope(idx_tm, z_tm);
        const double d2pi = (dpi_t - dpi_tm) / fmax(z_top_s - z_tm, 1e-6);
        const double dz = z_target - z_top_s;
        return static_cast<double>(pi_p[idx_t]) + (dpi_t + 0.5 * d2pi * dz) * dz;
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

__device__ inline double davies_weight(int32_t dist, int32_t rim_width, double tau_max) {
    if (rim_width <= 0 || dist >= rim_width) return 0.0;
    const double f = static_cast<double>(rim_width - dist) / static_cast<double>(rim_width);
    return tau_max * f * f;
}

__device__ inline double sponge_tau(double z, double z_top, double depth, double tau_max) {
    if (depth <= 0.0 || tau_max <= 0.0) return 0.0;
    const double z_start = z_top - depth;
    if (z <= z_start) return 0.0;
    const double s = sin(0.5 * 3.14159265358979323846 * (z - z_start) / depth);
    return tau_max * s * s;
}

__device__ inline double horiz_face_val_x(
    const float* f, int32_t k, int32_t j, int32_t i_face, double dir_u,
    int32_t nz, int32_t ny, int32_t nx, int32_t bc_x, int32_t bc_y, int32_t adv_order
) {
    // Reconstruct scalar at x-face i_face - 1/2 (between i_face-1 and i_face).
    if (dir_u >= 0.0) {
        if (adv_order >= 5) {
            return weno5_left(
                read_s(f, k, j, i_face - 3, nz, ny, nx, bc_x, bc_y),
                read_s(f, k, j, i_face - 2, nz, ny, nx, bc_x, bc_y),
                read_s(f, k, j, i_face - 1, nz, ny, nx, bc_x, bc_y),
                read_s(f, k, j, i_face,     nz, ny, nx, bc_x, bc_y),
                read_s(f, k, j, i_face + 1, nz, ny, nx, bc_x, bc_y)
            );
        }
        return up3_left(
            read_s(f, k, j, i_face - 2, nz, ny, nx, bc_x, bc_y),
            read_s(f, k, j, i_face - 1, nz, ny, nx, bc_x, bc_y),
            read_s(f, k, j, i_face,     nz, ny, nx, bc_x, bc_y)
        );
    } else {
        if (adv_order >= 5) {
            return weno5_left(
                read_s(f, k, j, i_face + 2, nz, ny, nx, bc_x, bc_y),
                read_s(f, k, j, i_face + 1, nz, ny, nx, bc_x, bc_y),
                read_s(f, k, j, i_face,     nz, ny, nx, bc_x, bc_y),
                read_s(f, k, j, i_face - 1, nz, ny, nx, bc_x, bc_y),
                read_s(f, k, j, i_face - 2, nz, ny, nx, bc_x, bc_y)
            );
        }
        return up3_left(
            read_s(f, k, j, i_face + 1, nz, ny, nx, bc_x, bc_y),
            read_s(f, k, j, i_face,     nz, ny, nx, bc_x, bc_y),
            read_s(f, k, j, i_face - 1, nz, ny, nx, bc_x, bc_y)
        );
    }
}

__device__ inline double horiz_face_val_y(
    const float* f, int32_t k, int32_t j_face, int32_t i, double dir_v,
    int32_t nz, int32_t ny, int32_t nx, int32_t bc_x, int32_t bc_y, int32_t adv_order
) {
    if (dir_v >= 0.0) {
        if (adv_order >= 5) {
            return weno5_left(
                read_s(f, k, j_face - 3, i, nz, ny, nx, bc_x, bc_y),
                read_s(f, k, j_face - 2, i, nz, ny, nx, bc_x, bc_y),
                read_s(f, k, j_face - 1, i, nz, ny, nx, bc_x, bc_y),
                read_s(f, k, j_face,     i, nz, ny, nx, bc_x, bc_y),
                read_s(f, k, j_face + 1, i, nz, ny, nx, bc_x, bc_y)
            );
        }
        return up3_left(
            read_s(f, k, j_face - 2, i, nz, ny, nx, bc_x, bc_y),
            read_s(f, k, j_face - 1, i, nz, ny, nx, bc_x, bc_y),
            read_s(f, k, j_face,     i, nz, ny, nx, bc_x, bc_y)
        );
    } else {
        if (adv_order >= 5) {
            return weno5_left(
                read_s(f, k, j_face + 2, i, nz, ny, nx, bc_x, bc_y),
                read_s(f, k, j_face + 1, i, nz, ny, nx, bc_x, bc_y),
                read_s(f, k, j_face,     i, nz, ny, nx, bc_x, bc_y),
                read_s(f, k, j_face - 1, i, nz, ny, nx, bc_x, bc_y),
                read_s(f, k, j_face - 2, i, nz, ny, nx, bc_x, bc_y)
            );
        }
        return up3_left(
            read_s(f, k, j_face + 1, i, nz, ny, nx, bc_x, bc_y),
            read_s(f, k, j_face,     i, nz, ny, nx, bc_x, bc_y),
            read_s(f, k, j_face - 1, i, nz, ny, nx, bc_x, bc_y)
        );
    }
}

__device__ inline double vert_tvd_face_val(
    const float* f, int32_t k_face, int32_t j, int32_t i, double dir_w,
    const double* dzeta, int32_t nz, int32_t ny, int32_t nx, int32_t bc_x, int32_t bc_y
) {
    // Reconstruct scalar onto interface k_face (between k_face-1 and k_face).
    if (k_face <= 0) return read_s(f, 0, j, i, nz, ny, nx, bc_x, bc_y);
    if (k_face >= nz) return read_s(f, nz - 1, j, i, nz, ny, nx, bc_x, bc_y);
    if (dir_w >= 0.0) {
        const int32_t km = k_face - 1;
        const double vc = read_s(f, km, j, i, nz, ny, nx, bc_x, bc_y);
        if (km == 0) return vc;
        const double vm = read_s(f, km - 1, j, i, nz, ny, nx, bc_x, bc_y);
        const double vp = read_s(f, km + 1, j, i, nz, ny, nx, bc_x, bc_y);
        const double dl = (vc - vm) / (0.5 * (dzeta[km] + dzeta[km - 1]));
        const double dr = (vp - vc) / (0.5 * (dzeta[km + 1] + dzeta[km]));
        return vc + 0.5 * dzeta[km] * koren_slope(dl, dr);
    } else {
        const int32_t kp = k_face;
        const double vc = read_s(f, kp, j, i, nz, ny, nx, bc_x, bc_y);
        if (kp == nz - 1) return vc;
        const double vm = read_s(f, kp - 1, j, i, nz, ny, nx, bc_x, bc_y);
        const double vp = read_s(f, kp + 1, j, i, nz, ny, nx, bc_x, bc_y);
        const double dl = (vc - vm) / (0.5 * (dzeta[kp] + dzeta[kp - 1]));
        const double dr = (vp - vc) / (0.5 * (dzeta[kp + 1] + dzeta[kp]));
        return vc - 0.5 * dzeta[kp] * koren_slope(dl, dr);
    }
}

__device__ inline void compute_contravariant_at_scalar_col(
    const NhGridConfig& cfg,
    const DeviceStaticGrid& g,
    const float* u,
    const float* v,
    const float* w,
    int32_t k_iface,
    int32_t j,
    int32_t i,
    double& gdotzeta_out
) {
    if (k_iface <= 0 || k_iface >= cfg.nz) {
        gdotzeta_out = 0.0;
        return;
    }
    const int32_t nz = cfg.nz, ny = cfg.ny, nx = cfg.nx;
    const double msf = read_2d(g.msf, j, i, ny, nx, cfg.bc_x, cfg.bc_y);
    const double dh_dx = (read_2d(g.hs, j, i + 1, ny, nx, cfg.bc_x, cfg.bc_y) -
                          read_2d(g.hs, j, i - 1, ny, nx, cfg.bc_x, cfg.bc_y)) / (2.0 * cfg.dx_m);
    const double dh_dy = (read_2d(g.hs, j + 1, i, ny, nx, cfg.bc_x, cfg.bc_y) -
                          read_2d(g.hs, j - 1, i, ny, nx, cfg.bc_x, cfg.bc_y)) / (2.0 * cfg.dy_m);

    const double ds_k = g.dzeta[k_iface];
    const double ds_km = g.dzeta[k_iface - 1];
    const double wt = ds_km / (ds_k + ds_km);

    const double u_lo = 0.5 * (read_u(u, k_iface - 1, j, i, nz, ny, nx, cfg.bc_x, cfg.bc_y) +
                               read_u(u, k_iface - 1, j, i + 1, nz, ny, nx, cfg.bc_x, cfg.bc_y));
    const double u_hi = 0.5 * (read_u(u, k_iface, j, i, nz, ny, nx, cfg.bc_x, cfg.bc_y) +
                               read_u(u, k_iface, j, i + 1, nz, ny, nx, cfg.bc_x, cfg.bc_y));
    const double v_lo = 0.5 * (read_v(v, k_iface - 1, j, i, nz, ny, nx, cfg.bc_x, cfg.bc_y) +
                               read_v(v, k_iface - 1, j + 1, i, nz, ny, nx, cfg.bc_x, cfg.bc_y));
    const double v_hi = 0.5 * (read_v(v, k_iface, j, i, nz, ny, nx, cfg.bc_x, cfg.bc_y) +
                               read_v(v, k_iface, j + 1, i, nz, ny, nx, cfg.bc_x, cfg.bc_y));

    const double u_w = (1.0 - wt) * u_lo + wt * u_hi;
    const double v_w = (1.0 - wt) * v_lo + wt * v_hi;
    const double w_val = read_w(w, k_iface, j, i, nz, ny, nx, cfg.bc_x, cfg.bc_y);
    const double bw = g.b_w[k_iface];
    gdotzeta_out = w_val - msf * bw * (u_w * dh_dx + v_w * dh_dy);
}

// ---------------------------------------------------------------------------
// Kernel 1: Slow tendencies (advection, Coriolis/LCC curvature, explicit diff,
// Davies relaxation, IAU)
// ---------------------------------------------------------------------------
__global__ void nh_slow_scalar_kernel(
    NhGridConfig cfg,
    DeviceStaticGrid g,
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

    const double msf = g.msf[j * nx + i];
    const double hs = g.hs[j * nx + i];
    const double g_s = 1.0 + g.db_s[k] * hs;

    const double uc = 0.5 * (read_u(u, k, j, i, nz, ny, nx, cfg.bc_x, cfg.bc_y) +
                             read_u(u, k, j, i + 1, nz, ny, nx, cfg.bc_x, cfg.bc_y));
    const double vc = 0.5 * (read_v(v, k, j, i, nz, ny, nx, cfg.bc_x, cfg.bc_y) +
                             read_v(v, k, j + 1, i, nz, ny, nx, cfg.bc_x, cfg.bc_y));
    double gz_lo = 0.0, gz_hi = 0.0;
    compute_contravariant_at_scalar_col(cfg, g, u, v, w, k, j, i, gz_lo);
    compute_contravariant_at_scalar_col(cfg, g, u, v, w, k + 1, j, i, gz_hi);
    const double zeta_dot = 0.5 * (gz_lo + gz_hi) / fmax(g_s, 1e-4);

    // Advect theta'
    const double th_xp = horiz_face_val_x(thp, k, j, i + 1, uc, nz, ny, nx, cfg.bc_x, cfg.bc_y, cfg.adv_order);
    const double th_xm = horiz_face_val_x(thp, k, j, i,     uc, nz, ny, nx, cfg.bc_x, cfg.bc_y, cfg.adv_order);
    const double th_yp = horiz_face_val_y(thp, k, j + 1, i, vc, nz, ny, nx, cfg.bc_x, cfg.bc_y, cfg.adv_order);
    const double th_ym = horiz_face_val_y(thp, k, j,     i, vc, nz, ny, nx, cfg.bc_x, cfg.bc_y, cfg.adv_order);
    const double th_zp = vert_tvd_face_val(thp, k + 1, j, i, zeta_dot, g.dzeta, nz, ny, nx, cfg.bc_x, cfg.bc_y);
    const double th_zm = vert_tvd_face_val(thp, k,     j, i, zeta_dot, g.dzeta, nz, ny, nx, cfg.bc_x, cfg.bc_y);

    double tend_th = -msf * (uc * (th_xp - th_xm) / cfg.dx_m + vc * (th_yp - th_ym) / cfg.dy_m)
                     - zeta_dot * (th_zp - th_zm) / g.dzeta[k];

    // Advect pi'
    const double pi_xp = horiz_face_val_x(pip, k, j, i + 1, uc, nz, ny, nx, cfg.bc_x, cfg.bc_y, cfg.adv_order);
    const double pi_xm = horiz_face_val_x(pip, k, j, i,     uc, nz, ny, nx, cfg.bc_x, cfg.bc_y, cfg.adv_order);
    const double pi_yp = horiz_face_val_y(pip, k, j + 1, i, vc, nz, ny, nx, cfg.bc_x, cfg.bc_y, cfg.adv_order);
    const double pi_ym = horiz_face_val_y(pip, k, j,     i, vc, nz, ny, nx, cfg.bc_x, cfg.bc_y, cfg.adv_order);
    const double pi_zp = vert_tvd_face_val(pip, k + 1, j, i, zeta_dot, g.dzeta, nz, ny, nx, cfg.bc_x, cfg.bc_y);
    const double pi_zm = vert_tvd_face_val(pip, k,     j, i, zeta_dot, g.dzeta, nz, ny, nx, cfg.bc_x, cfg.bc_y);

    double tend_pi = -msf * (uc * (pi_xp - pi_xm) / cfg.dx_m + vc * (pi_yp - pi_ym) / cfg.dy_m)
                     - zeta_dot * (pi_zp - pi_zm) / g.dzeta[k];

    // Explicit 3D Laplacian diffusion (used cleanly for Straka K=75 m^2/s)
    const double th0 = static_cast<double>(thp[idx]);
    const double pi0 = static_cast<double>(pip[idx]);
    if (cfg.kh_const > 0.0) {
        const double th_e = read_s(thp, k, j, i + 1, nz, ny, nx, cfg.bc_x, cfg.bc_y);
        const double th_w = read_s(thp, k, j, i - 1, nz, ny, nx, cfg.bc_x, cfg.bc_y);
        const double th_n = read_s(thp, k, j + 1, i, nz, ny, nx, cfg.bc_x, cfg.bc_y);
        const double th_s = read_s(thp, k, j - 1, i, nz, ny, nx, cfg.bc_x, cfg.bc_y);
        const double lap_h = msf * msf * (
            (th_e - 2.0 * th0 + th_w) / (cfg.dx_m * cfg.dx_m) +
            (th_n - 2.0 * th0 + th_s) / (cfg.dy_m * cfg.dy_m)
        );
        const double dz_c = g_s * g.dzeta[k];
        double lap_v = 0.0;
        if (k < nz - 1) {
            const double dz_p = 0.5 * (dz_c + (1.0 + g.db_s[k + 1] * hs) * g.dzeta[k + 1]);
            lap_v += (read_s(thp, k + 1, j, i, nz, ny, nx, cfg.bc_x, cfg.bc_y) - th0) / (dz_p * dz_c);
        }
        if (k > 0) {
            const double dz_m = 0.5 * (dz_c + (1.0 + g.db_s[k - 1] * hs) * g.dzeta[k - 1]);
            lap_v -= (th0 - read_s(thp, k - 1, j, i, nz, ny, nx, cfg.bc_x, cfg.bc_y)) / (dz_m * dz_c);
        }
        tend_th += cfg.kh_const * (lap_h + lap_v);
    }

    // Optional Davies lateral boundary relaxation
    if ((cfg.bc_x == 2 || cfg.bc_y == 2) && lbc_thp != nullptr && cfg.lbc_tau_max_s > 0.0) {
        double tau = 0.0;
        if (cfg.bc_x == 2) {
            if (cfg.lbc_edges & 1) tau = fmax(tau, davies_weight(i, cfg.lbc_rim_width, cfg.lbc_tau_max_s));
            if (cfg.lbc_edges & 2) tau = fmax(tau, davies_weight(nx - 1 - i, cfg.lbc_rim_width, cfg.lbc_tau_max_s));
        }
        if (cfg.bc_y == 2) {
            if (cfg.lbc_edges & 4) tau = fmax(tau, davies_weight(j, cfg.lbc_rim_width, cfg.lbc_tau_max_s));
            if (cfg.lbc_edges & 8) tau = fmax(tau, davies_weight(ny - 1 - j, cfg.lbc_rim_width, cfg.lbc_tau_max_s));
        }
        if (tau > 0.0) {
            tend_th -= tau * (th0 - static_cast<double>(lbc_thp[idx]));
            if (lbc_pip != nullptr) {
                tend_pi -= tau * (pi0 - static_cast<double>(lbc_pip[idx]));
            }
        }
    }

    if (cfg.has_iau && iau_thp != nullptr) tend_th += static_cast<double>(iau_thp[idx]);
    if (cfg.has_iau && iau_pip != nullptr) tend_pi += static_cast<double>(iau_pip[idx]);

    rth[idx] = static_cast<float>(tend_th);
    rpi[idx] = static_cast<float>(tend_pi);
}

__global__ void nh_slow_u_kernel(
    NhGridConfig cfg,
    DeviceStaticGrid g,
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
    const double msf = 0.5 * (g.msf[j * nx + il] + g.msf[j * nx + ir]);
    const double hs_u = 0.5 * (g.hs[j * nx + il] + g.hs[j * nx + ir]);
    const double g_u = 1.0 + g.db_s[k] * hs_u;

    const double uc = read_u(u, k, j, i, nz, ny, nx, cfg.bc_x, cfg.bc_y);
    const double vc = 0.25 * (
        read_v(v, k, j,     il, nz, ny, nx, cfg.bc_x, cfg.bc_y) +
        read_v(v, k, j + 1, il, nz, ny, nx, cfg.bc_x, cfg.bc_y) +
        read_v(v, k, j,     ir, nz, ny, nx, cfg.bc_x, cfg.bc_y) +
        read_v(v, k, j + 1, ir, nz, ny, nx, cfg.bc_x, cfg.bc_y)
    );
    double gz_ll = 0.0, gz_lh = 0.0, gz_rl = 0.0, gz_rh = 0.0;
    compute_contravariant_at_scalar_col(cfg, g, u, v, w, k,     j, il, gz_ll);
    compute_contravariant_at_scalar_col(cfg, g, u, v, w, k + 1, j, il, gz_lh);
    compute_contravariant_at_scalar_col(cfg, g, u, v, w, k,     j, ir, gz_rl);
    compute_contravariant_at_scalar_col(cfg, g, u, v, w, k + 1, j, ir, gz_rh);
    const double zeta_dot = 0.25 * (gz_ll + gz_lh + gz_rl + gz_rh) / fmax(g_u, 1e-4);

    // Centered/UP3 momentum advection on u-points
    auto u_get = [&](int32_t kk, int32_t jj, int32_t ii) -> double {
        return read_u(u, kk, jj, ii, nz, ny, nx, cfg.bc_x, cfg.bc_y);
    };
    double dudx = 0.0;
    if (uc >= 0.0) {
        dudx = (up3_left(u_get(k, j, i - 1), uc, u_get(k, j, i + 1)) -
                up3_left(u_get(k, j, i - 2), u_get(k, j, i - 1), uc)) / cfg.dx_m;
    } else {
        dudx = (up3_left(u_get(k, j, i + 2), u_get(k, j, i + 1), uc) -
                up3_left(u_get(k, j, i + 1), uc, u_get(k, j, i - 1))) / cfg.dx_m;
    }
    double dudy = 0.0;
    if (vc >= 0.0) {
        dudy = (up3_left(u_get(k, j - 1, i), uc, u_get(k, j + 1, i)) -
                up3_left(u_get(k, j - 2, i), u_get(k, j - 1, i), uc)) / cfg.dy_m;
    } else {
        dudy = (up3_left(u_get(k, j + 2, i), u_get(k, j + 1, i), uc) -
                up3_left(u_get(k, j + 1, i), uc, u_get(k, j - 1, i))) / cfg.dy_m;
    }
    double dudz = 0.0;
    if (zeta_dot >= 0.0 && k > 0) {
        dudz = (uc - u_get(k - 1, j, i)) / (0.5 * (g.dzeta[k] + g.dzeta[k - 1]));
    } else if (zeta_dot < 0.0 && k < nz - 1) {
        dudz = (u_get(k + 1, j, i) - uc) / (0.5 * (g.dzeta[k + 1] + g.dzeta[k]));
    }

    double tend = -msf * (uc * dudx + vc * dudy) - zeta_dot * dudz;

    if (cfg.use_coriolis) {
        const double f_u = 0.5 * (g.fcor[j * nx + il] + g.fcor[j * nx + ir]);
        const double dm_dy = (read_2d(g.msf, j + 1, il, ny, nx, cfg.bc_x, cfg.bc_y) -
                              read_2d(g.msf, j - 1, il, ny, nx, cfg.bc_x, cfg.bc_y) +
                              read_2d(g.msf, j + 1, ir, ny, nx, cfg.bc_x, cfg.bc_y) -
                              read_2d(g.msf, j - 1, ir, ny, nx, cfg.bc_x, cfg.bc_y)) / (4.0 * cfg.dy_m);
        const double dm_dx = (g.msf[j * nx + ir] - g.msf[j * nx + il]) / cfg.dx_m;
        tend += (f_u + uc * dm_dy - vc * dm_dx) * vc;
    }

    if (cfg.km_const > 0.0) {
        const double lap_h = msf * msf * (
            (u_get(k, j, i + 1) - 2.0 * uc + u_get(k, j, i - 1)) / (cfg.dx_m * cfg.dx_m) +
            (u_get(k, j + 1, i) - 2.0 * uc + u_get(k, j - 1, i)) / (cfg.dy_m * cfg.dy_m)
        );
        const double dz_c = g_u * g.dzeta[k];
        double lap_v = 0.0;
        if (k < nz - 1) {
            const double dz_p = 0.5 * (dz_c + (1.0 + g.db_s[k + 1] * hs_u) * g.dzeta[k + 1]);
            lap_v += (u_get(k + 1, j, i) - uc) / (dz_p * dz_c);
        }
        if (k > 0) {
            const double dz_m = 0.5 * (dz_c + (1.0 + g.db_s[k - 1] * hs_u) * g.dzeta[k - 1]);
            lap_v -= (uc - u_get(k - 1, j, i)) / (dz_m * dz_c);
        }
        tend += cfg.km_const * (lap_h + lap_v);
    }

    if ((cfg.bc_x == 2 || cfg.bc_y == 2) && lbc_u != nullptr && cfg.lbc_tau_max_s > 0.0) {
        double tau = 0.0;
        if (cfg.bc_x == 2) {
            if (cfg.lbc_edges & 1) tau = fmax(tau, davies_weight(i, cfg.lbc_rim_width, cfg.lbc_tau_max_s));
            if (cfg.lbc_edges & 2) tau = fmax(tau, davies_weight(nx - i, cfg.lbc_rim_width, cfg.lbc_tau_max_s));
        }
        if (cfg.bc_y == 2) {
            if (cfg.lbc_edges & 4) tau = fmax(tau, davies_weight(j, cfg.lbc_rim_width, cfg.lbc_tau_max_s));
            if (cfg.lbc_edges & 8) tau = fmax(tau, davies_weight(ny - 1 - j, cfg.lbc_rim_width, cfg.lbc_tau_max_s));
        }
        if (tau > 0.0) {
            tend -= tau * (uc - static_cast<double>(lbc_u[idx]));
        }
    }

    if (cfg.has_iau && iau_u != nullptr) tend += static_cast<double>(iau_u[idx]);
    ru[idx] = static_cast<float>(tend);
}

__global__ void nh_slow_v_kernel(
    NhGridConfig cfg,
    DeviceStaticGrid g,
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
    const double msf = 0.5 * (g.msf[jl * nx + i] + g.msf[jr * nx + i]);
    const double hs_v = 0.5 * (g.hs[jl * nx + i] + g.hs[jr * nx + i]);
    const double g_v = 1.0 + g.db_s[k] * hs_v;

    const double vc = read_v(v, k, j, i, nz, ny, nx, cfg.bc_x, cfg.bc_y);
    const double uc = 0.25 * (
        read_u(u, k, jl, i,     nz, ny, nx, cfg.bc_x, cfg.bc_y) +
        read_u(u, k, jl, i + 1, nz, ny, nx, cfg.bc_x, cfg.bc_y) +
        read_u(u, k, jr, i,     nz, ny, nx, cfg.bc_x, cfg.bc_y) +
        read_u(u, k, jr, i + 1, nz, ny, nx, cfg.bc_x, cfg.bc_y)
    );
    double gz_ll = 0.0, gz_lh = 0.0, gz_rl = 0.0, gz_rh = 0.0;
    compute_contravariant_at_scalar_col(cfg, g, u, v, w, k,     jl, i, gz_ll);
    compute_contravariant_at_scalar_col(cfg, g, u, v, w, k + 1, jl, i, gz_lh);
    compute_contravariant_at_scalar_col(cfg, g, u, v, w, k,     jr, i, gz_rl);
    compute_contravariant_at_scalar_col(cfg, g, u, v, w, k + 1, jr, i, gz_rh);
    const double zeta_dot = 0.25 * (gz_ll + gz_lh + gz_rl + gz_rh) / fmax(g_v, 1e-4);

    auto v_get = [&](int32_t kk, int32_t jj, int32_t ii) -> double {
        return read_v(v, kk, jj, ii, nz, ny, nx, cfg.bc_x, cfg.bc_y);
    };
    double dvdx = 0.0;
    if (uc >= 0.0) {
        dvdx = (up3_left(v_get(k, j, i - 1), vc, v_get(k, j, i + 1)) -
                up3_left(v_get(k, j, i - 2), v_get(k, j, i - 1), vc)) / cfg.dx_m;
    } else {
        dvdx = (up3_left(v_get(k, j, i + 2), v_get(k, j, i + 1), vc) -
                up3_left(v_get(k, j, i + 1), vc, v_get(k, j, i - 1))) / cfg.dx_m;
    }
    double dvdy = 0.0;
    if (vc >= 0.0) {
        dvdy = (up3_left(v_get(k, j - 1, i), vc, v_get(k, j + 1, i)) -
                up3_left(v_get(k, j - 2, i), v_get(k, j - 1, i), vc)) / cfg.dy_m;
    } else {
        dvdy = (up3_left(v_get(k, j + 2, i), v_get(k, j + 1, i), vc) -
                up3_left(v_get(k, j + 1, i), vc, v_get(k, j - 1, i))) / cfg.dy_m;
    }
    double dvdz = 0.0;
    if (zeta_dot >= 0.0 && k > 0) {
        dvdz = (vc - v_get(k - 1, j, i)) / (0.5 * (g.dzeta[k] + g.dzeta[k - 1]));
    } else if (zeta_dot < 0.0 && k < nz - 1) {
        dvdz = (v_get(k + 1, j, i) - vc) / (0.5 * (g.dzeta[k + 1] + g.dzeta[k]));
    }

    double tend = -msf * (uc * dvdx + vc * dvdy) - zeta_dot * dvdz;

    if (cfg.use_coriolis) {
        const double f_v = 0.5 * (g.fcor[jl * nx + i] + g.fcor[jr * nx + i]);
        const double dm_dx = (read_2d(g.msf, jl, i + 1, ny, nx, cfg.bc_x, cfg.bc_y) -
                              read_2d(g.msf, jl, i - 1, ny, nx, cfg.bc_x, cfg.bc_y) +
                              read_2d(g.msf, jr, i + 1, ny, nx, cfg.bc_x, cfg.bc_y) -
                              read_2d(g.msf, jr, i - 1, ny, nx, cfg.bc_x, cfg.bc_y)) / (4.0 * cfg.dx_m);
        const double dm_dy = (g.msf[jr * nx + i] - g.msf[jl * nx + i]) / cfg.dy_m;
        tend -= (f_v + uc * dm_dy - vc * dm_dx) * uc;
    }

    if (cfg.km_const > 0.0) {
        const double lap_h = msf * msf * (
            (v_get(k, j, i + 1) - 2.0 * vc + v_get(k, j, i - 1)) / (cfg.dx_m * cfg.dx_m) +
            (v_get(k, j + 1, i) - 2.0 * vc + v_get(k, j - 1, i)) / (cfg.dy_m * cfg.dy_m)
        );
        const double dz_c = g_v * g.dzeta[k];
        double lap_v = 0.0;
        if (k < nz - 1) {
            const double dz_p = 0.5 * (dz_c + (1.0 + g.db_s[k + 1] * hs_v) * g.dzeta[k + 1]);
            lap_v += (v_get(k + 1, j, i) - vc) / (dz_p * dz_c);
        }
        if (k > 0) {
            const double dz_m = 0.5 * (dz_c + (1.0 + g.db_s[k - 1] * hs_v) * g.dzeta[k - 1]);
            lap_v -= (vc - v_get(k - 1, j, i)) / (dz_m * dz_c);
        }
        tend += cfg.km_const * (lap_h + lap_v);
    }

    if ((cfg.bc_x == 2 || cfg.bc_y == 2) && lbc_v != nullptr && cfg.lbc_tau_max_s > 0.0) {
        double tau = 0.0;
        if (cfg.bc_x == 2) {
            if (cfg.lbc_edges & 1) tau = fmax(tau, davies_weight(i, cfg.lbc_rim_width, cfg.lbc_tau_max_s));
            if (cfg.lbc_edges & 2) tau = fmax(tau, davies_weight(nx - 1 - i, cfg.lbc_rim_width, cfg.lbc_tau_max_s));
        }
        if (cfg.bc_y == 2) {
            if (cfg.lbc_edges & 4) tau = fmax(tau, davies_weight(j, cfg.lbc_rim_width, cfg.lbc_tau_max_s));
            if (cfg.lbc_edges & 8) tau = fmax(tau, davies_weight(ny - j, cfg.lbc_rim_width, cfg.lbc_tau_max_s));
        }
        if (tau > 0.0) {
            tend -= tau * (vc - static_cast<double>(lbc_v[idx]));
        }
    }

    if (cfg.has_iau && iau_v != nullptr) tend += static_cast<double>(iau_v[idx]);
    rv[idx] = static_cast<float>(tend);
}

__global__ void nh_slow_w_kernel(
    NhGridConfig cfg,
    DeviceStaticGrid g,
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
    const double msf = g.msf[j * nx + i];
    const double hs = g.hs[j * nx + i];
    const double ds_k = g.dzeta[k];
    const double ds_km = g.dzeta[k - 1];
    const double wt = ds_km / (ds_k + ds_km);

    const double u_w = (1.0 - wt) * 0.5 * (read_u(u, k - 1, j, i, nz, ny, nx, cfg.bc_x, cfg.bc_y) +
                                           read_u(u, k - 1, j, i + 1, nz, ny, nx, cfg.bc_x, cfg.bc_y)) +
                       wt         * 0.5 * (read_u(u, k, j, i, nz, ny, nx, cfg.bc_x, cfg.bc_y) +
                                           read_u(u, k, j, i + 1, nz, ny, nx, cfg.bc_x, cfg.bc_y));
    const double v_w = (1.0 - wt) * 0.5 * (read_v(v, k - 1, j, i, nz, ny, nx, cfg.bc_x, cfg.bc_y) +
                                           read_v(v, k - 1, j + 1, i, nz, ny, nx, cfg.bc_x, cfg.bc_y)) +
                       wt         * 0.5 * (read_v(v, k, j, i, nz, ny, nx, cfg.bc_x, cfg.bc_y) +
                                           read_v(v, k, j + 1, i, nz, ny, nx, cfg.bc_x, cfg.bc_y));
    double gz_w = 0.0;
    compute_contravariant_at_scalar_col(cfg, g, u, v, w, k, j, i, gz_w);
    const double g_w = 1.0 + g.db_w[k] * hs;
    const double zeta_dot = gz_w / fmax(g_w, 1e-4);

    auto w_get = [&](int32_t kk, int32_t jj, int32_t ii) -> double {
        return read_w(w, kk, jj, ii, nz, ny, nx, cfg.bc_x, cfg.bc_y);
    };
    const double wc = w_get(k, j, i);
    double dwdx = 0.0;
    if (u_w >= 0.0) {
        dwdx = (up3_left(w_get(k, j, i - 1), wc, w_get(k, j, i + 1)) -
                up3_left(w_get(k, j, i - 2), w_get(k, j, i - 1), wc)) / cfg.dx_m;
    } else {
        dwdx = (up3_left(w_get(k, j, i + 2), w_get(k, j, i + 1), wc) -
                up3_left(w_get(k, j, i + 1), wc, w_get(k, j, i - 1))) / cfg.dx_m;
    }
    double dwdy = 0.0;
    if (v_w >= 0.0) {
        dwdy = (up3_left(w_get(k, j - 1, i), wc, w_get(k, j + 1, i)) -
                up3_left(w_get(k, j - 2, i), w_get(k, j - 1, i), wc)) / cfg.dy_m;
    } else {
        dwdy = (up3_left(w_get(k, j + 2, i), w_get(k, j + 1, i), wc) -
                up3_left(w_get(k, j + 1, i), wc, w_get(k, j - 1, i))) / cfg.dy_m;
    }
    double dwdz = 0.0;
    if (zeta_dot >= 0.0) {
        dwdz = (wc - w_get(k - 1, j, i)) / ds_km;
    } else {
        dwdz = (w_get(k + 1, j, i) - wc) / ds_k;
    }

    double tend = -msf * (u_w * dwdx + v_w * dwdy) - zeta_dot * dwdz;

    if (cfg.km_const > 0.0) {
        const double lap_h = msf * msf * (
            (w_get(k, j, i + 1) - 2.0 * wc + w_get(k, j, i - 1)) / (cfg.dx_m * cfg.dx_m) +
            (w_get(k, j + 1, i) - 2.0 * wc + w_get(k, j - 1, i)) / (cfg.dy_m * cfg.dy_m)
        );
        const double dz_w = g_w * 0.5 * (ds_k + ds_km);
        const double dz_p = (1.0 + g.db_s[k] * hs) * ds_k;
        const double dz_m = (1.0 + g.db_s[k - 1] * hs) * ds_km;
        const double lap_v = ((w_get(k + 1, j, i) - wc) / dz_p - (wc - w_get(k - 1, j, i)) / dz_m) / dz_w;
        tend += cfg.km_const * (lap_h + lap_v);
    }

    if (cfg.bc_x == 2 || cfg.bc_y == 2) {
        double tau = 0.0;
        if (cfg.bc_x == 2) {
            if (cfg.lbc_edges & 1) tau = fmax(tau, davies_weight(i, cfg.lbc_rim_width, cfg.lbc_tau_max_s));
            if (cfg.lbc_edges & 2) tau = fmax(tau, davies_weight(nx - 1 - i, cfg.lbc_rim_width, cfg.lbc_tau_max_s));
        }
        if (cfg.bc_y == 2) {
            if (cfg.lbc_edges & 4) tau = fmax(tau, davies_weight(j, cfg.lbc_rim_width, cfg.lbc_tau_max_s));
            if (cfg.lbc_edges & 8) tau = fmax(tau, davies_weight(ny - 1 - j, cfg.lbc_rim_width, cfg.lbc_tau_max_s));
        }
        tend -= tau * wc;
    }

    rw[idx] = static_cast<float>(tend);
}

// ---------------------------------------------------------------------------
// Kernel 2: 3D Divergence (for acoustic divergence damping) + acoustic horiz step
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
    const double msf = g.msf[j * nx + i];
    const double hs = g.hs[j * nx + i];
    const double g_s = 1.0 + g.db_s[k] * hs;

    const double u_r = read_u(u, k, j, i + 1, nz, ny, nx, cfg.bc_x, cfg.bc_y);
    const double u_l = read_u(u, k, j, i,     nz, ny, nx, cfg.bc_x, cfg.bc_y);
    const double v_n = read_v(v, k, j + 1, i, nz, ny, nx, cfg.bc_x, cfg.bc_y);
    const double v_s = read_v(v, k, j,     i, nz, ny, nx, cfg.bc_x, cfg.bc_y);

    const double msf_r = read_2d(g.msf, j, i + 1, ny, nx, cfg.bc_x, cfg.bc_y);
    const double msf_l = read_2d(g.msf, j, i - 1, ny, nx, cfg.bc_x, cfg.bc_y);
    const double msf_n = read_2d(g.msf, j + 1, i, ny, nx, cfg.bc_x, cfg.bc_y);
    const double msf_s = read_2d(g.msf, j - 1, i, ny, nx, cfg.bc_x, cfg.bc_y);

    const double hs_r = read_2d(g.hs, j, i + 1, ny, nx, cfg.bc_x, cfg.bc_y);
    const double hs_l = read_2d(g.hs, j, i - 1, ny, nx, cfg.bc_x, cfg.bc_y);
    const double hs_n = read_2d(g.hs, j + 1, i, ny, nx, cfg.bc_x, cfg.bc_y);
    const double hs_s = read_2d(g.hs, j - 1, i, ny, nx, cfg.bc_x, cfg.bc_y);

    const double gu_r = 1.0 + g.db_s[k] * 0.5 * (hs + hs_r);
    const double gu_l = 1.0 + g.db_s[k] * 0.5 * (hs + hs_l);
    const double gv_n = 1.0 + g.db_s[k] * 0.5 * (hs + hs_n);
    const double gv_s = 1.0 + g.db_s[k] * 0.5 * (hs + hs_s);

    const double mu_r = 0.5 * (msf + msf_r);
    const double mu_l = 0.5 * (msf + msf_l);
    const double mv_n = 0.5 * (msf + msf_n);
    const double mv_s = 0.5 * (msf + msf_s);

    const double hdiv = msf * msf * (
        (gu_r * u_r / mu_r - gu_l * u_l / mu_l) / cfg.dx_m +
        (gv_n * v_n / mv_n - gv_s * v_s / mv_s) / cfg.dy_m
    );

    double gz_lo = 0.0, gz_hi = 0.0;
    compute_contravariant_at_scalar_col(cfg, g, u, v, w, k,     j, i, gz_lo);
    compute_contravariant_at_scalar_col(cfg, g, u, v, w, k + 1, j, i, gz_hi);
    const double vdiv = (gz_hi - gz_lo) / g.dzeta[k];

    div3d[idx] = static_cast<float>((hdiv + vdiv) / fmax(g_s, 1e-4));
}

__global__ void nh_acoustic_u_kernel(
    NhGridConfig cfg,
    DeviceStaticGrid g,
    double dts,
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

    const double hs_l = g.hs[j * nx + il];
    const double hs_r = g.hs[j * nx + ir];
    const double msf_u = 0.5 * (g.msf[j * nx + il] + g.msf[j * nx + ir]);

    double dpidx_z = 0.0;
    if (cfg.bc_x == 1 || (i > 0 && i < nx)) {
        if (hs_l == hs_r) {
            dpidx_z = (static_cast<double>(pip[idxr]) - static_cast<double>(pip[idxl])) / cfg.dx_m;
        } else {
            const double z_l = g.zeta_s[k] + g.b_s[k] * hs_l;
            const double z_r = g.zeta_s[k] + g.b_s[k] * hs_r;
            const double z_u = 0.5 * (z_l + z_r);
            const double pi_l = interp_pi_at_z(cfg, g, pip, thp, qv, qc, qr, qi, qs, qg, k, j, il, hs_l, z_u);
            const double pi_r = interp_pi_at_z(cfg, g, pip, thp, qv, qc, qr, qi, qs, qg, k, j, ir, hs_r, z_u);
            dpidx_z = (pi_r - pi_l) / cfg.dx_m;
        }
    }

    const double z_u = g.zeta_s[k] + g.b_s[k] * 0.5 * (hs_l + hs_r);
    double thb, pib, dthdz, rhob, cs2b, qvl, qhl, qvr, qhr;
    eval_ref_state(z_u, cfg.theta_sfc_k, cfg.p_sfc_pa, cfg.n_ref_s, thb, pib, dthdz, rhob, cs2b);
    const double vf_l = moisture_load_scalar(cfg.has_moist, idxl, qv, qc, qr, qi, qs, qg, qvl, qhl);
    const double vf_r = moisture_load_scalar(cfg.has_moist, idxr, qv, qc, qr, qi, qs, qg, qvr, qhr);
    const double thv_u = 0.5 * (
        (thb + static_cast<double>(thp[idxl])) * vf_l +
        (thb + static_cast<double>(thp[idxr])) * vf_r
    );

    double ddiv_dx = 0.0;
    if (cfg.div_damp_coeff > 0.0 && div3d != nullptr && (cfg.bc_x == 1 || (i > 0 && i < nx))) {
        const double dmin = fmin(cfg.dx_m, cfg.dy_m);
        const double nu_div = cfg.div_damp_coeff * (dmin * dmin) / fmax(dts, 1e-6);
        ddiv_dx = nu_div * msf_u * (static_cast<double>(div3d[idxr]) - static_cast<double>(div3d[idxl])) / cfg.dx_m;
    }

    const double u_next = static_cast<double>(u[idx])
                        + dts * (static_cast<double>(ru[idx]) - CP_D * thv_u * msf_u * dpidx_z + ddiv_dx);
    u[idx] = static_cast<float>(u_next);
}

__global__ void nh_acoustic_v_kernel(
    NhGridConfig cfg,
    DeviceStaticGrid g,
    double dts,
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

    const double hs_l = g.hs[jl * nx + i];
    const double hs_r = g.hs[jr * nx + i];
    const double msf_v = 0.5 * (g.msf[jl * nx + i] + g.msf[jr * nx + i]);

    double dpidy_z = 0.0;
    if (cfg.bc_y == 1 || (j > 0 && j < ny)) {
        if (hs_l == hs_r) {
            dpidy_z = (static_cast<double>(pip[idxr]) - static_cast<double>(pip[idxl])) / cfg.dy_m;
        } else {
            const double z_l = g.zeta_s[k] + g.b_s[k] * hs_l;
            const double z_r = g.zeta_s[k] + g.b_s[k] * hs_r;
            const double z_v = 0.5 * (z_l + z_r);
            const double pi_l = interp_pi_at_z(cfg, g, pip, thp, qv, qc, qr, qi, qs, qg, k, jl, i, hs_l, z_v);
            const double pi_r = interp_pi_at_z(cfg, g, pip, thp, qv, qc, qr, qi, qs, qg, k, jr, i, hs_r, z_v);
            dpidy_z = (pi_r - pi_l) / cfg.dy_m;
        }
    }

    const double z_v = g.zeta_s[k] + g.b_s[k] * 0.5 * (hs_l + hs_r);
    double thb, pib, dthdz, rhob, cs2b, qvl, qhl, qvr, qhr;
    eval_ref_state(z_v, cfg.theta_sfc_k, cfg.p_sfc_pa, cfg.n_ref_s, thb, pib, dthdz, rhob, cs2b);
    const double vf_l = moisture_load_scalar(cfg.has_moist, idxl, qv, qc, qr, qi, qs, qg, qvl, qhl);
    const double vf_r = moisture_load_scalar(cfg.has_moist, idxr, qv, qc, qr, qi, qs, qg, qvr, qhr);
    const double thv_v = 0.5 * (
        (thb + static_cast<double>(thp[idxl])) * vf_l +
        (thb + static_cast<double>(thp[idxr])) * vf_r
    );

    double ddiv_dy = 0.0;
    if (cfg.div_damp_coeff > 0.0 && div3d != nullptr && (cfg.bc_y == 1 || (j > 0 && j < ny))) {
        const double dmin = fmin(cfg.dx_m, cfg.dy_m);
        const double nu_div = cfg.div_damp_coeff * (dmin * dmin) / fmax(dts, 1e-6);
        ddiv_dy = nu_div * msf_v * (static_cast<double>(div3d[idxr]) - static_cast<double>(div3d[idxl])) / cfg.dy_m;
    }

    const double v_next = static_cast<double>(v[idx])
                        + dts * (static_cast<double>(rv[idx]) - CP_D * thv_v * msf_v * dpidy_z + ddiv_dy);
    v[idx] = static_cast<float>(v_next);
}

// ---------------------------------------------------------------------------
// Kernel 3: Column-implicit HEVI vertical (w, pi') Thomas solver + theta' update
// Uses global coalesced d_tri_c and d_tri_d (zero per-thread nz arrays!).
// ---------------------------------------------------------------------------
__global__ void nh_acoustic_vert_implicit_kernel(
    NhGridConfig cfg,
    DeviceStaticGrid g,
    double dts,
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
    double* tri_c,
    double* tri_d
) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= cfg.ny || i >= cfg.nx) return;

    const int32_t nz = cfg.nz, ny = cfg.ny, nx = cfg.nx;
    const size_t col_stride = static_cast<size_t>(ny) * nx;
    const size_t h_idx = static_cast<size_t>(j) * nx + i;

    const double hs = g.hs[h_idx];
    const double msf = g.msf[h_idx];
    const double dh_dx = (read_2d(g.hs, j, i + 1, ny, nx, cfg.bc_x, cfg.bc_y) -
                          read_2d(g.hs, j, i - 1, ny, nx, cfg.bc_x, cfg.bc_y)) / (2.0 * cfg.dx_m);
    const double dh_dy = (read_2d(g.hs, j + 1, i, ny, nx, cfg.bc_x, cfg.bc_y) -
                          read_2d(g.hs, j - 1, i, ny, nx, cfg.bc_x, cfg.bc_y)) / (2.0 * cfg.dy_m);
    const double slope_h = msf * dh_dx;
    const double slope_y = msf * dh_dy;

    const double msf_r = read_2d(g.msf, j, i + 1, ny, nx, cfg.bc_x, cfg.bc_y);
    const double msf_l = read_2d(g.msf, j, i - 1, ny, nx, cfg.bc_x, cfg.bc_y);
    const double msf_n = read_2d(g.msf, j + 1, i, ny, nx, cfg.bc_x, cfg.bc_y);
    const double msf_s = read_2d(g.msf, j - 1, i, ny, nx, cfg.bc_x, cfg.bc_y);
    const double mu_r = 0.5 * (msf + msf_r), mu_l = 0.5 * (msf + msf_l);
    const double mv_n = 0.5 * (msf + msf_n), mv_s = 0.5 * (msf + msf_s);

    const double hs_r = read_2d(g.hs, j, i + 1, ny, nx, cfg.bc_x, cfg.bc_y);
    const double hs_l = read_2d(g.hs, j, i - 1, ny, nx, cfg.bc_x, cfg.bc_y);
    const double hs_n = read_2d(g.hs, j + 1, i, ny, nx, cfg.bc_x, cfg.bc_y);
    const double hs_s = read_2d(g.hs, j - 1, i, ny, nx, cfg.bc_x, cfg.bc_y);
    const double hsu_r = 0.5 * (hs + hs_r), hsu_l = 0.5 * (hs + hs_l);
    const double hsv_n = 0.5 * (hs + hs_n), hsv_s = 0.5 * (hs + hs_s);

    const double beta = cfg.beta_offcenter;
    const double omb = 1.0 - beta;

    auto omega_h_at = [&](int32_t k_iface) -> double {
        if (k_iface >= nz) return 0.0;
        double uw = 0.0, vw = 0.0;
        if (k_iface == 0) {
            uw = 0.5 * (read_u(u, 0, j, i, nz, ny, nx, cfg.bc_x, cfg.bc_y) +
                        read_u(u, 0, j, i + 1, nz, ny, nx, cfg.bc_x, cfg.bc_y));
            vw = 0.5 * (read_v(v, 0, j, i, nz, ny, nx, cfg.bc_x, cfg.bc_y) +
                        read_v(v, 0, j + 1, i, nz, ny, nx, cfg.bc_x, cfg.bc_y));
        } else {
            const double ds_k = g.dzeta[k_iface];
            const double ds_km = g.dzeta[k_iface - 1];
            const double wt = ds_km / (ds_k + ds_km);
            uw = (1.0 - wt) * 0.5 * (read_u(u, k_iface - 1, j, i, nz, ny, nx, cfg.bc_x, cfg.bc_y) +
                                     read_u(u, k_iface - 1, j, i + 1, nz, ny, nx, cfg.bc_x, cfg.bc_y)) +
                 wt         * 0.5 * (read_u(u, k_iface, j, i, nz, ny, nx, cfg.bc_x, cfg.bc_y) +
                                     read_u(u, k_iface, j, i + 1, nz, ny, nx, cfg.bc_x, cfg.bc_y));
            vw = (1.0 - wt) * 0.5 * (read_v(v, k_iface - 1, j, i, nz, ny, nx, cfg.bc_x, cfg.bc_y) +
                                     read_v(v, k_iface - 1, j + 1, i, nz, ny, nx, cfg.bc_x, cfg.bc_y)) +
                 wt         * 0.5 * (read_v(v, k_iface, j, i, nz, ny, nx, cfg.bc_x, cfg.bc_y) +
                                     read_v(v, k_iface, j + 1, i, nz, ny, nx, cfg.bc_x, cfg.bc_y));
        }
        return -g.b_w[k_iface] * (slope_h * uw + slope_y * vw);
    };

    const double w0_new = -omega_h_at(0);
    w[h_idx] = static_cast<float>(w0_new);
    w[static_cast<size_t>(nz) * col_stride + h_idx] = 0.0f;

    // Pass 1: Precompute scalar layer hat_pi[k] -> store temporarily in tri_d[k]?
    // Wait: tri_c & tri_d have size nz+1! Wait - during forward elimination k=1..nz-1 we ONLY
    // need hat_pi[k] at levels k-1 and k! So we can keep sliding variables (hat_pi_prev, a_p_prev, a_m_prev, pi_m_prev)
    // and compute hat_pi[k] on the fly in Pass 1 (storing hat_pi into pip only AFTER Thomas solve,
    // wait: NO, pip[k] IS needed to compute hat_pi[k], OR store hat_pi[k] in tri_c[k] during a pre-pass?
    // Wait: we need both hat_pi[k] and Thomas c'[k]! What about storing hat_pi[k] in pip[k]? NO, (1-beta)*pi_m + beta*hat_pi needs pi_m!).
    // Wait: compute (pi_eff[k] = (1-beta)*pi_m[k] + beta*hat_pi[k]) and store hat_pi[k] into d_tri_c / NO:
    // Look! At k=0..nz-1, computing pi_eff_k and storing pi_hat_k in w/pip:
    // Actually we can pass through k=0..nz-1 computing pi_hat_k in 2 lines on the fly from scalar u,v,w,
    // OR compute hat_pi_k at step k just once by keeping ONLY `prev` scalar registers in Pass 1?!
    // CHECK: Does Pass 2 (backward substitution) need hat_pi[k] to update pip[k] = hat_pi[k] - dts*beta*(a_p*w[k+1] - a_m*w[k])?
    // YES! So storing hat_pi[k] temporarily in pip[k] AFTER saving pi_eff[k] into RHS? Wait: why not check!
    // During Pass 1 at level k, once pi_hat[k] is computed:
    //   RHS[k] ONLY uses pi_eff[k] and pi_eff[k-1] (`pi_eff = (1-beta)*pi_m + beta*pi_hat`).
    //   Notice after RHS[k] uses pi_eff[k-1] and pi_eff[k], pip[k-1] is NEVER read again in Pass 1!
    //   AND Pass 2 updates pip[k] from pi_hat[k] and w[k+1], w[k]!
    //   SO in Pass 1, as soon as we compute pi_hat[k] from pi_m[k], we can overwrite pip[k-1] = pi_hat[k-1] (keeping pi_hat_prev in a register!),
    //   and overwrite pip[nz-1] = pi_hat[nz-1] at the end of Pass 1!! Zero extra arrays needed AND 100% coalesced!!

    double om_lo = omega_h_at(0);
    double ap_prev = 0.0, am_prev = 0.0, pi_eff_prev = 0.0, pi_hat_prev = 0.0;
    double wm_prev = static_cast<double>(w[h_idx]);

    for (int32_t k = 0; k < nz; ++k) {
        const size_t idx = static_cast<size_t>(k) * col_stride + h_idx;
        const double z_s = g.zeta_s[k] + g.b_s[k] * hs;
        const double g_s = 1.0 + g.db_s[k] * hs;
        const double dz_s = g_s * g.dzeta[k];

        double thb, pib, dthdz, rhob, cs2b;
        eval_ref_state(z_s, cfg.theta_sfc_k, cfg.p_sfc_pa, cfg.n_ref_s, thb, pib, dthdz, rhob, cs2b);
        const double c_ac = (R_D / CV_D) * pib;

        const double om_hi = omega_h_at(k + 1);
        const double gu_r = 1.0 + g.db_s[k] * hsu_r, gu_l = 1.0 + g.db_s[k] * hsu_l;
        const double gv_n = 1.0 + g.db_s[k] * hsv_n, gv_s = 1.0 + g.db_s[k] * hsv_s;
        const double u_r = read_u(u, k, j, i + 1, nz, ny, nx, cfg.bc_x, cfg.bc_y);
        const double u_l = read_u(u, k, j, i,     nz, ny, nx, cfg.bc_x, cfg.bc_y);
        const double v_n = read_v(v, k, j + 1, i, nz, ny, nx, cfg.bc_x, cfg.bc_y);
        const double v_s = read_v(v, k, j,     i, nz, ny, nx, cfg.bc_x, cfg.bc_y);

        const double d_h = (
            msf * msf * (
                (gu_r * u_r / mu_r - gu_l * u_l / mu_l) / cfg.dx_m +
                (gv_n * v_n / mv_n - gv_s * v_s / mv_s) / cfg.dy_m
            ) + (om_hi - om_lo) / g.dzeta[k]
        ) / fmax(g_s, 1e-4);
        om_lo = om_hi;

        const double ap = c_ac / dz_s - GRAVITY / (2.0 * CP_D * thb);
        const double am = c_ac / dz_s + GRAVITY / (2.0 * CP_D * thb);
        const double wm_cur = wm_prev;
        const double wm_nxt = (k + 1 < nz) ? static_cast<double>(w[static_cast<size_t>(k + 1) * col_stride + h_idx]) : 0.0;
        wm_prev = wm_nxt;

        const double pi_m = static_cast<double>(pip[idx]);
        const double pi_hat = pi_m + dts * (
            static_cast<double>(rpi[idx]) - c_ac * d_h - omb * (ap * wm_nxt - am * wm_cur)
        );
        const double pi_eff = omb * pi_m + beta * pi_hat;

        if (k > 0) {
            // Overwrite pip[k-1] with pi_hat_prev (safe: pi_m[k-1] already consumed!)
            pip[static_cast<size_t>(k - 1) * col_stride + h_idx] = static_cast<float>(pi_hat_prev);

            const size_t w_idx = static_cast<size_t>(k) * col_stride + h_idx;
            const double z_w = g.zeta_w[k] + g.b_w[k] * hs;
            const double g_w = 1.0 + g.db_w[k] * hs;
            const double dz_w = g_w * 0.5 * (g.dzeta[k] + g.dzeta[k - 1]);

            double thb_w, pib_w, dth_w, rhob_w, cs2_w;
            eval_ref_state(z_w, cfg.theta_sfc_k, cfg.p_sfc_pa, cfg.n_ref_s, thb_w, pib_w, dth_w, rhob_w, cs2_w);

            const size_t idx_km = static_cast<size_t>(k - 1) * col_stride + h_idx;
            double qvm, qhm, qvp, qhp;
            const double vfm = moisture_load_scalar(cfg.has_moist, idx_km, qv, qc, qr, qi, qs, qg, qvm, qhm);
            const double vfp = moisture_load_scalar(cfg.has_moist, idx,    qv, qc, qr, qi, qs, qg, qvp, qhp);

            double thb_km, pib_km, d_km, r_km, c_km;
            eval_ref_state(g.zeta_s[k - 1] + g.b_s[k - 1] * hs, cfg.theta_sfc_k, cfg.p_sfc_pa, cfg.n_ref_s, thb_km, pib_km, d_km, r_km, c_km);

            const double wt = g.dzeta[k - 1] / (g.dzeta[k] + g.dzeta[k - 1]);
            const double dthv_km = (thb_km + static_cast<double>(thp[idx_km])) * vfm - thb_km;
            const double dthv_kp = (thb    + static_cast<double>(thp[idx]))    * vfp - thb;
            const double dthv_w = (1.0 - wt) * dthv_km + wt * dthv_kp;
            const double thv_w = thb_w + dthv_w;
            const double buoy = GRAVITY * dthv_w / thb_w;

            const double pw = dts * (CP_D * thv_w) / dz_w;
            const double tau_sp = sponge_tau(z_w, cfg.z_top_m, cfg.sponge_depth_m, cfg.sponge_tau_max_s);

            const double A_k = -pw * dts * beta * beta * am_prev;
            const double B_k = 1.0 + dts * tau_sp + pw * dts * beta * beta * (am + ap_prev);
            const double C_k = (k < nz - 1) ? (-pw * dts * beta * beta * ap) : 0.0;

            double rhs_k = static_cast<double>(w[w_idx])
                         + dts * (static_cast<double>(rw[w_idx]) + buoy)
                         - pw * (pi_eff - pi_eff_prev);
            if (k == 1) rhs_k -= A_k * w0_new;

            if (k == 1) {
                tri_c[w_idx] = C_k / B_k;
                tri_d[w_idx] = rhs_k / B_k;
            } else {
                const size_t wm_idx = static_cast<size_t>(k - 1) * col_stride + h_idx;
                const double denom = B_k - A_k * tri_c[wm_idx];
                tri_c[w_idx] = C_k / denom;
                tri_d[w_idx] = (rhs_k - A_k * tri_d[wm_idx]) / denom;
            }
        }

        ap_prev = ap;
        am_prev = am;
        pi_eff_prev = pi_eff;
        pi_hat_prev = pi_hat;
    }
    pip[static_cast<size_t>(nz - 1) * col_stride + h_idx] = static_cast<float>(pi_hat_prev);

    // Pass 2: Backward substitution for w[nz-1..1], then update pi_p[nz-1..0] and theta_p[nz-1..0]
    double w_above = 0.0; // w[nz] = 0
    for (int32_t k = nz - 1; k >= 0; --k) {
        double w_below = w0_new;
        if (k > 0) {
            const size_t w_idx = static_cast<size_t>(k) * col_stride + h_idx;
            w_below = tri_d[w_idx] - ((k < nz - 1) ? tri_c[w_idx] * w_above : 0.0);
            w[w_idx] = static_cast<float>(w_below);
        }

        const size_t idx = static_cast<size_t>(k) * col_stride + h_idx;
        const double z_s = g.zeta_s[k] + g.b_s[k] * hs;
        const double g_s = 1.0 + g.db_s[k] * hs;
        const double dz_s = g_s * g.dzeta[k];
        double thb, pib, dthdz, rhob, cs2b;
        eval_ref_state(z_s, cfg.theta_sfc_k, cfg.p_sfc_pa, cfg.n_ref_s, thb, pib, dthdz, rhob, cs2b);
        const double c_ac = (R_D / CV_D) * pib;
        const double ap = c_ac / dz_s - GRAVITY / (2.0 * CP_D * thb);
        const double am = c_ac / dz_s + GRAVITY / (2.0 * CP_D * thb);

        const double pi_hat = static_cast<double>(pip[idx]);
        const double pi_new = pi_hat - dts * beta * (ap * w_above - am * w_below);
        pip[idx] = static_cast<float>(pi_new);

        // Symplectic Forward-Backward vertical theta_bar advection using updated w^{m+1}
        double thb_w_lo, thb_w_hi, pi0_d, d0_d, r0_d, c0_d;
        eval_ref_state(g.zeta_w[k]     + g.b_w[k]     * hs, cfg.theta_sfc_k, cfg.p_sfc_pa, cfg.n_ref_s, thb_w_lo, pi0_d, d0_d, r0_d, c0_d);
        eval_ref_state(g.zeta_w[k + 1] + g.b_w[k + 1] * hs, cfg.theta_sfc_k, cfg.p_sfc_pa, cfg.n_ref_s, thb_w_hi, pi0_d, d0_d, r0_d, c0_d);
        const double w_avg = 0.5 * (w_above + w_below);
        const double th_new = static_cast<double>(thp[idx])
                            + dts * (static_cast<double>(rth[idx]) - w_avg * (thb_w_hi - thb_w_lo) / dz_s);
        thp[idx] = static_cast<float>(th_new);

        w_above = w_below;
    }
}

// ---------------------------------------------------------------------------
// Kernel 4: Positive-definite conservative 3D FCT Zalesak tracer advection
// Conserves integral sum(rho_bar * G_s * dzeta / msf^2 * q) >= 0 everywhere.
// ---------------------------------------------------------------------------
__device__ inline double cell_mass_vol(
    const NhGridConfig& cfg, const DeviceStaticGrid& g, int32_t k, int32_t j, int32_t i
) {
    const double hs = g.hs[j * cfg.nx + i];
    const double msf = g.msf[j * cfg.nx + i];
    const double z_s = g.zeta_s[k] + g.b_s[k] * hs;
    const double g_s = 1.0 + g.db_s[k] * hs;
    double thb, pib, dthdz, rhob, cs2b;
    eval_ref_state(z_s, cfg.theta_sfc_k, cfg.p_sfc_pa, cfg.n_ref_s, thb, pib, dthdz, rhob, cs2b);
    return rhob * g_s * g.dzeta[k] / (msf * msf);
}

__device__ inline double face_flux_x(
    const NhGridConfig& cfg, const DeviceStaticGrid& g,
    const float* u, int32_t k, int32_t j, int32_t i_face
) {
    if (cfg.bc_x != 1 && (i_face <= 0 || i_face >= cfg.nx)) return 0.0;
    const int32_t il = (cfg.bc_x == 1) ? wrap_idx(i_face - 1, cfg.nx) : (i_face - 1);
    const int32_t ir = (cfg.bc_x == 1) ? wrap_idx(i_face, cfg.nx) : i_face;
    const double hs_u = 0.5 * (g.hs[j * cfg.nx + il] + g.hs[j * cfg.nx + ir]);
    const double msf_u = 0.5 * (g.msf[j * cfg.nx + il] + g.msf[j * cfg.nx + ir]);
    const double z_u = g.zeta_s[k] + g.b_s[k] * hs_u;
    const double g_u = 1.0 + g.db_s[k] * hs_u;
    double thb, pib, dthdz, rhob, cs2b;
    eval_ref_state(z_u, cfg.theta_sfc_k, cfg.p_sfc_pa, cfg.n_ref_s, thb, pib, dthdz, rhob, cs2b);
    const double uval = read_u(u, k, j, i_face, cfg.nz, cfg.ny, cfg.nx, cfg.bc_x, cfg.bc_y);
    return rhob * g_u * g.dzeta[k] * uval / msf_u;
}

__device__ inline double face_flux_y(
    const NhGridConfig& cfg, const DeviceStaticGrid& g,
    const float* v, int32_t k, int32_t j_face, int32_t i
) {
    if (cfg.bc_y != 1 && (j_face <= 0 || j_face >= cfg.ny)) return 0.0;
    const int32_t jl = (cfg.bc_y == 1) ? wrap_idx(j_face - 1, cfg.ny) : (j_face - 1);
    const int32_t jr = (cfg.bc_y == 1) ? wrap_idx(j_face, cfg.ny) : j_face;
    const double hs_v = 0.5 * (g.hs[jl * cfg.nx + i] + g.hs[jr * cfg.nx + i]);
    const double msf_v = 0.5 * (g.msf[jl * cfg.nx + i] + g.msf[jr * cfg.nx + i]);
    const double z_v = g.zeta_s[k] + g.b_s[k] * hs_v;
    const double g_v = 1.0 + g.db_s[k] * hs_v;
    double thb, pib, dthdz, rhob, cs2b;
    eval_ref_state(z_v, cfg.theta_sfc_k, cfg.p_sfc_pa, cfg.n_ref_s, thb, pib, dthdz, rhob, cs2b);
    const double vval = read_v(v, k, j_face, i, cfg.nz, cfg.ny, cfg.nx, cfg.bc_x, cfg.bc_y);
    return rhob * g_v * g.dzeta[k] * vval / msf_v;
}

__device__ inline double face_flux_z(
    const NhGridConfig& cfg, const DeviceStaticGrid& g,
    const float* u, const float* v, const float* w, int32_t k_face, int32_t j, int32_t i
) {
    if (k_face <= 0 || k_face >= cfg.nz) return 0.0;
    const double hs = g.hs[j * cfg.nx + i];
    const double msf = g.msf[j * cfg.nx + i];
    const double z_w = g.zeta_w[k_face] + g.b_w[k_face] * hs;
    double thb, pib, dthdz, rhob, cs2b, gz = 0.0;
    eval_ref_state(z_w, cfg.theta_sfc_k, cfg.p_sfc_pa, cfg.n_ref_s, thb, pib, dthdz, rhob, cs2b);
    compute_contravariant_at_scalar_col(cfg, g, u, v, w, k_face, j, i, gz);
    return rhob * gz / (msf * msf);
}

__device__ inline void compute_fluxes_for_cell(
    const NhGridConfig& cfg, const DeviceStaticGrid& g,
    const float* q, const float* u, const float* v, const float* w,
    int32_t k, int32_t j, int32_t i,
    double& fx_l_low, double& fx_l_hi,
    double& fy_l_low, double& fy_l_hi,
    double& fz_l_low, double& fz_l_hi
) {
    const int32_t nz = cfg.nz, ny = cfg.ny, nx = cfg.nx;
    // Left x-face (i - 1/2)
    const double ux = face_flux_x(cfg, g, u, k, j, i);
    const double q_xl_up = (ux >= 0.0)
        ? read_s(q, k, j, i - 1, nz, ny, nx, cfg.bc_x, cfg.bc_y)
        : read_s(q, k, j, i,     nz, ny, nx, cfg.bc_x, cfg.bc_y);
    const double q_xl_ho = horiz_face_val_x(q, k, j, i, ux, nz, ny, nx, cfg.bc_x, cfg.bc_y, cfg.adv_order);
    fx_l_low = ux * q_xl_up;
    fx_l_hi  = ux * fmax(0.0, q_xl_ho);

    // South y-face (j - 1/2)
    const double vy = face_flux_y(cfg, g, v, k, j, i);
    const double q_yl_up = (vy >= 0.0)
        ? read_s(q, k, j - 1, i, nz, ny, nx, cfg.bc_x, cfg.bc_y)
        : read_s(q, k, j,     i, nz, ny, nx, cfg.bc_x, cfg.bc_y);
    const double q_yl_ho = horiz_face_val_y(q, k, j, i, vy, nz, ny, nx, cfg.bc_x, cfg.bc_y, cfg.adv_order);
    fy_l_low = vy * q_yl_up;
    fy_l_hi  = vy * fmax(0.0, q_yl_ho);

    // Bottom z-face (k - 1/2)
    const double wz = face_flux_z(cfg, g, u, v, w, k, j, i);
    const double q_zl_up = (wz >= 0.0)
        ? read_s(q, k - 1, j, i, nz, ny, nx, cfg.bc_x, cfg.bc_y)
        : read_s(q, k,     j, i, nz, ny, nx, cfg.bc_x, cfg.bc_y);
    const double q_zl_ho = vert_tvd_face_val(q, k, j, i, wz, g.dzeta, nz, ny, nx, cfg.bc_x, cfg.bc_y);
    fz_l_low = wz * q_zl_up;
    fz_l_hi  = wz * fmax(0.0, q_zl_ho);
}

__global__ void nh_tracer_cfl_kernel(
    NhGridConfig cfg,
    DeviceStaticGrid g,
    double dt_s,
    const float* u,
    const float* v,
    const float* w,
    uint32_t* max_cfl_u32
) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    const int32_t k = blockIdx.z;
    float cfl = 0.0f;
    if (k < cfg.nz && j < cfg.ny && i < cfg.nx) {
        const double vol = fmax(cell_mass_vol(cfg, g, k, j, i), 1e-12);
        const double fx_r = face_flux_x(cfg, g, u, k, j, i + 1);
        const double fx_l = face_flux_x(cfg, g, u, k, j, i);
        const double fy_n = face_flux_y(cfg, g, v, k, j + 1, i);
        const double fy_s = face_flux_y(cfg, g, v, k, j, i);
        const double fz_t = face_flux_z(cfg, g, u, v, w, k + 1, j, i);
        const double fz_b = face_flux_z(cfg, g, u, v, w, k, j, i);
        const double out = (fmax(0.0, fx_r) + fmax(0.0, -fx_l)) / cfg.dx_m
                         + (fmax(0.0, fy_n) + fmax(0.0, -fy_s)) / cfg.dy_m
                         +  fmax(0.0, fz_t) + fmax(0.0, -fz_b);
        cfl = static_cast<float>((dt_s * out) / vol);
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
    double dt_q,
    const float* u,
    const float* v,
    const float* w,
    const float* q,
    float* q_low,
    float* r_out
) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    const int32_t k = blockIdx.z;
    if (k >= cfg.nz || j >= cfg.ny || i >= cfg.nx) return;

    const size_t idx = static_cast<size_t>(k) * cfg.ny * cfg.nx + static_cast<size_t>(j) * cfg.nx + i;
    const double vol = fmax(cell_mass_vol(cfg, g, k, j, i), 1e-12);

    double fxl_low, fxl_hi, fyl_low, fyl_hi, fzl_low, fzl_hi;
    double fxr_low, fxr_hi, fyr_low, fyr_hi, fzr_low, fzr_hi;
    compute_fluxes_for_cell(cfg, g, q, u, v, w, k, j, i, fxl_low, fxl_hi, fyl_low, fyl_hi, fzl_low, fzl_hi);

    // Right / North / Top face fluxes
    {
        const double ux = face_flux_x(cfg, g, u, k, j, i + 1);
        const double q_up = (ux >= 0.0)
            ? read_s(q, k, j, i,     cfg.nz, cfg.ny, cfg.nx, cfg.bc_x, cfg.bc_y)
            : read_s(q, k, j, i + 1, cfg.nz, cfg.ny, cfg.nx, cfg.bc_x, cfg.bc_y);
        const double q_ho = horiz_face_val_x(q, k, j, i + 1, ux, cfg.nz, cfg.ny, cfg.nx, cfg.bc_x, cfg.bc_y, cfg.adv_order);
        fxr_low = ux * q_up;
        fxr_hi  = ux * fmax(0.0, q_ho);
    }
    {
        const double vy = face_flux_y(cfg, g, v, k, j + 1, i);
        const double q_up = (vy >= 0.0)
            ? read_s(q, k, j,     i, cfg.nz, cfg.ny, cfg.nx, cfg.bc_x, cfg.bc_y)
            : read_s(q, k, j + 1, i, cfg.nz, cfg.ny, cfg.nx, cfg.bc_x, cfg.bc_y);
        const double q_ho = horiz_face_val_y(q, k, j + 1, i, vy, cfg.nz, cfg.ny, cfg.nx, cfg.bc_x, cfg.bc_y, cfg.adv_order);
        fyr_low = vy * q_up;
        fyr_hi  = vy * fmax(0.0, q_ho);
    }
    {
        const double wz = face_flux_z(cfg, g, u, v, w, k + 1, j, i);
        const double q_up = (wz >= 0.0)
            ? read_s(q, k,     j, i, cfg.nz, cfg.ny, cfg.nx, cfg.bc_x, cfg.bc_y)
            : read_s(q, k + 1, j, i, cfg.nz, cfg.ny, cfg.nx, cfg.bc_x, cfg.bc_y);
        const double q_ho = vert_tvd_face_val(q, k + 1, j, i, wz, g.dzeta, cfg.nz, cfg.ny, cfg.nx, cfg.bc_x, cfg.bc_y);
        fzr_low = wz * q_up;
        fzr_hi  = wz * fmax(0.0, q_ho);
    }

    const double q0 = fmax(0.0, static_cast<double>(q[idx]));
    const double div_low = (fxr_low - fxl_low) / cfg.dx_m + (fyr_low - fyl_low) / cfg.dy_m + (fzr_low - fzl_low);
    const double qlow_val = fmax(0.0, q0 - (dt_q / vol) * div_low);
    q_low[idx] = static_cast<float>(qlow_val);

    const double ax_r = fxr_hi - fxr_low;
    const double ax_l = fxl_hi - fxl_low;
    const double ay_n = fyr_hi - fyr_low;
    const double ay_s = fyl_hi - fyl_low;
    const double az_t = fzr_hi - fzr_low;
    const double az_b = fzl_hi - fzl_low;

    const double p_out = dt_q * (
        (fmax(0.0, ax_r) + fmax(0.0, -ax_l)) / cfg.dx_m +
        (fmax(0.0, ay_n) + fmax(0.0, -ay_s)) / cfg.dy_m +
        fmax(0.0, az_t) + fmax(0.0, -az_b)
    );
    const double q_avail = vol * qlow_val;
    r_out[idx] = static_cast<float>((p_out > 1e-30) ? fmin(1.0, 0.999999 * (q_avail / p_out)) : 1.0);
}

__global__ void nh_tracer_fct_pass2_kernel(
    NhGridConfig cfg,
    DeviceStaticGrid g,
    double dt_q,
    const float* u,
    const float* v,
    const float* w,
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
    const double vol = fmax(cell_mass_vol(cfg, g, k, j, i), 1e-12);
    const double rc = static_cast<double>(r_out[idx]);

    auto limited_x = [&](int32_t iface) -> double {
        if (cfg.bc_x != 1 && (iface <= 0 || iface >= nx)) return 0.0;
        const double ux = face_flux_x(cfg, g, u, k, j, iface);
        const double q_up = (ux >= 0.0)
            ? read_s(q_old, k, j, iface - 1, nz, ny, nx, cfg.bc_x, cfg.bc_y)
            : read_s(q_old, k, j, iface,     nz, ny, nx, cfg.bc_x, cfg.bc_y);
        const double q_ho = horiz_face_val_x(q_old, k, j, iface, ux, nz, ny, nx, cfg.bc_x, cfg.bc_y, cfg.adv_order);
        const double A = ux * (fmax(0.0, q_ho) - q_up);
        const double r_donor = (A >= 0.0)
            ? read_s(r_out, k, j, iface - 1, nz, ny, nx, cfg.bc_x, cfg.bc_y)
            : read_s(r_out, k, j, iface,     nz, ny, nx, cfg.bc_x, cfg.bc_y);
        return A * r_donor;
    };
    auto limited_y = [&](int32_t jface) -> double {
        if (cfg.bc_y != 1 && (jface <= 0 || jface >= ny)) return 0.0;
        const double vy = face_flux_y(cfg, g, v, k, jface, i);
        const double q_up = (vy >= 0.0)
            ? read_s(q_old, k, jface - 1, i, nz, ny, nx, cfg.bc_x, cfg.bc_y)
            : read_s(q_old, k, jface,     i, nz, ny, nx, cfg.bc_x, cfg.bc_y);
        const double q_ho = horiz_face_val_y(q_old, k, jface, i, vy, nz, ny, nx, cfg.bc_x, cfg.bc_y, cfg.adv_order);
        const double A = vy * (fmax(0.0, q_ho) - q_up);
        const double r_donor = (A >= 0.0)
            ? read_s(r_out, k, jface - 1, i, nz, ny, nx, cfg.bc_x, cfg.bc_y)
            : read_s(r_out, k, jface,     i, nz, ny, nx, cfg.bc_x, cfg.bc_y);
        return A * r_donor;
    };
    auto limited_z = [&](int32_t kface) -> double {
        if (kface <= 0 || kface >= nz) return 0.0;
        const double wz = face_flux_z(cfg, g, u, v, w, kface, j, i);
        const double q_up = (wz >= 0.0)
            ? read_s(q_old, kface - 1, j, i, nz, ny, nx, cfg.bc_x, cfg.bc_y)
            : read_s(q_old, kface,     j, i, nz, ny, nx, cfg.bc_x, cfg.bc_y);
        const double q_ho = vert_tvd_face_val(q_old, kface, j, i, wz, g.dzeta, nz, ny, nx, cfg.bc_x, cfg.bc_y);
        const double A = wz * (fmax(0.0, q_ho) - q_up);
        const double r_donor = (A >= 0.0)
            ? static_cast<double>(r_out[static_cast<size_t>(kface - 1) * ny * nx + j * nx + i])
            : static_cast<double>(r_out[static_cast<size_t>(kface)     * ny * nx + j * nx + i]);
        return A * r_donor;
    };

    const double ax_r = limited_x(i + 1);
    const double ax_l = limited_x(i);
    const double ay_n = limited_y(j + 1);
    const double ay_s = limited_y(j);
    const double az_t = limited_z(k + 1);
    const double az_b = limited_z(k);

    const double div_anti = (ax_r - ax_l) / cfg.dx_m + (ay_n - ay_s) / cfg.dy_m + (az_t - az_b);
    const double qn = static_cast<double>(q_low[idx]) - (dt_q / vol) * div_anti;
    q_new[idx] = static_cast<float>(fmax(0.0, qn));
    (void)rc;
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
        const double z = g.zeta_s[k] + g.b_s[k] * g.hs[j * cfg.nx + i];
        double thb, pib, dthdz, rhob, cs2b;
        eval_ref_state(z, cfg.theta_sfc_k, cfg.p_sfc_pa, cfg.n_ref_s, thb, pib, dthdz, rhob, cs2b);
        const double t_tot = fmax(50.0, (thb + static_cast<double>(thp[idx])) * fmax(1e-4, pib + static_cast<double>(pip[idx])));
        const double cs = sqrt(GAMMA_GAS * R_D * t_tot);
        cs_msf = static_cast<float>(cs * g.msf[j * cfg.nx + i]);
    }
    const unsigned int lane = threadIdx.x & 31;
    for (int offset = 16; offset > 0; offset >>= 1) {
        cs_msf = fmaxf(cs_msf, __shfl_down_sync(0xffffffffu, cs_msf, offset));
    }
    if (lane == 0 && cs_msf > 0.0f) {
        atomicMax(max_cfl_u32, __float_as_uint(cs_msf));
    }
}

inline int32_t advect_one_tracer(
    NhCudaHandle* h,
    float* q_ptr,
    double dt_s
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

inline int32_t advect_one_tracer(NhCudaHandle* h, float* q_ptr, double dt_s) {
    if (!q_ptr || dt_s <= 0.0) return 0;
    const int32_t nz = h->cfg.nz, ny = h->cfg.ny, nx = h->cfg.nx;
    const DeviceStaticGrid g = static_view(h);

    CUDA_CHECK(cudaMemset(h->d_max_u32, 0, sizeof(uint32_t)));
    dim3 b3(32, 8, 1);
    dim3 g3((nx + 31) / 32, (ny + 7) / 8, nz);
    nh_tracer_cfl_kernel<<<g3, b3>>>(h->cfg, g, dt_s, h->d_u, h->d_v, h->d_w, h->d_max_u32);
    CUDA_CHECK(cudaGetLastError());

    uint32_t cfl_u32 = 0;
    CUDA_CHECK(cudaMemcpy(&cfl_u32, h->d_max_u32, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    float cfl_max = 0.0f;
    memcpy(&cfl_max, &cfl_u32, sizeof(float));

    int32_t n_sub_q = static_cast<int32_t>(ceil(static_cast<double>(cfl_max) / 0.70));
    if (n_sub_q < 1) n_sub_q = 1;
    const double dt_q = dt_s / static_cast<double>(n_sub_q);

    for (int32_t s = 0; s < n_sub_q; ++s) {
        nh_tracer_fct_pass1_kernel<<<g3, b3>>>(
            h->cfg, g, dt_q, h->d_u, h->d_v, h->d_w, q_ptr, h->d_q_tmp, h->d_div
        );
        CUDA_CHECK(cudaGetLastError());
        nh_tracer_fct_pass2_kernel<<<g3, b3>>>(
            h->cfg, g, dt_q, h->d_u, h->d_v, h->d_w, q_ptr, h->d_q_tmp, h->d_div, h->d_q_tmp
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(q_ptr, h->d_q_tmp, h->n_s * sizeof(float), cudaMemcpyDeviceToDevice));
    }
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
    if (has_lbc) float32_arrays += 4;
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
    CUDA_CHECK_PTR(cudaMalloc(&h->d_tri_c, h->n_w * sizeof(double)));
    CUDA_CHECK_PTR(cudaMalloc(&h->d_tri_d, h->n_w * sizeof(double)));
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

    return h;
}

void nh_cuda_destroy(NhCudaHandle* h) {
    if (!h) return;
    auto cf = [](void* p) { if (p) cudaFree(p); };
    cf(h->d_zeta_w); cf(h->d_zeta_s); cf(h->d_dzeta);
    cf(h->d_b_w); cf(h->d_b_s); cf(h->d_db_w); cf(h->d_db_s);
    cf(h->d_hs); cf(h->d_msf); cf(h->d_fcor);
    cf(h->d_u); cf(h->d_v); cf(h->d_w); cf(h->d_thp); cf(h->d_pip);
    cf(h->d_qv); cf(h->d_qc); cf(h->d_qr); cf(h->d_qi); cf(h->d_qs); cf(h->d_qg);
    cf(h->d_u_n); cf(h->d_v_n); cf(h->d_w_n); cf(h->d_thp_n); cf(h->d_pip_n);
    cf(h->d_ru); cf(h->d_rv); cf(h->d_rw); cf(h->d_rth); cf(h->d_rpi);
    cf(h->d_div); cf(h->d_q_tmp); cf(h->d_tri_c); cf(h->d_tri_d); cf(h->d_max_u32);
    cf(h->d_lbc_u); cf(h->d_lbc_v); cf(h->d_lbc_thp); cf(h->d_lbc_pip);
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
        CUDA_CHECK(cudaMemset(h->d_max_u32, 0, sizeof(uint32_t)));
        nh_estimate_cs_max_kernel<<<grid_s, block>>>(h->cfg, g, h->d_thp, h->d_pip, h->d_max_u32);
        CUDA_CHECK(cudaGetLastError());
        uint32_t cs_u32 = 0;
        CUDA_CHECK(cudaMemcpy(&cs_u32, h->d_max_u32, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        float cs_val = 0.0f;
        memcpy(&cs_val, &cs_u32, sizeof(float));
        if (cs_val > 10.0f) cs_msf_max = static_cast<double>(cs_val);
    }

    if (copy_state_to_n(h) != 0) return -1;

    const double rk_dt[3] = {dt_s / 3.0, dt_s / 2.0, dt_s};
    const double dmin = fmin(h->cfg.dx_m, h->cfg.dy_m);

    for (int stage = 0; stage < 3; ++stage) {
        const double tau_stage = rk_dt[stage];

        // 1. Evaluate slow tendencies from current stage state (U^n, U^*, U^{**}).
        nh_slow_scalar_kernel<<<grid_s, block>>>(
            h->cfg, g, h->d_u, h->d_v, h->d_w, h->d_thp, h->d_pip,
            h->d_lbc_thp, h->d_lbc_pip, h->d_iau_thp, h->d_iau_pip, h->d_rth, h->d_rpi
        );
        nh_slow_u_kernel<<<grid_u, block>>>(
            h->cfg, g, h->d_u, h->d_v, h->d_w, h->d_lbc_u, h->d_iau_u, h->d_ru
        );
        nh_slow_v_kernel<<<grid_v, block>>>(
            h->cfg, g, h->d_u, h->d_v, h->d_w, h->d_lbc_v, h->d_iau_v, h->d_rv
        );
        nh_slow_w_kernel<<<grid_w, block>>>(
            h->cfg, g, h->d_u, h->d_v, h->d_w, h->d_rw
        );
        CUDA_CHECK(cudaGetLastError());

        // 2. Always start acoustic integration for stage s from U^n (Wicker & Skamarock 2002).
        if (stage > 0) {
            if (reset_state_from_n(h) != 0) return -1;
        }

        int32_t ns = n_substeps;
        if (ns <= 0) {
            ns = static_cast<int32_t>(ceil((tau_stage * cs_msf_max) / (0.38 * dmin)));
            if (ns < 1) ns = 1;
        }
        const double dts = tau_stage / static_cast<double>(ns);

        for (int32_t m = 0; m < ns; ++m) {
            if (h->cfg.div_damp_coeff > 0.0) {
                nh_compute_div_kernel<<<grid_s, block>>>(h->cfg, g, h->d_u, h->d_v, h->d_w, h->d_div);
                CUDA_CHECK(cudaGetLastError());
            }
            nh_acoustic_u_kernel<<<grid_u, block>>>(
                h->cfg, g, dts, h->d_ru, h->d_thp, h->d_pip, h->d_div,
                h->d_qv, h->d_qc, h->d_qr, h->d_qi, h->d_qs, h->d_qg, h->d_u
            );
            nh_acoustic_v_kernel<<<grid_v, block>>>(
                h->cfg, g, dts, h->d_rv, h->d_thp, h->d_pip, h->d_div,
                h->d_qv, h->d_qc, h->d_qr, h->d_qi, h->d_qs, h->d_qg, h->d_v
            );
            CUDA_CHECK(cudaGetLastError());

            if (h->cfg.bc_x == 1 || h->cfg.bc_y == 1) {
                nh_sync_periodic_faces_kernel<<<grid_per, 256>>>(h->cfg, h->d_u, h->d_v);
                CUDA_CHECK(cudaGetLastError());
            }

            nh_acoustic_vert_implicit_kernel<<<grid_col, block>>>(
                h->cfg, g, dts, h->d_u, h->d_v, h->d_rw, h->d_rth, h->d_rpi,
                h->d_qv, h->d_qc, h->d_qr, h->d_qi, h->d_qs, h->d_qg,
                h->d_w, h->d_thp, h->d_pip, h->d_tri_c, h->d_tri_d
            );
            CUDA_CHECK(cudaGetLastError());
        }
    }

    // 3. Conservative positive-definite FCT advection of all 6 moist species on full dt_s.
    if (h->cfg.has_moist) {
        if (advect_one_tracer(h, h->d_qv, dt_s) != 0) return -1;
        if (advect_one_tracer(h, h->d_qc, dt_s) != 0) return -1;
        if (advect_one_tracer(h, h->d_qr, dt_s) != 0) return -1;
        if (advect_one_tracer(h, h->d_qi, dt_s) != 0) return -1;
        if (advect_one_tracer(h, h->d_qs, dt_s) != 0) return -1;
        if (advect_one_tracer(h, h->d_qg, dt_s) != 0) return -1;
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
}

} // extern "C"
