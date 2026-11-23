"""Classical 3D Pressure-Level & Synoptic/Convective Forecast Diagnostics for Tempest-NWP v2.

Strictly 100% deterministic atmospheric physics and thermodynamics (0% AI / 0% ML).
All inputs and outputs strictly in standard SI units (K, Pa, m, s, kg/kg, kg/m^2, J/kg),
with RH [%] and isobaric pressure coordinate [Pa] following CF conventions.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from tempest_v2.ic_builder import unstagger_uv
from tempest_v2.projection import CP_D, GRAVITY_M_S2, KAPPA, P0_PA, R_D, DomainSpec, VerticalGrid

STANDARD_PLEV_HPA: tuple[int, ...] = (
    1000,
    925,
    850,
    700,
    600,
    500,
    400,
    300,
    250,
    200,
    150,
    100,
    50,
)
STANDARD_PLEV_PA: np.ndarray = np.array([h * 100.0 for h in STANDARD_PLEV_HPA], dtype=np.float32)

_EPS_MW = 0.62198
_LV0 = 2.501e6
_GAMMA_STD = 0.0065


def _eval_ref_3d(
    spec: DomainSpec,
    hs_m: np.ndarray,
    *,
    theta_sfc_ref: float = 288.15,
    p_sfc_ref: float = 101_325.0,
    n_ref_hz: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return `(z_s, dz_s, theta_bar, pi_bar)` with shape `(nz, ny, nx)` without allocating unused 3D reference arrays."""
    hs_2d = np.asarray(hs_m, dtype=np.float64)[None, :, :]
    z_w = spec.vertical.zeta_w[:, None, None] + spec.vertical.B_w[:, None, None] * hs_2d
    z_s = spec.vertical.zeta_s[:, None, None] + spec.vertical.B_s[:, None, None] * hs_2d
    dz_s = np.maximum(np.diff(z_w, axis=0), 1.0)
    del z_w
    n2 = float(n_ref_hz) * float(n_ref_hz)
    exp_fac = np.exp((n2 / GRAVITY_M_S2) * z_s)
    theta_bar = float(theta_sfc_ref) * exp_fac
    pi_sfc = (float(p_sfc_ref) / P0_PA) ** KAPPA
    g2_cp = (GRAVITY_M_S2 * GRAVITY_M_S2) / (CP_D * float(theta_sfc_ref) * max(n2, 1e-12))
    pi_bar = np.maximum(1e-4, pi_sfc - g2_cp * (1.0 - 1.0 / exp_fac))
    return z_s, dz_s, theta_bar, pi_bar


def _slice_spec_y(spec: DomainSpec, j0: int, j1: int) -> DomainSpec:
    from dataclasses import replace

    return replace(
        spec,
        ny=j1 - j0,
        y_m=spec.y_m[j0:j1],
        lat2d=spec.lat2d[j0:j1, :],
        lon2d=spec.lon2d[j0:j1, :],
        map_factor=spec.map_factor[j0:j1, :],
        coriolis_f=spec.coriolis_f[j0:j1, :],
    )


def _slice_state_y(state_3d: Dict[str, np.ndarray], j0: int, j1: int) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for k, arr in state_3d.items():
        if k == "v":
            out[k] = arr[:, j0 : j1 + 1, :]
        else:
            out[k] = arr[:, j0:j1, :]
    return out


def _es_water_pa(temp_k: np.ndarray) -> np.ndarray:
    tc = np.clip(temp_k - 273.15, -90.0, 60.0)
    return 611.2 * np.exp((17.67 * tc) / (tc + 243.5))


def _qsat_water(temp_k: np.ndarray, p_pa: np.ndarray) -> np.ndarray:
    es = np.minimum(_es_water_pa(temp_k), 0.90 * np.maximum(p_pa, 100.0))
    return (_EPS_MW * es) / np.maximum(p_pa - (1.0 - _EPS_MW) * es, 100.0)


def _bolton_theta_e_and_lcl(temp_k: np.ndarray, qv: np.ndarray, p_pa: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bolton (1980) equivalent potential temperature [K] and LCL temperature [K]."""
    q = np.clip(qv, 1e-7, 0.04)
    e_pa = np.maximum((q * p_pa) / (_EPS_MW + (1.0 - _EPS_MW) * q), 0.1)
    ln_e = np.log(e_pa / 611.2)
    t_d = np.minimum(temp_k, 273.15 + (243.5 * ln_e) / np.maximum(17.67 - ln_e, 1e-3))
    t_lcl = 1.0 / (1.0 / np.maximum(t_d - 56.0, 50.0) + np.log(np.maximum(temp_k / t_d, 1.0)) / 800.0) + 56.0
    p_hpa = np.maximum(p_pa * 0.01, 10.0)
    r_mix = q / np.maximum(1.0 - q, 0.5) * 1000.0  # g/kg
    theta_dl = temp_k * ((1000.0 / (p_hpa - e_pa * 0.01)) ** 0.2854) * ((temp_k / t_lcl) ** (0.28 * 1e-3 * r_mix))
    theta_e = theta_dl * np.exp((3.036 / np.maximum(t_lcl, 150.0) - 0.00178) * r_mix * (1.0 + 0.448e-3 * r_mix))
    return theta_e, t_lcl


def _moist_adiabat_t_from_theta_e(theta_e: np.ndarray, p_pa: np.ndarray) -> np.ndarray:
    """Invert Bolton theta_e along a saturated moist adiabat at pressure p_pa via 2 analytical-derivative Newton steps."""
    t_est = np.clip(theta_e * ((np.maximum(p_pa, 1000.0) / P0_PA) ** KAPPA), 150.0, 340.0)
    for _ in range(2):
        qs = _qsat_water(t_est, p_pa)
        te_cur, _ = _bolton_theta_e_and_lcl(t_est, qs, p_pa)
        t_safe = np.maximum(t_est, 150.0)
        d_te_dt = np.maximum(te_cur * (1.0 / t_safe + (_LV0 * _LV0 * qs) / (CP_D * 461.5 * (t_safe**3))), 0.8)
        t_est = np.clip(t_est + (theta_e - te_cur) / d_te_dt, 140.0, 340.0)
    return t_est


def compute_pressure_level_fields(
    spec: DomainSpec,
    hs_m: np.ndarray,
    state_3d: Dict[str, np.ndarray],
    *,
    plev_pa: Sequence[float] | np.ndarray = STANDARD_PLEV_PA,
    theta_sfc_ref: float = 288.15,
    p_sfc_ref: float = 101_325.0,
    n_ref_hz: float = 0.01,
) -> Dict[str, np.ndarray]:
    """Interpolate 3D prognostic state to standard isobaric pressure levels (`t_pl, gh_pl, q_pl, rh_pl, u_pl, v_pl, w_pl`)."""
    p_targets = np.asarray(plev_pa, dtype=np.float64)
    if spec.nz * spec.ny * spec.nx > 12_000_000 and spec.ny > 96:
        chunk_ny = 96
        np_out = len(p_targets)
        out_full: Optional[Dict[str, np.ndarray]] = None
        for j0 in range(0, spec.ny, chunk_ny):
            j1 = min(spec.ny, j0 + chunk_ny)
            sub = compute_pressure_level_fields(
                _slice_spec_y(spec, j0, j1),
                hs_m[j0:j1, :],
                _slice_state_y(state_3d, j0, j1),
                plev_pa=p_targets,
                theta_sfc_ref=theta_sfc_ref,
                p_sfc_ref=p_sfc_ref,
                n_ref_hz=n_ref_hz,
            )
            if out_full is None:
                out_full = {k: np.empty((np_out, spec.ny, spec.nx), dtype=np.float32) for k in sub}
            for k, arr in sub.items():
                out_full[k][:, j0:j1, :] = arr
        assert out_full is not None
        if spec.nx >= 2 and spec.ny >= 2:
            dx_eff = np.maximum(float(spec.dx_m) / np.maximum(spec.map_factor, 0.05), 1.0).astype(np.float32)
            dy_eff = float(spec.dy_m)
            for ip in range(np_out):
                u_i = out_full["u_pl"][ip]
                v_i = out_full["v_pl"][ip]
                du_dx = np.gradient(u_i, axis=1) / dx_eff
                dv_dx = np.gradient(v_i, axis=1) / dx_eff
                du_dy = np.gradient(u_i, dy_eff, axis=0)
                dv_dy = np.gradient(v_i, dy_eff, axis=0)
                out_full["vo_pl"][ip] = (dv_dx - du_dy).astype(np.float32)
                out_full["div_pl"][ip] = (du_dx + dv_dy).astype(np.float32)
        return out_full
    z_s, _dz_s, theta_bar, pi_bar = _eval_ref_3d(
        spec, hs_m, theta_sfc_ref=theta_sfc_ref, p_sfc_ref=p_sfc_ref, n_ref_hz=n_ref_hz
    )
    pi_tot = np.maximum(pi_bar + np.asarray(state_3d["pi_p"], dtype=np.float64), 1e-4)
    theta_tot = np.maximum(theta_bar + np.asarray(state_3d["theta_p"], dtype=np.float64), 50.0)
    t_3d = theta_tot * pi_tot
    p_3d = P0_PA * (pi_tot ** (1.0 / KAPPA))
    ln_p = np.log(np.maximum(p_3d, 1.0))

    qv = np.clip(np.asarray(state_3d.get("qv", np.zeros_like(t_3d)), dtype=np.float64), 1e-7, 0.05)
    e_v = (qv * p_3d) / (_EPS_MW + (1.0 - _EPS_MW) * qv)
    rh_3d = np.clip(100.0 * e_v / np.maximum(_es_water_pa(t_3d), 1.0), 0.0, 100.0)

    u_c, v_c = unstagger_uv(state_3d["u"], state_3d["v"])
    u_3d = u_c.astype(np.float64)
    v_3d = v_c.astype(np.float64)
    if spec.is_projected and spec.projection is not None:
        u_geo = np.empty_like(u_3d)
        v_geo = np.empty_like(v_3d)
        for k in range(spec.nz):
            u_geo[k], v_geo[k] = spec.projection.grid_to_geo_wind(u_3d[k], v_3d[k], spec.lon2d)
        u_3d, v_3d = u_geo, v_geo

    w_raw = np.asarray(state_3d["w"], dtype=np.float64)
    w_3d = 0.5 * (w_raw[:-1] + w_raw[1:]) if w_raw.shape[0] == spec.nz + 1 else w_raw

    nz, ny, nx = spec.nz, spec.ny, spec.nx
    qc_3d = np.maximum(np.asarray(state_3d.get("qc", np.zeros((nz, ny, nx))), dtype=np.float64), 0.0)
    qr_3d = np.maximum(np.asarray(state_3d.get("qr", np.zeros((nz, ny, nx))), dtype=np.float64), 0.0)
    qi_3d = np.maximum(np.asarray(state_3d.get("qi", np.zeros((nz, ny, nx))), dtype=np.float64), 0.0)
    qs_3d = np.maximum(np.asarray(state_3d.get("qs", np.zeros((nz, ny, nx))), dtype=np.float64), 0.0)
    qg_3d = np.maximum(np.asarray(state_3d.get("qg", np.zeros((nz, ny, nx))), dtype=np.float64), 0.0)

    np_out = len(p_targets)
    shape_out = (np_out, ny, nx)

    t_pl = np.empty(shape_out, dtype=np.float32)
    gh_pl = np.empty(shape_out, dtype=np.float32)
    z_pl = np.empty(shape_out, dtype=np.float32)
    theta_pl = np.empty(shape_out, dtype=np.float32)
    q_pl = np.empty(shape_out, dtype=np.float32)
    rh_pl = np.empty(shape_out, dtype=np.float32)
    u_pl = np.empty(shape_out, dtype=np.float32)
    v_pl = np.empty(shape_out, dtype=np.float32)
    w_pl = np.empty(shape_out, dtype=np.float32)
    omega_pl = np.empty(shape_out, dtype=np.float32)
    vo_pl = np.empty(shape_out, dtype=np.float32)
    div_pl = np.empty(shape_out, dtype=np.float32)
    qc_pl = np.empty(shape_out, dtype=np.float32)
    qr_pl = np.empty(shape_out, dtype=np.float32)
    qi_pl = np.empty(shape_out, dtype=np.float32)
    qs_pl = np.empty(shape_out, dtype=np.float32)
    qg_pl = np.empty(shape_out, dtype=np.float32)

    dx_eff = np.maximum(float(spec.dx_m) / np.maximum(spec.map_factor, 0.05), 1.0)
    dy_eff = float(spec.dy_m)

    jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    for ip, pt in enumerate(p_targets):
        pt_f = max(float(pt), 1.0)
        lpt = np.log(pt_f)
        k_hi = np.clip(np.sum(ln_p > lpt, axis=0), 1, nz - 1)
        k_lo = k_hi - 1

        lp_lo = ln_p[k_lo, jj, ii]
        lp_hi = ln_p[k_hi, jj, ii]
        wt = np.clip((lpt - lp_lo) / np.where(np.abs(lp_hi - lp_lo) > 1e-8, lp_hi - lp_lo, -1e-8), 0.0, 1.0)

        t_i = (1.0 - wt) * t_3d[k_lo, jj, ii] + wt * t_3d[k_hi, jj, ii]
        z_i = (1.0 - wt) * z_s[k_lo, jj, ii] + wt * z_s[k_hi, jj, ii]
        q_i = (1.0 - wt) * qv[k_lo, jj, ii] + wt * qv[k_hi, jj, ii]
        r_i = (1.0 - wt) * rh_3d[k_lo, jj, ii] + wt * rh_3d[k_hi, jj, ii]
        u_i = (1.0 - wt) * u_3d[k_lo, jj, ii] + wt * u_3d[k_hi, jj, ii]
        v_i = (1.0 - wt) * v_3d[k_lo, jj, ii] + wt * v_3d[k_hi, jj, ii]
        w_i = (1.0 - wt) * w_3d[k_lo, jj, ii] + wt * w_3d[k_hi, jj, ii]
        qc_i = (1.0 - wt) * qc_3d[k_lo, jj, ii] + wt * qc_3d[k_hi, jj, ii]
        qr_i = (1.0 - wt) * qr_3d[k_lo, jj, ii] + wt * qr_3d[k_hi, jj, ii]
        qi_i = (1.0 - wt) * qi_3d[k_lo, jj, ii] + wt * qi_3d[k_hi, jj, ii]
        qs_i = (1.0 - wt) * qs_3d[k_lo, jj, ii] + wt * qs_3d[k_hi, jj, ii]
        qg_i = (1.0 - wt) * qg_3d[k_lo, jj, ii] + wt * qg_3d[k_hi, jj, ii]

        below = pt > p_3d[0]
        if np.any(below):
            t0 = np.maximum(t_3d[0], 200.0)
            t_ext = t0 * ((pt_f / np.maximum(p_3d[0], 100.0)) ** (R_D * _GAMMA_STD / GRAVITY_M_S2))
            z_ext = z_s[0] - (t_ext - t0) / _GAMMA_STD
            t_i = np.where(below, t_ext, t_i)
            z_i = np.where(below, z_ext, z_i)
            q_i = np.where(below, qv[0], q_i)
            r_i = np.where(below, rh_3d[0], r_i)
            u_i = np.where(below, u_3d[0], u_i)
            v_i = np.where(below, v_3d[0], v_i)
            w_i = np.where(below, w_3d[0], w_i)
            qc_i = np.where(below, qc_3d[0], qc_i)
            qr_i = np.where(below, qr_3d[0], qr_i)
            qi_i = np.where(below, qi_3d[0], qi_i)
            qs_i = np.where(below, qs_3d[0], qs_i)
            qg_i = np.where(below, qg_3d[0], qg_i)

        above = pt < p_3d[-1]
        if np.any(above):
            t_top = np.maximum(t_3d[-1], 160.0)
            z_ext = z_s[-1] + (R_D * t_top / GRAVITY_M_S2) * np.log(np.maximum(p_3d[-1], 10.0) / pt_f)
            t_i = np.where(above, t_top, t_i)
            z_i = np.where(above, z_ext, z_i)
            q_i = np.where(above, qv[-1], q_i)
            r_i = np.where(above, rh_3d[-1], r_i)
            u_i = np.where(above, u_3d[-1], u_i)
            v_i = np.where(above, v_3d[-1], v_i)
            w_i = np.where(above, w_3d[-1], w_i)
            qc_i = np.where(above, qc_3d[-1], qc_i)
            qr_i = np.where(above, qr_3d[-1], qr_i)
            qi_i = np.where(above, qi_3d[-1], qi_i)
            qs_i = np.where(above, qs_3d[-1], qs_i)
            qg_i = np.where(above, qg_3d[-1], qg_i)

        q_clamped = np.clip(q_i, 0.0, 0.05)
        tv_i = np.maximum(t_i * (1.0 + 0.608 * q_clamped), 150.0)
        rho_i = pt_f / (R_D * tv_i)
        omega_i = -rho_i * GRAVITY_M_S2 * w_i
        theta_i = t_i * ((P0_PA / pt_f) ** KAPPA)

        if nx >= 2 and ny >= 2:
            du_dx = np.gradient(u_i, axis=1) / dx_eff
            dv_dx = np.gradient(v_i, axis=1) / dx_eff
            du_dy = np.gradient(u_i, dy_eff, axis=0)
            dv_dy = np.gradient(v_i, dy_eff, axis=0)
            vo_i = dv_dx - du_dy
            div_i = du_dx + dv_dy
        else:
            vo_i = np.zeros_like(u_i)
            div_i = np.zeros_like(u_i)

        t_pl[ip] = t_i.astype(np.float32)
        gh_pl[ip] = z_i.astype(np.float32)
        z_pl[ip] = (GRAVITY_M_S2 * z_i).astype(np.float32)
        theta_pl[ip] = theta_i.astype(np.float32)
        q_pl[ip] = q_clamped.astype(np.float32)
        rh_pl[ip] = np.clip(r_i, 0.0, 100.0).astype(np.float32)
        u_pl[ip] = u_i.astype(np.float32)
        v_pl[ip] = v_i.astype(np.float32)
        w_pl[ip] = w_i.astype(np.float32)
        omega_pl[ip] = omega_i.astype(np.float32)
        vo_pl[ip] = vo_i.astype(np.float32)
        div_pl[ip] = div_i.astype(np.float32)
        qc_pl[ip] = np.maximum(qc_i, 0.0).astype(np.float32)
        qr_pl[ip] = np.maximum(qr_i, 0.0).astype(np.float32)
        qi_pl[ip] = np.maximum(qi_i, 0.0).astype(np.float32)
        qs_pl[ip] = np.maximum(qs_i, 0.0).astype(np.float32)
        qg_pl[ip] = np.maximum(qg_i, 0.0).astype(np.float32)

    return {
        "t_pl": t_pl,
        "gh_pl": gh_pl,
        "z_pl": z_pl,
        "theta_pl": theta_pl,
        "q_pl": q_pl,
        "rh_pl": rh_pl,
        "u_pl": u_pl,
        "v_pl": v_pl,
        "w_pl": w_pl,
        "omega_pl": omega_pl,
        "vo_pl": vo_pl,
        "div_pl": div_pl,
        "qc_pl": qc_pl,
        "qr_pl": qr_pl,
        "qi_pl": qi_pl,
        "qs_pl": qs_pl,
        "qg_pl": qg_pl,
    }


def _parcel_cape_cin(
    t_env: np.ndarray,
    qv_env: np.ndarray,
    p_env: np.ndarray,
    z_s: np.ndarray,
    dz_s: np.ndarray,
    k_start: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute pseudoadiabatic parcel CAPE & CIN [J kg^-1] starting from level `k_start[j, i]`."""
    nz, ny, nx = t_env.shape
    jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    t_p0 = t_env[k_start, jj, ii]
    q_p0 = qv_env[k_start, jj, ii]
    p_p0 = p_env[k_start, jj, ii]
    theta_e0, t_lcl0 = _bolton_theta_e_and_lcl(t_p0, q_p0, p_p0)
    theta_dry0 = t_p0 * ((P0_PA / np.maximum(p_p0, 100.0)) ** KAPPA)
    p_lcl0 = P0_PA * ((t_lcl0 / np.maximum(theta_dry0, 50.0)) ** (1.0 / KAPPA))

    tv_env = t_env * (1.0 + 0.608 * qv_env)
    cape = np.zeros((ny, nx), dtype=np.float64)
    cin = np.zeros((ny, nx), dtype=np.float64)
    seen_pos = np.zeros((ny, nx), dtype=bool)
    neg_accum = np.zeros((ny, nx), dtype=np.float64)

    k = 0
    while k < nz:
        if float(np.min(z_s[k])) > 16_000.0:
            break
        step_k = 2 if (nz >= 60 and k >= 16 and k + 1 < nz) else 1
        dz_k = (dz_s[k] + dz_s[k + 1]) if step_k == 2 else dz_s[k]
        active = k > k_start
        if np.any(active):
            p_k = p_env[k]
            below_lcl = p_k >= p_lcl0
            t_dry = theta_dry0 * ((p_k / P0_PA) ** KAPPA)
            if np.all(below_lcl):
                t_par = t_dry
                q_par = q_p0
            else:
                t_moist = _moist_adiabat_t_from_theta_e(theta_e0, p_k)
                t_par = np.where(below_lcl, t_dry, t_moist)
                q_par = np.where(below_lcl, q_p0, _qsat_water(t_moist, p_k))
            tv_par = t_par * (1.0 + 0.608 * q_par)
            b_k = GRAVITY_M_S2 * (tv_par - tv_env[k]) / np.maximum(tv_env[k], 150.0) * dz_k

            pos = active & (b_k > 0.0)
            cape = np.where(pos, cape + b_k, cape)
            neg_pre = active & (b_k < 0.0) & (~seen_pos)
            neg_accum = np.where(neg_pre, neg_accum - b_k, neg_accum)
            seen_pos = seen_pos | pos
        k += step_k

    cin = np.where(cape > 1.0, neg_accum, 0.0)
    return cape.astype(np.float32), cin.astype(np.float32)


def compute_synoptic_convective_2d(
    spec: DomainSpec,
    hs_m: np.ndarray,
    state_3d: Dict[str, np.ndarray],
    surf_state: Dict[str, np.ndarray],
    *,
    theta_sfc_ref: float = 288.15,
    p_sfc_ref: float = 101_325.0,
    n_ref_hz: float = 0.01,
    prev_precip_total: Optional[np.ndarray] = None,
    interval_s: float = 0.0,
) -> Dict[str, np.ndarray]:
    """Compute classical 2D synoptic, thermodynamic, cloud, severe-convective & visibility diagnostics."""
    if spec.nz * spec.ny * spec.nx > 12_000_000 and spec.ny > 96:
        chunk_ny = 96
        out_full: Optional[Dict[str, np.ndarray]] = None
        for j0 in range(0, spec.ny, chunk_ny):
            j1 = min(spec.ny, j0 + chunk_ny)
            sub_surf = {
                k: (v[:, j0:j1, :] if np.asarray(v).ndim == 3 else np.asarray(v)[j0:j1, :])
                for k, v in surf_state.items()
            }
            sub_prev = prev_precip_total[j0:j1, :] if prev_precip_total is not None else None
            sub = compute_synoptic_convective_2d(
                _slice_spec_y(spec, j0, j1),
                hs_m[j0:j1, :],
                _slice_state_y(state_3d, j0, j1),
                sub_surf,
                theta_sfc_ref=theta_sfc_ref,
                p_sfc_ref=p_sfc_ref,
                n_ref_hz=n_ref_hz,
                prev_precip_total=sub_prev,
                interval_s=interval_s,
            )
            if out_full is None:
                out_full = {k: np.empty((spec.ny, spec.nx), dtype=np.float32) for k in sub}
            for k, arr in sub.items():
                out_full[k][j0:j1, :] = arr
        assert out_full is not None
        return out_full

    z_s, dz_s, theta_bar, pi_bar = _eval_ref_3d(
        spec, hs_m, theta_sfc_ref=theta_sfc_ref, p_sfc_ref=p_sfc_ref, n_ref_hz=n_ref_hz
    )
    pi_tot = np.maximum(pi_bar + np.asarray(state_3d["pi_p"], dtype=np.float64), 1e-4)
    theta_tot = np.maximum(theta_bar + np.asarray(state_3d["theta_p"], dtype=np.float64), 50.0)
    t_3d = theta_tot * pi_tot
    p_3d = P0_PA * (pi_tot ** (1.0 / KAPPA))

    qv = np.clip(np.asarray(state_3d.get("qv", np.zeros_like(t_3d)), dtype=np.float64), 1e-7, 0.05)
    qc = np.maximum(np.asarray(state_3d.get("qc", np.zeros_like(t_3d)), dtype=np.float64), 0.0)
    qr = np.maximum(np.asarray(state_3d.get("qr", np.zeros_like(t_3d)), dtype=np.float64), 0.0)
    qi = np.maximum(np.asarray(state_3d.get("qi", np.zeros_like(t_3d)), dtype=np.float64), 0.0)
    qs = np.maximum(np.asarray(state_3d.get("qs", np.zeros_like(t_3d)), dtype=np.float64), 0.0)
    qg = np.maximum(np.asarray(state_3d.get("qg", np.zeros_like(t_3d)), dtype=np.float64), 0.0)
    qhyd = qc + qr + qi + qs + qg
    rho = p_3d / (R_D * t_3d * np.maximum(1.0 + 0.608 * qv - qhyd, 0.5))

    # Surface 2m dewpoint & RH
    t2m = np.clip(np.asarray(surf_state.get("t2m", t_3d[0]), dtype=np.float64), 160.0, 340.0)
    q2m = np.clip(np.asarray(surf_state.get("q2m", qv[0]), dtype=np.float64), 1e-7, 0.05)
    p_sfc = p_3d[0] * np.exp(GRAVITY_M_S2 * np.maximum(z_s[0] - hs_m, 1.0) / (R_D * np.maximum(t_3d[0], 180.0)))
    e_2m = (q2m * p_sfc) / (_EPS_MW + (1.0 - _EPS_MW) * q2m)
    es_2m = _es_water_pa(t2m)
    rh2m = np.clip(100.0 * e_2m / np.maximum(es_2m, 1.0), 0.0, 100.0)
    ln_e2 = np.log(np.clip(e_2m, 0.5, es_2m) / 611.2)
    td2m = np.minimum(t2m, 273.15 + (243.5 * ln_e2) / np.maximum(17.67 - ln_e2, 1e-3))

    # Column moisture & cloud condensate paths
    tcwv = np.sum(rho * qv * dz_s, axis=0)
    lwp = np.sum(rho * (qc + qr) * dz_s, axis=0)
    iwp = np.sum(rho * (qi + qs) * dz_s, axis=0)

    # Tiered & total cloud cover from layer cloud fraction `1 - exp(-beta * q_cloud * rho * dz)`
    cld_3d = 1.0 - np.exp(-32.0 * rho * (qc + qi + 0.3 * qs) * dz_s)
    cld_3d = np.clip(cld_3d, 0.0, 1.0)
    low_m = p_3d >= 80_000.0
    mid_m = (p_3d < 80_000.0) & (p_3d >= 45_000.0)
    high_m = p_3d < 45_000.0
    lcc = 1.0 - np.prod(1.0 - np.where(low_m, cld_3d, 0.0), axis=0)
    mcc = 1.0 - np.prod(1.0 - np.where(mid_m, cld_3d, 0.0), axis=0)
    hcc = 1.0 - np.prod(1.0 - np.where(high_m, cld_3d, 0.0), axis=0)
    tcc = 1.0 - (1.0 - lcc) * (1.0 - mcc) * (1.0 - hcc)

    # Surface-based & Most-Unstable CAPE / CIN
    k0 = np.zeros((spec.ny, spec.nx), dtype=np.int64)
    cape_sb, cin_sb = _parcel_cape_cin(t_3d, qv, p_3d, z_s, dz_s, k0)
    theta_e_3d, _ = _bolton_theta_e_and_lcl(t_3d, qv, p_3d)
    low_300 = p_3d >= (p_sfc[None, :, :] - 30_000.0)
    k_mu = np.argmax(np.where(low_300, theta_e_3d, -1e9), axis=0)
    cape_mu, _cin_mu = _parcel_cape_cin(t_3d, qv, p_3d, z_s, dz_s, k_mu)

    # Winds, 0-6 km shear, 0-3 km Storm-Relative Helicity & 10m gust
    u_c, v_c = unstagger_uv(state_3d["u"], state_3d["v"])
    u_3d = u_c.astype(np.float64)
    v_3d = v_c.astype(np.float64)
    if spec.is_projected and spec.projection is not None:
        for k in range(spec.nz):
            u_3d[k], v_3d[k] = spec.projection.grid_to_geo_wind(u_3d[k], v_3d[k], spec.lon2d)

    agl = z_s - hs_m[None, :, :]
    jj, ii = np.meshgrid(np.arange(spec.ny), np.arange(spec.nx), indexing="ij")

    def _interp_agl(target_m: float, arr: np.ndarray) -> np.ndarray:
        k_hi = np.clip(np.sum(agl < target_m, axis=0), 1, spec.nz - 1)
        k_lo = k_hi - 1
        a_lo = agl[k_lo, jj, ii]
        a_hi = agl[k_hi, jj, ii]
        w = np.clip((target_m - a_lo) / np.where(np.abs(a_hi - a_lo) > 1.0, a_hi - a_lo, 1.0), 0.0, 1.0)
        return (1.0 - w) * arr[k_lo, jj, ii] + w * arr[k_hi, jj, ii]

    u_6km = _interp_agl(6000.0, u_3d)
    v_6km = _interp_agl(6000.0, v_3d)
    u_0km = u_3d[0]
    v_0km = v_3d[0]
    du_6 = u_6km - u_0km
    dv_6 = v_6km - v_0km
    shear_0_6km = np.hypot(du_6, dv_6)

    # Bunkers storm motion & 0-3 km SRH
    u_mean_6 = np.sum(np.where(agl <= 6000.0, u_3d * dz_s, 0.0), axis=0) / np.maximum(
        np.sum(np.where(agl <= 6000.0, dz_s, 0.0), axis=0), 1.0
    )
    v_mean_6 = np.sum(np.where(agl <= 6000.0, v_3d * dz_s, 0.0), axis=0) / np.maximum(
        np.sum(np.where(agl <= 6000.0, dz_s, 0.0), axis=0), 1.0
    )
    sh_mag = np.maximum(shear_0_6km, 1e-6)
    cx = u_mean_6 + 7.5 * (dv_6 / sh_mag)
    cy = v_mean_6 - 7.5 * (du_6 / sh_mag)
    srh_0_3km = np.zeros((spec.ny, spec.nx), dtype=np.float64)
    for k in range(spec.nz - 1):
        in3 = agl[k + 1] <= 3500.0
        ux_a = u_3d[k] - cx
        vy_a = v_3d[k] - cy
        ux_b = u_3d[k + 1] - cx
        vy_b = v_3d[k + 1] - cy
        contrib = ux_b * vy_a - ux_a * vy_b
        srh_0_3km = np.where(in3, srh_0_3km + contrib, srh_0_3km)

    u10 = np.asarray(surf_state.get("u10m", u_3d[0]), dtype=np.float64)
    v10 = np.asarray(surf_state.get("v10m", v_3d[0]), dtype=np.float64)
    wspd10 = np.hypot(u10, v10)
    ustar = np.maximum(np.asarray(surf_state.get("ustar", 0.04 * wspd10), dtype=np.float64), 0.0)
    # Beljaars (1987) / Brasseur (2001) / ECMWF IFS 3s peak gust (~3*sigma_u = 7.2*ustar) + convective downdraft term
    gust10m = wspd10 + 7.2 * ustar + 0.15 * np.sqrt(np.maximum(cape_sb, 0.0))

    # Highest freezing level [m ASL]
    freezing_level_m = np.zeros((spec.ny, spec.nx), dtype=np.float64)
    for k in range(spec.nz - 1):
        cross = (t_3d[k] >= 273.15) & (t_3d[k + 1] < 273.15)
        frac = (273.15 - t_3d[k]) / np.where(np.abs(t_3d[k + 1] - t_3d[k]) > 1e-5, t_3d[k + 1] - t_3d[k], -1.0)
        z_0c = z_s[k] + np.clip(frac, 0.0, 1.0) * (z_s[k + 1] - z_s[k])
        freezing_level_m = np.where(cross, z_0c, freezing_level_m)

    # WMO/ICAO 5% contrast (-ln 0.05 = 2.996) + Kunkel (1984) / Stoelinga-Warner (1999) + Clark (2008) Kohler haze
    lwc_g_m3 = 1000.0 * rho[0] * np.maximum(qc[0], 0.0)
    rwc_g_m3 = 1000.0 * rho[0] * np.maximum(qr[0], 0.0)
    iwc_g_m3 = 1000.0 * rho[0] * np.maximum(qi[0], 0.0)
    swc_g_m3 = 1000.0 * rho[0] * np.maximum(qs[0], 0.0)
    gwc_g_m3 = 1000.0 * rho[0] * np.maximum(qg[0], 0.0)
    beta_c = 144.7 * np.power(lwc_g_m3, 0.88)
    beta_r = 2.24 * np.power(rwc_g_m3, 0.75)
    beta_i = 163.9 * np.power(iwc_g_m3, 1.0)
    beta_s = 10.4 * np.power(swc_g_m3, 0.78)
    beta_g = 1.12 * np.power(gwc_g_m3, 0.75)
    # Clark et al. (2008) / IFS Kohler hygroscopic aerosol swelling (clean background beta_dry=0.020 km^-1, exp=-2/3)
    rh_frac = np.clip(rh2m / 100.0, 0.0, 0.999)
    beta_haze = 0.020 * np.power(1.0 - np.minimum(rh_frac, 0.992), -2.0 / 3.0)
    # Shallow 2m nocturnal radiation ground-fog (sub-grid condensate below k=0 when calm, inverted, and >= 98.8% RH)
    inv_2m = np.clip(t_3d[0] - t2m, 0.0, 5.0)
    calm_fac = np.clip((2.5 - wspd10) / 1.5, 0.0, 1.0)
    s_excess = np.clip((rh_frac - 0.988) / 0.012, 0.0, 1.0)
    lwc_2m_g_m3 = 0.045 * np.power(s_excess, 1.5) * calm_fac * np.clip(inv_2m / 0.5, 0.0, 1.0)
    beta_fog2m = 144.7 * np.power(lwc_2m_g_m3, 0.88)
    beta_tot = np.maximum(beta_c + beta_fog2m + beta_r + beta_i + beta_s + beta_g + beta_haze + 0.015, 0.05992)
    vis_m = np.clip(2996.0 / beta_tot, 10.0, 50_000.0)

    # Surface precipitation rate [kg m^-2 s^-1]
    p_tot = (
        np.asarray(surf_state.get("rain_acc", np.zeros((spec.ny, spec.nx))), dtype=np.float64)
        + np.asarray(surf_state.get("snow_acc", np.zeros((spec.ny, spec.nx))), dtype=np.float64)
        + np.asarray(surf_state.get("graupel_acc", np.zeros((spec.ny, spec.nx))), dtype=np.float64)
    )
    if prev_precip_total is not None and interval_s > 0.0:
        precip_rate = np.maximum(p_tot - np.asarray(prev_precip_total, dtype=np.float64), 0.0) / float(interval_s)
    else:
        precip_rate = np.zeros((spec.ny, spec.nx), dtype=np.float64)

    return {
        "td2m": td2m.astype(np.float32),
        "rh2m": rh2m.astype(np.float32),
        "gust10m": gust10m.astype(np.float32),
        "tcc": np.clip(tcc, 0.0, 1.0).astype(np.float32),
        "lcc": np.clip(lcc, 0.0, 1.0).astype(np.float32),
        "mcc": np.clip(mcc, 0.0, 1.0).astype(np.float32),
        "hcc": np.clip(hcc, 0.0, 1.0).astype(np.float32),
        "tcwv": tcwv.astype(np.float32),
        "lwp": lwp.astype(np.float32),
        "iwp": iwp.astype(np.float32),
        "cape_sb": np.asarray(cape_sb, dtype=np.float32),
        "cin_sb": np.asarray(cin_sb, dtype=np.float32),
        "cape_mu": np.asarray(cape_mu, dtype=np.float32),
        "shear_0_6km": shear_0_6km.astype(np.float32),
        "srh_0_3km": srh_0_3km.astype(np.float32),
        "freezing_level_m": freezing_level_m.astype(np.float32),
        "vis_m": vis_m.astype(np.float32),
        "precip_rate": precip_rate.astype(np.float32),
    }
