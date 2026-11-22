"""Linearized Physical Balance Control-Variable Operator K and Exact Discrete Adjoint K^T.

Transforms unbalanced control increments delta_u = (ps, theta_p, u, v, qv, ...)
into balanced physical model increments delta_x = K * delta_u:
1. Hydrostatic Exner balance: delta_ps + int_z (g / (cp * theta_bar^2)) * delta_theta_p dz -> delta_pi_p
2. Mid-latitude geostrophic wind balance (with smooth equatorial taper):
   delta_u_geo = -(cp * theta_bar * m / f_eff) * d(delta_pi_p)/dy
   delta_v_geo = +(cp * theta_bar * m / f_eff) * d(delta_pi_p)/dx

Strictly linear and 100% deterministic (0% AI / 0% ML). Satisfies exact discrete adjoint identity
<K u, v> == <u, K^T v> to floating-point machine precision.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from tempest_v2.projection import CP_D, GRAVITY_M_S2, KAPPA, DomainSpec, VerticalGrid


class BalanceOperator:
    """Exact discrete hydrostatic + geostrophic balance operator K and transpose K^T."""

    def __init__(
        self,
        spec: DomainSpec,
        hs_m: np.ndarray,
        *,
        theta_sfc_ref: float = 288.15,
        p_sfc_ref: float = 101_325.0,
        n_ref_hz: float = 0.01,
        alpha_hyd: float = 1.0,
        alpha_geo: float = 0.70,
        f0_taper_hz: float = 3.0e-5,
    ):
        self.spec = spec
        self.alpha_hyd = float(alpha_hyd)
        self.alpha_geo = float(alpha_geo)

        _, z_s, _ = spec.vertical.heights_3d(np.asarray(hs_m, dtype=np.float64))
        ref = VerticalGrid.reference_state(
            z_s, theta_sfc_k=theta_sfc_ref, p_sfc_pa=p_sfc_ref, brunt_vaisala_hz=n_ref_hz
        )
        self.theta_bar = ref["theta_bar"]  # (nz, ny, nx)
        pi0 = ref["pi_bar"][0]
        p0 = ref["p_bar"][0]

        z_agl = np.maximum(0.0, z_s - np.asarray(hs_m, dtype=np.float64)[None, :, :])
        # Surface pressure -> bottom Exner factor (c_ps retained for API compatibility)
        # with baroclinic exponential attenuation aloft to avoid stratospheric wind blow-up.
        self.c_ps = KAPPA * pi0 / np.maximum(p0, 1000.0)
        self.c_ps_3d = (
            (KAPPA * ref["pi_bar"] / np.maximum(p0, 1000.0)[None, :, :])
            * np.exp(-z_agl / 12_000.0)
        )

        # Layer thickness for hydrostatic integration: c_theta[k] = g * dz_k / (cp * theta_bar^2)
        dz_col = np.zeros_like(z_s)
        dz_col[1:] = z_s[1:] - z_s[:-1]
        self.c_theta = (GRAVITY_M_S2 * dz_col) / (CP_D * np.maximum(self.theta_bar, 150.0) ** 2)

        # Smooth equatorial taper: f_inv = f / (f^2 + f0^2), vanishes smoothly as f -> 0
        f = np.asarray(spec.coriolis_f, dtype=np.float64)
        f_inv = f / (f * f + f0_taper_hz * f0_taper_hz)

        # Taper geostrophic coupling weaker in lowest boundary layer (~0.3 -> 1.0 aloft)
        bl_taper = 0.30 + 0.70 * (1.0 - np.exp(-z_agl / 1200.0))
        self.geo_coeff = (
            self.alpha_geo
            * bl_taper
            * CP_D
            * self.theta_bar
            * np.asarray(spec.map_factor, dtype=np.float64)[None, :, :]
            * f_inv[None, :, :]
        )
        self.inv_2dx = 1.0 / (2.0 * float(spec.dx_m))
        self.inv_2dy = 1.0 / (2.0 * float(spec.dy_m))

    def forward(self, ctrl: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply physical balance transform delta_x = K * delta_u."""
        out = {k: v.astype(np.float64, copy=True) for k, v in ctrl.items()}
        ny, nx = self.spec.ny, self.spec.nx

        if "pi_p" in out:
            pi_b = out["pi_p"].copy()
        else:
            pi_b = np.zeros_like(self.theta_bar, dtype=np.float64)

        if self.alpha_hyd != 0.0:
            hyd = np.zeros_like(pi_b, dtype=np.float64)
            if "ps" in out:
                hyd += self.c_ps_3d * out["ps"][None, :, :]
            if "theta_p" in out:
                terms = np.zeros_like(pi_b, dtype=np.float64)
                terms[1:] = self.c_theta[1:] * out["theta_p"][1:]
                hyd += np.cumsum(terms, axis=0)
            pi_b = pi_b + self.alpha_hyd * hyd
            out["pi_p"] = pi_b

        if self.alpha_geo != 0.0 and ny >= 3 and nx >= 3:
            if "u" in out:
                dpi_dy = np.zeros_like(pi_b, dtype=np.float64)
                dpi_dy[:, 1:-1, 1:-1] = (pi_b[:, 2:, 1:-1] - pi_b[:, :-2, 1:-1]) * self.inv_2dy
                out["u"] = out["u"] - self.geo_coeff * dpi_dy
            if "v" in out:
                dpi_dx = np.zeros_like(pi_b, dtype=np.float64)
                dpi_dx[:, 1:-1, 1:-1] = (pi_b[:, 1:-1, 2:] - pi_b[:, 1:-1, :-2]) * self.inv_2dx
                out["v"] = out["v"] + self.geo_coeff * dpi_dx

        return out

    def adjoint(self, adj_phys: Dict[str, np.ndarray], template_keys=None) -> Dict[str, np.ndarray]:
        """Exact discrete transpose delta_u_adj = K^T * adj_phys satisfying <K u, v> == <u, K^T v>."""
        ny, nx = self.spec.ny, self.spec.nx
        keys = set(adj_phys.keys()) | (set(template_keys) if template_keys is not None else set())
        out = {
            k: adj_phys[k].astype(np.float64, copy=True) if k in adj_phys else np.zeros_like(adj_phys[next(iter(adj_phys))], dtype=np.float64)
            for k in keys
        }

        if "pi_p" in adj_phys:
            adj_pi_b = adj_phys["pi_p"].astype(np.float64, copy=True)
        else:
            adj_pi_b = np.zeros_like(self.theta_bar, dtype=np.float64)

        if self.alpha_geo != 0.0 and ny >= 3 and nx >= 3:
            if "u" in adj_phys:
                adj_dpi_dy = -self.geo_coeff * adj_phys["u"].astype(np.float64, copy=False)
                inner_y = adj_dpi_dy[:, 1:-1, 1:-1] * self.inv_2dy
                adj_pi_b[:, 2:, 1:-1] += inner_y
                adj_pi_b[:, :-2, 1:-1] -= inner_y
            if "v" in adj_phys:
                adj_dpi_dx = self.geo_coeff * adj_phys["v"].astype(np.float64, copy=False)
                inner_x = adj_dpi_dx[:, 1:-1, 1:-1] * self.inv_2dx
                adj_pi_b[:, 1:-1, 2:] += inner_x
                adj_pi_b[:, 1:-1, :-2] -= inner_x

        if self.alpha_hyd != 0.0:
            adj_hyd = self.alpha_hyd * adj_pi_b
            if "ps" in out:
                out["ps"] = out["ps"] + np.sum(self.c_ps_3d * adj_hyd, axis=0)
            if "theta_p" in out:
                cum_rev = np.cumsum(adj_hyd[::-1], axis=0)[::-1]
                out["theta_p"][1:] += self.c_theta[1:] * cum_rev[1:]

        if "pi_p" in out:
            out["pi_p"] = adj_pi_b

        return out
