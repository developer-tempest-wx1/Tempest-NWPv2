"""Deterministic Map Projections, Domain Definitions & Vertical Terrain-Following Coordinates.

Strictly 100% Classical Mathematical Geodesy (0% AI / 0% ML), SI Units Only:
- Spherical Lambert Conformal Conic (LCC) projection with exact forward/inverse transforms,
  map scale factor m(x, y), Coriolis f(x, y), and wind vector rotation angle gamma(lon).
- Smooth Klemp (2011) terrain-following height coordinate supporting nz = 65, 90, 137 levels
  with exact analytic hydrostatic reference atmosphere in IEEE 754 float64.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional, Tuple

import numpy as np

EARTH_RADIUS_M = 6_371_229.0
OMEGA_RAD_S = 7.292115e-5
GRAVITY_M_S2 = 9.80665
R_D = 287.058
CP_D = 1004.5
KAPPA = R_D / CP_D
P0_PA = 100_000.0


@dataclass(frozen=True)
class LCCProjection:
    """Spherical Lambert Conformal Conic projection for European/Benelux-DE LAM domains."""

    lat1_deg: float = 48.0
    lat2_deg: float = 54.0
    lat0_deg: float = 51.0
    lon0_deg: float = 8.5
    earth_radius_m: float = EARTH_RADIUS_M

    def _params(self) -> Tuple[float, float, float, float]:
        phi1 = math.radians(self.lat1_deg)
        phi2 = math.radians(self.lat2_deg)
        phi0 = math.radians(self.lat0_deg)
        if abs(phi1 - phi2) < 1e-8:
            cone_n = math.sin(phi1)
        else:
            t1 = math.tan(0.25 * math.pi + 0.5 * phi1)
            t2 = math.tan(0.25 * math.pi + 0.5 * phi2)
            cone_n = math.log(math.cos(phi1) / math.cos(phi2)) / math.log(t2 / t1)
        f_const = (math.cos(phi1) * (math.tan(0.25 * math.pi + 0.5 * phi1) ** cone_n)) / cone_n
        rho0 = self.earth_radius_m * f_const / (math.tan(0.25 * math.pi + 0.5 * phi0) ** cone_n)
        return cone_n, f_const, rho0, math.radians(self.lon0_deg)

    def latlon_to_xy(
        self, lat_deg: np.ndarray | float, lon_deg: np.ndarray | float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert geographic (lat, lon) [degrees] to projected (x, y) [m] (East, North)."""
        cone_n, f_const, rho0, _ = self._params()
        lat_arr = np.asarray(lat_deg, dtype=np.float64)
        lon_arr = np.asarray(lon_deg, dtype=np.float64)
        phi = np.radians(np.clip(lat_arr, -89.999, 89.999))
        dlon_deg = ((lon_arr - self.lon0_deg + 180.0) % 360.0) - 180.0
        theta = cone_n * np.radians(dlon_deg)
        rho = self.earth_radius_m * f_const / (np.tan(0.25 * math.pi + 0.5 * phi) ** cone_n)
        x_m = rho * np.sin(theta)
        y_m = rho0 - rho * np.cos(theta)
        return x_m, y_m

    def xy_to_latlon(
        self, x_m: np.ndarray | float, y_m: np.ndarray | float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert projected (x, y) [m] to geographic (lat, lon) [degrees]."""
        cone_n, f_const, rho0, _ = self._params()
        x_arr = np.asarray(x_m, dtype=np.float64)
        y_arr = np.asarray(y_m, dtype=np.float64)
        sign_n = 1.0 if cone_n >= 0.0 else -1.0
        dx = x_arr
        dy = rho0 - y_arr
        rho = sign_n * np.hypot(dx, dy)
        theta = np.arctan2(sign_n * dx, sign_n * dy)
        safe_rho = np.where(np.abs(rho) > 1e-12, rho, 1e-12)
        ratio = (self.earth_radius_m * f_const) / safe_rho
        phi = 2.0 * np.arctan(np.power(np.maximum(ratio, 1e-15), 1.0 / cone_n)) - 0.5 * math.pi
        lat_deg = np.degrees(phi)
        lon_deg = self.lon0_deg + np.degrees(theta / cone_n)
        lon_deg = ((lon_deg + 180.0) % 360.0) - 180.0
        return lat_deg, lon_deg

    def map_scale_factor(self, lat_deg: np.ndarray | float) -> np.ndarray:
        """Compute conformal map scale factor m = ds_grid / ds_earth (>= 1 between parallels ~0.999..1.004)."""
        cone_n, f_const, _, _ = self._params()
        phi = np.radians(np.clip(np.asarray(lat_deg, dtype=np.float64), -89.999, 89.999))
        rho = self.earth_radius_m * f_const / (np.tan(0.25 * math.pi + 0.5 * phi) ** cone_n)
        return np.abs(cone_n * rho) / (self.earth_radius_m * np.cos(phi))

    def coriolis_f(self, lat_deg: np.ndarray | float) -> np.ndarray:
        """Compute Coriolis parameter f = 2 Omega sin(phi) [s^-1]."""
        return 2.0 * OMEGA_RAD_S * np.sin(np.radians(np.asarray(lat_deg, dtype=np.float64)))

    def wind_rotation_angle_rad(self, lon_deg: np.ndarray | float) -> np.ndarray:
        """Meridian convergence angle gamma = n * (lon - lon0) [rad]."""
        cone_n, _, _, _ = self._params()
        dlon_deg = ((np.asarray(lon_deg, dtype=np.float64) - self.lon0_deg + 180.0) % 360.0) - 180.0
        return cone_n * np.radians(dlon_deg)

    def geo_to_grid_wind(
        self, u_geo: np.ndarray, v_geo: np.ndarray, lon_deg: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Rotate true Earth (east, north) wind components onto LCC (x, y) grid axes."""
        gamma = self.wind_rotation_angle_rad(lon_deg)
        c = np.cos(gamma)
        s = np.sin(gamma)
        u_grid = u_geo * c - v_geo * s
        v_grid = u_geo * s + v_geo * c
        return u_grid, v_grid

    def grid_to_geo_wind(
        self, u_grid: np.ndarray, v_grid: np.ndarray, lon_deg: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Rotate LCC (x, y) grid wind components back to true Earth (east, north)."""
        gamma = self.wind_rotation_angle_rad(lon_deg)
        c = np.cos(gamma)
        s = np.sin(gamma)
        u_geo = u_grid * c + v_grid * s
        v_geo = -u_grid * s + v_grid * c
        return u_geo, v_geo


@dataclass
class VerticalGrid:
    """Smooth Klemp (2011) terrain-following height grid and hydrostatic reference atmosphere."""

    nz: int = 137
    z_top_m: float = 31_000.0
    stretch_power: float = 1.5
    terrain_decay_power: float = 2.5

    def __post_init__(self) -> None:
        if self.nz < 4:
            raise ValueError(f"VerticalGrid requires nz >= 4, got {self.nz}")
        if self.z_top_m <= 0.0:
            raise ValueError(f"z_top_m must be positive, got {self.z_top_m}")

        s = np.linspace(0.0, 1.0, self.nz + 1, dtype=np.float64)
        self.zeta_w = self.z_top_m * np.power(s, self.stretch_power)
        self.zeta_w[0] = 0.0
        self.zeta_w[-1] = float(self.z_top_m)
        self.zeta_s = 0.5 * (self.zeta_w[:-1] + self.zeta_w[1:])
        self.dzeta = np.diff(self.zeta_w)

        def _b_func(zeta: np.ndarray) -> np.ndarray:
            return np.power(np.clip(1.0 - zeta / self.z_top_m, 0.0, 1.0), self.terrain_decay_power)

        def _db_func(zeta: np.ndarray) -> np.ndarray:
            if abs(self.terrain_decay_power - 1.0) < 1e-12:
                return np.full_like(zeta, -1.0 / self.z_top_m)
            base = np.clip(1.0 - zeta / self.z_top_m, 0.0, 1.0)
            return -(self.terrain_decay_power / self.z_top_m) * np.power(
                base, self.terrain_decay_power - 1.0
            )

        self.B_w = _b_func(self.zeta_w)
        self.B_s = _b_func(self.zeta_s)
        self.dB_dzeta_w = _db_func(self.zeta_w)
        self.dB_dzeta_s = _db_func(self.zeta_s)

    def heights_3d(self, hs_m: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute physical heights z_w(nz+1, ny, nx), z_s(nz, ny, nx), and Jacobian G_s(nz, ny, nx)."""
        hs = np.asarray(hs_m, dtype=np.float64)
        z_w = self.zeta_w[:, None, None] + hs[None, :, :] * self.B_w[:, None, None]
        z_s = self.zeta_s[:, None, None] + hs[None, :, :] * self.B_s[:, None, None]
        g_s = 1.0 + hs[None, :, :] * self.dB_dzeta_s[:, None, None]
        return z_w, z_s, g_s

    def check_jacobian_positive(self, hs_m: np.ndarray) -> float:
        """Return minimum Jacobian G across interfaces/centers; raises if inverted."""
        hs = np.asarray(hs_m, dtype=np.float64)
        g_min = float(np.min(1.0 + np.max(hs) * self.dB_dzeta_w))
        if g_min <= 0.05:
            raise ValueError(
                f"Coordinate folding: max(hs)={np.max(hs):.1f} m with z_top={self.z_top_m:.1f} m "
                f"yields min Jacobian G={g_min:.4f}"
            )
        return g_min

    @staticmethod
    def reference_state(
        z_m: np.ndarray,
        theta_sfc_k: float = 288.15,
        p_sfc_pa: float = 101_325.0,
        brunt_vaisala_hz: float = 0.01,
    ) -> Dict[str, np.ndarray]:
        """Exact analytic hydrostatic reference atmosphere for arbitrary heights z_m."""
        z = np.asarray(z_m, dtype=np.float64)
        pi_sfc = (p_sfc_pa / P0_PA) ** KAPPA
        n2 = brunt_vaisala_hz * brunt_vaisala_hz
        if n2 > 1e-12:
            exp_arg = (n2 * z) / GRAVITY_M_S2
            theta_bar = theta_sfc_k * np.exp(exp_arg)
            pi_bar = pi_sfc - (GRAVITY_M_S2 * GRAVITY_M_S2) / (
                CP_D * theta_sfc_k * n2
            ) * (1.0 - np.exp(-exp_arg))
        else:
            theta_bar = np.full_like(z, theta_sfc_k, dtype=np.float64)
            pi_bar = pi_sfc - (GRAVITY_M_S2 * z) / (CP_D * theta_sfc_k)
        pi_bar = np.maximum(pi_bar, 1e-4)
        p_bar = P0_PA * np.power(pi_bar, 1.0 / KAPPA)
        t_bar = theta_bar * pi_bar
        rho_bar = p_bar / (R_D * t_bar)
        cs2_bar = (CP_D / (CP_D - R_D)) * R_D * t_bar
        return {
            "theta_bar": theta_bar,
            "pi_bar": pi_bar,
            "p_bar": p_bar,
            "t_bar": t_bar,
            "rho_bar": rho_bar,
            "cs2_bar": cs2_bar,
        }


@dataclass(frozen=True)
class DomainSpec:
    """Horizontal + vertical specification for a Tempest-NWP v2 domain."""

    name: str
    ny: int
    nx: int
    dx_m: float
    dy_m: float
    is_projected: bool
    is_periodic_lon: bool
    projection: Optional[LCCProjection]
    x_m: np.ndarray
    y_m: np.ndarray
    lat2d: np.ndarray
    lon2d: np.ndarray
    map_factor: np.ndarray
    coriolis_f: np.ndarray
    vertical: VerticalGrid
    is_spherical: bool = False

    @property
    def nz(self) -> int:
        return self.vertical.nz


def _make_lcc_domain(
    name: str,
    ny: int,
    nx: int,
    dx_m: float,
    projection: LCCProjection,
    vertical: VerticalGrid,
    center_lat_deg: float = 51.0,
    center_lon_deg: float = 8.5,
) -> DomainSpec:
    cx, cy = projection.latlon_to_xy(center_lat_deg, center_lon_deg)
    x_m = float(cx) + (np.arange(nx, dtype=np.float64) - 0.5 * (nx - 1)) * dx_m
    y_m = float(cy) + (np.arange(ny, dtype=np.float64) - 0.5 * (ny - 1)) * dx_m
    xx, yy = np.meshgrid(x_m, y_m)
    lat2d, lon2d = projection.xy_to_latlon(xx, yy)
    map_factor = projection.map_scale_factor(lat2d)
    f_cor = projection.coriolis_f(lat2d)
    return DomainSpec(
        name=name,
        ny=ny,
        nx=nx,
        dx_m=dx_m,
        dy_m=dx_m,
        is_projected=True,
        is_periodic_lon=False,
        projection=projection,
        x_m=x_m,
        y_m=y_m,
        lat2d=lat2d,
        lon2d=lon2d,
        map_factor=map_factor,
        coriolis_f=f_cor,
        vertical=vertical,
    )


D1_DOMAIN_DEFAULT_DT_S: Dict[str, float] = {
    "d1_global_0p25": 60.0,
    "d1_global_0p4": 90.0,
    "d1_global_0p5": 120.0,
}


def get_domain_spec(
    name: str,
    nz: int = 137,
    z_top_m: Optional[float] = None,
    stretch_power: float = 1.5,
) -> DomainSpec:
    """Return canonical DomainSpec for D1 Global, D2 Europe LCC (7.5 km), or D3 Netherlands+buffer LCC (2.5 km default, 216x216; optional 'd3_benelux_2p5km' 432x432 or 'd3_1p5km' 720x720)."""
    key = name.strip().lower().replace("-", "_")
    if z_top_m is None:
        z_top_m = 31_000.0 if nz >= 90 else 22_000.0
    vgrid = VerticalGrid(nz=nz, z_top_m=z_top_m, stretch_power=stretch_power)

    if key in ("d3", "d3_nl", "d3_nl_2p5km", "nl_2p5km"):
        proj = LCCProjection(lat1_deg=50.5, lat2_deg=53.8, lat0_deg=52.15, lon0_deg=5.30)
        return _make_lcc_domain(
            "d3_nl_2p5km",
            ny=216,
            nx=216,
            dx_m=2500.0,
            projection=proj,
            vertical=vgrid,
            center_lat_deg=52.15,
            center_lon_deg=5.30,
        )

    if key in ("d3_2p5km", "d3_benelux_de_2p5km", "d3_benelux_2p5km", "benelux_de_2p5km", "benelux_2p5km"):
        proj = LCCProjection(lat1_deg=48.0, lat2_deg=54.0, lat0_deg=51.0, lon0_deg=8.5)
        return _make_lcc_domain(
            "d3_benelux_de_2p5km",
            ny=432,
            nx=432,
            dx_m=2500.0,
            projection=proj,
            vertical=vgrid,
            center_lat_deg=51.0,
            center_lon_deg=8.5,
        )

    if key in ("d3_1p5km", "d3_benelux_de_1p5km", "d3_benelux_1p5km", "benelux_de_1p5km", "benelux_1p5km"):
        proj = LCCProjection(lat1_deg=48.0, lat2_deg=54.0, lat0_deg=51.0, lon0_deg=8.5)
        return _make_lcc_domain(
            "d3_benelux_de_1p5km",
            ny=720,
            nx=720,
            dx_m=1500.0,
            projection=proj,
            vertical=vgrid,
            center_lat_deg=51.0,
            center_lon_deg=8.5,
        )

    if key in ("d2", "d2_eu_lcc", "d2_europe", "d2_europe_7p5km"):
        proj = LCCProjection(lat1_deg=48.0, lat2_deg=54.0, lat0_deg=51.0, lon0_deg=8.5)
        return _make_lcc_domain(
            "d2_eu_lcc",
            ny=400,
            nx=520,
            dx_m=7500.0,
            projection=proj,
            vertical=vgrid,
            center_lat_deg=51.0,
            center_lon_deg=8.5,
        )

    if key in ("d1", "d1_global", "d1_global_0p25", "global", "global025"):
        ny, nx = 721, 1440
        lats = np.linspace(-90.0, 90.0, ny, dtype=np.float64)
        lons = np.arange(nx, dtype=np.float64) * 0.25
        lon2d, lat2d = np.meshgrid(lons, lats)
        dy_m = EARTH_RADIUS_M * math.radians(0.25)
        cos_lat = np.maximum(np.cos(np.radians(lat2d)), math.cos(math.radians(89.5)))
        return DomainSpec(
            name="d1_global_0p25",
            ny=ny,
            nx=nx,
            dx_m=dy_m,
            dy_m=dy_m,
            is_projected=False,
            is_periodic_lon=True,
            projection=None,
            x_m=lons,
            y_m=lats,
            lat2d=lat2d,
            lon2d=lon2d,
            map_factor=1.0 / cos_lat,
            coriolis_f=2.0 * OMEGA_RAD_S * np.sin(np.radians(lat2d)),
            vertical=vgrid,
            is_spherical=True,
        )

    if key in ("d1_0p4", "d1_global_0p4"):
        ny, nx = 451, 900
        lats = np.linspace(-90.0, 90.0, ny, dtype=np.float64)
        lons = np.arange(nx, dtype=np.float64) * 0.4
        dy_m = EARTH_RADIUS_M * math.radians(0.4)
        lon2d, lat2d = np.meshgrid(lons, lats)
        cos_lat = np.maximum(np.cos(np.radians(lat2d)), math.cos(math.radians(89.5)))
        return DomainSpec(
            name="d1_global_0p4",
            ny=ny,
            nx=nx,
            dx_m=dy_m,
            dy_m=dy_m,
            is_projected=False,
            is_periodic_lon=True,
            projection=None,
            x_m=lons,
            y_m=lats,
            lat2d=lat2d,
            lon2d=lon2d,
            map_factor=1.0 / cos_lat,
            coriolis_f=2.0 * OMEGA_RAD_S * np.sin(np.radians(lat2d)),
            vertical=vgrid,
            is_spherical=True,
        )

    if key in ("d1_0p5", "d1_global_0p5"):
        ny, nx = 361, 720
        lats = np.linspace(-90.0, 90.0, ny, dtype=np.float64)
        lons = np.arange(nx, dtype=np.float64) * 0.5
        lon2d, lat2d = np.meshgrid(lons, lats)
        dy_m = EARTH_RADIUS_M * math.radians(0.5)
        cos_lat = np.maximum(np.cos(np.radians(lat2d)), math.cos(math.radians(89.5)))
        return DomainSpec(
            name="d1_global_0p5",
            ny=ny,
            nx=nx,
            dx_m=dy_m,
            dy_m=dy_m,
            is_projected=False,
            is_periodic_lon=True,
            projection=None,
            x_m=lons,
            y_m=lats,
            lat2d=lat2d,
            lon2d=lon2d,
            map_factor=1.0 / cos_lat,
            coriolis_f=2.0 * OMEGA_RAD_S * np.sin(np.radians(lat2d)),
            vertical=vgrid,
            is_spherical=True,
        )

    raise ValueError(f"Unknown domain spec '{name}'")
