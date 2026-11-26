"""Native GPU NEMO 5.0.2 3D Baroclinic Ocean + SI3 5-Category Sea-Ice Bridge.

Strictly 100% deterministic classical ocean fluid dynamics, TEOS-10 thermodynamics,
and sea-ice mechanics (0% AI / 0% ML), strict SI units, lossless WX-1 restarts.
"""

from __future__ import annotations

import ctypes
import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree
import xarray as xr
import wx1

from tempest_v2.cuda_bridge import load_cuda_library
from tempest_v2.projection import DomainSpec

SI3_NCAT = 5
SI3_NLAY = 4
SI3_RHOI = 917.0
SI3_RHOS = 330.0
O_RHO0 = 1026.0
O_CP0 = 3991.867957
SI3_CPI = 2096.7
SI3_LFI = 3.333601e5
SI3_RTMLT = 0.054
RT0 = 273.15

CAT_BOUNDS_M = (0.0, 0.45, 1.10, 2.10, 4.00, 100.0)

DEFAULT_OCEAN_DZ_M: Tuple[float, ...] = (
    5.0, 8.0, 12.0, 18.0, 25.0, 40.0, 60.0, 100.0,
    150.0, 220.0, 320.0, 450.0, 600.0, 750.0, 900.0, 1000.0,
)

OCEAN_3D_FIELDS = ("temp_k", "sal_psu", "u_oce", "v_oce", "tke", "avm", "avt")
OCEAN_2D_FIELDS = ("ssh", "ub", "vb", "mld")
OCEAN_FP64_FIELDS = frozenset(("temp_k", "sal_psu", "ssh"))
SI3_CAT_FIELDS = ("a_i", "v_i", "v_s", "sv_i", "t_su", "t_s", "a_ip", "v_ip")
SI3_2D_FIELDS = ("u_ice", "v_ice", "stress1", "stress2", "stress12", "ice_frac", "ice_thick", "ice_tsfc", "ice_albedo")
OCEAN_STATE_STRUCT_ORDER = OCEAN_3D_FIELDS + OCEAN_2D_FIELDS + SI3_CAT_FIELDS[:4] + ("e_i",) + SI3_CAT_FIELDS[4:] + SI3_2D_FIELDS


class NhOceanConfigC(ctypes.Structure):
    _fields_ = [
        ("nz", ctypes.c_int32),
        ("ny", ctypes.c_int32),
        ("nx", ctypes.c_int32),
        ("bc_x", ctypes.c_int32),
        ("bc_y", ctypes.c_int32),
        ("is_spherical", ctypes.c_int32),
        ("enable_dynamics", ctypes.c_int32),
        ("enable_si3", ctypes.c_int32),
        ("n_bt_substeps", ctypes.c_int32),
        ("n_evp_substeps", ctypes.c_int32),
        ("dx_m", ctypes.c_double),
        ("dy_m", ctypes.c_double),
        ("rho0", ctypes.c_double),
        ("cp_o", ctypes.c_double),
        ("ah_lap", ctypes.c_double),
        ("avm_bg", ctypes.c_double),
        ("avt_bg", ctypes.c_double),
        ("cd_bot", ctypes.c_double),
        ("p_ref_pa", ctypes.c_double),
        ("rn_pstar", ctypes.c_double),
        ("rn_crhg", ctypes.c_double),
        ("rn_ecc", ctypes.c_double),
        ("rn_cd_io", ctypes.c_double),
        ("rn_hinew", ctypes.c_double),
        ("rn_amax", ctypes.c_double),
    ]


class NhOceanStaticPtrsC(ctypes.Structure):
    _fields_ = [
        ("dz_m", ctypes.c_void_p),
        ("msf", ctypes.c_void_p),
        ("f_cor", ctypes.c_void_p),
        ("bathy_m", ctypes.c_void_p),
        ("ocean_mask", ctypes.c_void_p),
    ]


class NhOceanStatePtrsC(ctypes.Structure):
    _fields_ = [(k, ctypes.c_void_p) for k in OCEAN_STATE_STRUCT_ORDER]


class NhOceanForcingPtrsC(ctypes.Structure):
    _fields_ = [
        ("u10m", ctypes.c_void_p),
        ("v10m", ctypes.c_void_p),
        ("t2m", ctypes.c_void_p),
        ("q2m", ctypes.c_void_p),
        ("mslp", ctypes.c_void_p),
        ("sw_down", ctypes.c_void_p),
        ("lw_down", ctypes.c_void_p),
        ("precip_liq", ctypes.c_void_p),
        ("precip_ice", ctypes.c_void_p),
        ("cld_fra", ctypes.c_void_p),
    ]


_CONFIGURED_LIBS: set[int] = set()


def _configure_ocean_ctypes(lib: ctypes.CDLL) -> ctypes.CDLL:
    lid = id(lib)
    if lid in _CONFIGURED_LIBS:
        return lib
    lib.nh_ocean_si3_last_error.restype = ctypes.c_char_p
    lib.nh_ocean_si3_estimate_vram_bytes.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
    lib.nh_ocean_si3_estimate_vram_bytes.restype = ctypes.c_uint64
    lib.nh_ocean_si3_create.argtypes = [ctypes.POINTER(NhOceanConfigC), ctypes.POINTER(NhOceanStaticPtrsC)]
    lib.nh_ocean_si3_create.restype = ctypes.c_void_p
    lib.nh_ocean_si3_destroy.argtypes = [ctypes.c_void_p]
    lib.nh_ocean_si3_destroy.restype = None
    lib.nh_ocean_si3_upload_state.argtypes = [ctypes.c_void_p, ctypes.POINTER(NhOceanStatePtrsC)]
    lib.nh_ocean_si3_upload_state.restype = ctypes.c_int32
    lib.nh_ocean_si3_download_state.argtypes = [ctypes.c_void_p, ctypes.POINTER(NhOceanStatePtrsC)]
    lib.nh_ocean_si3_download_state.restype = ctypes.c_int32
    lib.nh_ocean_si3_step.argtypes = [ctypes.c_void_p, ctypes.POINTER(NhOceanForcingPtrsC), ctypes.c_double]
    lib.nh_ocean_si3_step.restype = ctypes.c_int32
    lib.nh_ocean_si3_teos10_rho.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double]
    lib.nh_ocean_si3_teos10_rho.restype = ctypes.c_double
    lib.nh_ocean_si3_freezing_temp_c.argtypes = [ctypes.c_double, ctypes.c_double]
    lib.nh_ocean_si3_freezing_temp_c.restype = ctypes.c_double
    _CONFIGURED_LIBS.add(lid)
    return lib


def ocean_dz_for_nz(nz: int) -> np.ndarray:
    """Return NEMO ocean vertical z-level thicknesses [m] with ~5 m surface layer."""
    if nz == len(DEFAULT_OCEAN_DZ_M):
        return np.asarray(DEFAULT_OCEAN_DZ_M, dtype=np.float64)
    if nz <= 0 or nz > 64:
        raise ValueError(f"Ocean nz must be in [1, 64], got {nz}")
    if nz == 1:
        return np.array([50.0], dtype=np.float64)
    # Geometric stretching starting at 5.0 m surface thickness
    return np.geomspace(5.0, max(50.0, 4658.0 / max(nz * 0.65, 1.0)), nz).astype(np.float64)


def teos10_rho(ct_c: float, sa_psu: float, depth_m: float = 0.0) -> float:
    """Exact Roquet (2015) TEOS-10 seawater density [kg m^-3] via CUDA library."""
    lib = _configure_ocean_ctypes(load_cuda_library())
    return float(lib.nh_ocean_si3_teos10_rho(float(ct_c), float(sa_psu), float(depth_m)))


def freezing_temp_c(sa_psu: float, depth_m: float = 0.0) -> float:
    """Exact NEMO 5.0.2 np_teos10 seawater freezing point [Celsius]."""
    lib = _configure_ocean_ctypes(load_cuda_library())
    return float(lib.nh_ocean_si3_freezing_temp_c(float(sa_psu), float(depth_m)))


def estimate_ocean_si3_vram_bytes(nz: int, ny: int, nx: int, enable_si3: bool = True) -> int:
    lib = _configure_ocean_ctypes(load_cuda_library())
    return int(lib.nh_ocean_si3_estimate_vram_bytes(int(nz), int(ny), int(nx), 1 if enable_si3 else 0))


def _bl99_enthalpy_vol(t_ice_c: float, s_ice_psu: float) -> float:
    tmlts = -SI3_RTMLT * max(0.0, float(s_ice_psu))
    tc = min(float(t_ice_c), tmlts - 1e-6)
    frac_melt = max(0.0, 1.0 - tmlts / min(tc, -1e-9)) if tmlts < -1e-9 else 1.0
    return SI3_RHOI * (SI3_CPI * (tmlts - tc) + SI3_LFI * frac_melt - O_CP0 * tmlts)


class NemoSi3GPUSolver:
    """Persistent GPU 3D NEMO 5.0.2 Ocean + SI3 5-Category Sea-Ice Solver."""

    def __init__(
        self,
        spec: DomainSpec,
        surface_static: Dict[str, np.ndarray],
        *,
        nz: int = 16,
        dz_m: Optional[Sequence[float]] = None,
        enable_dynamics: bool = True,
        enable_si3: bool = True,
        n_bt_substeps: int = 12,
        n_evp_substeps: int = 60,
        ah_lap: float = 400.0,
    ) -> None:
        self.lib = _configure_ocean_ctypes(load_cuda_library())
        self.spec = spec
        self.nz = int(nz)
        self.ny = int(spec.ny)
        self.nx = int(spec.nx)
        self.dz_m = np.asarray(dz_m if dz_m is not None else ocean_dz_for_nz(self.nz), dtype=np.float64)
        self.enable_si3 = bool(enable_si3)

        lsm = np.asarray(surface_static["lsm"], dtype=np.float32)
        lake = np.asarray(surface_static.get("lake_mask", np.zeros_like(lsm)), dtype=np.float32)
        bathy_in = np.asarray(surface_static.get("bathy_depth_m", np.where(lsm < 0.5, 40.0, 0.0)), dtype=np.float32)

        self.ocean_mask = np.where((lsm < 0.5) & (lake < 0.5) & (bathy_in > 1.0), 1.0, 0.0).astype(np.float32)
        self.bathy_m = np.where(self.ocean_mask >= 0.5, np.maximum(bathy_in, 5.0), 0.0).astype(np.float32)
        self.msf = np.ascontiguousarray(spec.map_factor, dtype=np.float32)
        self.fcor = np.ascontiguousarray(spec.coriolis_f, dtype=np.float32)

        self.cfg = NhOceanConfigC(
            nz=self.nz,
            ny=self.ny,
            nx=self.nx,
            bc_x=1 if spec.is_periodic_lon else 0,
            bc_y=0,
            is_spherical=1 if spec.is_spherical else 0,
            enable_dynamics=1 if enable_dynamics else 0,
            enable_si3=1 if enable_si3 else 0,
            n_bt_substeps=int(n_bt_substeps),
            n_evp_substeps=int(n_evp_substeps),
            dx_m=float(spec.dx_m),
            dy_m=float(spec.dy_m),
            rho0=O_RHO0,
            cp_o=O_CP0,
            ah_lap=float(ah_lap),
            avm_bg=1.2e-4,
            avt_bg=1.2e-5,
            cd_bot=2.5e-3,
            p_ref_pa=101325.0,
            rn_pstar=2.0e4,
            rn_crhg=20.0,
            rn_ecc=2.0,
            rn_cd_io=5.0e-3,
            rn_hinew=0.10,
            rn_amax=0.997,
        )
        stat_ptrs = NhOceanStaticPtrsC(
            dz_m=self.dz_m.ctypes.data,
            msf=self.msf.ctypes.data,
            f_cor=self.fcor.ctypes.data,
            bathy_m=self.bathy_m.ctypes.data,
            ocean_mask=self.ocean_mask.ctypes.data,
        )
        self._handle = self.lib.nh_ocean_si3_create(ctypes.byref(self.cfg), ctypes.byref(stat_ptrs))
        if not self._handle:
            err = self.lib.nh_ocean_si3_last_error()
            msg = err.decode("utf-8", errors="replace") if err else "unknown error"
            raise RuntimeError(f"Failed creating NemoSi3GPUSolver: {msg}")

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self.lib.nh_ocean_si3_destroy(self._handle)
            self._handle = None

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> "NemoSi3GPUSolver":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def empty_state(self) -> Dict[str, np.ndarray]:
        z_shape = (self.nz, self.ny, self.nx)
        s2 = (self.ny, self.nx)
        s_cat = (SI3_NCAT, self.ny, self.nx)
        s_e = (SI3_NCAT * SI3_NLAY, self.ny, self.nx)
        st: Dict[str, np.ndarray] = {
            k: np.zeros(z_shape, dtype=(np.float64 if k in OCEAN_FP64_FIELDS else np.float32))
            for k in OCEAN_3D_FIELDS
        }
        for k in OCEAN_2D_FIELDS:
            st[k] = np.zeros(s2, dtype=(np.float64 if k in OCEAN_FP64_FIELDS else np.float32))
        for k in SI3_CAT_FIELDS:
            st[k] = np.zeros(s_cat, dtype=np.float32)
        st["e_i"] = np.zeros(s_e, dtype=np.float32)
        for k in SI3_2D_FIELDS:
            st[k] = np.zeros(s2, dtype=np.float32)
        st["ice_tsfc"].fill(float(RT0 - 1.8))
        st["ice_albedo"].fill(0.066)
        st["t_su"].fill(float(RT0 - 1.8))
        st["t_s"].fill(float(RT0 - 1.8))
        return st

    def _pack_state_ptrs(self, state: Dict[str, np.ndarray]) -> Tuple[NhOceanStatePtrsC, list[np.ndarray]]:
        refs: list[np.ndarray] = []
        kwargs: Dict[str, int] = {}
        for k in OCEAN_STATE_STRUCT_ORDER:
            arr = state.get(k)
            if arr is None:
                kwargs[k] = 0
            else:
                dt = np.float64 if k in OCEAN_FP64_FIELDS else np.float32
                c_arr = np.ascontiguousarray(arr, dtype=dt)
                refs.append(c_arr)
                kwargs[k] = c_arr.ctypes.data
        return NhOceanStatePtrsC(**kwargs), refs

    def upload_state(self, state: Dict[str, np.ndarray]) -> None:
        ptrs, _refs = self._pack_state_ptrs(state)
        rc = self.lib.nh_ocean_si3_upload_state(self._handle, ctypes.byref(ptrs))
        if rc != 0:
            err = self.lib.nh_ocean_si3_last_error()
            raise RuntimeError(f"nh_ocean_si3_upload_state failed: {err}")

    def download_state(self) -> Dict[str, np.ndarray]:
        out = self.empty_state()
        ptrs = NhOceanStatePtrsC(**{k: out[k].ctypes.data for k in OCEAN_STATE_STRUCT_ORDER})
        rc = self.lib.nh_ocean_si3_download_state(self._handle, ctypes.byref(ptrs))
        if rc != 0:
            err = self.lib.nh_ocean_si3_last_error()
            raise RuntimeError(f"nh_ocean_si3_download_state failed: {err}")
        return out

    def step(self, forcing: Dict[str, np.ndarray], dt_s: float) -> None:
        refs: list[np.ndarray] = []
        kwargs: Dict[str, int] = {}
        for k in ("u10m", "v10m", "t2m", "q2m", "mslp", "sw_down", "lw_down", "precip_liq", "precip_ice", "cld_fra"):
            arr = forcing.get(k)
            if arr is None:
                kwargs[k] = 0
            else:
                c_arr = np.ascontiguousarray(arr, dtype=np.float32)
                refs.append(c_arr)
                kwargs[k] = c_arr.ctypes.data
        fptr = NhOceanForcingPtrsC(**kwargs)
        rc = self.lib.nh_ocean_si3_step(self._handle, ctypes.byref(fptr), float(dt_s))
        if rc != 0:
            err = self.lib.nh_ocean_si3_last_error()
            raise RuntimeError(f"nh_ocean_si3_step failed: {err}")

    def init_from_atmosphere_and_osisaf(
        self,
        surf_state: Dict[str, np.ndarray],
        *,
        osisaf_wx1_path: Optional[Path | str] = None,
        obs_dir: Optional[Path | str] = None,
    ) -> Dict[str, np.ndarray]:
        """Construct balanced 3D ocean T/S stratification + 5-cat SI3 ice from surface IC, Argo 3D CTD & OSI-SAF."""
        from tempest_v2.verification import _read_obs_prefix

        st = self.empty_state()
        wet = self.ocean_mask >= 0.5
        lat = self.spec.lat2d.astype(np.float64)
        lon = self.spec.lon2d.astype(np.float64)

        sst = np.asarray(surf_state.get("sst_bulk", surf_state.get("t2m", np.full((self.ny, self.nx), 285.0))), dtype=np.float64)
        sst = np.clip(sst, RT0 - 1.85, 309.0)

        # Realistic salinity distribution (subtropical saltier ~35.5, polar fresher ~33.0, Baltic brackish ~8.0)
        sss = 34.8 - 1.8 * (np.sin(np.radians(lat)) ** 2)
        baltic = (lat > 53.5) & (lat < 66.0) & (lon > 10.0) & (lon < 30.0)
        sss = np.where(baltic, 8.0, sss)
        sss = np.clip(sss, 2.0, 38.0)

        z_w = np.concatenate([[0.0], np.cumsum(self.dz_m)])
        z_t = 0.5 * (z_w[:-1] + z_w[1:])
        for k in range(self.nz):
            decay = np.exp(-z_t[k] / 450.0)
            t_k = 276.0 + (sst - 276.0) * decay
            t_k = np.where(sst < RT0, np.maximum(RT0 - 1.85, sst + (1.0 - decay) * 1.2), t_k)
            s_k = sss + (1.0 - decay) * 0.6
            st["temp_k"][k] = np.where(wet & (z_w[k] < self.bathy_m), t_k, 0.0).astype(np.float32)
            st["sal_psu"][k] = np.where(wet & (z_w[k] < self.bathy_m), s_k, 0.0).astype(np.float32)
            st["tke"][k] = np.where(wet, 1.0e-5, 0.0).astype(np.float32)
            st["avm"][k] = np.where(wet, 1.2e-4, 0.0).astype(np.float32)
            st["avt"][k] = np.where(wet, 1.2e-5, 0.0).astype(np.float32)

        # Assimilate 3D Argo float CTD (T, S) profiles when present in obs_dir or osisaf_wx1_path parent
        argo_root = Path(obs_dir) if obs_dir is not None else (Path(osisaf_wx1_path).parent if osisaf_wx1_path is not None else None)
        if argo_root is not None and argo_root.exists():
            argo = _read_obs_prefix(argo_root, "argo_ocean_profiles")
            if argo is not None and "temp" in argo and "salinity" in argo:
                a_lat = np.asarray(argo["lat"], dtype=np.float64)
                a_lon = np.asarray(argo["lon"], dtype=np.float64)
                a_t = np.asarray(argo["temp"], dtype=np.float64)
                a_s = np.asarray(argo["salinity"], dtype=np.float64)
                a_z = np.asarray(
                    argo.get(
                        "depths",
                        argo.get(
                            "depth",
                            np.array([5.0, 15.0, 25.0, 40.0, 60.0, 80.0, 100.0, 130.0, 170.0, 220.0, 280.0, 360.0, 460.0, 600.0, 800.0, 1050.0, 1300.0, 1550.0, 1800.0, 2000.0]),
                        ),
                    ),
                    dtype=np.float64,
                )
                if a_t.ndim == 2 and a_s.ndim == 2 and len(a_lat) == a_t.shape[0] and len(a_z) == a_t.shape[1]:
                    for idx_a in range(len(a_lat)):
                        if not (np.isfinite(a_lat[idx_a]) and np.isfinite(a_lon[idx_a])):
                            continue
                        dlat_r = np.radians(lat - float(a_lat[idx_a]))
                        dlon_r = np.radians(((lon - float(a_lon[idx_a]) + 180.0) % 360.0) - 180.0) * np.cos(np.radians(0.5 * (lat + float(a_lat[idx_a]))))
                        dist_km = 6371.0 * np.hypot(dlat_r, dlon_r)
                        w_h = np.where(wet & (dist_km < 350.0), 0.75 * np.exp(-0.5 * (dist_km / 150.0) ** 2), 0.0)
                        if not np.any(w_h > 1e-3):
                            continue
                        t_prof = np.interp(z_t, a_z, a_t[idx_a])
                        s_prof = np.interp(z_t, a_z, a_s[idx_a])
                        for k in range(self.nz):
                            if np.isfinite(t_prof[k]) and 271.0 <= t_prof[k] <= 310.0:
                                active_k = wet & (z_w[k] < self.bathy_m)
                                st["temp_k"][k] = np.where(active_k, (1.0 - w_h) * st["temp_k"][k] + w_h * t_prof[k], 0.0).astype(np.float32)
                            if np.isfinite(s_prof[k]) and 2.0 <= s_prof[k] <= 42.0:
                                active_k = wet & (z_w[k] < self.bathy_m)
                                st["sal_psu"][k] = np.where(active_k, (1.0 - w_h) * st["sal_psu"][k] + w_h * s_prof[k], 0.0).astype(np.float32)

        # Determine initial sea-ice fraction & thickness (OSI-SAF WX-1 -> surf_state -> thermal fallback)
        sic = np.asarray(surf_state.get("sea_ice_frac", np.zeros((self.ny, self.nx), dtype=np.float32)), dtype=np.float64)
        sit = np.asarray(surf_state.get("sea_ice_thick", np.zeros((self.ny, self.nx), dtype=np.float32)), dtype=np.float64)

        olat = np.asarray([], dtype=np.float64)
        olon = np.asarray([], dtype=np.float64)
        oconc = np.asarray([], dtype=np.float64)
        othk = np.asarray([], dtype=np.float64)
        if osisaf_wx1_path is not None and Path(osisaf_wx1_path).is_file():
            p_osi = Path(osisaf_wx1_path)
            with wx1.open_dataset(p_osi) as ds:
                if "lat" in ds and "lon" in ds:
                    olat = np.asarray(ds["lat"].values, dtype=np.float64)
                    olon = np.asarray(ds["lon"].values, dtype=np.float64)
                conc_k = next((k for k in ("ice_conc", "conc") if k in ds), None)
                thk_k = next((k for k in ("ice_thick", "thick") if k in ds), None)
                if conc_k is not None:
                    oconc = np.asarray(ds[conc_k].values, dtype=np.float64)
                if thk_k is not None:
                    othk = np.asarray(ds[thk_k].values, dtype=np.float64)
            if len(olat) == 0:
                osi_multi = _read_obs_prefix(p_osi.parent, p_osi.stem)
                if osi_multi is not None and "ice_conc" in osi_multi:
                    olat = np.asarray(osi_multi["lat"], dtype=np.float64)
                    olon = np.asarray(osi_multi["lon"], dtype=np.float64)
                    oconc = np.asarray(osi_multi["ice_conc"], dtype=np.float64)
                    othk = np.asarray(osi_multi.get("ice_thick", np.clip(0.4 + 1.8 * (np.maximum(oconc, 0.0) ** 1.5), 0.1, 4.5)), dtype=np.float64)
        elif obs_dir is not None and Path(obs_dir).exists():
            osi_multi = _read_obs_prefix(Path(obs_dir), "osi_saf_sea_ice")
            if osi_multi is not None and "ice_conc" in osi_multi:
                olat = np.asarray(osi_multi["lat"], dtype=np.float64)
                olon = np.asarray(osi_multi["lon"], dtype=np.float64)
                oconc = np.asarray(osi_multi["ice_conc"], dtype=np.float64)
                othk = np.asarray(osi_multi.get("ice_thick", np.clip(0.4 + 1.8 * (np.maximum(oconc, 0.0) ** 1.5), 0.1, 4.5)), dtype=np.float64)

        if len(olat) > 0 and len(oconc) == len(olat):
            if len(othk) != len(oconc):
                othk = np.clip(0.4 + 1.8 * (np.maximum(oconc, 0.0) ** 1.5), 0.1, 4.5)
            tree = cKDTree(np.column_stack([np.radians(olat), np.radians(((olon + 180.0) % 360.0) - 180.0)]))
            q_pts = np.column_stack([np.radians(lat.ravel()), np.radians(((lon.ravel() + 180.0) % 360.0) - 180.0)])
            dist, idx = tree.query(q_pts, distance_upper_bound=np.radians(1.5))
            valid = np.isfinite(dist) & (idx < len(oconc))
            sic_n = np.zeros_like(sic.ravel())
            sit_n = np.zeros_like(sit.ravel())
            sic_n[valid] = np.clip(oconc[idx[valid]], 0.0, 0.997)
            sit_n[valid] = np.clip(othk[idx[valid]], 0.0, 8.0)
            sic = np.where(sic_n.reshape(self.ny, self.nx) > 0.01, sic_n.reshape(self.ny, self.nx), sic)
            sit = np.where(sic_n.reshape(self.ny, self.nx) > 0.01, sit_n.reshape(self.ny, self.nx), sit)

        if float(np.max(sic)) < 1e-4:
            # Polar sea-ice thermal initialization when SST at freezing point and |lat| > 58 deg
            polar_cold = wet & (np.abs(lat) > 58.0) & (sst <= RT0 - 1.5)
            sic = np.where(polar_cold, 0.85, sic)
            sit = np.where(polar_cold, 1.20, sit)

        sic = np.where(wet, np.clip(sic, 0.0, 0.997), 0.0)
        sit = np.where(sic > 1e-4, np.maximum(sit, 0.15), 0.0)

        # Populate SI3 5-category ITD matching mean thickness `sit`
        if self.enable_si3 and float(np.max(sic)) > 1e-4:
            t_su_init = np.asarray(surf_state.get("sea_ice_tsfc", surf_state.get("t_skin", sst)), dtype=np.float64)
            t_su_init = np.clip(t_su_init, 220.0, RT0 - 0.2)
            for j in range(self.ny):
                for i in range(self.nx):
                    if not wet[j, i] or sic[j, i] <= 1e-4:
                        continue
                    a_tot = float(sic[j, i])
                    h_mean = float(sit[j, i])
                    cat = next((c for c in range(SI3_NCAT) if CAT_BOUNDS_M[c] <= h_mean < CAT_BOUNDS_M[c + 1]), SI3_NCAT - 1)
                    s_ice = min(6.0, 0.25 * float(sss[j, i]))
                    ts_c = min(-0.5, float(t_su_init[j, i]) - RT0)
                    e_vol = _bl99_enthalpy_vol(0.5 * (ts_c - 1.8), s_ice)
                    st["a_i"][cat, j, i] = np.float32(a_tot)
                    st["v_i"][cat, j, i] = np.float32(a_tot * h_mean)
                    st["v_s"][cat, j, i] = np.float32(a_tot * min(0.15, 0.1 * h_mean))
                    st["sv_i"][cat, j, i] = np.float32(a_tot * h_mean * s_ice)
                    st["t_su"][cat, j, i] = np.float32(RT0 + ts_c)
                    st["t_s"][cat, j, i] = np.float32(RT0 + ts_c)
                    for lay in range(SI3_NLAY):
                        st["e_i"][cat * SI3_NLAY + lay, j, i] = np.float32((a_tot * h_mean / SI3_NLAY) * e_vol)
                    st["ice_frac"][j, i] = np.float32(a_tot)
                    st["ice_thick"][j, i] = np.float32(h_mean)
                    st["ice_tsfc"][j, i] = np.float32(RT0 + ts_c)
                    st["ice_albedo"][j, i] = np.float32(0.75 if st["v_s"][cat, j, i] > 0 else 0.64)
                    st["temp_k"][0, j, i] = np.float32(max(float(st["temp_k"][0, j, i]), RT0 + freezing_temp_c(float(sss[j, i]), 2.5)))

        return st

    def build_forcing_from_atmosphere(
        self,
        spec: DomainSpec,
        atm_state: Dict[str, np.ndarray],
        surf_state: Dict[str, np.ndarray],
        *,
        mslp_pa: Optional[np.ndarray] = None,
        cld_fra: Optional[np.ndarray] = None,
        prev_rain: Optional[np.ndarray] = None,
        prev_snow: Optional[np.ndarray] = None,
        dt_interval_s: float = 1800.0,
    ) -> Dict[str, np.ndarray]:
        u10 = np.asarray(surf_state.get("u10m", np.zeros((self.ny, self.nx), dtype=np.float32)), dtype=np.float32)
        v10 = np.asarray(surf_state.get("v10m", np.zeros((self.ny, self.nx), dtype=np.float32)), dtype=np.float32)
        t2m = np.asarray(surf_state.get("t2m", np.full((self.ny, self.nx), 285.0, dtype=np.float32)), dtype=np.float32)
        q2m = np.asarray(surf_state.get("q2m", np.full((self.ny, self.nx), 0.005, dtype=np.float32)), dtype=np.float32)
        sw = np.asarray(surf_state.get("sw_down", np.zeros((self.ny, self.nx), dtype=np.float32)), dtype=np.float32)
        lw = np.asarray(surf_state.get("lw_down", np.full((self.ny, self.nx), 300.0, dtype=np.float32)), dtype=np.float32)

        if mslp_pa is None:
            mslp_pa = np.full((self.ny, self.nx), 101325.0, dtype=np.float32)
        if cld_fra is None:
            qc = np.asarray(atm_state.get("qc", np.zeros((spec.nz, self.ny, self.nx), dtype=np.float32)))
            qi = np.asarray(atm_state.get("qi", np.zeros((spec.nz, self.ny, self.nx), dtype=np.float32)))
            cwp = np.sum(np.maximum(qc + qi, 0.0), axis=0)
            cld_fra = np.clip(cwp / 0.0015, 0.0, 1.0).astype(np.float32)

        dt_f = max(float(dt_interval_s), 1.0)
        rain_tot = np.asarray(surf_state.get("rain_acc", np.zeros((self.ny, self.nx), dtype=np.float32)), dtype=np.float32)
        ice_tot = np.asarray(surf_state.get("snow_acc", np.zeros((self.ny, self.nx), dtype=np.float32)), dtype=np.float32) + np.asarray(
            surf_state.get("graupel_acc", np.zeros((self.ny, self.nx), dtype=np.float32)), dtype=np.float32
        )
        p_liq = np.maximum(0.0, (rain_tot - prev_rain) if prev_rain is not None else np.zeros_like(rain_tot)) / dt_f
        p_ice = np.maximum(0.0, (ice_tot - prev_snow) if prev_snow is not None else np.zeros_like(ice_tot)) / dt_f

        return {
            "u10m": u10,
            "v10m": v10,
            "t2m": t2m,
            "q2m": q2m,
            "mslp": np.asarray(mslp_pa, dtype=np.float32),
            "sw_down": sw,
            "lw_down": lw,
            "precip_liq": p_liq.astype(np.float32),
            "precip_ice": p_ice.astype(np.float32),
            "cld_fra": np.asarray(cld_fra, dtype=np.float32),
        }

    def couple_to_surface_state(
        self, oce_state: Dict[str, np.ndarray], surf_state: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Update atmospheric SST + SI3 sea-ice mosaic over wet ocean cells cleanly."""
        wet = self.ocean_mask >= 0.5
        out = dict(surf_state)
        sst_o = np.asarray(oce_state["temp_k"][0], dtype=np.float32)
        cur_sst = np.asarray(surf_state.get("sst_bulk", sst_o), dtype=np.float32)
        out["sst_bulk"] = np.where(wet, np.clip(sst_o, 270.5, 312.0), cur_sst).astype(np.float32)
        for okey, skey, fallback in (
            ("ice_frac", "sea_ice_frac", 0.0),
            ("ice_thick", "sea_ice_thick", 0.0),
            ("ice_tsfc", "sea_ice_tsfc", float(RT0 - 1.8)),
            ("ice_albedo", "sea_ice_albedo", 0.066),
        ):
            cur = np.asarray(surf_state.get(skey, np.full((self.ny, self.nx), fallback, dtype=np.float32)), dtype=np.float32)
            out[skey] = np.where(wet, np.asarray(oce_state[okey], dtype=np.float32), cur).astype(np.float32)
        return out

    def total_salt_kg_m2(self, oce_state: Dict[str, np.ndarray]) -> np.ndarray:
        """Exact column-integrated ocean + SI3 sea-ice salt mass [kg m^-2] per grid cell."""
        wet = self.ocean_mask >= 0.5
        h0 = np.maximum(0.5 * self.dz_m[0], self.dz_m[0] + np.asarray(oce_state["ssh"], dtype=np.float64))
        salt_oce = h0 * O_RHO0 * np.asarray(oce_state["sal_psu"][0], dtype=np.float64)
        for k in range(1, self.nz):
            b_k = wet & (float(np.sum(self.dz_m[:k])) < self.bathy_m)
            salt_oce += np.where(b_k, self.dz_m[k] * O_RHO0 * np.asarray(oce_state["sal_psu"][k], dtype=np.float64), 0.0)
        salt_ice = SI3_RHOI * np.sum(np.asarray(oce_state["sv_i"], dtype=np.float64), axis=0)
        # g/kg -> kg/m^2 = factor 1e-3
        return np.where(wet, (salt_oce + salt_ice) * 1.0e-3, 0.0)

    def save_restart_wx1(
        self,
        path: Path | str,
        valid_time_utc: datetime.datetime,
        oce_state: Dict[str, np.ndarray],
        *,
        sim_time_sec: float = 0.0,
    ) -> Path:
        out_p = Path(path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        v_dt = (
            valid_time_utc.astimezone(datetime.timezone.utc)
            if valid_time_utc.tzinfo
            else valid_time_utc.replace(tzinfo=datetime.timezone.utc)
        )
        t_coord = np.array([np.datetime64(int(v_dt.timestamp()), "s")], dtype="datetime64[s]")
        data_vars: Dict[str, Any] = {}
        for k in OCEAN_STATE_STRUCT_ORDER:
            if k not in oce_state:
                continue
            dt = np.float64 if k in OCEAN_FP64_FIELDS else np.float32
            arr = np.asarray(oce_state[k], dtype=dt)
            if k in OCEAN_3D_FIELDS:
                dims = ("z_oce", "y", "x")
            elif k in SI3_CAT_FIELDS:
                dims = ("ice_cat", "y", "x")
            elif k == "e_i":
                dims = ("ice_e_lay", "y", "x")
            else:
                dims = ("y", "x")
            data_vars[k] = (dims, arr)

        ds = xr.Dataset(
            data_vars=data_vars,
            coords={
                "time": ("time", t_coord),
                "z_oce": ("z_oce", np.arange(self.nz, dtype=np.int32)),
                "ice_cat": ("ice_cat", np.arange(SI3_NCAT, dtype=np.int32)),
                "ice_e_lay": ("ice_e_lay", np.arange(SI3_NCAT * SI3_NLAY, dtype=np.int32)),
                "y": ("y", self.spec.y_m.astype(np.float64)),
                "x": ("x", self.spec.x_m.astype(np.float64)),
            },
            attrs={
                "kind": "tempest_v2_ocean_si3_restart",
                "domain_name": self.spec.name,
                "valid_time_utc": v_dt.isoformat(),
                "sim_time_sec": f"{float(sim_time_sec):.6f}",
            },
        )
        wx1.write_dataset(out_p, ds, dataset_id=f"ocean_si3_{self.spec.name}", codec="zstd_shuffle", workers=4)
        return out_p

    @staticmethod
    def load_restart_wx1(path: Path | str) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        state: Dict[str, np.ndarray] = {}
        with wx1.open_dataset(path) as ds:
            meta = dict(ds.attrs)
            for k in OCEAN_STATE_STRUCT_ORDER:
                if k in ds:
                    dt = np.float64 if k in OCEAN_FP64_FIELDS else np.float32
                    state[k] = np.asarray(ds[k].values, dtype=dt)
        return state, meta
