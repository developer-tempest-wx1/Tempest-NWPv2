"""Python ctypes bridge and automatic 2D halo-tiling manager for tempest-cuda."""

from __future__ import annotations

import ctypes
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from tempest_v2.projection import VerticalGrid

DEFAULT_MAX_VRAM_BYTES = int(4.45 * (1024**3))  # 4.45 GiB budget on 6 GiB RTX 3060 (1 tile @ nz=65, 2x2 tiles @ nz=137)
DEFAULT_HALO_CELLS = 20  # >= 13 + 6*ns stencil radius for 1 full 3-stage WENO-5 RK3 + FCT step


class NhGridConfigC(ctypes.Structure):
    _fields_ = [
        ("nz", ctypes.c_int32),
        ("ny", ctypes.c_int32),
        ("nx", ctypes.c_int32),
        ("has_moist", ctypes.c_int32),
        ("bc_x", ctypes.c_int32),
        ("bc_y", ctypes.c_int32),
        ("lbc_edges", ctypes.c_int32),
        ("use_coriolis", ctypes.c_int32),
        ("adv_order", ctypes.c_int32),
        ("lbc_rim_width", ctypes.c_int32),
        ("has_iau", ctypes.c_int32),
        ("dx_m", ctypes.c_double),
        ("dy_m", ctypes.c_double),
        ("z_top_m", ctypes.c_double),
        ("theta_sfc_k", ctypes.c_double),
        ("p_sfc_pa", ctypes.c_double),
        ("n_ref_s", ctypes.c_double),
        ("beta_offcenter", ctypes.c_double),
        ("div_damp_coeff", ctypes.c_double),
        ("km_const", ctypes.c_double),
        ("kh_const", ctypes.c_double),
        ("hyperdiff_coeff", ctypes.c_double),
        ("sponge_depth_m", ctypes.c_double),
        ("sponge_tau_max_s", ctypes.c_double),
        ("lbc_tau_max_s", ctypes.c_double),
        ("is_spherical", ctypes.c_int32),
        ("polar_filter_lat_deg", ctypes.c_double),
    ]


class NhStatePtrsC(ctypes.Structure):
    _fields_ = [
        ("u", ctypes.c_void_p),
        ("v", ctypes.c_void_p),
        ("w", ctypes.c_void_p),
        ("theta_p", ctypes.c_void_p),
        ("pi_p", ctypes.c_void_p),
        ("qv", ctypes.c_void_p),
        ("qc", ctypes.c_void_p),
        ("qr", ctypes.c_void_p),
        ("qi", ctypes.c_void_p),
        ("qs", ctypes.c_void_p),
        ("qg", ctypes.c_void_p),
        ("tke", ctypes.c_void_p),
    ]


class NhPhysicsConfigC(ctypes.Structure):
    _fields_ = [
        ("enable_microphysics", ctypes.c_int32),
        ("enable_pbl", ctypes.c_int32),
        ("enable_radiation", ctypes.c_int32),
        ("enable_surface", ctypes.c_int32),
        ("rad_interval_s", ctypes.c_double),
        ("cloud_num_land", ctypes.c_double),
        ("cloud_num_sea", ctypes.c_double),
        ("julian_day", ctypes.c_double),
        ("utc_seconds", ctypes.c_double),
        ("co2_ppm", ctypes.c_double),
        ("pbl_last", ctypes.c_int32),
        ("cy46_physics", ctypes.c_int32),
    ]


class NhSurfaceStaticPtrsC(ctypes.Structure):
    _fields_ = [
        ("lat_deg", ctypes.c_void_p),
        ("lon_deg", ctypes.c_void_p),
        ("lsm", ctypes.c_void_p),
        ("lake_mask", ctypes.c_void_p),
        ("soil_type", ctypes.c_void_p),
        ("bathy_depth_m", ctypes.c_void_p),
        ("z0m_base", ctypes.c_void_p),
        ("albedo_base", ctypes.c_void_p),
        ("veg_frac", ctypes.c_void_p),
    ]


class NhSurfaceStatePtrsC(ctypes.Structure):
    _fields_ = [
        ("t_skin", ctypes.c_void_p),
        ("soil_t", ctypes.c_void_p),
        ("soil_m", ctypes.c_void_p),
        ("snow_swe", ctypes.c_void_p),
        ("sst_bulk", ctypes.c_void_p),
        ("dt_cool", ctypes.c_void_p),
        ("dt_warm", ctypes.c_void_p),
        ("lake_tbot", ctypes.c_void_p),
        ("lake_hmld", ctypes.c_void_p),
        ("lake_ctheta", ctypes.c_void_p),
        ("lake_hice", ctypes.c_void_p),
        ("rain_acc", ctypes.c_void_p),
        ("snow_acc", ctypes.c_void_p),
        ("graupel_acc", ctypes.c_void_p),
        ("sw_down", ctypes.c_void_p),
        ("lw_down", ctypes.c_void_p),
        ("shf", ctypes.c_void_p),
        ("lhf", ctypes.c_void_p),
        ("ustar", ctypes.c_void_p),
        ("pblh", ctypes.c_void_p),
        ("t2m", ctypes.c_void_p),
        ("q2m", ctypes.c_void_p),
        ("u10m", ctypes.c_void_p),
        ("v10m", ctypes.c_void_p),
        ("sea_ice_frac", ctypes.c_void_p),
        ("sea_ice_thick", ctypes.c_void_p),
        ("sea_ice_tsfc", ctypes.c_void_p),
        ("sea_ice_albedo", ctypes.c_void_p),
    ]


def _default_lib_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "crates" / "tempest-cuda" / "libtempest_cuda.so"


_LIB_CACHE: Optional[ctypes.CDLL] = None


def load_cuda_library(lib_path: Optional[Path] = None) -> ctypes.CDLL:
    """Load `libtempest_cuda.so` and configure exact C ABI signatures."""
    global _LIB_CACHE
    if _LIB_CACHE is not None and lib_path is None:
        return _LIB_CACHE

    path = Path(lib_path) if lib_path is not None else _default_lib_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Shared CUDA core library not found at {path}. "
            "Build it first via `cargo build --release -p tempest-cuda`."
        )

    lib = ctypes.CDLL(str(path))
    lib.nh_cuda_last_error.restype = ctypes.c_char_p
    lib.nh_cuda_vram_info.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
    ]
    lib.nh_cuda_vram_info.restype = ctypes.c_int32
    lib.nh_cuda_estimate_vram_bytes.argtypes = [
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    lib.nh_cuda_estimate_vram_bytes.restype = ctypes.c_uint64
    lib.nh_cuda_estimate_vram_bytes_ex.argtypes = [
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    lib.nh_cuda_estimate_vram_bytes_ex.restype = ctypes.c_uint64
    lib.nh_cuda_create.argtypes = [
        ctypes.POINTER(NhGridConfigC),
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.nh_cuda_create.restype = ctypes.c_void_p
    lib.nh_cuda_destroy.argtypes = [ctypes.c_void_p]
    lib.nh_cuda_destroy.restype = None
    lib.nh_cuda_upload_state.argtypes = [ctypes.c_void_p, ctypes.POINTER(NhStatePtrsC)]
    lib.nh_cuda_upload_state.restype = ctypes.c_int32
    lib.nh_cuda_download_state.argtypes = [ctypes.c_void_p, ctypes.POINTER(NhStatePtrsC)]
    lib.nh_cuda_download_state.restype = ctypes.c_int32
    lib.nh_cuda_set_lbc_target.argtypes = [ctypes.c_void_p, ctypes.POINTER(NhStatePtrsC)]
    lib.nh_cuda_set_lbc_target.restype = ctypes.c_int32
    lib.nh_cuda_set_iau_forcing.argtypes = [ctypes.c_void_p, ctypes.POINTER(NhStatePtrsC)]
    lib.nh_cuda_set_iau_forcing.restype = ctypes.c_int32
    lib.nh_cuda_clear_iau_forcing.argtypes = [ctypes.c_void_p]
    lib.nh_cuda_clear_iau_forcing.restype = ctypes.c_int32
    lib.nh_cuda_set_physics_config.argtypes = [ctypes.c_void_p, ctypes.POINTER(NhPhysicsConfigC)]
    lib.nh_cuda_set_physics_config.restype = ctypes.c_int32
    lib.nh_cuda_set_surface_static.argtypes = [ctypes.c_void_p, ctypes.POINTER(NhSurfaceStaticPtrsC)]
    lib.nh_cuda_set_surface_static.restype = ctypes.c_int32
    lib.nh_cuda_upload_surface.argtypes = [ctypes.c_void_p, ctypes.POINTER(NhSurfaceStatePtrsC)]
    lib.nh_cuda_upload_surface.restype = ctypes.c_int32
    lib.nh_cuda_download_surface.argtypes = [ctypes.c_void_p, ctypes.POINTER(NhSurfaceStatePtrsC)]
    lib.nh_cuda_download_surface.restype = ctypes.c_int32
    lib.nh_cuda_step_physics.argtypes = [ctypes.c_void_p, ctypes.c_double]
    lib.nh_cuda_step_physics.restype = ctypes.c_int32
    lib.nh_cuda_rebind_tile_grid.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(NhGridConfigC),
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.nh_cuda_rebind_tile_grid.restype = ctypes.c_int32
    lib.nh_cuda_estimate_cs_u32.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    lib.nh_cuda_estimate_cs_u32.restype = ctypes.c_int32
    lib.nh_cuda_sync_physics_cache.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int32,
    ]
    lib.nh_cuda_sync_physics_cache.restype = ctypes.c_int32
    lib.nh_cuda_step_rk3.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_int32]
    lib.nh_cuda_step_rk3.restype = ctypes.c_int32

    if lib_path is None:
        _LIB_CACHE = lib
    return lib


def cuda_vram_info() -> Tuple[int, int]:
    """Return `(free_bytes, total_bytes)` on active CUDA device."""
    lib = load_cuda_library()
    free_b = ctypes.c_uint64(0)
    total_b = ctypes.c_uint64(0)
    rc = lib.nh_cuda_vram_info(ctypes.byref(free_b), ctypes.byref(total_b))
    if rc != 0:
        err = lib.nh_cuda_last_error()
        raise RuntimeError(err.decode("utf-8", errors="replace") if err else "cudaMemGetInfo failed")
    return int(free_b.value), int(total_b.value)


def estimate_vram_bytes(
    nz: int,
    ny: int,
    nx: int,
    *,
    moist: bool = True,
    has_lbc: bool = False,
    has_iau: bool = False,
    has_physics: bool = False,
) -> int:
    """Exact GPU VRAM estimate in bytes for a tile `(nz, ny, nx)`."""
    lib = load_cuda_library()
    return int(
        lib.nh_cuda_estimate_vram_bytes_ex(
            int(nz),
            int(ny),
            int(nx),
            1 if moist else 0,
            1 if has_lbc else 0,
            1 if has_iau else 0,
            1 if has_physics else 0,
        )
    )


def plan_horizontal_tiling(
    nz: int,
    ny: int,
    nx: int,
    *,
    moist: bool = True,
    has_lbc: bool = False,
    has_iau: bool = False,
    has_physics: bool = False,
    max_vram_bytes: int = DEFAULT_MAX_VRAM_BYTES,
    halo: int = DEFAULT_HALO_CELLS,
    allow_x_split: bool = True,
) -> Tuple[int, int, int]:
    """Compute minimal `(nt_y, nt_x, est_tile_bytes)` keeping peak VRAM <= `max_vram_bytes`."""
    nt_y = 1
    nt_x = 1
    while True:
        core_y = math.ceil(ny / nt_y)
        core_x = math.ceil(nx / nt_x)
        pad_y = 0 if nt_y == 1 else (halo if nt_y == 2 else 2 * halo)
        pad_x = 0 if nt_x == 1 else (halo if nt_x == 2 else 2 * halo)
        tile_ny = min(ny, core_y + pad_y)
        tile_nx = min(nx, core_x + pad_x)
        est = estimate_vram_bytes(
            nz,
            tile_ny,
            tile_nx,
            moist=moist,
            has_lbc=has_lbc,
            has_iau=has_iau,
            has_physics=has_physics,
        )
        if est <= max_vram_bytes or ((not allow_x_split or core_x <= 16) and core_y <= 16):
            return nt_y, nt_x, est
        if allow_x_split and nt_x <= nt_y and core_x > 16:
            nt_x += 1
        elif core_y > 16:
            nt_y += 1
        else:
            return nt_y, nt_x, est


@dataclass(frozen=True)
class _TileBox:
    j0: int
    j1: int
    i0: int
    i1: int
    pj0: int
    pj1: int
    pi0: int
    pi1: int
    lbc_edges: int


def _pack_state_ptrs(state: Dict[str, np.ndarray], moist: bool) -> Tuple[NhStatePtrsC, List[np.ndarray]]:
    kept: List[np.ndarray] = []

    def ptr_of(name: str, optional: bool = False) -> ctypes.c_void_p:
        arr = state.get(name)
        if arr is None:
            if optional:
                return ctypes.c_void_p(0)
            raise KeyError(f"Missing required prognostic field '{name}'")
        c_arr = np.ascontiguousarray(arr, dtype=np.float32)
        kept.append(c_arr)
        return ctypes.c_void_p(c_arr.ctypes.data)

    ptrs = NhStatePtrsC(
        u=ptr_of("u"),
        v=ptr_of("v"),
        w=ptr_of("w"),
        theta_p=ptr_of("theta_p"),
        pi_p=ptr_of("pi_p"),
        qv=ptr_of("qv", optional=not moist),
        qc=ptr_of("qc", optional=not moist),
        qr=ptr_of("qr", optional=not moist),
        qi=ptr_of("qi", optional=not moist),
        qs=ptr_of("qs", optional=not moist),
        qg=ptr_of("qg", optional=not moist),
        tke=ptr_of("tke", optional=True),
    )
    return ptrs, kept


def _pack_struct_ptrs(
    cls: type,
    mapping: Dict[str, np.ndarray],
    field_names: Tuple[str, ...],
) -> Tuple[ctypes.Structure, List[np.ndarray]]:
    kept: List[np.ndarray] = []
    kwargs: Dict[str, ctypes.c_void_p] = {}
    for name in field_names:
        arr = mapping.get(name)
        if arr is None:
            kwargs[name] = ctypes.c_void_p(0)
        else:
            c_arr = np.ascontiguousarray(arr, dtype=np.float32)
            kept.append(c_arr)
            kwargs[name] = ctypes.c_void_p(c_arr.ctypes.data)
    return cls(**kwargs), kept


class NonHydrostaticGPUSolver:
    """
    GPU-accelerated split-explicit RK3 non-hydrostatic dynamical core with
    Phase B2 physics suite and automatic 2D horizontal halo tiling (`nz` kept intact).
    """

    SPECIES = ("qv", "qc", "qr", "qi", "qs", "qg")
    SURFACE_STATIC_FIELDS = (
        "lat_deg",
        "lon_deg",
        "lsm",
        "lake_mask",
        "soil_type",
        "bathy_depth_m",
        "z0m_base",
        "albedo_base",
        "veg_frac",
    )
    SURFACE_SOIL_FIELDS = ("soil_t", "soil_m")
    SURFACE_2D_FIELDS = (
        "t_skin",
        "snow_swe",
        "sst_bulk",
        "dt_cool",
        "dt_warm",
        "lake_tbot",
        "lake_hmld",
        "lake_ctheta",
        "lake_hice",
        "rain_acc",
        "snow_acc",
        "graupel_acc",
        "sw_down",
        "lw_down",
        "shf",
        "lhf",
        "ustar",
        "pblh",
        "t2m",
        "q2m",
        "u10m",
        "v10m",
        "sea_ice_frac",
        "sea_ice_thick",
        "sea_ice_tsfc",
        "sea_ice_albedo",
    )
    SURFACE_STATE_STRUCT_ORDER = (
        "t_skin",
        "soil_t",
        "soil_m",
        "snow_swe",
        "sst_bulk",
        "dt_cool",
        "dt_warm",
        "lake_tbot",
        "lake_hmld",
        "lake_ctheta",
        "lake_hice",
        "rain_acc",
        "snow_acc",
        "graupel_acc",
        "sw_down",
        "lw_down",
        "shf",
        "lhf",
        "ustar",
        "pblh",
        "t2m",
        "q2m",
        "u10m",
        "v10m",
        "sea_ice_frac",
        "sea_ice_thick",
        "sea_ice_tsfc",
        "sea_ice_albedo",
    )

    def __init__(
        self,
        vgrid: VerticalGrid,
        dx_m: float,
        dy_m: float,
        hs_m: np.ndarray,
        *,
        msf: Optional[np.ndarray] = None,
        f_cor: Optional[np.ndarray] = None,
        has_moist: bool = False,
        has_physics: bool = False,
        bc_x: int = 0,
        bc_y: int = 0,
        use_coriolis: bool = False,
        adv_order: int = 5,
        theta_sfc_k: float = 288.15,
        p_sfc_pa: float = 101_325.0,
        n_ref_s: float = 0.01,
        beta_offcenter: float = 0.55,
        div_damp_coeff: float = 0.04,
        km_const: float = 0.0,
        kh_const: float = 0.0,
        hyperdiff_coeff: float = 0.0,
        sponge_depth_m: float = 0.0,
        sponge_tau_max_s: float = 0.0,
        lbc_rim_width: int = 8,
        lbc_tau_max_s: float = 0.05,
        has_iau: bool = False,
        is_spherical: bool = False,
        polar_filter_lat_deg: float = 0.0,
        max_vram_bytes: int = DEFAULT_MAX_VRAM_BYTES,
        halo: int = DEFAULT_HALO_CELLS,
    ) -> None:
        self.lib = load_cuda_library()
        self.vgrid = vgrid
        self.hs_m = np.ascontiguousarray(hs_m, dtype=np.float64)
        self.ny, self.nx = self.hs_m.shape
        self.nz = vgrid.nz
        self.dx_m = float(dx_m)
        self.dy_m = float(dy_m)

        self.msf = (
            np.ones((self.ny, self.nx), dtype=np.float64)
            if msf is None
            else np.ascontiguousarray(msf, dtype=np.float64)
        )
        self.f_cor = (
            np.zeros((self.ny, self.nx), dtype=np.float64)
            if f_cor is None
            else np.ascontiguousarray(f_cor, dtype=np.float64)
        )
        if self.msf.shape != (self.ny, self.nx) or self.f_cor.shape != (self.ny, self.nx):
            raise ValueError("msf and f_cor must match hs_m shape (ny, nx)")

        self.has_moist = bool(has_moist)
        self.has_physics = bool(has_physics)
        self.bc_x = int(bc_x)
        self.bc_y = int(bc_y)
        self.use_coriolis = bool(use_coriolis)
        self.adv_order = int(adv_order)
        self.theta_sfc_k = float(theta_sfc_k)
        self.p_sfc_pa = float(p_sfc_pa)
        self.n_ref_s = float(n_ref_s)
        self.beta_offcenter = float(beta_offcenter)
        self.div_damp_coeff = float(div_damp_coeff)
        self.km_const = float(km_const)
        self.kh_const = float(kh_const)
        self.hyperdiff_coeff = float(hyperdiff_coeff)
        self.sponge_depth_m = float(sponge_depth_m)
        self.sponge_tau_max_s = float(sponge_tau_max_s)
        self.lbc_rim_width = int(lbc_rim_width)
        self.lbc_tau_max_s = float(lbc_tau_max_s)
        self.has_iau = bool(has_iau)
        self.is_spherical = bool(is_spherical)
        self.polar_filter_lat_deg = float(polar_filter_lat_deg)
        self.halo = int(halo)

        eff_max_vram = (
            min(int(max_vram_bytes), int(3.4 * (1024**3)))
            if (self.is_spherical and self.bc_x == 1 and int(max_vram_bytes) == DEFAULT_MAX_VRAM_BYTES)
            else int(max_vram_bytes)
        )
        try:
            free_b, _ = cuda_vram_info()
            eff_max_vram = min(eff_max_vram, max(512 * 1024 * 1024, int(free_b * 0.85)))
        except Exception:
            pass
        self.nt_y, self.nt_x, self.estimated_tile_vram_bytes = plan_horizontal_tiling(
            self.nz,
            self.ny,
            self.nx,
            moist=self.has_moist,
            has_lbc=(self.bc_x == 2 or self.bc_y == 2),
            has_iau=self.has_iau,
            has_physics=self.has_physics,
            max_vram_bytes=eff_max_vram,
            halo=self.halo,
            allow_x_split=(self.bc_x != 1),
        )
        if (self.nt_y > 1 and self.bc_y == 1) or (self.nt_x > 1 and self.bc_x == 1):
            raise ValueError(
                "2D halo tiling only supports non-periodic horizontal boundaries (bc=0 or bc=2)."
            )

        self._handle: Optional[int] = None
        self._tile_handles: List[int] = []
        self._shared_tile_handle: Optional[int] = None
        self._shape_tile_handles: Dict[Tuple[int, int], int] = {}
        self._tile_phys_caches: List[Dict[str, Any]] = []
        self._phys_accum_steps: int = 0
        self._state_host: Optional[Dict[str, np.ndarray]] = None
        self._lbc_host: Optional[Dict[str, np.ndarray]] = None
        self._iau_host: Optional[Dict[str, np.ndarray]] = None
        self._pcfg: Optional[NhPhysicsConfigC] = None
        self._surface_static_host: Optional[Dict[str, np.ndarray]] = None
        self._surface_state_host: Optional[Dict[str, np.ndarray]] = None
        self._tiles = self._build_tiles()

        if len(self._tiles) == 1:
            self._handle = self._create_tile_handle(self._tiles[0])
        else:
            total_tiles_vram = len(self._tiles) * self.estimated_tile_vram_bytes
            try:
                free_b, _ = cuda_vram_info()
            except Exception:
                free_b = DEFAULT_MAX_VRAM_BYTES
            resident_budget = min(DEFAULT_MAX_VRAM_BYTES, int(free_b * 0.85))
            if total_tiles_vram <= resident_budget:
                # All tiles fit in GPU VRAM simultaneously: keep every tile permanently on GPU!
                for t in self._tiles:
                    self._tile_handles.append(self._create_tile_handle(t))
            else:
                # Out-of-core tiling (e.g. 720x720x137 or 721x1440x137 on 6 GiB GPU):
                # allocate ONE persistent reusable handle per distinct tile shape (or fallback per-tile if > 1 shape doesn't fit)
                unique_shapes: List[Tuple[int, _TileBox]] = []
                seen_shapes: set[Tuple[int, int]] = set()
                for idx_t, t in enumerate(self._tiles):
                    sh = (t.pj1 - t.pj0, t.pi1 - t.pi0)
                    if sh not in seen_shapes:
                        seen_shapes.add(sh)
                        unique_shapes.append((idx_t, t))
                if len(unique_shapes) == 1 or len(unique_shapes) * self.estimated_tile_vram_bytes <= resident_budget:
                    for _idx_t, t in unique_shapes:
                        sh = (t.pj1 - t.pj0, t.pi1 - t.pi0)
                        self._shape_tile_handles[sh] = self._create_tile_handle(t)
                    if len(self._shape_tile_handles) == 1:
                        self._shared_tile_handle = next(iter(self._shape_tile_handles.values()))
                for t in self._tiles:
                    tny, tnx = t.pj1 - t.pj0, t.pi1 - t.pi0
                    self._tile_phys_caches.append(
                        {
                            "rad_dth_dt": np.zeros((self.nz, tny, tnx), dtype=np.float32),
                            "precip_liq_rate": np.zeros((tny, tnx), dtype=np.float32),
                            "precip_ice_rate": np.zeros((tny, tnx), dtype=np.float32),
                            "rad_init": ctypes.c_int32(0),
                            "rad_timer": ctypes.c_double(0.0),
                        }
                    )
        if self.has_physics:
            self.set_physics_config()

    @property
    def num_tiles(self) -> int:
        return len(self._tiles)

    @staticmethod
    def _uniform_padded_splits(n: int, nt: int, halo: int) -> List[Tuple[int, int, int, int]]:
        if nt <= 1:
            return [(0, n, 0, n)]
        if n > 2 * halo + nt and nt >= 2:
            rem = n - 2 * halo
            base = rem // nt
            extra = rem % nt
            widths = [
                base + (1 if t < extra else 0) + (halo if (t == 0 or t == nt - 1) else 0)
                for t in range(nt)
            ]
            splits = [0]
            for w in widths:
                splits.append(splits[-1] + w)
            p_target = base + 2 * halo + (1 if extra > 0 else 0)
        else:
            splits = [int(v) for v in np.linspace(0, n, nt + 1, dtype=int)]
            p_target = max(
                min(n, int(splits[t + 1]) + halo) - max(0, int(splits[t]) - halo)
                for t in range(nt)
            )

        out: List[Tuple[int, int, int, int]] = []
        for t in range(nt):
            j0, j1 = int(splits[t]), int(splits[t + 1])
            pj0 = max(0, j0 - halo)
            pj1 = min(n, j1 + halo)
            if (pj1 - pj0) == p_target - 1:
                if pj1 + 1 < n:
                    pj1 += 1
                elif pj0 - 1 > 0:
                    pj0 -= 1
            out.append((j0, j1, pj0, pj1))
        return out

    def _build_tiles(self) -> List[_TileBox]:
        if self.nt_y == 1 and self.nt_x == 1:
            edges = (1 if self.bc_x == 2 else 0) | (2 if self.bc_x == 2 else 0) | (
                4 if self.bc_y == 2 else 0
            ) | (8 if self.bc_y == 2 else 0)
            return [_TileBox(0, self.ny, 0, self.nx, 0, self.ny, 0, self.nx, edges)]

        y_boxes = self._uniform_padded_splits(self.ny, self.nt_y, self.halo)
        x_boxes = self._uniform_padded_splits(self.nx, self.nt_x, self.halo)
        tiles: List[_TileBox] = []
        for j0, j1, pj0, pj1 in y_boxes:
            for i0, i1, pi0, pi1 in x_boxes:
                edges = 0
                if self.bc_x == 2 and pi0 == 0:
                    edges |= 1
                if self.bc_x == 2 and pi1 == self.nx:
                    edges |= 2
                if self.bc_y == 2 and pj0 == 0:
                    edges |= 4
                if self.bc_y == 2 and pj1 == self.ny:
                    edges |= 8
                tiles.append(_TileBox(j0, j1, i0, i1, pj0, pj1, pi0, pi1, edges))
        return tiles

    def _make_cfg(self, tile: _TileBox) -> NhGridConfigC:
        return NhGridConfigC(
            nz=self.nz,
            ny=tile.pj1 - tile.pj0,
            nx=tile.pi1 - tile.pi0,
            has_moist=1 if self.has_moist else 0,
            bc_x=self.bc_x,
            bc_y=self.bc_y,
            lbc_edges=tile.lbc_edges,
            use_coriolis=1 if self.use_coriolis else 0,
            adv_order=self.adv_order,
            lbc_rim_width=self.lbc_rim_width,
            has_iau=1 if self.has_iau else 0,
            dx_m=self.dx_m,
            dy_m=self.dy_m,
            z_top_m=float(self.vgrid.z_top_m),
            theta_sfc_k=self.theta_sfc_k,
            p_sfc_pa=self.p_sfc_pa,
            n_ref_s=self.n_ref_s,
            beta_offcenter=self.beta_offcenter,
            div_damp_coeff=self.div_damp_coeff,
            km_const=self.km_const,
            kh_const=self.kh_const,
            hyperdiff_coeff=self.hyperdiff_coeff,
            sponge_depth_m=self.sponge_depth_m,
            sponge_tau_max_s=self.sponge_tau_max_s,
            lbc_tau_max_s=self.lbc_tau_max_s,
            is_spherical=1 if self.is_spherical else 0,
            polar_filter_lat_deg=self.polar_filter_lat_deg,
        )

    def _create_tile_handle(self, tile: _TileBox) -> int:
        cfg = self._make_cfg(tile)
        hs_sub = np.ascontiguousarray(self.hs_m[tile.pj0 : tile.pj1, tile.pi0 : tile.pi1], dtype=np.float64)
        msf_sub = np.ascontiguousarray(self.msf[tile.pj0 : tile.pj1, tile.pi0 : tile.pi1], dtype=np.float64)
        fcor_sub = np.ascontiguousarray(self.f_cor[tile.pj0 : tile.pj1, tile.pi0 : tile.pi1], dtype=np.float64)
        vg = self.vgrid
        zeta_w = np.ascontiguousarray(vg.zeta_w, dtype=np.float64)
        zeta_s = np.ascontiguousarray(vg.zeta_s, dtype=np.float64)
        dzeta = np.ascontiguousarray(vg.dzeta, dtype=np.float64)
        b_w = np.ascontiguousarray(vg.B_w, dtype=np.float64)
        b_s = np.ascontiguousarray(vg.B_s, dtype=np.float64)
        db_w = np.ascontiguousarray(vg.dB_dzeta_w, dtype=np.float64)
        db_s = np.ascontiguousarray(vg.dB_dzeta_s, dtype=np.float64)

        handle = self.lib.nh_cuda_create(
            ctypes.byref(cfg),
            zeta_w.ctypes.data,
            zeta_s.ctypes.data,
            dzeta.ctypes.data,
            b_w.ctypes.data,
            b_s.ctypes.data,
            db_w.ctypes.data,
            db_s.ctypes.data,
            hs_sub.ctypes.data,
            msf_sub.ctypes.data,
            fcor_sub.ctypes.data,
        )
        if not handle:
            err = self.lib.nh_cuda_last_error()
            raise RuntimeError(err.decode("utf-8", errors="replace") if err else "nh_cuda_create failed")
        return int(handle)

    def _raise_if_err(self, rc: int, msg: str) -> None:
        if rc != 0:
            err = self.lib.nh_cuda_last_error()
            detail = err.decode("utf-8", errors="replace") if err else msg
            raise RuntimeError(f"{msg}: {detail}")

    def empty_state(self) -> Dict[str, np.ndarray]:
        """Allocate zeroed host float32 state dict with exact C-grid shapes."""
        state = {
            "u": np.zeros((self.nz, self.ny, self.nx + 1), dtype=np.float32),
            "v": np.zeros((self.nz, self.ny + 1, self.nx), dtype=np.float32),
            "w": np.zeros((self.nz + 1, self.ny, self.nx), dtype=np.float32),
            "theta_p": np.zeros((self.nz, self.ny, self.nx), dtype=np.float32),
            "pi_p": np.zeros((self.nz, self.ny, self.nx), dtype=np.float32),
        }
        if self.has_moist:
            for sp in self.SPECIES:
                state[sp] = np.zeros((self.nz, self.ny, self.nx), dtype=np.float32)
        if self.has_physics:
            state["tke"] = np.full((self.nz, self.ny, self.nx), 0.01, dtype=np.float32)
        return state

    def _slice_tile_state(
        self, src: Dict[str, np.ndarray], tile: _TileBox, *, moist: Optional[bool] = None, include_qv: bool = False
    ) -> Dict[str, np.ndarray]:
        use_moist = self.has_moist if moist is None else bool(moist)
        out = {
            "u": np.ascontiguousarray(src["u"][:, tile.pj0 : tile.pj1, tile.pi0 : tile.pi1 + 1], dtype=np.float32),
            "v": np.ascontiguousarray(src["v"][:, tile.pj0 : tile.pj1 + 1, tile.pi0 : tile.pi1], dtype=np.float32),
            "w": np.ascontiguousarray(src["w"][:, tile.pj0 : tile.pj1, tile.pi0 : tile.pi1], dtype=np.float32),
            "theta_p": np.ascontiguousarray(src["theta_p"][:, tile.pj0 : tile.pj1, tile.pi0 : tile.pi1], dtype=np.float32),
            "pi_p": np.ascontiguousarray(src["pi_p"][:, tile.pj0 : tile.pj1, tile.pi0 : tile.pi1], dtype=np.float32),
        }
        if use_moist:
            for sp in self.SPECIES:
                out[sp] = np.ascontiguousarray(src[sp][:, tile.pj0 : tile.pj1, tile.pi0 : tile.pi1], dtype=np.float32)
        elif include_qv and "qv" in src:
            out["qv"] = np.ascontiguousarray(src["qv"][:, tile.pj0 : tile.pj1, tile.pi0 : tile.pi1], dtype=np.float32)
        if moist is None and "tke" in src:
            out["tke"] = np.ascontiguousarray(src["tke"][:, tile.pj0 : tile.pj1, tile.pi0 : tile.pi1], dtype=np.float32)
        return out

    def _write_tile_interior(self, dst: Dict[str, np.ndarray], tile_state: Dict[str, np.ndarray], tile: _TileBox) -> None:
        lj0, lj1 = tile.j0 - tile.pj0, tile.j1 - tile.pj0
        li0, li1 = tile.i0 - tile.pi0, tile.i1 - tile.pi0
        dst["w"][:, tile.j0 : tile.j1, tile.i0 : tile.i1] = tile_state["w"][:, lj0:lj1, li0:li1]
        dst["theta_p"][:, tile.j0 : tile.j1, tile.i0 : tile.i1] = tile_state["theta_p"][:, lj0:lj1, li0:li1]
        dst["pi_p"][:, tile.j0 : tile.j1, tile.i0 : tile.i1] = tile_state["pi_p"][:, lj0:lj1, li0:li1]
        if self.has_moist:
            for sp in self.SPECIES:
                dst[sp][:, tile.j0 : tile.j1, tile.i0 : tile.i1] = tile_state[sp][:, lj0:lj1, li0:li1]
        if "tke" in dst and "tke" in tile_state:
            dst["tke"][:, tile.j0 : tile.j1, tile.i0 : tile.i1] = tile_state["tke"][:, lj0:lj1, li0:li1]

        u_end = tile.i1 + (1 if tile.i1 == self.nx else 0)
        lu_end = li1 + (1 if tile.i1 == self.nx else 0)
        dst["u"][:, tile.j0 : tile.j1, tile.i0 : u_end] = tile_state["u"][:, lj0:lj1, li0:lu_end]

        v_end = tile.j1 + (1 if tile.j1 == self.ny else 0)
        lv_end = lj1 + (1 if tile.j1 == self.ny else 0)
        dst["v"][:, tile.j0 : v_end, tile.i0 : tile.i1] = tile_state["v"][:, lj0:lv_end, li0:li1]

    def upload_state(self, state: Dict[str, np.ndarray], *, copy: bool = True) -> None:
        if len(self._tiles) == 1:
            ptrs, kept = _pack_state_ptrs(state, self.has_moist)
            self._raise_if_err(self.lib.nh_cuda_upload_state(self._handle, ctypes.byref(ptrs)), "upload_state")
            del kept
        else:
            if copy:
                self._state_host = {k: np.ascontiguousarray(v, dtype=np.float32).copy() for k, v in state.items()}
            else:
                self._state_host = {
                    k: v if (isinstance(v, np.ndarray) and v.dtype == np.float32 and v.flags.c_contiguous)
                    else np.ascontiguousarray(v, dtype=np.float32)
                    for k, v in state.items()
                }
            if self.has_physics and "tke" not in self._state_host:
                self._state_host["tke"] = np.full((self.nz, self.ny, self.nx), 0.01, dtype=np.float32)
            if self._tile_handles:
                for h, tile in zip(self._tile_handles, self._tiles):
                    t_state = self._slice_tile_state(self._state_host, tile)
                    ptrs, kept = _pack_state_ptrs(t_state, self.has_moist)
                    self._raise_if_err(self.lib.nh_cuda_upload_state(h, ctypes.byref(ptrs)), "tile upload_state")
                    del kept

    def set_lbc_target(self, target: Dict[str, np.ndarray]) -> None:
        if len(self._tiles) == 1:
            ptrs, kept = _pack_state_ptrs(target, False)
            self._raise_if_err(self.lib.nh_cuda_set_lbc_target(self._handle, ctypes.byref(ptrs)), "set_lbc_target")
            del kept
        else:
            self._lbc_host = {k: np.ascontiguousarray(v, dtype=np.float32).copy() for k, v in target.items()}
            if self._tile_handles and (self.bc_x == 2 or self.bc_y == 2):
                has_qv = "qv" in self._lbc_host
                for h, tile in zip(self._tile_handles, self._tiles):
                    lbc_slice = self._slice_tile_state(self._lbc_host, tile, moist=False, include_qv=has_qv)
                    ptrs, kept = _pack_state_ptrs(lbc_slice, False)
                    self._raise_if_err(self.lib.nh_cuda_set_lbc_target(h, ctypes.byref(ptrs)), "tile set_lbc_target")
                    del kept

    def set_iau_forcing(self, ddt_state: Dict[str, np.ndarray]) -> None:
        if len(self._tiles) == 1:
            ptrs, kept = _pack_state_ptrs(ddt_state, False)
            self._raise_if_err(self.lib.nh_cuda_set_iau_forcing(self._handle, ctypes.byref(ptrs)), "set_iau_forcing")
            del kept
        else:
            self._iau_host = {k: np.ascontiguousarray(v, dtype=np.float32).copy() for k, v in ddt_state.items()}
            if self._tile_handles and self.has_iau:
                for h, tile in zip(self._tile_handles, self._tiles):
                    iau_slice = self._slice_tile_state(self._iau_host, tile, moist=False)
                    ptrs, kept = _pack_state_ptrs(iau_slice, False)
                    self._raise_if_err(self.lib.nh_cuda_set_iau_forcing(h, ctypes.byref(ptrs)), "tile set_iau_forcing")
                    del kept

    def clear_iau_forcing(self) -> None:
        """Disable IAU and free GPU/host IAU buffers once the IAU warm-up window ends."""
        self._iau_host = None
        self.has_iau = False
        if getattr(self, "_handle", None):
            self._raise_if_err(self.lib.nh_cuda_clear_iau_forcing(self._handle), "clear_iau_forcing")
        if getattr(self, "_tile_handles", None):
            for h in self._tile_handles:
                if h:
                    self._raise_if_err(self.lib.nh_cuda_clear_iau_forcing(h), "tile clear_iau_forcing")
        if getattr(self, "_shape_tile_handles", None):
            cleared: set[int] = set()
            for h in self._shape_tile_handles.values():
                if h and h not in cleared:
                    self._raise_if_err(self.lib.nh_cuda_clear_iau_forcing(h), "shared clear_iau_forcing")
                    cleared.add(h)

    def default_surface_static(
        self,
        *,
        lat_deg: float = 52.1,
        lon_deg: float = 5.18,
        lsm: float = 1.0,
        lake_mask: float = 0.0,
        soil_type: float = 9.0,
        bathy_depth_m: float = 4.5,
        z0m_base: float = 0.05,
        albedo_base: float = 0.20,
        veg_frac: float = 0.60,
    ) -> Dict[str, np.ndarray]:
        """Allocate default float32 2D surface static fields `(ny, nx)`."""
        shape = (self.ny, self.nx)
        return {
            "lat_deg": np.full(shape, lat_deg, dtype=np.float32),
            "lon_deg": np.full(shape, lon_deg, dtype=np.float32),
            "lsm": np.full(shape, lsm, dtype=np.float32),
            "lake_mask": np.full(shape, lake_mask, dtype=np.float32),
            "soil_type": np.full(shape, soil_type, dtype=np.float32),
            "bathy_depth_m": np.full(shape, bathy_depth_m, dtype=np.float32),
            "z0m_base": np.full(shape, z0m_base, dtype=np.float32),
            "albedo_base": np.full(shape, albedo_base, dtype=np.float32),
            "veg_frac": np.full(shape, veg_frac, dtype=np.float32),
        }

    def empty_surface_state(self) -> Dict[str, np.ndarray]:
        """Allocate physical float32 default 2D + 4-layer soil surface state dict."""
        s2 = (self.ny, self.nx)
        s4 = (4, self.ny, self.nx)
        out: Dict[str, np.ndarray] = {k: np.zeros(s2, dtype=np.float32) for k in self.SURFACE_2D_FIELDS}
        out["soil_t"] = np.full(s4, 285.0, dtype=np.float32)
        out["soil_m"] = np.full(s4, 0.28, dtype=np.float32)
        out["t_skin"].fill(286.0)
        out["sst_bulk"].fill(284.0)
        out["lake_tbot"].fill(281.0)
        out["lake_hmld"].fill(2.5)
        out["lake_ctheta"].fill(0.65)
        return out

    def _slice_tile_surface(self, src: Dict[str, np.ndarray], tile: _TileBox) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for k, v in src.items():
            if v.ndim == 3:
                out[k] = np.ascontiguousarray(v[:, tile.pj0 : tile.pj1, tile.pi0 : tile.pi1], dtype=np.float32)
            else:
                out[k] = np.ascontiguousarray(v[tile.pj0 : tile.pj1, tile.pi0 : tile.pi1], dtype=np.float32)
        return out

    def _write_tile_surface_interior(self, dst: Dict[str, np.ndarray], src: Dict[str, np.ndarray], tile: _TileBox) -> None:
        lj0, lj1 = tile.j0 - tile.pj0, tile.j1 - tile.pj0
        li0, li1 = tile.i0 - tile.pi0, tile.i1 - tile.pi0
        for k, v in src.items():
            if v.ndim == 3:
                dst[k][:, tile.j0 : tile.j1, tile.i0 : tile.i1] = v[:, lj0:lj1, li0:li1]
            else:
                dst[k][tile.j0 : tile.j1, tile.i0 : tile.i1] = v[lj0:lj1, li0:li1]

    def set_physics_config(
        self,
        config: Optional[NhPhysicsConfigC] = None,
        *,
        enable_microphysics: int = 1,
        enable_pbl: int = 1,
        enable_radiation: int = 1,
        enable_surface: int = 1,
        rad_interval_s: float = 900.0,
        cloud_num_land: float = 200.0e6,
        cloud_num_sea: float = 80.0e6,
        julian_day: float = 172.0,
        utc_seconds: float = 43200.0,
        co2_ppm: float = 425.0,
        pbl_last: int = 0,
        cy46_physics: int = 0,
    ) -> None:
        if config is None:
            config = NhPhysicsConfigC(
                enable_microphysics=int(enable_microphysics),
                enable_pbl=int(enable_pbl),
                enable_radiation=int(enable_radiation),
                enable_surface=int(enable_surface),
                rad_interval_s=float(rad_interval_s),
                cloud_num_land=float(cloud_num_land),
                cloud_num_sea=float(cloud_num_sea),
                julian_day=float(julian_day),
                utc_seconds=float(utc_seconds),
                co2_ppm=float(co2_ppm),
                pbl_last=int(pbl_last),
                cy46_physics=int(cy46_physics),
            )
        self._pcfg = config
        if len(self._tiles) == 1:
            self._raise_if_err(
                self.lib.nh_cuda_set_physics_config(self._handle, ctypes.byref(self._pcfg)),
                "set_physics_config",
            )
        elif self._tile_handles:
            for h in self._tile_handles:
                pcfg_copy = NhPhysicsConfigC.from_buffer_copy(self._pcfg)
                self._raise_if_err(
                    self.lib.nh_cuda_set_physics_config(h, ctypes.byref(pcfg_copy)),
                    "tile set_physics_config",
                )

    def set_surface_static(self, sstatic: Dict[str, np.ndarray]) -> None:
        if len(self._tiles) == 1:
            ptrs, kept = _pack_struct_ptrs(NhSurfaceStaticPtrsC, sstatic, self.SURFACE_STATIC_FIELDS)
            self._raise_if_err(self.lib.nh_cuda_set_surface_static(self._handle, ctypes.byref(ptrs)), "set_surface_static")
            del kept
        else:
            self._surface_static_host = {
                k: np.ascontiguousarray(v, dtype=np.float32).copy() for k, v in sstatic.items() if k in self.SURFACE_STATIC_FIELDS
            }
            if self._tile_handles:
                for h, tile in zip(self._tile_handles, self._tiles):
                    ss_slice = self._slice_tile_surface(self._surface_static_host, tile)
                    ptrs, kept = _pack_struct_ptrs(NhSurfaceStaticPtrsC, ss_slice, self.SURFACE_STATIC_FIELDS)
                    self._raise_if_err(self.lib.nh_cuda_set_surface_static(h, ctypes.byref(ptrs)), "tile set_surface_static")
                    del kept

    def upload_surface(self, surf: Dict[str, np.ndarray]) -> None:
        if len(self._tiles) == 1:
            ptrs, kept = _pack_struct_ptrs(NhSurfaceStatePtrsC, surf, self.SURFACE_STATE_STRUCT_ORDER)
            self._raise_if_err(self.lib.nh_cuda_upload_surface(self._handle, ctypes.byref(ptrs)), "upload_surface")
            del kept
        else:
            if self._surface_state_host is None:
                self._surface_state_host = self.empty_surface_state()
            for k, v in surf.items():
                if k in self.SURFACE_STATE_STRUCT_ORDER:
                    self._surface_state_host[k] = np.ascontiguousarray(v, dtype=np.float32).copy()
            if self._tile_handles:
                for h, tile in zip(self._tile_handles, self._tiles):
                    sf_slice = self._slice_tile_surface(self._surface_state_host, tile)
                    ptrs, kept = _pack_struct_ptrs(NhSurfaceStatePtrsC, sf_slice, self.SURFACE_STATE_STRUCT_ORDER)
                    self._raise_if_err(self.lib.nh_cuda_upload_surface(h, ctypes.byref(ptrs)), "tile upload_surface")
                    del kept

    def download_surface(self) -> Dict[str, np.ndarray]:
        if len(self._tiles) == 1:
            out = self.empty_surface_state()
            ptrs, kept = _pack_struct_ptrs(NhSurfaceStatePtrsC, out, self.SURFACE_STATE_STRUCT_ORDER)
            self._raise_if_err(self.lib.nh_cuda_download_surface(self._handle, ctypes.byref(ptrs)), "download_surface")
            del kept
            return out
        if self._tile_handles:
            out = self.empty_surface_state()
            for h, tile in zip(self._tile_handles, self._tiles):
                sf_slice = self._slice_tile_surface(out, tile)
                ptrs, kept = _pack_struct_ptrs(NhSurfaceStatePtrsC, sf_slice, self.SURFACE_STATE_STRUCT_ORDER)
                self._raise_if_err(self.lib.nh_cuda_download_surface(h, ctypes.byref(ptrs)), "tile download_surface")
                del kept
                self._write_tile_surface_interior(out, sf_slice, tile)
            self._surface_state_host = {k: v.copy() for k, v in out.items()}
            return out
        if self._surface_state_host is None:
            return self.empty_surface_state()
        return {k: v.copy() for k, v in self._surface_state_host.items()}

    def _advance_host_clock(self, dt_s: float) -> None:
        if self._pcfg is None:
            return
        self._pcfg.utc_seconds += float(dt_s)
        while self._pcfg.utc_seconds >= 86400.0:
            self._pcfg.utc_seconds -= 86400.0
            self._pcfg.julian_day += 1.0
            if self._pcfg.julian_day > 366.0:
                self._pcfg.julian_day -= 365.0

    def _estimate_host_cs_u32(self) -> int:
        """Compute exact IEEE-754 uint32 bit pattern of domain-max `cs * msf` in 64-row chunks (zero 3D float64 cache)."""
        assert self._state_host is not None
        zeta_s = self.vgrid.zeta_s[:, None, None].astype(np.float64)
        b_s = self.vgrid.B_s[:, None, None].astype(np.float64)
        n2 = float(self.n_ref_s) * float(self.n_ref_s)
        g = 9.80665
        cp = 1004.5
        rd = 287.058
        kappa = rd / cp
        gamma_rd = (cp / (cp - rd)) * rd
        pi_sfc = (float(self.p_sfc_pa) / 100_000.0) ** kappa
        g2_cp = (g * g) / (cp * float(self.theta_sfc_k) * max(n2, 1e-12))

        msf_eff = self.msf
        if self.is_spherical and 0.0 < self.polar_filter_lat_deg < 89.9:
            cos_crit = max(math.cos(math.radians(self.polar_filter_lat_deg)), 1e-3)
            msf_eff = np.minimum(self.msf, 1.0 / cos_crit)

        cs_max_f32 = np.float32(0.0)
        chunk_ny = 64
        th_p_all = self._state_host["theta_p"]
        pi_p_all = self._state_host["pi_p"]
        for j0 in range(0, self.ny, chunk_ny):
            j1 = min(self.ny, j0 + chunk_ny)
            z_c = zeta_s + b_s * self.hs_m[None, j0:j1, :]
            exp_f = np.exp((n2 / g) * z_c)
            th_tot = float(self.theta_sfc_k) * exp_f + th_p_all[:, j0:j1, :].astype(np.float64)
            pi_tot = np.maximum(1e-4, pi_sfc - g2_cp * (1.0 - 1.0 / exp_f) + pi_p_all[:, j0:j1, :].astype(np.float64))
            t_tot = np.maximum(50.0, th_tot * pi_tot)
            cs_c = (np.sqrt(gamma_rd * t_tot) * msf_eff[None, j0:j1, :]).astype(np.float32)
            m_c = np.float32(np.max(cs_c))
            if m_c > cs_max_f32:
                cs_max_f32 = m_c
        return int(cs_max_f32.view(np.uint32))

    def _run_multi_tile(
        self,
        dt_s: float,
        n_steps: int,
        n_substeps: int,
        run_dyn: bool,
        run_phys: bool,
        phys_dt_s: Optional[float] = None,
        on_step: Optional[Callable[[int, bool], None]] = None,
        on_tile: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        if self._state_host is None:
            raise RuntimeError("upload_state must be called before stepping multi-tile solver")
        if run_phys:
            if self._pcfg is None:
                self.set_physics_config()
            if self._surface_state_host is None:
                self._surface_state_host = self.empty_surface_state()
                if self._tile_handles:
                    self.upload_surface(self._surface_state_host)

        dt_p = float(dt_s if phys_dt_s is None else phys_dt_s)
        n_tiles = len(self._tiles)

        # Path A: All tile handles permanently resident in GPU VRAM
        if self._tile_handles:
            for step_i in range(n_steps):
                eff_substeps = int(n_substeps)
                if run_dyn and eff_substeps <= 0:
                    cs_max_u32 = 0
                    for h in self._tile_handles:
                        val = ctypes.c_uint32(0)
                        self._raise_if_err(self.lib.nh_cuda_estimate_cs_u32(h, ctypes.byref(val)), "tile estimate_cs_u32")
                        if int(val.value) > cs_max_u32:
                            cs_max_u32 = int(val.value)
                    if cs_max_u32 > 0:
                        eff_substeps = -cs_max_u32

                next_state = {k: np.empty_like(v) for k, v in self._state_host.items()}
                for t_idx, (h, tile) in enumerate(zip(self._tile_handles, self._tiles)):
                    if run_dyn:
                        self._raise_if_err(
                            self.lib.nh_cuda_step_rk3(h, float(dt_s), eff_substeps),
                            "tile step_rk3",
                        )
                    if run_phys:
                        self._raise_if_err(
                            self.lib.nh_cuda_step_physics(h, dt_p),
                            "tile step_physics",
                        )
                    t_state = self._slice_tile_state(self._state_host, tile)
                    dst_ptrs, kept_d = _pack_state_ptrs(t_state, self.has_moist)
                    self._raise_if_err(self.lib.nh_cuda_download_state(h, ctypes.byref(dst_ptrs)), "tile download")
                    del kept_d
                    self._write_tile_interior(next_state, t_state, tile)
                    if on_tile is not None:
                        on_tile(t_idx + 1, n_tiles)

                self._state_host = next_state
                for h, tile in zip(self._tile_handles, self._tiles):
                    t_state = self._slice_tile_state(self._state_host, tile)
                    ptrs, kept = _pack_state_ptrs(t_state, self.has_moist)
                    self._raise_if_err(self.lib.nh_cuda_upload_state(h, ctypes.byref(ptrs)), "tile halo sync")
                    del kept

                if run_phys:
                    self._advance_host_clock(dt_p)
                if on_step is not None:
                    on_step(step_i + 1, run_phys)
            return

        # Path B: Out-of-core tiling (reusing persistent shape handles with 0 cudaMalloc/cudaFree)
        inplace_1d = self.nt_x == 1 and all(
            self._tiles[idx + 1].pj0 >= self._tiles[idx].j0 for idx in range(len(self._tiles) - 1)
        )
        for step_i in range(n_steps):
            eff_substeps = int(n_substeps)
            if run_dyn and eff_substeps <= 0:
                cs_u32 = self._estimate_host_cs_u32()
                if cs_u32 > 0:
                    eff_substeps = -cs_u32

            next_state = self._state_host if inplace_1d else {k: np.empty_like(v) for k, v in self._state_host.items()}
            next_surf = (
                {k: np.empty_like(v) for k, v in self._surface_state_host.items()}
                if run_phys and self._surface_state_host is not None
                else None
            )
            lookahead_t_state = (
                self._slice_tile_state(self._state_host, self._tiles[0])
                if inplace_1d and len(self._tiles) > 0
                else None
            )
            for t_idx, tile in enumerate(self._tiles):
                t_shape = (tile.pj1 - tile.pj0, tile.pi1 - tile.pi0)
                sh_h = self._shape_tile_handles.get(t_shape, self._shared_tile_handle)
                reused = sh_h is not None
                if reused:
                    h = int(sh_h)
                    cfg = self._make_cfg(tile)
                    hs_sub = np.ascontiguousarray(self.hs_m[tile.pj0 : tile.pj1, tile.pi0 : tile.pi1], dtype=np.float64)
                    msf_sub = np.ascontiguousarray(self.msf[tile.pj0 : tile.pj1, tile.pi0 : tile.pi1], dtype=np.float64)
                    fcor_sub = np.ascontiguousarray(self.f_cor[tile.pj0 : tile.pj1, tile.pi0 : tile.pi1], dtype=np.float64)
                    self._raise_if_err(
                        self.lib.nh_cuda_rebind_tile_grid(
                            h,
                            ctypes.byref(cfg),
                            hs_sub.ctypes.data,
                            msf_sub.ctypes.data,
                            fcor_sub.ctypes.data,
                        ),
                        "tile rebind_grid",
                    )
                else:
                    h = self._create_tile_handle(tile)
                try:
                    t_state = (
                        lookahead_t_state
                        if (inplace_1d and lookahead_t_state is not None)
                        else self._slice_tile_state(self._state_host, tile)
                    )
                    ptrs, kept = _pack_state_ptrs(t_state, self.has_moist)
                    self._raise_if_err(self.lib.nh_cuda_upload_state(h, ctypes.byref(ptrs)), "tile upload")
                    del kept

                    if self._lbc_host is not None and (self.bc_x == 2 or self.bc_y == 2):
                        lbc_slice = self._slice_tile_state(
                            self._lbc_host, tile, moist=False, include_qv=("qv" in self._lbc_host)
                        )
                        lbc_ptrs, kept_l = _pack_state_ptrs(lbc_slice, False)
                        self._raise_if_err(self.lib.nh_cuda_set_lbc_target(h, ctypes.byref(lbc_ptrs)), "tile lbc")
                        del kept_l

                    if self._iau_host is not None and self.has_iau:
                        iau_slice = self._slice_tile_state(self._iau_host, tile, moist=False)
                        iau_ptrs, kept_i = _pack_state_ptrs(iau_slice, False)
                        self._raise_if_err(self.lib.nh_cuda_set_iau_forcing(h, ctypes.byref(iau_ptrs)), "tile iau")
                        del kept_i

                    if run_phys:
                        pcfg_copy = NhPhysicsConfigC.from_buffer_copy(self._pcfg)
                        self._raise_if_err(
                            self.lib.nh_cuda_set_physics_config(h, ctypes.byref(pcfg_copy)),
                            "tile set_physics_config",
                        )
                        if self._surface_static_host is not None:
                            ss_slice = self._slice_tile_surface(self._surface_static_host, tile)
                            ss_ptrs, kept_ss = _pack_struct_ptrs(NhSurfaceStaticPtrsC, ss_slice, self.SURFACE_STATIC_FIELDS)
                            self._raise_if_err(self.lib.nh_cuda_set_surface_static(h, ctypes.byref(ss_ptrs)), "tile set_surface_static")
                            del kept_ss
                        sf_slice = self._slice_tile_surface(self._surface_state_host, tile)
                        sf_ptrs, kept_sf = _pack_struct_ptrs(NhSurfaceStatePtrsC, sf_slice, self.SURFACE_STATE_STRUCT_ORDER)
                        self._raise_if_err(self.lib.nh_cuda_upload_surface(h, ctypes.byref(sf_ptrs)), "tile upload_surface")
                        del kept_sf
                        pcache = self._tile_phys_caches[t_idx]
                        self._raise_if_err(
                            self.lib.nh_cuda_sync_physics_cache(
                                h,
                                pcache["rad_dth_dt"].ctypes.data,
                                pcache["precip_liq_rate"].ctypes.data,
                                pcache["precip_ice_rate"].ctypes.data,
                                ctypes.byref(pcache["rad_init"]),
                                ctypes.byref(pcache["rad_timer"]),
                                1,
                            ),
                            "tile upload_phys_cache",
                        )

                    if run_dyn:
                        self._raise_if_err(
                            self.lib.nh_cuda_step_rk3(h, float(dt_s), eff_substeps),
                            "tile step_rk3",
                        )
                    if run_phys:
                        self._raise_if_err(
                            self.lib.nh_cuda_step_physics(h, dt_p),
                            "tile step_physics",
                        )
                        pcache = self._tile_phys_caches[t_idx]
                        self._raise_if_err(
                            self.lib.nh_cuda_sync_physics_cache(
                                h,
                                pcache["rad_dth_dt"].ctypes.data,
                                pcache["precip_liq_rate"].ctypes.data,
                                pcache["precip_ice_rate"].ctypes.data,
                                ctypes.byref(pcache["rad_init"]),
                                ctypes.byref(pcache["rad_timer"]),
                                0,
                            ),
                            "tile download_phys_cache",
                        )
                        sf_d_ptrs, kept_sfd = _pack_struct_ptrs(NhSurfaceStatePtrsC, sf_slice, self.SURFACE_STATE_STRUCT_ORDER)
                        self._raise_if_err(self.lib.nh_cuda_download_surface(h, ctypes.byref(sf_d_ptrs)), "tile download_surface")
                        del kept_sfd
                        self._write_tile_surface_interior(next_surf, sf_slice, tile)

                    dst_ptrs, kept_d = _pack_state_ptrs(t_state, self.has_moist)
                    self._raise_if_err(self.lib.nh_cuda_download_state(h, ctypes.byref(dst_ptrs)), "tile download")
                    del kept_d
                    if inplace_1d:
                        if t_idx + 1 < len(self._tiles):
                            lookahead_t_state = self._slice_tile_state(self._state_host, self._tiles[t_idx + 1])
                        else:
                            lookahead_t_state = None
                    self._write_tile_interior(next_state, t_state, tile)
                    del t_state
                finally:
                    if not reused:
                        self.lib.nh_cuda_destroy(h)
                if on_tile is not None:
                    on_tile(t_idx + 1, n_tiles)
            self._state_host = next_state
            if run_phys and next_surf is not None:
                self._surface_state_host = next_surf
                self._advance_host_clock(dt_p)
            if on_step is not None:
                on_step(step_i + 1, run_phys)

    def step_rk3(
        self,
        dt_s: float,
        n_steps: int = 1,
        n_substeps: int = 0,
        *,
        on_step: Optional[Callable[[int, bool], None]] = None,
        on_tile: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        if len(self._tiles) == 1:
            for step_i in range(n_steps):
                self._raise_if_err(
                    self.lib.nh_cuda_step_rk3(self._handle, float(dt_s), int(n_substeps)),
                    "step_rk3",
                )
                if on_step is not None:
                    on_step(step_i + 1, False)
            return
        self._run_multi_tile(dt_s, n_steps, n_substeps, run_dyn=True, run_phys=False, on_step=on_step, on_tile=on_tile)

    def step_physics(
        self,
        dt_s: float,
        n_steps: int = 1,
        *,
        on_step: Optional[Callable[[int, bool], None]] = None,
        on_tile: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        if self._pcfg is None:
            self.set_physics_config()
        if len(self._tiles) == 1:
            for step_i in range(n_steps):
                self._raise_if_err(
                    self.lib.nh_cuda_step_physics(self._handle, float(dt_s)),
                    "step_physics",
                )
                self._advance_host_clock(dt_s)
                if on_step is not None:
                    on_step(step_i + 1, True)
            return
        self._run_multi_tile(dt_s, n_steps, 0, run_dyn=False, run_phys=True, on_step=on_step, on_tile=on_tile)

    def step_full(
        self,
        dt_s: float,
        n_steps: int = 1,
        n_substeps: int = 0,
        *,
        physics_every: int = 1,
        flush_physics: bool = False,
        on_step: Optional[Callable[[int, bool], None]] = None,
        on_tile: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        """Advance coupled dynamics (split-explicit RK3 + FCT) + GPU physics suite.

        Maintains `self._phys_accum_steps` persistently across segment calls so
        `physics_every` cadence never shifts across LBC or output breakpoints.
        """
        if self._pcfg is None:
            self.set_physics_config()
        phys_cadence = max(1, int(physics_every))
        if len(self._tiles) == 1:
            for idx in range(n_steps):
                self._raise_if_err(
                    self.lib.nh_cuda_step_rk3(self._handle, float(dt_s), int(n_substeps)),
                    "step_rk3",
                )
                self._phys_accum_steps += 1
                ran_phys = bool(self._phys_accum_steps >= phys_cadence or (flush_physics and idx == n_steps - 1))
                if ran_phys:
                    dt_phys = float(dt_s) * self._phys_accum_steps
                    self._raise_if_err(
                        self.lib.nh_cuda_step_physics(self._handle, dt_phys),
                        "step_physics",
                    )
                    self._advance_host_clock(dt_phys)
                    self._phys_accum_steps = 0
                if on_step is not None:
                    on_step(idx + 1, ran_phys)
            return

        for idx in range(n_steps):
            self._phys_accum_steps += 1
            due = (self._phys_accum_steps >= phys_cadence) or (flush_physics and idx == n_steps - 1)
            dt_phys = float(dt_s) * self._phys_accum_steps if due else None
            self._run_multi_tile(dt_s, 1, n_substeps, run_dyn=True, run_phys=due, phys_dt_s=dt_phys, on_tile=on_tile)
            if due:
                self._phys_accum_steps = 0
            if on_step is not None:
                on_step(idx + 1, due)

    def download_state(self, *, copy: bool = True) -> Dict[str, np.ndarray]:
        if len(self._tiles) == 1:
            out = self.empty_state()
            ptrs, kept = _pack_state_ptrs(out, self.has_moist)
            self._raise_if_err(self.lib.nh_cuda_download_state(self._handle, ctypes.byref(ptrs)), "download_state")
            del kept
            return out
        if self._state_host is None:
            raise RuntimeError("No uploaded state to download")
        if not copy:
            return self._state_host
        return {k: v.copy() for k, v in self._state_host.items()}

    def tracer_mass(self, q: np.ndarray) -> float:
        """Exact conserved column-weighted tracer mass integral [kg]."""
        _, z_s, g_s = self.vgrid.heights_3d(self.hs_m)
        ref = VerticalGrid.reference_state(
            z_s,
            theta_sfc_k=self.theta_sfc_k,
            p_sfc_pa=self.p_sfc_pa,
            brunt_vaisala_hz=self.n_ref_s,
        )
        dzeta = self.vgrid.dzeta[:, None, None]
        msf_area = self.msf[None, :, :] if self.is_spherical else (self.msf[None, :, :] ** 2)
        cell_area = (self.dx_m * self.dy_m) / msf_area
        return float(np.sum(ref["rho_bar"] * g_s * dzeta * cell_area * np.asarray(q, dtype=np.float64)))

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self.lib.nh_cuda_destroy(self._handle)
            self._handle = None
        if getattr(self, "_tile_handles", None):
            for h in self._tile_handles:
                if h:
                    self.lib.nh_cuda_destroy(h)
            self._tile_handles = []
        if getattr(self, "_shape_tile_handles", None):
            destroyed = set()
            for h in self._shape_tile_handles.values():
                if h and h not in destroyed:
                    self.lib.nh_cuda_destroy(h)
                    destroyed.add(h)
            self._shape_tile_handles = {}
            self._shared_tile_handle = None
        elif getattr(self, "_shared_tile_handle", None):
            self.lib.nh_cuda_destroy(self._shared_tile_handle)
            self._shared_tile_handle = None

    def __del__(self) -> None:
        self.close()
