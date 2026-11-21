"""Python ctypes bridge and automatic 2D halo-tiling manager for tempest-cuda."""

from __future__ import annotations

import ctypes
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from tempest_v2.projection import VerticalGrid

DEFAULT_MAX_VRAM_BYTES = int(3.8 * (1024**3))  # Reserve >2.0 GB headroom on 6 GB RTX 3060


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
) -> int:
    """Exact GPU VRAM estimate in bytes for a tile `(nz, ny, nx)`."""
    lib = load_cuda_library()
    return int(
        lib.nh_cuda_estimate_vram_bytes(
            int(nz),
            int(ny),
            int(nx),
            1 if moist else 0,
            1 if has_lbc else 0,
            1 if has_iau else 0,
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
    max_vram_bytes: int = DEFAULT_MAX_VRAM_BYTES,
    halo: int = 6,
) -> Tuple[int, int, int]:
    """Compute minimal `(nt_y, nt_x, est_tile_bytes)` keeping peak VRAM <= `max_vram_bytes`."""
    nt_y = 1
    nt_x = 1
    while True:
        tile_ny = math.ceil(ny / nt_y) + (2 * halo if nt_y > 1 else 0)
        tile_nx = math.ceil(nx / nt_x) + (2 * halo if nt_x > 1 else 0)
        tile_ny = min(ny, tile_ny)
        tile_nx = min(nx, tile_nx)
        est = estimate_vram_bytes(
            nz, tile_ny, tile_nx, moist=moist, has_lbc=has_lbc, has_iau=has_iau
        )
        if est <= max_vram_bytes or (tile_ny <= 32 and tile_nx <= 32):
            return nt_y, nt_x, est
        if nt_x <= nt_y and tile_nx > 32:
            nt_x += 1
        elif tile_ny > 32:
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
    )
    return ptrs, kept


class NonHydrostaticGPUSolver:
    """
    GPU-accelerated split-explicit RK3 non-hydrostatic dynamical core with
    automatic 2D horizontal halo tiling (`nz` always kept intact per column).
    """

    SPECIES = ("qv", "qc", "qr", "qi", "qs", "qg")

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
        max_vram_bytes: int = DEFAULT_MAX_VRAM_BYTES,
        halo: int = 6,
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
        self.halo = int(halo)

        self.nt_y, self.nt_x, self.estimated_tile_vram_bytes = plan_horizontal_tiling(
            self.nz,
            self.ny,
            self.nx,
            moist=self.has_moist,
            has_lbc=(self.bc_x == 2 or self.bc_y == 2),
            has_iau=self.has_iau,
            max_vram_bytes=max_vram_bytes,
            halo=self.halo,
        )
        if (self.nt_y > 1 and self.bc_y == 1) or (self.nt_x > 1 and self.bc_x == 1):
            raise ValueError(
                "2D halo tiling only supports non-periodic horizontal boundaries (bc=0 or bc=2)."
            )

        self._handle: Optional[int] = None
        self._state_host: Optional[Dict[str, np.ndarray]] = None
        self._lbc_host: Optional[Dict[str, np.ndarray]] = None
        self._iau_host: Optional[Dict[str, np.ndarray]] = None
        self._tiles = self._build_tiles()

        if len(self._tiles) == 1:
            self._handle = self._create_tile_handle(self._tiles[0])

    @property
    def num_tiles(self) -> int:
        return len(self._tiles)

    def _build_tiles(self) -> List[_TileBox]:
        if self.nt_y == 1 and self.nt_x == 1:
            edges = (1 if self.bc_x == 2 else 0) | (2 if self.bc_x == 2 else 0) | (
                4 if self.bc_y == 2 else 0
            ) | (8 if self.bc_y == 2 else 0)
            return [_TileBox(0, self.ny, 0, self.nx, 0, self.ny, 0, self.nx, edges)]

        y_splits = np.linspace(0, self.ny, self.nt_y + 1, dtype=int)
        x_splits = np.linspace(0, self.nx, self.nt_x + 1, dtype=int)
        tiles: List[_TileBox] = []
        for ty in range(self.nt_y):
            j0, j1 = int(y_splits[ty]), int(y_splits[ty + 1])
            pj0 = max(0, j0 - self.halo)
            pj1 = min(self.ny, j1 + self.halo)
            for tx in range(self.nt_x):
                i0, i1 = int(x_splits[tx]), int(x_splits[tx + 1])
                pi0 = max(0, i0 - self.halo)
                pi1 = min(self.nx, i1 + self.halo)
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
        t_bc_x = self.bc_x
        t_bc_y = self.bc_y
        if len(self._tiles) > 1:
            # Open interior tile seams cleanly (no wall u=0 clamp inside interior halo edges)
            if tile.pi0 > 0 or tile.pi1 < self.nx:
                t_bc_x = 2
            if tile.pj0 > 0 or tile.pj1 < self.ny:
                t_bc_y = 2
        return NhGridConfigC(
            nz=self.nz,
            ny=tile.pj1 - tile.pj0,
            nx=tile.pi1 - tile.pi0,
            has_moist=1 if self.has_moist else 0,
            bc_x=t_bc_x,
            bc_y=t_bc_y,
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
        return state

    def _slice_tile_state(self, src: Dict[str, np.ndarray], tile: _TileBox) -> Dict[str, np.ndarray]:
        out = {
            "u": np.ascontiguousarray(src["u"][:, tile.pj0 : tile.pj1, tile.pi0 : tile.pi1 + 1], dtype=np.float32),
            "v": np.ascontiguousarray(src["v"][:, tile.pj0 : tile.pj1 + 1, tile.pi0 : tile.pi1], dtype=np.float32),
            "w": np.ascontiguousarray(src["w"][:, tile.pj0 : tile.pj1, tile.pi0 : tile.pi1], dtype=np.float32),
            "theta_p": np.ascontiguousarray(src["theta_p"][:, tile.pj0 : tile.pj1, tile.pi0 : tile.pi1], dtype=np.float32),
            "pi_p": np.ascontiguousarray(src["pi_p"][:, tile.pj0 : tile.pj1, tile.pi0 : tile.pi1], dtype=np.float32),
        }
        if self.has_moist:
            for sp in self.SPECIES:
                out[sp] = np.ascontiguousarray(src[sp][:, tile.pj0 : tile.pj1, tile.pi0 : tile.pi1], dtype=np.float32)
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

        u_end = tile.i1 + (1 if tile.i1 == self.nx else 0)
        lu_end = li1 + (1 if tile.i1 == self.nx else 0)
        dst["u"][:, tile.j0 : tile.j1, tile.i0 : u_end] = tile_state["u"][:, lj0:lj1, li0:lu_end]

        v_end = tile.j1 + (1 if tile.j1 == self.ny else 0)
        lv_end = lj1 + (1 if tile.j1 == self.ny else 0)
        dst["v"][:, tile.j0 : v_end, tile.i0 : tile.i1] = tile_state["v"][:, lj0:lv_end, li0:li1]

    def upload_state(self, state: Dict[str, np.ndarray]) -> None:
        if len(self._tiles) == 1:
            ptrs, kept = _pack_state_ptrs(state, self.has_moist)
            self._raise_if_err(self.lib.nh_cuda_upload_state(self._handle, ctypes.byref(ptrs)), "upload_state")
            del kept
        else:
            self._state_host = {k: np.ascontiguousarray(v, dtype=np.float32).copy() for k, v in state.items()}

    def set_lbc_target(self, target: Dict[str, np.ndarray]) -> None:
        if len(self._tiles) == 1:
            ptrs, kept = _pack_state_ptrs(target, False)
            self._raise_if_err(self.lib.nh_cuda_set_lbc_target(self._handle, ctypes.byref(ptrs)), "set_lbc_target")
            del kept
        else:
            self._lbc_host = {k: np.ascontiguousarray(v, dtype=np.float32).copy() for k, v in target.items()}

    def set_iau_forcing(self, ddt_state: Dict[str, np.ndarray]) -> None:
        if len(self._tiles) == 1:
            ptrs, kept = _pack_state_ptrs(ddt_state, False)
            self._raise_if_err(self.lib.nh_cuda_set_iau_forcing(self._handle, ctypes.byref(ptrs)), "set_iau_forcing")
            del kept
        else:
            self._iau_host = {k: np.ascontiguousarray(v, dtype=np.float32).copy() for k, v in ddt_state.items()}

    def step_rk3(self, dt_s: float, n_steps: int = 1, n_substeps: int = 0) -> None:
        if len(self._tiles) == 1:
            for _ in range(n_steps):
                self._raise_if_err(
                    self.lib.nh_cuda_step_rk3(self._handle, float(dt_s), int(n_substeps)),
                    "step_rk3",
                )
            return

        if self._state_host is None:
            raise RuntimeError("upload_state must be called before step_rk3")

        for _ in range(n_steps):
            next_host = {k: np.empty_like(v) for k, v in self._state_host.items()}
            for tile in self._tiles:
                h = self._create_tile_handle(tile)
                try:
                    t_state = self._slice_tile_state(self._state_host, tile)
                    ptrs, kept = _pack_state_ptrs(t_state, self.has_moist)
                    self._raise_if_err(self.lib.nh_cuda_upload_state(h, ctypes.byref(ptrs)), "tile upload")
                    del kept

                    if self._lbc_host is not None and (self.bc_x == 2 or self.bc_y == 2):
                        lbc_slice = self._slice_tile_state(self._lbc_host, tile)
                        lbc_ptrs, kept_l = _pack_state_ptrs(lbc_slice, False)
                        self._raise_if_err(self.lib.nh_cuda_set_lbc_target(h, ctypes.byref(lbc_ptrs)), "tile lbc")
                        del kept_l

                    if self._iau_host is not None and self.has_iau:
                        iau_slice = self._slice_tile_state(self._iau_host, tile)
                        iau_ptrs, kept_i = _pack_state_ptrs(iau_slice, False)
                        self._raise_if_err(self.lib.nh_cuda_set_iau_forcing(h, ctypes.byref(iau_ptrs)), "tile iau")
                        del kept_i

                    self._raise_if_err(
                        self.lib.nh_cuda_step_rk3(h, float(dt_s), int(n_substeps)),
                        "tile step_rk3",
                    )

                    dst_ptrs, kept_d = _pack_state_ptrs(t_state, self.has_moist)
                    self._raise_if_err(self.lib.nh_cuda_download_state(h, ctypes.byref(dst_ptrs)), "tile download")
                    del kept_d
                    self._write_tile_interior(next_host, t_state, tile)
                finally:
                    self.lib.nh_cuda_destroy(h)
            self._state_host = next_host

    def download_state(self) -> Dict[str, np.ndarray]:
        if len(self._tiles) == 1:
            out = self.empty_state()
            ptrs, kept = _pack_state_ptrs(out, self.has_moist)
            self._raise_if_err(self.lib.nh_cuda_download_state(self._handle, ctypes.byref(ptrs)), "download_state")
            del kept
            return out
        if self._state_host is None:
            raise RuntimeError("No uploaded state to download")
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
        cell_area = (self.dx_m * self.dy_m) / (self.msf[None, :, :] ** 2)
        return float(np.sum(ref["rho_bar"] * g_s * dzeta * cell_area * np.asarray(q, dtype=np.float64)))

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self.lib.nh_cuda_destroy(self._handle)
            self._handle = None

    def __del__(self) -> None:
        self.close()
