"""Classical 51+-Member Balanced Operational Ensemble System (0% AI / 0% ML).

Generates physical, flow-dependent, non-AI ensemble perturbations around a deterministic
Control Forecast (member 0 = `cf`):
  - Initial condition perturbations filtered with a self-adjoint `RecursiveSpatialFilter`
    and projected onto the discrete hydrostatic + geostrophic manifold via `BalanceOperator`.
  - Antithetic balanced pairs (`pf001`/`pf002`, `pf003`/`pf004`, ...) ensuring exact
    zero-mean initial/forcing perturbations across pairs.
  - Physical surface perturbations (`soil_m`, `sst_bulk`, `z0m_base`), stochastically
    perturbed parameters (SPP `cloud_num_land` / `cloud_num_sea`), and smooth 3D
    stochastic tendency forcing (`u`, `v`, `theta_p`).
  - Lossless WX-1 (`zstd_shuffle`) ensemble mean, sample spread, quantiles (p10/p50/p90),
    Control Forecast reference (`*_cf`), and exceedance probability products.
"""

from __future__ import annotations

import datetime
import gc
import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import wx1
import xarray as xr

from tempest_v2.da.balance import BalanceOperator
from tempest_v2.da.var3d import RecursiveSpatialFilter
from tempest_v2.ic_builder import stagger_uv
from tempest_v2.projection import DomainSpec

logger = logging.getLogger("tempest_v2.ensemble")


@dataclass
class EnsembleConfig:
    """Configuration for classical antithetic + balanced NWP ensemble."""

    n_members: int = 51  # Member 0 = Control Forecast (`cf`), 1..N-1 = `pf001..pfNNN`
    seed: int = 20261121
    length_scale_h_m: float = 60_000.0
    length_scale_v_m: float = 2_000.0
    amp_ps_pa: float = 60.0
    amp_theta_u_k: float = 0.35
    amp_wind_u_ms: float = 0.60
    amp_qv_rel: float = 0.05
    amp_soil_m: float = 0.02
    amp_sst_k: float = 0.35
    amp_z0m_log: float = 0.15
    amp_spp_ccn_log: float = 0.25
    amp_stoch_tend_u: float = 4.0e-5
    amp_stoch_tend_th: float = 1.5e-5
    stoch_forcing_duration_s: float = 1800.0
    antithetic: bool = True
    save_member_forecasts: bool = True
    base_cloud_num_land: float = 200.0e6
    base_cloud_num_sea: float = 80.0e6
    precip_thresholds_mm: Tuple[float, ...] = field(default_factory=lambda: (0.5, 2.0, 10.0, 25.0))
    wind_thresholds_ms: Tuple[float, ...] = field(default_factory=lambda: (10.0, 15.0, 20.0, 25.0))

    def __post_init__(self) -> None:
        if int(self.n_members) < 1:
            raise ValueError(f"n_members must be >= 1, got {self.n_members}")


def member_label(member_idx: int) -> str:
    return "cf" if int(member_idx) == 0 else f"pf{int(member_idx):03d}"


def _zero_mean_unit_std(arr: np.ndarray) -> np.ndarray:
    centered = np.asarray(arr, dtype=np.float64) - float(np.mean(arr))
    std = float(np.std(centered))
    if std <= 1e-12:
        return np.zeros_like(centered, dtype=np.float64)
    return centered / std


def _lead_tag(hour: float) -> str:
    return f"{int(round(hour)):03d}" if abs(hour - round(hour)) < 1e-4 else f"{hour:06.2f}"


def _thresh_tag(val: float) -> str:
    s = f"{float(val):g}"
    return s.replace(".", "p").replace("-", "m")


def generate_balanced_member_perturbation(
    member_idx: int,
    spec: DomainSpec,
    hs_m: np.ndarray,
    base_state_3d: Dict[str, np.ndarray],
    base_surf_state: Dict[str, np.ndarray],
    base_surface_static: Dict[str, np.ndarray],
    cfg: EnsembleConfig,
    *,
    balance_op: Optional[BalanceOperator] = None,
    spatial_filter: Optional[RecursiveSpatialFilter] = None,
) -> Dict[str, Any]:
    """Build IC + surface + SPP + stochastic tendency perturbations for one ensemble member."""
    idx = int(member_idx)
    label = member_label(idx)

    if idx == 0:
        return {
            "member_idx": 0,
            "member_name": "cf",
            "is_control": True,
            "sign": 0.0,
            "pair_id": -1,
            "state_3d": {k: np.array(v, copy=True) for k, v in base_state_3d.items()},
            "surf_state": {k: np.array(v, copy=True) for k, v in base_surf_state.items()},
            "static_overrides": {},
            "cloud_num_land": float(cfg.base_cloud_num_land),
            "cloud_num_sea": float(cfg.base_cloud_num_sea),
            "stoch_iau": None,
            "delta_raw": None,
        }

    nz, ny, nx = int(spec.nz), int(spec.ny), int(spec.nx)
    if cfg.antithetic:
        pair_id = (idx - 1) // 2
        sign = 1.0 if ((idx - 1) % 2 == 0) else -1.0
        rng_seed = (int(cfg.seed), int(pair_id))
    else:
        pair_id = idx - 1
        sign = 1.0
        rng_seed = (int(cfg.seed), int(idx))
    rng = np.random.default_rng(rng_seed)

    if spatial_filter is None:
        dz_med = float(np.median(np.diff(spec.vertical.zeta_s))) if nz >= 2 else 500.0
        spatial_filter = RecursiveSpatialFilter(
            spec.dx_m,
            spec.dy_m,
            dz_m=max(50.0, dz_med),
            length_scale_h_m=cfg.length_scale_h_m,
            length_scale_v_m=cfg.length_scale_v_m,
            num_passes=2,
        )
    if balance_op is None:
        balance_op = BalanceOperator(spec, hs_m)

    # 1. Spatially correlated control variables mapped onto hydrostatic + geostrophic manifold
    ps_pat = _zero_mean_unit_std(spatial_filter.apply_2d(rng.standard_normal((ny, nx))))
    th_pat = _zero_mean_unit_std(spatial_filter.apply_3d(rng.standard_normal((nz, ny, nx))))
    u_pat = _zero_mean_unit_std(spatial_filter.apply_3d(rng.standard_normal((nz, ny, nx))))
    v_pat = _zero_mean_unit_std(spatial_filter.apply_3d(rng.standard_normal((nz, ny, nx))))
    qv_pat = _zero_mean_unit_std(spatial_filter.apply_3d(rng.standard_normal((nz, ny, nx))))
    soil_pat = _zero_mean_unit_std(spatial_filter.apply_2d(rng.standard_normal((ny, nx))))
    sst_pat = _zero_mean_unit_std(spatial_filter.apply_2d(rng.standard_normal((ny, nx))))
    z0_pat = _zero_mean_unit_std(spatial_filter.apply_2d(rng.standard_normal((ny, nx))))
    tu_pat = _zero_mean_unit_std(spatial_filter.apply_3d(rng.standard_normal((nz, ny, nx))))
    tv_pat = _zero_mean_unit_std(spatial_filter.apply_3d(rng.standard_normal((nz, ny, nx))))
    tth_pat = _zero_mean_unit_std(spatial_filter.apply_3d(rng.standard_normal((nz, ny, nx))))
    ccn_norms = np.clip(rng.standard_normal(2), -2.5, 2.5)

    ctrl = {
        "ps": sign * cfg.amp_ps_pa * ps_pat,
        "theta_p": sign * cfg.amp_theta_u_k * th_pat,
        "pi_p": np.zeros((nz, ny, nx), dtype=np.float64),
        "u": sign * cfg.amp_wind_u_ms * u_pat,
        "v": sign * cfg.amp_wind_u_ms * v_pat,
    }
    bal = balance_op.forward(ctrl)
    du_stag, dv_stag = stagger_uv(bal["u"], bal["v"])
    dth = bal["theta_p"].astype(np.float32)
    dpi = bal["pi_p"].astype(np.float32)

    state_p = {k: np.array(v, copy=True) for k, v in base_state_3d.items()}
    state_p["u"] = (state_p["u"] + du_stag.astype(np.float32)).astype(np.float32)
    state_p["v"] = (state_p["v"] + dv_stag.astype(np.float32)).astype(np.float32)
    state_p["theta_p"] = (state_p["theta_p"] + dth).astype(np.float32)
    state_p["pi_p"] = (state_p["pi_p"] + dpi).astype(np.float32)

    if "qv" in state_p:
        dqv_raw = state_p["qv"].astype(np.float64) * (sign * cfg.amp_qv_rel * qv_pat)
        state_p["qv"] = np.maximum(0.0, state_p["qv"] + dqv_raw.astype(np.float32)).astype(np.float32)

    # 2. Surface initial condition perturbations (soil moisture, bulk SST)
    surf_p = {k: np.array(v, copy=True) for k, v in base_surf_state.items()}
    if "soil_m" in surf_p:
        d_sm = (sign * cfg.amp_soil_m * soil_pat).astype(np.float32)
        surf_p["soil_m"] = np.clip(surf_p["soil_m"] + d_sm[None, :, :], 0.02, 0.48).astype(np.float32)
    if "sst_bulk" in surf_p:
        d_sst = (sign * cfg.amp_sst_k * sst_pat).astype(np.float32)
        surf_p["sst_bulk"] = np.clip(surf_p["sst_bulk"] + d_sst, 271.15, 312.0).astype(np.float32)

    # 3. Roughness length & SPP cloud droplet number perturbations
    static_ov: Dict[str, np.ndarray] = {}
    if "z0m_base" in base_surface_static:
        z0_scale = np.exp(np.clip(sign * cfg.amp_z0m_log * z0_pat, -1.0, 1.0)).astype(np.float32)
        static_ov["z0m_base"] = np.clip(base_surface_static["z0m_base"] * z0_scale, 1e-4, 3.0).astype(np.float32)

    ccn_land = float(
        np.clip(cfg.base_cloud_num_land * np.exp(sign * cfg.amp_spp_ccn_log * float(ccn_norms[0])), 2.0e7, 2.0e9)
    )
    ccn_sea = float(
        np.clip(cfg.base_cloud_num_sea * np.exp(sign * cfg.amp_spp_ccn_log * float(ccn_norms[1])), 2.0e7, 2.0e9)
    )

    # 4. Smooth 3D stochastic physics tendency forcing (stochastic IAU)
    tu_s, tv_s = stagger_uv(sign * cfg.amp_stoch_tend_u * tu_pat, sign * cfg.amp_stoch_tend_u * tv_pat)
    stoch_iau = {
        "u": tu_s.astype(np.float32),
        "v": tv_s.astype(np.float32),
        "w": np.zeros((nz + 1, ny, nx), dtype=np.float32),
        "theta_p": (sign * cfg.amp_stoch_tend_th * tth_pat).astype(np.float32),
        "pi_p": np.zeros((nz, ny, nx), dtype=np.float32),
    }

    return {
        "member_idx": idx,
        "member_name": label,
        "is_control": False,
        "sign": sign,
        "pair_id": pair_id,
        "state_3d": state_p,
        "surf_state": surf_p,
        "static_overrides": static_ov,
        "cloud_num_land": ccn_land,
        "cloud_num_sea": ccn_sea,
        "stoch_iau": stoch_iau,
        "delta_raw": {
            "u": du_stag.astype(np.float32),
            "v": dv_stag.astype(np.float32),
            "theta_p": dth,
            "pi_p": dpi,
            "soil_m": (sign * cfg.amp_soil_m * soil_pat).astype(np.float32),
            "sst_bulk": (sign * cfg.amp_sst_k * sst_pat).astype(np.float32),
            "stoch_theta_p": stoch_iau["theta_p"],
        },
    }


def compute_ensemble_products_wx1(
    member_forecast_files: Sequence[Path | str],
    out_path: Path | str,
    spec: DomainSpec,
    valid_time_utc: datetime.datetime,
    lead_hours: float,
    cfg: EnsembleConfig,
) -> Path:
    """Aggregate member forecasts (0 = cf, 1..M-1 = pf*) into lossless WX-1 ensemble statistics."""
    files = [Path(f) for f in member_forecast_files]
    if not files:
        raise ValueError("member_forecast_files must not be empty")

    m_cnt = len(files)
    ny, nx = int(spec.ny), int(spec.nx)
    t2m_all = np.empty((m_cnt, ny, nx), dtype=np.float32)
    mslp_all = np.empty((m_cnt, ny, nx), dtype=np.float32)
    wspd_all = np.empty((m_cnt, ny, nx), dtype=np.float32)
    precip_all = np.empty((m_cnt, ny, nx), dtype=np.float32)

    for m, fp in enumerate(files):
        with wx1.open_dataset(fp) as ds:
            t2m_all[m] = np.asarray(ds["t2m"].values, dtype=np.float32)
            mslp_all[m] = np.asarray(ds["mslp"].values, dtype=np.float32)
            u10 = np.asarray(ds["u10m_geo"].values, dtype=np.float32)
            v10 = np.asarray(ds["v10m_geo"].values, dtype=np.float32)
            wspd_all[m] = np.hypot(u10, v10).astype(np.float32)
            pr = (
                np.asarray(ds["rain_acc"].values, dtype=np.float32)
                + np.asarray(ds["snow_acc"].values, dtype=np.float32)
                + np.asarray(ds["graupel_acc"].values, dtype=np.float32)
            )
            precip_all[m] = np.maximum(0.0, pr).astype(np.float32)

    ddof = 1 if m_cnt > 1 else 0
    data_vars: Dict[str, Any] = {}

    def _add_stats(prefix: str, arr: np.ndarray, units: str, with_p10_p50: bool = False) -> None:
        data_vars[f"{prefix}_cf"] = (("y", "x"), arr[0].astype(np.float32), {"units": units})
        data_vars[f"{prefix}_mean"] = (("y", "x"), np.mean(arr, axis=0, dtype=np.float64).astype(np.float32), {"units": units})
        data_vars[f"{prefix}_spread"] = (("y", "x"), np.std(arr, axis=0, ddof=ddof, dtype=np.float64).astype(np.float32), {"units": units})
        data_vars[f"{prefix}_p90"] = (("y", "x"), np.percentile(arr, 90.0, axis=0).astype(np.float32), {"units": units})
        if with_p10_p50:
            data_vars[f"{prefix}_p10"] = (("y", "x"), np.percentile(arr, 10.0, axis=0).astype(np.float32), {"units": units})
            data_vars[f"{prefix}_p50"] = (("y", "x"), np.percentile(arr, 50.0, axis=0).astype(np.float32), {"units": units})

    _add_stats("t2m", t2m_all, "K", with_p10_p50=True)
    _add_stats("mslp", mslp_all, "Pa", with_p10_p50=False)
    _add_stats("wind10m", wspd_all, "m s-1", with_p10_p50=False)
    _add_stats("precip", precip_all, "kg m-2", with_p10_p50=False)

    for th in cfg.precip_thresholds_mm:
        prob = np.mean(precip_all >= np.float32(th), axis=0, dtype=np.float64).astype(np.float32)
        data_vars[f"prob_precip_ge_{_thresh_tag(th)}mm"] = (("y", "x"), prob, {"units": "1"})

    for th in cfg.wind_thresholds_ms:
        prob = np.mean(wspd_all >= np.float32(th), axis=0, dtype=np.float64).astype(np.float32)
        data_vars[f"prob_wind10m_ge_{_thresh_tag(th)}ms"] = (("y", "x"), prob, {"units": "1"})

    prob_frost = np.mean(t2m_all < np.float32(273.15), axis=0, dtype=np.float64).astype(np.float32)
    data_vars["prob_frost_t2m_lt_0c"] = (("y", "x"), prob_frost, {"units": "1"})
    data_vars["lat"] = (("y", "x"), spec.lat2d.astype(np.float32), {"units": "degrees_north"})
    data_vars["lon"] = (("y", "x"), spec.lon2d.astype(np.float32), {"units": "degrees_east"})

    v_dt = (
        valid_time_utc.astimezone(datetime.timezone.utc)
        if valid_time_utc.tzinfo
        else valid_time_utc.replace(tzinfo=datetime.timezone.utc)
    )
    t_coord = np.array([np.datetime64(int(v_dt.timestamp()), "s")], dtype="datetime64[s]")

    ds_out = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": ("time", t_coord),
            "y": ("y", spec.y_m.astype(np.float64), {"units": "m"}),
            "x": ("x", spec.x_m.astype(np.float64), {"units": "m"}),
        },
        attrs={
            "kind": "tempest_v2_ensemble_stats",
            "model": "Tempest-NWP v2 Ensemble (0% AI/ML)",
            "domain_name": spec.name,
            "n_members": str(m_cnt),
            "control_member": "cf",
            "antithetic": "1" if cfg.antithetic else "0",
            "lead_hours": f"{float(lead_hours):.3f}",
            "valid_time_utc": v_dt.isoformat(),
        },
    )
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    wx1.write_dataset(p, ds_out, dataset_id=f"ens_{spec.name}_{lead_hours:06.2f}h", codec="zstd_shuffle", workers=4)
    return p


def run_ensemble_forecast(
    spec: DomainSpec,
    hs_m: np.ndarray,
    surface_static: Dict[str, np.ndarray],
    base_state_3d: Dict[str, np.ndarray],
    base_surf_state: Dict[str, np.ndarray],
    init_time_utc: datetime.datetime,
    *,
    forecast_hours: float,
    ens_dir: Path | str,
    cfg: EnsembleConfig,
    dt_s: float = 4.0,
    physics_every: int = 5,
    lbc_interpolator: Any = None,
    base_iau_forcing: Optional[Dict[str, np.ndarray]] = None,
    base_iau_duration_s: float = 0.0,
    output_hours: Optional[Sequence[float]] = None,
    enable_physics: bool = True,
    rad_interval_s: float = 900.0,
    existing_cf_outputs: Optional[Dict[float, Path]] = None,
    theta_sfc_ref: float = 288.15,
    p_sfc_ref: float = 101_325.0,
    n_ref_hz: float = 0.01,
) -> Dict[str, Any]:
    """Execute a classical M-member ensemble (`cf` + `pf001..pf{M-1:03d}`) and write WX-1 stats."""
    from tempest_v2.runner import forecast_output_schedule_hours, run_domain_forecast

    t0_wall = time.perf_counter()
    ens_root = Path(ens_dir)
    ens_root.mkdir(parents=True, exist_ok=True)
    stats_dir = ens_root / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    init_dt = (
        init_time_utc.astimezone(datetime.timezone.utc)
        if init_time_utc.tzinfo
        else init_time_utc.replace(tzinfo=datetime.timezone.utc)
    )
    hrs = (
        [float(h) for h in output_hours]
        if output_hours is not None
        else forecast_output_schedule_hours(forecast_hours, hourly=True)
    )
    if float(forecast_hours) not in hrs:
        hrs = sorted(set(hrs) | {float(forecast_hours)})

    dz_med = float(np.median(np.diff(spec.vertical.zeta_s))) if spec.nz >= 2 else 500.0
    spatial_filter = RecursiveSpatialFilter(
        spec.dx_m,
        spec.dy_m,
        dz_m=max(50.0, dz_med),
        length_scale_h_m=cfg.length_scale_h_m,
        length_scale_v_m=cfg.length_scale_v_m,
        num_passes=2,
    )
    balance_op = BalanceOperator(
        spec,
        hs_m,
        theta_sfc_ref=theta_sfc_ref,
        p_sfc_ref=p_sfc_ref,
        n_ref_hz=n_ref_hz,
    )

    member_files_by_hour: Dict[float, List[Path]] = {hr: [] for hr in hrs}
    member_summaries: List[Dict[str, Any]] = []

    for m_idx in range(cfg.n_members):
        m_name = member_label(m_idx)
        m_dir = ens_root / m_name
        m_dir.mkdir(parents=True, exist_ok=True)

        # Member 0 = deterministic Control Forecast (`cf`); reuse precomputed CF files when available
        reused_cf = False
        if m_idx == 0 and existing_cf_outputs is not None:
            matched: List[Tuple[float, Path]] = []
            for hr in hrs:
                src = existing_cf_outputs.get(hr)
                if src is None:
                    for k_hr, v_path in existing_cf_outputs.items():
                        if abs(float(k_hr) - float(hr)) < 5e-3:
                            src = v_path
                            break
                if src is not None and Path(src).exists():
                    matched.append((hr, Path(src)))
            if len(matched) == len(hrs):
                for hr, src_p in matched:
                    dst_p = m_dir / src_p.name
                    if src_p.resolve() != dst_p.resolve():
                        shutil.copy2(src_p, dst_p)
                    member_files_by_hour[hr].append(dst_p)
                reused_cf = True
                member_summaries.append(
                    {
                        "member_idx": 0,
                        "member_name": "cf",
                        "is_control": True,
                        "reused_control_run": True,
                        "elapsed_s": 0.0,
                        "outputs": [str(member_files_by_hour[hr][-1]) for hr in hrs],
                    }
                )

        if reused_cf:
            continue

        pert = generate_balanced_member_perturbation(
            m_idx,
            spec,
            hs_m,
            base_state_3d,
            base_surf_state,
            surface_static,
            cfg,
            balance_op=balance_op,
            spatial_filter=spatial_filter,
        )

        # Combine DA IAU forcing + smooth stochastic tendency forcing
        eff_iau: Optional[Dict[str, np.ndarray]] = None
        eff_iau_dur = 0.0
        stoch = pert["stoch_iau"]
        if base_iau_forcing is not None and stoch is not None:
            eff_iau = {
                k: (
                    np.asarray(base_iau_forcing.get(k, 0.0), dtype=np.float32)
                    + np.asarray(stoch.get(k, 0.0), dtype=np.float32)
                ).astype(np.float32)
                for k in ("u", "v", "w", "theta_p", "pi_p")
            }
            eff_iau_dur = max(float(base_iau_duration_s), float(cfg.stoch_forcing_duration_s))
        elif stoch is not None:
            eff_iau = stoch
            eff_iau_dur = float(cfg.stoch_forcing_duration_s)
        elif base_iau_forcing is not None:
            eff_iau = base_iau_forcing
            eff_iau_dur = float(base_iau_duration_s)

        res_m = run_domain_forecast(
            spec,
            hs_m,
            surface_static,
            pert["state_3d"],
            pert["surf_state"],
            init_dt,
            forecast_hours=forecast_hours,
            dt_s=dt_s,
            physics_every=physics_every,
            lbc_interpolator=lbc_interpolator,
            iau_forcing=eff_iau,
            iau_duration_s=eff_iau_dur,
            output_dir=m_dir,
            output_hours=hrs,
            enable_physics=enable_physics,
            rad_interval_s=rad_interval_s,
            cloud_num_land=float(pert["cloud_num_land"]),
            cloud_num_sea=float(pert["cloud_num_sea"]),
            static_overrides=pert["static_overrides"] or None,
            theta_sfc_ref=theta_sfc_ref,
            p_sfc_ref=p_sfc_ref,
            n_ref_hz=n_ref_hz,
        )

        expected_files: List[Path] = []
        for hr in hrs:
            fp = m_dir / f"forecast_{spec.name}_T+{_lead_tag(hr)}h.wx1"
            if not fp.exists():
                raise FileNotFoundError(f"Expected ensemble member output missing: {fp}")
            member_files_by_hour[hr].append(fp)
            expected_files.append(fp)

        member_summaries.append(
            {
                "member_idx": m_idx,
                "member_name": m_name,
                "is_control": bool(pert["is_control"]),
                "reused_control_run": False,
                "elapsed_s": float(res_m["elapsed_s"]),
                "cloud_num_land": float(pert["cloud_num_land"]),
                "cloud_num_sea": float(pert["cloud_num_sea"]),
                "outputs": [str(p) for p in expected_files],
            }
        )

        # Free host buffers + GPU allocations between members cleanly
        del pert, eff_iau, res_m
        gc.collect()

    stats_paths: List[Path] = []
    for hr in hrs:
        v_dt = init_dt + datetime.timedelta(seconds=float(hr) * 3600.0)
        sp = compute_ensemble_products_wx1(
            member_files_by_hour[hr],
            stats_dir / f"ens_stats_f{_lead_tag(hr)}.wx1",
            spec,
            v_dt,
            hr,
            cfg,
        )
        stats_paths.append(sp)

    if not cfg.save_member_forecasts:
        for m_idx in range(1, cfg.n_members):
            m_dir = ens_root / member_label(m_idx)
            if m_dir.exists():
                shutil.rmtree(m_dir, ignore_errors=True)

    manifest_path = ens_root / "ensemble_manifest.json"
    summary = {
        "domain": spec.name,
        "init_time_utc": init_dt.isoformat(),
        "n_members": int(cfg.n_members),
        "control_member": "cf",
        "antithetic": bool(cfg.antithetic),
        "seed": int(cfg.seed),
        "forecast_hours": float(forecast_hours),
        "output_hours": hrs,
        "stats_paths": [str(p) for p in stats_paths],
        "members": member_summaries,
        "elapsed_s": float(time.perf_counter() - t0_wall),
    }
    manifest_path.write_text(json.dumps(summary, indent=2))
    summary["manifest_path"] = str(manifest_path)
    return summary
