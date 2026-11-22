"""KNMI-Style Shifted ECMWF Open Data LBC Scheduler & WX-1 Rim Builder for Tempest-NWP v2.

Strictly 100% Classical Atmospheric Preprocessing (0% AI / 0% ML), SI Units Only:
- Resolves the latest complete ECMWF IFS Open Data cycle (0.1 deg default, 0.25 deg fallback)
  and shifts forecast step indices so a Tempest LAM cycle starting at T_init reads
  valid-time-matched ECMWF boundary conditions without waiting for a fresh IFS analysis.
- Crops global lat/lon GRIB2 fields to the LAM bounding box (in ic_builder._regrid_latlon_field),
  maps isobaric levels onto Klemp (2011) terrain-following levels with exact LCC wind rotation,
  and writes rim-packed zstd_shuffle WX-1 files directly consumable by LBCInterpolator.
"""

from __future__ import annotations

import datetime
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import xarray as xr

import wx1

from tempest_v2.ic_builder import (
    _read_cfgrib_var_map,
    _regrid_latlon_field,
    _vertical_map_isobaric_to_klemp,
    unstagger_uv,
)
from tempest_v2.ingest_ecmwf import LBC_SFC_VARS, STANDARD_SFC_VARS, download_ecmwf_data
from tempest_v2.projection import GRAVITY_M_S2, R_D, DomainSpec

logger = logging.getLogger("tempest_v2.ecmwf_lbc")


@dataclass(frozen=True)
class ECMWFLBCPlan:
    """Resolved ECMWF Open Data LBC schedule for a shifted-cycle LAM integration."""

    ecmwf_cycle_utc: datetime.datetime
    target_init_utc: datetime.datetime
    resol: str
    source: str
    lead_offset_hours: float
    ecmwf_steps: Tuple[int, ...]
    sim_times_sec: Tuple[float, ...]
    valid_times_utc: Tuple[datetime.datetime, ...]
    ic_step: int


def _to_utc(dt: datetime.datetime) -> datetime.datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=datetime.timezone.utc)
    return dt.astimezone(datetime.timezone.utc)


def ecmwf_available_steps(cycle_hour: int, resol: str = "0p1") -> List[int]:
    """Return ECMWF Open Data published IFS forecast steps [hours] for a synoptic cycle hour."""
    hr = int(cycle_hour)
    if hr in (0, 12):
        max_h = 240 if resol == "0p1" else 360
        return list(range(0, 145, 3)) + list(range(150, max_h + 1, 6))
    if hr in (6, 18):
        return list(range(0, 145, 3))
    raise ValueError(f"ECMWF IFS synoptic cycle hour must be 0, 6, 12, or 18; got {cycle_hour}")


def plan_ecmwf_lbc_steps(
    ecmwf_cycle_utc: datetime.datetime,
    target_init_utc: datetime.datetime,
    forecast_hours: float,
    *,
    lbc_interval_hours: float = 3.0,
    resol: str = "0p1",
    source: str = "azure",
) -> ECMWFLBCPlan:
    """Compute shifted ECMWF forecast steps & valid times covering `[target_init_utc, +forecast_hours]`."""
    cyc_dt = _to_utc(ecmwf_cycle_utc).replace(minute=0, second=0, microsecond=0)
    tgt_dt = _to_utc(target_init_utc)
    offset_h = (tgt_dt - cyc_dt).total_seconds() / 3600.0
    if offset_h < -1e-6:
        raise ValueError(
            f"ECMWF cycle {cyc_dt.isoformat()} is later than target init {tgt_dt.isoformat()}"
        )
    if abs(offset_h) < 1e-6:
        offset_h = 0.0

    avail = ecmwf_available_steps(cyc_dt.hour, resol=resol)
    start_cands = [s for s in avail if s <= offset_h + 1e-6]
    if not start_cands:
        raise ValueError(f"No ECMWF step <= offset {offset_h:.2f}h for {cyc_dt.isoformat()}")
    step_start = max(start_cands)

    need_end = offset_h + max(0.0, float(forecast_hours))
    end_cands = [s for s in avail if s >= need_end - 1e-6]
    if not end_cands:
        raise ValueError(
            f"ECMWF cycle {cyc_dt.isoformat()} ({resol}) max step {avail[-1]}h "
            f"cannot reach required {need_end:.2f}h"
        )
    step_end = min(end_cands)

    min_gap = max(3.0, float(lbc_interval_hours)) - 1e-6
    selected: List[int] = [step_start]
    for s in avail:
        if step_start < s <= step_end and (s - selected[-1]) >= min_gap:
            selected.append(s)
    if selected[-1] != step_end:
        selected.append(step_end)

    sim_times_sec = tuple((float(s) - offset_h) * 3600.0 for s in selected)
    valid_times_utc = tuple(cyc_dt + datetime.timedelta(hours=int(s)) for s in selected)

    return ECMWFLBCPlan(
        ecmwf_cycle_utc=cyc_dt,
        target_init_utc=tgt_dt,
        resol=resol,
        source=source,
        lead_offset_hours=float(offset_h),
        ecmwf_steps=tuple(selected),
        sim_times_sec=sim_times_sec,
        valid_times_utc=valid_times_utc,
        ic_step=int(step_start),
    )


def _default_ecmwf_latest_fn(source: str) -> Callable[[str, int], datetime.datetime]:
    from ecmwf.opendata import Client

    clients: Dict[str, Any] = {}

    def _query(res: str, step: int) -> datetime.datetime:
        if res not in clients:
            clients[res] = Client(source=source, model="ifs", resol=res)
        dt = clients[res].latest(type="fc", step=int(step), levtype="pl", levelist=[500], param=["t"])
        return _to_utc(dt)

    return _query


def resolve_ecmwf_lbc_plan(
    target_init_utc: datetime.datetime,
    forecast_hours: float,
    *,
    resol: str = "auto",
    lbc_interval_hours: float = 3.0,
    source: str = "azure",
    max_0p1_age_hours: float = 18.0,
    latest_fn: Optional[Callable[[str, int], datetime.datetime]] = None,
) -> ECMWFLBCPlan:
    """Resolve the newest complete ECMWF Open Data cycle (`0p1` default, `0p25` fallback)."""
    tgt_dt = _to_utc(target_init_utc)
    tgt_floor = tgt_dt.replace(hour=(tgt_dt.hour // 6) * 6, minute=0, second=0, microsecond=0)
    query_latest = latest_fn if latest_fn is not None else _default_ecmwf_latest_fn(source)
    resol_seq = ("0p1", "0p25") if resol == "auto" else (resol,)

    errors: List[str] = []
    for r in resol_seq:
        try:
            latest_c0 = _to_utc(query_latest(r, 0)).replace(minute=0, second=0, microsecond=0)
        except Exception as exc:
            errors.append(f"{r} latest(step=0) failed: {exc}")
            continue

        cand0 = min(latest_c0, tgt_floor)
        for k in range(4):
            cand = cand0 - datetime.timedelta(hours=6 * k)
            try:
                plan = plan_ecmwf_lbc_steps(
                    cand,
                    tgt_dt,
                    forecast_hours,
                    lbc_interval_hours=lbc_interval_hours,
                    resol=r,
                    source=source,
                )
            except ValueError as exc:
                errors.append(f"{r} {cand:%Y%m%d_%HZ}: {exc}")
                continue

            if resol == "auto" and r == "0p1" and plan.lead_offset_hours > float(max_0p1_age_hours):
                errors.append(
                    f"0p1 {cand:%Y%m%d_%HZ}: offset {plan.lead_offset_hours:.1f}h > {max_0p1_age_hours:.1f}h limit"
                )
                break

            max_step = int(plan.ecmwf_steps[-1])
            try:
                latest_end = _to_utc(query_latest(r, max_step)).replace(minute=0, second=0, microsecond=0)
            except Exception as exc:
                errors.append(f"{r} latest(step={max_step}) failed: {exc}")
                continue

            if latest_end >= cand:
                logger.info(
                    "Resolved ECMWF %s cycle %s for target %s (offset=%.1fh, steps=%s..%s, count=%d)",
                    r,
                    cand.strftime("%Y-%m-%d %HZ"),
                    tgt_dt.strftime("%Y-%m-%d %HZ"),
                    plan.lead_offset_hours,
                    plan.ecmwf_steps[0],
                    plan.ecmwf_steps[-1],
                    len(plan.ecmwf_steps),
                )
                return plan
            errors.append(
                f"{r} {cand:%Y%m%d_%HZ}: step {max_step}h only published through {latest_end:%Y%m%d_%HZ}"
            )

    raise RuntimeError(
        f"Failed to resolve complete ECMWF LBC cycle for {tgt_dt.isoformat()} ({'; '.join(errors)})"
    )


def _extract_rim_packed_lbc(
    child_spec: DomainSpec,
    state_3d: Dict[str, np.ndarray],
    ps_child: np.ndarray,
    rim_width: int,
) -> Tuple[Dict[str, Tuple[Tuple[str, ...], np.ndarray, Dict[str, str]]], Dict[str, np.ndarray], Dict[str, int]]:
    u_c, v_c = unstagger_uv(state_3d["u"], state_3d["v"])
    w_rim_ready = np.asarray(state_3d["w"], dtype=np.float32)
    ny_c, nx_c = int(child_spec.ny), int(child_spec.nx)
    use_rim = int(rim_width) > 0 and ny_c > 2 * int(rim_width) + 8 and nx_c > 2 * int(rim_width) + 8

    ps32 = np.asarray(ps_child, dtype=np.float32)
    if not use_rim:
        data_vars = {
            "u": (("z", "y", "x"), u_c, {"units": "m s-1"}),
            "v": (("z", "y", "x"), v_c, {"units": "m s-1"}),
            "w": (("z_w", "y", "x"), w_rim_ready, {"units": "m s-1"}),
            "theta_p": (("z", "y", "x"), state_3d["theta_p"], {"units": "K"}),
            "pi_p": (("z", "y", "x"), state_3d["pi_p"], {"units": "1"}),
            "qv": (("z", "y", "x"), state_3d["qv"], {"units": "kg kg-1"}),
            "ps": (("y", "x"), ps32, {"units": "Pa"}),
        }
        coords = {
            "y": child_spec.y_m.astype(np.float64),
            "x": child_spec.x_m.astype(np.float64),
        }
        return data_vars, coords, {"use_rim": 0}

    rw = int(rim_width)
    mask = np.zeros((ny_c, nx_c), dtype=bool)
    mask[:rw, :] = True
    mask[-rw:, :] = True
    mask[:, :rw] = True
    mask[:, -rw:] = True
    rj, ri = np.nonzero(mask)
    n_rim = len(rj)

    data_vars = {
        "u": (("z", "rim"), np.ascontiguousarray(u_c[:, rj, ri], dtype=np.float32), {"units": "m s-1"}),
        "v": (("z", "rim"), np.ascontiguousarray(v_c[:, rj, ri], dtype=np.float32), {"units": "m s-1"}),
        "w": (("z_w", "rim"), np.ascontiguousarray(w_rim_ready[:, rj, ri], dtype=np.float32), {"units": "m s-1"}),
        "theta_p": (("z", "rim"), np.ascontiguousarray(state_3d["theta_p"][:, rj, ri], dtype=np.float32), {"units": "K"}),
        "pi_p": (("z", "rim"), np.ascontiguousarray(state_3d["pi_p"][:, rj, ri], dtype=np.float32), {"units": "1"}),
        "qv": (("z", "rim"), np.ascontiguousarray(state_3d["qv"][:, rj, ri], dtype=np.float32), {"units": "kg kg-1"}),
        "ps": (("rim",), np.ascontiguousarray(ps32[rj, ri], dtype=np.float32), {"units": "Pa"}),
        "rim_j": (("rim",), rj.astype(np.int32), {"units": "1"}),
        "rim_i": (("rim",), ri.astype(np.int32), {"units": "1"}),
    }
    coords = {"rim": np.arange(n_rim, dtype=np.int32)}
    return data_vars, coords, {"use_rim": 1, "rim_width": rw, "n_rim": n_rim}


def ecmwf_grib_to_lbc_wx1(
    sfc_grib: Path | str,
    pl_grib: Path | str,
    child_spec: DomainSpec,
    child_hs_m: np.ndarray,
    output_dir: Path | str,
    step_index: int,
    sim_time_sec: float,
    valid_time_utc: datetime.datetime,
    *,
    resol: str = "0p1",
    ecmwf_cycle_utc: Optional[datetime.datetime] = None,
    ecmwf_step_hours: Optional[int] = None,
    rim_width: int = 16,
    theta_sfc_ref: float = 288.15,
    p_sfc_ref: float = 101_325.0,
    n_ref_hz: float = 0.01,
) -> Path:
    """Convert one ECMWF GRIB2 (sfc + pl) forecast step directly into a Tempest WX-1 LBC snapshot."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"lbc_{child_spec.name}_step_{int(step_index):04d}.wx1"

    pl_vars = _read_cfgrib_var_map(Path(pl_grib), use_cache=False)
    t_da, pl_lats, pl_lons = pl_vars["t"]
    p_coord = next(c for c in ("isobaricInhPa", "level") if c in t_da.coords)
    p_levels_pa = np.asarray(t_da.coords[p_coord].values, dtype=np.float64) * 100.0

    def _squeeze_pl(name: str) -> np.ndarray:
        arr = np.asarray(pl_vars[name][0].values)
        while arr.ndim > 3:
            arr = arr[0]
        return arr

    t_h = _regrid_latlon_field(_squeeze_pl("t"), pl_lats, pl_lons, child_spec, fill_nan_value=260.0)
    u_geo_h = _regrid_latlon_field(_squeeze_pl("u"), pl_lats, pl_lons, child_spec, fill_nan_value=0.0)
    v_geo_h = _regrid_latlon_field(_squeeze_pl("v"), pl_lats, pl_lons, child_spec, fill_nan_value=0.0)
    q_h = np.clip(_regrid_latlon_field(_squeeze_pl("q"), pl_lats, pl_lons, child_spec, fill_nan_value=1e-4), 1e-7, 0.04)
    gh_name = "gh" if "gh" in pl_vars else "z"
    gh_arr = _squeeze_pl(gh_name)
    if gh_name == "z":
        gh_arr = np.asarray(gh_arr, dtype=np.float64) / GRAVITY_M_S2
    z_h = _regrid_latlon_field(gh_arr, pl_lats, pl_lons, child_spec, fill_nan_value=0.0)
    del pl_vars

    sfc_vars = _read_cfgrib_var_map(Path(sfc_grib), use_cache=False)

    def _get_sfc(aliases: Tuple[str, ...], default_val: float) -> np.ndarray:
        for k in aliases:
            if k in sfc_vars:
                da, la, lo = sfc_vars[k]
                arr = np.asarray(da.values)
                while arr.ndim > 2:
                    arr = arr[0]
                return _regrid_latlon_field(arr, la, lo, child_spec, fill_nan_value=default_val)
        return np.full((child_spec.ny, child_spec.nx), default_val, dtype=np.float64)

    msl_h = _get_sfc(("msl", "mslp"), 101_325.0)
    t2m_h = _get_sfc(("t2m", "2t"), 285.0)
    del sfc_vars

    hs = np.asarray(child_hs_m, dtype=np.float64)
    ps_child = msl_h * np.exp(-(GRAVITY_M_S2 * hs) / (R_D * np.maximum(t2m_h, 200.0)))

    state_3d = _vertical_map_isobaric_to_klemp(
        child_spec,
        hs,
        p_levels_pa,
        z_h,
        t_h,
        u_geo_h,
        v_geo_h,
        q_h,
        ps_child,
        theta_sfc_ref=theta_sfc_ref,
        p_sfc_ref=p_sfc_ref,
        n_ref_hz=n_ref_hz,
    )

    data_vars, extra_coords, rim_info = _extract_rim_packed_lbc(child_spec, state_3d, ps_child, rim_width)
    v_dt = _to_utc(valid_time_utc)
    t_coord = np.array([np.datetime64(int(v_dt.timestamp()), "s")], dtype="datetime64[s]")
    coords: Dict[str, Any] = {
        "time": ("time", t_coord),
        "z": ("z", child_spec.vertical.zeta_s.astype(np.float32), {"units": "m"}),
        "z_w": ("z_w", child_spec.vertical.zeta_w.astype(np.float32), {"units": "m"}),
    }
    for ck, cv in extra_coords.items():
        coords[ck] = (ck, cv)

    attrs: Dict[str, str] = {
        "parent_domain": f"ecmwf_ifs_{resol}",
        "child_domain": child_spec.name,
        "step_index": str(int(step_index)),
        "sim_time_sec": f"{float(sim_time_sec):.3f}",
        "valid_time_utc": v_dt.isoformat(),
        "ecmwf_resol": str(resol),
        "child_nz": str(int(child_spec.nz)),
        "child_ny": str(int(child_spec.ny)),
        "child_nx": str(int(child_spec.nx)),
    }
    if ecmwf_cycle_utc is not None:
        attrs["ecmwf_cycle_utc"] = _to_utc(ecmwf_cycle_utc).isoformat()
    if ecmwf_step_hours is not None:
        attrs["ecmwf_step_hours"] = str(int(ecmwf_step_hours))
    if rim_info["use_rim"]:
        attrs["lbc_storage"] = "rim_packed"
        attrs["lbc_rim_width"] = str(int(rim_info["rim_width"]))
        chunks_map = {
            str(vn): [min(int(v[1].shape[0]), 64), min(int(v[1].shape[1]), 65536)]
            for vn, v in data_vars.items()
            if v[1].ndim == 2
        }
    else:
        kz_chunk = max(1, min(int(child_spec.nz), (16 * 1024 * 1024) // max(1, child_spec.ny * child_spec.nx * 4)))
        chunks_map = {
            str(vn): [min(int(v[1].shape[0]), kz_chunk), int(v[1].shape[1]), int(v[1].shape[2])]
            for vn, v in data_vars.items()
            if v[1].ndim == 3
        }

    ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
    wx1.write_dataset(
        out_path,
        ds,
        dataset_id=f"lbc_{child_spec.name}_{int(step_index):04d}",
        codec="zstd_shuffle",
        chunk_shapes=chunks_map,
        workers=4,
    )
    return out_path


def _cached_lbc_wx1_matches(
    path: Path,
    child_spec: DomainSpec,
    valid_dt: datetime.datetime,
    sim_time_sec: float,
) -> bool:
    if not path.is_file() or path.stat().st_size < 1024:
        return False
    try:
        with wx1.open_dataset(path) as ds:
            if ds.attrs.get("child_domain") != child_spec.name:
                return False
            if int(ds.attrs.get("child_ny", -1)) != int(child_spec.ny):
                return False
            if int(ds.attrs.get("child_nx", -1)) != int(child_spec.nx):
                return False
            if int(ds.sizes.get("z", -1)) != int(child_spec.nz):
                return False
            if abs(float(ds.attrs.get("sim_time_sec", "nan")) - float(sim_time_sec)) > 1.0:
                return False
            if ds.attrs.get("valid_time_utc") != _to_utc(valid_dt).isoformat():
                return False
        return True
    except Exception:
        return False


def prepare_ecmwf_lbc_series(
    child_spec: DomainSpec,
    child_hs_m: np.ndarray,
    target_init_utc: datetime.datetime,
    forecast_hours: float,
    lbc_dir: Path | str,
    *,
    resol: str = "auto",
    lbc_interval_hours: float = 3.0,
    source: str = "azure",
    grib_cache_dir: Optional[Path | str] = None,
    need_ic_grib: bool = False,
    rim_width: int = 16,
    plan: Optional[ECMWFLBCPlan] = None,
    latest_fn: Optional[Callable[[str, int], datetime.datetime]] = None,
    download_fn: Optional[Callable[..., Tuple[Path, Path]]] = None,
    theta_sfc_ref: float = 288.15,
    p_sfc_ref: float = 101_325.0,
    n_ref_hz: float = 0.01,
) -> Dict[str, Any]:
    """Download/cache shifted ECMWF Open Data steps and build rim-packed WX-1 LBCs for `child_spec`."""
    eff_plan = plan if plan is not None else resolve_ecmwf_lbc_plan(
        target_init_utc,
        forecast_hours,
        resol=resol,
        lbc_interval_hours=lbc_interval_hours,
        source=source,
        latest_fn=latest_fn,
    )
    out_lbc_dir = Path(lbc_dir)
    out_lbc_dir.mkdir(parents=True, exist_ok=True)
    cycle_stamp = f"{eff_plan.ecmwf_cycle_utc:%Y%m%d_%H}Z"
    cache_root = (
        Path(grib_cache_dir) if grib_cache_dir is not None else (out_lbc_dir / "raw_ecmwf")
    ) / f"{cycle_stamp}_{eff_plan.resol}"
    cache_root.mkdir(parents=True, exist_ok=True)

    dl = download_fn if download_fn is not None else download_ecmwf_data
    run_date = eff_plan.ecmwf_cycle_utc.strftime("%Y-%m-%d")
    run_time = eff_plan.ecmwf_cycle_utc.hour

    lbc_paths: List[Path] = []
    ic_sfc_grib: Optional[Path] = None
    ic_pl_grib: Optional[Path] = None

    for idx, (step_h, sim_s, v_dt) in enumerate(
        zip(eff_plan.ecmwf_steps, eff_plan.sim_times_sec, eff_plan.valid_times_utc)
    ):
        is_ic_step = need_ic_grib and int(step_h) == int(eff_plan.ic_step)
        expected_wx1 = out_lbc_dir / f"lbc_{child_spec.name}_step_{idx:04d}.wx1"
        if not is_ic_step and _cached_lbc_wx1_matches(expected_wx1, child_spec, v_dt, sim_s):
            logger.info("[%s] Reusing cached LBC %s", child_spec.name, expected_wx1.name)
            lbc_paths.append(expected_wx1)
            continue

        sfc_p, pl_p = dl(
            step=int(step_h),
            resol=eff_plan.resol,
            output_dir=cache_root,
            sfc_vars=STANDARD_SFC_VARS if is_ic_step else LBC_SFC_VARS,
            include_soil=bool(is_ic_step),
            source=eff_plan.source,
            run_date=run_date,
            run_time=run_time,
            skip_existing=True,
        )
        if is_ic_step:
            ic_sfc_grib = Path(sfc_p)
            ic_pl_grib = Path(pl_p)

        if _cached_lbc_wx1_matches(expected_wx1, child_spec, v_dt, sim_s):
            lbc_paths.append(expected_wx1)
            continue

        wx1_p = ecmwf_grib_to_lbc_wx1(
            sfc_p,
            pl_p,
            child_spec,
            child_hs_m,
            out_lbc_dir,
            idx,
            sim_s,
            v_dt,
            resol=eff_plan.resol,
            ecmwf_cycle_utc=eff_plan.ecmwf_cycle_utc,
            ecmwf_step_hours=int(step_h),
            rim_width=rim_width,
            theta_sfc_ref=theta_sfc_ref,
            p_sfc_ref=p_sfc_ref,
            n_ref_hz=n_ref_hz,
        )
        lbc_paths.append(wx1_p)

    return {
        "plan": eff_plan,
        "lbc_dir": out_lbc_dir,
        "lbc_paths": lbc_paths,
        "ic_sfc_grib": ic_sfc_grib,
        "ic_pl_grib": ic_pl_grib,
    }
