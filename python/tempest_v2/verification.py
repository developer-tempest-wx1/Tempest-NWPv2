"""Station Observation Loader, 3D-Var Driver, and Live KNMI/DWD Verification for Tempest-NWP v2.

Strictly 100% deterministic classical data assimilation & verification (0% AI / 0% ML, 0% mock data).
"""

from __future__ import annotations

import datetime
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import wx1

from tempest_v2.da.balance import BalanceOperator
from tempest_v2.da.lhn_radar import compute_lhn_tendencies
from tempest_v2.da.var3d import (
    RecursiveSpatialFilter,
    Var3DConfig,
    Var3DResult,
    VarObservation,
    VarObservationOperator,
    apply_analysis_increments_with_bounds,
    compute_iau_tendencies,
    solve_3dvar_pcg,
)
from tempest_v2.ic_builder import stagger_uv, unstagger_uv
from tempest_v2.projection import GRAVITY_M_S2, KAPPA, P0_PA, R_D, DomainSpec, VerticalGrid


def compute_mslp_pa(
    spec: DomainSpec,
    hs_m: np.ndarray,
    state_3d: Dict[str, np.ndarray],
    *,
    theta_sfc_ref: float = 288.15,
    p_sfc_ref: float = 101_325.0,
    n_ref_hz: float = 0.01,
) -> np.ndarray:
    """Compute Mean Sea Level Pressure [Pa] hypsometrically from bottom model layer."""
    hs = np.asarray(hs_m, dtype=np.float64)
    z_0 = float(spec.vertical.zeta_s[0]) + float(spec.vertical.B_s[0]) * hs
    ref0 = VerticalGrid.reference_state(
        z_0[None, :, :],
        theta_sfc_k=theta_sfc_ref,
        p_sfc_pa=p_sfc_ref,
        brunt_vaisala_hz=n_ref_hz,
    )
    th0 = ref0["theta_bar"][0] + np.asarray(state_3d["theta_p"][0], dtype=np.float64)
    pi0 = np.maximum(1e-4, ref0["pi_bar"][0] + np.asarray(state_3d["pi_p"][0], dtype=np.float64))
    t0 = np.maximum(160.0, th0 * pi0)
    p0 = P0_PA * np.power(pi0, 1.0 / KAPPA)
    qv0 = np.maximum(0.0, np.asarray(state_3d.get("qv", np.zeros_like(t0))[0], dtype=np.float64))
    tv0 = t0 * (1.0 + 0.608 * qv0)
    return (p0 * np.exp((GRAVITY_M_S2 * z_0) / (R_D * np.maximum(tv0, 160.0)))).astype(np.float32)


def _read_obs_prefix(obs_dir: Path, prefix: str) -> Optional[Dict[str, np.ndarray]]:
    search_dirs = [
        obs_dir,
        obs_dir / "sfc",
        obs_dir / "upperair",
        obs_dir / "satellite",
        obs_dir / "marine",
    ]
    found_dir: Optional[Path] = None
    main_path: Optional[Path] = None
    for d in search_dirs:
        cand = d / f"{prefix}.wx1"
        if cand.is_file():
            found_dir = d
            main_path = cand
            break
    if main_path is None or found_dir is None:
        return None

    all_keys = (
        "pressure",
        "lat",
        "lon",
        "latitude",
        "longitude",
        "elev",
        "temp",
        "u_wind",
        "v_wind",
        "time",
        "rh",
        "ztd",
        "sigma",
        "sst",
        "ice_conc",
        "ice_thick",
    )
    fields: Dict[str, np.ndarray] = {}
    with wx1.open_dataset(main_path) as ds:
        for k in all_keys:
            if k in ds:
                fields[k] = np.asarray(ds[k].values)
        if not fields and len(ds.data_vars) == 1:
            only_var = next(iter(ds.data_vars))
            fields[str(only_var)] = np.asarray(ds[only_var].values)

    for k in (
        "pressure",
        "lat",
        "lon",
        "elev",
        "temp",
        "u_wind",
        "v_wind",
        "time",
        "rh",
        "ztd",
        "sigma",
        "sst",
        "ice_conc",
        "ice_thick",
    ):
        if k not in fields:
            sub_p = found_dir / f"{prefix}_{k}.wx1"
            if sub_p.is_file():
                with wx1.open_dataset(sub_p) as sds:
                    if k in sds:
                        fields[k] = np.asarray(sds[k].values)
                    elif len(sds.data_vars) > 0:
                        first = next(iter(sds.data_vars))
                        fields[k] = np.asarray(sds[first].values)

    if "lat" not in fields and "latitude" in fields:
        fields["lat"] = fields["latitude"]
    if "lon" not in fields and "longitude" in fields:
        fields["lon"] = fields["longitude"]
    if "lat" not in fields or "lon" not in fields:
        return None
    return fields


def _qsat_water_scalar(t_k: float, p_pa: float) -> float:
    tc = max(173.15, min(373.15, t_k)) - 273.15
    es = 611.21 * math.exp(17.502 * tc / (tc + 240.97))
    es = min(es, 0.90 * p_pa)
    return 0.622 * es / max(100.0, p_pa - 0.378 * es)


def load_station_obs_wx1(
    obs_dir: Path | str,
    spec: DomainSpec,
    *,
    valid_time_utc: Optional[datetime.datetime] = None,
    max_window_hours: float = 6.0,
    theta_sfc_ref: float = 288.15,
    p_sfc_ref: float = 101_325.0,
    n_ref_hz: float = 0.01,
) -> Tuple[List[VarObservation], List[Dict[str, Any]]]:
    """Load QC-filtered surface, upper-air, GNSS ZTD, satellite wind, and marine WX-1 observations inside `spec`.

    Raw-only containers (e.g. `satellite_atms_microwave_radiances.wx1`) are intentionally excluded
    from assimilation in compliance with AGENTS.md Rule 3.
    """
    odir = Path(obs_dir)
    margin_x = 2.0 * spec.dx_m
    margin_y = 2.0 * spec.dy_m
    x_min, x_max = float(spec.x_m[0]) + margin_x, float(spec.x_m[-1]) - margin_x
    y_min, y_max = float(spec.y_m[0]) + margin_y, float(spec.y_m[-1]) - margin_y

    ref_ts: Optional[float] = None
    if valid_time_utc is not None:
        dt_utc = (
            valid_time_utc.astimezone(datetime.timezone.utc)
            if valid_time_utc.tzinfo
            else valid_time_utc.replace(tzinfo=datetime.timezone.utc)
        )
        ref_ts = dt_utc.timestamp()

    obs_list: List[VarObservation] = []
    stn_rows: List[Dict[str, Any]] = []

    def _in_domain_and_window(lat: float, lon: float, t_epoch: float) -> Optional[Tuple[float, float, float]]:
        if not (math.isfinite(lat) and math.isfinite(lon)):
            return None
        if spec.is_projected and spec.projection is not None:
            xm, ym = spec.projection.latlon_to_xy(lat, lon)
            if not (x_min <= float(xm) <= x_max and y_min <= float(ym) <= y_max):
                return None
        else:
            xm, ym = lon, lat
        t_off = 0.0
        if ref_ts is not None and math.isfinite(t_epoch) and t_epoch > 1e8:
            t_off = float(t_epoch - ref_ts)
            if abs(t_off) > max_window_hours * 3600.0:
                return None
        return float(xm), float(ym), t_off

    # 1. Surface synoptic / AWS / METAR / WIS2 / Marine buoy observations
    surface_sources = (
        ("knmi", "knmi_insitu_obs"),
        ("dwd", "dwd_poi_obs"),
        ("wis2", "wis2_obs"),
        ("metar", "global_metars"),
        ("global_sfc", "global_surface_obs"),
        ("marine", "marine_buoy_obs"),
    )
    for provider, prefix in surface_sources:
        data = _read_obs_prefix(odir, prefix)
        if data is None:
            continue
        n = len(data["lat"])
        lats = np.asarray(data["lat"], dtype=np.float64)
        lons = np.asarray(data["lon"], dtype=np.float64)
        elevs = np.asarray(data.get("elev", np.zeros(n)), dtype=np.float64)
        temps = np.asarray(data.get("temp", np.full(n, np.nan)), dtype=np.float64)
        press = np.asarray(data.get("pressure", np.full(n, np.nan)), dtype=np.float64)
        u_ws = np.asarray(data.get("u_wind", np.full(n, np.nan)), dtype=np.float64)
        v_ws = np.asarray(data.get("v_wind", np.full(n, np.nan)), dtype=np.float64)
        ssts = np.asarray(data.get("sst", np.full(n, np.nan)), dtype=np.float64)
        times = np.asarray(data.get("time", np.full(n, np.nan)), dtype=np.float64)

        for idx in range(n):
            lat = float(lats[idx])
            lon = float(lons[idx])
            chk = _in_domain_and_window(lat, lon, float(times[idx]))
            if chk is None:
                continue
            xm, ym, t_off = chk

            elev = float(elevs[idx]) if math.isfinite(float(elevs[idx])) else 0.0
            elev = float(np.clip(elev, -10.0, 4000.0))
            sid = f"{provider}_{idx:04d}"

            t_val = float(temps[idx])
            if math.isfinite(t_val) and 220.0 <= t_val <= 330.0:
                obs_list.append(
                    VarObservation(
                        station_id=sid,
                        var_type="t2m",
                        lat=lat,
                        lon=lon,
                        alt_m=elev,
                        value=t_val,
                        error_variance=1.2**2,
                        time_offset_sec=t_off,
                    )
                )
            else:
                t_val = float("nan")

            p_mslp = float(press[idx])
            if math.isfinite(p_mslp) and 92_000.0 <= p_mslp <= 106_500.0:
                t_col = t_val if math.isfinite(t_val) else 285.0
                p_stn = p_mslp * math.exp(-GRAVITY_M_S2 * elev / (R_D * t_col))
                obs_list.append(
                    VarObservation(
                        station_id=sid,
                        var_type="ps",
                        lat=lat,
                        lon=lon,
                        alt_m=elev,
                        value=p_stn,
                        error_variance=120.0**2,
                        time_offset_sec=t_off,
                    )
                )
            else:
                p_mslp = float("nan")

            sst_val = float(ssts[idx])
            if math.isfinite(sst_val) and 271.0 <= sst_val <= 310.0:
                obs_list.append(
                    VarObservation(
                        station_id=sid,
                        var_type="sst",
                        lat=lat,
                        lon=lon,
                        alt_m=0.0,
                        value=sst_val,
                        error_variance=0.8**2,
                        time_offset_sec=t_off,
                    )
                )

            u_v = float(u_ws[idx])
            v_v = float(v_ws[idx])
            if math.isfinite(u_v) and math.isfinite(v_v) and abs(u_v) <= 60.0 and abs(v_v) <= 60.0:
                obs_list.append(
                    VarObservation(
                        station_id=sid,
                        var_type="u_geo",
                        lat=lat,
                        lon=lon,
                        alt_m=elev,
                        value=u_v,
                        error_variance=1.8**2,
                        time_offset_sec=t_off,
                    )
                )
                obs_list.append(
                    VarObservation(
                        station_id=sid,
                        var_type="v_geo",
                        lat=lat,
                        lon=lon,
                        alt_m=elev,
                        value=v_v,
                        error_variance=1.8**2,
                        time_offset_sec=t_off,
                    )
                )
            else:
                u_v, v_v = float("nan"), float("nan")

            if provider in ("knmi", "dwd", "wis2", "metar", "marine"):
                stn_rows.append(
                    {
                        "provider": provider,
                        "station_id": sid,
                        "lat": lat,
                        "lon": lon,
                        "x_m": float(xm),
                        "y_m": float(ym),
                        "elev_m": elev,
                        "t2m_k": t_val,
                        "mslp_pa": p_mslp,
                        "u10_ms": u_v,
                        "v10_ms": v_v,
                    }
                )

    # 2. Upper-Air Radiosonde & SondeHub vertical soundings (theta_p, u_geo, v_geo, qv at pressure_pa)
    for provider, prefix in (("radiosonde", "radiosonde_obs"), ("sondehub", "sondehub_soundings")):
        data = _read_obs_prefix(odir, prefix)
        if data is None or "pressure" not in data:
            continue
        n = len(data["lat"])
        lats = np.asarray(data["lat"], dtype=np.float64)
        lons = np.asarray(data["lon"], dtype=np.float64)
        press = np.asarray(data["pressure"], dtype=np.float64)
        temps = np.asarray(data.get("temp", np.full(n, np.nan)), dtype=np.float64)
        rhs = np.asarray(data.get("rh", np.full(n, np.nan)), dtype=np.float64)
        u_ws = np.asarray(data.get("u_wind", np.full(n, np.nan)), dtype=np.float64)
        v_ws = np.asarray(data.get("v_wind", np.full(n, np.nan)), dtype=np.float64)
        times = np.asarray(data.get("time", np.full(n, np.nan)), dtype=np.float64)

        for idx in range(n):
            lat, lon, p_pa = float(lats[idx]), float(lons[idx]), float(press[idx])
            if not (math.isfinite(p_pa) and 5_000.0 <= p_pa <= 105_000.0):
                continue
            chk = _in_domain_and_window(lat, lon, float(times[idx]))
            if chk is None:
                continue
            _, _, t_off = chk
            sid = f"{provider}_{idx:05d}"

            # Estimate standard-atmosphere height for reference theta_bar(z)
            z_approx = max(0.0, -7400.0 * math.log(max(p_pa / p_sfc_ref, 1e-4)))
            ref_z = VerticalGrid.reference_state(
                np.array([z_approx], dtype=np.float64),
                theta_sfc_k=theta_sfc_ref,
                p_sfc_pa=p_sfc_ref,
                brunt_vaisala_hz=n_ref_hz,
            )
            th_bar_lvl = float(ref_z["theta_bar"][0])

            t_val = float(temps[idx])
            if math.isfinite(t_val) and 180.0 <= t_val <= 330.0:
                pi_obs = (p_pa / P0_PA) ** KAPPA
                theta_p_obs = (t_val / max(pi_obs, 1e-4)) - th_bar_lvl
                obs_list.append(
                    VarObservation(
                        station_id=sid,
                        var_type="theta_p",
                        lat=lat,
                        lon=lon,
                        pressure_pa=p_pa,
                        value=theta_p_obs,
                        error_variance=1.5**2,
                        time_offset_sec=t_off,
                    )
                )
                rh_val = float(rhs[idx])
                if math.isfinite(rh_val) and 0.0 <= rh_val <= 100.0 and p_pa >= 25_000.0:
                    qv_obs = (0.01 * rh_val) * _qsat_water_scalar(t_val, p_pa)
                    obs_list.append(
                        VarObservation(
                            station_id=sid,
                            var_type="qv",
                            lat=lat,
                            lon=lon,
                            pressure_pa=p_pa,
                            value=qv_obs,
                            error_variance=0.0012**2,
                            time_offset_sec=t_off,
                        )
                    )

            u_v, v_v = float(u_ws[idx]), float(v_ws[idx])
            if math.isfinite(u_v) and math.isfinite(v_v) and abs(u_v) <= 110.0 and abs(v_v) <= 110.0:
                obs_list.append(
                    VarObservation(
                        station_id=sid,
                        var_type="u_geo",
                        lat=lat,
                        lon=lon,
                        pressure_pa=p_pa,
                        value=u_v,
                        error_variance=2.2**2,
                        time_offset_sec=t_off,
                    )
                )
                obs_list.append(
                    VarObservation(
                        station_id=sid,
                        var_type="v_geo",
                        lat=lat,
                        lon=lon,
                        pressure_pa=p_pa,
                        value=v_v,
                        error_variance=2.2**2,
                        time_offset_sec=t_off,
                    )
                )

    # 3. European E-GVAP Ground GNSS Zenith Total Delay (ZTD)
    ztd_data = _read_obs_prefix(odir, "egvap_gnss_ztd")
    if ztd_data is not None and "ztd" in ztd_data:
        n = len(ztd_data["lat"])
        lats = np.asarray(ztd_data["lat"], dtype=np.float64)
        lons = np.asarray(ztd_data["lon"], dtype=np.float64)
        elevs = np.asarray(ztd_data.get("elev", np.zeros(n)), dtype=np.float64)
        ztds = np.asarray(ztd_data["ztd"], dtype=np.float64)
        sigmas = np.asarray(ztd_data.get("sigma", np.full(n, 0.012)), dtype=np.float64)
        times = np.asarray(ztd_data.get("time", np.full(n, np.nan)), dtype=np.float64)
        for idx in range(n):
            lat, lon, z_val = float(lats[idx]), float(lons[idx]), float(ztds[idx])
            if not (math.isfinite(z_val) and 1.5 <= z_val <= 3.0):
                continue
            chk = _in_domain_and_window(lat, lon, float(times[idx]))
            if chk is None:
                continue
            _, _, t_off = chk
            sig = float(sigmas[idx]) if math.isfinite(float(sigmas[idx])) else 0.012
            sig = max(0.006, min(0.050, sig))
            obs_list.append(
                VarObservation(
                    station_id=f"egvap_{idx:04d}",
                    var_type="ztd",
                    lat=lat,
                    lon=lon,
                    alt_m=float(np.clip(elevs[idx] if math.isfinite(float(elevs[idx])) else 0.0, -10.0, 4000.0)),
                    value=z_val,
                    error_variance=sig**2,
                    time_offset_sec=t_off,
                )
            )

    # 4. Spaceborne AMV & Scatterometer Winds (EUMETSAT MTG, Metop ASCAT, GOES, Himawari)
    sat_sources = (
        ("eumetsat_amv", "eumetsat"),
        ("ascat", "eumetsat_ascat"),
        ("goes_amv", "satellite_goes_amv_winds"),
        ("himawari_amv", "satellite_jma_himawari9_winds"),
    )
    for provider, prefix in sat_sources:
        data = _read_obs_prefix(odir, prefix)
        if data is None:
            continue
        n = len(data["lat"])
        lats = np.asarray(data["lat"], dtype=np.float64)
        lons = np.asarray(data["lon"], dtype=np.float64)
        press = np.asarray(data.get("pressure", np.full(n, np.nan)), dtype=np.float64)
        u_ws = np.asarray(data.get("u_wind", np.full(n, np.nan)), dtype=np.float64)
        v_ws = np.asarray(data.get("v_wind", np.full(n, np.nan)), dtype=np.float64)
        times = np.asarray(data.get("time", np.full(n, np.nan)), dtype=np.float64)
        for idx in range(n):
            lat, lon = float(lats[idx]), float(lons[idx])
            chk = _in_domain_and_window(lat, lon, float(times[idx]))
            if chk is None:
                continue
            _, _, t_off = chk
            u_v, v_v = float(u_ws[idx]), float(v_ws[idx])
            if not (math.isfinite(u_v) and math.isfinite(v_v) and abs(u_v) <= 110.0 and abs(v_v) <= 110.0):
                continue
            p_pa = float(press[idx])
            p_opt = p_pa if (math.isfinite(p_pa) and 5_000.0 <= p_pa <= 105_000.0) else None
            sid = f"{provider}_{idx:05d}"
            obs_list.append(
                VarObservation(
                    station_id=sid,
                    var_type="u_geo",
                    lat=lat,
                    lon=lon,
                    alt_m=10.0,
                    pressure_pa=p_opt,
                    value=u_v,
                    error_variance=2.5**2,
                    time_offset_sec=t_off,
                )
            )
            obs_list.append(
                VarObservation(
                    station_id=sid,
                    var_type="v_geo",
                    lat=lat,
                    lon=lon,
                    alt_m=10.0,
                    pressure_pa=p_opt,
                    value=v_v,
                    error_variance=2.5**2,
                    time_offset_sec=t_off,
                )
            )

    return obs_list, stn_rows


def run_domain_3dvar(
    spec: DomainSpec,
    state_3d: Dict[str, np.ndarray],
    surf_state: Dict[str, np.ndarray],
    hs_m: np.ndarray,
    lsm: np.ndarray,
    obs_list: List[VarObservation],
    *,
    theta_sfc_ref: float = 288.15,
    p_sfc_ref: float = 101_325.0,
    n_ref_hz: float = 0.01,
    length_scale_h_m: float = 45_000.0,
    max_iterations: int = 25,
    iau_tau_sec: float = 900.0,
    trajectory: Optional[Dict[float, Dict[str, np.ndarray]]] = None,
    obs_dir: Optional[Path | str] = None,
    radar_rain_rate_mm_h: Optional[np.ndarray] = None,
    model_rain_rate_mm_h: Optional[np.ndarray] = None,
    dtheta_dt_mp: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], Var3DResult]:
    """Run balanced 3D-Var / 4D-FGAT + Radar LHN + OSI-SAF cryosphere update and return analysis + IAU forcing."""
    hs = np.asarray(hs_m, dtype=np.float64)
    u_c, v_c = unstagger_uv(state_3d["u"], state_3d["v"])

    _, z_s, _ = spec.vertical.heights_3d(hs)
    ref0 = VerticalGrid.reference_state(
        z_s[:1], theta_sfc_k=theta_sfc_ref, p_sfc_pa=p_sfc_ref, brunt_vaisala_hz=n_ref_hz
    )
    th0 = ref0["theta_bar"][0] + np.asarray(state_3d["theta_p"][0], dtype=np.float64)
    pi0 = np.maximum(1e-4, ref0["pi_bar"][0] + np.asarray(state_3d["pi_p"][0], dtype=np.float64))
    p0 = P0_PA * np.power(pi0, 1.0 / KAPPA)
    tv0 = np.maximum(160.0, th0 * pi0 * (1.0 + 0.608 * np.maximum(0.0, np.asarray(state_3d["qv"][0], dtype=np.float64))))
    dz_sfc = np.maximum(0.0, z_s[0] - hs)
    ps_bg = p0 * np.exp((GRAVITY_M_S2 * dz_sfc) / (R_D * tv0))

    bg_centered: Dict[str, np.ndarray] = {
        "u": u_c,
        "v": v_c,
        "theta_p": np.asarray(state_3d["theta_p"], dtype=np.float32),
        "pi_p": np.asarray(state_3d["pi_p"], dtype=np.float32),
        "qv": np.asarray(state_3d["qv"], dtype=np.float32),
        "ps": ps_bg.astype(np.float32),
    }
    if "sst_bulk" in surf_state:
        bg_centered["sst"] = np.asarray(surf_state["sst_bulk"], dtype=np.float32)

    dz_avg = float(spec.vertical.z_top_m) / max(1, spec.nz)

    # Courtier et al. (1994) Multi-Incremental 3D-Var inner loop when horizontal grid >= 250k columns
    if spec.nx * spec.ny >= 250_000 and trajectory is None and spec.nx % 4 == 0 and spec.ny % 4 == 0:
        from dataclasses import replace
        from scipy.interpolate import RegularGridInterpolator

        stride = 4
        c_spec = replace(
            spec,
            ny=spec.ny // stride,
            nx=spec.nx // stride,
            dx_m=spec.dx_m * stride,
            dy_m=spec.dy_m * stride,
            x_m=spec.x_m[::stride].copy(),
            y_m=spec.y_m[::stride].copy(),
            lat2d=spec.lat2d[::stride, ::stride].copy(),
            lon2d=spec.lon2d[::stride, ::stride].copy(),
            map_factor=spec.map_factor[::stride, ::stride].copy(),
            coriolis_f=spec.coriolis_f[::stride, ::stride].copy(),
        )
        c_hs = hs[::stride, ::stride].copy()
        c_lsm = np.asarray(lsm)[::stride, ::stride] >= 0.5
        c_bg = {
            k: (v[:, ::stride, ::stride].copy() if v.ndim == 3 else v[::stride, ::stride].copy())
            for k, v in bg_centered.items()
        }
        c_hop = VarObservationOperator(
            c_spec, c_hs, c_bg, theta_sfc_ref=theta_sfc_ref, p_sfc_ref=p_sfc_ref, n_ref_hz=n_ref_hz
        )
        c_filter = RecursiveSpatialFilter(
            c_spec.dx_m, c_spec.dy_m, dz_m=dz_avg, length_scale_h_m=length_scale_h_m, length_scale_v_m=2_000.0, num_passes=2
        )
        c_bal = BalanceOperator(
            c_spec, c_hs, theta_sfc_ref=theta_sfc_ref, p_sfc_ref=p_sfc_ref, n_ref_hz=n_ref_hz
        )
        cfg = Var3DConfig(max_iterations=max_iterations, use_balance=True, iau_window_sec=iau_tau_sec)
        c_res = solve_3dvar_pcg(c_bg, obs_list, c_hop, c_filter, cfg, balance_op=c_bal, land_mask=c_lsm)

        # Bilinear prolongation of smooth L_h=45km inner-loop increments onto 1.5km outer grid
        yy_f, xx_f = np.meshgrid(spec.y_m, spec.x_m, indexing="ij")
        pts_f = np.column_stack([yy_f.ravel(), xx_f.ravel()])
        fine_inc: Dict[str, np.ndarray] = {}
        for k, arr_c in c_res.analysis_increments.items():
            if not np.any(arr_c):
                fine_inc[k] = np.zeros_like(bg_centered[k], dtype=np.float64)
                continue
            if arr_c.ndim == 2:
                itp = RegularGridInterpolator((c_spec.y_m, c_spec.x_m), arr_c, method="linear", bounds_error=False, fill_value=None)
                fine_inc[k] = itp(pts_f).reshape(spec.ny, spec.nx)
            else:
                out_3d = np.zeros((arr_c.shape[0], spec.ny, spec.nx), dtype=np.float64)
                for kz in range(arr_c.shape[0]):
                    if np.any(arr_c[kz]):
                        itp = RegularGridInterpolator((c_spec.y_m, c_spec.x_m), arr_c[kz], method="linear", bounds_error=False, fill_value=None)
                        out_3d[kz] = itp(pts_f).reshape(spec.ny, spec.nx)
                fine_inc[k] = out_3d
        res = Var3DResult(
            analysis_increments=fine_inc,
            initial_cost_j0=c_res.initial_cost_j0,
            final_cost_j=c_res.final_cost_j,
            cost_j_b=c_res.cost_j_b,
            cost_j_o=c_res.cost_j_o,
            iterations=c_res.iterations,
            converged=c_res.converged,
        )
    else:
        hop = VarObservationOperator(
            spec, hs, bg_centered, theta_sfc_ref=theta_sfc_ref, p_sfc_ref=p_sfc_ref, n_ref_hz=n_ref_hz
        )
        spatial_filter = RecursiveSpatialFilter(
            spec.dx_m, spec.dy_m, dz_m=dz_avg, length_scale_h_m=length_scale_h_m, length_scale_v_m=2_000.0, num_passes=2
        )
        balance_op = BalanceOperator(
            spec, hs, theta_sfc_ref=theta_sfc_ref, p_sfc_ref=p_sfc_ref, n_ref_hz=n_ref_hz
        )
        cfg = Var3DConfig(max_iterations=max_iterations, use_balance=True, iau_window_sec=iau_tau_sec)
        res = solve_3dvar_pcg(
            bg_centered,
            obs_list,
            hop,
            spatial_filter,
            cfg,
            balance_op=balance_op,
            land_mask=(np.asarray(lsm) >= 0.5),
            trajectory=trajectory,
        )

    updated_c = apply_analysis_increments_with_bounds(bg_centered, res.analysis_increments)
    u_st, v_st = stagger_uv(updated_c["u"], updated_c["v"])

    ana_state = {k: v.copy() for k, v in state_3d.items()}
    ana_state["u"] = u_st
    ana_state["v"] = v_st
    ana_state["theta_p"] = updated_c["theta_p"]
    ana_state["pi_p"] = updated_c["pi_p"]
    ana_state["qv"] = updated_c["qv"]

    # Adjust 2m temperature, skin temperature, and SST consistently with 3D-Var increments
    th0_ana = ref0["theta_bar"][0] + np.asarray(ana_state["theta_p"][0], dtype=np.float64)
    pi0_ana = np.maximum(1e-4, ref0["pi_bar"][0] + np.asarray(ana_state["pi_p"][0], dtype=np.float64))
    t0_ana = th0_ana * pi0_ana
    dt0 = (t0_ana - th0 * pi0).astype(np.float32)
    ana_surf = {k: v.copy() for k, v in surf_state.items()}
    if "t2m" in ana_surf and float(np.mean(np.abs(ana_surf["t2m"]))) > 150.0:
        ana_surf["t2m"] = t0_ana.astype(np.float32)
    if "t_skin" in ana_surf:
        is_land = np.asarray(lsm) >= 0.5
        ana_surf["t_skin"] = np.where(is_land, ana_surf["t_skin"] + 0.5 * dt0, ana_surf["t_skin"]).astype(np.float32)
    if "sst" in updated_c and "sst_bulk" in ana_surf:
        ana_surf["sst_bulk"] = updated_c["sst"].astype(np.float32)

    # Apply OSI-SAF cryosphere sea-ice concentration / thickness when available in obs_dir
    if obs_dir is not None:
        osi = _read_obs_prefix(Path(obs_dir), "osi_saf_sea_ice")
        if osi is not None and "ice_conc" in osi:
            ic_vals = np.asarray(osi["ice_conc"], dtype=np.float64)
            ic_lats = np.asarray(osi["lat"], dtype=np.float64)
            ic_lons = np.asarray(osi["lon"], dtype=np.float64)
            ice_mask = np.isfinite(ic_vals) & (ic_vals >= 0.15)
            if np.any(ice_mask) and spec.is_projected and spec.projection is not None:
                xm, ym = spec.projection.latlon_to_xy(ic_lats[ice_mask], ic_lons[ice_mask])
                ii = np.round((xm - float(spec.x_m[0])) / spec.dx_m).astype(np.int64)
                jj = np.round((ym - float(spec.y_m[0])) / spec.dy_m).astype(np.int64)
                valid_ij = (ii >= 0) & (ii < spec.nx) & (jj >= 0) & (jj < spec.ny)
                for j_pt, i_pt in zip(jj[valid_ij], ii[valid_ij]):
                    if float(lsm[j_pt, i_pt]) < 0.5 and "sst_bulk" in ana_surf:
                        ana_surf["sst_bulk"][j_pt, i_pt] = np.float32(271.35)

    inc_c = {
        "u": res.analysis_increments.get("u", np.zeros_like(u_c)),
        "v": res.analysis_increments.get("v", np.zeros_like(v_c)),
        "theta_p": res.analysis_increments.get("theta_p", np.zeros_like(ana_state["theta_p"])),
        "pi_p": res.analysis_increments.get("pi_p", np.zeros_like(ana_state["pi_p"])),
    }
    iau_raw = compute_iau_tendencies(inc_c, tau_sec=iau_tau_sec)

    # Radar Latent Heat Nudging (LHN) via compute_lhn_tendencies
    if radar_rain_rate_mm_h is None and obs_dir is not None:
        rad_p = Path(obs_dir) / "radar_obs.wx1"
        if rad_p.is_file():
            with wx1.open_dataset(rad_p) as rds:
                vname = "rain_rate_mm_h" if "rain_rate_mm_h" in rds else next(iter(rds.data_vars), None)
                if vname is not None:
                    arr_r = np.asarray(rds[vname].values, dtype=np.float64)
                    if arr_r.shape == (spec.ny, spec.nx):
                        radar_rain_rate_mm_h = arr_r

    if radar_rain_rate_mm_h is not None:
        mod_rr_mm_h = (
            np.asarray(model_rain_rate_mm_h, dtype=np.float64)
            if model_rain_rate_mm_h is not None
            else np.zeros((spec.ny, spec.nx), dtype=np.float64)
        )
        ref_full = VerticalGrid.reference_state(
            z_s, theta_sfc_k=theta_sfc_ref, p_sfc_pa=p_sfc_ref, brunt_vaisala_hz=n_ref_hz
        )
        dth_lhn, dqv_lhn = compute_lhn_tendencies(
            np.asarray(radar_rain_rate_mm_h, dtype=np.float64) / 3600.0,
            mod_rr_mm_h / 3600.0,
            z_s,
            hs,
            ref_full["rho_bar"],
            ana_state["theta_p"],
            ref_full["pi_bar"],
            ana_state["qv"],
            qc=ana_state.get("qc"),
            tau_lhn_sec=iau_tau_sec,
        )
        iau_raw["theta_p"] = (iau_raw["theta_p"] + dth_lhn).astype(np.float32)
        ana_state["qv"] = np.clip(ana_state["qv"] + dqv_lhn * float(iau_tau_sec), 0.0, 0.045).astype(np.float32)

    iau_u_st, iau_v_st = stagger_uv(iau_raw["u"], iau_raw["v"])
    iau_forcing = {
        "u": iau_u_st,
        "v": iau_v_st,
        "w": np.zeros_like(ana_state["w"]),
        "theta_p": iau_raw["theta_p"],
        "pi_p": iau_raw["pi_p"],
    }
    return ana_state, ana_surf, iau_forcing, res


def _bilinear_sample_2d(field: np.ndarray, spec: DomainSpec, x_pts: np.ndarray, y_pts: np.ndarray) -> np.ndarray:
    fx = (x_pts - float(spec.x_m[0])) / spec.dx_m
    fy = (y_pts - float(spec.y_m[0])) / spec.dy_m
    fx = np.clip(fx, 0.0, spec.nx - 1.000001)
    fy = np.clip(fy, 0.0, spec.ny - 1.000001)
    i0 = np.floor(fx).astype(np.int64)
    j0 = np.floor(fy).astype(np.int64)
    i1 = np.minimum(i0 + 1, spec.nx - 1)
    j1 = np.minimum(j0 + 1, spec.ny - 1)
    wx = fx - i0
    wy = fy - j0
    return (
        (1.0 - wy) * (1.0 - wx) * field[j0, i0]
        + (1.0 - wy) * wx * field[j0, i1]
        + wy * (1.0 - wx) * field[j1, i0]
        + wy * wx * field[j1, i1]
    )


def _metric_summary(obs: np.ndarray, mod: np.ndarray) -> Tuple[int, float, float, float]:
    valid = np.isfinite(obs) & np.isfinite(mod)
    if not np.any(valid):
        return 0, float("nan"), float("nan"), float("nan")
    diff = mod[valid] - obs[valid]
    return (
        int(valid.sum()),
        float(np.mean(diff)),
        float(np.sqrt(np.mean(diff * diff))),
        float(np.mean(np.abs(diff))),
    )


def verify_against_stations(
    spec: DomainSpec,
    state_3d: Dict[str, np.ndarray],
    surf_state: Dict[str, np.ndarray],
    hs_m: np.ndarray,
    station_rows: Sequence[Dict[str, Any]],
    *,
    theta_sfc_ref: float = 288.15,
    p_sfc_ref: float = 101_325.0,
    n_ref_hz: float = 0.01,
) -> Dict[str, Dict[str, float]]:
    """Evaluate bias, RMSE, and MAE against KNMI & DWD stations inside `spec`."""
    if not station_rows:
        empty = {
            "n_t2m": 0, "t2m_bias_k": float("nan"), "t2m_rmse_k": float("nan"), "t2m_mae_k": float("nan"),
            "n_mslp": 0, "mslp_bias_hpa": float("nan"), "mslp_rmse_hpa": float("nan"),
            "n_wind": 0, "wind_bias_ms": float("nan"), "wind_rmse_ms": float("nan"),
        }
        return {"all": dict(empty), "knmi": dict(empty), "dwd": dict(empty)}

    hs = np.asarray(hs_m, dtype=np.float64)
    z0 = float(spec.vertical.zeta_s[0]) + float(spec.vertical.B_s[0]) * hs
    ref0 = VerticalGrid.reference_state(
        z0[None, :, :], theta_sfc_k=theta_sfc_ref, p_sfc_pa=p_sfc_ref, brunt_vaisala_hz=n_ref_hz
    )
    th0 = ref0["theta_bar"][0] + np.asarray(state_3d["theta_p"][0], dtype=np.float64)
    pi0 = np.maximum(1e-4, ref0["pi_bar"][0] + np.asarray(state_3d["pi_p"][0], dtype=np.float64))
    t2m_grid = th0 * pi0
    mslp_grid = compute_mslp_pa(spec, hs, state_3d, theta_sfc_ref=theta_sfc_ref, p_sfc_ref=p_sfc_ref, n_ref_hz=n_ref_hz).astype(np.float64)

    uc, vc = unstagger_uv(state_3d["u"], state_3d["v"])
    u_grid, v_grid = uc[0].astype(np.float64), vc[0].astype(np.float64)
    if spec.is_projected and spec.projection is not None:
        u_geo_grid, v_geo_grid = spec.projection.grid_to_geo_wind(u_grid, v_grid, spec.lon2d)
    else:
        u_geo_grid, v_geo_grid = u_grid, v_grid

    x_pts = np.array([r["x_m"] for r in station_rows], dtype=np.float64)
    y_pts = np.array([r["y_m"] for r in station_rows], dtype=np.float64)
    elevs = np.array([r["elev_m"] for r in station_rows], dtype=np.float64)
    provs = np.array([r["provider"] for r in station_rows])

    hs_stn = _bilinear_sample_2d(hs, spec, x_pts, y_pts)
    mod_t2m = _bilinear_sample_2d(t2m_grid, spec, x_pts, y_pts) - 0.0065 * (elevs - hs_stn)
    mod_mslp_hpa = _bilinear_sample_2d(mslp_grid, spec, x_pts, y_pts) / 100.0
    mod_u = _bilinear_sample_2d(u_geo_grid, spec, x_pts, y_pts)
    mod_v = _bilinear_sample_2d(v_geo_grid, spec, x_pts, y_pts)
    mod_wspd = np.hypot(mod_u, mod_v)

    obs_t2m = np.array([r["t2m_k"] for r in station_rows], dtype=np.float64)
    obs_mslp_hpa = np.array([r["mslp_pa"] for r in station_rows], dtype=np.float64) / 100.0
    obs_wspd = np.hypot(
        np.array([r["u10_ms"] for r in station_rows], dtype=np.float64),
        np.array([r["v10_ms"] for r in station_rows], dtype=np.float64),
    )

    out: Dict[str, Dict[str, float]] = {}
    for key in ("all", "knmi", "dwd"):
        m = np.ones(len(station_rows), dtype=bool) if key == "all" else (provs == key)
        nt, tb, tr, tmae = _metric_summary(obs_t2m[m], mod_t2m[m])
        np_n, pb, pr, _ = _metric_summary(obs_mslp_hpa[m], mod_mslp_hpa[m])
        nw, wb, wr, _ = _metric_summary(obs_wspd[m], mod_wspd[m])
        out[key] = {
            "n_t2m": nt,
            "t2m_bias_k": tb,
            "t2m_rmse_k": tr,
            "t2m_mae_k": tmae,
            "n_mslp": np_n,
            "mslp_bias_hpa": pb,
            "mslp_rmse_hpa": pr,
            "n_wind": nw,
            "wind_bias_ms": wb,
            "wind_rmse_ms": wr,
        }
    return out
