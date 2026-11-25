"""Resumable Multi-Season, Multi-Weather-Type ARCO-ERA5 Hindcast & Dual Validation Suite.

Runs 24-hour classical physics-based Tempest-NWP hindcasts initialized from public ARCO-ERA5 across
diverse Dutch weather regimes and all 4 seasons, and scores each run against BOTH:
  1. Official KNMI hourly validated station observations (T2m, Td2m, RH, MSLP, Wind, Gust, SW_down,
     TCC, 1h precip, fog, snow, hail/graupel, thunder, freezing rain/ice), and
  2. Public ARCO-ERA5 0.25-deg surface & 3D upper-air (850/500 hPa) reference fields.

Strictly 100% classical deterministic NWP verification (0% AI / 0% ML).
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PY_ROOT = PROJECT_ROOT / "python"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from select_era5_cases import DEFAULT_CASES_DIR, rank_knmi_cases  # noqa: E402
from tempest_v2.ingest_era5 import _extract_lon_wrapped_subset, open_arco_era5_dataset  # noqa: E402
from tempest_v2.knmi_verify import (  # noqa: E402
    _ERA5_VERIFY_2D_PAIRS,
    _sanitize_era5_slice_for_wx1,
    verify_run_against_knmi,
)
from tempest_v2.runner import run_era5_hindcast_lam_pipeline  # noqa: E402
import wx1  # noqa: E402

logger = logging.getLogger("tempest_v2.benchmark_suite")

BENCHMARK_CATALOG: Tuple[Dict[str, str], ...] = (
    {"id": "winter_storm_eunice", "date": "2022-02-18", "season": "winter", "regime": "storm", "description": "Storm Eunice (gusts up to 40 m/s)"},
    {"id": "winter_storm_henk", "date": "2024-01-02", "season": "winter", "regime": "storm", "description": "Storm Henk (35 m/s gusts + heavy frontal rain)"},
    {"id": "summer_storm_poly", "date": "2023-07-05", "season": "summer", "regime": "summer_storm", "description": "Summer Storm Poly (41 m/s coastal gusts + thunder)"},
    {"id": "deep_conv_hail_supercell", "date": "2016-06-23", "season": "summer", "regime": "high_conv_thunder", "description": "Brabant giant-hail supercell (131 thunder hrs, 49.6 mm)"},
    {"id": "deep_conv_spring_mcs", "date": "2018-05-29", "season": "spring", "regime": "high_conv_thunder", "description": "Spring deep convection MCS (136 thunder hrs)"},
    {"id": "deep_conv_heat_thunder", "date": "2026-06-27", "season": "summer", "regime": "high_conv_thunder", "description": "Hot convective outbreak (36.5C, 129 thunder hrs)"},
    {"id": "high_conv_no_thunder_jun", "date": "2024-06-18", "season": "summer", "regime": "high_conv_no_thunder", "description": "Warm localized heavy showers just below thunder (34.5 mm, 0 thunder)"},
    {"id": "high_conv_no_thunder_aug", "date": "2015-08-16", "season": "summer", "regime": "high_conv_no_thunder", "description": "Warm convective showers without thunder (38.9 mm max, 0 thunder)"},
    {"id": "low_conv_thunder_nov", "date": "2024-11-22", "season": "autumn", "regime": "low_conv_thunder", "description": "Low-topped cold polar air convection with thunder & snow (47 thunder, 79 snow)"},
    {"id": "low_conv_thunder_feb", "date": "2020-02-10", "season": "winter", "regime": "low_conv_thunder", "description": "Winter trough low-topped convection with thunder & hail (42 thunder)"},
    {"id": "snowstorm_darcy", "date": "2021-02-07", "season": "winter", "regime": "snow", "description": "Snowstorm Darcy easterly blizzard (495 snow hrs, -5.9C)"},
    {"id": "snow_winter_2026", "date": "2026-01-07", "season": "winter", "regime": "snow", "description": "January 2026 widespread snow event (386 snow hrs)"},
    {"id": "severe_frost_snowpack", "date": "2021-02-09", "season": "winter", "regime": "severe_frost", "description": "Severe radiative frost over fresh snowpack (-14.5C min)"},
    {"id": "freezing_rain_ijzel", "date": "2016-01-05", "season": "winter", "regime": "freezing_mixed", "description": "North-NL freezing rain / ijzel transition (-4.4 to +8.3C)"},
    {"id": "mixed_snow_rain_spring", "date": "2023-03-10", "season": "spring", "regime": "freezing_mixed", "description": "Early-spring heavy wet snow to rain transition (243 snow hrs, 13.4 mm)"},
    {"id": "heavy_rain_limburg", "date": "2021-07-14", "season": "summer", "regime": "rain", "description": "Cut-off low extreme rain Limburg (40.6 mm at KNMI AWS)"},
    {"id": "radiation_fog_winter", "date": "2024-12-27", "season": "winter", "regime": "radiation_fog", "description": "Persistent winter radiation fog (490 fog hrs, 1.7 m/s mean wind)"},
    {"id": "radiation_fog_autumn", "date": "2015-11-02", "season": "autumn", "regime": "radiation_fog", "description": "Autumn radiation fog event (433 fog hrs, -0.3 to 15.9C)"},
    {"id": "clear_diurnal_spring", "date": "2025-03-19", "season": "spring", "regime": "clear_diurnal", "description": "Cloudless high-diurnal spring day (-4.7 to +18.6C, 0 rain)"},
    {"id": "heatwave_record", "date": "2019-07-25", "season": "summer", "regime": "heatwave", "description": "All-time Dutch national heatwave peak (+40.3C, dry)"},
)

_CATALOG_BY_ID: Dict[str, Dict[str, str]] = {c["id"]: dict(c) for c in BENCHMARK_CATALOG}

PRESETS: Dict[str, Tuple[str, ...]] = {
    "quick4": ("radiation_fog_winter", "clear_diurnal_spring", "deep_conv_hail_supercell", "low_conv_thunder_nov"),
    "core10": (
        "winter_storm_eunice",
        "summer_storm_poly",
        "deep_conv_hail_supercell",
        "high_conv_no_thunder_jun",
        "low_conv_thunder_nov",
        "heavy_rain_limburg",
        "snowstorm_darcy",
        "freezing_rain_ijzel",
        "radiation_fog_winter",
        "clear_diurnal_spring",
    ),
    "full20": tuple(c["id"] for c in BENCHMARK_CATALOG),
}

_EVENT_KEYS: Tuple[str, ...] = (
    "fog",
    "rain_0p1mm",
    "rain_2mm",
    "gust_15ms",
    "gust_20ms",
    "snow",
    "hail_graupel",
    "thunder",
    "thunder_25km",
    "ice_freezing_rain",
)

_METRIC_KEYS: Tuple[str, ...] = (
    "t2m_rmse",
    "t2m_bias",
    "t2m_day_bias",
    "t2m_ngt_bias",
    "td2m_rmse",
    "td2m_bias",
    "rh_bias",
    "mslp_rmse",
    "mslp_bias",
    "ws10_rmse",
    "ws10_bias",
    "gust10_rmse",
    "gust10_bias",
    "sw_down_rmse",
    "sw_down_bias",
    "tcc_rmse",
    "tcc_bias",
    "precip_rmse",
    "precip_bias",
    "ss_t2m_era5",
    "ss_mslp_era5",
    "ss_ws10_era5",
    "ss_precip_era5",
    "t850_rmse",
    "gh500_rmse",
    "wspd850_rmse",
)


def _safe_float(d: Any, *keys: str) -> float:
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return float("nan")
        cur = cur[k]
    try:
        v = float(cur)
        return v if math.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def _extract_t2m_day_night_bias(summary: Dict[str, Any]) -> Tuple[float, float, int, int]:
    """Compute sample-weighted daytime (09-17Z) and nighttime (21-05Z) land T2m bias."""
    from datetime import datetime, timezone

    hourly = summary.get("hourly") or []
    has_indep = any(bool(h.get("independent")) for h in hourly if isinstance(h, dict))
    day_sum = ngt_sum = 0.0
    day_n = ngt_n = 0
    for h in hourly:
        if not isinstance(h, dict):
            continue
        if has_indep and not bool(h.get("independent")):
            continue
        v_str = str(h.get("valid_time_utc", "")).strip()
        if not v_str:
            continue
        try:
            dt = datetime.fromisoformat(v_str.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            hr = dt.astimezone(timezone.utc).hour
        except Exception:
            continue
        t2m_b = ((h.get("land") or {}).get("t2m") or {})
        b = _safe_float(t2m_b, "bias")
        n = int(t2m_b.get("n", 0) or 0)
        if not math.isfinite(b):
            continue
        w = n if n > 0 else 1
        if 9 <= hr <= 17:
            day_sum += b * w
            day_n += w
        elif hr >= 21 or hr <= 5:
            ngt_sum += b * w
            ngt_n += w
    return (
        (day_sum / day_n) if day_n > 0 else float("nan"),
        (ngt_sum / ngt_n) if ngt_n > 0 else float("nan"),
        day_n,
        ngt_n,
    )


def _extract_case_with_pool(case_meta: Dict[str, str], summary: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    blk = summary.get("independent_land") or {}
    if not blk or int(_safe_float(blk, "t2m", "n") or 0) == 0:
        blk = summary.get("overall_land") or {}

    era5_cmp = summary.get("era5_comparison") or {}
    eblk = era5_cmp.get("independent_land") or era5_cmp.get("overall_land") or {}
    esk = eblk.get("skill") or {}

    eua_top = summary.get("era5_upper_air") or {}
    eua = eua_top.get("independent") or eua_top.get("overall") or {}

    ev_src = blk.get("events") or {}
    ev_counts: Dict[str, Dict[str, int]] = {}
    fog_b = blk.get("fog") or {}
    ev_counts["fog"] = {
        "hits": int(fog_b.get("hits", 0) or 0),
        "misses": int(fog_b.get("misses", 0) or 0),
        "false_alarms": int(fog_b.get("false_alarms", 0) or 0),
        "n": int(fog_b.get("n", 0) or 0),
    }
    for ek in _EVENT_KEYS[1:]:
        eb = ev_src.get(ek) or {}
        ev_counts[ek] = {
            "hits": int(eb.get("hits", 0) or 0),
            "misses": int(eb.get("misses", 0) or 0),
            "false_alarms": int(eb.get("false_alarms", 0) or 0),
            "n": int(eb.get("n", 0) or 0),
        }

    t2m_day_bias, t2m_ngt_bias, day_n, ngt_n = _extract_t2m_day_night_bias(summary)
    n_obs_t2m = int(_safe_float(blk, "t2m", "n") if math.isfinite(_safe_float(blk, "t2m", "n")) else 0)

    row: Dict[str, Any] = {
        "id": case_meta["id"],
        "date": case_meta["date"],
        "season": case_meta["season"],
        "regime": case_meta["regime"],
        "description": case_meta.get("description", ""),
        "verified_hours": int(summary.get("verified_hours", 0) or 0),
        "n_obs_t2m": n_obs_t2m,
        "t2m_rmse": _safe_float(blk, "t2m", "rmse"),
        "t2m_bias": _safe_float(blk, "t2m", "bias"),
        "t2m_day_bias": t2m_day_bias,
        "t2m_ngt_bias": t2m_ngt_bias,
        "td2m_rmse": _safe_float(blk, "td2m", "rmse"),
        "td2m_bias": _safe_float(blk, "td2m", "bias"),
        "rh_bias": _safe_float(blk, "rh", "bias"),
        "mslp_rmse": _safe_float(blk, "mslp", "rmse"),
        "mslp_bias": _safe_float(blk, "mslp", "bias"),
        "ws10_rmse": _safe_float(blk, "ws10", "rmse"),
        "ws10_bias": _safe_float(blk, "ws10", "bias"),
        "gust10_rmse": _safe_float(blk, "gust10", "rmse"),
        "gust10_bias": _safe_float(blk, "gust10", "bias"),
        "sw_down_rmse": _safe_float(blk, "sw_down", "rmse"),
        "sw_down_bias": _safe_float(blk, "sw_down", "bias"),
        "tcc_rmse": _safe_float(blk, "tcc", "rmse"),
        "tcc_bias": _safe_float(blk, "tcc", "bias"),
        "precip_rmse": _safe_float(blk, "precip", "rmse"),
        "precip_bias": _safe_float(blk, "precip", "bias"),
        "ss_t2m_era5": _safe_float(esk, "t2m", "skill_score"),
        "ss_mslp_era5": _safe_float(esk, "mslp", "skill_score"),
        "ss_ws10_era5": _safe_float(esk, "ws10", "skill_score"),
        "ss_precip_era5": _safe_float(esk, "precip", "skill_score"),
        "t850_rmse": _safe_float(eua, "t850_k", "rmse"),
        "gh500_rmse": _safe_float(eua, "gh500_m", "rmse"),
        "wspd850_rmse": _safe_float(eua, "wspd850_ms", "rmse"),
        "events": ev_counts,
    }

    def _n_for(block: Dict[str, Any], key: str) -> int:
        nv = block.get(key, {}).get("n", 0) if isinstance(block.get(key), dict) else 0
        try:
            return max(0, int(nv or 0))
        except Exception:
            return 0

    weights: Dict[str, int] = {
        "t2m_rmse": _n_for(blk, "t2m"),
        "t2m_bias": _n_for(blk, "t2m"),
        "t2m_day_bias": day_n,
        "t2m_ngt_bias": ngt_n,
        "td2m_rmse": _n_for(blk, "td2m"),
        "td2m_bias": _n_for(blk, "td2m"),
        "rh_bias": _n_for(blk, "rh"),
        "mslp_rmse": _n_for(blk, "mslp"),
        "mslp_bias": _n_for(blk, "mslp"),
        "ws10_rmse": _n_for(blk, "ws10"),
        "ws10_bias": _n_for(blk, "ws10"),
        "gust10_rmse": _n_for(blk, "gust10"),
        "gust10_bias": _n_for(blk, "gust10"),
        "sw_down_rmse": _n_for(blk, "sw_down"),
        "sw_down_bias": _n_for(blk, "sw_down"),
        "tcc_rmse": _n_for(blk, "tcc"),
        "tcc_bias": _n_for(blk, "tcc"),
        "precip_rmse": _n_for(blk, "precip"),
        "precip_bias": _n_for(blk, "precip"),
        "t850_rmse": _n_for(eua, "t850_k"),
        "gh500_rmse": _n_for(eua, "gh500_m"),
        "wspd850_rmse": _n_for(eua, "wspd850_ms"),
    }
    era5_pairs: Dict[str, Tuple[int, float, float]] = {}
    eblk_t = eblk.get("tempest") or {}
    eblk_e = eblk.get("era5") or {}
    for mk, vk in (
        ("ss_t2m_era5", "t2m"),
        ("ss_mslp_era5", "mslp"),
        ("ss_ws10_era5", "ws10"),
        ("ss_precip_era5", "precip"),
    ):
        tb = eblk_t.get(vk) or {}
        eb = eblk_e.get(vk) or {}
        rt = _safe_float(tb, "rmse")
        re = _safe_float(eb, "rmse")
        nt = int(tb.get("n", 0) or 0)
        ne = int(eb.get("n", 0) or 0)
        npair = min(nt, ne) if (nt > 0 and ne > 0) else 0
        era5_pairs[mk] = (npair, rt, re)
    return row, {"weights": weights, "era5_pairs": era5_pairs}


def _extract_case_metrics(case_meta: Dict[str, str], summary: Dict[str, Any]) -> Dict[str, Any]:
    row, _ = _extract_case_with_pool(case_meta, summary)
    return row


def _compute_pooled_metrics(rows: Sequence[Dict[str, Any]], pools: Optional[Sequence[Dict[str, Any]]]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for mk in _METRIC_KEYS:
        if mk.startswith("ss_") and pools is not None:
            mse_t_sum = mse_e_sum = 0.0
            n_sum = 0
            for p in pools:
                npair, rt, re = p.get("era5_pairs", {}).get(mk, (0, float("nan"), float("nan")))
                if npair > 0 and math.isfinite(rt) and math.isfinite(re):
                    mse_t_sum += npair * rt * rt
                    mse_e_sum += npair * re * re
                    n_sum += npair
            if n_sum > 0 and mse_e_sum > 1e-24:
                metrics[mk] = 1.0 - math.sqrt(mse_t_sum / n_sum) / math.sqrt(mse_e_sum / n_sum)
                continue

        pairs: List[Tuple[float, float]] = []
        for idx, r in enumerate(rows):
            v = float(r.get(mk, float("nan")))
            if not math.isfinite(v):
                continue
            w = float(pools[idx].get("weights", {}).get(mk, 0)) if pools is not None else 0.0
            pairs.append((v, w))
        if not pairs:
            metrics[mk] = float("nan")
            continue
        has_weights = all(w > 0.0 for _, w in pairs) or any(w > 0.0 for _, w in pairs)
        ws = [w if (has_weights and w > 0.0) else (0.0 if has_weights else 1.0) for _, w in pairs]
        wtot = sum(ws)
        if wtot <= 0.0:
            ws = [1.0 for _ in pairs]
            wtot = float(len(pairs))
        if mk.endswith("_rmse"):
            metrics[mk] = math.sqrt(sum(w * v * v for (v, _), w in zip(pairs, ws)) / wtot)
        else:
            metrics[mk] = sum(w * v for (v, _), w in zip(pairs, ws)) / wtot
    return metrics


def _aggregate_group(rows: Sequence[Dict[str, Any]], pools: Optional[Sequence[Dict[str, Any]]] = None) -> Dict[str, Any]:
    metrics = _compute_pooled_metrics(rows, pools)

    bootstrap_ci95: Dict[str, List[float]] = {}
    if len(rows) >= 3:
        rng = np.random.default_rng(42)
        idx_mat = rng.integers(0, len(rows), size=(1000, len(rows)))
        boot_buckets: Dict[str, List[float]] = {mk: [] for mk in _METRIC_KEYS}
        for b_idx in idx_mat:
            sub_rows = [rows[int(i)] for i in b_idx]
            sub_pools = [pools[int(i)] for i in b_idx] if pools is not None else None
            bm = _compute_pooled_metrics(sub_rows, sub_pools)
            for mk in _METRIC_KEYS:
                if math.isfinite(bm[mk]):
                    boot_buckets[mk].append(bm[mk])
        for mk in _METRIC_KEYS:
            vals = boot_buckets[mk]
            if len(vals) >= 10:
                lo, hi = np.percentile(vals, [2.5, 97.5])
                bootstrap_ci95[mk] = [float(lo), float(hi)]
            else:
                bootstrap_ci95[mk] = [float("nan"), float("nan")]

    ev_agg: Dict[str, Dict[str, Any]] = {}
    for ek in _EVENT_KEYS:
        h = sum(int(r["events"][ek]["hits"]) for r in rows)
        m = sum(int(r["events"][ek]["misses"]) for r in rows)
        fa = sum(int(r["events"][ek]["false_alarms"]) for r in rows)
        n = sum(int(r["events"][ek]["n"]) for r in rows)
        pod = float(h / (h + m)) if (h + m) > 0 else float("nan")
        far = float(fa / (h + fa)) if (h + fa) > 0 else float("nan")
        csi = float(h / (h + m + fa)) if (h + m + fa) > 0 else float("nan")
        fb = float((h + fa) / (h + m)) if (h + m) > 0 else float("nan")
        hits_rand = ((h + m) * (h + fa)) / float(n) if n > 0 else float("nan")
        denom = float(h + m + fa) - hits_rand if n > 0 else float("nan")
        ets = float((h - hits_rand) / denom) if (n > 0 and math.isfinite(denom) and denom > 1e-12) else float("nan")
        ev_agg[ek] = {
            "n": n,
            "obs_yes": h + m,
            "mod_yes": h + fa,
            "hits": h,
            "misses": m,
            "false_alarms": fa,
            "pod": pod,
            "far": far,
            "csi": csi,
            "ets": ets,
            "freq_bias": fb,
        }
    out: Dict[str, Any] = {"n_cases": len(rows), "metrics": metrics, "events": ev_agg}
    if bootstrap_ci95:
        out["bootstrap_ci95"] = bootstrap_ci95
    return out


def aggregate_benchmark_scorecard(case_records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate case-level KNMI + ERA5 verification summaries by case, season, regime, and overall."""
    extracted = [_extract_case_with_pool(rec["case"], rec["summary"]) for rec in case_records]
    rows = [r for r, _ in extracted]
    pools = [p for _, p in extracted]

    seasons_seen: Dict[str, List[Tuple[Dict[str, Any], Dict[str, Any]]]] = {}
    regimes_seen: Dict[str, List[Tuple[Dict[str, Any], Dict[str, Any]]]] = {}
    for r, p in extracted:
        seasons_seen.setdefault(r["season"], []).append((r, p))
        regimes_seen.setdefault(r["regime"], []).append((r, p))

    return {
        "n_cases": len(rows),
        "cases": rows,
        "by_season": {
            k: _aggregate_group([r for r, _ in pairs], [p for _, p in pairs])
            for k, pairs in sorted(seasons_seen.items())
        },
        "by_regime": {
            k: _aggregate_group([r for r, _ in pairs], [p for _, p in pairs])
            for k, pairs in sorted(regimes_seen.items())
        },
        "overall": _aggregate_group(rows, pools),
    }


def _plot_scorecard_png(scorecard: Dict[str, Any], out_png: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = scorecard.get("cases") or []
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.0), dpi=140)
    ax1, ax2, ax3, ax4 = axes.ravel()

    # 1. Per-case T2m & Wind10m RMSE
    if cases:
        labels = [f"{c['date']}\n({c['regime'][:10]})" for c in cases]
        x = np.arange(len(cases))
        w = 0.38
        t2m_vals = [c["t2m_rmse"] if math.isfinite(c["t2m_rmse"]) else 0.0 for c in cases]
        ws_vals = [c["ws10_rmse"] if math.isfinite(c["ws10_rmse"]) else 0.0 for c in cases]
        ax1.bar(x - w / 2, t2m_vals, width=w, label="T2m RMSE (K)")
        ax1.bar(x + w / 2, ws_vals, width=w, label="Wind10m RMSE (m/s)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=35, ha="right", fontsize=7.5)
        ax1.grid(True, axis="y", alpha=0.3)
        ax1.legend(fontsize=8)
    ax1.set_title("Per Case: T2m & Wind10m RMSE (Onafhankelijk)")

    # 2. Skill Score vs ARCO-ERA5 0.25-deg
    if cases:
        x = np.arange(len(cases))
        w = 0.26
        for off, mk, lbl in (
            (-w, "ss_t2m_era5", "SS T2m"),
            (0.0, "ss_mslp_era5", "SS MSLP"),
            (w, "ss_ws10_era5", "SS Wind10m"),
        ):
            vals = [c[mk] if math.isfinite(c[mk]) else 0.0 for c in cases]
            ax2.bar(x + off, vals, width=w, label=lbl)
        ax2.axhline(0.0, color="black", linewidth=0.8)
        ax2.set_xticks(x)
        ax2.set_xticklabels([c["date"] for c in cases], rotation=35, ha="right", fontsize=7.5)
        ax2.grid(True, axis="y", alpha=0.3)
        ax2.legend(fontsize=8)
    ax2.set_title("Skill Score vs ARCO-ERA5 0.25° (SS > 0 = Tempest wint)")

    # 3. Season summary T2m/Wind RMSE
    by_s = scorecard.get("by_season") or {}
    if by_s:
        s_names = list(by_s.keys())
        xs = np.arange(len(s_names))
        ax3.bar(xs - 0.18, [by_s[s]["metrics"]["t2m_rmse"] for s in s_names], width=0.36, label="T2m RMSE (K)")
        ax3.bar(xs + 0.18, [by_s[s]["metrics"]["ws10_rmse"] for s in s_names], width=0.36, label="Wind10m RMSE (m/s)")
        ax3.set_xticks(xs)
        ax3.set_xticklabels(s_names, fontsize=9)
        ax3.grid(True, axis="y", alpha=0.3)
        ax3.legend(fontsize=8)
    ax3.set_title("Seizoensgemiddelden: T2m & Wind10m RMSE")

    # 4. Overall event contingency POD & CSI
    ev_all = (scorecard.get("overall") or {}).get("events") or {}
    active_ev = [(k, v) for k, v in ev_all.items() if v.get("obs_yes", 0) > 0 or v.get("mod_yes", 0) > 0]
    if active_ev:
        xe = np.arange(len(active_ev))
        pods = [100.0 * v["pod"] if (v.get("obs_yes", 0) > 0 and math.isfinite(v["pod"])) else 0.0 for _, v in active_ev]
        csis = [100.0 * v["csi"] if (v.get("obs_yes", 0) > 0 and math.isfinite(v["csi"])) else 0.0 for _, v in active_ev]
        ax4.bar(xe - 0.18, pods, width=0.36, label="POD (%)")
        ax4.bar(xe + 0.18, csis, width=0.36, label="CSI (%)")
        for idx_ev, (_k, v_ev) in enumerate(active_ev):
            if int(v_ev.get("obs_yes", 0) or 0) == 0:
                ax4.text(idx_ev, 3.0, "n/a", ha="center", va="bottom", fontsize=7.5, color="dimgray", fontweight="bold")
        ax4.set_xticks(xe)
        ax4.set_xticklabels([k for k, _ in active_ev], rotation=28, ha="right", fontsize=8)
        ax4.set_ylim(0.0, 105.0)
        ax4.grid(True, axis="y", alpha=0.3)
        ax4.legend(fontsize=8)
    ax4.set_title("Weertypes & Convectie: Gebundelde POD / CSI (%)")

    fig.suptitle(f"Tempest-NWP Multi-Season ARCO-ERA5 Hindcast Benchmark ({scorecard['n_cases']} cases)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(out_png)
    plt.close(fig)
    return out_png


def _write_scorecard_artifacts(scorecard: Dict[str, Any], runs_root: Path) -> None:
    runs_root.mkdir(parents=True, exist_ok=True)
    csv_path = runs_root / "benchmark_scorecard.csv"
    json_path = runs_root / "benchmark_scorecard.json"
    png_path = runs_root / "benchmark_scorecard.png"

    rows = scorecard.get("cases") or []
    fieldnames = ["id", "date", "season", "regime", "verified_hours", "n_obs_t2m", *_METRIC_KEYS]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            out_r: Dict[str, Any] = {k: r.get(k, "") for k in ("id", "date", "season", "regime", "verified_hours", "n_obs_t2m")}
            for mk in _METRIC_KEYS:
                val = r.get(mk, float("nan"))
                out_r[mk] = round(float(val), 4) if math.isfinite(float(val)) else ""
            w.writerow(out_r)

    json_path.write_text(json.dumps(scorecard, indent=2, default=str), encoding="utf-8")
    if rows:
        _plot_scorecard_png(scorecard, png_path)


def _prefetch_era5_surface_cache(
    era5_ds: Any,
    valid_dts: Sequence[datetime.datetime],
    cache_dir: Path,
) -> Any:
    """Batch-fetch missing 2D ARCO-ERA5 verification slices and persist clean `.wx1` caches."""
    cdir = Path(cache_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    missing: List[Tuple[datetime.datetime, Path]] = []
    for dt in valid_dts:
        u_dt = dt.astimezone(datetime.timezone.utc) if dt.tzinfo is not None else dt.replace(tzinfo=datetime.timezone.utc)
        full_p = cdir / f"era5_verify_0p25_{u_dt:%Y%m%d_%H}Z.wx1"
        sfc_p = cdir / f"era5_verify_sfc_0p25_{u_dt:%Y%m%d_%H}Z.wx1"
        if (full_p.is_file() and full_p.stat().st_size > 1024) or (sfc_p.is_file() and sfc_p.stat().st_size > 1024):
            continue
        missing.append((u_dt, sfc_p))
    if not missing:
        return era5_ds

    if era5_ds is None:
        era5_ds = open_arco_era5_dataset()

    pairs = [
        (long_k, short_k)
        for long_k, short_k in _ERA5_VERIFY_2D_PAIRS
        if long_k in era5_ds.data_vars or short_k in era5_ds.data_vars
    ]
    if not pairs:
        return era5_ds

    src_names = [(long_k if long_k in era5_ds.data_vars else short_k) for long_k, _ in pairs]
    targets = [np.datetime64(dt.replace(tzinfo=None), "s") for dt, _ in missing]
    ds_sel = era5_ds[src_names].sel(time=targets, method="nearest")
    ds_sub = _extract_lon_wrapped_subset(ds_sel, 47.5, 57.5, -1.5, 12.0).load()
    rename_map = {
        (long_k if long_k in ds_sub.data_vars else short_k): short_k
        for long_k, short_k in pairs
        if (long_k if long_k in ds_sub.data_vars else short_k) != short_k
    }

    for idx_t, (u_dt, sfc_p) in enumerate(missing):
        ds_one = ds_sub.isel(time=idx_t) if "time" in ds_sub.dims else ds_sub
        if "time" in ds_one.coords:
            sel_sec = int(np.datetime64(ds_one["time"].values, "ns").astype("datetime64[s]").astype(np.int64))
            if abs(sel_sec - int(u_dt.timestamp())) > 1800:
                raise ValueError(f"ERA5 archive does not contain {u_dt.isoformat()}")
        if rename_map:
            ds_one = ds_one.rename(rename_map)
        ds_clean = _sanitize_era5_slice_for_wx1(ds_one, u_dt)
        wx1.write_dataset(
            sfc_p,
            ds_clean,
            dataset_id=f"era5_verify_sfc_{u_dt:%Y%m%d_%H}Z",
            codec="zstd_shuffle",
            workers=2,
        )
    return era5_ds


def _find_final_forecast_wx1(case_dir: Path, lam_domain: str, hours: float) -> Optional[Path]:
    tag = f"{int(round(hours)):03d}" if abs(hours - round(hours)) < 1e-4 else f"{hours:g}"
    fname = f"forecast_{lam_domain}_T+{tag}h.wx1"
    for sub in ("d3", "d2", "d1", "."):
        cand = case_dir / sub / fname
        if cand.is_file() and cand.stat().st_size > 1024:
            return cand
    return None


def resolve_selected_cases(
    *,
    preset: str = "core10",
    cases_arg: Optional[str] = None,
    dates_arg: Optional[str] = None,
    season: str = "all",
    extra_from_selector: int = 0,
    cases_dir: Path | str = DEFAULT_CASES_DIR,
) -> List[Dict[str, str]]:
    """Resolve ordered benchmark case definitions from preset, IDs, dates, season, and selector."""
    selected: List[Dict[str, str]] = []
    seen_dates: set[str] = set()

    if cases_arg:
        for cid in [x.strip() for x in cases_arg.split(",") if x.strip()]:
            if cid not in _CATALOG_BY_ID:
                raise ValueError(f"Unknown case ID {cid!r}; valid IDs: {sorted(_CATALOG_BY_ID)}")
            c = dict(_CATALOG_BY_ID[cid])
            if c["date"] not in seen_dates:
                selected.append(c)
                seen_dates.add(c["date"])
    elif dates_arg:
        for ds in [x.strip() for x in dates_arg.split(",") if x.strip()]:
            dt = datetime.date.fromisoformat(ds)
            mo = dt.month
            s_lbl = "winter" if mo in (12, 1, 2) else ("spring" if mo in (3, 4, 5) else ("summer" if mo in (6, 7, 8) else "autumn"))
            if ds not in seen_dates:
                selected.append(
                    {
                        "id": f"custom_{dt:%Y%m%d}",
                        "date": ds,
                        "season": s_lbl,
                        "regime": "custom",
                        "description": f"Custom date {ds}",
                    }
                )
                seen_dates.add(ds)
    else:
        for cid in PRESETS[preset]:
            c = dict(_CATALOG_BY_ID[cid])
            if c["date"] not in seen_dates:
                selected.append(c)
                seen_dates.add(c["date"])

    if season != "all":
        selected = [c for c in selected if c["season"] == season]

    if extra_from_selector > 0 and Path(cases_dir).is_dir():
        from select_era5_cases import rank_knmi_cases

        ranked = rank_knmi_cases(cases_dir, regime="all", season=season, top=extra_from_selector + 5)
        for reg in ("radiation_fog", "storm", "summer_storm", "rain", "clear_diurnal", "snow", "freezing_mixed", "high_conv_thunder", "high_conv_no_thunder", "low_conv_thunder"):
            added = 0
            for item in ranked.get(reg, []):
                if item["date"] in seen_dates:
                    continue
                selected.append(
                    {
                        "id": f"sel_{reg}_{item['ymd']}",
                        "date": item["date"],
                        "season": item["season"],
                        "regime": reg,
                        "description": f"Selector {reg} ({item['date']})",
                    }
                )
                seen_dates.add(item["date"])
                added += 1
                if added >= extra_from_selector:
                    break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Run resumable multi-season ARCO-ERA5 hindcast & KNMI/ERA5 validation suite.")
    parser.add_argument("--preset", choices=tuple(PRESETS), default="core10", help="Curated benchmark case preset")
    parser.add_argument("--cases", type=str, default=None, help="Comma-separated catalog case IDs (overrides --preset)")
    parser.add_argument("--dates", type=str, default=None, help="Comma-separated YYYY-MM-DD dates (overrides --preset)")
    parser.add_argument("--season", choices=("all", "winter", "spring", "summer", "autumn"), default="all")
    parser.add_argument("--extra-from-selector", type=int, default=0, help="Append N extra non-overlapping dates per regime from /era5_cases")
    parser.add_argument("--lam-domain", default="d3_nl_2p5km", choices=("d3_nl_2p5km", "lam_bnl_ns_de_2p5km", "d3_benelux_de_2p5km"))
    parser.add_argument("--hours", type=float, default=24.0)
    parser.add_argument("--nz", type=int, default=65)
    parser.add_argument("--lbc-interval-hours", type=float, default=3.0)
    parser.add_argument("--runs-root", type=Path, default=PROJECT_ROOT / "runs" / "era5_benchmark")
    parser.add_argument("--knmi-cache-dir", type=Path, default=PROJECT_ROOT / "runs" / "knmi_obs_cache")
    parser.add_argument("--era5-cache-dir", type=Path, default=PROJECT_ROOT / "runs" / "era5_cache")
    parser.add_argument("--no-cy46-physics", action="store_true", help="Disable Cy46 physics suite")
    parser.add_argument("--no-pbl-last", action="store_true", help="Disable IFS 50r1 PBL-last ordering")
    parser.add_argument("--no-stream-era5-lbc", action="store_true", help="Disable lean ERA5 LBC streaming")
    parser.add_argument("--no-compare-era5", action="store_true", help="Disable ARCO-ERA5 0.25-deg verification baseline")
    parser.add_argument("--era5-step-hours", type=float, default=3.0, help="Lead-time step [h] for ERA5 comparison")
    parser.add_argument("--verify-radar-fss", action="store_true", help="Also score KNMI 1 km radar FSS (slower)")
    parser.add_argument("--no-maps", action="store_true", help="Skip hourly PNG maps during verification")
    parser.add_argument("--verify-only", action="store_true", help="Only verify already-completed runs")
    parser.add_argument("--reverify", action="store_true", help="Re-run verification even if summary JSON already exists")
    parser.add_argument("--dry-run", action="store_true", help="List selected cases and cache status without running GPU solver")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    cases = resolve_selected_cases(
        preset=args.preset,
        cases_arg=args.cases,
        dates_arg=args.dates,
        season=args.season,
        extra_from_selector=args.extra_from_selector,
    )
    if not cases:
        raise SystemExit("No benchmark cases matched the requested filters.")

    use_era5_cmp = not args.no_compare_era5
    print(f"=== Tempest-NWP Multi-Season Benchmark Suite | cases={len(cases)} | domain={args.lam_domain} | hours={args.hours:g}h ===")
    for idx, c in enumerate(cases, start=1):
        ymd = c["date"].replace("-", "")
        case_dir = args.runs_root / f"{c['id']}_{ymd}_00Z"
        done_wx1 = _find_final_forecast_wx1(case_dir, args.lam_domain, args.hours) is not None
        sum_p = case_dir / "verification" / "knmi_hourly" / "knmi_verification_summary.json"
        print(
            f"  [{idx:02d}/{len(cases):02d}] {c['id']:<26s} | {c['date']} ({c['season']:<6s}) | "
            f"{c['regime']:<20s} | sim={'OK' if done_wx1 else 'TODO'} | verify={'OK' if sum_p.is_file() else 'TODO'} | {c['description']}"
        )

    if args.dry_run:
        return

    case_records: List[Dict[str, Any]] = []
    shared_era5_ds: Any = None
    try:
        for idx, c in enumerate(cases, start=1):
            ymd = c["date"].replace("-", "")
            case_dir = args.runs_root / f"{c['id']}_{ymd}_00Z"
            init_dt = datetime.datetime.fromisoformat(f"{c['date']}T00:00:00+00:00")
            final_wx1 = _find_final_forecast_wx1(case_dir, args.lam_domain, args.hours)

            if final_wx1 is None:
                if args.verify_only:
                    logger.info("[%d/%d] Skipping %s (not run yet and --verify-only active).", idx, len(cases), c["id"])
                    continue
                logger.info("[%d/%d] Running 24h ERA5 hindcast: %s (%s)...", idx, len(cases), c["id"], c["date"])
                run_era5_hindcast_lam_pipeline(
                    init_dt,
                    lam_domain=args.lam_domain,
                    forecast_hours=args.hours,
                    nz=args.nz,
                    run_dir=case_dir,
                    lbc_interval_hours=args.lbc_interval_hours,
                    era5_cache_dir=args.era5_cache_dir,
                    stream_lbc=not args.no_stream_era5_lbc,
                    cy46_physics=not args.no_cy46_physics,
                    pbl_last=not args.no_pbl_last,
                    show_progress=True,
                    log_steps=False,
                )

            sum_p = case_dir / "verification" / "knmi_hourly" / "knmi_verification_summary.json"
            summary: Optional[Dict[str, Any]] = None
            prior_upper_air: Optional[Dict[str, Any]] = None
            if sum_p.is_file():
                try:
                    loaded = json.loads(sum_p.read_text(encoding="utf-8"))
                    prior_upper_air = loaded.get("era5_upper_air")
                    if not args.reverify and ((not use_era5_cmp) or (loaded.get("era5_comparison") is not None)):
                        summary = loaded
                        logger.info("[%d/%d] Reusing cached verification %s", idx, len(cases), sum_p)
                except Exception:
                    summary = None

            if summary is None:
                include_upper_air = True
                if use_era5_cmp and args.reverify and prior_upper_air is not None:
                    include_upper_air = False
                    step_h = max(1.0, float(args.era5_step_hours))
                    n_era5 = int(round(float(args.hours) / step_h)) + 1
                    era5_dts = [init_dt + datetime.timedelta(hours=k * step_h) for k in range(n_era5)]
                    try:
                        shared_era5_ds = _prefetch_era5_surface_cache(shared_era5_ds, era5_dts, args.era5_cache_dir)
                    except Exception as exc:
                        logger.warning("[%d/%d] ERA5 2D surface prefetch failed (%s): %s", idx, len(cases), c["id"], exc)
                logger.info("[%d/%d] Verifying %s against KNMI%s...", idx, len(cases), c["id"], " + ARCO-ERA5" if use_era5_cmp else "")
                summary = verify_run_against_knmi(
                    case_dir,
                    knmi_cache_dir=args.knmi_cache_dir,
                    obs_source="auto",
                    make_maps=not args.no_maps,
                    compare_era5=use_era5_cmp,
                    era5_cache_dir=args.era5_cache_dir,
                    era5_step_hours=args.era5_step_hours,
                    era5_include_upper_air=include_upper_air,
                    existing_era5_upper_air=prior_upper_air if not include_upper_air else None,
                    verify_radar_fss=args.verify_radar_fss,
                )

            case_records.append({"case": c, "summary": summary})
            scorecard = aggregate_benchmark_scorecard(case_records)
            _write_scorecard_artifacts(scorecard, args.runs_root)
    finally:
        if shared_era5_ds is not None:
            try:
                shared_era5_ds.close()
            except Exception:
                pass

    if case_records:
        sc = aggregate_benchmark_scorecard(case_records)
        ov = sc["overall"]["metrics"]
        print(
            f"\n=== Benchmark Scorecard ({sc['n_cases']} cases) ==="
            f"\n  T2m RMSE={ov['t2m_rmse']:.2f} K (SS vs ERA5={ov['ss_t2m_era5']:+.3f}) | "
            f"Wind10m RMSE={ov['ws10_rmse']:.2f} m/s (SS={ov['ss_ws10_era5']:+.3f}) | "
            f"MSLP RMSE={ov['mslp_rmse']:.2f} hPa (SS={ov['ss_mslp_era5']:+.3f})"
        )
        print(f"  Scorecard CSV : {args.runs_root / 'benchmark_scorecard.csv'}")
        print(f"  Scorecard PNG : {args.runs_root / 'benchmark_scorecard.png'}")


if __name__ == "__main__":
    main()
