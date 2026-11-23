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

from tempest_v2.knmi_verify import verify_run_against_knmi  # noqa: E402
from tempest_v2.runner import run_era5_hindcast_lam_pipeline  # noqa: E402

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
    "ice_freezing_rain",
)

_METRIC_KEYS: Tuple[str, ...] = (
    "t2m_rmse",
    "t2m_bias",
    "td2m_rmse",
    "rh_bias",
    "mslp_rmse",
    "ws10_rmse",
    "ws10_bias",
    "gust10_rmse",
    "sw_down_rmse",
    "tcc_rmse",
    "precip_rmse",
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


def _extract_case_metrics(case_meta: Dict[str, str], summary: Dict[str, Any]) -> Dict[str, Any]:
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

    row: Dict[str, Any] = {
        "id": case_meta["id"],
        "date": case_meta["date"],
        "season": case_meta["season"],
        "regime": case_meta["regime"],
        "description": case_meta.get("description", ""),
        "verified_hours": int(summary.get("verified_hours", 0) or 0),
        "n_obs_t2m": int(_safe_float(blk, "t2m", "n") if math.isfinite(_safe_float(blk, "t2m", "n")) else 0),
        "t2m_rmse": _safe_float(blk, "t2m", "rmse"),
        "t2m_bias": _safe_float(blk, "t2m", "bias"),
        "td2m_rmse": _safe_float(blk, "td2m", "rmse"),
        "rh_bias": _safe_float(blk, "rh", "bias"),
        "mslp_rmse": _safe_float(blk, "mslp", "rmse"),
        "ws10_rmse": _safe_float(blk, "ws10", "rmse"),
        "ws10_bias": _safe_float(blk, "ws10", "bias"),
        "gust10_rmse": _safe_float(blk, "gust10", "rmse"),
        "sw_down_rmse": _safe_float(blk, "sw_down", "rmse"),
        "tcc_rmse": _safe_float(blk, "tcc", "rmse"),
        "precip_rmse": _safe_float(blk, "precip", "rmse"),
        "ss_t2m_era5": _safe_float(esk, "t2m", "skill_score"),
        "ss_mslp_era5": _safe_float(esk, "mslp", "skill_score"),
        "ss_ws10_era5": _safe_float(esk, "ws10", "skill_score"),
        "ss_precip_era5": _safe_float(esk, "precip", "skill_score"),
        "t850_rmse": _safe_float(eua, "t850_k", "rmse"),
        "gh500_rmse": _safe_float(eua, "gh500_m", "rmse"),
        "wspd850_rmse": _safe_float(eua, "wspd850_ms", "rmse"),
        "events": ev_counts,
    }
    return row


def _aggregate_group(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    metrics: Dict[str, float] = {}
    for mk in _METRIC_KEYS:
        vals = [float(r[mk]) for r in rows if math.isfinite(float(r.get(mk, float("nan"))))]
        metrics[mk] = float(np.mean(vals)) if vals else float("nan")

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
            "freq_bias": fb,
        }
    return {"n_cases": len(rows), "metrics": metrics, "events": ev_agg}


def aggregate_benchmark_scorecard(case_records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate case-level KNMI + ERA5 verification summaries by case, season, regime, and overall."""
    rows = [_extract_case_metrics(rec["case"], rec["summary"]) for rec in case_records]

    seasons_seen: Dict[str, List[Dict[str, Any]]] = {}
    regimes_seen: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        seasons_seen.setdefault(r["season"], []).append(r)
        regimes_seen.setdefault(r["regime"], []).append(r)

    return {
        "n_cases": len(rows),
        "cases": rows,
        "by_season": {k: _aggregate_group(v) for k, v in sorted(seasons_seen.items())},
        "by_regime": {k: _aggregate_group(v) for k, v in sorted(regimes_seen.items())},
        "overall": _aggregate_group(rows),
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
        pods = [100.0 * v["pod"] if math.isfinite(v["pod"]) else 0.0 for _, v in active_ev]
        csis = [100.0 * v["csi"] if math.isfinite(v["csi"]) else 0.0 for _, v in active_ev]
        ax4.bar(xe - 0.18, pods, width=0.36, label="POD (%)")
        ax4.bar(xe + 0.18, csis, width=0.36, label="CSI (%)")
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
    cases_dir: Path | str = Path("/home/j_ord/AI/Tempest/era5_cases"),
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
        if sum_p.is_file() and not args.reverify:
            try:
                loaded = json.loads(sum_p.read_text(encoding="utf-8"))
                if (not use_era5_cmp) or (loaded.get("era5_comparison") is not None):
                    summary = loaded
                    logger.info("[%d/%d] Reusing cached verification %s", idx, len(cases), sum_p)
            except Exception:
                summary = None

        if summary is None:
            logger.info("[%d/%d] Verifying %s against KNMI%s...", idx, len(cases), c["id"], " + ARCO-ERA5" if use_era5_cmp else "")
            summary = verify_run_against_knmi(
                case_dir,
                knmi_cache_dir=args.knmi_cache_dir,
                obs_source="auto",
                make_maps=not args.no_maps,
                compare_era5=use_era5_cmp,
                era5_cache_dir=args.era5_cache_dir,
                era5_step_hours=args.era5_step_hours,
                verify_radar_fss=args.verify_radar_fss,
            )

        case_records.append({"case": c, "summary": summary})
        scorecard = aggregate_benchmark_scorecard(case_records)
        _write_scorecard_artifacts(scorecard, args.runs_root)

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
