"""Fast Local Case Selector for ARCO-ERA5 Hindcasts Using KNMI Hourly CSV Archives.

Reads compressed KNMI hourly station archives (`knmi_uur_*.csv.gz`) in `/home/j_ord/AI/Tempest/era5_cases`
(legacy 0.1x integer-scaled CSV fields: FH, FX, T in 0.1 C, RH precipitation in 0.1 mm / -1 trace,
S=snow, O=thunder, M=fog) and ranks standout Dutch weather days by meteorological regime:
  - radiation_fog : widespread fog (M) with light wind (FH <= 4.0 m/s) & dry (rain <= 0.5 mm)
  - fog           : maximum widespread fog station-hours (M)
  - storm         : strongest widespread gusts (FX p95 & max) and mean wind (FH)
  - rain          : heaviest station-mean 24h precipitation (and thunder O)
  - clear_diurnal : largest station-mean diurnal T_max - T_min with dry conditions (< 0.1 mm)
  - snow          : widespread snowfall hours (S) with sub-zero T_min
"""

from __future__ import annotations

import argparse
import csv
import datetime
import gzip
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

DEFAULT_CASES_DIR = Path("/home/j_ord/AI/Tempest/era5_cases")
SEASON_MONTHS: Dict[str, Tuple[int, ...]] = {
    "winter": (12, 1, 2),
    "spring": (3, 4, 5),
    "summer": (6, 7, 8),
    "autumn": (9, 10, 11),
}
REGIMES: Tuple[str, ...] = (
    "radiation_fog",
    "fog",
    "storm",
    "summer_storm",
    "rain",
    "clear_diurnal",
    "snow",
    "severe_frost",
    "heatwave",
    "freezing_mixed",
    "high_conv_thunder",
    "high_conv_no_thunder",
    "low_conv_thunder",
)
_FILE_YEAR_RE = re.compile(r"knmi_uur_(\d{4})\.csv\.gz$")


def _month_to_season(month: int) -> str:
    for s_name, months in SEASON_MONTHS.items():
        if month in months:
            return s_name
    return "unknown"


def _parse_optional_float(raw: str) -> Optional[float]:
    s = raw.strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _percentile(sorted_vals: Sequence[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = (len(sorted_vals) - 1) * (q / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def _summarize_year_csv(
    csv_gz_path: Path,
    max_ymd: str,
) -> List[Dict[str, Any]]:
    days: Dict[str, Dict[str, Any]] = {}
    with gzip.open(csv_gz_path, "rt", encoding="utf-8", errors="ignore", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            ymd = (row.get("YYYYMMDD") or "").strip()
            if len(ymd) != 8 or ymd > max_ymd:
                continue
            stn = (row.get("STN") or "").strip()
            if not stn:
                continue

            d = days.get(ymd)
            if d is None:
                d = {
                    "fh_sum": 0.0,
                    "fh_cnt": 0,
                    "fx_vals": [],
                    "fx_max": 0.0,
                    "t_min": 99.0,
                    "t_max": -99.0,
                    "t_stn": {},
                    "rain_stn": {},
                    "fog_hrs": 0,
                    "snow_hrs": 0,
                    "thunder_hrs": 0,
                }
                days[ymd] = d

            fh_v = _parse_optional_float(row.get("FH") or "")
            if fh_v is not None and fh_v >= 0.0:
                d["fh_sum"] += fh_v * 0.1
                d["fh_cnt"] += 1

            fx_v = _parse_optional_float(row.get("FX") or "")
            if fx_v is not None and fx_v >= 0.0:
                fx_ms = fx_v * 0.1
                d["fx_vals"].append(fx_ms)
                if fx_ms > d["fx_max"]:
                    d["fx_max"] = fx_ms

            t_v = _parse_optional_float(row.get("T") or "")
            if t_v is not None:
                tc = t_v * 0.1
                if tc < d["t_min"]:
                    d["t_min"] = tc
                if tc > d["t_max"]:
                    d["t_max"] = tc
                st_rec = d["t_stn"].get(stn)
                if st_rec is None:
                    d["t_stn"][stn] = [tc, tc, 1]
                else:
                    if tc < st_rec[0]:
                        st_rec[0] = tc
                    if tc > st_rec[1]:
                        st_rec[1] = tc
                    st_rec[2] += 1

            rh_v = _parse_optional_float(row.get("RH") or "")
            if rh_v is not None:
                mm = 0.025 if (-1.5 <= rh_v < 0.0) else (max(0.0, rh_v * 0.1))
                d["rain_stn"][stn] = d["rain_stn"].get(stn, 0.0) + mm

            for raw_k, out_k in (("M", "fog_hrs"), ("S", "snow_hrs"), ("O", "thunder_hrs")):
                flag = _parse_optional_float(row.get(raw_k) or "")
                if flag is not None and flag >= 1.0:
                    d[out_k] += 1

    summaries: List[Dict[str, Any]] = []
    for ymd in sorted(days):
        d = days[ymd]
        fx_sorted = sorted(d["fx_vals"])
        fh_mean = (d["fh_sum"] / d["fh_cnt"]) if d["fh_cnt"] > 0 else float("nan")
        rain_vals = list(d["rain_stn"].values())
        rain_mean = (sum(rain_vals) / len(rain_vals)) if rain_vals else 0.0
        rain_max_stn = max(rain_vals) if rain_vals else 0.0
        diurnals = [hi - lo for lo, hi, cnt in d["t_stn"].values() if cnt >= 18]
        diurnal_c = (sum(diurnals) / len(diurnals)) if diurnals else 0.0
        t_min_c = d["t_min"] if d["t_min"] < 90.0 else float("nan")
        t_max_c = d["t_max"] if d["t_max"] > -90.0 else float("nan")
        iso_date = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}"
        mo = int(ymd[4:6])
        summaries.append(
            {
                "ymd": ymd,
                "month": mo,
                "season": _month_to_season(mo),
                "date": iso_date,
                "init_utc": f"{iso_date}T00:00Z",
                "fog_hrs": int(d["fog_hrs"]),
                "snow_hrs": int(d["snow_hrs"]),
                "thunder_hrs": int(d["thunder_hrs"]),
                "fh_mean_ms": fh_mean,
                "fx_p95_ms": _percentile(fx_sorted, 95.0),
                "fx_max_ms": float(d["fx_max"]),
                "rain_mean_mm": float(rain_mean),
                "rain_max_stn_mm": float(rain_max_stn),
                "diurnal_c": float(diurnal_c),
                "t_min_c": float(t_min_c),
                "t_max_c": float(t_max_c),
            }
        )
    return summaries


def rank_knmi_cases(
    cases_dir: Path | str = DEFAULT_CASES_DIR,
    *,
    regime: str = "all",
    season: str = "all",
    top: int = 5,
    min_year: Optional[int] = None,
    max_year: Optional[int] = None,
    max_date: str = "2026-11-12",
) -> Dict[str, List[Dict[str, Any]]]:
    """Scan KNMI hourly CSV archives and return top historical test cases per regime."""
    if regime != "all" and regime not in REGIMES:
        raise ValueError(f"Unsupported regime {regime!r}; expected 'all' or one of {REGIMES}")
    if season != "all" and season not in SEASON_MONTHS:
        raise ValueError(f"Unsupported season {season!r}; expected 'all' or one of {tuple(SEASON_MONTHS)}")
    root = Path(cases_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Cases directory not found: {root}")

    max_ymd = datetime.date.fromisoformat(max_date.strip()).strftime("%Y%m%d")
    selected_files: List[Path] = []
    for p in sorted(root.glob("knmi_uur_*.csv.gz")):
        m = _FILE_YEAR_RE.search(p.name)
        if m is not None:
            yr = int(m.group(1))
            if min_year is not None and yr < min_year:
                continue
            if max_year is not None and yr > max_year:
                continue
        selected_files.append(p)
    if not selected_files:
        raise FileNotFoundError(f"No matching knmi_uur_*.csv.gz files under {root}")

    all_days: List[Dict[str, Any]] = []
    for fp in selected_files:
        all_days.extend(_summarize_year_csv(fp, max_ymd=max_ymd))
    if season != "all":
        all_days = [d for d in all_days if d.get("season") == season]

    def _storm_score(d: Dict[str, Any]) -> Tuple[float, float]:
        fh_m = d["fh_mean_ms"] if d["fh_mean_ms"] == d["fh_mean_ms"] else 0.0
        return (d["fx_p95_ms"] + 0.6 * d["fx_max_ms"] + fh_m, d["fx_max_ms"])

    target_regimes: Iterable[str] = REGIMES if regime == "all" else (regime,)
    ranked: Dict[str, List[Dict[str, Any]]] = {}
    for reg in target_regimes:
        if reg == "radiation_fog":
            cands = [
                d for d in all_days
                if d["fog_hrs"] >= 1 and d["fh_mean_ms"] == d["fh_mean_ms"] and d["fh_mean_ms"] <= 4.0 and d["rain_mean_mm"] <= 0.5
            ]
            cands.sort(key=lambda d: (d["fog_hrs"], -d["fh_mean_ms"]), reverse=True)
        elif reg == "fog":
            cands = [d for d in all_days if d["fog_hrs"] >= 1]
            cands.sort(key=lambda d: (d["fog_hrs"], -(d["fh_mean_ms"] if d["fh_mean_ms"] == d["fh_mean_ms"] else 99.0)), reverse=True)
        elif reg == "storm":
            cands = [d for d in all_days if d["fx_max_ms"] > 0.0]
            cands.sort(key=_storm_score, reverse=True)
        elif reg == "summer_storm":
            cands = [d for d in all_days if d.get("month") in (5, 6, 7, 8, 9) and d["fx_max_ms"] > 0.0]
            cands.sort(key=_storm_score, reverse=True)
        elif reg == "rain":
            cands = [d for d in all_days if d["rain_mean_mm"] > 0.0]
            cands.sort(key=lambda d: (d["rain_mean_mm"], d["thunder_hrs"], d["rain_max_stn_mm"]), reverse=True)
        elif reg == "clear_diurnal":
            cands = [
                d for d in all_days
                if d["diurnal_c"] > 0.0 and d["rain_mean_mm"] < 0.1 and d["snow_hrs"] == 0 and d["thunder_hrs"] == 0
            ]
            cands.sort(key=lambda d: (d["diurnal_c"], -d["rain_mean_mm"], -d["fog_hrs"]), reverse=True)
        elif reg == "snow":
            cands = [
                d for d in all_days
                if d["snow_hrs"] >= 1 and d["t_min_c"] == d["t_min_c"] and d["t_min_c"] <= 0.5
            ]
            cands.sort(key=lambda d: (d["snow_hrs"], -d["t_min_c"]), reverse=True)
        elif reg == "severe_frost":
            cands = [d for d in all_days if d["t_min_c"] == d["t_min_c"] and d["t_min_c"] <= -5.0]
            cands.sort(key=lambda d: (d["t_min_c"], d["t_max_c"]))
        elif reg == "heatwave":
            cands = [d for d in all_days if d["t_max_c"] == d["t_max_c"] and d["t_max_c"] >= 30.0 and d["rain_mean_mm"] <= 1.0]
            cands.sort(key=lambda d: (d["t_max_c"], d["diurnal_c"]), reverse=True)
        elif reg == "freezing_mixed":
            cands = [
                d for d in all_days
                if d["t_min_c"] == d["t_min_c"] and d["t_max_c"] == d["t_max_c"]
                and d["t_min_c"] <= -1.0 and d["t_max_c"] >= 1.0
                and d["rain_mean_mm"] >= 1.5 and d["snow_hrs"] >= 5
            ]
            cands.sort(key=lambda d: (d["rain_mean_mm"], d["snow_hrs"]), reverse=True)
        elif reg == "high_conv_thunder":
            cands = [
                d for d in all_days
                if d.get("month") in (5, 6, 7, 8, 9) and d["t_max_c"] == d["t_max_c"]
                and d["t_max_c"] >= 24.0 and d["thunder_hrs"] >= 15
            ]
            cands.sort(key=lambda d: (d["thunder_hrs"], d["rain_max_stn_mm"], d["fx_max_ms"]), reverse=True)
        elif reg == "high_conv_no_thunder":
            cands = [
                d for d in all_days
                if d.get("month") in (5, 6, 7, 8, 9) and d["t_max_c"] == d["t_max_c"]
                and d["t_max_c"] >= 20.0 and d["thunder_hrs"] == 0
                and d["rain_max_stn_mm"] >= 15.0 and d["rain_mean_mm"] >= 1.0
                and (d["rain_max_stn_mm"] / max(d["rain_mean_mm"], 0.1)) >= 2.5
            ]
            cands.sort(key=lambda d: (d["rain_max_stn_mm"], d["t_max_c"]), reverse=True)
        else:  # low_conv_thunder
            cands = [
                d for d in all_days
                if d.get("month") in (11, 12, 1, 2, 3) and d["t_max_c"] == d["t_max_c"]
                and d["t_max_c"] <= 12.0 and d["thunder_hrs"] >= 3
            ]
            cands.sort(key=lambda d: (d["thunder_hrs"], d["snow_hrs"], d["fx_max_ms"]), reverse=True)
        ranked[reg] = cands[: max(1, int(top))]
    return ranked


def _parse_years_arg(raw: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
    if not raw:
        return None, None
    s = raw.strip()
    if "-" in s:
        a, b = s.split("-", 1)
        return int(a), int(b)
    y = int(s)
    return y, y


def main() -> None:
    parser = argparse.ArgumentParser(description="Select standout historical KNMI weather days for ARCO-ERA5 hindcasts.")
    parser.add_argument("--cases-dir", type=Path, default=DEFAULT_CASES_DIR, help="Directory with knmi_uur_*.csv.gz files")
    parser.add_argument("--regime", choices=("all",) + REGIMES, default="all", help="Meteorological regime to rank")
    parser.add_argument("--season", choices=("all",) + tuple(SEASON_MONTHS), default="all", help="Season filter")
    parser.add_argument("--top", type=int, default=5, help="Number of cases per regime (default: 5)")
    parser.add_argument("--years", type=str, default=None, help="Year or inclusive range, e.g. '2025' or '2024-2026'")
    parser.add_argument(
        "--max-date",
        type=str,
        default="2026-11-12",
        help="Latest date YYYY-MM-DD so 24-48h hindcasts stay inside ARCO-ERA5 coverage (default: 2026-11-12)",
    )
    args = parser.parse_args()

    min_y, max_y = _parse_years_arg(args.years)
    results = rank_knmi_cases(
        args.cases_dir,
        regime=args.regime,
        season=args.season,
        top=args.top,
        min_year=min_y,
        max_year=max_y,
        max_date=args.max_date,
    )

    for reg, rows in results.items():
        print(f"\n=== Regime: {reg} | season={args.season} (top {len(rows)}) ===")
        print(
            "Datum       (Seiz)   | Mist(u) | Sneeuw(u) | Onw(u) | Windgem | Stoot p95/max | Regen gem/max(mm) | Tmin/Tmax/Dag(C)"
        )
        print("-" * 124)
        for r in rows:
            fh_s = f"{r['fh_mean_ms']:4.1f}" if r["fh_mean_ms"] == r["fh_mean_ms"] else " n/a"
            print(
                f"{r['date']}  ({r['season']:<6s}) |  {r['fog_hrs']:5d}  |   {r['snow_hrs']:6d}  |  {r['thunder_hrs']:4d}  | "
                f"{fh_s} m/s | {r['fx_p95_ms']:4.1f} / {r['fx_max_ms']:4.1f}  | "
                f"{r['rain_mean_mm']:5.1f} / {r['rain_max_stn_mm']:5.1f}     | "
                f"{r['t_min_c']:+5.1f} / {r['t_max_c']:+5.1f} / {r['diurnal_c']:4.1f}"
            )
        if rows:
            top0 = rows[0]
            run_dir = f"runs/bnl_2p5km/{top0['ymd']}_00Z"
            print(
                "  -> Hindcast : "
                f"python scripts/run_tempest_v2.py --mode lam_era5_hindcast --lam-domain lam_bnl_ns_de_2p5km "
                f"--init-time {top0['init_utc']} --hours 24 --runs-root runs/bnl_2p5km --stream-era5-lbc"
            )
            print(
                "  -> Verify   : "
                f"python scripts/verify_knmi_wx1.py --run-dir {run_dir} --obs-source auto --compare-era5"
            )


if __name__ == "__main__":
    main()
