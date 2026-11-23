"""CLI for Automated Hourly Verification of Tempest-NWP .wx1 Forecasts Against KNMI 10-Min Data."""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PY_ROOT = PROJECT_ROOT / "python"
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))

from tempest_v2.knmi_verify import verify_run_against_knmi  # noqa: E402


def _fmt(val: float, spec: str = "+.2f") -> str:
    if val is None or not math.isfinite(float(val)):
        return "   n/a"
    return format(float(val), spec)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify Tempest-NWP .wx1 forecast outputs hourly against official KNMI 10-minute station observations."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--run-dir", type=Path, help="Tempest run directory (e.g. runs/bnl_2p5km/20261122_18Z)")
    src.add_argument("--wx1-dir", type=Path, help="Direct directory containing forecast_*_T+*h.wx1 files")
    parser.add_argument("--outdir", type=Path, default=None, help="Output directory for CSV/JSON/PNG verification artifacts")
    parser.add_argument("--knmi-cache-dir", type=Path, default=None, help="Cache dir for downloaded KNMI 10-min NetCDFs")
    parser.add_argument("--api-key", type=str, default=None, help="Optional KNMI Open Data API key override")
    parser.add_argument("--max-lead-hours", type=float, default=None, help="Optional max forecast lead hour to verify")
    parser.add_argument(
        "--independent-after-hour",
        type=float,
        default=None,
        help="Treat T >= hour as strictly independent (default: auto 4h when DA was active, else 2h)",
    )
    parser.add_argument(
        "--hourly-precip-6x",
        action="store_true",
        help="Fetch all 6x 10-min KNMI files per hour to verify exact 1-hour rainfall totals [mm]",
    )
    parser.add_argument("--land-only", action="store_true", help="Plot only land stations on hourly maps")
    parser.add_argument("--no-maps", action="store_true", help="Skip hourly PNG maps (write CSV/JSON/timeseries only)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    target = args.run_dir if args.run_dir is not None else args.wx1_dir
    assert target is not None
    summary = verify_run_against_knmi(
        target,
        outdir=args.outdir,
        knmi_cache_dir=args.knmi_cache_dir,
        api_key=args.api_key,
        max_lead_hours=args.max_lead_hours,
        independent_after_hour=args.independent_after_hour,
        hourly_precip_6x=args.hourly_precip_6x,
        land_only_plots=args.land_only,
        make_maps=not args.no_maps,
    )

    print(
        f"\n=== KNMI 10-min Verificatie | Domein: {summary['domain_name']} "
        f"| Uren geverifieerd: {summary['verified_hours']} (overgeslagen: {len(summary['skipped_hours'])}) ==="
    )
    print(
        "Lead  Valid UTC         Indep  N_land | T2m(bias/RMSE)   Td(bias)  RH(bias) | "
        "MSLP(bias/RMSE)  | Wind(bias/RMSE/DirMAE) | Mist(H/M/FA) | Neerslag int/rand(mm/u)"
    )
    print("-" * 138)
    for rec in summary["hourly"]:
        b = rec["land"]
        gp = rec["grid_precip"]
        flag = "JA " if rec["independent"] else "NEE"
        print(
            f"T+{rec['lead_hours']:02.0f}h {rec['valid_time_utc'][:16]}Z  {flag}    {b['n_stations']:3d}  | "
            f"{_fmt(b['t2m']['bias'])} / {_fmt(b['t2m']['rmse'], '.2f')} K | "
            f"{_fmt(b['td2m']['bias'])} K | {_fmt(b['rh']['bias'], '+.1f')}% | "
            f"{_fmt(b['mslp']['bias'])} / {_fmt(b['mslp']['rmse'], '.2f')} | "
            f"{_fmt(b['ws10']['bias'])} / {_fmt(b['ws10']['rmse'], '.2f')} / {_fmt(b['wdir']['mae_deg'], '.0f')}° | "
            f"{b['fog']['hits']:2d}/{b['fog']['misses']:2d}/{b['fog']['false_alarms']:2d}      | "
            f"{gp['int_max_mmh']:5.2f} / {gp['rim_max_mmh']:6.2f}"
        )

    for label, key in (
        ("Totaal (landstations)", "overall_land"),
        (f"Onafhankelijk T>={summary['independent_after_hour']:.0f}u (landstations)", "independent_land"),
        (f"Onafhankelijk T>={summary['independent_after_hour']:.0f}u (alle stations)", "independent_all"),
    ):
        blk = summary[key]
        print(f"\n[{label}] n_obs={blk['t2m']['n']}")
        print(
            f"  T2m: bias={_fmt(blk['t2m']['bias'])} K, RMSE={_fmt(blk['t2m']['rmse'], '.2f')} K, r={_fmt(blk['t2m']['corr'], '.2f')} | "
            f"Td2m bias={_fmt(blk['td2m']['bias'])} K | RH bias={_fmt(blk['rh']['bias'], '+.1f')}%"
        )
        print(
            f"  MSLP: bias={_fmt(blk['mslp']['bias'])} hPa, RMSE={_fmt(blk['mslp']['rmse'], '.2f')} hPa | "
            f"Wind10m: bias={_fmt(blk['ws10']['bias'])} m/s, RMSE={_fmt(blk['ws10']['rmse'], '.2f')} m/s, dir MAE={_fmt(blk['wdir']['mae_deg'], '.1f')}°"
        )
        fg = blk["fog"]
        print(
            f"  Mist (<1000m): H={fg['hits']}, M={fg['misses']}, FA={fg['false_alarms']}, "
            f"POD={_fmt(100.0 * fg['pod'] if math.isfinite(fg['pod']) else float('nan'), '.0f')}%, "
            f"FAR={_fmt(100.0 * fg['far'] if math.isfinite(fg['far']) else float('nan'), '.0f')}%, "
            f"CSI={_fmt(100.0 * fg['csi'] if math.isfinite(fg['csi']) else float('nan'), '.0f')}%"
        )

    if summary["worst_t2m_land_stations"]:
        print("\n[Top 5 landstations met grootste T2m RMSE (onafhankelijk)]")
        for st in summary["worst_t2m_land_stations"][:5]:
            print(
                f"  {st['station_id']} {st['station_name']:<22s} n={st['n_hours']:2d} | "
                f"T2m bias={st['t2m_bias_k']:+.2f}K RMSE={st['t2m_rmse_k']:.2f}K | "
                f"RH bias={st['rh_bias_pct']:+.1f}% | Wind bias={st['ws10_bias_ms']:+.2f}m/s"
            )

    art = summary["artifacts"]
    print(f"\n[Output] Overzicht PNG : {art['timeseries_png']}")
    print(f"[Output] Uur-CSV       : {art['hourly_csv']}")
    print(f"[Output] Station-CSV   : {art['station_csv']}")
    print(f"[Output] Kaarten       : {len(art['map_pngs'])} PNGs in {Path(summary['outdir']) / 'maps'}")


if __name__ == "__main__":
    main()
