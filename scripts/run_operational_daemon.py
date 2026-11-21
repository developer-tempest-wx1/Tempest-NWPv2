"""Continuous Operational Cycling Daemon CLI for Tempest-NWP v2 (0% AI / 0% ML)."""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PY_ROOT = PROJECT_ROOT / "python"
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))

from tempest_v2.daemon import DaemonConfig, latest_ready_cycle_time_utc, run_daemon_loop, run_single_cycle  # noqa: E402


def _parse_iso_utc(raw: str) -> datetime.datetime:
    dt = datetime.datetime.fromisoformat(raw.strip().replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt.astimezone(datetime.timezone.utc)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tempest-NWP v2 Automated Operational Cycling Daemon (0% AI / 0% ML)"
    )
    parser.add_argument("--runs-root", type=Path, default=PROJECT_ROOT / "runs")
    parser.add_argument("--cycles", type=int, default=None, help="Maximum cycles before exiting cleanly")
    parser.add_argument("--once", action="store_true", help="Execute one operational cycle immediately and exit")
    parser.add_argument("--init-time", type=str, default=None, help="Optional UTC ISO init time (with --once)")
    parser.add_argument("--cycle-hours", type=str, default="0,6,12,18", help="Comma-separated UTC cycle hours")
    parser.add_argument("--delay-minutes", type=int, default=35, help="Minutes after cycle hour before running")
    parser.add_argument("--d2-hours", type=float, default=24.0)
    parser.add_argument("--d3-hours", type=float, default=24.0)
    parser.add_argument("--nz", type=int, choices=[65, 90, 137], default=65)
    parser.add_argument("--physics-every", type=int, default=5)
    parser.add_argument("--cycle-restart-hours", type=str, default="6.0", help="Comma-separated cycle restart leads")
    parser.add_argument("--ensemble-members", type=int, default=0, help="0=CF only, 51=CF+50 perturbed members")
    parser.add_argument("--ensemble-hours", type=float, default=12.0)
    parser.add_argument("--keep-runs", type=int, default=16)
    parser.add_argument("--no-da", action="store_true")
    parser.add_argument("--no-radar", action="store_true")
    parser.add_argument("--no-notify", action="store_true")
    parser.add_argument("--notify-topic", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    cycle_hours = tuple(int(h.strip()) for h in args.cycle_hours.split(",") if h.strip())
    cyc_rst_hrs = tuple(float(h.strip()) for h in args.cycle_restart_hours.split(",") if h.strip())

    cfg = DaemonConfig(
        runs_root=args.runs_root,
        cycle_hours_utc=cycle_hours,
        cycle_delay_minutes=args.delay_minutes,
        keep_runs=args.keep_runs,
        d2_lead_hours=args.d2_hours,
        d3_lead_hours=args.d3_hours,
        nz=args.nz,
        physics_every=args.physics_every,
        cycle_restart_hours=cyc_rst_hrs,
        ensemble_members=args.ensemble_members,
        ensemble_lead_hours=args.ensemble_hours,
        enable_da=not args.no_da,
        enable_radar=not args.no_radar,
        enable_notifier=not args.no_notify,
        notify_topic=args.notify_topic,
    )

    if args.once:
        if args.init_time:
            init_dt = _parse_iso_utc(args.init_time)
        else:
            init_dt, _ = latest_ready_cycle_time_utc(
                datetime.datetime.now(datetime.timezone.utc),
                cfg.cycle_hours_utc,
                cfg.cycle_delay_minutes,
            )
        summary = run_single_cycle(cfg, init_dt)
        print(json.dumps(summary, indent=2, default=str))
        return

    explicit_times: Optional[list[datetime.datetime]] = None
    if args.init_time:
        explicit_times = [_parse_iso_utc(args.init_time)]
    results = run_daemon_loop(cfg, max_cycles=args.cycles, explicit_cycle_times=explicit_times)
    print(json.dumps({"completed_cycles": len(results), "runs": [r["run_id"] for r in results]}, indent=2))


if __name__ == "__main__":
    main()
