"""Operational CLI entrypoint for Tempest-NWP v2 (0% AI / 0% ML)."""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PY_ROOT = PROJECT_ROOT / "python"
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))

from tempest_v2.ic_builder import (  # noqa: E402
    aurora_wx1_to_ic,
    build_domain_static_fields,
    ecmwf_grib_to_domain_state,
    load_tempest_restart_wx1,
)
from tempest_v2.ingest_dwd_poi import ingest_dwd_poi_observations  # noqa: E402
from tempest_v2.ingest_ecmwf import download_ecmwf_cold_start  # noqa: E402
from tempest_v2.ingest_knmi_observations import ingest_knmi_observations  # noqa: E402
from tempest_v2.projection import get_domain_spec  # noqa: E402
from tempest_v2.runner import (  # noqa: E402
    forecast_output_schedule_hours,
    run_domain_forecast,
    run_three_nest_pipeline,
    run_two_nest_pipeline,
)
from tempest_v2.verification import (  # noqa: E402
    load_station_obs_wx1,
    run_domain_3dvar,
    verify_against_stations,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tempest-NWP v2 — Classical GPU Non-Hydrostatic NWP (0% AI / 0% ML)"
    )
    parser.add_argument("--mode", choices=["d1_to_d2_to_d3", "d2_to_d3", "d3", "d2", "d1"], default="d2_to_d3")
    parser.add_argument("--nz", type=int, choices=[65, 90, 137], default=65)
    parser.add_argument("--hours", type=float, default=6.0)
    parser.add_argument("--d1-hours", type=float, default=None, help="Optional global D1 forecast length [h]")
    parser.add_argument(
        "--ecmwf",
        action="store_true",
        help="Select ECMWF output schedule (3-hourly through T+144, then 6-hourly through T+360); physics/dynamics remain 100%% Tempest.",
    )
    parser.add_argument("--ic-source", choices=["ecmwf", "aurora", "restart"], default="ecmwf")
    parser.add_argument("--sfc-grib", type=Path, default=None)
    parser.add_argument("--pl-grib", type=Path, default=None)
    parser.add_argument("--aurora-ic", type=Path, default=None)
    parser.add_argument("--restart-in", type=Path, default=None)
    parser.add_argument("--osisaf-wx1", type=Path, default=None, help="Optional OSI-SAF sea-ice WX-1 observation file")
    parser.add_argument("--obs-dir", type=Path, default=PROJECT_ROOT / "data" / "observations")
    parser.add_argument("--download-live", action="store_true", help="Fetch live ECMWF/KNMI/DWD feeds before running")
    parser.add_argument("--enable-da", action="store_true", help="Assimilate KNMI + DWD station obs via Balanced 3D-Var + IAU")
    parser.add_argument("--enable-ocean-si3", action="store_true", help="Enable coupled NEMO 5.0.2 + SI3 GPU ocean/sea-ice")
    parser.add_argument("--no-ocean-si3", action="store_true", help="Disable coupled NEMO+SI3 even in D1 modes")
    parser.add_argument("--no-3d-diagnostics", action="store_true", help="Disable 3D pressure-level & CAPE/synoptic WX-1 diagnostics")
    parser.add_argument("--physics-every", type=int, default=5, help="Substep physics cadence every N dynamics steps")
    parser.add_argument("--dt-d1", type=float, default=60.0)
    parser.add_argument("--dt-d2", type=float, default=20.0)
    parser.add_argument("--dt-d3", type=float, default=4.0)
    parser.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "data" / "forecasts")
    parser.add_argument("--runs-root", type=Path, default=None, help="Isolated per-cycle runs root (YYYYMMDD_HHZ/...)")
    parser.add_argument("--cycle-restart-hours", type=str, default="6.0", help="Comma-separated cycle restart leads [h]")
    parser.add_argument("--ensemble-members", type=int, default=0, help="0=CF only, 51=CF + 50 perturbed members")
    parser.add_argument("--ensemble-hours", type=float, default=None, help="Ensemble forecast length [h]")
    parser.add_argument("--ensemble-seed", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    cyc_rst_hours = tuple(float(h.strip()) for h in args.cycle_restart_hours.split(",") if h.strip())

    sfc_grib = args.sfc_grib
    pl_grib = args.pl_grib
    obs_dir_eff = args.obs_dir
    if args.download_live:
        if args.ic_source == "ecmwf" and (sfc_grib is None or pl_grib is None):
            raw_dir = (args.runs_root / "raw_ecmwf") if args.runs_root is not None else (args.outdir / "raw_ecmwf")
            sfc_grib, pl_grib, _ = download_ecmwf_cold_start(
                step=0, prefer_0p1=False, output_dir=raw_dir
            )
        if args.enable_da:
            from tempest_v2.download_all_observations import run_observation_ingest_pipeline

            args.obs_dir.mkdir(parents=True, exist_ok=True)
            radar_target_spec = get_domain_spec("d3", nz=args.nz)
            obs_dir_eff, _ = run_observation_ingest_pipeline(
                args.obs_dir,
                da_hours=6.0,
                include_sondehub=True,
                spec_for_radar=radar_target_spec,
            )

    use_ocean = False if args.no_ocean_si3 else (
        True if (args.enable_ocean_si3 or args.mode in ("d1_to_d2_to_d3", "d1")) else False
    )
    include_3d_diag = not args.no_3d_diagnostics

    if args.mode == "d1_to_d2_to_d3":
        summary = run_three_nest_pipeline(
            forecast_hours=args.hours,
            d1_forecast_hours=args.d1_hours,
            nz=args.nz,
            ic_source=args.ic_source,
            sfc_grib=sfc_grib,
            pl_grib=pl_grib,
            aurora_wx1_path=args.aurora_ic,
            restart_path=args.restart_in,
            obs_dir=obs_dir_eff if args.enable_da else None,
            outdir=args.outdir,
            runs_root=args.runs_root,
            cycle_restart_hours=cyc_rst_hours,
            ensemble_members=args.ensemble_members,
            ensemble_lead_hours=args.ensemble_hours,
            ensemble_seed=args.ensemble_seed,
            ecmwf_schedule=args.ecmwf,
            enable_da=args.enable_da,
            dt_d1_s=args.dt_d1,
            dt_d2_s=args.dt_d2,
            dt_d3_s=args.dt_d3,
            physics_every=args.physics_every,
            enable_ocean_si3=use_ocean,
            include_3d_diagnostics=include_3d_diag,
            osisaf_wx1_path=args.osisaf_wx1,
        )
        printable = {
            "run_dir": str(summary.get("run_dir")),
            "d1_elapsed_s": summary["d1"]["elapsed_s"],
            "d2_elapsed_s": summary["d2"]["elapsed_s"],
            "d3_elapsed_s": summary["d3"]["elapsed_s"],
            "d1_outputs": [str(p) for p in summary["d1"]["output_paths"]],
            "d2_outputs": [str(p) for p in summary["d2"]["output_paths"]],
            "d3_outputs": [str(p) for p in summary["d3"]["output_paths"]],
            "ocean_restarts": {
                k: str(summary[k].get("ocean_restart_path"))
                for k in ("d1", "d2", "d3")
                if summary[k].get("ocean_restart_path") is not None
            },
            "cycle_restart_paths": [str(p) for p in summary.get("cycle_restart_paths", [])],
            "da": summary["da"],
            "ensemble": summary.get("ensemble"),
        }
        print(json.dumps(printable, indent=2, default=str))
        return

    if args.mode == "d2_to_d3":
        summary = run_two_nest_pipeline(
            forecast_hours=args.hours,
            nz=args.nz,
            ic_source=args.ic_source,
            sfc_grib=sfc_grib,
            pl_grib=pl_grib,
            aurora_wx1_path=args.aurora_ic,
            restart_path=args.restart_in,
            obs_dir=obs_dir_eff if args.enable_da else None,
            outdir=args.outdir,
            runs_root=args.runs_root,
            cycle_restart_hours=cyc_rst_hours,
            ensemble_members=args.ensemble_members,
            ensemble_lead_hours=args.ensemble_hours,
            ensemble_seed=args.ensemble_seed,
            ecmwf_schedule=args.ecmwf,
            enable_da=args.enable_da,
            dt_d2_s=args.dt_d2,
            dt_d3_s=args.dt_d3,
            physics_every=args.physics_every,
            enable_ocean_si3=use_ocean,
            include_3d_diagnostics=include_3d_diag,
            osisaf_wx1_path=args.osisaf_wx1,
        )
        printable = {
            "run_dir": str(summary.get("run_dir")),
            "d2_elapsed_s": summary["d2"]["elapsed_s"],
            "d3_elapsed_s": summary["d3"]["elapsed_s"],
            "d3_outputs": [str(p) for p in summary["d3"]["output_paths"]],
            "cycle_restart_paths": [str(p) for p in summary.get("cycle_restart_paths", [])],
            "da": summary["da"],
            "ensemble": summary.get("ensemble"),
        }
        print(json.dumps(printable, indent=2, default=str))
        return

    spec = get_domain_spec(args.mode, nz=args.nz)
    hs_m, sstatic = build_domain_static_fields(spec)
    if args.ic_source == "restart" and args.restart_in is not None:
        state_3d, surf_state, meta = load_tempest_restart_wx1(args.restart_in)
        init_dt = datetime.datetime.fromisoformat(meta["valid_time_utc"].replace("Z", "+00:00"))
    elif args.ic_source == "aurora" and args.aurora_ic is not None:
        state_3d, surf_state = aurora_wx1_to_ic(args.aurora_ic, spec, hs_m)
        init_dt = datetime.datetime.now(datetime.timezone.utc).replace(minute=0, second=0, microsecond=0)
    elif sfc_grib is not None and pl_grib is not None:
        state_3d, surf_state, init_dt = ecmwf_grib_to_domain_state(
            sfc_grib, pl_grib, spec, hs_m, surface_static=sstatic
        )
    else:
        raise SystemExit("Specify --sfc-grib/--pl-grib, --aurora-ic, --restart-in, or --download-live")

    da_report = {}
    iau_f = None
    if args.enable_da and obs_dir_eff.exists():
        obs_list, stn_rows = load_station_obs_wx1(obs_dir_eff, spec, valid_time_utc=init_dt)
        if obs_list:
            da_report["verify_bg"] = verify_against_stations(spec, state_3d, surf_state, hs_m, stn_rows)
            state_3d, surf_state, iau_f, vres = run_domain_3dvar(
                spec, state_3d, surf_state, hs_m, sstatic["lsm"], obs_list, obs_dir=obs_dir_eff
            )
            da_report["var3d"] = {"j0": vres.initial_cost_j0, "jf": vres.final_cost_j, "iters": vres.iterations}
            da_report["verify_ana"] = verify_against_stations(spec, state_3d, surf_state, hs_m, stn_rows)

    dt_s = {"d1": args.dt_d1, "d2": args.dt_d2, "d3": args.dt_d3}[args.mode]
    sched = forecast_output_schedule_hours(args.hours, ecmwf_schedule=args.ecmwf, hourly=True)
    res = run_domain_forecast(
        spec,
        hs_m,
        sstatic,
        state_3d,
        surf_state,
        init_dt,
        forecast_hours=args.hours,
        dt_s=dt_s,
        physics_every=args.physics_every,
        iau_forcing=iau_f,
        iau_duration_s=900.0 if iau_f is not None else 0.0,
        output_dir=args.outdir / f"{args.mode}_wx1",
        output_hours=sched,
        save_restart_path=args.outdir / f"restart_{spec.name}.wx1",
        enable_ocean_si3=use_ocean,
        osisaf_wx1_path=args.osisaf_wx1,
        obs_dir=obs_dir_eff if args.enable_da else None,
        ocean_restart_out=(args.outdir / f"restart_ocean_{spec.name}.wx1") if use_ocean else None,
        include_3d_diagnostics=include_3d_diag,
    )
    print(
        json.dumps(
            {
                "mode": args.mode,
                "elapsed_s": res["elapsed_s"],
                "num_tiles": res["num_tiles"],
                "outputs": [str(p) for p in res["output_paths"]],
                "restart": str(res["restart_path"]),
                "ocean_restart": str(res["ocean_restart_path"]) if res.get("ocean_restart_path") else None,
                "da": da_report,
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
