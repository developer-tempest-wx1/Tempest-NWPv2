"""Operational CLI entrypoint for Tempest-NWP v2 (0% AI / 0% ML)."""

from __future__ import annotations

import argparse
import datetime
import gc
import json
import logging
import sys
import time
from pathlib import Path

try:
    from tqdm import tqdm as _tqdm
except ImportError:  # pragma: no cover - optional terminal progress bar
    _tqdm = None


class _TqdmLoggingHandler(logging.Handler):
    """Route log lines via tqdm.write so live progress bars never get torn."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            if _tqdm is not None:
                _tqdm.write(msg, file=sys.stderr)
            else:
                sys.stderr.write(msg + "\n")
            sys.stderr.flush()
        except Exception:
            self.handleError(record)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PY_ROOT = PROJECT_ROOT / "python"
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))

from tempest_v2.ic_builder import (  # noqa: E402
    aurora_wx1_to_ic,
    build_domain_static_fields,
    clear_cfgrib_cache,
    ecmwf_grib_to_domain_state,
    load_tempest_restart_wx1,
)
from tempest_v2.ingest_dwd_poi import ingest_dwd_poi_observations  # noqa: E402
from tempest_v2.ingest_ecmwf import download_ecmwf_cold_start  # noqa: E402
from tempest_v2.ingest_knmi_observations import ingest_knmi_observations  # noqa: E402
from tempest_v2.projection import D1_DOMAIN_DEFAULT_DT_S, get_domain_spec  # noqa: E402
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
    parser.add_argument("--mode", choices=["d1_to_d2_to_d3", "d2_to_d3", "d3", "d3_benelux_2p5km", "d3_1p5km", "d2", "d1"], default="d2_to_d3")
    parser.add_argument(
        "--d1-domain",
        choices=list(D1_DOMAIN_DEFAULT_DT_S),
        default="d1_global_0p25",
        help="Global D1 grid resolution (0p4=44.5km 1-tile@nz65 / 2-tile@nz137)",
    )
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
    parser.add_argument("--d1-restart-in", type=Path, default=None, help="Optional D1 WX-1 restart file for warm-start cycling")
    parser.add_argument("--d2-restart-in", type=Path, default=None, help="Optional D2 WX-1 restart file for warm-start cycling")
    parser.add_argument("--osisaf-wx1", type=Path, default=None, help="Optional OSI-SAF sea-ice WX-1 observation file")
    parser.add_argument("--obs-dir", type=Path, default=PROJECT_ROOT / "data" / "observations")
    parser.add_argument("--download-live", action="store_true", help="Fetch live ECMWF/KNMI/DWD feeds before running")
    parser.add_argument("--enable-da", action="store_true", help="Assimilate KNMI + DWD station obs via Balanced 3D-Var + IAU")
    parser.add_argument("--enable-ocean-si3", action="store_true", help="Enable coupled NEMO 5.0.2 + SI3 GPU ocean/sea-ice")
    parser.add_argument("--no-ocean-si3", action="store_true", help="Disable coupled NEMO+SI3 even in D1 modes")
    parser.add_argument("--no-3d-diagnostics", action="store_true", help="Disable 3D pressure-level & CAPE/synoptic WX-1 diagnostics")
    parser.add_argument("--physics-every", type=int, default=5, help="Substep physics cadence every N dynamics steps")
    parser.add_argument("--dt-d1", type=float, default=None, help="D1 dynamics step [s] (default: auto 60/90/120s for 0p25/0p4/0p5)")
    parser.add_argument("--dt-d2", type=float, default=20.0)
    parser.add_argument("--dt-d3", type=float, default=6.0, help="D3 dynamics step [s] (default: 6.0 for 2.5 km D3; use 4.0 for d3_1p5km)")
    parser.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "data" / "forecasts")
    parser.add_argument("--runs-root", type=Path, default=None, help="Isolated per-cycle runs root (YYYYMMDD_HHZ/...)")
    parser.add_argument("--cycle-restart-hours", type=str, default="6.0", help="Comma-separated cycle restart leads [h]")
    parser.add_argument("--ensemble-members", type=int, default=0, help="0=CF only, 51=CF + 50 perturbed members")
    parser.add_argument("--ensemble-hours", type=float, default=None, help="Ensemble forecast length [h]")
    parser.add_argument("--ensemble-seed", type=int, default=None)
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")
    parser.add_argument(
        "--quiet-steps",
        action="store_true",
        help="Suppress per-step CLI log lines (keep phase/output logs and tqdm bar)",
    )
    parser.add_argument(
        "--pbl-last",
        action="store_true",
        default=False,
        help="IFS 50r1 physics ordering: execute vertical diffusion/PBL after moist microphysics",
    )
    parser.add_argument(
        "--cy46-physics",
        action="store_true",
        default=False,
        help="HARMONIE-AROME Cy46 physics: pressure CDNC profile (0.25x at k=0), Nielsen LW cloud optics, snow-melt f_sn>=0.25, XRIMAX=0.2",
    )
    parser.add_argument(
        "--mem-profile-every",
        type=int,
        default=0,
        help="Optional diagnostic interval [steps] logging RSS, swap, VRAM, live np.ndarray count, and GPU telemetry (0 = off)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[_TqdmLoggingHandler()],
        force=True,
    )
    logger = logging.getLogger("tempest_v2.cli")
    cyc_rst_hours = tuple(float(h.strip()) for h in args.cycle_restart_hours.split(",") if h.strip())
    show_progress = not args.no_progress
    log_steps = not args.quiet_steps

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
    dt_d1_eff = float(args.dt_d1) if args.dt_d1 is not None else D1_DOMAIN_DEFAULT_DT_S[args.d1_domain]

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
            d2_restart_path=args.d2_restart_in,
            d1_restart_path=args.d1_restart_in,
            obs_dir=obs_dir_eff if args.enable_da else None,
            outdir=args.outdir,
            runs_root=args.runs_root,
            cycle_restart_hours=cyc_rst_hours,
            ensemble_members=args.ensemble_members,
            ensemble_lead_hours=args.ensemble_hours,
            ensemble_seed=args.ensemble_seed,
            d1_spec=get_domain_spec(args.d1_domain, nz=args.nz),
            ecmwf_schedule=args.ecmwf,
            enable_da=args.enable_da,
            dt_d1_s=dt_d1_eff,
            dt_d2_s=args.dt_d2,
            dt_d3_s=args.dt_d3,
            physics_every=args.physics_every,
            enable_ocean_si3=use_ocean,
            include_3d_diagnostics=include_3d_diag,
            osisaf_wx1_path=args.osisaf_wx1,
            show_progress=show_progress,
            log_steps=log_steps,
            mem_profile_every=args.mem_profile_every,
            pbl_last=args.pbl_last,
            cy46_physics=args.cy46_physics,
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
            d2_restart_path=args.d2_restart_in,
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
            show_progress=show_progress,
            log_steps=log_steps,
            mem_profile_every=args.mem_profile_every,
            pbl_last=args.pbl_last,
            cy46_physics=args.cy46_physics,
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

    logger.info("=== Single-Domain Mode [%s] | nz=%d | hours=%.2fh ===", args.mode, args.nz, args.hours)
    t_prep0 = time.perf_counter()
    spec = get_domain_spec(args.d1_domain if args.mode == "d1" else args.mode, nz=args.nz)
    hs_m, sstatic = build_domain_static_fields(spec)
    logger.info("[%s] Building initial conditions from %s...", spec.name, args.ic_source)
    rst_single = (
        args.d1_restart_in if (args.mode == "d1" and args.d1_restart_in is not None)
        else (args.d2_restart_in if (args.mode == "d2" and args.d2_restart_in is not None) else args.restart_in)
    )
    if args.ic_source == "restart" and rst_single is not None:
        state_3d, surf_state, meta = load_tempest_restart_wx1(rst_single)
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
    logger.info("[%s] Static & IC ready in %.2fs (init=%s)", spec.name, time.perf_counter() - t_prep0, init_dt.isoformat())

    da_report = {}
    iau_f = None
    if args.enable_da and obs_dir_eff.exists():
        logger.info("[%s] Loading WX-1 DA observations from %s...", spec.name, obs_dir_eff)
        obs_list, stn_rows = load_station_obs_wx1(obs_dir_eff, spec, valid_time_utc=init_dt)
        logger.info("[%s] Loaded %d DA observations across %d surface stations", spec.name, len(obs_list), len(stn_rows))
        if obs_list:
            st_bg = state_3d
            da_report["verify_bg"] = verify_against_stations(spec, st_bg, surf_state, hs_m, stn_rows)
            use_iau = not (spec.is_spherical and spec.nz * spec.ny * spec.nx >= 50_000_000)
            ana_st, surf_state, iau_f, vres = run_domain_3dvar(
                spec, st_bg, surf_state, hs_m, sstatic["lsm"], obs_list, obs_dir=obs_dir_eff, compute_iau=use_iau
            )
            da_report["var3d"] = {"j0": vres.initial_cost_j0, "jf": vres.final_cost_j, "iters": vres.iterations}
            da_report["verify_ana"] = verify_against_stations(spec, ana_st, surf_state, hs_m, stn_rows)
            if iau_f is not None:
                # Bloom et al. (1996) IAU: start prognostic momentum/thermo from background so the increment is added once over the IAU window.
                init_st = dict(ana_st)
                for k_iau in ("u", "v", "theta_p", "pi_p"):
                    init_st[k_iau] = st_bg[k_iau]
                state_3d = init_st
            else:
                state_3d = ana_st
            del ana_st, st_bg
    clear_cfgrib_cache()
    gc.collect()

    dt_s = {
        "d1": dt_d1_eff,
        "d2": args.dt_d2,
        "d3": args.dt_d3,
        "d3_benelux_2p5km": args.dt_d3,
        "d3_1p5km": 4.0 if ("--dt-d3" not in sys.argv) else args.dt_d3,
    }[args.mode]
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
        show_progress=show_progress,
        log_steps=log_steps,
        mem_profile_every=args.mem_profile_every,
        pbl_last=args.pbl_last,
        cy46_physics=args.cy46_physics,
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
