"""Automated Continuous Operational Cycling Daemon for Tempest-NWP v2 (0% AI / 0% ML).

Manages 24/7 operational forecast cycling:
  - Per-cycle isolated directories (`runs/YYYYMMDD_HHZ/...`) with atomic `latest` symlink.
  - Automatic warm-start discovery via `find_latest_valid_cycle_restart` (e.g. from T+6h
    restarts of the prior cycle), with clean ECMWF step-0 cold-start fallback.
  - Exclusive `fcntl.flock` daemon lockfile so overlapping cron/systemd launches cannot corrupt state.
  - Bounded retention cleanup (`prune_old_runs`), signal-safe graceful shutdown, retry backoff,
    and optional `notifier.py` alerts.
"""

from __future__ import annotations

import contextlib
import datetime
import fcntl
import json
import logging
import os
import re
import shutil
import signal
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

from tempest_v2.notifier import notify_run_event
from tempest_v2.projection import DomainSpec, get_domain_spec
from tempest_v2.restart_validator import find_latest_valid_cycle_restart
from tempest_v2.runner import create_run_layout, run_three_nest_pipeline, run_two_nest_pipeline

logger = logging.getLogger("tempest_v2.daemon")

_CYCLE_DIR_RE = re.compile(r"^\d{8}_\d{2}Z(?:_.*)?$")


@dataclass
class DaemonConfig:
    """Configuration for continuous operational NWP cycling."""

    runs_root: Path = Path("runs")
    cycle_hours_utc: Tuple[int, ...] = (0, 6, 12, 18)
    cycle_delay_minutes: int = 35
    poll_interval_s: float = 60.0
    keep_runs: int = 16
    d1_lead_hours: float = 24.0
    d2_lead_hours: float = 24.0
    d3_lead_hours: float = 24.0
    nz: int = 65
    dt_d1_s: float = 60.0
    dt_d2_s: float = 20.0
    dt_d3_s: float = 4.0
    physics_every: int = 5
    enable_d1: bool = False
    enable_ocean_si3: bool = True
    include_3d_diagnostics: bool = True
    cycle_restart_hours: Tuple[float, ...] = (6.0,)
    restart_max_age_hours: float = 24.0
    ensemble_members: int = 0
    ensemble_lead_hours: float = 12.0
    ensemble_seed: Optional[int] = None
    enable_da: bool = True
    enable_radar: bool = True
    enable_notifier: bool = True
    notify_topic: Optional[str] = None
    max_retries: int = 3
    retry_backoff_s: float = 30.0
    lock_file: Optional[Path] = None

    def __post_init__(self) -> None:
        self.runs_root = Path(self.runs_root)
        if not self.cycle_hours_utc:
            raise ValueError("cycle_hours_utc must not be empty")


def _to_utc(dt: datetime.datetime) -> datetime.datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=datetime.timezone.utc)
    return dt.astimezone(datetime.timezone.utc)


def _candidate_cycles_near(
    now_utc: datetime.datetime, cycle_hours_utc: Sequence[int]
) -> List[datetime.datetime]:
    now = _to_utc(now_utc)
    base_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    hrs = sorted({int(h) % 24 for h in cycle_hours_utc})
    cands: List[datetime.datetime] = []
    for day_offset in (-1, 0, 1):
        day = base_day + datetime.timedelta(days=day_offset)
        for hr in hrs:
            cands.append(day.replace(hour=hr))
    cands.sort()
    return cands


def latest_ready_cycle_time_utc(
    now_utc: datetime.datetime,
    cycle_hours_utc: Sequence[int] = (0, 6, 12, 18),
    delay_minutes: int = 35,
) -> Tuple[datetime.datetime, datetime.datetime]:
    """Return the most recent `(cycle_init_utc, ready_at_utc)` whose ready time <= `now_utc`."""
    now = _to_utc(now_utc)
    delay = datetime.timedelta(minutes=float(delay_minutes))
    cands = _candidate_cycles_near(now, cycle_hours_utc)
    ready = [(c, c + delay) for c in cands if c + delay <= now]
    if ready:
        return ready[-1]
    return cands[0], cands[0] + delay


def next_cycle_time_utc(
    now_utc: datetime.datetime,
    cycle_hours_utc: Sequence[int] = (0, 6, 12, 18),
    delay_minutes: int = 35,
) -> Tuple[datetime.datetime, datetime.datetime]:
    """Return the next upcoming `(cycle_init_utc, ready_at_utc)` with ready time > `now_utc`."""
    now = _to_utc(now_utc)
    delay = datetime.timedelta(minutes=float(delay_minutes))
    for c in _candidate_cycles_near(now, cycle_hours_utc):
        r = c + delay
        if r > now:
            return c, r
    last = _candidate_cycles_near(now, cycle_hours_utc)[-1] + datetime.timedelta(days=1)
    return last, last + delay


@contextlib.contextmanager
def acquire_daemon_lock(lock_path: Path | str) -> Iterator[Path]:
    """Acquire a non-blocking exclusive advisory lock (`fcntl.flock`) on `lock_path`."""
    p = Path(lock_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(p), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"Another Tempest daemon already holds lock: {p}") from exc
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        payload = f"pid={os.getpid()} acquired={datetime.datetime.now(datetime.timezone.utc).isoformat()}\n"
        os.write(fd, payload.encode("utf-8"))
        yield p
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(fd)


def update_latest_symlink(runs_root: Path | str, run_dir: Path | str) -> Path:
    """Atomically update `runs_root/latest` symlink to point to `run_dir`."""
    root = Path(runs_root)
    target = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    link_path = root / "latest"
    tmp_link = root / f".latest.tmp.{os.getpid()}.{threading.get_ident()}"
    if tmp_link.exists() or tmp_link.is_symlink():
        tmp_link.unlink()

    link_target: str | Path
    try:
        link_target = target.resolve().relative_to(root.resolve())
    except ValueError:
        link_target = target.resolve()

    os.symlink(str(link_target), str(tmp_link))
    os.replace(str(tmp_link), str(link_path))
    return link_path


def prune_old_runs(
    runs_root: Path | str,
    keep_runs: int,
    protected_dirs: Sequence[Path | str] = (),
) -> List[Path]:
    """Remove old per-cycle run directories beyond newest `keep_runs`, preserving `protected_dirs`."""
    root = Path(runs_root)
    if not root.exists():
        return []

    protected_resolved = {Path(d).resolve() for d in protected_dirs}
    latest_link = root / "latest"
    if latest_link.is_symlink():
        try:
            protected_resolved.add(latest_link.resolve())
        except OSError:
            pass

    cycle_dirs = [
        d for d in root.iterdir()
        if d.is_dir() and not d.is_symlink() and _CYCLE_DIR_RE.match(d.name)
    ]
    cycle_dirs.sort(key=lambda d: (d.name[:12], d.stat().st_mtime), reverse=True)

    keep_n = max(0, int(keep_runs))
    keep_set = {d.resolve() for d in cycle_dirs[:keep_n]} | protected_resolved

    removed: List[Path] = []
    for d in cycle_dirs:
        if d.resolve() not in keep_set:
            shutil.rmtree(d)
            removed.append(d)
    return removed


def run_single_cycle(
    cfg: DaemonConfig,
    init_time_utc: datetime.datetime,
    *,
    run_tag: Optional[str] = None,
    d1_spec: Optional[DomainSpec] = None,
    d2_spec: Optional[DomainSpec] = None,
    d3_spec: Optional[DomainSpec] = None,
    pipeline_fn: Optional[Callable[..., Dict[str, Any]]] = None,
    obs_fn: Optional[Callable[[Path, datetime.datetime], Optional[Path]]] = None,
    ic_fn: Optional[Callable[[Path, datetime.datetime], Tuple[Path, Path]]] = None,
) -> Dict[str, Any]:
    """Execute one isolated operational cycle with automatic warm-start discovery & retention."""
    t0_wall = time.perf_counter()
    init_dt = _to_utc(init_time_utc)
    layout = create_run_layout(cfg.runs_root, init_dt, run_tag=run_tag)
    run_dir = layout["run_dir"]
    run_id = run_dir.name

    d1_name = d1_spec.name if d1_spec is not None else "d1_global_0p25"
    d2_name = d2_spec.name if d2_spec is not None else "d2_eu_lcc"
    d3_name = d3_spec.name if d3_spec is not None else "d3_benelux_de_1p5km"
    d1_shape = (int(d1_spec.nz), int(d1_spec.ny), int(d1_spec.nx)) if d1_spec is not None else None
    d2_shape = (int(d2_spec.nz), int(d2_spec.ny), int(d2_spec.nx)) if d2_spec is not None else None
    d3_shape = (int(d3_spec.nz), int(d3_spec.ny), int(d3_spec.nx)) if d3_spec is not None else None

    # 1. Discover valid warm-start restarts from prior cycles' `restarts/` directories
    d3_restart = find_latest_valid_cycle_restart(
        cfg.runs_root,
        d3_name,
        init_dt,
        max_age_hours=cfg.restart_max_age_hours,
        expected_shape=d3_shape,
        exact_valid_match=True,
    )
    d2_restart = find_latest_valid_cycle_restart(
        cfg.runs_root,
        d2_name,
        init_dt,
        max_age_hours=cfg.restart_max_age_hours,
        expected_shape=d2_shape,
        exact_valid_match=True,
    )
    d1_restart: Optional[Path] = None
    if cfg.enable_d1 and pipeline_fn is None:
        d1_restart = find_latest_valid_cycle_restart(
            cfg.runs_root,
            d1_name,
            init_dt,
            max_age_hours=cfg.restart_max_age_hours,
            expected_shape=d1_shape,
            exact_valid_match=True,
        )

    sfc_grib: Optional[Path] = None
    pl_grib: Optional[Path] = None
    if d3_restart is not None:
        ic_source = "restart"
        logger.info("[%s] Warm-starting cycle from %s (d2=%s, d1=%s)", run_id, d3_restart, d2_restart, d1_restart)
    else:
        ic_source = "ecmwf"
        logger.info("[%s] Cold-starting cycle via ECMWF step-0 analysis", run_id)
        if ic_fn is not None:
            sfc_grib, pl_grib = ic_fn(layout["config"], init_dt)
        else:
            from tempest_v2.ingest_ecmwf import download_ecmwf_cold_start

            sfc_grib, pl_grib, _ = download_ecmwf_cold_start(
                target_time=init_dt,
                step=0,
                prefer_0p1=False,
                output_dir=layout["config"] / "raw_ecmwf",
            )

    # 2. Observation & Radar QPE ingest into isolated cycle obs/ directory
    obs_dir_used: Optional[Path] = None
    if cfg.enable_da:
        if obs_fn is not None:
            obs_dir_used = obs_fn(layout["obs"], init_dt)
        else:
            from tempest_v2.download_all_observations import run_observation_ingest_pipeline

            radar_spec = None
            if cfg.enable_radar:
                radar_spec = d3_spec if d3_spec is not None else get_domain_spec("d3", nz=cfg.nz)
            run_obs_dir, _counts = run_observation_ingest_pipeline(
                layout["obs"],
                da_hours=6.0,
                zulu_time=init_dt.strftime("%Y%m%d_%H%MZ"),
                notify_topic=None,
                include_sondehub=True,
                spec_for_radar=radar_spec,
            )
            obs_dir_used = run_obs_dir

    osisaf_path: Optional[Path] = None
    if obs_dir_used is not None and Path(obs_dir_used).is_dir():
        main_osi = sorted(Path(obs_dir_used).rglob("osi_saf_sea_ice.wx1"))
        if main_osi:
            osisaf_path = main_osi[-1]
        else:
            osisaf_matches = sorted(Path(obs_dir_used).rglob("osisaf*.wx1"))
            if osisaf_matches:
                osisaf_path = osisaf_matches[-1]

    # 3. Execute operational 2-nest (D2->D3) or 3-nest (D1->D2->D3) + optional ensemble inside `layout["run_dir"]`
    if cfg.enable_notifier:
        notify_run_event(
            "start",
            run_id,
            cfg.d3_lead_hours,
            0.0,
            output_dir=run_dir,
            ntfy_topic=cfg.notify_topic,
        )

    common_kwargs: Dict[str, Any] = {
        "forecast_hours": cfg.d3_lead_hours,
        "nz": cfg.nz,
        "ic_source": ic_source,
        "sfc_grib": sfc_grib,
        "pl_grib": pl_grib,
        "restart_path": d3_restart,
        "d2_restart_path": d2_restart,
        "obs_dir": obs_dir_used,
        "run_dir": run_dir,
        "cycle_restart_hours": cfg.cycle_restart_hours,
        "ensemble_members": cfg.ensemble_members,
        "ensemble_lead_hours": cfg.ensemble_lead_hours,
        "ensemble_seed": cfg.ensemble_seed,
        "d2_spec": d2_spec,
        "d3_spec": d3_spec,
        "enable_da": cfg.enable_da,
        "dt_d2_s": cfg.dt_d2_s,
        "dt_d3_s": cfg.dt_d3_s,
        "physics_every": cfg.physics_every,
    }
    if pipeline_fn is not None:
        pipe_res = pipeline_fn(**common_kwargs)
    elif cfg.enable_d1:
        pipe_res = run_three_nest_pipeline(
            **common_kwargs,
            d1_forecast_hours=cfg.d1_lead_hours,
            d1_spec=d1_spec,
            d1_restart_path=d1_restart,
            dt_d1_s=cfg.dt_d1_s,
            enable_ocean_si3=cfg.enable_ocean_si3,
            include_3d_diagnostics=cfg.include_3d_diagnostics,
            osisaf_wx1_path=osisaf_path,
        )
    else:
        pipe_res = run_two_nest_pipeline(
            **common_kwargs,
            enable_ocean_si3=cfg.enable_ocean_si3,
            include_3d_diagnostics=cfg.include_3d_diagnostics,
            osisaf_wx1_path=osisaf_path,
        )

    latest_link = update_latest_symlink(cfg.runs_root, run_dir)
    pruned = prune_old_runs(cfg.runs_root, cfg.keep_runs, protected_dirs=[run_dir])
    elapsed_s = time.perf_counter() - t0_wall

    cycle_summary = {
        "run_id": run_id,
        "init_time_utc": init_dt.isoformat(),
        "run_dir": str(run_dir),
        "latest_symlink": str(latest_link),
        "ic_source": ic_source,
        "warm_started": bool(ic_source == "restart"),
        "restart_path": str(d3_restart) if d3_restart is not None else None,
        "d2_restart_path": str(d2_restart) if d2_restart is not None else None,
        "d1_restart_path": str(d1_restart) if d1_restart is not None else None,
        "pruned_runs": [str(p) for p in pruned],
        "elapsed_s": elapsed_s,
        "pipeline": pipe_res,
    }
    (layout["config"] / "daemon_cycle.json").write_text(json.dumps(cycle_summary, indent=2, default=str))

    if cfg.enable_notifier:
        notify_run_event(
            "success",
            run_id,
            cfg.d3_lead_hours,
            elapsed_s,
            metrics=pipe_res.get("da") if isinstance(pipe_res, dict) else None,
            output_dir=run_dir,
            ntfy_topic=cfg.notify_topic,
        )

    return cycle_summary


def run_daemon_loop(
    cfg: DaemonConfig,
    *,
    max_cycles: Optional[int] = None,
    stop_event: Optional[threading.Event] = None,
    explicit_cycle_times: Optional[Sequence[datetime.datetime]] = None,
    d1_spec: Optional[DomainSpec] = None,
    d2_spec: Optional[DomainSpec] = None,
    d3_spec: Optional[DomainSpec] = None,
    pipeline_fn: Optional[Callable[..., Dict[str, Any]]] = None,
    obs_fn: Optional[Callable[[Path, datetime.datetime], Optional[Path]]] = None,
    ic_fn: Optional[Callable[[Path, datetime.datetime], Tuple[Path, Path]]] = None,
) -> List[Dict[str, Any]]:
    """Run continuous 24/7 operational forecast cycling with locking, signal safety, and backoff."""
    cfg.runs_root.mkdir(parents=True, exist_ok=True)
    lock_path = cfg.lock_file if cfg.lock_file is not None else (cfg.runs_root / ".daemon.lock")
    ev = stop_event if stop_event is not None else threading.Event()

    prev_int = None
    prev_term = None
    in_main_thread = threading.current_thread() is threading.main_thread()

    def _on_signal(_signum: int, _frame: Any) -> None:
        logger.info("Shutdown signal received; stopping after active cycle completes.")
        ev.set()

    if in_main_thread:
        prev_int = signal.getsignal(signal.SIGINT)
        prev_term = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, _on_signal)
        signal.signal(signal.SIGTERM, _on_signal)

    results: List[Dict[str, Any]] = []
    explicit_idx = 0
    try:
        with acquire_daemon_lock(lock_path):
            while not ev.is_set():
                if max_cycles is not None and len(results) >= int(max_cycles):
                    break

                target_dt: Optional[datetime.datetime] = None
                if explicit_cycle_times is not None:
                    if explicit_idx >= len(explicit_cycle_times):
                        break
                    target_dt = _to_utc(explicit_cycle_times[explicit_idx])
                    explicit_idx += 1
                else:
                    now_utc = datetime.datetime.now(datetime.timezone.utc)
                    ready_dt, _ = latest_ready_cycle_time_utc(
                        now_utc, cfg.cycle_hours_utc, cfg.cycle_delay_minutes
                    )
                    cand_dir = cfg.runs_root / f"{ready_dt:%Y%m%d_%H}Z"
                    if not (cand_dir / "manifest.json").exists():
                        target_dt = ready_dt
                    else:
                        _nxt_dt, nxt_ready = next_cycle_time_utc(
                            now_utc, cfg.cycle_hours_utc, cfg.cycle_delay_minutes
                        )
                        wait_s = max(0.5, min(float(cfg.poll_interval_s), (nxt_ready - now_utc).total_seconds()))
                        ev.wait(timeout=wait_s)
                        continue

                if target_dt is None:
                    break

                attempt = 0
                while not ev.is_set():
                    attempt += 1
                    try:
                        res = run_single_cycle(
                            cfg,
                            target_dt,
                            d1_spec=d1_spec,
                            d2_spec=d2_spec,
                            d3_spec=d3_spec,
                            pipeline_fn=pipeline_fn,
                            obs_fn=obs_fn,
                            ic_fn=ic_fn,
                        )
                        results.append(res)
                        break
                    except Exception as exc:
                        logger.exception("Cycle %s attempt %d failed: %s", target_dt.isoformat(), attempt, exc)
                        if cfg.enable_notifier:
                            notify_run_event(
                                "error",
                                f"{target_dt:%Y%m%d_%H}Z",
                                cfg.d3_lead_hours,
                                0.0,
                                metrics={"error": str(exc), "attempt": attempt},
                                ntfy_topic=cfg.notify_topic,
                            )
                        if attempt >= max(1, int(cfg.max_retries)):
                            raise
                        ev.wait(timeout=float(cfg.retry_backoff_s) * attempt)
    finally:
        if in_main_thread and prev_int is not None and prev_term is not None:
            signal.signal(signal.SIGINT, prev_int)
            signal.signal(signal.SIGTERM, prev_term)

    return results
