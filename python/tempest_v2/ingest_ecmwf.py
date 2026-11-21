"""ECMWF Open Data Real-Time Operational Forecast Downloader.

Downloads 0.25° / 0.4° IFS (Integrated Forecasting System) real-time forecasts
directly from ECMWF's cloud mirrors (AWS / Azure).
"""

from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import Sequence

logger = logging.getLogger("tempest_v2.ingest_ecmwf")

# Full 14 isobaric levels in ECMWF OpenData (from 1000 hPa surface up to 10 hPa stratosphere)
STANDARD_PRESSURE_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50, 10]
STANDARD_PL_VARS = ["u", "v", "t", "q", "gh"]

# Complete surface, ocean, sea-ice, and cloud physics parameters
STANDARD_SFC_VARS = [
    "msl", "sp", "2t", "2d", "10u", "10v", "100u", "100v",
    "skt", "lsm", "z", "sithick", "zos",
    "sd", "rsn", "asn",
    "tclw", "tciw", "tcwv",
]

# 4-layer soil temperature and volumetric moisture (levels 1, 2, 3, 4)
STANDARD_SOIL_LEVELS = [1, 2, 3, 4]
STANDARD_SOIL_VARS = ["sot", "vsw"]


def download_ecmwf_data(
    step: int = 0,
    resol: str = "0p25",
    output_dir: str | Path = "data/raw_ecmwf",
    levels: Sequence[int] | None = None,
    pl_vars: Sequence[str] | None = None,
    sfc_vars: Sequence[str] | None = None,
    include_soil: bool = True,
    source: str = "azure",
    run_date: str | None = None,
    run_time: int | str | None = None,
) -> tuple[Path, Path]:
    """Downloads real-time operational ECMWF IFS forecast files in GRIB2 format.

    Args:
        step: Forecast step in hours (0 = initial analysis, 3, 6, 9...).
        resol: Resolution string ('0p1' for 0.1°, '0p25' for 0.25°, or '0p4' for 0.4°).
        output_dir: Target directory for raw downloaded files.
        levels: List of isobaric pressure levels in hPa.
        pl_vars: List of 3D upper-air parameters.
        sfc_vars: List of 2D surface parameters.
        include_soil: Whether to retrieve 4-layer soil moisture and temperature.
        source: Cloud mirror ('azure' or 'aws').
        run_date: Explicit UTC cycle date YYYY-MM-DD (optional).
        run_time: Explicit UTC cycle hour (00, 06, 12, 18; optional).

    Returns:
        Tuple of (sfc_grib_path, pl_grib_path).
    """
    from ecmwf.opendata import Client

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_levels = list(levels) if levels is not None else STANDARD_PRESSURE_LEVELS
    target_pl_vars = list(pl_vars) if pl_vars is not None else STANDARD_PL_VARS
    target_sfc_vars = list(sfc_vars) if sfc_vars is not None else STANDARD_SFC_VARS

    cycle_stamp = None
    if run_date is not None or run_time is not None:
        if run_date is None or run_time is None:
            raise ValueError("run_date and run_time must be supplied together")
        hour = int(str(run_time).replace("Z", ""))
        if hour not in (0, 6, 12, 18):
            raise ValueError("ECMWF IFS run_time must be one of 00, 06, 12, 18 UTC")
        cycle_stamp = f"{run_date.replace('-', '')}_{hour:02d}Z"

    # Preserve legacy 'latest' filenames so existing static/background cache
    # readers continue to locate the normal downloader's output. Explicit cycle
    # selection gets a unique timestamped file set and can safely coexist.
    if cycle_stamp is None:
        sfc_path = out_dir / f"ecmwf_ifs_{resol}_sfc_step{step:03d}.grib2"
        pl_path = out_dir / f"ecmwf_ifs_{resol}_pl_step{step:03d}.grib2"
        soil_path = out_dir / f"ecmwf_ifs_{resol}_soil_step{step:03d}.grib2"
        meta_path = out_dir / f"ecmwf_ifs_{resol}_step{step:03d}_metadata.json"
    else:
        sfc_path = out_dir / f"ecmwf_ifs_{resol}_{cycle_stamp}_sfc_step{step:03d}.grib2"
        pl_path = out_dir / f"ecmwf_ifs_{resol}_{cycle_stamp}_pl_step{step:03d}.grib2"
        soil_path = out_dir / f"ecmwf_ifs_{resol}_{cycle_stamp}_soil_step{step:03d}.grib2"
        meta_path = out_dir / f"ecmwf_ifs_{resol}_{cycle_stamp}_step{step:03d}_metadata.json"

    client = Client(source=source, model="ifs", resol=resol)

    # 1. Retrieve 2D surface fields. ecmwf-opendata returns the actual run datetime;
    # preserve it so a cold-start model cycle never invents a valid time.
    logger.info("Downloading ECMWF %s full surface parameters %s (step=%d)...", resol, target_sfc_vars, step)
    retrieve_cycle = {}
    if run_date is not None and run_time is not None:
        retrieve_cycle = {"date": run_date, "time": int(str(run_time).replace("Z", ""))}
    sfc_result = client.retrieve(
        **retrieve_cycle,
        step=step,
        type="fc",
        param=target_sfc_vars,
        target=str(sfc_path),
    )
    logger.info("Saved surface GRIB2 to %s (size: %.2f MB)", sfc_path, sfc_path.stat().st_size / 1e6)

    # 2. Retrieve 3D pressure level fields (all 14 levels including 10 hPa stratosphere)
    logger.info("Downloading ECMWF %s 3D pressure levels %s (step=%d)...", resol, target_levels, step)
    pl_result = client.retrieve(
        **retrieve_cycle,
        step=step,
        type="fc",
        levtype="pl",
        levelist=target_levels,
        param=target_pl_vars,
        target=str(pl_path),
    )
    logger.info("Saved pressure level GRIB2 to %s (size: %.2f MB)", pl_path, pl_path.stat().st_size / 1e6)

    # 3. Retrieve 4-layer soil variables if requested. Compare the provider-returned
    # cycle for every product because Open Data can move to a newer cycle between
    # independent requests. Never silently combine unlike valid times.
    soil_result = None
    if include_soil:
        logger.info("Downloading ECMWF %s 4-layer soil moisture & temperature %s (step=%d)...", resol, STANDARD_SOIL_VARS, step)
        try:
            soil_result = client.retrieve(
                **retrieve_cycle,
                step=step,
                type="fc",
                levtype="sol",
                levelist=STANDARD_SOIL_LEVELS,
                param=STANDARD_SOIL_VARS,
                target=str(soil_path),
            )
            logger.info("Saved soil GRIB2 to %s (size: %.2f MB)", soil_path, soil_path.stat().st_size / 1e6)
        except Exception as e:
            logger.warning("Could not download soil parameters: %s", e)

    def _provider_time(result: object | None) -> datetime.datetime | None:
        value = getattr(result, "datetime", None) if result is not None else None
        if value is None:
            return None
        if isinstance(value, str):
            value = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
        if not isinstance(value, datetime.datetime):
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=datetime.timezone.utc)
        return value.astimezone(datetime.timezone.utc)

    # Store the provider-reported forecast cycle (not wall-clock download time).
    sfc_dt = _provider_time(sfc_result)
    pl_dt = _provider_time(pl_result)
    soil_dt = _provider_time(soil_result)
    if sfc_dt is not None and pl_dt is not None:
        if sfc_dt != pl_dt:
            raise RuntimeError(
                "ECMWF surface and pressure files came from different forecast cycles: "
                f"{sfc_dt.isoformat()} vs {pl_dt.isoformat()}"
            )
        if soil_dt is not None and soil_dt != sfc_dt:
            raise RuntimeError(
                "ECMWF soil and atmospheric files came from different forecast cycles: "
                f"soil={soil_dt.isoformat()} atmosphere={sfc_dt.isoformat()}"
            )
        valid_dt = sfc_dt + datetime.timedelta(hours=int(step))
        meta_path.write_text(json.dumps({
            "provider": "ECMWF Open Data",
            "model": "ifs",
            "resolution": resol,
            "step_hours": int(step),
            "run_time_utc": sfc_dt.isoformat(),
            "valid_time_utc": valid_dt.isoformat(),
            "surface_file": sfc_path.name,
            "pressure_file": pl_path.name,
            "soil_file": soil_path.name if soil_result is not None and soil_path.exists() else None,
            "source": source,
        }, indent=2), encoding="utf-8")
        logger.info("ECMWF source cycle recorded as %s in %s", sfc_dt.isoformat(), meta_path)
    else:
        logger.warning(
            "ECMWF client returned no run datetime for all atmospheric products; "
            "strict source-time alignment will reject this background"
        )

    return sfc_path, pl_path


def download_ecmwf_cold_start(
    step: int = 0,
    output_dir: str | Path = "data/raw_ecmwf",
    prefer_0p1: bool = True,
    source: str = "azure",
    run_date: str | None = None,
    run_time: int | str | None = None,
) -> tuple[Path, Path, str]:
    """Download ECMWF IFS cold-start field, preferring 0.1 deg (~9 km) with clean fallback to 0.25 deg."""
    resol_order = ("0p1", "0p25") if prefer_0p1 else ("0p25",)
    last_exc: Exception | None = None
    for resol in resol_order:
        try:
            sfc_p, pl_p = download_ecmwf_data(
                step=step,
                resol=resol,
                output_dir=output_dir,
                include_soil=True,
                source=source,
                run_date=run_date,
                run_time=run_time,
            )
            return sfc_p, pl_p, resol
        except Exception as exc:
            logger.warning("ECMWF %s download failed (%s); trying next resolution fallback.", resol, exc)
            last_exc = exc
    raise RuntimeError(f"All ECMWF Open Data resolutions failed for step={step}: {last_exc}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download real-time operational ECMWF IFS initial analysis fields.")
    parser.add_argument("--step", type=int, default=0, help="Forecast step in hours (0 = initial analysis)")
    parser.add_argument("--resol", choices=["0p25", "0p1", "0p4"], default="0p25", help="Horizontal resolution")
    parser.add_argument("--outdir", default="data/raw_ecmwf", help="Output directory for downloaded GRIB2 files")
    parser.add_argument("--date", default=None, help="Explicit ECMWF UTC cycle date YYYY-MM-DD")
    parser.add_argument("--time", type=int, choices=[0, 6, 12, 18], default=None, help="Explicit ECMWF UTC cycle hour")
    args = parser.parse_args()

    if (args.date is None) != (args.time is None):
        parser.error("--date and --time must be supplied together")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    sfc, pl = download_ecmwf_data(step=args.step, resol=args.resol, output_dir=args.outdir, run_date=args.date, run_time=args.time)
    print("\n[SUCCESS] ECMWF Analysis Data Ingestion Complete!")
    print(f"  - 2D Surface GRIB2:        {sfc}")
    print(f"  - 3D Pressure Level GRIB2: {pl}")
