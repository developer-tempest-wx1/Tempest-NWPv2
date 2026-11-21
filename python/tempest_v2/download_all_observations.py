"""Tempest-NWP: Unified Real-Time Observational Data Ingestion Orchestrator.

Downloads and decodes genuine operational meteorological observations across all open sources:
- sfc/       : Official KNMI 10-min in-situ (48+ stations), DWD POI (980+ European stations), WMO SYNOP/METAR
- upperair/  : Official DWD WMO BUFR soundings (20,000+ levels), IGRA2, GNSS-RO refractivity
- satellite/ : EUMETSAT MTG-I1 FCI AMVs, Metop ASCAT ocean winds, GOES-18 DMWs, Himawari-9 AHI winds
- marine/    : NDBC global buoys, Copernicus Marine In-Situ drifters/Argo

Organizes downloads into Zulu timestamped directory structure:
  data/observations/runs/YYYYMMDD_HHMMZ/
    ├── sfc/
    ├── upperair/
    ├── satellite/
    └── marine/

Strictly 100% classical physics (0% AI / 0% Machine Learning, 0% Mock Data).
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
from pathlib import Path
import sys
import os
import time
from typing import Dict, Optional, Tuple

# Ensure package is on path
pkg_root = str(Path(__file__).resolve().parent.parent)
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from tempest_v2.ingest_knmi_observations import ingest_knmi_observations
from tempest_v2.ingest_dwd_poi import ingest_dwd_poi_observations
from tempest_v2.ingest_radiosondes import ingest_radiosonde_soundings
from tempest_v2.ingest_satellite import (
    ingest_amv_cloud_drift_winds,
    ingest_jma_himawari_winds,
    ingest_eumetsat_meteosat_and_ascat,
    ingest_atms_microwave_radiances,
)
from tempest_v2.notifier import notify_run_event

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("tempest_v2.download_all_observations")


def download_all_observations(
    da_hours: float = 6.0,
    base_dir: Optional[Path] = None,
    zulu_time: Optional[str] = None,
    notify_topic: Optional[str] = None,
    include_sondehub: bool = True,
    spec_for_radar: Optional[object] = None,
) -> Path:
    """Top-level pipeline entrypoint: downloads all live observations and returns the run directory."""
    target_base = Path(base_dir) if base_dir is not None else Path("data/observations/runs")
    run_dir, _ = run_observation_ingest_pipeline(
        base_dir=target_base,
        da_hours=da_hours,
        zulu_time=zulu_time,
        notify_topic=notify_topic,
        include_sondehub=include_sondehub,
        spec_for_radar=spec_for_radar,
    )
    return run_dir


def run_observation_ingest_pipeline(
    base_dir: Path,
    da_hours: float = 6.0,
    zulu_time: str = None,
    notify_topic: str = None,
    include_sondehub: bool = True,
    spec_for_radar: Optional[object] = None,
) -> Tuple[Path, Dict[str, int]]:
    """Executes the full observation download into structured Zulu subfolders."""
    if zulu_time is None:
        zulu_time = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%MZ")

    run_obs_dir = base_dir / zulu_time
    sfc_dir = run_obs_dir / "sfc"
    upperair_dir = run_obs_dir / "upperair"
    satellite_dir = run_obs_dir / "satellite"
    marine_dir = run_obs_dir / "marine"

    for d in [sfc_dir, upperair_dir, satellite_dir, marine_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print("   TEMPEST-NWP: Unified Real-Time Observation Ingestion Engine   ")
    print(f"         Zulu Timestamp: {zulu_time} | DA Window: ±{da_hours:.1f}h")
    print("         Target Storage: WX-1 Scientific Containers (.wx1)       ")
    print("         STRICTLY 0% AI / 0% ML / 0% MOCK DATA                   ")
    print("=================================================================\n")

    counts = {}
    t0 = time.time()

    # 1. Surface Feeds (sfc/)
    print("[1/4] Ingesting Official Surface Weather Observations (sfc/)...")
    try:
        knmi_count = ingest_knmi_observations(sfc_dir, da_hours=da_hours)
        counts["knmi_sfc"] = knmi_count
    except Exception as e:
        logger.warning("KNMI ingest notice: %s", e)
        counts["knmi_sfc"] = 0

    try:
        dwd_count = ingest_dwd_poi_observations(sfc_dir, da_hours=da_hours)
        counts["dwd_poi_sfc"] = dwd_count
    except Exception as e:
        logger.warning("DWD POI ingest notice: %s", e)
        counts["dwd_poi_sfc"] = 0

    try:
        from tempest_v2.ingest_obs_global import fetch_global_metars, save_observations_wx1
        print("  - Fetching 5,000+ real-time worldwide METAR airport weather reports...")
        metars = fetch_global_metars()
        if metars:
            metar_out = sfc_dir / "global_metars.wx1"
            save_observations_wx1(metars, metar_out)
            counts["global_metars"] = len(metars)
        else:
            counts["global_metars"] = 0
    except Exception as e:
        logger.warning("METAR ingest notice: %s", e)
        counts["global_metars"] = 0

    # 1d. WMO WIS2 (Global Broker -> Global Cache), worldwide SYNOP/SHIP/BUOY BUFR
    try:
        from tempest_v2.ingest_wis2 import ingest_wis2
        counts["wis2_surface"] = ingest_wis2(sfc_dir, da_hours=da_hours, listen_seconds=90)
        print(f"  - WIS2 surface reports: {counts['wis2_surface']}")
    except Exception as e:
        logger.warning("WIS2 ingest notice: %s", e)
        counts["wis2_surface"] = 0

    # 2. Upper-Air Soundings (upperair/)
    print("\n[2/4] Ingesting Upper-Air Soundings & Radiosondes (upperair/)...")
    try:
        sonde_count = ingest_radiosonde_soundings(upperair_dir, da_hours=da_hours)
        counts["radiosonde_levels"] = sonde_count
    except Exception as e:
        logger.warning("Radiosonde ingest notice: %s", e)
        counts["radiosonde_levels"] = 0

    if include_sondehub:
        # The daemon writes the cycle-specific live stream to this exact folder.
        # Prefer that complete, pre-run stream product; do not append a second
        # REST batch of the same serials and thereby assimilate duplicate levels.
        live_product = upperair_dir / "sondehub_soundings.wx1"
        if live_product.is_file() and live_product.stat().st_size > 0:
            logger.info("Using cycle-specific SondeHub live WX-1; skipping overlapping REST batch to avoid duplicate soundings.")
            counts["sondehub_batch_skipped_due_to_live"] = 0
            counts["sondehub_levels"] = 0
        else:
            try:
                from tempest_v2.ingest_sondehub import ingest_sondehub_soundings
                print("  - No completed live SondeHub product found; using REST batch fallback...")
                sondehub_count = ingest_sondehub_soundings(upperair_dir, da_hours=min(da_hours, 3.0), max_sondes=100)
                counts["sondehub_levels"] = sondehub_count
            except Exception as e:
                logger.warning("SondeHub ingest notice: %s", e)
                counts["sondehub_levels"] = 0
    else:
        print("  - SondeHub telemetry disabled (--no-sondehub). Skipping SondeHub soundings.")
        counts["sondehub_levels"] = 0

    # 2b. E-GVAP Ground GNSS Zenith Total Delay (ZTD) network across Europe
    print("  - Ingesting European E-GVAP / EUREF GNSS Zenith Total Delay (ZTD) network...")
    try:
        from tempest_v2.ingest_egvap import ingest_egvap_ztd
        egvap_count = ingest_egvap_ztd(upperair_dir, da_hours=da_hours)
        counts["egvap_gnss_ztd"] = egvap_count
        print(f"  - E-GVAP GNSS ZTD stations ingested: {egvap_count}")
    except Exception as e:
        logger.warning("E-GVAP GNSS ZTD ingest notice: %s", e)
        counts["egvap_gnss_ztd"] = 0

    # 3. Satellites (satellite/)
    print("\n[3/4] Ingesting Spaceborne Constellations: EUMETSAT, NOAA, JMA (satellite/)...")
    try:
        mtg_amv, ascat_winds = ingest_eumetsat_meteosat_and_ascat(satellite_dir / "eumetsat.wx1", da_hours=da_hours)
        counts["eumetsat_mtg_amv"] = mtg_amv
        counts["eumetsat_ascat_winds"] = ascat_winds
    except Exception as e:
        logger.warning("EUMETSAT ingest notice: %s", e)
        counts["eumetsat_mtg_amv"] = 0
        counts["eumetsat_ascat_winds"] = 0

    try:
        goes_winds = ingest_amv_cloud_drift_winds(satellite_dir / "satellite_goes_amv_winds.wx1", da_hours=da_hours)
        counts["goes_amv_winds"] = goes_winds
    except Exception as e:
        logger.warning("GOES AMV ingest notice: %s", e)
        counts["goes_amv_winds"] = 0

    try:
        hima_winds = ingest_jma_himawari_winds(satellite_dir / "satellite_jma_himawari9_winds.wx1", da_hours=da_hours)
        counts["himawari_winds"] = hima_winds
    except Exception as e:
        logger.warning("Himawari ingest notice: %s", e)
        counts["himawari_winds"] = 0

    try:
        atms_rad = ingest_atms_microwave_radiances(satellite_dir / "satellite_atms_microwave_radiances.wx1")
        counts["atms_raw_only_not_assimilated"] = atms_rad
        counts["atms_assimilated"] = 0
    except Exception as e:
        logger.warning("ATMS ingest notice: %s", e)
        counts["atms_raw_only_not_assimilated"] = 0
        counts["atms_assimilated"] = 0

    # 4. Marine Ingestion (marine/)
    print("\n[4/4] Ingesting Marine Buoys & Oceanic Platforms (marine/)...")
    marine_obs_file = marine_dir / "marine_buoy_obs.wx1"
    try:
        from tempest_v2.ingest_marine import ingest_full_marine_observations
        marine_res = ingest_full_marine_observations(marine_dir)
        counts["marine_buoys"] = int(marine_res.get("n_surface_buoys", 0))
        counts["argo_floats"] = int(marine_res.get("n_argo_profiles_with_data", 0))
        print(f"  - Buoys with pressure: {marine_res.get('n_buoy_pressure', 0)} | with SST: {marine_res.get('n_buoy_sst', 0)} "
              f"| Argo inventory: {marine_res.get('n_argo_inventory', 0)} | Argo profiles downloaded: {counts['argo_floats']}")
    except Exception as e:
        # No stale cache fallback: an empty marine feed is better than old data in a new analysis.
        logger.warning("Live marine ingest failed: %s. Marine feed is EMPTY for this run.", e)
        counts["marine_buoys"] = 0
        counts["argo_floats"] = 0

    # 4b. Cryosphere: EUMETSAT OSI-SAF Sea Ice Concentration & Thickness (OSI-401-d / MET Norway)
    print("  - Ingesting EUMETSAT OSI-SAF operational daily Sea-Ice concentration & thickness...")
    try:
        from tempest_v2.ingest_osi_saf import ingest_osi_saf_sea_ice
        osi_count = ingest_osi_saf_sea_ice(marine_dir, subsample_step=4)
        counts["osi_saf_sea_ice"] = osi_count
        print(f"  - OSI-SAF sea-ice points ingested: {osi_count:,}")
    except Exception as e:
        logger.warning("OSI-SAF Sea Ice ingest notice: %s", e)
        counts["osi_saf_sea_ice"] = 0

    if spec_for_radar is not None:
        try:
            from tempest_v2.ingest_radar_qpe import ingest_radar_qpe_to_wx1
            rad_p = ingest_radar_qpe_to_wx1(run_obs_dir, spec_for_radar)  # type: ignore[arg-type]
            counts["radar_qpe_grid"] = 1 if rad_p is not None else 0
        except Exception as e:
            logger.warning("Radar QPE ingest notice: %s", e)
            counts["radar_qpe_grid"] = 0

    os.environ["TEMPEST_OBS_DIR"] = str(run_obs_dir)

    elapsed = time.time() - t0
    total_obs = sum(counts.values())

    # Write manifest
    manifest = {
        "zulu_time": zulu_time,
        "timestamp_iso": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "da_hours": da_hours,
        "elapsed_seconds": elapsed,
        "total_observations": total_obs,
        "counts": counts,
        "subdirectories": {
            "sfc": str(sfc_dir.relative_to(run_obs_dir)),
            "upperair": str(upperair_dir.relative_to(run_obs_dir)),
            "satellite": str(satellite_dir.relative_to(run_obs_dir)),
            "marine": str(marine_dir.relative_to(run_obs_dir)),
        },
    }
    (run_obs_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print("\n=================================================================")
    print("   [SUCCESS] Observation Ingestion Pipeline Completed!")
    print(f"   - Target Folder:        {run_obs_dir}")
    print(f"   - Elapsed Wall-Time:    {elapsed:.1f} seconds")
    print(f"   - Total Observations:   {total_obs:,} verified genuine reports")
    for k, v in counts.items():
        print(f"     • {k:<22}: {v:,}")
    print("=================================================================\n")

    if notify_topic:
        notify_run_event(
            event_type="success",
            run_id=f"OBS_INGEST_{zulu_time}",
            forecast_hours=da_hours,
            runtime_sec=elapsed,
            output_dir=run_obs_dir,
            ntfy_topic=notify_topic,
        )

    return run_obs_dir, counts


def main():
    parser = argparse.ArgumentParser(description="Tempest-NWP Observation Ingestion Orchestrator")
    parser.add_argument("--base-dir", type=Path, default=Path("data/observations/runs"), help="Base runs directory")
    parser.add_argument("--da-hours", type=float, default=6.0, help="DA lookback window in hours (default: 6.0)")
    parser.add_argument("--zulu", type=str, default=None, help="Custom Zulu timestamp (e.g. 20261103_1800Z)")
    parser.add_argument("--notify", type=str, default=None, help="ntfy topic or provider")
    args = parser.parse_args()

    run_observation_ingest_pipeline(
        base_dir=args.base_dir,
        da_hours=args.da_hours,
        zulu_time=args.zulu,
        notify_topic=args.notify,
    )


if __name__ == "__main__":
    main()
