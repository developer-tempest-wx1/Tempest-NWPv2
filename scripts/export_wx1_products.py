"""CLI utility to export Tempest-NWP v2 WX-1 forecasts as CSV, maps, animations, meteograms & Skew-T.

Examples:
    # Interactive location selection on latest run:
    python scripts/export_wx1_products.py

    # Explicit locations via geopy + custom coordinates:
    python scripts/export_wx1_products.py \\
        --input runs/20261121_18Z --domain d1 \\
        --locations "Tilburg" "De Bilt" \\
        --points "Noordzee:52.5,3.5" \\
        --fps 2.5
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PY_ROOT = PROJECT_ROOT / "python"
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))

try:
    from tqdm import tqdm as _tqdm
except ImportError:  # pragma: no cover
    _tqdm = None

from tempest_v2.export_products import VALID_MAP_TYPES, export_wx1_products  # noqa: E402


class _TqdmLoggingHandler(logging.Handler):
    """Route log records through tqdm.write so progress bars stay intact."""

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


def _prompt_locations_interactive() -> List[str]:
    if not sys.stdin.isatty():
        return ["De Bilt", "Tilburg"]
    prompt = (
        "Voer locaties in (plaatsnamen via geopy of Naam:lat,lon, gescheiden door komma's) "
        "[standaard: De Bilt, Tilburg]: "
    )
    raw = input(prompt).strip()
    if not raw:
        return ["De Bilt", "Tilburg"]
    items = [part.strip() for part in re.split(r"[;\n]+|(?<!\d),(?!\s*[-+]?\d)", raw) if part.strip()]
    return items or ["De Bilt", "Tilburg"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tempest-NWP v2 WX-1 Exporter — CSV + Cartopy Kaarten + GIF Animaties + MetPy Skew-T (geopy)"
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=PROJECT_ROOT / "runs",
        help="Pad naar .wx1-bestand, domeinmap (d1/d2/d3), run-map of runs/-root (auto-selecteert nieuwste run)",
    )
    parser.add_argument("--domain", choices=["auto", "d1", "d2", "d3"], default="auto")
    parser.add_argument(
        "-l",
        "--locations",
        nargs="+",
        default=None,
        help="Plaatsnamen op te zoeken via geopy (bijv. --locations 'Tilburg' 'De Bilt' 'Utrecht')",
    )
    parser.add_argument(
        "-p",
        "--points",
        nargs="+",
        default=None,
        help="Extra handmatige coördinaten als 'Naam:lat,lon' of 'lat,lon'",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=Path,
        default=None,
        help="Uitvoermap (standaard: <run_dir>/exports/<domain>)",
    )
    parser.add_argument(
        "--maps",
        type=str,
        default="t2m_mslp,precip_wind,cape_shear,z500_t850",
        help=f"Komma-gescheiden kaarttypes ({','.join(VALID_MAP_TYPES)}) of 'none'",
    )
    parser.add_argument(
        "--zoom-deg",
        type=float,
        default=None,
        help="Kaart-zoom rond gekozen punten in graden (leeg=auto, 0=volledig domein)",
    )
    parser.add_argument(
        "--skewt-hours",
        type=str,
        default="all",
        help="Tijdstappen voor Skew-T diagrams ('all' of bijv. '0,6,12,24')",
    )
    parser.add_argument("--fps", type=float, default=2.5, help="Frames per seconde voor GIF/MP4 animaties")
    parser.add_argument("--dpi", type=int, default=150, help="Afbeeldingsresolutie in DPI")
    parser.add_argument("--mp4", action="store_true", help="Genereer naast GIF ook MP4-animaties (vereist ffmpeg)")
    parser.add_argument("--no-csv", action="store_true", help="Geen CSV-tijdreeksen en -profielen opslaan")
    parser.add_argument("--no-maps", action="store_true", help="Geen weerkaarten tekenen")
    parser.add_argument("--no-animations", action="store_true", help="Geen GIF/MP4 animaties maken")
    parser.add_argument("--no-skewt", action="store_true", help="Geen MetPy Skew-T + hodograaf diagrammen maken")
    parser.add_argument("--no-meteogram", action="store_true", help="Geen meteogram-PNG per locatie maken")
    parser.add_argument("--no-geopy", action="store_true", help="Geen online Nominatim geopy gebruiken (alleen cache/fallback/coords)")
    parser.add_argument("--no-progress", action="store_true", help="Schakel tqdm-voortgangsbalken uit")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[_TqdmLoggingHandler()],
        force=True,
    )
    logger = logging.getLogger("tempest_v2.export_cli")

    queries: List[str] = []
    if args.locations:
        queries.extend(args.locations)
    if args.points:
        queries.extend(args.points)
    if not queries:
        queries = _prompt_locations_interactive()
        logger.info("Gekozen locaties: %s", ", ".join(queries))

    raw_maps = args.maps.strip().lower()
    if raw_maps == "none" or args.no_maps:
        map_types: List[str] = []
    else:
        map_types = [m.strip().lower() for m in raw_maps.split(",") if m.strip()]

    skewt_hrs: Optional[List[float]] = None
    if args.skewt_hours.strip().lower() != "all":
        skewt_hrs = [float(x.strip()) for x in args.skewt_hours.split(",") if x.strip()]

    summary = export_wx1_products(
        args.input,
        queries,
        domain=args.domain,
        outdir=args.outdir,
        use_geopy=not args.no_geopy,
        map_types=map_types,
        zoom_deg=args.zoom_deg,
        skewt_hours=skewt_hrs,
        fps=args.fps,
        dpi=args.dpi,
        also_mp4=args.mp4,
        enable_csv=not args.no_csv,
        enable_meteograms=not args.no_meteogram,
        enable_maps=not args.no_maps and len(map_types) > 0,
        enable_skewt=not args.no_skewt,
        enable_animations=not args.no_animations,
        show_progress=not args.no_progress,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
