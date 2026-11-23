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
        description="Verify Tempest-NWP .wx1 forecast outputs hourly against official KNMI station observations."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--run-dir", type=Path, help="Tempest run directory (e.g. runs/bnl_2p5km/20261122_18Z)")
    src.add_argument("--wx1-dir", type=Path, help="Direct directory containing forecast_*_T+*h.wx1 files")
    parser.add_argument("--outdir", type=Path, default=None, help="Output directory for CSV/JSON/PNG verification artifacts")
    parser.add_argument("--knmi-cache-dir", type=Path, default=None, help="Cache dir for downloaded KNMI NetCDFs")
    parser.add_argument("--api-key", type=str, default=None, help="Optional KNMI Open Data API key override")
    parser.add_argument(
        "--obs-source",
        choices=["auto", "10min", "hourly-validated"],
        default="auto",
        help="KNMI obs dataset: auto (hourly-validated when >=24h old/missing 10min), 10min, or hourly-validated",
    )
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
    parser.add_argument(
        "--compare-ifs",
        action="store_true",
        help="Auto-discover ECMWF IFS GRIB2 cache from manifest.json and score side-by-side vs Tempest",
    )
    parser.add_argument(
        "--compare-ifs-dir",
        type=Path,
        default=None,
        help="Explicit directory containing ECMWF IFS *_sfc_*.grib2 files for side-by-side comparison",
    )
    parser.add_argument(
        "--compare-era5",
        action="store_true",
        help="Score Tempest side-by-side vs ARCO-ERA5 0.25-deg surface & 850/500 hPa fields",
    )
    parser.add_argument(
        "--era5-cache-dir",
        type=Path,
        default=None,
        help="Optional ERA5 cache directory for --compare-era5 (auto-discovered from manifest if omitted)",
    )
    parser.add_argument(
        "--era5-step-hours",
        type=float,
        default=3.0,
        help="Lead-time interval [h] for --compare-era5 sampling (default: 3.0; use 1.0 for every hour)",
    )
    parser.add_argument(
        "--verify-radar-fss",
        action="store_true",
        help="Fetch 12x5-min live gauge-adjusted KNMI radar QPE per hour and score Roberts & Lean (2008) FSS",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    target = args.run_dir if args.run_dir is not None else args.wx1_dir
    assert target is not None
    summary = verify_run_against_knmi(
        target,
        outdir=args.outdir,
        knmi_cache_dir=args.knmi_cache_dir,
        api_key=args.api_key,
        obs_source=args.obs_source,
        max_lead_hours=args.max_lead_hours,
        independent_after_hour=args.independent_after_hour,
        hourly_precip_6x=args.hourly_precip_6x,
        land_only_plots=args.land_only,
        make_maps=not args.no_maps,
        ifs_grib_dir=args.compare_ifs_dir,
        compare_ifs=args.compare_ifs,
        compare_era5=args.compare_era5,
        era5_cache_dir=args.era5_cache_dir,
        era5_step_hours=args.era5_step_hours,
        verify_radar_fss=args.verify_radar_fss,
    )

    rmeta = summary.get("run_metadata") or {}
    commit_str = f" | Commit: {rmeta['git_commit']}" if rmeta.get("git_commit") else ""
    sources_used = summary.get("obs_sources_used") or []
    if sources_used == ["hourly-validated"]:
        obs_hdr = "KNMI hourly-validated"
    elif sources_used == ["10min"] or not sources_used:
        obs_hdr = "KNMI 10-min"
    else:
        obs_hdr = "KNMI " + "+".join(sources_used)
    print(
        f"\n=== {obs_hdr} Verificatie | Domein: {summary['domain_name']}{commit_str} "
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
            f"T+{rec['lead_hours']:02.0f}h {rec['valid_time_utc'][:16]}Z  {flag}    {b.get('n_unique_stations', b['n_stations']):3d}  | "
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
        n_uniq = blk.get("n_unique_stations", blk["n_stations"])
        n_st_hrs = blk.get("n_station_hours", blk["t2m"]["n"])
        w_min = float(blk["wdir"].get("min_ws_ms", 3.0))
        print(f"\n[{label}] stations={n_uniq}, station_hours={n_st_hrs}, n_obs={blk['t2m']['n']}")
        print(
            f"  T2m: bias={_fmt(blk['t2m']['bias'])} K, RMSE={_fmt(blk['t2m']['rmse'], '.2f')} K, r={_fmt(blk['t2m']['corr'], '.2f')} | "
            f"Td2m bias={_fmt(blk['td2m']['bias'])} K | RH bias={_fmt(blk['rh']['bias'], '+.1f')}%"
        )
        print(
            f"  MSLP: bias={_fmt(blk['mslp']['bias'])} hPa, RMSE={_fmt(blk['mslp']['rmse'], '.2f')} hPa | "
            f"Wind10m: bias={_fmt(blk['ws10']['bias'])} m/s, RMSE={_fmt(blk['ws10']['rmse'], '.2f')} m/s, "
            f"gust bias={_fmt(blk['gust10']['bias'])} m/s, dir MAE (>={w_min:.0f} m/s)={_fmt(blk['wdir']['mae_deg'], '.1f')}°"
        )
        sw_blk = blk.get("sw_down") or {}
        if int(sw_blk.get("n", 0)) > 0:
            print(
                f"  Straling SW↓ (vs KNMI qg, n={sw_blk['n']}): "
                f"bias={_fmt(sw_blk['bias'], '+.1f')} W/m², RMSE={_fmt(sw_blk['rmse'], '.1f')} W/m², "
                f"obs_mean={_fmt(sw_blk['obs_mean'], '.1f')} vs mod_mean={_fmt(sw_blk['mod_mean'], '.1f')} W/m²"
            )
        tcc_blk = blk.get("tcc") or {}
        if int(tcc_blk.get("n", 0)) > 0:
            print(
                f"  Bewolking TCC (vs KNMI N/8, n={tcc_blk['n']}): "
                f"bias={_fmt(tcc_blk['bias'], '+.2f')}, RMSE={_fmt(tcc_blk['rmse'], '.2f')}, "
                f"obs_mean={_fmt(tcc_blk['obs_mean'], '.2f')} vs mod_mean={_fmt(tcc_blk['mod_mean'], '.2f')}"
            )
        fg = blk["fog"]
        print(
            f"  Mist (<1000m): H={fg['hits']}, M={fg['misses']}, FA={fg['false_alarms']}, "
            f"POD={_fmt(100.0 * fg['pod'] if math.isfinite(fg['pod']) else float('nan'), '.0f')}%, "
            f"FAR={_fmt(100.0 * fg['far'] if math.isfinite(fg['far']) else float('nan'), '.0f')}%, "
            f"CSI={_fmt(100.0 * fg['csi'] if math.isfinite(fg['csi']) else float('nan'), '.0f')}%"
        )

    indep_ev = (summary.get("independent_land") or {}).get("events") or {}
    indep_cd = (summary.get("independent_land") or {}).get("convection_diag") or {}
    ev_labels = (
        ("rain_0p1mm", "Regen >=0.1mm/u"),
        ("rain_2mm", "Zware bui >=2mm/u"),
        ("gust_15ms", "Stoot >=15 m/s"),
        ("gust_20ms", "Zware stoot >=20 m/s"),
        ("snow", "Sneeuw (W3 / acc)"),
        ("hail_graupel", "Hagel/Korrelhagel (WW/graupel)"),
        ("thunder", "Onweer (W5 / micro+CAPE)"),
        ("ice_freezing_rain", "IJzel (W6 / T<=0.5C+rain)"),
    )
    active_ev = [
        (lbl, indep_ev[k])
        for k, lbl in ev_labels
        if k in indep_ev and (indep_ev[k].get("obs_yes", 0) > 0 or indep_ev[k].get("mod_yes", 0) > 0)
    ]
    if active_ev or any(math.isfinite(float(indep_cd.get(ck, float("nan")))) for ck in ("mod_cape_mu_max", "mod_graupel_max_mm")):
        print(f"\n[Weertypes & Convectie T>={summary['independent_after_hour']:.0f}u (landstations)]")
        for lbl, ev in active_ev:
            pod_s = _fmt(100.0 * ev["pod"] if math.isfinite(ev["pod"]) else float("nan"), ".0f")
            far_s = _fmt(100.0 * ev["far"] if math.isfinite(ev["far"]) else float("nan"), ".0f")
            csi_s = _fmt(100.0 * ev["csi"] if math.isfinite(ev["csi"]) else float("nan"), ".0f")
            print(
                f"  {lbl:<32s}: obs={ev['obs_yes']:3d} mod={ev['mod_yes']:3d} | "
                f"H={ev['hits']:3d} M={ev['misses']:3d} FA={ev['false_alarms']:3d} | "
                f"POD={pod_s}% FAR={far_s}% CSI={csi_s}%"
            )
        print(
            f"  Convectie-diag: max MU-CAPE={_fmt(indep_cd.get('mod_cape_mu_max', float('nan')), '.0f')} J/kg | "
            f"max graupel={_fmt(indep_cd.get('mod_graupel_max_mm', float('nan')), '.2f')} mm/u | "
            f"max sneeuw={_fmt(indep_cd.get('mod_snow_max_mm', float('nan')), '.2f')} mm/u"
        )

    if summary["worst_t2m_land_stations"]:
        print("\n[Top 5 landstations met grootste T2m RMSE (onafhankelijk)]")
        for st in summary["worst_t2m_land_stations"][:5]:
            print(
                f"  {st['station_id']} {st['station_name']:<22s} n={st['n_hours']:2d} | "
                f"T2m bias={st['t2m_bias_k']:+.2f}K RMSE={st['t2m_rmse_k']:.2f}K | "
                f"RH bias={st['rh_bias_pct']:+.1f}% | Wind bias={st['ws10_bias_ms']:+.2f}m/s"
            )

    def _print_ref_section(cmp_blk: dict, *, ref_key: str, title: str, src_key: str, var_rows: tuple) -> None:
        print(f"\n=== {title} ({cmp_blk['matched_hours']} synchrone uren: {cmp_blk['matched_leads']}) ===")
        print(f"Bron: {cmp_blk.get(src_key, '')}")
        for label, bkey in (
            ("Synchroon Totaal (landstations)", "overall_land"),
            (f"Synchroon Onafhankelijk T>={summary['independent_after_hour']:.0f}u (landstations)", "independent_land"),
        ):
            blk = cmp_blk[bkey]
            ts, rs, sk = blk["tempest"], blk[ref_key], blk["skill"]
            print(f"\n[{label}] (SS > 0 betekent Tempest lagere RMSE dan {ref_key.upper()})")
            for var_key, var_lbl, unit in var_rows:
                if rs[var_key]["n"] == 0:
                    continue
                print(
                    f"  {var_lbl:<11s} (n={rs[var_key]['n']:3d}): "
                    f"Tempest bias={_fmt(ts[var_key]['bias'])} RMSE={_fmt(ts[var_key]['rmse'], '.2f')} {unit} | "
                    f"{ref_key.upper()} bias={_fmt(rs[var_key]['bias'])} RMSE={_fmt(rs[var_key]['rmse'], '.2f')} {unit} | "
                    f"SS={_fmt(sk[var_key]['skill_score'], '+.3f')}"
                )
            if rs["wdir"]["n"] > 0:
                print(
                    f"  Dir10m      (n={rs['wdir']['n']:3d}): "
                    f"Tempest MAE={_fmt(ts['wdir']['mae_deg'], '.1f')}° | "
                    f"{ref_key.upper()} MAE={_fmt(rs['wdir']['mae_deg'], '.1f')}° | "
                    f"SS={_fmt(sk['wdir']['skill_score'], '+.3f')}"
                )

    if summary.get("ifs_comparison"):
        icmp = summary["ifs_comparison"]
        _print_ref_section(
            icmp,
            ref_key="ifs",
            title="Tempest vs ECMWF IFS 0.1°",
            src_key="ifs_dir",
            var_rows=(
                ("t2m", "T2m", "K"),
                ("td2m", "Td2m", "K"),
                ("rh", "RH", "%"),
                ("mslp", "MSLP", "hPa"),
                ("ws10", "Wind10m", "m/s"),
            ),
        )

    if summary.get("era5_comparison"):
        ecmp = summary["era5_comparison"]
        _print_ref_section(
            ecmp,
            ref_key="era5",
            title="Tempest vs ARCO-ERA5 0.25°",
            src_key="era5_cache_dir",
            var_rows=(
                ("t2m", "T2m", "K"),
                ("td2m", "Td2m", "K"),
                ("rh", "RH", "%"),
                ("mslp", "MSLP", "hPa"),
                ("ws10", "Wind10m", "m/s"),
                ("gust10", "Gust10m", "m/s"),
                ("sw_down", "SW_down", "W/m2"),
                ("tcc", "TCC", "0..1"),
                ("precip", "Neerslag1h", "mm"),
            ),
        )
        eua = (summary.get("era5_upper_air") or {}).get("independent") or (summary.get("era5_upper_air") or {}).get("overall")
        if eua:
            print(f"\n[Tempest Binnengebied vs ARCO-ERA5 3D Bovenlucht ({eua['n_hours']} uren)]")
            for k_ua, lbl_ua, u_ua in (
                ("mslp_hpa", "MSLP 2D", "hPa"),
                ("t2m_c", "T2m 2D", "K"),
                ("t850_k", "T 850hPa", "K"),
                ("gh500_m", "GH 500hPa", "m"),
                ("wspd850_ms", "WSPD 850hPa", "m/s"),
                ("wspd500_ms", "WSPD 500hPa", "m/s"),
                ("q850_gkg", "Q 850hPa", "g/kg"),
            ):
                b = eua.get(k_ua) or {}
                if int(b.get("n_points", 0)) > 0:
                    print(f"  {lbl_ua:<12s}: bias={_fmt(b['bias'])} {u_ua}, RMSE={_fmt(b['rmse'], '.2f')} {u_ua}")

    if summary.get("radar_fss") and summary["radar_fss"].get("overall"):
        rf = summary["radar_fss"]["overall"]
        print(
            f"\n=== KNMI 1 km Radar QPE — Fractions Skill Score (Roberts & Lean 2008 | {rf['n_hours']} uren, dx={rf['dx_km']:.1f} km) ==="
        )
        print("Drempel | f_obs   f_mod   Bias  | FSS_unif |   2.5km    10km    25km    50km")
        print("-" * 80)
        for th_k, th_v in rf["thresholds"].items():
            sc = th_v["scales"]
            print(
                f"{th_k:<7s} | {th_v['f_obs']:6.4f}  {th_v['f_mod']:6.4f}  {_fmt(th_v['freq_bias'], '.2f')} | "
                f"  {th_v['fss_uniform']:5.3f}  | "
                f"{_fmt(sc.get('2.5km', {}).get('fss', float('nan')), '.3f'):>7s} "
                f"{_fmt(sc.get('10km', {}).get('fss', float('nan')), '.3f'):>7s} "
                f"{_fmt(sc.get('25km', {}).get('fss', float('nan')), '.3f'):>7s} "
                f"{_fmt(sc.get('50km', {}).get('fss', float('nan')), '.3f'):>7s}"
            )

    art = summary["artifacts"]
    print(f"\n[Output] Overzicht PNG : {art['timeseries_png']}")
    print(f"[Output] Uur-CSV       : {art['hourly_csv']}")
    print(f"[Output] Station-CSV   : {art['station_csv']}")
    print(f"[Output] Kaarten       : {len(art['map_pngs'])} PNGs in {Path(summary['outdir']) / 'maps'}")


if __name__ == "__main__":
    main()
