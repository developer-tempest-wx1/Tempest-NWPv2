# Tempest-NWP v2

**A 100% Classical, Deterministic, GPU-Accelerated Non-Hydrostatic Numerical Weather Prediction Model (0% AI / 0% Machine Learning)**

Tempest-NWP v2 solves the compressible non-hydrostatic Euler equations on an Arakawa C-grid with a smooth Klemp (2011) terrain-following height coordinate in **Rust + NVIDIA CUDA**, coupled to classical physical parameterizations, 3D-Var + IAU data assimilation, and the lossless **WX-1** (`.wx1`, `zstd_shuffle`) scientific container format.

> **Project Rule:** 0% neural networks, 0% ML surrogates, 0% data-driven dynamics. Every equation in the dynamical core and physics suite uses classical atmospheric physics, thermodynamics, and strict SI units (`K`, `Pa`, `m/s`, `kg/kg`, `m²/s²`, `kg/(m²·s)`).

---

## Key Architecture Highlights

- **Non-Hydrostatic GPU Dynamical Core (`crates/tempest-cuda`):** Split-explicit RK3 time integration with acoustic substepping, WENO-5 / monotonic positive-definite tracer advection ($q_v, q_c, q_r, q_i, q_s, q_g$), spherical Lambert Conformal Conic (LCC) & global latitude-longitude geometry, and coupled NEMO 5.0.2 + SI3 ocean/sea-ice GPU kernels.
- **Deterministic Mixed Precision (`FP64` / `FP32`):** Commit `1e8dfcb` introduced deterministic FP64/FP32 mixed precision while keeping prognostic state, hydrostatic reference fields, Exner pressure, and WX-1 restarts bit-exact IEEE 754 `float64` in the dynamical core:
  - *RTX 3060 benchmark (`65 × 216 × 216`, 800 m mountain, `physics_every=3`):*
    - `step_full` (60 steps): `225.46 ms/step` (4.435 step/s) → `204.73 ms/step` (**4.885 step/s, +10.1% throughput**)
    - `step_rk3` (60 steps): `210.89 ms/step` (4.742 step/s) → `191.28 ms/step` (**5.228 step/s, +10.3% throughput**, `max|w_dyn| = 0.0`, `max|pip_dyn| = 0.0`)
    - `step_phys` (20 calls): `47.05 ms/call` → `44.01 ms/call` (**+6.9%**)
    - `ocean_si3` (1800 s): `47.64 ms/step` → `48.47 ms/step` (unchanged FP64, within run-to-run noise)
- **High-Resolution Static Coupling:** Sub-grid topography and coastal masks from **Copernicus DEM GLO-30/90**, bathymetry from **GEBCO 2026** / **EMODnet**, 10 m land cover from **ESA WorldCover 2021** (`z0`, albedo, emissivity, urban/forest morphology, inland water), and 250 m soil properties from **ISRIC SoilGrids v2.0** + **FAO HWSD v2.2**.
- **Steps 1–6 Physics & Bias Calibration:** Dual verification against official KNMI AWS hourly observations and ARCO-ERA5 upper-air/surface fields drove systematic fixes across MSLP/hydrostatic balance, TripleClouds sub-grid radiation + climatological AOD, SBL/skin temperature/radiation fog/gust diagnostics, and microphysics/thunderstorm initiation. See [docs/BENCHMARK_IMPROVEMENTS_STAP1_TO_6.md](docs/BENCHMARK_IMPROVEMENTS_STAP1_TO_6.md).
- **WX-1 Storage Backbone:** All static containers, initial conditions, LBCs, lossless checkpoints, and synoptic/column outputs use [WX-1](https://github.com/developer-tempest-wx1/tempest-wx1) with lossless `zstd_shuffle` compression.

---

## Supported Domains

| Domain Key | Projection / Grid | Grid Shape (`ny × nx`) | Horizontal Spacing | Default Dynamics `dt` |
| :--- | :--- | :--- | :--- | :--- |
| `d1_global_0p25` | Global Lat-Lon (periodic) | `721 × 1440` | `0.25°` (~27.8 km) | `60.0 s` |
| `d1_global_0p4` | Global Lat-Lon (periodic) | `451 × 900` | `0.40°` (~44.5 km) | `90.0 s` |
| `d1_global_0p5` | Global Lat-Lon (periodic) | `361 × 720` | `0.50°` (~55.6 km) | `120.0 s` |
| `d2_eu_lcc` | Europe LCC | `400 × 520` | `7.5 km` | `20.0 s` |
| `lam_eu_5km` | Europe LCC | `480 × 560` | `5.0 km` | `15.0 s` |
| `lam_bnl_ns_de_2p5km` | Benelux / North Sea / DE LCC | `432 × 432` | `2.5 km` | `6.0 s` |
| `d3_benelux_de_2p5km` | Benelux / DE LCC | `432 × 432` | `2.5 km` | `6.0 s` |
| `d3_nl_2p5km` | Netherlands LCC | `216 × 216` | `2.5 km` | `6.0 s` |
| `d3_benelux_de_1p5km` | Benelux / DE LCC | `720 × 720` | `1.5 km` | `4.0 s` |

Vertical terrain-following coordinates support `nz ∈ {65, 90, 137}` levels (`z_top = 22 km` for `nz < 90`, `31 km` for `nz >= 90`).

---

## Prerequisites & Building

### 1. System Requirements

- **OS:** Linux (x86_64) or WSL2 with NVIDIA GPU driver & CUDA Toolkit
- **Rust:** `>= 1.85` (via `rustup`)
- **CUDA Toolkit (`nvcc`):** `>= 12.0` (`>= 12.8` required for native `sm_120` Blackwell compilation)
- **Python:** `>= 3.10` with `numpy`, `scipy`, `xarray`, `netCDF4`, `rasterio`, `eccodes` / `cfgrib` (see [pyproject.toml](pyproject.toml))
- **WX-1 storage library (`wx1`):** Install [`tempest-wx1`](https://github.com/developer-tempest-wx1/tempest-wx1) from PyPI (`pip install tempest-wx1`) or editable from a sibling clone (`pip install -e ../WX-1`)
- **Optional OS tools for FAO HWSD v2.2 & GRIB2:**
  ```bash
  sudo apt-get update && sudo apt-get install -y mdbtools libeccodes-dev
  ```

### 2. Clone & Build GPU Kernels

`crates/tempest-cuda/build.rs` compiles the CUDA kernels with `nvcc -O3 -std=c++17` and links `crates/tempest-cuda/libtempest_cuda.so`:

```bash
git clone https://github.com/developer-tempest-wx1/Tempest-NWPv2.git
cd Tempest-NWPv2

# Automatic GPU detection via nvidia-smi (default; falls back safely if older nvcc)
./scripts/build.sh --arch auto

# Build + run cargo test --release and pytest -q tests/ (+ optional editable pip install)
./scripts/build.sh --arch auto --test --editable-python
```

Supported `TEMPEST_CUDA_ARCH` / `--arch` targets:

| Spec | Compiled `sm_*` Targets | Intended Use |
| :--- | :--- | :--- |
| `auto` (default) | Detected `nvidia-smi` GPU(s) (headless fallback: `80,86,89,90,120`) | Local workstation / node |
| `all` / `portable` | Filtered subset of `75,80,86,89,90,100,120` supported by your `nvcc` | Redistributable fat binary |
| `sm_75` .. `sm_120` | Explicit GPU generation (e.g. `sm_86` RTX 30xx, `sm_89` RTX 40xx, `sm_90` Hopper, `sm_120` RTX 50xx Blackwell) | Explicit single/multi target (`sm_86,sm_120`) |

Manual Cargo build & full test suites:

```bash
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
TEMPEST_CUDA_ARCH=auto cargo build --release
cargo test --release
PYTHONPATH=python pytest -v tests/
```

---

## Static Geophysical Data Download & External Storage

[scripts/download_static_data.py](scripts/download_static_data.py) uses **Python standard library only** (no `geopandas`, `shapely`, or `requests` required) to download official open-access datasets into the directory layout expected by `tempest_v2`.

### Available Datasets & Flags

| Dataset | CLI Flag(s) | Options / Layers | Vertical Datum / Notes |
| :--- | :--- | :--- | :--- |
| **Copernicus DEM GLO-30 / GLO-90** | `--copernicus-dem`, `--glo90` | `--copernicus-layers`: `minimal` (`DEM,WBM`), `standard` (default: `DEM,WBM,EDM,FLM,HEM`), `all` (`DEM,WBM,EDM,FLM,HEM,ACM,XML,PREVIEW,INFO`) | Public AWS S3 (`copernicus-dem-30m`, `copernicus-dem-90m`); uses `tileList.txt` to skip ocean tiles; `INFO/eula_F.pdf` saved once per resolution |
| **Bathymetry (GEBCO / EMODnet)** | `--bathymetry` | `--bathy-source`: `gebco-erddap` (fast regional MSL), `gebco-2026-global` (`GEBCO_2026.nc` ~7 GiB), `gebco-2026-zip` (ZIP + extract), `gebco-2026-full` (`GEBCO_2026` + `GEBCO_2026_sub_ice` + `gebco_2026_tid`), `gebco-2025-zip`, `emodnet-wcs` | GEBCO = MSL; EMODnet = LAT (saved with `.json` metadata). `_pick_gebco_elevation_nc` automatically skips `TID` grids |
| **ESA WorldCover 10 m** | `--worldcover` | `--worldcover-year`: `2021` (`v200`, default) or `2020` (`v100`) | Public AWS S3 (`esa_worldcover_grid.geojson` intersection with 3° fallback); drives `z0`, albedo, emissivity, urban/forest & water masks |
| **ISRIC SoilGrids 250 m v2.0** | `--soilgrids` | `--soilgrids-mode`: `tempest` (14 layers used by Tempest), `surface` (10 layers `0-5cm` + WRB), `full` (55 layers: 9 properties × 6 depths + WRB), `--no-wrb` | ISRIC WCS 2.0.1 (`GEOTIFF_INT16`, EPSG:4326); includes XML/HTML exception guard |
| **FAO / IIASA HWSD v2.2** | `--hwsd` | Downloads `hwsd2.2_db.zip` + `hwsd2.2_raster.zip`, verifies ZIP CRCs & SHA-256 | Requires OS `mdbtools` (`mdb-tables`, `mdb-export`) during `--prepare-wx1` |
| **FABDEM v1.2** *(opt-in)* | `--fabdem` | 1° bare-earth DEM tiles from HuggingFace `links-ads/fabdem-v12` | Optional forest/building bias-removed DEM (not included in `--all`) |
| **PDOK AHN** *(opt-in, NL only)* | `--ahn` | `--ahn-product`: `dtm_05m` / `dsm_05m`; `--ahn-resolution`: spacing in m (default `5.0`) | Optional Netherlands high-res WCS (clipped to NL bounds; not in `--all`) |

> **Bulk / Alternative Mirrors:**
> - **Copernicus DEM CDSE:** Besides anonymous AWS S3, EU/ESA provides mirrors via Copernicus Data Space Ecosystem (`eodata`).
> - **SoilGrids Global VRTs:** For global or continental runs beyond WCS regional subsets, ISRIC publishes full VRT/COG hierarchies via WebDAV at `https://files.isric.org/soilgrids/latest/data/`.

### Download Commands

Preview URLs, tile counts, and target paths offline (`--dry-run` makes zero downloads):

```bash
python3 scripts/download_static_data.py --all --domain lam_bnl_ns_de_2p5km --dry-run
```

Download the standard regional static bundle (Copernicus GLO-30 `standard`, regional GEBCO, HWSD v2.2, ESA WorldCover 2021, SoilGrids 14-layer `tempest`) and compile lossless `.wx1` grids:

```bash
python3 scripts/download_static_data.py \
  --all \
  --domain lam_bnl_ns_de_2p5km \
  --prepare-wx1
```

Download the **complete** full archives (Copernicus GLO-30 + GLO-90 with all `DEM,WBM,EDM,FLM,HEM,ACM,XML,PREVIEW,INFO` layers, full GEBCO 2026 `ice_surface_elevation + sub_ice + tid`, full 55-layer SoilGrids 250 m + WRB, ESA WorldCover 2021, and HWSD v2.2):

```bash
python3 scripts/download_static_data.py \
  --copernicus-dem --glo90 --copernicus-layers all \
  --bathymetry --bathy-source gebco-2026-full \
  --worldcover --worldcover-year 2021 \
  --soilgrids --soilgrids-mode full \
  --hwsd \
  --domain lam_bnl_ns_de_2p5km \
  --prepare-wx1
```

### External Storage, Symlinks & Environment Variables

Large static archives (Copernicus DEM, full GEBCO 2026, ESA WorldCover, SoilGrids, HWSD v2.2) can live on an external SSD/HDD or shared data directory. Link them cleanly into `data/static/` (which is gitignored for raw/compiled binaries):

```bash
mkdir -p data/static
ln -s /path/to/external/COPERNICUS_DEM      data/static/copernicus_dem_raw
ln -s /path/to/external/bathymetry          data/static/bathymetry_raw
ln -s /path/to/external/esa_worldcover_2021 data/static/esa_worldcover_2021
ln -s /path/to/external/soilgrids_250m      data/static/soilgrids_250m
ln -s /path/to/external/hwsd_v22            data/static/hwsd_v22
```

Alternatively, override any directory explicitly via CLI flags (`--dem-dir`, `--bathy-dir`, `--worldcover-dir`, `--soilgrids-dir`, `--hwsd-dir`) or environment variables:

| Environment Variable | Default Path (`data/static/...`) | Purpose |
| :--- | :--- | :--- |
| `TEMPEST_COPERNICUS_DEM_DIR` | `data/static/copernicus_dem_raw` | Copernicus DEM GLO-30/90 root (`dem30_europe/`, `dem90_world/`) |
| `TEMPEST_BATHY_DIR` | `data/static/bathymetry_raw` | GEBCO 2026/2025/ERDDAP & EMODnet bathymetry files |
| `TEMPEST_WORLDCOVER_DIR` | `data/static/esa_worldcover_2021` | ESA WorldCover 10 m GeoTIFF tiles |
| `TEMPEST_SOILGRIDS_DIR` | `data/static/soilgrids_250m` | ISRIC SoilGrids 250 m `Q0p5` & `WRB_MostProbable` GeoTIFFs |
| `TEMPEST_HWSD_V22_DIR` | `data/static/hwsd_v22` | FAO HWSD v2.2 `.mdb` & `.bil` archives |
| `TEMPEST_ERA5_CASES_DIR` | `data/knmi_cases` (or `../era5_cases`) | KNMI hourly CSV archives (`knmi_uur_*.csv.gz`) for `select_era5_cases.py` |
| `TEMPEST_CUDA_ARCH` | `auto` | Target GPU compute capability (`auto`, `all`, or explicit `sm_75`..`sm_120`) |

---

## Running Forecasts & Benchmarks

```bash
# Operational 24h LAM forecast driven by ECMWF Open Data LBCs (100% Tempest physics;
# --ecmwf selects the 3h->T+144 / 6h->T+360 output cadence, not ECMWF physics)
PYTHONPATH=python python3 scripts/run_tempest_v2.py \
  --mode lam_ecmwf_lbc \
  --lam-domain lam_bnl_ns_de_2p5km \
  --hours 24 \
  --ecmwf

# Global D1 -> Europe D2 -> Benelux/DE D3 nested run with coupled NEMO+SI3
PYTHONPATH=python python3 scripts/run_tempest_v2.py \
  --mode d1_to_d2_to_d3 \
  --enable-ocean-si3 \
  --hours 24

# Multi-season ARCO-ERA5 hindcast & KNMI AWS verification benchmark (e.g. quick4 preset)
PYTHONPATH=python python3 scripts/run_benchmark_suite.py --preset quick4 --hours 24

# Find notable historical Dutch weather cases in local KNMI archives
python3 scripts/select_era5_cases.py --regime storm --top 10
```

Verify an existing forecast run directory against KNMI AWS stations & ERA5:

```bash
PYTHONPATH=python python3 scripts/verify_knmi_wx1.py --help
```

---

## Repository Layout

```text
Tempest-NWPv2/
├── Cargo.toml                         # Rust workspace (Apache-2.0)
├── pyproject.toml                     # Python package metadata & dependencies
├── LICENSE                            # Apache License 2.0
├── crates/
│   ├── tempest-core/                  # Core Rust dynamical/state orchestration & WX-1 bridge
│   └── tempest-cuda/                  # CUDA non-hydrostatic dycore, physics & NEMO/SI3 kernels
├── python/tempest_v2/                 # Projections, IC/LBC builders, static pipelines, DA, verification
├── scripts/
│   ├── build.sh                       # GPU & Rust release build helper
│   ├── download_static_data.py        # Open-access static geophysical data downloader
│   ├── run_tempest_v2.py              # Operational forecast CLI
│   ├── run_benchmark_suite.py         # Multi-regime ERA5 + KNMI validation suite
│   ├── select_era5_cases.py           # KNMI station archive case selector
│   └── verify_knmi_wx1.py             # Station + ERA5 verification CLI
├── docs/
│   ├── BENCHMARK_IMPROVEMENTS_STAP1_TO_6.md
│   └── PHYSICS_SOURCES.md
└── tests/                             # Rust + pytest verification suites
```

---

## License & Data Attribution

- **Source Code:** Licensed under the [Apache License, Version 2.0](LICENSE).
- **Static Datasets:** Subject to their respective upstream open licenses when downloaded by users (Copernicus DEM EULA, GEBCO Terms of Use, ESA WorldCover CC BY 4.0, ISRIC SoilGrids CC BY 4.0, FAO HWSD v2.2 Terms, FABDEM CC BY 4.0 / OGL, PDOK AHN CC0).
