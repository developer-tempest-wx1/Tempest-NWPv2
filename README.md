# Tempest-NWP v2 — Classical Non-Hydrostatic GPU NWP Model

> **Absolute Project Directive: 0% AI / 0% Machine Learning**  
> Tempest-NWP v2 is a 100% deterministic, classical physics-based Numerical Weather Prediction (NWP) system solving the compressible non-hydrostatic Euler / Navier-Stokes equations on GPU in strict SI units (`K`, `Pa`, `m/s`, `kg/kg`, `m^2/s^2`, `W/m^2`). No neural networks, deep learning surrogates, or data-driven parameterizations are used anywhere in the dynamical core or model physics.

---

## Key Architecture & Features

- **Self-Contained CUDA/Rust Solver (No External NEMO Clone Required):**  
  Both the atmospheric non-hydrostatic core (`nh_core.cu`), full atmospheric/land physics (`nh_physics.cu`), and the coupled **NEMO 5.0.2 + SI3** ocean & 5-category sea-ice solver (`nh_ocean_si3.cu`) are implemented directly in native CUDA C++ and compiled into a single shared library (`crates/tempest-cuda/libtempest_cuda.so`). You do **not** need to clone or build external NEMO/SI3 repositories on a new machine (`external_refs/` is strictly optional offline reference storage and gitignored).
- **Dynamical Core (`crates/tempest-cuda/src/cuda/nh_core.cu`):**
  - Arakawa C-grid staggered velocity/scalar formulation on spherical Lambert Conformal Conic (LCC) domains (`D2`, `D3`, standalone LAMs) and global latitude-longitude grids (`D1`).
  - Klemp (2011) smooth terrain-following vertical coordinates (`nz = 65`, `90`, `137`, IEEE 754 `float64` analytic hydrostatic reference state).
  - Split-explicit 3rd-order Wicker-Skamarock Runge-Kutta (`RK3`) time integration with WENO-5 / UP3 positive-definite tracer advection, 3D divergence damping, Rayleigh upper sponge, Davies lateral boundary relaxation (`bc_x=2, bc_y=2`), and Incremental Analysis Update (`IAU`).
  - Automatic 2D halo-tiling memory manager (`cuda_bridge.py`) so regional 2.5 km domains fit comfortably even on 6 GiB VRAM laptop GPUs (e.g. RTX 3060) and scale cleanly up to 24–32+ GiB RTX 4090 / 5080 / 5090 cards.
- **Physical Parameterizations (`crates/tempest-cuda/src/cuda/nh_physics.cu`):**
  - **Microphysics:** 6-species (`qv, qc, qr, qi, qs, qg`) mixed-phase scheme (WRF WSM6 / ICON / ICE-T principles) with Supercooled Liquid Water (SLW) preservation down to `-37 °C`, proportional sink scaling for positivity/conservation, subcycled precipitation fallout, and optional **HARMONIE-AROME Cy46** physics (`--cy46-physics`, `--pbl-last`).
  - **PBL & Convection:** Nonlocal YSU K-profile boundary layer + shallow EDMF thermal updraft mass-flux coupling.
  - **Radiation:** Two-stream delta-Eddington SW/LW radiative transfer (`ecRad` principles) with liquid/ice cloud optics.
  - **Surface, Land, Lakes & Ocean/Sea-Ice:** 4-layer soil thermo-hydrology (ECMWF HTESSEL/Noah depths `0.07, 0.21, 0.72, 1.89 m`, WMO/USDA 14-class FAO HWSD v2.2 lookup), FLake freshwater inland lakes, COARE 3.6/VOSKIN air-sea fluxes, and self-contained coupled 3D ocean + 5-category SI3 mEVP sea-ice (`nh_ocean_si3.cu` with TEOS-10 75-term EOS).
- **WX-1 Scientific Storage Backbone (`tempest-wx1` on PyPI):**  
  All gridded forecasts, restarts, initial/boundary conditions, static datasets, and observation containers use the open-source **`tempest-wx1`** container format (`pip install tempest-wx1>=0.1.8`), defaulting to lossless `zstd_shuffle` compression (`[1, nlat, nlon]` horizontal slice chunking & `[time, nlevels, 1, 1]` sounding chunking).
- **Open-Access Operational & Hindcast Initialization + Dual Verification:**
  - Operational cold-start & 3-hourly LBCs from **ECMWF Open Data** (`0p1`/`0p25` GRIB2, no account required).
  - Historical 24h+ hindcasts initialized & bounded from public **ARCO-ERA5** on Google Cloud Storage (zero-login Zarr streaming via `--stream-era5-lbc`, plus optional ERA5-Land soil).
  - Local Balanced 3D-Var + IAU data assimilation & TITAN-style spatial/temporal QC (`knmi`, `dwd`, radiosondes, WIS2, marine, OSI-SAF sea-ice).
  - Automated hourly verification (`scripts/verify_knmi_wx1.py` & `scripts/run_benchmark_suite.py`) against **KNMI official observations** (`10-min` / `hourly-validated` / local `knmi_uur_*.csv.gz`) side-by-side with **ARCO-ERA5** / **ECMWF IFS** baselines across 14+ metrics and categorical weather events (`fog`, `rain`, `gusts`, `snow`, `hail/graupel`, `thunder`, `freezing rain / ijzel`).

---

## `tempest-wx1` Storage Container (PyPI)

Tempest-NWP v2 uses the official [`tempest-wx1`](https://pypi.org/project/tempest-wx1/) package (`v0.1.8+`, Apache-2.0):

- **PyPI:** https://pypi.org/project/tempest-wx1/
- **GitHub:** https://github.com/developer-tempest-wx1/tempest-wx1
- **Install standalone or alongside Tempest-NWP v2:**
  ```bash
  pip install "tempest-wx1>=0.1.8"
  ```
- **Why WX-1?**
  - **Lossless prognostic storage (`zstd_shuffle`):** Byte-shuffled Zstandard preserves exact IEEE 754 `float32`/`float64` bits for restarts and checkpoints.
  - **Domain-specific chunk topologies:** Fast `[1, ny, nx]` synoptic horizontal slices for 2D/3D forecast maps and `[time, nz, 1, 1]` column chunks for atmospheric soundings.

---

## Supported GPUs (Including NVIDIA RTX 50xx Blackwell `sm_120`)

The Rust/CUDA build script (`crates/tempest-cuda/build.rs`) inspects both the installed `nvcc --list-gpu-arch` and `nvidia-smi` via `TEMPEST_CUDA_ARCH`:

| GPU Generation | Representative Cards | Compute Capability | `TEMPEST_CUDA_ARCH` | Minimum CUDA Toolkit |
| :--- | :--- | :---: | :---: | :--- |
| **RTX 50xx (Blackwell)** | **RTX 5090, 5080, 5070 Ti, 5070** | **`12.0`** | **`sm_120`** *(or `auto`)* | **CUDA 12.8+** (tested on **13.1**) |
| Datacenter Blackwell | B200 / GB200 | `10.0` | `sm_100` | CUDA 12.8+ |
| Hopper | H100 / H200 | `9.0` | `sm_90` | CUDA 12.0+ |
| RTX 40xx (Ada Lovelace) | RTX 4090, 4080, 4070, 4060 | `8.9` | `sm_89` | CUDA 11.8+ |
| RTX 30xx / Ampere | RTX 3060 / 3080 / 3090, A100 | `8.6` / `8.0` | `sm_86` / `sm_80` | CUDA 11.1+ |
| RTX 20xx (Turing) | RTX 2060 – 2080 Ti, T4 | `7.5` | `sm_75` | CUDA 10.0+ |

- `TEMPEST_CUDA_ARCH=auto` *(default)*: auto-detects local GPU(s) via `nvidia-smi`. If an RTX 50xx GPU (`sm_120`) is present alongside an older CUDA `< 12.8` toolkit, `build.rs` emits a warning advising CUDA `>= 12.8` and compiles the highest available virtual PTX (`compute_MAX`) for JIT forward compatibility.
- `TEMPEST_CUDA_ARCH=sm_120`: explicitly compiles native **RTX 50xx Blackwell** SASS + PTX (`-gencode arch=compute_120,code=sm_120 -gencode arch=compute_120,code=compute_120`).
- `TEMPEST_CUDA_ARCH=all`: builds a portable multi-architecture fatbin across `sm_75, sm_80, sm_86, sm_89, sm_90, sm_100, sm_120` (filtered to architectures supported by your `nvcc`) plus forward-compatible PTX.

### VRAM Sizing Guide
- `d3_nl_2p5km` (`216x216`, `nz=65`) & `lam_bnl_ns_de_2p5km` (`432x432`, `nz=65`): runs smoothly on **6 GiB VRAM** (e.g. RTX 3060 Laptop; large domains automatically halo-tile within VRAM budget).
- `d3_benelux_de_1p5km` (`720x720`, `nz=90/137`), `lam_eu_5km`, or full global `d1_global_0p25` + NEMO/SI3: recommended **12–32 GiB VRAM** (RTX 4070 Ti Super / 4090 / 5080 / 5090), or automatic multi-tile GPU paging on 6–8 GiB GPUs.

---

## Clean-Machine Installation & Build (Linux / WSL2)

### 1. System Prerequisites
```bash
sudo apt update
sudo apt install -y build-essential git curl pkg-config mdbtools libeccodes-dev
```
- **Rust (`>= 1.85`):** https://rustup.rs (`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`)
- **NVIDIA Driver + CUDA Toolkit:** **CUDA 12.8+** recommended (**CUDA 12.8+ required** for native RTX 50xx `sm_120` compilation). Ensure `nvcc` is in `PATH` or export `CUDA_HOME=/usr/local/cuda`.
- **Python (`>= 3.10`):** Conda, Miniconda, Mamba, or `venv`.

### 2. Clone, Install Python Package & Build CUDA/Rust Core
```bash
git clone https://github.com/developer-tempest-wx1/Tempest-NWPv2.git
cd Tempest-NWPv2

# Create/activate Python environment, then install Tempest + tempest-wx1 (PyPI) + optional deps
pip install -e ".[all]"

# Build Rust + CUDA solver + self-contained NEMO 5.0.2 / SI3 shared library
./scripts/build.sh --arch auto --test
```

To explicitly target an **RTX 5090 / 5080 / 5070 Ti / 5070** (`sm_120`):
```bash
./scripts/build.sh --arch sm_120 --test
```

To compile a portable multi-GPU binary (`RTX 20xx -> RTX 50xx`):
```bash
./scripts/build.sh --arch all --test
```

---

## Static Geophysical Data Setup

Static raw terrain, bathymetry, and soil files are excluded from Git (several GiB). Use the built-in deterministic downloader [`scripts/download_static_data.py`](scripts/download_static_data.py) — **no login or API key needed** for Copernicus DEM, EMODnet/GEBCO ERDDAP/CEDA, or FAO HWSD v2.2.

### Quick All-in-One Download for Benelux-DE 2.5 km + Lossless `.wx1` Build
```bash
# 1. Inspect resolved URLs and tile count without downloading
python scripts/download_static_data.py --all --domain lam_bnl_ns_de_2p5km --dry-run

# 2. Download Copernicus DEM GLO-30 + WBM, regional MSL Bathymetry, FAO HWSD v2.2, and build .wx1
python scripts/download_static_data.py --all --domain lam_bnl_ns_de_2p5km --workers 6 --prepare-wx1
```

### Dataset-by-Dataset Commands & Manual Sources

1. **Copernicus DEM GLO-30 (30 m) & GLO-90 (90 m) + Water Body Mask (`WBM.tif`):**
   - Source: Public AWS S3 Open Data buckets `copernicus-dem-30m` and `copernicus-dem-90m` (anonymous HTTPS, zero credentials).
   - Preserves below-sea-level polders (Flevoland / Zuidplaspolder) and distinguishes ocean (`WBM=1`), inland lakes (`WBM=2`), and rivers (`WBM=3`).
   ```bash
   # Download GLO-30 (+ optional --glo90 fallback) for European/Dutch domains
   python scripts/download_static_data.py --copernicus-dem --glo90 --domain lam_eu_5km --workers 8
   ```
   Expected layout (`TEMPEST_COPERNICUS_DEM_DIR` or `data/static/copernicus_dem_raw`):
   ```
   dem30_europe/{DEM,WBM}/Copernicus_DSM_COG_10_{N|S}XX_00_{E|W}XXX_00_{DEM|WBM}.tif
   dem90_world/{DEM,WBM}/Copernicus_DSM_COG_30_{N|S}XX_00_{E|W}XXX_00_{DEM|WBM}.tif
   ```

2. **Bathymetry (GEBCO 2026 / 2025 / ERDDAP MSL & EMODnet WCS LAT):**
   - **Fast regional MSL NetCDF subset (`--bathy-source gebco-erddap`, default, ~10 MB):**
     ```bash
     python scripts/download_static_data.py --bathymetry --bathy-source gebco-erddap --domain lam_bnl_ns_de_2p5km
     ```
   - **Full global `GEBCO_2026.nc` (~7.0 GiB, MSL) from BODC/CEDA (recommended for Global `D1` + NEMO/SI3):**
     ```bash
     python scripts/download_static_data.py --bathymetry --bathy-source gebco-2026-global
     ```
     *(Alternative global zip: `--bathy-source gebco-2025-zip` ~4.0 GiB from CEDA, or interactive/CEDA portals at https://www.gebco.net/data-products/gridded-bathymetry-data)*
   - **EMODnet Bathymetry Digital Terrain Model via OGC WCS (`emodnet:mean`):**
     ```bash
     python scripts/download_static_data.py --bathymetry --bathy-source emodnet-wcs --domain lam_bnl_ns_de_2p5km
     ```
     > **Datum Integrity Note:** EMODnet heights are referenced to **LAT** (Lowest Astronomical Tide), whereas GEBCO & Tempest ocean grids use **MSL** (Mean Sea Level). `tempest_v2.bathymetry.prepare_bathymetry()` strictly refuses to mix `lat`/`alat` inputs into MSL unless an explicit spatial `(MSL - LAT)` offset GeoTIFF is passed (`--vertical-offset-raster`).

3. **FAO / IIASA Harmonized World Soil Database v2.2 (HWSD v2.2):**
   - Requires OS package `mdbtools` (`mdb-tables`, `mdb-export`) to read `HWSD2.2.mdb` cleanly (no HWSD v1.x fallback mixing).
   ```bash
   sudo apt install -y mdbtools
   python scripts/download_static_data.py --hwsd
   ```
   Official archives fetched into `data/static/hwsd_v22/downloads/`:
   - `hwsd2.2_db.zip` (~11.1 MB, `HWSD2.2.mdb`)
   - `hwsd2.2_raster.zip` (~22.4 MB, `HWSD2.bil` 1 km global grid)

### Portable Path Environment Variables
If you store large raw static datasets or KNMI archives on another drive (e.g. `/data/...`), set:
```bash
export TEMPEST_COPERNICUS_DEM_DIR="/path/to/COPERNICUS_DEM"
export TEMPEST_BATHY_DIR="/path/to/bathymetrie"
export TEMPEST_HWSD_V22_DIR="/path/to/hwsd_v22"
export TEMPEST_ERA5_CASES_DIR="/path/to/knmi_uur_archives"  # optional local knmi_uur_*.csv.gz dir
export KNMI_API_KEY="your_optional_knmi_open_data_key"     # or config/knmi_api_key.txt (gitignored)
```

---

## Runnable Weather Cases & Multi-Season Verification

Tempest-NWP v2 supports both **live ECMWF operational forecasts** and **reproducible historical ARCO-ERA5 hindcasts** paired with automated **KNMI station + ARCO-ERA5 dual verification**.

### 1. Single 24h Hindcast Examples (ARCO-ERA5, Zero Login Required)

| Weather Regime / Case | Date (`00:00Z`) | Domain | Why It Matters |
| :--- | :---: | :---: | :--- |
| **Winter Storm Eunice** | `2022-02-18` | `lam_bnl_ns_de_2p5km` | Extreme coastal/inland gusts (~40 m/s), sting-jet dynamics |
| **Winter Storm Henk** | `2024-01-02` | `lam_bnl_ns_de_2p5km` | Strong frontal gusts (35 m/s) + heavy winter rain |
| **Summer Storm Poly** | `2023-07-05` | `lam_bnl_ns_de_2p5km` | Explosive North Sea coast mesoscale storm (41 m/s gusts + thunder) |
| **Brabant Hail Supercell** | `2016-06-23` | `d3_nl_2p5km` | Giant hail (`qs`/`qg` graupel + deep convection + 131 KNMI thunder hrs, 49.6 mm) |
| **Spring MCS Convection** | `2018-05-29` | `d3_nl_2p5km` | Organized spring deep convection (136 thunder hrs) |
| **Warm High Conv. (No Thunder)** | `2024-06-18` | `d3_nl_2p5km` | Heavy warm showers just below lightning initiation (34.5 mm max, 0 thunder) |
| **Cold Low-Topped Thunder** | `2024-11-22` | `d3_nl_2p5km` | Polar-air trough shallow convection with thunder & snow (47 thunder, 79 snow hrs) |
| **Snowstorm Darcy** | `2021-02-07` | `lam_bnl_ns_de_2p5km` | Easterly blizzard & snow-albedo coupling (495 KNMI snow hrs, `-5.9 °C`) |
| **Freezing Rain / IJzel** | `2016-01-05` | `d3_nl_2p5km` | Warm-nose aloft + sub-zero boundary layer ice transition |
| **Limburg Cut-Off Extreme Rain** | `2021-07-14` | `lam_bnl_ns_de_2p5km` | Quasistationary convective rainband (40.6 mm at KNMI AWS) |
| **Winter Radiation Fog** | `2024-12-27` | `d3_nl_2p5km` | Light-wind stable boundary layer & persistent radiation fog (490 fog hrs) |
| **Clear Spring Diurnal Cycle** | `2025-03-19` | `d3_nl_2p5km` | Cloudless radiative heating/cooling amplitude (`-4.7 °C` to `+18.6 °C`, 0 mm) |
| **All-Time Heatwave Record** | `2019-07-25` | `d3_nl_2p5km` | Extreme dry soil + sensible heat flux (`+40.3 °C` peak) |

Run any 24h hindcast directly with lean ERA5 LBC streaming (`--stream-era5-lbc` avoids downloading multi-GB raw caches) and verify against KNMI + ARCO-ERA5:

```bash
# Example A: 2016-06-23 Giant-Hail Supercell (24h, Benelux-NL 2.5 km)
python scripts/run_tempest_v2.py \
  --mode lam_era5_hindcast \
  --lam-domain d3_nl_2p5km \
  --init-time 2016-06-23T00:00Z \
  --hours 24 --nz 65 \
  --stream-era5-lbc --cy46-physics --pbl-last \
  --runs-root runs/examples/supercell_20160623

python scripts/verify_knmi_wx1.py \
  --run-dir runs/examples/supercell_20160623/20160623_00Z \
  --obs-source auto \
  --compare-era5

# Example B: 2022-02-18 Storm Eunice (24h, Benelux-North Sea-Germany 2.5 km)
python scripts/run_tempest_v2.py \
  --mode lam_era5_hindcast \
  --lam-domain lam_bnl_ns_de_2p5km \
  --init-time 2022-02-18T00:00Z \
  --hours 24 --nz 65 \
  --stream-era5-lbc --cy46-physics --pbl-last \
  --runs-root runs/examples/eunice_20220218

python scripts/verify_knmi_wx1.py \
  --run-dir runs/examples/eunice_20220218/20220218_00Z \
  --obs-source auto \
  --compare-era5
```

### 2. Multi-Season Benchmark Suite (`quick4`, `core10`, `full20`)
Run curated presets spanning winter/spring/summer/autumn + storm/convection/fog/snow/hail/freezing-rain/heatwave, automatically persisting all `.wx1` forecasts and producing `benchmark_scorecard.csv`, `benchmark_scorecard.json`, and `benchmark_scorecard.png` (resumable if interrupted):

```bash
# Inspect all cases and cache status without running the GPU solver
python scripts/run_benchmark_suite.py --preset full20 --dry-run

# Fast 4-regime smoke benchmark (radiation fog, clear diurnal, hail supercell, low-topped thunder)
python scripts/run_benchmark_suite.py --preset quick4 --lam-domain d3_nl_2p5km --hours 24

# Representative 10-case multi-season benchmark
python scripts/run_benchmark_suite.py --preset core10 --lam-domain d3_nl_2p5km --hours 24

# Full 20-case all-season, all-weather-type verification suite
python scripts/run_benchmark_suite.py --preset full20 --lam-domain d3_nl_2p5km --hours 24
```

*(Note: `--knmi-cache-dir` uses KNMI's open `hourly-in-situ-meteorological-observations-validated` dataset automatically for historical cases via `obs_source=auto`, or local `knmi_uur_*.csv.gz` archives if `TEMPEST_ERA5_CASES_DIR` is populated.)*

### 3. Operational Real-Time ECMWF LBC Forecast
Run a live 2.5 km Limited-Area forecast initialized from ECMWF Open Data (`0p1` with `0p25` fallback, `--ecmwf` selects ECMWF 3h/6h output schedule while physics/dynamics remain 100% Tempest):
```bash
python scripts/run_tempest_v2.py \
  --mode lam_ecmwf_lbc \
  --lam-domain lam_bnl_ns_de_2p5km \
  --lbc-resol auto \
  --lbc-interval-hours 3 \
  --hours 24 --nz 65 \
  --cy46-physics --pbl-last \
  --runs-root runs/operational_lam
```
Add `--enable-da --download-live` to assimilate live KNMI/DWD/radiosonde observations via Balanced 3D-Var + IAU, and `--enable-ocean-si3` on coupled global/marine configurations.

### 4. Global Coupled Atmosphere + NEMO 5.0.2 / SI3 (`D1`)
```bash
python scripts/run_tempest_v2.py \
  --mode d1 \
  --d1-domain d1_global_0p4 \
  --hours 24 --nz 65 \
  --enable-ocean-si3 \
  --runs-root runs/global_coupled_smoke
```

---

## Repository Structure

```text
├── AGENTS.md                        # Project engineering rules (0% AI/ML, SI units, WX-1, minimal design)
├── Cargo.toml / Cargo.lock          # Rust workspace (tempest-core + tempest-cuda)
├── pyproject.toml                   # Python packaging (depends on PyPI tempest-wx1>=0.1.8)
├── crates/
│   ├── tempest-core/                # Rust geodesy, thermodynamic constants & core utilities
│   └── tempest-cuda/
│       ├── build.rs                 # Auto-detects GPU arch, supports RTX 50xx sm_120 & multi-arch fatbin
│       └── src/cuda/
│           ├── nh_core.cu           # Non-hydrostatic split-explicit RK3 / WENO-5 dynamical core
│           ├── nh_physics.cu        # 6-class microphysics, YSU+EDMF PBL, ecRad SW/LW, 4L soil, FLake
│           └── nh_ocean_si3.cu      # Self-contained NEMO 5.0.2 3D ocean + SI3 5-category sea-ice solver
├── python/tempest_v2/               # Drivers, CUDA ctypes bridge, static preprocessors, DA, WX-1 I/O, verification
├── scripts/
│   ├── build.sh                     # One-command Rust + CUDA build & test helper
│   ├── download_static_data.py      # Copernicus DEM, GEBCO/EMODnet Bathymetry, FAO HWSD v2.2 downloader
│   ├── run_tempest_v2.py            # Main operational & ERA5 hindcast CLI
│   ├── run_benchmark_suite.py       # Resumable multi-season benchmark & KNMI/ERA5 scorecard runner
│   ├── select_era5_cases.py         # Historical KNMI archive weather-regime case selector
│   ├── verify_knmi_wx1.py           # Automated hourly KNMI + ARCO-ERA5/IFS verification CLI
│   ├── export_wx1_products.py       # WX-1 derived synoptic/sounding product exporter
│   └── run_operational_daemon.py    # Cycled operational forecast daemon
├── docs/                            # Physics equations provenance & NEMO/SI3 port notes
└── tests/                           # Pytest & CUDA regression suite
```

---

## GitHub Private-to-Public Checklist

Before switching a private mirror to **public**:
1. Confirm `git status` is clean and no secrets/keys (`KNMI_API_KEY`, `.env`, `config/knmi_api_key.txt`) or large `.wx1`/`.nc`/`.grib2`/`.tif`/`.mdb` binaries are tracked (all covered in `.gitignore`).
2. Review/remove any personal scratch notebooks (e.g. `notebooks/aurora15_colab_fixed_v4.ipynb`) or machine-local path links (e.g. in `REPORT.md`) if desired.
3. Add your preferred open-source `LICENSE` file (note `Cargo.toml` specifies `MIT OR Apache-2.0` and `pyproject.toml` specifies `Apache-2.0` — align both before public release).

---

## Running Tests

```bash
# Rust & CUDA kernel tests (runs on GPU)
cargo test --release

# Python unit, physics, static-data, and pipeline tests
pytest -q tests/
```
