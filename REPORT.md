# Tempest-NWP v2 — Onderzoeks-, Stabilisatie- en Verificatierapport (Fase 0–3)

**Datum:** 22 november 2026  
**Werkruimte:** `/home/j_ord/AI/Tempest/Tempest-NWPv2-kopie` (git HEAD voor rapport: `6d80672`)  
**Doelsysteem:** WSL2 Ubuntu (`23 GiB` RAM, `8.0 GiB` swap uit `32 GB` fysiek host-RAM; `.wslconfig`: `memory=24GB, processors=12, swap=8GB`) · NVIDIA RTX 3060 Laptop (`6143 MiB` VRAM)  
**Projectregel:** 0% AI/ML, 0% mock data; uitsluitend klassieke fysische NWP in SI-eenheden en verliesvrije WX-1-opslag.

---

## 1. Managementsamenvatting & Fact-check van Claude's hypothese

| Onderdeel van Claude's hypothese | Bevinding uit metingen & code | Oordeel |
|---|---|---|
| OOM / swap thrashing als oorzaak van `Killed` en oplopende stapduur bij stap 45 | Gedeeltelijk juist voor host-pieken (436 MB `zero_iau` piek op stap 45 + 440 MiB blijvend IAU-VRAM + gelijktijdige D2+D3 IC-allocatie + `cfgrib`-cache + 1-tile VRAM zonder veiligheidsmarge richting Windows/D3D12-gedeeld geheugen), maar in de 2-uurs gestabiliseerde run bleef `VmSwap = 0.0 MB` (`free -h`: `23 GiB` totaal, `21 GiB` beschikbaar, `924 KiB` swap gebruikt; `dmesg` leeg). | **Gedeeltelijk bevestigd** |
| Oplopende stapduur tot `16.4 s` door swap | In de 2-uurs D2-meting met `swap = 0.0 MB` zakte de GPU-klok door hitte van `1890 MHz` (`77.6 °C`, stap 1) naar `390–900 MHz` (`88.4–89.4 °C`, NVML throttle-vlag `0x20 SwThermalSlowdown`), waardoor `dt_step` opliep van `1.40 s` naar `4.26–4.57 s` in thermisch evenwicht. Op een 6+ uur durende run (of eerdere D2→D3 run met host-swapping boven `24 GB`) versterkten swap en thermal throttling elkaar. | **Deels thermische throttling, deels geheugenpiek** |
| Model gebruikt waarschijnlijk `fp64`, dus omzetten naar `fp32` halveert ~11 GB naar ~5,7 GB | **Onjuist.** Dynamica (`nh_core.cu`), fysica (`nh_physics.cu`) en NEMO+SI3 (`nh_ocean_si3.cu`) draaiden al in `float32` (`4849–5289 MiB` VRAM totaal voor `137×400×520`); alleen 1D/2D-geometrie en Thomas-hulpbuffers (`d_tri_c/d_tri_d`) zijn `double`, en 3D-Var (`var3d.py`) draait in `float64` op CPU-RAM. | **Onjuist (FP32/FP64 al aanwezig)** |
| Overige niet-genoemde fouten gevonden bij verificatie | (1) `ic_builder.py:387-398` klemt begintoestandsdruk >1000 hPa af (`-15.36 hPa` bias in hogedrukgebieden op `T+00h`) en laat `q2m` ongeïnitialiseerd (`Td2m = 203 K`); (2) `nh_core.cu:1775,1795,2677-2686` mist vocht-LBC's (`d_lbc_qv`), wat een droge westrandwand veroorzaakt (`TCWV 12.7 → 0.10–2.13 kg/m²` over 150–300 km in 6 uur); (3) DA accepteert ongedateerde METAR/GNSS-ZTD/AMV/ASCAT uit de toekomst (`+8h`). | **3 kritieke nieuwe bugs aangetoond** |

---

## 2. Fase 0 — Stabilisatie & 2-uurs D2-Groeicurve (`c6600f3`)

### 2.1 Oorzaken en doorgevoerde fixes (commit `c6600f3`)
1. **IAU VRAM- en RAM-piek na stap 45:** `runner.py` maakte bij het einde van de IAU-warmup een tijdelijke 436 MB host-dict `zero_iau` aan zonder de GPU-buffers (`d_iau_u/v/thp/pip`) vrij te geven. Vervangen door `nh_cuda_clear_iau_forcing` / `solver.clear_iau_forcing()` + `gc.collect()` in [`nh_core.h`](file:///home/j_ord/AI/Tempest/Tempest-NWPv2-kopie/crates/tempest-cuda/src/cuda/nh_core.h#L253), `nh_core.cu`, [`lib.rs`](file:///home/j_ord/AI/Tempest/Tempest-NWPv2-kopie/crates/tempest-cuda/src/lib.rs), [`cuda_bridge.py`](file:///home/j_ord/AI/Tempest/Tempest-NWPv2-kopie/python/tempest_v2/cuda_bridge.py#L848-L864) en [`runner.py`](file:///home/j_ord/AI/Tempest/Tempest-NWPv2-kopie/python/tempest_v2/runner.py).
2. **Dubbele IC-levenscyclus D2+D3 en `cfgrib`-cache:** `run_two_nest_pipeline` in [`runner.py`](file:///home/j_ord/AI/Tempest/Tempest-NWPv2-kopie/python/tempest_v2/runner.py#L1169) laadde D2 en D3 tegelijk in RAM; dit is sequentieel gemaakt (D2 afronden en vrijgeven voordat D3-IC geladen wordt) met `clear_cfgrib_cache()` in [`ic_builder.py`](file:///home/j_ord/AI/Tempest/Tempest-NWPv2-kopie/python/tempest_v2/ic_builder.py#L287-L289).
3. **VRAM-plafond voor 1-tile grids:** `NonHydrostaticGPUSolver.__init__` in `cuda_bridge.py` begrensde het VRAM-budget niet op live vrij VRAM; afgetopt op 85% van `nh_cuda_vram_info()` vóór `plan_horizontal_tiling`.
4. **Bloom (1996) IAU dubbeltelling in single-domain CLI + profiler:** [`scripts/run_tempest_v2.py`](file:///home/j_ord/AI/Tempest/Tempest-NWPv2-kopie/scripts/run_tempest_v2.py) startte single-domain DA met `ana_st` i.p.v. achtergrond `st_bg` tijdens IAU-forcing. Opt-in geheugen- en GPU-telemetrie toegevoegd via `--mem-profile-every N` (standaard `0`).

### 2.2 Gemeten 2-uurs D2-productierun (`137×400×520`, `dt=20 s`, 360 stappen, DA+IAU+NEMO/SI3)
Gemeten zonder swap (`VmSwap = 0.0 MB` op elke meting; totale wandtijd `1226.84 s`):

| Stap | Model-tijd | RSS (MB) | Swap (MB) | VRAM (MiB) | GPU Temp / Klok / Vermogen | `dt_step` | Gebeurtenis |
|---|---|---|---|---|---|---|---|
| 1 | 00:00:20 | 2909.1 | 0.0 | 5289 / 6143 | 77.6 °C · 1890 MHz · 46.0 W | 2.48 s | Start |
| 10 | 00:03:20 | 2909.1 | 0.0 | 5289 / 6143 | 84.1 °C · 1800 MHz · 75.3 W | 1.40 s | Opwarming GPU |
| 40 | 00:13:20 | 2909.1 | 0.0 | 5289 / 6143 | 86.9 °C · 1080 MHz · 51.1 W | 1.84 s | Vlak voor einde IAU |
| 45 | 00:15:00 | — | — | — | — | — | `IAU warm-up completed; IAU buffers freed` |
| 50 | 00:16:40 | 2909.1 | 0.0 | **4849** / 6143 | 87.7 °C · 900 MHz · 51.8 W | 2.23 s | **-440 MiB VRAM vrijgegeven; 0 MB RSS-piek** |
| 90 | 00:30:00 | 2909.1 | 0.0 | 4849 / 6143 | 88.5 °C · 900 MHz · 48.7 W | 3.02 s | Net voor 1e NEMO+SI3 koppeling (`0.34 s`) |
| 100 | 00:33:20 | 3960.4 | 0.0 | 4871 / 6143 | 88.4 °C · 900 MHz · 45.4 W | 3.21 s | Na 1e oceaan-download naar host (daarna vlak) |
| 180 | 01:00:00 | 3960.4 | 0.0 | 4871 / 6143 | 89.0 °C · 900 MHz · 33.3 W | 4.14 s | Vlak tot `T+01h` WX-1 output (`25.57 s`) |
| 190 | 01:03:20 | 4625.3 | 0.0 | 4871 / 6143 | 88.4 °C · 900 MHz · 34.2 W | 2.78 s | Output-diagnostiek in host-buffer (vlak t/m 270) |
| 280 | 01:33:20 | **4034.4** | 0.0 | 4871 / 6143 | 88.7 °C · 900 MHz · 28.8 W | 4.49 s | Teruggezakt naar stabiel plateau |
| 310 | 01:43:20 | 4034.4 | 0.0 | 4871 / 6143 | 88.6 °C · 562 MHz · 44.0 W | 4.57 s | Stabiel; `0x20 SwThermalSlowdown` |
| 340 | 01:53:20 | 4034.4 | 0.0 | 4871 / 6143 | 89.1 °C · 390 MHz · 27.9 W | 4.26 s | Stabiel |
| 360 | 02:00:00 | **4034.4** | **0.0** | **4871** / 6143 | 88.9 °C · 615 MHz · 44.4 W | **4.53 s** | Einde 2h run; alle 3 WX-1 & restarts geschreven |

*Transparantie-noot over `np_arrays=2`:* `gc.get_objects()` in CPython volgt alleen cyclische Python-containers en telt ruwe numerieke `numpy.ndarray`-allocaties daarom niet mee. Het echte geheugengebruik en lekgedrag worden betrouwbaar gemeten via OS-RSS (`4034.4 MB` exact vlak van stap 280 t/m 360), `VmSwap` (`0.0 MB`), en CUDA VRAM (`4871 MiB` exact vlak van stap 100 t/m 360). In thermisch evenwicht (stappen 280–360) bleef `dt_step` tussen `4.26 s` en `4.57 s` (±7% rond het gemiddelde, binnen de eis van 10%).

---

## 3. Fase 1 — Geverifieerde 2026-Kennisbank

Alle bevindingen uit de aangeleverde notities (`01` t/m `04` en de Cy46 Technical Note PDF) zijn op **22 november 2026** onafhankelijk online geverifieerd:

| Onderwerp | Kernbevinding & impact op Tempest | URL | Publicatiedatum | Bronstatus | Verificatie |
|---|---|---|---|---|---|
| **1a. KNMI HARMONIE Cy46 overgang** | Omschakeling van HA43 naar HA46 in november 2026 **uitgesteld tot ten vroegste half januari 2027** door technische uitdagingen en grote gebruikersverzoeken; HA43 blijft operationeel en Cy46 P1/P3/P5 blijven testdata. | https://english.knmidata.nl/latest/news/2026/10/14/transition-to-harmonie-46-in-early-2027 | 14 okt 2026 | Officieel (KNMI Data Platform) | **Bevestigd** |
| 1a. KNMI testdata & GRIB2-pakketten | Cy46 testdata (GRIB2; P1 NL `0.018°×0.029°`, P3/P5 EU) beschikbaar; testfase verlengd. | https://www.knmidata.nl/actueel/nieuws/2026/04/03/testdata-harmonie46-beschikbaar / https://dataplatform.knmi.nl/dataset/harmonie-arome-cy46-p1-1-0 | 3 apr 2026 / aug 2026 | Officieel (KNMI) | **Bevestigd** |
| 1a. Inhoud Cy46-verificatierapport (Cy46 vs Cy43) | Het interne testdata-verificatierapport zit gebundeld in de SURF/KDP-download en is niet los geïndexeerd in publieke zoekmachines. | — | — | — | **Niet gevonden** |
| **1b. ECMWF IFS 50r1 live** | Cy50r1 (+ AIFS v2) live sinds 12 mei 2026: verticale diffusie laatst, convectieve neerslag via wolkenschema, NEMO4-SI3 koppeling met sneeuw/ijsdikte terug naar atmosfeer, 15-min 4D-Var timeslots + outer-loop herscreening. | https://www.ecmwf.int/en/about/media-centre/news/2026/ifs-cycle-50r1-aifsv2-live | 12 mei 2026 | Officieel (ECMWF) | **Bevestigd** |
| 1b. IFS 50r1 nieuwe/vervallen parameters | Nieuwe parameters: `wbt` (wet-bulb T), `fscov` (snow cover fraction), 40+ NEMO4-SI3 `o2d`/`o3d` en sneeuw-op-zee-ijs; `cur` (urban cover) vervangt vervallen `vegdiff`; aparte HRES opgegaan in ENS-control (open-data-pad `ifs/0p25/oper` werkt nog). | https://www.ecmwf.int/en/newsletter/185/earth-system-science/upgrade-ifs-cycle-50r1 / https://events.ecmwf.int/event/531/attachments/3520/5948/50r1_AIFS2_presentation.pdf | okt 2025 / mei 2026 | Officieel (ECMWF) | **Bevestigd** |
| 1b. ECMWF IFS 50r2 GRIB2-only & eenheden | Doel Q2 2027 (6 mnd e-suite vanaf Q4 2026, wetenschappelijk gelijk aan 50r1, **uitsluitend GRIB2**). Tempest `ingest_ecmwf.py` gebruikt al GRIB2 via `cfgrib`, maar **50r2 wijzigt `tp` (`m → kg m⁻²`) en `tcc` (`0–1 → 0–100 %`)**: `ic_builder.py:527` checkt `"m" in units` voor `sd`, maar verificatie/export moet `units` van `tp` en `tcc` inspecteren. | https://confluence.ecmwf.int/plugins/viewsource/viewpagesrc.action?pageId=554145353 / https://forum.ecmwf.int/t/upcoming-migration-to-grib2-data-format/15034 | 2026 | Officieel (ECMWF) | **Bevestigd** |
| **1c. ACCORD 2026 & Cy46 Tech Note** | 8e ACCORD Nieuwsbrief + Cy46 Tech Note (*Meteorology* 2024 3(4), 354–390): **past in 0 MB extra VRAM:** CDNC-profiel (`250·P/1000`, ×0.25 niveau 0), Nielsen LW-optiek, sneeuwsmelt `0.25`, `XRIMAX=0.2`, `LFAKETREE`, OCND2/ICE-T. **Te zwaar voor 6 GB VRAM:** OOPS-EnVar, 3D-SPARTACUS ecRad, LIMA + 14 NRT CAMS-aerosolen. *Preprint-waarschuwing:* Tabel 1 rijen 2–5 missen mintekens in `c_λ`; rijen 1 en 6–13 kloppen exact met Fig. 4. | https://www.accord-nwp.org/?NL-editions / https://www.mdpi.com/2674-0494/3/4/18 | 8 apr 2026 / 2024 | Officieel / peer-reviewed | **Bevestigd** |
| **1c. NOAA GFS v17 uitstel** | Invoering GFS v17 verplaatst van oktober 2026 naar **eind 2027** (cloud-transitie); huidig operationeel model blijft GFS v16.x. Geen actie nodig voor GFS-ingest. | https://www.weather.gov/media/notification/pdf_2026/pns26-29_Updated_Science_for_GFSv17_aaa.pdf | 28 okt 2026 | Officieel (NOAA/NWS) | **Bevestigd** |
| 1c. NOAA UFS Land DA v3.0.0 / JEDI | CPU-gebaseerde bodem-/sneeuw-OI/LETKF uit `T2m`, `q2m` en sneeuwfractie past in <200 MB host-RAM (0 MB VRAM); Thompson-Eidhammer 2-moment + MOM6/CICE6/WW3 vallen buiten 6 GB VRAM. | https://epic.noaa.gov/ufs-land-da-v3-0-0-release/ | 2025/2026 | Officieel (NOAA EPIC) | **Bevestigd** |

---

## 4. Fase 2 — Code-audit, Vertaling & Baseline-Verificatie (D2 Europa)

### 4.1 Antwoorden op de 6 onderzoeksvragen uit `04_tempest_toepassing.md`

1. **Dynamica (expliciet vs semi-impliciet / semi-Lagrangiaans):**  
   Tempest is **niet** SISL, maar een **split-expliciete Wicker-Skamarock RK3**-oplosser (`nh_core.cu:nh_cuda_step_rk3`, L2848) met een horizontaal-expliciete / verticaal-impliciete (**HEVI**) Thomas-solver (`nh_acoustic_vert_implicit_kernel`, L1527) en **IEVA** verticale advectie (`nh_ieva_vert_advect_*_kernel`, L970–1243). Daarom kost een kleine tijdstap (`dt = 20 s` op D2, `4 s` op D3) meer rekentijd dan HARMONIE (`75 s` SISL).
2. **Precisie (`float32` vs `float64`):**  
   **Het verwachtingsmodel, de fysica én NEMO+SI3 draaien al in `float32` (FP32)** (`nh_core.cu:2644-2696`, `nh_physics.cu`, `nh_ocean_si3.cu:210-233`). Alleen 1D/2D-metriek en Thomas-hulpbuffers (`d_tri_c/d_tri_d`, ~456 MB op D2) zijn `double`. **De DA (`da/var3d.py`) draait al in `float64` (FP64).** Prioriteit #2 uit `04` is dus reeds geïmplementeerd.
3. **Referentietijd van het `±6 h` DA-venster (`verification.py:182-230` & `runner.py:972`):**  
   Het venster is gecentreerd op `init_time_utc`, maar de code bevat **drie ernstige bugs**:  
   - (i) `--download-live` haalt waarnemingen op het actuele kloktijdstip op (~08:13 UTC vs `00:00Z` analyse), waardoor `+3h..+6h` toekomstige WIS2/boei-data binnen `±6h` glippen;  
   - (ii) `global_metars` (~5.003), `egvap_gnss_ztd` (87, `time=0`), `eumetsat` AMV (~24.923) en `eumetsat_ascat` (267.648) missen geldige tijdstempels (`<= 1e8` of `NaN`) en **omzeilen de venstercheck** in `_in_domain_and_window` (`t_off = 0.0`), waardoor `08:13Z`-metingen de `00:00Z`-analyse vergiftigen;  
   - (iii) `_build_fgat_trajectory` (`verification.py:680-715`) maakt slechts 3 tijdspunten (`t_lo, 0, t_hi`) via één afgekapte Euler-stap i.p.v. 15-minuten-timeslots met outer-loop herscreening.
4. **Inhoud en cadans van `physics (every=2)` (`nh_physics.cu:2177-2290` & `runner.py:591-601`):**  
   Elke 40 s (`2 × 20 s`) draait `nh_cuda_step_physics`: (1) tweestrooms ecRad-achtige straling, **maar volledig herrekend om de `rad_interval_s = 900 s` (15 min)** (`nh_physics.cu:2195-2210` — Prioriteit #9 is al aanwezig!); (2) Oppervlak/LSM/COARE3.6/FLake/SI3; (3) YSU niet-lokale PBL + ondiepe EDMF; (4) 6-klasse gemengdfasige microfysica + neerslagval.
5. **Verschil tussen D2 en D3 fysica:**  
   **Geen enkel codeverschil**: beide roepen dezelfde `nh_cuda_step_physics` aan en hebben **geen diepe-convectieparametrisatie** (Prioriteit #7 uit `04` is n.v.t.). D3/D2 hebben wel constante CDNC `200e6/80e6` (met dubbeltelling van Seifert-Beheng en Kessler/Berry autoconversie in `nh_physics.cu:1694-1704`), momentum-PBL zonder `wscale`/vrije-atmosfeer `Ri`, geen subgrid-wolkenfractie, geen `XRIMAX` en geen sneeuwsmeltminimum `0.25`.
6. **Terugkoppeling van NEMO+SI3 naar de atmosfeer (`runner.py:712-714`, `ocean_si3_bridge.py:534-551`, `nh_ocean_si3.cu:298-316`):**  
   De atmosfeer **krijgt elke `1800 s` live `sst_bulk`, `sea_ice_frac`, `sea_ice_tsfc` en `sea_ice_albedo`** uit NEMO+SI3 terug (`couple_to_surface_state`). Wel blijven drie gaten open: (i) `nh_physics.cu:1062` negeert `sea_ice_thick` (`(void)sea_ice_thick;`) en sneeuwdikte `v_s` werkt alleen indirect via `ice_albedo`; (ii) `nh_ocean_si3.cu:298-316` berekent Large & Pond-fluxen zelf over i.p.v. de atmosferische COARE 3.6-fluxen over te nemen; (iii) `ic_builder.py` negeert ECMWF `sithick`/`zos`.

### 4.2 Volledige Vertaaltabel (`04` → Tempest-NWP v2)

| Verbetering | Bron | Tempest-onderdeel + bestand/functie | Wat er nu is | Benodigde aanpassing | Extra RAM/VRAM | Risico | Hoe te testen |
|---|---|---|---|---|---|---|---|
| Geheugengroei / levenscyclus | Logs | `nh_core.cu`, [`cuda_bridge.py`](file:///home/j_ord/AI/Tempest/Tempest-NWPv2-kopie/python/tempest_v2/cuda_bridge.py), [`runner.py`](file:///home/j_ord/AI/Tempest/Tempest-NWPv2-kopie/python/tempest_v2/runner.py) | Oude code lekte IAU-VRAM + laadde D2/D3 IC tegelijk | `nh_cuda_clear_iau_forcing`, 85% VRAM-cap, sequentieel D2→D3, `clear_cfgrib_cache` | Bespaart 440 MiB VRAM + GB's RAM | Laag (**Fase 0 opgelost**) | 2-uurs D2 profiling |
| `fp32` forecast / `fp64` DA | IFS 50r1, Cy46 | `nh_core.cu`, `nh_physics.cu`, `var3d.py` | **Al aanwezig** (alleen `tri_c/d` en 2D-metriek zijn `fp64`) | Geen codewijziging nodig | 0 | Geen | Code-inspectie bevestigd |
| Stralingsinterval 15 min | Cy46 | [`nh_physics.cu:2195`](file:///home/j_ord/AI/Tempest/Tempest-NWPv2-kopie/crates/tempest-cuda/src/cuda/nh_physics.cu#L2195), [`runner.py`](file:///home/j_ord/AI/Tempest/Tempest-NWPv2-kopie/python/tempest_v2/runner.py#L485) | **Al aanwezig** (`rad_interval_s = 900.0`) | Geen codewijziging nodig | 0 | Geen | Reeds actief in baseline |
| DA-venster ±3h, 15-min timeslots, missing-epoch check, 2-pass herscreening | IFS 50r1 | [`verification.py`](file:///home/j_ord/AI/Tempest/Tempest-NWPv2-kopie/python/tempest_v2/verification.py), [`da/var3d.py`](file:///home/j_ord/AI/Tempest/Tempest-NWPv2-kopie/python/tempest_v2/da/var3d.py) | ±6h venster, ongedateerde METAR/ZTD/ASCAT lekken als `t=0`, 3-punts Euler FGAT | `--da-strict-timeslots`, mappen-fallback, tijdsinflatie, 2-pass `solve_3dvar_pcg` (let op `bg_check` interactie — zie Fase 3 #1) | +0 VRAM, +36 MB host | Hoog bij ongecorrigeerde IC-druk | 6h `0813Z` & `2153Z` benchmarks |
| Verticale diffusie / PBL als laatste fysicastap | IFS 50r1 | [`nh_physics.cu:nh_cuda_step_physics`](file:///home/j_ord/AI/Tempest/Tempest-NWPv2-kopie/crates/tempest-cuda/src/cuda/nh_physics.cu), [`runner.py`](file:///home/j_ord/AI/Tempest/Tempest-NWPv2-kopie/python/tempest_v2/runner.py) | Volgorde was Rad → Sfc → PBL → Microphysics | Opt-in `--pbl-last`: Rad → Sfc → Microphysics → PBL | 0 MB | Zeer laag (**Fase 3 #2 gemerged**) | Unit-test + 6h A/B benchmark |
| Cy46 CDNC-profiel, Nielsen LW, sneeuwsmelt `0.25`, `XRIMAX=0.2` | Cy46 Note | [`nh_physics.cu`](file:///home/j_ord/AI/Tempest/Tempest-NWPv2-kopie/crates/tempest-cuda/src/cuda/nh_physics.cu) (`lw_layer_optics`, `solve_mo_sfclayrev`, microphysics) | Vaste CDNC `200/80 cm⁻³`, Kettler-hoge LW `k_abs`, `f_melt=1`, onbegrensde `Ri_b` | Opt-in `--cy46-physics` (tabel 1 rijen 1,6–13 met `/1.66` schaling) | 0 MB | Laag–Middel (**Fase 3 #3 gemerged**) | Mist unit-test + 6h A/B benchmark |
| IC 1000 hPa clamp & `q2m=0` fix (**nieuw ontdekt buiten `04`**) | Verificatie `T+00h` | [`ic_builder.py:387-408`](file:///home/j_ord/AI/Tempest/Tempest-NWPv2-kopie/python/tempest_v2/ic_builder.py#L387-L408) | `wt` klemt onder 1000 hPa af (`-15.36 hPa` bias hogedruk); `q2m` blijft `0` | Gebruik `ps_child` / hydrostatische extrapolatie naar onderkant; vul `q2m` uit `qv[0]`/`2d` | 0 MB | Middel (kern-IC) | `T+00h/03h/06h` MSLP & Z500 check |
| Vocht-LBC's (`d_lbc_qv`, **nieuw ontdekt buiten `04`**) | Verificatie `T+06h` | `nh_core.cu:1775,1795,2677-2686` | Alleen `u,v,w,thp,pip` hebben Davies-LBC; nulflux op randen droogt westrand uit | Voeg `d_lbc_qv` toe aan LBC-targets/relaxatie + interpolator | ~150 MiB VRAM op `137×400×520` | Middel | Westrand TCWV & neerslag na +6/+24h |
| `LFAKETREE` & nachtelijke momentum-PBL `km` | Cy46 Note | `ic_builder.py` / `nh_physics.cu` (`nh_pbl_momentum_*`) | `W10 land` +1.47..+2.33 m/s te hard, 200-300 m te zwak | Ruwheidsmenging / achtergrond-`km` beperken in SBL | 0 MB | Laag–Middel | 10 m & 50–300 m profiel vs HARMONIE |
| Bodemanalyse uit `T2m`/`q` (OI/LETKF) | GFS v17 / Land DA | Nieuw in `da/` | Geen bodem-DA | Eenvoudige OI voor bovenste `soil_t`/`soil_m` uit 3D-Var innovaties | <50 MB host | Middel | Nachtelijke `T2m`/`Td2m` |
| Convectieve neerslag via wolkenschema | IFS 50r1 | N.v.t. | Geen diepe convectieparametrisatie | Niet van toepassing zolang er geen diepe convectieschema is | — | — | N.v.t. |
| Sea-ice dikte `sea_ice_thick` & `sithick`/`zos` IC + `50r2` eenheidcheck | IFS 50r1/r2 | `ic_builder.py`, `nh_physics.cu:1062` | Dikte genegeerd in atmosfeer; geen `tp`/`tcc` unit check | Geleidend fluxterm over `sea_ice_thick`; `tp`/`tcc` eenheidslezer | 0 MB | Laag | Poolzeestations + `50r2` tests |
| SISL / Thompson-Eidhammer / LIMA / MOM6 | Cy46 / GFS v17 | Kernarchitectuur | GPU-VRAM is 6 GB; huidig `137×400×520` gebruikt al 4.8–5.3 GB | **Niet doen op huidige 6 GB GPU** | Overschrijdt 6 GB | Zeer hoog | — |

### 4.3 Herordende prioriteitenlijst voor 6 GB VRAM + WSL2
1. **[Fase 0 — Opgelost]** Geheugen- en VRAM-levenscyclus stabiliseren (`c6600f3`).
2. **[Urgent vervolg buiten `04`]** Fix `ic_builder.py:387-398` 1000 hPa clamp + `q2m=0` en `nh_core.cu` vocht-LBC `d_lbc_qv` (verklaren veruit de grootste `MSLP`/`Z500`/`TCWV`-fouten).
3. **[Fase 3 #1 — Getest, teruggedraaid]** Strict DA-venster, 15-min timeslots & 2-pass herscreening (eerst `bg_check` scheiden van tijdsinflatie én IC-clamp fixen!).
4. **[Fase 3 #2 — Gemerged `b94f34f`]** Verticale diffusie / PBL als laatste fysicastap (`--pbl-last`, 0 MB).
5. **[Fase 3 #3 — Gemerged `6d80672`]** Cy46 CDNC-profiel + Nielsen LW + sneeuwsmelt `0.25` + `XRIMAX=0.2` (`--cy46-physics`, 0 MB).
6. **[Reeds aanwezig in codebase]** FP32-verwachtingsmodel + FP64-DA (`04` #2) en 15-min straling (`04` #9).
7. **[Toekomst]** `LFAKETREE` / SBL `km`-profiel -> bodem-OI -> `sea_ice_thick` / `50r2`-guards. **Overgeslagen voor 6 GB:** Thompson-Eidhammer, LIMA, MOM6/CICE6, SISL.

### 4.4 Volledige 6-uurs Baseline-Verificatie (Europa D2 `65×400×520`, `2026-11-22 00Z → 06Z`)

#### Tegen ECMWF IFS `0p25` analyse/verwachting (binnengebied D2 excl. 20-cel rand, `n = 172.800`)
| Grootheid | `T+00h` (00Z) Bias / RMSE (`r`) | `T+03h` (03Z) Bias / RMSE (`r`) | `T+06h` (06Z) Bias / RMSE (`r`) | Diagnostische bevinding |
|---|---|---|---|---|
| **MSLP** (binnen) | `-3.95` / `7.42 hPa` (`0.6921`) | `-0.69` / `1.23 hPa` (`0.9936`) | `-1.08` / `1.63 hPa` (`0.9898`) | `T+00h` zwaar verstoord door 1000 hPa clamp; herstelt dynamisch naar ~`-1 hPa` |
| **MSLP** (Benelux `n=9101`) | `-7.09` / `8.98 hPa` (`0.2771`) | `-0.95` / `1.05 hPa` (`0.9730`) | `-1.49` / `1.54 hPa` (`0.9834`) | Hogedrukgebied over Benelux (>1015 hPa) op `T+00h` geclamped op 1000 hPa |
| **T2m** (binnen / land / zee) | `0.00` / `0.00 K` | `+0.00` / `1.05 K` (land `+0.09/1.33`, zee `-0.11/0.48`) | `-0.17` / `1.14 K` (land `-0.03/1.42`, zee `-0.34/0.65`) | Uitstekende thermische evolutie (~`1.1 K` RMSE op 6h) |
| **Td2m** (binnen) | `-74.00` / `74.21 K` | `-0.06` / `0.96 K` (`0.9850`) | `-0.47` / `2.12 K` (`0.9302`) | `surf["q2m"]` was ongeïnitialiseerd bij `T+00h` |
| **Wspd10m** (land / zee) | `+1.38/1.92` / `+1.02/1.41 m/s` | `+2.17/2.68` / `+0.14/1.32 m/s` | `+2.33/2.89` / `+0.46/1.68 m/s` | **Structureel +2.2..+2.3 m/s te harde landwind** |
| **Z500 / T500** | `-2.91/3.06 m` · `-0.02/0.07 K` | `-11.00/13.06 m` · `-0.01/0.38 K` | `-15.34/18.51 m` · `+0.07/0.57 K` | Kolom verzakt -11..-15 m door IC-drukclamp herstel |
| **Z850 / T850** | `-0.95/1.14 m` · `+0.02/0.13 K` | `-7.81/9.92 m` · `-0.10/0.61 K` | `-11.74/14.39 m` · `-0.17/0.72 K` | Kolom-inzinking onderin |
| **Z250 / T250** | `-2.76/3.11 m` · `+0.26/0.40 K` | `-6.92/11.22 m` · `+0.47/0.85 K` | `-9.40/16.47 m` · `+0.69/1.14 K` | Lichte bovenluchtopwarming |
| **Neerslag** (cumulatief) | — | Mod `0.273` vs EC `0.546 mm` | `-0.31` / `1.22 mm` (`r=0.8979`) | Zeer goede frontale structuur (`max 46.9 vs 39.8 mm`) |
| **Westrand-TCWV** `i=0,5,10,20,40` | `[12.57, 12.69, 12.57, 12.25, 14.37]` vs `~12.45` | `[0.10, 0.56, 1.18, 8.84, 13.47]` vs `~12.65` | `[0.10, 0.54, 1.06, 2.13, 9.82]` vs `~12.76` | **Droge wand rukt 225–300 km op door ontbrekende `d_lbc_qv`** |

#### Tegen KNMI HARMONIE-AROME Cy46 `05Z+1h (06Z)` (Benelux+W-DE `n=152.100`) en Live `06Z`-stations
| Referentie | Variabele | Bias | RMSE | `r` | Mod vs Ref | Fysische duiding |
|---|---|---|---|---|---|---|
| HARMONIE Cy46 (Benelux+W-DE) | `MSLP` | `-1.119 hPa` | `1.210 hPa` | `0.9783` | `1011.09 vs 1012.21` | Sterke synoptische overeenkomst |
| HARMONIE Cy46 | `T2m` (all / land / sea) | `+0.111 / +0.517 / -0.449 K` | `1.173 / 1.393 / 0.774 K` | `0.9548` | `274.82 vs 274.30` (land) | Te veel nachtelijke menging over land |
| HARMONIE Cy46 | `W10` (all / land / sea) | `+0.458 / +1.725 / -1.287 m/s` | `1.927 / 2.090 / 1.679 m/s` | `0.7881` | `4.18 vs 2.45` (land) | Landwind te hard, zeewind te zwak |
| HARMONIE profiel (land) | `50 → 100 → 200 → 300 m` | `W`: `+0.81 → 0.00 → -0.75 → -0.88 m/s` · `T`: `+0.13 → -0.18 → -0.43 → -0.56 K` | `W RMSE 1.58–1.75 m/s` | `0.81–0.93` | `300m W: 7.50 vs 8.38` | Bewijst overmatige nachtelijke schuifmenging (zwakke LLJ) |
| HARMONIE flux/BL/wolk | `SHF land/sea` · `LHF sea` · `BLH sea` · `TCC` | `-9.86 / -18.27 W/m²` · `-32.65 W/m²` · `-375.6 m` · `-0.237` | `20.3 / 23.2` · `37.9` · `499 m` · `0.497` | `0.49–0.87` | `BLH_sea 878 vs 1254 m` | Te weinig bewolking (`0.34 vs 0.57`) |
| Live `06Z` stations (`±1.5h`) | `all` (`n=832/172/748`) | `T2m +0.35 K`, `MSLP -1.13 hPa`, `W10 +1.75 m/s` | `1.85 K` / `1.74 hPa` / `2.82 m/s` | — | KNMI (`n=46/35/54`): `+1.52/2.26 K`, `-1.29/1.37 hPa`, `+0.84/1.44 m/s` | Bevestigt DWD (`T2m -0.29/1.37 K`, `W10 +1.45/2.07 m/s`) en METAR (`W10 +1.97/3.05 m/s`) |

---

## 5. Fase 3 — Geïmplementeerde Verbeteringen, Voor/Na-metingen & Merge-beslissingen

Alle Fase-3 benchmarks zijn identiek uitgevoerd op een **NW-Europa `7,5 km` subgrid (`140×200×65`, `dx=7.5 km`, `dt=20 s`, gekoppelde NEMO+SI3, 6h `00Z → 06Z`)** gevolgd door een **25-staps snelheids-/VRAM-meting (na 5 warmup-stappen) op volledig productie-D2 (`137×400×520`)**. Elke feature flag staat standaard **UIT**.

### 5.1 Verbetering #1: `--da-strict-timeslots` (IFS 50r1 venster, 15-min timeslots, missing-epoch check, 2-pass herscreening) — **NIET GEMERGED (teruggedraaid)**

| Scenario (6h NW-EU `00Z → 06Z`) | `EC MSLP` RMSE (Bias) | `EC T2m` RMSE (Bias) | `EC W10` RMSE (Bias) | `HARMONIE T2m` | `HARMONIE W10` | `OBS 06Z T2m` | `OBS 06Z W10` | Beslissing |
|---|---|---|---|---|---|---|---|---|
| `baseline_noda` (zonder DA) | 2.190 (+1.614) | 1.063 (+0.198) | 1.864 (+0.799) | 1.325 (+0.299) | 2.117 (+0.098) | 1.816 (+0.737) | 2.406 (+1.381) | Referentie |
| `0813Z` zonder flag (huidige bug) | 2.367 (+1.811) | 1.204 (-0.366) | 2.045 (+0.848) | 1.356 (-0.358) | 2.215 (+0.124) | 1.624 (-0.077) | 2.597 (+1.585) | Lekte 239 `t2m` + 220 wind van `+08:13Z` |
| `0813Z` mét `--da-strict-timeslots` | **2.190 (+1.614)** | **1.063 (+0.198)** | **1.864 (+0.799)** | **1.325 (+0.299)** | **2.117 (+0.098)** | 1.816 (+0.737) | **2.406 (+1.381)** | Weerde vervuilde `+08Z` obs 100% |
| `2153Z` zonder flag (`±6h`, 3- punts FGAT) | 2.214 (+1.601) | 1.408 (-0.229) | 1.984 (+0.813) | 1.527 (-0.150) | 2.244 (+0.298) | 1.863 (+0.234) | 2.674 (+1.674) | `bg_check` wees `62/1303` obs af |
| `2153Z` mét `--da-strict-timeslots` | 2.395 (+1.799) | 1.394 (-0.034) | **2.531 (+1.345)** | 1.674 (+0.053) | **2.728 (+0.928)** | **2.059 (+0.644)** | **3.212 (+2.237)** | **FAALT merge-gate** (`bg_check` wees maar `5/1303` af) |

**Exacte worteloorzaak van de `2153Z`-degradatie (reden van terugdraaien):**
1. Door de `1000 hPa`-klem in `ic_builder.py` had de `00Z`-achtergrond over Nederland/Benelux een foutieve `-13.14 hPa` drukafwijking (`verify_bg KNMI`). Met de standaard foutvariantie verwierp `var3d` background-QC dit terecht (`62/1303` afgewezen).
2. Met tijdsinflatie `(1 + (t_off/3600)^2) * 1.5` bij `-2h15m` werd `sig_tot` zodanig opgeblazen dat `-13.14 hPa` plots binnen `5 σ` viel (`slechts 5/1303` afgewezen!). Vervolgens zette `BalanceOperator` de onechte `+7 hPa` drukaanpassing om in kunstmatige windincrementen tot `25 m/s`, die `W10` en `MSLP` op `+6h` verslechterden (ook werd herhaald achterwaarts Euler-advectie over `-7200..-10800 s` instabieler).
3. *Conclusie:* De missende-epoch-check en `±3h` vensterbeveiliging zijn waardevol, maar `bg_check` moet op **ongeïnflateerde `sigma_o`** toetsen én de IC-clamp bug moet eerst verholpen zijn. Daarom conform merge-gate volledig teruggedraaid (niet gemerged).

### 5.2 Verbetering #2 (`--pbl-last`, IFS 50r1 verticale diffusie als laatste fysicastap) — **GEMERGED (`b94f34f`)**
### 5.3 Verbetering #3 (`--cy46-physics`, Cy46 CDNC-profiel, Nielsen LW, sneeuwsmelt `0.25`, `XRIMAX=0.2`) — **GEMERGED (`6d80672`)**

| Benchmark (6h NW-EU `00Z → 06Z` & Prod D2) | `baseline_noda` | `+ --pbl-last` (`b94f34f`) | `+ --cy46-physics` (`6d80672`) | Impact `--cy46-physics` vs baseline |
|---|---|---|---|---|
| **ECMWF MSLP** RMSE (Bias) | 2.1901 (+1.6141) | 2.1904 (+1.6141) | 2.2001 (+1.6246) | +0.010 hPa (neutraal) |
| **ECMWF T2m** (alle) RMSE (Bias) | 1.0633 (+0.1978) | **1.0603** (+0.1972) | **1.0283** (+0.1290) | **-0.0350 K RMSE** (bias -0.069 K) |
| **ECMWF T2m land** RMSE (Bias) | 1.2211 (+0.4128) | **1.2174** (+0.4118) | **1.1759** (+0.3121) | **-0.0452 K RMSE** (bias -0.101 K) |
| **ECMWF T2m sea** RMSE (Bias) | 0.6632 (-0.2142) | **0.6624** (-0.2140) | **0.6586** (-0.2219) | **-0.0046 K RMSE** |
| **ECMWF W10 / land** RMSE (Bias) | 1.8636 (+0.7994) / 1.9228 (+1.4682) | 1.8643 / 1.9230 | **1.8609** (+0.7764) / **1.9190** (+1.4351) | **-0.003 m/s RMSE**, land-bias -0.033 m/s |
| **ECMWF tp_6h** RMSE (Bias) | 1.0485 (-0.2031) | **1.0482** (-0.2024) | 1.0546 (-0.2048) | +0.006 mm (neutraal) |
| **HARMONIE T2m** (alle) RMSE (Bias) | 1.3253 (+0.2992) | **1.3209** (+0.2978) | **1.2987** (+0.2214) | **-0.0266 K RMSE** (bias -0.078 K) |
| **HARMONIE T2m land** RMSE (Bias) | 1.5522 (+0.6274) | **1.5461** (+0.6250) | **1.5164** (+0.5044) | **-0.0358 K RMSE** (bias -0.123 K) |
| **HARMONIE W10 land** RMSE (Bias) | 1.8669 (+1.3884) | **1.8662** (+1.3877) | **1.8660** (+1.3462) | **-0.001 m/s RMSE**, land-bias -0.042 m/s |
| **Live 06Z T2m** (`n=275`) RMSE (Bias) | 1.8162 (+0.7369) | 1.8186 (+0.7392) | **1.7649** (+0.6402) | **-0.0513 K RMSE** (bias -0.097 K) |
| **Live 06Z MSLP** (`n=127`) / **W10** (`n=264`) | 1.5028 / 2.4063 | 1.5030 / 2.4086 | 1.5095 / 2.4117 (bias `+1.38→+1.36`) | Neutraal (`±0.006`), windbias licht lager |
| **6h NW-EU looptijd / RSS** | ~143 s · 553→845.9 MB | **127.8 s** · 553.8→846.3 MB | **137.0 s** · 553.5→846.4 MB | **0 MB** RAM-overhead |
| **Prod D2 (`137×400×520`) s/step · VRAM** | 2.587 s/step · 4623 MiB | **2.007 s/step** · 4623 MiB | **2.189 s/step** · 4623 MiB | **0 MiB** VRAM-overhead, **0%** vertraging |

*(Opmerking: `baseline_noda` werd na koude/warme overgang gemeten op `88.8 °C / 900 MHz`; alle drie D2-benchmarks hadden exact `4623 MiB` VRAM, geen snelheidsregressie, 3/3 Rust- en 49/49 pytest-tests geslaagd en bit-identieke uitvoer wanneer de vlag uit staat.)*

### 5.4 Korte Changelog
- `c6600f3` — **Fase 0:** `nh_cuda_clear_iau_forcing` VRAM-vrijgave (-440 MiB na warmup), 85% live VRAM-cap, sequentiële D2→D3 IC-levenscyclus, `clear_cfgrib_cache`, Bloom IAU single-domain fix, `--mem-profile-every`.
- `b94f34f` — **Fase 3 (#2):** Opt-in `--pbl-last` (IFS 50r1: verticale diffusie/PBL ná microfysica).
- `6d80672` — **Fase 3 (#3):** Opt-in `--cy46-physics` (HARMONIE Cy46 drukafhankelijk CDNC-profiel met `0.25×` op `k=0`, Nielsen Tabel-1 LW wolkenoptiek, sneeuwsmeltminimum `f_sn ≥ 0.25`, `XRIMAX = 0.2`).
- *Niet gemerged (teruggedraaid):* `--da-strict-timeslots` (gedocumenteerd in §5.1).

---

## 6. Concrete Aanbevelingen voor Vervolg (in prioriteitsvolgorde)

1. **`ic_builder.py` IC-klem (`1000 hPa`) & `q2m=0` oplossen:** Vervang `wt = np.clip(..., 0.0, 1.0)` in [`_vertical_map_isobaric_to_klemp`](file:///home/j_ord/AI/Tempest/Tempest-NWPv2-kopie/python/tempest_v2/ic_builder.py#L387-L398) door hydrostatische neerwaartse integratie vanaf `ps_child` en vul initiële `surf["q2m"]` uit ECMWF `2d` / `qv[0]`. Haalt direct de `-15.36 hPa` hogedrukbias op `T+00h` en `-11..-15 m` Z500-verzakking weg.
2. **Vocht-LBC (`d_lbc_qv`) in `nh_core.cu` en `LBCInterpolator`:** Schrijf Davies-relaxatie ook uit voor `qv` langs de domeinranden zodat een 6- tot 120-uurs D2/D3-run geen westelijke droge wand (`12.7 → 0.1 kg/m²`) opbouwt.
3. **DA `--da-strict-timeslots` herintroduceren ná fix #1:** Scheid de missing-epoch-guard (`YYYYMMDD_HHMMZ` dir-check + `±3h` venster) van variantie-inflatie in `bg_check` (toets outliers op ongeïnflateerde `sigma_o`), en vervang meervoudige Euler-FGAT-stappen door een 2e orde semi-Lagrangiaanse interpolatie over korte intervallen.
4. **Landwind & nachtelijke schuifmenging (`LFAKETREE` + `nh_pbl_momentum_*`):** `W10 land` (`+1.7..+2.3 m/s`) en te zwakke `200–300 m` LLJ vereisen onderzoek naar `km` in stabiele omstandigheden en logaritmische ruwheidsverhoging over laag grasland/akkerbouw.
5. **Shuell MSLP-reductie + `sea_ice_thick` + `50r2`-eenhedenguards:** Verkleint berg-MSLP-bias (`+1.6 hPa` boven `800 m`), activeert warmtegeleiding door zee-ijsdikte in `nh_physics.cu:1062` en beveiligt `tp`/`tcc`-readers voor ECMWF 50r2.
6. **Gerichte D3 (`1,5 km` of `2,5 km`) verificatie:** Nu Optie 3 (IC-clamp + `q2m` + `d_lbc_qv`) is opgelost, verifiëren tegen KNMI HARMONIE Cy46 en live `06Z`-stations.

---

## 7. Addendum — Optie 3 kernfixes, 2,5 km analyse, CPU FP64, versnelling & opslagformaten

### 7.1 Optie 3: Oplossen van kernbugs uit Fase 2 (`5527ebb`, `18ac6f6`)

Beide in Fase 2 blootgelegde kernfouten zijn opgelost en geverifieerd (Cargo `3/3`, pytest `53/53`, en echte 6-uurs Europa-D2 verificatie op `65×400×520`, `2026-11-22 00Z → 06Z`, `dt = 20 s`, `physics_every = 5`, `--pbl-last --cy46-physics`, gekoppelde NEMO+SI3):

1. **Bug #1 (`5527ebb`, [`ic_builder.py`](file:///home/j_ord/AI/Tempest/Tempest-NWPv2-kopie/python/tempest_v2/ic_builder.py)):**
   - Verving de `1000 hPa`-klem (`wt = np.clip(..., 0.0, 1.0)`) in `_vertical_map_isobaric_to_klemp` bij `z < z_1000` door vochtig-hydrostatische neerwaartse extrapolatie (`γ = 0.0065 K/m` met virtuele temperatuur), en verving `1004.64` door `CP_D`.
   - Initialisering van `surf["q2m"]` toegevoegd via `_q_from_dewpoint_pa` vanuit ECMWF `2d`/`d2m` (fallback `qv[0]`).
2. **Bug #2 (`18ac6f6`, [`nh_internal.cuh`](file:///home/j_ord/AI/Tempest/Tempest-NWPv2-kopie/crates/tempest-cuda/src/cuda/nh_internal.cuh), [`nh_core.cu`](file:///home/j_ord/AI/Tempest/Tempest-NWPv2-kopie/crates/tempest-cuda/src/cuda/nh_core.cu), [`cuda_bridge.py`](file:///home/j_ord/AI/Tempest/Tempest-NWPv2-kopie/python/tempest_v2/cuda_bridge.py), [`runner.py`](file:///home/j_ord/AI/Tempest/Tempest-NWPv2-kopie/python/tempest_v2/runner.py)):**
   - Open tracer-randfluxen toegevoegd voor `bc == 2` (`face_flux_x_f`/`y_f`, 1e-orde upwind aan de domeinbuitenzijde van Pass 1 consistent met Pass 2).
   - `d_lbc_qv` buffer (met backward-compatible `has_lbc_qv` vlag wanneer `qv == NULL`) en exponentieel positief-definiete Davies-relaxatiekernel `nh_lbc_relax_qv_kernel` toegevoegd na horizontale/verticale FCT-advectie van `qv`, plus VRAM-schatter `+1` veld (`D2 137×400×520` met IAU blijft **1×1 tile**: `4188 MiB <= 4350 MiB`).

| Metriek (`65×400×520`, `2026-11-22 00Z → 06Z`) | Vóór fixes (Fase 2 §4.4) | Ná Optie 3 (`18ac6f6`) | Verbetering |
|---|---|---|---|
| **T+00h EC MSLP** binnen (`n=172.800`) Bias / RMSE (`r`) | `-3.95` / `7.42 hPa` (`0.6921`) | **`+0.112` / `0.560 hPa` (`0.9979`)** | **13,2× lagere RMSE** |
| **T+00h EC MSLP** Benelux Bias / RMSE (`r`) | `-7.09` / `8.98 hPa` (`0.2771`) | **`-0.029` / `0.068 hPa` (`0.9967`)** | **132× lagere RMSE** |
| **T+00h EC MSLP** hogedruk `>1015 hPa` Bias / RMSE | `-8.10 hPa` (`-15.36 hPa` piek) | **`+0.248` / `0.755 hPa`** | Hogedruk-afknijping weg |
| **T+00h EC Td2m** binnen Bias / RMSE (`r`) | `-74.00` / `74.21 K` | **`+0.005` / `0.017 K` (`1.0000`)** | `q2m=0` volledig verholpen |
| **T+06h Westrand-TCWV** `i=[0,1,5,10,20,40]` `[kg/m²]` | `[0.10, 0.16, 0.54, 1.06, 2.13, 9.82]` | **`[13.22, 13.24, 13.36, 13.47, 13.31, 14.57]`** *(EC: `[12.60..14.13]`)* | **Droge westrand 100% weg** |
| **T+06h EC Td2m** binnen Bias / RMSE (`r`) | `-0.47` / `2.12 K` (`0.9302`) | **`-0.072` / `1.089 K` (`0.9802`)** | RMSE gehalveerd |
| **T+06h EC MSLP / Benelux** Bias / RMSE | `-1.08/1.63` · `-1.49/1.54 hPa` | **`-0.984/1.602` · `-1.418/1.467 hPa`** | Lager drukverloop |
| **T+06h EC T2m** (binnen / land / zee) RMSE | `1.14 / 1.42 / 0.65 K` | **`1.122 / 1.397 / 0.618 K`** *(bias `-0.16/-0.03/-0.32`)* | Verbeterd op alle maskers |
| **T+06h EC Z500 / Z850** Bias / RMSE | `-15.34/18.51` · `-11.74/14.39 m` | **`-13.73/17.27` · `-10.71/13.59 m`** | Rest (~`-11..-14 m`) door vaste `00Z` LBC's |
| **T+06h HARMONIE Cy46** `MSLP / T2m / Td2m` RMSE (Bias) | `1.210 (-1.12) / 1.173 (+0.11)` / — | **`1.061 (-0.96) / 1.149 (+0.056) / 1.179 (+0.020)`** | Sterke overeenkomst; restpunt: `W10 land +1.72 m/s` |
| **Geheugen tijdens 1080 stappen** (`RSS / Swap / VRAM`) | — | **`2524.8 MB` vlak · `0.0 MB` swap · `3077 MiB` vlak** | `dt_step` `0.52→2.32 s` puur thermisch (`88.3°C / 772 MHz`) |

---

### 7.2 D3 Domeinanalyse: 2,5 km vs. 1,5 km op RTX 3060 (6 GB VRAM)

Gemeten direct met [`cuda_bridge.py`](file:///home/j_ord/AI/Tempest/Tempest-NWPv2-kopie/python/tempest_v2/cuda_bridge.py) en volledige gekoppelde dynamica+fysica (`--pbl-last --cy46-physics`, `physics_every=5`) op de RTX 3060 Laptop (`live 85% budget = 4350 MiB` uit `5118 MiB` vrij):

| Configuratie | Rooster (`nz×ny×nx`) | Nesting-ratio t.o.v. D2 | Totaal VRAM (`+qv`) | Tiles (IAU / na IAU) | `dt` | Gemeten stapduur (warm, 10 stappen) | Geschatte 6u-run |
|---|---|---|---|---|---|---|---|
| **D3 `1,5 km` (`nz=137`)** *(huidig)* | `137×720×720` (`71,0M`) | `5:1` (groot) | `9976` (`10247`) MiB | **`2×2` (4 tiles)** | `4,0 s` (5400 st.) | `9,89 s/stap` *(eerder gemeten bij `86.7°C`, `~7,7 GB` PCIe/stap)* | **~14,8 uur** (tot `20–30u` thermisch) |
| **D3 `1,5 km` (`nz=90`)** | `90×720×720` (`46,7M`) | `5:1` | `6630` (`6808`) MiB | `1×2` (2 tiles) | `4,0 s` (5400 st.) | *(niet getimed; wel nog 2-tile PCIe-kopie)* | *(niet gemeten)* |
| **D3 `2,5 km` (`nz=137`)** | `137×432×432` (`25,6M`) | **`3:1`** *(ideaal)* | `3673` (`3771`) MiB | **`1×1` (1 tile!)** | **`6,0 s`** (3600 st.) | **`0,985 s/stap`** *(VRAM `4228 MiB`)* | **59,1 min** *(**~15× sneller**)* |
| **D3 `2,5 km` (`nz=90`)** | `90×432×432` (`16,8M`) | **`3:1`** *(ideaal)* | `2469` (`2533`) MiB | **`1×1` (1 tile!)** | **`6,0 s`** (3600 st.) | **`0,624 s/stap`** *(VRAM `3190 MiB`)* | **37,5 min** *(**~24× sneller**)* |
| **D2 `7,5 km` (`nz=137`)** | `137×400×520` (`28,5M`) | — | `4079` (`4188`) MiB | `1×1` (1 tile) | `20,0 s` (1080 st.) | **`1,128 s/stap`** *(VRAM `4602 MiB`)* | **20,3 min** (koud) → `~36 min` (thermisch) |

**Meteorologische conclusie:** `2,5 km` (`432×432`) dekt exact hetzelfde Benelux+W-DE gebied (`1080×1080 km`), matcht KNMI HARMONIE/UWC-West (`2,5 km`), geeft de klassieke `3:1` nesting-verhouding vanaf `7,5 km` D2, en past in **één enkele GPU-tile**. Alleen voor zeer kleinschalige convectieve cellen of steile dalwind in Alpen/Ardennen biedt `1,5 km` meer detail.

---

### 7.3 CPU FP64-prestatie-inschatting (`Intel Core i5-12500H`, 12 threads, `23 GiB` RAM)

Tempest-NWP v2 heeft momenteel geen CPU-dynamica-oplosser (`crates/tempest-core` bevat alleen constanten, geodesie en thermodynamica). Om een eerlijke CPU FP64-looptijd te bepalen zijn echte OpenMP-microbenchmarks (`gcc -O3 -fopenmp -march=native`) op deze exacte machine gedraaid op een `7,12M`-cellen grid (`137×200×260`, buiten 18 MB L3-cache):
- **Gemeten bandbreedte (`STREAM FP64 triad`):** **`17,34 GB/s`** (12 threads)
- **Gemeten 3D 7-punts+WENO stencil (`float64`):** `1,48 ns/cel`
- **Gemeten strided verticale Thomas-solver (`k`-lus over `[nz,ny,nx]`):** `2,35 ns/cel`

Rekenend met ~`5,5 KB/cel` aan geheugenverkeer per volledige RK3+HEVI+IEVA+FCT-stap in FP64 (en ~`280 B/cel` voor de ~35 actieve 3D `float64`-arrays + Python/statiek/NEMO overhead):

| Domein (`nz=137`) | Host-RAM nodig (FP64) | Past in `23 GiB` WSL-RAM? | Bandbreedte-ondergrens | Realistische CPU FP64-tijd | Schatting 6u / 120u |
|---|---|---|---|---|---|
| **D2 `7,5 km` (`400×520`)** | `~8,0 GB` arrays (`~10–12 GB` totaal) | Ja | `~9 s/stap` | **`~12–18 s/stap`** | **3,5–5,5 uur** (`6u`) · **~3–4,5 dagen** (`120u`) |
| **D3 `2,5 km` (`432×432`)** | `~7,2 GB` arrays (`~9–11 GB` totaal) | Ja | `~8 s/stap` | **`~11–16 s/stap`** | **11–16 uur** (`6u`) · ~`9–13 dagen` (`120u`) |
| **D3 `1,5 km` (`720×720`)** | `~19,9 GB` arrays (`~26–30 GB` totaal) | **Nee — gaat swappen!** | `~22,5 s/stap` *(zonder swap)* | **`~30–45 s/stap`** *(veel trager met swap)* | **45–68 uur** (`6u`, theoretisch zonder swap) |

CPU FP64 is dus ruwweg **10–16× trager** op D2/D3-2,5 km dan de (thermisch gelimiteerde) RTX 3060, en op D3 `1,5 km` loopt `23 GiB` RAM bovendien vol.

---

### 7.4 Gerangschikte GPU-versnellingskansen (zonder fysicaverlies)

1. **`2,5 km` 1-tile voor D3 (~15× sneller) en optioneel `nz = 90` (~1,58× extra):** Vermijdt de ~`7,7 GB/stap` CPU↔GPU PCIe-kopieën over 4 tiles plus ruimere `dt` (`6 s` vs `4 s`).
2. **`d_tri_c/d_tri_d` en `nh_ieva_vert_advect_*` van `double` naar `float`:** De RTX 3060 (GA106) heeft een **1:64 FP64/FP32-verhouding**. Deze velden omzetten naar `float` bespaart direct `228 MiB` VRAM op D2 (`137×400×520`) en versnelt de 3×3 IEVA- en 4 PBL-Thomas-sweeps merkbaar.
3. **IEVA global short-circuit als `max(|CFL_z|) <= 0.45`:** In [`nh_core.cu:2988-3003`](file:///home/j_ord/AI/Tempest/Tempest-NWPv2-kopie/crates/tempest-cuda/src/cuda/nh_core.cu#L2988) draaien nu bij elke RK-stage (3×/stap) alle kolomsweeps van `nh_ieva_vert_advect_{scalar,u,v}_kernel` de volledige scanlus. Met een simpele reductievlag kan dit volledig overgeslagen worden in rustige atmosferische kolommen.
4. **1× per stap `moisture_load` voorberekenen in `d_q_tmp`:** Nu leest [`moisture_load_scalar_f`](file:///home/j_ord/AI/Tempest/Tempest-NWPv2-kopie/crates/tempest-cuda/src/cuda/nh_core.cu#L114) tijdens ~6 akoestische substeps × 3 stages × (`u`, `v`, `vert_implicit`) alle 6 vochtarrays (~`10–12 GB/stap` onnodig VRAM-verkeer op D2).
5. **Pointer-swap in `advect_one_tracer` + skippen van lege hydrometeoren (`qr/qi/qs/qg`):** Verwijder de 6 D2D `cudaMemcpy`'s per stap en sla FCT over als `max(q) == 0`.
6. **Pinned host-buffers + async CUDA streams** voor 4-tile modus (alleen relevant voor `1,5 km`) en betere laptopkoeling (houdt klok op `1600–1890 MHz` i.p.v. `390–900 MHz` throttling rond `88–89°C` — ruim `2×` winst op D2).

---

### 7.5 Opslagformaten: WX-1 vs. NetCDF4 vs. Zarr v3 vs. GRIB2

Direct gemeten op hetzelfde echte D2-vochtige dataset (`4` 3D-velden `65×400×520` + `4` 2D-velden `400×520`, `219,65 MB` oncomprimeerde `float32`):

| Formaat & Chunk-topologie | Bestandsgrootte (Compressie) | Schrijftijd | Lees volledige 3D | Lees 1 horizontale kaart (`z=0`) | Lees 1 verticale kolom (`65` lagen) | Lossless? (`max_err`) |
|---|---|---|---|---|---|---|
| **WX-1 `zstd_shuffle` `[1,400,520]`** | `107,39 MB` (2,05×) | `2,17 s` | `0,098 s` | **`3,52 ms`** *(snelste)* | `29,47 ms` | **Ja (`0.0`)** |
| **WX-1 `zstd_shuffle` `[8,100,130]`** | `107,91 MB` (2,04×) | `2,13 s` | `0,053 s` | `7,16 ms` | **`4,57 ms`** *(snelste)* | **Ja (`0.0`)** |
| NetCDF4 `zlib=4` `[1,400,520]` | `105,72 MB` (2,08×) | `2,93 s` | `0,137 s` | `6,57 ms` | `106,17 ms` *(23× trager)* | Ja (`0.0`) |
| NetCDF4 `zlib=4` `[8,100,130]` | `104,96 MB` (2,09×) | `2,86 s` | `0,151 s` | `19,37 ms` | `10,58 ms` | Ja (`0.0`) |
| Zarr v3 `zstd` `[1,400,520]` | `141,80 MB` (1,55×) *(vele losse chunk-bestanden)* | **`0,266 s`** | **`0,049 s`** | `4,24 ms` | `39,50 ms` | Ja (`0.0`) |
| Zarr v3 `zstd` `[8,100,130]` | `140,70 MB` (1,56×) | `0,453 s` | `0,092 s` | `11,47 ms` | `7,74 ms` | Ja (`0.0`) |
| GRIB2 `grid_simple` 16-bit *(1→4 vars geschat)* | `~108,21 MB` | `~0,45 s` *(`0,114 s`/var)* | `~0,35 s` *(`0,088 s`/var)* | 1 bericht | Full scan! | **Nee (`4,88e-4`, lossy)** |
| GRIB2 `grid_ccsds` 16-bit *(1→4 vars geschat)* | **`~66,60 MB`** | `~0,76 s` *(`0,190 s`/var)* | `~0,80 s` *(`0,200 s`/var)* | 1 bericht | Full scan! | **Nee (`4,88e-4`, lossy)** |

**Conclusie (conform `AGENTS.md` Regel 2):** Vervang **WX-1 niet** als interne backbone. Voor prognostische checkpoints, LBC's en model-I/O scoort WX-1 het best over de hele linie: één enkel compact bestand, even groot als NetCDF4 (`-24%` kleiner dan standaard Zarr v3), sneller bij zowel horizontale kaarten (`3,52 ms`) als verticale soundings (`4,57 ms`) en 100% verliesvrij (`max_err = 0.0`) — iets waar 16-bit GRIB2 afvalt door kwantisatieverlies en gebrek aan kolom-chunking. `NetCDF4` of `GRIB2` heeft uitsluitend nut als *optionele eind-export* voor externe tools (Metview, Panoply, QGIS, CDO).

---

### 7.6 Compact Nederland-D3 (`d3_nl_2p5km`, `216×216`, `2,5 km`) timing & 6-uurs verificatie

1. **Reden om `432×432×137` Benelux+DE af te breken (`14b9be5`):** Tijdens de geneste 6-uurs test liep de stapduur op `432×432×137` bij aanhoudende thermische throttling (`88,4 °C / 787–855 MHz / thr=0x20`) op naar **`3,69–3,77 s/stap`** (~`35 min` per forecast-uur, oftewel ~`3,5 uur` alleen voor D3 bovenop `28,6 min` voor D2). Daarom is standaard `d3` ingericht als **compact Nederland + ~120 km buffer (`d3_nl_2p5km`: `216×216`, `540×540 km`, `lat 49.67..54.57°N, lon 1.14..9.46°E`, `4×` minder cellen)** in [`projection.py`](file:///home/j_ord/AI/Tempest/Tempest-NWPv2-kopie/python/tempest_v2/projection.py#L280) (`2f089b7`), met behoud van `d3_benelux_2p5km` (`432×432`) en `d3_1p5km` (`720×720`).

2. **Gemeten `d3_nl_2p5km` (`216×216`, `2,5 km`, `dt = 6,0 s`, 1 tile, `--pbl-last --cy46-physics`) prestaties:**

| Lagen (`nz`) | Schatting VRAM (met IAU) | Gemeten live VRAM | Tiles | 10-staps snelheid (koel) | Geschatte 6u (3600 st.) |
|---|---|---|---|---|---|
| `nz = 65` | `565 MiB` | `1436 MiB` (`1521 MiB` met NEMO+SI3) | `1×1` | **`0,151 s/stap`** | `9,1 min` koel · **`18,26 min` gemeten thermisch (`88.6°C / 472–900 MHz`)** |
| `nz = 90` | `729 MiB` | `1630 MiB` | `1×1` | **`0,255 s/stap`** | `15,3 min` koel *(~`30 min` thermisch)* |
| `nz = 137` | `1039 MiB` | `1894 MiB` | `1×1` | **`0,410 s/stap`** | `24,6 min` koel *(~`45–50 min` thermisch)* |

3. **Volledige 6-uurs `d3_nl_2p5km` run (`65×216×216`, `dt=6,0 s`, 3600 stappen, `2026-11-22 00Z → 06Z`, standalone met vaste `00Z`-LBC's):**
   - Looptijd: **`1095,49 s` (`18,26 min`)**, **`num_tiles = 1`**, `RSS = 1817,4 MB` (stabiel), `Swap = 0,0 MB`, `VRAM = 1521/6143 MiB` (vlak van stap 900 t/m 3600).
   - Outputbestanden: `runs/20261122_00Z_d3_nl_2p5km/forecast_d3_nl_2p5km_T+000h.wx1`, `T+003h.wx1`, `T+006h.wx1`.

| Referentie (`d3_nl_2p5km`, binnengebied `n=30.976`) | `T+00h` Bias / RMSE (`r`) | `T+06h` Bias / RMSE (`r`) | Duiding |
|---|---|---|---|
| **ECMWF MSLP** `[hPa]` | **`-0.016` / `0.061` (`0.9970`)** | `+2.440` / `2.626` (`0.4733`) | `00Z→06Z` daalt EC over NL (`1015.47 → 1012.84`); vaste `00Z`-LBC houdt domein op `1015.28 hPa` |
| **ECMWF T2m** (int / land / sea) `[K]` | `0.000 / 0.000` (`1.0000`) | **`+0.263 / 1.052` (`0.9671`)** · land `+0.361/1.222` · zee `+0.104/0.691` | Sterk thermisch profiel ondanks vaste LBC's |
| **ECMWF Td2m** `[K]` | **`+0.002` / `0.004` (`1.0000`)** | **`+0.116 / 1.052` (`0.8634`)** | Vocht-IC en dauwpunt zeer zuiver |
| **ECMWF W10** (land / sea) `[m/s]` | `+1.782/2.141` · `+0.536/0.862` | `+0.796/1.603` · `-1.662/2.598` | Vaste `00Z`-LBC's blokkeren aantrekkende zeewind in klein domein |
| **ECMWF Westrand-TCWV** `i=[0,1,5,10,20,40]` | `[9.06..8.49]` vs `[8.20..7.44]` | **`[9.06, 9.04, 8.89, 8.52, 8.40, 8.23]`** *(EC `[8.64..7.50]`)* | Geen droge rand dankzij `d_lbc_qv` |
| **HARMONIE Cy46 (`2,5 km`) MSLP** `[hPa]` | — | `+2.897` / `3.108` (`0.3879`) *(mod `1015.28` vs `1012.39`)* | Zelfde synoptische `00Z`-LBC oorzaak (`--mode d2_to_d3` lost dit op via tijdvariërende D2-LBC's) |
| **HARMONIE Cy46 (`2,5 km`) T2m / Td2m** `[K]` | — | **`+0.534 / 1.430` (`0.9472`)** *(zee `+0.149/0.858`)* · Td2m **`+0.116 / 1.413`** | Goed 2,5 km thermisch en vochtgedrag |
| **HARMONIE Cy46 (`2,5 km`) W10** (land / zee) `[m/s]` | — | land `+0.988/1.697` · zee `-2.804/3.575` | Zelfde `00Z`-LBC zeewindblokkade bij standalone run zonder D2-koppeling |


