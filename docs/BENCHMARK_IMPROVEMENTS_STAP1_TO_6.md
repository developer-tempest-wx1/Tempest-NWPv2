# Tempest-NWP v2 — Overzicht van Opgeloste Problemen en Oude vs. Nieuwe Biases (Stap 1 t/m 6)

**Bijgewerkt:** 2026-11-26  
**Repository:** `/home/j_ord/AI/Tempest/Tempest-NWPv2-kopie` (branch `master`)  
**Referentie-baseline (ongewijzigd):** `runs/era5_benchmark/` (20 cases, 24h, KNMI LAND-stations)

## Statusoverzicht

* **Stappen 1 t/m 6** zijn volledig afgerond en gecommit (`866da61`, `3116f5e`, `30de140`, `86f1f9f`, `945e9ae`, `c9f114c`, `486a6f8`).
* **Stap 6** (`486a6f8`: microfysica, neerslag, CAPE, onweer/bliksem en ijzel) is volledig geïmplementeerd, opnieuw gecompileerd (CUDA + Rust release build), geslaagd voor de volledige testsuite (**121/121 passed** in `pytest`) en geverifieerd op alle drie de 12-uurs live verificatiecases (`deep_conv_hail_supercell`, `winter_storm_henk` v2 en `freezing_rain_ijzel` v2).

> **Belangrijke leeswijzer (Apples-to-Apples):** De oude 20-case baseline-scorecard (`benchmark_scorecard.json`) middelt over alle 20 cases van elk 24 uur. Alle onderstaande **"Oud → Nieuw"-vergelijkingen per stap** zijn daarentegen exact *apples-to-apples* berekend over precies dezelfde cases en dezelfde lead-uren (`T+01..T+06` voor Stap 2, `T+01..T+12` voor Stappen 3–6) op KNMI LAND-stations uit `knmi_verification_summary.json`. De nieuwe globale 20-case 24h-biases volgen pas na de volledige 20-case eindbenchmark. Alle fysicawijzigingen volgen strikt klassieke deterministische fysica in SI-eenheden (0% AI / 0% ML).

---

## 1. Oude 20-Case Baseline Scorecard (`runs/era5_benchmark/`, 24h LAND)

De oorspronkelijke 20-case benchmark (`n_cases = 20`) toonde de volgende hoofdfouten die als leidraad dienden voor Stappen 1 t/m 6:

### Continue variabelen (20 cases, 24h LAND)

| Variabele | Oude Bias (95%-BI) | Oude RMSE (95%-BI) | Belangrijkste diagnose |
| :--- | :--- | :--- | :--- |
| `T2m` (°C) | `+0.171` (`[-0.097, +0.458]`) | `1.778` (`[1.423, 2.135]`) | Dag `+0.543 °C` (te warm), nacht `-0.163 °C`; grote fouten bij stralingsmist (`+2.32 °C`) |
| `Td2m` (°C) | `-0.225` (`[-0.590, +0.164]`) | `1.649` (`[1.334, 1.948]`) | Grenslaag overdag/bij mist te droog |
| `RH` (%) | `-1.114` (`[-3.248, +1.063]`) | — | Sterke droge bias (`-7` tot `-8%`) in mistcases |
| `MSLP` (hPa) | `-0.466` (`[-0.615, -0.332]`) | `0.796` (`[0.658, 0.927]`) | Systematisch negatief in 19/20 cases (diabatische Exner/hydrostatische onbalans) |
| `Wspd10` (m/s) | `+0.161` (`[+0.032, +0.287]`) | `1.909` (`[1.681, 2.129]`) | Lichte overschatting en ruwheidskoppeling |
| `Gust10` (m/s) | `+1.442` (`[+0.835, +2.120]`) | `3.440` (`[2.739, 4.154]`) | Onvoorwaardelijke convective gust-term (`+0.15*sqrt(CAPE_sb)`) zonder actieve bui |
| `SW_down` (W/m²) | `+17.183` (`[+7.680, +28.529]`) | `97.479` (`[68.971, 124.402]`) | Te veel inkomend zonlicht door ontbrekende subgrid-bewolking |
| `TCC` (0–1) | `-0.300` (`[-0.381, -0.224]`) | `0.509` (`[0.440, 0.577]`) | Mist en lage stratus werden volledig gemist door 0/1-wolkenfractie |
| `Precip` (mm/h) | `-0.056` (`[-0.136, +0.022]`) | `1.439` (`[0.849, 2.037]`) | Dode autoconversie in `cy46_physics=1`, te weinig lichte/matige neerslag |
| Bovenlucht vs ERA5 | — | `T850`: `0.678 K` / `GH500`: `5.485 m` / `Wspd850`: `1.647 m/s` | Op basis van ERA5 upper-air scorecard |

### Weersverschijnselen (20 cases gepoold, 24h LAND)

| Event | N | Obs | Mod | Hits | FA | POD | FAR | CSI | Freq Bias (FB) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fog` | 11,265 | 953 | 98 | 36 | 62 | 3.8% | 63.3% | 3.5% | 0.10 |
| `rain_0p1mm` | 14,628 | 4,081 | 2,823 | 2,124 | 699 | 52.0% | 24.8% | 44.4% | 0.69 |
| `rain_2mm` | 14,628 | 543 | 389 | 133 | 256 | 24.5% | 65.8% | 16.6% | 0.72 |
| `gust_15ms` | 21,152 | 4,295 | 5,654 | 3,908 | 1,746 | 91.0% | 30.9% | 64.7% | 1.32 |
| `gust_20ms` | 21,152 | 2,014 | 2,378 | 1,755 | 623 | 87.1% | 26.2% | 66.6% | 1.18 |
| `snow` | 11,253 | 1,217 | 903 | 798 | 105 | 65.6% | 11.6% | 60.4% | 0.74 |
| `thunder` | 11,074 | 487 | 61 | 24 | 37 | 4.9% | 60.7% | 4.6% | 0.13 |
| `thunder_25km` | 11,074 | 487 | 516 | 132 | 384 | 27.1% | 74.4% | 15.2% | 1.06 |
| `ice_freezing_rain` | 11,253 | 134 | 19 | 15 | 4 | 11.2% | 21.1% | 10.9% | 0.14 |

---

## 2. Opgeloste Problemen en Verificatie per Stap

### Voorbereiding (`866da61`): Verificatie-uitlijning & Scorecard
* **Wat is opgelost:** KNMI 1-uurs accumulatie-tijdstempels en `SW_down`-uitlijning gecorrigeerd, mist/ijzel/onweer-verificatie toegevoegd, ERA5-cache-serialisatie robuust gemaakt en de gepoolde multi-season scorecard gebouwd.

---

### Stap 1 (`3116f5e`): Integratie ESA WorldCover 10 m, SoilGrids 250 m & Copernicus DEM
* **Wat is opgelost:**
  * **ESA WorldCover 2021 (v200, 10 m):** hoogresolutie landgebruik en vegetatiefractie gekoppeld aan de fysieke oppervlakteklassen.
  * **SoilGrids 250 m (55 lagen):** fysieke bodemtextuur (zand/klei/leem/silt/organisch koolstof/bulk-dichtheid) inclusief **veenklasse 14** gekoppeld aan het Land Surface Model (LSM).
  * **SMI bodemvocht-remapping:** Soil Moisture Index correct herverdeeld van ERA5-Land naar SoilGrids-hydraulische parameters (verwelkingspunt en veldcapaciteit).
  * **Copernicus DEM 30 m / 90 m → 250 m:** subgrid-orografie (SSO: standaarddeviatie, helling, oriëntatie) met pre-filtering geïntegreerd.
* **Verificatiestatus:** Op verzoek geen aparte snelle tussenbenchmark gedraaid voor alleen Stap 1; deze statische velden zijn actief in alle verificatieruns vanaf Stap 2.

---

### Stap 2 (`30de140`): Systematische MSLP- & GH500-bias
* **Oorzaak:** De thermische diabatische tendensen gebruikten niet-consistente druk-/Exner-koppeling en een kleine hydrostatische onbalans in IC/LBC.
* **Oplossing:** Compressibele diabatische Exner-koppeling toegevoegd en hydrostatische consistentie in de IC/LBC-opbouw hersteld.
* **Verificatie — `heavy_rain_limburg_20210714_00Z` (`T+01..T+06` LAND):**

| Metriek | Oud (`T+01..T+06`) | Nieuw (`T+01..T+06`) | Verschil |
| :--- | :--- | :--- | :--- |
| `MSLP` bias | `-0.618 hPa` | `-0.260 hPa` | `-0.358 hPa` minder bias (57.9% dichter bij 0) |
| `MSLP` RMSE | `0.712 hPa` | `0.387 hPa` | `-45.6%` RMSE |
| `T2m` bias / RMSE | `-0.881` / `1.020 °C` | `-0.807` / `0.961 °C` | `-0.059 °C` RMSE |
| `Td2m` bias / RMSE | `-0.988` / `1.105 °C` | `-0.961` / `1.082 °C` | `-0.023 °C` RMSE |
| `Gust10` bias / RMSE | `+2.096` / `2.895 m/s` | `+1.817` / `2.629 m/s` | `-0.266 m/s` RMSE |
| `SW_down` bias / RMSE | `-0.632` / `13.733 W/m²` | `+0.134` / `12.717 W/m²` | `-1.016 W/m²` RMSE |
| `Precip` bias / RMSE | `+0.157` / `1.362 mm/h` | `+0.170` / `1.609 mm/h` | `obs=0.074`, `mod: 0.231 → 0.245 mm/h` |
| Bovenlucht vs ERA5 (nieuw) | *(n.v.t. in oude summary)* | `GH500`: `-0.742 m` (RMSE `6.827 m`) / `T850`: `+0.288 K` (RMSE `0.644 K`) | — |

---

### Stap 3 (`86f1f9f`): Subgrid-bewolking (`TCC`) & Kortgolvige straling (`SW_down`)
* **Oorzaak:** Binaire (0/1) roostercel-wolken misten dunne lage stratus, mist en subgrid-bewolking (`TCC` bias `-0.300`, tot `-0.951` bij wintermist), wat leidde tot te veel `SW_down` (`+17.2 W/m²`).
* **Oplossing:** Sundqvist subgrid-wolkenfractie, ecRad Tripleclouds SW/LW-overlap (met fractale wolkinhomogeniteit) en 3-bands clear-sky SW-optica ingebouwd.
* **Verificatie — `radiation_fog_winter_20241227_00Z` (`T+01..T+12` LAND):**

| Metriek | Oud (`T+01..T+12`) | Nieuw Stap 3 (`T+01..T+12`) | Verschil |
| :--- | :--- | :--- | :--- |
| `TCC` bias / RMSE | `-0.951` / `0.956` | `-0.211` / `0.410` | `-57.1%` RMSE, bias `-77.8%` kleiner |
| `SW_down` bias / RMSE | `+22.000` / `47.579 W/m²` | `+8.406` / `29.059 W/m²` | `-38.9%` RMSE |
| `T2m` bias / RMSE | `+0.274` / `2.036 °C` | `+0.264` / `1.663 °C` | `-0.373 °C` RMSE |
| `Td2m` bias / RMSE | `-0.941` / `1.606 °C` | `-0.429` / `1.487 °C` | bias gehalveerd |
| `RH` bias / RMSE | `-7.310` / `10.912 %` | `-4.002` / `8.065 %` | `-2.847%` RMSE |
| `MSLP` bias / RMSE | `-0.207` / `0.311 hPa` | `+0.097` / `0.233 hPa` | bias rond 0 |
| `Fog` (`obs=267`) | H `22`, FA `0`, POD `8.2%`, CSI `8.2%`, FB `0.08` | H `107`, FA `6`, POD `40.1%`, FAR `5.3%`, CSI `39.2%`, FB `0.42` | CSI `+31.0 pp` |

---

### Stap 4 (`945e9ae`): Stabiele Grenslaag (SBL), LSM, Sneeuwisolatie, 2m-Profiel & Mist
* **Oorzaak:** De onderste modellagen begonnen zonder oppervlakte-verankering met een kunstmatige warme luchtbel boven koude stralingsmist/nachten; daarnaast misten stabiel/onstabiel onderscheid in huidgeleiding (`lambda_sk`), seriële sneeuwisolatie, winterrust in huidweerstand en stabiele Monin-Obukhov/TR-302 2m-interpolatie.
* **Oplossing:**
  1. Oppervlakte-verankerde SBL IC & LBC (onder eerste drukniveau > `hs + 15 m`) met 3-pass discrete hydrostatische balans.
  2. LSM-upgrades: ecLand stabiele vs. onstabiele huidgeleiding `lambda_sk`, seriële sneeuwpakket-isolatie + ijsverzadiging over koude sneeuw, koude/bevroren-bodem en Jarvis-winterrustdemping, vegetatie-afhankelijke thermische ruwheid `z0h`, en ECMWF Cy49r1 / KNMI TR-302 verplaatste 2m/10m-interpolatie.
  3. Deterministische WMO zicht- & mistdiagnostiek met Europese Köhler-haze, `k=1` ondiepe opgetilde mist (`z1 <= 55 m`), en begrensde isothermische stralingsmist zonder winderige/sneeuw-valse alarmen.

#### Verificatie 4a — `radiation_fog_winter_20241227_00Z` (`T+01..T+12` LAND)
| Metriek | Oud (`T+01..T+12`) | Nieuw Stap 4 (`T+01..T+12`) | Verschil |
| :--- | :--- | :--- | :--- |
| `T2m` bias / RMSE | `+0.274` / `2.036 °C` | `+0.094` / `1.461 °C` | `-28.2%` RMSE |
| `Td2m` bias / RMSE | `-0.941` / `1.606 °C` | `-0.510` / `1.399 °C` | `-0.207 °C` RMSE |
| `RH` bias / RMSE | `-7.310` / `10.912 %` | `-3.377` / `8.216 %` | `-2.696%` RMSE |
| `MSLP` bias / RMSE | `-0.207` / `0.311 hPa` | `+0.078` / `0.182 hPa` | `-0.129 hPa` RMSE |
| `SW_down` bias / RMSE | `+22.000` / `47.579 W/m²` | `+7.109` / `27.086 W/m²` | `-20.493 W/m²` RMSE |
| `TCC` bias / RMSE | `-0.951` / `0.956` | `-0.209` / `0.408` | `-0.548` RMSE |
| `Fog` (`obs=267`) | H `22`, FA `0`, POD `8.2%`, CSI `8.2%`, FB `0.08` | H `183`, FA `14`, POD `68.5%`, FAR `7.1%`, CSI `65.1%`, FB `0.74` | CSI `8.2% → 65.1%` (`+56.9 pp`) |

#### Verificatie 4b — `radiation_fog_autumn_20151102_00Z` (`T+01..T+12` LAND)
| Metriek | Oud (`T+01..T+12`) | Nieuw Stap 4 (`T+01..T+12`) | Verschil |
| :--- | :--- | :--- | :--- |
| `T2m` bias / RMSE | `+2.324` / `3.125 °C` | `+0.379` / `1.439 °C` | `-54.0%` RMSE, warme mistbias van `+2.32 → +0.38 °C` |
| `Td2m` bias / RMSE | `+0.941` / `1.996 °C` | `-0.325` / `1.808 °C` | `-0.188 °C` RMSE |
| `RH` bias / RMSE | `-8.145` / `11.056 %` | `-4.142` / `7.581 %` | `-3.475%` RMSE |
| `Wspd10` bias / RMSE | `+0.199` / `1.057 m/s` | `-0.107` / `0.898 m/s` | `-0.159 m/s` RMSE |
| `Gust10` bias / RMSE | `+0.128` / `1.310 m/s` | `+0.039` / `1.110 m/s` | `-0.200 m/s` RMSE |
| `SW_down` bias / RMSE | `+22.910` / `51.804 W/m²` | `-6.169` / `27.012 W/m²` | `-24.792 W/m²` RMSE |
| `TCC` bias / RMSE | `-0.214` / `0.381` | `+0.604` / `0.682` | *(aandachtspunt: overbewolking bij ondiepe herfstmist)* |
| `Fog` (`obs=231`) | H `15`, FA `0`, POD `6.5%`, CSI `6.5%`, FB `0.06` | H `163`, FA `0`, POD `70.6%`, FAR `0.0%`, CSI `70.6%`, FB `0.71` | CSI `6.5% → 70.6%` (`0` valse alarmen) |

---

### Stap 5 (`c9f114c`): Windstoten (`Gust10`) & SPPGUST Buoyancy-Schaling
* **Oorzaak:** De windstootdiagnostiek telde onvoorwaardelijk `+0.15 * sqrt(CAPE_sb)` op (ook bij droge warmte of lichte buien) en had op `T+00` een `ustar = 0`-bug bij koude start, wat zorgde voor een systematische `+1.44 m/s` `Gust10`-bias en overschatting van `15/20 m/s`-stoten.
* **Oplossing:**
  1. Vervanging door actief-convectieve Bechtold & Bidlot (2009) downdraft + 500–1500 m shear-transport, strikt gegate op actieve `precip_rate` / laagniveau `(qr + qg)`.
  2. Deling-vrije ECMWF IFS Cy48r1+ `SPPGUST` onstabiele buoyancy-schaling: `u_eff = cbrt(u*^3 + (0.5/12)*kappa*z_i*max(B_0, 0))`.
  3. `T+00` koude-start nul-`ustar`-bug verholpen met neutrale WMO open-terrein fallback (`0.0688 * wspd10`).
* **Verificatie — `summer_storm_poly_20230705_00Z` (`T+01..T+12` LAND):**

| Metriek | Oud (`T+01..T+12`) | Nieuw Stap 5 (`T+01..T+12`) | Verschil |
| :--- | :--- | :--- | :--- |
| `Gust10` bias / RMSE | `+2.553` / `3.954 m/s` | `+0.942` / `3.116 m/s` | `-1.61 m/s` bias, `-21.2%` RMSE |
| `Wspd10` bias / RMSE | `+0.656` / `2.685 m/s` | `+0.162` / `2.565 m/s` | bias bijna gehalveerd tot `+0.16 m/s` |
| `MSLP` bias / RMSE | `-0.857` / `1.186 hPa` | `-0.084` / `0.816 hPa` | `-0.370 hPa` RMSE |
| `TCC` bias / RMSE | `-0.282` / `0.501` | `-0.046` / `0.203` | `-0.298` RMSE |
| `SW_down` bias / RMSE | `+19.704` / `144.820 W/m²` | `-24.273` / `110.081 W/m²` | `-34.739 W/m²` RMSE |
| `T2m` bias / RMSE | `+0.564` / `1.193 °C` | `+0.684` / `1.236 °C` | vrijwel gelijk |
| `gust_15ms` (`obs=417`) | H `415`, FA `99`, FAR `19.3%`, CSI `80.4%`, FB `1.23` | H `407`, FA `76`, FAR `15.7%`, CSI `82.6%`, FB `1.16` | CSI `+2.2 pp`, FA `99 → 76` |
| `gust_20ms` (`obs=176`) | H `153`, FA `129`, FAR `45.7%`, CSI `50.2%`, FB `1.60` | H `127`, FA `52`, FAR `29.1%`, CSI `55.7%`, FB `1.02` | CSI `+5.5 pp`, FB `1.60 → 1.02`, FA `-60%` |
| `rain_0p1mm` (`obs=222`) | H `131`, CSI `56.5%`, FB `0.64` | H `136`, CSI `58.1%`, FB `0.67` | `+1.6 pp` CSI |

---

### Stap 6 (`486a6f8`, 121/121 pytest geslaagd, alle 3 live 12h cases afgerond): Microfysica, Neerslag, CAPE, Onweer/Bliksem & Ijzel
* **Oorzaken in oude model (`runs/era5_benchmark`):**
  1. Warme-regen autoconversie in `cy46_physics=1` was factor 1.000–10.000× te zwak, waardoor lichte/matige buien en frontale regen zwaar werden onderschat (`rain_0p1mm FB = 0.69`, `obs=0.286` vs `mod=0.209 mm/h`).
  2. Spurious extra `rho` factor in `qi → qs` aggregatie.
  3. Sneeuw- en graupelsmelt keek naar droge-boltemperatuur (`T > 0 °C`) i.p.v. psychrometrische natte-boltemperatuur (`T_w`).
  4. `_moist_adiabat_t_from_theta_e` had maar 2 iteraties (tot `+33.77 K` fout bij 900 hPa), waardoor `CAPE_mu` 2.2–3.0× werd opgeblazen tot `12,799 J/kg`.
  5. Er was nog geen prognostische bliksemproductie in de GPU-microfysica, en de 25 km-onweerdiagnostiek combineerde niet-gecolokaliseerde buurtmaxima (`thunder_25km FAR = 74.4%`).
* **Oplossingen:**
  1. Warme-regen botsings-coalescentie autoconversie (`Tripoli & Cotton 1980`, `r_c0 = 8.0 µm` met fysische land/zee `n_drop`) hersteld in `nh_physics.cu` met expliciete mistbescherming.
  2. Extra `rho` uit `qi → qs` verwijderd; psychrometrische natte-bolsmelt (`use_wb`) voor sneeuw en graupel; contactbevriezing (`sx2x[2][5]`) weer strikt op `dtemp > 4.0 K` gezet om onderkoelde ijzel (`-2.5 °C`) niet voortijdig in korrelsneeuw te bevriezen.
  3. Prognostische McCaul/Lopez oplaadzone-bliksem (`lightning_acc` [flashes/km²]) uit verticale snelheid `w` en graupel/ijs in `nh_microphysics_kernel`, plus `LPI`, `GWP` en `qg`-opname in `IWP`.
  4. Exacte 5-staps bisectie + 4-staps Newton voor `_moist_adiabat_t_from_theta_e` in `diagnostics.py`.
  5. Gecolokaliseerde onweerdiagnostiek in `knmi_verify.py`: `lt >= 0.01` OF gecolokaliseerde cel (`graupel >= 0.05 mm` & `CAPE_mu >= 50 J/kg`, of `CAPE_mu >= 300` & `IWP >= 0.25` & `precip >= 1.5 mm`), wat winter-korrelhagel zonder convectie uitsluit.

#### Verificatie 6a — `winter_storm_henk_20240102_00Z` (`T+01..T+12` LAND, **v2 definitief**, `runs/step6_verify_12h_v2`)
| Metriek | Oud (`T+01..T+12`) | Nieuw Stap 6 v2 (`T+01..T+12`) | Verschil |
| :--- | :--- | :--- | :--- |
| `rain_0p1mm` (`obs=255`) | H `169`, FA `9`, POD `66.3%`, FAR `5.1%`, CSI `64.0%`, FB `0.70` | H `195`, FA `17`, POD `76.5%`, FAR `8.0%`, CSI `71.7%`, FB `0.83` | CSI `+7.7 pp`, +26 hits, FB `0.70 → 0.83` |
| `rain_2mm` (`obs=62`) | H `36`, FA `9`, POD `58.1%`, CSI `50.7%`, FB `0.73` | H `35`, FA `10`, POD `56.5%`, CSI `48.6%`, FB `0.73` | gelijkwaardig |
| `Precip` bias / RMSE | `-0.225` / `0.603 mm/h` (`mod 0.622`) | `-0.189` / `0.576 mm/h` (`mod 0.659`, `obs 0.848`) | minder droge neerslagbias & lagere RMSE |
| `Gust10` bias / RMSE | `+1.092` / `2.385 m/s` | `+0.350` / `2.072 m/s` | `-0.742 m/s` bias, `-0.313 m/s` RMSE |
| `gust_15ms` (`obs=100`) | H `77`, FA `60`, FAR `43.8%`, CSI `48.1%`, FB `1.37` | H `63`, FA `31`, FAR `33.0%`, CSI `48.1%`, FB `0.94` | FA bijna gehalveerd (`60 → 31`), FB `0.94` |
| `Wspd10` bias / RMSE | `+0.381` / `1.617 m/s` | `+0.005` / `1.472 m/s` | windbias vrijwel `0.00 m/s` |
| `T2m` bias / RMSE | `-0.190` / `0.534 °C` | `-0.060` / `0.516 °C` | `-0.130 °C` biasverbetering |
| `TCC` bias / RMSE | `-0.018` / `0.082` | `+0.001` / `0.014` | vrijwel exact |
| `MSLP` bias / RMSE | `-0.115` / `0.517 hPa` | `+0.385` / `0.689 hPa` | *(aandachtspunt: lichte +bias bij diepe storm)* |
| Bovenlucht vs ERA5 (`T+01..T+12`) | — | `T850`: `-0.103 K` (`0.590 K`), `GH500`: `-0.967 m` (`4.770 m`), `Wspd850`: `-0.391 m/s` (`1.652 m/s`) | scherp bovenluchtprofiel |

#### Verificatie 6b — `deep_conv_hail_supercell_20160623_00Z` (`T+01..T+12` LAND, herverifieerd met definitieve `knmi_verify.py`, `runs/step6_verify_12h`)
| Metriek | Oud (`T+01..T+12`) | Nieuw Stap 6 (`T+01..T+12`) | Verschil |
| :--- | :--- | :--- | :--- |
| `CAPE_mu_max` (LAND) | `5456.4 J/kg` *(24h baseline: `10696.8`)* | `2441.2 J/kg` | fysisch realistische CAPE |
| `thunder_25km` (`obs=71`) | H `11`, FA `15`, POD `15.5%`, FAR `57.7%`, CSI `12.8%`, FB `0.37` | H `14`, FA `13`, POD `19.7%`, FAR `48.1%`, CSI `16.7%`, FB `0.38` | +3 hits, FAR `-9.6 pp`, CSI `+3.9 pp` |
| `rain_0p1mm` (`obs=122`) | H `18`, FA `5`, FAR `21.7%`, CSI `14.2%` | H `18`, FA `3`, FAR `14.3%`, CSI `14.4%` | lagere FAR |
| `Gust10` bias / RMSE | `+3.501` / `5.386 m/s` | `-1.092` / `3.542 m/s` | `-1.844 m/s` RMSE, `gust_15ms` FA `27 → 0` |
| `T2m` bias / RMSE | `+0.726` / `2.171 °C` | `+0.474` / `1.754 °C` | `-0.417 °C` RMSE |
| `MSLP` bias / RMSE | `-1.013` / `1.304 hPa` | `-0.528` / `0.825 hPa` | `-0.479 hPa` RMSE |
| `SW_down` bias / RMSE | `+75.417` / `168.594 W/m²` | `+18.434` / `118.986 W/m²` | `-56.98 W/m²` bias, `-49.61 W/m²` RMSE |
| `TCC` bias / RMSE | `-0.564` / `0.718` | `-0.017` / `0.196` | bias vrijwel verdwenen |
| `Fog` (`obs=3`) | H `0`, FA `0` | H `0`, FA `14` | *(aandachtspunt: mist-FA bij vochtige zomergrond)* |

#### Verificatie 6c — `freezing_rain_ijzel_20160105_00Z` (`T+01..T+12` LAND, **v2 definitief**, `runs/step6_verify_12h_v2`)
| Metriek | Oud (`T+01..T+12`) | Nieuw Stap 6 v2 (`T+01..T+12`) | Verschil |
| :--- | :--- | :--- | :--- |
| `T2m` bias / RMSE | `-1.011` / `1.984 °C` | `-0.196` / `1.424 °C` | `-28.2%` RMSE (`+0.815 °C` biasverbetering) |
| `Td2m` bias / RMSE | `-0.715` / `1.429 °C` | `-0.319` / `1.140 °C` | `-0.289 °C` RMSE |
| `RH` bias / RMSE | `+1.936` / `5.060 %` | `-0.748` / `4.304 %` | `-0.756%` RMSE |
| `MSLP` bias / RMSE | `-0.553` / `0.639 hPa` | `-0.267` / `0.378 hPa` | `-40.8%` RMSE |
| `SW_down` bias / RMSE | `+26.364` / `57.409 W/m²` | `+1.951` / `12.412 W/m²` | `-78.4%` RMSE, bias vrijwel `0` |
| `TCC` bias / RMSE | `-0.730` / `0.811` | `+0.007` / `0.165` | bias vrijwel verdwenen |
| `Precip` bias / RMSE | `+0.027` / `0.208 mm/h` (`mod 0.040`) | `+0.024` / `0.207 mm/h` (`mod 0.037`, `obs 0.013`) | gelijkwaardig |
| `thunder_25km` (`obs=0`) | Mod `2`, FA `2` (`FAR=100%`) | Mod `0`, FA `0` (`NEW=None`) | **alle winter-onweer-FA geëlimineerd** |
| `Fog` (`obs=2`) | H `0`, FA `4` | H `0`, FA `0` | mist-FA `4 → 0` |
| `ice_freezing_rain` (`obs=30`) | H `13`, M `17`, FA `1`, POD `43.3%`, FAR `7.1%`, CSI `41.9%`, FB `0.47` | H `9`, M `21`, FA `0`, POD `30.0%`, FAR `0.0%`, CSI `30.0%`, FB `0.30` | `FAR 7.1% → 0.0%`, 4 hits nipt gemist *(zie diagnose)* |

* **Diagnose van de 4 gemiste `ice_freezing_rain`-hits (`H 13 → 9`):** Het herstel van contactbevriezing op `dtemp > 4.0 K` verhinderde effectief het onterecht bevriezen van regen tot korrelsneeuw op Eelde (`06280`: `G = 0.000`) en Nieuw Beerta (`06286`: `G = 0.114 → 0.077`, gelijk aan de baseline `0.078`). De 4 gemiste uren op Eelde (`T+01`, `T+05..T+07`) hadden wél onderkoelde motregen (`R = 0.040–0.049 mm/h`, net onder de ongewijzigde `0.050 mm`-verificatiedrempel), terwijl een kleine noordoostwaartse verschuiving van de neerslagband meer regen naar Nieuw Beerta verplaatste (`0.327 → 0.453 mm/h` op `T+06` en `0.235 → 0.383 mm/h` op `T+07`).

---

## 3. Geconsolideerde Vergelijkingstabel (Oud → Nieuw over alle verificatiecases)

| Case (Venster LAND) | Stap | `T2m` bias (RMSE) °C | `MSLP` bias (RMSE) hPa | `Wspd10` bias (RMSE) m/s | `Gust10` bias (RMSE) m/s | `SW_down` bias (RMSE) W/m² | `TCC` bias (RMSE) | Kern-event CSI (Oud → Nieuw) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `heavy_rain_limburg` (6h) | 2 | `-0.88→-0.81` (`1.02→0.96`) | `-0.62→-0.26` (`0.71→0.39`) | `-0.05→-0.33` (`1.29→1.25`) | `+2.10→+1.82` (`2.90→2.63`) | `-0.6→+0.1` (`13.7→12.7`) | `-0.22→-0.23` (`0.38→0.38`) | `rain_0p1mm`: `21.1% → 14.3%` *(voor Stap 6)* |
| `radiation_fog_winter` (12h) | 3 | `+0.27→+0.26` (`2.04→1.66`) | `-0.21→+0.10` (`0.31→0.23`) | `-0.19→-0.47` (`0.85→0.93`) | `-1.12→-1.15` (`1.60→1.55`) | `+22.0→+8.4` (`47.6→29.1`) | `-0.95→-0.21` (`0.96→0.41`) | `fog`: `8.2% → 39.2%` |
| `radiation_fog_winter` (12h) | 4 | `+0.27→+0.09` (`2.04→1.46`) | `-0.21→+0.08` (`0.31→0.18`) | `-0.19→-0.39` (`0.85→0.84`) | `-1.12→-1.00` (`1.60→1.38`) | `+22.0→+7.1` (`47.6→27.1`) | `-0.95→-0.21` (`0.96→0.41`) | `fog`: `8.2% → 65.1%` |
| `radiation_fog_autumn` (12h) | 4 | `+2.32→+0.38` (`3.13→1.44`) | `-0.24→+0.13` (`0.43→0.43`) | `+0.20→-0.11` (`1.06→0.90`) | `+0.13→+0.04` (`1.31→1.11`) | `+22.9→-6.2` (`51.8→27.0`) | `-0.21→+0.60` (`0.38→0.68`) | `fog`: `6.5% → 70.6%` (`FA=0`) |
| `summer_storm_poly` (12h) | 5 | `+0.56→+0.68` (`1.19→1.24`) | `-0.86→-0.08` (`1.19→0.82`) | `+0.66→+0.16` (`2.69→2.57`) | `+2.55→+0.94` (`3.95→3.12`) | `+19.7→-24.3` (`144.8→110.1`) | `-0.28→-0.05` (`0.50→0.20`) | `gust_20ms`: `50.2% → 55.7%` |
| `deep_conv_hail_supercell` (12h) | 6 | `+0.73→+0.47` (`2.17→1.75`) | `-1.01→-0.53` (`1.30→0.83`) | `-0.10→-0.40` (`2.09→2.03`) | `+3.50→-1.09` (`5.39→3.54`) | `+75.4→+18.4` (`168.6→119.0`) | `-0.56→-0.02` (`0.72→0.20`) | `thunder_25km`: `12.8% → 16.7%` |
| `winter_storm_henk` (12h, v2) | 6 | `-0.19→-0.06` (`0.53→0.52`) | `-0.12→+0.39` (`0.52→0.69`) | `+0.38→+0.01` (`1.62→1.47`) | `+1.09→+0.35` (`2.39→2.07`) | `-8.6→-9.2` (`17.8→18.7`) | `-0.02→+0.00` (`0.08→0.01`) | `rain_0p1mm`: `64.0% → 71.7%` |
| `freezing_rain_ijzel` (12h, v2) | 6 | `-1.01→-0.20` (`1.98→1.42`) | `-0.55→-0.27` (`0.64→0.38`) | `-0.10→-0.30` (`1.47→1.48`) | `+0.16→+0.08` (`1.92→1.92`) | `+26.4→+2.0` (`57.4→12.4`) | `-0.73→+0.01` (`0.81→0.17`) | `ice`: `41.9% → 30.0%` (`FA 1→0`), `thunder_25km FA 2→0` |

---

## 4. Aandachtspunten & Vervolgstappen

1. **Volledige 20-case 24h-eindbenchmark:** Alleen daarmee verkrijgen we de definitieve globale 20-case biases en skill scores over alle seizoenen tegenover het ongewijzigde `runs/era5_benchmark` (naar een nieuwe uitvoerdirectory).
2. **Specifieke aandachtspunten om in de 20-case run te monitoren:**
   * Overbewolking (`TCC` `+0.604`) tijdens ondiepe herfstmist (`radiation_fog_autumn`).
   * Lichte positieve `MSLP`-bias (`+0.385 hPa`) tijdens diepe winterstorm (`winter_storm_henk`).
   * Zomerse zicht-/mist-valse alarmen (`14` FA in `deep_conv_hail_supercell`) onder vochtige oppervlakte-condities.
