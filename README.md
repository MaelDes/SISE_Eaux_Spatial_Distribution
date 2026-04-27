# SISE-Eaux Explorer

Python tools to process the French **SISE-Eaux** database (sanitary control of distributed drinking water), produce interactive annual maps, perform publication-ready statistical tests, and explore everything through a polished Streamlit web app.

> Companion code for the article *"Spatial distribution of water aggressiveness indices over France"* (Desrochers et al., in prep.).

| Component | Purpose |
|---|---|
| `sise_explorer.py` | **Streamlit web app** — interactive exploration of correlations, regressions, distributions, Piper diagrams, 3D surfaces, national maps, and per-commune profiles |
| `sise_pipeline.py` | Raw SISE files → yearly CSVs → static interactive HTML maps (elements and aggressiveness indices) |
| `sise_stats.py` | Statistical analysis by **BRGM geological zone** and **index grade**: Kruskal-Wallis, Dunn, ε², boxplots |
| `simplify_geojson.py` | One-shot script to reduce the 43 MB communes GeoJSON to ~5–8 MB for cloud deployment |

---

## Live demo

The app is deployed here : https://maeldes-sise-eaux-spatial-distribution-sise-explorer-5teaoj.streamlit.app/

---

## 1. Installation

Requires **Python 3.10+** (Python 3.12 recommended).

```powershell
git clone https://github.com/MaelDes/SISE_Eaux_Spatial_Distribution.git
cd SISE_Eaux_Spatial_Distribution

python -m venv .venv
.venv\Scripts\activate            # Windows
source .venv/bin/activate          # macOS / Linux

pip install -r requirements.txt
```

| Library | Used for |
|---|---|
| `streamlit` | Interactive web app |
| `streamlit-plotly-events` | Click-to-inspect on the maps |
| `pandas`, `numpy` | Data wrangling |
| `plotly` | Interactive HTML maps with year slider, all charts in the app |
| `geopandas`, `shapely` | Spatial joins (communes, geology), GeoJSON simplification |
| `scipy.stats` | Kruskal-Wallis, Mann-Whitney U, Pearson/Spearman correlations |
| `scikit-posthocs` | Dunn post-hoc tests (Bonferroni-Holm) |
| `matplotlib`, `seaborn` | Annotated boxplots, publication figures |
| `statsmodels` | OLS regression with confidence intervals |

---

## 2. Project folder layout

Recommended structure:

```
SISE_Eaux_Spatial_Distribution/
├── .venv/                                  # virtual environment
├── sise_explorer.py                        # Streamlit web app
├── sise_pipeline.py                        # CLI: raw files → CSVs → maps
├── sise_stats.py                           # CLI: full statistical analysis
├── simplify_geojson.py                     # Reduce GeoJSON for deployment
├── requirements.txt
├── README.md
│
├── communes.geojson                        # French communes (geocoding) — kept locally
├── GEO001M_CART_FR_S_FGEOL_2154.shp        # BRGM geological map (+ .dbf .shx .prj .cpg)
│
├── sise_brut/                              # raw SISE downloads
│   ├── DIS_RESULT_2020.txt
│   ├── DIS_PLV_2020.txt
│   └── ...
│
├── csv_traite/                             # output of `process` (auto-created)
│   └── Analyses_insee_20.csv
│
├── cartes/                                 # output of `map` (auto-created)
│   └── carte_CALCIUM.html
│
├── stats/                                  # output of `sise_stats.py run`
│   ├── classified_annual_data.csv
│   ├── summary_tests.csv
│   ├── summary_tests.tex
│   ├── boxplot_*.png
│   ├── figure_article_geology_boxplots.png
│   └── dunn_*.csv
│
└── .cache/                                 # auto-created by the app for downloads
    ├── classified_annual_data.csv
    └── communes.geojson
```

### External files to download once

| File | Source | Purpose |
|---|---|---|
| `communes.geojson` | https://github.com/gregoiredavid/france-geojson | INSEE-based geocoding for `sise_pipeline map` and the **Maps** tab of the app |
| BRGM geological map (`GEO001M_CART_FR_S_FGEOL_2154.*`) | https://infoterre.brgm.fr (1/1,000,000 map) | Lithological polygons for classification |

> ⚠️ A shapefile is **not a single file**. Keep together all files with the same base name: `.shp`, `.shx`, `.dbf`, `.prj`, `.cpg` (and optionally `.sbn`, `.sbx`, `.lyr`).

---

## 3. General workflow

```
┌──────────────────────┐   process   ┌────────────────────┐
│ DIS_RESULT_YYYY.txt  │────────────▶│ Analyses_insee_YY  │
│ DIS_PLV_YYYY.txt     │             │       .csv         │
└──────────────────────┘             └────────┬───────────┘
                                              │
              ┌───────────────────────────────┼───────────────────────────┐
              ▼                               ▼                           ▼
       sise_pipeline.py                 sise_stats.py              sise_explorer.py
             map                             run                    (Streamlit)
              │                               │                           │
              ▼                               ▼                           ▼
     carte_<param>.html              summary_tests.csv          Interactive web app
     (final HTML, archived)          boxplot_*.png              8 tabs: Correlation,
                                     figure_article_*.pdf       Scatter, Regression,
                                     classified_annual.csv      Distributions, Piper,
                                                                3D, Maps, Commune
```

Four steps:

1. **Once per year of data** — convert the two raw SISE files (RESULT + PLV) into a pivoted CSV
2. **For final figures of the paper** — generate definitive HTML maps for each parameter (`sise_pipeline map`)
3. **Once for the paper** — run the full statistical analysis (`sise_stats run`)
4. **Anytime** — launch the Streamlit explorer for interactive analysis (correlations, regressions, distributions, Piper, 3D, live Maps with parameter dropdown, and per-commune profile)

---

## 4. `sise_pipeline.py` — Processing and mapping

```powershell
.venv\Scripts\python.exe sise_pipeline.py --help
```

| Subcommand | Purpose |
|---|---|
| `process` | Convert the 2 raw SISE files of one year into a wide CSV |
| `map` | Generate an interactive HTML map for one parameter |
| `list-params` | List available parameters and indices |

### 4.1 `process` — Convert raw SISE files

```powershell
.venv\Scripts\python.exe sise_pipeline.py process `
    --result sise_brut\DIS_RESULT_2020.txt `
    --plv    sise_brut\DIS_PLV_2020.txt `
    --out    csv_traite\Analyses_insee_20.csv
```

| Argument | Required | Purpose |
|---|:-:|---|
| `--result` | ✅ | Path to `DIS_RESULT_YYYY.txt` (analyses file) |
| `--plv` | ✅ | Path to `DIS_PLV_YYYY.txt` (samplings file, includes INSEE codes) |
| `--out` | ✅ | Output CSV path (parent folder auto-created) |

Output: one row per sample, one column per parameter (e.g. `CALCIUM`, `HYDROGÉNOCARBONATES`, `PH `), plus ID columns `referenceprel`, `dateprel`, `nomcommuneprinc`, `inseecommuneprinc`.

**Batch processing (PowerShell loop):**

```powershell
$years = 20, 21, 22, 23
foreach ($y in $years) {
    .venv\Scripts\python.exe sise_pipeline.py process `
        --result sise_brut\DIS_RESULT_20$y.txt `
        --plv    sise_brut\DIS_PLV_20$y.txt `
        --out    csv_traite\Analyses_insee_$y.csv
}
```

### 4.2 `map` — Generate an interactive map

```powershell
.venv\Scripts\python.exe sise_pipeline.py map `
    --param CALCIUM `
    --csv   csv_traite\Analyses_insee_20.csv csv_traite\Analyses_insee_21.csv csv_traite\Analyses_insee_22.csv `
    --geojson communes.geojson `
    --out   cartes
```

| Argument | Required | Purpose |
|---|:-:|---|
| `--param` | ✅ | Parameter to map (see `list-params`) |
| `--csv` | ✅ | One or more yearly CSVs produced by `process` |
| `--geojson` | ⚠️ (one of two) | Communes GeoJSON (INSEE-based geocoding, recommended) |
| `--geocache` | ⚠️ (one of two) | Nominatim JSON cache (fallback if no GeoJSON) |
| `--out` | | Output folder (default `cartes/`) |
| `--iqr-factor` | | IQR outlier factor (default **1.5**; `3.0` = extremes only; `0` = disabled) |
| `--pause` | | Nominatim pause in seconds (default 1.0) |

**Common `--param` values:**

| Keyword | Description |
|---|---|
| `CALCIUM` | Calcium (mg/L) |
| `MAGNÉSIUM` | Magnesium (mg/L) |
| `HYDROGÉNOCARBONATES` | Bicarbonates (mg/L) |
| `SULFATES` | Sulfates (mg/L) |
| `CHLORURES` | Chlorides (mg/L) |
| `NITRATES (EN NO3)` | Nitrates (mg/L) |
| `"PH "` | pH ⚠️ **trailing space, quotes required** |
| `"TEMPÉRATURE DE L'EAU"` | Water temperature (°C) |
| `IL` | Langelier Saturation Index (auto-computed) |
| `ryznar` | Ryznar Stability Index (auto-computed) |
| `Larson` | Larson-Skold Index (auto-computed) |
| `Bason` | Basson Index (auto-computed) |
| `IL_calc` | LSI recomputed from Ca, HCO₃ and temperature |

Full list:

```powershell
.venv\Scripts\python.exe sise_pipeline.py list-params
```

**The HTML map includes:**
- a **yearly slider**
- a **dynamic title** showing `N_communes` and `N_measurements` for the selected year
- a **hover tooltip** on each commune with the number of measurements used for the mean
- a **bottom footer** with multi-year totals

**Batch all maps at once (PowerShell loop):**

```powershell
$params = @("CALCIUM", "MAGNÉSIUM", "HYDROGÉNOCARBONATES", "SULFATES", "PH ",
            "IL", "ryznar", "Larson", "Bason")

foreach ($p in $params) {
    Write-Host ">>> Map: $p" -ForegroundColor Cyan
    .venv\Scripts\python.exe sise_pipeline.py map --param $p `
        --csv csv_traite\Analyses_insee_20.csv csv_traite\Analyses_insee_21.csv csv_traite\Analyses_insee_22.csv `
        --geojson communes.geojson --out cartes
}
```

> **Tip** — For *interactive exploration* without re-running the CLI for each parameter, use the **Maps** tab in `sise_explorer.py` instead (see §6). This `map` command is best used to generate the final HTML files for archiving / sharing.

### 4.3 Auto-computed aggressiveness indices

When you pass `--param IL`, `--param ryznar`, `--param Larson` or `--param Bason`, the script derives the column from the available chemistry:

| Index | Formula | Required variables |
|---|---|---|
| **LSI (Langelier)** | `pH − pH_eq` | `PH `, `PH D'ÉQUILIBRE À LA T° ÉCHANTILLON` |
| **Ryznar** | `2·pH_eq − pH` | pH, pH_eq |
| **Larson-Skold** | `(SO₄ + Cl) / HCO₃` | sulfates, chlorides, bicarbonates |
| **Basson** | `f(pH, pH_eq, Ca, Mg, SO₄)` | pH, pH_eq, Ca, Mg, SO₄ |
| **IL_calc** | Langelier from Ca, HCO₃, T, TDS | Ca, HCO₃, T°, pH + other ions |

If a required column is missing in your CSV, the corresponding index is silently skipped.

---

## 5. `sise_stats.py` — Statistical analysis for the paper

Addresses the reviewer's request for formal statistical evidence (beyond GMM alone).

| Subcommand | Purpose |
|---|---|
| `inspect-geology` | Explore a geological file (list available fields) before running the analysis |
| `run` | Launch the full analysis |

### 5.1 `inspect-geology` — Prepare the BRGM shapefile

Always run this **before the first analysis** to identify the lithological field name:

```powershell
.venv\Scripts\python.exe sise_stats.py inspect-geology `
    --geology GEO001M_CART_FR_S_FGEOL_2154.shp
```

Expected output:

```
Available columns:
  DESCR        (1243 unique values)  ex: 'Calcaires du Bathonien'
  LITHO_SIMP   (  13 unique values)  ex: 'Sédiments et volcanites'
  NATURE       (   8 unique values)  ex: 'sédimentaire'
  ...
Recommended --geology-field: 'LITHO_SIMP'
```

**Rule of thumb**: for the BRGM 1:1,000,000 map, use `--geology-field LITHO_SIMP` (13 pre-simplified classes by BRGM, in French — the script translates them to English automatically).

### 5.2 `run` — Full statistical analysis

```powershell
.venv\Scripts\python.exe sise_stats.py run `
    --csv        csv_traite\Analyses_insee_20.csv csv_traite\Analyses_insee_21.csv csv_traite\Analyses_insee_22.csv `
    --communes   communes.geojson `
    --geology    GEO001M_CART_FR_S_FGEOL_2154.shp `
    --geology-field LITHO_SIMP `
    --no-simplify `
    --out        stats
```

| Argument | Required | Purpose |
|---|:-:|---|
| `--csv` | ✅ | Yearly CSVs from `sise_pipeline process` |
| `--communes` | ✅ | Communes GeoJSON (INSEE geocoding) |
| `--geology` | ✅ | BRGM shapefile or GeoJSON |
| `--geology-field` | | Lithology field name (default `DESCR`). For BRGM 1M → `LITHO_SIMP` |
| `--no-simplify` | | Skip keyword-based grouping (recommended with `LITHO_SIMP`, already simplified) |
| `--macro` | | Further coarsen to 5 macro-classes (Sedimentary, Volcanic, Plutonic, Metamorphic, Alluvium) |
| `--out` | | Output folder (default `stats/`) |
| `--params` | | Parameters to test (default: 6 major parameters) |
| `--min-group-size` | | Minimum group size to be included (default 10) |
| `--aggregate` | | `commune-year` (default, recommended) or `none` |

### What `run` does, step by step

1. **Load** yearly CSVs (filtered to relevant columns only — avoids memory issues)
2. **Compute** aggressiveness indices (LSI, Ryznar, Larson, Basson)
3. **Spatial join**: each commune is assigned to a BRGM polygon → column `geological_zone`
4. **Index grades**: each index is discretised according to literature thresholds → columns `Langelier_grade`, `Ryznar_grade`, `Larson_grade`, `Basson_grade`
5. **Commune × year aggregation** (if `--aggregate commune-year`): yearly mean per INSEE code to satisfy the independence assumption of Kruskal-Wallis
6. **Tests**: Kruskal-Wallis + Dunn post-hoc with Bonferroni-Holm correction, for each (parameter × classification)
7. **Effect size ε²** (Tomczak & Tomczak, 2014) systematically reported
8. **Outputs**: LaTeX table, detailed CSVs, annotated boxplots, compact paper figure

### 5.3 Output files in `stats/`

| File | Content |
|---|---|
| `summary_tests.csv` | One row per test: `parameter`, `grouping`, `N`, `H`, `p_value`, `epsilon^2`, `significant` |
| `summary_tests.tex` | Same table, LaTeX-ready |
| `classified_annual_data.csv` | Aggregated + classified data (used by the Streamlit app and to regenerate figures) |
| `dunn_<param>_<grouping>.csv` | Pairwise p-value matrix (Dunn post-hoc) |
| `boxplot_<param>_<grouping>.png` | Individual boxplot with significance stars |
| `figure_article_geology_boxplots.png` / `.pdf` | **Compact 4-panel figure** for the paper |

### Columns in `summary_tests.csv`

| Column | Meaning |
|---|---|
| `parameter` | Chemical parameter tested (`CALCIUM`, `HYDROGENOCARBONATES`, `PH `…) |
| `grouping` | Classification used: `geological_zone`, `Langelier_grade`, `Ryznar_grade`, `Larson_grade`, `Basson_grade` |
| `N` | Number of observations (commune × year after aggregation) |
| `n_groups` | Number of compared categories (after `min_group_size` filter) |
| `H` | Kruskal-Wallis statistic |
| `p_value` | Global p-value |
| `epsilon^2` | Effect size (0.01 small, 0.06 medium, 0.14 large) |
| `significant` | `True` if p < 0.05 |
| `shapiro_p` | Shapiro-Wilk p-value (informational, sub-sampled) |

---

## 6. `sise_explorer.py` — Interactive Streamlit web app

A polished interactive web app to explore the SISE-Eaux dataset: correlations, scatter plots, distributions, Piper diagrams, 3D response surfaces, an interactive national map, and a per-commune profile. Built with Streamlit + Plotly. This dashboard can be found here : https://maeldes-sise-eaux-spatial-distribution-sise-explorer-5teaoj.streamlit.app/

### 6.1 Launch

```powershell
streamlit run sise_explorer.py
```

The app opens automatically in your browser at `http://localhost:8501`. Press `Ctrl+C` in the terminal to stop it.

On first launch, it downloads the dataset (~35 MB) and the simplified GeoJSON (~5–8 MB) from this repo's GitHub Releases — about 30 seconds on a normal connection. Subsequent launches are instant.

### 6.2 Data source (sidebar)

Three options:

| Option | Use case |
|---|---|
| **Pre-aggregated file** (`classified_annual_data.csv`) | Recommended — already aggregated by commune × year, with geological zones and index grades |
| **Raw yearly CSVs** (auto-aggregate) | When you haven't run `sise_stats.py` yet |
| **Upload a CSV** | Any custom CSV file |

### 6.3 Global filters (sidebar)

These filters apply to **all 8 tabs simultaneously**.

| Filter | Purpose |
|---|---|
| **Year** | Restrict to specific years (multi-select) |
| **Categorical filter** | Keep only certain geological zones, index grades, etc. |
| **IQR outlier filter** | Automatically masks outlier values as NaN (see below) |

**IQR outlier filter** — A slider offers 4 levels:

| Value | Meaning |
|---|---|
| `0` | Disabled |
| `1.5` | Strict — masks typical outliers |
| `3.0` | **Recommended** — only masks clear errors (e.g. HCO₃ at 13,000 mg/L) |
| `5.0` | Very lenient — only the most extreme artefacts |

Outlier values are replaced with NaN; **rows are preserved** so that other columns remain available. A per-column breakdown of masked values is shown in an expandable panel.

### 6.4 The 8 tabs

| Tab | Purpose |
|---|---|
| **Correlation** | Heatmap with Pearson / Spearman / Kendall, optional hierarchical clustering of variables, CSV download |
| **Scatter** | X vs Y scatter colored by a 3rd variable (numeric or categorical), with optional marginal histograms, log axes |
| **Regression** | X vs Y scatter with OLS regression line, R², slope, intercept, p-value, and both Pearson & Spearman correlations displayed |
| **Distributions** | Build a filter (1–2 conditions) and compare histograms of the filtered subset against the full dataset, with Mann-Whitney U test |
| **Piper diagram** | Hydrochemical facies (Piper 1944) built from Ca, Mg, Na+K, HCO₃, Cl, SO₄ converted to meq/L, with optional coloring by a 4th variable |
| **3D surface** | X/Y/Z 3D scatter with optional quadratic response surface fitting (`Z = a + bX + cY + dX² + eY² + fXY`) |
| **Maps** | Interactive choropleth/scatter map of any parameter on French communes, with year slider, top-10 panels, **click any point to see the commune's full water profile** |
| **Commune** | Search any commune by name (autocomplete) and get its full chemistry, indices, geological context, and comparison to the national average |

### 6.5 The Maps tab

Pre-requisite: a `communes.geojson` (or `communes_simplified.geojson`) file. The app downloads it automatically from this repo's GitHub Releases on first launch if it's not found locally.

**Controls** at the top of the tab:

| Control | Options |
|---|---|
| Parameter | Any chemistry column (Ca, Mg, HCO₃, SO₄, Cl, NO₃, pH...) or computed index (Langelier, Ryznar, Larson, Basson) |
| Year | Single year or "All years" pooled |
| Render | **Fast (points)** ✅ default · **Detailed (polygons)** |

**Fast vs Detailed** — Two render modes balance speed and quality:

| Mode | Backend | Speed | When to use |
|---|---|---|---|
| **Fast (points)** | `Scattergeo` on commune centroids | **~10× faster** (<500ms per change) | Interactive exploration — every parameter/year change is near-instant |
| **Detailed (polygons)** | `Choropleth` with full polygons | Slower (2-4s) | Final article figure, screenshot for publication |

**Color scaling** is parameter-aware:
- **Chemistry parameters** → Viridis (continuous), bounds at 2-98th percentile
- **Aggressiveness indices** → RdBu_r diverging, **centered on the equilibrium value** (0 for Langelier, 6.5 for Ryznar, 0.5 for Larson, 300 for Basson). Blue = scaling/balanced, red = corrosive.

**Click any point** on the map to see a detailed panel below with:
- Full water composition (Ca, Mg, HCO₃, SO₄, Cl, Na, K, pH, ...)
- All 4 aggressiveness indices with classification grades
- Geological zone and lithology

**Bonus features**:
- 3 KPI cards at the top: number of communes, mean, median
- Two expandable "Top 10 communes" panels (highest and lowest values)
- **Download HTML** — export the interactive map as a standalone HTML file
- **Download CSV** — export per-commune mean values

### 6.6 The Commune tab

Search any of the ~33,000 French communes by name or INSEE code:

- Type **at least 2 characters** in the search box ("Lyon", "Saint-P...", "vichy", or an INSEE code like "03252")
- Up to 10 matches appear instantly — pick one
- A header card shows the commune name, INSEE code, years covered, geological zone and lithology
- A composition table lists all chemistry parameters with **value vs France average and delta** (e.g. `Ca²⁺ = 132.5 / France 78.2 / +54.3 (+69%)`)
- Aggressiveness indices show their classification grade

The Commune tab uses an optimized search index (~50 ms even on 33,000 communes, no lag).

### 6.7 Some interesting figures

|---|---|
| correlation matrix between indices | Tab **Correlation** → select IL, Basson, Ryznar → Pearson |
| scatter plots between indices | Tab **Regression** → X = IL, Y = ryznar (then repeat with other pairs) |
| divergent points | Tab **Scatter** → X = IL, Y = ryznar, color by `geological_zone` |
| filtered subset histograms | Tab **Distributions** → filter `IL > 0` AND `ryznar > 6.8`, variables: PH, CALCIUM, HCO₃ |
| 3D response surfaces | Tab **3D surface** → X = PH, Y = CALCIUM, Z = IL, color = ryznar, expand "Fit surface" |
| Piper diagram  | Tab **Piper diagram** → color by `geological_zone` |
| Any spatial map (Ca, Mg, HCO₃, indices...) | Tab **Maps** → pick parameter and year (use **Detailed** mode for export) |

### 6.8 Typical sanity checks

- `Pearson r` and `R²` appear as metrics in the Regression tab. For IL vs ryznar you should recover `R² ≈ 0.94` as in the article.
- In the Distributions tab, the summary table shows means of the filtered vs full dataset — useful to quickly quote numbers in the article text.
- The Correlation matrix tab has a "Download matrix as CSV" button to export the correlations for supplementary materials.
- The Maps tab in **Fast** mode should render any parameter in under 1 s once the GeoJSON is loaded into the session cache.

---

## 7. Streamlit app for data vizualisation**

https://maeldes-sise-eaux-spatial-distribution-sise-explorer-5teaoj.streamlit.app/

---

## 8. Interpreting statistical results

With SISE sample sizes (typically N = 15,000 to 100,000), **p-values are always extreme**. The useful information is **ε²**.

**Reading grid** (Tomczak & Tomczak, 2014):

| ε² | Interpretation | Comment |
|---|---|---|
| < 0.01 | Trivial | Mathematically real but no practical meaning |
| 0.01 – 0.06 | Small | Worth mentioning but not emphasizing |
| 0.06 – 0.14 | Medium | Interesting pattern, worth discussing |
| ≥ 0.14 | Large | Solid result to highlight |
| ≥ 0.30 | Very large | Strong finding |

---

## 9. Methodology

**Why non-parametric tests:**
- Water-chemistry distributions are strongly skewed with heavy tails (rare geogenic/anthropogenic extremes)
- Groups are highly unbalanced (N from 50 to >10,000 depending on lithology)
- ANOVA is ill-suited to these features

**Why ε² is reported alongside p:**
- With N > 15,000, any non-zero difference yields p < 0.001
- ε² quantifies the variance share explained by the grouping factor
- Distinguishes statistical significance from practical importance

**Why Bonferroni-Holm for Dunn post-hoc:**
- Controls the family-wise error rate (FWER) like Bonferroni
- More powerful (sequential step-down method)

**Tool versions used in the paper:** scipy 1.11, scikit-posthocs 0.8, pandas 2.1, GeoPandas 0.14 (`sjoin` with `within` predicate), WGS-84 CRS, statsmodels 0.14.

**Key references:**

- Piper, A.M. (1944). *A graphic procedure in the geochemical interpretation of water-analyses.* Eos Transactions AGU, 25(6), 914–928.
- BRGM (2004). *Carte géologique de la France à 1/1 000 000*, 6th ed.
- Dunn, O.J. (1964). *Multiple comparisons using rank sums.* Technometrics, 6, 241–252.
- Holm, S. (1979). *A simple sequentially rejective multiple test procedure.* Scandinavian Journal of Statistics, 6, 65–70.
- Tomczak, M., & Tomczak, E. (2014). *The need to report effect size estimates revisited.* Trends in Sport Sciences, 21, 19–25.

---

## 10. Commands cheat sheet

```powershell
# Activate the working directory
cd "C:\Users\MD287298\Desktop\Carto pipeline"

# === sise_pipeline.py ===
# List available parameters
.venv\Scripts\python.exe sise_pipeline.py list-params

# Process one year of SISE data
.venv\Scripts\python.exe sise_pipeline.py process --result sise_brut\DIS_RESULT_2020.txt --plv sise_brut\DIS_PLV_2020.txt --out csv_traite\Analyses_insee_20.csv

# Generate a calcium map
.venv\Scripts\python.exe sise_pipeline.py map --param CALCIUM --csv csv_traite\*.csv --geojson communes.geojson --out cartes

# Generate a Langelier Saturation Index map
.venv\Scripts\python.exe sise_pipeline.py map --param IL --csv csv_traite\*.csv --geojson communes.geojson --out cartes

# === sise_stats.py ===
# Inspect the BRGM geological file
.venv\Scripts\python.exe sise_stats.py inspect-geology --geology GEO001M_CART_FR_S_FGEOL_2154.shp

# Full statistical analysis for the paper
.venv\Scripts\python.exe sise_stats.py run --csv csv_traite\Analyses_insee_20.csv csv_traite\Analyses_insee_21.csv csv_traite\Analyses_insee_22.csv --communes communes.geojson --geology GEO001M_CART_FR_S_FGEOL_2154.shp --geology-field LITHO_SIMP --no-simplify --out stats

# === sise_explorer.py (Streamlit app) ===
streamlit run sise_explorer.py

# === Deployment helpers ===
# Reduce GeoJSON for cloud deployment
python simplify_geojson.py
```

---

## License

SISE-Eaux data: provided by the French Ministry of Health under [Licence Ouverte](https://www.etalab.gouv.fr/licence-ouverte-open-licence/).
GeoJSON of communes: from [gregoiredavid/france-geojson](https://github.com/gregoiredavid/france-geojson) (MIT).
BRGM geological map: BRGM, redistributed under research/educational use.
