# SISE_Eaux_Spatial_Distribution
Spatial Distribution of Potable Water Chemistry and Aggressiveness Indices in France Using the SISE-Eaux Database 
# SISE-Eaux Pipeline

Python tools to process the French **SISE-Eaux** database (sanitary control of distributed drinking water), produce interactive annual maps, and perform statistical tests.

Two independent but complementary scripts:

| Script | Purpose |
|---|---|
| `sise_pipeline.py` | Raw SISE files → yearly CSVs → **interactive maps** (elements and aggressiveness indices) |
| `sise_stats.py` | Statistical analysis by **BRGM geological zone** and **index grade**: Kruskal-Wallis, Dunn, ε², boxplots |

---

## 1. Installation

Requires **Python 3.10+** (ideally 3.11).

```powershell
cd "C:\path\to\your\folder"
python -m venv .venv
.venv\Scripts\pip.exe install pandas numpy plotly geopandas shapely scipy scikit-posthocs matplotlib seaborn geopy
```

| Library | Used for |
|---|---|
| `pandas`, `numpy` | Data wrangling |
| `plotly` | Interactive HTML maps with yearly slider |
| `geopandas`, `shapely` | Spatial joins (communes, geology) |
| `scipy.stats` | Kruskal-Wallis tests |
| `scikit-posthocs` | Dunn post-hoc tests (Bonferroni-Holm) |
| `matplotlib`, `seaborn` | Annotated boxplots |
| `geopy` | Nominatim fallback geocoding (rarely needed) |

---

## 2. Project folder layout

Recommended structure:

```
Carto pipeline/
├── .venv/                                  # virtual environment
├── sise_pipeline.py
├── sise_stats.py
│
├── communes.geojson                        # French communes (for INSEE geocoding)
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
└── stats/                                  # output of `sise_stats.py run` (auto-created)
    ├── summary_tests.csv
    ├── summary_tests.tex
    ├── classified_annual_data.csv
    ├── boxplot_CALCIUM_geological_zone.png
    ├── figure_article_geology_boxplots.png
    └── dunn_*.csv
```

### External files to download once

| File | Source | Purpose |
|---|---|---|
| `communes.geojson` | https://github.com/gregoiredavid/france-geojson | INSEE centroids for geocoding |
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
                      ┌───────────────────────┼───────────────────────┐
                      ▼                                               ▼
               sise_pipeline.py                                 sise_stats.py
                     map                                             run
                      │                                               │
                      ▼                                               ▼
              carte_<param>.html                              summary_tests.csv
                                                              boxplot_*.png
                                                              figure_article_*.pdf
```

Three steps:

1. **Once per year of data** — convert the two raw SISE files (RESULT + PLV) into a pivoted CSV
2. **As many times as needed** — generate interactive maps for each parameter or index
3. **Once for the paper** — run the full statistical analysis

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

---

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

---

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
| `MAGNESIUM` | Magnesium (mg/L) |
| `HYDROGENOCARBONATES` | Bicarbonates (mg/L) |
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
$params = @("CALCIUM", "MAGNESIUM", "HYDROGENOCARBONATES", "SULFATES", "PH ",
            "IL", "ryznar", "Larson", "Bason")

foreach ($p in $params) {
    Write-Host ">>> Map: $p" -ForegroundColor Cyan
    .venv\Scripts\python.exe sise_pipeline.py map --param $p `
        --csv csv_traite\Analyses_insee_20.csv csv_traite\Analyses_insee_21.csv csv_traite\Analyses_insee_22.csv `
        --geojson communes.geojson --out cartes
}
```

---

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

---

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

---

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

---

### 5.3 Output files in `stats/`

| File | Content |
|---|---|
| `summary_tests.csv` | One row per test: `parameter`, `grouping`, `N`, `H`, `p_value`, `epsilon^2`, `significant` |
| `summary_tests.tex` | Same table, LaTeX-ready |
| `classified_annual_data.csv` | Aggregated + classified data (useful to regenerate figures) |
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

## 6. Interpreting statistical results

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

## 7. FAQ and troubleshooting

### `MemoryError: Unable to allocate X GiB`

SISE CSVs can contain up to 1500 columns (one per unique parameter). The script automatically filters to the ~15 useful columns. If the error persists, check that your `load_csv_files` signature includes `keep_columns` (recent versions).

### Geological categories still appear in French in my boxplots

Make sure you use `LITHO_SIMP` as the field (`--geology-field LITHO_SIMP`) **and** pass `--no-simplify`. The script auto-translates the 13 BRGM classes via the `LITHO_SIMP_FR_EN` dictionary.

### How many years can I combine?

As many as you want — CSVs are simply concatenated. Adjust figure legends accordingly.

### Testing a parameter not in the default list

```powershell
.venv\Scripts\python.exe sise_stats.py run ... --params CALCIUM MAGNESIUM "NITRATES (EN NO3)"
```

### Running tests on raw measurements instead of commune × year

Use `--aggregate none`. Not recommended (violates independence) but useful for comparison.

---

## 8. Methodology (paper-ready)

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

**Key references:**

- BRGM (2004). *Carte géologique de la France à 1/1 000 000*, 6th ed.
- Dunn, O.J. (1964). Multiple comparisons using rank sums. *Technometrics* 6, 241–252.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics* 6, 65–70.
- Tomczak, M., & Tomczak, E. (2014). The need to report effect size estimates revisited. *Trends in Sport Sciences* 21, 19–25.

---

## 9. Commands cheat sheet

```powershell
# Activate the working directory
cd "C:\Users\MD287298\Desktop\Carto pipeline"

# List available parameters
.venv\Scripts\python.exe sise_pipeline.py list-params

# Inspect the BRGM geological file
.venv\Scripts\python.exe sise_stats.py inspect-geology --geology GEO001M_CART_FR_S_FGEOL_2154.shp

# Process one year of SISE data
.venv\Scripts\python.exe sise_pipeline.py process --result sise_brut\DIS_RESULT_2020.txt --plv sise_brut\DIS_PLV_2020.txt --out csv_traite\Analyses_insee_20.csv

# Generate a calcium map
.venv\Scripts\python.exe sise_pipeline.py map --param CALCIUM --csv csv_traite\*.csv --geojson communes.geojson --out cartes

# Generate a Langelier Saturation Index map
.venv\Scripts\python.exe sise_pipeline.py map --param IL --csv csv_traite\*.csv --geojson communes.geojson --out cartes

# Full statistical analysis for the paper
.venv\Scripts\python.exe sise_stats.py run --csv csv_traite\Analyses_insee_20.csv csv_traite\Analyses_insee_21.csv csv_traite\Analyses_insee_22.csv --communes communes.geojson --geology GEO001M_CART_FR_S_FGEOL_2154.shp --geology-field LITHO_SIMP --no-simplify --out stats
```
