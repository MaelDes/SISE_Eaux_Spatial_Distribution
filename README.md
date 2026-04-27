# SISE-Eaux Explorer

Interactive web app + statistical pipeline to explore the **French drinking-water chemistry database (SISE-Eaux)**: correlations, regressions, hydrochemical facies, response surfaces, and an interactive map of all French communes.

> Companion code for the article *"Spatial distribution of water aggressiveness indices over France"* (Desrochers et al., in prep.).

---

## Live demo

Try the deployed app: **[sise-eaux.streamlit.app](https://share.streamlit.io)** *(replace with your actual URL once deployed)*

The app loads ~33,000 communes × 3 years of measurements and lets you:

- Explore correlations between any pair of chemistry parameters (Pearson / Spearman / Kendall)
- Build OLS regressions with R² and p-values
- Compare distributions of filtered subsets vs. the full dataset
- Visualise the hydrochemical signature on a Piper (1944) diagram
- Plot 3D response surfaces (e.g. Langelier SI vs. pH × Ca²⁺)
- View any parameter as a national map with year slider, **click any commune to see its full water profile**
- Search any French commune by name and get its chemistry, indices, and geological context

---

## Repository contents

| File | Purpose |
|---|---|
| `sise_explorer.py` | Streamlit web app (8 tabs) |
| `sise_pipeline.py` | CLI to convert raw SISE files to clean CSVs and generate static HTML maps |
| `sise_stats.py` | CLI to run the full statistical analysis (Kruskal-Wallis + Dunn + ε² effect sizes) |
| `simplify_geojson.py` | One-shot script to reduce the 43 MB communes GeoJSON to ~5–8 MB |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |

The raw data files (~35 MB CSV + 30 MB GeoJSON) are too large for the Git repo and are hosted on **GitHub Releases**. The app downloads them automatically on first launch and caches them locally.

---

## Local installation

**1. Clone the repo**

```bash
git clone https://github.com/MaelDes/SISE_Eaux_Spatial_Distribution.git
cd SISE_Eaux_Spatial_Distribution
```

**2. Set up a Python environment** (Python 3.12 recommended)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell
source .venv/bin/activate        # macOS / Linux
pip install -r requirements.txt
```

**3. Launch the app**

```bash
streamlit run sise_explorer.py
```

The app opens at `http://localhost:8501`. On first launch, it downloads the dataset (~35 MB) and the GeoJSON (~30 MB) from this repo's GitHub Releases — about 30 seconds on a normal connection. Subsequent launches are instant.

---

## The 8 tabs

| Tab | Purpose |
|---|---|
| **Correlation** | Heatmap (Pearson / Spearman / Kendall) with optional hierarchical clustering of variables |
| **Scatter** | X vs Y scatter with colouring by a third variable (numeric or categorical), optional marginals |
| **Regression** | OLS fit with R², slope, intercept, p-value, and both Pearson & Spearman correlations |
| **Distributions** | Compare histograms of any filtered subset against the full dataset, with Mann-Whitney U test |
| **Piper diagram** | Hydrochemical facies (Piper 1944) from the major ions, with optional colouring |
| **3D surface** | X/Y/Z 3D scatter with optional quadratic surface fit (`Z = a + bX + cY + dX² + eY² + fXY`) |
| **Maps** | National choropleth/scatter of any parameter, year slider, click any point to see the commune's full water profile |
| **Commune** | Search any commune by name (autocomplete) and get its full chemistry, indices, geology, and comparison to the national average |

---

## Deploying your own copy on Streamlit Cloud

The repo is configured to deploy in 5 minutes on [Streamlit Community Cloud](https://share.streamlit.io) (free).

**1. Fork or clone this repo on your own GitHub account.**

**2. Generate the simplified GeoJSON locally** (only needed once):

```bash
python simplify_geojson.py
```

This produces `communes_simplified.geojson` (~5–8 MB) from the full 43 MB version. You'll need the source file (`communes.geojson`) — download from [gregoiredavid/france-geojson](https://github.com/gregoiredavid/france-geojson).

**3. Upload data files to a GitHub Release**

- On your repo: `Releases` → `Draft a new release`
- Tag: `v1.0`
- Drag-and-drop two assets:
  - `classified_annual_data.csv` (the pre-aggregated dataset)
  - `communes_simplified.geojson` (the simplified GeoJSON)
- Publish

**4. Update the URLs in `sise_explorer.py`**

Edit two constants at the top of the file:

```python
CLASSIFIED_URL = "https://github.com/<your_username>/<repo>/releases/download/v1.0/classified_annual_data.csv"
GEOJSON_URL    = "https://github.com/<your_username>/<repo>/releases/download/v1.0/communes_simplified.geojson"
```

**5. Push to GitHub, then deploy on Streamlit Cloud**

- Go to [share.streamlit.io](https://share.streamlit.io)
- Sign in with GitHub
- "New app" → select your repo, branch `main`, file `sise_explorer.py`
- Deploy

You'll get a public URL like `https://your-app.streamlit.app` to share.

---

## CLI tools (advanced)

For users who want to regenerate the dataset from raw SISE-Eaux files (RESULT + PLV format from [data.gouv.fr](https://www.data.gouv.fr)):

### `sise_pipeline.py`

```bash
# Convert raw SISE files to clean per-year CSVs
python sise_pipeline.py process \
    --plv DIS_PLV_2022.txt \
    --result DIS_RESULT_2022.txt \
    --communes communes.geojson \
    --out csv_traite/Analyses_insee_22.csv

# Generate static HTML map for any parameter
python sise_pipeline.py map \
    --param CALCIUM \
    --csv csv_traite/Analyses_insee_*.csv \
    --geojson communes.geojson \
    --out cartes/

# List all available parameters
python sise_pipeline.py list-params --csv csv_traite/Analyses_insee_22.csv
```

The pipeline auto-computes four water aggressiveness indices: **Langelier (IL)**, **Ryznar**, **Larson-Skold**, and **Basson**.

### `sise_stats.py`

```bash
# Inspect the BRGM lithology shapefile to choose a field
python sise_stats.py inspect-geology \
    --shapefile GEO001M_CART_FR_S_FGEOL_2154.shp

# Run the full statistical analysis
python sise_stats.py run \
    --csv csv_traite/Analyses_insee_*.csv \
    --geology GEO001M_CART_FR_S_FGEOL_2154.shp \
    --geology-field LITHO_SIMP \
    --out stats/
```

This produces:

- `stats/classified_annual_data.csv` — the pre-aggregated dataset with geological zones and index grades
- `stats/summary_tests.csv` — Kruskal-Wallis + Dunn post-hoc results with ε² effect sizes
- `stats/boxplot_*.png` — 30 boxplots (6 chemistry parameters × 5 classifications)
- `stats/figure_article_*.pdf` — publication-ready figures

---

## Methodology

**Statistical tests.** Kruskal-Wallis omnibus tests are followed by Dunn's post-hoc tests with Bonferroni-Holm correction for multiple comparisons. We report **ε² effect sizes** alongside p-values to quantify practical significance, since p-values inflate with sample size (cf. Tomczak & Tomczak 2014).

**Geological zones.** Communes are spatially joined (`GeoPandas.sjoin`, predicate `within`, WGS-84 CRS) to the BRGM 1:1,000,000 vector geological map of France (BRGM 2004). The simplified lithology field `LITHO_SIMP` (13 classes) is used.

**Aggressiveness indices.** Computed per-measurement, then averaged per commune × year. Formulas as in the article §2.2.

**Tool versions.** scipy 1.11, scikit-posthocs 0.8, pandas 2.1, GeoPandas 0.14, statsmodels 0.14.

---

## References

- Piper, A. M. (1944). *A graphic procedure in the geochemical interpretation of water-analyses.* Eos Transactions AGU, 25(6), 914–928.
- Dunn, O. J. (1964). *Multiple comparisons using rank sums.* Technometrics, 6(3), 241–252.
- Holm, S. (1979). *A simple sequentially rejective multiple test procedure.* Scandinavian Journal of Statistics, 6, 65–70.
- Tomczak, M., & Tomczak, E. (2014). *The need to report effect size estimates revisited.* Trends in Sport Sciences, 1(21), 19–25.
- BRGM (2004). *Carte géologique de la France à l'échelle 1/1 000 000* (6e édition révisée).

---

## License

Code: MIT.
Data: SISE-Eaux is provided by the French Ministry of Health under [Licence Ouverte](https://www.etalab.gouv.fr/licence-ouverte-open-licence/).
GeoJSON of communes: from [gregoiredavid/france-geojson](https://github.com/gregoiredavid/france-geojson) (MIT).
BRGM geological map: BRGM, redistributed under research/educational use.

---

## Citation

If you use this code in academic work, please cite the article (currently in preparation) and link to this repository.
