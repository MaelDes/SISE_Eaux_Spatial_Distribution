# -*- coding: utf-8 -*-
"""
sise_explorer.py
================
Streamlit web app to explore correlations, scatter plots, distributions,
Piper diagrams and 3D response surfaces from the SISE-Eaux dataset.

Run:
    streamlit run sise_explorer.py

Then open http://localhost:8501 in your browser.

Features
--------
- Tab 1: Correlation matrix (Pearson / Spearman / Kendall)
- Tab 2: Scatter X vs Y with coloring by a 3rd variable
- Tab 3: Regression scatter (X vs Y) with OLS, R^2 and p-value
- Tab 4: Distributions comparison (filtered subset vs. full dataset)
- Tab 5: Piper diagram (hydrochemical facies)
- Tab 6: 3D response surface (Langelier & Ryznar vs pH, Ca, CO2)

Data
----
Works on the commune x year aggregated dataset. You can either:
  (a) point the app to the yearly SISE CSVs (it will compute the aggregation)
  (b) point the app to the pre-aggregated `classified_annual_data.csv` produced
      by `sise_stats.py run` (faster, already has geological_zone + index grades)
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
import plotly.io as pio


# ---------------------------------------------------------------------------
# Premium Plotly theme
# ---------------------------------------------------------------------------

pio.templates["sise"] = pio.templates["plotly_dark"]
pio.templates["sise"].layout.update(
    font=dict(family="Inter, -apple-system, BlinkMacSystemFont, sans-serif",
              color="#e4e4e7", size=12),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.015)",
    colorway=["#a78bfa", "#22d3ee", "#34d399", "#fbbf24", "#f87171", "#f472b6"],
    title=dict(font=dict(size=14, color="#fafafa", family="Inter"),
               x=0.0, xanchor="left"),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.04)",
        zerolinecolor="rgba(255,255,255,0.08)",
        linecolor="rgba(255,255,255,0.08)",
        tickcolor="rgba(255,255,255,0.2)",
        tickfont=dict(color="#a1a1aa", size=11),
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.04)",
        zerolinecolor="rgba(255,255,255,0.08)",
        linecolor="rgba(255,255,255,0.08)",
        tickcolor="rgba(255,255,255,0.2)",
        tickfont=dict(color="#a1a1aa", size=11),
    ),
    legend=dict(
        bgcolor="rgba(24,24,27,0.7)",
        bordercolor="rgba(255,255,255,0.06)",
        borderwidth=1,
        font=dict(color="#d4d4d8", size=11),
    ),
    colorscale=dict(sequential="Viridis"),
    margin=dict(l=50, r=30, t=60, b=50),
    hoverlabel=dict(
        bgcolor="rgba(24,24,27,0.95)",
        bordercolor="rgba(167,139,250,0.4)",
        font=dict(color="#fafafa", family="Inter", size=12),
    ),
)
pio.templates.default = "sise"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="SISE Explorer",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Premium UI styling
# ---------------------------------------------------------------------------

_PREMIUM_CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
/* SISE Explorer - Premium UI */

/* ── Color tokens ── */
:root {
    --bg-base:       #09090b;
    --bg-elevated:   #18181b;
    --bg-hover:      #27272a;
    --bg-card:       rgba(24, 24, 27, 0.6);
    --border:        rgba(255, 255, 255, 0.06);
    --border-hover:  rgba(255, 255, 255, 0.12);
    --text-primary:  #fafafa;
    --text-secondary:#a1a1aa;
    --text-tertiary: #71717a;
    --accent:        #a78bfa;
    --accent-glow:   rgba(167, 139, 250, 0.15);
    --accent-cyan:   #22d3ee;
    --accent-green:  #34d399;
}

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    font-size: 14px;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    color: var(--text-primary);
    letter-spacing: -0.005em;
}

code, pre, .stCodeBlock, [data-testid="stCode"] {
    font-family: 'JetBrains Mono', ui-monospace, monospace !important;
    font-size: 12px !important;
}

/* ── App background — subtle gradient ── */
.stApp {
    background:
        radial-gradient(1200px 600px at 80% -10%, rgba(167, 139, 250, 0.06), transparent 60%),
        radial-gradient(800px 400px at 0% 20%, rgba(34, 211, 238, 0.04), transparent 50%),
        var(--bg-base);
    background-attachment: fixed;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(9, 9, 11, 0.85);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] > div { padding-top: 1.5rem; }
[data-testid="stSidebar"] h1 {
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.02em !important;
    margin-bottom: 0.25rem !important;
    border: none !important;
    padding: 0 !important;
    background: linear-gradient(135deg, #fafafa 0%, #a1a1aa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    color: var(--text-tertiary) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    margin-top: 1.5rem !important;
    margin-bottom: 0.75rem !important;
}
[data-testid="stSidebar"] .stMarkdown p {
    color: var(--text-secondary) !important;
    font-size: 12px !important;
    line-height: 1.6;
}
[data-testid="stSidebar"] label {
    color: var(--text-secondary) !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 500;
}
[data-testid="stSidebar"] .stRadio label {
    text-transform: none;
    letter-spacing: 0;
    font-size: 13px;
    font-weight: 400;
    color: var(--text-primary) !important;
}

/* ── Main content ── */
.main .block-container {
    padding-top: 2.5rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}

/* ── Headings ── */
h1 {
    font-size: 1.875rem !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.03em !important;
    margin-bottom: 0.5rem !important;
    border: none !important;
    padding: 0 !important;
    background: linear-gradient(135deg, #fafafa 0%, #d4d4d8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
h2 {
    font-size: 1.25rem !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.02em !important;
    margin-top: 1.5rem !important;
}
h3 {
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.015em !important;
}

/* ── Tabs — elegant pill style ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--border);
    border-radius: 10px;
    gap: 4px;
    padding: 4px;
    margin-bottom: 1.5rem;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: var(--text-secondary);
    font-size: 13px;
    font-weight: 500;
    padding: 0.5rem 1rem;
    border: none;
    border-radius: 7px;
    transition: all 0.15s ease;
    letter-spacing: -0.01em;
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--text-primary);
    background: rgba(255,255,255,0.03);
}
.stTabs [aria-selected="true"] {
    color: var(--text-primary) !important;
    background: rgba(255,255,255,0.06) !important;
    border: none !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.3), 0 0 0 1px rgba(255,255,255,0.04);
}
.stTabs [data-baseweb="tab-panel"] {
    padding-top: 0.5rem;
}

/* ── Metrics — premium cards ── */
[data-testid="stMetric"] {
    background: var(--bg-card);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    transition: border-color 0.15s ease, transform 0.15s ease;
}
[data-testid="stMetric"]:hover {
    border-color: var(--border-hover);
    transform: translateY(-1px);
}
[data-testid="stMetricLabel"] {
    color: var(--text-tertiary) !important;
    font-size: 10px !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 600;
}
[data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
    font-family: 'Inter', sans-serif;
}
[data-testid="stMetricDelta"] { font-size: 11px !important; }

/* ── Buttons ── */
.stButton > button {
    background: rgba(255, 255, 255, 0.06);
    color: var(--text-primary);
    border: 1px solid var(--border);
    border-radius: 8px;
    font-size: 13px;
    font-weight: 500;
    padding: 0.5rem 1rem;
    letter-spacing: -0.01em;
    transition: all 0.15s ease;
}
.stButton > button:hover {
    background: rgba(255, 255, 255, 0.1);
    border-color: var(--border-hover);
    transform: translateY(-1px);
}
.stButton > button:active { transform: translateY(0); }

/* ── Inputs ── */
.stTextInput input, .stNumberInput input {
    background: rgba(255, 255, 255, 0.03);
    color: var(--text-primary);
    border: 1px solid var(--border);
    border-radius: 8px;
    font-size: 13px;
    padding: 0.5rem 0.75rem;
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
}
.stTextInput input:focus, .stNumberInput input:focus {
    border-color: var(--accent);
    box-shadow: 0 0 0 3px var(--accent-glow);
    outline: none;
}

/* ── Select ── */
.stSelectbox [data-baseweb="select"] > div {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text-primary);
    font-size: 13px;
    transition: border-color 0.15s ease;
}
.stSelectbox [data-baseweb="select"] > div:hover { border-color: var(--border-hover); }
[data-baseweb="popover"] {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    box-shadow: 0 8px 24px rgba(0,0,0,0.4) !important;
}

/* ── Multiselect tags ── */
.stMultiSelect [data-baseweb="select"] > div {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid var(--border);
    border-radius: 8px;
}
[data-baseweb="tag"] {
    background: var(--accent-glow) !important;
    border: 1px solid rgba(167, 139, 250, 0.3) !important;
    border-radius: 5px !important;
}
[data-baseweb="tag"] span { color: var(--accent) !important; font-size: 11px; font-weight: 500; }

/* ── Slider ── */
.stSlider [data-baseweb="slider"] [role="slider"] {
    background: var(--accent);
    box-shadow: 0 0 0 4px var(--accent-glow);
    border: none;
}
.stSlider [data-baseweb="slider"] > div > div { background: var(--accent) !important; }

/* ── Checkbox ── */
.stCheckbox label { color: var(--text-primary) !important; font-size: 13px; }
.stCheckbox label > div:first-child {
    border: 1.5px solid var(--border-hover) !important;
    background: rgba(255, 255, 255, 0.03) !important;
    border-radius: 4px !important;
}

/* ── Alerts ── */
.stAlert, [data-testid="stNotification"] {
    background: var(--bg-card) !important;
    backdrop-filter: blur(12px);
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    font-size: 13px;
}
[data-baseweb="notification"][kind="info"] {
    border-left: 3px solid var(--accent-cyan) !important;
}
[data-baseweb="notification"][kind="success"] {
    border-left: 3px solid var(--accent-green) !important;
}
[data-baseweb="notification"][kind="warning"] {
    border-left: 3px solid #fbbf24 !important;
}
[data-baseweb="notification"][kind="error"] {
    border-left: 3px solid #f87171 !important;
}

/* ── Dataframe ── */
.stDataFrame {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid var(--border);
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    backdrop-filter: blur(12px);
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-secondary) !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    transition: border-color 0.15s ease;
}
.streamlit-expanderHeader:hover {
    border-color: var(--border-hover) !important;
}
.streamlit-expanderContent {
    background: rgba(255, 255, 255, 0.015);
    border: 1px solid var(--border);
    border-top: none;
    border-radius: 0 0 8px 8px;
}

/* ── Caption ── */
.stCaption, [data-testid="stCaptionContainer"] {
    color: var(--text-tertiary) !important;
    font-size: 12px !important;
    letter-spacing: -0.005em;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--bg-card);
    backdrop-filter: blur(12px);
    border: 1px dashed var(--border-hover);
    border-radius: 10px;
    transition: border-color 0.15s ease, background 0.15s ease;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent);
    background: rgba(167, 139, 250, 0.03);
}

/* ── Download button — accent style ── */
.stDownloadButton > button {
    background: linear-gradient(135deg, var(--accent), #8b5cf6);
    color: white;
    border: none;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 600;
    padding: 0.5rem 1rem;
    box-shadow: 0 1px 3px rgba(167, 139, 250, 0.3);
    transition: all 0.15s ease;
}
.stDownloadButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(167, 139, 250, 0.4);
}

/* ── Plotly chart container ── */
[data-testid="stPlotlyChart"] {
    background: var(--bg-card);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem;
    transition: border-color 0.15s ease;
}
[data-testid="stPlotlyChart"]:hover {
    border-color: var(--border-hover);
}

/* ── Scrollbars ── */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: var(--border-hover);
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover { background: var(--text-tertiary); }

/* ── Hide Streamlit branding ── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { background: transparent; }

/* ── Subtle fade-in animation for plot containers ── */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}
[data-testid="stPlotlyChart"],
[data-testid="stMetric"] {
    animation: fadeUp 0.4s ease-out;
}
</style>
"""

# Inject CSS. We use st.html() if available (Streamlit 1.33+) since it
# bypasses markdown processing and prevents stray characters from leaking
# into the visible page. Fall back to st.markdown otherwise.
try:
    st.html(_PREMIUM_CSS)
except AttributeError:
    st.markdown(_PREMIUM_CSS, unsafe_allow_html=True)

# Default candidate files in the project root
DEFAULT_CLASSIFIED = "stats/classified_annual_data.csv"
DEFAULT_CSV_GLOB   = "csv_traite/Analyses_insee_*.csv"

# ---------------------------------------------------------------------------
# Cloud deployment: the dataset is too large for the Git repo, so we host
# it in a GitHub Release and download it on first launch. Once cached on
# disk, subsequent loads are instant.
#
# REPLACE this URL with your own GitHub Release asset URL.
# ---------------------------------------------------------------------------
CLASSIFIED_URL   = (
    "https://github.com/MaelDes/SISE_Eaux_Spatial_Distribution/"
    "releases/download/v1.0/classified_annual_data.csv"
)
CACHE_CLASSIFIED = ".cache/classified_annual_data.csv"


def _resolve_classified_path(user_path: str) -> str | None:
    """
    Resolve the dataset path:
      1. The user-provided local path, if it exists
      2. The internal cache file, if it exists
      3. Download from CLASSIFIED_URL into the cache (cloud deployment)
    """
    import urllib.request, urllib.error

    if user_path and Path(user_path).exists():
        return user_path

    cache = Path(CACHE_CLASSIFIED)
    if cache.exists() and cache.stat().st_size > 100_000:
        return str(cache)

    if "<your" in CLASSIFIED_URL:
        return None  # placeholder URL, skip download

    cache.parent.mkdir(parents=True, exist_ok=True)
    try:
        with st.spinner(
            "First-time setup: downloading the dataset (~35 MB, one-time only)..."
        ):
            urllib.request.urlretrieve(CLASSIFIED_URL, cache)
        return str(cache)
    except (urllib.error.URLError, OSError) as e:
        st.error(
            f"Could not download the dataset.\n\n"
            f"**URL:** `{CLASSIFIED_URL}`\n\n"
            f"**Error:** `{e}`"
        )
        return None

# Columns that identify a sample (never used as numerical variables)
ID_COLS = {
    "referenceprel", "dateprel", "nomcommuneprinc", "inseecommuneprinc",
    "Year", "Annee", "longitude", "latitude",
}

# Pretty labels for common parameters
PRETTY = {
    "CALCIUM":             "Ca²⁺ (mg/L)",
    "MAGNESIUM":           "Mg²⁺ (mg/L)",
    "MAGNÉSIUM":           "Mg²⁺ (mg/L)",
    "HYDROGENOCARBONATES": "HCO₃⁻ (mg/L)",
    "HYDROGÉNOCARBONATES": "HCO₃⁻ (mg/L)",
    "SULFATES":            "SO₄²⁻ (mg/L)",
    "CHLORURES":           "Cl⁻ (mg/L)",
    "SODIUM":              "Na⁺ (mg/L)",
    "POTASSIUM":           "K⁺ (mg/L)",
    "NITRATES (EN NO3)":   "NO₃⁻ (mg/L)",
    "PH ":                 "pH",
    "IL":                  "Langelier SI",
    "IL_calc":             "Langelier SI (computed)",
    "ryznar":              "Ryznar SI",
    "Larson":              "Larson-Skold Index",
    "Bason":               "Basson Index",
    "TEMPÉRATURE DE L'EAU":"Water temperature (°C)",
    "TEMPERATURE DE L'EAU":"Water temperature (°C)",
    "N_mesures":           "N raw measurements",
}

def pretty(c: str) -> str:
    return PRETTY.get(c, c)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_classified(path: str) -> pd.DataFrame:
    """Load the pre-aggregated file produced by sise_stats.py"""
    df = pd.read_csv(path, low_memory=False)
    return df


@st.cache_data(show_spinner=False)
def load_and_aggregate(csv_paths: list[str]) -> pd.DataFrame:
    """Load yearly SISE CSVs and aggregate by commune x year."""
    from sise_pipeline import (
        load_csv_files, compute_indices,
        COL_PH, COL_PH_EQ, COL_TEMP, COL_CA, COL_MG, COL_HCO3,
        COL_SO4, COL_CL, COL_NO3, COL_NA, COL_K,
    )
    required = {
        COL_PH, COL_PH_EQ, COL_TEMP,
        COL_CA, COL_MG, COL_HCO3,
        COL_SO4, COL_CL, COL_NO3, COL_NA, COL_K,
    }
    df = load_csv_files(csv_paths, keep_columns=required)
    df = compute_indices(df)

    # aggregate to commune-year
    df["Year"] = pd.to_datetime(df["dateprel"], errors="coerce").dt.year
    df = df.dropna(subset=["Year", "inseecommuneprinc"])
    df["Year"] = df["Year"].astype(int)

    numeric = df.select_dtypes(include=np.number).columns.tolist()
    numeric = [c for c in numeric if c not in {"Year"}]
    name_col = {"nomcommuneprinc": "first"}
    agg = {c: "mean" for c in numeric} | name_col
    out = (
        df.groupby(["inseecommuneprinc", "Year"], as_index=False)
          .agg(agg)
    )
    return out


def _is_truly_numeric(series: pd.Series, threshold: float = 0.8) -> bool:
    """
    True if more than `threshold` fraction of non-null values can be coerced
    to a finite number.

    Unlike `pd.api.types.is_numeric_dtype`, this works reliably for:
      - pandas Categorical of numbers  -> treated as numeric
      - pandas Categorical of strings  -> NOT numeric
      - object dtype with numeric strings -> numeric
      - object dtype with actual strings -> NOT numeric
    """
    s = series.dropna()
    if len(s) == 0:
        return False
    # For Categorical: inspect the categories themselves, not the codes
    if isinstance(s.dtype, pd.CategoricalDtype):
        cats = s.cat.categories
        coerced = pd.to_numeric(pd.Series(cats), errors="coerce")
        return (1 - coerced.isna().mean()) >= threshold
    # For everything else: try coercing the values
    coerced = pd.to_numeric(s, errors="coerce")
    return (1 - coerced.isna().mean()) >= threshold


def numeric_columns(df: pd.DataFrame) -> list[str]:
    """All columns with actual numeric content (see _is_truly_numeric)."""
    return [c for c in df.columns
            if c not in ID_COLS and _is_truly_numeric(df[c])]


def categorical_columns(df: pd.DataFrame) -> list[str]:
    """All columns with actual non-numeric content."""
    return [c for c in df.columns
            if c not in ID_COLS and not _is_truly_numeric(df[c])]


# ---------------------------------------------------------------------------
# Sidebar : data source selection
# ---------------------------------------------------------------------------

st.sidebar.markdown("""
<div style='padding: 0 0 1.5rem 0; border-bottom: 1px solid rgba(255,255,255,0.06); margin-bottom: 1.25rem;'>
    <div style='display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.4rem;'>
        <div style='width: 28px; height: 28px; border-radius: 7px;
                    background: linear-gradient(135deg, #a78bfa, #22d3ee);
                    display: flex; align-items: center; justify-content: center;
                    box-shadow: 0 4px 12px rgba(167, 139, 250, 0.25);
                    font-size: 14px; font-weight: 700; color: white;'>S</div>
        <div style='font-size: 1rem; font-weight: 700; color: #fafafa;
                    letter-spacing: -0.02em;'>SISE Explorer</div>
    </div>
    <div style='font-size: 11px; color: #71717a; line-height: 1.5; padding-left: 0.1rem;'>
        Drinking-water chemistry analysis
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.header("Data source")

source = st.sidebar.radio(
    "Data source",
    options=[
        "Pre-aggregated file (classified_annual_data.csv)",
        "Raw yearly CSVs (will aggregate now)",
        "Upload a CSV",
    ],
    index=0,
)

df = None
status_msg = ""

if source.startswith("Pre-aggregated"):
    default_path = st.sidebar.text_input("Path", DEFAULT_CLASSIFIED)
    resolved = _resolve_classified_path(default_path)
    if resolved is not None:
        with st.spinner(f"Loading {resolved}..."):
            df = load_classified(resolved)
        status_msg = f"Loaded {len(df):,} rows from {resolved}"
    else:
        st.sidebar.warning(
            f"File not found: {default_path}\n\n"
            "Run `sise_stats.py run` first to generate it, "
            "or switch to 'Raw yearly CSVs'."
        )

elif source.startswith("Raw yearly CSVs"):
    import glob
    pattern = st.sidebar.text_input("Glob pattern", DEFAULT_CSV_GLOB)
    csv_files = sorted(glob.glob(pattern))
    st.sidebar.caption(f"{len(csv_files)} file(s) matched")
    if csv_files:
        if st.sidebar.button("Load & aggregate"):
            with st.spinner("Loading and aggregating..."):
                df = load_and_aggregate(csv_files)
            status_msg = f"Loaded and aggregated {len(df):,} commune×year rows"

else:  # Upload
    up = st.sidebar.file_uploader("CSV file", type=["csv"])
    if up is not None:
        df = pd.read_csv(up, low_memory=False)
        status_msg = f"Loaded {len(df):,} rows from upload"

if df is None:
    st.markdown("""
    <div style='padding: 4rem 0 2rem 0; text-align: center;'>
        <div style='display: inline-flex; align-items: center; gap: 0.75rem; padding: 0.5rem 1rem;
                    background: rgba(167, 139, 250, 0.08); border: 1px solid rgba(167, 139, 250, 0.2);
                    border-radius: 100px; font-size: 12px; color: #a78bfa; font-weight: 500;
                    margin-bottom: 1.5rem; letter-spacing: 0.02em;'>
            <span style='width: 6px; height: 6px; background: #a78bfa; border-radius: 50%;
                        box-shadow: 0 0 8px #a78bfa;'></span>
            INTERACTIVE DATA EXPLORER
        </div>
        <h1 style='font-size: 3rem !important; margin-bottom: 0.75rem !important;
                   background: linear-gradient(135deg, #fafafa 0%, #a1a1aa 100%);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                   font-weight: 800 !important; letter-spacing: -0.04em !important;'>
            SISE-Eaux Explorer
        </h1>
        <p style='color: #a1a1aa; font-size: 1.05rem; max-width: 600px; margin: 0 auto 2.5rem auto;
                  line-height: 1.6;'>
            Statistical exploration of French drinking-water chemistry —
            correlations, regressions, hydrochemical facies and response surfaces.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.info(
        "**Select a data source in the sidebar to begin.**  \n"
        "Recommended: point to `stats/classified_annual_data.csv` produced by "
        "`sise_stats.py run` — it already contains geological zones and index grades."
    )
    st.stop()

st.sidebar.success(status_msg)
st.sidebar.caption(f"Rows: **{len(df):,}** — Cols: **{len(df.columns)}**")


# ---------------------------------------------------------------------------
# Sidebar : optional filters
# ---------------------------------------------------------------------------

st.sidebar.header("Filters")

num_cols = numeric_columns(df)
cat_cols = categorical_columns(df)

# Year filter if Year column exists
for year_col in ("Year", "Annee"):
    if year_col in df.columns:
        years = sorted(df[year_col].dropna().unique())
        if len(years) > 1:
            sel_years = st.sidebar.multiselect(
                f"{year_col}", years, default=list(years)
            )
            df = df[df[year_col].isin(sel_years)]
        break

# Categorical filter (one at a time)
if cat_cols:
    filt_col = st.sidebar.selectbox(
        "Filter by category (optional)",
        ["(none)"] + cat_cols,
    )
    if filt_col != "(none)":
        options = sorted(df[filt_col].dropna().unique().astype(str))
        selected = st.sidebar.multiselect(
            f"{filt_col} values",
            options, default=options,
        )
        df = df[df[filt_col].astype(str).isin(selected)]

# Outlier filter (IQR-based)
st.sidebar.markdown("**Outlier filter (IQR)**")
iqr_factor = st.sidebar.select_slider(
    "Replace outliers beyond Q1/Q3 ± k × IQR with NaN",
    options=[0.0, 1.5, 3.0, 5.0],
    value=3.0,
    help=(
        "Typical values: 1.5 (strict, masks most outliers), "
        "3.0 (only masks clear errors, recommended), "
        "5.0 (only masks extreme artefacts), "
        "0 (disabled). "
        "The row is kept, only the aberrant value is set to NaN."
    ),
)
if iqr_factor > 0:
    # Winsorisation per column: masks individual outlying values (sets to NaN)
    # instead of dropping rows. Safer for multivariate analysis.
    target_cols = [c for c in num_cols if c not in {"Year", "Annee", "N_mesures",
                                                     "longitude", "latitude"}]
    if target_cols:
        df = df.copy()
        masked_total = 0
        masked_by = {}
        for col in target_cols:
            s = pd.to_numeric(df[col], errors="coerce")
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0 or pd.isna(iqr):
                continue
            lo, hi = q1 - iqr_factor * iqr, q3 + iqr_factor * iqr
            out_mask = s.notna() & ((s < lo) | (s > hi))
            n = int(out_mask.sum())
            if n > 0:
                df.loc[out_mask, col] = np.nan
                masked_by[col] = n
                masked_total += n
        if masked_total > 0:
            st.sidebar.caption(
                f"🧹 IQR × {iqr_factor}: masked **{masked_total:,}** values as NaN "
                "(rows preserved)"
            )
            with st.sidebar.expander("Details by column"):
                for c, n in sorted(masked_by.items(), key=lambda kv: -kv[1]):
                    st.caption(f"  {c}: {n:,} values")
        else:
            st.sidebar.caption(f"🧹 IQR × {iqr_factor}: no outlier detected")

st.sidebar.caption(f"After filters: **{len(df):,}** rows")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown(f"""
<div style='margin-bottom: 1.5rem;'>
    <div style='font-size: 11px; color: #a78bfa; font-weight: 600; letter-spacing: 0.12em;
                text-transform: uppercase; margin-bottom: 0.5rem;'>
        Exploratory analytics
    </div>
    <h1>SISE-Eaux Explorer</h1>
    <p style='color: #a1a1aa; font-size: 0.95rem; margin: 0; line-height: 1.5;'>
        French drinking-water chemistry · commune × year aggregated dataset ·
        <span style='color: #fafafa; font-weight: 500;'>{len(df):,}</span> observations ·
        <span style='color: #fafafa; font-weight: 500;'>{len(df.columns)}</span> variables
    </p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_corr, tab_scatter_c, tab_reg, tab_dist, tab_piper, tab_3d, tab_map, tab_commune = st.tabs([
    "Correlation",
    "Scatter",
    "Regression",
    "Distributions",
    "Piper diagram",
    "3D surface",
    "Maps",
    "Commune",
])


# ============================================================================
# Tab 1 : Correlation matrix (heatmap)
# ============================================================================

with tab_corr:
    st.subheader("Correlation matrix")
    st.caption("Select variables to include and a correlation method.")

    c1, c2 = st.columns([2, 1])
    with c1:
        default = [c for c in ["CALCIUM", "MAGNESIUM", "HYDROGENOCARBONATES",
                               "SULFATES", "CHLORURES", "PH ",
                               "IL", "ryznar", "Larson", "Bason"]
                   if c in num_cols]
        corr_vars = st.multiselect(
            "Variables",
            num_cols,
            default=default or num_cols[: min(10, len(num_cols))],
            format_func=pretty,
        )
    with c2:
        corr_method = st.radio(
            "Method",
            ["pearson", "spearman", "kendall"],
            index=0,
            help=(
                "Pearson: linear correlation (assumes normal data)\n"
                "Spearman: rank correlation (robust, non-linear monotonic)\n"
                "Kendall: rank correlation (more conservative than Spearman)"
            ),
        )
        show_values = st.checkbox("Show values", value=True)
        cluster = st.checkbox(
            "Cluster variables",
            value=False,
            help="Reorder rows/columns by hierarchical clustering",
        )

    if len(corr_vars) < 2:
        st.info("Pick at least 2 variables.")
    else:
        sub = df[corr_vars].apply(pd.to_numeric, errors="coerce").dropna()
        st.caption(
            f"N = {len(sub):,} rows (after dropping NaN on {len(corr_vars)} variables)"
        )

        if len(sub) < 3:
            st.warning("Not enough rows with all selected variables non-null.")
        else:
            corr = sub.corr(method=corr_method)

            if cluster:
                from scipy.spatial.distance import squareform
                from scipy.cluster.hierarchy import linkage, leaves_list
                dist = 1 - corr.abs()
                linkage_matrix = linkage(squareform(dist.values, checks=False),
                                         method="average")
                order = leaves_list(linkage_matrix)
                corr = corr.iloc[order, order]

            pretty_labels = [pretty(c) for c in corr.columns]
            fig = go.Figure(data=go.Heatmap(
                z=corr.values,
                x=pretty_labels, y=pretty_labels,
                colorscale="RdBu_r",
                zmin=-1, zmax=1,
                text=np.round(corr.values, 2) if show_values else None,
                texttemplate="%{text}" if show_values else None,
                hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>r = %{z:.3f}<extra></extra>",
                colorbar=dict(title="r"),
            ))
            fig.update_layout(
                height=max(500, 40 * len(corr)),
                width=max(500, 40 * len(corr)),
                title=f"Correlation ({corr_method.capitalize()}) — N = {len(sub):,}",
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Download matrix as CSV"):
                st.download_button(
                    "Download",
                    corr.to_csv().encode("utf-8"),
                    file_name=f"correlation_{corr_method}.csv",
                    mime="text/csv",
                )


# ============================================================================
# Tab 2 : Scatter X vs Y with color = Z
# ============================================================================

with tab_scatter_c:
    st.subheader("Scatter plot X vs Y, colored by Z")

    c1, c2, c3, c4 = st.columns(4)
    x_col = c1.selectbox("X", num_cols, index=num_cols.index("CALCIUM") if "CALCIUM" in num_cols else 0, key="sc_x", format_func=pretty)
    y_col = c2.selectbox("Y", num_cols, index=num_cols.index("HYDROGENOCARBONATES") if "HYDROGENOCARBONATES" in num_cols else 1, key="sc_y", format_func=pretty)

    color_options = ["(none)"] + num_cols + cat_cols
    z_col = c3.selectbox("Color by", color_options, index=0, key="sc_z", format_func=pretty)

    log_x = c4.checkbox("log X", value=False)
    log_y = c4.checkbox("log Y", value=False)
    marginals = c4.checkbox("Marginal distributions", value=True)

    # Build a deduplicated column list (handles X == Y or X == Z etc.)
    wanted_cols = [x_col, y_col]
    if z_col != "(none)":
        wanted_cols.append(z_col)
    unique_cols = list(dict.fromkeys(wanted_cols))  # preserve order, drop duplicates
    sub = df[unique_cols].dropna()

    st.caption(f"N = {len(sub):,} points")

    # Robust categorical detection based on the same helper used for column lists.
    color_is_categorical = False
    if z_col != "(none)" and z_col in sub.columns:
        color_is_categorical = not _is_truly_numeric(sub[z_col])

    if len(sub) < 2:
        st.warning("Not enough data.")
    elif x_col == y_col:
        st.warning("X and Y are the same variable. Pick a different Y.")
    else:
        # Prepare data: string-coerce categorical color column
        plot_df = sub.copy()
        if color_is_categorical and z_col != "(none)":
            plot_df[z_col] = plot_df[z_col].astype(str)

        # Build the scatter WITHOUT marginals (stable path for any color type)
        color_arg = None if z_col == "(none)" else z_col
        fig_scatter = px.scatter(
            plot_df, x=x_col, y=y_col,
            color=color_arg,
            opacity=0.55,
            log_x=log_x, log_y=log_y,
            color_continuous_scale=None if color_is_categorical else "Viridis",
            hover_data=plot_df.columns.tolist(),
            labels={x_col: pretty(x_col), y_col: pretty(y_col),
                    **({z_col: pretty(z_col)} if z_col != "(none)" else {})},
            height=650,
        )
        fig_scatter.update_traces(marker=dict(size=5), selector=dict(type="scatter"))

        if not marginals:
            fig = fig_scatter
        else:
            # Build marginals manually to sidestep the known Plotly bug with
            # histograms + categorical color (`color='V'` error).
            from plotly.subplots import make_subplots

            fig = make_subplots(
                rows=2, cols=2,
                column_widths=[0.82, 0.18],
                row_heights=[0.18, 0.82],
                shared_xaxes=True, shared_yaxes=True,
                horizontal_spacing=0.02, vertical_spacing=0.02,
                specs=[[{"type": "xy"}, None],
                       [{"type": "xy"}, {"type": "xy"}]],
            )

            # Top marginal: histogram of X (single color, no grouping)
            fig.add_trace(
                go.Histogram(
                    x=plot_df[x_col], nbinsx=40,
                    marker_color="rgba(167, 139, 250, 0.5)", showlegend=False,
                    hoverinfo="skip",
                ),
                row=1, col=1,
            )
            # Right marginal: histogram of Y
            fig.add_trace(
                go.Histogram(
                    y=plot_df[y_col], nbinsy=40,
                    marker_color="rgba(167, 139, 250, 0.5)", showlegend=False,
                    hoverinfo="skip",
                ),
                row=2, col=2,
            )
            # Main scatter traces (copied from the px figure we built)
            for tr in fig_scatter.data:
                fig.add_trace(tr, row=2, col=1)

            # Copy coloraxis/legend layout from px figure
            fig.update_layout(
                coloraxis=fig_scatter.layout.coloraxis,
                height=650,
                bargap=0.05,
            )
            fig.update_xaxes(title_text=pretty(x_col), row=2, col=1)
            fig.update_yaxes(title_text=pretty(y_col), row=2, col=1)
            if log_x:
                fig.update_xaxes(type="log", row=2, col=1)
            if log_y:
                fig.update_yaxes(type="log", row=2, col=1)

        fig.update_layout(
            title=f"{pretty(y_col)} vs {pretty(x_col)}"
                  + (f" — color: {pretty(z_col)}" if z_col != "(none)" else "")
                  + f"  (N = {len(sub):,})"
        )
        st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# Tab 3 : Regression (OLS) with R^2 and p-value
# ============================================================================

with tab_reg:
    st.subheader("Regression scatter with R², slope, p-value")

    c1, c2, c3 = st.columns(3)
    x_col = c1.selectbox("X", num_cols, index=num_cols.index("IL") if "IL" in num_cols else 0, key="reg_x", format_func=pretty)
    y_col = c2.selectbox("Y", num_cols, index=num_cols.index("ryznar") if "ryznar" in num_cols else 1, key="reg_y", format_func=pretty)
    reg_type = c3.radio("Regression", ["Linear (OLS)", "Pearson r only"], horizontal=True)

    if x_col == y_col:
        st.warning("X and Y are the same variable. Pick a different Y.")
    else:
        sub = df[[x_col, y_col]].dropna()
        st.caption(f"N = {len(sub):,} points")

        if len(sub) < 3:
            st.warning("Not enough data.")
        else:
            x = sub[x_col].values
            y = sub[y_col].values

            # Pearson correlation
            r_pearson, p_pearson = stats.pearsonr(x, y)
            r_spearman, p_spearman = stats.spearmanr(x, y)

            # OLS regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            r_squared = r_value ** 2

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Pearson r", f"{r_pearson:.3f}",
                      help=f"p = {p_pearson:.2e}")
            m2.metric("R²", f"{r_squared:.3f}")
            m3.metric("Slope", f"{slope:.3g}",
                      help=f"Intercept = {intercept:.3g}, std err = {std_err:.3g}")
            m4.metric("p-value", f"{p_value:.2e}",
                      delta="significant" if p_value < 0.05 else "n.s.",
                      delta_color="normal")

            # Scatter + regression line
            fig = px.scatter(
                sub, x=x_col, y=y_col,
                trendline="ols",
                opacity=0.4,
                labels={x_col: pretty(x_col), y_col: pretty(y_col)},
                height=600,
            )
            fig.update_traces(marker=dict(size=5), selector=dict(mode="markers"))
            fig.update_layout(
                title=(f"{pretty(y_col)} = {slope:.3g} × {pretty(x_col)} + {intercept:.3g}"
                       f"  |  R² = {r_squared:.3f}, p = {p_value:.2e}, "
                       f"N = {len(sub):,}")
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(
                f"**Pearson**: r = {r_pearson:.3f}, p = {p_pearson:.2e}  \n"
                f"**Spearman**: ρ = {r_spearman:.3f}, p = {p_spearman:.2e}"
            )


# ============================================================================
# Tab 4 : Distributions comparison (filtered vs all)
# ============================================================================

with tab_dist:
    st.subheader("Distribution comparison: filtered subset vs full dataset")
    st.caption(
        "Filter a subset using a condition (e.g. waters flagged as "
        "'scaling by Langelier but corrosive by Ryznar', cf. Article Figure 15), "
        "and compare its distributions against the whole dataset."
    )

    with st.expander("ℹ️ How to use"):
        st.markdown(
            "1. Build a filter using 1 or 2 numeric conditions on the left.\n"
            "2. Select which variables to display on the right.\n"
            "3. For each variable, a histogram of the filtered subset (red) is "
            "overlaid on the full dataset (blue) — with means reported.\n\n"
            "**Example** — reproduce Article Figure 15:\n"
            "- Condition 1: `IL > 0`\n"
            "- Condition 2: `ryznar > 6.8`\n"
            "- Variables: `PH`, `CALCIUM`, `HYDROGENOCARBONATES`"
        )

    left, right = st.columns([1, 2])

    with left:
        st.markdown("**Filter conditions**")

        # Condition 1
        f1_col = st.selectbox("Variable 1", num_cols,
                              index=num_cols.index("IL") if "IL" in num_cols else 0,
                              key="f1_col", format_func=pretty)
        f1_op  = st.selectbox("Op 1", [">", ">=", "<", "<=", "==", "!="], index=0, key="f1_op")
        f1_val = st.number_input("Value 1", value=0.0, key="f1_val")

        use_f2 = st.checkbox("Add a second condition", value=True)
        if use_f2:
            f2_col = st.selectbox("Variable 2", num_cols,
                                  index=num_cols.index("ryznar") if "ryznar" in num_cols else 0,
                                  key="f2_col", format_func=pretty)
            f2_op  = st.selectbox("Op 2", [">", ">=", "<", "<=", "==", "!="], index=0, key="f2_op")
            f2_val = st.number_input("Value 2", value=6.8, key="f2_val")

    def apply_condition(s: pd.Series, op: str, val: float) -> pd.Series:
        return {
            ">": s > val, ">=": s >= val,
            "<": s < val, "<=": s <= val,
            "==": s == val, "!=": s != val,
        }[op]

    mask = apply_condition(df[f1_col], f1_op, f1_val)
    if use_f2:
        mask &= apply_condition(df[f2_col], f2_op, f2_val)
    n_filt = int(mask.sum())
    n_tot  = len(df)

    with right:
        filter_expr = f"{f1_col} {f1_op} {f1_val}"
        if use_f2:
            filter_expr += f" AND {f2_col} {f2_op} {f2_val}"
        st.markdown(
            f"**Filter**: `{filter_expr}`  →  "
            f"**{n_filt:,} / {n_tot:,}** rows match "
            f"({100 * n_filt / max(n_tot, 1):.1f}%)"
        )

        default_vars = [c for c in ["PH ", "CALCIUM", "HYDROGENOCARBONATES"] if c in num_cols]
        dist_vars = st.multiselect(
            "Variables to display",
            num_cols,
            default=default_vars or num_cols[:3],
        )

        normalize = st.checkbox("Normalize histograms (density)", value=True)

    if n_filt < 5:
        st.warning("Filtered subset has fewer than 5 rows — relax the filter.")
    elif dist_vars:
        from plotly.subplots import make_subplots

        n = len(dist_vars)
        fig = make_subplots(rows=1, cols=n,
                            subplot_titles=[pretty(v) for v in dist_vars],
                            horizontal_spacing=0.08)

        norm = "probability density" if normalize else None
        stats_rows = []
        for i, var in enumerate(dist_vars, 1):
            all_vals = df[var].dropna()
            sub_vals = df.loc[mask, var].dropna()
            if len(all_vals) == 0 or len(sub_vals) == 0:
                continue

            fig.add_trace(
                go.Histogram(
                    x=all_vals, name="All",
                    marker_color="#4C78A8",
                    opacity=0.55, histnorm=norm, nbinsx=40,
                    showlegend=(i == 1),
                ), row=1, col=i,
            )
            fig.add_trace(
                go.Histogram(
                    x=sub_vals, name="Filtered",
                    marker_color="#E45756",
                    opacity=0.65, histnorm=norm, nbinsx=40,
                    showlegend=(i == 1),
                ), row=1, col=i,
            )

            # Means as vertical lines
            m_all = all_vals.mean(); m_sub = sub_vals.mean()
            fig.add_vline(x=m_all, line_dash="dash", line_color="#4C78A8",
                          row=1, col=i, annotation_text=f"μ={m_all:.1f}",
                          annotation_position="top left")
            fig.add_vline(x=m_sub, line_dash="dash", line_color="#E45756",
                          row=1, col=i, annotation_text=f"μ={m_sub:.1f}",
                          annotation_position="top right")

            # Mann-Whitney U test
            u_stat, u_p = stats.mannwhitneyu(all_vals, sub_vals, alternative="two-sided")
            stats_rows.append({
                "Variable": pretty(var),
                "Mean (all)":      f"{m_all:.2f}",
                "Mean (filtered)": f"{m_sub:.2f}",
                "Median (all)":    f"{all_vals.median():.2f}",
                "Median (filt.)":  f"{sub_vals.median():.2f}",
                "Mann-Whitney p":  f"{u_p:.2e}",
            })

        fig.update_layout(
            barmode="overlay",
            height=500,
            title=f"Filtered (N={n_filt:,}) vs All (N={n_tot:,})",
        )
        st.plotly_chart(fig, use_container_width=True)

        if stats_rows:
            st.markdown("**Summary statistics**")
            st.dataframe(pd.DataFrame(stats_rows), use_container_width=True)


# ============================================================================
# Tab 5 : Piper diagram
# ============================================================================

def piper_ternary_coords(ca, mg, na_k, cl, so4, hco3):
    """Convert meq/L percentages to Piper diagram x,y coords.

    Standard layout:
      - Left ternary (cations):  Ca, Mg, Na+K
      - Right ternary (anions):  HCO3, Cl, SO4
      - Central diamond: projection of both
    """
    # Cation ternary (left): we use (Ca, Mg, Na+K) summing to 100
    # Anion ternary (right): (HCO3, Cl, SO4) summing to 100
    # Classical Piper layout has triangles apex-up, centered below the diamond.

    # Cation ternary: apex Mg at top; bottom-left Ca, bottom-right Na+K
    cx = (na_k + 0.5 * mg) * 0.01
    cy = (mg * np.sqrt(3) / 2) * 0.01

    # Anion ternary: apex SO4 at top; bottom-left HCO3, bottom-right Cl
    ax = (cl + 0.5 * so4) * 0.01 + 1.2  # shifted right by 1.2
    ay = (so4 * np.sqrt(3) / 2) * 0.01

    # Diamond: constructed from projections.
    # Standard formula: project cation (Na+K) and anion (SO4+Cl) to diamond space.
    # We use the widely-used formulas (see Piper 1944 / USGS docs).
    d_x = 0.5 + 0.5 * ((na_k - ca) + (cl - hco3)) * 0.01 + 0.3
    d_y = 0.5 + 0.5 * ((na_k + ca) + (cl + hco3)) * 0.0 + \
          0.5 * (mg + so4) * 0.01 + 0.8
    # Simpler and well-known: diamond x = (%Na+K + %Cl) / 2
    #                          diamond y = (%Mg + %SO4) / 2 (+ offset)
    dx = ((na_k + cl) / 2) * 0.01 + 0.2
    dy = ((mg + so4) / 2) * 0.01 * np.sqrt(3) + 1.0

    return cx, cy, ax, ay, dx, dy


with tab_piper:
    st.subheader("Piper diagram (hydrochemical facies)")
    st.caption(
        "Major ions are converted to milliequivalents (meq/L) and plotted "
        "on the classical Piper (1944) trilinear diagram."
    )

    required_piper = ["CALCIUM", "MAGNESIUM", "SODIUM", "POTASSIUM",
                      "HYDROGENOCARBONATES", "CHLORURES", "SULFATES"]
    # Accept accented variants
    alt = {"MAGNESIUM": ["MAGNÉSIUM"], "HYDROGENOCARBONATES": ["HYDROGÉNOCARBONATES"]}

    def pick(df, name):
        if name in df.columns:
            return name
        for a in alt.get(name, []):
            if a in df.columns:
                return a
        return None

    names = {n: pick(df, n) for n in required_piper}
    missing = [n for n, c in names.items() if c is None]

    if missing:
        st.error(
            "The Piper diagram requires all major ions. Missing columns: "
            + ", ".join(missing)
            + "\n\nTip: re-run `sise_stats.py run` with all major ions enabled."
        )
    else:
        c1, c2 = st.columns([1, 2])
        with c1:
            color_opts = ["(none)"] + num_cols + cat_cols
            color_by = st.selectbox("Color by", color_opts, index=0, key="piper_color", format_func=pretty)
            n_max = st.slider("Max points", 500, min(20000, len(df)),
                              min(5000, len(df)), step=500)
            sample = df.sample(min(n_max, len(df)), random_state=42)

        # --- meq/L conversion ---
        # Atomic weights (g/mol) divided by absolute charge
        meq_factors = {
            "CALCIUM":             40.08 / 2,
            "MAGNESIUM":           24.31 / 2,
            "MAGNÉSIUM":           24.31 / 2,
            "SODIUM":              22.99 / 1,
            "POTASSIUM":           39.10 / 1,
            "HYDROGENOCARBONATES": 61.02 / 1,
            "HYDROGÉNOCARBONATES": 61.02 / 1,
            "CHLORURES":           35.45 / 1,
            "SULFATES":            96.06 / 2,
        }

        meq = pd.DataFrame(index=sample.index)
        for target, actual in names.items():
            f = meq_factors.get(actual, meq_factors.get(target, 1))
            meq[target] = pd.to_numeric(sample[actual], errors="coerce") / f

        meq = meq.dropna()
        if len(meq) < 2:
            st.warning("Not enough valid rows after meq/L conversion.")
        else:
            # --- Percentages of each cation / anion over their respective sum ---
            cat_sum = meq["CALCIUM"] + meq["MAGNESIUM"] + meq["SODIUM"] + meq["POTASSIUM"]
            an_sum  = meq["HYDROGENOCARBONATES"] + meq["CHLORURES"] + meq["SULFATES"]
            # Drop rows with null sums
            valid = (cat_sum > 0) & (an_sum > 0)
            meq = meq[valid]
            cat_sum = cat_sum[valid]
            an_sum  = an_sum[valid]

            # Fractions (0-1) — normalized
            ca   = (meq["CALCIUM"] / cat_sum).values
            mg   = (meq["MAGNESIUM"] / cat_sum).values
            nak  = ((meq["SODIUM"] + meq["POTASSIUM"]) / cat_sum).values
            hco3 = (meq["HYDROGENOCARBONATES"] / an_sum).values
            cl   = (meq["CHLORURES"] / an_sum).values
            so4  = (meq["SULFATES"] / an_sum).values

            # =====================================================
            # Canonical Piper diagram geometry (Piper 1944/Hill 1940)
            # =====================================================
            # Triangles of side L=1, height h = sqrt(3)/2
            # Cation triangle: base from (0,0) to (1,0), apex at (0.5, h)
            #   - Ca    -> bottom-left  (0, 0)
            #   - Na+K  -> bottom-right (1, 0)
            #   - Mg    -> apex         (0.5, h)
            # Anion triangle: shifted right by OFF = 2
            # Diamond: apex down at (1.5, h), apex up at (1.5, 3h),
            #          left apex (1, 2h), right apex (2, 2h)
            # -----------------------------------------------------
            h = np.sqrt(3) / 2
            OFF = 2.0            # horizontal offset between triangle centers
            GAP_Y = 0.25         # vertical gap between diamond bottom and triangle tops

            # Cation point in left triangle
            # Ca=(0,0), NaK=(1,0), Mg=(0.5, h)  -> barycentric
            cx = ca * 0 + nak * 1 + mg * 0.5
            cy = ca * 0 + nak * 0 + mg * h

            # Anion point in right triangle (shifted by OFF)
            # HCO3=(OFF,0), Cl=(OFF+1,0), SO4=(OFF+0.5,h)
            ax_ = hco3 * OFF + cl * (OFF + 1) + so4 * (OFF + 0.5)
            ay_ = hco3 * 0 + cl * 0 + so4 * h

            # Diamond projection (canonical Piper 1944 construction).
            # From the cation point, draw a line with slope +sqrt(3) (i.e. parallel
            # to the diamond's lower-left edge). From the anion point, draw a line
            # with slope -sqrt(3) (parallel to the lower-right edge). Their
            # intersection is the diamond position.
            # Solving the system analytically:
            sq3 = np.sqrt(3)
            dx = (ay_ - cy) / (2 * sq3) + (ax_ + cx) / 2
            dy = cy + sq3 * (dx - cx)

            with c2:
                fig = go.Figure()

                # --- Triangle outlines ---
                tri_color = "rgba(255,255,255,0.18)"
                # Cation triangle
                fig.add_trace(go.Scatter(
                    x=[0, 1, 0.5, 0], y=[0, 0, h, 0],
                    mode="lines", line=dict(color=tri_color, width=1.5),
                    hoverinfo="skip", showlegend=False,
                ))
                # Anion triangle
                fig.add_trace(go.Scatter(
                    x=[OFF, OFF + 1, OFF + 0.5, OFF],
                    y=[0, 0, h, 0],
                    mode="lines", line=dict(color=tri_color, width=1.5),
                    hoverinfo="skip", showlegend=False,
                ))
                # Diamond: 4 corners
                #   bottom apex at (1.5, h)
                #   left  apex at (1.0, 2h)
                #   top   apex at (1.5, 3h)
                #   right apex at (2.0, 2h)
                diamond_x = [1.5, 1.0, 1.5, 2.0, 1.5]
                diamond_y = [h,   2*h, 3*h, 2*h, h]
                fig.add_trace(go.Scatter(
                    x=diamond_x, y=diamond_y,
                    mode="lines", line=dict(color=tri_color, width=1.5),
                    hoverinfo="skip", showlegend=False,
                ))

                # --- Gridlines inside each shape (every 20%) ---
                # Principle: for a triangle ABC, a line parallel to side BC at
                # proportion t (0=A, 1=BC) goes from A+t*(B-A) to A+t*(C-A).

                def add_triangle_grid(A, B, C, step=0.2, color="rgba(255,255,255,0.06)"):
                    """Draw the 3 families of parallel gridlines inside a triangle."""
                    A, B, C = np.array(A), np.array(B), np.array(C)
                    for t in np.arange(step, 1.0 - 1e-9, step):
                        # Lines parallel to BC (opposite of A)
                        P1 = A + t * (B - A)
                        P2 = A + t * (C - A)
                        fig.add_trace(go.Scatter(
                            x=[P1[0], P2[0]], y=[P1[1], P2[1]],
                            mode="lines", line=dict(color=color, width=0.7),
                            hoverinfo="skip", showlegend=False,
                        ))
                        # Lines parallel to AC (opposite of B)
                        P1 = B + t * (A - B)
                        P2 = B + t * (C - B)
                        fig.add_trace(go.Scatter(
                            x=[P1[0], P2[0]], y=[P1[1], P2[1]],
                            mode="lines", line=dict(color=color, width=0.7),
                            hoverinfo="skip", showlegend=False,
                        ))
                        # Lines parallel to AB (opposite of C)
                        P1 = C + t * (A - C)
                        P2 = C + t * (B - C)
                        fig.add_trace(go.Scatter(
                            x=[P1[0], P2[0]], y=[P1[1], P2[1]],
                            mode="lines", line=dict(color=color, width=0.7),
                            hoverinfo="skip", showlegend=False,
                        ))

                # Cation triangle: Ca (bottom-left), Na+K (bottom-right), Mg (apex)
                add_triangle_grid((0, 0), (1, 0), (0.5, h))
                # Anion triangle: HCO3 (bottom-left), Cl (bottom-right), SO4 (apex)
                add_triangle_grid((OFF, 0), (OFF + 1, 0), (OFF + 0.5, h))

                # Diamond: 4 families of parallel lines (2 pairs).
                # The diamond has corners (bottom, left, top, right).
                # Family 1: parallel to the bottom-left edge -> lines from
                #           bottom->right-edge to left->top-edge at fraction t.
                # Family 2: parallel to the bottom-right edge -> lines from
                #           bottom->left-edge to right->top-edge at fraction t.
                B = np.array([1.5, h])      # bottom
                L = np.array([1.0, 2*h])    # left
                T = np.array([1.5, 3*h])    # top
                R = np.array([2.0, 2*h])    # right

                for t in np.arange(0.2, 1.0 - 1e-9, 0.2):
                    # Family 1: parallel to B-L
                    P1 = B + t * (R - B)   # on bottom-right edge
                    P2 = L + t * (T - L)   # on left-top edge
                    fig.add_trace(go.Scatter(
                        x=[P1[0], P2[0]], y=[P1[1], P2[1]],
                        mode="lines", line=dict(color="rgba(255,255,255,0.06)", width=0.7),
                        hoverinfo="skip", showlegend=False,
                    ))
                    # Family 2: parallel to B-R
                    P1 = B + t * (L - B)   # on bottom-left edge
                    P2 = R + t * (T - R)   # on right-top edge
                    fig.add_trace(go.Scatter(
                        x=[P1[0], P2[0]], y=[P1[1], P2[1]],
                        mode="lines", line=dict(color="rgba(255,255,255,0.06)", width=0.7),
                        hoverinfo="skip", showlegend=False,
                    ))

                # --- Hover text ---
                hover = [
                    f"Ca={c*100:.0f}% | Mg={m*100:.0f}% | Na+K={nk*100:.0f}%<br>"
                    f"HCO3={hc*100:.0f}% | Cl={cl_*100:.0f}% | SO4={s*100:.0f}%"
                    for c, m, nk, hc, cl_, s in zip(ca, mg, nak, hco3, cl, so4)
                ]

                # --- Color data ---
                color_values = None
                colorscale = None
                colorbar = None
                if color_by != "(none)" and color_by in sample.columns:
                    color_data = sample.loc[meq.index, color_by]
                    # Use the same robust numeric detection as Tab 2
                    if _is_truly_numeric(color_data):
                        color_values = pd.to_numeric(color_data, errors="coerce").values
                        colorscale = "Viridis"
                        colorbar = dict(title=pretty(color_by))
                    else:
                        # Categorical -> map to integers
                        cats = color_data.astype(str).astype("category")
                        color_values = cats.cat.codes.values
                        colorscale = "Turbo"
                        colorbar = dict(
                            title=pretty(color_by),
                            tickvals=list(range(len(cats.cat.categories))),
                            ticktext=list(cats.cat.categories),
                        )

                def scatter_trace(x, y, showcbar=False):
                    return go.Scatter(
                        x=x, y=y, mode="markers",
                        text=hover, hovertemplate="%{text}<extra></extra>",
                        marker=dict(
                            size=5,
                            opacity=0.55,
                            color=color_values if color_values is not None else "#1f77b4",
                            colorscale=colorscale,
                            colorbar=colorbar if showcbar else None,
                            showscale=showcbar,
                        ),
                        showlegend=False,
                    )

                fig.add_trace(scatter_trace(cx, cy, showcbar=False))
                fig.add_trace(scatter_trace(ax_, ay_, showcbar=False))
                fig.add_trace(scatter_trace(dx, dy, showcbar=(color_values is not None)))

                # --- Corner labels ---
                lbl = dict(size=14, color="#fafafa")
                # Cation triangle
                fig.add_annotation(x=0,    y=-0.06, text="<b>Ca²⁺</b>",   showarrow=False, font=lbl)
                fig.add_annotation(x=1,    y=-0.06, text="<b>Na⁺+K⁺</b>", showarrow=False, font=lbl)
                fig.add_annotation(x=0.5,  y=h + 0.06, text="<b>Mg²⁺</b>",showarrow=False, font=lbl)
                # Anion triangle
                fig.add_annotation(x=OFF,      y=-0.06, text="<b>HCO₃⁻</b>", showarrow=False, font=lbl)
                fig.add_annotation(x=OFF + 1,  y=-0.06, text="<b>Cl⁻</b>",   showarrow=False, font=lbl)
                fig.add_annotation(x=OFF+0.5,  y=h + 0.06, text="<b>SO₄²⁻</b>", showarrow=False, font=lbl)
                # Diamond corners
                fig.add_annotation(x=1.0 - 0.05, y=2*h, text="<b>Ca+Mg</b>",     showarrow=False, font=lbl, xanchor="right")
                fig.add_annotation(x=2.0 + 0.05, y=2*h, text="<b>Na+K+Cl+SO₄</b>", showarrow=False, font=lbl, xanchor="left")
                fig.add_annotation(x=1.5, y=3*h + 0.06, text="<b>SO₄+Cl</b>",   showarrow=False, font=lbl)
                fig.add_annotation(x=1.5, y=h - 0.06, text="<b>Ca+HCO₃</b>",    showarrow=False, font=lbl)

                fig.update_layout(
                    title=f"Piper diagram — N = {len(meq):,}",
                    xaxis=dict(visible=False, range=[-0.25, OFF + 1.25],
                               scaleanchor="y", scaleratio=1),
                    yaxis=dict(visible=False, range=[-0.15, 3*h + 0.25]),
                    height=750, showlegend=False,
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig, use_container_width=True)

                st.caption(
                    "Left triangle: cation facies (Ca, Mg, Na+K). "
                    "Right triangle: anion facies (HCO₃, Cl, SO₄). "
                    "Upper diamond: combined hydrochemical facies. "
                    "Each point is plotted three times (once in each sub-plot)."
                )


# ============================================================================
# Tab 6 : 3D response surface (Langelier & Ryznar vs pH, Ca, CO2)
# ============================================================================

with tab_3d:
    st.subheader("3D scatter plot")
    st.caption(
        "Explore relationships between three variables in 3D. "
        "Reproduces the spirit of Article Figure 16 (response surfaces for "
        "Langelier / Ryznar vs pH, Ca, CO₂)."
    )

    c1, c2, c3, c4 = st.columns(4)
    x3 = c1.selectbox("X", num_cols,
                      index=num_cols.index("PH ") if "PH " in num_cols else 0, key="3d_x", format_func=pretty)
    y3 = c2.selectbox("Y", num_cols,
                      index=num_cols.index("CALCIUM") if "CALCIUM" in num_cols else 1, key="3d_y", format_func=pretty)
    z3 = c3.selectbox("Z", num_cols,
                      index=num_cols.index("IL") if "IL" in num_cols else 2, key="3d_z", format_func=pretty)
    color3_options = ["(none)"] + num_cols + cat_cols
    color3 = c4.selectbox("Color by", color3_options,
                          index=color3_options.index("ryznar") if "ryznar" in color3_options else 0,
                          key="3d_color", format_func=pretty)

    n_max = st.slider("Max points (3D gets slow above ~5000)", 500, min(15000, len(df)),
                      min(3000, len(df)), step=500)
    # Deduplicate (handles X==Y or Z==color etc.)
    wanted = [x3, y3, z3]
    if color3 != "(none)":
        wanted.append(color3)
    unique_cols3 = list(dict.fromkeys(wanted))
    sub3 = df[unique_cols3].dropna()
    if len(sub3) > n_max:
        sub3 = sub3.sample(n_max, random_state=42)

    st.caption(f"N = {len(sub3):,} points displayed")

    if len({x3, y3, z3}) < 3:
        st.warning("X, Y and Z must be three different variables.")
    elif len(sub3) < 2:
        st.warning("Not enough data.")
    else:
        color_arg = None if color3 == "(none)" else color3
        fig = px.scatter_3d(
            sub3, x=x3, y=y3, z=z3,
            color=color_arg,
            color_continuous_scale="Viridis",
            opacity=0.7,
            height=700,
            labels={x3: pretty(x3), y3: pretty(y3), z3: pretty(z3),
                    **({color3: pretty(color3)} if color_arg else {})},
        )
        fig.update_traces(marker=dict(size=3), selector=dict(type="scatter3d"))
        fig.update_layout(
            title=f"{pretty(z3)} vs ({pretty(x3)}, {pretty(y3)})"
                  + (f" — color: {pretty(color3)}" if color_arg else "")
                  + f"  |  N = {len(sub3):,}",
            scene=dict(
                xaxis_title=pretty(x3),
                yaxis_title=pretty(y3),
                zaxis_title=pretty(z3),
            ),
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Optional: fit a response surface Z = f(X, Y)"):
            st.markdown(
                "Fits a quadratic surface `Z = a + bX + cY + dX² + eY² + fXY` "
                "to the data and overlays it on the scatter."
            )
            if st.button("Fit surface"):
                X = sub3[[x3, y3]].values
                Y = sub3[z3].values
                # quadratic design matrix
                A = np.column_stack([
                    np.ones(len(X)), X[:, 0], X[:, 1],
                    X[:, 0] ** 2, X[:, 1] ** 2, X[:, 0] * X[:, 1]
                ])
                coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
                yhat = A @ coef
                ss_res = np.sum((Y - yhat) ** 2)
                ss_tot = np.sum((Y - Y.mean()) ** 2)
                r2 = 1 - ss_res / ss_tot
                st.success(f"R² of the quadratic fit: {r2:.3f}")

                # Surface mesh
                gx = np.linspace(X[:, 0].min(), X[:, 0].max(), 30)
                gy = np.linspace(X[:, 1].min(), X[:, 1].max(), 30)
                gX, gY = np.meshgrid(gx, gy)
                gZ = (coef[0] + coef[1] * gX + coef[2] * gY
                      + coef[3] * gX ** 2 + coef[4] * gY ** 2 + coef[5] * gX * gY)
                fig.add_trace(go.Surface(x=gx, y=gy, z=gZ,
                                         opacity=0.35, showscale=False,
                                         colorscale="Greys", name="Fit"))
                st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# Tab 7 : Maps (choropleth on French communes)
# ============================================================================

# Default candidate path for the communes GeoJSON in the project root
DEFAULT_GEOJSON = "communes.geojson"
CACHE_GEOJSON   = ".cache/communes.geojson"
GEOJSON_URL     = (
    "https://github.com/MaelDes/SISE_Eaux_Spatial_Distribution/"
    "releases/download/v1.0/communes_simplified.geojson"
)


def _resolve_geojson_path(user_path: str) -> str | None:
    """
    Resolve the GeoJSON path with priority:
      1. The user-provided local path
      2. The internal cache from a previous download
      3. Download from GEOJSON_URL into the cache
    """
    import urllib.request, urllib.error

    if user_path and Path(user_path).exists():
        return user_path

    cache = Path(CACHE_GEOJSON)
    if cache.exists() and cache.stat().st_size > 100_000:
        return str(cache)

    if "<your" in GEOJSON_URL:
        return None

    cache.parent.mkdir(parents=True, exist_ok=True)
    try:
        with st.spinner(
            "First-time setup: downloading commune boundaries (~30 MB, one-time only)..."
        ):
            urllib.request.urlretrieve(GEOJSON_URL, cache)
        return str(cache)
    except (urllib.error.URLError, OSError) as e:
        st.error(
            f"Could not download the GeoJSON.\n\n"
            f"**URL:** `{GEOJSON_URL}`\n\n"
            f"**Error:** `{e}`"
        )
        return None


@st.cache_resource(show_spinner=False)
def _load_geojson(path: str) -> dict | None:
    """
    Load the communes GeoJSON ONCE per session. We use cache_resource (not
    cache_data) so the dict is kept in memory across reruns and not
    re-serialized to the browser at every interaction.
    """
    import json
    p = Path(path)
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def _precompute_centroids(path: str, _version: int = 3) -> pd.DataFrame | None:
    """
    Compute commune centroids once per session.
    Returns a small DataFrame (~35k rows × 4 cols, ~1 MB) instead of the
    heavy GeoJSON (10-30 MB) — used by the fast "points" rendering path.

    Auto-detects the INSEE code field by inspecting the first feature.

    The `_version` argument is used to invalidate the Streamlit cache when
    the function logic changes — bump it whenever you change this code.
    """
    import json, re
    p = Path(path)
    if not p.exists():
        return None

    with open(p, encoding="utf-8") as f:
        geojson = json.load(f)

    features = geojson.get("features", [])
    if not features:
        return None

    # --- Auto-detect the INSEE code field across the first 50 features ---
    pattern = re.compile(r"^[\dA-Za-z]{5}$")
    candidate_keys = {}
    for feat in features[:50]:
        props = feat.get("properties") or {}
        for k, v in props.items():
            if v is None:
                continue
            if pattern.match(str(v).strip()):
                candidate_keys[k] = candidate_keys.get(k, 0) + 1
    # Prefer keys whose name suggests an INSEE code, then by frequency
    def _key_score(k):
        name_bonus = 100 if re.search(r"code|insee|geo", k, re.I) else 0
        return name_bonus + candidate_keys[k]
    insee_key = max(candidate_keys, key=_key_score) if candidate_keys else None

    # Find a "name" field
    first_props = features[0].get("properties") or {}
    name_key = None
    for k in ("nom", "NOM_COM", "name", "libelle", "nom_com"):
        if k in first_props:
            name_key = k
            break

    # Fallback: top-level id
    use_top_id = (insee_key is None) and any(f.get("id") for f in features[:5])

    rows = []
    for feat in features:
        props = feat.get("properties") or {}
        if insee_key:
            code = props.get(insee_key)
        elif use_top_id:
            code = feat.get("id")
        else:
            code = None
        if code is None:
            continue

        geom = feat.get("geometry") or {}
        coords = geom.get("coordinates")
        if not coords:
            continue
        gtype = geom.get("type")
        # Extract a list of (lon, lat) pairs from the geometry
        if gtype == "Polygon":
            # coords = [outer_ring, hole1, hole2, ...] — take outer ring
            pts = coords[0] if coords else None
        elif gtype == "MultiPolygon":
            # coords = [poly1, poly2, ...], each poly = [outer_ring, ...]
            try:
                largest = max(coords, key=lambda poly: len(poly[0]) if poly and poly[0] else 0)
                pts = largest[0] if largest else None
            except (TypeError, IndexError):
                pts = None
        else:
            continue
        if not pts:
            continue
        try:
            arr = np.asarray(pts, dtype=float)
            if arr.ndim != 2 or arr.shape[1] < 2:
                continue
            lon, lat = float(arr[:, 0].mean()), float(arr[:, 1].mean())
        except Exception:
            continue
        name = (props.get(name_key) if name_key else "") or ""
        rows.append((str(code).strip().zfill(5), name, lon, lat))

    if not rows:
        return None
    return pd.DataFrame(rows, columns=["code", "name", "lon", "lat"])


@st.cache_data(show_spinner=False)
def _param_config() -> dict:
    """
    Return colorscale config per parameter.
    We try to import the canonical PARAM_CONFIG from sise_pipeline.
    If that fails (e.g. running standalone), we use a sensible default.
    """
    try:
        from sise_pipeline import PARAM_CONFIG
        return PARAM_CONFIG
    except Exception:
        return {
            "PH ":                 {"label": "pH",                 "cmin": 5,    "cmax": 10},
            "CALCIUM":             {"label": "Calcium (mg/L)",     "cmin": 0,    "cmax": 150},
            "MAGNESIUM":           {"label": "Magnesium (mg/L)",   "cmin": 0,    "cmax": 60},
            "MAGNÉSIUM":           {"label": "Magnesium (mg/L)",   "cmin": 0,    "cmax": 60},
            "HYDROGENOCARBONATES": {"label": "HCO3 (mg/L)",        "cmin": 0,    "cmax": 500},
            "HYDROGÉNOCARBONATES": {"label": "HCO3 (mg/L)",        "cmin": 0,    "cmax": 500},
            "SULFATES":            {"label": "Sulfates (mg/L)",    "cmin": 0,    "cmax": 80},
            "CHLORURES":           {"label": "Chlorides (mg/L)",   "cmin": 0,    "cmax": 100},
            "NITRATES (EN NO3)":   {"label": "Nitrates (mg/L)",    "cmin": 0,    "cmax": 60},
            "IL":                  {"label": "Langelier SI",       "cmin": -3,   "cmax": 2},
            "IL_calc":             {"label": "Langelier SI (calc)","cmin": -3,   "cmax": 2},
            "ryznar":              {"label": "Ryznar SI",          "cmin": 5,    "cmax": 13},
            "Bason":               {"label": "Basson Index",       "cmin": -200, "cmax": 1200},
            "Larson":              {"label": "Larson-Skold",       "cmin": 0,    "cmax": 3},
        }


def _is_index_param(param: str) -> bool:
    """Indices use a diverging colorscale (centered on equilibrium value)."""
    return param in {"IL", "IL_calc", "ryznar", "Larson", "Bason"}


def _index_zero(param: str) -> float | None:
    """Equilibrium value for diverging colorscales."""
    return {
        "IL": 0.0, "IL_calc": 0.0,
        "ryznar": 6.5,    # midpoint of the balanced zone (6.2–6.8)
        "Larson": 0.5,    # threshold low/moderate corrosion
        "Bason": 300.0,   # threshold balanced/moderate corrosion
    }.get(param)


@st.cache_data(show_spinner=False)
def _aggregate_for_map(
    df_pickle_key: str,
    df: pd.DataFrame,
    param: str,
    year: int | None,
    code_col: str,
) -> pd.DataFrame:
    """
    Aggregate the dataframe to (commune, mean_value, n_measures) for one
    parameter and optionally one year. The first arg is a pickle-key just to
    let Streamlit invalidate the cache when df changes.
    """
    work = df[[code_col, param]].copy()
    if year is not None and "Year" in df.columns:
        work = work.assign(Year=df["Year"])
        work = work[work["Year"] == year]
        work = work.drop(columns=["Year"])
    work = work.dropna(subset=[param])
    if work.empty:
        return work

    grouped = (
        work.groupby(code_col, as_index=False)
            .agg(value=(param, "mean"), n=(param, "size"))
    )
    grouped = grouped.rename(columns={code_col: "code"})
    # Pad INSEE codes to 5 chars to match GeoJSON (e.g. 1001 -> "01001")
    grouped["code"] = grouped["code"].astype(str).str.zfill(5)
    return grouped


with tab_map:
    st.markdown("""
    <div style='margin-bottom: 1rem;'>
      <div style='font-size: 11px; color: #a78bfa; font-weight: 600; letter-spacing: 0.12em;
                  text-transform: uppercase; margin-bottom: 0.4rem;'>
        Geographic visualisation
      </div>
      <h2 style='margin-top: 0 !important;'>Interactive choropleth map</h2>
      <p style='color: #a1a1aa; font-size: 13px; margin: 0.25rem 0 0 0;'>
        Mean value per commune for the selected parameter and year.
        Built directly from the loaded dataset — no pre-computation needed.
      </p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Advanced — GeoJSON source", expanded=False):
        st.caption(
            "By default, the app uses `communes.geojson` if present locally, "
            "otherwise downloads a simplified version automatically. Change "
            "only if you have a custom file."
        )
        geojson_path = st.text_input(
            "Custom GeoJSON path (optional)",
            value=DEFAULT_GEOJSON,
            key="map_geojson_path",
        )

    resolved_geojson = _resolve_geojson_path(geojson_path)
    if resolved_geojson is None:
        st.warning("GeoJSON not available. Place a `communes.geojson` file at the project root.")
        st.stop()
    geojson = _load_geojson(resolved_geojson)
    centroids = _precompute_centroids(resolved_geojson)

    if geojson is None:
        st.warning(
            f"GeoJSON not found at `{resolved_geojson}`. "
        )
    elif centroids is None or len(centroids) == 0:
        # Show a debug panel so the user can see what went wrong
        st.error(
            "Could not extract commune centroids from the GeoJSON. "
            "The Fast (points) mode requires this — switch to Detailed (polygons) for now."
        )
        with st.expander("Debug — GeoJSON structure"):
            sample = geojson["features"][0] if geojson.get("features") else {}
            st.write("**First feature properties keys:**")
            st.code(list((sample.get("properties") or {}).keys()))
            st.write("**First feature properties (first 10):**")
            st.code({k: v for k, v in (sample.get("properties") or {}).items()
                     if k in list((sample.get("properties") or {}).keys())[:10]})
            st.write("**Geometry type:**", sample.get("geometry", {}).get("type"))
            st.write("**Top-level `id` field:**", sample.get("id", "(none)"))
            st.info(
                "Send me the keys above and I will adapt the centroid extraction "
                "to your specific GeoJSON format."
            )
    else:
        # Identify INSEE code column in the dataframe
        code_col = next(
            (c for c in ("inseecommuneprinc", "code_insee", "INSEE_COM", "code")
             if c in df.columns),
            None,
        )
        if code_col is None:
            st.error(
                "No INSEE code column found in the dataset. "
                "Expected one of: `inseecommuneprinc`, `code_insee`, `INSEE_COM`, `code`."
            )
        else:
            # Identify GeoJSON feature key
            sample_props = geojson["features"][0].get("properties", {})
            if "code" in sample_props:
                feature_key = "properties.code"
            elif "INSEE_COM" in sample_props:
                feature_key = "properties.INSEE_COM"
            elif "codgeo" in sample_props:
                feature_key = "properties.codgeo"
            else:
                feature_key = "id"

            # --- Controls ---
            cfg = _param_config()
            available_params = [c for c in num_cols if c in cfg or c.upper() in cfg]
            chem_first = sorted([p for p in available_params if not _is_index_param(p)])
            indices    = sorted([p for p in available_params if     _is_index_param(p)])
            ordered_params = chem_first + indices

            ctl_param, ctl_year, ctl_mode = st.columns([2, 2, 1.4])
            with ctl_param:
                map_param = st.selectbox(
                    "Parameter",
                    ordered_params,
                    index=ordered_params.index("CALCIUM") if "CALCIUM" in ordered_params else 0,
                    key="map_param",
                    format_func=pretty,
                )
            with ctl_year:
                if "Year" in df.columns:
                    years = sorted(df["Year"].dropna().unique().astype(int))
                    if len(years) > 1:
                        map_year = st.select_slider(
                            "Year",
                            options=["All years"] + list(years),
                            value=years[-1],
                            key="map_year",
                        )
                        map_year = None if map_year == "All years" else int(map_year)
                    else:
                        map_year = int(years[0]) if years else None
                        st.caption(f"Single year: {map_year}")
                else:
                    map_year = None
            with ctl_mode:
                render_mode = st.radio(
                    "Render",
                    ["Fast (points)", "Detailed (polygons)"],
                    index=0,
                    key="map_mode",
                    help="Fast = scatter on commune centroids (renders ~10× faster, "
                         "best for interactive exploration). Detailed = full polygons "
                         "(slower but prettier, use it for the final article figure).",
                )

            # --- Aggregate ---
            df_key = f"{len(df)}_{map_param}_{map_year}_{iqr_factor}"
            with st.spinner("Aggregating commune values..."):
                agg = _aggregate_for_map(df_key, df, map_param, map_year, code_col)

            if agg.empty:
                st.warning("No data available for this parameter / year combination.")
            else:
                # Normalize codes on both sides (strip, zfill, str)
                agg = agg.copy()
                agg["code"] = agg["code"].astype(str).str.strip().str.zfill(5)
                cent = centroids.copy()
                cent["code"] = cent["code"].astype(str).str.strip().str.zfill(5)

                # Merge centroids on INSEE code (used for Fast mode)
                agg_pts = agg.merge(cent, on="code", how="left").dropna(
                    subset=["lon", "lat"]
                )

                # Diagnostic if matching failed
                n_unmatched = len(agg) - len(agg_pts)
                if n_unmatched > 0 and len(agg_pts) == 0:
                    st.error(
                        f"None of the {len(agg):,} communes could be matched to the GeoJSON. "
                        "This usually means the INSEE code format differs between your data "
                        "and the GeoJSON file."
                    )
                    with st.expander("Debug info"):
                        st.write("First 5 codes from your data:")
                        st.code(agg["code"].head().tolist())
                        st.write("First 5 codes from the GeoJSON:")
                        st.code(cent["code"].head().tolist())
                    st.stop()
                elif n_unmatched > 0:
                    st.caption(
                        f"⚠️ {n_unmatched:,} communes could not be located in the GeoJSON "
                        f"(matched {len(agg_pts):,}/{len(agg):,})"
                    )

                # --- Color scale ---
                cfg_p = cfg.get(map_param, {"label": map_param, "cmin": None, "cmax": None})
                label = cfg_p["label"]
                vals = agg["value"]
                lo, hi = vals.quantile(0.02), vals.quantile(0.98)
                if cfg_p["cmin"] is not None: lo = max(lo, cfg_p["cmin"])
                if cfg_p["cmax"] is not None: hi = min(hi, cfg_p["cmax"])

                if _is_index_param(map_param):
                    z0 = _index_zero(map_param)
                    span = max(abs(lo - z0), abs(hi - z0))
                    cmin, cmax, cmid = z0 - span, z0 + span, z0
                    colorscale = "RdBu_r"
                else:
                    cmin, cmax, cmid = lo, hi, None
                    colorscale = "Viridis"

                # --- KPI cards ---
                k1, k2, k3 = st.columns(3)
                k1.metric("Communes", f"{len(agg):,}")
                k2.metric("Mean", f"{vals.mean():.2f}")
                k3.metric("Median", f"{vals.median():.2f}")

                # --- Build map ---
                title_year = f"— {map_year}" if map_year else "— all years pooled"
                title_text = (f"<b>{label}</b> {title_year}  ·  "
                              f"N = {agg['n'].sum():,} measurements  ·  "
                              f"{len(agg):,} communes")

                if render_mode.startswith("Fast"):
                    # Scattergeo on centroids — much faster, no GeoJSON sent to browser
                    with st.spinner(f"Rendering {len(agg_pts):,} points..."):
                        # Pack everything for hover into customdata
                        customdata = np.column_stack([
                            agg_pts["code"].values,
                            agg_pts["name"].fillna("").values,
                            agg_pts["n"].values,
                        ])
                        fig = go.Figure(go.Scattergeo(
                            lon=agg_pts["lon"].values,
                            lat=agg_pts["lat"].values,
                            mode="markers",
                            marker=dict(
                                size=4,
                                color=agg_pts["value"].values,
                                colorscale=colorscale,
                                cmin=cmin, cmax=cmax, cmid=cmid,
                                showscale=True,
                                colorbar=dict(
                                    title=dict(text=label, font=dict(size=12)),
                                    thickness=14, len=0.7, x=1.02,
                                ),
                                line=dict(width=0),
                            ),
                            customdata=customdata,
                            hovertemplate=(
                                "<b>%{customdata[1]}</b> (INSEE %{customdata[0]})<br>"
                                f"{label}: " + "%{marker.color:.2f}<br>"
                                "Measurements: %{customdata[2]}<extra></extra>"
                            ),
                        ))
                        fig.update_geos(
                            visible=False,
                            bgcolor="rgba(0,0,0,0)",
                            showframe=False,
                            showcoastlines=True,
                            coastlinecolor="rgba(255,255,255,0.15)",
                            showcountries=True,
                            countrycolor="rgba(255,255,255,0.1)",
                            projection=dict(type="mercator"),
                            lonaxis=dict(range=[-5.5, 9.7]),  # mainland France bounds
                            lataxis=dict(range=[41.0, 51.5]),
                        )
                        fig.update_layout(
                            title=title_text,
                            height=720,
                            margin=dict(l=0, r=0, t=50, b=0),
                        )
                else:
                    # Choropleth (slow but pretty)
                    with st.spinner(f"Rendering polygons for {len(agg):,} communes..."):
                        fig = go.Figure(go.Choropleth(
                            geojson=geojson,
                            locations=agg["code"],
                            featureidkey=feature_key,
                            z=agg["value"],
                            zmin=cmin, zmax=cmax, zmid=cmid,
                            colorscale=colorscale,
                            marker_line_width=0,
                            customdata=np.column_stack([agg["n"].values]),
                            hovertemplate=(
                                "<b>INSEE %{location}</b><br>"
                                f"{label}: " + "%{z:.2f}<br>"
                                "Measurements: %{customdata[0]}<extra></extra>"
                            ),
                            colorbar=dict(
                                title=dict(text=label, font=dict(size=12)),
                                thickness=14, len=0.7, x=1.02,
                            ),
                        ))
                        fig.update_geos(
                            fitbounds="locations",
                            visible=False,
                            bgcolor="rgba(0,0,0,0)",
                            showframe=False,
                        )
                        fig.update_layout(
                            title=title_text,
                            height=720,
                            margin=dict(l=0, r=0, t=50, b=0),
                            geo=dict(
                                projection=dict(type="mercator"),
                                scope="europe",
                                center=dict(lat=46.5, lon=2.5),
                            ),
                        )

                st.plotly_chart(fig, use_container_width=True)

                # --- Bottom panel : top extreme communes + download ---
                bot_l, bot_r = st.columns([3, 1])
                with bot_l:
                    with st.expander("Top 10 communes — highest values"):
                        top_high = (
                            agg_pts.nlargest(10, "value")
                                   .reset_index(drop=True)
                                   .rename(columns={"code": "INSEE",
                                                    "name": "Commune",
                                                    "value": label,
                                                    "n": "N measurements"})
                                   [["INSEE", "Commune", label, "N measurements"]]
                        )
                        st.dataframe(top_high, use_container_width=True, hide_index=True)
                    with st.expander("Top 10 communes — lowest values"):
                        top_low = (
                            agg_pts.nsmallest(10, "value")
                                   .reset_index(drop=True)
                                   .rename(columns={"code": "INSEE",
                                                    "name": "Commune",
                                                    "value": label,
                                                    "n": "N measurements"})
                                   [["INSEE", "Commune", label, "N measurements"]]
                        )
                        st.dataframe(top_low, use_container_width=True, hide_index=True)
                with bot_r:
                    st.markdown("**Export**")
                    st.download_button(
                        "Download map (HTML)",
                        fig.to_html(include_plotlyjs="cdn").encode("utf-8"),
                        file_name=f"map_{map_param}_{map_year or 'all'}.html",
                        mime="text/html",
                    )
                    st.download_button(
                        "Download data (CSV)",
                        agg_pts.to_csv(index=False).encode("utf-8"),
                        file_name=f"map_{map_param}_{map_year or 'all'}.csv",
                        mime="text/csv",
                    )

# ============================================================================
# Tab 8 : Commune — fast searchable city profile
# ============================================================================

# --- Helpers (defined here, used only in this tab to keep things local) ---

def _commune_find_insee_col(df: pd.DataFrame) -> str | None:
    """Locate the INSEE column without ever scanning column contents."""
    for c in ("inseecommuneprinc", "code_insee", "INSEE_COM", "code", "insee",
              "INSEE", "code_commune", "commune_code", "CODE_INSEE"):
        if c in df.columns:
            return c
    return None


def _commune_find_name_col(df: pd.DataFrame) -> str | None:
    """Locate the commune-name column."""
    for c in ("nomcommuneprinc", "nom_commune", "NOM_COM", "nom", "commune",
              "name", "libelle", "COMMUNE", "Commune"):
        if c in df.columns:
            return c
    return None


@st.cache_data(show_spinner=False, max_entries=2)
def _commune_index_from_pairs(pairs: tuple) -> pd.DataFrame:
    """Build commune lookup table from a small hashable tuple of (insee, name)."""
    import unicodedata
    if not pairs:
        return pd.DataFrame()
    df_idx = pd.DataFrame(pairs, columns=["insee", "name"])
    df_idx["insee"] = df_idx["insee"].astype(str).str.strip().str.zfill(5)
    df_idx["name"]  = df_idx["name"].astype(str).str.strip()
    # Build a normalized search key: lowercase + accent-stripped
    def _normalize(s: str) -> str:
        s = s.lower()
        s = unicodedata.normalize("NFKD", s)
        s = "".join(ch for ch in s if not unicodedata.combining(ch))
        return s
    df_idx["search_key"] = df_idx["name"].map(_normalize)
    df_idx = df_idx.drop_duplicates(subset=["insee"]).reset_index(drop=True)
    return df_idx


@st.cache_data(show_spinner=False, max_entries=2)
def _commune_national_means(numeric_pairs: tuple) -> dict:
    """
    Pre-compute the national mean for every numeric column once per session.
    Lookup is then O(1).
    """
    return dict(numeric_pairs)


with tab_commune:
    st.markdown("""
    <div style='margin-bottom: 1rem;'>
      <div style='font-size: 11px; color: #a78bfa; font-weight: 600; letter-spacing: 0.12em;
                  text-transform: uppercase; margin-bottom: 0.4rem;'>
        Commune profile
      </div>
      <h2 style='margin-top: 0 !important;'>Search a French commune</h2>
      <p style='color: #a1a1aa; font-size: 13px; margin: 0.25rem 0 0 0;'>
        Find any commune by name, see its full water composition, aggressiveness
        indices and geological context.
      </p>
    </div>
    """, unsafe_allow_html=True)

    insee_col = _commune_find_insee_col(df)
    name_col  = _commune_find_name_col(df)

    if insee_col is None:
        st.warning(
            "Cannot build the commune search: no INSEE code column was "
            "detected in the dataset."
        )
        with st.expander("Debug — available columns"):
            st.code(list(df.columns))
    else:
        # Build a *small* tuple of (insee, name) pairs, hashable & cacheable.
        # If the dataset itself has no name column, fall back to the GeoJSON
        # which always carries the commune name in its properties.
        pairs = None
        if name_col:
            small = (df[[insee_col, name_col]]
                     .dropna(subset=[insee_col])
                     .drop_duplicates(subset=[insee_col]))
            pairs = tuple(zip(small[insee_col].astype(str),
                              small[name_col].astype(str)))
        else:
            # Try to enrich from the GeoJSON centroids (which include `name`)
            try:
                resolved = _resolve_geojson_path(DEFAULT_GEOJSON)
                if resolved:
                    cent = _precompute_centroids(resolved)
                    if cent is not None and len(cent):
                        # Map INSEE -> name from the GeoJSON
                        cent_lookup = dict(zip(
                            cent["code"].astype(str).str.zfill(5),
                            cent["name"].astype(str),
                        ))
                        small = df[[insee_col]].dropna().drop_duplicates()
                        codes = small[insee_col].astype(str).str.strip().str.zfill(5)
                        names = codes.map(cent_lookup).fillna("(unknown)")
                        pairs = tuple(zip(codes, names))
                        st.caption(
                            "ℹ️ Commune names enriched from the GeoJSON file "
                            "(your dataset has only INSEE codes)."
                        )
            except Exception:
                pass
            # Last resort: just show INSEE codes
            if pairs is None:
                small = df[[insee_col]].dropna().drop_duplicates()
                pairs = tuple((str(v), f"INSEE {v}") for v in small[insee_col])

        commune_idx = _commune_index_from_pairs(pairs)

        # Pre-compute national means once
        nat_means = _commune_national_means(
            tuple((c, float(df[c].mean()))
                  for c in num_cols if c in df.columns)
        )

        # ---------- Search box ----------
        col_search, col_count = st.columns([3, 1])
        with col_search:
            query = st.text_input(
                f"Search among {len(commune_idx):,} communes",
                placeholder="Type a name, e.g. 'Lyon', 'Saint-Pont', or an INSEE code",
                key="commune_search_text",
            )
        with col_count:
            st.metric("Communes indexed", f"{len(commune_idx):,}")

        # ---------- Filtering (CHEAP, runs only when query changes) ----------
        if not query or len(query.strip()) < 2:
            st.info(
                "Type at least 2 characters in the search box above to see "
                "matching communes."
            )
        else:
            # Normalize the query the same way as the index (lowercase + strip accents)
            import unicodedata
            q = query.strip().lower()
            q = unicodedata.normalize("NFKD", q)
            q = "".join(ch for ch in q if not unicodedata.combining(ch))
            # Match on name OR insee — pandas is fast enough on 33k rows
            mask = (
                commune_idx["search_key"].str.contains(q, regex=False, na=False)
                | commune_idx["insee"].str.startswith(q)
            )
            matches = commune_idx[mask].head(10)  # cap at 10 results

            if len(matches) == 0:
                st.warning(f"No commune matches `{query}`.")
                with st.expander("🔍 Debug — what's in the index?", expanded=True):
                    st.markdown("**Sample of names in the dataset:**")
                    sample = commune_idx[["insee", "name"]].head(10)
                    st.dataframe(sample, hide_index=True, use_container_width=True)
                    # Try a substring-anywhere match for debugging
                    rough = commune_idx[
                        commune_idx["search_key"].str.contains(q[:3], regex=False, na=False)
                    ].head(5)
                    if len(rough):
                        st.markdown(f"**Names containing `{q[:3]}`:**")
                        st.dataframe(rough[["insee", "name"]], hide_index=True,
                                     use_container_width=True)
                    st.caption(
                        "If the names look weird (encoding issues, all uppercase, "
                        "extra suffixes...), copy-paste this back to me."
                    )
            else:
                # Build a tiny selectbox with only the 10 candidates
                options = [
                    f"{r['name']} ({r['insee']})"
                    for _, r in matches.iterrows()
                ]
                chosen = st.radio(
                    f"{len(matches)} match(es) — pick one:",
                    options,
                    key="commune_match_pick",
                    horizontal=False,
                )

                # Recover the INSEE from the chosen label
                chosen_insee = chosen.split("(")[-1].rstrip(")")
                chosen_name  = chosen.rsplit(" (", 1)[0]

                # ---------- Subset for this commune ----------
                # boolean mask is fast (33k comparisons), no need to cache
                m = df[insee_col].astype(str).str.strip().str.zfill(5) == chosen_insee
                sub = df[m]

                if len(sub) == 0:
                    st.warning("No measurements found for this commune.")
                else:
                    n_records = len(sub)
                    years_present = (
                        sorted(sub["Year"].dropna().unique().astype(int).tolist())
                        if "Year" in sub.columns else []
                    )
                    geo_zone = None
                    if "geological_zone" in sub.columns:
                        gz = sub["geological_zone"].dropna()
                        if len(gz):
                            geo_zone = gz.mode().iloc[0]
                    litho = None
                    if "LITHO_SIMP" in sub.columns:
                        lt = sub["LITHO_SIMP"].dropna()
                        if len(lt):
                            litho = lt.mode().iloc[0]

                    # ---------- Header card ----------
                    meta_html = f"<div><b style='color:#fafafa'>{n_records}</b> record(s)</div>"
                    if years_present:
                        meta_html += f"<div>Years: <b style='color:#fafafa'>{', '.join(map(str, years_present))}</b></div>"
                    if geo_zone:
                        meta_html += f"<div>Zone: <b style='color:#fafafa'>{geo_zone}</b></div>"
                    if litho and litho != geo_zone:
                        meta_html += f"<div>Lithology: <b style='color:#fafafa'>{litho}</b></div>"

                    st.markdown(f"""
                    <div style='background: rgba(24,24,27,0.6);
                                border: 1px solid rgba(255,255,255,0.06);
                                border-radius: 12px; padding: 1.25rem 1.5rem;
                                margin: 0.75rem 0 1.5rem 0;
                                backdrop-filter: blur(12px);'>
                      <div style='display: flex; align-items: baseline;
                                  gap: 0.75rem; margin-bottom: 0.5rem;'>
                        <div style='font-size: 1.5rem; font-weight: 700;
                                    color: #fafafa; letter-spacing: -0.02em;'>{chosen_name}</div>
                        <div style='font-size: 12px; color: #71717a;
                                    font-family: monospace;'>INSEE {chosen_insee}</div>
                      </div>
                      <div style='display: flex; gap: 1.5rem; font-size: 12px;
                                  color: #a1a1aa; flex-wrap: wrap;'>
                        {meta_html}
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ---------- Two-column layout ----------
                    col_chem, col_idx = st.columns([3, 2])

                    with col_chem:
                        st.markdown("##### Water composition")
                        chem_params = [
                            "CALCIUM", "MAGNESIUM", "MAGNÉSIUM",
                            "HYDROGENOCARBONATES", "HYDROGÉNOCARBONATES",
                            "SULFATES", "CHLORURES", "SODIUM", "POTASSIUM",
                            "NITRATES (EN NO3)", "PH ", "PH",
                        ]
                        seen_pretty = set()
                        rows = []
                        for p in chem_params:
                            if p not in sub.columns or p not in num_cols:
                                continue
                            pn = pretty(p)
                            if pn in seen_pretty:
                                continue
                            seen_pretty.add(pn)
                            v = sub[p].mean()
                            if pd.isna(v):
                                continue
                            nat = nat_means.get(p)
                            if nat is not None and nat != 0 and not pd.isna(nat):
                                delta = v - nat
                                pct = delta / nat * 100
                                delta_str = f"{delta:+.2f} ({pct:+.0f}%)"
                            else:
                                delta_str = "—"
                            rows.append({
                                "Parameter": pn,
                                "This commune": f"{v:.2f}",
                                "France avg.": f"{nat:.2f}" if nat is not None else "—",
                                "Δ vs France": delta_str,
                            })
                        if rows:
                            st.dataframe(pd.DataFrame(rows), hide_index=True,
                                         use_container_width=True)
                        else:
                            st.caption("No chemistry data available.")

                    with col_idx:
                        st.markdown("##### Aggressiveness indices")
                        index_params = ["IL", "IL_calc", "ryznar", "Larson", "Bason"]
                        for p in index_params:
                            if p not in sub.columns or p not in num_cols:
                                continue
                            v = sub[p].mean()
                            if pd.isna(v):
                                continue
                            nat = nat_means.get(p)
                            delta_str = (f"{v - nat:+.2f} vs France"
                                         if nat is not None and not pd.isna(nat)
                                         else None)
                            grade_col = {"IL": "Langelier_grade",
                                         "ryznar": "Ryznar_grade",
                                         "Larson": "Larson_grade",
                                         "Bason": "Bason_grade"}.get(p)
                            grade = None
                            if grade_col and grade_col in sub.columns:
                                gg = sub[grade_col].dropna()
                                if len(gg):
                                    grade = gg.mode().iloc[0]
                            st.metric(
                                pretty(p), f"{v:.2f}",
                                delta=delta_str, delta_color="off",
                                help=f"Grade: **{grade}**" if grade else None,
                            )
