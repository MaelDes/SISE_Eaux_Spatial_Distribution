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


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="SISE-Eaux Explorer",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Default candidate files in the project root
DEFAULT_CLASSIFIED = "stats/classified_annual_data.csv"
DEFAULT_CSV_GLOB   = "csv_traite/Analyses_insee_*.csv"

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

st.sidebar.title("💧 SISE-Eaux Explorer")
st.sidebar.markdown("Correlation & distribution analysis of French drinking-water chemistry.")

st.sidebar.header("1. Data source")

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
    if Path(default_path).exists():
        with st.spinner(f"Loading {default_path}..."):
            df = load_classified(default_path)
        status_msg = f"Loaded {len(df):,} rows from {default_path}"
    else:
        st.sidebar.warning(
            f"File not found: {default_path}\n\n"
            "Run `sise_stats.py run` first, or switch to 'Raw yearly CSVs'."
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
    st.title("💧 SISE-Eaux Correlation Explorer")
    st.info(
        "👈 **Select a data source in the sidebar to start.**\n\n"
        "Recommended: point to `stats/classified_annual_data.csv` produced by "
        "`sise_stats.py run`. It already contains geological zones and index grades."
    )
    st.stop()

st.sidebar.success(status_msg)
st.sidebar.caption(f"Rows: **{len(df):,}** — Cols: **{len(df.columns)}**")


# ---------------------------------------------------------------------------
# Sidebar : optional filters
# ---------------------------------------------------------------------------

st.sidebar.header("2. Filters (optional)")

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

st.title("💧 SISE-Eaux Correlation Explorer")
st.caption(
    "Exploratory analysis of French drinking-water chemistry "
    "(commune × year aggregated data)."
)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_corr, tab_scatter_c, tab_reg, tab_dist, tab_piper, tab_3d = st.tabs([
    "📊 Correlation matrix",
    "🎨 Scatter + color",
    "📈 Regression",
    "🔍 Distributions",
    "🧪 Piper diagram",
    "🧊 3D response surface",
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
    x_col = c1.selectbox("X", num_cols, index=num_cols.index("CALCIUM") if "CALCIUM" in num_cols else 0, key="sc_x")
    y_col = c2.selectbox("Y", num_cols, index=num_cols.index("HYDROGENOCARBONATES") if "HYDROGENOCARBONATES" in num_cols else 1, key="sc_y")

    color_options = ["(none)"] + num_cols + cat_cols
    z_col = c3.selectbox("Color by", color_options, index=0, key="sc_z")

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
                    marker_color="#888", showlegend=False,
                    hoverinfo="skip",
                ),
                row=1, col=1,
            )
            # Right marginal: histogram of Y
            fig.add_trace(
                go.Histogram(
                    y=plot_df[y_col], nbinsy=40,
                    marker_color="#888", showlegend=False,
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
    x_col = c1.selectbox("X", num_cols, index=num_cols.index("IL") if "IL" in num_cols else 0, key="reg_x")
    y_col = c2.selectbox("Y", num_cols, index=num_cols.index("ryznar") if "ryznar" in num_cols else 1, key="reg_y")
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
                              key="f1_col")
        f1_op  = st.selectbox("Op 1", [">", ">=", "<", "<=", "==", "!="], index=0, key="f1_op")
        f1_val = st.number_input("Value 1", value=0.0, key="f1_val")

        use_f2 = st.checkbox("Add a second condition", value=True)
        if use_f2:
            f2_col = st.selectbox("Variable 2", num_cols,
                                  index=num_cols.index("ryznar") if "ryznar" in num_cols else 0,
                                  key="f2_col")
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
            color_by = st.selectbox("Color by", color_opts, index=0, key="piper_color")
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
                tri_color = "#333"
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

                def add_triangle_grid(A, B, C, step=0.2, color="#DDD"):
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
                        mode="lines", line=dict(color="#DDD", width=0.7),
                        hoverinfo="skip", showlegend=False,
                    ))
                    # Family 2: parallel to B-R
                    P1 = B + t * (L - B)   # on bottom-left edge
                    P2 = R + t * (T - R)   # on right-top edge
                    fig.add_trace(go.Scatter(
                        x=[P1[0], P2[0]], y=[P1[1], P2[1]],
                        mode="lines", line=dict(color="#DDD", width=0.7),
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
                lbl = dict(size=14, color="#222")
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
                    plot_bgcolor="white",
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
                      index=num_cols.index("PH ") if "PH " in num_cols else 0, key="3d_x")
    y3 = c2.selectbox("Y", num_cols,
                      index=num_cols.index("CALCIUM") if "CALCIUM" in num_cols else 1, key="3d_y")
    z3 = c3.selectbox("Z", num_cols,
                      index=num_cols.index("IL") if "IL" in num_cols else 2, key="3d_z")
    color3_options = ["(none)"] + num_cols + cat_cols
    color3 = c4.selectbox("Color by", color3_options,
                          index=color3_options.index("ryznar") if "ryznar" in color3_options else 0,
                          key="3d_color")

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
