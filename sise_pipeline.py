# -*- coding: utf-8 -*-
"""
sise_pipeline.py
================
Pipeline unifié SISE-Eaux : traitement → calcul d'indices → cartographie.


Usage CLI :
    # Étape 1 – traiter les fichiers bruts SISE (une fois par année)
    python sise_pipeline.py process \
        --result DIS_RESULT_2020.txt --plv DIS_PLV_2020.txt \
        --out csv_traite/Analyses_insee_20.csv

    # Étape 2 – carte via GeoJSON INSEE (recommandé, pas d'appel réseau)
    python sise_pipeline.py map --param CALCIUM \
        --csv csv_traite/Analyses_insee_20.csv csv_traite/Analyses_insee_21.csv \
        --geojson communes.geojson --out cartes/

    # Étape 2 – carte via Nominatim (fallback si pas de GeoJSON)
    python sise_pipeline.py map --param CALCIUM \
        --csv csv_traite/Analyses_insee_20.csv \
        --geocache geocache.json --out cartes/

    # Avec outlier removal IQR (x1.5 par défaut, x3 pour outliers extrêmes)
    python sise_pipeline.py map --param CALCIUM \
        --csv csv_traite/*.csv --geojson communes.geojson \
        --iqr-factor 1.5 --out cartes/

    # Lister les parametres/indices disponibles
    python sise_pipeline.py list-params

Telechargement du GeoJSON communes :
    https://github.com/gregoiredavid/france-geojson  ->  communes.geojson
    ou https://geo.api.gouv.fr/decoupage-administratif/communes?format=geojson&geometry=contour
Telechargement données géol https://infoterre.brgm.fr/formulaire/telechargement-carte-geologique-metropolitaine-11-000-000 
Ici on utilise GEO001M_CART_FR_S_FGEOL_2154.shp
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

ID_COLS = ["referenceprel", "dateprel", "nomcommuneprinc", "inseecommuneprinc"]

COL_PH    = "PH "
COL_PH_EQ = "PH D'EQUILIBRE A LA T° ECHANTILLON"
COL_TEMP  = "TEMPERATURE DE L'EAU"
COL_CA    = "CALCIUM"
COL_MG    = "MAGNESIUM"
COL_HCO3  = "HYDROGENOCARBONATES"
COL_SO4   = "SULFATES"
COL_CL    = "CHLORURES"
COL_NO3   = "NITRATES (EN NO3)"
COL_NA    = "SODIUM"
COL_K     = "POTASSIUM"

# Noms alternatifs avec accents (selon la version du fichier SISE)
_COL_ALIASES = {
    "PH D'ÉQUILIBRE À LA T° ÉCHANTILLON": COL_PH_EQ,
    "TEMPÉRATURE DE L'EAU":               COL_TEMP,
    "MAGNÉSIUM":                           COL_MG,
    "HYDROGÉNOCARBONATES":                 COL_HCO3,
}

COMPUTED_INDICES = {"IL", "IL_calc", "ryznar", "Bason", "Larson"}

_MG_CFG   = {"label": "Magnesium (mg/L)",           "cmin": 0,  "cmax": 60}
_HCO3_CFG = {"label": "Hydrogenocarbonates (mg/L)",  "cmin": 0,  "cmax": 500}
_TEMP_CFG = {"label": "Temperature (C)",              "cmin": 0,  "cmax": 30}
_PHEQ_CFG = {"label": "pH equilibre",                "cmin": 6,  "cmax": 9}

PARAM_CONFIG: dict[str, dict] = {
    COL_PH:   {"label": "pH",              "cmin": 5,    "cmax": 10},
    COL_CA:   {"label": "Calcium (mg/L)",  "cmin": 0,    "cmax": 150},
    COL_MG:   _MG_CFG,
    COL_HCO3: _HCO3_CFG,
    COL_SO4:  {"label": "Sulfates (mg/L)", "cmin": 0,    "cmax": 80},
    COL_NO3:  {"label": "Nitrates (mg/L)", "cmin": 0,    "cmax": 60},
    COL_TEMP: _TEMP_CFG,
    COL_PH_EQ: _PHEQ_CFG,
    "IL":     {"label": "Indice de Langelier",         "cmin": -3,   "cmax": 2},
    "IL_calc":{"label": "Indice de Langelier calcule", "cmin": -3,   "cmax": 2},
    "ryznar": {"label": "Indice de Ryznar",            "cmin": 5,    "cmax": 13},
    "Bason":  {"label": "Indice de Bason",             "cmin": -200, "cmax": 1200},
    "Larson": {"label": "Indice de Larson",            "cmin": 0,    "cmax": 3},
    # Aliases avec accents
    "MAGNESIUM":           _MG_CFG,
    "MAGNÉSIUM":        _MG_CFG,
    "HYDROGENOCARBONATES": _HCO3_CFG,
    "HYDROGÉNOCARBONATES": _HCO3_CFG,
    "TEMPERATURE DE L'EAU":   _TEMP_CFG,
    "TEMPÉRATURE DE L'EAU":  _TEMP_CFG,
}

COLORSCALE = [
    [0.0, "rgb(51,124,255)"],
    [0.2, "rgb(54,199,248)"],
    [0.4, "rgb(54,248,239)"],
    [0.6, "rgb(49,144,53)"],
    [0.8, "rgb(237,248,54)"],
    [1.0, "rgb(255,14,0)"],
]


# ---------------------------------------------------------------------------
# 1. Traitement des fichiers bruts SISE
# ---------------------------------------------------------------------------

def process_sise_files(
    result_txt: str | Path,
    plv_txt: str | Path,
    output_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Charge les exports RESULT et PLV SISE-Eaux, pivote en table large
    (1 ligne = 1 prelevement, 1 colonne = 1 parametre) et enregistre le CSV.
    """
    print(f"Chargement {Path(result_txt).name} + {Path(plv_txt).name} ...")

    brut = pd.read_csv(
        result_txt,
        delimiter=",",
        usecols=["referenceprel", "libmajparametre", "valtraduite"],
        low_memory=False,
    )
    brut["valtraduite"] = pd.to_numeric(brut["valtraduite"], errors="coerce")

    plv = pd.read_csv(
        plv_txt,
        delimiter=",",
        usecols=["referenceprel", "inseecommuneprinc", "nomcommuneprinc", "dateprel"],
        low_memory=False,
    )
    plv["dateprel"] = pd.to_datetime(plv["dateprel"], errors="coerce")
    plv = plv.drop_duplicates(subset="referenceprel")

    table = (
        pd.pivot_table(
            brut,
            index="referenceprel",
            columns="libmajparametre",
            values="valtraduite",
            aggfunc="first",
        )
        .reset_index()
    )

    merged = table.merge(plv, on="referenceprel", how="left")

    if output_csv is not None:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(output_csv, index=False)
        print(f"  -> {output_csv} ({len(merged):,} lignes)")

    return merged


# ---------------------------------------------------------------------------
# 2. Chargement multi-annees
# ---------------------------------------------------------------------------

def load_csv_files(
    csv_files: Sequence[str | Path],
    keep_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Concatene les CSV annuels en un seul DataFrame.

    Parametres
    ----------
    keep_columns : si fourni, ne charge que ces colonnes (+ les colonnes ID).
                   Indispensable si les CSV contiennent des centaines de
                   parametres SISE rares (evite MemoryError).
    """
    # Si on filtre, on construit la liste complete a demander a pd.read_csv
    usecols = None
    if keep_columns is not None:
        base = ["referenceprel", "dateprel", "nomcommuneprinc", "inseecommuneprinc"]
        # On ajoute aussi les variantes avec/sans accents des colonnes demandees
        wanted = set(base) | set(keep_columns)
        for original, alias in _COL_ALIASES.items():
            if alias in wanted:
                wanted.add(original)

        # On lit d'abord les headers pour ne garder que les colonnes existantes
        sample_cols = set(pd.read_csv(csv_files[0], nrows=0).columns)
        usecols = [c for c in wanted if c in sample_cols]
        missing = wanted - sample_cols - set(base)
        if missing:
            print(f"  /!\\ Colonnes demandees absentes des CSV : {sorted(missing)}")

    frames = [
        pd.read_csv(p, low_memory=False, usecols=usecols) for p in csv_files
    ]
    df = pd.concat(frames, ignore_index=True)
    df["dateprel"] = pd.to_datetime(df["dateprel"], errors="coerce")

    # Normalisation des noms de colonnes (accents optionnels selon version SISE)
    df = df.rename(columns=_COL_ALIASES)

    # Si apres renommage il y a des doublons (ex: deux variantes CALCIUM),
    # on fusionne en gardant la premiere valeur non nulle
    if df.columns.duplicated().any():
        print(f"  Fusion de colonnes dupliquees apres normalisation des accents...")
        df = df.T.groupby(level=0).first().T

    print(f"  -> {len(df):,} lignes x {len(df.columns)} colonnes")
    return df


# ---------------------------------------------------------------------------
# 3. Calcul des indices d'agressivite
# ---------------------------------------------------------------------------

def compute_indices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute IL, ryznar, TDS, IL_calc, Bason, Larson.
    Seuls les indices dont toutes les colonnes sources sont presentes sont calcules.
    """
    out = df.copy()
    cols = set(out.columns)

    # Langelier (IL) et Ryznar
    if {COL_PH, COL_PH_EQ}.issubset(cols):
        out["IL"] = out[COL_PH] - out[COL_PH_EQ]
        out.loc[(out["IL"] < -7) | (out["IL"] > 6), "IL"] = np.nan
        out["ryznar"] = 2 * out[COL_PH_EQ] - out[COL_PH]

    # TDS
    tds_cols = [COL_CA, COL_MG, COL_HCO3, COL_K, COL_NA, COL_SO4, COL_CL]
    if set(tds_cols).issubset(cols):
        out["TDS"] = out[tds_cols].sum(axis=1)

    # IL calcule
    if {COL_TEMP, "TDS", COL_CA, COL_HCO3, COL_PH}.issubset(set(out.columns)):
        t = out[COL_TEMP]
        s = out["TDS"]
        out["pKs_pKa"] = np.select(
            [t <= 10, (t >= 10) & (t < 18), (t >= 18) & (t < 20),
             (t >= 20) & (t < 25), (t >= 25) & (t < 30),
             (t >= 30) & (t < 40), t >= 40],
            [2.61, 2.34, 2.14, 2.10, 1.99, 1.90, 1.71],
            default=2.0,
        )
        out["eps"] = np.select(
            [(s >= 15) & (s < 120), (s >= 120) & (s < 200),
             (s >= 200) & (s < 500), (s >= 500) & (s < 1500),
             (s >= 1500) & (s < 3000), s >= 3000],
            [0.02, 0.05, 0.06, 0.09, 0.15, 0.21],
            default=0.02,
        )
        ca   = out[COL_CA].replace(0, np.nan)
        hco3 = out[COL_HCO3].replace(0, np.nan)
        out["IL_calc"] = out[COL_PH] - (
            -np.log10(ca / 4) - np.log10(hco3 / 12.2)
            + 7.7 + out["pKs_pKa"] + 2 * out["eps"]
        )

    # Bason
    if {COL_PH, COL_PH_EQ, COL_CA, COL_MG, COL_SO4}.issubset(cols):
        lcsi = (
            200 * (9.5 - out[COL_PH])
            + 2000 * (out[COL_PH_EQ] - out[COL_PH])
            + 2.2 * (500 - (out[COL_CA] / 20) * 50.045)
        ) / 3
        scsi = (0.6 * out[COL_MG] + 0.3 * out[COL_SO4]) / 3
        out["Bason"] = lcsi + scsi

    # Larson
    if {COL_SO4, COL_CL, COL_HCO3}.issubset(cols):
        out["Larson"] = (out[COL_SO4] + out[COL_CL]) / out[COL_HCO3].replace(0, np.nan)

    return out


# ---------------------------------------------------------------------------
# 4. Suppression des outliers (IQR + winsorisation)
# ---------------------------------------------------------------------------

def remove_outliers_iqr(
    df: pd.DataFrame,
    param: str,
    iqr_factor: float = 1.5,
) -> pd.DataFrame:
    """
    Winsorisation IQR sur les mesures brutes (avant agregation).

    Les valeurs hors [Q1 - factor*IQR, Q3 + factor*IQR] sont remplacees
    par NaN : elles n'entrent pas dans la moyenne par commune mais la
    ligne reste presente dans le DataFrame.

    Parametres
    ----------
    iqr_factor : 1.5 = outliers classiques  |  3.0 = seulement les extremes
    """
    if param not in df.columns:
        return df

    serie = df[param]
    q1  = serie.quantile(0.25)
    q3  = serie.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - iqr_factor * iqr
    upper = q3 + iqr_factor * iqr

    mask_out = (serie < lower) | (serie > upper)
    n_out = mask_out.sum()

    if n_out > 0:
        pct = 100 * n_out / serie.notna().sum()
        print(
            f"  Outliers IQR x{iqr_factor} sur '{param}' : "
            f"{n_out:,} valeurs ({pct:.1f}%) -> NaN  "
            f"[bornes : {lower:.3f} -- {upper:.3f}]"
        )
    else:
        print(f"  Aucun outlier detecte sur '{param}' (IQR x{iqr_factor})")

    out = df.copy()
    out.loc[mask_out, param] = np.nan
    return out


# ---------------------------------------------------------------------------
# 5. Agregation annuelle par commune
# ---------------------------------------------------------------------------

def aggregate_annual(df: pd.DataFrame, param: str) -> pd.DataFrame:
    """Moyenne annuelle du parametre par commune (code INSEE + nom).

    Ajoute une colonne 'N_mesures' = nombre de mesures non-NaN utilisees
    pour calculer la moyenne de chaque (commune, annee).
    """
    needed = ["nomcommuneprinc", "inseecommuneprinc", "dateprel", param]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Colonnes manquantes : {missing}")

    tmp = df[needed].copy()
    tmp = tmp.dropna(subset=["dateprel", param, "inseecommuneprinc"])
    tmp["Annee"] = tmp["dateprel"].dt.year

    return (
        tmp.groupby(["nomcommuneprinc", "inseecommuneprinc", "Annee"], as_index=False)
        .agg(**{param: (param, "mean"), "N_mesures": (param, "size")})
    )


# ---------------------------------------------------------------------------
# 6a. Geolocalisation via GeoJSON INSEE (recommande)
# ---------------------------------------------------------------------------

def geocode_from_geojson(
    df: pd.DataFrame,
    geojson_path: str | Path,
    code_col: str = "inseecommuneprinc",
) -> pd.DataFrame:
    """
    Ajoute longitude/latitude en calculant le centroide des polygones
    communaux a partir d'un GeoJSON (code INSEE -> centroide).

    Avantages vs Nominatim :
      - Pas d'appel reseau, instantane
      - Pas d'homonymie possible (code INSEE unique)
      - Pas de limite de requetes

    Necessite : pip install geopandas

    GeoJSON a telecharger :
        https://github.com/gregoiredavid/france-geojson  -> communes.geojson
    """
    try:
        import geopandas as gpd
    except ImportError:
        raise ImportError(
            "geopandas est requis pour ce mode.\n"
            "Installez-le : pip install geopandas"
        )

    print(f"Chargement GeoJSON : {Path(geojson_path).name} ...")
    gdf = gpd.read_file(geojson_path)

    # Detection automatique de la colonne code INSEE
    candidates = ["code", "insee", "codgeo", "COM", "code_commune", "INSEE_COM"]
    code_field = next((c for c in candidates if c in gdf.columns), None)
    if code_field is None:
        raise KeyError(
            f"Colonne code INSEE introuvable dans le GeoJSON.\n"
            f"Colonnes disponibles : {list(gdf.columns)}"
        )

    centroids = gdf[[code_field, "geometry"]].copy()
    centroids["geometry"] = centroids.geometry.centroid
    centroids["longitude"] = centroids.geometry.x
    centroids["latitude"]  = centroids.geometry.y
    centroids = centroids[[code_field, "longitude", "latitude"]].rename(
        columns={code_field: code_col}
    )

    out = df.copy()
    # Normalisation des codes INSEE sur 5 caracteres (ex: "75056" et non "75056.0")
    out[code_col]       = out[code_col].astype(str).str.split(".").str[0].str.zfill(5)
    centroids[code_col] = centroids[code_col].astype(str).str.split(".").str[0].str.zfill(5)

    merged = out.merge(centroids, on=code_col, how="left")

    n_missing = merged["longitude"].isna().sum()
    if n_missing:
        print(f"  /!\\ {n_missing} commune(s) sans correspondance dans le GeoJSON")

    result = merged.dropna(subset=["longitude", "latitude"])
    print(f"  -> {len(result):,} communes localisees")
    return result


# ---------------------------------------------------------------------------
# 6b. Geolocalisation via Nominatim + cache JSON (fallback)
# ---------------------------------------------------------------------------

def _load_geocache(path: str | Path) -> dict[str, tuple[float, float]]:
    p = Path(path)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            raw = json.load(f)
        return {k: tuple(v) for k, v in raw.items()}
    return {}


def _save_geocache(cache: dict, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def geocode_from_nominatim(
    df: pd.DataFrame,
    geocache_path: str | Path,
    pause: float = 1.0,
) -> pd.DataFrame:
    """
    Ajoute longitude/latitude via Nominatim, avec cache JSON persistant.
    Preferer geocode_from_geojson() pour eviter les homonymies.
    """
    try:
        from geopy.geocoders import Nominatim
    except ImportError:
        raise ImportError("Installez geopy : pip install geopy")

    cache     = _load_geocache(geocache_path)
    geolocator = Nominatim(user_agent="sise_pipeline_v2")

    nouvelles = [v for v in df["nomcommuneprinc"].dropna().unique() if v not in cache]
    if nouvelles:
        print(f"Geocodage Nominatim : {len(nouvelles)} nouvelle(s) commune(s)...")
        for i, ville in enumerate(nouvelles, 1):
            try:
                loc = geolocator.geocode(f"{ville}, France", timeout=10)
                cache[ville] = (loc.longitude, loc.latitude) if loc else (None, None)
            except Exception:
                cache[ville] = (None, None)
            if i % 50 == 0:
                print(f"  {i}/{len(nouvelles)}")
                _save_geocache(cache, geocache_path)
            time.sleep(pause)
        _save_geocache(cache, geocache_path)
        print(f"Cache mis a jour -> {geocache_path}")

    out = df.copy()
    out["longitude"] = df["nomcommuneprinc"].map(lambda v: cache.get(v, (None, None))[0])
    out["latitude"]  = df["nomcommuneprinc"].map(lambda v: cache.get(v, (None, None))[1])
    return out.dropna(subset=["longitude", "latitude"])


# ---------------------------------------------------------------------------
# 7. Cartographie
# ---------------------------------------------------------------------------

def make_map(
    df: pd.DataFrame,
    param: str,
    output_html: str | Path | None = None,
) -> go.Figure:
    """
    Carte interactive Plotly avec slider par annee.
    La colorscale est ajustee sur les percentiles 2-98 des donnees agregees
    (robuste aux outliers residuels apres winsorisation).
    """
    cfg   = PARAM_CONFIG.get(param, {"label": param, "cmin": None, "cmax": None})
    label = cfg["label"]

    # Bornes colorscale : percentiles 2-98 des moyennes communales
    vals = df[param].dropna()
    cmin = vals.quantile(0.02) if len(vals) else cfg["cmin"]
    cmax = vals.quantile(0.98) if len(vals) else cfg["cmax"]
    # On respecte les bornes metier si plus restrictives
    if cfg["cmin"] is not None:
        cmin = max(cmin, cfg["cmin"])
    if cfg["cmax"] is not None:
        cmax = min(cmax, cfg["cmax"])

    # Colonne annee (supporte "Annee" et "Année")
    annee_col = "Annee" if "Annee" in df.columns else "Année"
    annees = sorted(df[annee_col].dropna().unique().astype(int))

    # --- Traces (une par annee) ---
    traces = []
    for annee in annees:
        sub = df[df[annee_col] == annee]
        n_communes = len(sub)
        n_mesures = int(sub["N_mesures"].sum()) if "N_mesures" in sub.columns else 0

        # Texte du hover : nom + valeur + nb mesures si dispo
        if "N_mesures" in sub.columns:
            hover_text = sub.apply(
                lambda r: (
                    f"<b>{r['nomcommuneprinc']}</b><br>"
                    f"{label} : {r[param]:.2f}<br>"
                    f"N = {int(r['N_mesures'])} mesure(s)"
                ),
                axis=1,
            )
        else:
            hover_text = sub.apply(
                lambda r: f"<b>{r['nomcommuneprinc']}</b><br>{label} : {r[param]:.2f}",
                axis=1,
            )

        trace = go.Scattermapbox(
            lat=sub["latitude"],
            lon=sub["longitude"],
            mode="markers",
            name=f"{annee} (n={n_communes:,} communes)",
            visible=(annee == annees[0]),
            marker=dict(
                size=10,
                color=sub[param],
                colorscale=COLORSCALE,
                cmin=cmin,
                cmax=cmax,
                colorbar=dict(
                    title=dict(text=label, side="right"),
                    thickness=20,
                    len=0.8,
                    outlinewidth=0,
                ),
                opacity=0.8,
            ),
            text=hover_text,
            hoverinfo="text",
            customdata=[{"n_communes": n_communes, "n_mesures": n_mesures}] * len(sub),
        )
        traces.append(trace)

    # --- Stats globales pour le titre + sliders ---
    n_communes_by_year = {
        int(a): int((df[annee_col] == a).sum()) for a in annees
    }
    n_mesures_by_year = {}
    if "N_mesures" in df.columns:
        for a in annees:
            n_mesures_by_year[int(a)] = int(df.loc[df[annee_col] == a, "N_mesures"].sum())
    total_communes = sum(n_communes_by_year.values())
    total_mesures  = sum(n_mesures_by_year.values()) if n_mesures_by_year else 0

    title_base = f"Carte annuelle -- {label}"

    # Sliders avec N visible en cours de selection
    steps = []
    for i, a in enumerate(annees):
        a_int = int(a)
        n_c = n_communes_by_year[a_int]
        n_m = n_mesures_by_year.get(a_int, 0)
        label_slider = f"{a_int} (N_com={n_c:,}"
        if n_m:
            label_slider += f", N_mes={n_m:,}"
        label_slider += ")"
        steps.append(dict(
            method="update",
            label=str(a_int),
            args=[
                {"visible": [j == i for j in range(len(annees))]},
                {"title.text": f"{title_base} -- {label_slider}"},
            ],
        ))

    first_year = int(annees[0])
    n_c0 = n_communes_by_year[first_year]
    n_m0 = n_mesures_by_year.get(first_year, 0)
    title_first = f"{title_base} -- {first_year} (N_com={n_c0:,}"
    if n_m0:
        title_first += f", N_mes={n_m0:,}"
    title_first += ")"

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Annee : ", "font": {"size": 16}},
        pad={"t": 50},
        steps=steps,
    )]

    fig = go.Figure(data=traces)

    # Annotation globale en bas de carte avec les totaux
    annotations = []
    if total_mesures:
        annotations.append(dict(
            text=(f"Total sur la periode : {total_communes:,} "
                  f"observations commune x annee, {total_mesures:,} mesures brutes"),
            xref="paper", yref="paper", x=0.5, y=-0.02,
            xanchor="center", yanchor="top",
            showarrow=False,
            font=dict(size=12, color="grey"),
        ))

    fig.update_layout(
        title=dict(text=title_first, font=dict(size=18)),
        mapbox=dict(style="open-street-map", center=dict(lat=47.08, lon=2.4), zoom=5),
        sliders=sliders,
        annotations=annotations,
        margin=dict(r=0, t=60, l=0, b=40),
        height=800,
    )

    if output_html is not None:
        Path(output_html).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_html))
        print(f"Carte enregistree -> {output_html}")

    return fig


# ---------------------------------------------------------------------------
# 8. Pipeline de haut niveau
# ---------------------------------------------------------------------------

def run_map(
    csv_files: Sequence[str | Path],
    param: str,
    output_dir: str | Path = "cartes",
    geojson_path: str | Path | None = None,
    geocache_path: str | Path | None = None,
    iqr_factor: float | None = 1.5,
    nominatim_pause: float = 1.0,
) -> go.Figure:
    """
    Pipeline complet :
      chargement CSV -> (calcul indices) -> (outliers IQR winsorises)
      -> agregation annuelle -> geolocalisation -> carte HTML

    Parametres
    ----------
    geojson_path  : GeoJSON des communes (recommande, geocode par code INSEE)
    geocache_path : fallback Nominatim avec cache JSON (si pas de GeoJSON)
    iqr_factor    : facteur IQR pour la winsorisation (1.5 ou 3.0).
                    None = pas de filtre outliers.
    """
    print(f"\n{'='*55}")
    print(f"  Parametre : {param}")
    print(f"{'='*55}")

    df = load_csv_files(csv_files)
    n_annees = df["dateprel"].dt.year.nunique()
    print(f"Donnees chargees   : {len(df):,} lignes, {n_annees} annee(s)")

    # Calcul des indices si necessaire
    if param in COMPUTED_INDICES:
        print("Calcul des indices d'agressivite...")
        df = compute_indices(df)

    if param not in df.columns:
        available = sorted(c for c in df.columns if c not in ID_COLS)
        raise ValueError(
            f"Parametre '{param}' introuvable.\n"
            f"Disponibles : {available}"
        )

    # Outliers IQR sur les mesures brutes (avant agregation)
    if iqr_factor is not None:
        print(f"Suppression outliers IQR x{iqr_factor} (winsorisation)...")
        df = remove_outliers_iqr(df, param, iqr_factor=iqr_factor)

    # Agregation annuelle
    annual = aggregate_annual(df, param)
    print(f"Apres agregation   : {len(annual):,} communes x annees")

    # Geolocalisation
    if geojson_path is not None:
        annual = geocode_from_geojson(annual, geojson_path)
    elif geocache_path is not None:
        annual = geocode_from_nominatim(annual, geocache_path, pause=nominatim_pause)
    else:
        raise ValueError("Fournir --geojson ou --geocache pour la geolocalisation.")

    out_html = Path(output_dir) / f"carte_{param}.html"
    return make_map(annual, param=param, output_html=out_html)


# ---------------------------------------------------------------------------
# 9. Interface en ligne de commande (CLI)
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pipeline SISE-Eaux : traitement et cartographie",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- process ---
    p_proc = sub.add_parser("process", help="Traiter les fichiers bruts SISE")
    p_proc.add_argument("--result", required=True, help="DIS_RESULT_XXXX.txt")
    p_proc.add_argument("--plv",    required=True, help="DIS_PLV_XXXX.txt")
    p_proc.add_argument("--out",    required=True, help="CSV de sortie")

    # --- map ---
    p_map = sub.add_parser("map", help="Generer une carte pour un parametre")
    p_map.add_argument("--param", required=True,
                       help="Parametre ou indice (ex: CALCIUM, IL, ryznar)")
    p_map.add_argument("--csv",   required=True, nargs="+", help="CSV annuels")
    p_map.add_argument("--out",   default="cartes",
                       help="Dossier de sortie (defaut: cartes/)")

    geo = p_map.add_mutually_exclusive_group(required=True)
    geo.add_argument(
        "--geojson",
        help="GeoJSON des communes (geocode par code INSEE -- recommande)",
    )
    geo.add_argument(
        "--geocache",
        help="Fichier cache JSON Nominatim (fallback si pas de GeoJSON)",
    )

    p_map.add_argument(
        "--iqr-factor",
        type=float,
        default=1.5,
        metavar="FACTOR",
        help=(
            "Facteur IQR pour la winsorisation des outliers "
            "(defaut: 1.5 | 3.0 = extremes seulement | 0 = desactive)"
        ),
    )
    p_map.add_argument(
        "--pause",
        type=float,
        default=1.0,
        help="Pause entre requetes Nominatim en secondes (defaut: 1.0)",
    )

    # --- list-params ---
    sub.add_parser("list-params", help="Lister les parametres/indices disponibles")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "process":
        process_sise_files(args.result, args.plv, args.out)

    elif args.command == "map":
        iqr = args.iqr_factor if args.iqr_factor > 0 else None
        run_map(
            csv_files=args.csv,
            param=args.param,
            output_dir=args.out,
            geojson_path=getattr(args, "geojson", None),
            geocache_path=getattr(args, "geocache", None),
            iqr_factor=iqr,
            nominatim_pause=args.pause,
        )

    elif args.command == "list-params":
        print("\nParametres bruts SISE courants :")
        for p in sorted(k for k in PARAM_CONFIG if k not in COMPUTED_INDICES
                        and "QUILIBRE" not in k and "RATURE" not in k
                        and "GNESIUM" not in k and "GENO" not in k):
            print(f"  {p:40s} -> {PARAM_CONFIG[p]['label']}")
        print("\nIndices calcules :")
        for p in sorted(COMPUTED_INDICES):
            print(f"  {p:40s} -> {PARAM_CONFIG[p]['label']}")


if __name__ == "__main__":
    main()