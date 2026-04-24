# -*- coding: utf-8 -*-
"""
sise_stats.py
=============
Module de tests statistiques pour l'article SISE-Eaux.


Methodologie
------------
1. Classification des echantillons par ZONE GEOLOGIQUE (jointure spatiale
   avec un GeoJSON BRGM : sedimentaire / cristallin / volcanique / ...)
2. Classification par GRADE d'INDICE selon les seuils standards
   (Langelier, Ryznar, Larson, Bason)
3. Pour chaque couple (parametre, classification) :
      a) Kruskal-Wallis (H-test)  -> p-value globale
         + taille d'effet epsilon^2
      b) Si H significatif : post-hoc de Dunn avec correction Bonferroni-Holm
         -> matrice de p-values par paire
4. Sorties :
      - CSV : resultats detailles
      - Tableau LaTeX : format publication
      - Figures : boxplots par groupe avec etoiles de significativite

Tests non-parametriques justifies par :
  - Distributions des parametres chimiques non gaussiennes (verifie Shapiro)
  - Robustesse aux outliers residuels et aux tailles de groupes desequilibrees
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Constantes : classifications par grade d'indice
# ---------------------------------------------------------------------------

# Grades standards de la litterature (Langelier 1936, Ryznar 1944, Larson 1958,
# Basson 1974)
LANGELIER_GRADES = [
    (-np.inf, -0.1, "Aggressive"),
    (-0.1,    0.1, "Balanced"),
    ( 0.1, np.inf, "Scaling"),
]

RYZNAR_GRADES = [
    (-np.inf, 5.5,  "Highly scaling"),
    ( 5.5,    6.2, "Scaling"),
    ( 6.2,    6.8, "Balanced"),
    ( 6.8,    8.5, "Corrosive"),
    ( 8.5, np.inf, "Highly corrosive"),
]

LARSON_GRADES = [
    (-np.inf, 0.5, "Low corrosion"),
    ( 0.5,    1.0, "Moderate corrosion"),
    ( 1.0, np.inf, "High corrosion"),
]

BASON_GRADES = [
    (-np.inf,  300, "Balanced"),
    (  300,    800, "Moderate corrosion"),
    (  800,   1000, "High corrosion"),
    ( 1000, np.inf, "Very high corrosion"),
]

INDEX_GRADES = {
    "IL":     LANGELIER_GRADES,
    "ryznar": RYZNAR_GRADES,
    "Larson": LARSON_GRADES,
    "Bason":  BASON_GRADES,
}


@dataclass(slots=True)
class StatsResult:
    """Resultat complet d'un test comparatif sur un parametre et une classification."""
    parameter: str
    grouping: str                  # "geologie" ou "grade_IL", etc.
    n_samples: int
    n_groups: int
    group_sizes: dict[str, int]
    kruskal_H: float
    kruskal_p: float
    epsilon_squared: float
    shapiro_p: float | None       # test de normalite (pour justifier le choix)
    dunn_matrix: pd.DataFrame | None  # matrice des p-values par paire
    is_significant: bool


# ---------------------------------------------------------------------------
# 1a. Regroupement lithologique BRGM -> 5 grandes familles
# ---------------------------------------------------------------------------

# Mapping mot-cle (dans DESCR/NOTATION) -> famille lithologique simplifiee.
# L'ordre compte : la premiere regle qui matche gagne (ex: "argiles et gres"
# -> Sedimentaire avant "gres" seul).
# Sources : BRGM Descriptif cartes geologiques 1/50000 (Janjou 2004),
#           BRGM BD Million-Geol, Carte lithologique simplifiee 1/1 000 000.
LITHOLOGY_KEYWORDS: list[tuple[str, str]] = [
    # Quaternary / alluvium
    ("alluvion",      "Alluvium"),
    ("alluvial",      "Alluvium"),
    ("colluvion",     "Alluvium"),
    ("limon",         "Alluvium"),
    ("loess",         "Alluvium"),
    ("formations superficielles", "Alluvium"),
    ("quaternaire",   "Alluvium"),

    # Volcanic
    ("basalte",       "Volcanic"),
    ("basaltique",    "Volcanic"),
    ("volcan",        "Volcanic"),
    ("volcanique",    "Volcanic"),
    ("lave",          "Volcanic"),
    ("trachy",        "Volcanic"),
    ("rhyolite",      "Volcanic"),
    ("andesite",      "Volcanic"),
    ("tuf",           "Volcanic"),
    ("scorie",        "Volcanic"),
    ("pouzzolane",    "Volcanic"),

    # Plutonic / crystalline (granite, granodiorite, diorite, gabbro...)
    ("granite",       "Plutonic"),
    ("granit",        "Plutonic"),
    ("granod",        "Plutonic"),
    ("diorite",       "Plutonic"),
    ("gabbro",        "Plutonic"),
    ("syenite",       "Plutonic"),
    ("leucogranite",  "Plutonic"),
    ("monzogranite",  "Plutonic"),
    ("pegmatite",     "Plutonic"),
    ("plutonique",    "Plutonic"),

    # Metamorphic
    ("gneiss",        "Metamorphic"),
    ("micaschiste",   "Metamorphic"),
    ("schiste",       "Metamorphic"),
    ("ardoise",       "Metamorphic"),
    ("amphibolite",   "Metamorphic"),
    ("migmatite",     "Metamorphic"),
    ("metamorph",     "Metamorphic"),
    ("quartzite",     "Metamorphic"),
    ("phyllade",      "Metamorphic"),

    # Sedimentary carbonates
    ("calcaire",      "Carbonate sedimentary"),
    ("craie",         "Carbonate sedimentary"),
    ("dolomie",       "Carbonate sedimentary"),
    ("marno-calcaire","Carbonate sedimentary"),

    # Detrital sedimentary (sandstone, shale, marl, sand)
    ("gres",          "Detrital sedimentary"),
    ("argile",        "Detrital sedimentary"),
    ("marne",         "Detrital sedimentary"),
    ("sable",         "Detrital sedimentary"),
    ("silt",          "Detrital sedimentary"),
    ("conglomerat",   "Detrital sedimentary"),
    ("flysch",        "Detrital sedimentary"),
    ("molasse",       "Detrital sedimentary"),

    # Evaporites
    ("gypse",         "Evaporite"),
    ("anhydrite",     "Evaporite"),
    ("sel",           "Evaporite"),
    ("halite",        "Evaporite"),
    ("evaporite",     "Evaporite"),
]


def _strip_accents(s: str) -> str:
    """Normalise une chaine pour faire matcher les mots-cles (accents/casse)."""
    import unicodedata
    if not isinstance(s, str):
        return ""
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()


def simplify_lithology(description: str) -> str:
    """
    Mappe une description BRGM detaillee vers une famille simplifiee.

    Les descriptions BRGM melangent souvent plusieurs lithologies
    (ex: "calcaires et schistes"). On compte le nombre de mot-cles
    matches par famille et on retourne la famille dominante.

    En cas d'egalite, la priorite va aux familles dans l'ordre :
      Alluvions > Volcanique > Evaporite > Sedimentaire-carbonate >
      Sedimentaire-detritique > Cristallin > Metamorphique

    Retourne 'Inconnue' si aucun mot-cle ne matche.

    Exemples
    --------
    >>> simplify_lithology("Granite porphyroide de la Margeride")
    'Cristallin'
    >>> simplify_lithology("Calcaires et dolomies a silex")
    'Sedimentaire-carbonate'
    >>> simplify_lithology("Alluvions recentes de la Seine")
    'Alluvions'
    """
    if description is None or (isinstance(description, float) and np.isnan(description)):
        return "Unknown"

    normalized = _strip_accents(description)

    # Compter chaque famille matchee
    counts: dict[str, int] = {}
    for keyword, family in LITHOLOGY_KEYWORDS:
        if keyword in normalized:
            counts[family] = counts.get(family, 0) + 1

    if not counts:
        return "Unknown"

    # Ordre de priorite en cas d'egalite
    priority = {
        "Alluvium":               0,
        "Volcanic":               1,
        "Evaporite":              2,
        "Carbonate sedimentary":  3,
        "Detrital sedimentary":   4,
        "Plutonic":               5,
        "Metamorphic":            6,
        "Unknown":                7,
    }
    return max(counts, key=lambda f: (counts[f], -priority.get(f, 99)))


# Familles regroupees en macro-categories (pour tests avec moins de groupes)
LITHOLOGY_MACRO = {
    "Alluvium":              "Alluvium",
    "Volcanic":              "Volcanic (magmatic)",
    "Plutonic":              "Plutonic (magmatic)",
    "Metamorphic":           "Metamorphic",
    "Carbonate sedimentary": "Sedimentary",
    "Detrital sedimentary":  "Sedimentary",
    "Evaporite":             "Sedimentary",
    "Unknown":               "Unknown",
}


# Traduction des categories BRGM LITHO_SIMP (francais -> anglais).
# Utilisee quand on charge directement le champ LITHO_SIMP avec --no-simplify
# pour avoir des libelles anglais dans les figures.
LITHO_SIMP_FR_EN = {
    "Sédiments et volcanites":   "Sediments and volcanites",
    "Formations superficielles": "Surficial formations",
    "Roches métamorphiques":     "Metamorphic rocks",
    "Plutons varisques":         "Variscan plutons",
    "non documenté":             "Undocumented",
    "Roches orthogneissiques":   "Orthogneissic rocks",
    "Volcanites cénozoïques":    "Cenozoic volcanites",
    "Couverture sédimentaire":   "Sedimentary cover",
    "Socle varisque":            "Variscan basement",
    "Ophiolites alpines":        "Alpine ophiolites",
    "Plutons cadomiens":         "Cadomian plutons",
    "Socle cadomien":            "Cadomian basement",
    "Plutons alpins":            "Alpine plutons",
}


# ---------------------------------------------------------------------------
# 1b. Classification par zone geologique (jointure spatiale)
# ---------------------------------------------------------------------------

def assign_geological_zone(
    df: pd.DataFrame,
    geology_geojson: str | Path,
    zone_field: str = "DESCR",
    code_col: str = "inseecommuneprinc",
    communes_geojson: str | Path | None = None,
    simplify: bool = True,
    macro: bool = False,
) -> pd.DataFrame:
    """
    Ajoute une colonne 'zone_geologique' au DataFrame via jointure spatiale.

    Logique :
      1. Si pas de lon/lat, calcul des centroides via communes_geojson
      2. Jointure spatiale point-dans-polygone avec la carte BRGM
      3. Si simplify=True (defaut) : regroupement en familles
         (Sedimentaire-carbonate, Sedimentaire-detritique, Cristallin,
          Metamorphique, Volcanique, Alluvions, Evaporite)
      4. Si macro=True : regroupement supplementaire en 5 macro-categories
         (Sedimentaire, Magmatique-volcanique, Magmatique-plutonique,
          Metamorphique, Alluvions)

    Parametres
    ----------
    df              : DataFrame avec (longitude, latitude) OU inseecommuneprinc
    geology_geojson : GeoJSON/shapefile BRGM
                      - BD Million-Geol : utiliser zone_field="DESCR"
                      - LITHO_1M_SIMPLIFIEE : parametre deja simplifie,
                        mettre simplify=False et zone_field="LITHO" (ou le
                        nom reel du champ, que le script vous listera)
    zone_field      : champ lithologique dans le fichier BRGM
    simplify        : applique simplify_lithology() sur la valeur brute
    macro           : applique en plus le regroupement macro (5 categories)
    """
    try:
        import geopandas as gpd
    except ImportError:
        raise ImportError("pip install geopandas")

    # --- Si pas de longitude/latitude, on calcule depuis les communes ---
    if "longitude" not in df.columns or "latitude" not in df.columns:
        if communes_geojson is None:
            raise ValueError(
                "Le DataFrame n'a pas de coordonnees : fournir 'communes_geojson'"
            )
        from sise_pipeline import geocode_from_geojson
        df = geocode_from_geojson(df, communes_geojson, code_col=code_col)

    # --- Chargement des polygones geologiques ---
    print(f"Chargement geologie : {Path(geology_geojson).name} ...")
    geology = gpd.read_file(geology_geojson)

    if zone_field not in geology.columns:
        raise KeyError(
            f"Champ '{zone_field}' absent du fichier geologique.\n"
            f"Champs disponibles : {list(geology.columns)}\n"
            f"Conseil : pour la BD Million-Geol du BRGM, essayez 'DESCR' ou "
            f"'NOTATION'. Pour LITHO_1M_SIMPLIFIEE, essayez 'LITHO' ou 'CLASSE'."
        )

    # --- Points du DataFrame ---
    points = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )
    # Alignement des CRS
    if geology.crs != points.crs:
        geology = geology.to_crs(points.crs)

    # --- Jointure spatiale ---
    joined = gpd.sjoin(
        points,
        geology[[zone_field, "geometry"]],
        how="left",
        predicate="within",
    )
    # Si un point tombe sur une frontiere, sjoin peut dupliquer la ligne
    # -> on garde seulement la premiere correspondance
    joined = joined[~joined.index.duplicated(keep="first")]
    joined = joined.rename(columns={zone_field: "_litho_brute"})
    joined = joined.drop(columns=["geometry", "index_right"], errors="ignore")

    n_missing = joined["_litho_brute"].isna().sum()
    if n_missing:
        print(f"  /!\\ {n_missing} points with no geological zone (outside polygons)")

    # --- Simplification (regroupement en familles) ---
    if simplify:
        print("  Simplifying lithology into families...")
        joined["geological_zone"] = joined["_litho_brute"].apply(simplify_lithology)

        if macro:
            print("  Macro-grouping into 5 categories...")
            joined["geological_zone"] = joined["geological_zone"].map(LITHOLOGY_MACRO)

        # Stats sur le mapping
        counts = joined["geological_zone"].value_counts()
        n_unknown = (joined["geological_zone"] == "Unknown").sum()
        if n_unknown:
            unknowns = joined.loc[joined["geological_zone"] == "Unknown", "_litho_brute"].dropna().unique()[:5]
            print(f"  /!\\ {n_unknown} points classified as 'Unknown'. Examples of unrecognized descriptions:")
            for u in unknowns:
                print(f"      - {u[:100]}")
            print("  -> Add keywords to LITHOLOGY_KEYWORDS if needed")
        print(f"  Distribution: {counts.to_dict()}")

        joined = joined.drop(columns=["_litho_brute"])
    else:
        # Garde les valeurs BRGM brutes, mais traduit le champ LITHO_SIMP
        # en anglais si les libelles correspondent.
        joined = joined.rename(columns={"_litho_brute": "geological_zone"})
        joined["geological_zone"] = (
            joined["geological_zone"]
            .map(lambda v: LITHO_SIMP_FR_EN.get(v, v))
        )

    return pd.DataFrame(joined)


# ---------------------------------------------------------------------------
# 2. Classification par grade d'indice
# ---------------------------------------------------------------------------

def assign_index_grade(
    df: pd.DataFrame,
    index_col: str,
    grades: list[tuple[float, float, str]] | None = None,
) -> pd.DataFrame:
    """
    Ajoute une colonne 'grade_<index_col>' classant chaque valeur dans
    une categorie de risque (base sur les seuils de la litterature).

    Ex: IL < -0.5 -> "Agressive", -0.5 < IL < 0.5 -> "Equilibre", etc.
    """
    if grades is None:
        if index_col not in INDEX_GRADES:
            raise ValueError(
                f"No predefined grades for '{index_col}'. "
                f"Available: {list(INDEX_GRADES)}"
            )
        grades = INDEX_GRADES[index_col]

    bins   = [g[0] for g in grades] + [grades[-1][1]]
    labels = [g[2] for g in grades]

    # Nom de colonne en anglais : IL -> Langelier_grade, ryznar -> Ryznar_grade, etc.
    english_names = {
        "IL":     "Langelier_grade",
        "ryznar": "Ryznar_grade",
        "Larson": "Larson_grade",
        "Bason":  "Basson_grade",
    }
    grade_col = english_names.get(index_col, f"{index_col}_grade")

    out = df.copy()
    out[grade_col] = pd.cut(
        out[index_col],
        bins=bins,
        labels=labels,
        include_lowest=True,
        ordered=False,
    )
    return out


# ---------------------------------------------------------------------------
# 2b. Agregation commune x annee (pour tests stats rigoureux)
# ---------------------------------------------------------------------------

def aggregate_commune_year(
    df: pd.DataFrame,
    parameters: Sequence[str],
    grouping_cols: Sequence[str],
    code_col: str = "inseecommuneprinc",
) -> pd.DataFrame:
    """
    Reduit le DataFrame a une ligne par commune x annee, en moyennant chaque
    parametre. Les colonnes de classification (zone_geologique, grade_IL, etc.)
    sont conservees par majorite dans chaque commune x annee.

    Cette agregation est fortement recommandee avant les tests Kruskal-Wallis
    car les mesures brutes SISE sont non independantes : une meme commune peut
    avoir des dizaines de prelevements par an, ce qui pondere artificiellement
    les tests sur les mesures brutes (violation de l'hypothese d'independance).
    """
    if "dateprel" not in df.columns:
        raise KeyError("Column 'dateprel' required for aggregation")

    work = df.copy()
    work["Year"] = pd.to_datetime(work["dateprel"], errors="coerce").dt.year
    work = work.dropna(subset=["Year", code_col])
    work["Year"] = work["Year"].astype(int)

    n_before = len(work)

    numeric_cols = [c for c in parameters if c in work.columns]
    category_cols = [c for c in grouping_cols if c in work.columns]

    def _mode_first(s):
        s = s.dropna()
        if len(s) == 0:
            return np.nan
        m = s.mode()
        return m.iloc[0] if len(m) else s.iloc[0]

    agg_dict = {c: "mean" for c in numeric_cols}
    agg_dict.update({c: _mode_first for c in category_cols})

    aggregated = (
        work.groupby([code_col, "Year"], as_index=False)
            .agg(agg_dict)
    )

    print(
        f"  Commune x year aggregation: "
        f"{n_before:,} measurements -> {len(aggregated):,} observations "
        f"(reduction x{n_before/max(len(aggregated),1):.1f})"
    )
    return aggregated


# ---------------------------------------------------------------------------
# 3. Tests statistiques
# ---------------------------------------------------------------------------

def _epsilon_squared(H: float, n: int) -> float:
    """Taille d'effet pour Kruskal-Wallis (Tomczak & Tomczak, 2014).
    0.01 = petit, 0.06 = moyen, 0.14 = grand."""
    if n <= 1:
        return np.nan
    return max(0.0, (H - 0) / (n - 1))


def compare_groups(
    df: pd.DataFrame,
    parameter: str,
    grouping_col: str,
    min_group_size: int = 10,
    shapiro_subsample: int = 5000,
) -> StatsResult:
    """
    Teste si `parameter` diffre entre les groupes definis par `grouping_col`.

    Pipeline :
      1. Nettoyage (NaN, groupes trop petits)
      2. Shapiro-Wilk sur chaque groupe pour justifier le non-parametrique
      3. Kruskal-Wallis (H-test)
      4. Post-hoc de Dunn + correction Bonferroni-Holm (si H significatif)
      5. Taille d'effet epsilon^2

    Parametres
    ----------
    min_group_size : taille minimale d'un groupe pour qu'il soit inclus
                     (5 est le minimum absolu pour KW, 10 recommande)
    shapiro_subsample : Shapiro ne supporte pas N > ~5000, on sous-echantillonne
    """
    try:
        import scikit_posthocs as sp
    except ImportError:
        raise ImportError("pip install scikit-posthocs")

    data = df[[parameter, grouping_col]].dropna()

    # Filtrer les petits groupes
    sizes = data[grouping_col].value_counts()
    valid_groups = sizes[sizes >= min_group_size].index.tolist()
    data = data[data[grouping_col].isin(valid_groups)].copy()

    if len(valid_groups) < 2:
        raise ValueError(
            f"Moins de 2 groupes avec N >= {min_group_size} pour {parameter}"
        )

    groups = [data.loc[data[grouping_col] == g, parameter].values
              for g in valid_groups]

    # Shapiro-Wilk (agrege sur un echantillon global)
    sample = data[parameter].sample(
        n=min(len(data), shapiro_subsample), random_state=42
    )
    shapiro_p = stats.shapiro(sample).pvalue if len(sample) >= 3 else None

    # Kruskal-Wallis
    H, p_kruskal = stats.kruskal(*groups)
    eps2 = _epsilon_squared(H, len(data))

    # Post-hoc de Dunn
    dunn = None
    if p_kruskal < 0.05:
        dunn = sp.posthoc_dunn(
            data, val_col=parameter, group_col=grouping_col,
            p_adjust="holm",
        )

    return StatsResult(
        parameter=parameter,
        grouping=grouping_col,
        n_samples=len(data),
        n_groups=len(valid_groups),
        group_sizes={str(g): int(sizes[g]) for g in valid_groups},
        kruskal_H=float(H),
        kruskal_p=float(p_kruskal),
        epsilon_squared=float(eps2),
        shapiro_p=float(shapiro_p) if shapiro_p is not None else None,
        dunn_matrix=dunn,
        is_significant=bool(p_kruskal < 0.05),
    )


def run_full_analysis(
    df: pd.DataFrame,
    parameters: Sequence[str],
    grouping_cols: Sequence[str],
    output_dir: str | Path = "stats",
    min_group_size: int = 10,
) -> pd.DataFrame:
    """
    Lance tous les tests {parameter x grouping} et retourne un DataFrame
    recapitulatif + ecrit les tableaux LaTeX et les matrices de Dunn.

    Returns
    -------
    summary : DataFrame avec une ligne par test
              (parameter, grouping, N, H, p, eps^2, signif)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for grouping in grouping_cols:
        if grouping not in df.columns:
            print(f"  [SKIP] column '{grouping}' not in DataFrame")
            continue
        for param in parameters:
            if param not in df.columns:
                print(f"  [SKIP] parameter '{param}' not found")
                continue
            try:
                res = compare_groups(
                    df, parameter=param, grouping_col=grouping,
                    min_group_size=min_group_size,
                )
            except ValueError as e:
                print(f"  [SKIP] {param} / {grouping}: {e}")
                continue

            rows.append({
                "parameter":     param,
                "grouping":      grouping,
                "N":             res.n_samples,
                "n_groups":      res.n_groups,
                "H":             round(res.kruskal_H, 3),
                "p_value":       res.kruskal_p,
                "epsilon^2":     round(res.epsilon_squared, 4),
                "significant":   res.is_significant,
                "shapiro_p":     res.shapiro_p,
            })

            # Matrice de Dunn en CSV
            if res.dunn_matrix is not None:
                fname = output_dir / f"dunn_{param}_{grouping}.csv"
                res.dunn_matrix.to_csv(fname)

            print(
                f"  {param:20s} ~ {grouping:25s} | "
                f"N={res.n_samples:5d}  H={res.kruskal_H:7.2f}  "
                f"p={res.kruskal_p:.2e}  eps2={res.epsilon_squared:.3f}  "
                f"{'***' if res.kruskal_p < 0.001 else ('**' if res.kruskal_p < 0.01 else ('*' if res.kruskal_p < 0.05 else 'ns'))}"
            )

    summary = pd.DataFrame(rows)

    if not summary.empty:
        summary.to_csv(output_dir / "summary_tests.csv", index=False)
        _write_latex_table(summary, output_dir / "summary_tests.tex")

    return summary


# ---------------------------------------------------------------------------
# 4. Tableau LaTeX
# ---------------------------------------------------------------------------

def _sig_stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def _write_latex_table(summary: pd.DataFrame, output_path: Path) -> None:
    """Ecrit un tableau LaTeX publication-ready."""
    tmp = summary.copy()
    tmp["p_value"]   = tmp["p_value"].apply(lambda p: f"{p:.2e} {_sig_stars(p)}")
    tmp["epsilon^2"] = tmp["epsilon^2"].apply(lambda e: f"{e:.3f}")

    col_map = {
        "parameter":  "Parameter",
        "grouping":   "Grouping",
        "N":          "$N$",
        "n_groups":   "$k$",
        "H":          "$H$",
        "p_value":    "$p$-value",
        "epsilon^2":  "$\\varepsilon^2$",
    }
    tmp = tmp[list(col_map)].rename(columns=col_map)

    latex = tmp.to_latex(
        index=False,
        escape=False,
        caption=(
            "Kruskal-Wallis tests comparing water chemical parameters "
            "across geological zones and index grades. "
            "Significance: *** $p<0.001$, ** $p<0.01$, * $p<0.05$. "
            "Effect size $\\varepsilon^2$: 0.01 small, 0.06 medium, 0.14 large "
            "(Tomczak \\& Tomczak, 2014)."
        ),
        label="tab:kruskal",
        position="htbp",
    )
    output_path.write_text(latex, encoding="utf-8")
    print(f"Tableau LaTeX -> {output_path}")


# ---------------------------------------------------------------------------
# 5. Figures : boxplots avec annotations de significativite
# ---------------------------------------------------------------------------

def plot_boxplot_with_significance(
    df: pd.DataFrame,
    parameter: str,
    grouping_col: str,
    output_path: str | Path,
    min_group_size: int = 10,
    max_pairs_annotated: int = 10,
    figsize: tuple = (10, 6),
) -> None:
    """
    Boxplot du parametre par groupe, avec etoiles de significativite
    sur les paires significatives du post-hoc de Dunn.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    res = compare_groups(
        df, parameter=parameter, grouping_col=grouping_col,
        min_group_size=min_group_size,
    )

    groups_order = sorted(res.group_sizes, key=lambda g: -res.group_sizes[g])
    data = df[[parameter, grouping_col]].dropna()
    data = data[data[grouping_col].astype(str).isin(groups_order)]
    data[grouping_col] = data[grouping_col].astype(str)

    # Jolis labels pour les axes et titres
    param_labels = {
        "CALCIUM":             r"Ca$^{2+}$ (mg/L)",
        "MAGNESIUM":           r"Mg$^{2+}$ (mg/L)",
        "HYDROGENOCARBONATES": r"HCO$_3^{-}$ (mg/L)",
        "SULFATES":            r"SO$_4^{2-}$ (mg/L)",
        "CHLORURES":           r"Cl$^{-}$ (mg/L)",
        "PH ":                 "pH",
        "IL":                  "Langelier Saturation Index",
        "ryznar":              "Ryznar Stability Index",
        "Larson":              "Larson-Skold Index",
        "Bason":               "Basson Index",
    }
    grouping_labels = {
        "geological_zone":  "Geological zone",
        "Langelier_grade":  "Langelier grade",
        "Ryznar_grade":     "Ryznar grade",
        "Larson_grade":     "Larson grade",
        "Basson_grade":     "Basson grade",
    }
    y_label = param_labels.get(parameter, parameter)
    x_label = grouping_labels.get(grouping_col, grouping_col)
    title_param = param_labels.get(parameter, parameter)

    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(
        data=data, x=grouping_col, y=parameter,
        order=groups_order, showfliers=False, ax=ax,
        hue=grouping_col, palette="Set2", legend=False,
    )
    ax.set_title(
        f"{title_param} by {x_label.lower()}\n"
        f"Kruskal-Wallis: H={res.kruskal_H:.1f}, "
        f"p={res.kruskal_p:.2e} {_sig_stars(res.kruskal_p)}  "
        f"($\\varepsilon^2={res.epsilon_squared:.3f}$, N={res.n_samples:,})"
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.xticks(rotation=30, ha="right")

    # Annotations de significativite (paires les plus significatives)
    if res.dunn_matrix is not None:
        dunn = res.dunn_matrix
        pairs = []
        for i, g1 in enumerate(groups_order):
            for j, g2 in enumerate(groups_order):
                if j <= i or g1 not in dunn.index or g2 not in dunn.columns:
                    continue
                p = dunn.loc[g1, g2]
                if p < 0.05:
                    pairs.append((i, j, p))
        # On annote les plus significatives d'abord
        pairs.sort(key=lambda x: x[2])
        pairs = pairs[:max_pairs_annotated]

        y_max = data[parameter].quantile(0.95)
        y_min = data[parameter].quantile(0.05)
        span = y_max - y_min
        step = span * 0.07

        for k, (i, j, p) in enumerate(pairs):
            y = y_max + (k + 1) * step
            ax.plot([i, i, j, j], [y, y + step/3, y + step/3, y],
                    lw=1, c="black")
            ax.text((i + j) / 2, y + step/3, _sig_stars(p),
                    ha="center", va="bottom", fontsize=10)

        ax.set_ylim(top=y_max + (len(pairs) + 2) * step)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure -> {output_path}")


def plot_all_parameters(
    df: pd.DataFrame,
    parameters: Sequence[str],
    grouping_cols: Sequence[str],
    output_dir: str | Path,
    min_group_size: int = 10,
) -> None:
    """Genere un boxplot annote pour chaque couple (parameter, grouping)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for grouping in grouping_cols:
        if grouping not in df.columns:
            continue
        for param in parameters:
            if param not in df.columns:
                continue
            try:
                plot_boxplot_with_significance(
                    df, parameter=param, grouping_col=grouping,
                    output_path=output_dir / f"boxplot_{param}_{grouping}.png",
                    min_group_size=min_group_size,
                )
            except Exception as e:
                print(f"  [SKIP plot] {param} / {grouping} : {e}")

def plot_article_geology_figure(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    output_path: str | Path,
    grouping_col: str = "geological_zone",
    parameters: Sequence[str] = ("CALCIUM", "HYDROGENOCARBONATES", "SULFATES", "PH "),
) -> None:
    """
    Figure publication-ready en 4 panneaux :
    boxplots des moyennes annuelles communales par zone géologique.

    - Pas d'annotations de Dunn pour garder la figure lisible
    - Affiche seulement p-value globale et epsilon²
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if grouping_col not in df.columns:
        raise KeyError(f"Missing column: {grouping_col}")

    # Labels publication
    label_map = {
        "CALCIUM": r"Ca$^{2+}$ (mg/L)",
        "HYDROGENOCARBONATES": r"HCO$_3^-$ (mg/L)",
        "SULFATES": r"SO$_4^{2-}$ (mg/L)",
        "PH ": "pH",
    }

    # Ordre des groupes : par médiane de HCO3 si dispo, sinon alphabétique
    if "HYDROGENOCARBONATES" in df.columns:
        tmp_order = df[[grouping_col, "HYDROGENOCARBONATES"]].dropna().copy()
        tmp_order[grouping_col] = tmp_order[grouping_col].astype(str)
        order = (
            tmp_order.groupby(grouping_col)["HYDROGENOCARBONATES"]
            .median()
            .sort_values()
            .index
            .tolist()
        )
    else:
        order = sorted(df[grouping_col].dropna().astype(str).unique().tolist())

    # Résumé des tests pour la géologie
    summary_geo = summary[summary["grouping"] == grouping_col].copy()

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
    axes = axes.flatten()

    panel_letters = ["A", "B", "C", "D"]

    for i, (ax, param) in enumerate(zip(axes, parameters)):
        if param not in df.columns:
            ax.set_visible(False)
            continue

        data = df[[param, grouping_col]].dropna().copy()
        data[grouping_col] = data[grouping_col].astype(str)
        data = data[data[grouping_col].isin(order)]

        sns.boxplot(
            data=data,
            x=grouping_col,
            y=param,
            order=order,
            showfliers=False,
            ax=ax,
            color="white",
            linewidth=1.1,
        )

        ax.set_xlabel("")
        ax.set_ylabel(label_map.get(param, param), fontsize=11)
        ax.tick_params(axis="x", rotation=35, labelsize=9)
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(axis="y", alpha=0.25)

        # Lettre du panneau
        ax.text(
            0.01, 0.99, panel_letters[i],
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=12, fontweight="bold"
        )

        # Annotation statistique
        row = summary_geo[summary_geo["parameter"] == param]
        if not row.empty:
            row = row.iloc[0]
            p = row["p_value"]
            eps2 = row["epsilon^2"] if "epsilon^2" in row.index else row["epsilon^2"] if "epsilon^2" in summary_geo.columns else row["epsilon^2"]
            n = int(row["N"])

            if p < 0.001:
                p_txt = "p < 0.001"
            else:
                p_txt = f"p = {p:.3g}"

            stat_txt = f"Kruskal–Wallis\n{p_txt}\nε² = {eps2:.3f}\nN = {n}"
            ax.text(
                0.98, 0.98, stat_txt,
                transform=ax.transAxes,
                ha="right", va="top",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="0.6")
            )

    # Si moins de 4 variables
    for j in range(len(parameters), 4):
        axes[j].set_visible(False)

    fig.suptitle(
        "Major chemical parameters across geological zones\n"
        "Annual commune-scale means",
        fontsize=14,
        y=0.995
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    print(f"Figure article -> {output_path}")
    print(f"Figure article -> {output_path.with_suffix('.pdf')}")
# ---------------------------------------------------------------------------
# 6. Pipeline complet (tout en un)
# ---------------------------------------------------------------------------

def run_reviewer_response(
    df: pd.DataFrame,
    parameters: Sequence[str],
    geology_geojson: str | Path | None,
    output_dir: str | Path = "stats",
    communes_geojson: str | Path | None = None,
    geology_field: str = "DESCR",
    simplify_lithology: bool = True,
    macro_lithology: bool = False,
    index_cols: Sequence[str] = ("IL", "ryznar", "Larson", "Bason"),
    min_group_size: int = 10,
    aggregate: str = "commune-year",
) -> dict:
    """
    Pipeline complet repondant au commentaire du reviewer #4.

    Produit dans output_dir :
      - summary_tests.csv       : tous les tests Kruskal-Wallis
      - summary_tests.tex       : tableau publication LaTeX
      - dunn_<param>_<group>.csv: matrices de Dunn pour chaque test significatif
      - boxplot_<param>_<group>.png : figures annotees

    Parametres
    ----------
    parameters       : ["CALCIUM", "MAGNESIUM", "HYDROGENOCARBONATES", ...]
    geology_geojson  : GeoJSON BRGM pour classification geologique (ou None)
    index_cols       : indices pour lesquels on cree aussi une classification
                       par grade (Langelier, Ryznar, ...)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("  Statistical analysis (reviewer #4 response)")
    print(f"{'='*60}\n")

    work = df.copy()
    grouping_cols = []

    # --- Classification geologique ---
    if geology_geojson is not None:
        print("[1/3] Geological zone classification...")
        work = assign_geological_zone(
            work,
            geology_geojson=geology_geojson,
            zone_field=geology_field,
            communes_geojson=communes_geojson,
            simplify=simplify_lithology,
            macro=macro_lithology,
        )
        grouping_cols.append("geological_zone")

    # --- Classification par grade d'indice ---
    print("\n[2/3] Index grade classification...")
    english_grade_names = {
        "IL":     "Langelier_grade",
        "ryznar": "Ryznar_grade",
        "Larson": "Larson_grade",
        "Bason":  "Basson_grade",
    }
    for idx in index_cols:
        if idx in work.columns:
            work = assign_index_grade(work, idx)
            g_col = english_grade_names.get(idx, f"{idx}_grade")
            grouping_cols.append(g_col)
            print(f"  {g_col} created")

    # --- Agregation (recommandee pour satisfaire l'hypothese d'independance) ---
    if aggregate == "commune-year":
        print("\n[2b/3] Commune x year aggregation...")
        work = aggregate_commune_year(
            work, parameters=parameters, grouping_cols=grouping_cols,
        )
    elif aggregate == "none":
        print("\n[2b/3] No aggregation (tests on raw measurements)")
    else:
        raise ValueError(f"aggregate must be 'commune-year' or 'none' (got: {aggregate!r})")

    # --- Tests ---
    print("\n[3/3] Kruskal-Wallis tests + Dunn post-hoc...")
    summary = run_full_analysis(
        work, parameters=parameters, grouping_cols=grouping_cols,
        output_dir=output_dir, min_group_size=min_group_size,
    )

    # --- Figures ---
    print("\nGenerating annotated boxplots...")
    plot_all_parameters(
        work, parameters=parameters, grouping_cols=grouping_cols,
        output_dir=output_dir, min_group_size=min_group_size,
    )

    print(f"\n{'='*60}")
    print(f"  Done. Results in: {output_dir}")
    print(f"{'='*60}")
    # Sauvegarde du DataFrame classifie utilise pour les stats
    work.to_csv(output_dir / "classified_annual_data.csv", index=False)

    # Figure compacte pour l'article
    if "geological_zone" in work.columns:
        print("\nGenerating compact figure for the article...")
        plot_article_geology_figure(
            df=work,
            summary=summary,
            output_path=output_dir / "figure_article_geology_boxplots.png",
            grouping_col="geological_zone",
            parameters=("CALCIUM", "HYDROGENOCARBONATES", "SULFATES", "PH "),
        )
    return {"df_classified": work, "summary": summary}


# ---------------------------------------------------------------------------
# 7. Helper : inspection du fichier BRGM
# ---------------------------------------------------------------------------

def inspect_geology_file(geology_path: str | Path, n_samples: int = 8) -> None:
    """
    Affiche les colonnes et quelques exemples de valeurs du fichier BRGM.
    Utile pour identifier le bon --geology-field avant de lancer l'analyse.
    """
    try:
        import geopandas as gpd
    except ImportError:
        raise ImportError("pip install geopandas")

    print(f"\nInspection de : {geology_path}")
    print("=" * 60)
    gdf = gpd.read_file(geology_path)

    print(f"Nb de polygones : {len(gdf):,}")
    print(f"CRS             : {gdf.crs}")
    print(f"\nColonnes disponibles :")
    for col in gdf.columns:
        if col == "geometry":
            continue
        n_unique = gdf[col].nunique()
        sample = gdf[col].dropna().astype(str).iloc[0] if gdf[col].notna().any() else ""
        sample = sample[:60] + ("..." if len(sample) > 60 else "")
        print(f"  {col:20s}  ({n_unique:>5} valeurs uniques)  ex: {sample!r}")

    # Suggerer le meilleur candidat pour la lithologie
    candidates = [c for c in ["DESCR", "NOTATION", "LITHO", "LITHOLOGY", "CLASSE", "TYPE_GEOL"]
                  if c in gdf.columns]
    if candidates:
        best = candidates[0]
        print(f"\nChamp recommande pour --geology-field : '{best}'")
        print(f"Exemples de valeurs dans '{best}' :")
        for v in gdf[best].dropna().unique()[:n_samples]:
            fam = simplify_lithology(v)
            print(f"  {v[:80]!r:84s} -> {fam}")


# ---------------------------------------------------------------------------
# 8. CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyse statistique SISE-Eaux (reponse reviewer)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- inspect-geology ---
    p_ins = sub.add_parser(
        "inspect-geology",
        help="Lister les champs du fichier BRGM avant de lancer l'analyse",
    )
    p_ins.add_argument("--geology", required=True,
                       help="Fichier BRGM (.shp ou .geojson)")

    # --- run ---
    p_run = sub.add_parser("run", help="Lancer l'analyse statistique complete")
    p_run.add_argument("--csv", required=True, nargs="+",
                       help="CSV annuels (sortie de sise_pipeline process)")
    p_run.add_argument("--geology", required=True,
                       help="Fichier BRGM (.shp ou .geojson)")
    p_run.add_argument("--communes", required=True,
                       help="GeoJSON des communes francaises")
    p_run.add_argument("--geology-field", default="DESCR",
                       help="Champ lithologique (defaut: 'DESCR' -- voir inspect-geology)")
    p_run.add_argument("--out", default="stats",
                       help="Dossier de sortie (defaut: stats/)")
    p_run.add_argument("--params", nargs="+",
                       default=["CALCIUM", "MAGNESIUM", "HYDROGENOCARBONATES",
                                "SULFATES", "CHLORURES", "PH "],
                       help="Parametres a tester")
    p_run.add_argument("--min-group-size", type=int, default=10,
                       help="Taille min. d'un groupe (defaut: 10)")
    p_run.add_argument(
        "--no-simplify",
        action="store_true",
        help="Ne pas regrouper les descriptions BRGM en familles (garder brut)",
    )
    p_run.add_argument(
        "--macro",
        action="store_true",
        help="Regroupement grossier en 5 macro-categories "
             "(Sedimentaire, Magmatique-volcanique, Magmatique-plutonique, "
             "Metamorphique, Alluvions)",
    )
    p_run.add_argument(
        "--aggregate",
        choices=["commune-year", "none"],
        default="commune-year",
        help=(
            "Unite d'observation des tests. "
            "'commune-year' (defaut, recommande) : une moyenne par commune x annee "
            "-> satisfait l'hypothese d'independance de Kruskal-Wallis. "
            "'none' : teste les mesures brutes (surpondere les communes avec "
            "beaucoup de prelevements, viole l'independance)."
        ),
    )

    args = parser.parse_args()

    if args.command == "inspect-geology":
        inspect_geology_file(args.geology)

    elif args.command == "run":
        from sise_pipeline import (
            load_csv_files, compute_indices, geocode_from_geojson,
            COL_PH, COL_PH_EQ, COL_TEMP, COL_CA, COL_MG, COL_HCO3,
            COL_SO4, COL_CL, COL_NO3, COL_NA, COL_K,
        )

        # Colonnes indispensables pour calculer tous les indices demandes
        # + celles explicitement demandees en --params.
        # On charge uniquement celles-ci pour eviter MemoryError sur les
        # exports SISE qui ont des centaines de parametres rares.
        required_cols = {
            COL_PH, COL_PH_EQ, COL_TEMP,
            COL_CA, COL_MG, COL_HCO3,
            COL_SO4, COL_CL, COL_NO3, COL_NA, COL_K,
        } | set(args.params)

        df = load_csv_files(args.csv, keep_columns=required_cols)
        df = compute_indices(df)
        df = geocode_from_geojson(df, args.communes)

        run_reviewer_response(
            df=df,
            parameters=args.params,
            geology_geojson=args.geology,
            communes_geojson=args.communes,
            geology_field=args.geology_field,
            simplify_lithology=not args.no_simplify,
            macro_lithology=args.macro,
            output_dir=args.out,
            min_group_size=args.min_group_size,
            aggregate=args.aggregate,
        )