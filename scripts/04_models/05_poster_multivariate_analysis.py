#!/usr/bin/env python3
"""
multivariate_analysis.py — Multivariate Linear Regression Pipeline
====================================================================
Project : CVS Conversation Analysis (Dyadic Co-Viewing)
Purpose : Build a compact multivariate model predicting social connection
          quality from conversational features (structural + LLM-annotated).
Author  : [Your Name]
Date    : 2026-04-04

Pipeline overview
-----------------
1. Load & merge  — join structural, semantic, LLM, and outcome CSVs on dyad ID
2. QC / sanity   — check missingness, sample size, variable distributions
3. Composite     — PCA on outcome variables → "connection quality" composite_z
4. Screen        — univariate regressions against composite_z, keep p < .10
5. Model         — OLS linear regression (composite_z) with 3–4 features
6. Evaluate      — LOO-CV R², RMSE, bootstrap 95% CIs on coefficients
7. Figures       — PCA loadings, coefficient plot, actual vs predicted (poster-ready)

Usage
-----
    python multivariate_analysis.py

    Inputs are read from 04_data and 05_analysis_outputs.
    Outputs are written to 05_analysis_outputs/multivariate_output.

Reproducibility
---------------
- Random seed is fixed globally (SEED = 42).
- All stochastic steps (bootstrap, median split ties) are seeded.
- Requirements: numpy, pandas, scikit-learn, statsmodels, matplotlib, seaborn, scipy
- Tested with Python 3.10+

Notes for N=24
--------------
- With 24 observations, we limit to 3–4 predictors (~5–6 obs per predictor).
- Effect sizes (Cohen's d, pseudo-R²) are reported alongside p-values.
- Bootstrap CIs give stability estimates without relying on asymptotic p-values.
"""

# =============================================================================
# 0. IMPORTS AND CONFIGURATION
# =============================================================================
import os
import sys
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for script/CI use
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)

THEME = {
    'green':    '#6AAB9C',
    'salmon':   '#FA9284',
    'red':      '#E06C78',
    'blue':     '#5874DC',
    'navy':     '#384E78',
    'text':     '#384E78',
    'grid':     '#EEEEEE',
    'zero_line':'#AAAAAA',
}

# ---- User-configurable paths ------------------------------------------------
PROJECT = Path(__file__).resolve().parents[2]  # cvs_conversation/
DATA_DIR = PROJECT / "04_data"              # directory containing CSVs
# OUTPUT_DIR = Path("./multivariate_output")   # all outputs go here

# ---- Input file names --------------------------------------------------------
STRUCTURAL_FILE = DATA_DIR / "structural_dyad_analysis_mapped.csv"
SEMANTIC_FILE   = DATA_DIR / "scientific_dyad_analysis_results.csv"
LLM_FILE        = PROJECT / "05_analysis_outputs" / "llm_annotation_output" / "dyad_features.csv"
OUTCOMES_FILE   = DATA_DIR / "outcomes.csv"
ENGAGEMENT_FILE = PROJECT / "05_analysis_outputs" / "llm_annotation_output" / "conversation_level.csv"
OUTPUT_DIR = PROJECT / "05_analysis_outputs" / "multivariate_output"

# ---- Outcome columns (as they appear in outcomes.csv) ------------------------
OUTCOME_COLS = [
    "dyad_partner_eval_mean",
    "dyad_shared_reality_mean",
    "dyad_enjoyment_mean",
    # dyad_solo_mean excluded from PCA composite
]

# ---- Analysis parameters -----------------------------------------------------
SEED = 42                    # global random seed
BOOTSTRAP_N = 2000           # number of bootstrap resamples for CIs
MAX_PREDICTORS = 4           # hard cap on predictors in final model
UNIVARIATE_THRESHOLD = 0.10  # p-value cutoff for univariate screening
SOLO_REVERSE = True          # reverse-score solo if it loads negatively on PC1

# =============================================================================
# ▼▼▼  MANUAL FEATURE CONTROL — EDIT HERE  ▼▼▼
# =============================================================================

# -- Drop specific features before any analysis --------------------------------
# Add column names you want to exclude from the entire pipeline.
# Run once with FEATURES_TO_DROP = [] to see the full feature list printed
# by inspect_and_drop_features(), then add names here and re-run.
#
# Examples:
#   FEATURES_TO_DROP = ["ttr_asym", "q_count_mean", "sem_mean_sentiment"]
FEATURES_TO_DROP: list[str] = [
    'Unnamed: 0',     # CSV row index
    'soc_diff_film1', # unclear source
    'q_count_mean',   # punctuation-based question count, unreliable
    'q_count_asym',   # punctuation-based question count, unreliable
    'i_rate_mean',    # manually excluded
    'i_rate_asym',    # manually excluded
]

# -- Force specific features into the final model ------------------------------
# Set to a list of column names to bypass automatic screening + selection.
# Set to None to use automatic selection (Steps 4–5).
#
# Examples:
#   MANUAL_FEATURES = ["bc_rate_mean", "q_count_asym", "sem_semantic_similarity"]
MANUAL_FEATURES: list[str] | None = ["verbal_agreement_elaborated_n", "questions_asym", "hedging_asym"]

# =============================================================================
# ▲▲▲  END OF MANUAL FEATURE CONTROL  ▲▲▲
# =============================================================================

# ---- Reproducibility ---------------------------------------------------------
np.random.seed(SEED)

# ---- Plot style --------------------------------------------------------------
sns.set_theme(style="whitegrid", font_scale=1.1)
POSTER_FIG_DPI = 1200
COLOR_PALETTE = sns.color_palette("colorblind")


# =============================================================================
# 1. DATA LOADING AND MERGING
# =============================================================================
def load_and_merge(data_dir) -> pd.DataFrame:
    """
    Load the four source CSVs, extract pair_id, collapse condition-level
    files to dyad level (mean across conditions), and merge everything
    on pair_id.

    Returns
    -------
    df : pd.DataFrame
        One row per dyad, with all features + outcomes.
    """
    data_dir = Path(data_dir)  # safety cast: works whether str or Path is passed

    print("\n" + "=" * 60)
    print("STEP 1: LOADING AND MERGING DATA")
    print("=" * 60)

    # --- Helper: extract numeric pair_id from "Dyad ID" column ----------------
    def extract_pair_id(df: pd.DataFrame) -> pd.DataFrame:
        if "Dyad ID" in df.columns:
            df = df.copy()
            df["pair_id"] = (
                df["Dyad ID"]
                .str.extract(r"dyad(\d+)", expand=False)
                .astype(int)
            )
        return df

    # --- Helper: collapse to dyad level (mean across conditions) --------------
    def collapse_to_dyad(df: pd.DataFrame, id_col: str = "pair_id") -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if id_col in numeric_cols:
            numeric_cols.remove(id_col)
        return df.groupby(id_col)[numeric_cols].mean().reset_index()

    def resolve_input(path):
        path = Path(path)
        return path if path.is_absolute() else data_dir / path

    # --- Load each file -------------------------------------------------------
    paths = {
        "structural": resolve_input(STRUCTURAL_FILE),
        "semantic":   resolve_input(SEMANTIC_FILE),
        "llm":        resolve_input(LLM_FILE),
        "outcomes":   resolve_input(OUTCOMES_FILE),
    }

    for name, path in paths.items():
        if not path.exists():
            sys.exit(f"ERROR: {name} file not found at {path}")
        print(f"  ✓ Found {name}: {path.name}")

    structural = pd.read_csv(paths["structural"])
    semantic   = pd.read_csv(paths["semantic"])
    llm        = pd.read_csv(paths["llm"])
    outcomes   = pd.read_csv(paths["outcomes"])

    # --- Extract pair_id and collapse -----------------------------------------
    structural = collapse_to_dyad(extract_pair_id(structural))
    semantic   = collapse_to_dyad(extract_pair_id(semantic))
    llm        = collapse_to_dyad(extract_pair_id(llm))
    
    

    # Outcomes are already dyad-level; just ensure pair_id exists
    if "pair_id" not in outcomes.columns:
        sys.exit("ERROR: outcomes.csv must contain a 'pair_id' column.")

    print(f"\n  Dyad counts after collapsing:")
    print(f"    structural : {len(structural)}")
    print(f"    semantic   : {len(semantic)}")
    print(f"    llm        : {len(llm)}")
    print(f"    outcomes   : {len(outcomes)}")

    # --- Merge ----------------------------------------------------------------
    df = outcomes.copy()
    for source, name in [(structural, "structural"), (semantic, "semantic"), (llm, "llm")]:
        before = len(df)
        df = df.merge(source, on="pair_id", how="left", suffixes=("", f"_{name}_dup"))
        after = len(df)
        if after != before:
            print(f"  WARNING: merge with {name} changed row count {before} → {after}")

    # Drop any duplicate columns created by merge
    dup_cols = [c for c in df.columns if c.endswith("_dup")]
    if dup_cols:
        print(f"  Dropping duplicate columns: {dup_cols}")
        df.drop(columns=dup_cols, inplace=True)
    before = len(df)
    df = df[df["pair_id"] != 10].reset_index(drop=True)
    print(f"  Dropped dyad10 (missing feature data): {before} → {len(df)} dyads")

    print(f"\n  Final merged dataset: {df.shape[0]} dyads × {df.shape[1]} columns")
    return df


# =============================================================================
# 1.5. FEATURE RESTRUCTURING
# =============================================================================
def restructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    1. Convert per-speaker A/B features to mean + asym
    2. Drop redundant features
    """
    print("\n" + "=" * 60)
    print("STEP 1.5: FEATURE RESTRUCTURING")
    print("=" * 60)

    df = df.copy()

    # # ------------------------------------------------------------------
    # # 1. A/B → mean + asym
    # # ------------------------------------------------------------------
    # speaker_feats = {
    #     'ttr'    : ('ttr_A',     'ttr_B'),
    #     'i_rate' : ('i_rate_A',  'i_rate_B'),
    #     'we_rate': ('we_rate_A', 'we_rate_B'),
    #     'bc_rate': ('bc_rate_A', 'bc_rate_B'),
    # }

    # converted = []
    # for feat, (col_a, col_b) in speaker_feats.items():
    #     df[f'{feat}_mean'] = (df[col_a] + df[col_b]) / 2
    #     df[f'{feat}_asym'] = (df[col_a] - df[col_b]).abs()
    #     df.drop(columns=[col_a, col_b], inplace=True)
    #     converted.append(feat)
    #     print(f"  ✓ {col_a} + {col_b} → {feat}_mean, {feat}_asym")

    # ------------------------------------------------------------------
    # 2. Drop redundant features
    # ------------------------------------------------------------------
    to_drop = [
        # too many missing
        'on_topic_q4', 'on_topic_q5',
        # overlap with on_topic_rate
        'on_topic_q1', 'on_topic_q2', 'on_topic_q3',
        # overlap with questions_mean/asym
        'question_count',
        # standardized version of sem_delta; raw is sufficient
        'sem_delta_z',
        # cross-time change, not our research focus
        'sem_delta', 'sent_delta',
        # ddi series not used
        'ddi_partner_eval', 'ddi_shared_reality',
        'ddi_enjoyment', 'ddi_solo',
    ]

    existing_drops = [c for c in to_drop if c in df.columns]
    missing_drops  = [c for c in to_drop if c not in df.columns]

    df.drop(columns=existing_drops, inplace=True)
    print(f"\n  Dropped {len(existing_drops)} redundant features: {existing_drops}")
    if missing_drops:
        print(f"  ⚠ Not found (skipped): {missing_drops}")

    print(f"\n  Features after restructuring: {df.shape[1]} columns")
    return df


# =============================================================================
# 2. QC / SANITY CHECKS
# =============================================================================
def run_qc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run quality-control checks on the merged dataset:
    - Missingness per column
    - Sample size verification
    - Outcome variable distributions (mean, sd, skew, range)
    - Flag any constant or near-constant columns

    Returns the cleaned dataframe (drops columns with >50% missing).
    """
    print("\n" + "=" * 60)
    print("STEP 2: QUALITY CONTROL / SANITY CHECKS")
    print("=" * 60)

    n = len(df)
    print(f"\n  Sample size: N = {n}")
    if n < 20:
        print("  ⚠ WARNING: N < 20, multivariate results will be very unstable.")
    if n != 24:
        print(f"  ⚠ NOTE: Expected N=24 dyads, found N={n}.")

    # --- Missingness ----------------------------------------------------------
    missing = df.isnull().sum()
    missing_pct = (missing / n * 100).round(1)
    high_missing = missing_pct[missing_pct > 0].sort_values(ascending=False)
    if len(high_missing) > 0:
        print(f"\n  Columns with missing data:")
        for col, pct in high_missing.items():
            print(f"    {col}: {pct}% missing ({missing[col]}/{n})")
    else:
        print("\n  No missing data detected. ✓")

    # Drop columns with >50% missing
    drop_cols = missing_pct[missing_pct > 50].index.tolist()
    if drop_cols:
        print(f"\n  Dropping columns with >50% missing: {drop_cols}")
        df = df.drop(columns=drop_cols)

    # --- Outcome distributions ------------------------------------------------
    print(f"\n  Outcome variable distributions:")
    print(f"  {'Variable':<30} {'Mean':>7} {'SD':>7} {'Skew':>7} {'Min':>7} {'Max':>7}")
    print("  " + "-" * 75)
    for col in OUTCOME_COLS:
        if col in df.columns:
            s = df[col].dropna()
            print(f"  {col:<30} {s.mean():7.3f} {s.std():7.3f} "
                  f"{s.skew():7.3f} {s.min():7.3f} {s.max():7.3f}")
        else:
            print(f"  {col:<30}  *** NOT FOUND ***")

    # --- Constant / near-constant columns -------------------------------------
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    constant_cols = [c for c in numeric_cols if df[c].nunique() <= 1]
    if constant_cols:
        print(f"\n  ⚠ Constant columns (will be dropped): {constant_cols}")
        df = df.drop(columns=constant_cols)

    # --- Outcome correlation preview ------------------------------------------
    avail_outcomes = [c for c in OUTCOME_COLS if c in df.columns]
    if len(avail_outcomes) >= 2:
        print(f"\n  Outcome intercorrelations (Pearson r):")
        corr = df[avail_outcomes].corr()
        for i, c1 in enumerate(avail_outcomes):
            for c2 in avail_outcomes[i + 1:]:
                r = corr.loc[c1, c2]
                print(f"    {c1} × {c2}: r = {r:.3f}")

    return df


# =============================================================================
# 2b. INSPECT AND DROP FEATURES
# =============================================================================
def inspect_and_drop_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Print a full inventory of every feature column (mean, SD, missing %,
    min, max) so you can decide what to drop, then remove whatever is listed
    in FEATURES_TO_DROP.

    Call this after run_qc() and before build_composite().
    Edit FEATURES_TO_DROP at the top of the file and re-run to apply drops.
    """
    print("\n" + "=" * 60)
    print("STEP 2b: FEATURE INSPECTION & MANUAL DROP")
    print("=" * 60)

    exclude = (
        set(OUTCOME_COLS)
        | {"pair_id", "composite_z", "connection_group", "Order", "Dyad ID"}
    )
    feature_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude
    ]

    n = len(df)
    print(f"\n  All available features ({len(feature_cols)}):")
    print(f"\n  {'#':<4} {'Feature':<35} {'Mean':>9} {'SD':>9} "
          f"{'Missing%':>9} {'Min':>9} {'Max':>9}")
    print("  " + "─" * 87)
    for i, col in enumerate(feature_cols, 1):
        s       = df[col].dropna()
        missing = (df[col].isna().sum() / n * 100)
        print(f"  {i:<4} {col:<35} {s.mean():>9.3f} {s.std():>9.3f} "
              f"{missing:>8.1f}% {s.min():>9.3f} {s.max():>9.3f}")

    # ── Apply drops ──────────────────────────────────────────────────────────
    if not FEATURES_TO_DROP:
        print(f"\n  FEATURES_TO_DROP is empty — no features dropped.")
        print(f"  → Add column names to FEATURES_TO_DROP at the top of the file")
        print(f"    to exclude them from all downstream analysis.")
        return df

    not_found = [f for f in FEATURES_TO_DROP if f not in df.columns]
    if not_found:
        print(f"\n  ⚠  Not found in dataset (check spelling): {not_found}")

    to_drop = [f for f in FEATURES_TO_DROP if f in df.columns]
    if to_drop:
        df = df.drop(columns=to_drop)
        print(f"\n  Dropped {len(to_drop)} feature(s):")
        for f in to_drop:
            print(f"    ✗  {f}")
        print(f"\n  Remaining features: {len(feature_cols) - len(to_drop)}")
    return df


# =============================================================================
# 3. COMPOSITE OUTCOME (PCA)
# =============================================================================
def build_composite(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Build a composite "connection quality" score via PCA on outcome variables.

    Logic:
    - Standardize outcomes → run PCA
    - Inspect PC1 loadings; if solo loads negatively, reverse-score it
    - Report variance explained and loadings
    - Median-split composite into high/low groups for logistic regression

    Returns
    -------
    df : pd.DataFrame with new columns: 'composite_z', 'connection_group'
    pca_info : dict with loadings, variance explained, etc.
    """
    print("\n" + "=" * 60)
    print("STEP 3: COMPOSITE OUTCOME (PCA)")
    print("=" * 60)

    avail_outcomes = [c for c in OUTCOME_COLS if c in df.columns]
    if len(avail_outcomes) < 3:
        sys.exit(f"ERROR: Need at least 3 outcome variables, found {len(avail_outcomes)}")

    # --- Handle missing outcome data (listwise deletion for PCA) --------------
    outcome_data = df[avail_outcomes].copy()
    complete_mask = outcome_data.notnull().all(axis=1)
    n_dropped = (~complete_mask).sum()
    if n_dropped > 0:
        print(f"  ⚠ Dropping {n_dropped} dyads with missing outcome data for PCA.")

    outcome_complete = outcome_data[complete_mask]

    # --- Reverse-score solo if requested --------------------------------------
    if SOLO_REVERSE and "dyad_solo_mean" in avail_outcomes:
        print("  Reverse-scoring 'dyad_solo_mean' (high solo → low connection).")
        outcome_complete = outcome_complete.copy()
        outcome_complete["dyad_solo_mean"] = -outcome_complete["dyad_solo_mean"]

    # --- Standardize ----------------------------------------------------------
    scaler = StandardScaler()
    outcome_scaled = scaler.fit_transform(outcome_complete)

    # --- PCA ------------------------------------------------------------------
    pca = PCA(n_components=len(avail_outcomes), random_state=SEED)
    pca_scores = pca.fit_transform(outcome_scaled)

    # --- Report ---------------------------------------------------------------
    print(f"\n  PCA results ({len(avail_outcomes)} components):")
    print(f"  {'Component':<12} {'Var Explained':>14} {'Cumulative':>12}")
    print("  " + "-" * 40)
    cum = 0
    for i, ve in enumerate(pca.explained_variance_ratio_):
        cum += ve
        print(f"  {'PC' + str(i + 1):<12} {ve:14.1%} {cum:12.1%}")

    print(f"\n  PC1 loadings (connection quality interpretation):")
    loadings = pd.Series(pca.components_[0], index=avail_outcomes)
    for col, loading in loadings.items():
        direction = "+" if loading > 0 else "-"
        label = col.replace("dyad_", "").replace("_mean", "")
        print(f"    {label:<25} {direction}{abs(loading):.3f}")

    # --- Sanity check: does PC1 make conceptual sense? ------------------------
    #   We expect partner_eval, shared_reality, enjoyment to load positively
    #   and solo (if reversed) to also load positively.
    positive_outcomes = ["dyad_partner_eval_mean", "dyad_shared_reality_mean",
                         "dyad_enjoyment_mean"]
    positive_loadings = [loadings.get(c, 0) for c in positive_outcomes if c in loadings.index]
    if any(l < 0 for l in positive_loadings):
        print("  ⚠ NOTE: Some 'positive' outcomes load negatively on PC1.")
        print("    Consider inspecting PC2 or adjusting the composite.")
        print("    Proceeding with PC1 as-is; document this decision.")

    # --- Assign composite scores to full dataframe ----------------------------
    df = df.copy()
    df.loc[complete_mask, "composite_z"] = pca_scores[:, 0]

    print(f"\n  composite_z — mean: {df['composite_z'].mean():.3f}  "
          f"SD: {df['composite_z'].std():.3f}  "
          f"range: [{df['composite_z'].min():.2f}, {df['composite_z'].max():.2f}]")

    pca_info = {
        "loadings": loadings.to_dict(),
        "variance_explained": pca.explained_variance_ratio_.tolist(),
        "n_complete": int(complete_mask.sum()),
        "solo_reversed": SOLO_REVERSE,
    }

    return df, pca_info


# =============================================================================
# 4. UNIVARIATE SCREENING
# =============================================================================
def univariate_screen(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run univariate OLS regressions: each feature → composite_z.
    Control for 'Order' if available.
    Return a summary table sorted by p-value.
    """
    print("\n" + "=" * 60)
    print("STEP 4: UNIVARIATE SCREENING (feature → composite)")
    print("=" * 60)

    # Identify feature columns (everything that's not an outcome, ID, or composite)
    exclude = (
        set(OUTCOME_COLS)
        | {"pair_id", "composite_z", "connection_group", "Order", "Dyad ID"}
    )
    feature_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude
    ]
    print(f"  Screening {len(feature_cols)} features against composite_z")

    has_order = "Order" in df.columns
    if has_order:
        print("  Controlling for Order (counterbalancing covariate).")

    results = []
    for feat in feature_cols:
        subset = df[["composite_z", feat] + (["Order"] if has_order else [])].dropna()
        if len(subset) < 10:
            continue  # skip features with too few observations
        y = subset["composite_z"]
        X = subset[[feat]]
        if has_order:
            X = pd.concat([X, subset[["Order"]]], axis=1)
        X = sm.add_constant(X)

        try:
            model = sm.OLS(y, X).fit()
            beta = model.params[feat]
            p = model.pvalues[feat]
            r2 = model.rsquared
            n = len(subset)
            results.append({
                "feature": feat,
                "beta": beta,
                "std_beta": beta * subset[feat].std() / y.std(),
                "p_value": p,
                "r_squared": r2,
                "n": n,
            })
        except Exception as e:
            print(f"  ⚠ Skipping {feat}: {e}")

    screen_df = pd.DataFrame(results).sort_values("p_value").reset_index(drop=True)

    # --- Report top candidates ------------------------------------------------
    sig = screen_df[screen_df["p_value"] < UNIVARIATE_THRESHOLD]
    print(f"\n  Features passing p < {UNIVARIATE_THRESHOLD} threshold: {len(sig)}")
    if len(sig) > 0:
        print(f"\n  {'Feature':<30} {'Std β':>8} {'p':>8} {'R²':>8} {'N':>5}")
        print("  " + "-" * 65)
        for _, row in sig.iterrows():
            star = "***" if row["p_value"] < .001 else (
                   "**" if row["p_value"] < .01 else (
                   "*" if row["p_value"] < .05 else "†"))
            print(f"  {row['feature']:<30} {row['std_beta']:>+8.3f} "
                  f"{row['p_value']:>8.4f}{star} {row['r_squared']:>8.3f} "
                  f"{int(row['n']):>5}")
    else:
        print("  ⚠ No features passed screening. Consider raising threshold.")

    return screen_df


# =============================================================================
# 5. FEATURE SELECTION (GUIDED)
# =============================================================================
def select_features(screen_df: pd.DataFrame, df: pd.DataFrame) -> list[str]:
    """
    Select features for the final multivariate model.

    Strategy:
    - Take top candidates from screening (p < threshold).
    - Cap at MAX_PREDICTORS.
    - Check pairwise correlations; if |r| > .70, keep the one with lower p.
    - Allow manual override via MANUAL_FEATURES (set to None for auto).

    Returns list of selected feature names.
    """
    print("\n" + "=" * 60)
    print("STEP 5: FEATURE SELECTION")
    print("=" * 60)

    # --- Manual override (set MANUAL_FEATURES at the top of the file) ---------
    if MANUAL_FEATURES is not None:
        print(f"  Using manually specified features: {MANUAL_FEATURES}")
        return MANUAL_FEATURES

    # --- Automatic selection --------------------------------------------------
    candidates = screen_df[
        screen_df["p_value"] < UNIVARIATE_THRESHOLD
    ].sort_values("p_value")

    if len(candidates) == 0:
        # Fallback: take top 3 by p-value regardless of threshold
        print("  ⚠ No features below threshold; taking top 3 by p-value as fallback.")
        candidates = screen_df.head(3)

    selected = candidates["feature"].tolist()

    # --- Collinearity check ---------------------------------------------------
    if len(selected) > 1:
        feat_data = df[selected].dropna()
        corr = feat_data.corr().abs()
        dropped = set()
        for i, f1 in enumerate(selected):
            for f2 in selected[i + 1:]:
                if f2 in dropped:
                    continue
                r = corr.loc[f1, f2]
                if r > 0.70:
                    # Drop the one with higher p-value
                    p1 = candidates.loc[candidates["feature"] == f1, "p_value"].values[0]
                    p2 = candidates.loc[candidates["feature"] == f2, "p_value"].values[0]
                    drop = f2 if p2 > p1 else f1
                    print(f"  ⚠ Collinear pair: {f1} × {f2} (r={r:.2f}). Dropping {drop}.")
                    dropped.add(drop)
        selected = [f for f in selected if f not in dropped]

    # --- Cap at MAX_PREDICTORS ------------------------------------------------
    if len(selected) > MAX_PREDICTORS:
        print(f"  Capping at {MAX_PREDICTORS} predictors (had {len(selected)} candidates).")
        selected = selected[:MAX_PREDICTORS]

    print(f"\n  Final selected features ({len(selected)}):")
    for f in selected:
        row = screen_df[screen_df["feature"] == f].iloc[0]
        print(f"    {f:<30} std_β={row['std_beta']:+.3f}  p={row['p_value']:.4f}")

    return selected


# =============================================================================
# 5b. HIERARCHICAL REGRESSION (M1 baseline → M2 full model)
# =============================================================================
def hierarchical_regression(df: pd.DataFrame, behavioral_features: list[str]) -> dict:
    """
    Hierarchical regression comparing content-similarity predictors (M1)
    against content plus behavioral features (M2).

    M1 baseline: semantic_similarity + sent_alignment (content level)
    M2 full    : M1 + behavioral_features (behavioral level)

    Delta R^2 = M2 R^2 - M1 R^2, the incremental contribution of behavioral
    features above content similarity.

    semantic_similarity and sent_alignment come from
    scientific_dyad_analysis_results.csv. After collapse_to_dyad, semantic
    columns receive the sem_ prefix, e.g. sem_semantic_similarity and
    sem_sent_alignment.
    """
    print("\n" + "=" * 60)
    print("STEP 5b: HIERARCHICAL REGRESSION (M1 → M2)")
    print("=" * 60)

    # Confirm baseline column names. Semantic columns receive a sem_ prefix in
    # load_and_merge.
    candidate_sem  = ["sem_semantic_similarity", "semantic_similarity"]
    candidate_sent = ["sem_sent_alignment", "sent_alignment",
                      "sem_sentiment_synchrony", "sentiment_synchrony"]

    sem_col  = next((c for c in candidate_sem  if c in df.columns), None)
    sent_col = next((c for c in candidate_sent if c in df.columns), None)

    if sem_col is None or sent_col is None:
        print(f"  ⚠ Baseline columns not found.")
        print(f"    Looking for semantic: {candidate_sem}")
        print(f"    Looking for sentiment: {candidate_sent}")
        print(f"    Available columns: {[c for c in df.columns if 'sem' in c.lower() or 'sent' in c.lower()]}")
        print(f"  Skipping hierarchical regression.")
        return {}

    baseline_features = [sem_col, sent_col]
    print(f"  M1 baseline : {baseline_features}")
    print(f"  M2 adds     : {behavioral_features}")

    # Prepare data using the complete-case intersection for both models.
    all_cols = ["composite_z"] + baseline_features + behavioral_features
    available = [c for c in all_cols if c in df.columns]
    missing   = [c for c in all_cols if c not in df.columns]
    if missing:
        print(f"  ⚠ Missing columns (will be skipped): {missing}")

    model_df = df[available].dropna()
    n = len(model_df)
    print(f"  Complete cases: N = {n}")

    y = model_df["composite_z"].values

    def fit_ols(feature_cols):
        X_raw = model_df[feature_cols].values
        X_sc  = StandardScaler().fit_transform(X_raw)
        X_sm  = sm.add_constant(X_sc)
        return sm.OLS(y, X_sm).fit(), X_sc

    def loo_r2(feature_cols):
        X_raw = model_df[feature_cols].values
        X_sc  = StandardScaler().fit_transform(X_raw)
        preds = np.zeros(n)
        for tr, te in LeaveOneOut().split(X_sc):
            reg = LinearRegression().fit(X_sc[tr], y[tr])
            preds[te] = reg.predict(X_sc[te])
        return r2_score(y, preds)

    # M1: content similarity only.
    m1_feats = [c for c in baseline_features if c in model_df.columns]
    m1_model, _ = fit_ols(m1_feats)
    m1_loo_r2   = loo_r2(m1_feats)

    # M2: content similarity plus behavioral features.
    m2_feats = [c for c in baseline_features + behavioral_features if c in model_df.columns]
    m2_model, _ = fit_ols(m2_feats)
    m2_loo_r2   = loo_r2(m2_feats)

    # Delta-R^2 F test.
    delta_r2     = m2_model.rsquared - m1_model.rsquared
    delta_loo_r2 = m2_loo_r2 - m1_loo_r2
    n_added      = len(m2_feats) - len(m1_feats)
    df_num       = n_added
    df_den       = n - len(m2_feats) - 1
    if df_den > 0 and (1 - m1_model.rsquared) > 0:
        f_change = (delta_r2 / df_num) / ((1 - m2_model.rsquared) / df_den)
        from scipy.stats import f as f_dist
        p_change = 1 - f_dist.cdf(f_change, df_num, df_den)
    else:
        f_change, p_change = np.nan, np.nan

    # Print results.
    print(f"\n  {'':30} {'R²':>8} {'Adj R²':>8} {'LOO R²':>8}")
    print("  " + "─" * 58)
    print(f"  {'M1 (content similarity)':30} {m1_model.rsquared:>8.3f} "
          f"{m1_model.rsquared_adj:>8.3f} {m1_loo_r2:>8.3f}")
    print(f"  {'M2 (+ behavioral features)':30} {m2_model.rsquared:>8.3f} "
          f"{m2_model.rsquared_adj:>8.3f} {m2_loo_r2:>8.3f}")
    print(f"  {'ΔR² (behavioral added)':30} {delta_r2:>+8.3f} {'':>8} {delta_loo_r2:>+8.3f}")
    print(f"\n  F-change({df_num}, {df_den}) = {f_change:.3f},  p = {p_change:.4f}"
          if not np.isnan(f_change) else "  F-change: insufficient df")

    print(f"\n  M1 coefficients ({', '.join(m1_feats)}):")
    print(f"  {'Feature':<35} {'β':>8} {'p':>8}")
    print("  " + "─" * 55)
    for i, feat in enumerate(m1_feats):
        b = m1_model.params[i + 1]
        p = m1_model.pvalues[i + 1]
        star = "***" if p < .001 else ("**" if p < .01 else ("*" if p < .05 else ("." if p < .1 else "")))
        print(f"  {feat:<35} {b:>+8.3f} {p:>8.4f} {star}")

    print(f"\n  M2 coefficients (all features):")
    print(f"  {'Feature':<35} {'β':>8} {'p':>8}")
    print("  " + "─" * 55)
    for i, feat in enumerate(m2_feats):
        b = m2_model.params[i + 1]
        p = m2_model.pvalues[i + 1]
        star = "***" if p < .001 else ("**" if p < .01 else ("*" if p < .05 else ("." if p < .1 else "")))
        label = "(content)" if feat in baseline_features else "(behavior)"
        print(f"  {feat:<35} {b:>+8.3f} {p:>8.4f} {star}  {label}")

    print(f"\n  Interpretation:")
    print(f"  Content similarity alone explains {m1_model.rsquared:.1%} of connection-quality variance (LOO-CV: {m1_loo_r2:.1%})")
    print(f"  Adding behavioral features explains {m2_model.rsquared:.1%} (LOO-CV: {m2_loo_r2:.1%})")
    print(f"  Behavioral features add Delta R^2 = {delta_r2:+.3f} (LOO-CV Delta R^2 = {delta_loo_r2:+.3f})")

    return {
        "m1_model":      m1_model,
        "m2_model":      m2_model,
        "m1_r2":         m1_model.rsquared,
        "m2_r2":         m2_model.rsquared,
        "m1_loo_r2":     m1_loo_r2,
        "m2_loo_r2":     m2_loo_r2,
        "delta_r2":      delta_r2,
        "delta_loo_r2":  delta_loo_r2,
        "f_change":      f_change,
        "p_change":      p_change,
        "m1_feats":      m1_feats,
        "m2_feats":      m2_feats,
        "n":             n,
    }


# =============================================================================
# 6. LINEAR REGRESSION + EVALUATION
# =============================================================================
def fit_linear_model(
    df: pd.DataFrame,
    features: list[str],
) -> dict:
    """
    Fit OLS linear regression: features → composite_z (continuous).
    Evaluate with LOO-CV R²/RMSE, bootstrap CIs on coefficients.

    Returns dict with model results, CV metrics, and bootstrap CIs.
    """
    print("\n" + "=" * 60)
    print("STEP 6: LINEAR REGRESSION + EVALUATION")
    print("=" * 60)

    # --- Prepare data ---------------------------------------------------------
    model_cols = features + ["composite_z"]
    model_df = df[model_cols].dropna()
    n = len(model_df)
    print(f"  Complete cases for model: N = {n}")
    print(f"  Predictors: {features}")
    print(f"  Ratio: {n / len(features):.1f} observations per predictor")

    if n / len(features) < 5:
        print("  ⚠ WARNING: < 5 obs/predictor. Results may be unstable.")

    X = model_df[features].values
    y = model_df["composite_z"].values

    # Standardize features for comparable coefficients
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Full-sample OLS model (statsmodels for detailed output) --------------
    X_sm = sm.add_constant(X_scaled)
    ols_model = sm.OLS(y, X_sm).fit()

    print(f"\n  Full-sample OLS regression:")
    print(f"  R²:          {ols_model.rsquared:.3f}")
    print(f"  Adj R²:      {ols_model.rsquared_adj:.3f}")
    print(f"  F({ols_model.df_model:.0f}, {ols_model.df_resid:.0f}) = "
          f"{ols_model.fvalue:.2f}, p = {ols_model.f_pvalue:.4f}")
    print(f"  AIC: {ols_model.aic:.2f}   BIC: {ols_model.bic:.2f}")
    print(f"\n  {'Feature':<30} {'β':>8} {'SE':>8} {'t':>8} {'p':>8} "
          f"{'[95% CI]':>22}")
    print("  " + "-" * 92)
    for i, feat in enumerate(features):
        b   = ols_model.params[i + 1]
        se  = ols_model.bse[i + 1]
        t   = ols_model.tvalues[i + 1]
        p   = ols_model.pvalues[i + 1]
        lo, hi = ols_model.conf_int()[i + 1]
        star = "***" if p < .001 else ("**" if p < .01 else ("*" if p < .05 else ""))
        print(f"  {feat:<30} {b:>+8.3f} {se:>8.3f} {t:>8.2f} "
              f"{p:>8.4f}{star}  [{lo:>+8.3f}, {hi:>+8.3f}]")

    # --- Leave-One-Out Cross-Validation ---------------------------------------
    print(f"\n  Leave-One-Out Cross-Validation:")
    loo = LeaveOneOut()
    y_pred_loo = np.zeros(n)

    for train_idx, test_idx in loo.split(X_scaled):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train = y[train_idx]
        reg = LinearRegression()
        reg.fit(X_train, y_train)
        y_pred_loo[test_idx] = reg.predict(X_test)

    loo_r2   = r2_score(y, y_pred_loo)
    loo_rmse = np.sqrt(mean_squared_error(y, y_pred_loo))
    print(f"  LOO-CV R²:   {loo_r2:.3f}")
    print(f"  LOO-CV RMSE: {loo_rmse:.3f}")

    # --- Bootstrap CIs on OLS coefficients ------------------------------------
    print(f"\n  Bootstrap 95% CIs ({BOOTSTRAP_N} resamples):")
    boot_coefs = np.zeros((BOOTSTRAP_N, len(features)))
    rng = np.random.RandomState(SEED)

    for b in range(BOOTSTRAP_N):
        idx = rng.choice(n, size=n, replace=True)
        X_boot, y_boot = X_scaled[idx], y[idx]
        reg = LinearRegression()
        reg.fit(X_boot, y_boot)
        boot_coefs[b, :] = reg.coef_

    ci_lower    = np.percentile(boot_coefs, 2.5, axis=0)
    ci_upper    = np.percentile(boot_coefs, 97.5, axis=0)
    boot_median = np.median(boot_coefs, axis=0)

    print(f"\n  {'Feature':<30} {'Median':>8} {'95% CI':>22} {'Crosses 0?':>12}")
    print("  " + "-" * 78)
    for i, feat in enumerate(features):
        crosses_zero = "YES" if ci_lower[i] < 0 < ci_upper[i] else "no"
        print(f"  {feat:<30} {boot_median[i]:>+8.3f} "
              f"[{ci_lower[i]:>+8.3f}, {ci_upper[i]:>+8.3f}] "
              f"{crosses_zero:>12}")

    return {
        "model":       ols_model,
        "features":    features,
        "scaler":      scaler,
        "loo_r2":      loo_r2,
        "loo_rmse":    loo_rmse,
        "y_true":      y,
        "y_pred_loo":  y_pred_loo,
        "boot_coefs":  boot_coefs,
        "ci_lower":    ci_lower,
        "ci_upper":    ci_upper,
        "boot_median": boot_median,
        "r2":          ols_model.rsquared,
        "adj_r2":      ols_model.rsquared_adj,
        "n":           n,
    }


# =============================================================================
# 6b. CONFUSION MATRIX (median-split visualization, optional diagnostic)
# =============================================================================
def plot_confusion_matrix(y_true_continuous, y_pred_continuous, output_dir):
    """
    Split continuous predictions at the observed median, then plot a
    confusion matrix as a quick diagnostic of directional accuracy.
    """
    from sklearn.metrics import confusion_matrix as _cm
    median = np.median(y_true_continuous)
    y_true_bin = (y_true_continuous >= median).astype(int)
    y_pred_bin = (y_pred_continuous >= median).astype(int)

    cm = _cm(y_true_bin, y_pred_bin)
    accuracy = np.mean(y_true_bin == y_pred_bin)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Predicted Low", "Predicted High"],
        yticklabels=["Actual Low",    "Actual High"],
        ax=ax,
    )
    ax.set_title(
        f"LOO-CV Confusion Matrix (median split)\nAccuracy = {accuracy:.1%}",
        fontsize=12,
    )
    plt.tight_layout()
    out = Path(output_dir)
    fig.savefig(out / "fig4_confusion_matrix.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(out / "fig4_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    n_total = len(y_true_bin)
    print(f"  Accuracy: {accuracy:.1%} ({int(accuracy * n_total)}/{n_total} correct)")
    print("  ✓ Saved fig4_confusion_matrix.pdf/png")
    return cm, accuracy


# =============================================================================
# 7. POSTER-READY FIGURES
# =============================================================================
def make_figures(
    df: pd.DataFrame,
    pca_info: dict,
    model_results: dict,
    screen_df: pd.DataFrame,
    output_dir,
):
    """
    Generate three poster-ready figures:
    1. PCA loading plot (outcome structure)
    2. Logistic regression coefficient plot with bootstrap CIs
    3. ROC curve (LOO-CV)
    """
    output_dir = Path(output_dir)

    print("\n" + "=" * 60)
    print("STEP 7: GENERATING POSTER FIGURES")
    print("=" * 60)

    features = model_results["features"]

    # ---- Figure 1: PCA Loadings Bar Plot -------------------------------------
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    loadings = pca_info["loadings"]
    labels = [k.replace("dyad_", "").replace("_mean", "").replace("_", " ").title()
              for k in loadings.keys()]
    values = list(loadings.values())
    colors = [COLOR_PALETTE[0] if v > 0 else COLOR_PALETTE[3] for v in values]

    bars = ax1.barh(labels, values, color=colors, edgecolor="white", height=0.6)
    ax1.axvline(0, color="grey", linewidth=0.8, linestyle="--")
    ax1.set_xlabel("PC1 Loading", fontsize=12)
    ax1.set_title("Outcome Structure (PC1)\n"
                  f"Variance Explained: {pca_info['variance_explained'][0]:.1%}",
                  fontsize=13, fontweight="bold")
    ax1.tick_params(axis="y", labelsize=11)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        x_pos = val + 0.02 if val > 0 else val - 0.02
        ha = "left" if val > 0 else "right"
        ax1.text(x_pos, bar.get_y() + bar.get_height() / 2,
                 f"{val:.2f}", ha=ha, va="center", fontsize=10)

    plt.tight_layout()
    fig1.savefig(output_dir / "fig1_pca_loadings.pdf", dpi=POSTER_FIG_DPI, bbox_inches="tight")
    fig1.savefig(output_dir / "fig1_pca_loadings.png", dpi=POSTER_FIG_DPI, bbox_inches="tight")
    print("  ✓ Saved fig1_pca_loadings.pdf/png")

    # ---- Figure 2: Coefficient Plot with Bootstrap CIs -----------------------
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    feat_labels = [f.replace("_", " ").replace("asym", "(asym)")
                   .replace("mean", "(mean)").title() for f in features]
    coefs = model_results["boot_median"]
    ci_lo = model_results["ci_lower"]
    ci_hi = model_results["ci_upper"]
    y_pos = np.arange(len(features))

    # Color by whether CI excludes zero
    ci_colors = [COLOR_PALETTE[2] if (lo > 0 or hi < 0) else COLOR_PALETTE[7]
                 for lo, hi in zip(ci_lo, ci_hi)]

    ax2.barh(y_pos, coefs, xerr=[coefs - ci_lo, ci_hi - coefs],
             color=ci_colors, edgecolor="white", height=0.5, capsize=4,
             error_kw={"linewidth": 1.5})
    ax2.axvline(0, color="grey", linewidth=0.8, linestyle="--")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(feat_labels, fontsize=11)
    ax2.set_xlabel("Standardized OLS Coefficient (Bootstrap Median)", fontsize=11)
    ax2.set_title(f"Linear Regression: Predicting Connection Quality\n"
                  f"LOO-CV R² = {model_results['loo_r2']:.3f}  |  "
                  f"RMSE = {model_results['loo_rmse']:.3f}  |  "
                  f"N = {model_results['n']}",
                  fontsize=12, fontweight="bold")

    plt.tight_layout()
    fig2.savefig(output_dir / "fig2_coefficients.pdf", dpi=POSTER_FIG_DPI, bbox_inches="tight")
    fig2.savefig(output_dir / "fig2_coefficients.png", dpi=POSTER_FIG_DPI, bbox_inches="tight")
    print("  ✓ Saved fig2_coefficients.pdf/png")

    # ---- Figure 3: Actual vs Predicted (LOO-CV) ------------------------------
    fig3, ax3 = plt.subplots(figsize=(5, 5))
    y_true = model_results["y_true"]
    y_pred = model_results["y_pred_loo"]

    ax3.scatter(y_true, y_pred, alpha=0.75, edgecolors="k",
                linewidths=0.5, color=COLOR_PALETTE[0], s=60)
    lims = [min(y_true.min(), y_pred.min()) - 0.3,
            max(y_true.max(), y_pred.max()) + 0.3]
    ax3.plot(lims, lims, "k--", linewidth=1, alpha=0.5, label="Perfect prediction")
    ax3.set_xlim(lims); ax3.set_ylim(lims)
    ax3.set_xlabel("Actual composite_z", fontsize=12)
    ax3.set_ylabel("LOO-CV Predicted composite_z", fontsize=12)
    ax3.set_title(f"Actual vs Predicted (LOO-CV)\n"
                  f"R² = {model_results['loo_r2']:.3f}  |  "
                  f"RMSE = {model_results['loo_rmse']:.3f}",
                  fontsize=13, fontweight="bold")
    ax3.legend(fontsize=10, loc="upper left")
    ax3.set_aspect("equal")

    plt.tight_layout()
    fig3.savefig(output_dir / "fig3_actual_vs_predicted.pdf", dpi=POSTER_FIG_DPI, bbox_inches="tight")
    fig3.savefig(output_dir / "fig3_actual_vs_predicted.png", dpi=POSTER_FIG_DPI, bbox_inches="tight")
    print("  ✓ Saved fig3_actual_vs_predicted.pdf/png")

    plt.close("all")


# =============================================================================
# 8. EXPORT RESULTS
# =============================================================================
def export_results(
    df: pd.DataFrame,
    screen_df: pd.DataFrame,
    model_results: dict,
    pca_info: dict,
    output_dir,
):
    """
    Save all analysis outputs as CSVs + a JSON summary for reproducibility.
    """
    output_dir = Path(output_dir)

    print("\n" + "=" * 60)
    print("STEP 8: EXPORTING RESULTS")
    print("=" * 60)

    # --- Screening results ----------------------------------------------------
    screen_df.to_csv(output_dir / "univariate_screening.csv", index=False)
    print("  ✓ univariate_screening.csv")

    # --- Full dataset with composite ------------------------------------------
    df.to_csv(output_dir / "dataset_with_composite.csv", index=False)
    print("  ✓ dataset_with_composite.csv")

    # --- Model summary --------------------------------------------------------
    model = model_results["model"]
    features = model_results["features"]
    ci = model.conf_int()
    model_summary = []
    for i, feat in enumerate(features):
        model_summary.append({
            "feature":       feat,
            "beta":          model.params[i + 1],
            "se":            model.bse[i + 1],
            "t":             model.tvalues[i + 1],
            "p_value":       model.pvalues[i + 1],
            "ci_lower_95":   ci[i + 1, 0],
            "ci_upper_95":   ci[i + 1, 1],
            "boot_ci_lower": model_results["ci_lower"][i],
            "boot_ci_upper": model_results["ci_upper"][i],
            "boot_median":   model_results["boot_median"][i],
        })
    pd.DataFrame(model_summary).to_csv(output_dir / "linear_model_summary.csv", index=False)
    print("  ✓ linear_model_summary.csv")

    # --- JSON metadata (for reproducibility) ----------------------------------
    metadata = {
        "analysis_date": datetime.now().isoformat(),
        "script": "multivariate_analysis.py",
        "seed": SEED,
        "n_dyads": int(model_results["n"]),
        "features_selected": features,
        "pca": {
            "n_outcomes": len(pca_info["loadings"]),
            "variance_explained_pc1": pca_info["variance_explained"][0],
            "loadings": pca_info["loadings"],
            "solo_reversed": pca_info["solo_reversed"],
        },
        "model_performance": {
            "r2":        float(model_results["r2"]),
            "adj_r2":    float(model_results["adj_r2"]),
            "loo_cv_r2": float(model_results["loo_r2"]),
            "loo_cv_rmse": float(model_results["loo_rmse"]),
        },
        "bootstrap_n": BOOTSTRAP_N,
        "univariate_threshold": UNIVARIATE_THRESHOLD,
        "max_predictors": MAX_PREDICTORS,
    }
    with open(output_dir / "analysis_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("  ✓ analysis_metadata.json")

    # --- Statsmodels full summary text ----------------------------------------
    with open(output_dir / "linear_model_full_summary.txt", "w") as f:
        f.write(model.summary().as_text())
    print("  ✓ linear_model_full_summary.txt")


def make_descriptive_table(df, features, output_dir):
    """
    Generate Table 1 for model-entering predictors and outcomes.

    Reports M, SD, Min, Max, N, and each predictor's Pearson correlation with
    composite_z. Outcome rows leave the correlation column blank.
    """
    from scipy.stats import pearsonr
    output_dir = Path(output_dir)

    PREDICTOR_COLS = [f for f in features if f in df.columns]
    OUTCOME_COLS_TABLE = [
        "dyad_partner_eval_mean",
        "dyad_shared_reality_mean",
        "dyad_enjoyment_mean",
        "composite_z",
    ]
    labels = {
        "verbal_agreement_backchannel_n": "Verbal Agreement (Backchannel)",
        "verbal_agreement_affirm_n":      "Verbal Agreement (Affirm)",
        "verbal_agreement_elaborated_n":  "Verbal Agreement (Elaborated)",
        "questions_asym":                 "Question Asymmetry",
        "hedging_asym":                   "Hedging Asymmetry",
        "dyad_partner_eval_mean":         "Partner Evaluation",
        "dyad_shared_reality_mean":       "Shared Reality",
        "dyad_enjoyment_mean":            "Enjoyment",
        "composite_z":                    "Connection Quality (composite)",
    }

    def sig_star(p):
        if p < .001: return "***"
        if p < .01:  return "**"
        if p < .05:  return "*"
        if p < .10:  return "†"
        return ""

    rows = []
    for col in PREDICTOR_COLS + OUTCOME_COLS_TABLE:
        if col not in df.columns:
            continue
        s = df[col].dropna()

        # Pearson r with composite_z (predictors only)
        r_str = ""
        if col in PREDICTOR_COLS and "composite_z" in df.columns:
            pair = df[[col, "composite_z"]].dropna()
            if len(pair) > 4:
                r, p = pearsonr(pair[col], pair["composite_z"])
                r_str = f"{r:+.2f}{sig_star(p)}"

        rows.append({
            "Variable": labels.get(col, col),
            "M":   f"{s.mean():.2f}",
            "SD":  f"{s.std():.2f}",
            "Min": f"{s.min():.2f}",
            "Max": f"{s.max():.2f}",
            "N":   len(s),
            "r (composite_z)": r_str,
        })

    desc_df = pd.DataFrame(rows)
    desc_df.to_csv(output_dir / "descriptive_stats.csv", index=False)

    # Pretty-print to console
    print("\n  Table 1 — Descriptive Statistics")
    print(f"  {'Variable':<32} {'M':>6} {'SD':>6} {'Min':>6} {'Max':>6} "
          f"{'N':>4}  {'r (composite_z)':>16}")
    print("  " + "─" * 82)
    for _, row in desc_df.iterrows():
        print(f"  {row['Variable']:<32} {row['M']:>6} {row['SD']:>6} "
              f"{row['Min']:>6} {row['Max']:>6} {row['N']:>4}  "
              f"{row['r (composite_z)']:>16}")
    print("  Note: † p<.10  * p<.05  ** p<.01  *** p<.001")
    print("  ✓ Saved descriptive_stats.csv")
    return desc_df


def make_mean_asym_figure(screen_df, output_dir):
    output_dir = Path(output_dir)

    THEME = {
        'green':  '#6AAB9C',
        'salmon': '#FA9284',
        'blue':   '#5874DC',
        'navy':   '#384E78',
        'text':   '#000000',
        'grid':   '#EEEEEE',
    }

    sig = screen_df[screen_df['p_value'] < 0.1].copy()

    def label_type(feat):
        if feat.endswith('_asym'):  return 'Asymmetry'
        if feat.endswith('_mean'):  return 'Mean Level'
        return 'Count/Rate'

    def clean_label(feat):
        mapping = {
            'verbal_agreement_backchannel_n': 'Verbal Agreement (Backchannel)',
            'verbal_agreement_affirm_n':      'Verbal Agreement (Affirm)',
            'verbal_agreement_elaborated_n':  'Verbal Agreement (Elaborated)',
            'questions_asym':      'Question Asymmetry',
            'bc_rate_mean':        'Backchannel Rate',
            'total_turns':         'Total Turns',
            'participation_gini':  'Participation Gini',
            'we_rate_asym':        '"We" Usage Asymmetry',
            'sentiment_synchrony': 'Sentiment Synchrony',
            'ttr_mean':            'Vocabulary Diversity',
            'hedging_asym':        'Hedging Asymmetry',
        }
        return mapping.get(feat, feat.replace('_', ' ').title())

    sig['type']  = sig['feature'].apply(label_type)
    sig['label'] = sig['feature'].apply(clean_label)
    sig['sig']   = sig['p_value'].apply(
        lambda p: '*' if p < 0.05 else ('†' if p < 0.1 else '')
    )
    sig = sig.sort_values('std_beta')

    # Color mapping.
    colors = {
        'Asymmetry':  THEME['blue'],
        'Mean Level': THEME['green'],
        'Count/Rate': THEME['salmon'],
    }

    fig, ax = plt.subplots(figsize=(8, 5.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    for i, (_, row) in enumerate(sig.iterrows()):
        ax.barh(
            row['label'],
            row['std_beta'],
            color=colors[row['type']],
            alpha=0.88,
            edgecolor='white',
            linewidth=0.5,
            height=0.62,
        )
        # Significance marker.
        if row['sig']:
            x  = row['std_beta'] + 0.015 if row['std_beta'] > 0 else row['std_beta'] - 0.015
            ha = 'left' if row['std_beta'] > 0 else 'right'
            ax.text(x, i, row['sig'], va='center', ha=ha,
                    fontsize=12, color=THEME['navy'],
                    fontfamily='Helvetica', fontweight='bold')

    # Zero line.
    ax.axvline(0, color=THEME['grid'], linewidth=1.2,
               linestyle='-', alpha=1.0, zorder=0)

    # Axes.
    ax.set_xlabel('Standardized β', fontsize=12,
                  color=THEME['text'], fontfamily='Helvetica')
    ax.set_title('Conversational Predictors of Connection Quality',
                 fontsize=13, fontweight='bold',
                 color=THEME['navy'], fontfamily='Helvetica', pad=14)
    ax.tick_params(axis='y', labelrotation=45)

    # Tick styling.
    for label in ax.get_yticklabels() + ax.get_xticklabels():
        label.set_fontfamily('Helvetica')
        label.set_color(THEME['text'])
        label.set_fontsize(10.5)

    # Grid.
    ax.xaxis.grid(True, color=THEME['grid'], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_color(THEME['grid'])

    # Legend.
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=THEME['blue'],   label='Asymmetry',  edgecolor='white'),
        Patch(facecolor=THEME['green'],  label='Mean Level', edgecolor='white'),
        Patch(facecolor=THEME['salmon'], label='Count/Rate', edgecolor='white'),
    ]
    ax.legend(
        handles=legend_elements,
        fontsize=10,
        loc='lower right',
        framealpha=0.95,
        edgecolor=THEME['grid'],
        prop={'family': 'Helvetica'}
    )

    # Footnote.
    ax.text(0.01, -0.10,
            '* p < .05    † p < .10',
            transform=ax.transAxes,
            fontsize=9, color='#888888',
            fontfamily='Helvetica')

    plt.tight_layout()
    fig.savefig(output_dir / "fig5_mean_vs_asym.pdf",
                dpi=1200, bbox_inches="tight", facecolor='white')
    fig.savefig(output_dir / "fig5_mean_vs_asym.png",
                dpi=1200, bbox_inches="tight", facecolor='white')
    plt.close(fig)
    print("  ✓ Saved fig5_mean_vs_asym.pdf/png")
    
    
# =============================================================================
# 6c. EXTREME CASES CHECK
# =============================================================================
def check_extreme_cases(df, features, output_dir, n_extreme=3):
    """
    Inspect dyads with the highest and lowest composite_z values.
    """
    output_dir = Path(output_dir)
    
    key_cols = ['pair_id'] + features + [
        'composite_z',
        'dyad_partner_eval_mean',
        'dyad_shared_reality_mean',
        'dyad_enjoyment_mean',
    ]
    available = [c for c in key_cols if c in df.columns]
    extreme_df = df[available].dropna(subset=['composite_z'])
    extreme_df = extreme_df.sort_values('composite_z').reset_index(drop=True)
    
    print("\n" + "=" * 60)
    print("STEP 6c: EXTREME CASES CHECK")
    print("=" * 60)
    
    # Overall sample statistics.
    print(f"\n  composite_z distribution:")
    print(f"  Mean = {extreme_df['composite_z'].mean():.3f}")
    print(f"  SD   = {extreme_df['composite_z'].std():.3f}")
    print(f"  Min  = {extreme_df['composite_z'].min():.3f}")
    print(f"  Max  = {extreme_df['composite_z'].max():.3f}")
    
    # Bottom cases.
    print(f"\n  -- Lowest-connection {n_extreme} dyads --")
    bottom = extreme_df.head(n_extreme)
    print(bottom.to_string(index=False))
    
    # Top cases.
    print(f"\n  -- Highest-connection {n_extreme} dyads --")
    top = extreme_df.tail(n_extreme)
    print(top.to_string(index=False))
    
    # Outlier check.
    z_scores = np.abs(stats.zscore(extreme_df['composite_z']))
    outliers = extreme_df[z_scores > 2.5]
    if len(outliers) > 0:
        print(f"\n  Potential outliers (|z| > 2.5):")
        for _, row in outliers.iterrows():
            print(f"    pair_id={int(row['pair_id'])}  "
                  f"composite_z={row['composite_z']:.3f}  "
                  f"z={z_scores[row.name]:.2f}")
    else:
        print(f"\n  No clear outliers (|z| > 2.5)")
    
    # Compare feature means for high vs low groups.
    print(f"\n  -- High/low group feature comparison --")
    median = extreme_df['composite_z'].median()
    high = extreme_df[extreme_df['composite_z'] >= median]
    low  = extreme_df[extreme_df['composite_z'] <  median]
    
    print(f"\n  {'Feature':<25} {'Low (n=' + str(len(low)) + ')':>12} "
          f"{'High (n=' + str(len(high)) + ')':>13} {'Diff':>8}")
    print("  " + "-" * 62)
    for feat in features:
        if feat not in extreme_df.columns:
            continue
        lo_mean = low[feat].mean()
        hi_mean = high[feat].mean()
        diff    = hi_mean - lo_mean
        print(f"  {feat:<25} {lo_mean:>12.3f} {hi_mean:>13.3f} {diff:>+8.3f}")
    
    # Save the full sorted table.
    extreme_df.to_csv(output_dir / "extreme_cases.csv", index=False)
    print(f"\n  ✓ Saved extreme_cases.csv")
    
    return extreme_df

# =============================================================================
# 6d. SENSITIVITY ANALYSIS: PEARSON VS SPEARMAN
# =============================================================================
def sensitivity_analysis(df, features, output_dir):
    """
    Compare Pearson and Spearman correlations as a robustness check.
    """
    from scipy.stats import spearmanr, pearsonr
    output_dir = Path(output_dir)

    print("\n" + "=" * 60)
    print("STEP 6d: SENSITIVITY ANALYSIS (Pearson vs Spearman)")
    print("=" * 60)

    results = []

    print(f"\n  {'Feature':<25} {'Pearson r':>10} {'p':>8} {'Spearman r':>12} {'p':>8} {'Consistent?':>12}")
    print("  " + "-" * 80)

    for feat in features:
        subset = df[[feat, 'composite_z']].dropna()
        
        pearson_r,  pearson_p  = pearsonr(subset[feat], subset['composite_z'])
        spearman_r, spearman_p = spearmanr(subset[feat], subset['composite_z'])
        
        # Consistent if direction matches and significance status agrees.
        same_direction = (pearson_r > 0) == (spearman_r > 0)
        both_sig       = (pearson_p < 0.1) and (spearman_p < 0.1)
        neither_sig    = (pearson_p >= 0.1) and (spearman_p >= 0.1)
        consistent     = same_direction and (both_sig or neither_sig)
        
        flag = "✓" if consistent else "⚠"
        
        print(f"  {feat:<25} {pearson_r:>+10.3f} {pearson_p:>8.3f} "
              f"{spearman_r:>+12.3f} {spearman_p:>8.3f} {flag:>12}")
        
        results.append({
            'feature':    feat,
            'pearson_r':  pearson_r,
            'pearson_p':  pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'consistent': consistent,
        })

    results_df = pd.DataFrame(results)
    
    # Summary.
    n_consistent = results_df['consistent'].sum()
    print(f"\n  {n_consistent}/{len(features)} features consistent across methods")
    
    if n_consistent == len(features):
        print("  ✓ Results are robust to non-parametric Spearman correlations")
    else:
        inconsistent = results_df[~results_df['consistent']]['feature'].tolist()
        print(f"  ⚠ Inconsistent features: {inconsistent}")
        print("    → Check these for outlier influence")

    # Save.
    results_df.to_csv(output_dir / "sensitivity_analysis.csv", index=False)
    print(f"\n  ✓ Saved sensitivity_analysis.csv")

    return results_df
# =============================================================================
# 6e. NESTED LOO-CV (unbiased generalization estimate)
# =============================================================================
def nested_loo_cv(df: pd.DataFrame) -> dict:
    """
    Nested Leave-One-Out CV: feature selection happens *inside* each fold so
    the held-out test point never influences which features are chosen.

    Each fold:
      1. train = N-1 dyads
      2. univariate screen on train only  (feature → composite_z, OLS)
      3. select features (p < UNIVARIATE_THRESHOLD, collinearity |r|>.70, cap MAX_PREDICTORS)
      4. standardize on train, fit LinearRegression on train
      5. predict the single test dyad

    Returns
    -------
    dict with nested_loo_r2, nested_loo_rmse, y_true, y_pred,
    feature_selection_freq (how often each feature was picked across folds).
    """
    print("\n" + "=" * 60)
    print("STEP 6e: NESTED LOO-CV (unbiased generalization)")
    print("=" * 60)

    # If features are manually fixed, nesting adds nothing — just say so.
    if MANUAL_FEATURES is not None:
        print(f"  MANUAL_FEATURES is set → feature selection is theory-driven,")
        print(f"  not data-driven. Nested CV = standard LOO-CV on {MANUAL_FEATURES}.")

    # Identify all numeric candidate features (same exclusion as univariate_screen)
    exclude = (
        set(OUTCOME_COLS)
        | {"pair_id", "composite_z", "connection_group", "Order", "Dyad ID"}
    )
    all_feature_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude
    ]

    # Work on rows that have composite_z and at least one feature
    model_df = df[all_feature_cols + ["composite_z"] +
                  (["Order"] if "Order" in df.columns else [])].dropna(
                      subset=["composite_z"]
                  ).reset_index(drop=True)

    n = len(model_df)
    y_arr = model_df["composite_z"].values
    has_order = "Order" in model_df.columns

    print(f"  N = {n} dyads,  {len(all_feature_cols)} candidate features")
    print(f"  Threshold p < {UNIVARIATE_THRESHOLD},  max predictors = {MAX_PREDICTORS}")

    y_pred = np.full(n, np.nan)
    selected_per_fold: list[list[str]] = []

    for test_idx in range(n):
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_idx] = False
        train_df = model_df[train_mask]
        test_row  = model_df.iloc[[test_idx]]

        y_train = train_df["composite_z"].values

        # ── Step 1: univariate screen on train ──────────────────────────────
        if MANUAL_FEATURES is not None:
            fold_features = [f for f in MANUAL_FEATURES if f in train_df.columns]
        else:
            screen_results = []
            for feat in all_feature_cols:
                subset = train_df[[feat, "composite_z"] +
                                  (["Order"] if has_order else [])].dropna()
                if len(subset) < 8:
                    continue
                y_s = subset["composite_z"].values
                X_s = subset[[feat]].values
                if has_order:
                    X_s = np.column_stack([X_s, subset[["Order"]].values])
                X_s = sm.add_constant(X_s)
                try:
                    res = sm.OLS(y_s, X_s).fit()
                    screen_results.append({
                        "feature": feat,
                        "p_value": res.pvalues[1],  # index 1 = the feature (after const)
                    })
                except Exception:
                    pass

            screen_df_fold = pd.DataFrame(screen_results).sort_values("p_value")
            candidates = screen_df_fold[
                screen_df_fold["p_value"] < UNIVARIATE_THRESHOLD
            ]["feature"].tolist()

            if len(candidates) == 0:
                candidates = screen_df_fold.head(3)["feature"].tolist()

            # ── Step 2: collinearity check on train ─────────────────────────
            fold_features = candidates[:]
            if len(fold_features) > 1:
                corr = train_df[fold_features].corr().abs()
                dropped: set[str] = set()
                for i, f1 in enumerate(fold_features):
                    for f2 in fold_features[i + 1:]:
                        if f2 in dropped or f1 in dropped:
                            continue
                        if corr.loc[f1, f2] > 0.70:
                            p1 = screen_df_fold.loc[
                                screen_df_fold["feature"] == f1, "p_value"
                            ].values
                            p2 = screen_df_fold.loc[
                                screen_df_fold["feature"] == f2, "p_value"
                            ].values
                            if len(p1) and len(p2):
                                dropped.add(f2 if p2[0] > p1[0] else f1)
                fold_features = [f for f in fold_features if f not in dropped]

            # ── Step 3: cap ─────────────────────────────────────────────────
            fold_features = fold_features[:MAX_PREDICTORS]

        selected_per_fold.append(fold_features)

        if len(fold_features) == 0:
            y_pred[test_idx] = y_train.mean()
            continue

        # ── Step 4: standardize on train, fit ───────────────────────────────
        # Drop any train rows that have NaN in the selected features
        train_clean = train_df[fold_features + ["composite_z"]].dropna()
        if len(train_clean) < len(fold_features) + 2:
            # Not enough clean train rows to fit
            y_pred[test_idx] = np.nan
            continue

        X_train = train_clean[fold_features].values
        y_train_clean = train_clean["composite_z"].values

        # If test row has NaN in any selected feature, skip prediction
        test_vals = test_row[fold_features]
        if test_vals.isnull().values.any():
            y_pred[test_idx] = np.nan
            continue
        X_test = test_vals.values

        scaler_fold = StandardScaler().fit(X_train)
        X_train_sc  = scaler_fold.transform(X_train)
        X_test_sc   = scaler_fold.transform(X_test)

        reg = LinearRegression().fit(X_train_sc, y_train_clean)
        y_pred[test_idx] = reg.predict(X_test_sc)[0]

    # ── Metrics ─────────────────────────────────────────────────────────────
    valid = ~np.isnan(y_pred)
    nested_r2   = r2_score(y_arr[valid], y_pred[valid])
    nested_rmse = np.sqrt(mean_squared_error(y_arr[valid], y_pred[valid]))

    # Feature selection stability
    from collections import Counter
    freq = Counter(f for fold in selected_per_fold for f in fold)
    freq_df = pd.DataFrame(
        [{"feature": f, "times_selected": c, "pct": c / n}
         for f, c in freq.most_common()]
    )

    print(f"\n  Nested LOO-CV R²   = {nested_r2:.3f}")
    print(f"  Nested LOO-CV RMSE = {nested_rmse:.3f}")
    print(f"\n  Feature selection stability ({n} folds):")
    print(f"  {'Feature':<30} {'# selected':>12} {'% folds':>10}")
    print("  " + "-" * 56)
    for _, row in freq_df.iterrows():
        print(f"  {row['feature']:<30} {int(row['times_selected']):>12} "
              f"{row['pct']:>9.0%}")

    return {
        "nested_loo_r2":   nested_r2,
        "nested_loo_rmse": nested_rmse,
        "y_true":          y_arr,
        "y_pred":          y_pred,
        "freq_df":         freq_df,
        "selected_per_fold": selected_per_fold,
    }


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 60)
    print("MULTIVARIATE ANALYSIS PIPELINE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Seed: {SEED}")
    print("=" * 60)

    # --- Ensure paths are Path objects (safe if user edits to strings) ---------
    data_dir = Path(DATA_DIR)
    output_dir = Path(OUTPUT_DIR)

    # --- Setup output directory -----------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.resolve()}")

    # --- Pipeline -------------------------------------------------------------
    df = load_and_merge(data_dir)
    df = restructure_features(df)        # ← Step 1.5
    df = run_qc(df)
    df = inspect_and_drop_features(df)   # ← manual inspection / drop step
    df, pca_info = build_composite(df)
    screen_df = univariate_screen(df)
    features = select_features(screen_df, df)

    if len(features) == 0:
        sys.exit("ERROR: No features selected. Cannot fit model.")

    model_results = fit_linear_model(df, features)
    nested_results = nested_loo_cv(df)             # ← unbiased generalization
    hierarchical_results = hierarchical_regression(df, features)
    make_figures(df, pca_info, model_results, screen_df, output_dir)
    plot_confusion_matrix(model_results["y_true"], model_results["y_pred_loo"], output_dir)
    make_descriptive_table(df, features, output_dir)
    make_mean_asym_figure(screen_df, output_dir)
    check_extreme_cases(df, features, output_dir, n_extreme=3)
    sensitivity_analysis(df, features, output_dir)


    export_results(df, screen_df, model_results, pca_info, output_dir)

    # --- Final summary --------------------------------------------------------
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Model: {len(features)} features → composite_z (continuous)")
    print(f"  Features: {features}")
    print(f"  R²:                    {model_results['r2']:.3f}")
    print(f"  Adj R²:                {model_results['adj_r2']:.3f}")
    print(f"  LOO-CV R² (standard):  {model_results['loo_r2']:.3f}  ← biased (feature selection on full data)")
    print(f"  LOO-CV R² (nested):    {nested_results['nested_loo_r2']:.3f}  ← unbiased estimate")
    print(f"  LOO-CV RMSE (nested):  {nested_results['nested_loo_rmse']:.3f}")
    print(f"\n  Outputs saved to: {output_dir.resolve()}")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
