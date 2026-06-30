#!/usr/bin/env python3
"""
05_dissociation_test.py
=======================
The core question of the project (stated ASSOCIATIONALLY — this is a
cross-sectional correlational design, so it cannot establish causation,
direction, or "mechanism"; it tests covariation and incremental variance):

    Is VOCAL ALIGNMENT (sub-second prosodic coordination) a DISTINCT dimension
    of coordination (discriminant validity), and is it ASSOCIATED with social
    connection beyond turn-level conversational dynamics (incremental validity)?

Wording note: avoid "predicts ... through a mechanism separable from turn-level
dynamics" — that phrasing implies a causal pathway the design cannot support.
Use "is associated with" / "adds incremental variance over".

Given N=24 dyads, this script is deliberately PARSIMONIOUS and pre-specified, to
avoid the small-N traps your own code review flagged (feature-selection leakage,
uncorrected multiple comparisons). It does NOT screen features by their
correlation with the outcome. Instead each construct is reduced to ONE composite
predictor before any outcome is touched:

    turn-level composite   = PC1 of the turn-level feature block
    vocal-alignment composite = PC1 of the vocal-alignment feature block

Two vocal predictors are reported, kept strictly separate:
  (a) OMNIBUS (confirmatory): blind PC1 of all vocal metrics. Conservative; it
      blends opposite-valence metrics so it under-detects on purpose.
  (b) THEORY-SPECIFIED (exploratory / pre-registration candidate): an intensity
      turn-edge coordination index motivated by PRIOR entrainment literature
      (Levitan & Hirschberg; intensity entrainment), NOT chosen from this
      dataset's results. Within THIS pilot it is labeled exploratory; it must be
      PRE-REGISTERED and confirmed in a larger independent sample. Treating it as
      confirmatory here would be HARKing.

Analyses per outcome (partner_eval, shared_reality, enjoyment, solo):
  1. DISTINCTNESS   correlation (+ VIF) between the two composites and among raw
                    metrics -> is vocal alignment a separate dimension?
  2. HIERARCHICAL   M0 controls -> M1 +turn-level -> M2 +vocal alignment.
                    Report R^2, dR^2 (M2-M1), nested-F p, and a PERMUTATION p
                    (shuffle the vocal composite) robust to small N.
  3. PARTIAL r      vocal composite vs outcome, controlling turn-level + controls
                    (Pearson and Spearman).
  4. COMMONALITY    partition R^2 into unique-turn, unique-vocal, shared.
  5. CORRECTION     Benjamini-Hochberg FDR across the 4 outcomes on dR^2 p.

A DOUBLE DISSOCIATION (vocal predicts felt connection but not a cognitive
outcome, while turn-level does the reverse) is reported descriptively if present.

Inputs
------
  dyad_level_dataset.csv   turn-level features + outcomes (one row / pair_id)
  vocal_alignment_dyad.csv from 04_vocal_alignment.py (pair_id x condition)

Condition handling: outcomes here are dyad-level, so the vocal-alignment block is
aggregated to pair_id (mean across piper/cloudy) for the primary test, and a
condition-stratified sensitivity is printed. (If you later obtain per-condition
outcomes, switch UNIT to 'session' and enter condition as a control.)

Outputs
-------
  dissociation_results.csv   one row per outcome with all stats
  dissociation_results.json  full metadata (seed, blocks, loadings) for repro

Usage
-----
  python 05_dissociation_test.py
  python 05_dissociation_test.py --vocal ../04_data/vocal_alignment_dyad.csv
  python 05_dissociation_test.py --n-perm 10000 --seed 42
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent

CONFIG = {
    "turn_csv": PROJECT / "04_data" / "dyad_level_dataset.csv",
    "vocal_csv": PROJECT / "04_data" / "vocal_alignment_dyad.csv",
    "out_csv": PROJECT / "05_analysis_outputs" / "dissociation_results.csv",
    "out_json": PROJECT / "05_analysis_outputs" / "dissociation_results.json",
    # PRIMARY outcome = "connection quality" composite (PC1 of these 3), matching
    # poster_analysis_pipeline.py. solo is excluded from the composite (separate
    # construct) and reported only as a secondary outcome.
    "composite_cols": [
        "dyad_partner_eval_mean", "dyad_shared_reality_mean", "dyad_enjoyment_mean",
    ],
    # SECONDARY outcomes (sensitivity): the individual scales + solo.
    "outcomes": [
        "dyad_partner_eval_mean", "dyad_shared_reality_mean",
        "dyad_enjoyment_mean", "dyad_solo_mean",
    ],
    # optional dyad covariates (robustness model; off by default to save power)
    "covariate_csv": PROJECT / "04_data" / "covariates_dyad.csv",
    "extra_controls": ["age_mean", "gender_mixed"],
    "with_covariates": False,
    # THEORY-SPECIFIED vocal index (from prior entrainment literature, NOT from
    # this data's results). Intensity coordination at turn transitions. Reported
    # as EXPLORATORY / pre-registration candidate, never as confirmatory here.
    "theory_index_cols": ["va_int_edge_proximity", "va_int_edge_synchrony"],
    # pre-specified turn-level block (NOT chosen by outcome correlation)
    "turn_block": [
        "total_turns", "participation_gini", "turn_taking_density",
        "ttr_mean", "i_rate_mean", "we_rate_mean", "q_count_mean", "bc_rate_mean",
        "sem_semantic_similarity", "sem_sentiment_synchrony",
    ],
    # vocal block = every va_* column found in vocal_csv (whole-turn + edge)
    "controls": ["Order"],
    "n_perm": 5000,
    "seed": 42,
    "unit": "dyad",   # 'dyad' (default) | 'session'
}


# ─── OLS helpers (numpy; no statsmodels dependency) ───────────────────────────
def ols_r2(X, y):
    """R^2 and residual SS for OLS of y on X (X already includes intercept)."""
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sse = float(resid @ resid)
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - sse / sst if sst > 0 else np.nan
    return r2, sse, beta


def _design(*cols, n):
    """Stack a list of 1-D arrays into a design matrix with intercept."""
    mat = [np.ones(n)]
    for c in cols:
        if c is not None:
            mat.append(np.asarray(c, float))
    return np.column_stack(mat)


def nested_f(sse_r, sse_f, df_r, df_f, n):
    """F-test for nested models (reduced r inside full f). df_* = #params."""
    num = (sse_r - sse_f) / (df_f - df_r)
    den = sse_f / (n - df_f)
    if den <= 0 or num < 0:
        return np.nan, np.nan
    F = num / den
    p = stats.f.sf(F, df_f - df_r, n - df_f)
    return float(F), float(p)


def partial_corr(x, y, covars, method="pearson"):
    """Correlation of x,y after regressing both on covars (n x k)."""
    n = len(x)
    C = np.column_stack([np.ones(n), covars]) if covars.size else np.ones((n, 1))
    def resid(v):
        b, _, _, _ = np.linalg.lstsq(C, v, rcond=None)
        return v - C @ b
    rx, ry = resid(np.asarray(x, float)), resid(np.asarray(y, float))
    if method == "spearman":
        return stats.spearmanr(rx, ry)
    return stats.pearsonr(rx, ry)


def cronbach_alpha(X):
    """Cronbach's alpha for an n_subjects x n_items matrix (internal
    consistency of the composite's component scales)."""
    X = np.asarray(X, float)
    k = X.shape[1]
    item_var = X.var(axis=0, ddof=1).sum()
    total_var = X.sum(axis=1).var(ddof=1)
    if total_var == 0 or k < 2:
        return np.nan
    return float(k / (k - 1) * (1 - item_var / total_var))


def bh_fdr(pvals):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    prev = 1.0
    for rank, idx in enumerate(reversed(order), start=1):
        i = n - rank + 1
        val = p[idx] * n / i
        prev = min(prev, val)
        adj[idx] = prev
    return adj


def vif(block_df):
    """Variance inflation factor for each column (collinearity check)."""
    cols = block_df.columns
    Z = StandardScaler().fit_transform(block_df.values)
    out = {}
    for j, c in enumerate(cols):
        y = Z[:, j]
        X = np.column_stack([np.ones(len(y)), np.delete(Z, j, axis=1)])
        r2, _, _ = ols_r2(X, y)
        out[c] = float(1 / (1 - r2)) if r2 < 1 else np.inf
    return out


# ─── data prep ────────────────────────────────────────────────────────────────
def load_merge(cfg):
    turn = pd.read_csv(cfg["turn_csv"])
    vocal = pd.read_csv(cfg["vocal_csv"])
    va_cols = [c for c in vocal.columns if c.startswith("va_")]
    if not va_cols:
        raise SystemExit("No va_* columns in vocal_csv -- run 04 first.")

    if cfg["unit"] == "dyad":
        # aggregate vocal alignment across conditions -> one row / pair_id
        vag = vocal.groupby("pair_id")[va_cols].mean().reset_index()
        df = turn.merge(vag, on="pair_id", how="inner")
    else:  # session-level (requires session-level outcomes; advanced)
        df = turn.merge(vocal, on="pair_id", how="inner")
        if "condition" in df:
            cfg["controls"] = cfg["controls"] + ["condition_code"]
            df["condition_code"] = (df["condition"] == "piper").astype(int)

    # optional: merge dyad-level covariates (age, gender) for a robustness model
    if cfg.get("with_covariates") and Path(cfg["covariate_csv"]).exists():
        cov = pd.read_csv(cfg["covariate_csv"])
        keep = ["pair_id"] + [c for c in cfg["extra_controls"] if c in cov.columns]
        df = df.merge(cov[keep], on="pair_id", how="left")
        cfg["controls"] = cfg["controls"] + [c for c in cfg["extra_controls"]
                                             if c in cov.columns]
        print(f"[covariates] added {cfg['extra_controls']} from {cfg['covariate_csv']}")
    return df, va_cols


def make_composite(df, cols, name):
    """Standardize block -> PC1. Returns (scores, loadings, var_explained)."""
    sub = df[cols].apply(pd.to_numeric, errors="coerce")
    sub = sub.dropna(axis=1, how="all")
    Z = StandardScaler().fit_transform(sub.fillna(sub.mean()).values)
    pca = PCA(n_components=1, random_state=0).fit(Z)
    scores = pca.transform(Z)[:, 0]
    # orient PC so its largest-magnitude loading is positive (interpretability)
    load = pca.components_[0]
    if load[np.argmax(np.abs(load))] < 0:
        scores, load = -scores, -load
    return scores, dict(zip(sub.columns, load.round(3))), float(pca.explained_variance_ratio_[0])


# ─── main analysis per outcome ────────────────────────────────────────────────
def analyse_outcome(y, turn_c, vocal_c, controls, n, n_perm, rng):
    # controls-only design (intercept + any control columns)
    Xc = np.column_stack([np.ones(n)] + ([c for c in controls] if controls else []))
    X1 = np.column_stack([Xc, turn_c])             # + turn-level
    X2 = np.column_stack([X1, vocal_c])            # + vocal

    r2_0, sse0, _ = ols_r2(Xc, y)
    r2_1, sse1, _ = ols_r2(X1, y)
    r2_2, sse2, b2 = ols_r2(X2, y)
    dR2 = r2_2 - r2_1

    F, p_f = nested_f(sse1, sse2, X1.shape[1], X2.shape[1], n)

    # permutation p for dR2 (shuffle vocal composite)
    perm_ge = 0
    for _ in range(n_perm):
        vp = rng.permutation(vocal_c)
        r2p, _, _ = ols_r2(np.column_stack([X1, vp]), y)
        if (r2p - r2_1) >= dR2 - 1e-12:
            perm_ge += 1
    p_perm = (perm_ge + 1) / (n_perm + 1)

    # partial correlations of vocal composite, controlling turn + controls
    cov = np.column_stack([turn_c] + ([c for c in controls] if controls else []))
    pr_p = partial_corr(vocal_c, y, cov, "pearson")
    pr_s = partial_corr(vocal_c, y, cov, "spearman")

    # commonality (2-block): unique + shared R^2 above controls
    r2_turn_only, _, _ = ols_r2(np.column_stack([Xc, turn_c]), y)
    r2_vocal_only, _, _ = ols_r2(np.column_stack([Xc, vocal_c]), y)
    uniq_turn = r2_2 - r2_vocal_only
    uniq_vocal = r2_2 - r2_turn_only
    common = (r2_turn_only - r2_0) + (r2_vocal_only - r2_0) - (r2_2 - r2_0)

    return {
        "R2_controls": r2_0, "R2_turn": r2_1, "R2_full": r2_2,
        "dR2_vocal": dR2, "nestedF": F, "p_nestedF": p_f, "p_perm": p_perm,
        "partial_r_pearson": pr_p.statistic, "partial_p_pearson": pr_p.pvalue,
        "partial_r_spearman": pr_s.statistic, "partial_p_spearman": pr_s.pvalue,
        "commonality_unique_turn": uniq_turn,
        "commonality_unique_vocal": uniq_vocal,
        "commonality_shared": common,
        "beta_vocal": float(b2[-1]),
    }


def run(cfg):
    rng = np.random.default_rng(cfg["seed"])
    df, va_cols = load_merge(cfg)
    n = len(df)
    print(f"Merged dataset: N={n} dyads, {len(va_cols)} vocal-alignment metrics.")
    if n < 15:
        print("[warn] very small N; treat all inference as exploratory.")

    turn_block = [c for c in cfg["turn_block"] if c in df.columns]
    turn_c, turn_load, turn_ve = make_composite(df, turn_block, "turn")
    vocal_c, vocal_load, vocal_ve = make_composite(df, va_cols, "vocal")

    # standardize composites
    turn_c = (turn_c - turn_c.mean()) / turn_c.std(ddof=0)
    vocal_c = (vocal_c - vocal_c.mean()) / vocal_c.std(ddof=0)

    # DISTINCTNESS
    r_comp = stats.pearsonr(turn_c, vocal_c)
    print(f"\n[Distinctness] corr(turn composite, vocal composite) = "
          f"{r_comp.statistic:+.3f} (p={r_comp.pvalue:.3f}); "
          f"PC1 var explained: turn {turn_ve:.0%}, vocal {vocal_ve:.0%}")
    print(f"  -> low |r| supports vocal alignment as a DISTINCT dimension.")

    # PRIMARY outcome: "connection quality" = pre-specified EQUAL-WEIGHT mean of
    # the 3 z-scored social scales. Equal weighting is more reproducible and
    # less sample-dependent than PCA loadings estimated on N=24. The PCA version
    # is kept as a sensitivity check.
    comp_cols = [c for c in cfg["composite_cols"] if c in df.columns]
    Zc = StandardScaler().fit_transform(df[comp_cols])
    df["connection_composite"] = Zc.mean(axis=1)                 # PRIMARY
    alpha = cronbach_alpha(df[comp_cols].values)
    comp_scores, comp_load, comp_ve = make_composite(df, comp_cols, "outcome")
    df["connection_composite_pca"] = comp_scores                # sensitivity
    r_ew_pca = stats.pearsonr(df["connection_composite"],
                              df["connection_composite_pca"]).statistic
    print(f"\n[Primary outcome] connection = equal-weight z-mean of "
          f"{[c.replace('dyad_','').replace('_mean','') for c in comp_cols]}; "
          f"Cronbach alpha={alpha:.2f}; agrees with PCA version r={r_ew_pca:.2f} "
          f"(PCA PC1 var {comp_ve:.0%}).")

    controls = []
    for c in cfg["controls"]:
        if c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce").fillna(0).values
            controls.append((v - v.mean()) / (v.std(ddof=0) or 1))

    def _one(out_name, tier, vocal_pred=None):
        vp = vocal_c if vocal_pred is None else vocal_pred
        y = pd.to_numeric(df[out_name], errors="coerce").values
        mask = np.isfinite(y)
        res = analyse_outcome(
            y[mask], turn_c[mask], vp[mask],
            [c[mask] for c in controls], mask.sum(), cfg["n_perm"], rng,
        )
        res["outcome"] = out_name
        res["tier"] = tier
        res["n"] = int(mask.sum())
        return res

    # primary (composite) first, then the individual scales as secondary
    rows = [_one("connection_composite", "primary")]
    for out in cfg["outcomes"]:
        if out not in df.columns:
            print(f"[skip] outcome {out} not found")
            continue
        rows.append(_one(out, "secondary"))

    # THEORY-SPECIFIED index (exploratory / pre-registration candidate):
    # intensity coordination at turn edges, motivated by prior entrainment
    # literature -- NOT chosen from this dataset. Tested on the primary outcome
    # only. Reported as exploratory; confirmation requires a preregistered sample.
    tcols = [c for c in cfg["theory_index_cols"] if c in df.columns]
    if len(tcols) >= 1:
        Zt = StandardScaler().fit_transform(df[tcols].fillna(df[tcols].mean()))
        theory_vocal = Zt.mean(axis=1)
        theory_vocal = (theory_vocal - theory_vocal.mean()) / (theory_vocal.std(ddof=0) or 1)
        r_t = _one("connection_composite", "exploratory", theory_vocal)
        r_t["outcome"] = "connection_composite [intensity-edge index]"
        rows.append(r_t)
        print(f"\n[Theory-specified index] intensity-edge coordination "
              f"({[c.replace('va_','') for c in tcols]}) vs connection: "
              f"dR2={r_t['dR2_vocal']:.3f}, p_perm={r_t['p_perm']:.3f}, "
              f"partial_r={r_t['partial_r_pearson']:+.2f}  "
              f"[EXPLORATORY -- pre-register to confirm]")

    # FDR is applied ONLY across the secondary scales (primary is a single
    # pre-specified test, so it is reported uncorrected).
    sec = [r for r in rows if r["tier"] == "secondary"]
    padj = bh_fdr([r["p_perm"] for r in sec]) if sec else []
    for r, pa in zip(sec, padj):
        r["p_perm_FDR"] = float(pa)
    for r in rows:
        if r["tier"] != "secondary":
            r["p_perm_FDR"] = np.nan  # primary + exploratory: not FDR-corrected

    res_df = pd.DataFrame(rows)
    front = ["tier", "outcome", "n", "R2_turn", "R2_full", "dR2_vocal",
             "p_nestedF", "p_perm", "p_perm_FDR",
             "partial_r_pearson", "partial_p_pearson"]
    res_df = res_df[front + [c for c in res_df.columns if c not in front]]

    Path(cfg["out_csv"]).parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(cfg["out_csv"], index=False)
    meta = {
        "n": n, "seed": cfg["seed"], "n_perm": cfg["n_perm"], "unit": cfg["unit"],
        "primary_outcome": "connection_composite (equal-weight z-mean)",
        "vocal_omnibus": "PC1 of all vocal metrics (confirmatory, conservative)",
        "theory_index_cols": tcols if 'tcols' in dir() else [],
        "theory_index_status": "EXPLORATORY / pre-registration candidate "
                               "(intensity-edge; from prior literature, not this data)",
        "composite_cols": comp_cols, "composite_cronbach_alpha": alpha,
        "composite_pca_loadings": comp_load, "composite_PC1_var": comp_ve,
        "equalweight_vs_pca_r": float(r_ew_pca),
        "turn_block": turn_block, "turn_loadings": turn_load,
        "turn_PC1_var": turn_ve, "vocal_block": va_cols,
        "vocal_loadings": vocal_load, "vocal_PC1_var": vocal_ve,
        "composite_corr": r_comp.statistic, "controls": cfg["controls"],
    }
    with open(cfg["out_json"], "w") as f:
        json.dump(meta, f, indent=2)

    # readable summary
    print("\n=== Incremental value of vocal alignment (beyond turn-level) ===")
    print(f"{'tier':10s} {'outcome':24s} {'R2_turn':>8s} {'R2_full':>8s} "
          f"{'dR2':>7s} {'p_perm':>7s} {'FDR':>6s} {'partial_r':>10s}")
    for r in rows:
        fdr = f"{r['p_perm_FDR']:6.3f}" if np.isfinite(r['p_perm_FDR']) else "   -- "
        print(f"{r['tier']:10s} {r['outcome']:24s} {r['R2_turn']:8.3f} "
              f"{r['R2_full']:8.3f} {r['dR2_vocal']:7.3f} {r['p_perm']:7.3f} "
              f"{fdr} {r['partial_r_pearson']:+10.3f}")
    print("(primary = connection composite, uncorrected single test; "
          "secondary scales FDR-corrected)")
    print(f"\nWrote {cfg['out_csv']}\n      {cfg['out_json']}")

    # descriptive double-dissociation hint
    _double_dissociation_hint(rows)
    return res_df


def _double_dissociation_hint(rows):
    if not rows:
        return
    by = {r["outcome"]: r for r in rows}
    vocal_sig = [o for o, r in by.items() if r["p_perm"] < 0.05]
    turn_dom = [o for o, r in by.items()
                if r["commonality_unique_turn"] > r["commonality_unique_vocal"]]
    if vocal_sig and turn_dom and set(vocal_sig) - set(turn_dom):
        print("[pattern] Possible dissociation: vocal alignment uniquely predicts "
              f"{sorted(set(vocal_sig) - set(turn_dom))}, while turn-level dominates "
              f"{sorted(set(turn_dom) - set(vocal_sig))}. Interpret cautiously at this N.")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--turn", default=str(CONFIG["turn_csv"]))
    p.add_argument("--vocal", default=str(CONFIG["vocal_csv"]))
    p.add_argument("--out-csv", default=str(CONFIG["out_csv"]))
    p.add_argument("--out-json", default=str(CONFIG["out_json"]))
    p.add_argument("--unit", choices=["dyad", "session"], default=CONFIG["unit"])
    p.add_argument("--n-perm", type=int, default=CONFIG["n_perm"])
    p.add_argument("--seed", type=int, default=CONFIG["seed"])
    p.add_argument("--with-covariates", action="store_true",
                   help="add age + gender from covariates_dyad.csv (robustness model)")
    a = p.parse_args()
    cfg = dict(CONFIG)
    cfg.update({"turn_csv": a.turn, "vocal_csv": a.vocal, "out_csv": a.out_csv,
                "out_json": a.out_json, "unit": a.unit, "n_perm": a.n_perm,
                "seed": a.seed, "with_covariates": a.with_covariates})
    return cfg


if __name__ == "__main__":
    run(parse_args())
