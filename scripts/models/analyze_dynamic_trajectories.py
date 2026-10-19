"""Analyze within-conversation dynamic trajectory predictors of connection.

Tests whether trajectory summaries (slope, early-late shift, variability)
predict partner_eval beyond static means.  Dyad-level regressions (N=24)
with permutation inference and bootstrap CIs.

Inputs
------
- 05_analysis_outputs/dynamic_trajectory/trajectory_summaries.csv
  (from build_dynamic_trajectory_features.py)

Outputs
-------
- 05_analysis_outputs/dynamic_trajectory/model_results.csv
- 05_analysis_outputs/dynamic_trajectory/summary.md
- 05_analysis_outputs/dynamic_trajectory/analysis_manifest.json

Example
-------
    python scripts/models/analyze_dynamic_trajectories.py
    python scripts/models/analyze_dynamic_trajectories.py --window question --nperm 10000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

ANALYSIS = Path(os.environ.get("CVS_ANALYSIS_OUTPUTS", PROJECT / "05_analysis_outputs"))
IN_DIR = ANALYSIS / "dynamic_trajectory"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42

TRAJECTORY_FEATURES = [
    "mean_turn_dur", "floor_share", "floor_share_imbalance",
    "backchannel_rate", "response_latency", "lexical_overlap",
    "turn_rate", "f0_mean", "int_mean", "rate_sylps",
]

DYNAMIC_SUFFIXES = ["slope", "early_late_diff", "sd"]

# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def z(v: np.ndarray) -> np.ndarray:
    """Z-score an array; returns NaN if constant."""
    v = np.asarray(v, float)
    sd = np.nanstd(v, ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        return np.full_like(v, np.nan)
    return (v - np.nanmean(v)) / sd


def ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """OLS coefficients via numpy."""
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b


def r_squared(X: np.ndarray, y: np.ndarray, b: np.ndarray) -> float:
    """R-squared from design matrix and coefficients."""
    yhat = X @ b
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    if ss_tot == 0:
        return np.nan
    return float(1 - ss_res / ss_tot)


def _perm_test_dyad(
    X: np.ndarray,
    y: np.ndarray,
    dyad_ids: np.ndarray,
    j: int,
    n_perm: int,
    rng: np.random.Generator,
) -> float:
    """Dyad-level sign-flip permutation test for coefficient j.

    Permutes outcome values at the dyad level (since outcome is dyad-level,
    each dyad has a single partner_eval value).
    """
    b_obs = ols(X, y)[j]
    unique_dyads = np.unique(dyad_ids)
    n_dyads = len(unique_dyads)

    exceed = 0
    for _ in range(n_perm):
        # permute outcome by shuffling dyad assignment
        perm_order = rng.permutation(n_dyads)
        y_perm = np.empty_like(y)
        for target_idx, source_idx in enumerate(perm_order):
            target_mask = dyad_ids == unique_dyads[target_idx]
            source_mask = dyad_ids == unique_dyads[source_idx]
            y_perm[target_mask] = y[source_mask]
        b_perm = ols(X, y_perm)[j]
        if abs(b_perm) >= abs(b_obs):
            exceed += 1

    return (1 + exceed) / (n_perm + 1)


def _bootstrap_ci(
    X: np.ndarray,
    y: np.ndarray,
    j: int,
    n_boot: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Bootstrap CI for coefficient j (case resampling)."""
    n = len(y)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        try:
            boots[b] = ols(X[idx], y[idx])[j]
        except np.linalg.LinAlgError:
            boots[b] = np.nan
    lo = float(np.nanpercentile(boots, 100 * alpha / 2))
    hi = float(np.nanpercentile(boots, 100 * (1 - alpha / 2)))
    return lo, hi


def _delta_r2_perm(
    X_full: np.ndarray,
    X_reduced: np.ndarray,
    y: np.ndarray,
    n_perm: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Permutation test for delta-R-squared (full vs reduced model)."""
    b_full = ols(X_full, y)
    b_red = ols(X_reduced, y)
    r2_full = r_squared(X_full, y, b_full)
    r2_red = r_squared(X_reduced, y, b_red)
    delta_obs = r2_full - r2_red

    exceed = 0
    for _ in range(n_perm):
        # residualize y under reduced model, permute residuals
        resid = y - X_reduced @ b_red
        resid_perm = rng.permutation(resid)
        y_perm = X_reduced @ b_red + resid_perm

        b_f = ols(X_full, y_perm)
        b_r = ols(X_reduced, y_perm)
        delta_perm = r_squared(X_full, y_perm, b_f) - r_squared(X_reduced, y_perm, b_r)
        if delta_perm >= delta_obs:
            exceed += 1

    p = (1 + exceed) / (n_perm + 1)
    return delta_obs, p


def bh_fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for r, i in enumerate(o[::-1]):
        rank = n - r
        prev = min(prev, p[i] * n / rank)
        q[i] = prev
    return q


# ---------------------------------------------------------------------------
# Model runners
# ---------------------------------------------------------------------------

def run_models(
    df: pd.DataFrame,
    window_type: str,
    n_perm: int,
    n_boot: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Run the full model sequence for one window type.

    Models:
    1. static-only:  partner_eval ~ mean_{feature}_z
    2. dynamic-only: partner_eval ~ {dynamic_summary}_{feature}_z
    3. static+dynamic: partner_eval ~ mean_{feature}_z + slope_{feature}_z
       with delta-R2 test for the slope increment
    """
    sub = df[df["window_type"] == window_type].copy()
    if len(sub) == 0:
        print(f"  No data for window_type={window_type}, skipping.")
        return pd.DataFrame()

    y_raw = sub["partner_eval"].values
    pair_ids = sub["pair_id"].values
    n = len(sub)

    results: list[dict] = []

    for feat in TRAJECTORY_FEATURES:
        mean_col = f"{feat}_mean"
        if mean_col not in sub.columns:
            continue

        # --- Model 1: static-only ---
        x_raw = sub[mean_col].values
        mask = np.isfinite(x_raw) & np.isfinite(y_raw)
        if mask.sum() < 5:
            continue

        y = z(y_raw[mask])
        x = z(x_raw[mask])
        X = np.column_stack([np.ones(mask.sum()), x])
        b = ols(X, y)
        r2 = r_squared(X, y, b)
        p_perm = _perm_test_dyad(X, y, pair_ids[mask], 1, n_perm, rng)
        ci_lo, ci_hi = _bootstrap_ci(X, y, 1, n_boot, rng)
        # Pearson r
        r_val = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else np.nan

        results.append({
            "window_type": window_type,
            "feature": feat,
            "model": "static_only",
            "predictor": f"{feat}_mean",
            "beta": float(b[1]),
            "ci95_lo": ci_lo,
            "ci95_hi": ci_hi,
            "r": r_val,
            "r2": r2,
            "delta_r2": np.nan,
            "p_perm": p_perm,
            "p_delta_r2": np.nan,
            "n": int(mask.sum()),
        })

        # --- Model 2: dynamic-only (slope, early_late_diff, sd) ---
        for suf in DYNAMIC_SUFFIXES:
            dyn_col = f"{feat}_{suf}"
            if dyn_col not in sub.columns:
                continue
            xd_raw = sub[dyn_col].values
            mask_d = np.isfinite(xd_raw) & np.isfinite(y_raw)
            if mask_d.sum() < 5:
                continue

            yd = z(y_raw[mask_d])
            xd = z(xd_raw[mask_d])
            Xd = np.column_stack([np.ones(mask_d.sum()), xd])
            bd = ols(Xd, yd)
            r2d = r_squared(Xd, yd, bd)
            p_d = _perm_test_dyad(Xd, yd, pair_ids[mask_d], 1, n_perm, rng)
            ci_lo_d, ci_hi_d = _bootstrap_ci(Xd, yd, 1, n_boot, rng)
            r_d = float(np.corrcoef(xd, yd)[0, 1]) if len(xd) > 2 else np.nan

            results.append({
                "window_type": window_type,
                "feature": feat,
                "model": f"dynamic_only_{suf}",
                "predictor": dyn_col,
                "beta": float(bd[1]),
                "ci95_lo": ci_lo_d,
                "ci95_hi": ci_hi_d,
                "r": r_d,
                "r2": r2d,
                "delta_r2": np.nan,
                "p_perm": p_d,
                "p_delta_r2": np.nan,
                "n": int(mask_d.sum()),
            })

        # --- Model 3: static + slope (incremental test) ---
        slope_col = f"{feat}_slope"
        if slope_col not in sub.columns:
            continue
        xs_raw = sub[mean_col].values
        xsl_raw = sub[slope_col].values
        mask_3 = np.isfinite(xs_raw) & np.isfinite(xsl_raw) & np.isfinite(y_raw)
        if mask_3.sum() < 6:
            continue

        y3 = z(y_raw[mask_3])
        xs3 = z(xs_raw[mask_3])
        xsl3 = z(xsl_raw[mask_3])
        X_full = np.column_stack([np.ones(mask_3.sum()), xs3, xsl3])
        X_red = np.column_stack([np.ones(mask_3.sum()), xs3])

        b3 = ols(X_full, y3)
        r2_3 = r_squared(X_full, y3, b3)
        r2_red = r_squared(X_red, y3, ols(X_red, y3))
        delta_r2, p_delta = _delta_r2_perm(X_full, X_red, y3, n_perm, rng)
        # CI for slope coefficient in combined model
        ci_lo_3, ci_hi_3 = _bootstrap_ci(X_full, y3, 2, n_boot, rng)

        results.append({
            "window_type": window_type,
            "feature": feat,
            "model": "static_plus_slope",
            "predictor": f"{feat}_mean + {feat}_slope",
            "beta": float(b3[2]),  # slope coefficient
            "ci95_lo": ci_lo_3,
            "ci95_hi": ci_hi_3,
            "r": np.nan,
            "r2": r2_3,
            "delta_r2": delta_r2,
            "p_perm": np.nan,
            "p_delta_r2": p_delta,
            "n": int(mask_3.sum()),
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def write_summary(results: pd.DataFrame, out: Path) -> None:
    """Write a markdown summary of results."""
    lines: list[str] = [
        "# Dynamic Trajectory Analysis — Summary",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
    ]

    for wtype in results["window_type"].unique():
        sub = results[results["window_type"] == wtype]
        lines.append(f"## Window type: {wtype}")
        lines.append("")

        # Static-only results
        static = sub[sub["model"] == "static_only"].sort_values("p_perm")
        lines.append("### Static-only models (partner_eval ~ mean_feature_z)")
        lines.append("")
        if len(static) > 0:
            cols = ["feature", "beta", "r", "r2", "p_perm", "n"]
            lines.append(static[cols].round(3).to_markdown(index=False))
        else:
            lines.append("No results.")
        lines.append("")

        # Best dynamic predictors
        dynamic = sub[sub["model"].str.startswith("dynamic_only")].sort_values("p_perm")
        lines.append("### Dynamic-only models (partner_eval ~ dynamic_summary_z)")
        lines.append("")
        if len(dynamic) > 0:
            cols = ["feature", "model", "predictor", "beta", "r", "r2", "p_perm", "n"]
            lines.append(dynamic[cols].round(3).to_markdown(index=False))
        else:
            lines.append("No results.")
        lines.append("")

        # Incremental: static + slope
        incr = sub[sub["model"] == "static_plus_slope"].sort_values("p_delta_r2")
        lines.append("### Static + Slope (incremental delta-R2 test)")
        lines.append("")
        if len(incr) > 0:
            cols = ["feature", "beta", "r2", "delta_r2", "p_delta_r2", "n"]
            lines.append(incr[cols].round(3).to_markdown(index=False))
        else:
            lines.append("No results.")
        lines.append("")

        # Flag significant incremental findings
        sig_incr = incr[incr["p_delta_r2"] < 0.05] if len(incr) > 0 else pd.DataFrame()
        if len(sig_incr) > 0:
            lines.append("**Significant incremental dynamic predictors (p < .05):**")
            for _, row in sig_incr.iterrows():
                lines.append(
                    f"- {row['feature']}: delta-R2 = {row['delta_r2']:.3f}, "
                    f"p = {row['p_delta_r2']:.3f}"
                )
        else:
            lines.append("No dynamic feature adds significant incremental variance "
                         "beyond the static mean.")
        lines.append("")

    # Descriptive statistics
    lines.append("## Interpretation")
    lines.append("")
    lines.append("N = 24 dyads. All tests are exploratory given the small sample.")
    lines.append("Permutation p-values use dyad-level outcome shuffling.")
    lines.append("Bootstrap CIs use case resampling (2000 draws).")
    lines.append("")

    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--window", choices=["question", "time_bin", "both"],
                    default="both",
                    help="Which window type(s) to analyze")
    ap.add_argument("--input", type=Path, default=IN_DIR / "trajectory_summaries.csv",
                    help="Path to trajectory_summaries.csv")
    ap.add_argument("--out", type=Path, default=IN_DIR,
                    help="Output directory")
    ap.add_argument("--nperm", type=int, default=5000,
                    help="Number of permutations")
    ap.add_argument("--nboot", type=int, default=2000,
                    help="Number of bootstrap draws")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("Loading trajectory summaries...")
    df = pd.read_csv(args.input)
    print(f"  {len(df)} rows, {df['pair_id'].nunique()} dyads, "
          f"window types: {sorted(df['window_type'].unique())}")

    if "partner_eval" not in df.columns:
        sys.exit("trajectory_summaries.csv is missing partner_eval — "
                 "run build_dynamic_trajectory_features.py first")

    windows_to_run = []
    if args.window in ("question", "both"):
        windows_to_run.append("question")
    if args.window in ("time_bin", "both"):
        windows_to_run.append("time_bin")

    all_results = []
    for wtype in windows_to_run:
        print(f"\nRunning models for window_type={wtype}...")
        res = run_models(df, wtype, args.nperm, args.nboot, rng)
        if len(res) > 0:
            all_results.append(res)
            print(f"  {len(res)} model rows")

    if not all_results:
        sys.exit("No results produced.")

    results = pd.concat(all_results, ignore_index=True)

    # FDR correction across all permutation p-values
    perm_mask = results["p_perm"].notna()
    if perm_mask.any():
        results.loc[perm_mask, "q_perm_BH"] = bh_fdr(
            results.loc[perm_mask, "p_perm"].values
        )
    delta_mask = results["p_delta_r2"].notna()
    if delta_mask.any():
        results.loc[delta_mask, "q_delta_r2_BH"] = bh_fdr(
            results.loc[delta_mask, "p_delta_r2"].values
        )

    results = results.round(4)
    results.to_csv(args.out / "model_results.csv", index=False)
    print(f"\nWrote {args.out / 'model_results.csv'} ({len(results)} rows)")

    # Summary
    write_summary(results, args.out)
    print(f"Wrote {args.out / 'summary.md'}")

    # Print headline
    static_q = results[
        (results["model"] == "static_only")
        & (results["feature"] == "mean_turn_dur")
        & (results["window_type"] == "question")
    ]
    if len(static_q) > 0:
        row = static_q.iloc[0]
        print(f"\nReplication check — mean_turn_dur static-only (question windows):")
        print(f"  r = {row['r']:.3f}, R2 = {row['r2']:.3f}, p_perm = {row['p_perm']:.4f}")

    # Best dynamic predictor
    dyn = results[results["model"].str.startswith("dynamic_only")].sort_values("p_perm")
    if len(dyn) > 0:
        best = dyn.iloc[0]
        print(f"\nStrongest dynamic predictor:")
        print(f"  {best['predictor']}: r = {best['r']:.3f}, p_perm = {best['p_perm']:.4f}")

    # Best incremental
    incr = results[results["model"] == "static_plus_slope"].sort_values("p_delta_r2")
    if len(incr) > 0:
        best_i = incr.iloc[0]
        print(f"\nBest incremental (static + slope):")
        print(f"  {best_i['feature']}: delta-R2 = {best_i['delta_r2']:.3f}, "
              f"p = {best_i['p_delta_r2']:.4f}")

    # Analysis manifest
    manifest = {
        "status": "COMPLETE",
        "script": str(Path(__file__).resolve().relative_to(PROJECT)),
        "created": datetime.now().isoformat(timespec="seconds"),
        "input": str(args.input),
        "n_dyads": int(df["pair_id"].nunique()),
        "window_types": windows_to_run,
        "n_perm": args.nperm,
        "n_boot": args.nboot,
        "seed": args.seed,
        "n_model_rows": len(results),
        "python": sys.version,
    }
    (args.out / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.out / 'analysis_manifest.json'}")
    print("Done.")


if __name__ == "__main__":
    main()
