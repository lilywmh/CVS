"""Incremental validity of selected video-behavior variables (dyad level).

Purpose
-------
Implements 07_writeups/multimodal_analysis_plan_2026-07-09.md: does each of
the four reliability-selected video variables add explained variance in the
connection composite beyond the turn-level (transcript) composite and the
vocal intensity-edge index? Mirrors the vocal-stage framework in
test_vocal_alignment_incremental_validity.py.

Inputs
------
- 04_data/multimodal_dyad_table.csv           (video + vocal + outcomes; dyad level)
- 04_data/multimodal_dyad_session_table.csv   (for the QC-gated sensitivity arm)
- 04_data/dyad_level_dataset.csv              (turn-level block + Order, as vocal stage)

Outputs -> 05_analysis_outputs/multimodal_video_incremental/
-------
- incremental_results.csv    one row per outcome x video variable x model arm
- distinctness_matrix.csv    correlations among predictors/composites
- qc_sensitivity.csv         primary models rerun on QC-gated video values
- analysis_metadata.json     seed, inputs (sha256), loadings, versions
- multimodal_video_incremental_report.md

Example
-------
    python scripts/models/analyze_multimodal_video_incremental.py
    python scripts/models/analyze_multimodal_video_incremental.py --n-perm 10000

How to check correctness
------------------------
- n = 24 dyads in every primary row; permutation p reproducible with seed 42.
- Turn PC1 loadings in the metadata should be sign-aligned with the vocal
  stage run (PC1 sign is normalized to load positively on total_turns).
- Rerunning with unchanged inputs must produce byte-identical CSVs.

Common failure cases
--------------------
- Missing multimodal tables -> run scripts/models/build_multimodal_dyad_table.py.
- scipy absent -> parametric p columns are NaN; permutation inference still runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:  # optional; permutation inference is primary
    from scipy import stats as scipy_stats
except ImportError:  # pragma: no cover
    scipy_stats = None

SCRIPT_VERSION = "1.0.0 (2026-07-09)"
PROJECT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("CVS_DATA", PROJECT / "04_data"))
OUT_DIR = Path(os.environ.get("CVS_ANALYSIS_OUTPUTS", PROJECT / "05_analysis_outputs")) / "multimodal_video_incremental"

TURN_BLOCK = [
    "total_turns", "participation_gini", "turn_taking_density",
    "ttr_mean", "i_rate_mean", "we_rate_mean", "q_count_mean", "bc_rate_mean",
    "sem_semantic_similarity", "sem_sentiment_synchrony",
]
VOCAL_INDEX_COLS = ["va_int_edge_proximity", "va_int_edge_synchrony"]
VIDEO_VARS = [
    "video_delta_wrist_speed_px_s",
    "video_delta_active_motion_rate",
    "video_speaker_arm_openness",
    "video_mutual_partner_orientation",
]
COMPOSITE_COLS = ["dyad_partner_eval_mean", "dyad_shared_reality_mean", "dyad_enjoyment_mean"]
SECONDARY_OUTCOMES = COMPOSITE_COLS + ["dyad_solo_mean"]
CONTROLS = ["Order"]

logger = logging.getLogger("analyze_multimodal_video_incremental")


def fail(msg: str) -> None:
    logger.error(msg)
    sys.exit(1)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def zscore(x: pd.Series | np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    sd = np.nanstd(x, ddof=1)
    if sd == 0 or not np.isfinite(sd):
        fail("Zero-variance column encountered during z-scoring.")
    return (x - np.nanmean(x)) / sd


def pc1(df_block: pd.DataFrame, anchor: str) -> tuple[np.ndarray, dict[str, float], float]:
    """First principal component via SVD on the z-scored block.

    Sign is normalized so the loading on `anchor` is positive, making the
    composite's direction reproducible across runs and datasets.
    """
    Z = np.column_stack([zscore(df_block[c]) for c in df_block.columns])
    U, S, Vt = np.linalg.svd(Z - Z.mean(axis=0), full_matrices=False)
    loadings = Vt[0]
    scores = Z @ loadings
    anchor_idx = list(df_block.columns).index(anchor)
    if loadings[anchor_idx] < 0:
        loadings, scores = -loadings, -scores
    var_explained = float(S[0] ** 2 / (S ** 2).sum())
    return zscore(scores), dict(zip(df_block.columns, np.round(loadings, 4))), var_explained


def ols_r2(X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """R^2 and SSE for OLS with intercept."""
    X1 = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ beta
    sse = float(resid @ resid)
    sst = float(((y - y.mean()) ** 2).sum())
    return 1.0 - sse / sst, sse


def nested_f(sse_r: float, sse_f: float, df_extra: int, df_resid_full: int) -> tuple[float, float]:
    if df_resid_full <= 0 or sse_f <= 0:
        return np.nan, np.nan
    f_stat = ((sse_r - sse_f) / df_extra) / (sse_f / df_resid_full)
    p = float(scipy_stats.f.sf(f_stat, df_extra, df_resid_full)) if scipy_stats else np.nan
    return float(f_stat), p


def incremental_test(y: np.ndarray, base: np.ndarray, added: np.ndarray,
                     n_perm: int, rng: np.random.Generator) -> dict[str, float]:
    """DeltaR^2 of `added` over `base` with permutation p (shuffle `added`)."""
    n = len(y)
    r2_base, sse_base = ols_r2(base, y)
    full = np.column_stack([base, added])
    r2_full, sse_full = ols_r2(full, y)
    d_r2 = r2_full - r2_base
    f_stat, f_p = nested_f(sse_base, sse_full, 1, n - full.shape[1] - 1)
    perm_d = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(added)
        r2_p, _ = ols_r2(np.column_stack([base, perm]), y)
        perm_d[i] = r2_p - r2_base
    p_perm = float((np.sum(perm_d >= d_r2) + 1) / (n_perm + 1))
    return {"n": n, "r2_base": round(r2_base, 4), "r2_full": round(r2_full, 4),
            "delta_r2": round(d_r2, 4), "f_stat": round(f_stat, 3) if np.isfinite(f_stat) else np.nan,
            "f_p": round(f_p, 4) if np.isfinite(f_p) else np.nan, "p_perm": round(p_perm, 4)}


def bh_fdr(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(p)
    out[order] = np.minimum(ranked, 1.0)
    return [round(v, 4) for v in out]


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = {
        "dyad": DATA / "multimodal_dyad_table.csv",
        "session": DATA / "multimodal_dyad_session_table.csv",
        "turn": DATA / "dyad_level_dataset.csv",
    }
    for label, p in paths.items():
        if not p.exists():
            fail(f"Missing input ({label}): {p}. "
                 "Run scripts/models/build_multimodal_dyad_table.py first.")
    dyad = pd.read_csv(paths["dyad"])
    turn = pd.read_csv(paths["turn"])
    missing = [c for c in TURN_BLOCK + CONTROLS if c not in turn.columns]
    if missing:
        fail(f"{paths['turn']} lacks columns {missing}.")
    for col in VIDEO_VARS + VOCAL_INDEX_COLS:
        if col not in dyad.columns:
            fail(f"{paths['dyad']} lacks column {col}.")
    df = turn[["pair_id"] + TURN_BLOCK + CONTROLS + SECONDARY_OUTCOMES].merge(
        dyad[["pair_id"] + VIDEO_VARS + VOCAL_INDEX_COLS
             + ["video_arm_qc_n_warning", "video_min_arm_detection"]],
        on="pair_id", how="inner")
    if df[TURN_BLOCK + VIDEO_VARS + VOCAL_INDEX_COLS + SECONDARY_OUTCOMES].isna().any().any():
        na_cols = df.columns[df.isna().any()].tolist()
        fail(f"Missing values in analysis columns {na_cols}; inspect the merge.")
    session = pd.read_csv(paths["session"])
    return df.reset_index(drop=True), session


def qc_gated_video(session: pd.DataFrame) -> pd.DataFrame:
    ok = (session["video_arm_qc_n_warning"] == 0) & (session["video_min_arm_detection"] >= 0.5)
    logger.info("QC-gated sensitivity: keeping %d/%d sessions.", int(ok.sum()), len(session))
    gated = (session[ok].groupby("pair_id")[VIDEO_VARS].mean()
             .add_suffix("_qcgated").reset_index())
    return gated


def run(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    df, session = load_inputs()
    n = len(df)
    logger.info("Analysis dataset: %d dyads.", n)

    # Composites (definitions identical to vocal stage)
    turn_c, turn_loadings, turn_ve = pc1(df[TURN_BLOCK], anchor="total_turns")
    vocal_index = zscore(np.mean([zscore(df[c]) for c in VOCAL_INDEX_COLS], axis=0))
    Zc = np.column_stack([zscore(df[c]) for c in COMPOSITE_COLS])
    df["connection_composite"] = Zc.mean(axis=1)
    controls = np.column_stack([zscore(df[c]) for c in CONTROLS])

    # 1. Distinctness matrix
    named = {"turn_composite": turn_c, "vocal_intensity_edge": vocal_index}
    named.update({v: zscore(df[v]) for v in VIDEO_VARS})
    names = list(named)
    dist = pd.DataFrame(
        [[round(pearson(named[a], named[b]), 3) for b in names] for a in names],
        index=names, columns=names)

    # 2. Incremental validity
    rows = []
    outcomes = {"connection_composite": df["connection_composite"].to_numpy()}
    outcomes.update({o: zscore(df[o]) for o in SECONDARY_OUTCOMES})
    for outcome_name, y in outcomes.items():
        m1 = np.column_stack([turn_c, controls])            # turn + Order
        m2 = np.column_stack([turn_c, controls, vocal_index])
        for video_var in VIDEO_VARS:
            v = zscore(df[video_var])
            res_m2 = incremental_test(y, m2, v, args.n_perm, rng)
            res_m1 = incremental_test(y, m1, v, args.n_perm, rng)
            rows.append({
                "outcome": outcome_name, "video_variable": video_var,
                "r_zero_order": round(pearson(v, y), 3),
                **{f"{k}_overM2": val for k, val in res_m2.items()},
                "delta_r2_overM1": res_m1["delta_r2"], "p_perm_overM1": res_m1["p_perm"],
            })
    results = pd.DataFrame(rows)
    primary = results["outcome"] == "connection_composite"
    results.loc[primary, "p_perm_overM2_fdr"] = bh_fdr(
        results.loc[primary, "p_perm_overM2"].tolist())

    # 3. QC-gated sensitivity (primary outcome, M2 arm)
    gated = qc_gated_video(session)
    dfg = df.merge(gated, on="pair_id", how="inner")
    qc_rows = []
    yg = dfg["connection_composite"].to_numpy()
    idx = df["pair_id"].isin(dfg["pair_id"]).to_numpy()
    m2g = np.column_stack([turn_c[idx], controls[idx], vocal_index[idx]])
    for video_var in VIDEO_VARS:
        vg = zscore(dfg[f"{video_var}_qcgated"])
        res = incremental_test(yg, m2g, vg, args.n_perm, rng)
        qc_rows.append({"outcome": "connection_composite",
                        "video_variable": f"{video_var}_qcgated", **res})
    qc_results = pd.DataFrame(qc_rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT_DIR / "incremental_results.csv", index=False)
    dist.to_csv(OUT_DIR / "distinctness_matrix.csv")
    qc_results.to_csv(OUT_DIR / "qc_sensitivity.csv", index=False)

    metadata = {
        "script": "scripts/models/analyze_multimodal_video_incremental.py",
        "script_version": SCRIPT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": args.seed, "n_perm": args.n_perm, "n_dyads": n,
        "turn_pc1_loadings": turn_loadings,
        "turn_pc1_var_explained": round(turn_ve, 3),
        "vocal_index_cols": VOCAL_INDEX_COLS, "video_vars": VIDEO_VARS,
        "scipy_available": scipy_stats is not None,
        "inputs": [{"path": str(p), "sha256": sha256_of(p)} for p in [
            DATA / "multimodal_dyad_table.csv",
            DATA / "multimodal_dyad_session_table.csv",
            DATA / "dyad_level_dataset.csv"]],
    }
    (OUT_DIR / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    write_report(results, dist, qc_results, metadata)
    logger.info("Wrote outputs -> %s", OUT_DIR)


def write_report(results: pd.DataFrame, dist: pd.DataFrame,
                 qc_results: pd.DataFrame, metadata: dict) -> None:
    primary = results[results["outcome"] == "connection_composite"].copy()
    lines = [
        "# Multimodal Video Incremental-Validity Results",
        "",
        f"Generated {metadata['created_at']} · seed {metadata['seed']} · "
        f"{metadata['n_perm']} permutations · N = {metadata['n_dyads']} dyads · EXPLORATORY",
        "",
        "Plan: `07_writeups/multimodal_analysis_plan_2026-07-09.md`. "
        "Permutation p is the primary inference; parametric F descriptive. "
        "delta_active_motion is a sensitivity twin of delta_wrist_speed, not an "
        "independent discovery.",
        "",
        "## Distinctness (correlations among predictors)",
        "",
        dist.to_markdown(),
        "",
        "## Primary outcome (connection composite): video over turn + Order + vocal",
        "",
        primary[["video_variable", "r_zero_order", "r2_base_overM2", "delta_r2_overM2",
                 "p_perm_overM2", "p_perm_overM2_fdr", "delta_r2_overM1", "p_perm_overM1"]]
        .to_markdown(index=False),
        "",
        "## QC-gated sensitivity (primary outcome)",
        "",
        qc_results[["video_variable", "n", "delta_r2", "p_perm"]].to_markdown(index=False),
        "",
        "## Secondary outcomes (uncorrected, exploratory)",
        "",
        results[results["outcome"] != "connection_composite"][
            ["outcome", "video_variable", "r_zero_order", "delta_r2_overM2", "p_perm_overM2"]]
        .to_markdown(index=False),
        "",
        "## Reading guide",
        "",
        "- `delta_r2_overM2`: variance added beyond turn-level composite, Order, "
        "and the vocal intensity-edge index.",
        "- A null here is informative: variables were selected for reliability, "
        "not outcome association, so estimates are unbiased by selection.",
        "",
    ]
    (OUT_DIR / "multimodal_video_incremental_report.md").write_text(
        "\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-perm", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args)


if __name__ == "__main__":
    main()
