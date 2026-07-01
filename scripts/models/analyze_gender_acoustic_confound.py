#!/usr/bin/env python3
"""
analyze_gender_acoustic_confound.py
===================================
Exploratory check for whether gender composition and raw speaker pitch
difference explain the vocal-alignment association with connection outcomes.

Research question
-----------------
Same-gender dyads may naturally have closer baseline pitch and may also build
rapport more easily. This script tests that potential confound by:

1. Comparing same-gender vs mixed-gender dyads on connection outcomes, vocal
   alignment metrics, and raw speaker pitch-difference.
2. Running nested regressions for the connection composite:
      outcome ~ acoustic predictor
      outcome ~ acoustic predictor + gender_mixed
      outcome ~ acoustic predictor + gender_mixed + raw pitch difference

Inputs
------
  ${CVS_DATA:-04_data}/covariates_dyad.csv
  ${CVS_DATA:-04_data}/dyad_level_dataset.csv
  ${CVS_DATA:-04_data}/vocal_alignment_dyad.csv
  ${CVS_DATA:-04_data}/acoustic_turns.csv

Outputs
-------
  ${CVS_ANALYSIS_OUTPUTS:-05_analysis_outputs}/gender_mixed_analysis.csv
  ${CVS_ANALYSIS_OUTPUTS:-05_analysis_outputs}/gender_pitch_confound_check.csv

Usage
-----
  python scripts/models/analyze_gender_acoustic_confound.py
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


PROJECT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("CVS_DATA", PROJECT / "04_data"))
ANALYSIS_OUTPUTS = Path(
    os.environ.get("CVS_ANALYSIS_OUTPUTS", PROJECT / "05_analysis_outputs")
)

OUTCOME_COLS = [
    "dyad_partner_eval_mean",
    "dyad_shared_reality_mean",
    "dyad_enjoyment_mean",
]

FEATURE_COLS = [
    "total_turns",
    "participation_gini",
    "turn_taking_density",
    "ttr_mean",
    "i_rate_mean",
    "we_rate_mean",
    "q_count_mean",
    "q_count_asym",
    "bc_rate_mean",
    "sem_semantic_similarity",
    "sem_sentiment_synchrony",
]

ACOUSTIC_PREDICTORS = [
    "vocal_intensity_edge_index",
    "va_f0_proximity",
    "va_f0_edge_proximity",
    "speaker_f0_mean_diff_hz",
]


def zscore(series: pd.Series) -> pd.Series:
    return (series - series.mean()) / series.std(ddof=0)


def add_connection_composite(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    mask = out[OUTCOME_COLS].notna().all(axis=1)
    if not mask.any():
        out["connection_composite"] = np.nan
        return out

    z = StandardScaler().fit_transform(out.loc[mask, OUTCOME_COLS])
    pc1 = PCA(n_components=1).fit_transform(z)[:, 0]
    mean_connection = out.loc[mask, OUTCOME_COLS].mean(axis=1)
    if np.corrcoef(pc1, mean_connection)[0, 1] < 0:
        pc1 = -pc1
    out.loc[mask, "connection_composite"] = pc1
    return out


def add_vocal_summaries(df: pd.DataFrame, vocal: pd.DataFrame) -> pd.DataFrame:
    va_cols = [c for c in vocal.columns if c.startswith("va_")]
    vocal_dyad = vocal.groupby("pair_id")[va_cols].mean().reset_index()

    index_cols = ["va_int_edge_proximity", "va_int_edge_synchrony"]
    mask = vocal_dyad[index_cols].notna().all(axis=1)
    if mask.any():
        z = StandardScaler().fit_transform(vocal_dyad.loc[mask, index_cols])
        vocal_dyad.loc[mask, "vocal_intensity_edge_index"] = z.mean(axis=1)

    return df.merge(vocal_dyad, on="pair_id", how="left")


def add_pitch_difference(df: pd.DataFrame, turns: pd.DataFrame) -> pd.DataFrame:
    required = ["pair_id", "dyad_id", "condition", "speaker", "f0_mean"]
    rows = []
    for (pair_id, _dyad, condition), group in (
        turns.dropna(subset=required)
        .groupby(["pair_id", "dyad_id", "condition"])
    ):
        speaker_means = group.groupby("speaker")["f0_mean"].mean().dropna()
        if len(speaker_means) >= 2:
            values = speaker_means.values[:2]
            rows.append({
                "pair_id": pair_id,
                "condition": condition,
                "speaker_f0_mean_diff_hz": abs(values[0] - values[1]),
            })

    if not rows:
        df["speaker_f0_mean_diff_hz"] = np.nan
        return df

    pitch_diff = (
        pd.DataFrame(rows)
        .groupby("pair_id")["speaker_f0_mean_diff_hz"]
        .mean()
        .reset_index()
    )
    return df.merge(pitch_diff, on="pair_id", how="left")


def add_turn_composite(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    available = [c for c in FEATURE_COLS if c in out.columns]
    mask = out[available].notna().all(axis=1)
    if not available or not mask.any():
        out["turn_composite"] = np.nan
        return out

    z = StandardScaler().fit_transform(out.loc[mask, available])
    pc1 = PCA(n_components=1).fit_transform(z)[:, 0]
    if "total_turns" in out and np.corrcoef(pc1, out.loc[mask, "total_turns"])[0, 1] < 0:
        pc1 = -pc1
    out.loc[mask, "turn_composite"] = pc1
    return out


def cohen_d(mixed: pd.Series, same: pd.Series) -> float:
    pooled = np.sqrt(
        ((len(mixed) - 1) * mixed.var(ddof=1) + (len(same) - 1) * same.var(ddof=1))
        / (len(mixed) + len(same) - 2)
    )
    return (mixed.mean() - same.mean()) / pooled if pooled and np.isfinite(pooled) else np.nan


def permutation_p(values: np.ndarray, labels: np.ndarray, observed: float, n_perm: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        permuted = rng.permutation(labels)
        diff = values[permuted == 1].mean() - values[permuted == 0].mean()
        if abs(diff) >= abs(observed):
            count += 1
    return (count + 1) / (n_perm + 1)


def gender_group_comparisons(df: pd.DataFrame, n_perm: int = 10000, seed: int = 42) -> pd.DataFrame:
    variables = (
        ["connection_composite", *OUTCOME_COLS, "dyad_solo_mean"]
        + [c for c in FEATURE_COLS if c in df.columns]
        + [
            "speaker_f0_mean_diff_hz",
            "vocal_intensity_edge_index",
            "va_f0_proximity",
            "va_f0_edge_proximity",
            "va_int_edge_proximity",
            "va_int_edge_synchrony",
        ]
    )

    rows = []
    for var in variables:
        if var not in df.columns:
            continue
        sub = df[["pair_id", "gender_mixed", var]].dropna()
        sub = sub[sub["gender_mixed"].isin([0, 1])]
        if sub["gender_mixed"].nunique() < 2:
            continue

        same = sub[sub["gender_mixed"] == 0][var]
        mixed = sub[sub["gender_mixed"] == 1][var]
        if len(same) < 2 or len(mixed) < 2:
            continue

        observed = mixed.mean() - same.mean()
        t_stat, p_welch = stats.ttest_ind(mixed, same, equal_var=False)
        p_perm = permutation_p(
            sub[var].to_numpy(),
            sub["gender_mixed"].to_numpy(),
            observed,
            n_perm=n_perm,
            seed=seed,
        )
        rows.append({
            "variable": var,
            "n": len(sub),
            "n_same": len(same),
            "n_mixed": len(mixed),
            "same_mean": same.mean(),
            "mixed_mean": mixed.mean(),
            "mixed_minus_same": observed,
            "cohens_d": cohen_d(mixed, same),
            "welch_t": t_stat,
            "welch_p": p_welch,
            "perm_p": p_perm,
        })

    return pd.DataFrame(rows).sort_values("perm_p")


def fit_standardized_ols(df: pd.DataFrame, y_col: str, x_cols: list[str]):
    data = df[[y_col, *x_cols]].dropna().copy()
    y = zscore(data[y_col])
    x = pd.DataFrame(index=data.index)
    for col in x_cols:
        x[col] = data[col] if col == "gender_mixed" else zscore(data[col])
    model = sm.OLS(y, sm.add_constant(x)).fit()
    return model, data


def regression_confound_checks(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for predictor in ACOUSTIC_PREDICTORS:
        needed = ["connection_composite", predictor, "gender_mixed"]
        if predictor != "speaker_f0_mean_diff_hz":
            needed.append("speaker_f0_mean_diff_hz")
        if any(c not in df.columns for c in needed):
            continue

        sub = df[needed].dropna().copy()
        sub = sub[sub["gender_mixed"].isin([0, 1])]
        if len(sub) < 12:
            continue

        model_0, _ = fit_standardized_ols(sub, "connection_composite", [predictor])
        model_1, _ = fit_standardized_ols(sub, "connection_composite", [predictor, "gender_mixed"])
        controls = [predictor, "gender_mixed"]
        if predictor != "speaker_f0_mean_diff_hz":
            controls.append("speaker_f0_mean_diff_hz")
        model_2, _ = fit_standardized_ols(sub, "connection_composite", controls)

        rows.append({
            "predictor": predictor,
            "n": len(sub),
            "r2_predictor_only": model_0.rsquared,
            "predictor_beta_only": model_0.params[predictor],
            "predictor_p_only": model_0.pvalues[predictor],
            "r2_plus_gender": model_1.rsquared,
            "delta_r2_gender": model_1.rsquared - model_0.rsquared,
            "predictor_beta_plus_gender": model_1.params[predictor],
            "predictor_p_plus_gender": model_1.pvalues[predictor],
            "gender_p_plus_gender": model_1.pvalues.get("gender_mixed", np.nan),
            "r2_plus_gender_pitchdiff": model_2.rsquared,
            "predictor_beta_full": model_2.params[predictor],
            "predictor_p_full": model_2.pvalues[predictor],
            "gender_p_full": model_2.pvalues.get("gender_mixed", np.nan),
            "pitchdiff_p_full": model_2.pvalues.get("speaker_f0_mean_diff_hz", np.nan),
        })

    return pd.DataFrame(rows)


def load_inputs() -> pd.DataFrame:
    covariates = pd.read_csv(DATA / "covariates_dyad.csv")
    dyad = pd.read_csv(DATA / "dyad_level_dataset.csv")
    vocal = pd.read_csv(DATA / "vocal_alignment_dyad.csv")
    turns = pd.read_csv(DATA / "acoustic_turns.csv")

    cov_cols = ["pair_id", "gender_mixed", "age_mean"]
    df = dyad.merge(covariates[cov_cols], on="pair_id", how="left")
    df = add_connection_composite(df)
    df = add_vocal_summaries(df, vocal)
    df = add_pitch_difference(df, turns)
    df = add_turn_composite(df)
    return df


def main() -> None:
    ANALYSIS_OUTPUTS.mkdir(parents=True, exist_ok=True)
    df = load_inputs()

    group_table = gender_group_comparisons(df)
    regression_table = regression_confound_checks(df)

    group_path = ANALYSIS_OUTPUTS / "gender_mixed_analysis.csv"
    regression_path = ANALYSIS_OUTPUTS / "gender_pitch_confound_check.csv"
    group_table.to_csv(group_path, index=False)
    regression_table.to_csv(regression_path, index=False)

    print(f"Wrote {group_path}")
    print(f"Wrote {regression_path}")

    print("\nGender-mixed coding: 0=same gender, 1=mixed gender")
    focus = [
        "connection_composite",
        "speaker_f0_mean_diff_hz",
        "vocal_intensity_edge_index",
        "va_f0_proximity",
        "va_f0_edge_proximity",
        "va_int_edge_proximity",
    ]
    display_cols = [
        "variable", "n", "n_same", "n_mixed", "same_mean", "mixed_mean",
        "mixed_minus_same", "cohens_d", "welch_p", "perm_p",
    ]
    focus_table = group_table[group_table["variable"].isin(focus)][display_cols].copy()
    for col in focus_table.columns:
        if col not in {"variable", "n", "n_same", "n_mixed"}:
            focus_table[col] = focus_table[col].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
    print(focus_table.to_string(index=False))

    print("\nConnection composite regression confound checks:")
    reg = regression_table.copy()
    for col in reg.columns:
        if col not in {"predictor", "n"}:
            reg[col] = reg[col].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
    print(reg.to_string(index=False))


if __name__ == "__main__":
    main()
