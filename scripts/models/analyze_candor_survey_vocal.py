#!/usr/bin/env python3
"""Test CANDOR vocal intensity-edge alignment against connection outcomes.

This is the survey-vocal stage of the independent CANDOR replication. The
analysis follows ``07_writeups/candor_feasibility_2026-07-11.md``:

- Primary outcome: z-mean of dyad-rated ``i_like_you``, ``in_common``, and
  ``how_enjoyable``.
- Primary predictor: z-mean of intensity edge proximity and synchrony.
- Primary test: zero-order Pearson association with a permutation p-value.
- Robustness: HC3 OLS and Freedman-Lane permutation inference after available
  pre-specified survey/demographic/technical covariates.
- Secondary outcomes and component metrics are reported with BH-FDR correction.

This script does not claim the planned turn-level incremental test. That test
requires a separately constructed CANDOR transcript-feature composite.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests


PRIMARY_OUTCOME_ITEMS = [
    "candor_i_like_you_mean",
    "candor_in_common_mean",
    "candor_how_enjoyable_mean",
]
PRIMARY_PREDICTOR_ITEMS = [
    "va_int_edge_proximity",
    "va_int_edge_synchrony",
]
SECONDARY_OUTCOMES = [
    "candor_i_like_you_mean",
    "candor_in_common_mean",
    "candor_how_enjoyable_mean",
    "candor_i_felt_close_to_my_partner_mean",
    "candor_i_would_like_to_become_friends_mean",
    "candor_shared_reality_mean",
    "candor_responsive_mean",
    "candor_connection_composite_z_exploratory_mean",
]
ROBUSTNESS_CONTROLS = [
    "candor_pre_affect_mean",
    "candor_pre_arousal_mean",
    "candor_technical_quality_min",
    "candor_conv_length_mean",
    "candor_age_mean",
    "same_reported_sex",
]


def zscore(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    std = numeric.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return pd.Series(np.nan, index=series.index, dtype=float)
    return (numeric - numeric.mean()) / std


def require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise SystemExit(f"Input table lacks required columns: {missing}")


def bool_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    mapped = series.astype(str).str.lower().map({"true": 1.0, "false": 0.0})
    numeric = pd.to_numeric(series, errors="coerce")
    return mapped.fillna(numeric)


def prepare_sample(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    require_columns(
        frame,
        ["dyad_id", "n_survey_respondents", "n_switch"]
        + PRIMARY_OUTCOME_ITEMS
        + PRIMARY_PREDICTOR_ITEMS,
    )
    sample = frame.copy()
    sample["same_reported_sex"] = bool_numeric(sample["same_reported_sex"])

    qc = {
        "n_input": int(len(sample)),
        "n_fewer_than_two_survey_respondents": int(
            (pd.to_numeric(sample["n_survey_respondents"], errors="coerce") < 2).sum()
        ),
        "n_below_min_switches": int(
            (pd.to_numeric(sample["n_switch"], errors="coerce") < 5).sum()
        ),
    }
    sample = sample[
        (pd.to_numeric(sample["n_survey_respondents"], errors="coerce") == 2)
        & (pd.to_numeric(sample["n_switch"], errors="coerce") >= 5)
    ].copy()

    primary_z = [zscore(sample[column]) for column in PRIMARY_OUTCOME_ITEMS]
    primary_matrix = pd.concat(primary_z, axis=1)
    sample["candor_connection_primary_z"] = primary_matrix.mean(axis=1)
    sample.loc[primary_matrix.notna().sum(axis=1) != len(primary_z),
               "candor_connection_primary_z"] = np.nan

    predictor_z = [zscore(sample[column]) for column in PRIMARY_PREDICTOR_ITEMS]
    predictor_matrix = pd.concat(predictor_z, axis=1)
    sample["vocal_intensity_edge_index"] = predictor_matrix.mean(axis=1)
    sample.loc[predictor_matrix.notna().sum(axis=1) != len(predictor_z),
               "vocal_intensity_edge_index"] = np.nan
    qc["n_after_design_qc"] = int(len(sample))
    qc["n_primary_complete"] = int(
        sample[["candor_connection_primary_z", "vocal_intensity_edge_index"]]
        .dropna().shape[0]
    )
    return sample, qc


def permutation_correlation(x: np.ndarray, y: np.ndarray, n_perm: int,
                            rng: np.random.Generator) -> float:
    observed = abs(float(stats.pearsonr(x, y).statistic))
    exceed = 0
    for _ in range(n_perm):
        exceed += abs(float(np.corrcoef(x, rng.permutation(y))[0, 1])) >= observed
    return (exceed + 1) / (n_perm + 1)


def correlation_row(frame: pd.DataFrame, predictor: str, outcome: str,
                    n_perm: int, rng: np.random.Generator) -> dict:
    complete = frame[[predictor, outcome]].dropna()
    if len(complete) < 4:
        return {"predictor": predictor, "outcome": outcome, "n": len(complete)}
    x = complete[predictor].to_numpy(float)
    y = complete[outcome].to_numpy(float)
    pearson = stats.pearsonr(x, y)
    spearman = stats.spearmanr(x, y)
    return {
        "predictor": predictor,
        "outcome": outcome,
        "n": len(complete),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "permutation_p": permutation_correlation(x, y, n_perm, rng),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def available_controls(frame: pd.DataFrame) -> tuple[list[str], dict]:
    included = []
    excluded = {}
    for column in ROBUSTNESS_CONTROLS:
        if column not in frame:
            excluded[column] = "missing"
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.notna().sum() < 20:
            excluded[column] = "fewer than 20 observations"
        elif numeric.nunique(dropna=True) < 2:
            excluded[column] = "zero variance"
        else:
            frame[column] = numeric
            included.append(column)
    return included, excluded


def incremental_model(frame: pd.DataFrame, predictor: str, outcome: str,
                      controls: list[str], n_perm: int,
                      rng: np.random.Generator) -> dict:
    columns = [outcome, predictor] + controls
    complete = frame[columns].dropna().copy()
    n = len(complete)
    if n <= len(controls) + 3:
        return {"predictor": predictor, "outcome": outcome, "n": n}

    y = zscore(complete[outcome]).to_numpy(float)
    x = zscore(complete[predictor]).to_numpy(float)
    control_matrix = np.column_stack(
        [zscore(complete[column]).to_numpy(float) for column in controls]
    ) if controls else np.empty((n, 0))

    reduced_x = sm.add_constant(control_matrix, has_constant="add")
    full_x = np.column_stack([reduced_x, x])
    reduced = sm.OLS(y, reduced_x).fit()
    full = sm.OLS(y, full_x).fit(cov_type="HC3")
    observed_delta = float(full.rsquared - reduced.rsquared)
    predictor_ci = np.asarray(full.conf_int())[-1]

    fitted = reduced.fittedvalues
    residual = reduced.resid
    exceed = 0
    for _ in range(n_perm):
        y_perm = fitted + rng.permutation(residual)
        reduced_perm = sm.OLS(y_perm, reduced_x).fit()
        full_perm = sm.OLS(y_perm, full_x).fit()
        delta_perm = full_perm.rsquared - reduced_perm.rsquared
        exceed += delta_perm >= observed_delta

    return {
        "predictor": predictor,
        "outcome": outcome,
        "n": n,
        "controls": ";".join(controls),
        "r2_reduced": float(reduced.rsquared),
        "r2_full": float(full.rsquared),
        "delta_r2": observed_delta,
        "standardized_beta": float(full.params[-1]),
        "hc3_se": float(full.bse[-1]),
        "hc3_p": float(full.pvalues[-1]),
        "ci95_low": float(predictor_ci[0]),
        "ci95_high": float(predictor_ci[1]),
        "freedman_lane_p": (exceed + 1) / (n_perm + 1),
    }


def add_fdr(frame: pd.DataFrame, p_column: str) -> pd.DataFrame:
    frame = frame.copy()
    if p_column not in frame:
        frame[p_column] = np.nan
    frame["fdr_p"] = np.nan
    valid = frame[p_column].notna()
    if valid.any():
        frame.loc[valid, "fdr_p"] = multipletests(
            frame.loc[valid, p_column], method="fdr_bh"
        )[1]
    return frame


def write_report(path: Path, qc: dict, primary: dict, robust: dict,
                 secondary: pd.DataFrame, controls: list[str],
                 excluded_controls: dict) -> None:
    def f(value, digits=3):
        return "NA" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"

    lines = [
        "# CANDOR survey-vocal analysis",
        "",
        "## Scope",
        "",
        "This stage tests the pre-specified intensity-edge association with CANDOR survey outcomes. "
        "It does not yet test incremental validity over transcript-derived turn-level features.",
        "",
        "## Sample QC",
        "",
        f"- Input conversations: {qc['n_input']}",
        f"- Excluded for fewer than two survey respondents: {qc['n_fewer_than_two_survey_respondents']}",
        f"- Excluded for fewer than five speaker switches: {qc['n_below_min_switches']}",
        f"- Complete primary cases: {qc['n_primary_complete']}",
        "",
        "## Primary test",
        "",
        "Outcome = z-mean of i_like_you, in_common, and how_enjoyable. "
        "Predictor = z-mean of intensity edge proximity and synchrony.",
        "",
        f"- n = {primary.get('n', 'NA')}",
        f"- Pearson r = {f(primary.get('pearson_r'))}",
        f"- Parametric p = {f(primary.get('pearson_p'), 4)}",
        f"- Permutation p = {f(primary.get('permutation_p'), 4)}",
        f"- Spearman rho = {f(primary.get('spearman_rho'))}, p = {f(primary.get('spearman_p'), 4)}",
        "",
        "## Covariate robustness",
        "",
        f"Controls retained: {', '.join(controls) if controls else 'none'}.",
        f"Controls omitted: {excluded_controls if excluded_controls else 'none'}.",
        "",
        f"- n = {robust.get('n', 'NA')}",
        f"- Standardized beta = {f(robust.get('standardized_beta'))}",
        f"- HC3 95% CI = [{f(robust.get('ci95_low'))}, {f(robust.get('ci95_high'))}]",
        f"- Delta R2 = {f(robust.get('delta_r2'))}",
        f"- HC3 p = {f(robust.get('hc3_p'), 4)}",
        f"- Freedman-Lane permutation p = {f(robust.get('freedman_lane_p'), 4)}",
        "",
        "## Secondary outcomes",
        "",
        "Secondary tests use BH-FDR across the listed survey outcomes. See "
        "`secondary_outcome_correlations.csv` for complete estimates.",
        "",
        "| Outcome | n | r | permutation p | FDR p |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in secondary.iterrows():
        lines.append(
            f"| {row['outcome']} | {int(row['n'])} | {f(row.get('pearson_r'))} | "
            f"{f(row.get('permutation_p'), 4)} | {f(row.get('fdr_p'), 4)} |"
        )
    lines.extend([
        "",
        "## Interpretation guardrail",
        "",
        "These are cross-sectional associations. Report association, not causation or mechanism. "
        "The full replication requires the planned transcript turn-level composite.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--n-perm", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_perm < 100:
        raise SystemExit("--n-perm must be at least 100")
    frame = pd.read_csv(args.input)
    sample, qc = prepare_sample(frame)
    rng = np.random.default_rng(args.seed)

    primary = correlation_row(
        sample, "vocal_intensity_edge_index", "candor_connection_primary_z",
        args.n_perm, rng,
    )
    controls, excluded_controls = available_controls(sample)
    robust = incremental_model(
        sample, "vocal_intensity_edge_index", "candor_connection_primary_z",
        controls, args.n_perm, rng,
    )

    secondary_rows = [
        correlation_row(sample, "vocal_intensity_edge_index", outcome,
                        args.n_perm, rng)
        for outcome in SECONDARY_OUTCOMES if outcome in sample
    ]
    secondary = add_fdr(pd.DataFrame(secondary_rows), "permutation_p")

    component_rows = []
    component_outcomes = ["candor_connection_primary_z"] + [
        outcome for outcome in SECONDARY_OUTCOMES if outcome in sample
    ]
    for predictor in PRIMARY_PREDICTOR_ITEMS:
        for outcome in component_outcomes:
            component_rows.append(
                correlation_row(sample, predictor, outcome, args.n_perm, rng)
            )
    components = add_fdr(pd.DataFrame(component_rows), "permutation_p")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([primary]).to_csv(args.out_dir / "primary_association.csv", index=False)
    pd.DataFrame([robust]).to_csv(args.out_dir / "primary_covariate_robustness.csv", index=False)
    secondary.to_csv(args.out_dir / "secondary_outcome_correlations.csv", index=False)
    components.to_csv(args.out_dir / "component_metric_correlations.csv", index=False)
    sample.to_csv(args.out_dir / "analysis_sample.csv", index=False)

    metadata = {
        "input": str(args.input),
        "seed": args.seed,
        "n_permutations": args.n_perm,
        "primary_outcome_items": PRIMARY_OUTCOME_ITEMS,
        "primary_predictor_items": PRIMARY_PREDICTOR_ITEMS,
        "robustness_controls_included": controls,
        "robustness_controls_excluded": excluded_controls,
        "qc": qc,
        "scope": "survey-vocal association; turn-level incremental model not yet run",
    }
    (args.out_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    write_report(
        args.out_dir / "candor_survey_vocal_report.md", qc, primary, robust,
        secondary, controls, excluded_controls,
    )

    print(f"Primary: n={primary.get('n')} r={primary.get('pearson_r', np.nan):+.3f} "
          f"permutation_p={primary.get('permutation_p', np.nan):.4f}")
    print(f"Robustness: beta={robust.get('standardized_beta', np.nan):+.3f} "
          f"delta_R2={robust.get('delta_r2', np.nan):.3f} "
          f"permutation_p={robust.get('freedman_lane_p', np.nan):.4f}")
    print(f"Results -> {args.out_dir}")


if __name__ == "__main__":
    main()
