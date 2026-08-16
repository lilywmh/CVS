#!/usr/bin/env python3
"""Positive-control replication of CANDOR turn interval -> partner enjoyment.

This is a pipeline validation, not a novel analysis. Reece et al. (2023)
reported that longer between-speaker gaps predicted lower partner enjoyment,
whereas overlap duration did not show the same association. The script checks
whether the locally prepared CANDOR transcript and survey tables recover that
direction before any new transcript/LLM construct is tested.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def _ordered(group: pd.DataFrame) -> pd.DataFrame:
    return group.sort_values("t_start", kind="mergesort").reset_index(drop=True)


def build_interval_table(turns: pd.DataFrame, exclude_backchannels: bool) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for convo_id, group in turns.groupby("dyad_id", sort=True):
        ordered = _ordered(group)
        by_speaker: dict[str, list[float]] = {}
        for index in range(1, len(ordered)):
            previous, current = ordered.iloc[index - 1], ordered.iloc[index]
            if str(previous["speaker"]) == str(current["speaker"]):
                continue
            if exclude_backchannels and "candor_turn_kind" in ordered:
                if previous.get("candor_turn_kind") == "backchannel" or current.get(
                    "candor_turn_kind"
                ) == "backchannel":
                    continue
            interval = float(current["t_start"] - previous["t_end"])
            if not np.isfinite(interval):
                continue
            by_speaker.setdefault(str(current["speaker"]), []).append(interval)

        for speaker, values_list in by_speaker.items():
            values = np.asarray(values_list, dtype=float)
            if len(values) < 5:
                continue
            gaps = values[values >= 0]
            overlaps = values[values < 0]
            rows.append(
                {
                    "convo_id": str(convo_id),
                    "interval_owner_id": speaker,
                    "n_intervals": len(values),
                    "mean_interval_s": float(np.mean(values)),
                    "median_interval_s": float(np.median(values)),
                    "is_mean_gapper": int(np.mean(values) >= 0),
                    "mean_gap_s": float(np.mean(gaps)) if len(gaps) else np.nan,
                    "mean_overlap_s": (
                        float(np.mean(overlaps)) if len(overlaps) else np.nan
                    ),
                    "overlap_fraction": float(np.mean(values < 0)),
                }
            )
    return pd.DataFrame(rows)


def merge_partner_enjoyment(
    intervals: pd.DataFrame, survey: pd.DataFrame
) -> pd.DataFrame:
    required = {"convo_id", "user_id", "partner_id", "how_enjoyable"}
    missing = required - set(survey.columns)
    if missing:
        raise SystemExit(f"survey table missing columns: {sorted(missing)}")

    survey = survey.copy()
    for column in ("convo_id", "user_id", "partner_id"):
        survey[column] = survey[column].astype(str)
    survey["how_enjoyable"] = pd.to_numeric(
        survey["how_enjoyable"], errors="coerce"
    )
    if survey.duplicated(["convo_id", "user_id"]).any():
        raise SystemExit("survey table has duplicate convo_id/user_id rows")

    owner = survey[["convo_id", "user_id", "partner_id"]].rename(
        columns={"user_id": "interval_owner_id"}
    )
    partner = survey[["convo_id", "user_id", "how_enjoyable"]].rename(
        columns={
            "user_id": "partner_id",
            "how_enjoyable": "partner_how_enjoyable",
        }
    )
    merged = intervals.merge(
        owner, on=["convo_id", "interval_owner_id"], how="left", validate="one_to_one"
    ).merge(partner, on=["convo_id", "partner_id"], how="left", validate="one_to_one")
    return merged


def _linear_combination(result, weights: dict[str, float]) -> dict[str, float]:
    names = list(result.params.index)
    vector = np.asarray([weights.get(name, 0.0) for name in names])
    estimate = float(vector @ result.params.to_numpy())
    variance = float(vector @ result.cov_params().to_numpy() @ vector)
    se = float(np.sqrt(max(variance, 0.0)))
    z = estimate / se if se > 0 else np.nan
    from scipy import stats

    p = float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan
    return {
        "estimate": estimate,
        "std_error": se,
        "ci_low": estimate - 1.96 * se,
        "ci_high": estimate + 1.96 * se,
        "p_value": p,
    }


def fit_models(table: pd.DataFrame, analysis: str) -> pd.DataFrame:
    data = table.dropna(
        subset=["partner_how_enjoyable", "mean_interval_s", "median_interval_s"]
    ).copy()
    formulas = {
        "mean_interval": "partner_how_enjoyable ~ mean_interval_s",
        "median_interval": "partner_how_enjoyable ~ median_interval_s",
        "gap_overlap_interaction": (
            "partner_how_enjoyable ~ mean_interval_s * is_mean_gapper"
        ),
    }
    rows: list[dict[str, object]] = []
    fitted = {}
    for model_name, formula in formulas.items():
        result = smf.ols(formula, data=data).fit(cov_type="HC3")
        fitted[model_name] = result
        for term in result.params.index:
            rows.append(
                {
                    "analysis": analysis,
                    "model": model_name,
                    "estimator": "OLS_HC3",
                    "term": term,
                    "estimate": result.params[term],
                    "std_error": result.bse[term],
                    "ci_low": result.conf_int().loc[term, 0],
                    "ci_high": result.conf_int().loc[term, 1],
                    "p_value": result.pvalues[term],
                    "n": int(result.nobs),
                }
            )

    interaction = fitted["gap_overlap_interaction"]
    for label, weights in {
        "overlapper_interval_slope": {"mean_interval_s": 1.0},
        "gapper_interval_slope": {
            "mean_interval_s": 1.0,
            "mean_interval_s:is_mean_gapper": 1.0,
        },
    }.items():
        values = _linear_combination(interaction, weights)
        rows.append(
            {
                "analysis": analysis,
                "model": "gap_overlap_interaction",
                "estimator": "OLS_HC3_linear_combination",
                "term": label,
                **values,
                "n": int(interaction.nobs),
            }
        )

    # Dependence-aware sensitivity: crossed random intercept approximations for
    # interval owner, partner, and conversation. Failure is recorded rather
    # than hiding the reproducible OLS positive control.
    try:
        mixed = smf.mixedlm(
            formulas["mean_interval"],
            data=data,
            groups=data["interval_owner_id"],
            re_formula="1",
            vc_formula={
                "partner": "0 + C(partner_id)",
                "conversation": "0 + C(convo_id)",
            },
        ).fit(reml=False, method="lbfgs", maxiter=500, disp=False)
        for term in mixed.fe_params.index:
            rows.append(
                {
                    "analysis": analysis,
                    "model": "mean_interval",
                    "estimator": "MixedLM_owner_partner_conversation_RI",
                    "term": term,
                    "estimate": mixed.fe_params[term],
                    "std_error": mixed.bse_fe[term],
                    "ci_low": mixed.conf_int().loc[term, 0],
                    "ci_high": mixed.conf_int().loc[term, 1],
                    "p_value": mixed.pvalues[term],
                    "n": int(mixed.nobs),
                }
            )
    except Exception as exc:  # pragma: no cover - optimizer/data dependent
        rows.append(
            {
                "analysis": analysis,
                "model": "mean_interval",
                "estimator": "MixedLM_owner_partner_conversation_RI",
                "term": "FIT_FAILED",
                "estimate": np.nan,
                "std_error": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "p_value": np.nan,
                "n": len(data),
                "note": str(exc),
            }
        )
    return pd.DataFrame(rows)


def _term(results: pd.DataFrame, analysis: str, model: str, term: str) -> pd.Series:
    row = results[
        results["analysis"].eq(analysis)
        & results["model"].eq(model)
        & results["estimator"].eq("OLS_HC3")
        & results["term"].eq(term)
    ]
    if len(row) != 1:
        raise RuntimeError(f"missing unique result: {analysis}/{model}/{term}")
    return row.iloc[0]


def write_report(
    path: Path, table: pd.DataFrame, results: pd.DataFrame, verdict: str
) -> None:
    primary_table = table[table["analysis"].eq("all_original_adjacent")]
    mean = _term(results, "all_original_adjacent", "mean_interval", "mean_interval_s")
    median = _term(
        results, "all_original_adjacent", "median_interval", "median_interval_s"
    )
    gap = results[
        results["analysis"].eq("all_original_adjacent")
        & results["term"].eq("gapper_interval_slope")
    ].iloc[0]
    lines = [
        "# CANDOR turn-interval positive control",
        "",
        "This reproduces an already published CANDOR association. It validates",
        "the local transcript-survey merge and model direction; it is not a new",
        "scientific contribution.",
        "",
        "## Sample",
        "",
        f"- Analysis rows: {primary_table['partner_how_enjoyable'].notna().sum()}",
        f"- Conversations: {primary_table['convo_id'].nunique()}",
        f"- Interval owners: {primary_table['interval_owner_id'].nunique()}",
        "",
        "## Primary checks (OLS HC3)",
        "",
        f"- Mean interval slope: {mean['estimate']:.4f}, 95% CI "
        f"[{mean['ci_low']:.4f}, {mean['ci_high']:.4f}], p={mean['p_value']:.4g}",
        f"- Median interval slope: {median['estimate']:.4f}, 95% CI "
        f"[{median['ci_low']:.4f}, {median['ci_high']:.4f}], p={median['p_value']:.4g}",
        f"- Gapper slope: {gap['estimate']:.4f}, 95% CI "
        f"[{gap['ci_low']:.4f}, {gap['ci_high']:.4f}], p={gap['p_value']:.4g}",
        f"- Positive-control verdict: **{verdict}**",
        "",
        "## Interpretation",
        "",
        "A successful check requires negative, p<.05 mean, median, and gapper",
        "slopes. Exact coefficients need not match the paper because this pipeline",
        "uses the prepared Backbiter transcript rather than the paper's complete",
        "10-ms communication-state reconstruction. The mixed-effects rows in",
        "`model_results.csv` are dependence-aware sensitivity estimates.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    turns = pd.read_csv(args.turns)
    required_turns = {"dyad_id", "speaker", "t_start", "t_end"}
    missing = required_turns - set(turns.columns)
    if missing:
        raise SystemExit(f"turn table missing columns: {sorted(missing)}")
    survey = pd.read_csv(args.survey)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tables = []
    results = []
    for analysis, exclude_backchannels in (
        ("all_original_adjacent", False),
        ("exclude_explicit_backchannel_transitions", True),
    ):
        interval = build_interval_table(turns, exclude_backchannels)
        merged = merge_partner_enjoyment(interval, survey)
        merged["analysis"] = analysis
        tables.append(merged)
        results.append(fit_models(merged, analysis))

    table = pd.concat(tables, ignore_index=True)
    model_results = pd.concat(results, ignore_index=True)
    table.to_csv(out / "participant_turn_interval_table.csv", index=False)
    model_results.to_csv(out / "model_results.csv", index=False)

    checks = [
        _term(model_results, "all_original_adjacent", "mean_interval", "mean_interval_s"),
        _term(
            model_results,
            "all_original_adjacent",
            "median_interval",
            "median_interval_s",
        ),
        model_results[
            model_results["analysis"].eq("all_original_adjacent")
            & model_results["term"].eq("gapper_interval_slope")
        ].iloc[0],
    ]
    reproduced = all(row["estimate"] < 0 and row["p_value"] < 0.05 for row in checks)
    verdict = "REPRODUCED" if reproduced else "NOT REPRODUCED"
    write_report(
        out / "turn_interval_positive_control_report.md", table, model_results, verdict
    )
    metadata = {
        "turns": str(Path(args.turns).resolve()),
        "survey": str(Path(args.survey).resolve()),
        "verdict": verdict,
        "known_result_not_novel": True,
        "primary_analysis": "all_original_adjacent",
    }
    (out / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print((out / "turn_interval_positive_control_report.md").read_text())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--turns", required=True)
    parser.add_argument("--survey", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
