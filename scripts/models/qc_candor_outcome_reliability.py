#!/usr/bin/env python3
"""
qc_candor_outcome_reliability.py
================================
Measurement-validity QC for the CANDOR survey connection outcome, run BEFORE
any incremental-validity test.

Motivation
----------
The processed-audio analysis found vocal alignment unrelated to survey
connection. Before that null can be interpreted theoretically, the OUTCOME has
to be shown to be a usable measurement. Three things can silently produce a
null on the outcome side:

  1. LOW INTERNAL CONSISTENCY
     The connection composite averages several items. If Cronbach's alpha is
     low, the composite is mostly item-specific noise.

  2. CEILING / RESTRICTED RANGE
     Paid Prolific participants rating a stranger after a friendly 30-minute
     conversation tend to pile up at the top of the scale. Restricted variance
     mechanically attenuates every correlation.

  3. WEAK PARTNER AGREEMENT
     The dyad-level outcome is the mean of two partners' ratings. That mean is
     most naturally interpreted as shared perception when the partners agree.
     The partner ICC quantifies that agreement. If ICC is near zero, the dyad
     mean is a weak measure of shared experience; this does not make prediction
     impossible, but it favors participant-level outcomes and dependence-aware
     models over a strong shared-connection interpretation.

The script also reports the reliability of the dyad mean (Spearman-Brown on the
partner ICC with k = 2) and the resulting attenuation ceiling. Pass
--vocal-reliability-csv to combine it with the predictor-side reliability from
scripts/models/validate_candor_intensity_edge.py for the joint ceiling.

Statistics
----------
Cronbach's alpha
    alpha = k/(k-1) * (1 - sum(item variances) / variance of the total score),
    computed on complete cases across the item set.

Partner ICC(1), one-way random effects (Shrout & Fleiss 1979)
    From a one-way ANOVA over conversations with k = 2 raters:
        ICC(1)   = (MSB - MSW) / (MSB + (k-1) * MSW)
    This is the proportion of variance attributable to the conversation.
    ICC(k) = (MSB - MSW) / MSB is the reliability of the k-rater mean and is
    reported as `reliability_of_dyad_mean`. Only conversations with exactly two
    non-missing ratings contribute.

Attenuation ceiling
    max observable r between the dyad-mean outcome and a perfectly reliable
    predictor is sqrt(reliability_of_dyad_mean).

Inputs
------
  --participant  candor_survey_participant.csv from build_candor_survey_outcomes.py
                 (one row per participant; needs convo_id, user_id and the
                 outcome item columns)
  --vocal-reliability-csv  (optional) intensity_edge_split_half_reliability.csv
                 from scripts/models/validate_candor_intensity_edge.py. That
                 script covers the PREDICTOR side of measurement validity
                 (pseudo-dyad and time-shift nulls, split-half reliability);
                 this script covers the OUTCOME side. Supplying it makes the
                 attenuation ceiling reflect error on both sides at once.

Outputs (timestamped run directory, never overwritten)
------------------------------------------------------
  item_descriptives.csv    n, mean, sd, skew, ceiling/floor rates per item
  composite_alpha.csv      Cronbach alpha for each composite definition
  partner_icc.csv          ICC(1), ICC(k), F, p per outcome and composite
  attenuation_ceiling.csv  max observable r for each composite
  qc_report.md             human-readable summary with verdicts
  manifest.json            inputs, config, git commit, row counts

Example (CARC)
--------------
  python scripts/models/qc_candor_outcome_reliability.py \
    --participant /scratch1/$USER/betterup_raw_media/outputs/candor_full_processed_resumable/candor_survey_participant.csv \
    --out-root    /scratch1/$USER/betterup_raw_media/outputs/outcome_qc
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT = Path(__file__).resolve().parents[2]

# The composite actually used as the primary outcome in the survey-vocal
# analysis (standardised mean of three items).
PRIMARY_COMPOSITE = ["i_like_you", "in_common", "how_enjoyable"]

# The broader composite defined in build_candor_survey_outcomes.py. Kept here
# so the two definitions can be compared side by side -- they currently differ,
# which is itself something to resolve before pre-registering.
EXPLORATORY_COMPOSITE = [
    "how_enjoyable", "i_like_you", "in_common",
    "i_felt_close_to_my_partner", "shared_reality", "responsive",
]

ALL_OUTCOMES = [
    "how_enjoyable", "i_like_you", "you_like_me", "in_common",
    "i_felt_close_to_my_partner", "i_would_like_to_become_friends",
    "interested_in_exchanging_contact_info", "shared_reality", "responsive",
]

ALPHA_ADEQUATE = 0.70
ALPHA_MARGINAL = 0.60
ICC_ADEQUATE = 0.30      # partner agreement good enough to call it dyadic
ICC_MARGINAL = 0.10
CEILING_CONCERN = 0.30   # >30% of ratings at the scale maximum


# --------------------------------------------------------------------------
def cronbach_alpha(frame: pd.DataFrame) -> tuple[float, int, int]:
    """Standardized alpha on complete cases, matching the z-item composite."""
    data = frame.apply(pd.to_numeric, errors="coerce").dropna()
    k = data.shape[1]
    n = data.shape[0]
    if k < 2 or n < 3:
        return float("nan"), n, k
    standard_deviation = data.std(axis=0, ddof=0)
    data = (data - data.mean(axis=0)) / standard_deviation.replace(0, np.nan)
    data = data.dropna(axis=1)
    k = data.shape[1]
    if k < 2:
        return float("nan"), n, k
    item_variances = data.var(axis=0, ddof=1).sum()
    total_variance = data.sum(axis=1).var(ddof=1)
    if total_variance <= 0:
        return float("nan"), n, k
    alpha = (k / (k - 1)) * (1 - item_variances / total_variance)
    return float(alpha), int(n), int(k)


def partner_icc(participant: pd.DataFrame, column: str,
                group_key: str = "convo_id") -> dict:
    """One-way random-effects ICC over conversations with exactly 2 raters.

    Returns ICC(1) (single-rater agreement) and ICC(k) (reliability of the
    two-partner mean), plus the ANOVA F test of whether conversations differ.
    """
    values = participant[[group_key, column]].dropna()
    sizes = values.groupby(group_key)[column].size()
    complete_groups = sizes[sizes == 2].index
    values = values[values[group_key].isin(complete_groups)]
    n_groups = len(complete_groups)
    result = {
        "outcome": column, "n_dyads_complete": int(n_groups), "k_raters": 2,
        "icc1": np.nan, "reliability_of_dyad_mean": np.nan,
        "f_stat": np.nan, "p_value": np.nan,
        "ms_between": np.nan, "ms_within": np.nan,
        "sd_between_dyad_mean": np.nan,
    }
    if n_groups < 10:
        result["note"] = "fewer than 10 conversations with two complete ratings"
        return result

    k = 2
    grand_mean = values[column].mean()
    group_means = values.groupby(group_key)[column].mean()

    ss_between = float((k * (group_means - grand_mean) ** 2).sum())
    merged = values.merge(group_means.rename("_gm"), on=group_key)
    ss_within = float(((merged[column] - merged["_gm"]) ** 2).sum())

    df_between = n_groups - 1
    df_within = n_groups * (k - 1)
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within if df_within > 0 else np.nan

    if not np.isfinite(ms_within) or ms_within <= 0:
        # Degenerate: partners agree exactly. ICC is 1 when there is real
        # between-dyad variance, and undefined when every value is identical.
        if np.isfinite(ms_between) and ms_between > 0:
            result.update({
                "icc1": 1.0, "reliability_of_dyad_mean": 1.0,
                "f_stat": np.inf, "p_value": 0.0,
                "ms_between": round(ms_between, 5), "ms_within": 0.0,
                "sd_between_dyad_mean": round(float(group_means.std(ddof=1)), 5),
                "note": "zero within-dyad variance; partners agree exactly",
            })
        else:
            result["note"] = "no variance in this outcome; ICC undefined"
        return result

    icc1 = (ms_between - ms_within) / (ms_between + (k - 1) * ms_within)
    icc_k = (ms_between - ms_within) / ms_between if ms_between > 0 else np.nan
    f_stat = ms_between / ms_within
    p_value = float(stats.f.sf(f_stat, df_between, df_within))

    result.update({
        "icc1": round(float(icc1), 4),
        # Negative ICC means partners disagree more than chance; the dyad mean
        # then has no reliable common variance, so clamp at 0 for downstream
        # attenuation use but keep the raw ICC visible above.
        "reliability_of_dyad_mean": round(float(max(icc_k, 0.0)), 4),
        "f_stat": round(float(f_stat), 4),
        "p_value": float(f"{p_value:.4g}"),
        "ms_between": round(ms_between, 5),
        "ms_within": round(ms_within, 5),
        "sd_between_dyad_mean": round(float(group_means.std(ddof=1)), 5),
        "note": "",
    })
    return result


def item_descriptives(participant: pd.DataFrame, items: list[str],
                      scale_max: float | None) -> pd.DataFrame:
    rows = []
    for item in items:
        if item not in participant.columns:
            continue
        series = pd.to_numeric(participant[item], errors="coerce").dropna()
        if series.empty:
            continue
        observed_max = float(series.max())
        observed_min = float(series.min())
        pct_at_scale_max = (
            float((series >= scale_max).mean()) if scale_max is not None else np.nan
        )
        rows.append({
            "item": item,
            "n": int(series.size),
            "mean": round(float(series.mean()), 4),
            "sd": round(float(series.std(ddof=1)), 4),
            "min": observed_min,
            "max": observed_max,
            "skew": round(float(stats.skew(series)), 4),
            "kurtosis": round(float(stats.kurtosis(series)), 4),
            "pct_at_observed_max": round(float((series >= observed_max).mean()), 4),
            "pct_at_scale_max": (
                round(pct_at_scale_max, 4) if np.isfinite(pct_at_scale_max) else np.nan
            ),
            "pct_at_bottom": round(float((series <= observed_min).mean()), 4),
            "ceiling_flag": (
                bool(pct_at_scale_max > CEILING_CONCERN)
                if scale_max is not None else None
            ),
        })
    return pd.DataFrame(rows)


def build_composite(participant: pd.DataFrame, items: list[str],
                    name: str, min_items: int) -> pd.Series:
    """Standardise each item across participants, then average.

    Matches build_candor_survey_outcomes.py: z-score at the participant level
    first so items on different scales contribute equally, then require a
    minimum number of non-missing items.
    """
    present = [i for i in items if i in participant.columns]
    if not present:
        return pd.Series(np.nan, index=participant.index, name=name)
    z = pd.DataFrame(index=participant.index)
    for item in present:
        series = pd.to_numeric(participant[item], errors="coerce")
        sd = series.std(ddof=0)
        z[item] = (series - series.mean()) / sd if np.isfinite(sd) and sd > 0 else np.nan
    composite = z.mean(axis=1, skipna=True)
    composite[z.notna().sum(axis=1) < min_items] = np.nan
    composite.name = name
    return composite


def read_vocal_reliability(path: str | None) -> dict[str, float]:
    """Load Spearman-Brown reliabilities from validate_candor_intensity_edge.py.

    Expects that script's ``intensity_edge_split_half_reliability.csv``, which
    has one row per metric (proximity / synchrony / composite) and a
    ``spearman_brown_reliability`` column. Returns {} if the file is absent or
    unreadable so the ceiling still reports the outcome-only case.
    """
    if not path:
        return {}
    file_path = Path(path)
    if not file_path.exists():
        print(f"[warn] vocal reliability file not found: {file_path}")
        return {}
    try:
        frame = pd.read_csv(file_path)
        return {
            str(row["metric"]): float(row["spearman_brown_reliability"])
            for _, row in frame.iterrows()
            if np.isfinite(pd.to_numeric(row.get("spearman_brown_reliability"),
                                         errors="coerce"))
        }
    except Exception as exc:  # noqa: BLE001 - QC helper must not crash the run
        print(f"[warn] could not parse {file_path}: {exc}")
        return {}


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(PROJECT),
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def _verdict(value: float, adequate: float, marginal: float) -> str:
    if not np.isfinite(value):
        return "UNKNOWN"
    if value >= adequate:
        return "ADEQUATE"
    if value >= marginal:
        return "MARGINAL"
    return "INADEQUATE"


# --------------------------------------------------------------------------
def write_report(out_dir: Path, descriptives: pd.DataFrame,
                 alpha_table: pd.DataFrame, icc_table: pd.DataFrame,
                 attenuation: pd.DataFrame, cfg: dict,
                 n_participants: int, n_dyads: int) -> Path:
    lines = [
        "# CANDOR outcome reliability QC",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Input: `{cfg['participant']}`",
        f"Survey rows: {cfg['n_response_rows']}; unique participants: {n_participants}; "
        f"repeat participants: {cfg['n_repeat_participants']}; conversations: {n_dyads}",
        "",
        "## Why this exists",
        "",
        "The vocal-alignment null is only interpretable if the outcome is a",
        "usable measurement. Low alpha, a ceiling effect, or near-zero partner",
        "agreement each produce a null for reasons that have nothing to do with",
        "the theory being tested.",
        "Because some participants occur in multiple conversations, ICC p-values",
        "that treat conversations as independent are descriptive. The formal",
        "outcome model must account for participant and conversation dependence.",
        "",
        "## 1. Item descriptives (ceiling / range restriction)",
        "",
        "| item | n | mean | sd | skew | % at observed max | % at scale max | ceiling flag |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for _, row in descriptives.iterrows():
        lines.append(
            f"| `{row['item']}` | {row['n']} | {row['mean']} | {row['sd']} | "
            f"{row['skew']} | {row['pct_at_observed_max']} | "
            f"{row['pct_at_scale_max']} | "
            f"{'UNKNOWN' if pd.isna(row['ceiling_flag']) else ('YES' if row['ceiling_flag'] else 'NO')} |"
        )

    lines += [
        "",
        "## 2. Internal consistency",
        "",
        "| composite | k items | n complete | Cronbach alpha | verdict |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for _, row in alpha_table.iterrows():
        lines.append(
            f"| {row['composite']} | {row['k_items']} | {row['n_complete']} | "
            f"{row['alpha']} | "
            f"{_verdict(row['alpha'], ALPHA_ADEQUATE, ALPHA_MARGINAL)} |"
        )

    lines += [
        "",
        "## 3. Partner agreement (is connection a property of the conversation?)",
        "",
        "| outcome | dyads | ICC(1) | reliability of dyad mean | F | p | verdict |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for _, row in icc_table.iterrows():
        lines.append(
            f"| `{row['outcome']}` | {row['n_dyads_complete']} | {row['icc1']} | "
            f"{row['reliability_of_dyad_mean']} | {row['f_stat']} | "
            f"{row['p_value']} | "
            f"{_verdict(row['icc1'], ICC_ADEQUATE, ICC_MARGINAL)} |"
        )

    lines += [
        "",
        "## 4. Attenuation ceiling",
        "",
        "Maximum correlation the dyad-mean outcome could show with a perfectly",
        "reliable conversation-level predictor: `sqrt(reliability of dyad mean)`.",
        "",
        "Rows naming a vocal metric use the split-half reliabilities from",
        "`validate_candor_intensity_edge.py`, giving the JOINT ceiling",
        "`sqrt(r_xx * r_yy)` with measurement error on both sides.",
        "This is a classical-test-theory diagnostic under independent-error",
        "assumptions, not a literal hard upper bound when those assumptions fail.",
        "",
        "| composite | vocal metric | r_xx (vocal) | r_yy (dyad mean) | max observable r |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for _, row in attenuation.iterrows():
        lines.append(
            f"| {row['composite']} | {row['vocal_metric']} | "
            f"{row['reliability_vocal']} | {row['reliability_of_dyad_mean']} | "
            f"{row['max_observable_r']} |"
        )

    lines += [
        "",
        "## How to read this",
        "",
        "- **ICC near zero is the most serious outcome.** It means the two",
        "  partners do not agree strongly about how the conversation went, so",
        "  the dyad mean is a weak measure of shared perception. This does not",
        "  make prediction mathematically impossible, but it weakens a shared-",
        "  connection interpretation. In that case prioritize an individual-",
        "  level model with participant and conversation dependence.",
        "- **A strong ceiling effect** restricts variance and attenuates every",
        "  correlation. Report the observed SD alongside the null.",
        "- Compare `max observable r` with the observed r = .011. If the",
        "  ceiling is small, frame the result as an equivalence test with a",
        "  pre-specified smallest effect size of interest, not as evidence of",
        "  absence.",
        "- The primary and exploratory composites differ in item content. Pick",
        "  ONE as primary and lock it before running any outcome model.",
        "",
    ]
    path = out_dir / "qc_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run(cfg: dict) -> Path:
    participant_path = Path(cfg["participant"])
    if not participant_path.exists():
        raise SystemExit(f"Input not found: {participant_path}")

    participant = pd.read_csv(participant_path)
    required = {"convo_id", "user_id"}
    missing = required - set(participant.columns)
    if missing:
        raise SystemExit(
            f"{participant_path} missing required columns: {sorted(missing)}")

    available = [c for c in ALL_OUTCOMES if c in participant.columns]
    if not available:
        raise SystemExit(
            f"{participant_path} contains none of the expected outcome items: "
            f"{ALL_OUTCOMES}")
    for column in available:
        participant[column] = pd.to_numeric(participant[column], errors="coerce")

    n_response_rows = len(participant)
    n_participants = int(participant["user_id"].nunique())
    n_dyads = participant["convo_id"].nunique()
    repeat_counts = participant.groupby("user_id")["convo_id"].nunique()
    n_repeat_participants = int((repeat_counts > 1).sum())
    cfg["n_response_rows"] = int(n_response_rows)
    cfg["n_repeat_participants"] = n_repeat_participants
    print(
        f"Loaded {n_response_rows:,} survey rows from {n_participants:,} unique "
        f"participants across {n_dyads:,} conversations"
    )
    print(f"Outcome items present: {available}")

    out_root = Path(cfg["out_root"])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_root / f"outcome_qc_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)
    print(f"Run directory: {out_dir}")

    print("[1/4] item descriptives ...")
    descriptives = item_descriptives(participant, available, cfg["scale_max"])
    descriptives.to_csv(out_dir / "item_descriptives.csv", index=False)

    print("[2/4] internal consistency ...")
    composites = {
        "primary_3item": (PRIMARY_COMPOSITE, 3),
        "exploratory_6item": (EXPLORATORY_COMPOSITE, 4),
    }
    alpha_rows = []
    for name, (items, required_items) in composites.items():
        present = [i for i in items if i in participant.columns]
        alpha, n_complete, k = cronbach_alpha(participant[present])
        alpha_rows.append({
            "composite": name,
            "items": ";".join(present),
            "k_items": k,
            "n_complete": n_complete,
            "alpha": round(alpha, 4) if np.isfinite(alpha) else np.nan,
            "required_items_for_score": required_items,
        })
    alpha_table = pd.DataFrame(alpha_rows)
    alpha_table.to_csv(out_dir / "composite_alpha.csv", index=False)

    print("[3/4] partner ICC ...")
    for name, (items, required_items) in composites.items():
        participant[f"composite_{name}"] = build_composite(
            participant, items, f"composite_{name}", required_items)
    icc_targets = available + [f"composite_{n}" for n in composites]
    icc_table = pd.DataFrame([partner_icc(participant, column)
                              for column in icc_targets])
    icc_table.to_csv(out_dir / "partner_icc.csv", index=False)

    print("[4/4] attenuation ceiling ...")
    # Optional: pair the outcome reliability with the vocal-measure reliability
    # produced by scripts/models/validate_candor_intensity_edge.py so the
    # ceiling reflects measurement error on BOTH sides.
    vocal_reliabilities = read_vocal_reliability(cfg["vocal_reliability_csv"])
    attenuation_rows = []
    for name in composites:
        row = icc_table[icc_table["outcome"] == f"composite_{name}"]
        r_yy = row["reliability_of_dyad_mean"].iloc[0] if len(row) else np.nan
        ceiling = (round(float(np.sqrt(r_yy)), 4)
                   if np.isfinite(r_yy) and r_yy > 0 else np.nan)
        base = {
            "composite": name,
            "vocal_metric": "(perfectly reliable predictor)",
            "reliability_vocal": 1.0,
            "reliability_of_dyad_mean": r_yy,
            "max_observable_r": ceiling,
        }
        attenuation_rows.append(base)
        for metric, r_xx in vocal_reliabilities.items():
            joint = (round(float(np.sqrt(max(r_xx, 0.0) * r_yy)), 4)
                     if np.isfinite(r_yy) and r_yy > 0 and np.isfinite(r_xx)
                     else np.nan)
            attenuation_rows.append({
                "composite": name,
                "vocal_metric": metric,
                "reliability_vocal": round(float(r_xx), 4),
                "reliability_of_dyad_mean": r_yy,
                "max_observable_r": joint,
            })
    attenuation = pd.DataFrame(attenuation_rows)
    attenuation.to_csv(out_dir / "attenuation_ceiling.csv", index=False)

    report = write_report(out_dir, descriptives, alpha_table, icc_table,
                          attenuation, cfg, n_participants, n_dyads)

    manifest = {
        "script": str(Path(__file__).resolve().relative_to(PROJECT)),
        "generated": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "python": sys.version.split()[0],
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "config": {k: (str(v) if isinstance(v, Path) else v)
                   for k, v in cfg.items()},
        "n_participants": int(n_participants),
        "n_response_rows": int(n_response_rows),
        "n_repeat_participants": n_repeat_participants,
        "n_conversations": int(n_dyads),
        "outcome_items_present": available,
        "outputs": sorted(p.name for p in out_dir.iterdir()),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                           encoding="utf-8")

    print("\n--- headline ---")
    for _, row in alpha_table.iterrows():
        print(f"  alpha  {row['composite']:20s} {row['alpha']}  "
              f"[{_verdict(row['alpha'], ALPHA_ADEQUATE, ALPHA_MARGINAL)}]")
    for name in composites:
        row = icc_table[icc_table["outcome"] == f"composite_{name}"]
        if len(row):
            icc = row["icc1"].iloc[0]
            print(f"  ICC(1) {name:20s} {icc}  "
                  f"[{_verdict(icc, ICC_ADEQUATE, ICC_MARGINAL)}]")
    for _, row in attenuation.iterrows():
        print(f"  ceiling {row['composite']:19s} x {row['vocal_metric']:28s} "
              f"max observable r = {row['max_observable_r']}")
    print(f"\nReport: {report}")
    return out_dir


def parse_args() -> dict:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--participant", required=True,
                   help="candor_survey_participant.csv from build_candor_survey_outcomes.py")
    p.add_argument("--out-root", required=True,
                   help="parent directory; a timestamped run dir is created inside")
    p.add_argument("--scale-max", type=float, default=None,
                   help="scale maximum for the ceiling calculation; "
                        "without it, ceiling flags are reported as UNKNOWN")
    p.add_argument("--vocal-reliability-csv", default=None,
                   help="intensity_edge_split_half_reliability.csv from "
                        "validate_candor_intensity_edge.py; if given, the "
                        "attenuation table reports the JOINT ceiling")
    a = p.parse_args()
    return {
        "participant": a.participant,
        "out_root": a.out_root,
        "scale_max": a.scale_max,
        "vocal_reliability_csv": a.vocal_reliability_csv,
    }


if __name__ == "__main__":
    run(parse_args())
