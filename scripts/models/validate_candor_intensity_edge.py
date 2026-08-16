#!/usr/bin/env python3
"""Validate the CANDOR intensity-edge vocal-alignment construct.

This stage must run before costly transcript annotation. It asks whether the
pre-specified intensity-edge metrics distinguish real adjacent floor transfers
from two null processes:

1. cross-conversation pseudo dyads with no shared participant; and
2. within-conversation circular time shifts that break local adjacency.

It also estimates odd/even split-half reliability. The composite used in the
survey analysis is standardized across conversations, so above-chance evidence
is evaluated for its raw proximity and synchrony components separately. The
split-half table additionally reports reliability of the standardized composite.

The input is the existing per-turn acoustic table. Audio is not re-extracted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from scripts.acoustic_alignment.compute_vocal_alignment_floor_transfer import (
    _adjacent_pairs,
    _expected_abs_difference,
    add_primary_floor_transfer_flags,
    classify_backchannels,
    zscore_within_speaker,
)


RELIABILITY_ADEQUATE = 0.60
RELIABILITY_MARGINAL = 0.40
SURROGATE_ALPHA = 0.05

def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 4 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    denominator = np.sqrt(
        np.dot(x_centered, x_centered) * np.dot(y_centered, y_centered)
    )
    return float(np.dot(x_centered, y_centered) / denominator)


def _proximity(
    offset: np.ndarray,
    onset: np.ndarray,
) -> float:
    if len(offset) < 4:
        return np.nan
    observed = float(np.mean(np.abs(offset - onset)))
    return float(_expected_abs_difference(offset, onset) - observed)


def _metrics(
    offset: np.ndarray,
    onset: np.ndarray,
) -> tuple[float, float]:
    n = min(len(offset), len(onset))
    if n < 4:
        return np.nan, np.nan
    offset_index = np.linspace(0, len(offset) - 1, n).round().astype(int)
    onset_index = np.linspace(0, len(onset) - 1, n).round().astype(int)
    offset = np.asarray(offset, dtype=float)[offset_index]
    onset = np.asarray(onset, dtype=float)[onset_index]
    valid = np.isfinite(offset) & np.isfinite(onset)
    offset, onset = offset[valid], onset[valid]
    if len(offset) < 4:
        return np.nan, np.nan
    return (
        _proximity(offset, onset),
        _safe_corr(offset, onset),
    )


def _edge_series(group: pd.DataFrame) -> dict | None:
    turns = group.sort_values("t_start", kind="mergesort").reset_index(drop=True).copy()
    eligible = turns["_pair_eligible"].fillna(False).astype(bool)
    turns.loc[~eligible, ["onset_int", "offset_int"]] = np.nan
    turns["onset_int_z"] = zscore_within_speaker(turns, "onset_int")
    turns["offset_int_z"] = zscore_within_speaker(turns, "offset_int")

    offset, onset, raw_offset, raw_onset = [], [], [], []
    source_speaker, target_speaker, direction, gaps = [], [], [], []
    for previous, current in _adjacent_pairs(turns):
        off, on = previous["offset_int_z"], current["onset_int_z"]
        if not np.isfinite(off) or not np.isfinite(on):
            continue
        offset.append(float(off))
        onset.append(float(on))
        raw_offset.append(float(previous["offset_int"]))
        raw_onset.append(float(current["onset_int"]))
        source_speaker.append(str(previous["speaker"]))
        target_speaker.append(str(current["speaker"]))
        direction.append(f"{previous['speaker']}->{current['speaker']}")
        gaps.append(float(current["t_start"] - previous["t_end"]))

    if len(offset) < 5:
        return None
    return {
        "offset": np.asarray(offset),
        "onset": np.asarray(onset),
        "raw_offset": np.asarray(raw_offset),
        "raw_onset": np.asarray(raw_onset),
        "source_speaker": np.asarray(source_speaker),
        "target_speaker": np.asarray(target_speaker),
        "direction": np.asarray(direction),
        "gap_s": np.asarray(gaps),
        "speakers": frozenset(group["_participant_id"].astype(str).unique()),
    }


def _normalize_by_speaker(values: np.ndarray, speakers: np.ndarray) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=float)
    for speaker in np.unique(speakers):
        mask = speakers == speaker
        subset = values[mask]
        sd = np.std(subset, ddof=0)
        if np.isfinite(sd) and sd > 0:
            out[mask] = (subset - np.mean(subset)) / sd
    return out


def _independent_half_metrics(
    series: dict, indices: np.ndarray
) -> tuple[float, float]:
    if len(indices) < 4:
        return np.nan, np.nan
    offset = _normalize_by_speaker(
        series["raw_offset"][indices], series["source_speaker"][indices]
    )
    onset = _normalize_by_speaker(
        series["raw_onset"][indices], series["target_speaker"][indices]
    )
    return _metrics(offset, onset)


def _balanced_odd_even(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Alternate transitions within direction so both halves contain A->B/B->A."""
    first, second = [], []
    for value in np.unique(direction):
        indices = np.flatnonzero(direction == value)
        first.extend(indices[::2])
        second.extend(indices[1::2])
    return np.asarray(sorted(first)), np.asarray(sorted(second))


def _zscore(values: pd.Series) -> pd.Series:
    sd = values.std(ddof=0)
    return (values - values.mean()) / sd if sd and np.isfinite(sd) else np.nan


def _p_values(observed: float, null: np.ndarray) -> tuple[float, float]:
    null = null[np.isfinite(null)]
    if not len(null):
        return np.nan, np.nan
    one_sided = (np.sum(null >= observed) + 1) / (len(null) + 1)
    centered = null - np.mean(null)
    two_sided = (np.sum(np.abs(centered) >= abs(observed - np.mean(null))) + 1) / (
        len(null) + 1
    )
    return float(one_sided), float(two_sided)


def _participant_disjoint_permutation(
    sessions: list[dict], rng: np.random.Generator, max_attempts: int = 10000
) -> np.ndarray:
    """Return a bijection where no session is paired with a shared participant."""
    n = len(sessions)
    for _ in range(max_attempts):
        assignment = rng.permutation(n)
        if all(
            i != int(j)
            and sessions[i]["speakers"].isdisjoint(sessions[int(j)]["speakers"])
            for i, j in enumerate(assignment)
        ):
            return assignment
    raise RuntimeError(
        "Could not construct a participant-disjoint pseudo-session permutation."
    )


def _direction_preserving_shift(
    session: dict, rng: np.random.Generator
) -> np.ndarray:
    """Circularly shift onsets within A->B/B->A direction separately."""
    shifted = session["onset"].copy()
    for direction in np.unique(session["direction"]):
        indices = np.flatnonzero(session["direction"] == direction)
        if len(indices) < 2:
            shifted[indices] = np.nan
            continue
        shift = int(rng.integers(1, len(indices)))
        shifted[indices] = np.roll(session["onset"][indices], shift)
    return shifted


def _checkpoint_fingerprint(sessions: list[dict], seed: int) -> str:
    payload = []
    for session in sessions:
        content = hashlib.sha256()
        content.update(np.asarray(session["offset"], dtype="<f8").tobytes())
        content.update(np.asarray(session["onset"], dtype="<f8").tobytes())
        content.update("\0".join(session["direction"].astype(str)).encode("utf-8"))
        payload.append({
            "dyad_id": str(session["dyad_id"]),
            "pair_id": str(session["pair_id"]),
            "condition": str(session["condition"]),
            "speakers": sorted(session["speakers"]),
            "n_edges": len(session["offset"]),
            "content_sha256": content.hexdigest(),
        })
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    encoded = json.dumps(
        {"sessions": payload, "seed": seed, "script_hash": script_hash},
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _null_distributions(
    sessions: list[dict],
    n_perm: int,
    seed: int,
    checkpoint: Path,
) -> pd.DataFrame:
    fingerprint = _checkpoint_fingerprint(sessions, seed)
    checkpoint_meta = checkpoint.with_suffix(".metadata.json")
    rows = []
    if checkpoint.exists():
        if not checkpoint_meta.exists():
            raise SystemExit(
                "Null checkpoint has no metadata; use --restart-null."
            )
        previous_meta = json.loads(checkpoint_meta.read_text(encoding="utf-8"))
        if previous_meta.get("fingerprint") != fingerprint:
            raise SystemExit(
                "Null checkpoint does not match current input/code/config; "
                "use --restart-null."
            )
        previous = pd.read_csv(checkpoint)
        rows = previous[previous["permutation"] < n_perm].to_dict("records")
        print(f"Resuming null simulation from {len(rows)} completed permutations.")
    completed = {int(row["permutation"]) for row in rows}

    for permutation in range(n_perm):
        if permutation in completed:
            continue
        rng = np.random.default_rng(np.random.SeedSequence([seed, permutation, 104729]))
        assignment = _participant_disjoint_permutation(sessions, rng)
        pseudo_proximity, pseudo_synchrony = [], []
        shifted_proximity, shifted_synchrony = [], []
        for i, session in enumerate(sessions):
            partner = sessions[int(assignment[i])]
            prox, sync = _metrics(session["offset"], partner["onset"])
            pseudo_proximity.append(prox)
            pseudo_synchrony.append(sync)

            prox, sync = _metrics(
                session["offset"], _direction_preserving_shift(session, rng)
            )
            shifted_proximity.append(prox)
            shifted_synchrony.append(sync)

        rows.append({
            "permutation": permutation,
            "pseudo_proximity_mean": np.nanmean(pseudo_proximity),
            "pseudo_synchrony_mean": np.nanmean(pseudo_synchrony),
            "time_shift_proximity_mean": np.nanmean(shifted_proximity),
            "time_shift_synchrony_mean": np.nanmean(shifted_synchrony),
        })
        if (permutation + 1) % 25 == 0 or permutation + 1 == n_perm:
            current = pd.DataFrame(rows).sort_values("permutation")
            temporary = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
            current.to_csv(temporary, index=False)
            temporary.replace(checkpoint)
            checkpoint_meta.write_text(
                json.dumps({"fingerprint": fingerprint}, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"Completed {len(current)} / {n_perm} null permutations.", flush=True)
    return pd.DataFrame(rows).sort_values("permutation").reset_index(drop=True)


def _reliability(session_table: pd.DataFrame) -> pd.DataFrame:
    table = session_table.dropna(subset=[
        "proximity_half_1", "proximity_half_2",
        "synchrony_half_1", "synchrony_half_2",
    ]).copy()
    table["composite_half_1"] = (
        _zscore(table["proximity_half_1"]) + _zscore(table["synchrony_half_1"])
    ) / 2
    table["composite_half_2"] = (
        _zscore(table["proximity_half_2"]) + _zscore(table["synchrony_half_2"])
    ) / 2

    rows = []
    for metric in ("proximity", "synchrony", "composite"):
        x, y = table[f"{metric}_half_1"], table[f"{metric}_half_2"]
        pearson = stats.pearsonr(x, y)
        spearman = stats.spearmanr(x, y)
        corrected = (
            min(1.0, 2 * pearson.statistic / (1 + pearson.statistic))
            if pearson.statistic > 0 else 0.0
        )
        rows.append({
            "metric": metric,
            "n_sessions": len(table),
            "split_half_pearson_r": pearson.statistic,
            "split_half_pearson_p": pearson.pvalue,
            "spearman_brown_reliability": corrected,
            "split_half_spearman_rho": spearman.statistic,
            "split_half_spearman_p": spearman.pvalue,
            "reliability_verdict": (
                "ADEQUATE" if corrected >= RELIABILITY_ADEQUATE else
                "MARGINAL" if corrected >= RELIABILITY_MARGINAL else
                "INADEQUATE"
            ),
        })
    return pd.DataFrame(rows)


def _write_report(
    path: Path,
    summary: pd.DataFrame,
    reliability: pd.DataFrame,
    transition_qc: pd.DataFrame,
    gate: dict,
    n_input: int,
    n_sessions: int,
    dataset_label: str,
) -> None:
    def markdown(frame: pd.DataFrame) -> str:
        columns = list(frame.columns)

        def render(value) -> str:
            if isinstance(value, (float, np.floating)):
                return "NA" if not np.isfinite(value) else f"{value:.4f}"
            return str(value)

        rows = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
        ]
        rows.extend(
            "| " + " | ".join(render(value) for value in row) + " |"
            for row in frame.itertuples(index=False, name=None)
        )
        return "\n".join(rows)

    lines = [
        f"# {dataset_label} intensity-edge measurement validation",
        "",
        "This stage validates the acoustic construct before any large-scale LLM annotation.",
        "",
        "## Sample",
        "",
        f"- Input dyads/conversations: {n_input}",
        f"- Valid sessions with at least five intensity-edge switches: {n_sessions}",
        "- Pseudo pairs cannot share a participant.",
        "- Time shifts are performed within speaker-transition direction.",
        "",
        "## Null tests",
        "",
        markdown(summary),
        "",
        "## Split-half reliability",
        "",
        markdown(reliability),
        "",
        "## Transition QC",
        "",
        markdown(transition_qc),
        "",
        "## Decision rule",
        "",
        "This is a prospective rule for the corrected measurement rerun, not an original preregistration.",
        f"- Surrogate alpha: {SURROGATE_ALPHA:.2f}; proximity and synchrony must each beat both the participant-disjoint pseudo null and the within-session time-shift null.",
        f"- Composite Spearman-Brown >= {RELIABILITY_ADEQUATE:.2f}: adequate; {RELIABILITY_MARGINAL:.2f}-{RELIABILITY_ADEQUATE - 0.01:.2f}: marginal; below {RELIABILITY_MARGINAL:.2f}: inadequate.",
        f"- Overall gate: **{gate['verdict']}**.",
        f"- Reason: {gate['reason']}",
        "",
        "Proceed to costly transcript annotation only after a PASS. MARGINAL remains exploratory and requires an explicit budget/scientific decision. A survey null is not interpretable as evidence against the theory when the acoustic construct fails this gate.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    turns = pd.read_csv(args.turns)
    required = {
        "dyad_id", "pair_id", "condition", "speaker", "t_start", "t_end",
        "onset_int", "offset_int", "text",
    }
    missing = required - set(turns.columns)
    if missing:
        raise SystemExit(f"{args.turns} missing columns: {sorted(missing)}")
    missing_participant = set(args.participant_cols) - set(turns.columns)
    if missing_participant:
        raise SystemExit(
            f"{args.turns} missing participant identity columns: "
            f"{sorted(missing_participant)}"
        )
    if turns[args.participant_cols].isna().any(axis=None):
        raise SystemExit(
            "Participant identity columns contain missing values: "
            f"{args.participant_cols}"
        )
    turns["_participant_id"] = turns[args.participant_cols].astype(str).agg(
        "::".join, axis=1
    )

    config = {
        "bc_max_words": args.bc_max_words,
        "min_turn_s": args.min_turn_s,
        "min_coverage": args.min_coverage,
    }
    turns = classify_backchannels(turns, args.bc_max_words)
    turns = add_primary_floor_transfer_flags(turns, config)
    turns["_pair_eligible"] = turns["is_primary_floor_transfer"]
    turns["_exclude_overlap"] = True

    keys = ["dyad_id", "pair_id", "condition"]
    sessions, rows, transition_rows = [], [], []
    for key, group in turns.groupby(keys, sort=True):
        ordered = group.sort_values("t_start", kind="mergesort").reset_index(drop=True)
        original_switches = 0
        eligible_switches = 0
        overlap_switches = 0
        for idx in range(1, len(ordered)):
            previous, current = ordered.iloc[idx - 1], ordered.iloc[idx]
            if previous["speaker"] == current["speaker"]:
                continue
            original_switches += 1
            if not (previous["_pair_eligible"] and current["_pair_eligible"]):
                continue
            eligible_switches += 1
            overlap_switches += current["t_start"] < previous["t_end"]

        series = _edge_series(group)
        if series is None:
            continue
        prox, sync = _metrics(series["offset"], series["onset"])
        half_1, half_2 = _balanced_odd_even(series["direction"])
        p1, s1 = _independent_half_metrics(series, half_1)
        p2, s2 = _independent_half_metrics(series, half_2)

        overlap_group = group.copy()
        overlap_group["_exclude_overlap"] = False
        overlap_series = _edge_series(overlap_group)
        overlap_prox, overlap_sync = (
            _metrics(overlap_series["offset"], overlap_series["onset"])
            if overlap_series is not None else (np.nan, np.nan)
        )
        series.update(dict(zip(keys, key)))
        sessions.append(series)
        rows.append({
            **dict(zip(keys, key)),
            "n_edge_switches": len(series["offset"]),
            "intensity_edge_proximity": prox,
            "intensity_edge_synchrony": sync,
            "overlap_inclusive_proximity": overlap_prox,
            "overlap_inclusive_synchrony": overlap_sync,
            "proximity_half_1": p1,
            "proximity_half_2": p2,
            "synchrony_half_1": s1,
            "synchrony_half_2": s2,
        })
        transition_rows.append({
            **dict(zip(keys, key)),
            "n_original_speaker_switches": original_switches,
            "n_switches_with_eligible_turns": eligible_switches,
            "n_overlapping_eligible_switches_excluded": overlap_switches,
            "n_nonoverlap_switches_with_missing_edges": max(
                eligible_switches - overlap_switches - len(series["offset"]), 0
            ),
            "n_primary_nonoverlap_edges": len(series["offset"]),
            "median_primary_gap_s": float(np.median(series["gap_s"])),
            "pct_primary_gaps_over_2s": float(np.mean(series["gap_s"] > 2.0)),
        })

    if len(sessions) < 3:
        raise SystemExit("Fewer than three valid sessions; cannot run validation.")
    session_table = pd.DataFrame(rows)
    transition_session = pd.DataFrame(transition_rows)
    transition_qc = pd.DataFrame([{
        "n_sessions": len(transition_session),
        "original_switches": int(transition_session["n_original_speaker_switches"].sum()),
        "eligible_turn_switches": int(transition_session["n_switches_with_eligible_turns"].sum()),
        "overlap_switches_excluded": int(transition_session["n_overlapping_eligible_switches_excluded"].sum()),
        "nonoverlap_switches_with_missing_edges": int(transition_session["n_nonoverlap_switches_with_missing_edges"].sum()),
        "primary_nonoverlap_edges": int(transition_session["n_primary_nonoverlap_edges"].sum()),
        "median_session_gap_s": float(transition_session["median_primary_gap_s"].median()),
        "median_session_pct_gaps_over_2s": float(transition_session["pct_primary_gaps_over_2s"].median()),
        "strict_vs_overlap_proximity_r": float(session_table[["intensity_edge_proximity", "overlap_inclusive_proximity"]].corr().iloc[0, 1]),
        "strict_vs_overlap_synchrony_r": float(session_table[["intensity_edge_synchrony", "overlap_inclusive_synchrony"]].corr().iloc[0, 1]),
    }])
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "intensity_edge_null_distributions.csv"
    if args.restart_null and checkpoint.exists():
        checkpoint.unlink()
        checkpoint.with_suffix(".metadata.json").unlink(missing_ok=True)
    session_table.to_csv(out / "intensity_edge_session_metrics.csv", index=False)
    nulls = _null_distributions(
        sessions, args.n_permutations, args.seed, checkpoint
    )

    summary_rows = []
    for metric in ("proximity", "synchrony"):
        observed = session_table[f"intensity_edge_{metric}"].mean()
        for null_name in ("pseudo", "time_shift"):
            null = nulls[f"{null_name}_{metric}_mean"].to_numpy()
            p_one, p_two = _p_values(observed, null)
            summary_rows.append({
                "metric": metric,
                "null": null_name,
                "real_mean": observed,
                "null_mean": np.nanmean(null),
                "mean_difference": observed - np.nanmean(null),
                "p_one_sided_real_gt_null": p_one,
                "p_two_sided": p_two,
                "passes_one_sided_alpha": bool(p_one < SURROGATE_ALPHA),
                "n_sessions": session_table[f"intensity_edge_{metric}"].notna().sum(),
                "n_permutations": len(null),
            })
    summary = pd.DataFrame(summary_rows)
    reliability = _reliability(session_table)
    component_pass = bool(summary["passes_one_sided_alpha"].all())
    composite_row = reliability[reliability["metric"] == "composite"]
    composite_reliability = (
        float(composite_row["spearman_brown_reliability"].iloc[0])
        if len(composite_row) else np.nan
    )
    if component_pass and composite_reliability >= RELIABILITY_ADEQUATE:
        gate = {"verdict": "PASS", "reason": "both components beat both nulls and composite reliability is adequate"}
    elif component_pass and composite_reliability >= RELIABILITY_MARGINAL:
        gate = {"verdict": "MARGINAL", "reason": "null tests pass but composite reliability is only marginal"}
    elif component_pass:
        gate = {
            "verdict": "FAIL",
            "reason": "all null tests pass but composite reliability is inadequate",
        }
    else:
        gate = {
            "verdict": "FAIL",
            "reason": "one or more component null tests fail",
        }

    nulls.to_csv(out / "intensity_edge_null_distributions.csv", index=False)
    summary.to_csv(out / "intensity_edge_null_summary.csv", index=False)
    reliability.to_csv(out / "intensity_edge_split_half_reliability.csv", index=False)
    transition_session.to_csv(out / "intensity_edge_transition_qc_by_session.csv", index=False)
    transition_qc.to_csv(out / "intensity_edge_transition_qc_summary.csv", index=False)
    metadata = {
        "turns": str(Path(args.turns).resolve()),
        "seed": args.seed,
        "n_permutations": args.n_permutations,
        "n_input_conversations": int(turns["dyad_id"].nunique()),
        "n_valid_sessions": len(session_table),
        "participant_identity_columns": args.participant_cols,
        "measurement_gate": gate,
        "acceptance_criteria": {
            "status": "prospective rule for the corrected measurement rerun; not an original preregistration",
            "surrogate_one_sided_alpha": SURROGATE_ALPHA,
            "all_four_component_null_tests_must_pass": True,
            "reliability_adequate": RELIABILITY_ADEQUATE,
            "reliability_marginal": RELIABILITY_MARGINAL,
        },
    }
    (out / "intensity_edge_validation_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    _write_report(
        out / "intensity_edge_measurement_report.md",
        summary,
        reliability,
        transition_qc,
        gate,
        metadata["n_input_conversations"],
        len(session_table),
        args.dataset_label,
    )
    print(summary.to_string(index=False))
    print("\nSplit-half reliability")
    print(reliability.to_string(index=False))
    print(f"\nWrote measurement-validation outputs to {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--turns", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--bc-max-words", type=int, default=2)
    parser.add_argument("--min-turn-s", type=float, default=0.30)
    parser.add_argument("--min-coverage", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-label", default="CANDOR")
    parser.add_argument(
        "--participant-cols", nargs="+", default=["speaker"],
        help=(
            "columns that jointly identify a person across sessions; CANDOR "
            "uses speaker UUIDs, while the local CVS data uses pair_id role"
        ),
    )
    parser.add_argument(
        "--restart-null", action="store_true",
        help="discard an existing null-distribution checkpoint",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
