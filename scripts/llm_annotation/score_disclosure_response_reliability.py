#!/usr/bin/env python3
"""Score two independent disclosure-response coding sheets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix


DEPTHS = ["0_none", "1_low", "2_moderate", "3_high"]
RESPONSE_TYPES = [
    "validation_acknowledgment",
    "elaborative_question",
    "related_reciprocal_disclosure",
    "engaged_advice_interpretation",
    "minimal_continuer",
    "deflection_topic_shift",
    "unclear",
]
LINKAGE = ["yes", "no", "unclear"]
RESPONSIVE_TYPES = set(RESPONSE_TYPES[:4])


def _read(path: str, suffix: str) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False, dtype=str)
    required = {
        "sequence_id", "disclosure_depth", "response_type", "topically_linked"
    }
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(f"{path} missing columns: {sorted(missing)}")
    if frame["sequence_id"].duplicated().any():
        raise SystemExit(f"{path} has duplicate sequence_id")
    for column, allowed in (
        ("disclosure_depth", DEPTHS),
        ("response_type", RESPONSE_TYPES),
        ("topically_linked", LINKAGE),
    ):
        invalid = ~frame[column].isin(allowed)
        if invalid.any():
            examples = frame.loc[invalid, column].value_counts().head().to_dict()
            raise SystemExit(
                f"{path}: incomplete/invalid {column} values: {examples}"
            )
    keep = ["sequence_id", "disclosure_depth", "response_type", "topically_linked"]
    return frame[keep].rename(columns={column: f"{column}_{suffix}" for column in keep[1:]})


def _derive(frame: pd.DataFrame, suffix: str) -> None:
    depth = frame[f"disclosure_depth_{suffix}"]
    response = frame[f"response_type_{suffix}"]
    linked = frame[f"topically_linked_{suffix}"]
    frame[f"primary_disclosure_{suffix}"] = depth.isin(["2_moderate", "3_high"])
    frame[f"responsive_binary_{suffix}"] = (
        frame[f"primary_disclosure_{suffix}"]
        & response.isin(RESPONSIVE_TYPES)
        & linked.eq("yes")
    )


def run(args: argparse.Namespace) -> None:
    coder_a = _read(args.coder_a, "a")
    coder_b = _read(args.coder_b, "b")
    merged = coder_a.merge(coder_b, on="sequence_id", how="outer", indicator=True)
    if not merged["_merge"].eq("both").all():
        raise SystemExit("coder sheets do not contain identical sequence IDs")
    merged = merged.drop(columns="_merge")
    _derive(merged, "a")
    _derive(merged, "b")

    common_primary = merged[
        merged["primary_disclosure_a"] & merged["primary_disclosure_b"]
    ]
    metrics = {
        "n_sequences": len(merged),
        "n_common_primary_disclosures": len(common_primary),
        "depth_weighted_kappa": cohen_kappa_score(
            merged["disclosure_depth_a"],
            merged["disclosure_depth_b"],
            labels=DEPTHS,
            weights="quadratic",
        ),
        "primary_disclosure_kappa": cohen_kappa_score(
            merged["primary_disclosure_a"], merged["primary_disclosure_b"]
        ),
        "responsive_binary_full_sample_kappa": cohen_kappa_score(
            merged["responsive_binary_a"], merged["responsive_binary_b"]
        ),
        "responsive_binary_common_disclosure_kappa": (
            cohen_kappa_score(
                common_primary["responsive_binary_a"],
                common_primary["responsive_binary_b"],
            )
            if len(common_primary) >= 10 else None
        ),
        "response_type_common_disclosure_kappa": (
            cohen_kappa_score(
                common_primary["response_type_a"],
                common_primary["response_type_b"],
                labels=RESPONSE_TYPES,
            )
            if len(common_primary) >= 10 else None
        ),
    }
    responsive_kappa = metrics["responsive_binary_common_disclosure_kappa"]
    gate = (
        metrics["primary_disclosure_kappa"] >= 0.70
        and responsive_kappa is not None
        and responsive_kappa >= 0.70
    )
    metrics["human_reliability_gate"] = "PASS" if gate else "FAIL"

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    merged["any_disagreement"] = (
        merged["disclosure_depth_a"].ne(merged["disclosure_depth_b"])
        | merged["response_type_a"].ne(merged["response_type_b"])
        | merged["topically_linked_a"].ne(merged["topically_linked_b"])
    )
    merged.to_csv(out / "coder_comparison.csv", index=False)
    merged[merged["any_disagreement"]].to_csv(out / "disagreements.csv", index=False)
    (out / "reliability_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    confusion = confusion_matrix(
        merged["primary_disclosure_a"], merged["primary_disclosure_b"]
    )
    pd.DataFrame(
        confusion,
        index=["coder_a_not_primary", "coder_a_primary"],
        columns=["coder_b_not_primary", "coder_b_primary"],
    ).to_csv(out / "primary_disclosure_confusion.csv")
    print(json.dumps(metrics, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coder-a", required=True)
    parser.add_argument("--coder-b", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
