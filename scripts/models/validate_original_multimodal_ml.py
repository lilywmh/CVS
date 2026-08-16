#!/usr/bin/env python3
"""Leakage-controlled nested ML validation for the original CVS dataset.

The analysis unit is one dyad. Outer leave-one-dyad-out evaluation estimates
generalization; all imputation, scaling, and hyperparameter selection occur on
the outer training data. Whole-pipeline outcome permutations repeat the nested
Ridge procedure. Elastic Net is an observed-data sparsity sensitivity analysis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "04_data"
ANALYSIS = PROJECT / "05_analysis_outputs"

OUTCOME_COLS = [
    "dyad_partner_eval_mean",
    "dyad_shared_reality_mean",
    "dyad_enjoyment_mean",
]

TRANSCRIPT_COLS = [
    "Order",
    "sem_semantic_similarity",
    "sem_sentiment_synchrony",
    "sem_mean_sentiment",
    "total_turns",
    "participation_gini",
    "turn_taking_density",
    "ttr_mean",
    "ttr_asym",
    "i_rate_mean",
    "i_rate_asym",
    "we_rate_mean",
    "we_rate_asym",
    "q_count_mean",
    "q_count_asym",
    "bc_rate_mean",
    "bc_rate_asym",
]

ACOUSTIC_COLS = [
    "va_f0_proximity",
    "va_f0_convergence",
    "va_f0_synchrony",
    "va_f0_edge_proximity",
    "va_f0_edge_convergence",
    "va_f0_edge_synchrony",
    "va_int_proximity",
    "va_int_convergence",
    "va_int_synchrony",
    "va_int_edge_proximity",
    "va_int_edge_convergence",
    "va_int_edge_synchrony",
    "va_rate_proximity",
    "va_rate_convergence",
    "va_rate_synchrony",
]

VIDEO_COLS = [
    "video_delta_wrist_speed_px_s",
    "video_delta_active_motion_rate",
    "video_speaker_arm_openness",
    "video_mutual_partner_orientation",
]

LLM_BASE_COLS = [
    "depth_deep_mean",
    "depth_deep_asym",
    "personal_mean",
    "personal_asym",
    "questions_mean",
    "questions_asym",
    "disclosure_mean",
    "disclosure_asym",
    "responsive_mean",
    "responsive_asym",
    "hedging_mean",
    "hedging_asym",
    "converge_mean",
    "converge_asym",
    "turn_asym",
]

RIDGE_GRID = np.logspace(-4, 4, 9)
ELASTIC_ALPHA_GRID = np.logspace(-3, 2, 6)
ELASTIC_L1_GRID = [0.1, 0.5, 0.9, 1.0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate_llm(path: Path) -> pd.DataFrame:
    turns = pd.read_csv(path)
    required = {"pair_id", "total_turns", *LLM_BASE_COLS,
                "verbal_agreement_elaborated_n"}
    missing = sorted(required - set(turns.columns))
    if missing:
        raise ValueError(f"LLM table lacks columns: {missing}")

    rows: list[dict[str, float]] = []
    for pair_id, group in turns.groupby("pair_id", sort=True):
        weights = pd.to_numeric(group["total_turns"], errors="coerce").fillna(0)
        denominator = float(weights.sum())
        row: dict[str, float] = {"pair_id": pair_id}
        for column in LLM_BASE_COLS:
            values = pd.to_numeric(group[column], errors="coerce")
            valid = values.notna() & (weights > 0)
            row[f"llm_{column}"] = (
                float(np.average(values[valid], weights=weights[valid]))
                if valid.any() else np.nan
            )
        row["llm_elaborated_agreement_rate"] = (
            float(pd.to_numeric(
                group["verbal_agreement_elaborated_n"], errors="coerce"
            ).fillna(0).sum() / denominator)
            if denominator > 0 else np.nan
        )
        rows.append(row)
    return pd.DataFrame(rows)


def load_analysis_table(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    dyad = pd.read_csv(args.dyad)
    multimodal = pd.read_csv(args.multimodal)
    llm = aggregate_llm(args.llm)

    for label, frame in [("dyad", dyad), ("multimodal", multimodal), ("llm", llm)]:
        if "pair_id" not in frame:
            raise ValueError(f"{label} table has no pair_id")
        if frame["pair_id"].isna().any() or frame["pair_id"].duplicated().any():
            raise ValueError(f"{label} table has null or duplicate pair_id")

    needed_dyad = ["pair_id", *OUTCOME_COLS, *TRANSCRIPT_COLS]
    missing = sorted(set(needed_dyad) - set(dyad.columns))
    if missing:
        raise ValueError(f"Dyad table lacks columns: {missing}")
    needed_multi = ["pair_id", *ACOUSTIC_COLS, *VIDEO_COLS]
    missing = sorted(set(needed_multi) - set(multimodal.columns))
    if missing:
        raise ValueError(f"Multimodal table lacks columns: {missing}")

    table = (
        dyad[needed_dyad]
        .merge(multimodal[needed_multi], on="pair_id", how="inner", validate="one_to_one")
        .merge(llm, on="pair_id", how="left", validate="one_to_one")
        .sort_values("pair_id")
        .reset_index(drop=True)
    )
    if len(table) != len(dyad):
        raise ValueError(f"Merge lost dyads: {len(dyad)} -> {len(table)}")
    if table[OUTCOME_COLS].isna().any().any():
        raise ValueError("Primary outcome components contain missing values")
    numeric = table.select_dtypes(include=[np.number])
    if np.isinf(numeric.to_numpy()).any():
        raise ValueError("Analysis table contains infinite numeric values")

    llm_cols = [f"llm_{column}" for column in LLM_BASE_COLS]
    llm_cols.append("llm_elaborated_agreement_rate")
    blocks = {
        "transcript": TRANSCRIPT_COLS,
        "transcript_acoustic": TRANSCRIPT_COLS + ACOUSTIC_COLS,
        "transcript_acoustic_video": TRANSCRIPT_COLS + ACOUSTIC_COLS + VIDEO_COLS,
        "full_exploratory_llm": TRANSCRIPT_COLS + ACOUSTIC_COLS + VIDEO_COLS + llm_cols,
    }
    for block, columns in blocks.items():
        constant = [column for column in columns if table[column].nunique(dropna=True) < 2]
        if constant:
            raise ValueError(f"{block} has constant features: {constant}")
    return table, blocks


def outcome_from_training_reference(
    train_components: np.ndarray,
    test_components: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(train_components, axis=0)
    sd = np.std(train_components, axis=0, ddof=0)
    if np.any(~np.isfinite(sd)) or np.any(sd == 0):
        raise ValueError("Outcome component has zero or invalid training-fold variance")
    return (
        np.mean((train_components - mean) / sd, axis=1),
        np.mean((test_components - mean) / sd, axis=1),
    )


def make_pipeline(algorithm: str, seed: int) -> tuple[Pipeline, dict[str, list[float]]]:
    if algorithm == "ridge":
        model = Ridge()
        grid = {"model__alpha": RIDGE_GRID.tolist()}
    elif algorithm == "elastic_net":
        model = ElasticNet(max_iter=100_000, random_state=seed)
        grid = {
            "model__alpha": ELASTIC_ALPHA_GRID.tolist(),
            "model__l1_ratio": ELASTIC_L1_GRID,
        }
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("model", model),
    ]), grid


def build_ridge_cache(X: pd.DataFrame, seed: int) -> list[dict]:
    """Cache only outcome-independent, fold-valid feature transformations."""
    def ridge_geometry(X_train: np.ndarray, X_test: np.ndarray) -> dict:
        kernel = X_train @ X_train.T
        eigenvalues, eigenvectors = np.linalg.eigh(kernel)
        return {
            "X_train": X_train,
            "cross_kernel": X_test @ X_train.T,
            "eigenvalues": np.maximum(eigenvalues, 0.0),
            "eigenvectors": eigenvectors,
        }

    values = X.to_numpy(dtype=float)
    cache = []
    for fold, (outer_train, outer_test) in enumerate(LeaveOneOut().split(values)):
        raw_train, raw_test = values[outer_train], values[outer_test]
        outer_imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        outer_train_imputed = outer_imputer.fit_transform(raw_train)
        outer_test_imputed = outer_imputer.transform(raw_test)
        outer_scaler = StandardScaler().fit(outer_train_imputed)
        inner_rows = []
        inner = KFold(n_splits=5, shuffle=True, random_state=seed + fold)
        for inner_train, inner_valid in inner.split(raw_train):
            imputer = SimpleImputer(strategy="median", keep_empty_features=True)
            train_imputed = imputer.fit_transform(raw_train[inner_train])
            valid_imputed = imputer.transform(raw_train[inner_valid])
            scaler = StandardScaler().fit(train_imputed)
            train_scaled = scaler.transform(train_imputed)
            valid_scaled = scaler.transform(valid_imputed)
            inner_rows.append({
                "train_index": inner_train,
                "valid_index": inner_valid,
                **ridge_geometry(train_scaled, valid_scaled),
            })
        outer_train_scaled = outer_scaler.transform(outer_train_imputed)
        outer_test_scaled = outer_scaler.transform(outer_test_imputed)
        cache.append({
            "fold": fold,
            "outer_train_index": outer_train,
            "outer_test_index": outer_test,
            **ridge_geometry(outer_train_scaled, outer_test_scaled),
            "inner": inner_rows,
        })
    return cache


def cached_ridge_fit_predict(
    geometry: dict,
    y_train: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit sklearn-equivalent Ridge in dual form using cached X geometry."""
    intercept = float(np.mean(y_train))
    centered = y_train - intercept
    eigenvectors = geometry["eigenvectors"]
    dual = eigenvectors @ (
        (eigenvectors.T @ centered) / (geometry["eigenvalues"] + alpha)
    )
    predictions = intercept + geometry["cross_kernel"] @ dual
    coefficients = geometry["X_train"].T @ dual
    return predictions, coefficients


def nested_ridge_cached(
    cache: list[dict],
    feature_names: list[str],
    outcome_components: np.ndarray,
    keep_coefficients: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_rows = []
    tuning_rows = []
    coefficient_rows = []
    for fold_data in cache:
        train_index = fold_data["outer_train_index"]
        test_index = fold_data["outer_test_index"]
        y_train, y_test = outcome_from_training_reference(
            outcome_components[train_index], outcome_components[test_index]
        )
        alpha_mse = []
        for alpha in RIDGE_GRID:
            squared_errors = []
            for inner in fold_data["inner"]:
                prediction, _ = cached_ridge_fit_predict(
                    inner, y_train[inner["train_index"]], float(alpha)
                )
                error = y_train[inner["valid_index"]] - prediction
                squared_errors.extend(np.square(error).tolist())
            alpha_mse.append(float(np.mean(squared_errors)))
        best_position = int(np.argmin(alpha_mse))
        best_alpha = float(RIDGE_GRID[best_position])
        fitted_prediction, fitted_coefficients = cached_ridge_fit_predict(
            fold_data, y_train, best_alpha
        )
        prediction_rows.append({
            "fold": int(fold_data["fold"]),
            "row_index": int(test_index[0]),
            "y_true": float(y_test[0]),
            "y_pred": float(fitted_prediction[0]),
            "baseline_pred": float(np.mean(y_train)),
        })
        tuning_rows.append({
            "fold": int(fold_data["fold"]),
            "row_index": int(test_index[0]),
            "best_inner_mse": alpha_mse[best_position],
            "alpha": best_alpha,
        })
        if keep_coefficients:
            for feature, coefficient in zip(feature_names, fitted_coefficients):
                coefficient_rows.append({
                    "fold": int(fold_data["fold"]),
                    "row_index": int(test_index[0]),
                    "feature": feature,
                    "coefficient": float(coefficient),
                    "selected_nonzero": bool(abs(coefficient) > 1e-10),
                })
    return (
        pd.DataFrame(prediction_rows),
        pd.DataFrame(tuning_rows),
        pd.DataFrame(coefficient_rows),
    )


def nested_predictions(
    X: pd.DataFrame,
    outcome_components: np.ndarray,
    algorithm: str,
    seed: int,
    keep_coefficients: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outer = LeaveOneOut()
    prediction_rows = []
    tuning_rows = []
    coefficient_rows = []

    for fold, (train_index, test_index) in enumerate(outer.split(X)):
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_train, y_test = outcome_from_training_reference(
            outcome_components[train_index], outcome_components[test_index]
        )
        inner = KFold(n_splits=5, shuffle=True, random_state=seed + fold)
        pipeline, grid = make_pipeline(algorithm, seed + fold)
        search = GridSearchCV(
            pipeline,
            grid,
            scoring="neg_mean_squared_error",
            cv=inner,
            n_jobs=1,
            refit=True,
            error_score="raise",
        )
        search.fit(X_train, y_train)
        prediction = float(search.predict(X_test)[0])
        prediction_rows.append({
            "fold": fold,
            "row_index": int(test_index[0]),
            "y_true": float(y_test[0]),
            "y_pred": prediction,
            "baseline_pred": float(np.mean(y_train)),
        })
        tuning_rows.append({
            "fold": fold,
            "row_index": int(test_index[0]),
            "best_inner_mse": float(-search.best_score_),
            **{key.replace("model__", ""): value
               for key, value in search.best_params_.items()},
        })
        if keep_coefficients:
            model = search.best_estimator_.named_steps["model"]
            for feature, coefficient in zip(X.columns, model.coef_):
                coefficient_rows.append({
                    "fold": fold,
                    "row_index": int(test_index[0]),
                    "feature": feature,
                    "coefficient": float(coefficient),
                    "selected_nonzero": bool(abs(coefficient) > 1e-10),
                })
    return (
        pd.DataFrame(prediction_rows),
        pd.DataFrame(tuning_rows),
        pd.DataFrame(coefficient_rows),
    )


def performance(predictions: pd.DataFrame) -> dict[str, float]:
    y_true = predictions["y_true"].to_numpy()
    y_pred = predictions["y_pred"].to_numpy()
    baseline = predictions["baseline_pred"].to_numpy()
    pearson = stats.pearsonr(y_true, y_pred)
    spearman = stats.spearmanr(y_true, y_pred)
    return {
        "n": int(len(predictions)),
        "cv_r2": float(r2_score(y_true, y_pred)),
        "cv_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "cv_mae": float(mean_absolute_error(y_true, y_pred)),
        "baseline_rmse": float(np.sqrt(mean_squared_error(y_true, baseline))),
        "prediction_pearson_r": float(pearson.statistic),
        "prediction_pearson_p": float(pearson.pvalue),
        "prediction_spearman_rho": float(spearman.statistic),
        "prediction_spearman_p": float(spearman.pvalue),
    }


def coefficient_summary(coefficients: pd.DataFrame) -> pd.DataFrame:
    if coefficients.empty:
        return pd.DataFrame()
    return (
        coefficients.groupby("feature", as_index=False)
        .agg(
            mean_coefficient=("coefficient", "mean"),
            median_coefficient=("coefficient", "median"),
            coefficient_sd=("coefficient", "std"),
            positive_fraction=("coefficient", lambda value: float((value > 0).mean())),
            nonzero_fraction=("selected_nonzero", "mean"),
        )
        .assign(mean_abs_coefficient=lambda frame: frame["mean_coefficient"].abs())
        .sort_values("mean_abs_coefficient", ascending=False)
        .reset_index(drop=True)
    )


def bh_fdr(values: pd.Series) -> pd.Series:
    p = values.to_numpy(dtype=float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty_like(ranked)
    adjusted[order] = np.minimum(ranked, 1.0)
    return pd.Series(adjusted, index=values.index)


def run_permutations(
    table: pd.DataFrame,
    blocks: dict[str, list[str]],
    ridge_caches: dict[str, list[dict]],
    args: argparse.Namespace,
) -> pd.DataFrame:
    checkpoint = args.out_dir / "ridge_whole_pipeline_permutations.csv"
    checkpoint_metadata = args.out_dir / "ridge_whole_pipeline_permutations.meta.json"
    signature = {
        "script_sha256": sha256(Path(__file__)),
        "seed": args.seed,
        "ridge_grid": RIDGE_GRID.tolist(),
        "outcome_columns": OUTCOME_COLS,
        "feature_blocks": blocks,
        "inputs": {
            str(path.resolve()): sha256(path)
            for path in [args.dyad, args.multimodal, args.llm]
        },
    }
    rows: list[dict] = []
    if checkpoint.exists() and not args.restart_permutations:
        if not checkpoint_metadata.exists():
            raise RuntimeError(
                f"Checkpoint metadata is missing: {checkpoint_metadata}. "
                "Use --restart-permutations to start a verified run."
            )
        previous_signature = json.loads(checkpoint_metadata.read_text(encoding="utf-8"))
        if previous_signature != signature:
            raise RuntimeError(
                "Permutation checkpoint does not match the current inputs/config. "
                "Use --restart-permutations to start a clean run."
            )
        existing = pd.read_csv(checkpoint)
        required = {"permutation", "feature_block", "cv_r2", "cv_rmse",
                    "prediction_pearson_r"}
        if not required.issubset(existing.columns):
            raise RuntimeError("Permutation checkpoint has an invalid schema")
        if existing.duplicated(["permutation", "feature_block"]).any():
            raise RuntimeError("Permutation checkpoint contains duplicate rows")
        unknown_blocks = set(existing["feature_block"]) - set(blocks)
        if unknown_blocks:
            raise RuntimeError(f"Permutation checkpoint has unknown blocks: {unknown_blocks}")
        rows = existing[existing["permutation"] < args.n_perm].to_dict("records")
        print(f"[resume] {len(existing)} permutation-block rows found")
    else:
        checkpoint_metadata.write_text(
            json.dumps(signature, indent=2) + "\n", encoding="utf-8"
        )
    completed = {
        (int(row["permutation"]), str(row["feature_block"])) for row in rows
    }
    outcomes = table[OUTCOME_COLS].to_numpy(dtype=float)

    for permutation in range(args.n_perm):
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, permutation, 7919]))
        permuted = outcomes[rng.permutation(len(outcomes))]
        for block, columns in blocks.items():
            if (permutation, block) in completed:
                continue
            predictions, _, _ = nested_ridge_cached(
                ridge_caches[block], columns, permuted, False
            )
            metrics = performance(predictions)
            rows.append({
                "permutation": permutation,
                "feature_block": block,
                "cv_r2": metrics["cv_r2"],
                "cv_rmse": metrics["cv_rmse"],
                "prediction_pearson_r": metrics["prediction_pearson_r"],
            })
        if (permutation + 1) % 10 == 0 or permutation + 1 == args.n_perm:
            frame = pd.DataFrame(rows).sort_values(["permutation", "feature_block"])
            temporary = checkpoint.with_suffix(".csv.tmp")
            frame.to_csv(temporary, index=False)
            temporary.replace(checkpoint)
            print(f"[permutation] {permutation + 1}/{args.n_perm}", flush=True)
    return pd.DataFrame(rows).sort_values(["permutation", "feature_block"])


def write_report(
    path: Path,
    results: pd.DataFrame,
    incremental: pd.DataFrame,
    feature_counts: dict[str, int],
    n_perm: int,
) -> None:
    def value(row: pd.Series, column: str, digits: int = 3) -> str:
        item = row.get(column, np.nan)
        return "NA" if not np.isfinite(item) else f"{item:.{digits}f}"

    lines = [
        "# Leakage-controlled multimodal ML validation",
        "",
        "Outer evaluation uses leave-one-dyad-out. Imputation, scaling, and "
        "hyperparameter selection are fitted without the held-out dyad. Ridge "
        f"permutation p-values repeat the complete nested pipeline ({n_perm} permutations).",
        "",
        "## Feature blocks",
        "",
    ]
    for block, count in feature_counts.items():
        lines.append(f"- `{block}`: {count} prespecified predictors")
    lines.extend([
        "",
        "The final block contains unvalidated LLM-derived proxies and is exploratory.",
        "",
        "## Nested cross-validation",
        "",
        "| Algorithm | Feature block | n | CV R2 | RMSE | Baseline RMSE | prediction r | permutation p | FDR p |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for _, row in results.iterrows():
        lines.append(
            f"| {row['algorithm']} | {row['feature_block']} | {int(row['n'])} | "
            f"{value(row, 'cv_r2')} | {value(row, 'cv_rmse')} | "
            f"{value(row, 'baseline_rmse')} | {value(row, 'prediction_pearson_r')} | "
            f"{value(row, 'permutation_p')} | {value(row, 'permutation_p_fdr')} |"
        )
    lines.extend([
        "",
        "## Incremental feature-block tests",
        "",
        "Positive RMSE gain means the expanded block predicted better. P-values "
        "compare that gain with paired whole-pipeline outcome permutations.",
        "",
        "| Added block | Base block | RMSE gain | permutation p | FDR p |",
        "| --- | --- | ---: | ---: | ---: |",
    ])
    for _, row in incremental.iterrows():
        lines.append(
            f"| {row['expanded_block']} | {row['base_block']} | "
            f"{value(row, 'observed_rmse_gain')} | "
            f"{value(row, 'permutation_p')} | {value(row, 'permutation_p_fdr')} |"
        )
    ridge = results[results["algorithm"] == "ridge"]
    best = ridge.sort_values("cv_rmse").iloc[0]
    lines.extend([
        "",
        "## Interpretation",
        "",
        f"The lowest-RMSE Ridge block was `{best['feature_block']}` "
        f"(CV R2={value(best, 'cv_r2')}, RMSE={value(best, 'cv_rmse')}, "
        f"whole-pipeline permutation p={value(best, 'permutation_p')}).",
        "",
        "Positive CV R2 and a small whole-pipeline permutation p would be needed "
        "to claim above-chance predictive feasibility. With 24 dyads, even that "
        "would remain exploratory rather than evidence of external generalization.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table, blocks = load_analysis_table(args)
    table.to_csv(args.out_dir / "analysis_table.csv", index=False)
    (args.out_dir / "feature_blocks.json").write_text(
        json.dumps(blocks, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[data] {len(table)} dyads; blocks="
          f"{ {name: len(columns) for name, columns in blocks.items()} }")
    ridge_caches = {
        block: build_ridge_cache(table[columns], args.seed)
        for block, columns in blocks.items()
    }

    observed_rows = []
    all_predictions = []
    all_tuning = []
    all_coefficients = []
    outcomes = table[OUTCOME_COLS].to_numpy(dtype=float)
    for algorithm in ["ridge", "elastic_net"]:
        for block, columns in blocks.items():
            print(f"[observed] {algorithm} / {block}", flush=True)
            if algorithm == "ridge":
                predictions, tuning, coefficients = nested_ridge_cached(
                    ridge_caches[block], columns, outcomes, True
                )
            else:
                predictions, tuning, coefficients = nested_predictions(
                    table[columns], outcomes, algorithm, args.seed, True
                )
            metrics = performance(predictions)
            observed_rows.append({"algorithm": algorithm, "feature_block": block, **metrics})
            predictions.insert(0, "algorithm", algorithm)
            predictions.insert(1, "feature_block", block)
            predictions["pair_id"] = table.iloc[predictions["row_index"]]["pair_id"].to_numpy()
            tuning.insert(0, "algorithm", algorithm)
            tuning.insert(1, "feature_block", block)
            coefficients.insert(0, "algorithm", algorithm)
            coefficients.insert(1, "feature_block", block)
            all_predictions.append(predictions)
            all_tuning.append(tuning)
            all_coefficients.append(coefficients)

    observed = pd.DataFrame(observed_rows)
    predictions = pd.concat(all_predictions, ignore_index=True)
    tuning = pd.concat(all_tuning, ignore_index=True)
    coefficients = pd.concat(all_coefficients, ignore_index=True)
    predictions.to_csv(args.out_dir / "outer_predictions.csv", index=False)
    tuning.to_csv(args.out_dir / "outer_tuning.csv", index=False)
    coefficients.to_csv(args.out_dir / "outer_coefficients.csv", index=False)
    summaries = []
    for (algorithm, block), group in coefficients.groupby(["algorithm", "feature_block"]):
        summary = coefficient_summary(group)
        summary.insert(0, "algorithm", algorithm)
        summary.insert(1, "feature_block", block)
        summaries.append(summary)
    pd.concat(summaries, ignore_index=True).to_csv(
        args.out_dir / "coefficient_stability.csv", index=False
    )

    permutations = run_permutations(table, blocks, ridge_caches, args)
    ridge_mask = observed["algorithm"] == "ridge"
    for index, row in observed[ridge_mask].iterrows():
        null = permutations[
            permutations["feature_block"] == row["feature_block"]
        ]
        observed.loc[index, "permutation_p"] = (
            (null["cv_rmse"] <= row["cv_rmse"]).sum() + 1
        ) / (len(null) + 1)
    observed.loc[ridge_mask, "permutation_p_fdr"] = bh_fdr(
        observed.loc[ridge_mask, "permutation_p"]
    )
    observed.to_csv(args.out_dir / "nested_ml_performance.csv", index=False)

    ridge_observed = observed[ridge_mask].set_index("feature_block")
    permutation_wide = permutations.pivot(
        index="permutation", columns="feature_block", values="cv_rmse"
    )
    comparisons = [
        ("transcript", "transcript_acoustic"),
        ("transcript_acoustic", "transcript_acoustic_video"),
        ("transcript_acoustic_video", "full_exploratory_llm"),
    ]
    incremental_rows = []
    for base, expanded in comparisons:
        observed_gain = float(
            ridge_observed.loc[base, "cv_rmse"]
            - ridge_observed.loc[expanded, "cv_rmse"]
        )
        null_gain = permutation_wide[base] - permutation_wide[expanded]
        incremental_rows.append({
            "base_block": base,
            "expanded_block": expanded,
            "observed_rmse_gain": observed_gain,
            "permutation_p": float(
                ((null_gain >= observed_gain).sum() + 1) / (len(null_gain) + 1)
            ),
        })
    incremental = pd.DataFrame(incremental_rows)
    incremental["permutation_p_fdr"] = bh_fdr(incremental["permutation_p"])
    incremental.to_csv(
        args.out_dir / "incremental_feature_block_tests.csv", index=False
    )

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": str(Path(__file__).relative_to(PROJECT)),
        "script_sha256": sha256(Path(__file__)),
        "seed": args.seed,
        "n_permutations": args.n_perm,
        "n_dyads": len(table),
        "outer_cv": "LeaveOneOut",
        "inner_cv": "KFold(n_splits=5, shuffle=True, random_state=seed+outer_fold)",
        "outcome_scaling": "fit on each outer training fold; applied to held-out dyad",
        "inputs": [
            {"path": str(path), "sha256": sha256(path)}
            for path in [args.dyad, args.multimodal, args.llm]
        ],
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    (args.out_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    write_report(
        args.out_dir / "nested_ml_validation_report.md",
        observed,
        incremental,
        {name: len(columns) for name, columns in blocks.items()},
        args.n_perm,
    )
    print(f"[done] {args.out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dyad", type=Path, default=DATA / "dyad_level_dataset.csv")
    parser.add_argument("--multimodal", type=Path, default=DATA / "multimodal_dyad_table.csv")
    parser.add_argument(
        "--llm", type=Path,
        default=ANALYSIS / "llm_annotation_output" / "dyad_features.csv",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=ANALYSIS / "final_ml_validation",
    )
    parser.add_argument("--n-perm", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--restart-permutations", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
