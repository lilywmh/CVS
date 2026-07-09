---
name: cvs-next-stage-analysis
description: Guide next-stage scientific analysis for the CVS conversation project. Use when planning, auditing, explaining, or extending the dyadic co-viewing conversation analyses, especially sample-size reconciliation, reproducibility checks, result interpretation, mentor-ready summaries, manuscript logic, and deciding whether a new model or figure is scientifically justified.
---

# CVS Next Stage Analysis

## Purpose

Use this skill to keep the CVS conversation project scientifically organized before adding new analyses. Prioritize checking whether the existing data, scripts, outputs, and claims align.

The core rule is: first reconcile the analysis pipeline and sample sizes, then interpret results, then decide whether new analyses are needed.

## Starting Workflow

1. Read the project-level docs first: `README.md`, `PIPELINE.md`, `scripts/README.md`, and `data/README.md`.
2. Inspect generated local outputs in `04_data/`, `05_analysis_outputs/`, and `06_figures/`.
3. Build a compact audit of row counts, dyad/session IDs, missingness, and output timestamps.
4. Explain any sample-size changes across tables before treating model results as final.
5. Separate confirmatory claims from exploratory checks.
6. Summarize conclusions in plain English and, when useful for the user's mentor, simple Chinese.

## Scientific Checking Logic

Treat sample-size mismatches as the first issue to understand, not automatically as an error. This project can have different units:

- session-level feature rows, often around two conditions per dyad
- dyad-level outcome rows
- turn-level acoustic rows
- final model rows after merging complete cases

For each major analysis, identify:

- what unit is being modeled: turn, session, dyad, or participant
- which ID columns define that unit, such as `pair_id`, `dyad_id`, `Dyad ID`, `condition`, or `Order`
- which input tables are merged
- which rows are dropped and why
- final `n` used in each model or figure

Do not recommend new models until this reconciliation is clear.

## Project-Building Logic

When helping plan the next stage, prefer this order:

1. Reproducibility audit: confirm inputs, outputs, row counts, timestamps, and script run order.
2. Sample inclusion memo: document which dyads/sessions enter each result and which are excluded.
3. Results memo: translate statistical outputs into conservative scientific claims.
4. Figure/table inventory: list which outputs are manuscript-ready and which are exploratory.
5. Robustness plan: add targeted checks only where the audit shows a real uncertainty.
6. Manuscript integration: write method/result language only after the above steps are stable.

## Communication Style

Use simple terms with the user. Avoid over-technical explanations unless requested.

Good short explanation:

> The next step is to check that all results are using the expected data. Some files have more rows than the final models, so we need to confirm which dyads were included, which were excluded, and why.

Simple Chinese version:

> 下一步不是马上加新模型，而是先检查数据和结果是否对得上。我们要确认每个分析用了哪些 dyad，哪些 dyad 因为缺数据被排除，以及最终样本量为什么是 23 或 24。这样后面的结论才可靠，也方便和导师解释。

## Result Interpretation Rules

- Describe small-sample findings cautiously.
- Use "associated with" rather than causal wording unless the design supports causality.
- Label LLM-derived annotations and vocal-alignment analyses according to their evidential status.
- If an analysis is exploratory, say so directly.
- Preserve the distinction between generated local outputs and public reproducible code.
- Flag privacy-sensitive files before suggesting sharing or committing data.

## Reference

For the current project state and concrete audit checks, read `references/current-project-audit.md`.
