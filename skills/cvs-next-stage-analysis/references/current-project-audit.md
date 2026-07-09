# Current Project Audit Reference

Use this reference for the CVS conversation repository at:

`/Users/Meihui/Downloads/sync/CVS/cvs_conversation`

## Known Repository Shape

The repository contains a reproducible dyadic conversation analysis pipeline:

- `scripts/text_features/`: semantic, sentiment, and structural features
- `scripts/llm_annotation/`: Claude and Qwen annotation workflows
- `scripts/acoustic_alignment/`: manual label alignment, acoustic extraction, vocal alignment
- `scripts/models/`: statistical models, covariates, robustness/confound checks
- `scripts/figures/`: publication and exploratory figures
- `04_data/`, `05_analysis_outputs/`, `06_figures/`: local generated/private outputs

Private recordings, transcripts, participant data, and generated outputs are intentionally not public.

## Current Observed Counts

From the latest workspace inspection:

- `04_data/outcomes.csv`: 25 rows
- `04_data/scientific_dyad_analysis_results.csv`: 47 rows
- `04_data/structural_dyad_analysis_mapped.csv`: 47 rows
- `05_analysis_outputs/llm_annotation_output/dyad_features.csv`: 47 rows
- `04_data/vocal_alignment_dyad.csv`: 46 rows
- `04_data/acoustic_turns.csv`: 5425 rows
- `05_analysis_outputs/multivariate_output/univariate_screening.csv`: 51 rows
- `05_analysis_outputs/multivariate_output/linear_model_summary.csv`: 3 rows
- `05_analysis_outputs/dissociation_results.csv`: 6 rows
- `05_analysis_outputs/gender_pitch_confound_check.csv`: 4 rows

These differences may be scientifically legitimate because the analysis unit changes across stages. The next-stage task is to explain them clearly.

## Current Key Outputs

`05_analysis_outputs/multivariate_output/analysis_metadata.json` reported:

- analysis date: `2026-06-30`
- final `n_dyads`: 23
- selected features:
  - `verbal_agreement_elaborated_n`
  - `questions_asym`
  - `hedging_asym`
- model performance:
  - `R2`: about 0.456
  - adjusted `R2`: about 0.370
  - leave-one-out CV `R2`: about 0.172

`05_analysis_outputs/dissociation_results.json` reported:

- `n`: 24
- primary outcome: `connection_composite`
- composite alpha: about 0.724
- equal-weight vs PCA outcome composite correlation: about 0.999
- vocal omnibus composite correlation: about 0.190
- vocal PC1 variance explained: about 0.272

## Next Audit Questions

Answer these before adding new analysis:

1. Why do feature tables have 46-47 rows while final dyad models use 23-24 rows?
2. Is each row a dyad, condition/session, turn, or model coefficient?
3. Which dyads are missing outcomes, acoustic features, LLM features, or covariates?
4. Are the same IDs represented consistently across `pair_id`, `dyad_id`, and `Dyad ID`?
5. Are generated files current relative to the scripts that produce them?
6. Which figures are confirmatory, which are exploratory, and which are older legacy outputs?
7. Are any private or identifiable files at risk of being shared or committed?

## Preferred Deliverables

For next-stage analysis, produce these artifacts when useful:

- a sample-size reconciliation table
- a missingness/exclusion memo
- a result-claim memo in conservative language
- a figure/table inventory
- a bilingual mentor summary in simple English and Chinese
