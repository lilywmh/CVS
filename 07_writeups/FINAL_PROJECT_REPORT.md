# Detecting Coordination Is Not the Same as Measuring It

## A Reliability-First Multimodal Analysis of Dyadic Conversation

**Meihui (Lily) Wu**  
**Final Project Report | August 2026**

## Abstract

Computational studies of conversation often treat acoustic, linguistic, and visual coordination scores as stable properties of a pair and use them to predict social connection. This project tested that assumption before interpreting predictive models. The primary dataset contained 24 previously unacquainted dyads who completed 46 structured post-film discussions with corrected transcripts, shared-microphone audio, fixed-camera video, and postconversation surveys. A second analysis used 1,656 natural video conversations from the CANDOR corpus. The pipeline first tested whether an acoustic intensity-edge measure distinguished real turn transitions from participant-disjoint pseudo-pairs and within-session time shifts, then tested whether it reliably ranked sessions and dyads, and only then examined survey associations and out-of-sample prediction. Real transitions in the laboratory data showed greater proximity and synchrony than both null processes (all permutation *p* = .001), but split-half reliability was zero and cross-session test-retest correlations were negative. The dyad-level connection composite was also unreliable (Spearman-Brown reliability = .36), although participant-level multi-item scales were internally consistent. An in-sample acoustic increment of approximately .20 in *R*² did not translate into a reliable nested leave-one-dyad-out gain, and the corresponding CANDOR survey association was essentially zero. Pose-based video measures also added little predictive information. A corrected post hoc analysis identified shorter average turns as the strongest remaining correlate of partner evaluation (*r* = -.70), but this estimate is exploratory because it was discovered after the primary validation work, uses only 24 dyads, and relies on a marginally reliable dyad-level outcome. A CANDOR replication of the dynamic trajectory analysis found no replication of the laboratory turn-duration association and only negligible incremental dynamic effects. A semantic reflection-convergence test found weak evidence that discussion talk moved toward the partner's pre-discussion film interpretation (pseudo-dyad *p* = .032; within-dyad sign-flip *p* = .184). The central result is methodological: detecting coordination in aggregate does not establish a reliable individual-difference measure or a generalizable predictor.

## 1. Research Question

Conversation partners coordinate across multiple channels. They may adjust timing, pitch, intensity, wording, posture, and movement in response to one another. Acoustic-prosodic entrainment research commonly separates local proximity, convergence over time, and synchrony across successive turns (Levitan & Hirschberg, 2011). A practical pipeline may therefore extract alignment scores and test whether pairs with higher scores also report stronger social connection.

That workflow contains an important hidden assumption: a measure that detects coordination across many conversations must also rank individual pairs reliably. These are different statistical questions. A population mean can differ from a null distribution even when the relative ordering of sessions is unstable. If the ordering is mostly measurement noise, correlation and machine-learning analyses can still produce attractive in-sample results, especially with many predictors and a small sample.

This project therefore adopted a reliability-first question:

> Can multimodal conversational features be extracted reproducibly, shown to represent more than a null pairing process, measured reliably at the intended level, and used to explain or predict social connection without leakage?

The work is framed as a pipeline-validation study and exploratory empirical pilot, not as a definitive test of whether coordination causes social connection.

## 2. Data

### 2.1 Laboratory co-viewing study

The analyzed sample comprised 24 dyads and 46 discussion sessions. Dyad members were previously unacquainted. Each pair watched two short animated films, *Piper* and *Cloudy*, in counterbalanced order and discussed each film using the same five prompts. Twenty-two dyads contributed both discussions; two contributed one usable discussion. Audio was recorded through one shared microphone at 16 kHz, and a fixed camera captured both participants.

The analysis table contained 5,425 manually corrected speaker turns. WhisperX word timestamps were aligned to the corrected turns. Postconversation outcomes were available for 25 dyads, but only 24 had corresponding analyzed transcript and acoustic data. The primary dyad-level connection composite was the equal-weight mean of standardized partner evaluation, shared reality, and enjoyment. Equal weighting was used instead of sample-estimated principal-component weights to limit overfitting.

### 2.2 CANDOR corpus

CANDOR is a public multimodal corpus of 1,656 natural English-language video conversations between strangers, including transcripts, separate audio streams, video, and extensive pre- and postconversation surveys (Reece et al., 2023). The processed-audio analysis used all 1,656 conversations for feature extraction and 1,653 complete cases for the primary survey association. CANDOR complemented the laboratory study with much greater sample size and separate speaker audio, while the laboratory study uniquely provided two repeated discussions for most dyads.

![Figure 1. Multimodal analysis workflow.](/Users/Meihui/Downloads/sync/CVS/cvs_conversation/06_figures/chi_report/fig1_methods_workflow.png)

**Figure 1.** Multimodal analysis workflow. Features were aggregated only after modality-specific quality checks and were evaluated with association tests, CANDOR analyses, and nested leave-one-dyad-out prediction.

## 3. Methods

### 3.1 Transcript and outcome processing

Corrected speaker turns were summarized using prespecified structural and semantic variables, including turn count and duration, participation balance, lexical diversity, question and backchannel rates, semantic similarity, and sentiment coordination. A transcript-level principal component was used in the original hierarchical models. LLM-generated dynamics labels were retained only as exploratory proxies because the planned human validation set was not completed.

Survey scoring was reconstructed from the item-level source file and verified against the archived dyad means and within-dyad differences to a maximum absolute error of 8.9 x 10⁻¹⁶. Partner evaluation used 13 items with one reverse-coded item; shared reality used eight items; enjoyment and solo experience were single items. Internal consistency and agreement between the two members of each dyad were evaluated separately because consistency within a person's scale does not establish agreement between partners.

### 3.2 Acoustic feature extraction

Because the laboratory audio used a shared microphone, simultaneous speaker-specific acoustics could not be isolated. The primary analysis therefore focused on nonoverlapping floor transfers. For each transfer, the last 500 ms of the outgoing speaker's turn and first 500 ms of the incoming speaker's turn were extracted. Fundamental frequency, intensity, and speaking rate were measured with Praat/Parselmouth. Invalid timestamps, low-coverage or very short turns, overlaps, and minimal backchannels were excluded from the primary transfer analysis.

The focal intensity-edge index combined two standardized components:

- **Proximity:** how similar the two speakers' intensity values were at adjacent turn edges.
- **Synchrony:** whether deviations from each speaker's own typical intensity moved in the same direction at the handoff.

Continuous acoustic features were normalized within speaker before pair-level summaries were computed.

### 3.3 Measurement-validation sequence

The acoustic analysis followed three ordered stages. The first two stages were computed without using survey outcomes.

1. **Detectability.** Real adjacent transfers were compared with participant-disjoint pseudo-pairs and direction-preserving time shifts within sessions.
2. **Reliability.** Odd and even transfers provided within-session split halves, corrected with the Spearman-Brown formula. The two film discussions provided cross-session test-retest correlations for 22 dyads.
3. **Association and prediction.** Survey analyses were interpreted only in light of the first two stages.

The prospective reliability rule was Spearman-Brown reliability ≥ .60 as adequate, .40-.59 as marginal, and below .40 as inadequate. This rule was adopted during the validation rerun rather than preregistered before the original exploratory analyses.

### 3.4 Video features

Fixed-camera video was processed with MediaPipe Pose and FaceMesh-derived proxies. Participant mapping followed the verified seating convention: participant A appeared on the video right and participant B on the video left. Primary variables included speaker-minus-listener wrist speed and active-motion rate, arm openness while speaking, and mutual partner-orientation rate. The last variable is a head-orientation proxy, not calibrated eye gaze. Hand-landmark measures were not used as primary predictors because detection varied substantially across sessions.

### 3.5 Statistical analysis

Original dyad-level association models entered film order and the transcript composite before adding acoustic or video blocks. Incremental variance was summarized as change in *R*². Acoustic sensitivity used Freedman-Lane residual permutation and leave-one-dyad influence checks. Video tests used permutation inference and Benjamini-Hochberg false-discovery-rate correction (Benjamini & Hochberg, 1995; Freedman & Lane, 1983). A post hoc dynamic-trajectory analysis divided each discussion into five question-windows and tested whether within-conversation slopes, early-late shifts, and variability predicted partner evaluation beyond static session means. A reflection-convergence analysis tested whether each speaker's discussion text moved toward their partner's pre-discussion written film interpretation, using sentence-transformer embeddings (Reimers & Gurevych, 2019) and a partner-specific attraction index benchmarked against same-film pseudo-partner reflections.

Predictive validation used leave-one-dyad-out outer folds. Missing-value imputation, scaling, and hyperparameter tuning occurred inside each training fold. Ridge and Elastic Net models were evaluated across four cumulative blocks: transcript, transcript plus acoustic, transcript plus acoustic and video, and a final exploratory block adding LLM proxies. Five hundred whole-pipeline outcome permutations repeated the nested procedure.

A later participant-level analysis corrected an initial actor-partner model. It tested the partner-minus-actor contrast directly rather than comparing separate significance levels (Gelman & Stern, 2006). Pair-constant between-dyad associations were estimated at the dyad level; within-dyad contrasts used participant rows and member-swap sign-flip tests. Because this analysis was developed after reviewing earlier results, it is reported as post hoc exploration.

## 4. Results

### 4.1 Pipeline checks

The local CANDOR implementation reproduced the published direction of the turn-interval association with enjoyment. Across 1,655 conversations and 3,255 participant-level rows, mean interval, median interval, and gapper coefficients were negative: mean slope = -.577, 95% CI [-.789, -.365], *p* = 1.0 × 10⁻⁷; median slope = -1.477, *p* = 8.9 × 10⁻⁸; and gapper slope = -.481, *p* = .011. This was a positive control for the transcript-survey merge and model direction, not a new contribution and not a validation of acoustic extraction.

In the laboratory data, real intensity-edge proximity averaged .109 compared with .006 for pseudo-pairs and .008 for time shifts. Synchrony averaged .184 compared with .004 and .012, respectively. All four one-sided permutation tests gave *p* = .001. The pipeline therefore detected local adjacent coordination beyond both implemented null processes.

### 4.2 Detectability did not produce reliable rankings

The same acoustic indices failed the reliability stage. Across 43 eligible sessions, split-half correlations were -.178 for proximity, approximately zero for synchrony, and -.088 for the composite. Spearman-Brown reliability was zero for all three. Across the two films, dyad-level test-retest correlations were also negative: -.271 for proximity, -.230 for synchrony, and -.266 for the composite (*n* = 22; all *p* > .20).

This pattern was not inevitable for every behavioral measure in the dataset. Selected video behaviors showed cross-session correlations of .52-.79, and simple transcript behaviors later showed Spearman-Brown estimates above .65 for most variables. The acoustic failure therefore cannot be explained only by having two short discussions or a small sample.

![Figure 2. Acoustic measurement validation.](/Users/Meihui/Downloads/sync/CVS/cvs_conversation/06_figures/chi_report/fig2_acoustic_measurement_validation.png)

**Figure 2.** Measurement validation separates aggregate detectability from dyad-level reliability. Real transfers exceeded both null distributions, while split-half correlations were near zero or negative.

The CANDOR measurement audit produced a similar warning under a different recording design. Its intensity-edge split-half reliability was approximately .19, below the prespecified threshold. Separate audio tracks also exposed a major operational difference: 68% of candidate CANDOR handoffs involved overlap and were excluded from the strict transfer definition, whereas the shared-microphone laboratory representation identified no overlap exclusions. Strict and overlap-inclusive CANDOR scores correlated only about .50. Thus, the two datasets did not operationalize exactly the same set of interaction events.

### 4.3 The dyad-level outcome was also weakly supported

At the participant level, partner evaluation and shared reality had good internal consistency (Cronbach's alpha = .891 and .790). Partner agreement was much weaker. The intraclass correlation was .360 for partner evaluation, .213 for enjoyment, .120 for shared reality, and -.226 for solo experience. The corresponding reliability of a two-person dyad mean was .529, .351, .214, and 0. The equal-weight connection composite had reliability .359.

These results do not imply that participants answered carelessly. They show that two people in the same conversation can have internally coherent but different experiences. Partner evaluation was the best dyad-level outcome, but its reliability was only marginal and its confidence interval was wide.

### 4.4 In-sample association did not establish prediction

The original transcript-plus-order model explained *R*² = .260 of the connection composite. Adding the intensity-edge index increased in-sample variance explained by .198; a 50,000-iteration Freedman-Lane test gave *p* = .012. The increment remained positive in every leave-one-dyad omission and was .231 in the 22 complete dyads. This association is reproducible from the current local inputs, but it is not evidence that the acoustic index is a reliable general predictor: both predictor and outcome failed their measurement gates.

Nested prediction made the gap clear. Ridge leave-one-dyad-out *R*² was .004 for transcript only, .076 after adding acoustic features, .050 after adding video, and .130 after adding exploratory LLM proxies. The minimum FDR-adjusted whole-pipeline *p* across blocks was .096. Elastic Net *R*² was negative for every block (-.081 to -.252 for the three expanded models). Paired whole-pipeline tests found no reliable incremental RMSE gain from acoustic (*p* = .208), video (*p* = .731), or LLM features (*p* = .184).

![Figure 3. Leakage-controlled predictive validation.](/Users/Meihui/Downloads/sync/CVS/cvs_conversation/06_figures/chi_report/fig3_predictive_validation.png)

**Figure 3.** Leakage-controlled prediction. Ridge results were small and model-dependent, Elastic Net results were negative, and observed modality gains fell within their permutation null intervals.

The processed-audio CANDOR survey analysis was also near zero: *n* = 1,653, Pearson *r* = .011, permutation *p* = .645; covariate-adjusted standardized beta = .004, 95% CI [-.044, .052], Freedman-Lane *p* = .861. Because the CANDOR acoustic reliability gate failed, this is best described as a precise null association for this operational measure, not as evidence that acoustic coordination in general is unrelated to connection.

### 4.5 Video variables showed no detectable increment

Pose availability was effectively complete, and 94% of participants moved faster while speaking than while listening, supporting the interpretation of the speaker-listener movement contrast. However, over the transcript-plus-acoustic model, all four selected video variables added no more than .017 in *R*², all permutation *p* values were at least .45, and all FDR-adjusted values were approximately .93. Results were also small over the transcript-only model and in a stricter tracking-QC subset. The study therefore found no detectable incremental association for these proxies, but it cannot establish that visible behavior is irrelevant.

### 4.6 Post hoc turn-structure analysis

The corrected actor-partner analysis found no evidence that the same behavior had different actor and partner effects. For example, the partner-minus-actor contrast was .247 for question rate (95% CI [-.191, .685], sign-flip *p* = .277) and -.368 for mean turn duration (95% CI [-.838, .103], *p* = .170). The earlier claim that questions benefited the person being asked rather than the person asking was therefore withdrawn.

At the between-dyad level, the strongest exploratory association was simpler. Dyads with shorter average turns reported higher partner evaluation: *r* = -.704, 95% CI [-.863, -.420], permutation *p* = .0001, BH-adjusted *q* = .0008. Turn count was positively associated with partner evaluation (*r* = .603, *q* = .006), but it correlated -.80 with mean turn duration. In a joint model, mean turn duration contributed unique Δ*R*² = .137, compared with .004 for turn count. The behavior itself was stable across sessions (Spearman-Brown reliability = .939).

This is the strongest remaining empirical signal, but it is not a confirmatory result. The analysis was formulated after earlier models were inspected, the sample contained only 24 dyads, several nested levels were aggregated, and dyad-mean partner evaluation had only marginal reliability. The defensible interpretation is that conversational granularity is a replication target, not that shorter turns have been established as a cause or general predictor of connection.

### 4.7 Within-conversation dynamic trajectory analysis

The preceding analyses treated each conversation as a single summary (session mean). This section asks whether *how features change across the conversation* adds predictive information beyond those static means. Given the post hoc status and small sample, this analysis is exploratory.

**Method.** Each of the 46 discussions was divided into five windows corresponding to the five researcher-provided discussion questions. Within each question-window, ten features were aggregated per speaker or per dyad: mean turn duration, floor share, floor-share imbalance, backchannel rate, response latency, lexical overlap (content-word overlap with the prior partner turn, excluding stop words), turn rate, mean pitch, mean intensity, and mean speech rate. Three dynamic summaries were then computed from the five-window trajectory of each feature: (a) the OLS slope across windows 1–5, indexing a linear trend; (b) the early–late difference (mean of windows 4–5 minus mean of windows 1–2), indexing a shift; and (c) the within-conversation standard deviation, indexing variability. These summaries were averaged across speakers and conditions to the dyad level (N = 24) and tested against partner evaluation — the only dyad-level outcome with marginal reliability (Spearman-Brown = .529). Inference used 5,000-permutation *p* values and 2,000-draw bootstrap confidence intervals with Benjamini-Hochberg FDR correction. As a robustness check, the same analysis was repeated using five equal-duration time bins rather than question boundaries.

**Results.** The static-mean replication was consistent: mean turn duration correlated *r* = -.69 with partner evaluation (permutation *p* = .001), close to the earlier between-dyad estimate of -.70.

The strongest dynamic predictor was floor-share variability (within-conversation SD): dyads in which the balance of speaking time fluctuated more across questions reported lower partner evaluation (*r* = -.67, *R*² = .44, permutation *p* < .001, *q*_BH = .005). Turn-duration variability was similarly associated (*r* = -.64, *R*² = .41, *p* = .001, *q*_BH = .009). Both survived FDR correction. Slope and early–late summaries were weaker: turn-duration slope reached *r* = .45 (*p* = .039) but did not survive FDR correction (*q*_BH = .18).

However, no dynamic summary added significant incremental variance beyond the corresponding static mean. The largest increment was for floor-share slope (Δ*R*² = .12, *p* = .065). In the equal-duration time-bin robustness check, floor-share variability was again the strongest dynamic predictor (*r* = -.75, *p* < .001), and no incremental test reached significance.

**Topic drift tells the same story.** A separate analysis measured how far each conversational window drifted from the film being discussed, using TF-IDF cosine similarity between sliding windows and the film synopsis (see `scripts/llm_simulation/topic_drift.py`). Conversations did wander progressively further from each question's anchor (question-drift slope = 0.098, *p* < .001), confirming that within-conversation topic trajectories exist. Yet the *process* of drifting did not predict connection. What mattered was the *state*: dyads whose talk was less film-focused overall reported higher partner evaluation (Spearman ρ = -.49), consistent with pairs who moved beyond the scripted prompts into personal territory. But the proportion of off-film windows — indexing whether the dyad *actually left* the film topic — was unrelated to connection (ρ = .02). In other words, connection tracked *where* the conversation landed, not the trajectory by which it got there.

**Interpretation.** Across both acoustic–structural and topical dimensions, the pattern converges: connection is associated with the steady state of a conversation — short balanced turns, talk that goes beyond the prompt — not with how the conversation evolves over time. Conversational instability (fluctuating floor share, erratic turn length) correlates with lower connection, but this information is largely redundant with the static session mean. Slope, early–late shift, and topic-drift rate add no incremental variance. The dynamic summaries do not establish a distinct temporal mechanism; rather, sessions with extreme means also tend to be more variable, and sessions that drift more also tend to be less film-focused overall. The result narrows the search: if a dynamic signal exists, it is not captured by simple trajectory summaries at N = 24.

**CANDOR replication.** The same equal-duration time-bin analysis was applied to 1,603 CANDOR conversations. The laboratory turn-duration association did not replicate: mean turn duration correlated *r* = .032 with partner evaluation (*p* = .195). The strongest static predictors were speech rate (*r* = .166), pitch (*r* = .145), and intensity (*r* = .128), all *p* < .001 but all *R*² < .03. The strongest dynamic-only predictor was speech-rate variability (*r* = -.064, *p* = .010). Two dynamic summaries added significant incremental variance beyond the corresponding static mean: response-latency slope (Δ*R*² = .005, *p* = .004) and pitch slope (Δ*R*² = .003, *p* = .040). However, all incremental effect sizes were below .005 in *R*² — statistically detectable at *N* = 1,603 but practically negligible. The CANDOR replication therefore confirmed the laboratory conclusion: within-conversation dynamic trajectory summaries do not add meaningful predictive information beyond static session means.

### 4.8 Reflection convergence analysis

The preceding analyses measured structural and acoustic coordination. This section asks a content-level question: does each speaker's discussion talk move toward the partner's pre-discussion interpretation of the film?

**Method.** Before each discussion, participants wrote a short reflection (Q6: "most memorable scene and why") about the film they had just watched. These 96 reflections (48 participants × 2 films) were embedded with sentence-transformers (all-MiniLM-L6-v2, 384-d; Reimers & Gurevych, 2019). Discussion text was pooled per speaker per question-window. A partner-specific attraction (PSA) index was computed for each speaker-question window as the cosine similarity between the speaker's discussion text and the partner's pre-discussion reflection, minus the mean cosine similarity between the same discussion text and all same-film non-partner reflections. This subtraction controls for shared film content. The analysis used 387 valid windows (22 excluded for < 30 words), 96 reflections, and 46 pseudo-anchor reflections per person-film. Inference used sign-flip permutation (10,000 draws) at the dyad level and bootstrap confidence intervals (2,000 draws). A pseudo-dyad null (1,000 random re-pairings) tested whether real pairings exceeded chance. Robustness checks used mpnet embeddings, TF-IDF, a 50-word minimum, and concatenation of all discussion questions.

**Results.** Mean PSA was 0.0052, 95% CI [-0.0058, 0.0165]; the within-dyad sign-flip *p* = .184, indicating that PSA was not reliably different from zero across dyads. However, the real pairing ranked 32nd of 1,000 pseudo-dyad re-pairings (*p* = .032), suggesting that the real pairing showed more attraction than would be expected by chance reassignment. No temporal dynamics were detected: the Q1–5 linear slope was -0.0008 (*p* = .565), and the Q1–3 vs. Q4–5 contrast was -0.0051 (*p* = .794). Cross-film consistency was near zero (*r* = -.036, Spearman-Brown = -.075). The all-question concatenation robustness check approached significance (PSA = 0.0107, *p* = .053). Results were consistent across embedding models (mpnet *p* = .228; TF-IDF *p* = .177).

**Interpretation.** The evidence is mixed. Discussion talk in real pairs showed slightly more attraction toward the partner's reflection than pseudo-dyad re-pairings, but PSA was not reliably above zero within dyads. Short reflections (median ~33 words for Q6) produce noisy embeddings, and N = 24 limits power. There was no temporal trajectory — speakers did not progressively converge on their partner's interpretation across discussion questions — and no cross-film consistency. The analysis does not establish a content-level convergence mechanism but provides a methodological template for larger samples with richer pre-discussion text.

## 5. Discussion

The project produced a clearer result by beginning with measurement rather than prediction. Real laboratory conversations contained detectable local intensity coordination, yet the derived score did not consistently rank the same sessions or dyads. Aggregate detectability and individual-difference reliability are therefore empirically separable. A null comparison answers whether a pattern exists on average; reliability answers whether observed differences between pairs can be reproduced. Only the second supports using the score as a stable predictor.

This distinction explains why the original in-sample association and later predictive audit are not contradictory. The hierarchical regression describes the 24 observed dyads and remains reproducible as an in-sample association. Nested cross-validation asks whether the fitted relationship transfers to a held-out dyad when preprocessing and tuning are repeated without that dyad. The latter evidence was weak and model-dependent. It is more accurate to say that the association did not translate into reliable out-of-sample prediction than to say that it disappeared or was false.

The outcome audit changed the interpretation as well. High alpha showed that participants used the partner-evaluation and shared-reality items coherently. Low partner ICC showed that their experiences were not interchangeable. Participant-level outcomes are therefore preferable for actor-partner questions, while any dyad-mean analysis must state that it models a weakly supported shared construct.

The post hoc turn-duration result offers a more promising next step because the behavioral measure itself was stable and interpretable. It also avoids the failed claim that actor and partner coefficients differed. A future study should preregister mean turn duration or a less mechanically dependent measure of turn exchange, preserve question and session levels, and collect repeated participant outcomes. Replication should test whether the effect survives independent data, alternative turn segmentation, and adjustment for total talk time without treating a ratio identity as a psychological mechanism.

The CANDOR dynamic trajectory replication sharpened this caution. The laboratory turn-duration association (*r* = -.70) did not appear in 1,603 CANDOR conversations (*r* = .03), and dynamic trajectory summaries added negligible incremental variance even with 60-fold greater sample size. This non-replication may reflect differences in conversation structure (prompted five-question discussions vs. unstructured video calls), microphone configuration, or outcome measurement, but it means that mean turn duration should not yet be treated as a general predictor of connection. The reflection-convergence analysis similarly found no robust content-level mechanism: discussion talk showed only weak attraction toward the partner's pre-discussion interpretation, with no temporal trajectory and no cross-film consistency.

For pipeline design, three practices should become default. First, compare real coordination with nulls that preserve relevant structure. Second, report reliability at the exact level used in the downstream model. Third, nest every data-dependent preprocessing and tuning step inside held-out evaluation. LLM-generated labels should not be treated as validated behavioral codes until compared with independent human judgments.

## 6. Limitations

The laboratory sample was small, with 24 analyzed dyads, so effect estimates and out-of-sample metrics have high sampling variability. Outcomes were collected once, preventing temporal reliability assessment. Two dyads had only one usable discussion. The shared microphone prevented speaker-specific analysis during overlap and made the laboratory transfer definition different from the separate-track CANDOR definition. CANDOR audio may also contain platform processing such as automatic gain control or noise suppression, but this was not directly tested.

The acoustic odd-even split is one operational reliability estimate and may not capture every meaningful time scale. The between-dyad turn analysis compressed turn, question, and session levels into one dyad mean because the sample was too small for a well-supported four-level model. Mean turn duration and turn count are mathematically related to total speaking time, so the term conversational granularity should remain descriptive. Finally, the LLM feature block lacked human reference labels, and the video head-orientation measure was not calibrated as eye gaze.

## 7. Conclusion

The pipeline successfully extracted multimodal conversational behavior and detected local acoustic coordination above structured nulls. It did not validate intensity-edge coordination as a stable dyad-level measure or as a generalizable predictor of social connection. Video and exploratory LLM blocks also showed no reliable incremental predictive gain. The most promising new observation was an exploratory association between shorter average turns and higher partner evaluation, supported by a stable behavioral measure but limited by post hoc selection, marginal outcome reliability, and *n* = 24. The project's main contribution is therefore a validation framework and a caution: before asking what a conversational feature predicts, establish that it measures a reproducible difference at the level where the prediction will be made.

## Reproducibility and Source Map

- Acoustic measurement validation: `05_analysis_outputs/own_data_intensity_edge_validation/intensity_edge_measurement_report.md`
- Outcome scoring and agreement: `05_analysis_outputs/s2_outcome_validation/report.md`
- Corrected between/within analysis: `05_analysis_outputs/s4_apim/report.md`
- Video incremental analysis: `05_analysis_outputs/multimodal_video_incremental/multimodal_video_incremental_report.md`
- Nested predictive validation: `05_analysis_outputs/final_ml_validation/nested_ml_validation_report.md`
- Full audit and sample reconciliation: `07_writeups/ORIGINAL_PROJECT_VALIDATION_AUDIT_2026-08-15.md`
- Dynamic trajectory analysis: `05_analysis_outputs/dynamic_trajectory/` (windowed features, trajectory summaries, model results, summary)
- Topic drift analysis: `05_analysis_outputs/llm_sim_partner/drift_human_tfidf_20261012_201541/summary.md`
- CANDOR dynamic trajectory replication: `05_analysis_outputs/dynamic_trajectory/candor_run_13238729/summary.md`
- Reflection convergence: `05_analysis_outputs/reflection_convergence/run_20261019_1700/` (analysis plan, summary, per-window measures, figures)
- CANDOR results: completed CARC processed-audio runs documented in the project analysis log and report outline.

## References

Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: A practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B, 57*(1), 289-300. https://doi.org/10.1111/j.2517-6161.1995.tb02031.x

Cameron, A. C., & Miller, D. L. (2015). A practitioner's guide to cluster-robust inference. *Journal of Human Resources, 50*(2), 317-372. https://doi.org/10.3368/jhr.50.2.317

Freedman, D. A., & Lane, D. (1983). A nonstochastic interpretation of reported significance levels. *Journal of Business & Economic Statistics, 1*(4), 292-298. https://doi.org/10.1080/07350015.1983.10509354

Gelman, A., & Stern, H. (2006). The difference between significant and not significant is not itself statistically significant. *The American Statistician, 60*(4), 328-331.

Levitan, R., & Hirschberg, J. (2011). Measuring acoustic-prosodic entrainment with respect to multiple levels and dimensions. In *Proceedings of Interspeech 2011* (pp. 3081-3084). https://doi.org/10.21437/Interspeech.2011-771

Reece, A., Cooney, G., Bull, P., Chung, C., Dawson, B., Fitzpatrick, C., Glazer, T., Knox, D., Liebscher, A., Marin, S., & Sefidgar, Y. (2023). The CANDOR corpus: Insights from a large multimodal dataset of naturalistic conversation. *Science Advances, 9*(13), eadf3197. https://doi.org/10.1126/sciadv.adf3197

Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In *Proceedings of EMNLP-IJCNLP 2019* (pp. 3982-3992). https://doi.org/10.18653/v1/D19-1410
