import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parent.parent
SEMANTIC_PATH   = PROJECT / '04_data' / 'scientific_dyad_analysis_results.csv'
STRUCTURAL_PATH = PROJECT / '04_data' / 'structural_dyad_analysis_mapped.csv'
OUTCOMES_PATH   = PROJECT / '04_data' / 'outcomes.csv'
OUTPUT_DIR      = PROJECT / '05_analysis_outputs' / 'dyad_analysis_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTCOME_COLS = [
    'dyad_partner_eval_mean',
    'dyad_shared_reality_mean',
    'dyad_enjoyment_mean',
    'dyad_solo_mean'
]

# ─── STEP 1: COLLAPSE FEATURES TO DYAD LEVEL ─────────────────────────────────

def collapse_to_dyad_level():
    sem    = pd.read_csv(SEMANTIC_PATH)
    struct = pd.read_csv(STRUCTURAL_PATH)

    # Extract numeric dyad ID to match outcomes pair_id
    for df in [sem, struct]:
        df['pair_id'] = df['Dyad ID'].str.extract(r'dyad(\d+)').astype(int)

    # --- Semantic features: average across conditions ---
    sem_features = ['semantic_similarity', 'sentiment_synchrony', 'mean_sentiment']
    sem_dyad = (
        sem.groupby('pair_id')[sem_features]
        .mean()
        .reset_index()
        .rename(columns=lambda c: f'sem_{c}' if c != 'pair_id' else c)
    )

    # --- Structural features ---
    # Dyad-level features: average across conditions
    dyad_level_feats = ['total_turns', 'participation_gini', 'turn_taking_density']
    struct_dyad = (
        struct.groupby('pair_id')[dyad_level_feats]
        .mean()
        .reset_index()
    )

    # Per-speaker features: compute dyad mean and asymmetry (|A - B|)
    # Mean captures overall level; asymmetry captures imbalance between partners
    speaker_feats = ['ttr', 'i_rate', 'we_rate', 'q_count', 'bc_rate']
    for feat in speaker_feats:
        col_a, col_b = f'{feat}_A', f'{feat}_B'
        # Average first across conditions, then compute mean and asymmetry
        a_mean = struct.groupby('pair_id')[col_a].mean()
        b_mean = struct.groupby('pair_id')[col_b].mean()
        struct_dyad[f'{feat}_mean']  = ((a_mean + b_mean) / 2).values
        struct_dyad[f'{feat}_asym']  = (a_mean - b_mean).abs().values

    # --- Order: take the value (same within dyad) ---
    order = struct.groupby('pair_id')['Order'].first().reset_index()

    # --- Merge everything ---
    features_dyad = (
        sem_dyad
        .merge(struct_dyad, on='pair_id')
        .merge(order,       on='pair_id')
    )

    return features_dyad


# ─── STEP 2: MERGE WITH OUTCOMES ─────────────────────────────────────────────

def build_analysis_dataset(features_dyad):
    outcomes = pd.read_csv(OUTCOMES_PATH)
    df = features_dyad.merge(outcomes[['pair_id'] + OUTCOME_COLS], on='pair_id', how='inner')
    print(f"Analysis dataset: {df.shape[0]} dyads, {df.shape[1]} columns")
    return df


# ─── STEP 3: REGRESSION ───────────────────────────────────────────────────────

def run_regressions(df, feature_cols):
    """
    Univariate OLS for each feature-outcome pair.
    Standardized beta weights, controlling for Order.
    Given N=24, this is intentionally univariate — adding more predictors
    would overfit badly at this sample size.
    """
    results = []

    for outcome in OUTCOME_COLS:
        for feat in feature_cols:
            subset = df[[feat, outcome, 'Order']].dropna()
            if len(subset) < 10:
                continue

            # Standardize predictor and outcome (z-score)
            z_feat = (subset[feat]    - subset[feat].mean())    / subset[feat].std()
            z_out  = (subset[outcome] - subset[outcome].mean()) / subset[outcome].std()

            X = sm.add_constant(pd.DataFrame({
                'Feature': z_feat,
                'Order':   subset['Order']
            }))
            model = sm.OLS(z_out, X).fit()

            results.append({
                'Feature':   feat,
                'Outcome':   outcome,
                'Beta':      model.params['Feature'],
                'SE':        model.bse['Feature'],
                'P':         model.pvalues['Feature'],
                'R2':        model.rsquared,
                'N':         len(subset)
            })

    report = pd.DataFrame(results)

    # FDR correction across all tests
    if not report.empty:
        _, report['P_adj_FDR'], _, _ = multipletests(report['P'], method='fdr_bh')

    return report


# ─── STEP 4: VISUALIZATION ────────────────────────────────────────────────────

def plot_heatmap(report):
    pivot_beta = report.pivot(index='Feature', columns='Outcome', values='Beta')
    pivot_p    = report.pivot(index='Feature', columns='Outcome', values='P')
    pivot_padj = report.pivot(index='Feature', columns='Outcome', values='P_adj_FDR')

    # Annotation: show beta + significance markers
    annot = pd.DataFrame('', index=pivot_beta.index, columns=pivot_beta.columns)
    for feat in pivot_beta.index:
        for out in pivot_beta.columns:
            b    = pivot_beta.loc[feat, out]
            p    = pivot_p.loc[feat, out]
            padj = pivot_padj.loc[feat, out]
            raw_star  = '*' if p < .05 else ''
            fdr_marker = '†' if padj < .05 else ''
            annot.loc[feat, out] = f'{b:.2f}{raw_star}{fdr_marker}'

    fig, ax = plt.subplots(figsize=(10, 12))
    sns.heatmap(
        pivot_beta, annot=annot, fmt='', cmap='RdYlGn',
        center=0, linewidths=.5, ax=ax
    )
    ax.set_title(
        'Standardized Beta Weights (controlling for Order)\n'
        '* p < .05 (uncorrected)   † p < .05 (FDR corrected)',
        fontsize=12, pad=15
    )
    ax.set_xlabel('Social Outcome', fontsize=11)
    ax.set_ylabel('Linguistic Feature', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'beta_heatmap.pdf'), dpi=150)
    plt.close()
    print("Heatmap saved.")


def plot_significant_scatters(df, report, feature_cols):
    """Scatter plots for any uncorrected p < .05 findings."""
    sig = report[report['P'] < .05]
    if sig.empty:
        print("No uncorrected significant findings to plot.")
        return

    n = len(sig)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (_, row) in zip(axes, sig.iterrows()):
        feat, out = row['Feature'], row['Outcome']
        subset = df[[feat, out]].dropna()
        ax.scatter(subset[feat], subset[out], alpha=0.7, edgecolors='k', linewidths=0.5)

        # Regression line
        m, b_int = np.polyfit(subset[feat], subset[out], 1)
        x_line = np.linspace(subset[feat].min(), subset[feat].max(), 100)
        ax.plot(x_line, m * x_line + b_int, color='tomato', linewidth=1.5)

        ax.set_xlabel(feat, fontsize=9)
        ax.set_ylabel(out, fontsize=9)
        ax.set_title(f'β={row["Beta"]:.2f}, p={row["P"]:.3f}', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'significant_scatters.pdf'), dpi=150)
    plt.close()
    print(f"{n} scatter plot(s) saved.")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    print("Step 1: Collapsing features to dyad level...")
    features_dyad = collapse_to_dyad_level()

    print("Step 2: Merging with outcomes...")
    df = build_analysis_dataset(features_dyad)

    # Save merged dataset for inspection
    df.to_csv(os.path.join(OUTPUT_DIR, 'dyad_level_dataset.csv'), index=False)

    # Identify feature columns (everything except IDs, outcomes, Order)
    exclude = set(OUTCOME_COLS + ['pair_id', 'Order'])
    feature_cols = [c for c in df.columns if c not in exclude]
    print(f"\nFeatures entering analysis ({len(feature_cols)}):")
    print(feature_cols)

    print("\nStep 3: Running regressions...")
    report = run_regressions(df, feature_cols)
    report.to_csv(os.path.join(OUTPUT_DIR, 'regression_report.csv'), index=False)

    print("\nStep 4: Plotting...")
    plot_heatmap(report)
    plot_significant_scatters(df, report, feature_cols)

    # Summary
    sig_raw = report[report['P'] < .05].sort_values('Beta', ascending=False)
    sig_fdr = report[report['P_adj_FDR'] < .05]

    print(f"\nDone. Results in '{OUTPUT_DIR}'")
    print(f"\nUncorrected significant findings (p < .05): {len(sig_raw)}")
    if not sig_raw.empty:
        print(sig_raw[['Feature', 'Outcome', 'Beta', 'P', 'P_adj_FDR', 'N']].to_string(index=False))

    print(f"\nFDR-corrected significant findings: {len(sig_fdr)}")
    if not sig_fdr.empty:
        print(sig_fdr[['Feature', 'Outcome', 'Beta', 'P', 'P_adj_FDR']].to_string(index=False))
    else:
        print("None — given N=24 this is expected. Focus on effect sizes and uncorrected p-values as exploratory.")
