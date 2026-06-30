import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parents[2]  # cvs_conversation/
DATA = Path(os.environ.get("CVS_DATA", PROJECT / '04_data'))
ANALYSIS_OUTPUTS = Path(
    os.environ.get("CVS_ANALYSIS_OUTPUTS", PROJECT / '05_analysis_outputs')
)
FEATURES_PATH = ANALYSIS_OUTPUTS / 'llm_annotation_output' / 'dyad_features.csv'
OUTCOMES_PATH = DATA / 'outcomes.csv'
OUTPUT_DIR    = ANALYSIS_OUTPUTS / 'llm_regression_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTCOME_COLS = [
    'dyad_partner_eval_mean',
    'dyad_shared_reality_mean',
    'dyad_enjoyment_mean',
    'dyad_solo_mean'
]

# Theoretically motivated feature set
SELECTED_FEATURES = [
    'depth_deep_mean',       # average discussion depth
    'depth_deep_asym',       # depth imbalance between speakers
    'personal_mean',         # average personal stance rate
    'question_count',        # genuine questions asked
    'questions_asym',        # question asking imbalance
    'on_topic_rate',         # overall topic adherence
    'on_topic_q2',           # Q2 (personal resonance) topic adherence
    'disclosure_mid_n',      # mid-level self disclosure turns
    'epistemic_low_rate',    # hedging / uncertainty rate
    'hedging_asym',          # hedging imbalance between speakers
    'converge_rate',         # rate of convergence turns
    'converge_asym',         # convergence imbalance between speakers
    'yeah_elaborated_n',     # yeah + new content turns
    'yeah_agreement_n',      # pure agreement turns
    'turn_asym',             # turn-taking imbalance
    'personal_asym',         # personal stance imbalance
]

# Columns to exclude from any analysis
EXCLUDE = {
    'pair_id', 'total_turns', 'responsive_rate',
    'responsive_mean', 'responsive_asym'
}

def add_composite_features(df):
    """
    Create theoretically motivated composite features.
    imbalance_score: overall conversational imbalance
    (average of normalized asym indicators — higher = more unequal)
    """
    asym_cols = ['turn_asym', 'depth_deep_asym', 'hedging_asym',
                 'converge_asym', 'questions_asym', 'personal_asym']
    available = [c for c in asym_cols if c in df.columns]

    # Z-score each asym col then average — so they contribute equally
    normed = pd.DataFrame()
    for c in available:
        std = df[c].std()
        if std > 0:
            normed[c] = (df[c] - df[c].mean()) / std

    df['imbalance_score'] = normed.mean(axis=1)
    print(f"  imbalance_score built from: {available}")
    return df


# ─── STEP 1: COLLAPSE TO DYAD LEVEL ──────────────────────────────────────────
def collapse_to_dyad(df):
    """Average features across conditions (piper + cloudy) per dyad."""
    feat_cols = [c for c in df.columns if c not in ['dyad_id', 'condition', 'pair_id']]
    dyad = df.groupby('pair_id')[feat_cols].mean().reset_index()
    print(f"Collapsed to {len(dyad)} dyads, {len(feat_cols)} features")
    return dyad


# ─── STEP 2: MERGE WITH OUTCOMES ─────────────────────────────────────────────
def merge_with_outcomes(dyad, outcomes):
    merged = dyad.merge(outcomes[['pair_id'] + OUTCOME_COLS], on='pair_id', how='inner')
    print(f"Merged dataset: {merged.shape[0]} dyads, {merged.shape[1]} columns")
    return merged


# ─── STEP 3: SELECT FEATURES ─────────────────────────────────────────────────
def select_features(df):
    available = [f for f in SELECTED_FEATURES if f in df.columns]
    missing   = [f for f in SELECTED_FEATURES if f not in df.columns]
    zero_var  = [f for f in available if df[f].std() == 0]

    if missing:
        print(f"  Missing features: {missing}")
    if zero_var:
        print(f"  Skipping (zero variance): {zero_var}")

    feature_cols = [f for f in available if f not in zero_var]
    print(f"Features entering regression: {len(feature_cols)}")
    return feature_cols


# ─── STEP 4: REGRESSION ───────────────────────────────────────────────────────
def run_regressions(df, feature_cols):
    """
    Univariate OLS for each feature-outcome pair.
    Standardized beta weights (z-scored predictor and outcome).
    Given N=24, intentionally univariate to avoid overfitting.
    """
    results = []
    for outcome in OUTCOME_COLS:
        for feat in feature_cols:
            subset = df[[feat, outcome]].dropna()
            if len(subset) < 10:
                continue
            if subset[feat].std() == 0:
                continue

            z_feat = (subset[feat] - subset[feat].mean()) / subset[feat].std()
            z_out  = (subset[outcome] - subset[outcome].mean()) / subset[outcome].std()

            X     = sm.add_constant(z_feat.rename('Feature'))
            model = sm.OLS(z_out, X).fit()

            results.append({
                'Feature': feat,
                'Outcome': outcome,
                'Beta':    round(model.params['Feature'], 3),
                'SE':      round(model.bse['Feature'], 3),
                'P':       round(model.pvalues['Feature'], 4),
                'R2':      round(model.rsquared, 3),
                'N':       len(subset)
            })

    report = pd.DataFrame(results)
    if not report.empty:
        _, report['P_adj_FDR'], _, _ = multipletests(report['P'], method='fdr_bh')
        report['P_adj_FDR'] = report['P_adj_FDR'].round(4)

    return report


# ─── STEP 5: VISUALIZATIONS ───────────────────────────────────────────────────
def plot_heatmap(report):
    pivot_beta = report.pivot(index='Feature', columns='Outcome', values='Beta')
    pivot_p    = report.pivot(index='Feature', columns='Outcome', values='P')
    pivot_padj = report.pivot(index='Feature', columns='Outcome', values='P_adj_FDR')

    annot = pd.DataFrame('', index=pivot_beta.index, columns=pivot_beta.columns)
    for feat in pivot_beta.index:
        for out in pivot_beta.columns:
            b    = pivot_beta.loc[feat, out]
            p    = pivot_p.loc[feat, out]
            padj = pivot_padj.loc[feat, out]
            star = '*' if p < .05 else ''
            fdr  = '†' if padj < .05 else ''
            annot.loc[feat, out] = f'{b:.2f}{star}{fdr}'

    fig, ax = plt.subplots(figsize=(10, max(8, len(pivot_beta) * 0.35)))
    sns.heatmap(
        pivot_beta, annot=annot, fmt='', cmap='RdYlGn',
        center=0, linewidths=0.4, ax=ax,
        cbar_kws={'label': 'Standardized Beta'}
    )
    ax.set_title(
        'LLM Feature Beta Weights\n* p<.05 (uncorrected)   † p<.05 (FDR corrected)',
        fontsize=12, pad=12
    )
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'llm_beta_heatmap.pdf')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved: {path}")


def plot_significant_scatters(df, report):
    sig = report[report['P'] < .05].sort_values('Beta', ascending=False)
    if sig.empty:
        print("No significant findings to plot.")
        return

    n = len(sig)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for ax, (_, row) in zip(axes, sig.iterrows()):
        feat, out = row['Feature'], row['Outcome']
        subset = df[[feat, out]].dropna()
        ax.scatter(subset[feat], subset[out], alpha=0.7,
                   edgecolors='#555', linewidths=0.5, s=50, color='#2C6FAC')
        m, b = np.polyfit(subset[feat], subset[out], 1)
        x_line = np.linspace(subset[feat].min(), subset[feat].max(), 100)
        ax.plot(x_line, m * x_line + b, color='tomato', linewidth=1.5)
        ax.set_xlabel(feat, fontsize=8)
        ax.set_ylabel(out.replace('dyad_','').replace('_mean',''), fontsize=8)
        ax.set_title(f'β={row["Beta"]:.2f}, p={row["P"]:.3f}', fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # hide unused axes
    for ax in axes[n:]:
        ax.set_visible(False)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'llm_significant_scatters.pdf')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Scatter plots saved: {path}")


# ─── STEP 6: PRINT SUMMARY ────────────────────────────────────────────────────
def print_summary(report):
    sig_raw = report[report['P'] < .05].sort_values('Beta', ascending=False)
    sig_fdr = report[report['P_adj_FDR'] < .05]

    print(f"\n{'─'*60}")
    print(f"REGRESSION SUMMARY")
    print(f"{'─'*60}")
    print(f"Total tests run    : {len(report)}")
    print(f"Uncorrected p<.05  : {len(sig_raw)}")
    print(f"FDR corrected p<.05: {len(sig_fdr)}")

    print(f"\n── Uncorrected Significant Findings ──")
    if not sig_raw.empty:
        print(sig_raw[['Feature','Outcome','Beta','P','P_adj_FDR','N']].to_string(index=False))
    else:
        print("  None")

    if not sig_fdr.empty:
        print(f"\n── FDR Corrected Findings ──")
        print(sig_fdr[['Feature','Outcome','Beta','P','P_adj_FDR']].to_string(index=False))

    print(f"\nNote: N=24 dyads. All results are exploratory.")
    print(f"      No FDR-corrected significant findings is expected at this sample size.")
    print(f"      Focus on effect sizes and consistent patterns across outcomes.")


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Loading data...")
    features_df = pd.read_csv(FEATURES_PATH)
    outcomes_df = pd.read_csv(OUTCOMES_PATH)

    print("\nStep 1: Collapsing to dyad level...")
    dyad_df = collapse_to_dyad(features_df)

    print("\nStep 2: Adding composite features...")
    dyad_df = add_composite_features(dyad_df)

    print("\nStep 3: Merging with outcomes...")
    df = merge_with_outcomes(dyad_df, outcomes_df)

    print("\nStep 4: Selecting features...")
    feature_cols = select_features(df)

    print("\nStep 5: Running regressions...")
    report = run_regressions(df, feature_cols)
    report.to_csv(os.path.join(OUTPUT_DIR, 'llm_regression_report.csv'), index=False)
    print(f"Full report saved.")

    print("\nStep 6: Plotting...")
    plot_heatmap(report)
    plot_significant_scatters(df, report)

    print_summary(report)
