import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

# ─── PATHS ────────────────────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parents[2]  # cvs_conversation/
DATA = Path(os.environ.get("CVS_DATA", PROJECT / "04_data"))
ANALYSIS_OUTPUTS = Path(
    os.environ.get("CVS_ANALYSIS_OUTPUTS", PROJECT / "05_analysis_outputs")
)
SEMANTIC_PATH   = DATA / 'scientific_dyad_analysis_results.csv'
STRUCTURAL_PATH = DATA / 'structural_dyad_analysis_mapped.csv'
OUTCOMES_PATH   = DATA / 'outcomes.csv'
OUTPUT_DIR      = ANALYSIS_OUTPUTS / 'dyad_feature_outcome_figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── STYLE ────────────────────────────────────────────────────────────────────
BLUE       = '#2C6FAC'
ORANGE     = '#E8834A'
LIGHT_BLUE = '#A8C8E8'
LIGHT_ORG  = '#F5C4A4'
GREY       = '#666666'
BG         = '#FAFAFA'

plt.rcParams.update({
    'font.family':       'Helvetica Neue',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.facecolor':    BG,
    'figure.facecolor':  'white',
    'axes.labelsize':    13,
    'axes.titlesize':    14,
    'xtick.labelsize':   11,
    'ytick.labelsize':   11,
})

# ─── DATA PREP ────────────────────────────────────────────────────────────────
def load_dyad_data():
    sem    = pd.read_csv(SEMANTIC_PATH)
    struct = pd.read_csv(STRUCTURAL_PATH)
    outcomes = pd.read_csv(OUTCOMES_PATH)

    for df in [sem, struct]:
        df['pair_id'] = df['Dyad ID'].str.extract(r'dyad(\d+)').astype(int)

    # Collapse to dyad level
    sem_dyad = sem.groupby('pair_id')[['semantic_similarity','sentiment_synchrony','mean_sentiment']].mean().reset_index()
    sem_dyad.columns = ['pair_id','sem_semantic_similarity','sem_sentiment_synchrony','sem_mean_sentiment']

    dyad_feats = ['total_turns','participation_gini','turn_taking_density']
    struct_dyad = struct.groupby('pair_id')[dyad_feats].mean().reset_index()

    speaker_feats = ['ttr','i_rate','we_rate','q_count','bc_rate']
    for feat in speaker_feats:
        a = struct.groupby('pair_id')[f'{feat}_A'].mean()
        b = struct.groupby('pair_id')[f'{feat}_B'].mean()
        struct_dyad[f'{feat}_mean'] = ((a + b) / 2).values
        struct_dyad[f'{feat}_asym'] = (a - b).abs().values

    order = struct.groupby('pair_id')['Order'].first().reset_index()

    df = sem_dyad.merge(struct_dyad, on='pair_id').merge(order, on='pair_id')
    df = df.merge(outcomes[['pair_id','dyad_partner_eval_mean','dyad_enjoyment_mean',
                             'dyad_shared_reality_mean','dyad_solo_mean']], on='pair_id')
    return df


# ─── PLOT 1: INTERACTION PROFILE ──────────────────────────────────────────────
def plot_interaction_profile(df):
    """
    Split dyads into high vs low q_count_asym groups.
    Compare their partner_eval and enjoyment means with error bars.
    """
    median_asym = df['q_count_asym'].median()
    df['asym_group'] = np.where(df['q_count_asym'] > median_asym, 'High\nAsymmetry', 'Low\nAsymmetry')

    outcomes = ['dyad_partner_eval_mean', 'dyad_enjoyment_mean']
    labels   = ['Partner Evaluation', 'Enjoyment']
    colors   = [BLUE, ORANGE]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=False)
    fig.suptitle('Social Outcomes by Question Asymmetry', fontsize=16, fontweight='bold', y=1.02)

    for ax, outcome, label, color in zip(axes, outcomes, labels, colors):
        groups = ['Low\nAsymmetry', 'High\nAsymmetry']
        means  = [df[df['asym_group'] == g][outcome].mean() for g in groups]
        sems   = [df[df['asym_group'] == g][outcome].sem()  for g in groups]

        bars = ax.bar(groups, means, yerr=sems, capsize=6,
                      color=[LIGHT_BLUE, color], edgecolor=['#2C6FAC', color],
                      linewidth=1.5, error_kw={'linewidth': 2, 'color': GREY},
                      width=0.5, zorder=2)

        # Overlay individual data points
        for i, g in enumerate(groups):
            subset = df[df['asym_group'] == g][outcome]
            jitter = np.random.uniform(-0.08, 0.08, size=len(subset))
            ax.scatter(np.full(len(subset), i) + jitter, subset,
                       color='white', edgecolors=GREY, s=40, zorder=3, linewidths=0.8, alpha=0.8)

        # t-test annotation
        lo = df[df['asym_group'] == 'Low\nAsymmetry'][outcome]
        hi = df[df['asym_group'] == 'High\nAsymmetry'][outcome]
        t, p = stats.ttest_ind(lo, hi)
        sig = '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else 'ns'
        p_str = f'p = {p:.3f}' if p >= .001 else 'p < .001'
        label = f't({len(lo)+len(hi)-2}) = {t:.2f}, {p_str}  {sig}'

        y_max = max(means) + max(sems) + 0.15
        ax.plot([0, 0, 1, 1], [y_max, y_max + 0.05, y_max + 0.05, y_max],
                color=GREY, linewidth=1.2)
        ax.text(0.5, y_max + 0.08, label, ha='center', va='bottom', fontsize=10, color=GREY)

        ax.set_title(label, fontweight='bold', pad=10)
        ax.set_ylabel('Mean Rating', labelpad=8)
        ax.set_ylim(bottom=df[outcome].min() - 0.3)
        ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, '1_interaction_profile.pdf')
    plt.savefig(path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Saved: {path}")


# ─── PLOT 2: CORRELATION MATRIX ───────────────────────────────────────────────
def plot_correlation_matrix(df):
    """
    Correlation matrix between key features and outcomes.
    Clean, readable, readable.
    """
    feature_map = {
        'sem_semantic_similarity':  'Semantic\nSimilarity',
        'sem_sentiment_synchrony':  'Sentiment\nSynchrony',
        'total_turns':              'Total\nTurns',
        'turn_taking_density':      'Turn-Taking\nDensity',
        'ttr_mean':                 'Lexical\nDiversity',
        'q_count_mean':             'Question\nCount',
        'q_count_asym':             'Question\nAsymmetry',
        'bc_rate_mean':             'Backchannel\nRate',
        'dyad_partner_eval_mean':   'Partner\nEvaluation',
        'dyad_shared_reality_mean': 'Shared\nReality',
        'dyad_enjoyment_mean':      'Enjoyment',
    }

    cols = list(feature_map.keys())
    sub  = df[cols].rename(columns=feature_map)
    corr = sub.corr()

    # Mask upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(12, 10))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    sns.heatmap(
        corr, mask=mask, cmap=cmap, center=0,
        vmin=-1, vmax=1, square=True, linewidths=0.5,
        linecolor='white', annot=True, fmt='.2f',
        annot_kws={'size': 9}, ax=ax,
        cbar_kws={'shrink': 0.6, 'label': 'Pearson r'}
    )

    ax.set_title('Feature & Outcome Correlation Matrix', fontsize=16,
                 fontweight='bold', pad=15)
    ax.tick_params(axis='x', rotation=0, labelsize=10)
    ax.tick_params(axis='y', rotation=0, labelsize=10)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, '2_correlation_matrix.pdf')
    plt.savefig(path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Saved: {path}")


# ─── PLOT 3: VIOLIN — FEATURE DISTRIBUTION BY OUTCOME GROUP ──────────────────
def plot_violin_by_outcome(df):
    """
    Split dyads into high/low partner_eval.
    Show distribution of key features in each group.
    """
    median_eval = df['dyad_partner_eval_mean'].median()
    df['eval_group'] = np.where(
        df['dyad_partner_eval_mean'] > median_eval,
        'High Partner\nEvaluation', 'Low Partner\nEvaluation'
    )

    features = {
        'q_count_asym':            'Question Asymmetry',
        'q_count_mean':            'Question Count',
        'total_turns':             'Total Turns',
        'bc_rate_mean':            'Backchannel Rate',
        'sem_sentiment_synchrony': 'Sentiment Synchrony',
        'ttr_mean':                'Lexical Diversity',
    }

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle('Feature Distributions by Partner Evaluation Group',
                 fontsize=16, fontweight='bold', y=1.02)
    axes = axes.flatten()

    palette = {'Low Partner\nEvaluation': LIGHT_BLUE, 'High Partner\nEvaluation': BLUE}

    for ax, (feat, label) in zip(axes, features.items()):
        groups = ['Low Partner\nEvaluation', 'High Partner\nEvaluation']

        vp = ax.violinplot(
            [df[df['eval_group'] == g][feat].dropna().values for g in groups],
            positions=[0, 1], showmedians=True, showextrema=False
        )

        for i, (body, color) in enumerate(zip(vp['bodies'], [LIGHT_BLUE, BLUE])):
            body.set_facecolor(color)
            body.set_alpha(0.8)
            body.set_edgecolor(GREY)
        vp['cmedians'].set_color(GREY)
        vp['cmedians'].set_linewidth(2)

        # Overlay points
        for i, g in enumerate(groups):
            subset = df[df['eval_group'] == g][feat].dropna()
            jitter = np.random.uniform(-0.06, 0.06, size=len(subset))
            ax.scatter(np.full(len(subset), i) + jitter, subset,
                       color='white', edgecolors=GREY, s=35, zorder=3,
                       linewidths=0.8, alpha=0.9)

        # t-test
        lo = df[df['eval_group'] == 'Low Partner\nEvaluation'][feat].dropna()
        hi = df[df['eval_group'] == 'High Partner\nEvaluation'][feat].dropna()
        t, p = stats.ttest_ind(lo, hi)
        sig = '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else 'ns'
        p_str = f'p = {p:.3f}' if p >= .001 else 'p < .001'
        label = f't({len(lo)+len(hi)-2}) = {t:.2f}, {p_str} {sig}'

        y_max = df[feat].max() + df[feat].std() * 0.3
        ax.text(0.5, y_max, label, ha='center', va='bottom', fontsize=8.5, color=GREY)

        ax.set_title(label, fontweight='bold', pad=8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(groups, fontsize=9)
        ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, '3_violin_by_outcome.pdf')
    plt.savefig(path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Saved: {path}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    np.random.seed(42)
    df = load_dyad_data()

    print("Generating dyad feature-outcome figures...")
    plot_interaction_profile(df)
    plot_correlation_matrix(df)
    plot_violin_by_outcome(df)
    print(f"\nAll figures saved to '{OUTPUT_DIR}/'")
