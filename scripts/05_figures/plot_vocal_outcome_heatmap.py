#!/usr/bin/env python3
"""
plot_vocal_outcome_heatmap.py
=============================
Exploratory heatmap: correlation of each vocal-alignment metric with each
connection outcome (N=24 dyads). Red = positive, blue = negative, * = p<.05
(uncorrected, exploratory). Also writes the underlying numbers as a CSV.

EXPLORATORY ONLY -- not the confirmatory composite test (see 05).
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parents[2]  # cvs_conversation/
DATA = PROJECT / "04_data"
FIG = PROJECT / "06_figures"

# PRIMARY = connection composite (PC1 of the 3 social scales, matching
# 05_poster_multivariate_analysis.py); the 4 individual scales follow as secondary.
COMPOSITE_COLS = ["dyad_partner_eval_mean", "dyad_shared_reality_mean",
                  "dyad_enjoyment_mean"]
OUTC = ["connection_composite", "dyad_partner_eval_mean",
        "dyad_shared_reality_mean", "dyad_enjoyment_mean", "dyad_solo_mean"]
LABELS = {"connection_composite": "CONNECTION\n(composite)",
          "dyad_partner_eval_mean": "partner_eval",
          "dyad_shared_reality_mean": "shared_reality",
          "dyad_enjoyment_mean": "enjoyment", "dyad_solo_mean": "solo"}

turn = pd.read_csv(DATA / "dyad_level_dataset.csv")
vocal = pd.read_csv(DATA / "vocal_alignment_dyad.csv")
va = [c for c in vocal.columns if c.startswith("va_")]
vag = vocal.groupby("pair_id")[va].mean().reset_index()
df = turn.merge(vag, on="pair_id")

# build the connection composite (PC1, oriented so connection is positive)
Z = StandardScaler().fit_transform(df[COMPOSITE_COLS])
pca = PCA(n_components=3, random_state=42).fit(Z)
pc1 = pca.transform(Z)[:, 0]
load = pca.components_[0]
if load[np.argmax(np.abs(load))] < 0:
    pc1 = -pc1
df["connection_composite"] = pc1
print(f"connection composite: PC1 var {pca.explained_variance_ratio_[0]:.0%}")

R = np.zeros((len(va), len(OUTC)))
P = np.zeros_like(R)
for i, m in enumerate(va):
    for j, o in enumerate(OUTC):
        R[i, j], P[i, j] = stats.pearsonr(df[m], df[o])

# save numbers
out = pd.DataFrame(R, index=va, columns=[LABELS[o] for o in OUTC]).round(3)
out.to_csv(DATA / "vocal_outcome_correlations.csv")

# nicer row labels: va_int_edge_synchrony -> int · edge · synchrony
def pretty(m):
    return m.replace("va_", "").replace("_", " · ")

fig, ax = plt.subplots(figsize=(8.5, 8))
im = ax.imshow(R, cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
ax.set_xticks(range(len(OUTC)))
ax.set_xticklabels([LABELS[o] for o in OUTC], rotation=30, ha="right", fontsize=10)
# bold the primary (composite) column label
ax.get_xticklabels()[0].set_fontweight("bold")
ax.set_yticks(range(len(va)))
ax.set_yticklabels([pretty(m) for m in va], fontsize=9)
for i in range(len(va)):
    for j in range(len(OUTC)):
        star = "*" if P[i, j] < 0.05 else ""
        txt = f"{R[i, j]:+.2f}{star}"
        ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                color="white" if abs(R[i, j]) > 0.4 else "black",
                fontweight="bold" if star else "normal")
# divider line separating PRIMARY composite (col 0) from secondary scales
ax.axvline(0.5, color="black", lw=2)
cbar = fig.colorbar(im, ax=ax, shrink=0.6)
cbar.set_label("Pearson r", fontsize=9)
ax.set_title("Vocal alignment × connection (exploratory, N=24)\n"
             "left col = primary composite | red=+  blue=−  * = p<.05 uncorrected",
             fontsize=11)
fig.tight_layout()
FIG.mkdir(exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"vocal_outcome_heatmap.{ext}", dpi=150, bbox_inches="tight")
print("Wrote:")
print(" ", FIG / "vocal_outcome_heatmap.png")
print(" ", FIG / "vocal_outcome_heatmap.pdf")
print(" ", DATA / "vocal_outcome_correlations.csv")
