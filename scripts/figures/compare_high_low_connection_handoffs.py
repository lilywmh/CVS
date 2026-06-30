#!/usr/bin/env python3
"""
compare_high_low_connection_handoffs.py
==========================
ILLUSTRATIVE comparison: intensity contour at turn transitions for a HIGH-
connection vs a LOW-connection dyad. Exemplars selected by the connection
composite; their intensity-edge metric values are printed in the titles so the
selection is transparent. NOT evidence (n=2 examples) — the quantitative
evidence is in 05 / the scatter. For a talk/intro figure.

scipy-only (reads 16 kHz WAV directly); intensity = short-time RMS in dB.
"""
from pathlib import Path
import os
import numpy as np
import pandas as pd
from scipy.io import wavfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

PROJECT = Path(__file__).resolve().parents[2]  # cvs_conversation/
DATA = Path(os.environ.get("CVS_DATA", PROJECT / "04_data"))
WAV = Path(os.environ.get("CVS_WAV_DIR", PROJECT / "01_pipeline" / "_wav"))
FIG = Path(os.environ.get("CVS_FIGURES", PROJECT / "06_figures"))
EDGE = 0.5
WINDOW = 240.0                      # long window: capture many handoffs
C = {0: "#2c7fb8", 1: "#d95f0e"}    # speaker 00 / 01

# (pair_id, label, connection z, int-edge prox, int-edge sync) — from analysis
EXEMPLARS = [
    {"pid": 19, "dyad": "dyad19_251015", "tag": "HIGH connection",
     "conn": +0.98, "prox": +0.43, "sync": +0.52},
    {"pid": 18, "dyad": "dyad18_251014", "tag": "LOW connection",
     "conn": -2.05, "prox": -0.09, "sync": -0.12},
]


def rms_db(y, sr, win=0.04, hop=0.01):
    n, h = int(win * sr), int(hop * sr)
    t, v = [], []
    for i in range(0, len(y) - n, h):
        fr = y[i:i + n]
        v.append(20 * np.log10(np.sqrt(np.mean(fr ** 2)) + 1e-6))
        t.append((i + n / 2) / sr)
    return np.array(t), np.array(v)


def find_wav(dyad):
    for cond in ("piper", "cloudy"):
        p = WAV / f"{dyad}_{cond}_discussion_16k.wav"
        if p.exists():
            return p, cond
    return None, None


def best_window(turns):
    """Pick the WINDOW-second span with the most speaker switches."""
    turns = turns.sort_values("t_start").reset_index(drop=True)
    best, best_n = (turns.t_start.iloc[0], 0), -1
    for start in np.arange(turns.t_start.min(), turns.t_end.max() - WINDOW, 2.0):
        seg = turns[(turns.t_start >= start) & (turns.t_start < start + WINDOW)]
        sw = (seg.speaker.values[1:] != seg.speaker.values[:-1]).sum() if len(seg) > 1 else 0
        if sw > best_n:
            best, best_n = (start, sw), sw
    return best[0]


def panel(ax, ex, turns_all):
    wav, cond = find_wav(ex["dyad"])
    sr, y = wavfile.read(wav); y = y.astype(float); y /= (np.max(np.abs(y)) or 1)
    turns = turns_all[(turns_all.dyad_id == ex["dyad"]) & (turns_all.condition == cond)]
    t0 = best_window(turns); t1 = t0 + WINDOW
    seg = turns[(turns.t_end > t0) & (turns.t_start < t1)].sort_values("t_start").reset_index(drop=True)
    ti, iv = rms_db(y[int(t0 * sr):int(t1 * sr)], sr); ti += t0

    spk_idx = {s: i for i, s in enumerate(sorted(turns.speaker.unique()))}
    # raw contour FAINT (long window is dense), colored by speaker
    for _, r in seg.iterrows():
        ci = C[spk_idx.get(r.speaker, 0)]
        m = (ti >= r.t_start) & (ti < r.t_end)
        ax.plot(ti[m], iv[m], color=ci, lw=0.5, alpha=0.35)

    # at each handoff: A-offset level + B-onset level (thick bars) + connector.
    # FLAT connector = levels match; STEEP = mismatch.
    gaps = []
    for k in range(1, len(seg)):
        if seg.loc[k, "speaker"] == seg.loc[k - 1, "speaker"]:
            continue
        bnd = seg.loc[k, "t_start"]
        lvls = {}
        for side, (lo, hi, spk) in {"off": (bnd - EDGE, bnd, seg.loc[k - 1, "speaker"]),
                                    "on": (bnd, bnd + EDGE, seg.loc[k, "speaker"])}.items():
            m = (ti >= max(lo, t0)) & (ti < min(hi, t1))
            if m.any():
                lvls[side] = np.nanmean(iv[m])
                ax.hlines(lvls[side], max(lo, t0), min(hi, t1),
                          color=C[spk_idx.get(spk, 0)], lw=5, alpha=0.95)
        if "off" in lvls and "on" in lvls:
            ax.plot([bnd, bnd], [lvls["off"], lvls["on"]], color="k", lw=1.2, alpha=0.6)
            gaps.append(abs(lvls["off"] - lvls["on"]))
    mean_gap = np.mean(gaps) if gaps else np.nan
    ax.set_title(f"{ex['tag']} — {ex['dyad']} ({cond})   |   connection z={ex['conn']:+.2f}, "
                 f"int-edge prox={ex['prox']:+.2f}/sync={ex['sync']:+.2f}   |   "
                 f"mean handoff level-gap = {mean_gap:.1f} dB ({len(gaps)} handoffs)",
                 fontsize=9)
    ax.set_ylabel("intensity (dB)"); ax.set_xlim(t0, t1)


def scatter_compare():
    """Per-dyad offset->onset intensity scatter (within-speaker z), high vs low.
    Uses ALL of each dyad's speaker switches -> the clean view of 'matching'."""
    from scipy import stats
    d = pd.read_csv(DATA / "acoustic_turns.csv")
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    for a, ex in zip(ax, EXEMPLARS):
        g = d[d.dyad_id == ex["dyad"]].copy()
        pts = []
        for cond, gc in g.groupby("condition"):
            gc = gc.sort_values("t_start").reset_index(drop=True)
            for col in ["offset_int", "onset_int"]:
                gc[col + "_z"] = gc.groupby("speaker")[col].transform(
                    lambda s: (s - s.mean()) / (s.std(ddof=0) or np.nan))
            for i in range(1, len(gc)):
                if gc.loc[i, "speaker"] != gc.loc[i - 1, "speaker"]:
                    pts.append((gc.loc[i - 1, "offset_int_z"], gc.loc[i, "onset_int_z"]))
        arr = np.array([p for p in pts if np.isfinite(p[0]) and np.isfinite(p[1])])
        a.scatter(arr[:, 0], arr[:, 1], s=40, alpha=0.6, color="#2c7fb8", edgecolors="w")
        if len(arr) >= 4:
            r, p = stats.pearsonr(arr[:, 0], arr[:, 1])
            sl, ic = np.polyfit(arr[:, 0], arr[:, 1], 1)
            xs = np.linspace(arr[:, 0].min(), arr[:, 0].max(), 30)
            a.plot(xs, sl * xs + ic, color="#d95f0e", lw=2)
            sub = f"r={r:+.2f} (n={len(arr)} switches)"
        else:
            sub = f"n={len(arr)} switches"
        a.axhline(0, color="#ddd", lw=.6); a.axvline(0, color="#ddd", lw=.6)
        a.set_title(f"{ex['tag']} — {ex['dyad']}\nconnection z={ex['conn']:+.2f}; {sub}",
                    fontsize=10)
        a.set_xlabel("A — turn-OFFSET intensity (z)")
    ax[0].set_ylabel("B — turn-ONSET intensity (z)")
    fig.suptitle("Intensity matching at handoffs: high- vs low-connection dyad "
                 "(illustrative; all of each dyad's switches)", fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"high_low_connection_intensity_scatter.{ext}", dpi=150, bbox_inches="tight")
    print("Wrote", FIG / "high_low_connection_intensity_scatter.png")


def handoffs_zoom(maxh=10, W=1.0, gap=0.5):
    """Splice out ONLY the +/- W s around each speaker switch (drop the long
    monologue middles) and lay the handoff zooms side by side. The thick bars
    are the 0.5 s edge levels; the black connector flat=matched, steep=mismatch."""
    turns_all = pd.read_csv(DATA / "labeled_turns.csv")
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5))
    for ax, ex in zip(axes, EXEMPLARS):
        wav, cond = find_wav(ex["dyad"])
        sr, y = wavfile.read(wav); y = y.astype(float); y /= (np.max(np.abs(y)) or 1)
        s = turns_all[(turns_all.dyad_id == ex["dyad"]) & (turns_all.condition == cond)]
        s = s.sort_values("t_start").reset_index(drop=True)
        spk_idx = {sp: i for i, sp in enumerate(sorted(s.speaker.unique()))}
        bnds = [(s.loc[k, "t_start"], s.loc[k - 1, "speaker"], s.loc[k, "speaker"])
                for k in range(1, len(s)) if s.loc[k, "speaker"] != s.loc[k - 1, "speaker"]]
        bnds = bnds[:maxh]
        xcur, ticks, gaps = 0.0, [], []
        for bnd, prev, nxt in bnds:
            a, b = int((bnd - W) * sr), int((bnd + W) * sr)
            if a < 0 or b > len(y):
                continue
            ti, iv = rms_db(y[a:b], sr)            # ti in 0..2W
            x = ti + xcur
            off = ti < W
            ax.plot(x[off], iv[off], color=C[spk_idx[prev]], lw=1.0)
            ax.plot(x[~off], iv[~off], color=C[spk_idx[nxt]], lw=1.0)
            ax.axvline(xcur + W, color="k", lw=0.6, alpha=0.4)
            offm, onm = (ti >= W - EDGE) & (ti < W), (ti >= W) & (ti < W + EDGE)
            if offm.any() and onm.any():
                lo, on = np.nanmean(iv[offm]), np.nanmean(iv[onm])
                ax.hlines(lo, xcur + W - EDGE, xcur + W, color=C[spk_idx[prev]], lw=5)
                ax.hlines(on, xcur + W, xcur + W + EDGE, color=C[spk_idx[nxt]], lw=5)
                ax.plot([xcur + W, xcur + W], [lo, on], color="k", lw=1.3, alpha=0.7)
                gaps.append(abs(lo - on))
            ticks.append(xcur + W)
            xcur += 2 * W
            ax.axvspan(xcur, xcur + gap, color="#eeeeee")   # "//" omitted middle
            xcur += gap
        mg = np.mean(gaps) if gaps else np.nan
        ax.set_xticks(ticks); ax.set_xticklabels([f"H{i+1}" for i in range(len(ticks))], fontsize=8)
        ax.set_ylabel("intensity (dB)")
        ax.set_title(f"{ex['tag']} — {ex['dyad']}   |   connection z={ex['conn']:+.2f}   |   "
                     f"mean handoff level-gap = {mg:.1f} dB across {len(gaps)} handoffs "
                     f"(grey = omitted middle)", fontsize=9)
    lo = min(a.get_ylim()[0] for a in axes); hi = max(a.get_ylim()[1] for a in axes)
    for a in axes:
        a.set_ylim(lo, hi)
    fig.suptitle("Turn-handoffs only (long middles omitted): high- vs low-connection dyad "
                 "— flat black connector = loudness matched at handoff", fontsize=11)
    # legend INSIDE the HIGH-connection (top) panel, upper-right corner
    axes[0].legend(handles=[Patch(color=C[0], label="Speaker 00"),
                            Patch(color=C[1], label="Speaker 01"),
                            plt.Line2D([0], [0], color="k", lw=4, label="0.5 s edge level"),
                            plt.Line2D([0], [0], color="k", lw=1.3, label="offset→onset connector")],
                   loc="upper right", fontsize=7, ncol=2, framealpha=0.9,
                   borderaxespad=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"high_low_connection_handoff_zoom.{ext}", dpi=150, bbox_inches="tight")
    print("Wrote", FIG / "high_low_connection_handoff_zoom.png")


def main():
    turns = pd.read_csv(DATA / "labeled_turns.csv")
    fig, ax = plt.subplots(2, 1, figsize=(11, 6.5))
    for a, ex in zip(ax, EXEMPLARS):
        panel(a, ex, turns)
    lo = min(a.get_ylim()[0] for a in ax); hi = max(a.get_ylim()[1] for a in ax)
    for a in ax:
        a.set_ylim(lo, hi)
    ax[1].set_xlabel("time (s)")
    fig.suptitle("Intensity at turn transitions: high- vs low-connection dyad "
                 "(illustrative exemplars; thick bars = 0.5 s turn-edge level)",
                 fontsize=11, y=1.0)
    fig.legend(handles=[Patch(color=C[0], label="Speaker 00"),
                        Patch(color=C[1], label="Speaker 01"),
                        plt.Line2D([0], [0], color="k", lw=5, label="turn-edge level")],
               loc="upper right", fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"high_low_connection_intensity_comparison.{ext}", dpi=150, bbox_inches="tight")
    print("Wrote", FIG / "high_low_connection_intensity_comparison.png")
    scatter_compare()
    handoffs_zoom()


if __name__ == "__main__":
    main()
