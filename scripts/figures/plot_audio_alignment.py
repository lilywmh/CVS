#!/usr/bin/env python3
"""
plot_audio_alignment.py
==========================
Illustrative audio-visualisation figures for the vocal-alignment story.

Fig 1 (example): for one dyad segment, waveform + intensity(dB) + pitch(f0)
contours, shaded by speaker, with turn-edge windows highlighted to show
INTENSITY MATCHING at turn transitions (the headline finding, illustrated).

Fig 2 (dataset): scatter of speaker-A turn-OFFSET intensity vs the next
speaker-B turn-ONSET intensity across ALL switches -> the entrainment, shown
quantitatively (within-speaker z-scored).

Uses only scipy + numpy + matplotlib (no parselmouth/librosa needed).
f0 via autocorrelation; intensity via short-time RMS.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import wavfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

PROJECT = Path(__file__).resolve().parents[2]  # cvs_conversation/
WAV_DIR = PROJECT / "01_pipeline" / "_wav"
FIG = PROJECT / "06_figures"
FIG.mkdir(exist_ok=True)

EX_DYAD, EX_COND = "dyad12_250928", "piper"
T0, T1 = 9.0, 19.0           # window (s) with two clean handoffs
F0_MIN, F0_MAX = 75, 400
EDGE = 0.5                    # edge window (s)
C = {"SPEAKER_00": "#2c7fb8", "SPEAKER_01": "#d95f0e"}   # blue / orange


# ---------- short-time features ----------
def rms_db(y, sr, win=0.04, hop=0.01):
    n, h = int(win * sr), int(hop * sr)
    t, v = [], []
    for i in range(0, len(y) - n, h):
        fr = y[i:i + n]
        v.append(20 * np.log10(np.sqrt(np.mean(fr ** 2)) + 1e-6))
        t.append((i + n / 2) / sr)
    return np.array(t), np.array(v)


def f0_autocorr(y, sr, win=0.04, hop=0.01):
    n, h = int(win * sr), int(hop * sr)
    lo, hi = int(sr / F0_MAX), int(sr / F0_MIN)
    t, f = [], []
    for i in range(0, len(y) - n, h):
        fr = y[i:i + n] * np.hanning(n)
        fr = fr - fr.mean()
        if np.sqrt(np.mean(fr ** 2)) < 0.01:        # silence
            t.append((i + n / 2) / sr); f.append(np.nan); continue
        ac = np.correlate(fr, fr, "full")[n - 1:]
        seg = ac[lo:hi]
        if seg.size == 0 or ac[0] == 0:
            t.append((i + n / 2) / sr); f.append(np.nan); continue
        lag = lo + np.argmax(seg)
        t.append((i + n / 2) / sr)
        f.append(sr / lag if ac[lag] / ac[0] > 0.3 else np.nan)  # voicing gate
    return np.array(t), np.array(f)


def speaker_at(turns, t):
    r = turns[(turns.t_start <= t) & (turns.t_end > t)]
    return r.iloc[0]["speaker"] if len(r) else None


# ---------- Figure 1: example segment ----------
def fig_example():
    sr, y = wavfile.read(WAV_DIR / f"{EX_DYAD}_{EX_COND}_discussion_16k.wav")
    y = y.astype(float)
    y /= (np.max(np.abs(y)) or 1)
    a, b = int(T0 * sr), int(T1 * sr)
    yw = y[a:b]
    tw = np.arange(len(yw)) / sr + T0

    turns = pd.read_csv(PROJECT / "04_data" / "labeled_turns.csv")
    turns = turns[(turns.dyad_id == EX_DYAD) & (turns.condition == EX_COND)]
    seg_turns = turns[(turns.t_end > T0) & (turns.t_start < T1)].sort_values("t_start")

    ti, iv = rms_db(yw, sr)
    tf, f0 = f0_autocorr(yw, sr)
    ti += T0; tf += T0

    fig, ax = plt.subplots(3, 1, figsize=(11, 7), sharex=True,
                           gridspec_kw={"height_ratios": [1, 1.2, 1.2]})

    def shade(axx):
        for _, r in seg_turns.iterrows():
            axx.axvspan(max(r.t_start, T0), min(r.t_end, T1),
                        color=C.get(r.speaker, "#999"), alpha=0.08)
            x = max(r.t_start, T0)
            if x > T0:
                axx.axvline(x, color="k", lw=0.8, alpha=0.4)

    # waveform
    ax[0].plot(tw, yw, lw=0.4, color="#444")
    shade(ax[0]); ax[0].set_ylabel("waveform"); ax[0].set_yticks([])
    ax[0].set_title(f"Vocal alignment at turn transitions — {EX_DYAD} ({EX_COND}), "
                    f"{T0:.0f}–{T1:.0f}s", fontsize=12)

    # intensity, colored by current speaker
    for _, r in seg_turns.iterrows():
        m = (ti >= r.t_start) & (ti < r.t_end)
        ax[1].plot(ti[m], iv[m], color=C.get(r.speaker, "#999"), lw=2)
    shade(ax[1]); ax[1].set_ylabel("intensity (dB)")

    # mark edge windows + their mean level at each handoff
    sw = seg_turns.reset_index(drop=True)
    for k in range(1, len(sw)):
        if sw.loc[k, "speaker"] == sw.loc[k - 1, "speaker"]:
            continue
        bnd = sw.loc[k, "t_start"]
        for (lo, hi, spk) in [(bnd - EDGE, bnd, sw.loc[k - 1, "speaker"]),
                              (bnd, bnd + EDGE, sw.loc[k, "speaker"])]:
            m = (ti >= max(lo, T0)) & (ti < min(hi, T1))
            if m.any():
                lvl = np.nanmean(iv[m])
                ax[1].hlines(lvl, max(lo, T0), min(hi, T1),
                             color=C.get(spk), lw=4, alpha=0.9)
        ax[1].annotate("handoff", (bnd, ax[1].get_ylim()[1]),
                       ha="center", va="top", fontsize=8, color="k")

    # f0
    for _, r in seg_turns.iterrows():
        m = (tf >= r.t_start) & (tf < r.t_end)
        ax[2].plot(tf[m], f0[m], ".", ms=3, color=C.get(r.speaker, "#999"))
    shade(ax[2]); ax[2].set_ylabel("pitch f0 (Hz)"); ax[2].set_xlabel("time (s)")
    ax[2].set_ylim(F0_MIN, F0_MAX)

    fig.legend(handles=[Patch(color=C["SPEAKER_00"], label="Speaker 00"),
                        Patch(color=C["SPEAKER_01"], label="Speaker 01"),
                        plt.Line2D([0], [0], color="k", lw=4,
                                   label="turn-edge window (0.5s)")],
               loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"audio_alignment_example.{ext}", dpi=150,
                    bbox_inches="tight")
    print("Wrote", FIG / "audio_alignment_example.png")


# ---------- Figure 2: dataset-level offset->onset scatter ----------
def fig_scatter():
    d = pd.read_csv(PROJECT / "04_data" / "acoustic_turns.csv")
    rows = []
    for (dy, cond), g in d.groupby(["dyad_id", "condition"]):
        g = g.sort_values("t_start").reset_index(drop=True)
        # z-score offset/onset intensity within speaker (per session)
        for col in ["offset_int", "onset_int"]:
            g[col + "_z"] = g.groupby("speaker")[col].transform(
                lambda s: (s - s.mean()) / (s.std(ddof=0) or np.nan))
        for i in range(1, len(g)):
            if g.loc[i, "speaker"] != g.loc[i - 1, "speaker"]:
                rows.append((g.loc[i - 1, "offset_int_z"], g.loc[i, "onset_int_z"]))
    arr = np.array([r for r in rows if np.isfinite(r[0]) and np.isfinite(r[1])])
    x, y = arr[:, 0], arr[:, 1]
    from scipy import stats
    r, p = stats.pearsonr(x, y)
    sl, ic = np.polyfit(x, y, 1)

    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    ax.scatter(x, y, s=10, alpha=0.25, color="#2c7fb8", edgecolors="none")
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, sl * xs + ic, color="#d95f0e", lw=2,
            label=f"r = {r:+.2f} (p={p:.1e})")
    ax.set_xlabel("Speaker A — turn-OFFSET intensity (z)")
    ax.set_ylabel("Speaker B — turn-ONSET intensity (z)")
    ax.set_title(f"Intensity matching at turn transitions\n"
                 f"all speaker switches, N={len(arr)}", fontsize=11)
    ax.axhline(0, color="#ccc", lw=0.6); ax.axvline(0, color="#ccc", lw=0.6)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"intensity_edge_scatter.{ext}", dpi=150,
                    bbox_inches="tight")
    print("Wrote", FIG / "intensity_edge_scatter.png", f"| r={r:+.2f} p={p:.2g}")


if __name__ == "__main__":
    fig_example()
    fig_scatter()
