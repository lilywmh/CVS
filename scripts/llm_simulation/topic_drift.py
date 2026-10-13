#!/usr/bin/env python3
"""
Topic drift in real vs LLM-simulated CVS conversations
======================================================

Purpose
-------
Measures how far a stretch of talk is from the film being discussed, without
any hand or LLM labeling, and compares real strangers with LLM-simulated ones.

Film score of a window w (an utterance plus up to WINDOW-1 preceding utterances
in the same question):

    film_score(w) = cos(e_w, e_own_film) - cos(e_w, e_other_film)

e = sentence embedding; the anchors are fixed short synopses of the two films
(FILM_ANCHORS below). Subtracting the other film removes what all movie talk
shares. High = talk specific to this film; near zero = about neither film
(drift). Similarity to the question text is reported alongside, because Q2
("did anything remind you of your own life?") invites personal stories.

Subcommands
-----------
human     Score every real window in all 46 discussions.
          RQ1 description, a validation sheet (highest / lowest windows), and the
          exploratory dyad-level link to connection outcomes.
compare   Score one simulation run made by sim_partner_turing.py.
          --run <topic run>: RQ2, segment film score and within-segment drift
                             slope, real vs simulated (paired, dyad-level tests).
          --run <turn run> : RQ3, pull-back: the same preceding context ending in
                             the simulated vs the real reply; also "following"
                             (reply score regressed on context score).

Backends
--------
--backend st     sentence-transformers model (default all-MiniLM-L6-v2);
                 run on CARC with weights pre-downloaded.
--backend tfidf  dependency-free lexical TF-IDF version of the same score;
                 a robustness check, and lets the pipeline run anywhere.

Inputs
------
04_data/acoustic_turns.csv, 04_data/outcomes.csv (human),
05_analysis_outputs/llm_sim_partner/<run>/ (compare)

Outputs (new timestamped dir, never overwritten)
-------
05_analysis_outputs/llm_sim_partner/drift_<human|topic|turn>_<backend>_<ts>/
    manifest.json, summary.md, plus per-subcommand CSVs:
    human  : windows_human.csv, validation_extremes.csv, connection_exploratory.csv,
             trajectory_human.csv
    topic  : segments.csv, trajectory_real_vs_sim.csv, trajectory.png (if matplotlib)
    turn   : items.csv

Example commands
----------------
python scripts/llm_simulation/topic_drift.py human --backend tfidf
python scripts/llm_simulation/topic_drift.py human --backend st
python scripts/llm_simulation/topic_drift.py compare --run 05_analysis_outputs/llm_sim_partner/<topic run>
python scripts/llm_simulation/topic_drift.py compare --run 05_analysis_outputs/llm_sim_partner/<turn run> \
    --backend st --model sentence-transformers/all-mpnet-base-v2 --window 5
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim_partner_turing as spt  # noqa: E402
import topic_lexicon as lex  # noqa: E402

FILM_ANCHORS = {
    "piper": (
        "Piper, a Pixar short film. A baby sandpiper bird lives on a beach with its "
        "mother and the flock. The birds run to the shore to dig for food in the wet "
        "sand when the waves go out. A big wave crashes over the little bird and it "
        "becomes afraid of the water and the ocean. The mother encourages it. The "
        "baby bird meets tiny hermit crabs that burrow into the sand, copies them, "
        "sees the shells and clams under the water, overcomes its fear, finds lots "
        "of food, and feeds the other birds."
    ),
    "cloudy": (
        "Partly Cloudy, a Pixar short film. Clouds in the sky make babies, and storks "
        "deliver them to families: puppies, kittens, lambs and human babies. A dark "
        "grey storm cloud named Gus makes dangerous babies: a baby alligator, a ram, "
        "a porcupine, an electric eel and a shark. His loyal stork Peck delivers them "
        "and keeps getting bitten, butted, poked and shocked. Peck flies away to "
        "another cloud, and Gus is sad and angry, thinking he was abandoned, but Peck "
        "comes back wearing football pads and a helmet, and Gus gives him a baby shark."
    ),
}
OTHER = {"piper": "cloudy", "cloudy": "piper"}
EMB_LABEL_MIN_WORDS = 8


def keyword_features(text: str, condition: str) -> dict:
    """Lexicon v1 core hits, oriented to the film being discussed."""
    h = lex.keyword_hits(text)
    off = {f"kw_{c}": h[c] for c in lex.OFF_FILM_CATS}
    return {"kw_own_film": h[lex.OWN_FILM[condition]],
            "kw_other_film": h[lex.OTHER_FILM[condition]],
            "kw_film_general": h["film_general"],
            "kw_off_film": sum(off.values()), **off,
            "kw_ambiguous": sum(h[f"{c}_amb"] for c in lex.LEXICON)}


def auc(pos: pd.Series, neg: pd.Series) -> float:
    """Probability that a random 'pos' value exceeds a random 'neg' value
    (Mann-Whitney AUC, ties count half)."""
    pos, neg = pos.dropna().to_numpy(), neg.dropna().to_numpy()
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    allv = np.concatenate([pos, neg])
    ranks = pd.Series(allv).rank().to_numpy()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
TOKEN_RE = re.compile(r"[a-z']+")


# --------------------------------------------------------------------------- #
# Embedding backends                                                          #
# --------------------------------------------------------------------------- #
class TfidfBackend:
    """Plain TF-IDF over unigrams, fit on every text it is asked to encode
    (windows + anchors), English stop words removed, sublinear tf, L2-normed."""

    name = "tfidf"

    def encode(self, texts: list[str]) -> np.ndarray:
        docs = [[w for w in TOKEN_RE.findall(t.lower()) if w not in spt.STOP and len(w) > 2]
                for t in texts]
        df = Counter(w for d in docs for w in set(d))
        vocab = {w: i for i, w in enumerate(sorted(df))}
        n = len(docs)
        idf = np.array([math.log((1 + n) / (1 + df[w])) + 1 for w in sorted(df)])
        mat = np.zeros((n, len(vocab)), dtype=np.float32)
        for r, d in enumerate(docs):
            for w, c in Counter(d).items():
                mat[r, vocab[w]] = (1 + math.log(c)) * idf[vocab[w]]
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return mat / norms


class STBackend:
    name = "st"

    def __init__(self, model: str, device: str | None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            sys.exit("sentence-transformers is not installed. On CARC: conda activate "
                     "cvs-vllm && pip install sentence-transformers. Or use --backend tfidf.")
        self.model = SentenceTransformer(model, device=device)
        self.name = f"st:{model}"

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, batch_size=64, normalize_embeddings=True,
                                            show_progress_bar=False), dtype=np.float32)


class Scorer:
    """Scores many windows in one pass so TF-IDF is fit on a shared vocabulary."""

    def __init__(self, backend):
        self.backend = backend

    def score(self, rows: list[dict], return_embeddings: bool = False):
        """rows: dicts with 'text', 'condition', 'question'. Adds film_score,
        own_sim, other_sim, question_sim.

        If *return_embeddings* is True, returns (DataFrame, emb_matrix, question_embs)
        where emb_matrix is (n_rows, dim) and question_embs maps question int
        to its anchor embedding vector.
        """
        anchors = {f"film:{c}": t for c, t in FILM_ANCHORS.items()}
        anchors |= {f"q:{q}": t for q, t in spt.QUESTIONS.items()}
        anchors |= {f"cat:{c}": t for c, t in lex.CATEGORY_ANCHORS.items()}
        keys = list(anchors)
        uniq = list(dict.fromkeys(r["text"] for r in rows))
        emb = self.backend.encode([anchors[k] for k in keys] + uniq)
        a = {k: emb[i] for i, k in enumerate(keys)}
        idx = {t: len(keys) + i for i, t in enumerate(uniq)}
        out = []
        row_embs = []
        for r in rows:
            e = emb[idx[r["text"]]]
            own = float(e @ a[f"film:{r['condition']}"])
            oth = float(e @ a[f"film:{OTHER[r['condition']]}"])
            qs = float(e @ a[f"q:{int(r['question'])}"]) if f"q:{int(r['question'])}" in a else np.nan
            cats = {c: float(e @ a[f"cat:{c}"]) for c in lex.OFF_FILM_CATS}
            top_cat = max(cats, key=cats.get)
            n_words = r.get("n_words", len(spt.words(r["text"])))
            if n_words < EMB_LABEL_MIN_WORDS:
                label = "short"
            else:
                label = "film" if own >= cats[top_cat] else top_cat
            out.append({**r, "own_sim": own, "other_sim": oth,
                        "film_score": own - oth, "question_sim": qs,
                        "offfilm_sim": cats[top_cat], "offfilm_top": top_cat,
                        "emb_label": label, **keyword_features(r["text"], r["condition"])})
            row_embs.append(e)
        df = pd.DataFrame(out)
        if return_embeddings:
            emb_matrix = np.stack(row_embs)
            question_embs = {}
            for q in spt.QUESTIONS:
                k = f"q:{q}"
                if k in a:
                    question_embs[q] = a[k]
            return df, emb_matrix, question_embs
        return df


def compute_walk_metrics(df: pd.DataFrame, emb_matrix: np.ndarray,
                         question_embs: dict[int, np.ndarray],
                         segment_keys: list[str] | None = None) -> pd.DataFrame:
    """Add step_size, cumulative_walk, question_drift per window within each
    question segment.

    *df* and *emb_matrix* must be row-aligned.  *segment_keys* defaults to
    ['dyad_id', 'condition', 'question'] (human) but callers can override
    (e.g. add 'source' or 'item_id').
    """
    if segment_keys is None:
        segment_keys = ["dyad_id", "condition", "question"]
    step_size = np.full(len(df), np.nan)
    cumulative_walk = np.full(len(df), np.nan)
    question_drift = np.full(len(df), np.nan)
    for _, idxs in df.groupby(segment_keys, sort=False).groups.items():
        idxs = sorted(idxs)  # preserve window order
        q = int(df.loc[idxs[0], "question"])
        q_emb = question_embs.get(q)
        cum = 0.0
        for j, ix in enumerate(idxs):
            e = emb_matrix[ix]
            if j == 0:
                step_size[ix] = 0.0
            else:
                prev = emb_matrix[idxs[j - 1]]
                norm_prod = (np.linalg.norm(e) * np.linalg.norm(prev))
                cos_sim = float(e @ prev / norm_prod) if norm_prod > 0 else 1.0
                step_size[ix] = max(0.0, 1.0 - cos_sim)
            cum += step_size[ix]
            cumulative_walk[ix] = cum
            if q_emb is not None:
                norm_prod = np.linalg.norm(e) * np.linalg.norm(q_emb)
                question_drift[ix] = 1.0 - float(e @ q_emb / norm_prod) if norm_prod > 0 else np.nan
            else:
                question_drift[ix] = np.nan
    df = df.copy()
    df["step_size"] = step_size
    df["cumulative_walk"] = cumulative_walk
    df["question_drift"] = question_drift
    return df


def make_backend(a) -> object:
    if a.backend == "tfidf":
        return TfidfBackend()
    device = None if a.device == "auto" else a.device
    return STBackend(a.model, device)


# --------------------------------------------------------------------------- #
# Windows                                                                     #
# --------------------------------------------------------------------------- #
def segment_windows(units: list[dict], window: int) -> list[dict]:
    """units: [{'role','text'}] of ONE question segment, in order. One window per
    unit: that unit plus up to window-1 preceding units. pos = 0..1 in segment."""
    n = len(units)
    out = []
    for i in range(n):
        chunk = units[max(0, i - window + 1): i + 1]
        out.append({"pos_idx": i, "pos": i / (n - 1) if n > 1 else 0.0,
                    "text": " ".join(u["text"] for u in chunk),
                    "n_words": sum(len(spt.words(u["text"])) for u in chunk)})
    return out


def slope(x: pd.Series, y: pd.Series) -> float:
    m = x.notna() & y.notna()
    if m.sum() < 3 or x[m].nunique() < 2:
        return np.nan
    return float(np.polyfit(x[m].astype(float), y[m].astype(float), 1)[0])


def new_out_dir(kind: str, backend_name: str) -> Path:
    tag = re.sub(r"[^A-Za-z0-9]+", "-", backend_name.split("/")[-1]).strip("-")
    out = spt.OUT_ROOT / f"drift_{kind}_{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=False)
    return out


def write_manifest(out: Path, a, extra: dict) -> None:
    (out / "manifest.json").write_text(json.dumps({
        "script": str(Path(__file__).resolve().relative_to(spt.PROJECT)),
        "created": datetime.now().isoformat(timespec="seconds"),
        "args": {k: str(v) for k, v in vars(a).items()},
        "film_anchors": FILM_ANCHORS, **extra}, indent=2))


def fmt(df: pd.DataFrame) -> str:
    return df.round(3).to_markdown(index=False)


def tests_row(label: str, d: pd.DataFrame, col: str, null: float = 0.0) -> dict:
    mean, lo, hi = spt.cluster_boot_mean(d, col)
    p, nd = spt.dyad_signflip(d, col, null=null)
    return {"measure": label, "mean": mean, "95% CI low": lo, "95% CI high": hi,
            "p (dyad sign-flip)": p, "n": int(d[col].notna().sum()), "n dyads": nd}


# --------------------------------------------------------------------------- #
# human                                                                       #
# --------------------------------------------------------------------------- #
def run_human(a) -> None:
    units = spt.load_floor_units(spt.INPUT_CSV).sort_values(["dyad_id", "condition", "unit"])
    rows = []
    for (d, c, q), g in units.groupby(["dyad_id", "condition", "question"], sort=False):
        segs = [{"role": r.role, "text": r.text} for r in g.itertuples()]
        for w in segment_windows(segs, a.window):
            rows.append({"dyad_id": d, "pair_id": int(g.pair_id.iloc[0]), "condition": c,
                         "order": g.order.iloc[0], "question": int(q),
                         "seg_n_units": len(segs), **w})
    scorer = Scorer(make_backend(a))
    df, emb_matrix, question_embs = scorer.score(rows, return_embeddings=True)
    df = df[df.n_words >= a.min_words].copy()
    keep_idx = df.index.to_numpy()
    emb_matrix = emb_matrix[keep_idx]
    df = df.reset_index(drop=True)
    df = compute_walk_metrics(df, emb_matrix, question_embs)
    out = new_out_dir("human", scorer.backend.name)
    df.to_csv(out / "windows_human.csv", index=False)

    # RQ1 tables
    by_q = df.groupby("question").agg(windows=("film_score", "size"),
                                      film_score=("film_score", "mean"),
                                      question_sim=("question_sim", "mean")).reset_index()
    df["third"] = pd.cut(df.pos, [-0.01, 1 / 3, 2 / 3, 1.0], labels=["early", "middle", "late"])
    by_pos = df.groupby("third", observed=True).film_score.agg(["size", "mean"]).reset_index()
    by_order = df.groupby("order").film_score.agg(["size", "mean"]).reset_index()
    seg = (df[df.seg_n_units >= 4].groupby(["dyad_id", "condition", "question"])
             .apply(lambda g: slope(g.pos, g.film_score), include_groups=False)
             .rename("drift_slope").reset_index())
    q1_med = df[df.question == 1].groupby(["dyad_id", "condition"]).film_score.median().rename("q1_median")
    df = df.join(q1_med, on=["dyad_id", "condition"])
    df["below_q1_median"] = (df.film_score < df.q1_median).astype(float)
    below = df[df.question > 1].groupby("question").below_q1_median.mean().reset_index()
    traj = (df.assign(bin=pd.cut(df.pos, 5, labels=False)).groupby(["question", "bin"])
              .film_score.mean().reset_index())
    traj.to_csv(out / "trajectory_human.csv", index=False)

    # walk metrics: question-relative random walk
    walk_by_q = df.groupby("question").agg(
        mean_cumulative_walk=("cumulative_walk", "mean"),
        mean_question_drift=("question_drift", "mean"),
        mean_step_size=("step_size", "mean")).reset_index()
    # walk slope: does question_drift increase within a segment?
    walk_seg = (df[df.seg_n_units >= 4].groupby(["dyad_id", "condition", "question"])
                  .apply(lambda g: slope(g.pos, g.question_drift), include_groups=False)
                  .rename("qd_slope").reset_index())
    walk_seg_test = tests_row("question-drift slope per segment", walk_seg, "qd_slope")
    # trajectory by question for walk metrics
    walk_traj = (df.assign(bin=pd.cut(df.pos, 5, labels=False))
                   .groupby(["question", "bin"])
                   .agg(question_drift=("question_drift", "mean"),
                        cumulative_walk=("cumulative_walk", "mean")).reset_index())
    walk_traj.to_csv(out / "trajectory_walk_human.csv", index=False)

    # keyword validation of the film score (lexicon v1, core terms only)
    own_only = df[(df.kw_own_film > 0) & (df.kw_off_film == 0)]
    off_only = df[(df.kw_off_film > 0) & (df.kw_own_film == 0)]
    kw_groups = pd.DataFrame([
        {"group": "names this film's things only", "windows": len(own_only),
         "mean film score": own_only.film_score.mean()},
        {"group": "names off-film topics only", "windows": len(off_only),
         "mean film score": off_only.film_score.mean()},
        {"group": "both", "windows": int(((df.kw_own_film > 0) & (df.kw_off_film > 0)).sum()),
         "mean film score": df[(df.kw_own_film > 0) & (df.kw_off_film > 0)].film_score.mean()},
        {"group": "neither", "windows": int(((df.kw_own_film == 0) & (df.kw_off_film == 0)).sum()),
         "mean film score": df[(df.kw_own_film == 0) & (df.kw_off_film == 0)].film_score.mean()}])
    kw_auc = auc(own_only.film_score, off_only.film_score)
    # where off-film talk goes: units (not overlapping windows) with any core hit
    ucat = []
    for r in units.itertuples():
        f = keyword_features(r.text, r.condition)
        ucat.append({"question": int(r.question), "n_words": r.n_words,
                     **{c: int(f[f"kw_{c}"] > 0) for c in lex.OFF_FILM_CATS},
                     "any_off_film": int(f["kw_off_film"] > 0),
                     "own_film": int(f["kw_own_film"] > 0)})
    ucat = pd.DataFrame(ucat)
    where = (ucat.groupby("question")[["own_film", "any_off_film"] + lex.OFF_FILM_CATS].mean()
                 .mul(100).round(1).reset_index())
    emb_lab = (df[df.emb_label != "short"].groupby("question").emb_label
                 .value_counts(normalize=True).mul(100).round(1).unstack(fill_value=0).reset_index())

    # validation sheet: extremes by film score (text included; stays local)
    ext = pd.concat([df.nlargest(20, "film_score").assign(rank_group="highest"),
                     df.nsmallest(20, "film_score").assign(rank_group="lowest")])
    ext[["rank_group", "dyad_id", "condition", "question", "pos", "film_score",
         "question_sim", "text"]].assign(looks_right="", notes="").to_csv(
        out / "validation_extremes.csv", index=False)

    # exploratory link to connection outcomes (dyad level)
    conn_md = ["Outcomes file not found; skipped."]
    oc_path = spt.PROJECT / "04_data" / "outcomes.csv"
    if oc_path.exists():
        oc = pd.read_csv(oc_path)
        dy = df.groupby("pair_id").agg(mean_film_score=("film_score", "mean"),
                                       off_film_window_share=("kw_off_film", lambda x: (x > 0).mean())
                                       ).reset_index()
        m = dy.merge(oc, on="pair_id", how="inner")
        rows_c = []
        rng = np.random.default_rng(42)
        for pred in ["mean_film_score", "off_film_window_share"]:
            for col in ["dyad_partner_eval_mean", "dyad_shared_reality_mean", "dyad_enjoyment_mean"]:
                if col not in m:
                    continue
                sub = m[[pred, col]].dropna()
                rho = sub.rank().corr().iloc[0, 1]
                boots = [sub.iloc[rng.integers(0, len(sub), len(sub))].rank().corr().iloc[0, 1]
                         for _ in range(2000)]
                lo, hi = np.nanpercentile(boots, [2.5, 97.5])
                rows_c.append({"drift measure": pred, "outcome": col, "spearman_rho": rho,
                               "95% CI low": lo, "95% CI high": hi, "n dyads": len(sub)})
        conn = pd.DataFrame(rows_c)
        conn.to_csv(out / "connection_exploratory.csv", index=False)
        conn_md = ["For mean_film_score, negative rho = dyads that drift more report more of the "
                   "outcome; for off_film_window_share, positive rho means the same. That is the "
                   "direction Speer et al. (2024) predict. Exploratory: "
                   f"{len(m)} dyads, effect sizes only.", "", fmt(conn)]

    seg_test = tests_row("within-segment drift slope (film score per segment)", seg, "drift_slope")
    md = [f"# Topic drift, human data ({scorer.backend.name}, window {a.window})", "",
          f"- windows scored: {len(df)} (min {a.min_words} words) from "
          f"{df.groupby(['dyad_id', 'condition']).ngroups} discussions, {df.dyad_id.nunique()} dyads",
          "- film score = cos(window, own film) - cos(window, other film); lower = further from the film",
          "", "## Film score by question", "", fmt(by_q), "",
          "## Film score by position within the question", "", fmt(by_pos), "",
          "## Does film score fall across a question? (segments with >= 4 units)", "",
          fmt(pd.DataFrame([seg_test])), "", "Negative mean = talk moves away from the film as a question goes on.", "",
          "## Share of windows below the discussion's own Q1 median", "", fmt(below), "",
          "## First vs second film", "", fmt(by_order), "",
          f"## Keyword check of the film score (lexicon {lex.VERSION}, core terms)", "",
          fmt(kw_groups), "",
          f"AUC (film score separates 'this film only' from 'off-film only' windows): {kw_auc:.3f}. "
          "0.5 = no separation, 1 = perfect. Validates the high end of the score; "
          "'neither' windows are mostly generic reactions.", "",
          "## Where talk goes: % of utterances naming each topic (core keywords)", "",
          fmt(where), "",
          f"## Embedding labels by question (% of windows with >= {EMB_LABEL_MIN_WORDS} words; "
          "label = film if closer to this film's synopsis than to any off-film topic)", "",
          fmt(emb_lab), "",
          "## Question-relative random walk", "",
          "These metrics are question-normalized: each question's anchor is its own text, "
          "so Q2's 'personal life' baseline is separated from Q1's 'film reaction' baseline.", "",
          "- step_size = cosine distance between consecutive window embeddings",
          "- cumulative_walk = running sum of step_size (total path length)",
          "- question_drift = cosine distance from the question anchor", "",
          "### Walk metrics by question", "", fmt(walk_by_q), "",
          "### Does question drift increase within a segment? (>= 4 units)", "",
          fmt(pd.DataFrame([walk_seg_test])), "",
          "Positive mean = conversation wanders further from the question over time.", "",
          "### Walk trajectory: `trajectory_walk_human.csv`", "",
          "## Exploratory: drift and connection (dyad level)", ""] + conn_md + [
          "", "## Validation", "",
          "Open `validation_extremes.csv`: the 20 highest and 20 lowest windows. Fill `looks_right` "
          "(Y/N): highest should be about this film, lowest should be about something else."]
    (out / "summary.md").write_text("\n".join(md) + "\n")
    write_manifest(out, a, {"input": str(spt.INPUT_CSV.relative_to(spt.PROJECT)),
                            "input_sha256": spt.sha256(spt.INPUT_CSV), "n_windows": len(df),
                            "lexicon_version": lex.VERSION, "keyword_auc": kw_auc})
    print(f"human: {len(df)} windows -> {out}")


# --------------------------------------------------------------------------- #
# compare                                                                     #
# --------------------------------------------------------------------------- #
def run_compare(a) -> None:
    run = a.run.resolve()
    cfg = json.loads((run / "manifest.json").read_text())["config"]
    items = {it["item_id"]: it for it in spt.read_jsonl(run / "items.jsonl")}
    act = {r["item_id"]: r for r in spt.read_jsonl(run / "actor.jsonl")}
    if not act:
        sys.exit(f"No simulations in {run}/actor.jsonl")
    scorer = Scorer(make_backend(a))
    if cfg["level"] == "topic":
        compare_topic(a, run, items, act, scorer)
    else:
        compare_turn(a, run, items, act, scorer)


def compare_topic(a, run, items, act, scorer) -> None:
    rows = []
    for iid, r in act.items():
        it = items.get(iid)
        if not it or not r.get("sim_segment"):
            continue
        for src, seg in (("real", it["real_segment"]), ("sim", r["sim_segment"])):
            for w in segment_windows(seg, a.window):
                rows.append({"item_id": iid, "dyad_id": it["dyad_id"], "condition": it["condition"],
                             "question": it["question"], "source": src, "seg_n_units": len(seg), **w})
    df, emb_matrix, question_embs = scorer.score(rows, return_embeddings=True)
    df = df[df.n_words >= a.min_words].copy()
    keep_idx = df.index.to_numpy()
    emb_matrix = emb_matrix[keep_idx]
    df = df.reset_index(drop=True)
    df = compute_walk_metrics(df, emb_matrix, question_embs,
                              segment_keys=["item_id", "source", "dyad_id", "condition", "question"])
    out = new_out_dir("topic", scorer.backend.name)

    per = []
    for (iid, src), g in df.groupby(["item_id", "source"]):
        per.append({"item_id": iid, "source": src, "dyad_id": g.dyad_id.iloc[0],
                    "question": g.question.iloc[0], "condition": g.condition.iloc[0],
                    "mean_film_score": g.film_score.mean(),
                    "drift_slope": slope(g.pos, g.film_score) if g.seg_n_units.iloc[0] >= 4 else np.nan,
                    "mean_question_sim": g.question_sim.mean(),
                    "mean_cumulative_walk": g.cumulative_walk.iloc[-1] if len(g) else np.nan,
                    "mean_question_drift": g.question_drift.mean(),
                    "qd_slope": slope(g.pos, g.question_drift) if g.seg_n_units.iloc[0] >= 4 else np.nan})
    per = pd.DataFrame(per)
    kw = []
    for iid, r in act.items():
        it = items.get(iid)
        if not it or not r.get("sim_segment"):
            continue
        for src, seg in (("real", it["real_segment"]), ("sim", r["sim_segment"])):
            fs = [keyword_features(u["text"], it["condition"]) for u in seg]
            row = {"item_id": iid, "source": src,
                   "off_film_unit_share": np.mean([f["kw_off_film"] > 0 for f in fs]),
                   "own_film_unit_share": np.mean([f["kw_own_film"] > 0 for f in fs])}
            row |= {f"{c}_unit_share": np.mean([f[f"kw_{c}"] > 0 for f in fs]) for c in lex.OFF_FILM_CATS}
            kw.append(row)
    per = per.merge(pd.DataFrame(kw), on=["item_id", "source"], how="left")
    walk_cols = ["mean_cumulative_walk", "mean_question_drift", "qd_slope"]
    kw_cols = ["off_film_unit_share", "own_film_unit_share"] + [f"{c}_unit_share" for c in lex.OFF_FILM_CATS]
    wide = per.pivot_table(index=["item_id", "dyad_id", "question", "condition"], columns="source",
                           values=["mean_film_score", "drift_slope", "mean_question_sim"] + walk_cols + kw_cols
                           ).reset_index()
    wide.columns = ["_".join(c).strip("_") for c in wide.columns]
    for m in ["mean_film_score", "drift_slope", "mean_question_sim"] + walk_cols + kw_cols:
        wide[f"{m}_diff"] = wide.get(f"{m}_sim") - wide.get(f"{m}_real")
    wide.to_csv(out / "segments.csv", index=False)

    tests = pd.DataFrame([
        tests_row("segment film score, sim - real", wide, "mean_film_score_diff"),
        tests_row("drift slope, sim - real", wide, "drift_slope_diff"),
        tests_row("question similarity, sim - real", wide, "mean_question_sim_diff"),
        tests_row("share of utterances naming off-film topics, sim - real", wide, "off_film_unit_share_diff"),
        tests_row("share of utterances naming this film's things, sim - real", wide, "own_film_unit_share_diff"),
    ])
    walk_tests = pd.DataFrame([
        tests_row("cumulative walk (path length), sim - real", wide, "mean_cumulative_walk_diff"),
        tests_row("question drift, sim - real", wide, "mean_question_drift_diff"),
        tests_row("question-drift slope, sim - real", wide, "qd_slope_diff"),
    ])
    cat_tab = pd.DataFrame([{"topic": c, "real %": 100 * wide[f"{c}_unit_share_real"].mean(),
                             "simulated %": 100 * wide[f"{c}_unit_share_sim"].mean()}
                            for c in lex.OFF_FILM_CATS])
    means = pd.DataFrame([{"source": s, "segment film score": wide[f"mean_film_score_{s}"].mean(),
                           "drift slope": wide[f"drift_slope_{s}"].mean()} for s in ("real", "sim")])
    by_q = wide.groupby("question")[["mean_film_score_real", "mean_film_score_sim"]].mean().reset_index()

    traj = (df.assign(bin=pd.cut(df.pos, 5, labels=False)).groupby(["source", "bin"])
              .film_score.agg(["mean", "size"]).reset_index())
    traj.to_csv(out / "trajectory_real_vs_sim.csv", index=False)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 3.6))
        for s, lab in (("real", "Real strangers"), ("sim", "LLM agents")):
            t = traj[traj.source == s]
            ax.plot((t["bin"] + 0.5) / 5, t["mean"], marker="o", label=lab)
        ax.set_xlabel("Position within the question (0 = start, 1 = end)")
        ax.set_ylabel("Film score (own - other film)")
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(out / "trajectory.png", dpi=200)
    except ImportError:
        pass

    md = [f"# Topic drift, topic level: {run.name} ({scorer.backend.name}, window {a.window})", "",
          f"- actor `{json.loads((run / 'manifest.json').read_text())['config']['actor_model']}`; "
          f"segments with both versions: {wide.mean_film_score_diff.notna().sum()} from "
          f"{wide.dyad_id.nunique()} dyads",
          "- film score: lower = further from the film. Drift slope: negative = moves away "
          "from the film across the segment.", "",
          "## Real vs simulated (means)", "", fmt(means), "",
          "## Paired tests, simulated minus real", "", fmt(tests), "",
          "RQ2 prediction: segment film score diff > 0 (LLM stays closer to the film) and "
          "drift slope diff > 0 (LLM drifts less).", "",
          "## Segment film score by question", "", fmt(by_q), "",
          f"## Off-film topics named (lexicon {lex.VERSION}, % of utterances)", "", fmt(cat_tab), "",
          "## Question-relative random walk (sim vs real)", "",
          "- cumulative_walk: total path length in embedding space within the segment",
          "- question_drift: cosine distance from the question anchor (comparable across questions)",
          "- qd_slope: does question_drift increase within the segment?", "",
          "### Paired tests, simulated minus real", "", fmt(walk_tests), "",
          "Negative cumulative_walk diff = LLM wanders less. "
          "Negative question_drift diff = LLM stays closer to the question. "
          "Negative qd_slope diff = LLM drifts from the question less.", "",
          "Trajectory by position: `trajectory_real_vs_sim.csv` and `trajectory.png`."]
    (out / "summary.md").write_text("\n".join(md) + "\n")
    write_manifest(out, a, {"run": str(run.relative_to(spt.PROJECT)), "n_segments": len(wide)})
    print(f"compare[topic]: {len(wide)} segments -> {out}")


def compare_turn(a, run, items, act, scorer) -> None:
    rows, meta = [], []
    for iid, r in act.items():
        it = items.get(iid)
        if not it or not r.get("sim_text"):
            continue
        q = it["question"]
        ctx = [h for h in it["history"] if h["question"] == q][-(a.window - 1):] if a.window > 1 else []
        if not ctx:
            ctx = [h for h in it["history"] if h["question"] == q][-1:]
        ctx_text = " ".join(h["text"] for h in ctx)
        base = {"item_id": iid, "condition": it["condition"], "question": q}
        rows += [{**base, "kind": "context", "text": ctx_text},
                 {**base, "kind": "real_win", "text": f"{ctx_text} {it['real_text']}".strip()},
                 {**base, "kind": "sim_win", "text": f"{ctx_text} {r['sim_text']}".strip()},
                 {**base, "kind": "real_reply", "text": it["real_text"]},
                 {**base, "kind": "sim_reply", "text": r["sim_text"]}]
        fr, fs = keyword_features(it["real_text"], it["condition"]), keyword_features(r["sim_text"], it["condition"])
        meta.append({"item_id": iid, "dyad_id": it["dyad_id"], "condition": it["condition"],
                     "question": q, "real_words": len(spt.words(it["real_text"])),
                     "sim_words": len(spt.words(r["sim_text"])),
                     "real_off_film_kw": int(fr["kw_off_film"] > 0), "sim_off_film_kw": int(fs["kw_off_film"] > 0),
                     "real_own_film_kw": int(fr["kw_own_film"] > 0), "sim_own_film_kw": int(fs["kw_own_film"] > 0)})
    sc, emb_matrix, question_embs = scorer.score(rows, return_embeddings=True)
    # compute question_drift per row (each row is a single window, not a segment sequence)
    qd_vals = []
    for i in range(len(sc)):
        e = emb_matrix[i]
        q = int(sc.iloc[i]["question"])
        q_emb = question_embs.get(q)
        if q_emb is not None:
            norm_prod = np.linalg.norm(e) * np.linalg.norm(q_emb)
            qd_vals.append(1.0 - float(e @ q_emb / norm_prod) if norm_prod > 0 else np.nan)
        else:
            qd_vals.append(np.nan)
    sc["question_drift"] = qd_vals
    wide = sc.pivot(index="item_id", columns="kind", values="film_score").add_suffix("_score")
    wide_qd = sc.pivot(index="item_id", columns="kind", values="question_drift").add_suffix("_qd")
    df = pd.DataFrame(meta).merge(wide.reset_index(), on="item_id").merge(wide_qd.reset_index(), on="item_id")
    df["pull_back"] = df.sim_win_score - df.real_win_score
    df["pull_back_reply_only"] = df.sim_reply_score - df.real_reply_score
    df["walk_pull_back"] = df.sim_win_qd - df.real_win_qd
    cut = df.context_score.quantile([1 / 3, 2 / 3]).to_list()
    df["context_third"] = pd.cut(df.context_score, [-np.inf, cut[0], cut[1], np.inf],
                                 labels=["off-film (lowest third)", "middle", "on-film (highest third)"])
    out = new_out_dir("turn", scorer.backend.name)
    df.to_csv(out / "items.csv", index=False)

    df["off_film_kw_diff"] = df.sim_off_film_kw - df.real_off_film_kw
    df["own_film_kw_diff"] = df.sim_own_film_kw - df.real_own_film_kw
    tests = [tests_row("pull-back, all test points", df, "pull_back"),
             tests_row("pull-back, off-film contexts (primary)",
                       df[df.context_third == "off-film (lowest third)"], "pull_back"),
             tests_row("pull-back, reply alone (secondary)", df, "pull_back_reply_only"),
             tests_row("reply names an off-film topic, sim - real", df, "off_film_kw_diff"),
             tests_row("reply names this film's things, sim - real", df, "own_film_kw_diff"),
             tests_row("walk pull-back (question drift, sim_win - real_win)", df, "walk_pull_back")]
    by_third = df.groupby("context_third", observed=True).agg(
        n=("pull_back", "size"), real_window=("real_win_score", "mean"),
        sim_window=("sim_win_score", "mean"), pull_back=("pull_back", "mean")).reset_index()

    # following: slope of reply-window score on context score, real vs sim, dyad bootstrap
    def fol(d):
        return slope(d.context_score, d.real_win_score), slope(d.context_score, d.sim_win_score)
    r_s, s_s = fol(df)
    rng = np.random.default_rng(42)
    groups = [g for _, g in df.groupby("dyad_id")]
    diffs = []
    for _ in range(2000):
        bs = pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        rr, ss = fol(bs)
        diffs.append(ss - rr)
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])

    md = [f"# Topic drift, turn level: {run.name} ({scorer.backend.name}, window {a.window})", "",
          f"- test points: {len(df)} from {df.dyad_id.nunique()} dyads",
          f"- pull-back = film score of [context + simulated reply] minus [same context + real reply]; "
          f"context = up to {a.window - 1} preceding utterances in the same question", "",
          "## Pull-back tests", "", fmt(pd.DataFrame(tests)), "",
          "RQ3 prediction: pull-back > 0 in off-film contexts (the LLM steers back to the film).", "",
          "## By how on-film the preceding context was", "", fmt(by_third), "",
          "## Following the context", "",
          f"Slope of reply-window film score on context film score: real {r_s:.3f}, simulated {s_s:.3f}; "
          f"difference (sim - real) {s_s - r_s:.3f}, 95% dyad-bootstrap CI [{lo:.3f}, {hi:.3f}]. "
          "A smaller simulated slope means the LLM's reply depends less on where the conversation "
          "already was.", "",
          "Note: the context is part of both windows, so pull-back isolates the reply. "
          "`pull_back_reply_only` scores the reply without context (short replies are noisy).", "",
          "## Question-relative walk pull-back", "",
          "walk_pull_back = question_drift of [context + sim reply] minus [context + real reply]. "
          "Negative = the LLM reply stays closer to the question anchor than the real reply did. "
          "This is question-normalized, so Q1 and Q2 are on comparable scales."]
    (out / "summary.md").write_text("\n".join(md) + "\n")
    write_manifest(out, a, {"run": str(run.relative_to(spt.PROJECT)), "n_items": len(df)})
    print(f"compare[turn]: {len(df)} test points -> {out}")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Topic drift: real vs LLM-simulated CVS conversations")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("human", "compare"):
        p = sub.add_parser(name)
        p.add_argument("--backend", default="st", choices=["st", "tfidf"])
        p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
        p.add_argument("--device", default="auto", help="auto, cpu or cuda")
        p.add_argument("--window", type=int, default=3, help="utterances per window")
        p.add_argument("--min-words", type=int, default=3,
                       help="drop windows with fewer words (embeddings of 1-2 words are noise)")
        if name == "compare":
            p.add_argument("--run", type=Path, required=True,
                           help="a run dir made by sim_partner_turing.py")
    a = ap.parse_args()
    if a.window < 1:
        sys.exit("--window must be >= 1")
    run_human(a) if a.cmd == "human" else run_compare(a)


if __name__ == "__main__":
    main()
