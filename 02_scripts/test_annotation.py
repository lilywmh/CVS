"""
Full-Scale LLM Annotation Pipeline
====================================
Runs turn-level and conversation-level annotation across all dyads and conditions.
Saves per-dyad turn CSVs + a collapsed dyad-level feature CSV for analysis.

Usage:
    python annotation_pipeline.py

Outputs (in OUTPUT_DIR):
    turns/          — one CSV per dyad-condition with all turn annotations
    dyad_features.csv  — collapsed dyad-level features ready for regression
    conversation_level.csv — engagement ratings per dyad-condition
"""

import os
import re
import glob
import json
import time
from pathlib import Path
import pandas as pd
import numpy as np
from anthropic import Anthropic

# ─── CONFIG ───────────────────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parent.parent
PIPER_DIR  = str(PROJECT / '01_pipeline' / 'all_srt' / 'piper')
CLOUDY_DIR = str(PROJECT / '01_pipeline' / 'all_srt' / 'cloudy')
OUTPUT_DIR = str(PROJECT / '05_analysis_outputs' / 'llm_annotation_output')
TURNS_DIR  = os.path.join(OUTPUT_DIR, 'turns')
os.makedirs(TURNS_DIR, exist_ok=True)

BATCH_SIZE = 8
client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# ─── QUESTION CONTEXT ─────────────────────────────────────────────────────────
QUESTION_CONTEXT = {
    1: {
        "question": "What did you think about and how did you feel while watching?",
        "expects": "emotional reactions and personal feelings",
        "depth_note": "surface=naming an emotion, interpretive=explaining why they felt that way, abstract=connecting to broader themes",
        "stance_note": "personal stance is expected and appropriate here"
    },
    2: {
        "question": "Did anything resonate with you or remind you of something in your own life?",
        "expects": "personal memories, experiences, or connections to own life",
        "depth_note": "surface=vague connection, interpretive=specific personal experience, abstract=reflection on life pattern or value",
        "stance_note": "generic film description without personal connection is off-topic"
    },
    3: {
        "question": "Did anything surprise you or turn out differently than expected?",
        "expects": "specific moments or plot points that were unexpected",
        "depth_note": "surface=naming what surprised them, interpretive=explaining why it was surprising, abstract=reflecting on narrative structure",
        "stance_note": "describing plot without saying what was surprising counts as partial"
    },
    4: {
        "question": "Were there scenes that were emotionally challenging?",
        "expects": "evaluation of emotional intensity and which scenes felt demanding",
        "depth_note": "surface=yes/no, interpretive=specific scenes + why emotionally hard, abstract=reflecting on what makes content emotionally challenging",
        "stance_note": "comparison to other films is on-topic here"
    },
    5: {
        "question": "Were there scenes that were intellectually challenging?",
        "expects": "evaluation of cognitive difficulty and which parts required more effort",
        "depth_note": "surface=yes/no, interpretive=what was confusing and why, abstract=reflecting on narrative ambiguity or artistic intent",
        "stance_note": "saying something was unclear without explaining why counts as surface"
    }
}

# ─── PROMPTS ──────────────────────────────────────────────────────────────────
TURN_SYSTEM = """You are annotating a conversation transcript. Two people just watched a short animated film together and are answering discussion questions about it.

For each numbered turn, classify TEN things:

1. d (depth): Pick highest level present.
   "surface"=names feeling or describes plot, no explanation
   "interpretive"=explains why or analyzes motivation/logic
   "abstract"=broader theme, life principle, or reflection on the film as a creation

2. st (stance): "personal"=own opinion/feeling, "neutral"=no personal angle

3. verbal_agreement: Does this turn express agreement with the partner?
   "backchannel"=pure filler (yeah, mm-hm) with no clear agreement signal
   "affirm"=explicitly agrees with partner, no new content
   "elaborated"=agrees with partner + adds new content
   "neutral"=no agreement expressed

4. ot (on_topic): vs the question being answered. "yes"/"partial"/"no"

5. q (is_question): true if genuinely asking partner for info/opinion/reaction. false if rhetorical or backchannel.

6. sd (self_disclosure): speaker sharing personal experience/memory/feeling from own life?
   "high"=clear personal share, "mid"=implied personal connection, "low"=none

7. ra (responsive): does this turn genuinely engage with what the previous speaker said?
   "true"=directly responds to prior turn, "false"=talks past it or ignores it

8. s (sentiment): "pos"/"neg"/"neu"/"mix"

9. ep (epistemic): how certain does the speaker sound?
   "high"=confident ("definitely", "clearly"), "low"=hedging ("I guess", "I don't know", "maybe"), "mid"=neutral

10. cv (convergence): "converge"=agreeing/aligning, "diverge"=introducing contrast, "neutral"=neither

--- FEW-SHOT EXAMPLES ---

Q1 (how did you feel while watching):
Turn: "But on the other hand, the cloud didn't take into account what the star can do and cannot, and that it's kind of like hurting him."
{"d":"interpretive","st":"personal","y":"na","ot":"yes","q":false,"sd":"low","ra":"true","s":"neg","ep":"mid","cv":"diverge"}

Q2 (did anything resonate with your own life):
Turn: "Yeah, maybe try to accommodate and find a solution somewhere in the middle."
{"d":"abstract","st":"personal","y":"elaborated","ot":"yes","q":false,"sd":"low","ra":"true","s":"pos","ep":"low","cv":"converge"}

Q3 (did anything surprise you):
Turn: "Yeah I think this story was much more abstract than the first one and it's just like made up."
{"d":"surface","st":"personal","y":"backchannel","ot":"yes","q":false,"sd":"low","ra":"false","s":"neu","ep":"low","cv":"neutral"}

Q2 (did anything resonate with your own life):
Turn: "I'm not like an alligator."
{"d":"abstract","st":"personal","y":"na","ot":"yes","q":false,"sd":"mid","ra":"false","s":"neu","ep":"high","cv":"diverge"}

Q4 (were there emotionally challenging scenes):
Turn: "Yeah, I feel that this scene was more... I don't know, because it's kind of weird."
{"d":"surface","st":"personal","y":"elaborated","ot":"yes","q":false,"sd":"low","ra":"true","s":"mix","ep":"low","cv":"converge"}

--- END EXAMPLES ---

Respond ONLY as compact JSON array — no markdown, no explanation:
[{"i":0,"d":"surface","st":"personal","y":"na","ot":"yes","q":false,"sd":"low","ra":"true","s":"neu","ep":"mid","cv":"neutral"},...]"""

CONVERSATION_SYSTEM = """You are evaluating a full conversation between two people who just watched an animated film together.

Rate the OVERALL conversation on engagement: how invested and active do both speakers seem?
"high"=both actively contributing, building on each other, genuine interest
"medium"=one person more engaged, or both somewhat going through the motions
"low"=minimal effort, short answers, not really connecting

Respond ONLY as JSON — no markdown:
{"engagement":"high"/"medium"/"low","reason":"one sentence","dominant_speaker":"00"/"01"/"balanced"}"""

# ─── PARSING ──────────────────────────────────────────────────────────────────
def parse_transcript(filepath):
    turns = []
    curr_q = 0
    speaker_regex = re.compile(r"^(SPEAKER_\d+):\s*(.*)", re.IGNORECASE)
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip().replace('\u201c', '"').replace('\u201d', '"').replace('\u2018', "'").replace('\u2019', "'")
            if re.match(r"Question_\d+", line):
                m = re.search(r"(\d+)", line)
                if m: curr_q = int(m.group(1))
                continue
            match = speaker_regex.match(line)
            if match and match.group(2).strip():
                turns.append({
                    "question_num": curr_q,
                    "speaker":      match.group(1).strip(),
                    "text":         match.group(2).strip()
                })
    return turns

# ─── TURN ANNOTATION ──────────────────────────────────────────────────────────
def annotate_turns(turns, dyad_id, condition):
    all_annotations = {}
    batches = [turns[i:i+BATCH_SIZE] for i in range(0, len(turns), BATCH_SIZE)]

    for b_idx, batch in enumerate(batches):
        offset = b_idx * BATCH_SIZE
        q_num  = batch[0]['question_num']
        q_info = QUESTION_CONTEXT.get(q_num, {
            "question": "General discussion", "expects": "general film discussion",
            "depth_note": "standard", "stance_note": "standard"
        })
        prior     = turns[max(0, offset-2):offset]
        def _clean(text):
            return text.encode('ascii', 'replace').decode('ascii')

        prior_str = "\n".join([f"{t['speaker']}: {_clean(t['text'])}" for t in prior])
        batch_str = "\n".join([
            f"{offset+i}. [Q{t['question_num']}] {t['speaker']}: {_clean(t['text'])}"
            for i, t in enumerate(batch)
        ])
        prompt = f"""Discussion question: "{q_info['question']}"
Expected content: {q_info['expects']}
Depth guide: {q_info['depth_note']}
Stance guide: {q_info['stance_note']}

Prior context:
{prior_str if prior_str else 'None'}

Turns to annotate:
{batch_str}"""

        try:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=BATCH_SIZE * 80,
                system=TURN_SYSTEM,
                messages=[{"role": "user", "content": prompt}]
            )
            raw = response.content[0].text.strip()
            raw = re.sub(r'```json\s*', '', raw)
            raw = re.sub(r'```\s*', '', raw).strip()
            if '[' in raw and not raw.rstrip().endswith(']'):
                last_brace = raw.rfind('}')
                if last_brace != -1:
                    raw = raw[:last_brace+1] + ']'
            match = re.search(r'\[.*\]', raw, re.DOTALL)
            if not match:
                raise ValueError(f"No JSON array found")
            results = json.loads(match.group())
            for r in results:
                all_annotations[r['i']] = r
        except Exception as e:
            print(f"    Batch {b_idx} error: {e}")
            for i in range(len(batch)):
                idx = offset + i
                all_annotations[idx] = {
                    "i": idx, "d": "surface", "st": "neutral", "y": "na",
                    "ot": "yes", "q": False, "sd": "low", "ra": "true",
                    "s": "neu", "ep": "mid", "cv": "neutral"
                }
        time.sleep(0.1)

    rows = []
    for i, t in enumerate(turns):
        ann = all_annotations.get(i, {})
        rows.append({
            "dyad_id":         dyad_id,
            "condition":       condition,
            "turn_index":      i,
            "question_num":    t['question_num'],
            "speaker":         t['speaker'],
            "text":            t['text'],
            "depth":           ann.get("d",  "surface"),
            "stance":          ann.get("st", "neutral"),
            "verbal_agreement": ann.get("y",  "na"),
            "on_topic":        ann.get("ot", "yes"),
            "is_question":     ann.get("q",  False),
            "self_disclosure": ann.get("sd", "low"),
            "responsive":      ann.get("ra", "true"),
            "sentiment":       ann.get("s",  "neu"),
            "epistemic":       ann.get("ep", "mid"),
            "convergence":     ann.get("cv", "neutral"),
        })
    return pd.DataFrame(rows)

# ─── CONVERSATION ANNOTATION ──────────────────────────────────────────────────
def annotate_conversation(turns, dyad_id, condition):
    full_text = "\n".join([f"[Q{t['question_num']}] {t['speaker']}: {t['text']}" for t in turns])
    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            system=CONVERSATION_SYSTEM,
            messages=[{"role": "user", "content": full_text}]
        )
        raw = response.content[0].text.strip()
        raw = re.sub(r'```json\s*', '', raw)
        raw = re.sub(r'```\s*', '', raw).strip()
        result = json.loads(raw)
    except Exception as e:
        print(f"    Conversation annotation error: {e}")
        result = {"engagement": "medium", "reason": "error", "dominant_speaker": "balanced"}

    result["dyad_id"]   = dyad_id
    result["condition"] = condition
    return result

# ─── COLLAPSE TO DYAD LEVEL ───────────────────────────────────────────────────

def collapse_to_dyad(df):
    """
    Aggregate turn-level annotations to dyad-condition level.
    Produces one row per dyad-condition with feature rates and asymmetry scores.
    """
    rows = []
    if "yeah_type" in df.columns and "verbal_agreement" not in df.columns:
        df = df.rename(columns={"yeah_type": "verbal_agreement"})
    df["is_question"] = df["is_question"].astype(bool)

    for (dyad_id, condition), grp in df.groupby(['dyad_id', 'condition']):
        total = len(grp)
        spk_ids = grp['speaker'].unique()

        # ── Overall rates ──
        row = {
            "dyad_id":              dyad_id,
            "condition":            condition,
            "total_turns":          total,
            "depth_surface_rate":   (grp['depth'] == 'surface').sum() / total,
            "depth_interpret_rate": (grp['depth'] == 'interpretive').sum() / total,
            "depth_abstract_rate":  (grp['depth'] == 'abstract').sum() / total,
            "personal_stance_rate": (grp['stance'] == 'personal').sum() / total,
            "verbal_agreement_backchannel_n": (grp['verbal_agreement'] == 'backchannel').sum(),
            "verbal_agreement_affirm_n":      (grp['verbal_agreement'] == 'affirm').sum(),
            "verbal_agreement_elaborated_n":  (grp['verbal_agreement'] == 'elaborated').sum(),
            "on_topic_rate":        (grp['on_topic'] == 'yes').sum() / total,
            "off_topic_rate":       (grp['on_topic'] == 'no').sum() / total,
            "question_count":       grp['is_question'].sum(),
            "disclosure_high_n":    (grp['self_disclosure'] == 'high').sum(),
            "disclosure_mid_n":     (grp['self_disclosure'] == 'mid').sum(),
            "responsive_rate":      (grp['responsive'] == 'true').sum() / total,
            "sentiment_pos_rate":   (grp['sentiment'] == 'pos').sum() / total,
            "sentiment_neg_rate":   (grp['sentiment'] == 'neg').sum() / total,
            "epistemic_low_rate":   (grp['epistemic'] == 'low').sum() / total,
            "epistemic_high_rate":  (grp['epistemic'] == 'high').sum() / total,
            "converge_rate":        (grp['convergence'] == 'converge').sum() / total,
            "diverge_rate":         (grp['convergence'] == 'diverge').sum() / total,
        }

        # ── On-topic rate per question block ──
        for q in range(1, 6):
            block = grp[grp['question_num'] == q]
            if len(block) > 0:
                row[f"on_topic_q{q}"] = (block['on_topic'] == 'yes').sum() / len(block)
            else:
                row[f"on_topic_q{q}"] = np.nan

        # ── Per-speaker features + asymmetry ──
        if len(spk_ids) == 2:
            spk_a, spk_b = sorted(spk_ids)
            for feat, col in [
                ("depth_deep",    lambda s: s['depth'].isin(['interpretive','abstract'])),
                ("personal",      lambda s: s['stance'] == 'personal'),
                ("questions",     lambda s: s['is_question']),
                ("disclosure",    lambda s: s['self_disclosure'].isin(['high','mid'])),
                ("responsive",    lambda s: s['responsive'] == 'true'),
                ("hedging",       lambda s: s['epistemic'] == 'low'),
                ("converge",      lambda s: s['convergence'] == 'converge'),
            ]:
                sub_a = grp[grp['speaker'] == spk_a]
                sub_b = grp[grp['speaker'] == spk_b]
                rate_a = col(sub_a).sum() / max(len(sub_a), 1)
                rate_b = col(sub_b).sum() / max(len(sub_b), 1)
                row[f"{feat}_mean"]  = (rate_a + rate_b) / 2
                row[f"{feat}_asym"]  = abs(rate_a - rate_b)

            # Turn count asymmetry
            row["turn_asym"] = abs(len(grp[grp['speaker'] == spk_a]) - len(grp[grp['speaker'] == spk_b])) / total

        rows.append(row)

    return pd.DataFrame(rows)

# ─── MAIN PIPELINE ────────────────────────────────────────────────────────────
def run():
    all_turns = []
    all_conv  = []

    # Find all transcript files
    files = []
    for directory, condition in [(PIPER_DIR, 'piper'), (CLOUDY_DIR, 'cloudy')]:
        if not os.path.exists(directory):
            print(f"Warning: {directory} not found, skipping")
            continue
        for f in sorted(glob.glob(os.path.join(directory, "*.txt"))):
            files.append((f, condition))

    print(f"Found {len(files)} transcript files\n")

    for filepath, condition in files:
        filename = os.path.basename(filepath)
        id_match = re.search(r"(dyad\d+_\d+)", filename)
        dyad_id  = id_match.group(1) if id_match else filename.replace('.txt', '')

        # Skip if already done
        cache_path = os.path.join(TURNS_DIR, f"{dyad_id}_{condition}.csv")
        if os.path.exists(cache_path):
            print(f"  Skipping {dyad_id} {condition} (already annotated)")
            df_cached = pd.read_csv(cache_path)
            all_turns.append(df_cached)
            continue

        print(f"  Annotating {dyad_id} [{condition}]...")
        turns = parse_transcript(filepath)
        if not turns:
            print(f"    No turns found, skipping")
            continue

        # Turn-level annotation
        df_turns = annotate_turns(turns, dyad_id, condition)
        df_turns.to_csv(cache_path, index=False)
        all_turns.append(df_turns)

        # Conversation-level annotation
        conv = annotate_conversation(turns, dyad_id, condition)
        all_conv.append(conv)

        print(f"    Done: {len(turns)} turns annotated")
        time.sleep(0.2)

    # Save all turns
    if all_turns:
        df_all_turns = pd.concat(all_turns, ignore_index=True)
        df_all_turns.to_csv(os.path.join(OUTPUT_DIR, 'all_turns.csv'), index=False)
        print(f"\nAll turns saved: {len(df_all_turns)} total")

        # Collapse to dyad level
        df_dyad = collapse_to_dyad(df_all_turns)

        # Add pair_id for merging with outcomes
        df_dyad['pair_id'] = df_dyad['dyad_id'].str.extract(r'dyad(\d+)').astype(int)
        df_dyad.to_csv(os.path.join(OUTPUT_DIR, 'dyad_features.csv'), index=False)
        print(f"Dyad features saved: {len(df_dyad)} rows, {len(df_dyad.columns)} features")

    # Save conversation-level ratings
    if all_conv:
        df_conv = pd.DataFrame(all_conv)
        df_conv.to_csv(os.path.join(OUTPUT_DIR, 'conversation_level.csv'), index=False)
        print(f"Conversation ratings saved: {len(df_conv)} rows")

    print(f"\nAll outputs in '{OUTPUT_DIR}/'")
    print("Next step: merge dyad_features.csv with your outcomes.csv using pair_id")

if __name__ == "__main__":
    run()
