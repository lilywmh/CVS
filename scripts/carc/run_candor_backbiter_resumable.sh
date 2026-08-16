#!/usr/bin/env bash
set -euo pipefail

# Resumable CANDOR vocal-alignment runner for CARC.
#
# Checkpoints:
#   1. labeled_turns_backbiter.csv is reused if present.
#   2. speaker WAVs are reused by prepare_candor_for_vocal_alignment.py.
#   3. acoustic features are written one conversation at a time under
#      $CANDOR_OUT_DIR/acoustic_chunks; each completed conversation gets a
#      matching .done marker. Re-running skips completed chunks.
#   4. final acoustic and vocal-alignment CSVs are rebuilt from chunks.

CANDOR_INPUT="${CANDOR_INPUT:-/scratch1/$USER/candor/raw_media_part_*.zip}"
CANDOR_OUT_DIR="${CANDOR_OUT_DIR:-data/derived/candor}"
CANDOR_WAV_DIR="${CANDOR_WAV_DIR:-data/interim/candor_wav}"
CANDOR_LIMIT="${CANDOR_LIMIT:-}"
CANDOR_FORCE_PREP="${CANDOR_FORCE_PREP:-0}"
CANDOR_AUDIO_SOURCE="${CANDOR_AUDIO_SOURCE:-processed}"

mkdir -p "$CANDOR_OUT_DIR" "$CANDOR_WAV_DIR" "$CANDOR_OUT_DIR/acoustic_chunks" "$CANDOR_OUT_DIR/acoustic_done"

labeled="$CANDOR_OUT_DIR/labeled_turns_backbiter.csv"
acoustic="$CANDOR_OUT_DIR/acoustic_turns_backbiter.csv"
source_marker="$CANDOR_OUT_DIR/.candor_audio_source"

if [[ -s "$source_marker" ]]; then
  recorded_source="$(<"$source_marker")"
  if [[ "$recorded_source" != "$CANDOR_AUDIO_SOURCE" ]]; then
    echo "Audio-source mismatch: output records '$recorded_source', requested '$CANDOR_AUDIO_SOURCE'." >&2
    echo "Use a new CANDOR_OUT_DIR and CANDOR_WAV_DIR." >&2
    exit 1
  fi
elif [[ -s "$labeled" || -n "$(find "$CANDOR_OUT_DIR/acoustic_done" -name '*.done' -print -quit)" ]]; then
  echo "Existing output predates audio-source tracking." >&2
  echo "Use a new CANDOR_OUT_DIR and CANDOR_WAV_DIR; do not reuse old acoustic chunks." >&2
  exit 1
else
  printf '%s\n' "$CANDOR_AUDIO_SOURCE" > "$source_marker"
fi

echo "[config] CANDOR audio source: $CANDOR_AUDIO_SOURCE"

limit_args=()
if [[ -n "$CANDOR_LIMIT" ]]; then
  limit_args=(--limit "$CANDOR_LIMIT")
fi

if [[ "$CANDOR_FORCE_PREP" == "1" || ! -s "$labeled" ]]; then
  echo "[prep] creating $labeled"
  python scripts/acoustic_alignment/prepare_candor_for_vocal_alignment.py \
    --input "$CANDOR_INPUT" \
    --out "$labeled" \
    --wav-dir "$CANDOR_WAV_DIR" \
    --audio-source "$CANDOR_AUDIO_SOURCE" \
    --extract-audio \
    "${limit_args[@]}"
else
  echo "[prep] reusing existing $labeled"
fi

echo "[chunks] processing per-conversation acoustic features"
while IFS= read -r dyad_id; do
  chunk_turns="$CANDOR_OUT_DIR/acoustic_chunks/${dyad_id}.turns.csv"
  chunk_out="$CANDOR_OUT_DIR/acoustic_chunks/${dyad_id}.acoustic.csv"
  done_marker="$CANDOR_OUT_DIR/acoustic_done/${dyad_id}.done"

  if [[ -s "$chunk_out" && -e "$done_marker" ]]; then
    echo "[skip] $dyad_id"
    continue
  fi

  echo "[run] $dyad_id"
  python - "$labeled" "$dyad_id" "$chunk_turns" <<'PY'
from pathlib import Path
import sys
import pandas as pd

labeled = Path(sys.argv[1])
dyad_id = sys.argv[2]
out = Path(sys.argv[3])
df = pd.read_csv(labeled)
sub = df[df["dyad_id"].astype(str).eq(dyad_id)].copy()
if sub.empty:
    raise SystemExit(f"no rows for {dyad_id}")
sub.to_csv(out, index=False)
PY

  rm -f "$done_marker" "$chunk_out.tmp"
  python scripts/acoustic_alignment/extract_acoustic_features.py \
    --turns-csv "$chunk_turns" \
    --out "$chunk_out.tmp"
  mv "$chunk_out.tmp" "$chunk_out"
  touch "$done_marker"
done < <(python - "$labeled" "$CANDOR_OUT_DIR/acoustic_chunks" <<'PY'
from pathlib import Path
import sys
import pandas as pd

labeled = Path(sys.argv[1])
chunk_dir = Path(sys.argv[2])
chunk_dir.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(labeled, usecols=["dyad_id"])
for dyad_id in sorted(df["dyad_id"].dropna().astype(str).unique()):
    print(dyad_id)
PY
)

echo "[combine] rebuilding $acoustic"
python - "$CANDOR_OUT_DIR/acoustic_chunks" "$acoustic" <<'PY'
from pathlib import Path
import sys
import pandas as pd

chunk_dir = Path(sys.argv[1])
out = Path(sys.argv[2])
files = sorted(chunk_dir.glob("*.acoustic.csv"))
if not files:
    raise SystemExit("No acoustic chunk files found.")
df = pd.concat((pd.read_csv(path) for path in files), ignore_index=True)
tmp = out.with_suffix(out.suffix + ".tmp")
df.to_csv(tmp, index=False)
tmp.replace(out)
print(f"Wrote {len(df)} turns from {df['dyad_id'].nunique()} conversations -> {out}")
PY

echo "[alignment] computing final vocal-alignment tables"
python scripts/acoustic_alignment/compute_vocal_alignment_floor_transfer.py \
  --in "$acoustic" \
  --out "$CANDOR_OUT_DIR/vocal_alignment_floor_transfer_backbiter.csv" \
  --excluded-turns-out "$CANDOR_OUT_DIR/vocal_alignment_floor_transfer_excluded_turns_backbiter.csv"
