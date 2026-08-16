#!/usr/bin/env python3
"""
Prepare CANDOR media downloads for this repo's vocal-alignment pipeline.

This converts CANDOR transcript CSVs into the timestamped-turn schema consumed
by extract_acoustic_features.py transfer mode:

  dyad_id, pair_id, condition, Order, turn_idx, speaker, role,
  t_start, t_end, dur, n_words, text, coverage, wav_path

It can read either extracted CANDOR conversation folders or BetterUp media zip
parts directly. Audio extraction defaults to the synchronized per-speaker files
under ``processed/``. The unsynchronized ``raw/`` streams have speaker-specific
``startTimeOffset`` values and must not be sampled with transcript timestamps
without explicit offset correction. One 16 kHz mono WAV is written per speaker,
matching the ``wav_path`` column.

Examples:
  python scripts/acoustic_alignment/prepare_candor_for_vocal_alignment.py \
      --input /Users/Meihui/Downloads/candor/raw_media_part_001.zip \
      --out data/derived/candor/labeled_turns_backbiter.csv \
      --wav-dir data/interim/candor_wav \
      --limit 1

  python scripts/acoustic_alignment/prepare_candor_for_vocal_alignment.py \
      --input "/Users/Meihui/Downloads/candor/raw_media_part_*.zip" \
      --extract-audio
"""
from __future__ import annotations

import argparse
import csv
import glob
import io
import json
import shutil
import subprocess
import tempfile
import wave
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = PROJECT / "data" / "derived" / "candor" / "labeled_turns_backbiter.csv"
DEFAULT_WAV_DIR = PROJECT / "data" / "interim" / "candor_wav"


@dataclass(frozen=True)
class ConversationSource:
    conversation_id: str
    root: Path | None = None
    zip_path: Path | None = None


def _iter_sources(patterns: list[str]) -> list[ConversationSource]:
    sources: list[ConversationSource] = []
    seen: set[tuple[str, str]] = set()
    for pattern in patterns:
        matches = [Path(p) for p in glob.glob(pattern)] or [Path(pattern)]
        for path in matches:
            if path.is_dir():
                for meta in sorted(path.glob("*/metadata.json")):
                    cid = meta.parent.name
                    key = ("dir", str(meta.parent))
                    if key not in seen:
                        seen.add(key)
                        sources.append(ConversationSource(cid, root=meta.parent))
            elif path.suffix.lower() == ".zip":
                with zipfile.ZipFile(path) as zf:
                    cids = sorted({
                        name.split("/", 1)[0]
                        for name in zf.namelist()
                        if name.endswith("/metadata.json")
                    })
                for cid in cids:
                    key = ("zip", f"{path}:{cid}")
                    if key not in seen:
                        seen.add(key)
                        sources.append(ConversationSource(cid, zip_path=path))
            else:
                raise SystemExit(f"Unsupported input: {path}")
    return sources


def _read_text(src: ConversationSource, rel: str) -> str:
    if src.root is not None:
        return (src.root / rel).read_text(encoding="utf-8", errors="replace")
    assert src.zip_path is not None
    with zipfile.ZipFile(src.zip_path) as zf:
        with zf.open(f"{src.conversation_id}/{rel}") as f:
            return f.read().decode("utf-8", errors="replace")


def _read_csv(src: ConversationSource, rel: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(_read_text(src, rel)))


def _metadata(src: ConversationSource) -> dict:
    return json.loads(_read_text(src, "metadata.json"))


def _speaker_roles(meta: dict) -> dict[str, str]:
    roles = {}
    for speaker in meta.get("speakers", []):
        user_id = speaker.get("user_id")
        channel = speaker.get("channel")
        if user_id and channel:
            roles[str(user_id)] = str(channel)
    return roles


def _speaker_media_files(meta: dict, audio_source: str) -> dict[str, str]:
    files = {}
    for speaker in meta.get("speakers", []):
        user_id = speaker.get("user_id")
        if not user_id:
            continue
        if audio_source == "processed":
            files[str(user_id)] = f"{user_id}.mp4"
        else:
            speaker_files = speaker.get("files") or []
            if speaker_files:
                files[str(user_id)] = str(speaker_files[0]["filename"])
    return files


def _media_exists(src: ConversationSource, rel: str) -> bool:
    if src.root is not None:
        return (src.root / rel).is_file()
    assert src.zip_path is not None
    member = f"{src.conversation_id}/{rel}"
    with zipfile.ZipFile(src.zip_path) as zf:
        try:
            zf.getinfo(member)
        except KeyError:
            return False
    return True


def _processed_media_spec(
    src: ConversationSource, meta: dict, speaker: str
) -> tuple[str, str | None, bool]:
    per_speaker = f"processed/{speaker}.mp4"
    if _media_exists(src, per_speaker):
        return per_speaker, None, False

    # Some CANDOR records omit a per-user processed MP4 but retain the
    # synchronized stereo mix. Extract the participant's mapped L/R channel;
    # never fall back to the unsynchronized raw stream.
    combined = f"processed/{src.conversation_id}.mp3"
    role = _speaker_roles(meta).get(speaker)
    channel_index = {"L": 0, "R": 1}.get(role)
    if _media_exists(src, combined) and channel_index is not None:
        return combined, f"pan=mono|c0=c{channel_index}", True
    raise FileNotFoundError(
        f"No synchronized processed media for {src.conversation_id}/{speaker}: "
        f"missing {per_speaker} and usable {combined} channel"
    )


def _wav_path(wav_dir: Path, conversation_id: str, speaker: str) -> Path:
    return wav_dir / f"{conversation_id}__{speaker}.wav"


def _turn_rows(
    src: ConversationSource,
    variant: str,
    wav_dir: Path,
    include_backchannels: bool,
    pair_index: int,
    audio_source: str,
) -> list[dict]:
    meta = _metadata(src)
    roles = _speaker_roles(meta)
    transcript = _read_csv(src, f"transcription/transcript_{variant}.csv")
    rows: list[dict] = []
    order = 0

    def add_row(turn_idx, speaker, start, stop, text, n_words, source_turn_id, kind):
        nonlocal order
        if pd.isna(speaker) or pd.isna(start) or pd.isna(stop):
            return
        text = "" if pd.isna(text) else str(text)
        start_f = float(start)
        stop_f = float(stop)
        rows.append({
            "dyad_id": src.conversation_id,
            "pair_id": pair_index,
            "condition": f"candor_{variant}",
            "Order": order,
            "turn_idx": turn_idx,
            "speaker": str(speaker),
            "role": roles.get(str(speaker)),
            "t_start": round(start_f, 3),
            "t_end": round(stop_f, 3),
            "dur": round(stop_f - start_f, 3),
            "n_words": int(n_words) if pd.notna(n_words) else len(text.split()),
            "text": text,
            "coverage": 1.0,
            "wav_path": str(_wav_path(wav_dir, src.conversation_id, str(speaker))),
            "candor_turn_id": source_turn_id,
            "candor_turn_kind": kind,
            "candor_audio_source": audio_source,
        })
        order += 1

    for _, row in transcript.iterrows():
        add_row(
            turn_idx=order,
            speaker=row.get("speaker"),
            start=row.get("start"),
            stop=row.get("stop"),
            text=row.get("utterance"),
            n_words=row.get("n_words"),
            source_turn_id=row.get("turn_id"),
            kind="turn",
        )
        if include_backchannels and "backchannel" in row and pd.notna(row.get("backchannel")):
            add_row(
                turn_idx=order,
                speaker=row.get("backchannel_speaker"),
                start=row.get("backchannel_start"),
                stop=row.get("backchannel_stop"),
                text=row.get("backchannel"),
                n_words=len(str(row.get("backchannel")).split()),
                source_turn_id=row.get("turn_id"),
                kind="backchannel",
            )
    rows.sort(key=lambda r: (r["t_start"], r["t_end"], r["speaker"]))
    for idx, row in enumerate(rows):
        row["turn_idx"] = idx
    return rows


def _valid_wav(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 78:
        return False
    try:
        with wave.open(str(path), "rb") as wav:
            return wav.getnframes() > 0 and wav.getframerate() == 16000
    except (wave.Error, EOFError):
        return False


def _extract_audio(
    src: ConversationSource, wav_dir: Path, audio_source: str
) -> tuple[int, list[dict]]:
    meta = _metadata(src)
    speaker_files = _speaker_media_files(meta, audio_source)
    wav_dir.mkdir(parents=True, exist_ok=True)
    n_done = 0
    records = []
    for speaker, raw_name in speaker_files.items():
        if audio_source == "processed":
            media_rel, audio_filter, fallback = _processed_media_spec(src, meta, speaker)
        else:
            media_rel = f"raw/{raw_name}"
            audio_filter = None
            fallback = False
        media_name = Path(media_rel).name
        out = _wav_path(wav_dir, src.conversation_id, speaker)
        if _valid_wav(out):
            records.append({
                "dyad_id": src.conversation_id,
                "speaker": speaker,
                "audio_source": audio_source,
                "media_rel": media_rel,
                "channel_filter": audio_filter,
                "used_stereo_fallback": fallback,
                "wav_path": str(out),
                "status": "reused",
            })
            continue
        if out.exists():
            out.unlink()
        tmp_out = out.with_name(f"{out.stem}.tmp{out.suffix}")
        if tmp_out.exists():
            tmp_out.unlink()
        def convert(input_path: Path) -> int:
            cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(input_path), "-vn",
            ]
            if audio_filter:
                cmd.extend(["-af", audio_filter])
            cmd.extend(["-ac", "1", "-ar", "16000", str(tmp_out)])
            return subprocess.run(cmd, check=False).returncode

        if src.root is not None:
            returncode = convert(src.root / media_rel)
        else:
            # MP4 demuxing requires seekable input, so a zip member cannot be
            # streamed to ffmpeg over stdin. CARC uses extracted roots; this
            # temporary-file path supports local zip smoke tests safely.
            assert src.zip_path is not None
            with tempfile.TemporaryDirectory(prefix="candor_media_") as tmp_dir:
                input_path = Path(tmp_dir) / media_name
                with zipfile.ZipFile(src.zip_path) as zf:
                    member = f"{src.conversation_id}/{media_rel}"
                    with zf.open(member) as source, input_path.open("wb") as target:
                        shutil.copyfileobj(source, target)
                returncode = convert(input_path)

        valid_wav = returncode == 0 and _valid_wav(tmp_out)
        if returncode != 0 or not valid_wav:
            if tmp_out.exists():
                tmp_out.unlink()
            raise RuntimeError(
                f"ffmpeg failed for {src.conversation_id}/{media_rel}"
            )
        tmp_out.replace(out)
        n_done += 1
        records.append({
            "dyad_id": src.conversation_id,
            "speaker": speaker,
            "audio_source": audio_source,
            "media_rel": media_rel,
            "channel_filter": audio_filter,
            "used_stereo_fallback": fallback,
            "wav_path": str(out),
            "status": "created",
        })
    return n_done, records


def run(args: argparse.Namespace) -> pd.DataFrame:
    sources = _iter_sources(args.input)
    if args.limit:
        sources = sources[: args.limit]
    if not sources:
        raise SystemExit("No CANDOR conversations found.")

    all_rows: list[dict] = []
    n_audio = 0
    audio_records: list[dict] = []
    for pair_index, src in enumerate(sources, start=1):
        all_rows.extend(
            _turn_rows(
                src=src,
                variant=args.variant,
                wav_dir=args.wav_dir,
                include_backchannels=args.include_backchannels,
                pair_index=pair_index,
                audio_source=args.audio_source,
            )
        )
        if args.extract_audio:
            n_new, records = _extract_audio(src, args.wav_dir, args.audio_source)
            n_audio += n_new
            audio_records.extend(records)

    df = pd.DataFrame(all_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, quoting=csv.QUOTE_MINIMAL)
    print(
        f"Wrote {len(df)} turns from {df['dyad_id'].nunique()} CANDOR conversations "
        f"-> {args.out}"
    )
    missing_wav = sum(not Path(p).exists() for p in df["wav_path"].dropna().unique())
    if args.extract_audio:
        manifest_path = args.out.with_name("candor_audio_manifest.csv")
        manifest = pd.DataFrame(audio_records)
        manifest.to_csv(manifest_path, index=False)
        n_fallback = int(manifest["used_stereo_fallback"].sum()) if len(manifest) else 0
        print(
            f"Extracted {n_audio} synchronized speaker WAV files from "
            f"{args.audio_source}/ -> {args.wav_dir}"
        )
        print(
            f"Audio manifest: {len(manifest)} speakers, {n_fallback} synchronized "
            f"stereo fallbacks -> {manifest_path}"
        )
    elif missing_wav:
        print(f"[warn] {missing_wav} speaker WAV files are not present yet; rerun with --extract-audio.")
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Extracted CANDOR root/folder, zip file, or glob. Repeatable.",
    )
    parser.add_argument(
        "--variant",
        choices=["backbiter", "audiophile", "cliffhanger"],
        default="backbiter",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--wav-dir", type=Path, default=DEFAULT_WAV_DIR)
    parser.add_argument(
        "--audio-source",
        choices=["processed", "raw"],
        default="processed",
        help=(
            "Speaker media source. Default: processed (synchronized to transcript "
            "timestamps). Raw streams are retained only for explicit offset audits."
        ),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--include-backchannels", action="store_true")
    parser.add_argument("--extract-audio", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
