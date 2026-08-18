#!/usr/bin/env python3
"""
Quality check for a generated transcript. Catches the failure modes we actually
observed while evaluating ASR backends, so a future change that reintroduces
them does not pass unnoticed.

Checks:
  1. Repetition-loop hallucinations (Whisper getting stuck repeating a token
     instead of transcribing speech). The validated WhisperX path produces
     zero of these beyond natural emphatic repeats; MLX variants produced
     loops of 200+ repeats.
  2. Trailing content after the last substantial turn (hallucinated filler
     over closing silence, e.g. "chao chao chao ...").
  3. UNKNOWN speaker blocks (words falling outside any diarization turn).
  4. Basic stats for eyeballing against expectations.

Usage:
    .venv/bin/python check_transcript.py path/to/transcript.md
"""

import argparse
import re
import sys
from pathlib import Path

# A token repeated 4+ times in a row, tolerating punctuation between repeats.
# Natural speech does produce short emphatic runs ("no, no, no, no"), so the
# count matters more than the presence: report all, flag the long ones.
LOOP_RE = re.compile(r"\b(\w+)\b(?:[,.!?¡¿]?\s+\1\b){3,}", re.IGNORECASE)
BLOCK_RE = re.compile(r"^\*\*(?P<speaker>[^*]+)\*\* `(?P<start>[\d:]+) - (?P<end>[\d:]+)`$")

# Runs at or above this length are very unlikely to be real speech.
SEVERE_THRESHOLD = 8


def parse_blocks(text: str) -> list[dict]:
    blocks = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        match = BLOCK_RE.match(line.strip())
        if match:
            body = lines[i + 1].strip() if i + 1 < len(lines) else ""
            blocks.append({
                "speaker": match.group("speaker"),
                "start": match.group("start"),
                "end": match.group("end"),
                "text": body,
            })
    return blocks


def to_seconds(timestamp: str) -> int:
    parts = [int(p) for p in timestamp.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("transcript", type=Path)
    parser.add_argument("--quiet", action="store_true", help="Only print problems, no stats")
    args = parser.parse_args()

    if not args.transcript.exists():
        sys.exit(f"File not found: {args.transcript}")

    text = args.transcript.read_text(encoding="utf-8")
    blocks = parse_blocks(text)
    problems = 0

    # 1. repetition loops
    loops = []
    for match in LOOP_RE.finditer(text):
        token = match.group(1)
        repeats = len(re.findall(rf"\b{re.escape(token)}\b", match.group(0), re.IGNORECASE))
        loops.append((token, repeats, match.start()))

    severe = [loop for loop in loops if loop[1] >= SEVERE_THRESHOLD]
    if severe:
        problems += len(severe)
        print(f"HALLUCINATION LOOPS: {len(severe)} severe (>={SEVERE_THRESHOLD} repeats), {len(loops)} total")
        for token, repeats, pos in severe:
            context = text[max(0, pos - 60):pos].replace("\n", " ")
            print(f'  "{token}" x{repeats}  after: ...{context}')
    elif loops:
        print(f"Repetition runs: {len(loops)}, all short ({max(l[1] for l in loops)} max), likely natural emphasis")
    else:
        print("Repetition loops: none")

    # 2. trailing filler: short blocks after a gap, at the tail
    if blocks:
        substantial = [b for b in blocks if len(b["text"].split()) >= 10]
        if substantial:
            last_real_end = to_seconds(substantial[-1]["end"])
            trailing = [b for b in blocks if to_seconds(b["start"]) > last_real_end]
            if trailing:
                tail_span = to_seconds(trailing[-1]["end"]) - to_seconds(trailing[0]["start"])
                if tail_span > 30:
                    problems += 1
                    print(f"TRAILING FILLER: {len(trailing)} block(s) spanning {tail_span}s after the last "
                          f"substantial turn at {substantial[-1]['end']}")
                    for b in trailing:
                        print(f"  [{b['start']} - {b['end']}] {b['speaker']}: {b['text'][:70]}")
                else:
                    print("Trailing filler: none")
            else:
                print("Trailing filler: none")

    # 3. UNKNOWN speaker blocks
    unknown = [b for b in blocks if b["speaker"] == "UNKNOWN"]
    if unknown:
        unknown_words = sum(len(b["text"].split()) for b in unknown)
        print(f"UNKNOWN speaker: {len(unknown)} block(s), {unknown_words} words unattributed "
              f"(words outside any diarization turn)")
    else:
        print("UNKNOWN speaker blocks: none")

    # 4. stats
    if not args.quiet and blocks:
        speakers = sorted({b["speaker"] for b in blocks})
        words = sum(len(b["text"].split()) for b in blocks)
        print()
        print(f"Turns: {len(blocks)} | Speakers: {len(speakers)} ({', '.join(speakers)}) | Words: {words}")
        print(f"Span: {blocks[0]['start']} to {blocks[-1]['end']}")

    print()
    print("PASS: no blocking problems found" if problems == 0 else f"REVIEW: {problems} problem(s) found")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
