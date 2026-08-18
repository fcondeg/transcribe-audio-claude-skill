---
name: transcribe
description: "Turn a meeting recording (audio or video) into a speaker-tagged Markdown transcript, running entirely on the local machine so no audio or transcript ever leaves it. Use when the user wants a recording transcribed, a meeting written up from audio, or speaker-separated text from a call. Handles its own first-run dependency setup on macOS, Windows and Linux."
---

# Transcribe a meeting recording

Produces a speaker-tagged `.md` transcript from any audio or video file, fully
offline. Nothing is uploaded: this exists so client calls and internal 1:1s can be
transcribed without the content reaching a third-party API.

## Scope

**Step 1 only**: transcription with generic speaker labels (`SPEAKER_00`,
`SPEAKER_01`, ...). It does not work out who is who, and does not extract decisions
or action items. Offer those as a follow-up pass over the transcript; do not do them
unprompted.

## Step 0: first-run setup (check every time, it is cheap)

The skill folder is `<SKILL_DIR>` (the directory containing this file). Check
whether the environment is ready:

- Ready if **both** `<SKILL_DIR>/.venv` exists **and** `<SKILL_DIR>/.setup-stamp`
  contains the current stamp version (`2`).
- If either is missing, run the setup for the platform and **tell the user it is a
  one-off that downloads a few GB**, so a long wait is expected:

  macOS / Linux:
  ```bash
  bash "<SKILL_DIR>/scripts/setup.sh"
  ```
  Windows:
  ```powershell
  powershell -ExecutionPolicy Bypass -File "<SKILL_DIR>\scripts\setup.ps1"
  ```

The setup scripts are idempotent, so re-running is harmless. They install `ffmpeg`
and `uv` (Homebrew on macOS, winget on Windows, apt or the official installer on
Linux), create a Python 3.11 virtualenv, install the pinned packages, and add
`mlx-whisper` only on Apple Silicon. On Windows, if an NVIDIA GPU is present,
`setup.ps1` also swaps in a CUDA build of PyTorch (the default Windows wheel is
CPU-only) — that's a further multi-GB download on top of the base setup, first
run only. If it can't complete that step it says so and leaves the skill working
on CPU rather than failing setup outright.

**The Hugging Face token is the one thing setup cannot do for the user.** Speaker
diarization uses a gated model. If setup reports the token is missing, relay its
instructions and stop: create a read token at huggingface.co/settings/tokens, accept
the terms at huggingface.co/pyannote/speaker-diarization-community-1, and save it as
`HF_TOKEN=hf_...` in `<SKILL_DIR>/.env`. Reassure them the token only unlocks a
one-time model download; no recording data is ever sent anywhere. Never ask them to
paste the token into the chat if they can write the file themselves.

## Step 1: gather what the run needs

1. **Resolve the input path.** Recordings usually sit in `~/Downloads`. If the file
   does not exist, say so rather than guessing at a similar name.

2. **Report the expected duration** before starting, from:
   ```bash
   ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "<file>"
   ```
   Rough guide: on Apple Silicon (`--fast`) about **8% of the recording's length**
   (a 90 minute meeting takes ~7 minutes). On `--safe` roughly **real time** on CPU,
   much quicker with an NVIDIA GPU.

3. **Speaker count.** If the user said how many people were in the meeting, or it is
   obviously a 1:1, pass `--min-speakers N --max-speakers N`. If unknown, ask in one
   line. Do not guess: a wrong bound degrades diarization.

4. **Language.** Autodetect is reliable on long recordings and has misfired on very
   short ones. For anything under ~5 minutes, or when the language is known, pass
   `--language es` / `--language en` explicitly.

## Step 2: run it in the background

It takes far too long to block on. Use the venv's Python directly:

macOS / Linux:
```bash
"<SKILL_DIR>/.venv/bin/python" "<SKILL_DIR>/scripts/transcribe.py" "<input>" \
  -o "<output>.md" --min-speakers N --max-speakers N --keep-json
```
Windows:
```powershell
& "<SKILL_DIR>\.venv\Scripts\python.exe" "<SKILL_DIR>\scripts\transcribe.py" "<input>" `
  -o "<output>.md" --min-speakers N --max-speakers N --keep-json
```

Name the output `YYYY-MM-DD-<slug>.md` using the **meeting's** date, not today's if
they differ. Keep `--keep-json`: the sidecar carries word-level timestamps and
per-word speaker data that a later speaker-identification pass needs.

Mode selection: `--fast` is the default and correct almost always, so omit the flag.
On non-Apple-Silicon machines the script falls back to `--safe` automatically and
says so. Reach for `--safe` explicitly only when a transcript must be maximally
defensible, when cross-checking a `--fast` result that looks wrong, or for
recordings outside what `--fast` was validated on (3+ speakers with crosstalk,
non-Spanish, heavy code-switching, distant room audio).

A harmless `torchcodec`/`ffmpeg` dylib warning prints on startup. Ignore it: this
pipeline always hands pyannote a preloaded waveform, never a file path.

## Step 3: quality-check before declaring success

```bash
"<SKILL_DIR>/.venv/bin/python" "<SKILL_DIR>/scripts/check_transcript.py" "<output>.md"
```

It flags the failure modes measured while building this: repetition-loop
hallucinations, invented filler over closing silence, and words left unattributed.
Exit code is non-zero when something is found. **If it reports problems, surface
them rather than presenting the transcript as good.** A transcript that silently
invents or drops text is worse than a slow one.

Two things that look alarming but usually are not, worth checking before raising
them as faults:

- **Transcript ends well before the file does.** Usually the recording simply ran on
  after the conversation. Confirm with the audio level in the tail: silence sits
  around -45 dB or below, speech around -30 dB.
  ```bash
  ffmpeg -hide_banner -nostats -ss <seconds> -i "<file>" -af volumedetect -f null - 2>&1 | grep mean_volume
  ```
- **Word count looks low.** Compare words per minute of *speech*, not per minute of
  file. A recording that is half silence looks sparse but is fine.

## Step 4: report back

Give the output path, duration, speakers detected, turn count, and the
quality-check verdict. Then offer the follow-up (identifying speakers by name,
extracting decisions and actions), without doing it unasked.

## Notes for whoever maintains this

- `--fast` is **not** "MLX instead of WhisperX". It is WhisperX's *architecture*
  (VAD-fixed chunk boundaries, independent per-chunk decode, no timestamp sampling)
  running on the MLX engine. Calling MLX-Whisper the obvious way was tried first and
  fabricated text badly, including silently dropping explicit instructions from a
  real meeting. Every setting in `_run_fast_asr_and_align` has a measurement behind
  it in `EXPERIMENTS.md`. Do not "tidy" them without re-running that comparison.
- Transcripts are confidential meeting content. Quote what is needed; do not paste
  whole transcripts into other contexts.
