# transcribe

A [Claude Code](https://claude.com/claude-code) skill that turns a meeting
recording into a speaker-tagged Markdown transcript, **entirely on your own
machine**. No audio and no transcript is ever uploaded anywhere, so data privacy 
is kept at a maximum.

```
**SPEAKER_00** `00:02:40 - 00:10:30`
That's ok, for what you are saying, we are on track...

**SPEAKER_01** `00:10:32 - 00:20:39`
Ok, let's run the checklist to be sure...
```

Give it an audio or video file; it gives back speaker-separated text with
timestamps. It does **not** figure out who's who by name, and it does not
summarize or extract action items: that's a deliberate scope cut, meant as a
clean first pass for a follow-up step to build on.

## How it works, in short

Two independent stages run in parallel and merge at the end:

- **Speech-to-text**, via [WhisperX](https://github.com/m-bain/whisperX)
  (`--safe`, works everywhere) or a custom pipeline that runs WhisperX's
  algorithm on Apple's MLX engine (`--fast`, Apple Silicon only, ~7x faster).
- **Speaker diarization**, via
  [pyannote.audio](https://github.com/pyannote/pyannote-audio).

Why it's built this way, including a failed first attempt and how it was
diagnosed and fixed, is documented in detail in
[`EXPERIMENTS.md`](EXPERIMENTS.md).

## Install

1. Copy this whole `transcribe` folder into your Claude Code skills directory:

   - **macOS / Linux:** `~/.claude/skills/transcribe`
   - **Windows:** `%USERPROFILE%\.claude\skills\transcribe`

   ```bash
   mkdir -p ~/.claude/skills
   cp -R /path/to/transcribe ~/.claude/skills/
   ```

   (Or drop it in a project's `.claude/skills/` folder if you only want it
   available there. Or `git clone` this repo straight into the skills
   directory.)

2. Start Claude Code and ask it to transcribe something, for example:
   > *"transcribe ~/Downloads/my-meeting.m4a, it was a 1:1"*

   **The first run sets itself up automatically**: installs `ffmpeg` and
   `uv`, creates a Python 3.11 environment, installs the pinned packages, and
   (on Windows, if an NVIDIA GPU is present) installs a CUDA build of
   PyTorch. See [What gets installed](#what-gets-installed) below for exact
   sizes — budget several GB and a slow first run. Every run after that
   starts instantly.

3. **One manual step only you can do: a free Hugging Face token.** Speaker
   diarization uses a gated model, so an unauthenticated download will fail.

   - Create a **read** token at <https://huggingface.co/settings/tokens>
   - While logged in, accept the terms at
     <https://huggingface.co/pyannote/speaker-diarization-community-1>
   - Save it in `.env` inside this folder:
     ```bash
     echo 'HF_TOKEN=hf_your_token_here' > ~/.claude/skills/transcribe/.env
     chmod 600 ~/.claude/skills/transcribe/.env
     ```

   The token only authorizes the one-time model download. Your recordings
   are never uploaded anywhere, and neither is the token.

You can also run setup by hand, without going through Claude Code:

```bash
bash ~/.claude/skills/transcribe/scripts/setup.sh          # macOS / Linux
```
```powershell
powershell -ExecutionPolicy Bypass -File "$env:USERPROFILE\.claude\skills\transcribe\scripts\setup.ps1"   # Windows
```

Both scripts are safe to re-run any time — they only install what's
missing and pick up where they left off.

## What gets installed

Nothing here is bundled in this repo; setup downloads all of it on first
run, measured on a real install:

| Component | Installed size | When |
|---|---|---|
| `ffmpeg` | ~650 MB | setup, once |
| `uv` | ~50 MB | setup, once |
| Python 3.11 runtime (managed by `uv`) | ~80 MB | setup, once |
| Python packages (WhisperX, PyTorch CPU, etc.) | **~1.9 GB** on CPU-only platforms | setup, once |
| ↳ same, but with CUDA PyTorch | **~7.7 GB** (Windows/Linux + NVIDIA GPU only) | setup, once |
| `faster-whisper large-v3` model (`--safe` ASR) | ~2.9 GB | first transcription |
| `pyannote` speaker diarization model | ~32 MB | first transcription |
| `wav2vec2` word-alignment model | ~360 MB per language | first time each language is used |

**Rough total for a first real transcription:** about **6 GB** on a CPU-only
machine (Linux, Windows without an NVIDIA GPU, Intel Mac), or about **11–12
GB** on Windows/Linux with an NVIDIA GPU, since the CUDA build of PyTorch
bundles its own copy of the CUDA/cuDNN runtime libraries. On Apple Silicon,
`--fast` additionally pulls MLX Whisper weights instead of the
`faster-whisper` model above — a comparable multi-GB download, not
separately measured here.

None of this touches this repo's size on disk (see
[Repository layout](#repository-layout)); it all lives in `.venv/` and your
Hugging Face / torch cache directories, both of which are safe to delete if
you ever want to reclaim the space (setup will just redo the download).

## What you get on your hardware

| | `--fast` (default) | `--safe` |
|---|---|---|
| Where it runs | Apple Silicon GPU | CPU, or NVIDIA GPU via CUDA |
| 90-minute meeting | **~7 minutes** | ~90 min on CPU, much less on CUDA |
| Available on | Apple Silicon Macs (M1 and later) | everything |

**Apple Silicon Mac:** both modes, `--fast` by default.
**Intel Mac / Windows / Linux:** `--safe` only. The script detects this and
falls back automatically, telling you why. MLX is Apple-only, so there's no
way around it. With an NVIDIA GPU, `--safe` is genuinely fast; on CPU,
budget roughly the length of the recording.

**Windows + NVIDIA GPU:** `setup.ps1` detects the GPU and installs a CUDA
build of PyTorch for you automatically (the default Windows wheel from PyPI
is CPU-only, so this is a separate step, done right). If it can't complete
that step (no network, driver too old), setup still finishes successfully
and the skill falls back to CPU — it tells you plainly which happened, and
you can just re-run setup later to retry.

## Using it

Normally you just ask Claude — see [`SKILL.md`](SKILL.md) for exactly what
it does on your behalf. Directly, from the command line:

```bash
# from the skill folder
.venv/bin/python scripts/transcribe.py ~/Downloads/meeting.m4a \
  --min-speakers 2 --max-speakers 2 --keep-json
```

Flags worth knowing:

| Flag | Why |
|---|---|
| `--min-speakers N --max-speakers N` | Tell it how many people were in the room. Materially improves diarization. |
| `--language es` | Skip autodetect. Recommended under ~5 minutes, where autodetect has misfired. |
| `--keep-json` | Word-level timestamps and per-word speaker data, needed by any later processing pass. |
| `--safe` | Force the slower, longest-validated path. |

Check any transcript before trusting it:

```bash
.venv/bin/python scripts/check_transcript.py output.md
```

This flags repetition-loop hallucinations, invented filler over closing
silence, and unattributed words. It exits non-zero if it finds something.

## Two things that look like bugs and usually are not

**The transcript ends before the recording does.** Usually the recording
just kept rolling after everyone stopped talking. Check the audio level in
the tail: silence sits around -45 dB or lower, speech around -30 dB.

```bash
ffmpeg -hide_banner -nostats -ss 2900 -i recording.m4a -af volumedetect -f null - 2>&1 | grep mean_volume
```

**The word count looks low.** Compare words per minute of *speech*, not per
minute of file. One validated recording was only 51% speech and looked
sparse until normalized, at which point it matched a denser recording
closely.

## Known limitations

- Install of dependencies validated on Mac and Windows, untested on Linux.
- Validated on 2-speaker 1:1s. **3+ speakers with crosstalk is untested**
  and is where VAD-based segmentation is most likely to struggle.
- Proper nouns are the weakest spot in both modes.
- Alignment loads one language model per run, so heavy mid-sentence
  code-switching degrades timestamp precision, though not the text itself.
- Speaker labels are generic (`SPEAKER_00`, `SPEAKER_01`, ...). Naming them
  is a separate, later pass this skill deliberately doesn't attempt.

## Repository layout

```
transcribe/
├── SKILL.md              # instructions Claude Code follows to run this skill
├── README.md / README.txt
├── EXPERIMENTS.md         # the engineering log behind the design decisions
├── LICENSE                 # MIT
├── .env.example           # copy to .env and add your Hugging Face token
└── scripts/
    ├── setup.sh            # macOS / Linux first-run setup
    ├── setup.ps1           # Windows first-run setup
    ├── requirements.txt
    ├── transcribe.py       # the actual pipeline
    └── check_transcript.py # post-hoc quality check
```

Everything setup creates — the `.venv/` Python environment, your `.env`
token, `.setup-stamp`, and any `output/` you generate — is listed in
[`.gitignore`](.gitignore) and never belongs in version control: `.venv/` is
multiple GB and platform-specific, and `.env` holds a personal credential.
A fresh `git clone` plus one setup run rebuilds all of it locally, on
whichever machine you're on.

## License

[MIT](LICENSE) — do what you want with it, just keep the copyright notice.

## Why it's built this way

`--fast` is **not** "MLX instead of WhisperX". It's WhisperX's *architecture*
(VAD-fixed chunk boundaries, independent per-chunk decoding, no timestamp
sampling) running on the MLX engine. Calling MLX-Whisper the obvious way was
tried first and fabricated text badly: repetition loops up to 220 tokens,
and, once those were fixed, silently dropping explicit instructions from a
real meeting.

[`EXPERIMENTS.md`](EXPERIMENTS.md) has the full record: what was measured,
what was rejected and why, the errors made along the way, and a ranked list
of what to try next. Read it before changing the ASR path — several settings
that look arbitrary are load-bearing.
