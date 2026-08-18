# Backend evaluation: what was measured, what was rejected, what to try next

Record of the pipeline's development, 2026-08-04 to 2026-08-05. Read this before
changing the ASR path: several settings that look arbitrary are load-bearing, and
several plausible-sounding "improvements" were measured and found harmful.

This is an engineering log, not a benchmark paper: everything below was measured on
a small number of real recordings used for local validation during development,
never uploaded anywhere. Names, companies and other identifying details from those
recordings have been generalized or removed; only what mattered for engineering
decisions (word counts, timings, which text was invented or dropped) is kept.

**Test material.** Nearly everything below was measured on one recording ("Recording
1"): 87 minutes (5450s), Spanish, 2 speakers, phone-quality `.m4a`. A second Spanish
1:1 ("Recording 2", 56 minutes) was added at the end as independent validation.
Hardware: M5 Pro, 18 cores, 48GB unified memory.

**Reference transcript.** Produced by the all-WhisperX CPU path. Not hand-verified 
ground truth, just the longest-validated output, and later found to *miss* some content 
that a better pipeline caught. Comparisons phrased as "vs reference" mean vs that file.

## Shipped configuration

Two modes, both running ASR+alignment in one process concurrently with pyannote
diarization on MPS in another:

| | `--fast` (default) | `--safe` |
|---|---|---|
| ASR | MLX-Whisper `large-v3` on the GPU, VAD-fixed chunks, independent per-chunk decode | faster-whisper `large-v3`, CPU, `int8` |
| Alignment | wav2vec2 (WhisperX), CPU | same |
| Diarization | pyannote `speaker-diarization-community-1`, MPS | same |
| 87-min recording, end to end | ~6 to 7 min | 27m26s |

## Part 1: speed wins that cost nothing

| Configuration | Wall clock | Output vs reference |
|---|---|---|
| Sequential, all CPU | 79m00s | reference |
| Parallel (ASR+align ‖ diarize), all CPU | 63m39s | byte-for-byte identical |
| Parallel, diarization on MPS | 27m26s | byte-for-byte identical |

1. **Parallelization.** ASR and diarization do not depend on each other.
   Diarization needs only the waveform, not the text; they meet at the final
   `assign_word_speakers` merge. Running them as two processes: 79 to 64 minutes.
   The gain is below the theoretical max because both compete for the same cores
   (CPU averaged 622% of 1800% available).

2. **Diarization on MPS.** 183s versus ~44 min on CPU, roughly 14x. Validated
   rather than trusted, because pyannote on MPS has a documented history of wrong
   timestamps and silently corrupted output. Raw turn boundaries do jitter versus
   CPU (mean 0.9s, max 10.5s), but the jitter lives in silence between words: the
   final `.md` came out byte-for-byte identical. **The correct test was the merged
   output, not the intermediate boundaries.**

### Component timings

| Step | Device | Time |
|---|---|---|
| faster-whisper ASR (`large-v3`, `int8`) | CPU | ~33.5 min |
| MLX-Whisper ASR, VAD-fixed chunks | GPU | **~4 min** |
| wav2vec2 word alignment | CPU | ~1.5 to 2 min |
| pyannote VAD | MPS | ~14s |
| pyannote diarization | CPU | ~44 min |
| pyannote diarization | MPS | ~3 min |

## Part 2: the MLX detour, and what it taught

MLX-Whisper called the obvious way, `mlx_whisper.transcribe()`, is much faster than
faster-whisper but **fabricated text**. Three attempts to fix it by preprocessing:

| Variant | ASR time | Severe loops (>=8 repeats) | Other damage |
|---|---|---|---|
| MLX plain | 575.6s | 14, worst `"la"` x220 | ~2 min invented "chao chao chao" over closing silence |
| MLX + VAD chunking via `clip_timestamps` | 584.4s | 11, **worst overall**: `"sí"` x221 and x215, `"pues"` x19 | 14% fewer words; excluded real interior speech |
| MLX + `loudnorm` audio | 710.7s | 8, worst `"absolutamente"` x61 | ~3.5 min invented tail; 24% slower |

Notes that mattered later:

- The `"pues"` x19 loop landed **inside substantive content**, mid-way through
  a speaker's own opening self-assessment. Not garbage in an obvious place.
- **Audio normalization is not the fix.** Reasonable hypothesis (both loaders do a
  bare 16kHz mono ffmpeg decode with no gain handling, so it was untested), and it
  was the best of the three, but it still hallucinated and cost 24% more time.
- **`clip_timestamps` chunking is not WhisperX chunking.** It looked equivalent but
  routes clips through the same sequential loop with the same shared state. This
  mattered: it is why the first VAD attempt failed, and why the second (Part 4)
  succeeded.

## Part 3: decoding guards

| Variant | ASR time | Severe loops | Words | Name mentions |
|---|---|---|---|---|
| MLX plain | 575.6s | 14, `"la"` x220 | 14934 | 65 |
| **A: `condition_on_previous_text=False`** | **295.6s** | **0** | 13194 | 62 |
| B: A + `hallucination_silence_threshold=2.0` | 350.8s | 0 | 12933 | 53 |
| C: A + `initial_prompt` primer | 292.5s | 1, an invented name x51 | 13279 | 60 |
| D: C + `no_speech_threshold=0.9` | 299.5s | 1, an invented name x51 | 13335 | 64 |
| E: A + trailing-silence trim | 280.8s + 14s VAD | **0** | 13084 | 62 |

- **`condition_on_previous_text=False` (A) removed every severe loop, and halved
  runtime.** The docstring warns carryover makes the model "prone to getting stuck
  in a failure loop". The speedup was unexpected: loops burn decode tokens, and
  carryover lengthens every window's prompt. Note WhisperX sets this `False` too.
- **`hallucination_silence_threshold` (B) is a trap.** It does nothing unless
  `word_timestamps=True` (verified in source: the block sits inside an
  `if word_timestamps:` gate). Enabled properly it cost recall badly and did not
  fix the trailing fabrication it targets.
- **`initial_prompt` priming (C, D) made things worse**, introducing a garbled
  repeated-name loop (one invented name, repeated x51) absent from the meeting.
  See Part 4 for the actual mechanism; the primer itself only reaches the first
  window.
- **Trailing-silence trim (E) works and is nearly free.** VAD located the end of
  speech at 5217s against the reference's 5216s.

### Why E still was not good enough

E is clean of hallucinations and 7x faster, and was shipped opt-in with warnings.
But its residual word loss was **concentrated, not spread**:

| Window | ref | E/A | delta |
|---|---|---|---|
| 70-80 min | 1602 | 1301 | **-19%** |
| 40-50 min | 1769 | 1590 | -10% |
| 20-30 min | 1703 | 1571 | -8% |
| all others | | | -5% to +5% |

Reading that window against the reference, the deficit was three of one speaker's
explicit instructions about a plan.

A 220-repeat loop is loud and catchable. Silently dropping a mandate is neither.

## Part 4: the actual fix, fixed segmentation (variant F, now `--fast`)

**Root cause.** In the reference Whisper loop that `mlx_whisper.transcribe()`
implements, the next window's position is derived from the timestamps just decoded:

```
seek = round(last_word_end * FRAMES_PER_SECOND)     # transcribe.py:429
seek += last_timestamp_pos * input_stride           # transcribe.py:390
```

Segmentation is therefore **output-dependent and self-reinforcing**. One bad window
shifts every later boundary, so errors cascade, and any perturbation reshuffles them
across the entire file. This explains the whole pattern of results above: why losses
concentrate in unpredictable windows, and why the `initial_prompt` primer appeared
to "cause" a loop 41 minutes into the recording when it only reaches window one. It
changed window one, which moved `seek`, which re-cut every subsequent window.

WhisperX's real insight is not its engine, it is that **VAD fixes every boundary
before any decoding happens**, so segmentation cannot drift and is reproducible.

**What `--fast` does**, mirroring `whisperx/asr.py` exactly:

1. pyannote VAD, `merge_chunks(chunk_size=30)`, giving 201 fixed chunks for the test
   recording (avg 24.9s). Boundaries set before any decoding.
2. Per chunk: slice the audio, pad to 30s, `model.decode()` **independently**,
   bypassing `transcribe()` entirely (WhisperX does the same with CTranslate2). No
   cascade can exist because no state crosses chunks.
3. `without_timestamps=True`, matching WhisperX: the model never samples timestamp
   tokens, which is what drove the cascade. Segment times come from VAD, word times
   from alignment.
4. Temperature fallback `[0.0 ... 1.0]` on `compression_ratio > 2.4` or
   `avg_logprob < -1.0`, same thresholds as WhisperX.
5. `whisperx.align()` downstream, unchanged.

**Result on Recording 1 (87 minutes):**

| | reference (`--safe`) | A/E (rejected) | **F (`--fast`)** |
|---|---|---|---|
| ASR time | ~2010s | 295.6s | **260.9s** (14s VAD + 247s ASR) |
| Words | 13837 | 13194 (95.4%) | **13717 (99.1%)** |
| Name mentions | 73 | 62 (85%) | **76 (104%)** |
| Severe loops | 0 | 0 | **0** |
| 70-80 min window | 1602 | 1301 (81%) | **1545 (96.4%)** |
| One speaker's three mandates | present | **lost** | **present, verbatim** |
| Chunks needing fallback | n/a | n/a | **2 of 201** |

All three mandates were verified by reading the passage, not by keyword counting,
and match the reference word for word. One recurring proper name appeared 28 times
(reference: 24); each of the 28 was checked individually: all distinct, real
contexts, so F recovered four mentions the reference itself missed.

Fixed segmentation was the entire mechanism. It is also *faster* than the cascading
loop, since no compute is spent decoding silence or re-decoding drifted windows.

## Part 5: independent validation on a second recording

`--fast` run on Recording 2, a second 1:1, 56 minutes (3362s), Spanish, 2 speakers,
never used during development.

| | result |
|---|---|
| Wall clock, full pipeline | **3m24s** for 56 minutes of audio |
| Language detection | correct (`es`), by vote across the longest opening speech chunks |
| Hallucination loops | none |
| Trailing filler | none |
| Unattributed words | 1 |
| Speech captured | all of it, verified (below) |

Two things looked alarming at first and both turned out to be properties of the
recording, not pipeline faults. Recording the reasoning because the same alarms will
recur:

1. **Transcript ends at 48:06 on a 56:02 file.** VAD places the last speech at
   2886.0s, exactly where the transcript ends. The recording simply ran for ~8
   minutes after the conversation finished.
2. **Word density looked half the other recording** (85 words per wall-clock minute
   versus 159). Normalised against actual speech time the two agree closely: **167
   versus 185 words per minute of speech**. This recording is only 51% speech
   (24 min of talking in a 48 min span) against 86% for Recording 1.

That second point prompted a check of whether VAD was under-detecting quiet speech,
since `--fast` depends entirely on VAD. There were two very large excluded gaps,
847.8s (14.1 min) and 302.0s (5 min). Measured loudness inside them:

| Region | mean volume |
|---|---|
| Gap 1 (1922-2770s, untranscribed) | -46.6 dB |
| Gap 2 (1555-1857s, untranscribed) | -45.3 dB |
| Known silence (tail, 2890s+) | -47.4 dB |
| Known speech (600-720s) | -29.8 dB |

The gaps match silence, not speech, so VAD was correct. Overall file loudness is
also comparable between the two recordings (-31.4 dB vs -29.2 dB), ruling out a
systematic quiet-audio effect. (All the silent regions do show loud transient peaks
around -3 to -6 dB, clicks and bumps, with no sustained speech.)

**A validation-design point worth remembering: `--safe` is not an independent check
on VAD.** WhisperX runs the same pyannote VAD with the same onset/offset internally,
so speech that VAD misses is invisible to *both* modes. Cross-checking the two modes
tests the decode path only. Testing VAD recall requires something else, either
audio-level analysis as above, or transcribing without VAD and looking for coherent
content inside the excluded regions.

Minor observed artifact: Proper nouns remain the weakest spot in both
modes.

## Part 6: Windows `--safe` was silently running CPU-only (2026-08-18)

Different platform (Windows, RTX 5070 Ti), different failure class from everything
above: not an accuracy problem, a wiring problem. `--safe` ran entirely on CPU
(`whisperx cpu/int8` and diarization on `cpu`) on a machine with a capable GPU
sitting idle.

**Root cause.** `transcribe.py` picks its device from `torch.cuda.is_available()`
(line 56), which cascades into `diarize_device`, `asr_device`, and `compute_type`
(lines 343-347) — correct code, fed a wrong premise. The venv's `torch` was
`2.8.0+cpu`. On Windows, the default PyPI index only serves CPU torch wheels; CUDA
builds exist solely on `download.pytorch.org`. `setup.ps1` detected the GPU but
only printed a hint suggesting a manual `uv pip install`, it never ran it. The old
version of that hint was also wrong for current hardware: it pointed at `cu124`,
which installs cleanly on an RTX 5070 Ti (Blackwell, `sm_120`) and then fails at
first kernel launch with "no kernel image is available for execution". `cu128` is
the correct index; its wheels cover `sm_70` through `sm_120`, so no
compute-capability branching is needed.

One thing that was *not* broken: `ctranslate2` 4.8.1 (the `--safe` ASR backend)
ships its own bundled CUDA runtime and `cudnn64_9.dll`, and reported one usable
CUDA device the whole time. Only the torch-mediated stages — VAD, alignment,
diarization — were stranded on CPU.

**A second trap on the fix itself.** The obvious repair,
`uv pip install --index-url .../cu128 torch==2.8.0 ...`, silently no-ops on a
machine that already has `2.8.0+cpu` installed: PEP 440 treats the bare
`==2.8.0` specifier as satisfied by `2.8.0+cpu`, since the requirement carries no
local-version segment. `uv` reports "Checked N packages" and does nothing.
`--reinstall` is required to force the swap.

**Fix shipped in `setup.ps1` (stamp bumped `1` → `2`):** if `nvidia-smi` reports a
GPU and the venv's torch has no CUDA build, force-reinstall
`torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0` from the cu128 index with
`--reinstall`, ordered after the base `requirements.txt` install (which pulls the
CPU torch as a `whisperx` dependency — installing CUDA torch first would just get
overwritten). Idempotency check is a dedicated
`python -c "import torch, sys; sys.exit(0 if torch.version.cuda else 1)"`, not the
existing `Test-PackagesPresent` (which only imports `whisperx` and is happy with a
CPU-only install). Failure to install (no network, driver too old) is reported
plainly and setup still exits 0 — CPU is a valid, working fallback, not an error
state.

**Verified before/after on this machine:**

| | before | after |
|---|---|---|
| `torch.__version__` | `2.8.0+cpu` | `2.8.0+cu128` |
| `torch.version.cuda` | `None` | `12.8` |
| `torch.cuda.is_available()` | `False` | `True` |
| `torch.cuda.get_device_name(0)` | n/a | `NVIDIA GeForce RTX 5070 Ti` |
| transcribe.py log line | `safe: whisperx cpu/int8, 8 threads` / diarization (cpu) | `safe: whisperx cuda/float16, 8 threads` / diarization (cuda) |
| GPU memory during a run | ~0 | ~9-10 GB |

## Corrections made along the way

Two errors worth recording, because both were caught by verifying a proxy against
the underlying text, and both had pointed the wrong way:

1. **Name counts were case-sensitive.** MLX lowercases proper nouns more often than
   WhisperX, so every early name-count comparison understated MLX's recall. All 
   figures above are case-insensitive. The reference has 2 mentions of one company
   name, not 1. Hallucination findings were unaffected, since invented text does not
   depend on capitalization.
2. **"The primer caused the invented-name loop" was wrong.** With carryover disabled, an
   `initial_prompt` cannot reach minute 41. The real mechanism was the segmentation
   cascade, which is what led to Part 4.

The general lesson: proxy metrics (word counts, name counts, a loop-detection
regex) were good enough to reject a backend emitting 220-token loops, but the
decisive question, "did it drop anything that matters", could only be answered by
reading two transcripts side by side.

## Experimentation path, highest value first

### 1. Validate `--fast` more widely (the current gap)

`--fast` is default on the strength of two Spanish 2-speaker 1:1s. Untested and
most likely to stress VAD-based segmentation:

- **3+ speakers with crosstalk and overlapping turns.** VAD merges overlapping
  speech into one region, so a chunk may contain two people talking at once. Both
  ASR and the downstream speaker assignment could degrade.
- **Other languages, and code-switching.** Alignment loads one language model per
  run, so heavy Spanish/English mixing mid-sentence degrades timestamp precision
  (not text). Plausible in any meeting with bilingual speakers.
- **Short recordings.** Language detection now votes across the longest of the first
  six speech chunks, which is more robust than sampling the head of the file, but is
  untested under 5 minutes. `--language` remains the reliable override.
- **Poor audio.** Both test recordings are phone-quality but close-mic. Room audio
  with distance and echo is the harder case.

Cheapest useful check: run any new recording both ways and diff. `--safe` exists
partly for that.

### 2. A real accuracy harness

The whole evaluation rests on proxies, one of which was wrong in a flattering
direction, and the decisive comparison had to be done by hand. That does not scale
and cannot be re-run automatically after a change.

`pyannote.metrics` is already installed (collar-aware DER). `jiwer` handles WER. The
missing piece is one hand-corrected reference transcript, which is human listening
time, not code. Both test recordings are candidates.

### 3. Batching `--fast` (speed only, no quality effect)

`--fast` decodes chunks one at a time in a Python loop. WhisperX batches 8. Batched
decoding is **already implemented** in mlx_whisper: `decode()` accepts `mel.ndim == 3`,
`n_audio = mel.shape[0]`, and the internal tensors carry `(n_audio, n_group, seq)`
shapes with lockstep generation and masking handled.

What remains: `decode_with_fallback` compares scalars (`.compression_ratio`,
`.avg_logprob`), so per-element retry needs writing, run the batch at temperature 0,
select failures by index, re-run only those hotter, merge. Roughly 40 lines plus mel
stacking. Payoff is unpredictable (one 30s chunk may already be saturating the GPU),
so measure before investing.

### 4. Other engines (low value now)

- **whisper.cpp**: also a reference-algorithm port, so it inherits the same cascade
  and would need the same Part 4 treatment. Its one real edge is Core ML / ANE for
  the encoder. But with `--fast` ASR at ~4 min, ASR is no longer the bottleneck, so
  the ceiling on any engine swap is now small. Skip.
- **lightning-whisper-mlx**: same caveat, same required scrutiny.
- **NVIDIA NeMo / parakeet**: CUDA-only, not viable on Macs.

### 5. Diarization refinements

- **`UNKNOWN` speaker blocks**: words falling outside every diarization turn. The
  `--safe` reference run had none; MLX runs produced a handful (5 blocks, 7 words).
  If they grow on multi-speaker recordings, options are padding turn boundaries or
  nearest-turn assignment for orphaned words.
- **pyannote `precision-2`** (paid pyannoteAI API) scores better on DER but sends
  audio off-device. Rejected on confidentiality grounds, not quality. Not
  revisitable for client meetings.

## Method note

Where a claim above says "identical", it means `diff` produced zero differences on
the final `.md`. Where it says "0 severe loops", it means zero found by one detector
(`check_transcript.py`) for one failure mode, on one or two recordings, which is not
a guarantee of no hallucination. `check_transcript.py` was itself verified in both
directions: it passes the reference and flags a synthetic bad transcript with a
non-zero exit code.
