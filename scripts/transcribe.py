#!/usr/bin/env python3
"""
Recording -> speaker-tagged .md transcript. Runs entirely on your machine:
no audio, and no transcript, ever leaves it.

Two modes:
  --fast (default, Apple Silicon only)
      WhisperX's architecture running on the MLX engine, so the Apple GPU does
      the ASR. Roughly 7x faster than --safe, and matched reference quality on
      the recordings it was validated against.
  --safe
      The all-WhisperX path. Slower, works on every platform, and uses CUDA
      automatically if you have an NVIDIA GPU.

On anything that is not Apple Silicon, --fast is unavailable and the script
falls back to --safe automatically (MLX is Apple-only).

Speakers come out labelled SPEAKER_00, SPEAKER_01, ... Identifying who is who,
and pulling out decisions or action items, is a separate later pass.

Usage:
    python transcribe.py recording.m4a
    python transcribe.py recording.mp3 --min-speakers 2 --max-speakers 4
    python transcribe.py recording.mp4 --language es --safe
"""

import argparse
import concurrent.futures
import datetime
import importlib.util
import json
import os
import platform
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

SAMPLE_RATE = 16000
VAD_ONSET, VAD_OFFSET, CHUNK_SIZE = 0.500, 0.363, 30
# WhisperX's own fallback ladder and thresholds (whisperx/asr.py)
TEMPERATURES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
COMPRESSION_RATIO_THRESHOLD, LOGPROB_THRESHOLD = 2.4, -1.0

IS_APPLE_SILICON = platform.system() == "Darwin" and platform.machine() == "arm64"


def refresh_windows_path() -> None:
    """A shell that predates a first-run winget install (ffmpeg, uv) won't see the
    new PATH entries until it restarts: Windows only broadcasts PATH changes to new
    processes. Rebuild PATH from the registry so this process, and anything it
    spawns (whisperx's internal ffmpeg call, the worker processes below), find
    freshly installed tools without the user having to open a new terminal."""
    if platform.system() != "Windows":
        return
    import winreg
    segments = [os.environ.get("PATH", "")]
    for hive, subkey in (
        (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
        (winreg.HKEY_CURRENT_USER, "Environment"),
    ):
        try:
            with winreg.OpenKey(hive, subkey) as key:
                value, _ = winreg.QueryValueEx(key, "Path")
                segments.append(value)
        except OSError:
            pass  # best-effort: if the registry read fails, the original PATH still applies
    os.environ["PATH"] = ";".join(s for s in segments if s)


def fast_mode_available() -> bool:
    """--fast needs MLX, which only exists on Apple Silicon."""
    return IS_APPLE_SILICON and importlib.util.find_spec("mlx_whisper") is not None


def pick_accelerator() -> str:
    """Best available device for the PyTorch-based stages (VAD, diarization, alignment)."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------

def load_env_file(env_path: Path) -> dict:
    values = {}
    if not env_path.exists():
        return values
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def get_hf_token(cli_token) -> str:
    if cli_token:
        return cli_token
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    for candidate in (HERE / ".env", HERE.parent / ".env"):
        token = load_env_file(candidate).get("HF_TOKEN")
        if token:
            return token
    sys.exit(
        "No Hugging Face token found.\n"
        "Diarization uses a gated model, so a free read token is required:\n"
        "  1. Create one at https://huggingface.co/settings/tokens\n"
        "  2. Accept the terms at "
        "https://huggingface.co/pyannote/speaker-diarization-community-1\n"
        f"  3. Put HF_TOKEN=hf_... in {HERE.parent / '.env'}\n"
        "     (or pass --hf-token, or export HF_TOKEN)"
    )


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def format_timestamp(seconds: float) -> str:
    total = round(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def merge_consecutive_speakers(segments: list) -> list:
    """Collapse consecutive segments from the same speaker into one readable block."""
    blocks = []
    for seg in segments:
        speaker = seg.get("speaker", "UNKNOWN")
        text = seg.get("text", "").strip()
        if not text:
            continue
        if blocks and blocks[-1]["speaker"] == speaker:
            blocks[-1]["end"] = seg["end"]
            blocks[-1]["text"] += " " + text
        else:
            blocks.append({"speaker": speaker, "start": seg["start"], "end": seg["end"], "text": text})
    return blocks


def render_markdown(audio_path: Path, result: dict, engine: str, language: str) -> str:
    segments = result.get("segments", [])
    blocks = merge_consecutive_speakers(segments)
    speakers = sorted({b["speaker"] for b in blocks})
    duration = segments[-1]["end"] if segments else 0

    lines = [
        f"# Transcript: {audio_path.name}",
        "",
        f"*Processed {datetime.date.today().isoformat()} with {engine} "
        f"+ pyannote/speaker-diarization-community-1*",
        f"*Detected language: {language}*",
        f"*Duration: {format_timestamp(duration)}*",
        f"*Speakers detected: {len(speakers)} ({', '.join(speakers)}), not yet identified by name*",
        "",
        "---",
        "",
    ]
    for block in blocks:
        lines.append(f"**{block['speaker']}** `{format_timestamp(block['start'])} - "
                     f"{format_timestamp(block['end'])}`")
        lines.append(block["text"])
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# shared VAD helpers
# --------------------------------------------------------------------------

def _vad_chunks(audio, hf_token: str, vad_device: str):
    """Speech regions merged to <=CHUNK_SIZE. Boundaries are fixed here, before any
    decoding: that is the whole point of the fast path (see EXPERIMENTS.md)."""
    import torch
    from whisperx.vads import Pyannote

    vad = Pyannote(torch.device(vad_device), token=hf_token,
                   vad_onset=VAD_ONSET, vad_offset=VAD_OFFSET)
    raw = vad({"waveform": vad.preprocess_audio(audio), "sample_rate": SAMPLE_RATE})
    return Pyannote.merge_chunks(raw, chunk_size=CHUNK_SIZE, onset=VAD_ONSET, offset=VAD_OFFSET)


def _chunk_mel(audio, chunk, n_mels):
    import mlx.core as mx
    from mlx_whisper.audio import N_FRAMES, N_SAMPLES, log_mel_spectrogram, pad_or_trim

    clip = audio[int(chunk["start"] * SAMPLE_RATE):int(chunk["end"] * SAMPLE_RATE)]
    mel = log_mel_spectrogram(clip, n_mels=n_mels, padding=max(0, N_SAMPLES - clip.shape[0]))
    return pad_or_trim(mel, N_FRAMES, axis=-2).astype(mx.float16)


def _detect_language(model, audio, chunks):
    """Detect on real speech rather than the head of the file, which may be silence.
    Short samples misdetect (measured: a 4s English clip came back as Spanish), so
    vote across the longest of the opening chunks."""
    from collections import Counter

    candidates = sorted(chunks[:6], key=lambda c: c["end"] - c["start"], reverse=True)[:3]
    votes = []
    for chunk in candidates:
        # detect_language unwraps to a single dict for non-batched (2D) mel input
        _, probs = model.detect_language(_chunk_mel(audio, chunk, model.dims.n_mels))
        votes.append(max(probs, key=probs.get))
    return Counter(votes).most_common(1)[0][0]


# --------------------------------------------------------------------------
# ASR paths (each runs in its own process, concurrently with diarization)
# --------------------------------------------------------------------------

def _run_fast_asr_and_align(audio_path: str, mlx_model: str, align_device: str, language,
                             hf_token: str, vad_device: str, threads: int):
    """
    WhisperX's architecture on the MLX engine. Apple Silicon only.

    Why this exists, and what must not be "simplified" (every claim here has a
    measurement behind it in EXPERIMENTS.md):

    The reference Whisper loop that mlx_whisper.transcribe() implements advances its
    window using the timestamps it just decoded, so segmentation is output-dependent
    and self-reinforcing: one bad window shifts every later boundary, errors cascade,
    and any small perturbation reshuffles them across the whole file. Measured
    consequences on a real 87 minute meeting: repetition loops up to 220 repeats of a
    single token, and once those were fixed, a concentrated 19% word loss in one
    passage that silently dropped three explicit instructions.

    This path fixes every boundary with VAD *before* decoding anything and decodes
    each chunk independently, exactly as WhisperX does with CTranslate2. That
    recovered 99.1% of the reference's words, 104% of its name mentions and all three
    dropped instructions verbatim, with zero severe loops, while being faster than
    the cascading loop (no compute wasted on silence or on re-decoding drifted
    windows).
    """
    import torch
    torch.set_num_threads(threads)
    import mlx.core as mx
    import whisperx
    from mlx_whisper.decoding import DecodingOptions
    from mlx_whisper.transcribe import ModelHolder

    audio = whisperx.load_audio(audio_path)
    chunks = _vad_chunks(audio, hf_token, vad_device)
    if not chunks:
        raise RuntimeError("VAD found no speech in this recording, nothing to transcribe")

    model = ModelHolder.get_model(mlx_model, mx.float16)
    lang = language or _detect_language(model, audio, chunks)

    segments = []
    for chunk in chunks:
        mel = _chunk_mel(audio, chunk, model.dims.n_mels)
        result = None
        for temperature in TEMPERATURES:
            result = model.decode(mel, DecodingOptions(
                language=lang, task="transcribe", temperature=temperature,
                # the model never samples timestamp tokens: that is what drove the
                # cascade. Segment times come from VAD, word times from alignment.
                without_timestamps=True, suppress_blank=True, suppress_tokens=[-1], fp16=True,
            ))
            degenerate = (result.compression_ratio > COMPRESSION_RATIO_THRESHOLD
                          or result.avg_logprob < LOGPROB_THRESHOLD)
            if not degenerate:
                break
        text = result.text.strip()
        if text:
            segments.append({"start": float(chunk["start"]), "end": float(chunk["end"]), "text": text})

    model_a, metadata = whisperx.load_align_model(language_code=lang, device=align_device)
    aligned = whisperx.align(segments, model_a, metadata, audio, align_device,
                             return_char_alignments=False)
    return aligned, lang


def _run_safe_asr_and_align(audio_path: str, model_name: str, device: str, compute_type: str,
                             batch_size: int, language, threads: int):
    """The all-WhisperX path. Works everywhere; uses CUDA when available."""
    import torch
    torch.set_num_threads(threads)
    import whisperx

    audio = whisperx.load_audio(audio_path)
    model = whisperx.load_model(model_name, device, compute_type=compute_type, threads=threads)
    result = model.transcribe(audio, batch_size=batch_size, language=language)
    lang = result["language"]
    model_a, metadata = whisperx.load_align_model(language_code=lang, device=device)
    aligned = whisperx.align(result["segments"], model_a, metadata, audio, device,
                             return_char_alignments=False)
    return aligned, lang


def _run_diarize(audio_path: str, hf_token: str, device: str, min_speakers, max_speakers, threads: int):
    """Independent of ASR: needs only the raw audio, not the transcribed text."""
    import torch
    torch.set_num_threads(threads)
    import whisperx
    from whisperx.diarize import DiarizationPipeline

    audio = whisperx.load_audio(audio_path)
    diarize_model = DiarizationPipeline(token=hf_token, device=device)
    return diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)


# --------------------------------------------------------------------------

def main():
    refresh_windows_path()

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", type=Path, help="Path to the recording (audio or video)")
    parser.add_argument("-o", "--output", type=Path, default=None,
                         help="Output .md path (default: alongside the input)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fast", dest="mode", action="store_const", const="fast",
                       help="default on Apple Silicon. WhisperX's architecture on the MLX "
                            "engine, so the Apple GPU does the ASR. About 7x faster than --safe")
    mode.add_argument("--safe", dest="mode", action="store_const", const="safe",
                       help="the all-WhisperX path. Works on every platform, uses CUDA when "
                            "available. The longest-validated option")
    parser.set_defaults(mode="fast")
    parser.add_argument("--model", default="large-v3", help="Whisper model for --safe (default: large-v3)")
    parser.add_argument("--mlx-model", default="mlx-community/whisper-large-v3-mlx",
                         help="HF repo for the MLX weights (used by --fast)")
    parser.add_argument("--language", default=None,
                         help="Force a language code (es, en, ...). Default: autodetect. "
                              "Recommended for recordings under ~5 minutes")
    parser.add_argument("--min-speakers", type=int, default=None)
    parser.add_argument("--max-speakers", type=int, default=None)
    parser.add_argument("--device", default=None,
                         help="Device for ASR/alignment (default: cuda if available, else cpu)")
    parser.add_argument("--diarize-device", default=None,
                         help="Device for VAD/diarization (default: cuda, else mps, else cpu)")
    parser.add_argument("--compute-type", default=None,
                         help="faster-whisper compute type (default: float16 on cuda, int8 on cpu)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hf-token", default=None, help="Hugging Face token (overrides env/.env)")
    parser.add_argument("--keep-json", action="store_true",
                         help="Also write the structured result (word-level timestamps, per-word speaker)")
    parser.add_argument("--threads", type=int, default=None,
                         help="Total CPU threads to split between the two processes")
    args = parser.parse_args()

    if not args.audio.exists():
        sys.exit(f"File not found: {args.audio}")

    # --- resolve mode against what this machine can actually do ---
    if args.mode == "fast" and not fast_mode_available():
        explicitly_asked = "--fast" in sys.argv
        reason = ("MLX runs only on Apple Silicon"
                  if not IS_APPLE_SILICON else
                  "mlx-whisper is not installed (re-run the setup script)")
        if explicitly_asked:
            sys.exit(f"--fast is unavailable here: {reason}.\nUse --safe instead.")
        print(f"Note: --fast is unavailable here ({reason}). Falling back to --safe, "
              f"which is slower but works everywhere.")
        args.mode = "safe"

    accelerator = pick_accelerator()
    diarize_device = args.diarize_device or accelerator
    # CTranslate2 (the --safe ASR backend) has CUDA and CPU builds, but no Metal:
    # on Apple Silicon its work stays on the CPU regardless of MPS being present.
    asr_device = args.device or ("cuda" if accelerator == "cuda" else "cpu")
    compute_type = args.compute_type or ("float16" if asr_device == "cuda" else "int8")

    hf_token = get_hf_token(args.hf_token)

    total_threads = args.threads or os.cpu_count() or 8
    asr_threads = max(1, total_threads // 2)
    # a GPU-bound diarization barely uses CPU threads; a CPU-bound one wants its share
    diarize_threads = max(1, total_threads - asr_threads) if diarize_device == "cpu" else 4

    label = ("fast: mlx gpu, VAD-fixed chunks" if args.mode == "fast"
             else f"safe: whisperx {asr_device}/{compute_type}, {asr_threads} threads")
    print(f"Running ASR+alignment ({label}) and diarization ({diarize_device}) in parallel")

    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        if args.mode == "fast":
            asr_future = executor.submit(
                _run_fast_asr_and_align, str(args.audio), args.mlx_model, asr_device,
                args.language, hf_token, diarize_device, asr_threads,
            )
        else:
            asr_future = executor.submit(
                _run_safe_asr_and_align, str(args.audio), args.model, asr_device,
                compute_type, args.batch_size, args.language, asr_threads,
            )
        diarize_future = executor.submit(
            _run_diarize, str(args.audio), hf_token, diarize_device,
            args.min_speakers, args.max_speakers, diarize_threads,
        )
        result, language = asr_future.result()
        diarize_segments = diarize_future.result()

    print(f"Merging transcription (language={language}) with speaker labels")
    import whisperx
    result = whisperx.assign_word_speakers(diarize_segments, result)

    # record how it was made, so a reader months later knows which pipeline
    # (and therefore which known limitations) applied
    engine = (f"--fast mode: MLX-Whisper ({args.mlx_model.split('/')[-1]}) with fixed VAD "
              f"segmentation + WhisperX alignment"
              if args.mode == "fast" else f"--safe mode: WhisperX ({args.model}, {asr_device})")

    output_path = args.output or args.audio.with_suffix(".md")
    output_path.write_text(render_markdown(args.audio, result, engine, language), encoding="utf-8")
    print(f"Wrote {output_path}")

    if args.keep_json:
        json_path = output_path.with_suffix(".json")
        json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
