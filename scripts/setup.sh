#!/usr/bin/env bash
# Idempotent bootstrap for the transcribe skill (macOS / Linux).
# Safe to re-run: it installs only what is missing and re-stamps the version.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
VENV="$ROOT/.venv"
STAMP="$ROOT/.setup-stamp"
STAMP_VERSION="1"          # bump to force everyone to re-run the Python install

OS="$(uname -s)"
ARCH="$(uname -m)"
APPLE_SILICON=false
[ "$OS" = "Darwin" ] && [ "$ARCH" = "arm64" ] && APPLE_SILICON=true

say()  { printf '  %s\n' "$*"; }
step() { printf '\n==> %s\n' "$*"; }

step "Checking environment"
say "OS: $OS ($ARCH)"
if $APPLE_SILICON; then
  say "Apple Silicon detected: the fast GPU path (--fast) will be available."
else
  say "Not Apple Silicon: --fast is unavailable (MLX is Apple-only)."
  say "The skill will use --safe, which is slower but works here."
fi

# --- system packages ------------------------------------------------------
step "System dependencies (ffmpeg, uv)"

install_with_brew() {
  for pkg in "$@"; do
    if brew list --formula "$pkg" >/dev/null 2>&1; then
      say "$pkg: already installed"
    else
      say "$pkg: installing via Homebrew..."
      brew install "$pkg"
    fi
  done
}

need_ffmpeg=true;  command -v ffmpeg >/dev/null 2>&1 && need_ffmpeg=false
need_uv=true;      command -v uv     >/dev/null 2>&1 && need_uv=false

if $need_ffmpeg || $need_uv; then
  if command -v brew >/dev/null 2>&1; then
    pkgs=()
    $need_ffmpeg && pkgs+=(ffmpeg)
    $need_uv     && pkgs+=(uv)
    install_with_brew "${pkgs[@]}"
  elif [ "$OS" = "Darwin" ]; then
    cat <<'EOF'
  Homebrew is not installed, and it is the standard package manager on macOS.
  Install it first (this is a one-off, and it will ask for your password):

    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

  Then run this setup again.
EOF
    exit 1
  else
    # Linux without brew: apt is the common case, and uv has its own installer
    if $need_ffmpeg; then
      if command -v apt-get >/dev/null 2>&1; then
        say "ffmpeg: installing via apt (may ask for your password)..."
        sudo apt-get update -qq && sudo apt-get install -y ffmpeg
      else
        say "ffmpeg is missing and no supported package manager was found."
        say "Install ffmpeg with your distribution's package manager, then re-run."
        exit 1
      fi
    fi
    if $need_uv; then
      say "uv: installing via the official installer..."
      curl -LsSf https://astral.sh/uv/install.sh | sh
      export PATH="$HOME/.local/bin:$PATH"
    fi
  fi
else
  say "ffmpeg: already installed"
  say "uv: already installed"
fi

command -v uv >/dev/null 2>&1 || { say "uv is still not on PATH. Open a new terminal and re-run."; exit 1; }

# --- python environment ---------------------------------------------------
step "Python environment"
if [ -d "$VENV" ]; then
  say "venv: found at $VENV"
else
  say "venv: creating with Python 3.11 (uv downloads it if needed)..."
  uv venv --python 3.11 "$VENV"
fi

# Trust the stamp only if the venv is actually populated. A stale stamp next to
# an empty/recreated venv (deleted by hand, restored from backup, etc.) must not
# skip installation, that was a real bug: it reported success into a venv with
# nothing in it.
packages_present() {
  "$VENV/bin/python" -c "import whisperx" >/dev/null 2>&1
}

if [ -f "$STAMP" ] && [ "$(cat "$STAMP" 2>/dev/null)" = "$STAMP_VERSION" ] && packages_present; then
  say "packages: already installed and up to date (stamp $STAMP_VERSION)"
else
  say "packages: installing (this downloads a few GB the first time, please be patient)..."
  uv pip install --python "$VENV/bin/python" -r "$HERE/requirements.txt"
  if $APPLE_SILICON; then
    say "packages: adding mlx-whisper for the fast Apple GPU path..."
    uv pip install --python "$VENV/bin/python" mlx-whisper
  fi
  printf '%s' "$STAMP_VERSION" > "$STAMP"
fi

# --- hugging face token ---------------------------------------------------
step "Hugging Face token"
TOKEN_OK=false
if [ -f "$ROOT/.env" ] && grep -q '^HF_TOKEN=hf_' "$ROOT/.env" 2>/dev/null; then
  TOKEN_OK=true
fi
[ -n "${HF_TOKEN:-}" ] && TOKEN_OK=true

if $TOKEN_OK; then
  say "token: found"
else
  cat <<EOF
  Not found. Speaker diarization uses a gated model, so a free token is needed.
  This is a one-off, and only you can do it:

    1. Create a READ token at https://huggingface.co/settings/tokens
    2. While logged in, accept the terms at
       https://huggingface.co/pyannote/speaker-diarization-community-1
    3. Write it into $ROOT/.env :

       echo 'HF_TOKEN=hf_your_token_here' > "$ROOT/.env"
       chmod 600 "$ROOT/.env"

  The token only unlocks the one-time model download. Nothing about your
  recordings is ever sent to Hugging Face, or anywhere else.
EOF
fi

step "Done"
if $TOKEN_OK; then
  say "Setup complete. The skill is ready to use."
else
  say "Setup complete except for the Hugging Face token, see above."
fi
