# Idempotent bootstrap for the transcribe skill (Windows, PowerShell).
# Safe to re-run: installs only what is missing and re-stamps the version.
#
# Note on this platform: the --fast GPU path is Apple Silicon only (it uses MLX,
# which Apple ships for its own chips). On Windows the skill runs --safe, which
# is slower but uses your NVIDIA GPU automatically if you have one.

$ErrorActionPreference = 'Stop'

$Here  = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root  = Split-Path -Parent $Here
$Venv  = Join-Path $Root '.venv'
$Stamp = Join-Path $Root '.setup-stamp'
$StampVersion = '2'        # bump to force everyone to re-run the Python install

function Say  { param($m) Write-Host "  $m" }
function Step { param($m) Write-Host "`n==> $m" -ForegroundColor Cyan }

Step 'Checking environment'
Say "OS: Windows ($env:PROCESSOR_ARCHITECTURE)"
Say 'The fast Apple GPU path (--fast) is not available on Windows.'
Say 'The skill will use --safe, which works here and uses CUDA if you have an NVIDIA GPU.'

function Test-Command { param($n) $null -ne (Get-Command $n -ErrorAction SilentlyContinue) }

function Install-WithWinget {
    param($Id, $Label)
    if (-not (Test-Command 'winget')) {
        Say "$Label is missing, and winget is not available to install it."
        Say 'Install winget (App Installer) from the Microsoft Store, or install'
        Say "$Label manually, then re-run this script."
        exit 1
    }
    Say "${Label}: installing via winget..."
    winget install --id $Id --silent --accept-package-agreements --accept-source-agreements
}

# --- system packages ------------------------------------------------------
Step 'System dependencies (ffmpeg, uv)'

if (Test-Command 'ffmpeg') { Say 'ffmpeg: already installed' }
else { Install-WithWinget -Id 'Gyan.FFmpeg' -Label 'ffmpeg' }

if (Test-Command 'uv') { Say 'uv: already installed' }
else { Install-WithWinget -Id 'astral-sh.uv' -Label 'uv' }

# winget updates PATH for new shells, not the current one
$env:Path = [System.Environment]::GetEnvironmentVariable('Path','Machine') + ';' +
            [System.Environment]::GetEnvironmentVariable('Path','User')

if (-not (Test-Command 'uv')) {
    Say 'uv is installed but not yet on PATH in this shell.'
    Say 'Close this window, open a new PowerShell, and run the setup again.'
    exit 1
}
if (-not (Test-Command 'ffmpeg')) {
    Say 'ffmpeg is installed but not yet on PATH in this shell.'
    Say 'Close this window, open a new PowerShell, and run the setup again.'
    exit 1
}

# --- python environment ---------------------------------------------------
Step 'Python environment'
if (Test-Path $Venv) { Say "venv: found at $Venv" }
else {
    Say 'venv: creating with Python 3.11 (uv downloads it if needed)...'
    uv venv --python 3.11 $Venv
}

$VenvPython = Join-Path $Venv 'Scripts\python.exe'

# Trust the stamp only if the venv is actually populated. A stale stamp next to
# an empty/recreated venv (deleted by hand, restored from backup, etc.) must not
# skip installation, that was a real bug: it reported success into a venv with
# nothing in it.
function Test-PackagesPresent {
    & $VenvPython -c "import whisperx" 2>$null
    return $LASTEXITCODE -eq 0
}

if ((Test-Path $Stamp) -and ((Get-Content $Stamp -Raw).Trim() -eq $StampVersion) -and (Test-PackagesPresent)) {
    Say "packages: already installed and up to date (stamp $StampVersion)"
} else {
    Say 'packages: installing (this downloads a few GB the first time, please be patient)...'
    uv pip install --python $VenvPython -r (Join-Path $Here 'requirements.txt')
    Say 'note: mlx-whisper is skipped, it is Apple Silicon only'
    Set-Content -Path $Stamp -Value $StampVersion -NoNewline
}

# --- GPU (CUDA torch) ------------------------------------------------------
# The default PyPI torch wheel on Windows is CPU-only; CUDA builds only exist
# on the PyTorch index. ctranslate2 (the --safe ASR backend) already ships its
# own bundled CUDA runtime and works regardless, but VAD/alignment/diarization
# go through torch directly, so torch itself has to be the CUDA build for the
# GPU to actually get used.
Step 'GPU'

function Test-NvidiaGpu {
    if (-not (Test-Command 'nvidia-smi')) { return $false }
    nvidia-smi -L *> $null
    return $LASTEXITCODE -eq 0
}

function Test-CudaTorch {
    & $VenvPython -c "import torch, sys; sys.exit(0 if torch.version.cuda else 1)" 2>$null
    return $LASTEXITCODE -eq 0
}

if (-not (Test-NvidiaGpu)) {
    Say 'No NVIDIA GPU detected: transcription will run on CPU (roughly real-time).'
} elseif (Test-CudaTorch) {
    Say 'NVIDIA GPU detected. torch is already a CUDA build, nothing to do.'
} else {
    Say 'NVIDIA GPU detected, but torch is CPU-only. Installing the CUDA build'
    Say '(cu128, downloads a few GB the first time)...'
    # cu128 wheels cover sm_70 through sm_120 (Blackwell / RTX 50-series included),
    # so there is no need to branch on compute capability. Do not use cu124: it
    # installs fine on Blackwell cards but fails at runtime with "no kernel image
    # is available for execution".
    #
    # --reinstall matters: torch==2.8.0 as a bare version spec is satisfied by an
    # already-installed 2.8.0+cpu (PEP 440 local-version matching), so without it
    # this silently no-ops and leaves the CPU build in place.
    uv pip install --python $VenvPython --index-url https://download.pytorch.org/whl/cu128 `
        --reinstall torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
    $installExit = $LASTEXITCODE

    if ($installExit -eq 0 -and (Test-CudaTorch)) {
        Say 'torch: CUDA build installed and active.'
    } else {
        Say 'torch: still CPU-only after the install attempt.'
        Say 'The skill still works, just slower (roughly real time on CPU instead of'
        Say 'GPU-accelerated). Common causes: no network access to download.pytorch.org,'
        Say 'or an NVIDIA driver too old for CUDA 12.8 (update the driver and re-run'
        Say 'this script to retry).'
    }
}

# --- hugging face token ---------------------------------------------------
Step 'Hugging Face token'
$EnvFile = Join-Path $Root '.env'
$TokenOk = $false
if (Test-Path $EnvFile) {
    if ((Get-Content $EnvFile -Raw) -match 'HF_TOKEN=hf_') { $TokenOk = $true }
}
if ($env:HF_TOKEN) { $TokenOk = $true }

if ($TokenOk) { Say 'token: found' }
else {
    Write-Host @"
  Not found. Speaker diarization uses a gated model, so a free token is needed.
  This is a one-off, and only you can do it:

    1. Create a READ token at https://huggingface.co/settings/tokens
    2. While logged in, accept the terms at
       https://huggingface.co/pyannote/speaker-diarization-community-1
    3. Write it into $EnvFile :

       Set-Content -Path "$EnvFile" -Value "HF_TOKEN=hf_your_token_here"

  The token only unlocks the one-time model download. Nothing about your
  recordings is ever sent to Hugging Face, or anywhere else.
"@
}

Step 'Done'
if ($TokenOk) { Say 'Setup complete. The skill is ready to use.' }
else { Say 'Setup complete except for the Hugging Face token, see above.' }
