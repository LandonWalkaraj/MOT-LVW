#!/usr/bin/env bash
#SBATCH --job-name=lorat-v7-a100
#SBATCH --account=rc_general
#SBATCH --partition=gpu-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --signal=TERM@600
#SBATCH --gres=gpu:1
#SBATCH --output=/work/%u/lorat-v7-a100-%j.out

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Refusing to run outside a Slurm allocation. Submit this script with sbatch."
    exit 2
fi

BUNDLE="${BUNDLE:-/work/$USER/theia_lorat_v7_bundle.tar.gz}"
RUN_ROOT="${RUN_ROOT:-/work/$USER/lorat-v7-job-$SLURM_JOB_ID}"
RESULT_ROOT="${RESULT_ROOT:-/work/$USER/lorat-v7-results-full}"
VENV_DIR="${VENV_DIR:-/work/$USER/lorat-v7-venv-cu128}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-/work/$USER/.cache/pip}"
EXPECT_GPU_NAME="${EXPECT_GPU_NAME:-A100}"
CLEANUP_RUN_ROOT="${CLEANUP_RUN_ROOT:-1}"

TRACK_COUNTS="${TRACK_COUNTS:-1,2,3,4,5}"
MAX_FRAMES="${MAX_FRAMES:-0}"
SEQUENCE="${SEQUENCE:-dancetrack0065}"
COMPARE_CONFIGS="${COMPARE_CONFIGS:-B-224 L-224 g-224}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-25}"
AREA_SAMPLE_INTERVAL="${AREA_SAMPLE_INTERVAL:-10}"
IDENTITY_SAMPLE_INTERVAL="${IDENTITY_SAMPLE_INTERVAL:-10}"
SAVE_VIDEO="${SAVE_VIDEO:-1}"

export PIP_CACHE_DIR

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Bundle: $BUNDLE"
echo "Run root: $RUN_ROOT"
echo "Result root: $RESULT_ROOT"
echo "Track counts: $TRACK_COUNTS"
echo "Compare configs: $COMPARE_CONFIGS"
echo "Max frames: $MAX_FRAMES"
echo "Save video: $SAVE_VIDEO"

if [[ ! -f "$BUNDLE" ]]; then
    echo "Missing bundle: $BUNDLE"
    exit 1
fi

mkdir -p "$RUN_ROOT" "$RESULT_ROOT" "$PIP_CACHE_DIR"
cleanup_run_root() {
    if [[ "$CLEANUP_RUN_ROOT" == "1" && "$RUN_ROOT" == "/work/$USER"/lorat-v7-job-* && -d "$RUN_ROOT" ]]; then
        rm -rf "$RUN_ROOT"
    fi
}
trap cleanup_run_root EXIT

tar -xzf "$BUNDLE" -C "$RUN_ROOT"
cd "$RUN_ROOT/mot-v7"

if command -v module >/dev/null 2>&1; then
    module purge || true
    module load python3/anaconda/3.12
fi

python --version

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    python -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision --index-url "$PYTORCH_INDEX_URL"
python -m pip install -r external/LoRAT-main/requirements.txt
python -m pip install -r requirements.txt

python - <<PY
import torch

print(f"torch version: {torch.__version__}")
print(f"torch cuda version: {torch.version.cuda}")
print(f"torch cuda available: {torch.cuda.is_available()}")
print(f"torch cuda device count: {torch.cuda.device_count()}")

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in this Slurm job; aborting benchmark.")

gpu_name = torch.cuda.get_device_name(0)
print(f"torch cuda device 0: {gpu_name}")
expected = "${EXPECT_GPU_NAME}"
if expected and expected.lower() not in gpu_name.lower():
    raise SystemExit(f"Expected GPU name containing {expected!r}, got {gpu_name!r}; aborting benchmark.")
PY

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
fi

read -r -a COMPARE_CONFIG_ARRAY <<< "$COMPARE_CONFIGS"
VIDEO_ARGS=()
if [[ "$SAVE_VIDEO" == "1" || "$SAVE_VIDEO" == "true" || "$SAVE_VIDEO" == "TRUE" ]]; then
    VIDEO_ARGS+=(--save-video)
fi

echo "Selected sequence: $SEQUENCE"

python programs/benchmark_lorat_v7.py \
    --dataset-root "$PWD/data/raw/DanceTrack" \
    --lorat-root "$PWD/external/LoRAT-main" \
    --device cuda:0 \
    --gpu-profile hpc \
    --sequence "$SEQUENCE" \
    --track-counts "$TRACK_COUNTS" \
    --compare-configs "${COMPARE_CONFIG_ARRAY[@]}" \
    --max-frames "$MAX_FRAMES" \
    --progress-interval "$PROGRESS_INTERVAL" \
    --area-sample-interval "$AREA_SAMPLE_INTERVAL" \
    --identity-sample-interval "$IDENTITY_SAMPLE_INTERVAL" \
    --output-root "$RESULT_ROOT" \
    "${VIDEO_ARGS[@]}"

echo "Benchmark complete. Summaries:"
find "$RESULT_ROOT" -maxdepth 3 -type f -name "summary.md" -print
