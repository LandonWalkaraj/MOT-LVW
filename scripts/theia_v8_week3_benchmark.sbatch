#!/usr/bin/env bash
#SBATCH --job-name=lorat-v8-week3
#SBATCH --account=rc_general
#SBATCH --partition=gpu-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --signal=TERM@600
#SBATCH --gres=gpu:1
#SBATCH --output=/work/%u/lorat-v8-week3-%j.out

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Refusing to run outside a Slurm allocation. Submit this script with sbatch."
    exit 2
fi

BUNDLE="${BUNDLE:-/work/$USER/theia_lorat_v8_train_bundle.tar.gz}"
UPDATE_ROOT="${UPDATE_ROOT:-/work/$USER/lorat-v8-live-updates}"
RUN_ROOT="${RUN_ROOT:-/work/$USER/lorat-v8-week3-job-$SLURM_JOB_ID}"
RESULT_ROOT="${RESULT_ROOT:-/work/$USER/lorat-v8-week3-results-full}"
VENV_DIR="${VENV_DIR:-/work/$USER/lorat-v8-venv-cu128-$SLURM_JOB_ID}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-/work/$USER/.cache/pip}"
EXPECT_GPU_NAME="${EXPECT_GPU_NAME:-A100}"
HEAD_WEIGHTS_ROOT="${HEAD_WEIGHTS_ROOT:-/work/$USER/lorat-v8-train-results/checkpoints}"
HEAD_WEIGHTS_ROOT_AUTO="${HEAD_WEIGHTS_ROOT_AUTO:-1}"
HEAD_CHECKPOINT="${HEAD_CHECKPOINT:-best}"
DATASET_ROOT="${DATASET_ROOT:-}"
CLEANUP_RUN_ROOT="${CLEANUP_RUN_ROOT:-1}"

TRACK_COUNTS="${TRACK_COUNTS:-1,2,3,4,5}"
MAX_FRAMES="${MAX_FRAMES:-0}"
SEQUENCES="${SEQUENCES:-dancetrack0065}"
SPLIT="${SPLIT:-val}"
COMPARE_CONFIGS="${COMPARE_CONFIGS:-B-224}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-25}"
AREA_SAMPLE_INTERVAL="${AREA_SAMPLE_INTERVAL:-10}"
IDENTITY_SAMPLE_INTERVAL="${IDENTITY_SAMPLE_INTERVAL:-1}"
SAVE_VIDEO="${SAVE_VIDEO:-1}"

export PIP_CACHE_DIR

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Bundle: $BUNDLE"
echo "Update root: $UPDATE_ROOT"
echo "Run root: $RUN_ROOT"
echo "Result root: $RESULT_ROOT"
echo "Head weights root: $HEAD_WEIGHTS_ROOT"
echo "Head weights auto-detect: $HEAD_WEIGHTS_ROOT_AUTO"
echo "Head checkpoint preference: $HEAD_CHECKPOINT"
echo "Dataset root override: ${DATASET_ROOT:-auto}"
echo "Split: $SPLIT"
echo "Sequences: $SEQUENCES"
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
    if [[ "$CLEANUP_RUN_ROOT" == "1" && "$RUN_ROOT" == "/work/$USER"/lorat-v8-week3-job-* && -d "$RUN_ROOT" ]]; then
        rm -rf "$RUN_ROOT"
    fi
}
trap cleanup_run_root EXIT

tar -xzf "$BUNDLE" -C "$RUN_ROOT"
cd "$RUN_ROOT/mot-v8-train"

if [[ -d "$UPDATE_ROOT" ]]; then
    echo "Applying live updates from: $UPDATE_ROOT"
    cp -a "$UPDATE_ROOT"/. "$PWD"/
fi

if command -v module >/dev/null 2>&1; then
    module purge || true
    module load python3/anaconda/3.12
fi

python --version

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    python -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel
python -m pip install "setuptools<82"
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

if [[ -z "$DATASET_ROOT" ]]; then
    if [[ -d "$PWD/data/raw/Combined" ]]; then
        DATASET_ROOT="$PWD/data/raw/Combined"
    elif [[ -d "$PWD/data/raw/DanceTrack" ]]; then
        DATASET_ROOT="$PWD/data/raw/DanceTrack"
    else
        echo "Could not find bundled dataset root under $PWD/data/raw"
        find "$PWD/data/raw" -maxdepth 3 -type d 2>/dev/null | head -80 || true
        exit 1
    fi
fi
echo "Resolved dataset root: $DATASET_ROOT"

read -r -a COMPARE_CONFIG_ARRAY <<< "$COMPARE_CONFIGS"
head_root_has_configs() {
    local root="$1"
    local config key best_path latest_path
    for config in "${COMPARE_CONFIG_ARRAY[@]}"; do
        key="${config//-/_}"
        best_path="$root/$key/v8_head_${key}_best_by_val_iou.pt"
        latest_path="$root/$key/v8_head_${key}_latest.pt"
        if [[ ! -f "$best_path" && ! -f "$latest_path" ]]; then
            return 1
        fi
    done
    return 0
}

if ! head_root_has_configs "$HEAD_WEIGHTS_ROOT"; then
    if [[ "$HEAD_WEIGHTS_ROOT_AUTO" == "1" || "$HEAD_WEIGHTS_ROOT_AUTO" == "true" || "$HEAD_WEIGHTS_ROOT_AUTO" == "TRUE" ]]; then
        echo "Configured head root does not contain all requested heads; searching latest training outputs."
        while IFS= read -r candidate_root; do
            if head_root_has_configs "$candidate_root"; then
                HEAD_WEIGHTS_ROOT="$candidate_root"
                echo "Auto-detected head weights root: $HEAD_WEIGHTS_ROOT"
                break
            fi
        done < <(ls -td /work/"$USER"/lorat-v8-train-results*/checkpoints 2>/dev/null || true)
    fi
fi

for config in "${COMPARE_CONFIG_ARRAY[@]}"; do
    key="${config//-/_}"
    preferred_suffix="latest"
    fallback_suffix="best_by_val_iou"
    if [[ "$HEAD_CHECKPOINT" == "best" || "$HEAD_CHECKPOINT" == "best_by_val_iou" ]]; then
        preferred_suffix="best_by_val_iou"
        fallback_suffix="latest"
    fi
    head_path="$HEAD_WEIGHTS_ROOT/$key/v8_head_${key}_${preferred_suffix}.pt"
    fallback_path="$HEAD_WEIGHTS_ROOT/$key/v8_head_${key}_${fallback_suffix}.pt"
    if [[ ! -f "$head_path" && -f "$fallback_path" ]]; then
        head_path="$fallback_path"
    fi
    if [[ ! -f "$head_path" ]]; then
        echo "Missing trained V8 head for $config. Checked: $head_path and $fallback_path"
        exit 1
    fi
    echo "Using V8 head for $config: $head_path"
done

SEQUENCE_ARGS=()
read -r -a SEQUENCE_ARRAY <<< "$SEQUENCES"
for sequence in "${SEQUENCE_ARRAY[@]}"; do
    SEQUENCE_ARGS+=(--sequence "$sequence")
done

VIDEO_ARGS=()
if [[ "$SAVE_VIDEO" == "1" || "$SAVE_VIDEO" == "true" || "$SAVE_VIDEO" == "TRUE" ]]; then
    VIDEO_ARGS+=(--save-video)
fi

python programs/benchmark_lorat_week3.py \
    --dataset-root "$DATASET_ROOT" \
    --split "$SPLIT" \
    --lorat-root "$PWD/external/LoRAT-main" \
    --device cuda:0 \
    --gpu-profile hpc \
    "${SEQUENCE_ARGS[@]}" \
    --track-counts "$TRACK_COUNTS" \
    --compare-configs "${COMPARE_CONFIG_ARRAY[@]}" \
    --v8-head-weights-root "$HEAD_WEIGHTS_ROOT" \
    --v8-head-checkpoint "$HEAD_CHECKPOINT" \
    --max-frames "$MAX_FRAMES" \
    --progress-interval "$PROGRESS_INTERVAL" \
    --area-sample-interval "$AREA_SAMPLE_INTERVAL" \
    --identity-sample-interval "$IDENTITY_SAMPLE_INTERVAL" \
    --output-root "$RESULT_ROOT" \
    "${VIDEO_ARGS[@]}"

echo "Week 3 benchmark complete. Summaries:"
find "$RESULT_ROOT" -maxdepth 3 -type f -name "summary.md" -print
