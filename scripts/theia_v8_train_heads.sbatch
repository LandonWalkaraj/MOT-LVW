#!/usr/bin/env bash
#SBATCH --job-name=lorat-v8-train
#SBATCH --account=rc_general
#SBATCH --partition=gpu-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --signal=TERM@600
#SBATCH --gres=gpu:1
#SBATCH --output=/work/%u/lorat-v8-train-%j.out

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Refusing to run outside a Slurm allocation. Submit this script with sbatch."
    exit 2
fi

BUNDLE="${BUNDLE:-/work/$USER/theia_lorat_v8_train_bundle.tar.gz}"
UPDATE_ROOT="${UPDATE_ROOT:-/work/$USER/lorat-v8-live-updates}"
RUN_ROOT="${RUN_ROOT:-/work/$USER/lorat-v8-train-job-$SLURM_JOB_ID}"
RESULT_ROOT="${RESULT_ROOT:-/work/$USER/lorat-v8-train-results}"
VENV_DIR="${VENV_DIR:-/work/$USER/lorat-v8-venv-cu128}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-/work/$USER/.cache/pip}"
EXPECT_GPU_NAME="${EXPECT_GPU_NAME:-A100}"
CLEANUP_RUN_ROOT="${CLEANUP_RUN_ROOT:-1}"

TRAIN_CONFIGS="${TRAIN_CONFIGS:-B-224 L-224 g-224}"
DATASET_ROOT="${DATASET_ROOT:-}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
VAL_SPLIT="${VAL_SPLIT:-val}"
MAX_OBJECTS="${MAX_OBJECTS:-8}"
TARGET_REGION_MODE="${TARGET_REGION_MODE:-mixed}"
TARGET_REGIONS_PER_OBJECT="${TARGET_REGIONS_PER_OBJECT:-2}"
TEMPLATE_AREA_FACTOR="${TEMPLATE_AREA_FACTOR:-2.0}"
SEARCH_AREA_FACTOR="${SEARCH_AREA_FACTOR:-4.0}"
SEARCH_SCALE_JITTER="${SEARCH_SCALE_JITTER:-0.25}"
SEARCH_TRANSLATION_JITTER="${SEARCH_TRANSLATION_JITTER:-3.0}"
SEARCH_MIN_OBJECT_SIZE="${SEARCH_MIN_OBJECT_SIZE:-10.0}"
TEMPLATE_SAMPLING="${TEMPLATE_SAMPLING:-mixed}"
PREVIOUS_BOX_JITTER="${PREVIOUS_BOX_JITTER:-0.10}"
SEQUENCE_WINDOW_LENGTH="${SEQUENCE_WINDOW_LENGTH:-5}"
LOST_TARGET_PROBABILITY="${LOST_TARGET_PROBABILITY:-0.10}"
MAX_LOST_TARGETS_PER_FRAME="${MAX_LOST_TARGETS_PER_FRAME:-2}"
MAX_MISSING_GAP_FRAMES="${MAX_MISSING_GAP_FRAMES:-30}"
FRAME_STRIDE="${FRAME_STRIDE:-5}"
VAL_FRAME_STRIDE="${VAL_FRAME_STRIDE:-15}"
MAX_SEQUENCES="${MAX_SEQUENCES:-0}"
MAX_VAL_SEQUENCES="${MAX_VAL_SEQUENCES:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-512}"
EPOCHS="${EPOCHS:-12}"
MAX_STEPS="${MAX_STEPS:-0}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
BOX_LOSS_WEIGHT="${BOX_LOSS_WEIGHT:-1.0}"
LTRB_LOSS_WEIGHT="${LTRB_LOSS_WEIGHT:-1.0}"
REID_LOSS_WEIGHT="${REID_LOSS_WEIGHT:-0.25}"
CENTER_POSITIVE_WEIGHT="${CENTER_POSITIVE_WEIGHT:-0.5}"
NEGATIVE_LOSS_WEIGHT="${NEGATIVE_LOSS_WEIGHT:-1.0}"
HARD_NEGATIVE_LOSS_WEIGHT="${HARD_NEGATIVE_LOSS_WEIGHT:-2.0}"
FOCAL_LOSS_GAMMA="${FOCAL_LOSS_GAMMA:-2.0}"
IOU_AWARE_WARMUP_STEPS="${IOU_AWARE_WARMUP_STEPS:-250}"
GAUSSIAN_SIGMA="${GAUSSIAN_SIGMA:-1.25}"
CLIP_MAX_NORM="${CLIP_MAX_NORM:-1.0}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-250}"
TRAIN_DIAGNOSTIC_SAMPLES="${TRAIN_DIAGNOSTIC_SAMPLES:-64}"
OVERFIT_SMOKE="${OVERFIT_SMOKE:-0}"
OVERFIT_SMOKE_SAMPLES="${OVERFIT_SMOKE_SAMPLES:-16}"
QUALITY_GATE_MEAN_IOU="${QUALITY_GATE_MEAN_IOU:-0.0}"
QUALITY_GATE_IOU50="${QUALITY_GATE_IOU50:-0.0}"
HEAD_HIDDEN_DIM="${HEAD_HIDDEN_DIM:-256}"
HEAD_LORA_RANK="${HEAD_LORA_RANK:-16}"
HEAD_RANK="${HEAD_RANK:-5}"
SEARCH_RADIUS_FACTOR="${SEARCH_RADIUS_FACTOR:-2.25}"

export PIP_CACHE_DIR

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Bundle: $BUNDLE"
echo "Update root: $UPDATE_ROOT"
echo "Run root: $RUN_ROOT"
echo "Result root: $RESULT_ROOT"
echo "Train configs: $TRAIN_CONFIGS"
echo "Train split: $TRAIN_SPLIT"
echo "Val split: $VAL_SPLIT"
echo "Max sequences: $MAX_SEQUENCES"
echo "Max val sequences: $MAX_VAL_SEQUENCES"
echo "Target region mode: $TARGET_REGION_MODE"
echo "Target regions per object: $TARGET_REGIONS_PER_OBJECT"
echo "Template area factor: $TEMPLATE_AREA_FACTOR"
echo "Search area factor: $SEARCH_AREA_FACTOR"
echo "Search scale jitter: $SEARCH_SCALE_JITTER"
echo "Search translation jitter: $SEARCH_TRANSLATION_JITTER"
echo "Search min object size: $SEARCH_MIN_OBJECT_SIZE"
echo "Template sampling: $TEMPLATE_SAMPLING"
echo "Previous box jitter: $PREVIOUS_BOX_JITTER"
echo "Sequence window length: $SEQUENCE_WINDOW_LENGTH"
echo "Lost target probability: $LOST_TARGET_PROBABILITY"
echo "Max lost targets per frame: $MAX_LOST_TARGETS_PER_FRAME"
echo "Max missing gap frames: $MAX_MISSING_GAP_FRAMES"
echo "Epochs: $EPOCHS"
echo "Max steps: $MAX_STEPS"
echo "Loss weights: box=$BOX_LOSS_WEIGHT ltrb=$LTRB_LOSS_WEIGHT reid=$REID_LOSS_WEIGHT center_positive=$CENTER_POSITIVE_WEIGHT negative=$NEGATIVE_LOSS_WEIGHT hard_negative=$HARD_NEGATIVE_LOSS_WEIGHT focal_gamma=$FOCAL_LOSS_GAMMA"
echo "Overfit smoke: $OVERFIT_SMOKE samples=$OVERFIT_SMOKE_SAMPLES quality_gate_mean_iou=$QUALITY_GATE_MEAN_IOU quality_gate_iou50=$QUALITY_GATE_IOU50"

if [[ ! -f "$BUNDLE" ]]; then
    echo "Missing bundle: $BUNDLE"
    exit 1
fi

mkdir -p "$RUN_ROOT" "$RESULT_ROOT" "$PIP_CACHE_DIR"
cleanup_run_root() {
    if [[ "$CLEANUP_RUN_ROOT" == "1" && "$RUN_ROOT" == "/work/$USER"/lorat-v8-train-job-* && -d "$RUN_ROOT" ]]; then
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

if [[ -z "$DATASET_ROOT" ]]; then
    if [[ -d "$PWD/data/raw/Combined" ]]; then
        DATASET_ROOT="$PWD/data/raw/Combined"
    else
        DATASET_ROOT="$PWD/data/raw/DanceTrack"
    fi
fi
echo "Dataset root: $DATASET_ROOT"

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
    raise SystemExit("CUDA is not available in this Slurm job; aborting training.")

gpu_name = torch.cuda.get_device_name(0)
print(f"torch cuda device 0: {gpu_name}")
expected = "${EXPECT_GPU_NAME}"
if expected and expected.lower() not in gpu_name.lower():
    raise SystemExit(f"Expected GPU name containing {expected!r}, got {gpu_name!r}; aborting training.")
PY

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
fi

mkdir -p "$RESULT_ROOT/models/lorat" "$RESULT_ROOT/checkpoints" "$RESULT_ROOT/logs"
read -r -a TRAIN_CONFIG_ARRAY <<< "$TRAIN_CONFIGS"

for CONFIG in "${TRAIN_CONFIG_ARRAY[@]}"; do
    CONFIG_SAFE="${CONFIG//-/_}"
    OUTPUT="$RESULT_ROOT/models/lorat/v8_head_${CONFIG_SAFE}.pt"
    CHECKPOINT_DIR="$RESULT_ROOT/checkpoints/$CONFIG_SAFE"
    LOG_PATH="$RESULT_ROOT/logs/train_${CONFIG_SAFE}.log"

    echo "============================================================"
    echo "Training V8 head for $CONFIG"
    echo "Output: $OUTPUT"
    echo "Checkpoint dir: $CHECKPOINT_DIR"
    echo "Log: $LOG_PATH"
    echo "============================================================"

    EXTRA_ARGS=()
    if [[ "$OVERFIT_SMOKE" == "1" ]]; then
        EXTRA_ARGS+=(--overfit-smoke)
    fi

    python programs/train_lorat_v8_head.py \
        --dataset-root "$DATASET_ROOT" \
        --split "$TRAIN_SPLIT" \
        --val-split "$VAL_SPLIT" \
        --lorat-root "$PWD/external/LoRAT-main" \
        --lorat-config "$CONFIG" \
        --device cuda:0 \
        --max-objects "$MAX_OBJECTS" \
        --target-region-mode "$TARGET_REGION_MODE" \
        --target-regions-per-object "$TARGET_REGIONS_PER_OBJECT" \
        --template-area-factor "$TEMPLATE_AREA_FACTOR" \
        --search-area-factor "$SEARCH_AREA_FACTOR" \
        --search-scale-jitter "$SEARCH_SCALE_JITTER" \
        --search-translation-jitter "$SEARCH_TRANSLATION_JITTER" \
        --search-min-object-size "$SEARCH_MIN_OBJECT_SIZE" \
        --template-sampling "$TEMPLATE_SAMPLING" \
        --previous-box-jitter "$PREVIOUS_BOX_JITTER" \
        --sequence-window-length "$SEQUENCE_WINDOW_LENGTH" \
        --lost-target-probability "$LOST_TARGET_PROBABILITY" \
        --max-lost-targets-per-frame "$MAX_LOST_TARGETS_PER_FRAME" \
        --max-missing-gap-frames "$MAX_MISSING_GAP_FRAMES" \
        --frame-stride "$FRAME_STRIDE" \
        --val-frame-stride "$VAL_FRAME_STRIDE" \
        --max-sequences "$MAX_SEQUENCES" \
        --max-val-sequences "$MAX_VAL_SEQUENCES" \
        --max-samples "$MAX_SAMPLES" \
        --max-val-samples "$MAX_VAL_SAMPLES" \
        --epochs "$EPOCHS" \
        --max-steps "$MAX_STEPS" \
        --learning-rate "$LEARNING_RATE" \
        --weight-decay "$WEIGHT_DECAY" \
        --box-loss-weight "$BOX_LOSS_WEIGHT" \
        --ltrb-loss-weight "$LTRB_LOSS_WEIGHT" \
        --reid-loss-weight "$REID_LOSS_WEIGHT" \
        --center-positive-weight "$CENTER_POSITIVE_WEIGHT" \
        --negative-loss-weight "$NEGATIVE_LOSS_WEIGHT" \
        --hard-negative-loss-weight "$HARD_NEGATIVE_LOSS_WEIGHT" \
        --focal-loss-gamma "$FOCAL_LOSS_GAMMA" \
        --iou-aware-warmup-steps "$IOU_AWARE_WARMUP_STEPS" \
        --gaussian-sigma "$GAUSSIAN_SIGMA" \
        --clip-max-norm "$CLIP_MAX_NORM" \
        --v8-head-hidden-dim "$HEAD_HIDDEN_DIM" \
        --v8-head-lora-rank "$HEAD_LORA_RANK" \
        --v8-head-rank "$HEAD_RANK" \
        --v8-search-radius-factor "$SEARCH_RADIUS_FACTOR" \
        --checkpoint-interval "$CHECKPOINT_INTERVAL" \
        --checkpoint-dir "$CHECKPOINT_DIR" \
        --train-diagnostic-samples "$TRAIN_DIAGNOSTIC_SAMPLES" \
        --overfit-smoke-samples "$OVERFIT_SMOKE_SAMPLES" \
        --quality-gate-mean-iou "$QUALITY_GATE_MEAN_IOU" \
        --quality-gate-iou50 "$QUALITY_GATE_IOU50" \
        --diagnostic-csv "$RESULT_ROOT/logs/train_${CONFIG_SAFE}_diagnostics.csv" \
        --output "$OUTPUT" \
        "${EXTRA_ARGS[@]}" 2>&1 | tee "$LOG_PATH"

    echo "Finished V8 head for $CONFIG"
done

echo "Training complete. Result files:"
find "$RESULT_ROOT" -maxdepth 4 -type f -print
