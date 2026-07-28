#!/usr/bin/env bash
#SBATCH --job-name=lorat-v9-b224
#SBATCH --account=rc_general
#SBATCH --partition=gpu-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --signal=TERM@600
#SBATCH --gres=gpu:1
#SBATCH --output=/work/%u/lorat-v9-train-b224-24h-%j.out

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Refusing to run outside a Slurm allocation. Submit this script with sbatch."
    exit 2
fi

BUNDLE="${BUNDLE:-/work/$USER/theia_lorat_v8_train_bundle.tar.gz}"
UPDATE_ROOT="${UPDATE_ROOT:-/work/$USER/lorat-v9-live-updates}"
RUN_ROOT="${RUN_ROOT:-/work/$USER/lorat-v9-train-job-$SLURM_JOB_ID}"
RESULT_ROOT="${RESULT_ROOT:-/work/$USER/lorat-v9-train-results-b224-tao-24h-$SLURM_JOB_ID}"
VENV_DIR="${VENV_DIR:-/work/$USER/lorat-v9-venv-cu128}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-/work/$USER/.cache/pip}"
EXPECT_GPU_NAME="${EXPECT_GPU_NAME:-A100}"
CLEANUP_RUN_ROOT="${CLEANUP_RUN_ROOT:-1}"

DATASET_ROOT="${DATASET_ROOT:-/work/$USER/lorat-data/raw/DanceTrack}"
MOT17_ROOT="${MOT17_ROOT:-/work/$USER/lorat-data/raw/MOT17}"
TAO_ROOT="${TAO_ROOT:-/work/$USER/lorat-data/raw/TAO_OW_SUBSET}"
LASOT_ROOT="${LASOT_ROOT:-/work/$USER/lorat-data/raw/LaSOT_subset}"
LASOT_ARCHIVE="${LASOT_ARCHIVE:-/work/$USER/LaSOT_subset.tar.gz}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
VAL_SPLIT="${VAL_SPLIT:-val}"
MOT17_SPLIT="${MOT17_SPLIT:-train}"
MOT17_VAL_SPLIT="${MOT17_VAL_SPLIT:-train}"
TAO_SPLIT="${TAO_SPLIT:-train}"
TAO_VAL_SPLIT="${TAO_VAL_SPLIT:-validation}"
TAO_USE_FREEFORM="${TAO_USE_FREEFORM:-1}"
LASOT_SPLIT="${LASOT_SPLIT:-train}"
LASOT_VAL_SPLIT="${LASOT_VAL_SPLIT:-val}"
LASOT_VAL_FRACTION="${LASOT_VAL_FRACTION:-0.20}"
TRAIN_CONFIG="${TRAIN_CONFIG:-B-224}"
CONFIG_SAFE="${TRAIN_CONFIG//-/_}"

MAX_OBJECTS="${MAX_OBJECTS:-8}"
TARGET_REGION_MODE="${TARGET_REGION_MODE:-mixed}"
TARGET_REGIONS_PER_OBJECT="${TARGET_REGIONS_PER_OBJECT:-6}"
TEMPLATE_AREA_FACTOR="${TEMPLATE_AREA_FACTOR:-2.0}"
SEARCH_AREA_FACTOR="${SEARCH_AREA_FACTOR:-4.0}"
SEARCH_SCALE_JITTER="${SEARCH_SCALE_JITTER:-0.25}"
SEARCH_TRANSLATION_JITTER="${SEARCH_TRANSLATION_JITTER:-3.0}"
SEARCH_MIN_OBJECT_SIZE="${SEARCH_MIN_OBJECT_SIZE:-10.0}"
SEARCH_ANCHOR_MODE="${SEARCH_ANCHOR_MODE:-union}"
REPAIR_SEARCH_TO_TARGET="${REPAIR_SEARCH_TO_TARGET:-1}"
SEARCH_TARGET_PADDING_FRACTION="${SEARCH_TARGET_PADDING_FRACTION:-0.05}"
TEMPLATE_SAMPLING="${TEMPLATE_SAMPLING:-mixed}"
PREVIOUS_BOX_JITTER="${PREVIOUS_BOX_JITTER:-0.10}"
SEQUENCE_WINDOW_LENGTH="${SEQUENCE_WINDOW_LENGTH:-5}"
LOST_TARGET_PROBABILITY="${LOST_TARGET_PROBABILITY:-0.10}"
MAX_LOST_TARGETS_PER_FRAME="${MAX_LOST_TARGETS_PER_FRAME:-2}"
MAX_MISSING_GAP_FRAMES="${MAX_MISSING_GAP_FRAMES:-30}"
DRIFT_NEGATIVE_PROBABILITY="${DRIFT_NEGATIVE_PROBABILITY:-0.35}"
DRIFT_NEGATIVE_START_EPOCH="${DRIFT_NEGATIVE_START_EPOCH:-3}"
DRIFT_NEGATIVE_MAX_PER_FRAME="${DRIFT_NEGATIVE_MAX_PER_FRAME:-2}"
DRIFT_NEGATIVE_SHIFT_FACTOR="${DRIFT_NEGATIVE_SHIFT_FACTOR:-3.0}"

FRAME_STRIDE="${FRAME_STRIDE:-5}"
VAL_FRAME_STRIDE="${VAL_FRAME_STRIDE:-15}"
MOT17_FRAME_STRIDE="${MOT17_FRAME_STRIDE:-5}"
MOT17_VAL_FRAME_STRIDE="${MOT17_VAL_FRAME_STRIDE:-15}"
TAO_FRAME_STRIDE="${TAO_FRAME_STRIDE:-2}"
TAO_VAL_FRAME_STRIDE="${TAO_VAL_FRAME_STRIDE:-5}"
LASOT_FRAME_STRIDE="${LASOT_FRAME_STRIDE:-5}"
LASOT_VAL_FRAME_STRIDE="${LASOT_VAL_FRAME_STRIDE:-15}"
MAX_SEQUENCES="${MAX_SEQUENCES:-0}"
MAX_VAL_SEQUENCES="${MAX_VAL_SEQUENCES:-0}"
MOT17_MAX_SEQUENCES="${MOT17_MAX_SEQUENCES:-0}"
MOT17_MAX_VAL_SEQUENCES="${MOT17_MAX_VAL_SEQUENCES:-0}"
TAO_MAX_SEQUENCES="${TAO_MAX_SEQUENCES:-0}"
TAO_MAX_VAL_SEQUENCES="${TAO_MAX_VAL_SEQUENCES:-0}"
LASOT_MAX_SEQUENCES="${LASOT_MAX_SEQUENCES:-0}"
LASOT_MAX_VAL_SEQUENCES="${LASOT_MAX_VAL_SEQUENCES:-2}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-512}"
MOT17_MAX_SAMPLES="${MOT17_MAX_SAMPLES:-0}"
MOT17_MAX_VAL_SAMPLES="${MOT17_MAX_VAL_SAMPLES:-512}"
TAO_MAX_SAMPLES="${TAO_MAX_SAMPLES:-0}"
TAO_MAX_VAL_SAMPLES="${TAO_MAX_VAL_SAMPLES:-512}"
LASOT_MAX_SAMPLES="${LASOT_MAX_SAMPLES:-0}"
LASOT_MAX_VAL_SAMPLES="${LASOT_MAX_VAL_SAMPLES:-256}"

EPOCHS="${EPOCHS:-250}"
MAX_TRAIN_SAMPLES_PER_EPOCH="${MAX_TRAIN_SAMPLES_PER_EPOCH:-2048}"
SOURCE_SAMPLING_WEIGHTS="${SOURCE_SAMPLING_WEIGHTS:-MOTFrameHeadDataset=1.0,TAOFrameHeadDataset=4.0,LaSOTFrameHeadDataset=1.5}"
MAX_STEPS="${MAX_STEPS:-0}"
MAX_WALL_HOURS="${MAX_WALL_HOURS:-23.25}"
EVAL_INTERVAL_EPOCHS="${EVAL_INTERVAL_EPOCHS:-2}"
TRAIN_DIAGNOSTIC_SAMPLES="${TRAIN_DIAGNOSTIC_SAMPLES:-96}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
BOX_LOSS_WEIGHT="${BOX_LOSS_WEIGHT:-1.0}"
LTRB_LOSS_WEIGHT="${LTRB_LOSS_WEIGHT:-1.0}"
REID_LOSS_WEIGHT="${REID_LOSS_WEIGHT:-0.50}"
CENTER_POSITIVE_WEIGHT="${CENTER_POSITIVE_WEIGHT:-0.5}"
NEGATIVE_LOSS_WEIGHT="${NEGATIVE_LOSS_WEIGHT:-1.0}"
HARD_NEGATIVE_LOSS_WEIGHT="${HARD_NEGATIVE_LOSS_WEIGHT:-3.0}"
FOCAL_LOSS_GAMMA="${FOCAL_LOSS_GAMMA:-2.0}"
SMALL_TARGET_LOSS_WEIGHT="${SMALL_TARGET_LOSS_WEIGHT:-3.0}"
SMALL_TARGET_AREA_THRESHOLD="${SMALL_TARGET_AREA_THRESHOLD:-4096.0}"
SMALL_TARGET_MAX_SIDE="${SMALL_TARGET_MAX_SIDE:-96.0}"
DCFST_DISCRIMINATION_WEIGHT="${DCFST_DISCRIMINATION_WEIGHT:-0.70}"
ASSIGNMENT_DISCRIMINATION_WEIGHT="${ASSIGNMENT_DISCRIMINATION_WEIGHT:-0.50}"
ASSIGNMENT_MARGIN="${ASSIGNMENT_MARGIN:-0.25}"
CANDIDATE_RANKING_LOSS_WEIGHT="${CANDIDATE_RANKING_LOSS_WEIGHT:-0.60}"
CANDIDATE_RANKING_START_EPOCH="${CANDIDATE_RANKING_START_EPOCH:-3}"
CANDIDATE_RANKING_MARGIN="${CANDIDATE_RANKING_MARGIN:-0.25}"
CANDIDATE_RANKING_TOPK="${CANDIDATE_RANKING_TOPK:-16}"
CANDIDATE_RANKING_POSITIVE_IOU="${CANDIDATE_RANKING_POSITIVE_IOU:-0.35}"
CANDIDATE_RANKING_OTHER_IOU="${CANDIDATE_RANKING_OTHER_IOU:-0.30}"
ROLLOUT_IDENTITY_SWITCH_PENALTY="${ROLLOUT_IDENTITY_SWITCH_PENALTY:-0.75}"
ROLLOUT_WINDOW_MISS_PENALTY="${ROLLOUT_WINDOW_MISS_PENALTY:-0.25}"
ROLLOUT_VALIDATION_MODE="${ROLLOUT_VALIDATION_MODE:-sequence}"
GEOMETRY_ONLY_EPOCHS="${GEOMETRY_ONLY_EPOCHS:-2}"
HARD_NEGATIVE_START_EPOCH="${HARD_NEGATIVE_START_EPOCH:-3}"
REID_START_EPOCH="${REID_START_EPOCH:-5}"
DCFST_START_EPOCH="${DCFST_START_EPOCH:-3}"
ASSIGNMENT_START_EPOCH="${ASSIGNMENT_START_EPOCH:-3}"
CLOSED_LOOP_START_EPOCH="${CLOSED_LOOP_START_EPOCH:-5}"
CLOSED_LOOP_PROBABILITY="${CLOSED_LOOP_PROBABILITY:-0.35}"
CLIP_MAX_NORM="${CLIP_MAX_NORM:-1.0}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-1000}"
CHECKPOINT_RETENTION="${CHECKPOINT_RETENTION:-latest}"
HEAD_HIDDEN_DIM="${HEAD_HIDDEN_DIM:-256}"
HEAD_LORA_RANK="${HEAD_LORA_RANK:-16}"
HEAD_RANK="${HEAD_RANK:-5}"
SEARCH_RADIUS_FACTOR="${SEARCH_RADIUS_FACTOR:-2.25}"
WINDOW_PENALTY_RATIO="${WINDOW_PENALTY_RATIO:-0.25}"
LOCAL_GRID_SIZE="${LOCAL_GRID_SIZE:-17}"
TRAINING_MEMORY_SLOTS="${TRAINING_MEMORY_SLOTS:-3}"
AUTO_RESUME_HEAD="${AUTO_RESUME_HEAD:-1}"
RESUME_HEAD="${RESUME_HEAD:-}"

if [[ -z "$RESUME_HEAD" && "$AUTO_RESUME_HEAD" == "1" ]]; then
    while IFS= read -r candidate; do
        if [[ -n "$candidate" && -f "$candidate" && "$candidate" != "$RESULT_ROOT"* ]]; then
            RESUME_HEAD="$candidate"
            break
        fi
    done < <(
        {
            ls -t /work/"$USER"/lorat-v9-train-results*/checkpoints/"$CONFIG_SAFE"/v9_local_head_"$CONFIG_SAFE"_best_by_rollout_identity.pt 2>/dev/null || true
            ls -t /work/"$USER"/lorat-v9-train-results*/checkpoints/"$CONFIG_SAFE"/v9_local_head_"$CONFIG_SAFE"_best_by_val_iou.pt 2>/dev/null || true
            ls -t /work/"$USER"/lorat-v9-train-results*/checkpoints/"$CONFIG_SAFE"/v9_local_head_"$CONFIG_SAFE"_latest.pt 2>/dev/null || true
        }
    )
fi

export PIP_CACHE_DIR

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Bundle: $BUNDLE"
echo "Update root: $UPDATE_ROOT"
echo "Run root: $RUN_ROOT"
echo "Result root: $RESULT_ROOT"
echo "Train config: $TRAIN_CONFIG"
echo "Epochs: $EPOCHS"
echo "Max wall hours: $MAX_WALL_HOURS"
echo "Max train samples per epoch: $MAX_TRAIN_SAMPLES_PER_EPOCH"
echo "Source sampling weights: ${SOURCE_SAMPLING_WEIGHTS:-none}"
echo "Eval interval epochs: $EVAL_INTERVAL_EPOCHS"
echo "Checkpoint interval: $CHECKPOINT_INTERVAL"
echo "Checkpoint retention: $CHECKPOINT_RETENTION"
echo "Candidate ranking loss weight: $CANDIDATE_RANKING_LOSS_WEIGHT"
echo "Candidate ranking start epoch: $CANDIDATE_RANKING_START_EPOCH"
echo "Candidate ranking margin/topk: $CANDIDATE_RANKING_MARGIN / $CANDIDATE_RANKING_TOPK"
echo "Drift negatives probability/start/max/shift: $DRIFT_NEGATIVE_PROBABILITY / $DRIFT_NEGATIVE_START_EPOCH / $DRIFT_NEGATIVE_MAX_PER_FRAME / $DRIFT_NEGATIVE_SHIFT_FACTOR"
echo "Rollout identity/window penalties: $ROLLOUT_IDENTITY_SWITCH_PENALTY / $ROLLOUT_WINDOW_MISS_PENALTY"
echo "Rollout validation mode: $ROLLOUT_VALIDATION_MODE"
echo "Resume head: ${RESUME_HEAD:-none}"
echo "TAO root: ${TAO_ROOT:-none}"
echo "LaSOT root: ${LASOT_ROOT:-none}"
echo "LaSOT archive: ${LASOT_ARCHIVE:-none}"

if [[ ! -f "$BUNDLE" ]]; then
    echo "Missing bundle: $BUNDLE"
    exit 1
fi

mkdir -p "$RUN_ROOT" "$RESULT_ROOT" "$PIP_CACHE_DIR"
cleanup_run_root() {
    if [[ "$CLEANUP_RUN_ROOT" == "1" && "$RUN_ROOT" == "/work/$USER"/lorat-v9-train-job-* && -d "$RUN_ROOT" ]]; then
        rm -rf "$RUN_ROOT"
    fi
}
trap cleanup_run_root EXIT

tar -xzf "$BUNDLE" -C "$RUN_ROOT"
cd "$RUN_ROOT/mot-v8-train"

if [[ -d "$UPDATE_ROOT" ]]; then
    echo "Applying live V9 updates from: $UPDATE_ROOT"
    cp -a "$UPDATE_ROOT"/. "$PWD"/
fi

if [[ -z "$DATASET_ROOT" ]]; then
    if [[ -d "/work/$USER/lorat-data/raw/Combined" ]]; then
        DATASET_ROOT="/work/$USER/lorat-data/raw/Combined"
    elif [[ -d "/work/$USER/lorat-data/raw/DanceTrack" ]]; then
        DATASET_ROOT="/work/$USER/lorat-data/raw/DanceTrack"
    elif [[ -d "$PWD/data/raw/Combined" ]]; then
        DATASET_ROOT="$PWD/data/raw/Combined"
    else
        DATASET_ROOT="$PWD/data/raw/DanceTrack"
    fi
fi
if [[ -z "$MOT17_ROOT" ]]; then
    if [[ -d "/work/$USER/lorat-data/raw/MOT17" ]]; then
        MOT17_ROOT="/work/$USER/lorat-data/raw/MOT17"
    elif [[ -d "$PWD/data/raw/MOTChallenge/MOT17" ]]; then
        MOT17_ROOT="$PWD/data/raw/MOTChallenge/MOT17"
    elif [[ -d "$PWD/data/raw/MOT17" ]]; then
        MOT17_ROOT="$PWD/data/raw/MOT17"
    fi
fi
if [[ -z "${TAO_ROOT:-}" && -d "/work/$USER/lorat-data/raw/TAO_OW_SUBSET" ]]; then
    TAO_ROOT="/work/$USER/lorat-data/raw/TAO_OW_SUBSET"
elif [[ -z "${TAO_ROOT:-}" && -d "$PWD/data/raw/TAO_OW_SUBSET" ]]; then
    TAO_ROOT="$PWD/data/raw/TAO_OW_SUBSET"
fi
if [[ -n "${LASOT_ROOT:-}" && ! -d "$LASOT_ROOT" && -f "${LASOT_ARCHIVE:-}" ]]; then
    echo "Extracting LaSOT subset archive on compute node: $LASOT_ARCHIVE"
    mkdir -p "$(dirname "$LASOT_ROOT")"
    tar -xzf "$LASOT_ARCHIVE" -C "$(dirname "$LASOT_ROOT")"
fi
if [[ -z "${LASOT_ROOT:-}" && -d "/work/$USER/lorat-data/raw/LaSOT_subset" ]]; then
    LASOT_ROOT="/work/$USER/lorat-data/raw/LaSOT_subset"
elif [[ -z "${LASOT_ROOT:-}" && -d "$PWD/data/raw/LaSOT_subset" ]]; then
    LASOT_ROOT="$PWD/data/raw/LaSOT_subset"
fi

echo "Resolved primary dataset root: $DATASET_ROOT"
echo "Resolved MOT17 root: ${MOT17_ROOT:-none}"
echo "Resolved TAO root: ${TAO_ROOT:-none}"
echo "Resolved LaSOT root: ${LASOT_ROOT:-none}"

prepare_tao_split_zip() {
    local split_dir="$1"
    local zip_path="$split_dir/YFCC100M.zip"
    local extract_dir="$split_dir/YFCC100M"

    if [[ -d "$extract_dir" ]]; then
        local existing_count
        existing_count="$(find "$extract_dir" -type f -print -quit 2>/dev/null | wc -l)"
        if [[ "$existing_count" -gt 0 ]]; then
            echo "TAO frames already extracted: $extract_dir"
            return
        fi
    fi

    if [[ -f "$zip_path" ]]; then
        echo "Extracting TAO/YFCC frames from $zip_path to $extract_dir"
        mkdir -p "$extract_dir"
        if command -v unzip >/dev/null 2>&1; then
            unzip -q "$zip_path" -d "$extract_dir"
        else
            python3 - "$zip_path" "$extract_dir" <<'PY'
import sys
import zipfile
from pathlib import Path

zip_path = Path(sys.argv[1])
extract_dir = Path(sys.argv[2])
with zipfile.ZipFile(zip_path) as zf:
    zf.extractall(extract_dir)
PY
        fi
    fi
}

if [[ -n "${TAO_ROOT:-}" && -d "$TAO_ROOT" ]]; then
    prepare_tao_split_zip "$TAO_ROOT/frames/train"
    prepare_tao_split_zip "$TAO_ROOT/frames/val"
    prepare_tao_split_zip "$TAO_ROOT/frames/validation"
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
OUTPUT="$RESULT_ROOT/models/lorat/v9_local_head_${CONFIG_SAFE}.pt"
CHECKPOINT_DIR="$RESULT_ROOT/checkpoints/$CONFIG_SAFE"
LOG_PATH="$RESULT_ROOT/logs/train_${CONFIG_SAFE}.log"

EXTRA_ARGS=()
if [[ "$REPAIR_SEARCH_TO_TARGET" != "1" ]]; then
    EXTRA_ARGS+=(--disable-search-target-repair)
fi
if [[ "$TAO_USE_FREEFORM" == "1" ]]; then
    EXTRA_ARGS+=(--tao-use-freeform)
fi
if [[ -z "${MOT17_ROOT:-}" || ! -d "$MOT17_ROOT" ]]; then
    EXTRA_ARGS+=(--disable-mot17)
fi
if [[ -z "${TAO_ROOT:-}" || ! -d "$TAO_ROOT" ]]; then
    EXTRA_ARGS+=(--disable-tao)
fi
if [[ -z "${LASOT_ROOT:-}" || ! -d "$LASOT_ROOT" ]]; then
    EXTRA_ARGS+=(--disable-lasot)
fi
if [[ -n "${RESUME_HEAD:-}" && -f "$RESUME_HEAD" ]]; then
    EXTRA_ARGS+=(--resume-head "$RESUME_HEAD")
else
    EXTRA_ARGS+=(--auto-resume-latest)
fi

echo "============================================================"
echo "Training V9 local-search head for $TRAIN_CONFIG"
echo "Output: $OUTPUT"
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "Log: $LOG_PATH"
echo "============================================================"

python programs/train_lorat_v9_local_search_head.py \
    --dataset-root "$DATASET_ROOT" \
    --split "$TRAIN_SPLIT" \
    --val-split "$VAL_SPLIT" \
    --mot17-root "${MOT17_ROOT:-}" \
    --mot17-split "$MOT17_SPLIT" \
    --mot17-val-split "$MOT17_VAL_SPLIT" \
    --tao-root "${TAO_ROOT:-}" \
    --tao-split "$TAO_SPLIT" \
    --tao-val-split "$TAO_VAL_SPLIT" \
    --lasot-root "${LASOT_ROOT:-}" \
    --lasot-split "$LASOT_SPLIT" \
    --lasot-val-split "$LASOT_VAL_SPLIT" \
    --lasot-val-fraction "$LASOT_VAL_FRACTION" \
    --lorat-root "$PWD/external/LoRAT-main" \
    --lorat-config "$TRAIN_CONFIG" \
    --device cuda:0 \
    --max-objects "$MAX_OBJECTS" \
    --target-region-mode "$TARGET_REGION_MODE" \
    --target-regions-per-object "$TARGET_REGIONS_PER_OBJECT" \
    --template-area-factor "$TEMPLATE_AREA_FACTOR" \
    --search-area-factor "$SEARCH_AREA_FACTOR" \
    --search-scale-jitter "$SEARCH_SCALE_JITTER" \
    --search-translation-jitter "$SEARCH_TRANSLATION_JITTER" \
    --search-min-object-size "$SEARCH_MIN_OBJECT_SIZE" \
    --search-anchor-mode "$SEARCH_ANCHOR_MODE" \
    --search-target-padding-fraction "$SEARCH_TARGET_PADDING_FRACTION" \
    --template-sampling "$TEMPLATE_SAMPLING" \
    --previous-box-jitter "$PREVIOUS_BOX_JITTER" \
    --sequence-window-length "$SEQUENCE_WINDOW_LENGTH" \
    --lost-target-probability "$LOST_TARGET_PROBABILITY" \
    --max-lost-targets-per-frame "$MAX_LOST_TARGETS_PER_FRAME" \
    --max-missing-gap-frames "$MAX_MISSING_GAP_FRAMES" \
    --drift-negative-probability "$DRIFT_NEGATIVE_PROBABILITY" \
    --drift-negative-start-epoch "$DRIFT_NEGATIVE_START_EPOCH" \
    --drift-negative-max-per-frame "$DRIFT_NEGATIVE_MAX_PER_FRAME" \
    --drift-negative-shift-factor "$DRIFT_NEGATIVE_SHIFT_FACTOR" \
    --frame-stride "$FRAME_STRIDE" \
    --val-frame-stride "$VAL_FRAME_STRIDE" \
    --mot17-frame-stride "$MOT17_FRAME_STRIDE" \
    --mot17-val-frame-stride "$MOT17_VAL_FRAME_STRIDE" \
    --tao-frame-stride "$TAO_FRAME_STRIDE" \
    --tao-val-frame-stride "$TAO_VAL_FRAME_STRIDE" \
    --lasot-frame-stride "$LASOT_FRAME_STRIDE" \
    --lasot-val-frame-stride "$LASOT_VAL_FRAME_STRIDE" \
    --max-sequences "$MAX_SEQUENCES" \
    --max-val-sequences "$MAX_VAL_SEQUENCES" \
    --mot17-max-sequences "$MOT17_MAX_SEQUENCES" \
    --mot17-max-val-sequences "$MOT17_MAX_VAL_SEQUENCES" \
    --tao-max-sequences "$TAO_MAX_SEQUENCES" \
    --tao-max-val-sequences "$TAO_MAX_VAL_SEQUENCES" \
    --lasot-max-sequences "$LASOT_MAX_SEQUENCES" \
    --lasot-max-val-sequences "$LASOT_MAX_VAL_SEQUENCES" \
    --max-samples "$MAX_SAMPLES" \
    --max-val-samples "$MAX_VAL_SAMPLES" \
    --mot17-max-samples "$MOT17_MAX_SAMPLES" \
    --mot17-max-val-samples "$MOT17_MAX_VAL_SAMPLES" \
    --tao-max-samples "$TAO_MAX_SAMPLES" \
    --tao-max-val-samples "$TAO_MAX_VAL_SAMPLES" \
    --lasot-max-samples "$LASOT_MAX_SAMPLES" \
    --lasot-max-val-samples "$LASOT_MAX_VAL_SAMPLES" \
    --epochs "$EPOCHS" \
    --max-train-samples-per-epoch "$MAX_TRAIN_SAMPLES_PER_EPOCH" \
    --source-sampling-weights "$SOURCE_SAMPLING_WEIGHTS" \
    --max-steps "$MAX_STEPS" \
    --max-wall-hours "$MAX_WALL_HOURS" \
    --eval-interval-epochs "$EVAL_INTERVAL_EPOCHS" \
    --train-diagnostic-samples "$TRAIN_DIAGNOSTIC_SAMPLES" \
    --learning-rate "$LEARNING_RATE" \
    --weight-decay "$WEIGHT_DECAY" \
    --box-loss-weight "$BOX_LOSS_WEIGHT" \
    --ltrb-loss-weight "$LTRB_LOSS_WEIGHT" \
    --reid-loss-weight "$REID_LOSS_WEIGHT" \
    --center-positive-weight "$CENTER_POSITIVE_WEIGHT" \
    --negative-loss-weight "$NEGATIVE_LOSS_WEIGHT" \
    --hard-negative-loss-weight "$HARD_NEGATIVE_LOSS_WEIGHT" \
    --focal-loss-gamma "$FOCAL_LOSS_GAMMA" \
    --small-target-loss-weight "$SMALL_TARGET_LOSS_WEIGHT" \
    --small-target-area-threshold "$SMALL_TARGET_AREA_THRESHOLD" \
    --small-target-max-side "$SMALL_TARGET_MAX_SIDE" \
    --dcfst-discrimination-weight "$DCFST_DISCRIMINATION_WEIGHT" \
    --assignment-discrimination-weight "$ASSIGNMENT_DISCRIMINATION_WEIGHT" \
    --assignment-margin "$ASSIGNMENT_MARGIN" \
    --candidate-ranking-loss-weight "$CANDIDATE_RANKING_LOSS_WEIGHT" \
    --candidate-ranking-start-epoch "$CANDIDATE_RANKING_START_EPOCH" \
    --candidate-ranking-margin "$CANDIDATE_RANKING_MARGIN" \
    --candidate-ranking-topk "$CANDIDATE_RANKING_TOPK" \
    --candidate-ranking-positive-iou "$CANDIDATE_RANKING_POSITIVE_IOU" \
    --candidate-ranking-other-iou "$CANDIDATE_RANKING_OTHER_IOU" \
    --rollout-identity-switch-penalty "$ROLLOUT_IDENTITY_SWITCH_PENALTY" \
    --rollout-window-miss-penalty "$ROLLOUT_WINDOW_MISS_PENALTY" \
    --rollout-validation-mode "$ROLLOUT_VALIDATION_MODE" \
    --geometry-only-epochs "$GEOMETRY_ONLY_EPOCHS" \
    --hard-negative-start-epoch "$HARD_NEGATIVE_START_EPOCH" \
    --reid-start-epoch "$REID_START_EPOCH" \
    --dcfst-start-epoch "$DCFST_START_EPOCH" \
    --assignment-start-epoch "$ASSIGNMENT_START_EPOCH" \
    --closed-loop-start-epoch "$CLOSED_LOOP_START_EPOCH" \
    --closed-loop-probability "$CLOSED_LOOP_PROBABILITY" \
    --v8-head-hidden-dim "$HEAD_HIDDEN_DIM" \
    --v8-head-lora-rank "$HEAD_LORA_RANK" \
    --v8-head-rank "$HEAD_RANK" \
    --v8-search-radius-factor "$SEARCH_RADIUS_FACTOR" \
    --v8-window-penalty-ratio "$WINDOW_PENALTY_RATIO" \
    --v9-local-grid-size "$LOCAL_GRID_SIZE" \
    --training-memory-slots "$TRAINING_MEMORY_SLOTS" \
    --checkpoint-interval "$CHECKPOINT_INTERVAL" \
    --checkpoint-retention "$CHECKPOINT_RETENTION" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --diagnostic-csv "$RESULT_ROOT/logs/train_${CONFIG_SAFE}_diagnostics.csv" \
    --output "$OUTPUT" \
    "${EXTRA_ARGS[@]}" 2>&1 | tee "$LOG_PATH"

echo "Training complete. Result files:"
find "$RESULT_ROOT" -maxdepth 4 -type f -print
