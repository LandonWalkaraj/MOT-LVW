#!/usr/bin/env bash
#SBATCH --job-name=lorat-v9-bench
#SBATCH --account=rc_general
#SBATCH --partition=gpu-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --signal=TERM@600
#SBATCH --gres=gpu:1
#SBATCH --output=/work/%u/lorat-v9-benchmark-%j.out

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Refusing to run outside a Slurm allocation. Submit this script with sbatch."
    exit 2
fi

BUNDLE="${BUNDLE:-/work/$USER/theia_lorat_v8_train_bundle.tar.gz}"
UPDATE_ROOT="${UPDATE_ROOT:-/work/$USER/lorat-v9-live-updates}"
RUN_ROOT="${RUN_ROOT:-/work/$USER/lorat-v9-benchmark-job-$SLURM_JOB_ID}"
RESULT_ROOT="${RESULT_ROOT:-/work/$USER/lorat-v9-benchmark-results-$SLURM_JOB_ID}"
VENV_DIR="${VENV_DIR:-/work/$USER/lorat-v9-venv-cu128}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-/work/$USER/.cache/pip}"
EXPECT_GPU_NAME="${EXPECT_GPU_NAME:-A100}"
HEAD_WEIGHTS_ROOT="${HEAD_WEIGHTS_ROOT:-}"
HEAD_CHECKPOINT="${HEAD_CHECKPOINT:-best_by_rollout_identity}"
DATASET_ROOT="${DATASET_ROOT:-}"
CLEANUP_RUN_ROOT="${CLEANUP_RUN_ROOT:-1}"
TAO_ROOT="${TAO_ROOT:-}"
TAO_SPLIT="${TAO_SPLIT:-validation}"
TAO_USE_FREEFORM="${TAO_USE_FREEFORM:-1}"
TAO_EXAMPLE_ROOT="${TAO_EXAMPLE_ROOT:-}"
TAO_EXAMPLE_VIDEOS="${TAO_EXAMPLE_VIDEOS:-1}"
TAO_EXAMPLE_MIN_TRACKS="${TAO_EXAMPLE_MIN_TRACKS:-1}"
TAO_EXAMPLE_MIN_ANNOTATED_FRAMES="${TAO_EXAMPLE_MIN_ANNOTATED_FRAMES:-10}"
TAO_EXAMPLE_MAX_FRAMES="${TAO_EXAMPLE_MAX_FRAMES:-300}"
TAO_EXAMPLE_COPY_MODE="${TAO_EXAMPLE_COPY_MODE:-copy}"
LASOT_ROOT="${LASOT_ROOT:-}"
LASOT_SPLIT="${LASOT_SPLIT:-val}"
LASOT_EXAMPLE_ROOT="${LASOT_EXAMPLE_ROOT:-}"
LASOT_EXAMPLE_VIDEOS="${LASOT_EXAMPLE_VIDEOS:-1}"
LASOT_EXAMPLE_MIN_VISIBLE_FRAMES="${LASOT_EXAMPLE_MIN_VISIBLE_FRAMES:-10}"
LASOT_EXAMPLE_MAX_FRAMES="${LASOT_EXAMPLE_MAX_FRAMES:-300}"
LASOT_EXAMPLE_COPY_MODE="${LASOT_EXAMPLE_COPY_MODE:-copy}"
LASOT_VAL_FRACTION="${LASOT_VAL_FRACTION:-0.20}"
BDD100K_ROOT="${BDD100K_ROOT:-}"
BDD100K_SPLIT="${BDD100K_SPLIT:-val}"
BDD100K_EXAMPLE_ROOT="${BDD100K_EXAMPLE_ROOT:-}"
BDD100K_EXAMPLE_VIDEOS="${BDD100K_EXAMPLE_VIDEOS:-1}"
BDD100K_EXAMPLE_MIN_TRACKS="${BDD100K_EXAMPLE_MIN_TRACKS:-5}"
BDD100K_EXAMPLE_MIN_ANNOTATED_FRAMES="${BDD100K_EXAMPLE_MIN_ANNOTATED_FRAMES:-10}"
BDD100K_EXAMPLE_MAX_FRAMES="${BDD100K_EXAMPLE_MAX_FRAMES:-300}"
BDD100K_EXAMPLE_COPY_MODE="${BDD100K_EXAMPLE_COPY_MODE:-copy}"
BDD100K_CATEGORIES="${BDD100K_CATEGORIES:-pedestrian,rider,car,truck,bus,train,motorcycle,bicycle}"
BDD100K_INCLUDE_DISTRACTORS="${BDD100K_INCLUDE_DISTRACTORS:-0}"

TRACK_COUNTS="${TRACK_COUNTS:-1,2,3,4,5}"
MAX_FRAMES="${MAX_FRAMES:-0}"
SEQUENCES="${SEQUENCES:-dancetrack0065}"
SPLIT="${SPLIT:-val}"
COMPARE_CONFIGS="${COMPARE_CONFIGS:-B-224}"
INIT_SELECTION="${INIT_SELECTION:-largest}"
INIT_MIN_AREA="${INIT_MIN_AREA:-0}"
INIT_MAX_AREA="${INIT_MAX_AREA:-0}"
INIT_TRACK_IDS="${INIT_TRACK_IDS:-}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-25}"
AREA_SAMPLE_INTERVAL="${AREA_SAMPLE_INTERVAL:-10}"
IDENTITY_SAMPLE_INTERVAL="${IDENTITY_SAMPLE_INTERVAL:-1}"
SAVE_VIDEO="${SAVE_VIDEO:-1}"
DINOV2_CROP_REID="${DINOV2_CROP_REID:-1}"
DINOV2_CROP_REID_BATCH="${DINOV2_CROP_REID_BATCH:-16}"
V9_LOCAL_GRID_SIZE="${V9_LOCAL_GRID_SIZE:-17}"
V9_DIAGNOSTIC_MODES="${V9_DIAGNOSTIC_MODES:-normal}"
DRAW_CANDIDATE_DIAGNOSTICS="${DRAW_CANDIDATE_DIAGNOSTICS:-1}"
REID_ABLATION="${REID_ABLATION:-1}"
FULL_AREA_OBSERVATIONS="${FULL_AREA_OBSERVATIONS:-1}"
CONTROLLED_OCCLUSION_DURATIONS="${CONTROLLED_OCCLUSION_DURATIONS:-0,5,10,20,40,80}"
CONTROLLED_OCCLUSION_TRIALS_PER_DURATION="${CONTROLLED_OCCLUSION_TRIALS_PER_DURATION:-3}"
CONTROLLED_OCCLUSION_WARMUP_FRAMES="${CONTROLLED_OCCLUSION_WARMUP_FRAMES:-10}"
CONTROLLED_OCCLUSION_RECOVERY_FRAMES="${CONTROLLED_OCCLUSION_RECOVERY_FRAMES:-30}"
CONTROLLED_OCCLUSION_MASK="${CONTROLLED_OCCLUSION_MASK:-black}"

export PIP_CACHE_DIR

if [[ -z "${HEAD_WEIGHTS_ROOT:-}" || "$HEAD_WEIGHTS_ROOT" == "auto" ]]; then
    mapfile -t HEAD_WEIGHT_ROOT_CANDIDATES < <(
        {
            find "/work/$USER/lorat-models/v9-heads" -maxdepth 3 -type d -name "checkpoints" -printf '%T@ %p\n' 2>/dev/null || true
            find "/work/$USER" -maxdepth 3 -type d -path "/work/$USER/lorat-v9-train-results*/checkpoints" -printf '%T@ %p\n' 2>/dev/null || true
        } \
            | sort -nr \
            | awk '{ $1=""; sub(/^ /, ""); print }'
    )
    if [[ "${#HEAD_WEIGHT_ROOT_CANDIDATES[@]}" -eq 0 ]]; then
        echo "Could not auto-resolve HEAD_WEIGHTS_ROOT. Checked /work/$USER/lorat-models/v9-heads/**/checkpoints and /work/$USER/lorat-v9-train-results*/checkpoints"
        exit 1
    fi
    HEAD_WEIGHTS_ROOT="${HEAD_WEIGHT_ROOT_CANDIDATES[0]}"
fi

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Bundle: $BUNDLE"
echo "Update root: $UPDATE_ROOT"
echo "Run root: $RUN_ROOT"
echo "Result root: $RESULT_ROOT"
echo "Head weights root: $HEAD_WEIGHTS_ROOT"
echo "Head checkpoint preference: $HEAD_CHECKPOINT"
echo "Dataset root override: ${DATASET_ROOT:-auto}"
echo "TAO root override: ${TAO_ROOT:-none}"
echo "LaSOT root override: ${LASOT_ROOT:-none}"
echo "BDD100K root override: ${BDD100K_ROOT:-none}"
echo "Split: $SPLIT"
echo "Sequences: $SEQUENCES"
echo "Track counts: $TRACK_COUNTS"
echo "Compare configs: $COMPARE_CONFIGS"
echo "Init selection: $INIT_SELECTION"
echo "Max frames: $MAX_FRAMES"
echo "Save video: $SAVE_VIDEO"
echo "V9 local grid size: $V9_LOCAL_GRID_SIZE"
echo "V9 diagnostic modes: $V9_DIAGNOSTIC_MODES"
echo "Draw candidate diagnostics: $DRAW_CANDIDATE_DIAGNOSTICS"
echo "ReID ablation: $REID_ABLATION"
echo "Controlled occlusion durations: $CONTROLLED_OCCLUSION_DURATIONS"

if [[ ! -f "$BUNDLE" ]]; then
    echo "Missing bundle: $BUNDLE"
    exit 1
fi

mkdir -p "$RUN_ROOT" "$RESULT_ROOT" "$PIP_CACHE_DIR"
cleanup_run_root() {
    if [[ "$CLEANUP_RUN_ROOT" == "1" && "$RUN_ROOT" == "/work/$USER"/lorat-v9-benchmark-job-* && -d "$RUN_ROOT" ]]; then
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

echo "V9 benchmark source hashes after live updates:"
sha256sum \
    programs/benchmark_lorat_v9.py \
    programs/benchmark_lorat_v9_harness.py \
    programs/benchmark_lorat_mot.py \
    programs/bounding_box_v9_lorat_local_search.py \
    programs/bounding_box_v9_runtime_base.py \
    programs/mot_common.py \
    programs/export_bdd100k_to_mot_sequences.py \
    programs/export_lasot_to_mot_sequences.py \
    programs/export_tao_to_mot_sequences.py \
    2>/dev/null || true

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
    if [[ -d "/work/$USER/lorat-data/raw/Combined" ]]; then
        DATASET_ROOT="/work/$USER/lorat-data/raw/Combined"
    elif [[ -d "/work/$USER/lorat-data/raw/DanceTrack" ]]; then
        DATASET_ROOT="/work/$USER/lorat-data/raw/DanceTrack"
    elif [[ -d "$PWD/data/raw/Combined" ]]; then
        DATASET_ROOT="$PWD/data/raw/Combined"
    elif [[ -d "$PWD/data/raw/DanceTrack" ]]; then
        DATASET_ROOT="$PWD/data/raw/DanceTrack"
    else
        echo "Could not find dataset root under /work/$USER/lorat-data/raw or bundled $PWD/data/raw"
        find "/work/$USER/lorat-data/raw" -maxdepth 3 -type d 2>/dev/null | head -80 || true
        find "$PWD/data/raw" -maxdepth 3 -type d 2>/dev/null | head -80 || true
        exit 1
    fi
fi
echo "Resolved dataset root: $DATASET_ROOT"

SEQUENCE_ARGS=()
NORMALIZED_SEQUENCES="${SEQUENCES//,/ }"
read -r -a SEQUENCE_ARRAY <<< "$NORMALIZED_SEQUENCES"
for sequence in "${SEQUENCE_ARRAY[@]}"; do
    SEQUENCE_ARGS+=(--sequence "$sequence")
done

DATA_SOURCE_ARGS=()
if [[ -n "${BDD100K_ROOT:-}" ]]; then
    if [[ ! -d "$BDD100K_ROOT" ]]; then
        echo "BDD100K_ROOT was set but does not exist: $BDD100K_ROOT"
        exit 1
    fi
    if [[ -z "$BDD100K_EXAMPLE_ROOT" ]]; then
        BDD100K_EXAMPLE_ROOT="$RESULT_ROOT/_bdd100k_mot_examples"
    fi
    DATA_SOURCE_ARGS=(
        --bdd100k-root "$BDD100K_ROOT"
        --bdd100k-split "$BDD100K_SPLIT"
        --bdd100k-example-root "$BDD100K_EXAMPLE_ROOT"
        --bdd100k-example-videos "$BDD100K_EXAMPLE_VIDEOS"
        --bdd100k-example-min-tracks "$BDD100K_EXAMPLE_MIN_TRACKS"
        --bdd100k-example-min-annotated-frames "$BDD100K_EXAMPLE_MIN_ANNOTATED_FRAMES"
        --bdd100k-example-max-frames "$BDD100K_EXAMPLE_MAX_FRAMES"
        --bdd100k-example-copy-mode "$BDD100K_EXAMPLE_COPY_MODE"
        --bdd100k-categories "$BDD100K_CATEGORIES"
    )
    if [[ "$BDD100K_INCLUDE_DISTRACTORS" == "1" || "$BDD100K_INCLUDE_DISTRACTORS" == "true" || "$BDD100K_INCLUDE_DISTRACTORS" == "TRUE" ]]; then
        DATA_SOURCE_ARGS+=(--bdd100k-include-distractors)
    fi
elif [[ -n "${TAO_ROOT:-}" ]]; then
    if [[ ! -d "$TAO_ROOT" ]]; then
        echo "TAO_ROOT was set but does not exist: $TAO_ROOT"
        exit 1
    fi
    if [[ -z "$TAO_EXAMPLE_ROOT" ]]; then
        TAO_EXAMPLE_ROOT="$RESULT_ROOT/_tao_mot_examples"
    fi
    DATA_SOURCE_ARGS=(
        --tao-root "$TAO_ROOT"
        --tao-split "$TAO_SPLIT"
        --tao-example-root "$TAO_EXAMPLE_ROOT"
        --tao-example-videos "$TAO_EXAMPLE_VIDEOS"
        --tao-example-min-tracks "$TAO_EXAMPLE_MIN_TRACKS"
        --tao-example-min-annotated-frames "$TAO_EXAMPLE_MIN_ANNOTATED_FRAMES"
        --tao-example-max-frames "$TAO_EXAMPLE_MAX_FRAMES"
        --tao-example-copy-mode "$TAO_EXAMPLE_COPY_MODE"
    )
    if [[ "$TAO_USE_FREEFORM" == "1" || "$TAO_USE_FREEFORM" == "true" || "$TAO_USE_FREEFORM" == "TRUE" ]]; then
        DATA_SOURCE_ARGS+=(--tao-use-freeform)
    fi
elif [[ -n "${LASOT_ROOT:-}" ]]; then
    if [[ ! -d "$LASOT_ROOT" ]]; then
        echo "LASOT_ROOT was set but does not exist: $LASOT_ROOT"
        exit 1
    fi
    if [[ -z "$LASOT_EXAMPLE_ROOT" ]]; then
        LASOT_EXAMPLE_ROOT="$RESULT_ROOT/_lasot_mot_examples"
    fi
    DATA_SOURCE_ARGS=(
        --lasot-root "$LASOT_ROOT"
        --lasot-split "$LASOT_SPLIT"
        --lasot-example-root "$LASOT_EXAMPLE_ROOT"
        --lasot-example-videos "$LASOT_EXAMPLE_VIDEOS"
        --lasot-example-min-visible-frames "$LASOT_EXAMPLE_MIN_VISIBLE_FRAMES"
        --lasot-example-max-frames "$LASOT_EXAMPLE_MAX_FRAMES"
        --lasot-example-copy-mode "$LASOT_EXAMPLE_COPY_MODE"
        --lasot-val-fraction "$LASOT_VAL_FRACTION"
    )
else
    DATA_SOURCE_ARGS=(--dataset-root "$DATASET_ROOT" --split "$SPLIT" "${SEQUENCE_ARGS[@]}")
fi

NORMALIZED_COMPARE_CONFIGS="${COMPARE_CONFIGS//,/ }"
read -r -a COMPARE_CONFIG_ARRAY <<< "$NORMALIZED_COMPARE_CONFIGS"
for config in "${COMPARE_CONFIG_ARRAY[@]}"; do
    key="${config//-/_}"
    checkpoint_suffixes=("best_by_rollout_identity" "best_by_val_iou" "latest")
    if [[ "$HEAD_CHECKPOINT" == "best" ]]; then
        checkpoint_suffixes=("best_by_rollout_identity" "best_by_val_iou" "latest")
    elif [[ "$HEAD_CHECKPOINT" == "best_by_val_iou" ]]; then
        checkpoint_suffixes=("best_by_val_iou")
    elif [[ "$HEAD_CHECKPOINT" == "best_by_rollout_identity" ]]; then
        checkpoint_suffixes=("best_by_rollout_identity")
    elif [[ "$HEAD_CHECKPOINT" == "latest" ]]; then
        checkpoint_suffixes=("latest")
    fi
    candidates=()
    for suffix in "${checkpoint_suffixes[@]}"; do
        candidates+=(
            "$HEAD_WEIGHTS_ROOT/$key/v9_local_head_${key}_${suffix}.pt"
            "$HEAD_WEIGHTS_ROOT/checkpoints/$key/v9_local_head_${key}_${suffix}.pt"
            "$HEAD_WEIGHTS_ROOT/v9_local_head_${key}_${suffix}.pt"
            "$HEAD_WEIGHTS_ROOT/$key/v9_head_${key}_${suffix}.pt"
        )
    done
    candidates+=("$HEAD_WEIGHTS_ROOT/models/lorat/v9_local_head_${key}.pt")
    found=""
    for candidate in "${candidates[@]}"; do
        if [[ -f "$candidate" ]]; then
            found="$candidate"
            break
        fi
    done
    if [[ -z "$found" ]]; then
        echo "Missing trained V9 head for $config under $HEAD_WEIGHTS_ROOT"
        printf '  checked: %s\n' "${candidates[@]}"
        exit 1
    fi
    checkpoint_kind="unknown"
    for suffix in "${checkpoint_suffixes[@]}"; do
        if [[ "$found" == *"_$suffix.pt" ]]; then
            checkpoint_kind="$suffix"
            break
        fi
    done
    echo "Using V9 head for $config: $found"
    echo "Resolved checkpoint preference result for $config: $checkpoint_kind"
    python - "$found" "$checkpoint_kind" <<'PY'
import sys
import torch

path = sys.argv[1]
kind = sys.argv[2]
try:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
except Exception as exc:
    print(f"Resolved checkpoint metadata: file={path} preference={kind} load_error={exc}", flush=True)
    raise
if not isinstance(checkpoint, dict):
    print(f"Resolved checkpoint metadata: file={path} preference={kind} non_dict_checkpoint", flush=True)
else:
    print(
        "Resolved checkpoint metadata: "
        f"file={path} preference={kind} "
        f"epoch={checkpoint.get('epoch')} "
        f"steps={checkpoint.get('steps')} "
        f"val_iou={checkpoint.get('val_mean_iou')} "
        f"best_val_iou={checkpoint.get('best_val_iou')} "
        f"rollout_correct={checkpoint.get('val_rollout_correct_rate_iou30')} "
        f"rollout_switch={checkpoint.get('val_rollout_identity_switch_rate')} "
        f"rollout_track_loss={checkpoint.get('val_rollout_track_loss_rate')} "
        f"rollout_frames_until_loss={checkpoint.get('val_rollout_mean_frames_until_loss')} "
        f"rollout_mode={checkpoint.get('rollout_validation_mode')} "
        f"checkpoint_selection_score={checkpoint.get('checkpoint_selection_score')}",
        flush=True,
    )
PY
done

INIT_SELECTION_ARGS=(
    --init-selection "$INIT_SELECTION"
    --init-min-area "$INIT_MIN_AREA"
    --init-max-area "$INIT_MAX_AREA"
)
if [[ -n "$INIT_TRACK_IDS" ]]; then
    IFS=',' read -r -a INIT_TRACK_ID_ARRAY <<< "$INIT_TRACK_IDS"
    for track_id in "${INIT_TRACK_ID_ARRAY[@]}"; do
        if [[ -n "$track_id" ]]; then
            INIT_SELECTION_ARGS+=(--init-track-id "$track_id")
        fi
    done
fi

VIDEO_ARGS=()
if [[ "$SAVE_VIDEO" == "1" || "$SAVE_VIDEO" == "true" || "$SAVE_VIDEO" == "TRUE" ]]; then
    VIDEO_ARGS+=(--save-video)
fi

CROP_REID_ARGS=(--v8-dinov2-crop-reid-batch "$DINOV2_CROP_REID_BATCH")
if [[ "$DINOV2_CROP_REID" == "0" || "$DINOV2_CROP_REID" == "false" || "$DINOV2_CROP_REID" == "FALSE" ]]; then
    CROP_REID_ARGS+=(--disable-v8-dinov2-crop-reid)
fi

WEEK3_ARGS=()
if [[ "$REID_ABLATION" == "1" || "$REID_ABLATION" == "true" || "$REID_ABLATION" == "TRUE" ]]; then
    WEEK3_ARGS+=(--reid-ablation)
fi
if [[ "$FULL_AREA_OBSERVATIONS" == "1" || "$FULL_AREA_OBSERVATIONS" == "true" || "$FULL_AREA_OBSERVATIONS" == "TRUE" ]]; then
    WEEK3_ARGS+=(--full-area-observations)
fi

CONTROLLED_OCCLUSION_ARGS=(
    --controlled-occlusion-durations "$CONTROLLED_OCCLUSION_DURATIONS"
    --controlled-occlusion-trials-per-duration "$CONTROLLED_OCCLUSION_TRIALS_PER_DURATION"
    --controlled-occlusion-warmup-frames "$CONTROLLED_OCCLUSION_WARMUP_FRAMES"
    --controlled-occlusion-recovery-frames "$CONTROLLED_OCCLUSION_RECOVERY_FRAMES"
    --controlled-occlusion-mask "$CONTROLLED_OCCLUSION_MASK"
)
if [[ "$CONTROLLED_OCCLUSION_DURATIONS" == "none" || "$CONTROLLED_OCCLUSION_DURATIONS" == "NONE" ]]; then
    CONTROLLED_OCCLUSION_ARGS=(--controlled-occlusion-durations "")
fi

NORMALIZED_DIAGNOSTIC_MODES="${V9_DIAGNOSTIC_MODES//,/ }"
read -r -a DIAGNOSTIC_MODE_ARRAY <<< "$NORMALIZED_DIAGNOSTIC_MODES"
if [[ "${#DIAGNOSTIC_MODE_ARRAY[@]}" -eq 0 ]]; then
    DIAGNOSTIC_MODE_ARRAY=(normal)
fi
HAS_NORMAL_DIAGNOSTIC_MODE=0
for diagnostic_mode in "${DIAGNOSTIC_MODE_ARRAY[@]}"; do
    if [[ "$diagnostic_mode" == "normal" ]]; then
        HAS_NORMAL_DIAGNOSTIC_MODE=1
        break
    fi
done
if [[ "$HAS_NORMAL_DIAGNOSTIC_MODE" -eq 0 ]]; then
    echo "V9 diagnostic modes did not include real runtime mode 'normal'; prepending normal before GT-assisted diagnostics."
    DIAGNOSTIC_MODE_ARRAY=(normal "${DIAGNOSTIC_MODE_ARRAY[@]}")
fi
echo "Resolved V9 diagnostic mode run order: ${DIAGNOSTIC_MODE_ARRAY[*]}"

for diagnostic_mode in "${DIAGNOSTIC_MODE_ARRAY[@]}"; do
    if [[ -z "$diagnostic_mode" ]]; then
        continue
    fi
    DIAGNOSTIC_ARGS=(--v9-diagnostic-mode "$diagnostic_mode")
    if [[ "$DRAW_CANDIDATE_DIAGNOSTICS" == "1" || "$DRAW_CANDIDATE_DIAGNOSTICS" == "true" || "$DRAW_CANDIDATE_DIAGNOSTICS" == "TRUE" ]]; then
        DIAGNOSTIC_ARGS+=(--draw-candidate-diagnostics)
    fi
    MODE_RESULT_ROOT="$RESULT_ROOT"
    if [[ "${#DIAGNOSTIC_MODE_ARRAY[@]}" -gt 1 ]]; then
        MODE_RESULT_ROOT="$RESULT_ROOT/$diagnostic_mode"
    fi
    echo "Running V9 benchmark diagnostic_mode=$diagnostic_mode result_root=$MODE_RESULT_ROOT"
    python programs/benchmark_lorat_v9.py \
        --lorat-root "$PWD/external/LoRAT-main" \
        --device cuda:0 \
        --gpu-profile hpc \
        "${DATA_SOURCE_ARGS[@]}" \
        --track-counts "$TRACK_COUNTS" \
        --compare-configs "${COMPARE_CONFIG_ARRAY[@]}" \
        --v9-head-weights-root "$HEAD_WEIGHTS_ROOT" \
        --v9-head-checkpoint "$HEAD_CHECKPOINT" \
        --v9-local-grid-size "$V9_LOCAL_GRID_SIZE" \
        "${DIAGNOSTIC_ARGS[@]}" \
        "${INIT_SELECTION_ARGS[@]}" \
        --max-frames "$MAX_FRAMES" \
        --progress-interval "$PROGRESS_INTERVAL" \
        --area-sample-interval "$AREA_SAMPLE_INTERVAL" \
        --identity-sample-interval "$IDENTITY_SAMPLE_INTERVAL" \
        "${CROP_REID_ARGS[@]}" \
        "${WEEK3_ARGS[@]}" \
        "${CONTROLLED_OCCLUSION_ARGS[@]}" \
        --output-root "$MODE_RESULT_ROOT" \
        "${VIDEO_ARGS[@]}"
done

echo "V9 benchmark complete. Summaries and videos:"
find "$RESULT_ROOT" -maxdepth 8 -type f \( -name "summary.md" -o -name "*.mp4" \) -print
