#!/usr/bin/env bash
#SBATCH --job-name=lorat-v8-overfit
#SBATCH --account=rc_general
#SBATCH --partition=gpu-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --signal=TERM@300
#SBATCH --gres=gpu:1
#SBATCH --output=/work/%u/lorat-v8-overfit-%j.out

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Refusing to run outside a Slurm allocation. Submit this script with sbatch."
    exit 2
fi

BASE_SCRIPT="${BASE_SCRIPT:-/work/$USER/theia_v8_train_heads.sbatch}"
if [[ ! -f "$BASE_SCRIPT" ]]; then
    echo "Missing base V8 training script: $BASE_SCRIPT"
    exit 1
fi

# Use the existing full training bundle already staged on /work, then overlay the
# current local code uploaded specifically for this diagnostic run.
export BUNDLE="${BUNDLE:-/work/$USER/theia_lorat_v8_train_bundle.tar.gz}"
export UPDATE_ROOT="${UPDATE_ROOT:-/work/$USER/lorat-v8-overfit-diagnostic-updates}"
export RUN_ROOT="${RUN_ROOT:-/work/$USER/lorat-v8-overfit-job-$SLURM_JOB_ID}"
export RESULT_ROOT="${RESULT_ROOT:-/work/$USER/lorat-v8-overfit-diagnostic-$SLURM_JOB_ID}"
export CLEANUP_RUN_ROOT="${CLEANUP_RUN_ROOT:-1}"

# Tiny B-224-only overfit probe. This should answer whether the V8 head can
# memorize selected-object geometry before we spend time on a full run.
export TRAIN_CONFIGS="${TRAIN_CONFIGS:-B-224}"
export MAX_OBJECTS="${MAX_OBJECTS:-8}"
export MAX_SEQUENCES="${MAX_SEQUENCES:-1}"
export MAX_VAL_SEQUENCES="${MAX_VAL_SEQUENCES:-1}"
export MAX_SAMPLES="${MAX_SAMPLES:-32}"
export MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-32}"
export EPOCHS="${EPOCHS:-60}"
export MAX_STEPS="${MAX_STEPS:-0}"

# Keep V8's intended training path, but make this friendly to overfitting.
export LEARNING_RATE="${LEARNING_RATE:-3e-4}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
export TARGET_REGION_MODE="${TARGET_REGION_MODE:-mixed}"
export TARGET_REGIONS_PER_OBJECT="${TARGET_REGIONS_PER_OBJECT:-2}"
export TEMPLATE_SAMPLING="${TEMPLATE_SAMPLING:-mixed}"
export SEARCH_ANCHOR_MODE="${SEARCH_ANCHOR_MODE:-current}"
export REPAIR_SEARCH_TO_TARGET="${REPAIR_SEARCH_TO_TARGET:-1}"
export SEARCH_TARGET_PADDING_FRACTION="${SEARCH_TARGET_PADDING_FRACTION:-0.05}"

# Diagnostic proof and hard stop.
export OVERFIT_SMOKE="${OVERFIT_SMOKE:-1}"
export OVERFIT_SINGLE_OBJECT="${OVERFIT_SINGLE_OBJECT:-1}"
export OVERFIT_SMOKE_SAMPLES="${OVERFIT_SMOKE_SAMPLES:-32}"
export TRAIN_DIAGNOSTIC_SAMPLES="${TRAIN_DIAGNOSTIC_SAMPLES:-32}"
export QUALITY_GATE_MEAN_IOU="${QUALITY_GATE_MEAN_IOU:-0.50}"
export QUALITY_GATE_IOU50="${QUALITY_GATE_IOU50:-0.50}"
export DEBUG_VISUAL_SAMPLES="${DEBUG_VISUAL_SAMPLES:-32}"
export DEBUG_VISUAL_EVERY_EPOCH="${DEBUG_VISUAL_EVERY_EPOCH:-1}"
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-500}"

echo "Launching V8 B-224 overfit diagnostic through $BASE_SCRIPT"
echo "Result root: $RESULT_ROOT"
echo "Update root: $UPDATE_ROOT"
echo "Quality gate: mean IoU >= $QUALITY_GATE_MEAN_IOU and IoU@0.5 >= $QUALITY_GATE_IOU50"

bash "$BASE_SCRIPT"
