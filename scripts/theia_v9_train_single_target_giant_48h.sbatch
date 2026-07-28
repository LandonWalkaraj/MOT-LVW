#!/usr/bin/env bash
#SBATCH --job-name=lorat-v9-g378-n1
#SBATCH --account=rc_general
#SBATCH --partition=gpu-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --signal=TERM@600
#SBATCH --gres=gpu:1
#SBATCH --output=/work/%u/lorat-v9-single-target-g378-48h-%j.out

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Refusing to run outside a Slurm allocation. Submit this script with sbatch."
    exit 2
fi

# This is a controlled isolation run for the question:
# "Can V9 track one selected person if we use the largest LoRaT backbone?"
#
# It reuses the main mixed-data V9 trainer, but changes the experimental setup:
# - one selected target only, not MOT conflict training;
# - full-person target regions, not part/head/tiny-region augmentation;
# - the largest available LoRaT config by default: g-378;
# - no global auto-resume by default, so old multi-object checkpoints cannot
#   silently contaminate the diagnostic run.

TRAIN_CONFIG="${TRAIN_CONFIG:-g-378}"
CONFIG_SAFE="${TRAIN_CONFIG//-/_}"
BASE_SCRIPT="${BASE_SCRIPT:-/work/$USER/theia_v9_train_b224_48h_tao.sbatch}"

export TRAIN_CONFIG
export MAX_OBJECTS="${MAX_OBJECTS:-1}"
export TARGET_REGION_MODE="${TARGET_REGION_MODE:-full}"
export TARGET_REGIONS_PER_OBJECT="${TARGET_REGIONS_PER_OBJECT:-1}"
export TEMPLATE_SAMPLING="${TEMPLATE_SAMPLING:-mixed}"
export SEARCH_ANCHOR_MODE="${SEARCH_ANCHOR_MODE:-union}"
export SEQUENCE_WINDOW_LENGTH="${SEQUENCE_WINDOW_LENGTH:-5}"
export LOST_TARGET_PROBABILITY="${LOST_TARGET_PROBABILITY:-0.08}"
export MAX_LOST_TARGETS_PER_FRAME="${MAX_LOST_TARGETS_PER_FRAME:-1}"

# g-378 is much heavier than B-224. Keep the per-epoch budget smaller so the
# run produces useful checkpoints inside a 48-hour wall-clock limit.
export EPOCHS="${EPOCHS:-250}"
export MAX_TRAIN_SAMPLES_PER_EPOCH="${MAX_TRAIN_SAMPLES_PER_EPOCH:-1024}"
export MAX_WALL_HOURS="${MAX_WALL_HOURS:-47.25}"
export EVAL_INTERVAL_EPOCHS="${EVAL_INTERVAL_EPOCHS:-2}"
export TRAIN_DIAGNOSTIC_SAMPLES="${TRAIN_DIAGNOSTIC_SAMPLES:-64}"
export MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-256}"
export MOT17_MAX_VAL_SAMPLES="${MOT17_MAX_VAL_SAMPLES:-256}"
export TAO_MAX_VAL_SAMPLES="${TAO_MAX_VAL_SAMPLES:-256}"
export LASOT_MAX_VAL_SAMPLES="${LASOT_MAX_VAL_SAMPLES:-256}"

# Tilt toward selected-target SOT behavior while retaining person/open-world
# examples. The source names are the trainer dataset class names.
export SOURCE_SAMPLING_WEIGHTS="${SOURCE_SAMPLING_WEIGHTS:-MOTFrameHeadDataset=2.0,TAOFrameHeadDataset=1.0,LaSOTFrameHeadDataset=2.0}"

# Start clean by default. Set AUTO_RESUME_HEAD=1 or RESUME_HEAD=/path/to.pt if
# we intentionally want a continuation run later.
export AUTO_RESUME_HEAD="${AUTO_RESUME_HEAD:-0}"
export RESUME_HEAD="${RESUME_HEAD:-}"

export RESULT_ROOT="${RESULT_ROOT:-/work/$USER/lorat-v9-train-results-single-target-${CONFIG_SAFE}-48h-$SLURM_JOB_ID}"

echo "Single-target largest-LoRaT V9 training wrapper"
echo "Base script: $BASE_SCRIPT"
echo "Train config: $TRAIN_CONFIG"
echo "Max objects: $MAX_OBJECTS"
echo "Target region mode: $TARGET_REGION_MODE"
echo "Result root: $RESULT_ROOT"
echo "Auto resume head: $AUTO_RESUME_HEAD"
echo "Resume head: ${RESUME_HEAD:-none}"

if [[ ! -f "$BASE_SCRIPT" ]]; then
    echo "Missing base V9 training script: $BASE_SCRIPT"
    exit 1
fi

bash "$BASE_SCRIPT"
