#!/usr/bin/env bash
#SBATCH --job-name=lorat-v8-b224-tao-48h
#SBATCH --account=rc_general
#SBATCH --partition=gpu-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --signal=TERM@600
#SBATCH --gres=gpu:1
#SBATCH --output=/work/%u/lorat-v8-train-b224-tao-48h-%j.out

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Refusing to run outside a Slurm allocation. Submit this script with sbatch."
    exit 2
fi

export TRAIN_CONFIGS="${TRAIN_CONFIGS:-B-224}"
export EPOCHS="${EPOCHS:-250}"
export MAX_WALL_HOURS="${MAX_WALL_HOURS:-47.25}"
export MAX_TRAIN_SAMPLES_PER_EPOCH="${MAX_TRAIN_SAMPLES_PER_EPOCH:-4096}"
export EVAL_INTERVAL_EPOCHS="${EVAL_INTERVAL_EPOCHS:-5}"
export DIAGNOSTIC_INTERVAL_EPOCHS="${DIAGNOSTIC_INTERVAL_EPOCHS:-10}"
export RESULT_ROOT="${RESULT_ROOT:-/work/$USER/lorat-v8-train-results-b224-mixed-tao-yfcc-48h-$SLURM_JOB_ID}"

# TAO/YFCC is intentionally added as a mixed-data source, not as a replacement
# for the existing DanceTrack/MOT-style Combined dataset inside the training bundle.
export TAO_SPLIT="${TAO_SPLIT:-train}"
export TAO_VAL_SPLIT="${TAO_VAL_SPLIT:-validation}"
export TAO_USE_FREEFORM="${TAO_USE_FREEFORM:-1}"
export TAO_FRAME_STRIDE="${TAO_FRAME_STRIDE:-1}"
export TAO_VAL_FRAME_STRIDE="${TAO_VAL_FRAME_STRIDE:-1}"
export TAO_MAX_SEQUENCES="${TAO_MAX_SEQUENCES:-0}"
export TAO_MAX_VAL_SEQUENCES="${TAO_MAX_VAL_SEQUENCES:-0}"
export TAO_MAX_SAMPLES="${TAO_MAX_SAMPLES:-0}"
export TAO_MAX_VAL_SAMPLES="${TAO_MAX_VAL_SAMPLES:-256}"

export DEBUG_VISUAL_EVERY_EPOCH="${DEBUG_VISUAL_EVERY_EPOCH:-0}"
export DEBUG_VISUAL_SAMPLES="${DEBUG_VISUAL_SAMPLES:-12}"
export TRAIN_DIAGNOSTIC_SAMPLES="${TRAIN_DIAGNOSTIC_SAMPLES:-64}"
export MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-256}"
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-100}"

exec bash "/work/$USER/theia_v8_train_heads.sbatch"
