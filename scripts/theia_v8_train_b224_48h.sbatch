#!/usr/bin/env bash
#SBATCH --job-name=lorat-v8-b224-48h
#SBATCH --account=rc_general
#SBATCH --partition=gpu-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --signal=TERM@600
#SBATCH --gres=gpu:1
#SBATCH --output=/work/%u/lorat-v8-train-b224-48h-%j.out

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Refusing to run outside a Slurm allocation. Submit this script with sbatch."
    exit 2
fi

export TRAIN_CONFIGS="${TRAIN_CONFIGS:-B-224}"
export EPOCHS="${EPOCHS:-12}"
export RESULT_ROOT="${RESULT_ROOT:-/work/$USER/lorat-v8-train-results-b224-anchor-dcfst-48h-$SLURM_JOB_ID}"
export DEBUG_VISUAL_EVERY_EPOCH="${DEBUG_VISUAL_EVERY_EPOCH:-1}"
export DEBUG_VISUAL_SAMPLES="${DEBUG_VISUAL_SAMPLES:-12}"
export TRAIN_DIAGNOSTIC_SAMPLES="${TRAIN_DIAGNOSTIC_SAMPLES:-128}"
export MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-512}"
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-250}"

exec bash "/work/$USER/theia_v8_train_heads.sbatch"
