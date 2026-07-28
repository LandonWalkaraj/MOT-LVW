#!/usr/bin/env bash
#SBATCH --job-name=lorat-v9-g378
#SBATCH --account=rc_general
#SBATCH --partition=gpu-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --signal=TERM@600
#SBATCH --gres=gpu:1
#SBATCH --output=/work/%u/lorat-v9-train-g378-24h-%j.out

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Refusing to run outside a Slurm allocation. Submit this script with sbatch."
    exit 2
fi

# G-378 mixed-data training wrapper.
#
# This intentionally reuses the B-224 mixed TAO/LaSOT/MOT training script and
# changes only the LoRaT backbone config/result naming. That keeps the g-378
# run trained like the current B-224 run:
# - multi-object samples by default;
# - mixed target regions;
# - TAO-heavy mixed source sampling;
# - rollout/checkpoint/auto-resume behavior from the shared B-224 script.

TRAIN_CONFIG="${TRAIN_CONFIG:-g-378}"
CONFIG_SAFE="${TRAIN_CONFIG//-/_}"
BASE_SCRIPT="${BASE_SCRIPT:-/work/$USER/theia_v9_train_b224_48h_tao.sbatch}"

export TRAIN_CONFIG
export RESULT_ROOT="${RESULT_ROOT:-/work/$USER/lorat-v9-train-results-${CONFIG_SAFE}-tao-24h-$SLURM_JOB_ID}"

echo "G-378 mixed-data V9 training wrapper"
echo "Base script: $BASE_SCRIPT"
echo "Train config: $TRAIN_CONFIG"
echo "Result root: $RESULT_ROOT"
echo "Recipe: inherited from B-224 mixed TAO/LaSOT/MOT training"
echo "Max objects: ${MAX_OBJECTS:-base-default}"
echo "Target region mode: ${TARGET_REGION_MODE:-base-default}"
echo "Target regions per object: ${TARGET_REGIONS_PER_OBJECT:-base-default}"
echo "Source sampling weights: ${SOURCE_SAMPLING_WEIGHTS:-base-default}"
echo "Max train samples per epoch: ${MAX_TRAIN_SAMPLES_PER_EPOCH:-base-default}"
echo "Auto resume head: ${AUTO_RESUME_HEAD:-base-default}"
echo "Resume head: ${RESUME_HEAD:-none}"

if [[ ! -f "$BASE_SCRIPT" ]]; then
    echo "Missing base V9 training script: $BASE_SCRIPT"
    exit 1
fi

bash "$BASE_SCRIPT"
