#!/usr/bin/env bash
#SBATCH --job-name=lorat-v8-large-small
#SBATCH --account=rc_general
#SBATCH --partition=gpu-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --signal=TERM@600
#SBATCH --gres=gpu:1
#SBATCH --output=/work/%u/lorat-v8-large-small-%j.out

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Refusing to run outside a Slurm allocation. Submit this script with sbatch."
    exit 2
fi

BASE_SCRIPT="${BASE_SCRIPT:-/work/$USER/theia_v8_week3_benchmark.sbatch}"
RESULT_ROOT_BASE="${RESULT_ROOT:-/work/$USER/lorat-v8-week3-largest-smallest-$SLURM_JOB_ID}"
INIT_SELECTION_ORDER="${INIT_SELECTION_ORDER:-largest smallest}"
INIT_MIN_AREA="${INIT_MIN_AREA:-0}"
INIT_MAX_AREA="${INIT_MAX_AREA:-0}"
INIT_TRACK_IDS="${INIT_TRACK_IDS:-}"

echo "Largest/smallest V8 Week 3 benchmark wrapper"
echo "Job ID: $SLURM_JOB_ID"
echo "Base script: $BASE_SCRIPT"
echo "Result root base: $RESULT_ROOT_BASE"
echo "Init selection order: $INIT_SELECTION_ORDER"
echo "Init area window: min=$INIT_MIN_AREA max=$INIT_MAX_AREA"
echo "Init track ids: ${INIT_TRACK_IDS:-none}"

if [[ ! -f "$BASE_SCRIPT" ]]; then
    echo "Missing base benchmark script: $BASE_SCRIPT"
    exit 1
fi

mkdir -p "$RESULT_ROOT_BASE"

for mode in $INIT_SELECTION_ORDER; do
    case "$mode" in
        largest|smallest|area-window|middle)
            ;;
        *)
            echo "Unsupported init selection mode: $mode"
            exit 2
            ;;
    esac

    case_root="$RESULT_ROOT_BASE/$mode"
    echo ""
    echo "============================================================"
    echo "Running init-selection=$mode"
    echo "Case result root: $case_root"
    echo "============================================================"

    INIT_SELECTION="$mode" \
    INIT_MIN_AREA="$INIT_MIN_AREA" \
    INIT_MAX_AREA="$INIT_MAX_AREA" \
    INIT_TRACK_IDS="$INIT_TRACK_IDS" \
    RESULT_ROOT="$case_root" \
    bash "$BASE_SCRIPT"
done

echo ""
echo "Largest/smallest benchmark complete. Summaries:"
find "$RESULT_ROOT_BASE" -maxdepth 4 -type f -name "summary.md" -print
