#!/usr/bin/env bash
#SBATCH --job-name=lorat-v8-demo-video
#SBATCH --account=rc_general
#SBATCH --partition=gpu-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --signal=TERM@600
#SBATCH --gres=gpu:1
#SBATCH --output=/work/%u/lorat-v8-demo-video-%j.out

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Refusing to run outside a Slurm allocation. Submit this script with sbatch."
    exit 2
fi

# This wrapper is intentionally presentation/demo oriented:
# - B-224 only, because the 48h improved training run produced a B-224 head.
# - Full DanceTrack0065 sequence.
# - Save videos, summaries, and CSVs for proof/review.
# - Use the explicit latest trained checkpoint root instead of auto-detecting.
export COMPARE_CONFIGS="${COMPARE_CONFIGS:-B-224}"
export HEAD_WEIGHTS_ROOT="${HEAD_WEIGHTS_ROOT:-/work/$USER/lorat-v8-train-results-b224-anchor-dcfst-48h-670262/checkpoints}"
export HEAD_CHECKPOINT="${HEAD_CHECKPOINT:-best}"
export RESULT_ROOT="${RESULT_ROOT:-/work/$USER/lorat-v8-demo-video-results-$SLURM_JOB_ID}"
export TRACK_COUNTS="${TRACK_COUNTS:-1,2,3,4,5}"
export SEQUENCES="${SEQUENCES:-dancetrack0065}"
export INIT_SELECTION="${INIT_SELECTION:-largest}"
export INIT_MIN_AREA="${INIT_MIN_AREA:-0}"
export INIT_MAX_AREA="${INIT_MAX_AREA:-0}"
export INIT_TRACK_IDS="${INIT_TRACK_IDS:-}"
export MAX_FRAMES="${MAX_FRAMES:-0}"
export SAVE_VIDEO="${SAVE_VIDEO:-1}"
export DINOV2_CROP_REID="${DINOV2_CROP_REID:-1}"
export DINOV2_CROP_REID_BATCH="${DINOV2_CROP_REID_BATCH:-16}"
export AREA_SAMPLE_INTERVAL="${AREA_SAMPLE_INTERVAL:-10}"
export IDENTITY_SAMPLE_INTERVAL="${IDENTITY_SAMPLE_INTERVAL:-1}"
export CONTROLLED_OCCLUSION_DURATIONS="${CONTROLLED_OCCLUSION_DURATIONS:-0,5,10,20,40}"
export CONTROLLED_OCCLUSION_TRIALS_PER_DURATION="${CONTROLLED_OCCLUSION_TRIALS_PER_DURATION:-2}"
export CONTROLLED_OCCLUSION_WARMUP_FRAMES="${CONTROLLED_OCCLUSION_WARMUP_FRAMES:-10}"
export CONTROLLED_OCCLUSION_RECOVERY_FRAMES="${CONTROLLED_OCCLUSION_RECOVERY_FRAMES:-30}"

echo "Demo video benchmark wrapper"
echo "Result root: $RESULT_ROOT"
echo "Head weights root: $HEAD_WEIGHTS_ROOT"
echo "Head checkpoint: $HEAD_CHECKPOINT"
echo "Track counts: $TRACK_COUNTS"
echo "Sequences: $SEQUENCES"
echo "Init selection: $INIT_SELECTION"
echo "Init area window: min=$INIT_MIN_AREA max=$INIT_MAX_AREA"
echo "Init track ids: ${INIT_TRACK_IDS:-none}"

exec bash "/work/$USER/theia_v8_week3_benchmark.sbatch"
