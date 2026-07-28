#!/usr/bin/env bash
# Open-world-focused V9 benchmark runner for NCSA Delta.
#
# Upload this file plus delta_v9_benchmark_all_in_one.sbatch and the runtime
# bundle to the same directory, then submit this wrapper with sbatch.

#SBATCH --job-name=lorat-v9-ow
#SBATCH --account=CHANGE_ME_DELTA_ACCOUNT
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --constraint="work"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --output=lorat-v9-open-world-%j.out

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Refusing to run outside a Slurm allocation. Submit this file with sbatch."
    exit 2
fi

DELTA_ROOT="${DELTA_ROOT:-/u/lwalkaraj/lorat-mot}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALL_IN_ONE_SCRIPT="${ALL_IN_ONE_SCRIPT:-$SCRIPT_DIR/delta_v9_benchmark_all_in_one.sbatch}"

if [[ ! -f "$ALL_IN_ONE_SCRIPT" ]]; then
    echo "Missing all-in-one Delta benchmark script: $ALL_IN_ONE_SCRIPT"
    echo "Upload delta_v9_benchmark_all_in_one.sbatch next to this file."
    exit 1
fi

export WORK_ROOT="${WORK_ROOT:-$DELTA_ROOT}"
export BUNDLE="${BUNDLE:-$DELTA_ROOT/lorat_v9_delta_benchmark_bundle.tar.gz}"
export DATA_ROOT="${DATA_ROOT:-$DELTA_ROOT/lorat-data/raw}"
export HEAD_WEIGHTS_ROOT="${HEAD_WEIGHTS_ROOT:-$DELTA_ROOT/lorat-models/v9-heads/current/checkpoints}"
export HEAD_CHECKPOINT="${HEAD_CHECKPOINT:-best_by_rollout_identity}"
export RESULT_ROOT="${RESULT_ROOT:-$DELTA_ROOT/lorat-v9-open-world-results-$SLURM_JOB_ID}"
export COMPARE_CONFIGS="${COMPARE_CONFIGS:-B-224,g-378}"
export TRACK_COUNTS="${TRACK_COUNTS:-1,2,3,4,5}"
export MAX_FRAMES="${MAX_FRAMES:-300}"
export SOURCE_VIDEOS_PER_DATASET="${SOURCE_VIDEOS_PER_DATASET:-3}"
export DIAGNOSTIC_MODES="${DIAGNOSTIC_MODES:-normal}"
export DRAW_CANDIDATE_DIAGNOSTICS="${DRAW_CANDIDATE_DIAGNOSTICS:-1}"
export REID_ABLATION="${REID_ABLATION:-1}"
export FULL_AREA_OBSERVATIONS="${FULL_AREA_OBSERVATIONS:-1}"

# This wrapper is intentionally open-world focused. These nonexistent roots
# make the all-in-one runner skip pedestrian-only baselines for this job.
export DANCETRACK_ROOT="${DANCETRACK_ROOT:-$DATA_ROOT/__skip_for_open_world/DanceTrack}"
export COMBINED_ROOT="${COMBINED_ROOT:-$DATA_ROOT/__skip_for_open_world/Combined}"
export MOT17_ROOT="${MOT17_ROOT:-$DATA_ROOT/__skip_for_open_world/MOT17}"

export BDD100K_ROOT="${BDD100K_ROOT:-$DATA_ROOT/BDD100K}"
export BDD100K_SPLIT="${BDD100K_SPLIT:-val}"
export TAO_ROOT="${TAO_ROOT:-$DATA_ROOT/TAO_OW_SUBSET}"
export TAO_SPLIT="${TAO_SPLIT:-validation}"
export LASOT_ROOT="${LASOT_ROOT:-$DATA_ROOT/LaSOT_subset}"
export LASOT_SPLIT="${LASOT_SPLIT:-val}"

echo "Delta V9 open-world benchmark wrapper"
echo "Delta root: $DELTA_ROOT"
if [[ "$DELTA_ROOT" == /u/* ]]; then
    echo "WARNING: DELTA_ROOT is under /u. Delta docs recommend /work/hdd/<project>/<user> for datasets, job I/O, and results."
fi
echo "All-in-one script: $ALL_IN_ONE_SCRIPT"
echo "Data root: $DATA_ROOT"
echo "Head weights root: $HEAD_WEIGHTS_ROOT"
echo "Result root: $RESULT_ROOT"
echo "Source order inside all-in-one: BDD100K if present, then TAO-OW and LaSOT; DanceTrack/MOT17 skipped."

bash "$ALL_IN_ONE_SCRIPT"
