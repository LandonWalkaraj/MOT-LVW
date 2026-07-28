#!/usr/bin/env bash
#SBATCH --job-name=v9-b224-g378
#SBATCH --account=rc_general
#SBATCH --partition=gpu-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --signal=TERM@600
#SBATCH --gres=gpu:1
#SBATCH --output=/work/%u/lorat-v9-b224-g378-all-datasets-%j.out

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Refusing to run outside a Slurm allocation. Submit this script with sbatch."
    exit 2
fi

REMOTE_WORK_ROOT="${REMOTE_WORK_ROOT:-/work/$USER}"
SOURCE_BENCHMARK_SCRIPT="${SOURCE_BENCHMARK_SCRIPT:-$REMOTE_WORK_ROOT/theia_v9_training_sources_benchmark.sbatch}"
BASE_RESULT_ROOT="${BASE_RESULT_ROOT:-/work/$USER/lorat-v9-b224-g378-all-datasets-$SLURM_JOB_ID}"

B224_HEAD_WEIGHTS_ROOT="${B224_HEAD_WEIGHTS_ROOT:-/work/$USER/lorat-models/v9-heads/b224-704072/checkpoints}"
B224_HEAD_CHECKPOINT="${B224_HEAD_CHECKPOINT:-best_by_val_iou}"
G378_HEAD_WEIGHTS_ROOT="${G378_HEAD_WEIGHTS_ROOT:-/work/$USER/lorat-models/v9-heads/g378-706102/checkpoints}"
G378_HEAD_CHECKPOINT="${G378_HEAD_CHECKPOINT:-best_by_rollout_identity}"

SOURCE_TRACK_COUNTS="${SOURCE_TRACK_COUNTS:-1,2,3,4,5}"
SOURCE_MAX_FRAMES="${SOURCE_MAX_FRAMES:-300}"
SOURCE_DIAGNOSTIC_MODES="${SOURCE_DIAGNOSTIC_MODES:-normal,gt_window,gt_identity}"
DRAW_CANDIDATE_DIAGNOSTICS="${DRAW_CANDIDATE_DIAGNOSTICS:-1}"
SOURCE_VIDEOS_PER_DATASET="${SOURCE_VIDEOS_PER_DATASET:-3}"

DANCETRACK_DATASET_ROOT="${DANCETRACK_DATASET_ROOT:-/work/$USER/lorat-data/raw/DanceTrack}"
DANCETRACK_SEQUENCE="${DANCETRACK_SEQUENCE:-auto}"
DANCETRACK_SPLIT="${DANCETRACK_SPLIT:-val}"
COMBINED_ROOT="${COMBINED_ROOT:-/work/$USER/lorat-data/raw/Combined}"
COMBINED_SPLIT="${COMBINED_SPLIT:-val}"
COMBINED_SEQUENCE="${COMBINED_SEQUENCE:-auto}"
MOT17_ROOT="${MOT17_ROOT:-/work/$USER/lorat-data/raw/MOT17}"
MOT17_SPLIT="${MOT17_SPLIT:-train}"
MOT17_SEQUENCE="${MOT17_SEQUENCE:-auto}"
TAO_ROOT="${TAO_ROOT:-/work/$USER/lorat-data/raw/TAO_OW_SUBSET}"
LASOT_ROOT="${LASOT_ROOT:-/work/$USER/lorat-data/raw/LaSOT_subset}"
BDD100K_ROOT="${BDD100K_ROOT:-/work/$USER/lorat-data/raw/BDD100K}"

CONFIG_STATUS_FILE="$BASE_RESULT_ROOT/config_status.tsv"
CONFIG_FAILURES=0

run_config() {
    local label="$1"
    local compare_config="$2"
    local head_root="$3"
    local checkpoint="$4"
    local result_root="$BASE_RESULT_ROOT/$label"

    echo "============================================================"
    echo "Running V9 all-dataset benchmark config: $compare_config"
    echo "Config label: $label"
    echo "Head weights root: $head_root"
    echo "Head checkpoint: $checkpoint"
    echo "Result root: $result_root"
    echo "============================================================"

    set +e
    env \
        REMOTE_WORK_ROOT="$REMOTE_WORK_ROOT" \
        BASE_RESULT_ROOT="$result_root" \
        HEAD_WEIGHTS_ROOT="$head_root" \
        HEAD_CHECKPOINT="$checkpoint" \
        COMPARE_CONFIGS="$compare_config" \
        SOURCE_TRACK_COUNTS="$SOURCE_TRACK_COUNTS" \
        SOURCE_MAX_FRAMES="$SOURCE_MAX_FRAMES" \
        SOURCE_DIAGNOSTIC_MODES="$SOURCE_DIAGNOSTIC_MODES" \
        DRAW_CANDIDATE_DIAGNOSTICS="$DRAW_CANDIDATE_DIAGNOSTICS" \
        SOURCE_VIDEOS_PER_DATASET="$SOURCE_VIDEOS_PER_DATASET" \
        DANCETRACK_DATASET_ROOT="$DANCETRACK_DATASET_ROOT" \
        DANCETRACK_SEQUENCE="$DANCETRACK_SEQUENCE" \
        DANCETRACK_SPLIT="$DANCETRACK_SPLIT" \
        COMBINED_ROOT="$COMBINED_ROOT" \
        COMBINED_SPLIT="$COMBINED_SPLIT" \
        COMBINED_SEQUENCE="$COMBINED_SEQUENCE" \
        MOT17_ROOT="$MOT17_ROOT" \
        MOT17_SPLIT="$MOT17_SPLIT" \
        MOT17_SEQUENCE="$MOT17_SEQUENCE" \
        TAO_ROOT="$TAO_ROOT" \
        LASOT_ROOT="$LASOT_ROOT" \
        BDD100K_ROOT="$BDD100K_ROOT" \
        bash "$SOURCE_BENCHMARK_SCRIPT"
    local status=$?
    set -e

    if [[ "$status" -eq 0 ]]; then
        printf '%s\t%s\t%s\t%s\t%s\n' "$label" "$compare_config" "ok" "$status" "$result_root" >> "$CONFIG_STATUS_FILE"
    else
        printf '%s\t%s\t%s\t%s\t%s\n' "$label" "$compare_config" "failed" "$status" "$result_root" >> "$CONFIG_STATUS_FILE"
        CONFIG_FAILURES=$((CONFIG_FAILURES + 1))
    fi
}

if [[ ! -f "$SOURCE_BENCHMARK_SCRIPT" ]]; then
    echo "Missing source benchmark script: $SOURCE_BENCHMARK_SCRIPT"
    exit 1
fi

mkdir -p "$BASE_RESULT_ROOT"
printf 'label\tcompare_config\tstatus\texit_code\tresult_root\n' > "$CONFIG_STATUS_FILE"

echo "V9 B-224 + g-378 all-dataset benchmark single job"
echo "Slurm job id: $SLURM_JOB_ID"
echo "Base result root: $BASE_RESULT_ROOT"
echo "Output file: /work/$USER/lorat-v9-b224-g378-all-datasets-$SLURM_JOB_ID.out"
echo "Config order: B-224 -> g-378"
echo "Dataset roots:"
echo "  DanceTrack: $DANCETRACK_DATASET_ROOT"
echo "  Combined:   $COMBINED_ROOT"
echo "  MOT17:      $MOT17_ROOT"
echo "  BDD100K:    $BDD100K_ROOT"
echo "  TAO-OW:     $TAO_ROOT"
echo "  LaSOT:      $LASOT_ROOT"
echo "Source videos per dataset target: $SOURCE_VIDEOS_PER_DATASET"
echo "Source track counts: $SOURCE_TRACK_COUNTS"

run_config "B_224" "B-224" "$B224_HEAD_WEIGHTS_ROOT" "$B224_HEAD_CHECKPOINT"
run_config "g_378" "g-378" "$G378_HEAD_WEIGHTS_ROOT" "$G378_HEAD_CHECKPOINT"

echo "============================================================"
echo "Single-job B-224 + g-378 benchmark complete."
echo "Config status table:"
cat "$CONFIG_STATUS_FILE"
echo "Summary/video outputs:"
find "$BASE_RESULT_ROOT" -maxdepth 9 -type f \( -name "summary.md" -o -name "*.mp4" -o -name "source_status.tsv" \) -print

if [[ "$CONFIG_FAILURES" -gt 0 ]]; then
    echo "One or more configs failed. See $CONFIG_STATUS_FILE and per-config source_status.tsv files."
    exit 1
fi
