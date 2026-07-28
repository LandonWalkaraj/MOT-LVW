#!/usr/bin/env bash
#SBATCH --job-name=v9-src-video
#SBATCH --account=rc_general
#SBATCH --partition=gpu-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --signal=TERM@600
#SBATCH --gres=gpu:1
#SBATCH --output=/work/%u/lorat-v9-source-videos-%j.out

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Refusing to run outside a Slurm allocation. Submit this script with sbatch."
    exit 2
fi

REMOTE_WORK_ROOT="${REMOTE_WORK_ROOT:-/work/$USER}"
BASE_RESULT_ROOT="${BASE_RESULT_ROOT:-/work/$USER/lorat-v9-training-source-videos-$SLURM_JOB_ID}"
BENCHMARK_SCRIPT="${BENCHMARK_SCRIPT:-$REMOTE_WORK_ROOT/theia_v9_benchmark.sbatch}"

HEAD_WEIGHTS_ROOT="${HEAD_WEIGHTS_ROOT:-}"
HEAD_CHECKPOINT="${HEAD_CHECKPOINT:-best_by_rollout_identity}"
COMPARE_CONFIGS="${COMPARE_CONFIGS:-B-224}"
SOURCE_TRACK_COUNTS="${SOURCE_TRACK_COUNTS:-1,2,3,4,5}"
SOURCE_MAX_FRAMES="${SOURCE_MAX_FRAMES:-300}"
SOURCE_DIAGNOSTIC_MODES="${SOURCE_DIAGNOSTIC_MODES:-normal,gt_window,gt_identity}"
DRAW_CANDIDATE_DIAGNOSTICS="${DRAW_CANDIDATE_DIAGNOSTICS:-1}"
SOURCE_VIDEOS_PER_DATASET="${SOURCE_VIDEOS_PER_DATASET:-3}"

DANCETRACK_SEQUENCE="${DANCETRACK_SEQUENCE:-auto}"
DANCETRACK_SPLIT="${DANCETRACK_SPLIT:-val}"
DANCETRACK_DATASET_ROOT="${DANCETRACK_DATASET_ROOT:-/work/$USER/lorat-data/raw/DanceTrack}"

COMBINED_ROOT="${COMBINED_ROOT:-/work/$USER/lorat-data/raw/Combined}"
COMBINED_SPLIT="${COMBINED_SPLIT:-val}"
COMBINED_SEQUENCE="${COMBINED_SEQUENCE:-auto}"

MOT17_ROOT="${MOT17_ROOT:-/work/$USER/lorat-data/raw/MOT17}"
MOT17_SPLIT="${MOT17_SPLIT:-train}"
MOT17_SEQUENCE="${MOT17_SEQUENCE:-auto}"

TAO_ROOT="${TAO_ROOT:-/work/$USER/lorat-data/raw/TAO_OW_SUBSET}"
TAO_SPLIT="${TAO_SPLIT:-validation}"
TAO_USE_FREEFORM="${TAO_USE_FREEFORM:-1}"
TAO_EXAMPLE_VIDEOS="${TAO_EXAMPLE_VIDEOS:-$SOURCE_VIDEOS_PER_DATASET}"
TAO_EXAMPLE_MIN_TRACKS="${TAO_EXAMPLE_MIN_TRACKS:-5}"

LASOT_ROOT="${LASOT_ROOT:-/work/$USER/lorat-data/raw/LaSOT_subset}"
LASOT_SPLIT="${LASOT_SPLIT:-val}"
LASOT_EXAMPLE_VIDEOS="${LASOT_EXAMPLE_VIDEOS:-$SOURCE_VIDEOS_PER_DATASET}"
BDD100K_ROOT="${BDD100K_ROOT:-/work/$USER/lorat-data/raw/BDD100K}"
BDD100K_SPLIT="${BDD100K_SPLIT:-val}"
BDD100K_EXAMPLE_VIDEOS="${BDD100K_EXAMPLE_VIDEOS:-$SOURCE_VIDEOS_PER_DATASET}"
BDD100K_EXAMPLE_MIN_TRACKS="${BDD100K_EXAMPLE_MIN_TRACKS:-5}"
BDD100K_EXAMPLE_MIN_ANNOTATED_FRAMES="${BDD100K_EXAMPLE_MIN_ANNOTATED_FRAMES:-10}"
BDD100K_CATEGORIES="${BDD100K_CATEGORIES:-pedestrian,rider,car,truck,bus,train,motorcycle,bicycle}"

SOURCE_STATUS_FILE="$BASE_RESULT_ROOT/source_status.tsv"
SOURCE_FAILURES=0

record_source_status() {
    local label="$1"
    local status="$2"
    local exit_code="$3"
    local detail="$4"
    local result_root="$5"
    printf '%s\t%s\t%s\t%s\t%s\n' "$label" "$status" "$exit_code" "$detail" "$result_root" >> "$SOURCE_STATUS_FILE"
}

run_source_case() {
    local label="$1"
    shift
    local result_root="$BASE_RESULT_ROOT/$label"
    echo "============================================================"
    echo "Running V9 training-source video benchmark: $label"
    echo "Result root: $result_root"
    echo "============================================================"
    set +e
    env \
    DATASET_ROOT= \
    TAO_ROOT= \
    LASOT_ROOT= \
    BDD100K_ROOT= \
    RESULT_ROOT="$result_root" \
    HEAD_WEIGHTS_ROOT="$HEAD_WEIGHTS_ROOT" \
    HEAD_CHECKPOINT="$HEAD_CHECKPOINT" \
    COMPARE_CONFIGS="$COMPARE_CONFIGS" \
    TRACK_COUNTS="$SOURCE_TRACK_COUNTS" \
    MAX_FRAMES="$SOURCE_MAX_FRAMES" \
    SAVE_VIDEO=1 \
    V9_DIAGNOSTIC_MODES="$SOURCE_DIAGNOSTIC_MODES" \
    DRAW_CANDIDATE_DIAGNOSTICS="$DRAW_CANDIDATE_DIAGNOSTICS" \
    CONTROLLED_OCCLUSION_DURATIONS=none \
    REID_ABLATION=1 \
    FULL_AREA_OBSERVATIONS=1 \
    "$@" \
    bash "$BENCHMARK_SCRIPT"
    local status=$?
    set -e
    if [[ "$status" -eq 0 ]]; then
        echo "Completed V9 training-source video benchmark: $label"
        record_source_status "$label" "ok" "$status" "completed" "$result_root"
    else
        echo "FAILED V9 training-source video benchmark: $label exit_code=$status"
        record_source_status "$label" "failed" "$status" "benchmark_error" "$result_root"
        SOURCE_FAILURES=$((SOURCE_FAILURES + 1))
    fi
    return 0
}

find_mot_sequences() {
    local root="$1"
    local split="$2"
    local limit="$3"
    local split_dir="$root/$split"
    local path
    if [[ ! -d "$split_dir" ]]; then
        return 0
    fi
    local count=0
    for path in "$split_dir"/*; do
        [[ -d "$path" ]] || continue
        if [[ -d "$path/img1" && -f "$path/gt/gt.txt" ]]; then
            basename "$path"
            count=$((count + 1))
            if [[ "$limit" -gt 0 && "$count" -ge "$limit" ]]; then
                return 0
            fi
        fi
    done
}

run_optional_mot_case() {
    local label="$1"
    local root="$2"
    local split="$3"
    local sequence="$4"
    if [[ ! -d "$root" ]]; then
        echo "Skipping $label source video benchmark; missing root=$root"
        record_source_status "$label" "skipped" "0" "missing_root:$root" "$BASE_RESULT_ROOT/$label"
        return 0
    fi
    if [[ "$sequence" == "auto" || -z "$sequence" ]]; then
        local auto_sequences=()
        mapfile -t auto_sequences < <(find_mot_sequences "$root" "$split" "$SOURCE_VIDEOS_PER_DATASET")
        sequence="$(IFS=,; echo "${auto_sequences[*]}")"
    fi
    if [[ -z "$sequence" ]]; then
        echo "Skipping $label source video benchmark; no MOT-style sequence found under $root/$split"
        record_source_status "$label" "skipped" "0" "no_mot_sequence:$root/$split" "$BASE_RESULT_ROOT/$label"
        return 0
    fi
    echo "$label selected MOT-style sequence(s): $sequence"
    run_source_case "$label" \
        DATASET_ROOT="$root" \
        SPLIT="$split" \
        SEQUENCES="$sequence"
}

if [[ ! -f "$BENCHMARK_SCRIPT" ]]; then
    echo "Missing benchmark script: $BENCHMARK_SCRIPT"
    exit 1
fi

mkdir -p "$BASE_RESULT_ROOT"
printf 'source\tstatus\texit_code\tdetail\tresult_root\n' > "$SOURCE_STATUS_FILE"
echo "Source benchmark order: dancetrack -> combined_mot -> mot17 -> bdd100k -> tao_ow -> lasot"
echo "Source videos per dataset target: $SOURCE_VIDEOS_PER_DATASET"
echo "Source track counts: $SOURCE_TRACK_COUNTS"

run_optional_mot_case "dancetrack" "$DANCETRACK_DATASET_ROOT" "$DANCETRACK_SPLIT" "$DANCETRACK_SEQUENCE"

run_optional_mot_case "combined_mot" "$COMBINED_ROOT" "$COMBINED_SPLIT" "$COMBINED_SEQUENCE"
run_optional_mot_case "mot17" "$MOT17_ROOT" "$MOT17_SPLIT" "$MOT17_SEQUENCE"

if [[ -d "$BDD100K_ROOT" ]]; then
    run_source_case "bdd100k" \
        BDD100K_ROOT="$BDD100K_ROOT" \
        BDD100K_SPLIT="$BDD100K_SPLIT" \
        BDD100K_EXAMPLE_VIDEOS="$BDD100K_EXAMPLE_VIDEOS" \
        BDD100K_EXAMPLE_MIN_TRACKS="$BDD100K_EXAMPLE_MIN_TRACKS" \
        BDD100K_EXAMPLE_MIN_ANNOTATED_FRAMES="$BDD100K_EXAMPLE_MIN_ANNOTATED_FRAMES" \
        BDD100K_EXAMPLE_MAX_FRAMES="$SOURCE_MAX_FRAMES" \
        BDD100K_CATEGORIES="$BDD100K_CATEGORIES"
else
    echo "Skipping BDD100K source video benchmark; missing BDD100K_ROOT=$BDD100K_ROOT"
    record_source_status "bdd100k" "skipped" "0" "missing_root:$BDD100K_ROOT" "$BASE_RESULT_ROOT/bdd100k"
fi

if [[ -d "$TAO_ROOT" ]]; then
    run_source_case "tao_ow" \
        TAO_ROOT="$TAO_ROOT" \
        TAO_SPLIT="$TAO_SPLIT" \
        TAO_USE_FREEFORM="$TAO_USE_FREEFORM" \
        TAO_EXAMPLE_VIDEOS="$TAO_EXAMPLE_VIDEOS" \
        TAO_EXAMPLE_MIN_TRACKS="$TAO_EXAMPLE_MIN_TRACKS"
else
    echo "Skipping TAO source video benchmark; missing TAO_ROOT=$TAO_ROOT"
    record_source_status "tao_ow" "skipped" "0" "missing_root:$TAO_ROOT" "$BASE_RESULT_ROOT/tao_ow"
fi

if [[ -d "$LASOT_ROOT" ]]; then
    run_source_case "lasot" \
        LASOT_ROOT="$LASOT_ROOT" \
        LASOT_SPLIT="$LASOT_SPLIT" \
        LASOT_EXAMPLE_VIDEOS="$LASOT_EXAMPLE_VIDEOS"
else
    echo "Skipping LaSOT source video benchmark; missing LASOT_ROOT=$LASOT_ROOT"
    record_source_status "lasot" "skipped" "0" "missing_root:$LASOT_ROOT" "$BASE_RESULT_ROOT/lasot"
fi

echo "Training-source video benchmark complete."
echo "Source status table:"
cat "$SOURCE_STATUS_FILE"
find "$BASE_RESULT_ROOT" -maxdepth 8 -type f \( -name "summary.md" -o -name "*.mp4" \) -print

if [[ "$SOURCE_FAILURES" -gt 0 ]]; then
    echo "One or more source benchmarks failed. See $SOURCE_STATUS_FILE for per-source status."
    exit 1
fi
