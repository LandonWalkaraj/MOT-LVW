#!/usr/bin/env bash
# V9 all-dataset benchmark runner for NCSA Delta.
#
# Intended Open OnDemand use:
#   1. Upload this file and lorat_v9_delta_benchmark_bundle.tar.gz to your Delta
#      work/scratch directory.
#   2. Edit the account line below or submit with:
#        sbatch --account=<your_delta_account> scripts/delta_v9_benchmark_all_in_one.sbatch
#   3. Keep datasets under DATA_ROOT and V9 heads under HEAD_WEIGHTS_ROOT.
#
# Delta account and partition names depend on your allocation. The common GPU
# partitions include gpuA100x4, gpuA100x8, gpuA40x4, gpuMI100x8, and newer H200
# partitions on some allocations.

#SBATCH --job-name=lorat-v9-delta-bench
#SBATCH --account=CHANGE_ME_DELTA_ACCOUNT
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --constraint="work"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --output=lorat-v9-delta-benchmark-%j.out

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Refusing to run outside a Slurm allocation. Submit this file with sbatch."
    exit 2
fi

WORK_ROOT="${WORK_ROOT:-${SCRATCH:-$PWD}/lorat-mot}"
BUNDLE="${BUNDLE:-$WORK_ROOT/lorat_v9_delta_benchmark_bundle.tar.gz}"
RUN_ROOT="${RUN_ROOT:-$WORK_ROOT/lorat-v9-delta-run-$SLURM_JOB_ID}"
RESULT_ROOT="${RESULT_ROOT:-$WORK_ROOT/lorat-v9-delta-results-$SLURM_JOB_ID}"
DATA_ROOT="${DATA_ROOT:-$WORK_ROOT/lorat-data/raw}"
HEAD_WEIGHTS_ROOT="${HEAD_WEIGHTS_ROOT:-$WORK_ROOT/lorat-models/v9-heads}"
HEAD_CHECKPOINT="${HEAD_CHECKPOINT:-best_by_rollout_identity}"
COMPARE_CONFIGS="${COMPARE_CONFIGS:-B-224,g-378}"
TRACK_COUNTS="${TRACK_COUNTS:-1,2,3,4,5}"
MAX_FRAMES="${MAX_FRAMES:-300}"
SOURCE_VIDEOS_PER_DATASET="${SOURCE_VIDEOS_PER_DATASET:-3}"
DIAGNOSTIC_MODES="${DIAGNOSTIC_MODES:-normal}"
DRAW_CANDIDATE_DIAGNOSTICS="${DRAW_CANDIDATE_DIAGNOSTICS:-1}"
REID_ABLATION="${REID_ABLATION:-1}"
FULL_AREA_OBSERVATIONS="${FULL_AREA_OBSERVATIONS:-1}"
V9_LOCAL_GRID_SIZE="${V9_LOCAL_GRID_SIZE:-17}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
VENV_DIR="${VENV_DIR:-$WORK_ROOT/lorat-v9-delta-venv-cu128}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-$WORK_ROOT/.cache/pip}"
EXPECT_GPU_NAME="${EXPECT_GPU_NAME:-}"

DANCETRACK_ROOT="${DANCETRACK_ROOT:-$DATA_ROOT/DanceTrack}"
DANCETRACK_SPLIT="${DANCETRACK_SPLIT:-val}"
COMBINED_ROOT="${COMBINED_ROOT:-$DATA_ROOT/Combined}"
COMBINED_SPLIT="${COMBINED_SPLIT:-val}"
MOT17_ROOT="${MOT17_ROOT:-$DATA_ROOT/MOT17}"
MOT17_SPLIT="${MOT17_SPLIT:-train}"
BDD100K_ROOT="${BDD100K_ROOT:-$DATA_ROOT/BDD100K}"
BDD100K_SPLIT="${BDD100K_SPLIT:-val}"
TAO_ROOT="${TAO_ROOT:-$DATA_ROOT/TAO_OW_SUBSET}"
TAO_SPLIT="${TAO_SPLIT:-validation}"
LASOT_ROOT="${LASOT_ROOT:-$DATA_ROOT/LaSOT_subset}"
LASOT_SPLIT="${LASOT_SPLIT:-val}"

export PIP_CACHE_DIR

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Work root: $WORK_ROOT"
echo "Bundle: $BUNDLE"
echo "Run root: $RUN_ROOT"
echo "Result root: $RESULT_ROOT"
echo "Data root: $DATA_ROOT"
echo "Head weights root: $HEAD_WEIGHTS_ROOT"
echo "Compare configs: $COMPARE_CONFIGS"
echo "Track counts: $TRACK_COUNTS"
echo "Max frames: $MAX_FRAMES"
echo "Datasets: DanceTrack, Combined, MOT17, BDD100K, TAO-OW, LaSOT"

if [[ ! -f "$BUNDLE" ]]; then
    echo "Missing runtime bundle: $BUNDLE"
    exit 1
fi

mkdir -p "$RUN_ROOT" "$RESULT_ROOT" "$PIP_CACHE_DIR"
tar -xzf "$BUNDLE" -C "$RUN_ROOT"

PROJECT_DIR=""
if [[ -d "$RUN_ROOT/mot-v9-delta" ]]; then
    PROJECT_DIR="$RUN_ROOT/mot-v9-delta"
elif [[ -d "$RUN_ROOT/mot-v8-train" ]]; then
    PROJECT_DIR="$RUN_ROOT/mot-v8-train"
else
    PROJECT_DIR="$(find "$RUN_ROOT" -maxdepth 2 -type f -path '*/programs/benchmark_lorat_v9.py' -printf '%h\n' | sed 's#/programs$##' | head -1)"
fi
if [[ -z "$PROJECT_DIR" || ! -f "$PROJECT_DIR/programs/benchmark_lorat_v9.py" ]]; then
    echo "Could not locate benchmark_lorat_v9.py after extracting bundle."
    find "$RUN_ROOT" -maxdepth 3 -type f | head -80
    exit 1
fi
cd "$PROJECT_DIR"

echo "Benchmark project directory: $PROJECT_DIR"
echo "V9 source hashes:"
sha256sum \
    programs/benchmark_lorat_v9.py \
    programs/benchmark_lorat_v9_harness.py \
    programs/bounding_box_v9_lorat_local_search.py \
    programs/bounding_box_v9_runtime_base.py \
    programs/export_bdd100k_to_mot_sequences.py \
    programs/export_lasot_to_mot_sequences.py \
    programs/export_tao_to_mot_sequences.py \
    2>/dev/null || true

if command -v module >/dev/null 2>&1; then
    module reset || true
    module load python/3.10 2>/dev/null || module load python 2>/dev/null || true
fi

python --version
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    python -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel
python -m pip install "setuptools<82"
python -m pip install torch torchvision --index-url "$PYTORCH_INDEX_URL"
python -m pip install -r external/LoRAT-main/requirements.txt
python -m pip install -r requirements.txt

python - <<PY
import torch

print(f"torch version: {torch.__version__}")
print(f"torch cuda version: {torch.version.cuda}")
print(f"torch cuda available: {torch.cuda.is_available()}")
print(f"torch cuda device count: {torch.cuda.device_count()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available inside this Slurm job.")
name = torch.cuda.get_device_name(0)
print(f"torch cuda device 0: {name}")
expected = "${EXPECT_GPU_NAME}"
if expected and expected.lower() not in name.lower():
    raise SystemExit(f"Expected GPU name containing {expected!r}, got {name!r}")
PY

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
fi

COMPARE_CONFIG_ARRAY=()
IFS=',' read -r -a _COMPARE_CONFIG_TOKENS <<< "$COMPARE_CONFIGS"
for item in "${_COMPARE_CONFIG_TOKENS[@]}"; do
    item="${item//[[:space:]]/}"
    if [[ -n "$item" ]]; then
        COMPARE_CONFIG_ARRAY+=("$item")
    fi
done
if [[ "${#COMPARE_CONFIG_ARRAY[@]}" -eq 0 ]]; then
    COMPARE_CONFIG_ARRAY=(B-224)
fi

DIAGNOSTIC_MODE_ARRAY=()
IFS=',' read -r -a _DIAGNOSTIC_MODE_TOKENS <<< "$DIAGNOSTIC_MODES"
for item in "${_DIAGNOSTIC_MODE_TOKENS[@]}"; do
    item="${item//[[:space:]]/}"
    if [[ -n "$item" ]]; then
        DIAGNOSTIC_MODE_ARRAY+=("$item")
    fi
done
if [[ "${#DIAGNOSTIC_MODE_ARRAY[@]}" -eq 0 ]]; then
    DIAGNOSTIC_MODE_ARRAY=(normal)
fi

echo "Resolved compare config array: ${COMPARE_CONFIG_ARRAY[*]}"
echo "Resolved diagnostic mode array: ${DIAGNOSTIC_MODE_ARRAY[*]}"

SOURCE_STATUS_FILE="$RESULT_ROOT/source_status.tsv"
printf 'source\tstatus\texit_code\tdetail\tresult_root\n' > "$SOURCE_STATUS_FILE"
SOURCE_FAILURES=0

record_source_status() {
    local label="$1"
    local status="$2"
    local exit_code="$3"
    local detail="$4"
    local result_root="$5"
    printf '%s\t%s\t%s\t%s\t%s\n' "$label" "$status" "$exit_code" "$detail" "$result_root" >> "$SOURCE_STATUS_FILE"
}

find_mot_sequences() {
    local root="$1"
    local split="$2"
    local limit="$3"
    local split_dir="$root/$split"
    local count=0
    local path
    [[ -d "$split_dir" ]] || return 0
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

run_python_benchmark() {
    local label="$1"
    shift
    local case_result_root="$RESULT_ROOT/$label"
    mkdir -p "$case_result_root"
    echo "============================================================"
    echo "Running Delta V9 benchmark source: $label"
    echo "Result root: $case_result_root"
    echo "============================================================"
    local extra_args=()
    if [[ "$DRAW_CANDIDATE_DIAGNOSTICS" == "1" || "$DRAW_CANDIDATE_DIAGNOSTICS" == "true" ]]; then
        extra_args+=(--draw-candidate-diagnostics)
    fi
    if [[ "$REID_ABLATION" == "1" || "$REID_ABLATION" == "true" ]]; then
        extra_args+=(--reid-ablation)
    fi
    if [[ "$FULL_AREA_OBSERVATIONS" == "1" || "$FULL_AREA_OBSERVATIONS" == "true" ]]; then
        extra_args+=(--full-area-observations)
    fi
    local source_failed=0
    local diagnostic_mode
    for diagnostic_mode in "${DIAGNOSTIC_MODE_ARRAY[@]}"; do
        local mode_result_root="$case_result_root"
        if [[ "${#DIAGNOSTIC_MODE_ARRAY[@]}" -gt 1 ]]; then
            mode_result_root="$case_result_root/$diagnostic_mode"
        fi
        mkdir -p "$mode_result_root"
        echo "Running $label diagnostic_mode=$diagnostic_mode"
        set +e
        python programs/benchmark_lorat_v9.py \
            "$@" \
            --lorat-root "$PROJECT_DIR/external/LoRAT-main" \
            --device cuda:0 \
            --gpu-profile hpc \
            --track-counts "$TRACK_COUNTS" \
            --compare-configs "${COMPARE_CONFIG_ARRAY[@]}" \
            --v9-head-weights-root "$HEAD_WEIGHTS_ROOT" \
            --v9-head-checkpoint "$HEAD_CHECKPOINT" \
            --v9-local-grid-size "$V9_LOCAL_GRID_SIZE" \
            --v9-diagnostic-mode "$diagnostic_mode" \
            --max-frames "$MAX_FRAMES" \
            --progress-interval 25 \
            --area-sample-interval 10 \
            --identity-sample-interval 1 \
            --controlled-occlusion-durations "" \
            --save-video \
            "${extra_args[@]}" \
            --output-root "$mode_result_root"
        local status=$?
        set -e
        if [[ "$status" -ne 0 ]]; then
            record_source_status "$label/$diagnostic_mode" "failed" "$status" "benchmark_error" "$mode_result_root"
            source_failed=1
        else
            record_source_status "$label/$diagnostic_mode" "ok" "$status" "completed" "$mode_result_root"
        fi
    done
    if [[ "$source_failed" -ne 0 ]]; then
        SOURCE_FAILURES=$((SOURCE_FAILURES + 1))
    fi
}

run_optional_mot_case() {
    local label="$1"
    local root="$2"
    local split="$3"
    if [[ ! -d "$root" ]]; then
        echo "Skipping $label; missing root=$root"
        record_source_status "$label" "skipped" "0" "missing_root:$root" "$RESULT_ROOT/$label"
        return 0
    fi
    local sequences=()
    mapfile -t sequences < <(find_mot_sequences "$root" "$split" "$SOURCE_VIDEOS_PER_DATASET")
    if [[ "${#sequences[@]}" -eq 0 ]]; then
        echo "Skipping $label; no MOT-style sequences under $root/$split"
        record_source_status "$label" "skipped" "0" "no_mot_sequences:$root/$split" "$RESULT_ROOT/$label"
        return 0
    fi
    local sequence_args=(--dataset-root "$root" --split "$split")
    local sequence
    for sequence in "${sequences[@]}"; do
        sequence_args+=(--sequence "$sequence")
    done
    echo "$label selected sequence(s): ${sequences[*]}"
    run_python_benchmark "$label" "${sequence_args[@]}"
}

run_optional_mot_case "dancetrack" "$DANCETRACK_ROOT" "$DANCETRACK_SPLIT"
run_optional_mot_case "combined_mot" "$COMBINED_ROOT" "$COMBINED_SPLIT"
run_optional_mot_case "mot17" "$MOT17_ROOT" "$MOT17_SPLIT"

if [[ -d "$BDD100K_ROOT" ]]; then
    run_python_benchmark "bdd100k" \
        --bdd100k-root "$BDD100K_ROOT" \
        --bdd100k-split "$BDD100K_SPLIT" \
        --bdd100k-example-root "$RESULT_ROOT/_bdd100k_mot_examples" \
        --bdd100k-example-videos "$SOURCE_VIDEOS_PER_DATASET" \
        --bdd100k-example-min-tracks 5 \
        --bdd100k-example-min-annotated-frames 10 \
        --bdd100k-example-max-frames "$MAX_FRAMES" \
        --bdd100k-example-copy-mode copy
else
    echo "Skipping bdd100k; missing root=$BDD100K_ROOT"
    record_source_status "bdd100k" "skipped" "0" "missing_root:$BDD100K_ROOT" "$RESULT_ROOT/bdd100k"
fi

if [[ -d "$TAO_ROOT" ]]; then
    run_python_benchmark "tao_ow" \
        --tao-root "$TAO_ROOT" \
        --tao-split "$TAO_SPLIT" \
        --tao-example-root "$RESULT_ROOT/_tao_mot_examples" \
        --tao-example-videos "$SOURCE_VIDEOS_PER_DATASET" \
        --tao-example-min-tracks 5 \
        --tao-example-min-annotated-frames 10 \
        --tao-example-max-frames "$MAX_FRAMES" \
        --tao-example-copy-mode copy \
        --tao-use-freeform
else
    echo "Skipping tao_ow; missing root=$TAO_ROOT"
    record_source_status "tao_ow" "skipped" "0" "missing_root:$TAO_ROOT" "$RESULT_ROOT/tao_ow"
fi

if [[ -d "$LASOT_ROOT" ]]; then
    run_python_benchmark "lasot" \
        --lasot-root "$LASOT_ROOT" \
        --lasot-split "$LASOT_SPLIT" \
        --lasot-example-root "$RESULT_ROOT/_lasot_mot_examples" \
        --lasot-example-videos "$SOURCE_VIDEOS_PER_DATASET" \
        --lasot-example-min-visible-frames 10 \
        --lasot-example-max-frames "$MAX_FRAMES" \
        --lasot-example-copy-mode copy
else
    echo "Skipping lasot; missing root=$LASOT_ROOT"
    record_source_status "lasot" "skipped" "0" "missing_root:$LASOT_ROOT" "$RESULT_ROOT/lasot"
fi

echo "Delta V9 benchmark complete."
echo "Source status:"
cat "$SOURCE_STATUS_FILE"
echo "Videos and summaries:"
find "$RESULT_ROOT" -maxdepth 8 -type f \( -name "summary.md" -o -name "*.mp4" -o -name "mot_paper_metrics_summary.csv" \) -print

if [[ "$SOURCE_FAILURES" -gt 0 ]]; then
    echo "One or more source benchmarks failed."
    exit 1
fi
