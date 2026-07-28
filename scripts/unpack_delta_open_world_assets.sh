#!/usr/bin/env bash
# Organize uploaded V9 open-world benchmark assets on NCSA Delta.
#
# Best practice on Delta: submit scripts/delta_unpack_open_world_assets.sbatch
# instead of running this directly on a login node. Direct shell use is only
# appropriate for very small test packs.

set -euo pipefail

DELTA_ROOT="${DELTA_ROOT:-$(pwd)}"
UPLOAD_DIR="${UPLOAD_DIR:-$(pwd)}"
DATA_ROOT="${DATA_ROOT:-$DELTA_ROOT/lorat-data/raw}"
HEAD_ROOT="${HEAD_ROOT:-$DELTA_ROOT/lorat-models/v9-heads/current}"
CLEAN_ARCHIVES_AFTER_UNPACK="${CLEAN_ARCHIVES_AFTER_UNPACK:-0}"

mkdir -p "$DELTA_ROOT" "$DATA_ROOT" "$HEAD_ROOT" "$DELTA_ROOT/lorat-model-zips" "$DELTA_ROOT/datasets"

copy_if_present() {
    local src="$1"
    local dst="$2"
    if [[ -f "$src" && "$src" != "$dst" ]]; then
        mkdir -p "$(dirname "$dst")"
        cp -n "$src" "$dst" || true
    fi
}

extract_tar_to_data_root() {
    local archive="$1"
    if [[ -f "$archive" ]]; then
        echo "Extracting dataset archive to $DATA_ROOT: $archive"
        tar -xzf "$archive" -C "$DATA_ROOT"
    fi
}

extract_zip_to_bdd_root() {
    local archive="$1"
    if [[ -f "$archive" ]]; then
        echo "Extracting BDD100K archive to $DATA_ROOT/BDD100K: $archive"
        mkdir -p "$DATA_ROOT/BDD100K"
        unzip -q -o "$archive" -d "$DATA_ROOT/BDD100K"
    fi
}

echo "Delta open-world asset unpack"
echo "Upload dir: $UPLOAD_DIR"
echo "Delta root: $DELTA_ROOT"
if [[ "$DELTA_ROOT" == /u/* ]]; then
    echo "WARNING: DELTA_ROOT is under /u. Delta docs recommend /work/hdd/<project>/<user> for datasets and job I/O."
fi
echo "Data root: $DATA_ROOT"
echo "Head root: $HEAD_ROOT"
echo "Clean archives after unpack: $CLEAN_ARCHIVES_AFTER_UNPACK"

copy_if_present "$UPLOAD_DIR/lorat_v9_delta_benchmark_bundle.tar.gz" "$DELTA_ROOT/lorat_v9_delta_benchmark_bundle.tar.gz"
copy_if_present "$UPLOAD_DIR/delta_v9_benchmark_all_in_one.sbatch" "$DELTA_ROOT/delta_v9_benchmark_all_in_one.sbatch"
copy_if_present "$UPLOAD_DIR/delta_v9_open_world_benchmark.sbatch" "$DELTA_ROOT/delta_v9_open_world_benchmark.sbatch"

if compgen -G "$UPLOAD_DIR/lorat-model-zips/*.zip" >/dev/null; then
    cp -n "$UPLOAD_DIR"/lorat-model-zips/*.zip "$DELTA_ROOT/lorat-model-zips/" || true
fi
if compgen -G "$DELTA_ROOT/lorat-model-zips/*.zip" >/dev/null; then
    echo "Extracting V9 head checkpoint zips to $HEAD_ROOT"
    for archive in "$DELTA_ROOT"/lorat-model-zips/*.zip; do
        unzip -q -o "$archive" -d "$HEAD_ROOT"
    done
    if [[ "$CLEAN_ARCHIVES_AFTER_UNPACK" == "1" || "$CLEAN_ARCHIVES_AFTER_UNPACK" == "true" ]]; then
        rm -f "$DELTA_ROOT"/lorat-model-zips/*.zip
    fi
fi

if [[ -f "$UPLOAD_DIR/datasets/LaSOT_subset.tar.gz" ]]; then
    copy_if_present "$UPLOAD_DIR/datasets/LaSOT_subset.tar.gz" "$DELTA_ROOT/datasets/LaSOT_subset.tar.gz"
fi
if [[ -f "$UPLOAD_DIR/datasets/TAO_OW_SUBSET.tar.gz" ]]; then
    copy_if_present "$UPLOAD_DIR/datasets/TAO_OW_SUBSET.tar.gz" "$DELTA_ROOT/datasets/TAO_OW_SUBSET.tar.gz"
fi

extract_tar_to_data_root "$DELTA_ROOT/datasets/LaSOT_subset.tar.gz"
extract_tar_to_data_root "$DELTA_ROOT/datasets/TAO_OW_SUBSET.tar.gz"
if [[ "$CLEAN_ARCHIVES_AFTER_UNPACK" == "1" || "$CLEAN_ARCHIVES_AFTER_UNPACK" == "true" ]]; then
    rm -f "$DELTA_ROOT/datasets/LaSOT_subset.tar.gz" "$DELTA_ROOT/datasets/TAO_OW_SUBSET.tar.gz"
    rm -rf "$DELTA_ROOT/datasets/TAO_OW_SUBSET.tar.gz.parts"
fi

if compgen -G "$UPLOAD_DIR/bdd100k/*.zip" >/dev/null; then
    mkdir -p "$DELTA_ROOT/bdd100k"
    cp -n "$UPLOAD_DIR"/bdd100k/*.zip "$DELTA_ROOT/bdd100k/" || true
fi
if compgen -G "$DELTA_ROOT/bdd100k/*.zip" >/dev/null; then
    for archive in "$DELTA_ROOT"/bdd100k/*.zip; do
        extract_zip_to_bdd_root "$archive"
    done
    if [[ "$CLEAN_ARCHIVES_AFTER_UNPACK" == "1" || "$CLEAN_ARCHIVES_AFTER_UNPACK" == "true" ]]; then
        rm -f "$DELTA_ROOT"/bdd100k/*.zip
    fi
fi
if compgen -G "$UPLOAD_DIR/bdd100k/*.tar.gz" >/dev/null; then
    mkdir -p "$DELTA_ROOT/bdd100k"
    cp -n "$UPLOAD_DIR"/bdd100k/*.tar.gz "$DELTA_ROOT/bdd100k/" || true
fi
if compgen -G "$DELTA_ROOT/bdd100k/*.tar.gz" >/dev/null; then
    mkdir -p "$DATA_ROOT/BDD100K"
    for archive in "$DELTA_ROOT"/bdd100k/*.tar.gz; do
        echo "Extracting BDD100K archive to $DATA_ROOT/BDD100K: $archive"
        tar -xzf "$archive" -C "$DATA_ROOT/BDD100K"
    done
    if [[ "$CLEAN_ARCHIVES_AFTER_UNPACK" == "1" || "$CLEAN_ARCHIVES_AFTER_UNPACK" == "true" ]]; then
        rm -f "$DELTA_ROOT"/bdd100k/*.tar.gz
    fi
fi

echo "Final expected roots:"
printf '  bundle: '; ls -lh "$DELTA_ROOT/lorat_v9_delta_benchmark_bundle.tar.gz" 2>/dev/null || true
printf '  all-in-one sbatch: '; ls -lh "$DELTA_ROOT/delta_v9_benchmark_all_in_one.sbatch" 2>/dev/null || true
printf '  open-world sbatch: '; ls -lh "$DELTA_ROOT/delta_v9_open_world_benchmark.sbatch" 2>/dev/null || true
printf '  heads: '; find "$HEAD_ROOT" -maxdepth 3 -type f -name 'v9_local_head_*.pt' | head -10 || true
printf '  BDD100K: '; find "$DATA_ROOT/BDD100K" -maxdepth 3 -type d 2>/dev/null | head -10 || true
printf '  TAO-OW: '; find "$DATA_ROOT/TAO_OW_SUBSET" -maxdepth 2 -type d 2>/dev/null | head -10 || true
printf '  LaSOT: '; find "$DATA_ROOT/LaSOT_subset" -maxdepth 2 -type d 2>/dev/null | head -10 || true

echo ""
echo "Submit the open-world benchmark with your real Delta account, for example:"
echo "  cd $DELTA_ROOT"
echo "  sbatch --account=<your_delta_account> --partition=gpuA100x4 delta_v9_open_world_benchmark.sbatch"
