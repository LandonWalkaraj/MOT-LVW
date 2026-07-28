#!/usr/bin/env bash
# Slurm-managed unpack step for NCSA Delta open-world benchmark assets.
#
# Run this after uploading the prepared asset pack. Keep this as a separate
# compute-node job so large archive extraction does not run on Delta login nodes.

#SBATCH --job-name=lorat-unpack
#SBATCH --account=CHANGE_ME_DELTA_ACCOUNT
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --constraint="work"
#SBATCH --output=lorat-delta-unpack-%j.out

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Refusing to run outside a Slurm allocation. Submit this file with sbatch."
    exit 2
fi

DELTA_ROOT="${DELTA_ROOT:-$PWD}"
CLEAN_ARCHIVES_AFTER_UNPACK="${CLEAN_ARCHIVES_AFTER_UNPACK:-1}"

cd "$DELTA_ROOT"

echo "Delta V9 asset unpack"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Delta root: $DELTA_ROOT"
echo "Clean archives after unpack: $CLEAN_ARCHIVES_AFTER_UNPACK"
if [[ "$DELTA_ROOT" == /u/* ]]; then
    echo "WARNING: unpacking under /u. Delta docs recommend /work/hdd/<project>/<user> for datasets and job I/O."
fi

if command -v module >/dev/null 2>&1; then
    module reset || true
    module list || true
fi

CLEAN_ARCHIVES_AFTER_UNPACK="$CLEAN_ARCHIVES_AFTER_UNPACK" \
DELTA_ROOT="$DELTA_ROOT" \
bash unpack_delta_open_world_assets.sh

echo "Unpack complete. Largest files under DELTA_ROOT:"
du -sh "$DELTA_ROOT" 2>/dev/null || true
find "$DELTA_ROOT" -maxdepth 4 -type f -printf '%s %p\n' 2>/dev/null | sort -nr | head -20 || true
