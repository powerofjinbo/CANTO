#!/bin/bash
# HPC3: seed the acorn 60-epoch continuation (S1TAG=acorn_long, plan_acorn_long.txt; DECISIONS 2026-10-04 07:04 UTC):
# copy the baseline run's epoch-30 state epochs/epoch29.ckpt (the last.ckpt of a finished run holds its best epoch) to
# stage1_acorn_long/ignn60/arm<A>_s<S>/gnn/artifacts/last.ckpt.
# Usage: sbatch slurm/hpc3/seed_ignn60.sbatch ARM SEED
#SBATCH -J seed_ignn60
#SBATCH -p free
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -t 00:30:00
#SBATCH -o /pub/jinboz1/hnl/logs_hpc3/%x_%j.out
export HNL_ROOT=${HNL_ROOT:-/pub/jinboz1/hnl} HNL_DATA=${HNL_DATA:-/pub/jinboz1/hnl_lrt}
set -euo pipefail
source $HNL_ROOT/hnl-displaced/env/env.sh || exit 1
src=$HNL_RUNS/baseline_v1/arm$1_s$2/gnn/artifacts/epochs/epoch29.ckpt
dst=$HNL_RUNS/stage1_acorn_long/ignn60/arm$1_s$2/gnn/artifacts
[ -e "$HNL_RUNS/baseline_v1/arm$1_s$2/gnn_train.done" ] || { echo "baseline arm $1 seed $2 has not finished its GNN training"; exit 1; }
[ -e "$src" ] || { echo "missing $src"; exit 1; }
mkdir -p "$dst"
[ -e "$dst/last.ckpt" ] || cp -p "$src" "$dst/last.ckpt"
sha256sum "$src" "$dst/last.ckpt"
touch "$HNL_RUNS/stage1_acorn_long/ignn60/arm$1_s$2/seeded.done"
