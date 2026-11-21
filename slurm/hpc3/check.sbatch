#!/bin/bash
# HPC3: run one Python script of scripts/hpc3 or analysis/ (CPU).
#   sbatch [-J name -c N --mem=XG -t HH:MM:SS] slurm/hpc3/check.sbatch scripts/hpc3/<script>.py [args...]
#SBATCH -J hpc3_check
#SBATCH -p free
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH -t 03:00:00
#SBATCH -o /pub/jinboz1/hnl/logs_hpc3/%x_%j.out
export HNL_ROOT=${HNL_ROOT:-/pub/jinboz1/hnl} HNL_DATA=${HNL_DATA:-/pub/jinboz1/hnl_lrt}
set -euo pipefail
source $HNL_ROOT/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
echo "git $(git rev-parse HEAD); python -B $*"
python -B "$@"
