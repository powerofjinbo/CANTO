#!/bin/bash
# HPC3: the chain watcher (scripts/hpc3/chain.py watch): resubmits steps that ended without their marker (preempted on
# free-gpu / free) and rewires --dependency=afterok. Light (squeue/sacct every POLL s), 1 CPU, on -p standard so that it
# is not preempted itself (charged to asterenb_lab: at most 1 core x 3 days = 72 SU per job). Stop: touch $HNL_OUT/_chain/STOP
#SBATCH -J chain_watch
#SBATCH -p standard
#SBATCH -A asterenb_lab
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -t 3-00:00:00
#SBATCH -o /pub/jinboz1/hnl/logs_hpc3/%x_%j.out
export HNL_ROOT=${HNL_ROOT:-/pub/jinboz1/hnl} HNL_DATA=${HNL_DATA:-/pub/jinboz1/hnl_lrt}
set -uo pipefail
source $HNL_ROOT/hnl-displaced/env/env.sh || exit 1
cd "$HNL_REPO"
# a file-system error can still end the process (2026-11-19: twice within 30 min): start it again, forced, since the
# heartbeat left behind is its own
python3 -B scripts/hpc3/chain.py watch --poll "${POLL:-120}" ${FORCE:+--force}
rc=$?
while [ $rc -ne 0 ] && [ $rc -ne 3 ] && [ ! -e "$HNL_OUT/_chain/STOP" ]; do
    echo "watcher exited with $rc at $(date -u +%FT%TZ): restarting in 60 s"
    sleep 60
    python3 -B scripts/hpc3/chain.py watch --poll "${POLL:-120}" --force
    rc=$?
done
exit $rc
