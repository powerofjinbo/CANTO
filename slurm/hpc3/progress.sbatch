#!/bin/bash
# HPC3: live progress page (scripts/hpc3/progress.py): rewrites PROGRESS.html / PROGRESS.txt in the repo every 60 s.
# Read-only on the chain; 1 core on -p standard so it is not preempted (at most 48 SU of asterenb_lab per job).
#SBATCH -J progress_page
#SBATCH -p standard
#SBATCH -A asterenb_lab
#SBATCH -c 1
#SBATCH --mem=1G
#SBATCH -t 2-00:00:00
#SBATCH -o /pub/jinboz1/hnl/logs_hpc3/%x_%j.out
cd /pub/jinboz1/hnl/hnl-displaced
exec python3 -B scripts/hpc3/progress.py --loop 60
