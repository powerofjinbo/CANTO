# Source (bash) in every job script and interactive shell:
#     source /pub/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
# = env/env.sh (all caches / TMPDIR under $HNL_ROOT, refuses /home) + the venv built by env/make_env.sh.
# Thread counts follow the Slurm allocation; outside Slurm (login node) they default to 4.

source "$(dirname "${BASH_SOURCE[0]}")/env.sh" || return 1
if [ ! -x "$HNL_ENV/venv/bin/python" ]; then
    echo "activate.sh: no venv at $HNL_ENV/venv (run env/make_env.sh)" >&2
    return 1
fi
# shellcheck disable=SC1091
source "$HNL_ENV/venv/bin/activate"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}
export NUMBA_NUM_THREADS=${NUMBA_NUM_THREADS:-$OMP_NUM_THREADS}
# HPC3: record where a job runs (host, partition, GPU model) at the top of every job log
if [ -n "${SLURM_JOB_ID:-}" ] && [ -z "${HNL_JOBINFO_DONE:-}" ]; then
    export HNL_JOBINFO_DONE=1
    echo "jobinfo: job $SLURM_JOB_ID${SLURM_ARRAY_TASK_ID:+ task $SLURM_ARRAY_TASK_ID} host $(hostname) partition ${SLURM_JOB_PARTITION:-?} account ${SLURM_JOB_ACCOUNT:-?} gpu $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | paste -sd';' - || echo none) out $HNL_OUT" >&2
fi
# HPC3 GPU rule: training and evaluation run on A30 or A100; V100 only with HNL_ALLOW_V100=1 (after a job
# has shown that the step fits in V100 memory); timing / throughput steps require A30 (HNL_REQUIRE_A30=1).
if [ -n "${SLURM_JOB_ID:-}" ] && command -v nvidia-smi > /dev/null 2>&1; then
    _hnl_gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1)
    case "$_hnl_gpu" in
        *V100*) if [ "${HNL_ALLOW_V100:-0}" != 1 ]; then echo "activate.sh: refusing to run on $_hnl_gpu (HNL_ALLOW_V100 is not 1)" >&2; unset _hnl_gpu; return 1; fi ;;
    esac
    if [ "${HNL_REQUIRE_A30:-0}" = 1 ]; then
        case "$_hnl_gpu" in *A30*) ;; *) echo "activate.sh: this step needs an A30, got '$_hnl_gpu'" >&2; unset _hnl_gpu; return 1 ;; esac
    fi
    unset _hnl_gpu
fi
