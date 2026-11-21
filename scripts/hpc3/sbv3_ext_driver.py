"""HPC3: the data-dependent grid-extension loop of hnl.eval.strong_baselines_v3 (follow-up round 3, task A2) as ONE CPU
step of the chain (scripts/hpc3/chain.py), with no sbatch from inside.

    python3 -B scripts/hpc3/sbv3_ext_driver.py --marker FILE [--max-steps 6] [--n 2] [--keep-local]

Run by hand, the loop is: `strong_baselines_v3 inventory --step K --families ...` wrote the step's plan
(feedback_v3/strong_baselines_a2/extensions/step<K>.json: for every family choice at a grid edge, the edge slice's
configurations with the next values of the parameter, run on all acorn runs), `ext-jobs --step K` printed one
slurm/feedback/fb3_grid_val.sbatch job per run and group, and the next step's inventory followed once those grids were
written, until no choice sat at an edge (interior, or the parameter's range ends). Here, per step K = 1, 2, ...:

  1. the plan of step K: `inventory(families, OUT_DIR, K)` (strong_baselines_v3; the same function and rule), or the
     existing step<K>.json of an interrupted attempt (its grids are those of steps < K, all finished before it was
     written, so it is the same plan);
  2. an empty plan ends the loop: every choice of every family on every run is interior or at its range end;
  3. otherwise every grid of the step (strong_baselines_v3.ext_tasks: one per run and group, configuration lists
     next to the plan) is evaluated here, as fb3_grid_val.sbatch does: the run's val scores (RUN/gnn_scores/valset.tar)
     extracted to node-local /tmp, strong_baselines_v3.run_grid with the group's worker count (acorn's CCandWalk: half
     the CPUs, for memory) and, for learned merging, the run's merge model; then RUN/<grid>/provenance.json (the
     finish signal; a grid directory without it is from an interrupted attempt and is evaluated again).

Families: strong_baselines_v3.FAMILIES (acwalk, mwslmerge). Preemption-safe: a resubmission resumes at the first
unfinished grid. The marker (JSON: steps, grids, final inventory) is written only after the loop ended with an empty
plan; more than --max-steps steps stop the job (exit 1). One thread per worker process (OMP / NUMBA = 1, as
fb3_grid_val.sbatch: outputs identical).
"""
from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"          # before numpy: one thread per worker process (fb3_grid_val.sbatch)
os.environ["NUMBA_NUM_THREADS"] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--marker", required=True)
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--n", type=int, default=2, help="values per edge and step (strong_baselines_v3 inventory --n)")
    ap.add_argument("--cpus", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    ap.add_argument("--keep-local", action="store_true")
    a = ap.parse_args(argv)
    from hnl.eval import strong_baselines_v3 as S3

    out_dir = os.path.join(S3.FB3, "extensions")
    fams = [f for _, f, *_ in S3.FAMILIES]
    job = os.environ.get("SLURM_JOB_ID", "local")
    print(f"{now()} sbv3_ext_driver: git {S3._git()}; job {job}; host {os.uname().nodename}; cpus {a.cpus}; "
          f"families {fams}; arms {[x for x, _ in S3.ARMS]}; plans {out_dir}", flush=True)
    # inputs: every run's base grids of the families (acorn_acwalk_all_merge, acorn_lmerge_*) and merge model
    missing = []
    for run in S3.runs_all():
        for g in S3.MENU_PARTS:
            if not os.path.exists(os.path.join(run, g, "configs.json")):
                missing.append(os.path.join(run, g))
        if "mwslmerge" in fams and not os.path.exists(S3.merge_model(run)):
            missing.append(S3.merge_model(run))
        if not os.path.exists(os.path.join(run, "gnn_scores", "valset.tar")):
            missing.append(os.path.join(run, "gnn_scores", "valset.tar"))
    if missing:
        print("missing inputs:\n  " + "\n  ".join(missing), file=sys.stderr)
        return 1
    if os.path.exists(a.marker):
        print(f"{a.marker} exists: the loop is finished")
        return 0
    local = os.path.join("/tmp", os.environ.get("USER", "u"), f"sbv3_ext_{job}")
    done_grids, steps = [], []
    try:
        for k in range(1, a.max_steps + 2):
            path = os.path.join(out_dir, f"step{k}.json")
            if os.path.exists(path):
                plan = json.load(open(path))
                if plan.get("families") != fams or int(plan.get("n_values_per_edge", -1)) != a.n:
                    print(f"{path}: families {plan.get('families')} / n {plan.get('n_values_per_edge')} are not "
                          f"{fams} / {a.n}", file=sys.stderr)
                    return 1
                print(f"{now()} step {k}: plan {path} exists (resumed)", flush=True)
            else:
                print(f"{now()} step {k}: inventory", flush=True)
                plan = S3.inventory(fams, out_dir, k, a.n)["plan"]
            n_new = {f: len(v) for f, v in plan["configs"].items()}
            steps.append({"step": k, "plan": path, "new_configs": n_new})
            if not any(n_new.values()):
                print(f"{now()} step {k}: no family choice at an extendable grid edge: loop finished", flush=True)
                break
            if k > a.max_steps:
                print(f"step {k} would be needed (more than --max-steps {a.max_steps}): stopping", file=sys.stderr)
                return 1
            tasks = S3.ext_tasks(k, out_dir)
            print(f"{now()} step {k}: {n_new}; {len(tasks)} grids to evaluate", flush=True)
            for run in dict.fromkeys(t["run"] for t in tasks):
                graphs = os.path.join(local, os.path.basename(run))
                shutil.rmtree(graphs, ignore_errors=True)
                os.makedirs(graphs)
                subprocess.run(["tar", "-xf", os.path.join(run, "gnn_scores", "valset.tar"), "-C", graphs], check=True)
                for t in (x for x in tasks if x["run"] == run):
                    d = os.path.join(run, t["grid"])
                    if os.path.exists(d):                      # an interrupted attempt (no provenance.json)
                        print(f"{d}: unfinished (no provenance.json): evaluated again", flush=True)
                        shutil.rmtree(d)
                    cfgs = S3.parse_configs(open(t["configs_file"]).read())
                    workers = max(1, a.cpus // 2) if t["workers"] else a.cpus
                    t0 = time.time()
                    print(f"{now()} {run} {t['grid']}: {len(cfgs)} configurations, {workers} workers, merge model "
                          f"{t['merge_model'] or 'none'}", flush=True)
                    S3.run_grid(graphs, d, cfgs, workers, t["merge_model"])
                    S3.grid_provenance(run, t["grid"], t["configs_file"], t["merge_model"], job)
                    done_grids.append({"run": run, "grid": t["grid"], "elapsed_s": round(time.time() - t0, 1)})
                if not a.keep_local:
                    shutil.rmtree(graphs, ignore_errors=True)
        else:
            return 1
    finally:
        if not a.keep_local:
            shutil.rmtree(local, ignore_errors=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.marker)), exist_ok=True)
    with open(a.marker + ".tmp", "w") as f:
        json.dump({"finished_utc": now(), "slurm_job": job, "git": S3._git(), "families": fams,
                   "arms": [x for x, _ in S3.ARMS], "steps": steps, "grids_this_job": done_grids,
                   "parts": {os.path.basename(r): S3.ext_parts(r) for r in S3.runs_all()}}, f, indent=1)
    os.replace(a.marker + ".tmp", a.marker)
    print(f"{now()} wrote {a.marker}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
