"""Graph-stage ceiling per |d0| and r_v bin for the acorn and Canto graphs (validation events only). Not
preregistered; no selection depends on it.

The graph-stage ceiling of a particle is 1 when connected components on the chain-true edges present in the run's graph
give a double-majority match (eval_val/ceiling_particle.parquet, column dm_ceiling; docs/definitions.md section 6).
This script only bins that existing per-particle column, for every seed of:
  acorn (ttbar) = baseline_v1/armi_s*, acorn (HNL) = baseline_v1/armii_s*,
  Canto (ttbar) = stage1_final_learned/l1tl/armi_s* (the learned triplet-filter union graph of the frozen Canto).
Populations as hnl.eval.baseline_val.populations: hnl_targets (all HNL-decay targets, binned in |d0| and r_v), primary
(the HNL daughters, |d0| >= 10 mm or r_v >= 100 mm), w_muons, ttbar_prompt (integrated only).

    python analysis/paper1/graph_ceiling_bins.py --out results/paper1_graph_ceiling_val_v1.csv

The runs are read from hnl.paths.RUNS. A relative --out "results/..." goes to HNL_RESULTS (hnl.paths.repo_path); an
--out inside the repo is refused.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from hnl import paths as _hnl_paths
from hnl.eval.baseline_val import populations
from hnl.eval.particles import particle_table

RUNS = _hnl_paths.RUNS
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS = {
    "acorn (ttbar)": "baseline_v1/armi_s{}",
    "acorn (HNL)": "baseline_v1/armii_s{}",
    "Canto (ttbar)": "stage1_final_learned/l1tl/armi_s{}",
}
BINS = {"d0": [0, 0.1, 1, 3, 10, 30, 100, 300, np.inf],          # |d0| bins of the paper (mm)
        "rv": [0, 10, 30, 100, 200, 300, np.inf]}                # r_v bins of the paper (mm; "no vertex" separately)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    a.out = os.path.abspath(_hnl_paths.repo_path(a.out, REPO))
    if (os.path.realpath(a.out) + "/").startswith(os.path.realpath(REPO) + "/"):
        raise SystemExit(f"refusing --out {a.out}: inside the repo (its results/ are read-only)")
    rows, info, sources = [], None, {}
    for model, pat in MODELS.items():
        for seed in (0, 1, 2):
            path = os.path.join(RUNS, pat.format(seed), "eval_val", "ceiling_particle.parquet")
            c = pd.read_parquet(path)
            sources[f"{model} s{seed}"] = path
            if info is None:
                info = particle_table(np.unique(c.event_id))
            q = c.merge(info, on=["event_id", "particle_id"], how="inner")
            if len(q) != len(c):
                raise SystemExit(f"{path}: {len(c) - len(q)} particles without a truth row")
            P = populations(q)
            for pop in ("hnl_targets", "primary"):
                sel = q[P[pop]]
                for var, edges in BINS.items():
                    x = sel.d0.abs() if var == "d0" else sel.rv
                    b = pd.cut(x, edges, right=False).astype(str)
                    if var == "rv":
                        b[x.isna().to_numpy()] = "no vertex"
                    g = sel.groupby(b.to_numpy()).dm_ceiling.agg(["sum", "size"])
                    for lab, r in g.iterrows():
                        rows.append(dict(model=model, seed=seed, population=pop, variable=var, bin=lab,
                                         k=int(r["sum"]), n=int(r["size"])))
            for pop in ("primary", "hnl_targets", "w_muons", "ttbar_prompt"):
                x = q[P[pop]]
                rows.append(dict(model=model, seed=seed, population=pop, variable="all", bin="all",
                                 k=int(x.dm_ceiling.sum()), n=len(x)))
    out = pd.DataFrame(rows)
    out["ceiling"] = out.k / out.n
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    out.to_csv(a.out, index=False, float_format="%.6g")
    git = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    meta = dict(script="analysis/paper1/graph_ceiling_bins.py", git=git, job=os.environ.get("SLURM_JOB_ID"),
                split="validation (2,999 events)", sources=sources, rows=len(out))
    json.dump(meta, open(os.path.splitext(a.out)[0] + "_meta.json", "w"), indent=1)
    piv = out[(out.population == "primary") & (out.variable == "all")].pivot_table(index="model", columns="seed",
                                                                                     values="ceiling")
    print(piv.round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
