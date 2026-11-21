"""Collect the validation read-outs of slurm/paper1/readouts.sbatch (a job script that is not part of this snapshot)
into $HNL_RESULTS (not preregistered; no selection depends on them).

1. Edge efficiency per truth |d0| bin at one edge false-positive rate for every model: the official e* of
   results/working_points_v1.json (the e* of claim S1b), each run at its own cut that gives it. The acorn runs (arms
   of HNL_ARMS, three seeds each: six runs with the default i, ii) are read out by that job (hnl.eval.edge_tpr --wp;
   new files under derived/paper1_v1/edge_tpr/); the Canto runs with the same command
   (runs/stage1_final_learned/l1tl/armi_s*/edge_tpr_estar.csv).
   -> results/paper1_edge_tpr_val_v1.csv (the paper does not use it; not part of this snapshot)
2. The headroom report of hnl.eval.headroom (graph-stage ceiling, CC with merged tracks split by truth, CC) at the CC
   cuts of results/working_points_v1.json (cuts_at_FR_star), run again by this job into derived/paper1_v1/headroom/ and
   compared with the copy in runs/baseline_v1/analysis_val/headroom/.
   -> results/paper1_headroom_val_v1_{losses,losses_per_seed,pairs_per_seed}.csv

    python analysis/paper1/gather.py

RUNS = hnl.paths.RUNS, NEW = hnl.paths.DERIVED/paper1_v1, and the tables go to hnl.paths.RESULTS (HNL_RESULTS), never
into the repo's results/. Models: the acorn arms of HNL_ARMS (default i, ii) and Canto (ttbar), seeds 0-2. The headroom
tables are compared with the copy in RUNS/baseline_v1/analysis_val/headroom (meta key comparison).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
from hnl import paths as _hnl_paths  # noqa: E402

RUNS = _hnl_paths.RUNS
NEW = os.path.join(_hnl_paths.DERIVED, "paper1_v1")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = _hnl_paths.RESULTS
ARMS = [a for a in os.environ.get("HNL_ARMS", "i,ii").split(",") if a]
ACORN = {k: v for k, v in {"i": "acorn (ttbar)", "ii": "acorn (HNL)", "iii": "acorn (ttbar+HNL)"}.items() if k in ARMS}
CANTO = {("Canto (ttbar)", s): f"stage1_final_learned/l1tl/armi_s{s}" for s in (0, 1, 2)}
if (os.path.realpath(RES) + "/").startswith(os.path.realpath(REPO) + "/"):
    raise SystemExit(f"gather: HNL_RESULTS {RES} is inside the repo: refusing")


def read_edge(csv: str, model: str, seed: int, run: str) -> pd.DataFrame:
    t = pd.read_csv(csv)
    eff = json.load(open(os.path.splitext(csv)[0] + "_efpr.json"))
    tpr = [c for c in t.columns if c.startswith("tpr_in_graph@")]
    edge = [c for c in t.columns if c.startswith("edge_eff@")]
    if len(tpr) != 1 or len(edge) != 1:
        raise SystemExit(f"{csv}: expected one evaluation cut, got {tpr} {edge}")
    cut = float(re.sub(r".*@", "", tpr[0]))
    if not np.isclose(cut, eff["cut"]):
        raise SystemExit(f"{csv}: column cut {cut} != json cut {eff['cut']}")
    t = t.rename(columns={tpr[0]: "tpr_in_graph", edge[0]: "edge_eff"})
    t.insert(0, "model", model)
    t.insert(1, "seed", seed)
    t.insert(2, "run", run)
    t["cut"], t["efpr"], t["target_efpr"] = eff["cut"], eff["efpr"], eff["target_efpr"]
    t["reachable"], t["dD"] = eff["reachable"], eff["dD"]
    return t


def edges() -> None:
    parts = []
    for arm, model in ACORN.items():
        for s in (0, 1, 2):
            parts.append(read_edge(os.path.join(NEW, "edge_tpr", f"arm{arm}_s{s}.csv"), model, s,
                                   f"baseline_v1/arm{arm}_s{s}"))
    for (model, s), run in CANTO.items():
        parts.append(read_edge(os.path.join(RUNS, run, "edge_tpr_estar.csv"), model, s, run))
    t = pd.concat(parts, ignore_index=True)
    if t.target_efpr.nunique() != 1:
        raise SystemExit(f"more than one target edge FPR: {t.target_efpr.unique()}")
    out = os.path.join(RES, "paper1_edge_tpr_val_v1.csv")
    os.makedirs(RES, exist_ok=True)
    t.to_csv(out, index=False, float_format="%.6g")
    s = t.drop_duplicates(["model", "seed"])[["model", "seed", "cut", "efpr", "reachable", "dD"]]
    print(s.to_string(index=False))
    meta = dict(script="analysis/paper1/gather.py", git=git(), job=os.environ.get("SLURM_JOB_ID"),
                split="validation (2,999 events)", target_efpr=float(t.target_efpr.iloc[0]),
                acorn_inputs=os.path.join(NEW, "edge_tpr"), canto_inputs=sorted(set(CANTO.values())), rows=len(t))
    json.dump(meta, open(out[:-4] + "_meta.json", "w"), indent=1)


def headroom() -> None:
    old = os.path.join(RUNS, "baseline_v1", "analysis_val", "headroom")
    report = {}
    for name in ("losses", "losses_per_seed", "pairs_per_seed"):
        new = pd.read_csv(os.path.join(NEW, "headroom", f"{name}.csv"))
        ref = pd.read_csv(os.path.join(old, f"{name}.csv"))
        num = new.select_dtypes("number").columns
        same = new.shape == ref.shape and list(new.columns) == list(ref.columns) and \
            np.allclose(new[num].to_numpy(float), ref[num].to_numpy(float), equal_nan=True, atol=1e-12)
        report[name] = dict(rows=len(new), identical_to_w1_analysis_val=bool(same))
        new.to_csv(os.path.join(RES, f"paper1_headroom_val_v1_{name}.csv"), index=False, float_format="%.6g")
    print(json.dumps(report, indent=1))
    meta = dict(script="analysis/paper1/gather.py", git=git(), job=os.environ.get("SLURM_JOB_ID"),
                command="python -m hnl.eval.headroom --runs $HNL_RUNS/baseline_v1 --wp $HNL_RESULTS/working_points_v1.json "
                        "--out $HNL_DERIVED/paper1_v1/headroom",
                split="validation (2,999 events)", cuts="cuts_at_FR_star of $HNL_RESULTS/working_points_v1.json",
                comparison=report)
    json.dump(meta, open(os.path.join(RES, "paper1_headroom_val_v1_meta.json"), "w"), indent=1)


def git() -> str:
    return subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


if __name__ == "__main__":
    edges()
    headroom()
    sys.exit(0)
