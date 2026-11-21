"""Canto-L trained on HNL (arm ii), seeds 0, 1 and 2: the validation settings and the read-out spec of its test on the
HNL test sets, so that the model has three seeds on test like Canto-L (ttbar). Post-freeze, descriptive, no claim;
nothing is chosen on test.

  select   validation only. One file with seeds 0, 1 and 2 of arm (ii), all three by the same calls: per seed the
           inputs of slurm/stage1/final_test_shard.sbatch and final_test_merge.sbatch, i.e. the FR* configuration
           (builder rule at FR* with the one-standard-error tie rule; hnl.eval.final_select.run_rows over the val grid
           trackbuild_val_lmerge (--grids), family mwslmerge, with the guard baseline of
           hnl.eval.training_sample_board and hnl.eval.training_sample_3seed), the LLP-blind configuration of the same
           family (section-6c rule on ttbar val, the same call), the classifier's e* cut (hnl.eval.final_select.cut_e)
           and the run's merge model, with the frozen working points (--wp, default
           $HNL_RESULTS/working_points_v1.json). The classifier checkpoint (best checkpoint of the run) must be the
           one that scored the run's val graphs. run = source_run =
           $HNL_RUNS/stage1_final_learned_ts/l1tl/armii_s<S>, graphs_run = graphs_source_run =
           $HNL_RUNS/baseline_v1_ts/armii_s<S>. A separate gate step checks and freezes the file. The guard baseline
           of run_rows is $HNL_RUNS/baseline_v1/armi_s0 for every seed. Default --out
           $HNL_RESULTS/candidate/training_sample_test_selection_hpc3_v1.json.
  spec     from that one frozen selection (--selection, default
           $HNL_RESULTS/training_sample_test_selection_hpc3_v1.json, its .sha256 required); label, short and group as
           in the selection. Default --out $HNL_RESULTS/training_sample_spec_test_3seed_v1.json.

    python -m hnl.eval.training_sample_seeds12 select
    python -m hnl.eval.training_sample_seeds12 spec
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os

import numpy as np
import pandas as pd

from hnl.eval import bootstrap as BS
from hnl.eval import bootstrap_tables as BT
from hnl.eval import per_sample as PSM
from hnl.eval import published_defs as PD
from hnl.eval import second_population as SP
from hnl.eval import training_sample_test as TST
from hnl.eval.particles import particle_table
import hnl.paths as _hnl_paths

REPO = BT.REPO
RUNS = _hnl_paths.RUNS
SEL_NEW = "results/training_sample_test_selection_hpc3_v1.json"      # the frozen selection, seeds 0, 1, 2
SEL_CANDIDATE = "results/candidate/training_sample_test_selection_hpc3_v1.json"   # select's output (unfrozen)
SPEC_3 = "results/training_sample_spec_test_3seed_v1.json"
WP = "results/working_points_v1.json"
GRIDS_HPC3 = "trackbuild_val_lmerge"
FAMILY = "mwslmerge"
ARMS = {"ii": ("l1tl_ii", "Canto-L (HNL)", "arm (ii), HNL")}
SEEDS = ("0", "1", "2")
SUB = TST.SUB


def _repo(p: str) -> str:
    return BT._repo(p)


def _sha(p: str) -> str:
    return PSM.sha256(p)


# --------------------------------------------------------------------------------------------------------- select
def _val_record_ckpt(run: str) -> dict:
    fs = sorted(glob.glob(os.path.join(run, "gnn_scores", "infer_*.json")))
    if len(fs) != 1:
        raise SystemExit(f"{run}/gnn_scores: {len(fs)} val inference records")
    return json.load(open(fs[0]))


def cmd_select(a) -> None:
    from hnl.eval import final_select as FS
    from hnl.graphs.helix_union import best_ckpt
    out_path = _repo(a.out)
    if os.path.exists(out_path) and not a.check_only:
        raise SystemExit(f"{out_path} exists: refusing to overwrite")
    wp_path = os.path.abspath(_repo(a.wp))
    wpj = json.load(open(wp_path))
    FS.matching_baseline = lambda run: os.path.join(RUNS, "baseline_v1", "armi_s0")   # the board's guard baseline
    checks: list = []
    pipelines = {}
    for arm, (pname, label, sample) in ARMS.items():
        seeds_out = {}
        for s in SEEDS:
            src = os.path.join(RUNS, "stage1_final_learned_ts", "l1tl", f"arm{arm}_s{s}")
            t, _ = FS.run_rows(src, a.grids, wpj, wpj["reference_arm"])
            fam = t[t.family == FAMILY]
            TST.check(checks, f"{label} s{s}: one {FAMILY} row in {a.grids}", len(fam), 1, tol=0.0)
            if len(fam) != 1:
                continue
            r = fam.iloc[0]
            cfg, llp, ce = str(r.config), str(r.llp_blind_config), FS.cut_e(src, wpj)
            wp_txt = f"{FAMILY} {cfg.split(',', 1)[1]} ({r.grid})"
            print(f"{label} s{s}: FR* {cfg} ({r.grid}); LLP-blind {llp}; e* cut {ce}", flush=True)
            TST.check(checks, f"{label} s{s}: e* cut computed (RUN/edge_tpr_estar_efpr.json)", int(ce is not None), 1,
                      tol=0.0)
            gnn = best_ckpt(src, "gnn")
            rec = _val_record_ckpt(src)
            gsha = _sha(gnn)
            TST.check(checks, f"{label} s{s}: best checkpoint = val scores' checkpoint", gnn, rec["ckpt"], how="str")
            TST.check(checks, f"{label} s{s}: checkpoint sha256 = val record", gsha, rec["ckpt_sha256"], how="str")
            gsrc = os.path.join(RUNS, "baseline_v1_ts", f"arm{arm}_s{s}")
            ml_run = os.path.join(RUNS, "baseline_v1", f"arm{arm}_s{s}")
            ub = json.load(open(os.path.join(gsrc, "ltf_union_graphs", "report_val", "build.json")))
            TST.check(checks, f"{label} s{s}: union ML checkpoint = baseline best ML checkpoint",
                      os.path.realpath(ub["ml_ckpt"]), os.path.realpath(best_ckpt(ml_run, "ml")), how="str")
            TST.check(checks, f"{label} s{s}: union triplet model is the run's", ub["triplet_model"],
                      os.path.join(gsrc, "learned_triplet_filter_v2", "model.joblib"), how="str")
            TST.check(checks, f"{label} s{s}: acorn one-time test done", int(os.path.exists(
                os.path.join(ml_run, "test_eval.done"))), 1, tol=0.0)
            merge = os.path.join(src, "lmerge", "model.joblib")
            TST.check(checks, f"{label} s{s}: merge model exists", int(os.path.exists(merge)), 1, tol=0.0)
            if not os.path.exists(merge):
                continue
            seeds_out[s] = {
                "run": src, "source_run": src, "graphs_run": gsrc, "graphs_source_run": gsrc,
                "ml_run": ml_run, "test_graphs_source": os.path.join(ml_run, "gnn_scores_test", "testset.tar"),
                "cut_e": ce, "merge_model": merge, "merge_model_sha256": _sha(merge), "config": cfg,
                "llp_blind_config": llp, "val_working_point": wp_txt, "gnn_ckpt": gnn, "gnn_ckpt_sha256": gsha,
                "triplet_model": ub["triplet_model"], "triplet_model_sha256": _sha(ub["triplet_model"]),
                "tau": ub["tau"], "k": ub["k"], "r_wide": ub["r_wide"], "nsig": ub["nsig"], "pt_min": ub["pt_min"],
                "union_ml_ckpt": ub["ml_ckpt"]}
        pipelines[pname] = {"dir": os.path.join(RUNS, "stage1_final_learned_ts", "l1tl"), "graphs": "ltf_union_graphs",
                            "training_sample": sample, "family": FAMILY, "seeds": seeds_out}
    TST.stop_if_failed(checks, "selection checks")
    out = {"created_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "git": PSM.git_state(),
           "host": os.uname().nodename, "slurm_job": os.environ.get("SLURM_JOB_ID"),
           "purpose": "post-freeze, descriptive test of "
                      "Canto-L (HNL), seeds 0, 1 and 2. Not a "
                      "selection: final_selection_v1.json and the claims are unchanged. Validation settings only; input "
                      "of slurm/stage1/final_test_shard.sbatch and final_test_merge.sbatch (pipeline l1tl_ii, seeds 0-2) "
                      "once frozen by the gate step.",
           "working_points": wp_path, "working_points_sha256": _sha(wp_path), "FR_star": wpj["FR_star"],
           "e_star": wpj["e_star"], "reference_arm": wpj["reference_arm"],
           "grid": a.grids,
           "rules": "family mwslmerge; per run on val: FR* configuration by the builder rule with the one-standard-error "
                    "tie rule and LLP-blind configuration by the section-6c rule (hnl.eval.final_select.run_rows, as "
                    "hnl.eval.training_sample_board and training_sample_3seed), e* cut from RUN/edge_tpr_estar_efpr.json "
                    "(final_select.cut_e); the same calls for seeds 0, 1 and 2",
           "outputs": "run = source_run (the trained run) and graphs_run = graphs_source_run: every test output is "
                      "written into the trained runs",
           "candidate": None, "pipelines": pipelines,
           "checks": f"{len(checks)} of {len(checks)} ok", "check_list": [[c[0], str(c[1]), str(c[2]), c[3]] for c in checks]}
    if a.check_only:
        print(json.dumps({k: v for k, v in out.items() if k != "check_list"}, indent=1))
        return
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path + ".tmp", "w") as f:
        json.dump(out, f, indent=1)
    os.replace(out_path + ".tmp", out_path)
    print(f"wrote {out_path}")


def cmd_spec(a) -> None:
    """The read-out spec of the three-seed model (format of results/training_sample_spec_test_v1.json) from the one
    frozen selection (seeds 0, 1, 2; its .sha256 must match); RUN/trackbuild_test, configuration 0."""
    out_path = _repo(a.out)
    if os.path.exists(out_path):
        raise SystemExit(f"{out_path} exists: refusing to overwrite")
    sel_path = _repo(a.selection)
    sha_file = sel_path + ".sha256"
    if not os.path.exists(sha_file) or open(sha_file).read().split()[:1] != [_sha(sel_path)]:
        raise SystemExit(f"{sel_path}: not frozen ({sha_file} missing or different; scripts/hpc3/check_ts_selection.py)")
    sel = json.load(open(sel_path))
    spec = []
    for pname, label, _ in ARMS.values():
        ents = []
        for s in SEEDS:
            d = sel["pipelines"][pname]["seeds"][s]
            fam, params = d["config"].split(",", 1)
            ents.append({"run": d["run"], "sub": SUB, "cut": 0, "wp": f"{fam} {params} ({SUB}, configuration 0)"})
        spec.append({"label": f"{label}: learned triplet union, learned merging (training-sample study, three seeds, "
                              "post-freeze)", "short": label, "group": "Canto (training sample)", "entries": ents})
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path + ".tmp", "w") as f:
        json.dump(spec, f, indent=1)
        f.write("\n")
    os.replace(out_path + ".tmp", out_path)
    print(json.dumps(spec, indent=1))
    print(f"wrote {out_path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("select")
    p.add_argument("--out", default=_repo(SEL_CANDIDATE), help="results/... goes to $HNL_RESULTS")
    p.add_argument("--wp", default=_repo(WP), help="the frozen working points (default %(default)s)")
    p.add_argument("--grids", default=GRIDS_HPC3, help="val grids of run_rows (default %(default)s)")
    p.add_argument("--check-only", action="store_true", help="checks and the would-be file on stdout; nothing written")
    p = sub.add_parser("spec")
    p.add_argument("--selection", default=_repo(SEL_NEW), help="the frozen selection (default %(default)s)")
    p.add_argument("--out", default=_repo(SPEC_3), help="results/... goes to $HNL_RESULTS")
    a = ap.parse_args(argv)
    {"select": cmd_select, "spec": cmd_spec}[a.cmd](a)


if __name__ == "__main__":
    main()
