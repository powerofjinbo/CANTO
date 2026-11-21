"""Robustness check of the acorn training length (DECISIONS 2026-10-04 07:04 UTC): the acorn IGNNs continued to 60
epochs (runs/stage1_acorn_long/ignn60) against the 30-epoch baselines (runs/baseline_v1), val, per arm and seed: CC at
the best cut <= FR* and at the section-4 rule, and the Canto builders on the grid the 30- and 60-epoch runs share (MWS-layer + merging
with c in 0.03-0.2, walkthrough + merging); one-standard-error rule.

    python -m hnl.eval.acorn_long_compare armi,armii results/acorn_long_60ep_val.csv [--wp results/working_points_v1.json]

The runs are read from hnl.paths.RUNS and FR* from the frozen working points (--wp; "results/..." -> HNL_RESULTS, the
w1_gate copy), never a hard-coded value. The 30-epoch Canto builders come from one grid on every run,
RUN/trackbuild_val_mwslayer (builders cc,walkmergeS,mwslayer,mwslayermerge); the 60-epoch ones from RUN60/trackbuild_val
(walkmergeS,mwslayer,mwslayermerge). A missing grid or a failed ranking is a structural failure (exit 1, nothing
written). A relative "results/..." output goes to HNL_RESULTS; an output inside the repo is refused.
"""
import argparse
import json
import os
import sys

import pandas as pd

from hnl import paths as _hnl_paths
from hnl.eval.ranking import rank

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
GRID30 = "trackbuild_val_mwslayer"
GRID60 = "trackbuild_val"
METHODS = [("CC best", {"source": "eval_val", "rule": "best"}, {"source": "eval_val", "rule": "best"}),
           ("CC s4", {"source": "eval_val", "rule": "s4"}, {"source": "eval_val", "rule": "s4"}),
           ("MWS-layer+merge", {"source": [GRID30], "family": "mwslayermerge"},
            {"source": [GRID60], "family": "mwslayermerge"}),
           ("walk+merge", {"source": [GRID30], "family": "walkmerge"}, {"source": [GRID60], "family": "walkmerge"})]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", help="comma list, e.g. armi,armii")
    ap.add_argument("out")
    ap.add_argument("--wp", default="results/working_points_v1.json")
    a = ap.parse_args(argv)
    out = os.path.abspath(_hnl_paths.repo_path(a.out, REPO))
    if (os.path.realpath(out) + "/").startswith(os.path.realpath(REPO) + "/"):
        raise SystemExit(f"refusing output {out}: inside the repo (its results/ are read-only)")
    wp_path = _hnl_paths.repo_path(a.wp, REPO)
    fr = float(json.load(open(wp_path))["FR_star"])
    print(f"FR* = {fr:.6f} from {wp_path}")
    rows, bad = [], []
    for arm in a.arms.split(","):
        for s in (0, 1, 2):
            for ep, base, k in (("30", "baseline_v1", 1), ("60", "stage1_acorn_long/ignn60", 2)):
                run = os.path.join(_hnl_paths.RUNS, base, f"{arm}_s{s}")
                for m in METHODS:
                    lab, e = m[0], m[k]
                    srcs = [e["source"]] if isinstance(e["source"], str) else e["source"]
                    miss = [g for g in srcs if not os.path.exists(os.path.join(run, g, "per_event.parquet"))]
                    if miss:
                        bad.append(f"{run}: missing {miss} ({lab})")
                        continue
                    try:
                        _, ent = rank([{"label": "x", "entries": [{"run": run, **e}]}], fr, tie_se=True)
                    except Exception as ex:  # noqa: BLE001  (reported, then a structural failure)
                        bad.append(f"{run} {lab}: {ex!r}"[-300:])
                        continue
                    r = ent.iloc[0]
                    rows.append(dict(arm=arm, seed=s, epochs=ep, method=lab, eff_primary=r.eff_primary,
                                     eff_ttbar_prompt=r.eff_ttbar_prompt, fake_rate_hnl=r.fake_rate_hnl,
                                     wp=str(r.working_point)))
    if bad:
        print("\n".join(bad))
        raise SystemExit(f"acorn_long_compare: {len(bad)} inputs missing or failing; nothing written")
    t = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    t.to_csv(out, index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    print(t.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
