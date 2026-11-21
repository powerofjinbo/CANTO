"""Producer of the CC_REF file of hnl.eval.per_sample (chain step ro_cc_ref).

The file is the entries table of hnl.eval.ranking.rank() on cantol_cc_spec.json, i.e. Canto-L's classifier +
connected components on the Canto-L runs' eval_val, entries without "rule" (so rule s4: the largest fake rate <= FR*):

    python3 -B scripts/hpc3/cc_ref.py --selection $HNL_RESULTS/final_selection_v1.json \
        --wp $HNL_RESULTS/working_points_v1.json --out $HNL_RESULTS/per_sample_cc_ref

The runs are those of the frozen selection (pipeline l1tl, seeds 0, 1, 2; hnl.eval.per_sample matches the rows by run).
FR* is that of the frozen working points (its FR* must equal the selection's). Writes OUT/cantol_cc_spec.json and, last,
OUT/cantol_cc_entries.csv (the chain marker). Refuses to overwrite cantol_cc_entries.csv. Read-only on everything else.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from hnl.eval import ranking as RK

LABEL = "Canto-L classifier + CC (best cut <= FR*)"   # the spec's label (per_sample compares by run only)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selection", required=True, help="the frozen final_selection_v1.json")
    ap.add_argument("--wp", required=True, help="the frozen working_points_v1.json")
    ap.add_argument("--out", required=True, help="output directory")
    a = ap.parse_args(argv)
    out_csv = os.path.join(a.out, "cantol_cc_entries.csv")
    if os.path.exists(out_csv):
        sys.exit(f"{out_csv} exists: refusing to overwrite")
    sel = json.load(open(a.selection))
    wp = json.load(open(a.wp))
    fr = float(wp["FR_star"])
    if abs(fr - float(sel["FR_star"])) > 1e-15:
        sys.exit(f"FR* of {a.wp} ({fr}) and {a.selection} ({sel['FR_star']}) differ")
    seeds = sel["pipelines"]["l1tl"]["seeds"]
    runs = [seeds[s]["run"] for s in ("0", "1", "2")]
    for r in runs:
        for f in ("per_particle.parquet", "per_event.parquet"):
            if not os.path.exists(os.path.join(r, "eval_val", f)):
                sys.exit(f"{r}/eval_val/{f} missing")
    spec = [{"label": LABEL, "group": "Canto", "short": "Canto-L CC*",
             "entries": [{"run": r, "source": "eval_val"} for r in runs]}]
    _, ent = RK.rank(spec, fr)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "cantol_cc_spec.json"), "w") as f:
        json.dump(spec, f, indent=1)
    tmp = f"{out_csv}.tmp{os.getpid()}"
    ent.to_csv(tmp, index=False)
    os.replace(tmp, out_csv)
    print(ent.to_string())
    print(f"wrote {out_csv} (FR* {fr})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
