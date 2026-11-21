"""Copy records to the names the read-out code reads in $HNL_RESULTS (read-only copies with a .sha256).

    python3 -B scripts/hpc3/ro_copy.py --marker FILE [--same A B ...] SRC DST [SRC DST ...]

--same A B: A and B must be byte-identical before anything is copied (e.g. the frozen selection and the candidate it
was frozen from, so that the candidate's seed table belongs to the frozen selection). A DST that exists with the same
content is kept; with other content the copy is refused (exit 1). Every DST is written atomically, made read-only and
gets DST.sha256; the marker is written last (chain step done). Read-out steps:
  ro_seeds_copy   $HNL_RESULTS/candidate/final_selection_v1_seeds.csv -> $HNL_RESULTS/final_selection_v1_seeds.csv
                  (--same: $HNL_RESULTS/final_selection_v1.json and candidate/final_selection_v1.json)
  ro_claims_copy  $HNL_RUNS/final_test_v1/claims/claims.json -> $HNL_RESULTS/claims_test_v1.json
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import stat
import sys
import time


def sha(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--marker", required=True)
    ap.add_argument("--same", nargs=2, action="append", default=[], metavar=("A", "B"))
    ap.add_argument("pairs", nargs="+")
    a = ap.parse_args(argv)
    if len(a.pairs) % 2:
        sys.exit("give SRC DST pairs")
    for x, y in a.same:
        if sha(x) != sha(y):
            print(f"{x} and {y} differ: refusing to copy", file=sys.stderr)
            return 1
    pairs = list(zip(a.pairs[::2], a.pairs[1::2]))
    for src, dst in pairs:
        if os.path.exists(dst) and sha(dst) != sha(src):
            print(f"{dst} exists with other content than {src}: refusing", file=sys.stderr)
            return 1
    lines = []
    for src, dst in pairs:
        os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
        if not os.path.exists(dst):
            tmp = f"{dst}.tmp{os.getpid()}"
            shutil.copyfile(src, tmp)
            os.chmod(tmp, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
            os.replace(tmp, dst)
        h = sha(dst)
        with open(dst + ".sha256", "w") as f:
            f.write(f"{h}  {dst}\n")
        lines.append(f"{src} -> {dst} sha256 {h}")
        print(lines[-1])
    os.makedirs(os.path.dirname(os.path.abspath(a.marker)), exist_ok=True)
    with open(a.marker, "w") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} job {os.environ.get('SLURM_JOB_ID', 'local')}\n"
                + "\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
