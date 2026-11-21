"""Paths of the analysis. Inputs (raw/, unpacked/, generated/) are read from HNL_DATA; the output roots HNL_RUNS,
HNL_DERIVED, HNL_SCRATCH and HNL_RUNS_EXTRA are under HNL_OUT (default $HNL_DATA/hpc3_v1). A few modules hold default
paths of their own under HNL_DATA. env/env.sh exports the same variables."""
from __future__ import annotations

import os

ROOT = os.environ.get("HNL_ROOT", "/pub/jinboz1/hnl")
DATA = os.environ.get("HNL_DATA", "/pub/jinboz1/hnl_lrt")
OUT = os.environ.get("HNL_OUT", os.path.join(DATA, "hpc3_v1"))
RUNS = os.environ.get("HNL_RUNS", os.path.join(OUT, "runs"))
DERIVED = os.environ.get("HNL_DERIVED", os.path.join(OUT, "derived"))
SCRATCH = os.environ.get("HNL_SCRATCH", os.path.join(OUT, "scratch"))
RUNS_EXTRA = os.environ.get("HNL_RUNS_EXTRA", os.path.join(OUT, "runs_extra"))
# read-out tables: the code's "results/..." paths are read and written under HNL_RESULTS (default $HNL_OUT/results),
# not in the repo's results/
RESULTS = os.environ.get("HNL_RESULTS", os.path.join(OUT, "results"))


def repo_path(p: str, repo: str) -> str:
    """A repo-relative path of the read-out code: "results/..." goes to RESULTS, anything else stays in the repo."""
    if os.path.isabs(p):
        return p
    q = os.path.normpath(p)
    if q == "results" or q.startswith("results" + os.sep):
        return os.path.join(RESULTS, os.path.relpath(q, "results"))
    return os.path.join(repo, p)
