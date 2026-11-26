# Canto: a graph neural network tracker for displaced tracks

Graph neural network (GNN) trackers learn what a track looks like from simulated Standard Model events, in which
almost every track starts at the collision point, and they can miss the displaced tracks of long-lived particles.
In a uniform magnetic field a displaced track has the same shape as a prompt track and differs in where it starts.
Canto is a GNN tracker that recognizes a track by its shape: it is trained on Standard Model top-quark pair
(ttbar) events only and tested on a public benchmark of heavy neutral lepton (HNL) decays.

This repository holds code, configurations, the event lists of the splits and result tables of the analysis
reported in the paper "Canto: a graph neural network tracker for displaced tracks". The analysis was run on the
HPC3 cluster of the Research Cyberinfrastructure Center at the University of California, Irvine (UCI).

## Results

All numbers are for the HNL test sets (6999 events). The measure is the efficiency for the 5013 displaced
daughters, the muons from an HNL decay with |d0| >= 10 mm or decay radius r_v >= 100 mm. Every number is the mean
of three trainings with different seeds; the uncertainty combines an event bootstrap with the spread of the seeds.
Differences are in percentage points.

| Result of Canto | Value | Baseline |
|---|---|---|
| Efficiency of Canto (ttbar) | 98.82 ± 0.14 % | standard tracker, acorn (ttbar) with CC: 89.75 ± 0.39 % |
| Lead over the tracker trained on the signal | 7.16 ± 1.52 points | acorn (HNL) with CC at the cut rule: 91.66 ± 1.52 % |
| Lead over the same tracker with the builder of Canto | 1.30 ± 0.23 points | acorn (HNL) with LM: 97.53 ± 0.24 % |
| Gain of Canto from training on the signal | 0.36 ± 0.14 points | Canto (HNL), 99.18 ± 0.12 %, against Canto (ttbar) |

The fake rate of Canto (ttbar) is 1.00 ± 0.02 %, against 3.02 ± 0.05 % for the standard tracker.

Here acorn is the Exa.TrkX pipeline in its acorn implementation, trained on the ttbar sample or on the two labeled
HNL samples. CC is the connected-components track builder, and LM is the builder of Canto (a mutex watershed with
same-layer repulsion, followed by learned merging of track fragments). The standard tracker is acorn (ttbar) with
CC. The cut rule and the limit on the fake rate are defined in the paper and in `docs/preregistration.md`.

The benchmark has no pileup, a fast detector simulation and a uniform magnetic field. It holds 620 HNL daughters
with |d0| >= 100 mm in the test sets, and for these the lead of Canto over acorn (HNL) with LM is not established.

## Layout

| Path | Content |
|---|---|
| `src/hnl/` | The Python package `hnl` (see below) |
| `configs/baseline/` | acorn configurations of the graph (metric learning) and of the edge classifier |
| `configs/stage1/` | Configurations of the edge classifiers trained for this work; `triplet_symregion_trunc.yaml` is the Canto classifier |
| `splits/` | Event identifiers of every split; `ttbar_*`, `hnl10_100_*`, `hnl15_200_*` and `hnl15_100_external` are the splits of the paper, and the other lists are not used in it |
| `patches/acorn/` | Seven patches to acorn at its pinned commit, applied at install time; `patches/acts_v15/` holds two patches to ACTS that the results of the paper do not need |
| `env/` | Scripts that build and activate the environment, and its lock file |
| `slurm/hpc3/`, `scripts/hpc3/` | Part of the job chain used on the cluster |
| `scripts/diagnostics/` | Three checks of the samples (truth, overlap between samples) |
| `analysis/paper1/` | Validation read-outs behind Fig. 3(a) |
| `tests/` | Unit tests |
| `docs/` | The metric definitions (`definitions.md`), the rules fixed in advance (`preregistration.md`) and the list of dropped events |
| `results/` | Result tables of the analysis; `results/README.md` names the files behind Tables II to IV and Figs. 2, 3 and 5 and behind other numbers of the text |
| `figures/paper1/` | The five figures of the paper as PNG files |

Inside `src/hnl/`:

- `data/` reads the samples and writes the splits; `truth/` fits helices to the true hits and computes d0 and r_v.
- `graphs/learned_triplet_filter.py` adds the triplet legs to the acorn graph.
- `models/triplet_gnn.py` is the edge classifier with triplet messages, and `models/augment.py` the inner-hit
  truncation used in its training.
- `trackbuild/mutex_ws.py` and `trackbuild/learned_merge.py` are the two stages of LM; `trackbuild/acorn_walk.py`
  is the CC+walk builder of acorn on our graphs.
- `train/` runs the acorn stages; `eval/` holds the matching, the bootstrap (`eval/bootstrap.py`), the working
  points, the truncation probe and the read-outs, with the comparisons fixed in advance in `eval/claims.py`.
- `plots/` holds helpers for validation and comparison plots. The scripts that draw the five figures of the paper
  are not in this snapshot.
- Not every module is used by the paper, and some modules need files that are not in this snapshot.

Names in the code and in the tables differ from those of the paper: Canto appears as `Canto-L` or `new`
(pipeline `l1tl`), acorn (ttbar) as arm (i), acorn (HNL) as arm (ii) or `ref`, CC+walk as `acwalk`, and LM as
`mwslmerge`.

## About this snapshot

This snapshot was prepared for release with the paper. It holds code of the analysis, the rules fixed in advance and
the result tables behind the paper. In preparing it, cluster paths were partly replaced by placeholders, comments were edited,
and modules, checks and job scripts that the paper does not use were left out. The code was not run end to end in
this form, and the tables under `results/` were produced before these edits. Only the unit tests under `tests/` were
run on the code in this form (see Installation).

Some code that produced tables of the paper is also absent. The chain definitions in `scripts/hpc3/` call the modules
`hnl.eval.acorn_long_lm` (behind `results/acorn_long_60ep_lm_val.*`) and `hnl.eval.probe_dense_report` (behind
`results/probe_dense_v1.csv`), which are not in this snapshot, and the scripts that draw the figures are not here
either.

## Installation

The environment of the analysis was built on HPC3 with an earlier form of `env/make_env.sh`, and `env/lock.txt`
records it: Python 3.12, PyTorch 2.11.0 with CUDA 12.6, PyTorch Geometric 2.7.0, Lightning 2.6.1, acorn at commit
`99d15341` with the patches of `patches/acorn/`, and FRNN at a pinned commit. The build needs Linux, GCC 9 to 13
and an NVIDIA GPU. The script here differs from the one that was run: site paths were replaced, and the pip
constraints are now read from `env/lock.txt` by default (or from the file named by `HNL_PIP_CONSTRAINTS`), where the
script that was run read a pip freeze that is not in this snapshot. In this form the script has not been run.

```bash
export HNL_ROOT=/path/to/work       # environment, caches and logs; must not lie under the home directory
export HNL_DATA=/path/to/data       # samples and outputs
bash env/make_env.sh --stage deps            # needs internet access, no GPU
bash env/make_env.sh --stage build,smoke     # on a GPU node
source env/activate.sh
python -m pytest tests -k "not gpu"
```

Notes:

- The default locations in `env/env.sh` and `src/hnl/paths.py` are those used on the cluster; set `HNL_ROOT` and
  `HNL_DATA` to your own. The job scripts of `slurm/hpc3/` expect this repository at `$HNL_ROOT/hnl-displaced`.
  They and the scripts of `scripts/hpc3/` also contain fixed cluster paths (the log directory in the `#SBATCH -o`
  lines, a `cd` in `progress.sbatch`, a path to figure code in `install_tex.sbatch`) and a Slurm account name,
  which these two variables do not change. A few constants in the code are placeholders (`/path/to/...`,
  `PARTITION`, `EXCLUDED_NODES`) that must be set before the code that uses them can run.
- The chain definitions `scripts/hpc3/defs_*.sh` submit job scripts under `slurm/stage1/`, `slurm/feedback/`,
  `slurm/paper1/` and `slurm/diagnostics/`, `slurm/run_py.sbatch` and `analysis/paper1/model_table.py`. None of
  these is in this snapshot, so the job chain cannot be started from it.
- The comments of `env/make_env.sh` name two job scripts that are not in this snapshot; start the stages directly
  as above. The `smoke` stage reads one unpacked HNL event under `$HNL_DATA/unpacked` and writes `env/lock.txt` anew.
- The `From:` line of each file in `patches/acorn/` was replaced for this release, so the SHA-256 sums in
  `patches/acorn/SHA256SUMS` differ from those listed in `env/lock.txt`; the other lines of the patches are those
  that were applied.
- These steps have not been tried from this snapshot, on HPC3 or on another system.
- Tests marked `data` need the unpacked samples and are skipped without them. The tests of `tests/` were run for this
  snapshot on a CPU node of HPC3, from the root of the snapshot, in the environment of the analysis (activated with
  the `env/activate.sh` of the working copy of the analysis, not with the one here) and with `HNL_ROOT` and
  `HNL_DATA` at their cluster defaults. The command was

  ```bash
  PYTHONPATH=$PWD/src PYTHONPYCACHEPREFIX=<directory outside the snapshot> \
      python -m pytest -q $PWD/tests -k "not gpu" -p no:cacheprovider --basetemp=<directory outside the snapshot>
  ```

  457 tests passed, 8 were skipped, 6 failed and 18 were deselected by `-k "not gpu"`. Two of the failures read a
  table under `$HNL_RESULTS` that is neither there nor in this snapshot, one expects three acorn arms where
  `env/env.sh` sets two, and three come from tests that are out of step with the code or with the headers of the
  split lists.

The samples are public ("Large Radius Tracking Events in pp collisions with a Generic detector",
doi:10.5281/zenodo.6812533) and are not included; `NOTICE` gives the full reference.

## Rules fixed in advance and their outcome

`docs/definitions.md` holds the metric definitions. `docs/preregistration.md` holds the bins, the rules for the
working points, the statistics and the three comparisons of Canto with the acorn models under CC; its header states
that it was committed on 3 October 2026, before the first evaluation on a test split. In these two documents "the
paper" means the publication of the acorn pipeline on these samples, not the Canto paper. The preregistration cites
the definitions at a development commit (`89eb354`), and both files name documents such as `DECISIONS.md`,
`RESULTS.md` and `docs/paper_facts.md`. Neither the development history nor these documents are part of this
repository, and `docs/definitions.md` has not been compared with that commit for this release.

`results/claims_test_v1.json` is the outcome file of the three comparisons on the test sets (`primary`, `S1` and
`S1b`), written by `python -m hnl.eval.claims`. It records each difference with its bootstrap and seed terms, and
whether each requirement is met.

The reported analysis and the preregistration differ in these points:

- The preregistration lists three acorn arms and takes the reference baseline from among them. Arm (iii), trained
  on ttbar and HNL events together, is not part of the reported analysis, and the tables here hold arms (i) and
  (ii) only.
- The comparisons fixed in advance name CC only. The rows of the tables with CC+walk or LM as the builder of the
  acorn models are not among them, and `results/strong_baselines_v3.csv` marks the working points that were
  evaluated after the freeze on the stored test scores ("post-freeze").
- Fig. 5 of the paper is drawn in four bins per panel, with a last bin from 100 mm on
  (`results/fig5_rebin_v2.csv`, binnings `rv_4` and `d0_4`). These bins are coarser than those of Section 2 of the
  preregistration. `results/test_binned_v1.csv` holds the bins of Section 2, and `results/fig5_rebin_v1.csv` holds
  an intermediate choice with a last bin from 200 mm on, which the paper does not use.
- Section 7 of the preregistration states that every final model is evaluated on each test set once. This snapshot
  does not document the uses of the test sets.

## What is not in this snapshot

Model checkpoints, simulated events, the log of the job chain and other provenance files are not included, and the
job chain used on the cluster is included only in part. The tables in `results/` therefore cannot be regenerated
from this repository alone. Files in `results/`, files in `docs/` and comments in the code may name files that are
absent.

## Licence

The code is under the MIT License (see `LICENSE`). Third-party material, among it the acorn patches under the
Apache License 2.0, is listed in `NOTICE`.
