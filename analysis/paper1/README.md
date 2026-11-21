# Validation read-outs behind Fig. 3(a) of the paper

Fig. 3(a) of the paper shows where the standard tracker loses displaced tracks. The quantities behind it exist per
particle in the validation outputs of the analysis; the two scripts in this folder collect and bin them.

Both scripts:

- read validation events only (2,999 events);
- change no selection, cut or model, and are not preregistered;
- write new files under `$HNL_RESULTS` and refuse to write into this repository.

In the analysis they were run in one CPU job, `slurm/paper1/readouts.sbatch`. That job script is not part of this
snapshot.

| step | code | output |
|---|---|---|
| headroom report at the CC cuts of `working_points_v1.json` (`cuts_at_FR_star`) | `python -m hnl.eval.headroom`, then `gather.py` | `$HNL_DERIVED/paper1_v1/headroom/`, copied to `$HNL_RESULTS/paper1_headroom_val_v1_*.csv` and compared with the copy in `$HNL_RUNS/baseline_v1/analysis_val/headroom/` |
| graph-stage ceiling per abs(d0) and r_v bin for acorn (ttbar), acorn (HNL) and Canto (ttbar), three seeds each | `graph_ceiling_bins.py` | `$HNL_RESULTS/paper1_graph_ceiling_val_v1.csv` |

The copies of these tables in this repository are `results/paper1_headroom_val_v1_{losses,losses_per_seed,pairs_per_seed}.csv`
and `results/paper1_graph_ceiling_val_v1.csv`. The graph-ceiling table comes with
`results/paper1_graph_ceiling_val_v1_meta.json` (commit, Slurm job and input files); the `_meta.json` file of the
headroom tables is not part of this snapshot.

`gather.py` also collects edge efficiencies per abs(d0) bin for the six acorn runs (two training samples, three seeds)
and the three Canto runs. The paper does not use them, and their table is not part of this snapshot.
