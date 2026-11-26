# Result tables

These are result tables of the analysis reported in the paper, which was run on the UCI HPC3 cluster; each file is
an unedited copy of an output of that analysis.

| Table or figure of the paper | Files |
|---|---|
| Table I (events in each split) | no file here; the numbers are the lengths of the lists in `../splits/` |
| Table II (30 against 60 training epochs, HNL validation events) | `acorn_long_60ep_lm_val.md`, with the cuts and efficiencies of the single trainings in `acorn_long_60ep_val.csv` |
| Table III (the effect of each change of Canto, HNL validation events) | `strong_baselines_v3_differences.csv`, `bootstrap_v1.csv` and `acorn_long_60ep_lm_val.md` |
| Table IV (HNL test sets) | `strong_baselines_v3_differences.csv`, `bootstrap_ts_test_3seed_v1.csv` and `bootstrap_v1.csv`, with `training_sample_paired_test_3seed_v1.csv` for the comparisons between training samples and `claims_test_v1.json` for the comparisons fixed in advance (`../docs/preregistration.md`) |
| Fig. 1 (event displays) | no file here |
| Fig. 2 (displacement of the targets) | the histograms in `sample_truth_v1_hist.csv`, the counts and the quoted fractions in `sample_truth_v1.md` |
| Fig. 3 (losses of the standard tracker, truncation probe) | the drawn points in `figures/fig03_bars.csv`; the losses and the probe quoted with it in `paper1_headroom_val_v1_losses.csv` and `probe_dense_v1.csv` |
| Fig. 4 (diagram of the pipeline) | no file here |
| Fig. 5 (efficiency in the HNL test sets) | the drawn points in `figures/fig05_rv_d0_ge100_lines.csv`, read from `fig5_rebin_v2.csv` (four bins per panel, binnings `rv_4` and `d0_4`; paired differences in `fig5_rebin_v2_differences.csv`). The same efficiencies in the finer bins of `../docs/preregistration.md` are in `test_binned_v1.md` and `figures/fig08_test_rv_pts.csv`. `fig5_rebin_v1.csv`, `fig5_rebin_v1_differences.csv` and `figures/fig05_rv_d0_rebinned.csv` hold five bins per panel and are not used in the paper |

Other numbers of the text are read from `working_points_v1.json` (the fake-rate limit FR* and the reference cut),
`ceiling_table2_v1.md` (the graph ceilings), `strong_baselines_v3.csv` (the efficiencies quoted in the abstract) and
`strong_baselines_v3_test_per_set.csv` (the differences per test set).

The other files here are per-seed entries, selections, specifications and checks; `ranking/` holds the ranking tables of
the models on the validation events and on the test sets, with two plots of the validation ranking, and `probe_outer_v3.csv` a further read-out of
the truncation probe. The paper does not use every file.

Paths written inside the files (`/pub/.../hpc3_v1/...`) are locations on the cluster and are not part of this
repository. Some outputs and provenance files of the analysis are not part of this snapshot, among them the log of
the job chain and the reports that accompany several of these tables, so a file here may name a file that is absent
(for example a `.sha256` file without the file it hashes), and this directory is not a complete record of the
analysis.
