# HNL-daughter efficiency in |d0| and r_v bins, 3 models of paper Fig. 5 (test_binned_v1)

`python -m hnl.eval.test_binned --split test` (module docstring for the method). Ledger R17 (docs/feedback/test_use_ledger.md): post-freeze, descriptive read-out of stored test outputs; no network run, no track building, nothing chosen on test, no claim.

- Set: pooled HNL test sets (hnl10_100_test + hnl15_200_test + hnl15_100_external); 7999 events read (with ttbar_test for the identity checks); replicate tables: preregistered test tables (B = 500, seed 12345).
- Population: HNL-daughter targets (pT > 1 GeV, |eta| < 4, >= 5 space points); |d0| and r_v bins of docs/definitions.md section 4 (truth-hit fit; r_v = two-track vertex radius, "no vertex" when the pair has none).
- value = mean of the three seeds; combined = sqrt(boot sd^2 + seed sd^2 / 3) (Eq. (2) of the paper); 68 % CP = Clopper-Pearson on the per-seed count.
- Models: Canto-L = Canto-L (frozen: learned merging); acorn (ttbar) LM = acorn (ttbar) + MWS-layer + learned merging (Canto-L's builder); acorn (HNL) LM = acorn (HNL) + MWS-layer + learned merging (Canto-L's builder); working points per seed in the CSV.
- Run: job 59807180, host hpc3-15-21, 2026-11-20T20:55:22Z to 2026-11-20T20:55:28Z; git e63ff6ac7a8960c8893e9bfab84d3d2dccdac45b; ../../../../dfs6b/pub/jinboz1/hnl/hnl-displaced/src/hnl/eval/test_binned.py sha256 b90450c2297876a486526aaa7b3fa8d667947c5ebcb8cb9c58fb444a43bffcb8; slurm/feedback/test_binned.sbatch sha256 c47e49e032bdbcaed91f2b17aeb2b47b41fb49eb13fbf1f08615a32587b9e748.

## Efficiency [%] in the |d0| bins [mm]

Cells: value [68 % CP] (combined sigma, points); n = HNL daughters per seed.

| bin | n | Canto-L | acorn (ttbar) LM | acorn (HNL) LM |
|---|---|---|---|---|
| 0-0.1 | 95 | 100.00 [98.08, 100.00] (0.00) | 100.00 [98.08, 100.00] (0.00) | 100.00 [98.08, 100.00] (0.00) |
| 0.1-1 | 506 | 99.08 [98.43, 99.49] (0.42) | 99.01 [98.35, 99.44] (0.38) | 98.68 [97.96, 99.18] (0.48) |
| 1-3 | 650 | 99.33 [98.84, 99.64] (0.31) | 98.41 [97.75, 98.89] (0.46) | 98.56 [97.93, 99.02] (0.45) |
| 3-10 | 1403 | 99.12 [98.79, 99.37] (0.35) | 98.95 [98.61, 99.22] (0.27) | 99.07 [98.74, 99.33] (0.26) |
| 10-30 | 2013 | 99.32 [99.08, 99.50] (0.20) | 98.18 [97.83, 98.48] (0.26) | 98.34 [98.01, 98.63] (0.27) |
| 30-100 | 2009 | 98.76 [98.46, 99.00] (0.20) | 95.17 [94.64, 95.65] (0.51) | 97.16 [96.74, 97.53] (0.40) |
| 100-300 | 584 | 98.00 [97.24, 98.58] (0.68) | 90.81 [89.45, 92.02] (1.21) | 96.40 [95.46, 97.17] (0.74) |
| >= 300 | 36 | 87.96 [79.91, 93.42] (6.21) | 88.89 [80.97, 94.13] (4.75) | 93.52 [86.45, 97.43] (3.56) |

## Efficiency [%] in the r_v bins [mm]

Cells: value [68 % CP] (combined sigma, points); n = HNL daughters per seed.

| bin | n | Canto-L | acorn (ttbar) LM | acorn (HNL) LM |
|---|---|---|---|---|
| 0-10 | 599 | 99.67 [99.23, 99.88] (0.24) | 99.67 [99.23, 99.88] (0.24) | 99.61 [99.16, 99.85] (0.25) |
| 10-30 | 1050 | 99.49 [99.16, 99.70] (0.23) | 99.05 [98.64, 99.34] (0.30) | 99.14 [98.75, 99.42] (0.24) |
| 30-100 | 2278 | 99.08 [98.83, 99.28] (0.22) | 97.63 [97.27, 97.95] (0.32) | 98.43 [98.13, 98.69] (0.27) |
| 100-200 | 1494 | 98.86 [98.52, 99.13] (0.23) | 96.50 [95.95, 96.97] (0.50) | 98.13 [97.71, 98.48] (0.40) |
| 200-300 | 848 | 99.10 [98.64, 99.41] (0.37) | 95.28 [94.44, 96.01] (0.69) | 97.72 [97.08, 98.23] (0.45) |
| >= 300 | 468 | 99.64 [99.11, 99.89] (0.21) | 96.08 [94.97, 96.98] (0.81) | 97.22 [96.24, 97.98] (0.72) |
| no vertex | 559 | 96.18 [95.19, 96.99] (0.88) | 92.01 [90.69, 93.17] (1.07) | 93.68 [92.47, 94.71] (1.00) |

## Identity checks

- 551 of 551 passed (per-seed entries and model rows of results/strong_baselines_v3_entries.csv and results/strong_baselines_v3.csv; Canto-L against results/bootstrap_v1.csv (test_release_card); bin sums; the same particles for every model).

## Stored outputs read (sha256)

- /pub/jinboz1/hnl_lrt/hpc3_v1/runs/stage1_final_learned/l1tl/armi_s0/trackbuild_test/per_particle.parquet 6f5ee0903efad3993cb2f2eef542f04848bb0f8ec1c3050b317ca090371cd299
- /pub/jinboz1/hnl_lrt/hpc3_v1/runs/stage1_final_learned/l1tl/armi_s0/trackbuild_test/per_event.parquet fb885958b685aec3a860139c827eef5f66d7f2aee863d3aca1bed3da4803151d
- /pub/jinboz1/hnl_lrt/hpc3_v1/runs/stage1_final_learned/l1tl/armi_s1/trackbuild_test/per_particle.parquet 5784a0d0f7d03cc991cfcc982e4fe9b3a03cf4bcb5b50b731e6eb7246c8b8119
- /pub/jinboz1/hnl_lrt/hpc3_v1/runs/stage1_final_learned/l1tl/armi_s1/trackbuild_test/per_event.parquet ba6da70eaa9af9d68dcbb852c96febb4045c53d15f3405730d12d82a8d482847
- /pub/jinboz1/hnl_lrt/hpc3_v1/runs/stage1_final_learned/l1tl/armi_s2/trackbuild_test/per_particle.parquet 369ec6b07458e208941041f6393d723c7ab31a1d4a5e721360ca37e6bd02f32e
- /pub/jinboz1/hnl_lrt/hpc3_v1/runs/stage1_final_learned/l1tl/armi_s2/trackbuild_test/per_event.parquet 9ee7669f0a7920c387960a98241d3126da3f49ccaec7c9ba0c16c310cb64187a
- /pub/jinboz1/hnl_lrt/hpc3_v1/runs/feedback_v3/strong_baselines_a2/armi_s0/lmerge_test/per_particle.parquet d6678c0e0f0c7f2d508b3d609e55aebf7adcc75229067153727bdd6a170ccb25
- /pub/jinboz1/hnl_lrt/hpc3_v1/runs/feedback_v3/strong_baselines_a2/armi_s0/lmerge_test/per_event.parquet 4d5bf5acec33bfe7d67125a879a65494cfdbe833e84b8da6efdab6bb4c1ac376
- /pub/jinboz1/hnl_lrt/hpc3_v1/runs/feedback_v3/strong_baselines_a2/armi_s1/lmerge_test/per_particle.parquet 58ebf99e4402265b057b5a23e4c51acc7238812e86d73842ec89f7b280195ea9
- /pub/jinboz1/hnl_lrt/hpc3_v1/runs/feedback_v3/strong_baselines_a2/armi_s1/lmerge_test/per_event.parquet d64929dee8167004c87ad14e3a8c2d6337dbde37406d9d6cc4494b1871316eca
- /pub/jinboz1/hnl_lrt/hpc3_v1/runs/feedback_v3/strong_baselines_a2/armi_s2/lmerge_test/per_particle.parquet b7212d8f4a539dcff7c59972c7669c1b03aabd22dd652a9d5d812407b664e1ee
- /pub/jinboz1/hnl_lrt/hpc3_v1/runs/feedback_v3/strong_baselines_a2/armi_s2/lmerge_test/per_event.parquet ee6831420f3e5e67373b924ca211c546d2d5b978bd91a1d659fb88e5b81512ec
- /pub/jinboz1/hnl_lrt/hpc3_v1/runs/feedback_v3/strong_baselines_a2/armii_s0/lmerge_test/per_particle.parquet 8be926d2ad0613d2596d0da97c255b50b293de36913f819a43f610405f5417c4
- /pub/jinboz1/hnl_lrt/hpc3_v1/runs/feedback_v3/strong_baselines_a2/armii_s0/lmerge_test/per_event.parquet 6bc14352c463e953a8fc4b18b291f8f8ffe4b52d4f7cc1ce961b2b1a31bf5246
- /pub/jinboz1/hnl_lrt/hpc3_v1/runs/feedback_v3/strong_baselines_a2/armii_s1/lmerge_test/per_particle.parquet 104282b7b6d6f243c49d442b0089428e8f58fa00f1da7742948ce9d654fcfeae
- /pub/jinboz1/hnl_lrt/hpc3_v1/runs/feedback_v3/strong_baselines_a2/armii_s1/lmerge_test/per_event.parquet a6d4373837eab9a266d7c0fc28c47e0214e842e5aae592c5e21d35d1688835f2
- /pub/jinboz1/hnl_lrt/hpc3_v1/runs/feedback_v3/strong_baselines_a2/armii_s2/lmerge_test/per_particle.parquet 883eaa1cd3dd5ec318b05e8f2203fe5af465dada1d6b30a87d752520454586da
- /pub/jinboz1/hnl_lrt/hpc3_v1/runs/feedback_v3/strong_baselines_a2/armii_s2/lmerge_test/per_event.parquet d88e290e887663743567aeb4061407cdd91739314ab218ae2b10926f6daa88d8
