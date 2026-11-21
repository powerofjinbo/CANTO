# Primary efficiency per sample: validation and test read out separately

Primary = HNL-daughter targets with |d0| >= 10 mm or r_v >= 100 mm. Every model at its committed per-seed working point (three seeds; nothing is chosen here, on validation or on test): Canto-L (frozen: learned merging; results/final_selection_v1.json); acorn (HNL) and acorn (ttbar) at their val-best builder configuration (acorn (HNL) + val-best builder of the full menu (chosen per seed); acorn (ttbar) + val-best builder of the full menu (chosen per seed); results/strong_baselines_v3_val_entries.csv and results/strong_baselines_v3_spec_test.json); acorn (HNL) + CC at the preregistered cut (the claims' reference) and acorn (ttbar) + CC (preregistered cut = best cut <= FR*). Validation only: the Canto-L classifier (Canto-L's scores on its learned triplet union graph) + CC on the Canto-L runs' eval_val, at the best cut <= FR* = 0.030900 (hnl.eval.ranking rule 'best': highest primary efficiency with fake rate <= FR*; cuts 0.43 / 0.36 / 0.39) and at the cut of the preregistered rule (rule 's4': largest fake rate <= FR*; cuts 0.47 / 0.45 / 0.44), and acorn (HNL) + CC, best cut <= FR*.

Each cell: seed mean ± total sigma, sigma^2 = sigma_boot^2 + s_seeds^2 / 3 (hnl.eval.bootstrap.seed_mean); differences new minus baseline in percentage points with the paired event bootstrap of the difference of seed means plus both seed terms (linear_combination). Validation per sample: each sample's own val events resampled (B = 1000, SeedSequence(20261008), one table per sample shared by every model and seed; the subset construction of hnl.eval.round4_checks); pooled validation: the committed val tables (B = 1000, seed 20261005; both samples' event i together), as results/strong_baselines_v3_val. Test: the preregistered tables (B = 500, seed 12345; hnl10_100_test and hnl15_200_test on the joint units, as hnl.eval.strong_baselines.per_set); post-freeze, descriptive, not claims (results/claims_test_v1.json unchanged). hnl15_100_external (m_N = 15 GeV, c tau = 100 mm) entered no training and no selection.

## Validation

### Primary efficiency (%)

| model | hnl10_100 val (1,000 events; 692 per seed) | hnl15_200 val (999 events; 639 per seed) | pooled HNL val (1,999 events; 1,331 per seed) |
|---|---|---|---|
| Canto-L (frozen: learned merging) | 99.04 ± 0.40 | 97.86 ± 0.55 | 98.47 ± 0.36 |
| acorn (HNL) val-best (MWS-layer + learned merging) | 98.65 ± 0.44 | 97.18 ± 0.57 | 97.95 ± 0.39 |
| acorn (ttbar) val-best (MWS-layer + learned merging) | 97.74 ± 0.48 | 94.94 ± 0.82 | 96.39 ± 0.49 |
| acorn (HNL) + CC, preregistered cut (claims' reference) | 93.30 ± 1.38 | 92.28 ± 1.47 | 92.81 ± 1.32 |
| acorn (ttbar) + CC, preregistered cut (= best cut) | 92.73 ± 1.09 | 89.36 ± 1.04 | 91.11 ± 0.83 |
| Canto-L classifier + CC, best cut <= FR* (validation only) | 98.36 ± 0.39 | 96.87 ± 0.60 | 97.65 ± 0.35 |
| Canto-L classifier + CC, preregistered cut rule (validation only) | 97.98 ± 0.45 | 96.97 ± 0.63 | 97.50 ± 0.39 |
| acorn (HNL) + CC, best cut <= FR* (validation only) | 96.05 ± 0.78 | 94.37 ± 0.80 | 95.24 ± 0.57 |

### Differences (points)

| new | baseline | hnl10_100 val | hnl15_200 val | pooled HNL val |
|---|---|---|---|---|
| Canto-L (frozen: learned merging) | acorn (HNL) val-best (MWS-layer + learned merging) | +0.39 ± 0.54 (z 0.7) | +0.68 ± 0.60 (z 1.1) | +0.53 ± 0.43 (z 1.2) |
| Canto-L (frozen: learned merging) | acorn (ttbar) val-best (MWS-layer + learned merging) | +1.30 ± 0.58 (z 2.3) | +2.92 ± 0.83 (z 3.5) | +2.08 ± 0.51 (z 4.1) |
| Canto-L (frozen: learned merging) | acorn (HNL) + CC, preregistered cut (claims' reference) | +5.73 ± 1.43 (z 4.0) | +5.58 ± 1.45 (z 3.8) | +5.66 ± 1.32 (z 4.3) |
| Canto-L (frozen: learned merging) | acorn (ttbar) + CC, preregistered cut (= best cut) | +6.31 ± 1.14 (z 5.5) | +8.50 ± 1.03 (z 8.3) | +7.36 ± 0.82 (z 9.0) |
| Canto-L classifier + CC, best cut <= FR* (validation only) | acorn (HNL) + CC, best cut <= FR* (validation only) | +2.31 ± 0.81 (z 2.9) | +2.50 ± 0.80 (z 3.1) | +2.40 ± 0.55 (z 4.4) |
| Canto-L classifier + CC, best cut <= FR* (validation only) | acorn (ttbar) + CC, preregistered cut (= best cut) | +5.64 ± 1.08 (z 5.2) | +7.51 ± 1.02 (z 7.4) | +6.54 ± 0.77 (z 8.5) |
| Canto-L classifier + CC, preregistered cut rule (validation only) | acorn (HNL) + CC, preregistered cut (claims' reference) | +4.67 ± 1.40 (z 3.3) | +4.69 ± 1.47 (z 3.2) | +4.68 ± 1.31 (z 3.6) |
| Canto-L classifier + CC, preregistered cut rule (validation only) | acorn (ttbar) + CC, preregistered cut (= best cut) | +5.25 ± 1.10 (z 4.8) | +7.62 ± 1.01 (z 7.5) | +6.39 ± 0.76 (z 8.4) |

### Counts per seed

Last column: the bootstrap sd of the seed mean when the sample is read out of the pooled val table (both samples' event i resampled together) instead of its own table (cross-check of the resampling choice).

| model | set | n per seed | k per seed (s0 / s1 / s2) | efficiency per seed (%) | mean ± sigma (%) | bootstrap sd | seed sd | bootstrap sd, pooled val table |
|---|---|---|---|---|---|---|---|---|
| Canto-L (frozen: learned merging) | hnl10_100 val | 692 | 688 / 685 / 683 | 99.42 / 98.99 / 98.70 | 99.04 ± 0.40 | 0.34 | 0.36 | 0.35 |
| Canto-L (frozen: learned merging) | hnl15_200 val | 639 | 627 / 624 / 625 | 98.12 / 97.65 / 97.81 | 97.86 ± 0.55 | 0.53 | 0.24 | 0.52 |
| Canto-L (frozen: learned merging) | pooled HNL val | 1331 | 1315 / 1309 / 1308 | 98.80 / 98.35 / 98.27 | 98.47 ± 0.36 | 0.32 | 0.28 | - |
| acorn (HNL) val-best (MWS-layer + learned merging) | hnl10_100 val | 692 | 682 / 686 / 680 | 98.55 / 99.13 / 98.27 | 98.65 ± 0.44 | 0.35 | 0.44 | 0.37 |
| acorn (HNL) val-best (MWS-layer + learned merging) | hnl15_200 val | 639 | 622 / 621 / 620 | 97.34 / 97.18 / 97.03 | 97.18 ± 0.57 | 0.56 | 0.16 | 0.57 |
| acorn (HNL) val-best (MWS-layer + learned merging) | pooled HNL val | 1331 | 1304 / 1307 / 1300 | 97.97 / 98.20 / 97.67 | 97.95 ± 0.39 | 0.36 | 0.26 | - |
| acorn (ttbar) val-best (MWS-layer + learned merging) | hnl10_100 val | 692 | 677 / 676 / 676 | 97.83 / 97.69 / 97.69 | 97.74 ± 0.48 | 0.48 | 0.08 | 0.50 |
| acorn (ttbar) val-best (MWS-layer + learned merging) | hnl15_200 val | 639 | 611 / 606 / 603 | 95.62 / 94.84 / 94.37 | 94.94 ± 0.82 | 0.74 | 0.63 | 0.73 |
| acorn (ttbar) val-best (MWS-layer + learned merging) | pooled HNL val | 1331 | 1288 / 1282 / 1279 | 96.77 / 96.32 / 96.09 | 96.39 ± 0.49 | 0.45 | 0.34 | - |
| acorn (HNL) + CC, preregistered cut (claims' reference) | hnl10_100 val | 692 | 661 / 640 / 636 | 95.52 / 92.49 / 91.91 | 93.30 ± 1.38 | 0.81 | 1.94 | 0.80 |
| acorn (HNL) + CC, preregistered cut (claims' reference) | hnl15_200 val | 639 | 605 / 583 / 581 | 94.68 / 91.24 / 90.92 | 92.28 ± 1.47 | 0.84 | 2.08 | 0.88 |
| acorn (HNL) + CC, preregistered cut (claims' reference) | pooled HNL val | 1331 | 1266 / 1223 / 1217 | 95.12 / 91.89 / 91.44 | 92.81 ± 1.32 | 0.64 | 2.01 | - |
| acorn (ttbar) + CC, preregistered cut (= best cut) | hnl10_100 val | 692 | 648 / 646 / 631 | 93.64 / 93.35 / 91.18 | 92.73 ± 1.09 | 0.77 | 1.34 | 0.81 |
| acorn (ttbar) + CC, preregistered cut (= best cut) | hnl15_200 val | 639 | 572 / 572 / 569 | 89.51 / 89.51 / 89.05 | 89.36 ± 1.04 | 1.03 | 0.27 | 1.03 |
| acorn (ttbar) + CC, preregistered cut (= best cut) | pooled HNL val | 1331 | 1220 / 1218 / 1200 | 91.66 / 91.51 / 90.16 | 91.11 ± 0.83 | 0.67 | 0.83 | - |
| Canto-L classifier + CC, best cut <= FR* (validation only) | hnl10_100 val | 692 | 679 / 681 / 682 | 98.12 / 98.41 / 98.55 | 98.36 ± 0.39 | 0.37 | 0.22 | 0.38 |
| Canto-L classifier + CC, best cut <= FR* (validation only) | hnl15_200 val | 639 | 620 / 620 / 617 | 97.03 / 97.03 / 96.56 | 96.87 ± 0.60 | 0.58 | 0.27 | 0.55 |
| Canto-L classifier + CC, best cut <= FR* (validation only) | pooled HNL val | 1331 | 1299 / 1301 / 1299 | 97.60 / 97.75 / 97.60 | 97.65 ± 0.35 | 0.35 | 0.09 | - |
| Canto-L classifier + CC, preregistered cut rule (validation only) | hnl10_100 val | 692 | 679 / 676 / 679 | 98.12 / 97.69 / 98.12 | 97.98 ± 0.45 | 0.42 | 0.25 | 0.44 |
| Canto-L classifier + CC, preregistered cut rule (validation only) | hnl15_200 val | 639 | 620 / 622 / 617 | 97.03 / 97.34 / 96.56 | 96.97 ± 0.63 | 0.58 | 0.39 | 0.57 |
| Canto-L classifier + CC, preregistered cut rule (validation only) | pooled HNL val | 1331 | 1299 / 1298 / 1296 | 97.60 / 97.52 / 97.37 | 97.50 ± 0.39 | 0.38 | 0.11 | - |
| acorn (HNL) + CC, best cut <= FR* (validation only) | hnl10_100 val | 692 | 660 / 664 / 670 | 95.38 / 95.95 / 96.82 | 96.05 ± 0.78 | 0.65 | 0.73 | 0.64 |
| acorn (HNL) + CC, best cut <= FR* (validation only) | hnl15_200 val | 639 | 607 / 599 / 603 | 94.99 / 93.74 / 94.37 | 94.37 ± 0.80 | 0.72 | 0.63 | 0.74 |
| acorn (HNL) + CC, best cut <= FR* (validation only) | pooled HNL val | 1331 | 1267 / 1263 / 1273 | 95.19 / 94.89 / 95.64 | 95.24 ± 0.57 | 0.52 | 0.38 | - |

## Test (post-freeze, descriptive; not claims)

### Primary efficiency (%)

| model | hnl10_100_test (1,000 events; 603 per seed) | hnl15_200_test (1,000 events; 549 per seed) | hnl15_100_external (4,999 events; 3,861 per seed) | pooled HNL test (6,999 events; 5,013 per seed) |
|---|---|---|---|---|
| Canto-L (frozen: learned merging) | 99.12 ± 0.46 | 98.06 ± 0.55 | 98.89 ± 0.16 | 98.82 ± 0.14 |
| acorn (HNL) val-best (MWS-layer + learned merging) | 98.01 ± 0.53 | 96.42 ± 0.76 | 97.61 ± 0.26 | 97.53 ± 0.24 |
| acorn (ttbar) val-best (MWS-layer + learned merging) | 97.07 ± 0.71 | 94.05 ± 1.01 | 96.11 ± 0.30 | 96.00 ± 0.28 |
| acorn (HNL) + CC, preregistered cut (claims' reference) | 92.10 ± 1.84 | 90.16 ± 2.37 | 91.81 ± 1.43 | 91.66 ± 1.52 |
| acorn (ttbar) + CC, preregistered cut (= best cut) | 92.04 ± 1.04 | 87.49 ± 1.25 | 89.72 ± 0.45 | 89.75 ± 0.39 |

### Differences (points)

| new | baseline | hnl10_100_test | hnl15_200_test | hnl15_100_external | pooled HNL test |
|---|---|---|---|---|---|
| Canto-L (frozen: learned merging) | acorn (HNL) val-best (MWS-layer + learned merging) | +1.11 ± 0.60 (z 1.8) | +1.64 ± 0.79 (z 2.1) | +1.28 ± 0.25 (z 5.0) | +1.30 ± 0.23 (z 5.6) |
| Canto-L (frozen: learned merging) | acorn (ttbar) val-best (MWS-layer + learned merging) | +2.05 ± 0.78 (z 2.6) | +4.01 ± 0.92 (z 4.4) | +2.77 ± 0.30 (z 9.3) | +2.82 ± 0.26 (z 10.7) |
| Canto-L (frozen: learned merging) | acorn (HNL) + CC, preregistered cut (claims' reference) | +7.02 ± 1.87 (z 3.8) | +7.89 ± 2.37 (z 3.3) | +7.08 ± 1.43 (z 4.9) | +7.16 ± 1.52 (z 4.7) |
| Canto-L (frozen: learned merging) | acorn (ttbar) + CC, preregistered cut (= best cut) | +7.08 ± 1.10 (z 6.4) | +10.56 ± 1.21 (z 8.8) | +9.17 ± 0.45 (z 20.2) | +9.07 ± 0.39 (z 23.5) |

### Counts per seed

| model | set | n per seed | k per seed (s0 / s1 / s2) | efficiency per seed (%) | mean ± sigma (%) | bootstrap sd | seed sd |
|---|---|---|---|---|---|---|---|
| Canto-L (frozen: learned merging) | hnl10_100_test | 603 | 594 / 600 / 599 | 98.51 / 99.50 / 99.34 | 99.12 ± 0.46 | 0.34 | 0.53 |
| Canto-L (frozen: learned merging) | hnl15_200_test | 549 | 538 / 540 / 537 | 98.00 / 98.36 / 97.81 | 98.06 ± 0.55 | 0.53 | 0.28 |
| Canto-L (frozen: learned merging) | hnl15_100_external | 3861 | 3821 / 3814 / 3819 | 98.96 / 98.78 / 98.91 | 98.89 ± 0.16 | 0.15 | 0.09 |
| Canto-L (frozen: learned merging) | pooled HNL test | 5013 | 4953 / 4954 / 4955 | 98.80 / 98.82 / 98.84 | 98.82 ± 0.14 | 0.14 | 0.02 |
| acorn (HNL) val-best (MWS-layer + learned merging) | hnl10_100_test | 603 | 590 / 591 / 592 | 97.84 / 98.01 / 98.18 | 98.01 ± 0.53 | 0.52 | 0.17 |
| acorn (HNL) val-best (MWS-layer + learned merging) | hnl15_200_test | 549 | 530 / 532 / 526 | 96.54 / 96.90 / 95.81 | 96.42 ± 0.76 | 0.69 | 0.56 |
| acorn (HNL) val-best (MWS-layer + learned merging) | hnl15_100_external | 3861 | 3771 / 3776 / 3759 | 97.67 / 97.80 / 97.36 | 97.61 ± 0.26 | 0.22 | 0.23 |
| acorn (HNL) val-best (MWS-layer + learned merging) | pooled HNL test | 5013 | 4891 / 4899 / 4877 | 97.57 / 97.73 / 97.29 | 97.53 ± 0.24 | 0.20 | 0.22 |
| acorn (ttbar) val-best (MWS-layer + learned merging) | hnl10_100_test | 603 | 588 / 582 / 586 | 97.51 / 96.52 / 97.18 | 97.07 ± 0.71 | 0.65 | 0.51 |
| acorn (ttbar) val-best (MWS-layer + learned merging) | hnl15_200_test | 549 | 517 / 513 / 519 | 94.17 / 93.44 / 94.54 | 94.05 ± 1.01 | 0.96 | 0.56 |
| acorn (ttbar) val-best (MWS-layer + learned merging) | hnl15_100_external | 3861 | 3721 / 3710 / 3702 | 96.37 / 96.09 / 95.88 | 96.11 ± 0.30 | 0.27 | 0.25 |
| acorn (ttbar) val-best (MWS-layer + learned merging) | pooled HNL test | 5013 | 4826 / 4805 / 4807 | 96.27 / 95.85 / 95.89 | 96.00 ± 0.28 | 0.25 | 0.23 |
| acorn (HNL) + CC, preregistered cut (claims' reference) | hnl10_100_test | 603 | 574 / 548 / 544 | 95.19 / 90.88 / 90.22 | 92.10 ± 1.84 | 0.98 | 2.70 |
| acorn (HNL) + CC, preregistered cut (claims' reference) | hnl15_200_test | 549 | 517 / 491 / 477 | 94.17 / 89.44 / 86.89 | 90.16 ± 2.37 | 1.02 | 3.70 |
| acorn (HNL) + CC, preregistered cut (claims' reference) | hnl15_100_external | 3861 | 3650 / 3505 / 3479 | 94.54 / 90.78 / 90.11 | 91.81 ± 1.43 | 0.37 | 2.39 |
| acorn (HNL) + CC, preregistered cut (claims' reference) | pooled HNL test | 5013 | 4741 / 4544 / 4500 | 94.57 / 90.64 / 89.77 | 91.66 ± 1.52 | 0.33 | 2.56 |
| acorn (ttbar) + CC, preregistered cut (= best cut) | hnl10_100_test | 603 | 551 / 557 / 557 | 91.38 / 92.37 / 92.37 | 92.04 ± 1.04 | 0.99 | 0.57 |
| acorn (ttbar) + CC, preregistered cut (= best cut) | hnl15_200_test | 549 | 482 / 481 / 478 | 87.80 / 87.61 / 87.07 | 87.49 ± 1.25 | 1.23 | 0.38 |
| acorn (ttbar) + CC, preregistered cut (= best cut) | hnl15_100_external | 3861 | 3471 / 3450 / 3471 | 89.90 / 89.36 / 89.90 | 89.72 ± 0.45 | 0.42 | 0.31 |
| acorn (ttbar) + CC, preregistered cut (= best cut) | pooled HNL test | 5013 | 4504 / 4488 / 4506 | 89.85 / 89.53 / 89.89 | 89.75 ± 0.39 | 0.37 | 0.20 |

## Validation to test, per labelled sample

Test minus validation (points; the mean of the per-seed changes, seed i the same trained network on both). The validation and test tables are independent: sigma^2 = sigma_boot,val^2 + sigma_boot,test^2 + s^2 / 3, s the seed sd of the per-seed changes; for the change of a difference the bootstrap terms are those of the paired differences on validation and on test and the seed term is s_new^2 / 3 + s_base^2 / 3.

### Change of each model

| model | hnl10_100: val -> test (%) | change (points) | hnl15_200: val -> test (%) | change (points) |
|---|---|---|---|---|
| Canto-L (frozen: learned merging) | 99.04 -> 99.12 | +0.08 ± 0.69 (z 0.1) | 97.86 -> 98.06 | +0.20 ± 0.79 (z 0.2) |
| acorn (HNL) val-best (MWS-layer + learned merging) | 98.65 -> 98.01 | -0.64 ± 0.70 (z -0.9) | 97.18 -> 96.42 | -0.77 ± 0.93 (z -0.8) |
| acorn (ttbar) val-best (MWS-layer + learned merging) | 97.74 -> 97.07 | -0.67 ± 0.85 (z -0.8) | 94.94 -> 94.05 | -0.89 ± 1.32 (z -0.7) |
| acorn (HNL) + CC, preregistered cut (claims' reference) | 93.30 -> 92.10 | -1.21 ± 1.35 (z -0.9) | 92.28 -> 90.16 | -2.12 ± 1.68 (z -1.3) |
| acorn (ttbar) + CC, preregistered cut (= best cut) | 92.73 -> 92.04 | -0.69 ± 1.61 (z -0.4) | 89.36 -> 87.49 | -1.87 ± 1.61 (z -1.2) |

### Change of the difference (Canto-L minus baseline)

| baseline | hnl10_100: val -> test difference (points) | change (points) | hnl15_200: val -> test difference (points) | change (points) |
|---|---|---|---|---|
| acorn (HNL) val-best (MWS-layer + learned merging) | +0.39 -> +1.11 | +0.72 ± 0.88 (z 0.8) | +0.68 -> +1.64 | +0.96 ± 0.98 (z 1.0) |
| acorn (ttbar) val-best (MWS-layer + learned merging) | +1.30 -> +2.05 | +0.74 ± 1.01 (z 0.7) | +2.92 -> +4.01 | +1.09 ± 1.26 (z 0.9) |
| acorn (HNL) + CC, preregistered cut (claims' reference) | +5.73 -> +7.02 | +1.29 ± 1.47 (z 0.9) | +5.58 -> +7.89 | +2.31 ± 1.67 (z 1.4) |
| acorn (ttbar) + CC, preregistered cut (= best cut) | +6.31 -> +7.08 | +0.77 ± 1.71 (z 0.4) | +8.50 -> +10.56 | +2.06 ± 1.57 (z 1.3) |

## Checks

631 of 631 identity checks passed (tolerance 1e-12, counts exact; results/per_sample_v1_checks.csv): pooled val and test rows and differences reproduce results/strong_baselines_v3*; the per-set test rows reproduce hnl.eval.strong_baselines.per_set for every pair and results/strong_baselines_v3_test_per_set.csv; per-sample counts add up to the pooled counts for every model and seed; working points, cuts and test-output provenance as committed; the Canto-L classifier + CC entries of rule 's4' equal the reference read-out of that row (meta: cc_reference; it was made with hnl.eval.ranking's default rule, 's4'), every column.

Slurm job 59807179, 2026-11-20T20:55:22Z - 2026-11-20T20:57:43Z (UTC), git e63ff6ac7a8960c8893e9bfab84d3d2dccdac45b.

