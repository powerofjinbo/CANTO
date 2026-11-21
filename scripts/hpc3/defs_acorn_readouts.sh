#!/bin/bash
# HPC3 chain definitions (scripts/hpc3/chain.py) of the acorn read-out layer: strong_baselines_v3 for
# acorn (ttbar) / (HNL) with CC, CC + walkthrough (acwalk) and learned merging (LM), its Canto-L / Canto-B rows, the
# in-scope ranking v3 / ranking_test_v1, the acorn CC* post-freeze test, and the claims / seed-table copies into
# $HNL_RESULTS. Defines steps only: nothing is submitted here (the
# watcher's `tick` submits them). Run from the repo: bash scripts/hpc3/defs_acorn_readouts.sh [--replace]
#
# Order of the steps (keys; "after" lists in each add):
#   acorn_acwalkx_{i,ii}_s{0,1,2}   acorn CCandWalk lower-edge grid (trackbuild_val_acwalk_ext)
#   acorn_acwalk_all_merge          trackbuild_val_acwalk_all = acwalk + acwalk_ext on the six runs
#   sbv3_ext                        the grid-extension loop (inventory / ext grids until every choice is interior)
#   ro_seeds_copy                   frozen selection's seed table -> $HNL_RESULTS/final_selection_v1_seeds.csv
#   ro_ranking_val_v3               in-scope val ranking v3 ($HNL_RESULTS/ranking/{spec_val_v3.json,ranking_val_v3*})
#   acorn_cc_bestcut_test           acorn (HNL) / (ttbar) CC at the val best cut on the stored test scores
#   sbv3_val                        fb3_val: merge, val (runs ONCE: it refuses after test outputs exist), jobs
#   acorn_lm_test_{i,ii}_s{0,1,2}   fb3_test of the val-chosen LM configuration (NAME=@: read from the spec)
#   acorn_acwalk_test_{i,ii}_s{0,1,2}  the same for CC + walkthrough
#   ro_claims_copy                  $HNL_RUNS/final_test_v1/claims/claims.json -> $HNL_RESULTS/claims_test_v1.json
#   ro_ranking_test_v1              ranking_test spec + rank ($HNL_RESULTS/ranking/{spec_test_v1.json,ranking_test_v1*})
#   sbv3_test                       fb3_summary: test tables, differences, md
source "$(dirname "${BASH_SOURCE[0]}")/../../env/env.sh" || exit 1
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
C="python3 -B scripts/hpc3/chain.py"
REPL=${1:-}
add() { $C add ${REPL:+--replace} "$@"; }
M=$HNL_OUT/markers
RES=$HNL_RESULTS
BASE=$HNL_RUNS/baseline_v1
FB3=$HNL_RUNS/feedback_v3/strong_baselines_a2
SPEC=$RES/strong_baselines_v3_spec_test.json
ARMS="i ii"
SEEDS="0 1 2"
csv() { local IFS=,; echo "$*"; }
each() { local p=$1 x=$2 out=(); for a in $ARMS; do for s in $SEEDS; do out+=("${p}_${a}_s${s}${x}"); done; done; csv "${out[@]}"; }
canto() { local p=$1 x=$2 out=(); for s in $SEEDS; do out+=("${p}_i_s${s}${x}"); done; csv "${out[@]}"; }

# B.2: acorn's CCandWalk, lower-edge extension of the val grid (acwalkX: acwalk and acwalkg, 8 configurations each)
for a in $ARMS; do for s in $SEEDS; do
    RUN=$BASE/arm${a}_s${s}
    add --order 160 --after bl_${a}_s${s}_gnn_infer,data_truth_lookup \
        --marker "$RUN/trackbuild_val_acwalk_ext.done" acorn_acwalkx_${a}_s${s} -- \
        -c 8 --mem=16G -t 03:00:00 slurm/feedback/acwalk_val.sbatch "$RUN" acwalkX trackbuild_val_acwalk_ext
done; done

# B.3: the acwalk grid union of the six runs (hnl.eval.strong_baselines merge: arms ii, i; a data copy, idempotent)
add --order 230 --after "$(each acorn_acwalk ""),$(each acorn_acwalkx "")" \
    --marker "$M/acorn_acwalk_all_merge.done" acorn_acwalk_all_merge -- \
    --export=ALL,MARKER=$M/acorn_acwalk_all_merge.done -c 2 --mem=16G -t 01:00:00 \
    slurm/run_py.sbatch hnl.eval.strong_baselines merge

# B.4: the data-dependent extension loop as one CPU step (scripts/hpc3/sbv3_ext_driver.py runs every grid itself)
add --order 231 --after "acorn_acwalk_all_merge,$(each acorn_lmerge ""),w1_gate,bl_ii_s0_eval,data_truth_lookup" \
    --marker "$M/sbv3_ext.done" sbv3_ext -- \
    -c 16 --mem=64G -t 24:00:00 slurm/hpc3/check.sbatch scripts/hpc3/sbv3_ext_driver.py --marker "$M/sbv3_ext.done"

# C: the frozen selection's seed table next to the selection (canto_select_gate copies only the JSON)
add --order 232 --after canto_select_gate --marker "$M/ro_seeds_copy.done" ro_seeds_copy -- \
    -c 1 --mem=2G -t 00:20:00 slurm/hpc3/check.sbatch scripts/hpc3/ro_copy.py --marker "$M/ro_seeds_copy.done" \
    --same "$RES/final_selection_v1.json" "$RES/candidate/final_selection_v1.json" \
    "$RES/candidate/final_selection_v1_seeds.csv" "$RES/final_selection_v1_seeds.csv"

# C1a: the val ranking v3 (acorn (ttbar) / (HNL) CC at both cuts, Canto-B and Canto-L R6)
add --order 233 --after "w1_gate,$(each bl _eval),$(canto canto_l1tl _lmerge),$(canto canto_l1t_base _lmerge)" \
    --marker "$M/ro_ranking_val_v3.done" ro_ranking_val_v3 -- \
    --export=ALL,MARKER=$M/ro_ranking_val_v3.done -c 4 --mem=32G -t 02:00:00 slurm/run_py.sbatch hnl.eval.ranking \
    --build-spec v3-hpc3 --spec "$RES/ranking/spec_val_v3.json" --fr-wp "$RES/working_points_v1.json" \
    --out "$RES/ranking" --name ranking_val_v3 --tie-se

# B.7: acorn CC at its val best cut <= FR* on the stored test scores (only the cuts eval_test does not hold)
add --order 234 --after "w1_gate,$(each bl _eval),$(each bl _test_eval),data_outlier_hits,ro_ranking_val_v3" \
    --marker "$M/acorn_cc_bestcut_test.done" acorn_cc_bestcut_test -- \
    --export=ALL,MARKER=$M/acorn_cc_bestcut_test.done -c 8 --mem=32G -t 03:00:00 \
    slurm/run_py.sbatch hnl.eval.published_defs bestcut --arms ii,i --seeds 0,1,2

# B.8: val (merge of the grid parts, the val tables / differences / edges, the val-fixed test spec); runs once
add --order 235 --after "$(each acorn_lmerge ""),acorn_acwalk_all_merge,sbv3_ext,$(each bl _eval),$(each bl _test_eval),w1_gate,$(canto canto_l1tl _lmerge),$(canto canto_l1t_base _lmerge),$(canto canto_l1t_base ""),canto_select_gate,ro_seeds_copy,ro_ranking_val_v3,acorn_cc_bestcut_test" \
    --marker "$M/sbv3_val.done" sbv3_val -- \
    --export=ALL,MARKER=$M/sbv3_val.done -c 4 --mem=32G -t 04:00:00 slurm/feedback/fb3_val.sbatch

# B.8: the post-freeze test outputs of the val-chosen LM and CC + walkthrough configurations (one per run)
for a in $ARMS; do for s in $SEEDS; do
    RUN=$BASE/arm${a}_s${s}
    add --order 236 --after sbv3_val,bl_${a}_s${s}_test_eval,acorn_lmerge_${a}_s${s} \
        --marker "$FB3/arm${a}_s${s}/lmerge_test/provenance.json" acorn_lm_test_${a}_s${s} -- \
        --export=ALL,SPEC=$SPEC -c 8 --mem=32G -t 04:00:00 slurm/feedback/fb3_test.sbatch "$RUN" "$FB3/arm${a}_s${s}" \
        lmerge_test=@
    add --order 236 --after sbv3_val,bl_${a}_s${s}_test_eval,acorn_acwalk_all_merge \
        --marker "$FB3/arm${a}_s${s}/acwalk_test/provenance.json" acorn_acwalk_test_${a}_s${s} -- \
        --export=ALL,SPEC=$SPEC -c 8 --mem=32G -t 04:00:00 slurm/feedback/fb3_test.sbatch "$RUN" "$FB3/arm${a}_s${s}" \
        acwalk_test=@
done; done

# C: the claims under the name the read-outs read
add --order 237 --after canto_claims --marker "$M/ro_claims_copy.done" ro_claims_copy -- \
    -c 1 --mem=2G -t 00:20:00 slurm/hpc3/check.sbatch scripts/hpc3/ro_copy.py --marker "$M/ro_claims_copy.done" \
    "$HNL_RUNS/final_test_v1/claims/claims.json" "$RES/claims_test_v1.json"

# C1b: the test release card (Canto-L, acorn (HNL) / (ttbar) CC at both cuts), identity checks against the claims
add --order 238 --after "ro_ranking_val_v3,acorn_cc_bestcut_test,$(canto canto_final_test ""),$(each bl _test_eval),canto_select_gate,w1_gate,ro_claims_copy" \
    --marker "$M/ro_ranking_test_v1.done" ro_ranking_test_v1 -- \
    --export=ALL,MARKER=$M/ro_ranking_test_v1.done -c 4 --mem=32G -t 02:00:00 slurm/run_py.sbatch hnl.eval.ranking_test \
    all --spec "$RES/ranking/spec_test_v1.json" --out "$RES/ranking" --name ranking_test_v1 \
    --claims "$RES/claims_test_v1.json"

# B.8: the test tables, differences and markdown of strong_baselines_v3
add --order 239 --after "sbv3_val,$(each acorn_lm_test ""),$(each acorn_acwalk_test ""),$(each bl _test_eval),acorn_cc_bestcut_test,$(canto canto_final_test ""),canto_claims,ro_claims_copy,ro_ranking_test_v1" \
    --marker "$M/sbv3_test.done" sbv3_test -- \
    --export=ALL,MARKER=$M/sbv3_test.done -c 4 --mem=32G -t 03:00:00 slurm/feedback/fb3_summary.sbatch
