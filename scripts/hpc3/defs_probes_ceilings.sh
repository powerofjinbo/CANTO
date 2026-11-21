#!/bin/bash
# HPC3 chain definitions of the read-outs C5-C10:
# truncation probes (Fig. 3(b,c); FB28, FB31, FB35, FB36, FF09, NV26, DP01, DP02), the perfect-classifier ceilings
# (CE01, CE02, PF04a/b, PF06a), the Fig. 3(a) read-outs (PF03), the model table (MD03, G09, G12) and the 60-epoch acorn
# check (V07, EP01, EP05, EP08, EP09, EP12, EP16, FE26a, FE26m).
#
#     bash scripts/hpc3/defs_probes_ceilings.sh          define the steps (scripts/hpc3/chain.py add; nothing submitted)
#     DRY=1 bash scripts/hpc3/defs_probes_ceilings.sh    print the add commands only
#
# Run it from the main checkout (chain.py records the working directory, and the jobs run the scripts of
# $HNL_REPO). Markers are written only on success: by the scripts themselves (truncation_probe
# summary.csv, probe meta.json, trackbuild .done, ceiling block meta.json, the copied events.json and plan file, both
# written atomically) or by run_py.sbatch / readouts.sbatch / model_table.py at the end (MARKER / --marker).
# The probes resume from OUT/partial after a preemption or a time limit.
# Prerequisites defined elsewhere: sbv3_val (strong_baselines_v3 val entries, spec, differences),
# data_bootstrap_replicates (bootstrap tables; bootstrap_tables.OUT_RUNS must point into HNL_RUNS).
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
source "$HERE/env/env.sh" || exit 1
set -euo pipefail
cd "$HNL_REPO"
if [ -z "${DRY:-}" ] && ! grep -q "def copy_events" src/hnl/eval/probe_outer.py; then
    echo "defs_probes_ceilings: $HNL_REPO does not hold hnl.eval.probe_outer copy-events"; exit 1
fi
C="python3 -B scripts/hpc3/chain.py"
add() { if [ -n "${DRY:-}" ]; then echo "$C add $*"; else $C add "$@"; fi; }

R=$HNL_RUNS
PO=$R/feedback_v1/probe_outer_v1
PD=$R/feedback_v1/probe_dense_v1
MK=$HNL_OUT/markers
WP=$HNL_RESULTS/working_points_v1.json
SEEDS="0 1 2"

# --- C10 truncation probes -----------------------------------------------------------------------------------------
# the probe's 300 HNL val events: an input definition (event list), copied after checking the ids in the tar
add --order 500 --after bl_i_s0_ml_infer --marker "$PO/events.json" probe_events_copy -- \
    -p free -c 1 --mem=4G -t 00:30:00 slurm/run_py.sbatch hnl.eval.probe_outer copy-events \
    --src "$HNL_DATA/runs/feedback_v1/probe_outer_v1/events.json" --dst "$PO/events.json" \
    --tar "$R/baseline_v1/armi_s0/ml_graphs/valset.tar"

# original inner probe (RUN/truncation_probe/summary.csv; the probe_outer report's reproduction check).
tp() {  # key run after
    add --order 505 --after "$3" --marker "$2/truncation_probe/summary.csv" "$1" -- \
        -p free -c 8 --mem=24G -t 02:00:00 slurm/stage1/truncation_probe.sbatch "$2"
}
tp trunc_probe_bl_i_s0 "$R/baseline_v1/armi_s0" bl_i_s0_gnn_train,bl_i_s0_ml_infer,data_truth_lookup
tp trunc_probe_bl_i_s1 "$R/baseline_v1/armi_s1" bl_i_s1_gnn_train,bl_i_s1_ml_infer,data_truth_lookup
tp trunc_probe_bl_ii_s0 "$R/baseline_v1/armii_s0" bl_ii_s0_gnn_train,bl_ii_s0_ml_infer,data_truth_lookup
tp trunc_probe_l1_scr_i_s0 "$R/stage1_screen_ttbar/l1/armi_s0" probe_ctrl_l1_scr_i_s0,bl_i_s0_ml_infer,data_truth_lookup
tp trunc_probe_ignn_trunc_scr_i_s0 "$R/stage1_screen_ttbar/ignn_trunc/armi_s0" \
    probe_ctrl_ignn_trunc_scr_i_s0,bl_i_s0_ml_infer,data_truth_lookup

# mirror / endpoint probe, modes inner, outer, middle at m = 0, 2, 4 (probe_outer.sbatch: -c 16 --mem=16G)
po() {  # tag run graphs time after
    add --order 510 --after "probe_events_copy,$5" --marker "$PO/$1/meta.json" "po_$1" -- \
        -p free -c 16 --mem=16G -t "$4" slurm/stage1/probe_outer.sbatch "$1" "$2" "$3"
}
po acorn_ttbar_s0 "$R/baseline_v1/armi_s0" "$R/baseline_v1/armi_s0/ml_graphs/valset.tar" 03:00:00 \
    bl_i_s0_gnn_train,bl_i_s0_ml_infer,data_truth_lookup
po acorn_ttbar_s1 "$R/baseline_v1/armi_s1" "$R/baseline_v1/armi_s1/ml_graphs/valset.tar" 01:00:00 \
    bl_i_s1_gnn_train,bl_i_s1_ml_infer,data_truth_lookup
po acorn_ttbar_s2 "$R/baseline_v1/armi_s2" "$R/baseline_v1/armi_s2/ml_graphs/valset.tar" 01:30:00 \
    bl_i_s2_gnn_train,bl_i_s2_ml_infer,data_truth_lookup
po acorn_hnl_s0 "$R/baseline_v1/armii_s0" "$R/baseline_v1/armii_s0/ml_graphs/valset.tar" 01:00:00 \
    bl_ii_s0_gnn_train,bl_ii_s0_ml_infer,data_truth_lookup
for S in $SEEDS; do
    T=$([ "$S" = 0 ] && echo 08:00:00 || echo 02:00:00)
    po "canto_l_s$S" "$R/stage1_final_learned/l1tl/armi_s$S" "$R/baseline_v1/armi_s$S/ltf_union_graphs/valset.tar" "$T" \
        "canto_l1tl_i_s$S,canto_union_i_s$S,data_truth_lookup"
done
po l1_scr "$R/stage1_screen_ttbar/l1/armi_s0" "$R/baseline_v1/armi_s0/ml_graphs/valset.tar" 01:30:00 \
    probe_ctrl_l1_scr_i_s0,bl_i_s0_ml_infer,data_truth_lookup
po ignn_trunc_scr "$R/stage1_screen_ttbar/ignn_trunc/armi_s0" "$R/baseline_v1/armi_s0/ml_graphs/valset.tar" 01:00:00 \
    probe_ctrl_ignn_trunc_scr_i_s0,bl_i_s0_ml_infer,data_truth_lookup
po ignn147_full "$R/stage1_capacity/ignn147/armi_s0" "$R/baseline_v1/armi_s0/ml_graphs/valset.tar" 02:30:00 \
    probe_ctrl_ignn147_i_s0,bl_i_s0_ml_infer,data_truth_lookup
PO_KEYS=po_acorn_ttbar_s0,po_acorn_ttbar_s1,po_acorn_ttbar_s2,po_acorn_hnl_s0,po_canto_l_s0,po_canto_l_s1,po_canto_l_s2
PO_KEYS=$PO_KEYS,po_l1_scr,po_ignn_trunc_scr,po_ignn147_full
TP_KEYS=trunc_probe_bl_i_s0,trunc_probe_bl_i_s1,trunc_probe_bl_ii_s0,trunc_probe_l1_scr_i_s0,trunc_probe_ignn_trunc_scr_i_s0
add --order 530 --after "$PO_KEYS,$TP_KEYS" --marker "$MK/po_report_v3.done" po_report_v3 -- \
    -p free -c 2 --mem=8G -t 00:30:00 --export=ALL,MARKER="$MK/po_report_v3.done" slurm/run_py.sbatch \
    hnl.eval.probe_outer_report --base "$PO" --out "$HNL_RESULTS/probe_outer_v3"

# dense probe, modes inner and outer at m = 0-4 (probe_dense.sbatch: -c 16 --mem=16G). acorn: one job per model;
# Canto-L: one job per mode (TAG_inner / TAG_outer; the report joins the halves).
pd() {  # key tag run graphs time after [modes]
    add --order 520 --after "probe_events_copy,$6" --marker "$PD/$2/meta.json" "$1" -- \
        -p free -c 16 --mem=16G -t "$5" slurm/stage1/probe_dense.sbatch "$2" "$3" "$4" ${7:+"$7"}
}
PD_KEYS=""
for S in $SEEDS; do
    pd "pd_acorn_ttbar_s$S" "acorn_ttbar_s$S" "$R/baseline_v1/armi_s$S" "$R/baseline_v1/armi_s$S/ml_graphs/valset.tar" \
        01:30:00 "bl_i_s${S}_gnn_train,bl_i_s${S}_ml_infer,data_truth_lookup"
    pd "pd_acorn_hnl_s$S" "acorn_hnl_s$S" "$R/baseline_v1/armii_s$S" "$R/baseline_v1/armii_s$S/ml_graphs/valset.tar" \
        01:30:00 "bl_ii_s${S}_gnn_train,bl_ii_s${S}_ml_infer,data_truth_lookup"
    T=$([ "$S" = 0 ] && echo 10:00:00 || echo 02:30:00)
    # 2026-11-18: no Canto-L dense-probe jobs (the paper prints no Canto-L dense row; Fig. 3(b,c) draws acorn only)
    for M in $([ "${HNL_PD_CANTO_L:-0}" = 1 ] && echo inner outer); do
        pd "pd_canto_l_s${S}_$M" "canto_l_s${S}_$M" "$R/stage1_final_learned/l1tl/armi_s$S" \
            "$R/baseline_v1/armi_s$S/ltf_union_graphs/valset.tar" "$T" \
            "canto_l1tl_i_s$S,canto_union_i_s$S,data_truth_lookup" "$M"
    done
    PD_KEYS="$PD_KEYS,pd_acorn_ttbar_s$S,pd_acorn_hnl_s$S"
done
add --order 531 --after "po_report_v3${PD_KEYS}" --marker "$MK/pd_report.done" pd_report -- \
    -p free -c 2 --mem=8G -t 00:30:00 --export=ALL,MARKER="$MK/pd_report.done" slurm/run_py.sbatch \
    hnl.eval.probe_dense_report --base "$PD" --v1 "$PO" --v3 "$HNL_RESULTS/probe_outer_v3.csv" \
    --out "$HNL_RESULTS/probe_dense_v1" --models acorn_ttbar,acorn_hnl

# --- C8 / C9 the 60-epoch acorn check --------------------------------------------------------------------------------
TB30_KEYS="" TB60_KEYS="" EVAL_KEYS="" IGNN60_KEYS="" LM60_KEYS=""
for A in i ii; do
    for S in $SEEDS; do
        # one 30-epoch grid on all six runs
        add --order 540 --after "bl_${A}_s${S}_gnn_infer,data_truth_lookup,data_outlier_hits" \
            --marker "$R/baseline_v1/arm${A}_s$S/trackbuild_val_mwslayer.done" "acorn30_tb_mwslayer_${A}_s$S" -- \
            -p free -c 8 --mem=16G -t 01:00:00 slurm/stage1/trackbuild.sbatch "$R/baseline_v1/arm${A}_s$S" \
            cc,walkmergeS,mwslayer,mwslayermerge trackbuild_val_mwslayer
        # 60 epochs
        add --order 541 --after "ignn60_${A}_s$S,data_truth_lookup,data_outlier_hits" \
            --marker "$R/stage1_acorn_long/ignn60/arm${A}_s$S/trackbuild_val.done" "acorn60_tb_${A}_s$S" -- \
            -p free -c 8 --mem=16G -t 01:30:00 slurm/stage1/trackbuild.sbatch "$R/stage1_acorn_long/ignn60/arm${A}_s$S" \
            walkmergeS,mwslayer,mwslayermerge trackbuild_val
        TB30_KEYS="$TB30_KEYS,acorn30_tb_mwslayer_${A}_s$S"
        TB60_KEYS="$TB60_KEYS,acorn60_tb_${A}_s$S"
        EVAL_KEYS="$EVAL_KEYS,bl_${A}_s${S}_eval"
        IGNN60_KEYS="$IGNN60_KEYS,ignn60_${A}_s$S"
        LM60_KEYS="$LM60_KEYS,acorn60_lm_${A}_s$S"
    done
done
add --order 545 --after "w1_gate${EVAL_KEYS}${IGNN60_KEYS}${TB30_KEYS}${TB60_KEYS}" \
    --marker "$MK/ro_acorn_long_compare.done" ro_acorn_long_compare -- \
    -p free -c 4 --mem=24G -t 01:30:00 --export=ALL,MARKER="$MK/ro_acorn_long_compare.done" slurm/run_py.sbatch \
    hnl.eval.acorn_long_compare armi,armii "$HNL_RESULTS/acorn_long_60ep_val.csv" --wp "$WP"
add --order 546 --after "w1_gate,sbv3_val,data_bootstrap_replicates${EVAL_KEYS}${IGNN60_KEYS}${LM60_KEYS}" \
    --marker "$MK/ro_acorn_long_lm.done" ro_acorn_long_lm -- \
    -p free -c 4 --mem=32G -t 01:30:00 --export=ALL,MARKER="$MK/ro_acorn_long_lm.done" slurm/run_py.sbatch \
    hnl.eval.acorn_long_lm "$HNL_RESULTS/acorn_long_60ep_lm_val" --arms ii,i

# --- C7 model table ---------------------------------------------------------------------------------------------
MT_AFTER="canto_l1tl_ii_s0${EVAL_KEYS}"
for S in $SEEDS; do MT_AFTER="$MT_AFTER,canto_l1tl_i_s$S,canto_l1t_base_i_s$S,canto_union_i_s$S,canto_union_ii_s$S"; done
add --order 550 --after "$MT_AFTER" --marker "$MK/ro_model_table.done" ro_model_table -- \
    -p free -c 2 --mem=16G -t 00:30:00 slurm/hpc3/check.sbatch analysis/paper1/model_table.py \
    --out "$HNL_RESULTS/model_table_v1.md" --marker "$MK/ro_model_table.done"

# --- C6 Fig. 3(a) read-outs: headroom, graph-ceiling bins, edge TPR at e*, gather (9 runs) --------------------
RO_AFTER="w1_gate${EVAL_KEYS}"
for A in i ii; do for S in $SEEDS; do RO_AFTER="$RO_AFTER,bl_${A}_s${S}_gnn_infer"; done; done
for S in $SEEDS; do RO_AFTER="$RO_AFTER,canto_l1tl_i_s$S,canto_estar_i_s$S"; done
add --order 555 --after "$RO_AFTER" --marker "$MK/ro_paper1_readouts.done" ro_paper1_readouts -- \
    -p free -c 8 --mem=24G -t 03:00:00 --export=ALL,MARKER="$MK/ro_paper1_readouts.done" slurm/paper1/readouts.sbatch

# --- C5 perfect-classifier ceilings (plan from the sbv3 val entries; one step per run; summary) --------------------
CT=$R/ceiling_table2_v1
CT_AFTER="sbv3_val"
for A in i ii; do for S in $SEEDS; do CT_AFTER="$CT_AFTER,acorn_lmerge_${A}_s$S"; done; done
for S in $SEEDS; do CT_AFTER="$CT_AFTER,canto_l1tl_i_s${S}_lmerge"; done
add --order 560 --after "$CT_AFTER" --marker "$CT/plan.txt" ro_ceiling_table2_plan -- \
    -p free -c 1 --mem=4G -t 00:20:00 slurm/run_py.sbatch hnl.eval.ceiling_table2 plan --out "$CT/plan.txt"
CT_KEYS=""
ct() {  # key plan_tag after
    add --order 561 --after "ro_ceiling_table2_plan,data_truth_lookup,$3" --marker "$CT/$2/block_0of1/meta.json" "$1" -- \
        -p free -c 16 --mem=32G -t 01:00:00 --export=ALL,PLAN="$CT/plan.txt",PLAN_TAG="$2" \
        slurm/diagnostics/ceiling_table2.sbatch
    CT_KEYS="$CT_KEYS,$1"
}
for S in $SEEDS; do
    ct "ro_ceiling_table2_acorn_ttbar_s$S" "acorn_ttbar_s$S" "bl_i_s${S}_gnn_infer,acorn_lmerge_i_s$S"
    ct "ro_ceiling_table2_acorn_hnl_s$S" "acorn_HNL_s$S" "bl_ii_s${S}_gnn_infer,acorn_lmerge_ii_s$S"
    ct "ro_ceiling_table2_cantol_ttbar_s$S" "Canto-L_ttbar_s$S" "canto_l1tl_i_s$S,canto_l1tl_i_s${S}_lmerge"
done
add --order 562 --after "sbv3_val,data_bootstrap_replicates${CT_KEYS}" --marker "$MK/ro_ceiling_table2.done" \
    ro_ceiling_table2 -- -p free -c 4 --mem=32G -t 01:00:00 --export=ALL,MARKER="$MK/ro_ceiling_table2.done" \
    slurm/run_py.sbatch hnl.eval.ceiling_table2 summarize --base "$CT" --out "$HNL_RESULTS/ceiling_table2_v1"
