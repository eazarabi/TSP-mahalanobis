#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# RDT Experimental Pipeline — 10 models mirroring the main paper's structure
# ══════════════════════════════════════════════════════════════════════════════

set -e

PYTHON="/home/elnur/Desktop/Coding/venv/bin/python -u"
export PYTHONUNBUFFERED=1

LORA_RANK=8
WARM_EPOCHS=15
COLD_EPOCHS=30

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# ─── Timer ───
PIPELINE_START=$(date +%s)

elapsed_since_start() {
    local now=$(date +%s)
    local e=$((now - PIPELINE_START))
    printf "%dm %ds" $((e/60)) $((e%60))
}

# ─── Live epoch progress bar for one training run ───
run_training() {
    local LABEL="$1"
    local TOTAL_EPOCHS="$2"
    shift 2
    local STEP_START=$(date +%s)

    echo "  Training: $LABEL"

    "$@" 2>&1 | while IFS= read -r line; do
        case "$line" in
            Epoch*)
                ep=$(echo "$line" | sed -n 's/.*Epoch *\([0-9]*\).*/\1/p')
                val=$(echo "$line" | sed -n 's/.*val_L=\([0-9.]*\).*/\1/p')
                marker=""
                case "$line" in *best*) marker=" *";; esac
                if [ -n "$ep" ]; then
                    pct=$((ep * 100 / TOTAL_EPOCHS))
                    filled=$((pct / 5))
                    empty=$((20 - filled))
                    bar=$(printf '%0.s█' $(seq 1 $filled))
                    spc=$(printf '%0.s░' $(seq 1 $empty))
                    printf "\r    [%s%s] %3d%% Epoch %d/%d  val_L=%s%s  " "$bar" "$spc" "$pct" "$ep" "$TOTAL_EPOCHS" "$val" "$marker"
                fi
                ;;
        esac
    done
    echo ""

    local STEP_END=$(date +%s)
    local dur=$((STEP_END - STEP_START))
    echo "    Done in $((dur/60))m $((dur%60))s (total: $(elapsed_since_start))"
    echo ""
}

# ─── Helper that picks warm or cold epoch count and launches with progress bar ───
run_one() {
    local metric=$1
    local T=$2
    local init_ckpt=$3
    local extra_flags=$4
    local idx=$5
    local label=$6

    local epochs=$COLD_EPOCHS
    local init_flag=""
    if [ -n "$init_ckpt" ] && [ -f "$init_ckpt" ]; then
        epochs=$WARM_EPOCHS
        init_flag="--init-from $init_ckpt"
    fi

    echo "  [$idx/10] $label (T=$T, $epochs epochs)"
    run_training "$label" "$epochs" \
        $PYTHON experimental/train_rdt.py \
        --metric $metric \
        --thinking-steps $T \
        --lora-rank $LORA_RANK \
        --epochs $epochs \
        $init_flag \
        $extra_flags
}

echo "══════════════════════════════════════════════════════════════════"
echo "  RDT EXPERIMENTAL PIPELINE (10 models)"
echo "══════════════════════════════════════════════════════════════════"
echo "  LoRA rank: $LORA_RANK | Epochs: $WARM_EPOCHS warm / $COLD_EPOCHS cold"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "══════════════════════════════════════════════════════════════════"

mkdir -p experimental/checkpoints experimental/results

# ══════════════════════════════════════════════════════════════════════
# PHASE 1: EUCLIDEAN DEPTH SWEEP (5 models)
# ══════════════════════════════════════════════════════════════════════
echo ""
echo "┌──────────────────────────────────────────────────────────────┐"
echo "│  PHASE 1/4: EUCLIDEAN DEPTH SWEEP (5 models)                 │"
echo "└──────────────────────────────────────────────────────────────┘"
echo ""

EUC_INIT="checkpoints/glimpse_rollout_euclidean_best.pt"
EUC_ENT_INIT="checkpoints/glimpse_rollout_euclidean_ent_best.pt"

run_one euclidean 1 "$EUC_INIT" "" 1 "Euclidean T=1 (control)"
run_one euclidean 2 "$EUC_INIT" "" 2 "Euclidean T=2"
run_one euclidean 4 "$EUC_INIT" "" 3 "Euclidean T=4"
run_one euclidean 8 "$EUC_INIT" "" 4 "Euclidean T=8 (deep probe)"
run_one euclidean 4 "$EUC_ENT_INIT" "--entropy-coef 0.01" 5 "Euclidean T=4 + entropy"

echo "  Phase 1 complete."

# ══════════════════════════════════════════════════════════════════════
# PHASE 2: CROSS-METRIC (2 models)
# ══════════════════════════════════════════════════════════════════════
echo ""
echo "┌──────────────────────────────────────────────────────────────┐"
echo "│  PHASE 2/4: CROSS-METRIC (2 models)                          │"
echo "└──────────────────────────────────────────────────────────────┘"
echo ""

run_one manhattan 4 "checkpoints/glimpse_rollout_manhattan_best.pt" "" 6 "Manhattan T=4"
run_one weighted_euclidean 4 "checkpoints/glimpse_rollout_weighted_euclidean_best.pt" "" 7 "Weighted Euclidean T=4"

echo "  Phase 2 complete."

# ══════════════════════════════════════════════════════════════════════
# PHASE 3: MAHALANOBIS 3-WAY ABLATION (3 models)
# ══════════════════════════════════════════════════════════════════════
echo ""
echo "┌──────────────────────────────────────────────────────────────┐"
echo "│  PHASE 3/4: MAHALANOBIS 3-WAY ABLATION (3 models)            │"
echo "└──────────────────────────────────────────────────────────────┘"
echo ""

run_one mahalanobis 4 "checkpoints/glimpse_rollout_mahalanobis_best.pt" "--no-whitening --no-spatial" 8 "Mahalanobis original T=4"
run_one mahalanobis 4 "checkpoints/glimpse_rollout_mahalanobis_whiten_best.pt" "--no-spatial" 9 "Mahalanobis whitened T=4"
run_one mahalanobis 4 "checkpoints/glimpse_rollout_mahalanobis_whiten_pomo_spatial_best.pt" "" 10 "Mahalanobis whitened+spatial T=4"

echo "  Phase 3 complete."

# ══════════════════════════════════════════════════════════════════════
# PHASE 4: EVALUATION
# ══════════════════════════════════════════════════════════════════════
echo ""
echo "┌──────────────────────────────────────────────────────────────┐"
echo "│  PHASE 4/4: FULL EVALUATION                                  │"
echo "└──────────────────────────────────────────────────────────────┘"
echo ""

echo "  [Eval 1/4] Euclidean (T=1,2,4,8)..."
STEP_START=$(date +%s)
$PYTHON experimental/evaluate_rdt.py --metric euclidean --thinking-steps 1 2 4 8 \
    2>&1 | tee experimental/results/rdt_euclidean_eval.log | grep -E "^(T=| *T |-|  \d)" || true
STEP_END=$(date +%s); echo "    Done in $(( (STEP_END-STEP_START)/60 ))m $(( (STEP_END-STEP_START)%60 ))s"

echo ""
echo "  [Eval 2/4] Manhattan (T=4)..."
STEP_START=$(date +%s)
$PYTHON experimental/evaluate_rdt.py --metric manhattan --thinking-steps 4 \
    2>&1 | tee experimental/results/rdt_manhattan_eval.log | grep -E "^(T=| *T |-|  \d)" || true
STEP_END=$(date +%s); echo "    Done in $(( (STEP_END-STEP_START)/60 ))m $(( (STEP_END-STEP_START)%60 ))s"

echo ""
echo "  [Eval 3/4] Weighted Euclidean (T=4)..."
STEP_START=$(date +%s)
$PYTHON experimental/evaluate_rdt.py --metric weighted_euclidean --thinking-steps 4 \
    2>&1 | tee experimental/results/rdt_weighted_euclidean_eval.log | grep -E "^(T=| *T |-|  \d)" || true
STEP_END=$(date +%s); echo "    Done in $(( (STEP_END-STEP_START)/60 ))m $(( (STEP_END-STEP_START)%60 ))s"

echo ""
echo "  [Eval 4/4] Mahalanobis (T=4)..."
STEP_START=$(date +%s)
$PYTHON experimental/evaluate_rdt.py --metric mahalanobis --thinking-steps 4 \
    2>&1 | tee experimental/results/rdt_mahalanobis_eval.log | grep -E "^(T=| *T |-|  \d)" || true
STEP_END=$(date +%s); echo "    Done in $(( (STEP_END-STEP_START)/60 ))m $(( (STEP_END-STEP_START)%60 ))s"

# ══════════════════════════════════════════════════════════════════════
# DONE
# ══════════════════════════════════════════════════════════════════════
PIPELINE_END=$(date +%s)
TOTAL_TIME=$((PIPELINE_END - PIPELINE_START))
TOTAL_H=$((TOTAL_TIME / 3600))
TOTAL_M=$(( (TOTAL_TIME % 3600) / 60 ))
TOTAL_S=$((TOTAL_TIME % 60))

echo ""
echo "══════════════════════════════════════════════════════════════════"
echo "  RDT PIPELINE COMPLETE"
echo "══════════════════════════════════════════════════════════════════"
echo "  Total time: ${TOTAL_H}h ${TOTAL_M}m ${TOTAL_S}s"
echo ""
echo "  Checkpoints:  experimental/checkpoints/rdt_T*_<metric>*_best.pt"
echo "  Results:      experimental/results/rdt_<metric>_full.json"
echo "  Eval logs:    experimental/results/rdt_<metric>_eval.log"
echo "══════════════════════════════════════════════════════════════════"
