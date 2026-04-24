#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# Encoder-side RDT pipeline — 10 models mirroring main paper structure
# Loops the encoder stack T times (with shared weights + per-iteration LoRA).
# Decoder is unchanged (standard glimpse).
# ══════════════════════════════════════════════════════════════════════════════

set -e
PYTHON="/home/elnur/Desktop/Coding/venv/bin/python -u"
export PYTHONUNBUFFERED=1

LORA_RANK=8
WARM_EPOCHS=15
COLD_EPOCHS=30

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

PIPELINE_START=$(date +%s)
elapsed() { local e=$(($(date +%s) - PIPELINE_START)); printf "%dm %ds" $((e/60)) $((e%60)); }

run_training() {
    local LABEL="$1"; local EPOCHS="$2"; shift 2
    local SS=$(date +%s)
    echo "  Training: $LABEL"
    "$@" 2>&1 | while IFS= read -r line; do
        case "$line" in
            Epoch*)
                ep=$(echo "$line" | sed -n 's/.*Epoch *\([0-9]*\).*/\1/p')
                val=$(echo "$line" | sed -n 's/.*val_L=\([0-9.]*\).*/\1/p')
                m=""; case "$line" in *best*) m=" *";; esac
                if [ -n "$ep" ]; then
                    pct=$((ep * 100 / EPOCHS)); f=$((pct/5)); e=$((20-f))
                    bar=$(printf '%0.s█' $(seq 1 $f)); spc=$(printf '%0.s░' $(seq 1 $e))
                    printf "\r    [%s%s] %3d%% Epoch %d/%d  val_L=%s%s  " "$bar" "$spc" "$pct" "$ep" "$EPOCHS" "$val" "$m"
                fi
                ;;
        esac
    done
    echo ""
    local d=$(($(date +%s) - SS))
    echo "    Done in $((d/60))m $((d%60))s (total: $(elapsed))"
    echo ""
}

run_one() {
    local metric=$1; local T=$2; local init=$3; local extra=$4; local idx=$5; local label=$6
    local epochs=$COLD_EPOCHS; local init_flag=""
    if [ -n "$init" ] && [ -f "$init" ]; then epochs=$WARM_EPOCHS; init_flag="--init-from $init"; fi
    echo "  [$idx/10] $label (T=$T, $epochs epochs)"
    run_training "$label" "$epochs" \
        $PYTHON experimental/train_rdt_any.py --variant encoder \
        --metric $metric --T $T --lora-rank $LORA_RANK --epochs $epochs $init_flag $extra
}

echo "══════════════════════════════════════════════════════════════════"
echo "  ENCODER-RDT PIPELINE (10 models)"
echo "══════════════════════════════════════════════════════════════════"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "══════════════════════════════════════════════════════════════════"

mkdir -p experimental/encoder_rdt/checkpoints experimental/encoder_rdt/results

echo ""
echo "┌──────────────────────────────────────────────────────────────┐"
echo "│  PHASE 1/4: EUCLIDEAN DEPTH SWEEP (5 models)                 │"
echo "└──────────────────────────────────────────────────────────────┘"
echo ""

EUC="checkpoints/glimpse_rollout_euclidean_best.pt"
EUC_ENT="checkpoints/glimpse_rollout_euclidean_ent_best.pt"
run_one euclidean 1 "$EUC" "" 1 "Encoder T=1 (control)"
run_one euclidean 2 "$EUC" "" 2 "Encoder T=2"
run_one euclidean 4 "$EUC" "" 3 "Encoder T=4"
run_one euclidean 8 "$EUC" "" 4 "Encoder T=8"
run_one euclidean 4 "$EUC_ENT" "--entropy-coef 0.01" 5 "Encoder T=4 + entropy"

echo "┌──────────────────────────────────────────────────────────────┐"
echo "│  PHASE 2/4: CROSS-METRIC (2 models)                          │"
echo "└──────────────────────────────────────────────────────────────┘"
echo ""
run_one manhattan 4 "checkpoints/glimpse_rollout_manhattan_best.pt" "" 6 "Manhattan Encoder T=4"
run_one weighted_euclidean 4 "checkpoints/glimpse_rollout_weighted_euclidean_best.pt" "" 7 "Weighted Euc Encoder T=4"

echo "┌──────────────────────────────────────────────────────────────┐"
echo "│  PHASE 3/4: MAHALANOBIS 3-WAY (3 models)                     │"
echo "└──────────────────────────────────────────────────────────────┘"
echo ""
run_one mahalanobis 4 "checkpoints/glimpse_rollout_mahalanobis_best.pt" "--no-whitening --no-spatial" 8 "Mah original Encoder T=4"
run_one mahalanobis 4 "checkpoints/glimpse_rollout_mahalanobis_whiten_best.pt" "--no-spatial" 9 "Mah whitened Encoder T=4"
run_one mahalanobis 4 "checkpoints/glimpse_rollout_mahalanobis_whiten_pomo_spatial_best.pt" "" 10 "Mah whitened+spatial Encoder T=4"

echo "┌──────────────────────────────────────────────────────────────┐"
echo "│  PHASE 4/4: EVALUATION                                       │"
echo "└──────────────────────────────────────────────────────────────┘"
echo ""
for metric in euclidean manhattan weighted_euclidean mahalanobis; do
    echo "  Evaluating $metric..."
    if [ "$metric" = "euclidean" ]; then
        $PYTHON experimental/evaluate_rdt_any.py --variant encoder --metric $metric --T 1 2 4 8 \
            2>&1 | tee experimental/encoder_rdt/results/eval_${metric}.log
    else
        $PYTHON experimental/evaluate_rdt_any.py --variant encoder --metric $metric --T 4 \
            2>&1 | tee experimental/encoder_rdt/results/eval_${metric}.log
    fi
done

TOTAL=$(($(date +%s) - PIPELINE_START))
echo ""
echo "══════════════════════════════════════════════════════════════════"
echo "  ENCODER-RDT PIPELINE COMPLETE"
echo "  Total: $((TOTAL/3600))h $((TOTAL%3600/60))m $((TOTAL%60))s"
echo "══════════════════════════════════════════════════════════════════"
