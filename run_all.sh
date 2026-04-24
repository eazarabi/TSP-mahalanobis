#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# TSP Transformer RL — Full Pipeline Script
# ══════════════════════════════════════════════════════════════════════════════
# Trains all models, runs evaluation, generates results for the report.
#
# Usage:
#   chmod +x run_all.sh
#   ./run_all.sh
# ══════════════════════════════════════════════════════════════════════════════

set -e

# ─── Configuration ───
PYTHON="/home/elnur/Desktop/Coding/venv/bin/python -u"
EPOCHS=100
TRAIN_SIZE=50000
export PYTHONUNBUFFERED=1

# ─── Timer ───
PIPELINE_START=$(date +%s)

elapsed_since_start() {
    local now=$(date +%s)
    local e=$((now - PIPELINE_START))
    printf "%dm %ds" $((e/60)) $((e%60))
}

# ─── Training wrapper: shows live epoch progress bar ───
run_training() {
    local LABEL="$1"
    shift
    local STEP_START=$(date +%s)

    echo "  Training: $LABEL"

    # Run training, show live epoch progress
    "$@" 2>&1 | while IFS= read -r line; do
        case "$line" in
            Epoch*)
                # Extract epoch number and val_L from line like:
                # "Epoch  42/100 | train_L=... | val_L=15.4108 | ..."
                ep=$(echo "$line" | sed -n 's/.*Epoch *\([0-9]*\).*/\1/p')
                val=$(echo "$line" | sed -n 's/.*val_L=\([0-9.]*\).*/\1/p')
                marker=""
                case "$line" in *best*) marker=" *";; esac
                if [ -n "$ep" ]; then
                    pct=$((ep * 100 / EPOCHS))
                    filled=$((pct / 5))
                    empty=$((20 - filled))
                    bar=$(printf '%0.s█' $(seq 1 $filled))
                    spc=$(printf '%0.s░' $(seq 1 $empty))
                    printf "\r    [%s%s] %3d%% Epoch %d/%d  val_L=%s%s  " "$bar" "$spc" "$pct" "$ep" "$EPOCHS" "$val" "$marker"
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

echo "══════════════════════════════════════════════════════════════════"
echo "  TSP TRANSFORMER RL — FULL TRAINING & EVALUATION PIPELINE"
echo "══════════════════════════════════════════════════════════════════"
echo "  Python: $PYTHON"
echo "  Epochs: $EPOCHS | Train size: $TRAIN_SIZE"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "══════════════════════════════════════════════════════════════════"
echo ""

# ─── Clean old results ───
echo "Cleaning old results..."
rm -f results/*.json results/*.png results/*.txt
echo "  Done."
echo ""

# ══════════════════════════════════════════════════════════════════
# PHASE 1: EUCLIDEAN ARCHITECTURE ABLATION (5 models)
# ══════════════════════════════════════════════════════════════════
echo "┌──────────────────────────────────────────────────────────────┐"
echo "│  PHASE 1/4: EUCLIDEAN ARCHITECTURE ABLATION (5 models)       │"
echo "└──────────────────────────────────────────────────────────────┘"
echo ""

echo "  [1/5]"
run_training "Simple + Rollout (Euclidean)" \
    $PYTHON train.py --metric euclidean --baseline rollout \
    --epochs $EPOCHS --train_size $TRAIN_SIZE \
    --no-pomo --no-spatial --no-whiten

echo "  [2/5]"
run_training "Simple + Critic (Euclidean)" \
    $PYTHON train.py --metric euclidean --baseline critic \
    --epochs $EPOCHS --train_size $TRAIN_SIZE \
    --no-pomo --no-spatial --no-whiten

echo "  [3/5]"
run_training "Glimpse + Rollout (Euclidean)" \
    $PYTHON train.py --metric euclidean --glimpse --baseline rollout \
    --epochs $EPOCHS --train_size $TRAIN_SIZE \
    --no-pomo --no-spatial --no-whiten

echo "  [4/5]"
run_training "Glimpse + Critic (Euclidean)" \
    $PYTHON train.py --metric euclidean --glimpse --baseline critic \
    --epochs $EPOCHS --train_size $TRAIN_SIZE \
    --no-pomo --no-spatial --no-whiten

echo "  [5/5]"
run_training "Glimpse + Rollout + Entropy (Euclidean)" \
    $PYTHON -c "
import sys; sys.path.insert(0, '.')
from config import Config
from train import train_improved

cfg = Config()
cfg.distance_metric = 'euclidean'
cfg.data_distribution = 'uniform'
cfg.num_epochs = $EPOCHS
cfg.train_size = $TRAIN_SIZE
cfg.entropy_coef = 0.01
cfg.use_whitening = False
cfg.use_spatial_encoding = False

train_improved(cfg, use_glimpse=True, baseline_type='rollout',
               run_name='glimpse_rollout_euclidean_ent', use_pomo=False)
"

echo "  Phase 1 complete."
echo ""

# ══════════════════════════════════════════════════════════════════
# PHASE 2: CROSS-METRIC COMPARISON (2 models)
# ══════════════════════════════════════════════════════════════════
echo "┌──────────────────────────────────────────────────────────────┐"
echo "│  PHASE 2/4: CROSS-METRIC COMPARISON (2 models)               │"
echo "└──────────────────────────────────────────────────────────────┘"
echo ""

echo "  [1/2]"
run_training "Glimpse + Rollout (Manhattan)" \
    $PYTHON train.py --metric manhattan --glimpse --baseline rollout \
    --epochs $EPOCHS --train_size $TRAIN_SIZE \
    --no-pomo --no-spatial --no-whiten

echo "  [2/2]"
run_training "Glimpse + Rollout (Weighted Euclidean)" \
    $PYTHON train.py --metric weighted_euclidean --glimpse --baseline rollout \
    --epochs $EPOCHS --train_size $TRAIN_SIZE \
    --no-pomo --no-spatial --no-whiten

echo "  Phase 2 complete."
echo ""

# ══════════════════════════════════════════════════════════════════
# PHASE 3: MAHALANOBIS — Original + Improved (2 models)
# ══════════════════════════════════════════════════════════════════
echo "┌──────────────────────────────────────────────────────────────┐"
echo "│  PHASE 3/4: MAHALANOBIS ORIGINAL + IMPROVED (2 models)       │"
echo "└──────────────────────────────────────────────────────────────┘"
echo ""

echo "  [1/2]"
run_training "Glimpse + Rollout (Mahalanobis ORIGINAL)" \
    $PYTHON train.py --metric mahalanobis --glimpse --baseline rollout \
    --epochs $EPOCHS --train_size $TRAIN_SIZE \
    --no-pomo --no-spatial --no-whiten

echo "  [2/2]"
run_training "Glimpse + Rollout (Mahalanobis IMPROVED — whitening + spatial)" \
    $PYTHON train.py --metric mahalanobis --glimpse --baseline rollout \
    --epochs $EPOCHS --train_size $TRAIN_SIZE \
    --no-pomo --no-spatial

echo "  Phase 3 complete."
echo ""

# ══════════════════════════════════════════════════════════════════
# PHASE 4: EVALUATION
# ══════════════════════════════════════════════════════════════════
echo "┌──────────────────────────────────────────────────────────────┐"
echo "│  PHASE 4/4: FULL EVALUATION                                  │"
echo "└──────────────────────────────────────────────────────────────┘"
echo ""
echo "  Running: evaluate.py (all metrics, all decoding strategies, OR-Tools)"
echo ""

$PYTHON evaluate.py 2>&1 | tee results/evaluation_log.txt | while IFS= read -r line; do
    # Show key progress lines
    if echo "$line" | grep -qE "^(  METRIC:|Evaluating|  Greedy:|  Best-of|  POMO|  8x|  Beam|  Nearest|  2-opt|  OR-Tools|    Mean)"; then
        echo "  $line"
    fi
done

echo ""
echo "  Phase 4 complete."
echo ""

# ══════════════════════════════════════════════════════════════════
# DONE
# ══════════════════════════════════════════════════════════════════
PIPELINE_END=$(date +%s)
TOTAL_TIME=$((PIPELINE_END - PIPELINE_START))
TOTAL_H=$((TOTAL_TIME / 3600))
TOTAL_M=$(( (TOTAL_TIME % 3600) / 60 ))
TOTAL_S=$((TOTAL_TIME % 60))

echo "══════════════════════════════════════════════════════════════════"
echo "  ALL DONE"
echo "══════════════════════════════════════════════════════════════════"
echo ""
echo "  Total time: ${TOTAL_H}h ${TOTAL_M}m ${TOTAL_S}s"
echo ""
echo "  Output:"
echo "    results/evaluation_full.json     — Numerical results"
echo "    results/tours_*.png              — Tour visualizations"
echo "    results/attention_*.png          — Attention heatmaps"
echo "    results/sample_efficiency_*.png  — Learning curves"
echo "    results/length_distribution_*.png"
echo "══════════════════════════════════════════════════════════════════"
