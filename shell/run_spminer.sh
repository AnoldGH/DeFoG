#!/usr/bin/env bash
# Run the full SPMiner Part 1 pipeline for one dataset.
#
# Step 1 (DeFoG env):   Export training graphs to topology-only NetworkX pickle.
# Step 2 (spminer env): Mine frequent motifs with SPMiner; save patterns pickle.
#
# Run from WSL terminal:
#   bash /mnt/d/DeFoGPlus/DeFoGPlus/shell/run_spminer.sh [dataset]
#
# Supported datasets: qm9 (default), zinc, moses, guacamol
#
# Prerequisites:
#   - conda activate defog    environment must exist (DeFoG)
#   - conda activate spminer  environment must exist (run spminer_setup.sh first)
#   - Training data must already be downloaded (run DeFoG training at least once)

set -e

DATASET="${1:-qm9}"
DEFOG_ROOT="/mnt/d/DeFoGPlus/DeFoGPlus"
SPMINER_ROOT="/mnt/d/DeFoGPlus/neural-subgraph-learning-GNN"
DEFOG_PYTHON=~/miniconda3/envs/defog/bin/python
SPMINER_PYTHON=~/miniconda3/envs/spminer/bin/python

# ── Validate dataset arg ──────────────────────────────────────────────────────
case "$DATASET" in
    qm9)      EXPERIMENT="qm9_no_h" ;;
    zinc)     EXPERIMENT="zinc"     ;;
    moses)    EXPERIMENT="moses"    ;;
    guacamol) EXPERIMENT="guacamol" ;;
    planar)   EXPERIMENT="planar"   ;;
    sbm)      EXPERIMENT="sbm"      ;;
    *)
        echo "ERROR: Unknown dataset '$DATASET'."
        echo "Usage: $0 [qm9|zinc|moses|guacamol|planar|sbm]"
        exit 1
        ;;
esac

NX_GRAPHS_PATH="$DEFOG_ROOT/data/$DATASET/training_graphs_nx.pkl"
MOTIFS_PATH="$DEFOG_ROOT/data/$DATASET/spminer_motifs.pkl"

# ── Set maximum & minimum subgraph size ──────────────────────────────────────

case "$DATASET" in 
    planar) 
        MIN_PATTERN_SIZE=5
        MAX_PATTERN_SIZE=8
        ;;
    *)
        MIN_PATTERN_SIZE=3
        MAX_PATTERN_SIZE=6
        ;;
esac

# ── Step 1: Export training graphs (DeFoG env) ───────────────────────────────
echo "=== Step 1: Exporting $DATASET training graphs to NetworkX ==="
echo "    experiment config : $EXPERIMENT"
echo "    output            : $NX_GRAPHS_PATH"
echo ""

$DEFOG_PYTHON "$DEFOG_ROOT/scripts/export_nx_graphs.py" \
    +experiment="$EXPERIMENT" \
    "dataset=$DATASET" \
    hydra.job.chdir=False \
    "model.spminer_nx_graphs_path=$NX_GRAPHS_PATH"

# ── Step 2: Mine motifs (spminer env) ────────────────────────────────────────
echo ""
echo "=== Step 2: Mining motifs with SPMiner ==="
echo "    input  : $NX_GRAPHS_PATH"
echo "    output : $MOTIFS_PATH"
echo ""

$SPMINER_PYTHON "$DEFOG_ROOT/scripts/mine_motifs.py" \
    --spminer_root  "$SPMINER_ROOT" \
    --nx_graphs_path "$NX_GRAPHS_PATH" \
    --out_path       "$MOTIFS_PATH" \
    --dataset_name   "$DATASET" \
    --max_pattern_size "$MAX_PATTERN_SIZE" \
    --min_pattern_size "$MIN_PATTERN_SIZE"

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "=== Part 1 complete ==="
echo "    NetworkX graphs : $NX_GRAPHS_PATH"
echo "    Mined motifs    : $MOTIFS_PATH"
echo ""
echo "Next (Part 2):"
echo "  1. Add spminer_motifs_path and spminer_top_k to your experiment config."
echo "  2. Run: python scripts/compute_motif_marginals.py \\"
echo "          +experiment=$EXPERIMENT \\"
echo "          hydra.job.chdir=False \\"
echo "          model.transition=motif_edited_spminer \\"
echo "          model.spminer_motifs_path=$MOTIFS_PATH \\"
echo "          model.motif_marginals_path=data/$DATASET/spminer_motif_marginals.pt"
