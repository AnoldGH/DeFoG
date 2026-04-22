#!/usr/bin/env bash
# Precompute motif-edited marginal distributions for Plan A (motif_edited_A).
#
# Ring specs chosen relative to each dataset's molecule size:
#   QM9      (mean  8.8 nodes, max  9): [(6,1)]        — 1×6-ring  (+6  nodes)
#   MOSES    (mean 21.7 nodes, max 27): [(5,1),(6,1)]  — 1×5 + 1×6 (+11 nodes)
#   Guacamol (mean 27.8 nodes, max 88): [(5,1),(6,2)]  — 1×5 + 2×6 (+17 nodes)
#
# Output files are written relative to the repo root (hydra chdir is disabled).
# Run from the repo root:
#   bash shell/approximate_motif_edited_marginal.sh

set -e

PYTHON=~/miniconda3/envs/defog/bin/python
SCRIPT=scripts/compute_motif_marginals.py
COMMON="hydra.job.chdir=False model.transition=motif_edited_A model.motif_num_samples=10000 model.motif_sample_batch_size=256"

echo "=== QM9 (no-H): 1x ring-6 ==="
$PYTHON $SCRIPT \
    +experiment=qm9_no_h \
    $COMMON \
    "model.motif_ring_specs=[[6,1]]" \
    model.motif_marginals_path=data/qm9/motif_marginals_6x1.pt

echo ""
echo "=== MOSES: 1x ring-5 + 1x ring-6 ==="
$PYTHON $SCRIPT \
    +experiment=moses \
    $COMMON \
    "model.motif_ring_specs=[[5,1],[6,1]]" \
    model.motif_marginals_path=data/moses/motif_marginals_5x1_6x1.pt

echo ""
echo "=== Guacamol: 1x ring-5 + 2x ring-6 ==="
$PYTHON $SCRIPT \
    +experiment=guacamol \
    $COMMON \
    "model.motif_ring_specs=[[5,1],[6,2]]" \
    model.motif_marginals_path=data/guacamol/motif_marginals_5x1_6x2.pt

echo ""
echo "Done. Output files:"
echo "  data/qm9/motif_marginals_6x1.pt"
echo "  data/moses/motif_marginals_5x1_6x1.pt"
echo "  data/guacamol/motif_marginals_5x1_6x2.pt"
