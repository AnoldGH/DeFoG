#!/usr/bin/env bash
# Create the isolated SPMiner conda environment in WSL.
#
# Run from WSL terminal:
#   bash /mnt/d/DeFoGPlus/DeFoGPlus/shell/spminer_setup.sh
#
# Package selection rationale:
#   Pinned (must match ckpt/model.pt and each other's C extensions):
#     torch==1.4.0, torch_geometric==1.4.3, deepsnap==0.1.2
#   Latest with Python 3.7 binary wheels (avoids build-from-source failures):
#     numpy 1.21.6, scipy 1.7.3, scikit-learn 1.0.2, matplotlib 3.5.3
#   Pure Python (any recent version works):
#     networkx, seaborn, tqdm, test_tube

set -e

# ── Locate conda shell integration ───────────────────────────────────────────
# conda is not in PATH for non-login bash sessions; probe known locations first.
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    CONDA_BASE="$HOME/miniconda3"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    CONDA_BASE="$HOME/anaconda3"
elif command -v conda &>/dev/null; then
    CONDA_BASE=$(conda info --base)
else
    echo "ERROR: conda not found. Expected \$HOME/miniconda3 or \$HOME/anaconda3."
    exit 1
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"

# ── Environment creation ─────────────────────────────────────────────────────
echo "=== Creating conda environment: spminer (Python 3.7) ==="
conda create -n spminer python=3.7 -y
conda activate spminer

# ── Step 1: PyTorch 1.4.0 (CPU wheel, pinned to match ckpt/model.pt) ────────
echo ""
echo "=== Step 1: PyTorch 1.4.0 (CPU) ==="
pip install \
    torch==1.4.0 \
    torchvision==0.5.0 \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html

# ── Step 2: PyTorch Geometric 1.4.3 C++ extensions ──────────────────────────
# Wheel filenames encode the exact torch version; must match step 1 precisely.
echo ""
echo "=== Step 2: PyTorch Geometric 1.4.3 ==="
pip install \
    torch-scatter==2.0.4 \
    torch-sparse==0.6.1 \
    torch-cluster==1.5.2 \
    torch-spline-conv==1.2.0 \
    -f https://pytorch-geometric.com/whl/torch-1.4.0+cpu.html
pip install torch-geometric==1.4.3

# ── Step 3: DeepSNAP (pinned, pure Python, API-compatible with PyG 1.4.3) ───
echo ""
echo "=== Step 3: DeepSNAP 0.1.2 ==="
pip install deepsnap==0.1.2

# ── Step 4: Scientific Python — latest versions with py37 binary wheels ──────
# numpy 1.21.6 and scipy 1.7.3 are the last releases shipping cp37 manylinux
# wheels. scikit-learn 1.0.2 is the last with a cp37 wheel.
echo ""
echo "=== Step 4: Scientific Python (binary wheels) ==="
pip install \
    "numpy==1.21.6" \
    "scipy==1.7.3" \
    "scikit-learn==1.0.2"

# ── Step 5: Visualisation + utilities ────────────────────────────────────────
# matplotlib 3.5.3 is the last release with a cp37 manylinux wheel.
# networkx, seaborn, tqdm, test_tube are pure Python — install latest stable.
echo ""
echo "=== Step 5: Visualisation and utilities ==="
pip install \
    "matplotlib==3.5.3" \
    "networkx>=2.4,<3.0" \
    seaborn \
    tqdm \
    test_tube

# ── Verification ─────────────────────────────────────────────────────────────
echo ""
echo "=== Verifying installation ==="
python - <<'EOF'
import pkg_resources
import torch, torch_geometric, deepsnap, networkx, numpy, scipy, sklearn, matplotlib
print("  torch           ", torch.__version__)
print("  torch_geometric ", torch_geometric.__version__)
print("  deepsnap        ", pkg_resources.get_distribution("deepsnap").version)
print("  networkx        ", networkx.__version__)
print("  numpy           ", numpy.__version__)
print("  scipy           ", scipy.__version__)
print("  scikit-learn    ", sklearn.__version__)
print("  matplotlib      ", matplotlib.__version__)
print("OK - all packages imported successfully.")
EOF

echo ""
echo "Setup complete. Activate with: conda activate spminer"
