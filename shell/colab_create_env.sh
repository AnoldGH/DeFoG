#!/usr/bin/env bash
# Build the defog conda environment from scratch on a Colab GPU instance.
# Called by 01_env_setup.ipynb after Miniforge is installed.
#
# Install order is intentional and must not be changed:
#   1. conda: graph-tool, rdkit, numpy, scipy, matplotlib, networkx, conda-pack
#      Anything that conda-forge resolves transitively for graph-tool/rdkit
#      MUST live in this layer. If pip installs an older version of the same
#      package afterwards, it will partially overwrite conda's files and
#      create a broken hybrid (conda's compiled extensions expect the newer
#      ABI, but pip leaves stale bytecode/headers).
#   2. pip: PyTorch CUDA wheel
#      Uses Colab's live CUDA driver. conda-forge's pytorch carries its own
#      full CUDA stack (3-5 GB extra in the tarball for no benefit).
#   3. pip: remaining Python deps (must NOT re-pin anything from Step 2)

set -euo pipefail

CONDA=/content/miniconda3/bin/conda
PYTHON=/content/miniconda3/envs/defog/bin/python
# NOTE: invoke pip as "$PYTHON -m pip" throughout — the bin/pip script's
# shebang line can be broken after conda-unpack relocates the environment.

# ── Detect Colab's CUDA version ───────────────────────────────────────────────
# Prefer nvcc (exact toolkit version); fall back to nvidia-smi driver report.
if command -v nvcc &>/dev/null; then
    CUDA_VERSION=$(nvcc --version | grep -oP 'release \K[\d.]+' | head -1)
else
    CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[\d.]+' | head -1 || echo "12.1")
fi
CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d. -f2)
echo "Detected CUDA ${CUDA_MAJOR}.${CUDA_MINOR}"

# PyTorch 2.4.0 official CUDA wheel variants: cu118, cu121, cu124.
# CUDA 12.2 / 12.3 drivers are backward-compatible with the cu121 wheel.
if   [ "$CUDA_MAJOR" -ge 12 ] && [ "$CUDA_MINOR" -ge 4 ]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu124"
elif [ "$CUDA_MAJOR" -ge 12 ]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu121"
else
    TORCH_INDEX="https://download.pytorch.org/whl/cu118"
fi
echo "PyTorch wheel index: $TORCH_INDEX"

# ── Step 1: Create the conda environment ──────────────────────────────────────
echo ""
echo "=== Step 1: Creating conda environment (Python 3.9) ==="
$CONDA create -n defog python=3.9 -y

# ── Step 2: conda-forge packages ──────────────────────────────────────────────
# Every package conda-forge installs as a transitive dep of graph-tool/rdkit
# is pinned here so pip cannot silently downgrade it later.
#
#   numpy >= 1.25     — graph_tool 2.45 needs numpy._utils
#   scipy >= 1.12     — torch_geometric needs scipy._lib._array_api
#   matplotlib >= 3.8 — graph_tool's drawing module needs backends.registry
#   networkx >= 3.0   — graph-tool ecosystem expects modern networkx API
echo ""
echo "=== Step 2: Installing conda-forge packages ==="
echo "    (This is the slowest step — ~10 min to solve + download.)"
$CONDA install -n defog -c conda-forge \
    "graph-tool=2.45" \
    "rdkit=2023.03.2" \
    "numpy>=1.25,<2.0" \
    "scipy>=1.12" \
    "matplotlib>=3.8,<4.0" \
    "networkx>=3.0" \
    "conda-pack" \
    -y

# ── Step 3: PyTorch via pip ───────────────────────────────────────────────────
echo ""
echo "=== Step 3: Installing PyTorch 2.4.0 (CUDA wheel) ==="
$PYTHON -m pip install torch==2.4.0 torchvision --index-url "$TORCH_INDEX"

# ── Step 4: Remaining pip dependencies ───────────────────────────────────────
# DO NOT add numpy, scipy, matplotlib, or networkx here — they are managed
# by conda (Step 2). Re-pinning them via pip would recreate the hybrid bug.
#
# Version notes:
#   torch_geometric 2.3.1   — matches DeFoG's original environment
#   pytorch-lightning 2.0.4 — stable with torch 2.4
#   torchmetrics 0.11.4     — last version compatible with lightning 2.0.x
#   pandas 2.0.3            — first version ABI-safe with numpy >= 1.25
#   scikit-learn 1.3.2      — compatible with numpy >= 1.25, scipy >= 1.12
#   imageio 2.30+           — required by analysis.visualization
echo ""
echo "=== Step 4: Installing remaining pip dependencies ==="
$PYTHON -m pip install \
    "torch_geometric==2.3.1" \
    "pytorch-lightning==2.0.4" \
    "torchmetrics==0.11.4" \
    "hydra-core==1.3.2" \
    "omegaconf==2.3.0" \
    "wandb==0.15.4" \
    "scikit-learn==1.3.2" \
    "pandas==2.0.3" \
    "seaborn==0.13.2" \
    "tqdm==4.65.0" \
    "pyemd==1.0.0" \
    "PyGSP==0.5.1" \
    "imageio>=2.30" \
    "Pillow" \
    "PyYAML==6.0.2" \
    "gpustat==0.6.0"

# ── Step 5: Smoke test ────────────────────────────────────────────────────────
# IMPORTANT: graph_tool MUST be imported after torch's libgomp is preloaded,
# otherwise torch's bundled libgomp.so.1 (older) shadows conda's version and
# graph_tool fails with "GOMP_5.0 not found". The notebook handles this via
# LD_PRELOAD in RUN_ENV; here we set it inline for the smoke test only.
echo ""
echo "=== Step 5: Smoke test ==="
LD_PRELOAD=/content/miniconda3/envs/defog/lib/libgomp.so.1 \
$PYTHON -c "
import torch, graph_tool, rdkit, scipy, numpy, matplotlib, networkx, imageio
print('torch     ', torch.__version__)
print('graph_tool', graph_tool.__version__)
print('rdkit     ', rdkit.__version__)
print('numpy     ', numpy.__version__)
print('scipy     ', scipy.__version__)
print('matplotlib', matplotlib.__version__)
print('networkx  ', networkx.__version__)
print('imageio   ', imageio.__version__)
print('All imports OK.')
"
