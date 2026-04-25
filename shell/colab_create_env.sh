#!/usr/bin/env bash
# Build the defog conda environment from scratch on a Colab GPU instance.
#
# Design principle (learned the hard way after a dozen hybrid-install bugs):
#   "If a package is on conda-forge, conda owns it. If it isn't, pip does.
#    Never both."
# Pinning a pip version of a package that conda-forge installs transitively
# creates a broken hybrid (Python files from one version, .so files from
# another). Symptoms: `numpy._utils` missing, `scipy._lib._array_api`
# missing, `matplotlib.backends.registry` missing, `pandas._libs.pandas_parser`
# missing — all the same root cause.
#
# Layout:
#   Step 1: Use mamba if available (much faster solver), else conda.
#   Step 2: One conda install — every conda-forge-resolvable dep at once,
#           no version pins. Lets the solver pick a self-consistent set.
#   Step 3: pip — only DeFoG-specific deps that aren't on conda-forge.
#           Loose ranges, never exact pins on anything conda might own.

set -euo pipefail

CONDA=/content/miniconda3/bin/conda
MAMBA=/content/miniconda3/bin/mamba
PYTHON=/content/miniconda3/envs/defog/bin/python

# Pick the fastest available solver.
if [ -x "$MAMBA" ]; then
    SOLVER="$MAMBA"
    echo "Using mamba (fast solver)."
else
    SOLVER="$CONDA"
    echo "mamba not found, using conda."
fi

# ── Detect Colab's CUDA version (for PyTorch wheel selection) ─────────────────
if command -v nvcc &>/dev/null; then
    CUDA_VERSION=$(nvcc --version | grep -oP 'release \K[\d.]+' | head -1)
else
    CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[\d.]+' | head -1 || echo "12.1")
fi
CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d. -f2)
echo "Detected CUDA ${CUDA_MAJOR}.${CUDA_MINOR}"

if   [ "$CUDA_MAJOR" -ge 12 ] && [ "$CUDA_MINOR" -ge 4 ]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu124"
elif [ "$CUDA_MAJOR" -ge 12 ]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu121"
else
    TORCH_INDEX="https://download.pytorch.org/whl/cu118"
fi
echo "PyTorch wheel index: $TORCH_INDEX"

# ── Step 1: Create env (skip if already present from a prior run) ─────────────
echo ""
echo "=== Step 1: Creating conda environment (Python 3.9) ==="
if [ -d /content/miniconda3/envs/defog ]; then
    echo "defog env already exists, skipping create."
else
    "$SOLVER" create -n defog python=3.9 -y
fi

# ── Step 2: Install everything conda-forge can resolve, in one solve ──────────
# No version pins — let the solver pick a self-consistent set.
# Anything that has a chance of being a transitive dep of graph-tool/rdkit
# is listed explicitly to prevent pip from later trying to manage it.
echo ""
echo "=== Step 2: Installing conda-forge packages (one solve) ==="
"$SOLVER" install -n defog -c conda-forge --override-channels \
    graph-tool \
    rdkit \
    numpy \
    scipy \
    scikit-learn \
    matplotlib \
    networkx \
    pandas \
    seaborn \
    pillow \
    pyyaml \
    imageio \
    tqdm \
    -y

# ── Step 3: pip — DeFoG-specific deps not on conda-forge ──────────────────────
# Use $PYTHON -m pip (not the bin/pip script) because pip's shebang line can be
# unreliable. Loose lower bounds throughout — pip should never override the
# conda-installed numpy/scipy/etc., so dep resolution will pick whatever's
# compatible.
echo ""
echo "=== Step 3: pip — PyTorch CUDA wheel ==="
$PYTHON -m pip install --upgrade pip
$PYTHON -m pip install "torch==2.4.0" "torchvision" --index-url "$TORCH_INDEX"

echo ""
echo "=== Step 3: pip — DeFoG framework deps ==="
$PYTHON -m pip install \
    "torch_geometric==2.3.1" \
    "pytorch-lightning==2.0.4" \
    "torchmetrics==0.11.4" \
    "hydra-core>=1.3,<1.4" \
    "omegaconf>=2.3,<2.4" \
    "wandb>=0.15" \
    "pyemd" \
    "PyGSP" \
    "gpustat"

# ── Step 4: Smoke test ────────────────────────────────────────────────────────
# MPLBACKEND=Agg overrides Jupyter's inherited inline backend (which doesn't
# exist in the conda env). LD_PRELOAD forces conda's libgomp to load before
# torch's bundled one, otherwise graph_tool errors on `GOMP_5.0 not found`.
echo ""
echo "=== Step 4: Smoke test ==="
MPLBACKEND=Agg \
LD_PRELOAD=/content/miniconda3/envs/defog/lib/libgomp.so.1 \
$PYTHON -c "
import torch, graph_tool, rdkit, scipy, numpy, matplotlib, networkx, pandas, imageio
import torch_geometric, pytorch_lightning, hydra, wandb, torchmetrics
print('torch            ', torch.__version__)
print('graph_tool       ', graph_tool.__version__)
print('rdkit            ', rdkit.__version__)
print('numpy            ', numpy.__version__)
print('scipy            ', scipy.__version__)
print('matplotlib       ', matplotlib.__version__)
print('networkx         ', networkx.__version__)
print('pandas           ', pandas.__version__)
print('imageio          ', imageio.__version__)
print('torch_geometric  ', torch_geometric.__version__)
print('pytorch_lightning', pytorch_lightning.__version__)
print('torchmetrics     ', torchmetrics.__version__)
print('CUDA available   :', torch.cuda.is_available())
print('All imports OK.')
"
