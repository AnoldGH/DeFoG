#!/usr/bin/env bash
# Restore the cached defog conda environment from Google Drive.
# Called by 02_session_init.ipynb after Miniconda is installed.
#
# The tarball was created by 01_env_setup.ipynb using conda-pack.
# It is always extracted to the same absolute path that was used during
# packing (/content/miniconda3/envs/defog), so conda-unpack's path
# rewriting is minimal and fast.

set -euo pipefail

TARBALL="/content/drive/MyDrive/DeFoGColab/defog_env.tar.gz"
ENV_DIR="/content/miniconda3/envs/defog"

# ── Validate tarball ──────────────────────────────────────────────────────────
if [ ! -f "$TARBALL" ]; then
    echo "ERROR: environment tarball not found at:" >&2
    echo "  $TARBALL" >&2
    echo "" >&2
    echo "Run 01_env_setup.ipynb first to create and cache the environment." >&2
    exit 1
fi
echo "Tarball found: $(du -sh "$TARBALL" | cut -f1)"

# ── Extract ───────────────────────────────────────────────────────────────────
echo ""
echo "=== Extracting environment (3-5 min depending on Drive read speed) ==="
mkdir -p "$ENV_DIR"
tar -xzf "$TARBALL" -C "$ENV_DIR"
echo "Extraction complete."

# ── Fix hardcoded paths ───────────────────────────────────────────────────────
# conda-unpack rewrites any absolute paths baked into activation scripts and
# compiled Python extensions. Since the extract path matches the original pack
# path, this is fast (mostly a no-op but still required).
echo ""
echo "=== Running conda-unpack ==="
"$ENV_DIR/bin/conda-unpack"
echo "conda-unpack complete."

# ── Quick sanity check ────────────────────────────────────────────────────────
echo ""
echo "=== Environment check ==="
"$ENV_DIR/bin/python" --version
"$ENV_DIR/bin/python" -c "import torch; print('torch', torch.__version__, '| CUDA', torch.cuda.is_available())"
echo "Restore complete."
