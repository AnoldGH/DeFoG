"""Mine frequent subgraph motifs from DeFoG training graphs using SPMiner.

Runs in the isolated SPMiner conda environment (spminer) via WSL.
Loads topology-only nx.Graph objects produced by export_nx_graphs.py,
calls SPMiner's pattern_growth pipeline, and saves the discovered motif
patterns to a pickle file consumed by Part 2 (motif_edited_spminer).

The output file contains list[nx.Graph] where each graph has a node
attribute "anchor" (1 for the seed/attachment node, 0 for all others).

Usage (from WSL, in the spminer conda environment):
    python /mnt/d/DeFoGPlus/DeFoGPlus/scripts/mine_motifs.py \\
        --nx_graphs_path /mnt/d/DeFoGPlus/DeFoGPlus/data/qm9/training_graphs_nx.pkl \\
        --out_path       /mnt/d/DeFoGPlus/DeFoGPlus/data/qm9/spminer_motifs.pkl \\
        --dataset_name   qm9

Implementation notes:
  - use_whole_graphs is intentionally NOT used: it triggers a NameError in
    pattern_growth (start_time undefined). Instead, neighborhood sizes are
    tuned per dataset to stay within each dataset's graph size range.
  - matplotlib is switched to the Agg (non-interactive) backend before any
    import of pyplot, which is required in headless WSL environments.
  - The script changes CWD to the SPMiner repo root so that all relative
    paths inside pattern_growth (plots/cluster/, ckpt/model.pt, results/)
    resolve correctly.
"""
import argparse
import os
import pickle
import sys
from types import SimpleNamespace

# Must set non-interactive backend before pyplot is imported anywhere.
import matplotlib
matplotlib.use("Agg")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="SPMiner motif mining for DeFoG")
    p.add_argument(
        "--spminer_root",
        default="/mnt/d/DeFoGPlus/neural-subgraph-learning-GNN",
        help="Absolute path to the SPMiner repo root in WSL",
    )
    p.add_argument(
        "--nx_graphs_path",
        required=True,
        help="Path to training_graphs_nx.pkl produced by export_nx_graphs.py",
    )
    p.add_argument(
        "--out_path",
        required=True,
        help="Output path for mined motif patterns (.pkl)",
    )
    p.add_argument(
        "--dataset_name",
        default="qm9",
        choices=["qm9", "zinc", "moses", "guacamol"],
        help="Dataset name — controls neighborhood size defaults",
    )
    p.add_argument(
        "--model_path",
        default="ckpt/model.pt",
        help="Path to pretrained SPMiner checkpoint, relative to spminer_root",
    )
    # Search parameters
    p.add_argument("--min_pattern_size", type=int, default=3,
                   help="Minimum motif size (nodes)")
    p.add_argument("--max_pattern_size", type=int, default=6,
                   help="Maximum motif size (nodes)")
    p.add_argument("--n_neighborhoods", type=int, default=5000,
                   help="Number of node neighborhoods to embed (must be "
                        "divisible by --emb_batch_size)")
    p.add_argument("--emb_batch_size", type=int, default=1000,
                   help="Batch size for computing neighborhood embeddings")
    p.add_argument("--n_trials", type=int, default=500,
                   help="Number of pattern-growth search trials")
    p.add_argument("--out_batch_size", type=int, default=5,
                   help="Patterns returned per pattern size")
    p.add_argument("--search_strategy", default="greedy",
                   choices=["greedy", "mcts"])
    return p.parse_args()


# ── Neighborhood size defaults per dataset ───────────────────────────────────

# Tuned so that max_neighborhood_size <= dataset max_n_nodes.
# QM9 (no-H): max 9 nodes. ZINC: max 38. MOSES: max 27. Guacamol: max 88.
_NEIGH_DEFAULTS = {
    "qm9":      (4,  8),
    "zinc":     (8, 25),
    "moses":    (8, 20),
    "guacamol": (10, 35),
}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Validate n_neighborhoods is divisible by emb_batch_size so no embeddings
    # are silently dropped (pattern_growth only processes full batches).
    if args.n_neighborhoods % args.emb_batch_size != 0:
        raise ValueError(
            f"--n_neighborhoods ({args.n_neighborhoods}) must be divisible by "
            f"--emb_batch_size ({args.emb_batch_size})."
        )

    # ── Change to SPMiner root ────────────────────────────────────────────────
    # Required so that relative paths inside pattern_growth resolve:
    #   ckpt/model.pt, plots/cluster/, results/
    os.chdir(args.spminer_root)
    sys.path.insert(0, args.spminer_root)

    os.makedirs("plots/cluster", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # ── Import SPMiner (after sys.path is set) ────────────────────────────────
    from subgraph_mining.decoder import pattern_growth

    # ── Load training graphs ──────────────────────────────────────────────────
    print(f"Loading graphs from: {args.nx_graphs_path}")
    with open(args.nx_graphs_path, "rb") as f:
        graphs = pickle.load(f)
    print(f"Loaded {len(graphs)} training graphs")

    # ── Build args namespace for pattern_growth ───────────────────────────────
    # These encoder defaults must match the architecture baked into ckpt/model.pt.
    min_neigh, max_neigh = _NEIGH_DEFAULTS[args.dataset_name]

    pg_args = SimpleNamespace(
        # ── Encoder (must match ckpt/model.pt) ──
        conv_type="SAGE",
        method_type="order",
        n_layers=8,
        hidden_dim=64,
        skip="learnable",
        dropout=0.0,
        margin=0.1,
        model_path=args.model_path,
        node_anchored=True,
        # ── Decoder ──
        sample_method="tree",
        use_whole_graphs=False,     # see module docstring for rationale
        radius=3,
        subgraph_sample_size=0,
        min_pattern_size=args.min_pattern_size,
        max_pattern_size=args.max_pattern_size,
        min_neighborhood_size=min_neigh,
        max_neighborhood_size=max_neigh,
        n_neighborhoods=args.n_neighborhoods,
        n_trials=args.n_trials,
        search_strategy=args.search_strategy,
        out_batch_size=args.out_batch_size,
        out_path=args.out_path,     # pattern_growth saves here
        analyze=False,
        batch_size=args.emb_batch_size,
    )

    print(
        f"\nSPMiner config:"
        f"\n  dataset:           {args.dataset_name}"
        f"\n  pattern sizes:     {args.min_pattern_size}–{args.max_pattern_size}"
        f"\n  neighborhood sizes:{min_neigh}–{max_neigh}"
        f"\n  n_neighborhoods:   {args.n_neighborhoods}"
        f"\n  n_trials:          {args.n_trials}"
        f"\n  out_batch_size:    {args.out_batch_size} per size"
        f"\n  model:             {args.model_path}"
        f"\n  output:            {args.out_path}\n"
    )

    # ── Ensure output directory exists ───────────────────────────────────────
    out_dir = os.path.dirname(args.out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # ── Run SPMiner ───────────────────────────────────────────────────────────
    # pattern_growth handles: model loading, neighborhood sampling, embedding,
    # search, and saving the result to pg_args.out_path.
    pattern_growth(graphs, "graph", pg_args)

    # ── Summary ───────────────────────────────────────────────────────────────
    with open(args.out_path, "rb") as f:
        patterns = pickle.load(f)
    print(f"\nMined {len(patterns)} motif patterns → {args.out_path}")
    for i, p in enumerate(patterns):
        print(f"  pattern {i}: {p.number_of_nodes()} nodes, {p.number_of_edges()} edges")


if __name__ == "__main__":
    main()
