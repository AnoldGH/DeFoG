"""
Precompute empirical node/edge type marginals of the motif-edited distribution.

Supports two transition types:

  motif_edited_A (ring-based):
    python scripts/compute_motif_marginals.py \\
        +experiment=qm9_no_h \\
        hydra.job.chdir=False \\
        model.transition=motif_edited_A \\
        model.motif_ring_specs=[[6,1]] \\
        model.motif_marginals_path=data/qm9/motif_marginals.pt

  motif_edited_spminer (SPMiner patterns):
    python scripts/compute_motif_marginals.py \\
        +experiment=qm9_no_h \\
        hydra.job.chdir=False \\
        model.transition=motif_edited_spminer \\
        model.spminer_motifs_path=data/qm9/spminer_motifs.pkl \\
        model.spminer_top_k=3 \\
        model.motif_marginals_path=data/qm9/spminer_motif_marginals.pt

The script samples graphs from the dataset marginal, applies the motif-editing
function, and accumulates per-type counts to estimate x_limit_motif and
e_limit_motif.  Output is saved to cfg.model.motif_marginals_path.
"""
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from src import utils
from flow_matching import flow_matching_utils
from flow_matching.motif_editor import add_rings, add_motif_patterns


def load_dataset_infos(cfg):
    """Load only the dataset statistics needed for marginal computation.

    All four molecular datasets have their node/edge type distributions and
    node-count distributions hardcoded in their *Infos constructors, so we
    can pass datamodule=None and avoid downloading the actual graphs.
    """
    dataset_name = cfg["dataset"]["name"]
    if "qm9" in dataset_name:
        from datasets import qm9_dataset
        return qm9_dataset.QM9infos(datamodule=None, cfg=cfg)
    elif dataset_name == "guacamol":
        from datasets import guacamol_dataset
        return guacamol_dataset.Guacamolinfos(datamodule=None, cfg=cfg)
    elif dataset_name == "moses":
        from datasets import moses_dataset
        return moses_dataset.MOSESinfos(datamodule=None, cfg=cfg)
    elif "zinc" in dataset_name:
        from datasets import zinc_dataset
        return zinc_dataset.ZINCinfos(datamodule=None, cfg=cfg)
    else:
        raise ValueError(f"Dataset '{dataset_name}' not supported for motif marginal precomputation.")


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transition = cfg.model.transition
    if transition not in ("motif_edited_A", "motif_edited_spminer"):
        raise ValueError(
            f"compute_motif_marginals.py requires transition='motif_edited_A' or "
            f"'motif_edited_spminer', got {transition!r}."
        )

    dataset_infos = load_dataset_infos(cfg)

    # Original marginal limit_dist — same construction as "marginal" transition.
    node_types = dataset_infos.node_types.float()
    x_orig = (node_types / node_types.sum()).to(device)
    edge_types = dataset_infos.edge_types.float()
    e_orig = (edge_types / edge_types.sum()).to(device)
    original_limit_dist = utils.PlaceHolder(X=x_orig, E=e_orig, y=torch.zeros(0, device=device))

    # ------------------------------------------------------------------
    # Dispatch: build (apply_motif_fn, node_budget) for the chosen transition.
    # ------------------------------------------------------------------
    if transition == "motif_edited_A":
        ring_specs = [tuple(rs) for rs in cfg.model.motif_ring_specs]
        node_budget = sum(rs * rc for rs, rc in ring_specs)

        def apply_motif_fn(z, nm, orig):
            return add_rings(z, nm, ring_specs, orig)

        print(f"Transition : motif_edited_A")
        print(f"Ring specs : {ring_specs}")
        print(f"Node budget: {node_budget}")

    else:  # motif_edited_spminer
        motifs_path = cfg.model.spminer_motifs_path
        if motifs_path is None:
            raise ValueError(
                "model.spminer_motifs_path must be set for motif_edited_spminer."
            )
        with open(motifs_path, "rb") as f:
            motif_graphs = pickle.load(f)
        top_k = cfg.model.spminer_top_k
        node_budget = sum(G.number_of_nodes() for G in motif_graphs[:top_k])

        def apply_motif_fn(z, nm, orig):
            return add_motif_patterns(z, nm, motif_graphs, orig, top_k=top_k)

        print(f"Transition  : motif_edited_spminer")
        print(f"Motifs path : {motifs_path}")
        print(f"Top-k       : {top_k}  ({len(motif_graphs)} patterns available)")
        print(f"Node budget : {node_budget}")

    num_samples = cfg.model.motif_num_samples
    batch_size = cfg.model.motif_sample_batch_size
    output_path = cfg.model.motif_marginals_path

    dx = x_orig.shape[0]
    de = e_orig.shape[0]
    X_counts = torch.zeros(dx, device=device)
    E_counts = torch.zeros(de, device=device)

    samples_done = 0
    total_batches = (num_samples + batch_size - 1) // batch_size

    for _ in tqdm(range(total_batches), desc="Sampling edited graphs"):
        bs = min(batch_size, num_samples - samples_done)
        samples_done += bs

        # Inflate n_max to guarantee node_budget free slots for every graph.
        n_nodes = dataset_infos.nodes_dist.sample_n(bs, device)
        n_max = int(torch.max(n_nodes).item()) + node_budget
        arange = torch.arange(n_max, device=device).unsqueeze(0).expand(bs, -1)
        node_mask = arange < n_nodes.unsqueeze(1)

        z = flow_matching_utils.sample_discrete_feature_noise(original_limit_dist, node_mask)
        z, node_mask = apply_motif_fn(z, node_mask, original_limit_dist)

        # Accumulate node-type counts over all active nodes.
        X_counts += z.X[node_mask].sum(dim=0)

        # Accumulate edge-type counts over active, off-diagonal pairs.
        active_edge = node_mask.unsqueeze(2) & node_mask.unsqueeze(1)  # (B, N, N)
        diag = torch.eye(n_max, dtype=torch.bool, device=device).unsqueeze(0).expand(bs, -1, -1)
        E_counts += z.E[active_edge & ~diag].sum(dim=0)

    x_limit_motif = X_counts / X_counts.sum()
    e_limit_motif = E_counts / E_counts.sum()

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save({"X": x_limit_motif.cpu(), "E": e_limit_motif.cpu()}, output_path)

    print(f"\nSaved motif marginals to {output_path}")
    print(f"Node marginal: {x_limit_motif}")
    print(f"Edge marginal: {e_limit_motif}")


if __name__ == "__main__":
    main()
