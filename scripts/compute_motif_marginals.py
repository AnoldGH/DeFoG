"""
Precompute empirical node/edge type marginals of the ring-edited distribution.

Run from the repo root with the same experiment config used for sampling:
    python scripts/compute_motif_marginals.py +experiment=qm9_no_h

The script samples graphs from the dataset marginal, applies add_rings, and
accumulates per-type counts to estimate x_limit_motif and e_limit_motif.
Output is saved to cfg.model.motif_marginals_path.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from src import utils
from flow_matching import flow_matching_utils
from flow_matching.motif_editor import add_rings


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

    dataset_infos = load_dataset_infos(cfg)

    # Original marginal limit_dist — same construction as "marginal" transition.
    node_types = dataset_infos.node_types.float()
    x_orig = (node_types / node_types.sum()).to(device)
    edge_types = dataset_infos.edge_types.float()
    e_orig = (edge_types / edge_types.sum()).to(device)
    original_limit_dist = utils.PlaceHolder(X=x_orig, E=e_orig, y=torch.zeros(0, device=device))

    ring_specs = [tuple(rs) for rs in cfg.model.motif_ring_specs]
    ring_budget = sum(rs * rc for rs, rc in ring_specs)
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

        # Inflate n_max to guarantee ring_budget free slots for every graph.
        n_nodes = dataset_infos.nodes_dist.sample_n(bs, device)
        n_max = int(torch.max(n_nodes).item()) + ring_budget
        arange = torch.arange(n_max, device=device).unsqueeze(0).expand(bs, -1)
        node_mask = arange < n_nodes.unsqueeze(1)

        z = flow_matching_utils.sample_discrete_feature_noise(original_limit_dist, node_mask)
        z, node_mask = add_rings(z, node_mask, ring_specs, original_limit_dist)

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

    print(f"Saved motif marginals to {output_path}")
    print(f"Node marginal: {x_limit_motif}")
    print(f"Edge marginal: {e_limit_motif}")


if __name__ == "__main__":
    main()
