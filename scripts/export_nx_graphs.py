"""Export DeFoG training graphs as topology-only NetworkX graphs for SPMiner.

Runs in the DeFoG conda environment. Loads the training split via the existing
Hydra/datamodule infrastructure and converts each PyG Data object to an
undirected nx.Graph containing only node/edge topology (no atom types or bond
types — SPMiner's model operates on structure only).

Output: a Python pickle file containing list[nx.Graph].

Usage (from repo root):
    python scripts/export_nx_graphs.py \\
        +experiment=qm9_no_h \\
        hydra.job.chdir=False \\
        model.spminer_nx_graphs_path=data/qm9/training_graphs_nx.pkl

    python scripts/export_nx_graphs.py \\
        +experiment=zinc \\
        hydra.job.chdir=False \\
        model.spminer_nx_graphs_path=data/zinc/training_graphs_nx.pkl
"""
import os
import sys
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import hydra
import networkx as nx
from omegaconf import DictConfig
from tqdm import tqdm


def build_datamodule(cfg):
    dataset_name = cfg["dataset"]["name"]
    if "qm9" in dataset_name:
        from datasets import qm9_dataset
        return qm9_dataset.QM9DataModule(cfg)
    elif "zinc" in dataset_name:
        from datasets import zinc_dataset
        return zinc_dataset.ZINCDataModule(cfg)
    elif dataset_name == "guacamol":
        from datasets import guacamol_dataset
        return guacamol_dataset.GuacaMolDataModule(cfg)
    elif dataset_name == "moses":
        from datasets import moses_dataset
        return moses_dataset.MOSESDataModule(cfg)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name!r}")


def pyg_to_nx_topology(data):
    """Convert a PyG Data object to a topology-only undirected nx.Graph.

    Strips all node and edge attributes — only the adjacency structure is kept.
    Self-loops present in edge_index are removed.
    """
    n = data.num_nodes
    G = nx.Graph()
    G.add_nodes_from(range(n))
    if data.edge_index.numel() > 0:
        edges = data.edge_index.t().tolist()
        G.add_edges_from([(int(u), int(v)) for u, v in edges if u != v])
    return G


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    out_path = cfg.model.spminer_nx_graphs_path
    if out_path is None:
        raise ValueError(
            "Specify the output path on the command line:\n"
            "  model.spminer_nx_graphs_path=data/<dataset>/training_graphs_nx.pkl"
        )

    datamodule = build_datamodule(cfg)
    datamodule.prepare_data()
    datamodule.setup(stage='fit')

    nx_graphs = []
    for batch in tqdm(datamodule.train_dataloader(), desc="Exporting training graphs"):
        for data in batch.to_data_list():
            G = pyg_to_nx_topology(data)
            # Skip degenerate graphs (isolated single node or no edges)
            if G.number_of_nodes() > 1 and G.number_of_edges() > 0:
                nx_graphs.append(G)

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_path, "wb") as f:
        pickle.dump(nx_graphs, f)

    sizes = [G.number_of_nodes() for G in nx_graphs]
    print(f"Exported {len(nx_graphs)} graphs → {out_path}")
    print(f"Node count range: {min(sizes)}–{max(sizes)}, mean {sum(sizes)/len(sizes):.1f}")


if __name__ == "__main__":
    main()
