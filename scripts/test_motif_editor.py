"""
Self-contained correctness tests for add_rings and the motif_edited_A
NoiseDistribution path. No dataset loading required.

Run from repo root:
    /home/maxlyu/miniconda3/envs/defog/bin/python scripts/test_motif_editor.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from src import utils
from flow_matching.motif_editor import add_rings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_graph(B, N, dx, de, n_nodes_list, device="cpu"):
    """Return a zero-feature PlaceHolder + node_mask with given active node counts."""
    node_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
    for b, n in enumerate(n_nodes_list):
        node_mask[b, :n] = True

    X = torch.zeros(B, N, dx, device=device)
    E = torch.zeros(B, N, N, de, device=device)
    # Mark "no bond" (class 0) for all pairs — mimics encode_no_edge output.
    for b in range(B):
        n = n_nodes_list[b]
        for i in range(n):
            for j in range(n):
                if i != j:
                    E[b, i, j, 0] = 1.0
    y = torch.zeros(B, 0, device=device)
    graph = utils.PlaceHolder(X=X, E=E, y=y)
    return graph, node_mask


def make_limit_dist(dx, de, device="cpu"):
    """Uniform limit distribution."""
    X = torch.ones(dx, device=device) / dx
    E = torch.ones(de, device=device) / de
    y = torch.zeros(0, device=device)
    return utils.PlaceHolder(X=X, E=E, y=y)


def active_node_counts(node_mask):
    return node_mask.sum(dim=1).tolist()


def check_symmetry(E):
    assert torch.allclose(E, E.transpose(1, 2)), "E is not symmetric"


def check_one_hot(T, name):
    sums = T.sum(dim=-1)
    active = sums > 0
    if active.any():
        assert torch.allclose(sums[active], torch.ones_like(sums[active])), \
            f"{name} active entries are not one-hot (sums={sums[active]})"


# ---------------------------------------------------------------------------
# Test 1: basic ring addition — node count and symmetry
# ---------------------------------------------------------------------------
def test_basic_ring():
    B, dx, de = 4, 5, 4
    n_nodes = [5, 6, 7, 8]
    N = 15  # enough padding for 1 ring of size 6

    graph, node_mask = make_graph(B, N, dx, de, n_nodes)
    limit_dist = make_limit_dist(dx, de)

    ring_specs = [(6, 1)]
    z, new_mask = add_rings(graph, node_mask, ring_specs, limit_dist)

    expected = [n + 6 for n in n_nodes]
    actual = active_node_counts(new_mask)
    assert actual == expected, f"Expected {expected} active nodes, got {actual}"

    check_symmetry(z.E)
    check_one_hot(z.X, "X")
    print("PASS  test_basic_ring")


# ---------------------------------------------------------------------------
# Test 2: capacity skip — ring not added when insufficient free slots
# ---------------------------------------------------------------------------
def test_capacity_skip():
    B, dx, de = 2, 5, 4
    N = 10
    # Graph 0: 5 active → 5 free, ring size 6 → skip  (5 < 6)
    # Graph 1: 4 active → 6 free, ring size 6 → add   (6 >= 6)
    n_nodes = [5, 4]

    graph, node_mask = make_graph(B, N, dx, de, n_nodes)
    limit_dist = make_limit_dist(dx, de)

    ring_specs = [(6, 1)]
    z, new_mask = add_rings(graph, node_mask, ring_specs, limit_dist)

    actual = active_node_counts(new_mask)
    assert actual[0] == 5, f"Graph 0 should be skipped, got {actual[0]} nodes"
    assert actual[1] == 10, f"Graph 1 should have 10 nodes, got {actual[1]}"

    check_symmetry(z.E)
    print("PASS  test_capacity_skip")


# ---------------------------------------------------------------------------
# Test 3: multiple ring sizes — correct total node count
# ---------------------------------------------------------------------------
def test_multi_ring_specs():
    B, dx, de = 2, 5, 4
    n_nodes = [3, 3]
    ring_budget = 5 + 6  # one ring of 5, one ring of 6
    N = max(n_nodes) + ring_budget + 2  # +2 extra slack

    graph, node_mask = make_graph(B, N, dx, de, n_nodes)
    limit_dist = make_limit_dist(dx, de)

    ring_specs = [(5, 1), (6, 1)]
    z, new_mask = add_rings(graph, node_mask, ring_specs, limit_dist)

    expected = [n + ring_budget for n in n_nodes]
    actual = active_node_counts(new_mask)
    assert actual == expected, f"Expected {expected}, got {actual}"

    check_symmetry(z.E)
    print("PASS  test_multi_ring_specs")


# ---------------------------------------------------------------------------
# Test 4: ring count > 1 — two rings of the same size
# ---------------------------------------------------------------------------
def test_ring_count():
    B, dx, de = 2, 5, 4
    n_nodes = [2, 2]
    ring_budget = 6 * 2
    N = max(n_nodes) + ring_budget + 2

    graph, node_mask = make_graph(B, N, dx, de, n_nodes)
    limit_dist = make_limit_dist(dx, de)

    ring_specs = [(6, 2)]
    z, new_mask = add_rings(graph, node_mask, ring_specs, limit_dist)

    expected = [n + ring_budget for n in n_nodes]
    actual = active_node_counts(new_mask)
    assert actual == expected, f"Expected {expected}, got {actual}"

    check_symmetry(z.E)
    print("PASS  test_ring_count")


# ---------------------------------------------------------------------------
# Test 5: ring edges actually exist (non-zero, non-class-0)
# ---------------------------------------------------------------------------
def test_ring_edges_exist():
    B, dx, de = 1, 5, 4
    n_nodes = [3]
    N = 12

    graph, node_mask = make_graph(B, N, dx, de, n_nodes)
    limit_dist = make_limit_dist(dx, de)

    ring_specs = [(6, 1)]
    z, new_mask = add_rings(graph, node_mask, ring_specs, limit_dist)

    # Ring occupies slots 3..8 (0-indexed). Check that internal ring edges
    # are non-zero (at least one class > 0 for each pair).
    ring_indices = list(range(3, 9))
    for k in range(6):
        i = ring_indices[k]
        j = ring_indices[(k + 1) % 6]
        edge_ij = z.E[0, i, j]
        assert edge_ij.sum() > 0, f"Ring edge ({i},{j}) is all-zero"
        assert edge_ij.argmax().item() != 0 or edge_ij[0] < 1.0, \
            f"Ring edge ({i},{j}) is 'no bond' (class 0 is 1.0)"

    # Check connecting edge (node 0 or any original node) to ring node u
    # exists — at least one edge from original nodes to ring nodes is nonzero.
    orig_to_ring = z.E[0, :3, 3:9]  # (3, 6, de)
    assert orig_to_ring.sum() > 0, "No connecting edge found between original and ring"

    check_symmetry(z.E)
    print("PASS  test_ring_edges_exist")


# ---------------------------------------------------------------------------
# Test 6: tensor shape unchanged
# ---------------------------------------------------------------------------
def test_shape_unchanged():
    B, N, dx, de = 3, 20, 6, 5
    n_nodes = [4, 5, 6]

    graph, node_mask = make_graph(B, N, dx, de, n_nodes)
    limit_dist = make_limit_dist(dx, de)

    ring_specs = [(5, 1)]
    z, new_mask = add_rings(graph, node_mask, ring_specs, limit_dist)

    assert z.X.shape == (B, N, dx), f"X shape changed: {z.X.shape}"
    assert z.E.shape == (B, N, N, de), f"E shape changed: {z.E.shape}"
    assert new_mask.shape == (B, N), f"node_mask shape changed: {new_mask.shape}"
    print("PASS  test_shape_unchanged")


# ---------------------------------------------------------------------------
# Test 7: NoiseDistribution motif_edited_A path (mock cfg + dataset_infos)
# ---------------------------------------------------------------------------
def test_noise_distribution_motif_edited_A():
    from flow_matching.noise_distribution import NoiseDistribution
    from types import SimpleNamespace

    dx, de = 5, 4
    B, N = 3, 20
    n_nodes_list = [4, 6, 8]

    # Precomputed marginals file
    tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    torch.save({
        "X": torch.ones(dx) / dx,
        "E": torch.ones(de) / de,
    }, tmp.name)
    tmp.close()

    # Mock dataset_infos
    dataset_infos = SimpleNamespace(
        output_dims={"X": dx, "E": de, "y": 0},
        node_types=torch.ones(dx),
        edge_types=torch.ones(de),
    )

    # Mock cfg
    cfg = SimpleNamespace(model=SimpleNamespace(
        transition="motif_edited_A",
        motif_ring_specs=[[6, 1]],
        motif_marginals_path=tmp.name,
    ))

    nd = NoiseDistribution("motif_edited_A", dataset_infos, cfg=cfg)

    # Build node_mask with ring_budget inflation (as sample_batch would)
    ring_budget = 6
    n_max = max(n_nodes_list) + ring_budget
    arange = torch.arange(n_max).unsqueeze(0).expand(B, -1)
    node_mask = arange < torch.tensor(n_nodes_list).unsqueeze(1)

    z_T, updated_mask = nd.sample_initial_noise(node_mask)

    # Each graph should gain 6 nodes
    expected = [n + 6 for n in n_nodes_list]
    actual = active_node_counts(updated_mask)
    assert actual == expected, f"Expected {expected}, got {actual}"

    assert z_T.X.shape == (B, n_max, dx)
    assert z_T.E.shape == (B, n_max, n_max, de)
    check_symmetry(z_T.E)

    os.unlink(tmp.name)
    print("PASS  test_noise_distribution_motif_edited_A")


# ---------------------------------------------------------------------------
# Test 8: non-motif transition unchanged by sample_initial_noise
# ---------------------------------------------------------------------------
def test_noise_distribution_passthrough():
    from flow_matching.noise_distribution import NoiseDistribution
    from types import SimpleNamespace

    dx, de = 5, 4
    B, N = 2, 10

    dataset_infos = SimpleNamespace(
        output_dims={"X": dx, "E": de, "y": 0},
        node_types=torch.ones(dx),
        edge_types=torch.ones(de),
    )

    nd = NoiseDistribution("marginal", dataset_infos)

    node_mask = torch.ones(B, N, dtype=torch.bool)
    z_T, returned_mask = nd.sample_initial_noise(node_mask)

    assert (returned_mask == node_mask).all(), "node_mask should be unchanged for non-motif transitions"
    assert z_T.X.shape == (B, N, dx)
    print("PASS  test_noise_distribution_passthrough")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_basic_ring()
    test_capacity_skip()
    test_multi_ring_specs()
    test_ring_count()
    test_ring_edges_exist()
    test_shape_unchanged()
    test_noise_distribution_motif_edited_A()
    test_noise_distribution_passthrough()
    print("\nAll tests passed.")
