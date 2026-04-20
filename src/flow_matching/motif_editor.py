import torch
import torch.nn.functional as F

from utils import PlaceHolder


def add_rings(
    graph: PlaceHolder,
    node_mask: torch.Tensor,
    ring_specs: list[tuple[int, int]],
    limit_dist: PlaceHolder,
) -> tuple[PlaceHolder, torch.Tensor]:
    """
    Attach rings to a batch of graphs by activating free (padding) node slots.

    Ring nodes and bond types are sampled from limit_dist (the same distribution
    used to sample the original graph). Specifically, E[0] is the "no bond" class
    and is excluded when sampling bond types for ring-internal and connecting edges,
    since those edges must be actual bonds by definition.

    The growing-graph interpretation is used: each ring's attachment node v is
    sampled uniformly from all currently active nodes (including previously added
    ring nodes). Rings are added left-to-right through ring_specs, and within each
    size, one at a time. If a graph has fewer than ring_size free slots, that size's
    additions are skipped for that graph (and all subsequent additions of that size).

    Args:
        graph:      PlaceHolder — X (B, N, dx) one-hot, E (B, N, N, de) one-hot,
                    y (B, dy).
        node_mask:  (B, N) bool — True = active node.
        ring_specs: [(ring_size, count), ...] sorted by increasing ring_size.
        limit_dist: PlaceHolder — X (dx,) node-type marginals,
                    E (de,) edge-type marginals with E[0] = no-bond probability.

    Returns:
        (modified PlaceHolder, updated node_mask)
    """
    B, N, dx = graph.X.shape
    de = graph.E.shape[-1]
    device = graph.X.device

    X = graph.X.clone()       # (B, N, dx)
    E = graph.E.clone()       # (B, N, N, de)
    node_mask = node_mask.clone()  # (B, N) bool

    # Bond distribution: exclude class 0 (no bond) and renormalize.
    bond_dist = limit_dist.E.clone()
    bond_dist[0] = 0.0
    bond_dist = bond_dist / bond_dist.sum()  # (de,)

    b_idx = torch.arange(B, device=device)  # (B,)

    for ring_size, count in ring_specs:
        for _ in range(count):
            # How many free slots each graph has.
            free_nodes = N - node_mask.sum(dim=1)   # (B,)
            can_add = free_nodes >= ring_size        # (B,) bool
            if not can_add.any():
                break

            # ------------------------------------------------------------------
            # Identify the first ring_size free slots for each graph.
            # free_rank[b, i] = 1-indexed rank of position i among free positions
            # in graph b (0 if position i is active).
            # ------------------------------------------------------------------
            free_mask = ~node_mask                              # (B, N)
            free_rank = free_mask.long().cumsum(dim=1)         # (B, N)

            # ring_node_indices[b, k] = position of the (k+1)-th free slot in graph b.
            # Shape: (B, ring_size). Fully vectorized via broadcasting.
            k_vals = torch.arange(1, ring_size + 1, device=device)  # (ring_size,)
            matches = free_mask.unsqueeze(2) & (free_rank.unsqueeze(2) == k_vals)  # (B, N, ring_size)
            ring_node_indices = matches.long().argmax(dim=1)   # (B, ring_size)

            # Ring slot mask: positions that will become ring nodes.
            ring_pos_mask = free_mask & (free_rank <= ring_size)   # (B, N)
            ring_pos_mask = ring_pos_mask & can_add.unsqueeze(1)   # zero out skipped graphs

            # ------------------------------------------------------------------
            # Sample attachment point v (growing graph: sample from all active nodes).
            # ------------------------------------------------------------------
            active_prob = node_mask.float()                         # (B, N)
            active_prob = active_prob / active_prob.sum(dim=1, keepdim=True).clamp(min=1)
            v = active_prob.multinomial(1).squeeze(1)               # (B,)

            # ------------------------------------------------------------------
            # Sample attachment point u within ring (uniform over ring nodes).
            # ------------------------------------------------------------------
            u_rank = torch.randint(0, ring_size, (B,), device=device)   # (B,) 0-indexed
            u_idx = ring_node_indices[b_idx, u_rank]                     # (B,)

            # ------------------------------------------------------------------
            # Sample node types for ring nodes and write into X.
            # ------------------------------------------------------------------
            node_probs = limit_dist.X.unsqueeze(0).unsqueeze(0).expand(B, ring_size, -1)
            node_types = node_probs.reshape(B * ring_size, -1).multinomial(1).reshape(B, ring_size)
            X_ring = F.one_hot(node_types, num_classes=dx).float()   # (B, ring_size, dx)

            # Map each position to the corresponding ring-node feature via free_rank.
            # (free_rank - 1) gives a 0-indexed rank; clamp avoids negatives at active slots,
            # but those positions are masked out by ring_pos_mask anyway.
            rank0 = (free_rank - 1).clamp(min=0, max=ring_size - 1)  # (B, N)
            rank0_exp = rank0.unsqueeze(2).expand(-1, -1, dx)        # (B, N, dx)
            X_write = torch.gather(X_ring, 1, rank0_exp)             # (B, N, dx)
            X = X + X_write * ring_pos_mask.unsqueeze(2).float()

            # ------------------------------------------------------------------
            # Sample ring-internal bond types and write into E.
            # Edges form a cycle: r0-r1, r1-r2, ..., r_{k-1}-r0.
            # ------------------------------------------------------------------
            bond_probs = bond_dist.unsqueeze(0).unsqueeze(0).expand(B, ring_size, -1)
            ring_bond_types = bond_probs.reshape(B * ring_size, -1).multinomial(1).reshape(B, ring_size)
            E_ring = F.one_hot(ring_bond_types, num_classes=de).float()  # (B, ring_size, de)

            r_src = ring_node_indices                                    # (B, ring_size)
            r_dst = torch.roll(ring_node_indices, -1, dims=1)           # (B, ring_size)

            b_exp = b_idx.unsqueeze(1).expand(-1, ring_size)            # (B, ring_size)
            can_exp = can_add.unsqueeze(1).expand(-1, ring_size)        # (B, ring_size)

            valid = can_exp.reshape(-1)
            vb   = b_exp.reshape(-1)[valid]
            vsrc = r_src.reshape(-1)[valid]
            vdst = r_dst.reshape(-1)[valid]
            ve   = E_ring.reshape(-1, de)[valid]

            E[vb, vsrc, vdst] = ve
            E[vb, vdst, vsrc] = ve   # enforce symmetry

            # ------------------------------------------------------------------
            # Sample and write connecting edge v <-> u.
            # ------------------------------------------------------------------
            conn_type = bond_dist.unsqueeze(0).expand(B, -1).multinomial(1).squeeze(1)  # (B,)
            E_conn = F.one_hot(conn_type, num_classes=de).float()       # (B, de)

            act = can_add
            E[b_idx[act], v[act], u_idx[act]] = E_conn[act]
            E[b_idx[act], u_idx[act], v[act]] = E_conn[act]   # enforce symmetry

            # ------------------------------------------------------------------
            # Activate ring node slots.
            # ------------------------------------------------------------------
            node_mask = node_mask | ring_pos_mask

    return PlaceHolder(X=X, E=E, y=graph.y), node_mask
