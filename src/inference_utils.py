"""Helpers to construct `subgraph_cond` tensors for sampling.

Four entry points:
  from_null          — zeros, matches CFG-dropout token
  from_idx           — condition on one training graph per sample
  from_idx_max       — elementwise-max over k training graphs (primary = [:,0])
  from_anchor_emb    — synthetic motif: user-supplied 64-d SPMiner vec + size

All return `[bs, sub_feats.feature_dim]` tensors on `device`, with the scalar
tail width and layout determined by the SubgraphEmbeddingFeatures instance
(may be 0, 1, or 2 scalars — see sub_feats.idx_tail()).
"""

import torch


def from_null(sub_feats, bs, device):
    """Null / unconditional token — zeros of the correct shape."""
    return torch.zeros(bs, sub_feats.feature_dim, device=device)


def from_idx(sub_feats, idx, device):
    """Condition each row on a single training graph's precomputed pattern."""
    if not isinstance(idx, torch.Tensor):
        idx = torch.as_tensor(idx, dtype=torch.long)
    idx = idx.cpu().long()
    pat = sub_feats.precomputed[idx].to(device).float()  # [bs, pattern_dim]
    tail = sub_feats.idx_tail(idx).to(device).float()  # [bs, n_extra]
    return torch.cat([pat, tail], dim=-1)


def from_idx_max(sub_feats, idx_matrix, device):
    """Elementwise-max over k training graphs per sample.

    Pattern is max'd across k; scalar tail (if any) is taken from the
    primary column (idx_matrix[:, 0]), matching how aggregation is
    handled at training time in _build_row (tail stays own).
    """
    if not isinstance(idx_matrix, torch.Tensor):
        idx_matrix = torch.as_tensor(idx_matrix, dtype=torch.long)
    idx_matrix = idx_matrix.cpu().long()
    bs, k = idx_matrix.shape
    pats = sub_feats.precomputed[idx_matrix.view(-1)].view(bs, k, -1)
    pat = pats.max(dim=1).values.to(device).float()  # [bs, pattern_dim]
    primary = idx_matrix[:, 0]
    tail = sub_feats.idx_tail(primary).to(device).float()  # [bs, n_extra]
    return torch.cat([pat, tail], dim=-1)


def from_anchor_emb(sub_feats, anchor_emb, n_nodes, n_edges=None, device=None):
    """Synthetic motif: user-supplied 64-d SPMiner vector + target size(s).

    TODO: layout mismatch vs training. SubgraphEmbeddingFeatures._pattern_for
    single-anchor branch flattens per[a] of shape [S, H] — each of the S size
    slots gets a *different* embedding (BFS neighborhoods at different sizes
    around the same anchor). Here we tile one [H] vector across all n_sizes
    slots, so per-size variation is lost. Proper fix: accept
    `anchor_per_size: [bs, n_sizes, H]` and broadcast only across reductions.

    anchor_emb: [bs, hidden_dim] — one SPMiner encoder output per sample.
    n_nodes:    [bs] long — target graph size per sample (drives log_n).
    n_edges:    [bs] long — optional, required if sub_feats.include_log_e=True.
    """
    if device is None:
        device = anchor_emb.device if torch.is_tensor(anchor_emb) else torch.device("cpu")
    if not isinstance(anchor_emb, torch.Tensor):
        anchor_emb = torch.as_tensor(anchor_emb)
    if not isinstance(n_nodes, torch.Tensor):
        n_nodes = torch.as_tensor(n_nodes, dtype=torch.long)
    bs = anchor_emb.shape[0]
    H = sub_feats.hidden_dim
    n_red = len(sub_feats.pooling)
    n_sizes = len(sub_feats.sizes)
    if anchor_emb.shape[-1] != H:
        raise ValueError(
            f"anchor_emb last dim = {anchor_emb.shape[-1]}, expected {H}"
        )
    pat = anchor_emb.view(bs, 1, H).expand(bs, n_red * n_sizes, H).reshape(bs, -1)
    pat = pat.to(device).float()

    # Build tail from provided target_n_nodes / n_edges, since we don't have
    # a training idx to look up.
    parts = []
    if sub_feats.include_log_n:
        parts.append(torch.log(n_nodes.float()) / sub_feats.log_max_n)
    if sub_feats.include_log_e:
        if n_edges is None:
            raise ValueError(
                "sub_feats.include_log_e=True requires n_edges for from_anchor_emb"
            )
        if not isinstance(n_edges, torch.Tensor):
            n_edges = torch.as_tensor(n_edges, dtype=torch.long)
        parts.append(
            torch.log(n_edges.float().clamp(min=1)) / sub_feats.log_max_e
        )
    if not parts:
        tail = torch.zeros(bs, 0, device=device)
    else:
        tail = torch.stack(parts, dim=-1).to(device).float()

    return torch.cat([pat, tail], dim=-1)
