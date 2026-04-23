"""Helpers to construct `subgraph_cond` tensors for sampling.

Three modes match the design doc's inference strategies (plus null):
  1. from_idx           — condition on one training graph per sample
  2. from_idx_max       — elementwise-max over k training graphs (primary = [:,0])
  3. from_anchor_emb    — synthetic motif: user-supplied 64-d SPMiner vec + size
  null: from_null       — zeros, matches CFG dropout token

All return [bs, feature_dim] tensors on `device`, ready to pass to
`model.sample_batch(..., subgraph_cond=cond)`.
"""

import math
import torch


def _log_n_scaled(sub_feats, n_nodes_tensor):
    """log(n) / log_max_n, matching SubgraphEmbeddingFeatures._log_n scaling."""
    return torch.log(n_nodes_tensor.float()) / sub_feats.log_max_n


def from_null(sub_feats, bs, device):
    """Null / unconditional token — zeros of the correct shape."""
    return torch.zeros(bs, sub_feats.feature_dim, device=device)


def from_idx(sub_feats, idx, device):
    """Condition each row on a single training graph's precomputed pattern.

    idx: [bs] long tensor or iterable of graph indices (into sub_feats tables).
    """
    if not isinstance(idx, torch.Tensor):
        idx = torch.as_tensor(idx, dtype=torch.long)
    idx = idx.cpu().long()
    pat = sub_feats.precomputed[idx].to(device).float()  # [bs, pattern_dim]
    log_n = _log_n_scaled(sub_feats, sub_feats.n_nodes[idx]).to(device).unsqueeze(-1)
    return torch.cat([pat, log_n], dim=-1)


def from_idx_max(sub_feats, idx_matrix, device):
    """Elementwise-max over k training graphs per sample.

    idx_matrix: [bs, k] long tensor or list-of-lists. log_n is taken from
    column 0 (the 'primary' target) — the aggregation only affects pattern.
    """
    if not isinstance(idx_matrix, torch.Tensor):
        idx_matrix = torch.as_tensor(idx_matrix, dtype=torch.long)
    idx_matrix = idx_matrix.cpu().long()
    bs, k = idx_matrix.shape
    # precomputed: [N, pattern_dim]; gather [bs, k, pattern_dim], then max over k
    pats = sub_feats.precomputed[idx_matrix.view(-1)].view(bs, k, -1)  # cpu
    pat = pats.max(dim=1).values.to(device).float()  # [bs, pattern_dim]
    primary = idx_matrix[:, 0]
    log_n = _log_n_scaled(sub_feats, sub_feats.n_nodes[primary]).to(device).unsqueeze(-1)
    return torch.cat([pat, log_n], dim=-1)


def from_anchor_emb(sub_feats, anchor_emb, n_nodes, device):
    """Synthetic motif: user-supplied 64-d SPMiner vector + target size.

    TODO: layout mismatch vs training. SubgraphEmbeddingFeatures._pattern_for
    single-anchor branch flattens per[a] of shape [S, H] — each of the S size
    slots gets a *different* embedding (BFS neighborhoods at different sizes
    around the same anchor). Here we tile one [H] vector across all n_sizes
    slots, so per-size variation is lost. Proper fix: accept
    `anchor_per_size: [bs, n_sizes, H]` and broadcast only across reductions.

    anchor_emb: [bs, hidden_dim] — one SPMiner encoder output per sample.
    n_nodes:    [bs] long — target graph size per sample (drives log_n).
    """
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
    # Tile anchor across (size, reduction) axes; SP miner single-anchor collapses
    # every reduction to the anchor itself -> repeat across n_red * n_sizes blocks.
    pat = anchor_emb.view(bs, 1, H).expand(bs, n_red * n_sizes, H).reshape(bs, -1)
    pat = pat.to(device).float()  # [bs, n_red*n_sizes*H] = pattern_dim
    log_n = _log_n_scaled(sub_feats, n_nodes).to(device).unsqueeze(-1)
    return torch.cat([pat, log_n], dim=-1)
