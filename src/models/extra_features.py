import math

import torch
from src import utils


_REDUCTIONS = {
    "mean": lambda x: x.mean(dim=0),
    "max": lambda x: x.max(dim=0).values,
    "min": lambda x: x.min(dim=0).values,
    "std": lambda x: x.std(dim=0, unbiased=False),
}


def _parse_reduction(name):
    """Return a callable that reduces [n, S, H] -> [S, H] over the anchor axis."""
    if name in _REDUCTIONS:
        return _REDUCTIONS[name]
    if name.startswith("q") and name[1:].isdigit():
        q = int(name[1:]) / 100.0
        if not 0.0 <= q <= 1.0:
            raise ValueError(f"quantile out of range: {name}")
        return lambda x, q=q: x.quantile(q, dim=0)
    raise ValueError(f"unknown pooling reduction: {name!r}")


class SubgraphEmbeddingFeatures:
    """Graph-level conditioning from precomputed SPMiner order embeddings.

    Lookup path: `noisy_data["idx"]` -> pattern vector (concat of size-wise
    per-anchor-pool SPMiner embeddings for each reduction in `pooling`) +
    log(n_nodes). Returned via y; X and E empty.

    Training augmentation (noisy_data["subgraph_training"] == True), per row:
      - `cfg_dropout` prob: replace with zeros (CFG null token)
      - `aggregation_prob` prob: elementwise-max with 1..k-1 other graphs'
        pattern vectors (keeps own log_n — size is the target, not aggregated)
      - `anchor_aug_prob` prob: substitute a single-anchor vector (all pool
        slots collapse to the anchor value — consistent with "n_anchors=1")

    Eval / sampling: deterministic lookup (no augmentation). If no idx and no
    `subgraph_cond` override, returns zeros.

    `pooling` is an ordered list of reduction names applied across anchors.
    Supported: "mean", "max", "min", "std", or "qNN" for quantile (e.g. "q25").
    Result dim = len(pooling) * len(sizes) * hidden_dim + 1 (+1 = log_n).
    """

    def __init__(
        self,
        emb_path,
        pooling=("max",),
        cfg_dropout=0.1,
        aggregation_prob=0.2,
        aggregation_k_max=3,
        anchor_aug_prob=0.3,
    ):
        data = torch.load(emb_path, weights_only=False)
        self.n_nodes = data["n_nodes"].long()  # [N]
        self.per_anchor = data.get("per_anchor", None)  # list of [n_i, S, 64]
        if self.per_anchor is None:
            raise ValueError(
                "SubgraphEmbeddingFeatures requires 'per_anchor' in the embedding file; "
                "re-run Phase 2 precomputation with per-anchor storage."
            )
        self.sizes = list(data["sizes"])
        self.hidden_dim = self.per_anchor[0].shape[-1]
        self.pooling = list(pooling)
        self.reductions = [_parse_reduction(r) for r in self.pooling]
        n_red = len(self.reductions)
        n_sizes = len(self.sizes)
        self.pattern_dim = n_red * n_sizes * self.hidden_dim
        self.feature_dim = self.pattern_dim + 1
        self.log_max_n = math.log(float(self.n_nodes.max().item()))
        self.cfg_dropout = cfg_dropout
        self.aggregation_prob = aggregation_prob
        self.aggregation_k_max = aggregation_k_max
        self.anchor_aug_prob = anchor_aug_prob

        # Precompute per-graph pattern: [N, n_red * n_sizes * hidden_dim].
        # For each graph, for each reduction, reduce along anchor axis per size.
        N = len(self.per_anchor)
        self.precomputed = torch.zeros(N, self.pattern_dim)
        for g, per in enumerate(self.per_anchor):  # per: [n_i, S, H]
            slots = []
            for red in self.reductions:
                pooled = red(per.float())  # [S, H]
                slots.append(pooled.reshape(-1))  # [S*H]
            self.precomputed[g] = torch.cat(slots, dim=0)  # [n_red*S*H]

    def _log_n(self, idx):
        return torch.log(self.n_nodes[idx].float()) / self.log_max_n

    def _pattern_for(self, gid, use_random_anchor):
        """Return a [pattern_dim] tensor for graph `gid` (CPU)."""
        if use_random_anchor and self.per_anchor is not None:
            per = self.per_anchor[gid]  # [n, S, H]
            a = int(torch.randint(0, per.shape[0], (1,)).item())
            single = per[a].reshape(-1).float()  # [S*H]
            # For a single anchor, every reduction collapses to the anchor value.
            return single.repeat(len(self.reductions))  # [n_red*S*H]
        return self.precomputed[gid]  # [n_red*S*H]

    def _build_row(self, own_gid, N_total):
        """Sample one row: returns ([pattern_dim], scalar log_n)."""
        r = float(torch.rand(1).item())
        own_log_n = self._log_n(own_gid)

        if r < self.cfg_dropout:
            return torch.zeros(self.pattern_dim), torch.zeros(())  # null row (own log_n dropped too)

        use_rand_anchor = float(torch.rand(1).item()) < self.anchor_aug_prob
        own_pat = self._pattern_for(own_gid, use_rand_anchor)

        if r < self.cfg_dropout + self.aggregation_prob and self.aggregation_k_max > 1:
            k_extra = int(torch.randint(1, self.aggregation_k_max, (1,)).item())
            extra_gids = torch.randint(0, N_total, (k_extra,)).tolist()
            pats = [own_pat]
            for g in extra_gids:
                use_ra = float(torch.rand(1).item()) < self.anchor_aug_prob
                pats.append(self._pattern_for(g, use_ra))
            own_pat = torch.stack(pats, dim=0).max(dim=0).values

        return own_pat, own_log_n

    def _training_lookup(self, idx, device):
        N_total = self.precomputed.shape[0]
        pat_rows, size_rows = [], []
        for gid in idx.tolist():
            pat, log_n = self._build_row(int(gid), N_total)
            pat_rows.append(pat)
            size_rows.append(log_n)
        pat = torch.stack(pat_rows, dim=0).to(device)  # [bs, S*64]
        log_n = torch.stack(size_rows, dim=0).unsqueeze(-1).to(device)  # [bs, 1]
        return torch.cat([pat, log_n], dim=-1).float()

    def _deterministic_lookup(self, idx, device):
        vec = self.precomputed[idx].to(device)  # [bs, pattern_dim]
        log_n = self._log_n(idx).to(device).unsqueeze(-1)  # [bs, 1]
        return torch.cat([vec, log_n], dim=-1).float()

    def __call__(self, noisy_data):
        X = noisy_data["X_t"]
        E = noisy_data["E_t"]
        bs = X.shape[0]
        device = X.device

        override = noisy_data.get("subgraph_cond", None)
        if override is not None:
            y = override.to(device).float()
        elif "idx" in noisy_data and noisy_data["idx"] is not None:
            idx = noisy_data["idx"].cpu().long()
            if noisy_data.get("subgraph_training", False):
                y = self._training_lookup(idx, device)
            else:
                y = self._deterministic_lookup(idx, device)
        else:
            y = torch.zeros(bs, self.feature_dim, device=device)

        empty_x = X.new_zeros((*X.shape[:-1], 0))
        empty_e = E.new_zeros((*E.shape[:-1], 0))
        return utils.PlaceHolder(X=empty_x, E=empty_e, y=y)


class CombinedExtraFeatures:
    """Call multiple ExtraFeatures-style callables and concat their outputs."""

    def __init__(self, *features):
        self.features = features

    def __call__(self, noisy_data):
        outs = [f(noisy_data) for f in self.features]
        return utils.PlaceHolder(
            X=torch.cat([o.X for o in outs], dim=-1),
            E=torch.cat([o.E for o in outs], dim=-1),
            y=torch.cat([o.y for o in outs], dim=-1),
        )


class DummyExtraFeatures:
    def __init__(self):
        """This class does not compute anything, just returns empty tensors."""

    def __call__(self, noisy_data):
        X = noisy_data["X_t"]
        E = noisy_data["E_t"]
        y = noisy_data["y_t"]
        empty_x = X.new_zeros((*X.shape[:-1], 0))
        empty_e = E.new_zeros((*E.shape[:-1], 0))
        empty_y = y.new_zeros((y.shape[0], 0))
        return utils.PlaceHolder(X=empty_x, E=empty_e, y=empty_y)


class ExtraFeatures:
    def __init__(self, extra_features_type, rrwp_steps, dataset_info):
        self.max_n_nodes = dataset_info.max_n_nodes
        self.ncycles = NodeCycleFeatures()
        self.features_type = extra_features_type
        self.rrwp_steps = rrwp_steps
        self.RRWP = RRWPFeatures()
        self.RWP = RRWPFeatures(normalize=False)
        if extra_features_type in ["eigenvalues", "all"]:
            self.eigenfeatures = EigenFeatures(mode=extra_features_type)

    def __call__(self, noisy_data):
        n = noisy_data["node_mask"].sum(dim=1).unsqueeze(1) / self.max_n_nodes
        x_cycles, y_cycles = self.ncycles(noisy_data)  # (bs, n_cycles)

        if self.features_type == "cycles":
            E = noisy_data["E_t"]
            extra_edge_attr = torch.zeros((*E.shape[:-1], 0)).type_as(E)
            return utils.PlaceHolder(
                X=x_cycles, E=extra_edge_attr, y=torch.hstack((n, y_cycles))
            )

        elif self.features_type == "eigenvalues":
            eigenfeatures = self.eigenfeatures(noisy_data)
            E = noisy_data["E_t"]
            extra_edge_attr = torch.zeros((*E.shape[:-1], 0)).type_as(E)
            n_components, batched_eigenvalues = eigenfeatures  # (bs, 1), (bs, 10)
            return utils.PlaceHolder(
                X=x_cycles,
                E=extra_edge_attr,
                y=torch.hstack((n, y_cycles)),
            )

        elif self.features_type == "rrwp":
            E = noisy_data["E_t"].float()[..., 1:].sum(-1)  # bs, n, n
            rrwp_edge_attr = self.RRWP(E, k=self.rrwp_steps)
            diag_index = torch.arange(rrwp_edge_attr.shape[1])
            rrwp_node_attr = rrwp_edge_attr[:, diag_index, diag_index, :]
            self.eigenfeatures = EigenFeatures(mode="all")

            return utils.PlaceHolder(
                X=rrwp_node_attr,
                E=rrwp_edge_attr,
                y=torch.hstack((n, y_cycles)),
            )

        elif self.features_type == "rrwp_double":
            E = noisy_data["E_t"].float()[..., 1:].sum(-1)  # bs, n, n
            rrwp_edge_attr = self.RRWP(E, k=self.rrwp_steps)
            rrwp_edge_attr_wo_norm = self.RWP(E, k=self.rrwp_steps)

            # Normalize the rrwp_edge_attr_wo_norm
            max_value = rrwp_edge_attr_wo_norm.max(dim=1, keepdim=True).values
            max_value = max_value.max(dim=2, keepdim=True).values
            rrwp_edge_attr_wo_norm = rrwp_edge_attr_wo_norm / max_value

            rrwp_edge_attr = torch.cat((rrwp_edge_attr, rrwp_edge_attr_wo_norm), dim=-1)
            diag_index = torch.arange(rrwp_edge_attr.shape[1])
            rrwp_node_attr = rrwp_edge_attr[:, diag_index, diag_index, :]
            # self.eigenfeatures = EigenFeatures(mode='all')

            return utils.PlaceHolder(
                X=rrwp_node_attr,
                E=rrwp_edge_attr,
                y=torch.hstack((n, y_cycles)),
            )

        elif self.features_type == "rrwp_only":
            E = noisy_data["E_t"].float()[..., 1:].sum(-1)  # bs, n, n
            rrwp_edge_attr = self.RRWP(E, k=self.rrwp_steps)
            diag_index = torch.arange(rrwp_edge_attr.shape[1])
            rrwp_node_attr = rrwp_edge_attr[:, diag_index, diag_index, :]

            return utils.PlaceHolder(
                X=rrwp_node_attr,
                E=rrwp_edge_attr,
                y=n,
            )

        elif self.features_type == "rrwp_comp":
            E = noisy_data["E_t"].float()[..., 1:].sum(-1)  # bs, n, n
            rrwp_edge_attr = self.RRWP(E, k=int(self.rrwp_steps / 2))
            diag_index = torch.arange(rrwp_edge_attr.shape[1])
            rrwp_node_attr = rrwp_edge_attr[:, diag_index, diag_index, :]

            comp_E = 1 - noisy_data["E_t"].float()[..., 1:].sum(-1)  # bs, n, n
            comp_rrwp_edge_attr = self.RRWP(comp_E, k=int(self.rrwp_steps / 2))
            comp_rrwp_node_attr = comp_rrwp_edge_attr[:, diag_index, diag_index, :]

            return utils.PlaceHolder(
                X=torch.cat((rrwp_node_attr, comp_rrwp_node_attr), dim=-1),
                E=torch.cat((rrwp_edge_attr, comp_rrwp_edge_attr), dim=-1),
                y=torch.hstack((n, y_cycles)),
            )

        elif self.features_type == "all":
            eigenfeatures = self.eigenfeatures(noisy_data)
            E = noisy_data["E_t"]
            extra_edge_attr = torch.zeros((*E.shape[:-1], 0)).type_as(E)
            n_components, batched_eigenvalues, nonlcc_indicator, k_lowest_eigvec = (
                eigenfeatures  # (bs, 1), (bs, 10),
            )

            return utils.PlaceHolder(
                X=torch.cat(
                    (x_cycles, nonlcc_indicator, k_lowest_eigvec),
                    dim=-1,
                ),
                E=extra_edge_attr,
                y=torch.hstack((n, y_cycles, n_components, batched_eigenvalues)),
            )

        else:
            raise ValueError(f"Features type {self.features_type} not implemented")


class RRWPFeatures:
    def __init__(self, k=10, normalize=True):
        self.k = k
        self.normalize = normalize

    def __call__(self, E, k=None):
        k = k or self.k

        (
            bs,
            n,
            _,
        ) = E.shape
        if self.normalize:
            degree = torch.zeros(bs, n, n, device=E.device)
            to_fill = 1 / (E.sum(dim=-1).float())
            to_fill[E.sum(dim=-1).float() == 0] = 0
            degree = torch.diagonal_scatter(degree, to_fill, dim1=1, dim2=2)
            E = degree @ E

        id = torch.eye(n, device=E.device).unsqueeze(0).repeat(bs, 1, 1)
        rrwp_list = [id]

        for i in range(k - 1):
            cur_rrwp = rrwp_list[-1] @ E
            rrwp_list.append(cur_rrwp)

        return torch.stack(rrwp_list, -1)


class NodeCycleFeatures:
    def __init__(self):
        self.kcycles = KNodeCycles()

    def __call__(self, noisy_data):
        adj_matrix = noisy_data["E_t"][..., 1:].sum(dim=-1).float()

        x_cycles, y_cycles = self.kcycles.k_cycles(
            adj_matrix=adj_matrix
        )  # (bs, n_cycles)
        x_cycles = x_cycles.type_as(adj_matrix) * noisy_data["node_mask"].unsqueeze(-1)
        # Avoid large values when the graph is dense
        x_cycles = x_cycles / 10
        y_cycles = y_cycles / 10
        x_cycles[x_cycles > 1] = 1
        y_cycles[y_cycles > 1] = 1
        return x_cycles, y_cycles


class EigenFeatures:
    """
    Code taken from : https://github.com/Saro00/DGN/blob/master/models/pytorch/eigen_agg.py
    """

    def __init__(self, mode):
        """mode: 'eigenvalues' or 'all'"""
        self.mode = mode

    def __call__(self, noisy_data):
        E_t = noisy_data["E_t"]
        mask = noisy_data["node_mask"]
        A = E_t[..., 1:].sum(dim=-1).float() * mask.unsqueeze(1) * mask.unsqueeze(2)
        # L = compute_laplacian(A, normalize="sym")
        L = compute_laplacian(A, normalize=False)
        mask_diag = 2 * L.shape[-1] * torch.eye(A.shape[-1]).type_as(L).unsqueeze(0)
        mask_diag = mask_diag * (~mask.unsqueeze(1)) * (~mask.unsqueeze(2))
        L = L * mask.unsqueeze(1) * mask.unsqueeze(2) + mask_diag

        if self.mode == "eigenvalues":
            eigvals = torch.linalg.eigvalsh(L)  # bs, n
            eigvals = eigvals.type_as(A) / torch.sum(mask, dim=1, keepdim=True)

            n_connected_comp, batch_eigenvalues = get_eigenvalues_features(
                eigenvalues=eigvals
            )
            return n_connected_comp.type_as(A), batch_eigenvalues.type_as(A)

        elif self.mode == "all":
            eigvals, eigvectors = torch.linalg.eigh(L)
            # print(eigvals)
            eigvals = eigvals.type_as(A) / torch.sum(mask, dim=1, keepdim=True)
            eigvectors = eigvectors * mask.unsqueeze(2) * mask.unsqueeze(1)
            # Retrieve eigenvalues features
            n_connected_comp, batch_eigenvalues = get_eigenvalues_features(
                eigenvalues=eigvals
            )

            # Retrieve eigenvectors features
            nonlcc_indicator, k_lowest_eigenvector = get_eigenvectors_features(
                vectors=eigvectors,
                node_mask=noisy_data["node_mask"],
                n_connected=n_connected_comp,
            )
            return (
                n_connected_comp,
                batch_eigenvalues,
                nonlcc_indicator,
                k_lowest_eigenvector,
            )
        else:
            raise NotImplementedError(f"Mode {self.mode} is not implemented")


def compute_laplacian(adjacency, normalize: bool):
    """
    adjacency : batched adjacency matrix (bs, n, n)
    normalize: can be None, 'sym' or 'rw' for the combinatorial, symmetric normalized or random walk Laplacians
    Return:
        L (n x n ndarray): combinatorial or symmetric normalized Laplacian.
    """
    diag = torch.sum(adjacency, dim=-1)  # (bs, n)
    n = diag.shape[-1]
    D = torch.diag_embed(diag)  # Degree matrix      # (bs, n, n)
    combinatorial = D - adjacency  # (bs, n, n)

    if not normalize:
        return (combinatorial + combinatorial.transpose(1, 2)) / 2

    diag0 = diag.clone()
    diag[diag == 0] = 1e-12

    diag_norm = 1 / torch.sqrt(diag)  # (bs, n)
    D_norm = torch.diag_embed(diag_norm)  # (bs, n, n)
    L = torch.eye(n, device=adjacency.device).unsqueeze(0) - D_norm @ adjacency @ D_norm
    L[diag0 == 0] = 0
    return (L + L.transpose(1, 2)) / 2


def get_eigenvalues_features(eigenvalues, k=5):
    """
    values : eigenvalues -- (bs, n)
    node_mask: (bs, n)
    k: num of non zero eigenvalues to keep
    """
    ev = eigenvalues
    bs, n = ev.shape
    n_connected_components = (ev < 1e-5).sum(dim=-1)
    try:
        assert (n_connected_components > 0).all(), (n_connected_components, ev)
    except:
        import pdb

        pdb.set_trace()

    to_extend = max(n_connected_components) + k - n
    if to_extend > 0:
        eigenvalues = torch.hstack(
            (eigenvalues, 2 * torch.ones(bs, to_extend).type_as(eigenvalues))
        )
    indices = torch.arange(k).type_as(eigenvalues).long().unsqueeze(
        0
    ) + n_connected_components.unsqueeze(1)
    first_k_ev = torch.gather(eigenvalues, dim=1, index=indices)
    return n_connected_components.unsqueeze(-1), first_k_ev


def get_eigenvectors_features(vectors, node_mask, n_connected, k=2):
    """
    vectors (bs, n, n) : eigenvectors of Laplacian IN COLUMNS
    returns:
        not_lcc_indicator : indicator vectors of largest connected component (lcc) for each graph  -- (bs, n, 1)
        k_lowest_eigvec : k first eigenvectors for the largest connected component   -- (bs, n, k)
    """
    bs, n = vectors.size(0), vectors.size(1)

    # Create an indicator for the nodes outside the largest connected components
    first_ev = torch.round(vectors[:, :, 0], decimals=3) * node_mask  # bs, n
    # Add random value to the mask to prevent 0 from becoming the mode
    random = torch.randn(bs, n, device=node_mask.device) * (~node_mask)  # bs, n
    first_ev = first_ev + random
    most_common = torch.mode(first_ev, dim=1).values  # values: bs -- indices: bs
    mask = ~(first_ev == most_common.unsqueeze(1))
    not_lcc_indicator = (mask * node_mask).unsqueeze(-1).float()

    # Get the eigenvectors corresponding to the first nonzero eigenvalues
    to_extend = max(n_connected) + k - n
    if to_extend > 0:
        vectors = torch.cat(
            (vectors, torch.zeros(bs, n, to_extend).type_as(vectors)), dim=2
        )  # bs, n , n + to_extend
    indices = torch.arange(k).type_as(vectors).long().unsqueeze(0).unsqueeze(
        0
    ) + n_connected.unsqueeze(
        2
    )  # bs, 1, k
    indices = indices.expand(-1, n, -1)  # bs, n, k
    first_k_ev = torch.gather(vectors, dim=2, index=indices)  # bs, n, k
    first_k_ev = first_k_ev * node_mask.unsqueeze(2)

    return not_lcc_indicator, first_k_ev


def batch_trace(X):
    """
    Expect a matrix of shape B N N, returns the trace in shape B
    :param X:
    :return:
    """
    diag = torch.diagonal(X, dim1=-2, dim2=-1)
    trace = diag.sum(dim=-1)
    return trace


def batch_diagonal(X):
    """
    Extracts the diagonal from the last two dims of a tensor
    :param X:
    :return:
    """
    return torch.diagonal(X, dim1=-2, dim2=-1)


class KNodeCycles:
    """Builds cycle counts for each node in a graph."""

    def __init__(self):
        super().__init__()

    def calculate_kpowers(self):
        self.k1_matrix = self.adj_matrix.float()
        self.d = self.adj_matrix.sum(dim=-1)
        self.k2_matrix = self.k1_matrix @ self.adj_matrix.float()
        self.k3_matrix = self.k2_matrix @ self.adj_matrix.float()
        self.k4_matrix = self.k3_matrix @ self.adj_matrix.float()
        self.k5_matrix = self.k4_matrix @ self.adj_matrix.float()
        self.k6_matrix = self.k5_matrix @ self.adj_matrix.float()

    def k3_cycle(self):
        """tr(A ** 3)."""
        c3 = batch_diagonal(self.k3_matrix)
        return (c3 / 2).unsqueeze(-1).float(), (torch.sum(c3, dim=-1) / 6).unsqueeze(
            -1
        ).float()

    def k4_cycle(self):
        diag_a4 = batch_diagonal(self.k4_matrix)
        c4 = (
            diag_a4
            - self.d * (self.d - 1)
            - (self.adj_matrix @ self.d.unsqueeze(-1)).sum(dim=-1)
        )
        return (c4 / 2).unsqueeze(-1).float(), (torch.sum(c4, dim=-1) / 8).unsqueeze(
            -1
        ).float()

    def k5_cycle(self):
        diag_a5 = batch_diagonal(self.k5_matrix)
        triangles = batch_diagonal(self.k3_matrix)
        c5 = (
            diag_a5
            - 2 * triangles * self.d
            - (self.adj_matrix @ triangles.unsqueeze(-1)).sum(dim=-1)
            + triangles
        )
        return (c5 / 2).unsqueeze(-1).float(), (c5.sum(dim=-1) / 10).unsqueeze(
            -1
        ).float()

    def k6_cycle(self):
        term_1_t = batch_trace(self.k6_matrix)
        term_2_t = batch_trace(self.k3_matrix**2)
        term3_t = torch.sum(self.adj_matrix * self.k2_matrix.pow(2), dim=[-2, -1])
        d_t4 = batch_diagonal(self.k2_matrix)
        a_4_t = batch_diagonal(self.k4_matrix)
        term_4_t = (d_t4 * a_4_t).sum(dim=-1)
        term_5_t = batch_trace(self.k4_matrix)
        term_6_t = batch_trace(self.k3_matrix)
        term_7_t = batch_diagonal(self.k2_matrix).pow(3).sum(-1)
        term8_t = torch.sum(self.k3_matrix, dim=[-2, -1])
        term9_t = batch_diagonal(self.k2_matrix).pow(2).sum(-1)
        term10_t = batch_trace(self.k2_matrix)

        c6_t = (
            term_1_t
            - 3 * term_2_t
            + 9 * term3_t
            - 6 * term_4_t
            + 6 * term_5_t
            - 4 * term_6_t
            + 4 * term_7_t
            + 3 * term8_t
            - 12 * term9_t
            + 4 * term10_t
        )
        return None, (c6_t / 12).unsqueeze(-1).float()

    def k_cycles(self, adj_matrix, verbose=False):
        self.adj_matrix = adj_matrix
        self.calculate_kpowers()

        k3x, k3y = self.k3_cycle()
        assert (k3x >= -0.1).all()

        k4x, k4y = self.k4_cycle()
        assert (k4x >= -0.1).all()

        k5x, k5y = self.k5_cycle()
        assert (k5x >= -0.1).all(), k5x

        _, k6y = self.k6_cycle()
        assert (k6y >= -0.1).all()

        kcyclesx = torch.cat([k3x, k4x, k5x], dim=-1)
        kcyclesy = torch.cat([k3y, k4y, k5y, k6y], dim=-1)
        return kcyclesx, kcyclesy
