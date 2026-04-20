import torch

from src import utils
from flow_matching import flow_matching_utils
from flow_matching.motif_editor import add_rings


class NoiseDistribution:

    def __init__(self, model_transition, dataset_infos, cfg=None):

        self.x_num_classes = dataset_infos.output_dims["X"]
        self.e_num_classes = dataset_infos.output_dims["E"]
        self.y_num_classes = dataset_infos.output_dims["y"]
        self.x_added_classes = 0
        self.e_added_classes = 0
        self.y_added_classes = 0
        self.transition = model_transition

        if model_transition == "uniform":
            x_limit = torch.ones(self.x_num_classes) / self.x_num_classes
            e_limit = torch.ones(self.e_num_classes) / self.e_num_classes

        elif model_transition == "absorbfirst":
            x_limit = torch.zeros(self.x_num_classes)
            x_limit[0] = 1
            e_limit = torch.zeros(self.e_num_classes)
            e_limit[0] = 1

        elif model_transition == "argmax":
            node_types = dataset_infos.node_types.float()
            x_marginals = node_types / torch.sum(node_types)

            edge_types = dataset_infos.edge_types.float()
            e_marginals = edge_types / torch.sum(edge_types)

            x_max_dim = torch.argmax(x_marginals)
            e_max_dim = torch.argmax(e_marginals)
            x_limit = torch.zeros(self.x_num_classes)
            x_limit[x_max_dim] = 1
            e_limit = torch.zeros(self.e_num_classes)
            e_limit[e_max_dim] = 1

        elif model_transition == "absorbing":
            # only add virtual classes when there are several
            if self.x_num_classes > 1:
                # if self.x_num_classes >= 1:
                self.x_num_classes += 1
                self.x_added_classes = 1
            if self.e_num_classes > 1:
                self.e_num_classes += 1
                self.e_added_classes = 1

            x_limit = torch.zeros(self.x_num_classes)
            x_limit[-1] = 1
            e_limit = torch.zeros(self.e_num_classes)
            e_limit[-1] = 1

        elif model_transition == "marginal":

            node_types = dataset_infos.node_types.float()
            x_limit = node_types / torch.sum(node_types)

            edge_types = dataset_infos.edge_types.float()
            e_limit = edge_types / torch.sum(edge_types)

        elif model_transition == "motif_edited_A":
            # Original marginals — used to sample z_T before ring edits and
            # as the distribution passed to add_rings.
            node_types = dataset_infos.node_types.float()
            original_x = node_types / node_types.sum()
            edge_types = dataset_infos.edge_types.float()
            original_e = edge_types / edge_types.sum()
            # y_limit is set below; use a zero-length placeholder here so the
            # original_limit_dist can be constructed after y_limit is assigned.
            self._original_x = original_x
            self._original_e = original_e

            # Precomputed edited marginals — used as limit_dist for the rate matrix.
            saved = torch.load(cfg.model.motif_marginals_path, map_location="cpu", weights_only=True)
            x_limit = saved["X"]
            e_limit = saved["E"]

            self.ring_specs = [tuple(rs) for rs in cfg.model.motif_ring_specs]

        elif model_transition == "edge_marginal":
            x_limit = torch.ones(self.x_num_classes) / self.x_num_classes

            edge_types = dataset_infos.edge_types.float()
            e_limit = edge_types / torch.sum(edge_types)

        elif model_transition == "node_marginal":
            e_limit = torch.ones(self.e_num_classes) / self.e_num_classes

            node_types = dataset_infos.node_types.float()
            x_limit = node_types / torch.sum(node_types)

        else:
            raise ValueError(f"Unknown transition model: {model_transition}")

        y_limit = torch.ones(self.y_num_classes) / self.y_num_classes  # typically dummy
        print(
            f"Limit distribution of the classes | Nodes: {x_limit} | Edges: {e_limit}"
        )
        self.limit_dist = utils.PlaceHolder(X=x_limit, E=e_limit, y=y_limit)

        if model_transition == "motif_edited_A":
            self.original_limit_dist = utils.PlaceHolder(
                X=self._original_x, E=self._original_e, y=y_limit
            )

    def sample_initial_noise(self, node_mask):
        """Sample z_T and return (z_T, node_mask).

        For motif_edited_A: samples from the original marginal then applies
        add_rings, returning the edited graph and the updated node_mask.
        For all other transitions: delegates to sample_discrete_feature_noise
        and returns node_mask unchanged.
        """
        if self.transition == "motif_edited_A":
            device = node_mask.device
            orig = utils.PlaceHolder(
                X=self.original_limit_dist.X.to(device),
                E=self.original_limit_dist.E.to(device),
                y=self.original_limit_dist.y.to(device),
            )
            z = flow_matching_utils.sample_discrete_feature_noise(orig, node_mask)
            z, updated_mask = add_rings(z, node_mask, self.ring_specs, orig)
            return z, updated_mask
        else:
            z = flow_matching_utils.sample_discrete_feature_noise(self.limit_dist, node_mask)
            return z, node_mask

    def update_input_output_dims(self, input_dims):
        input_dims["X"] += self.x_added_classes
        input_dims["E"] += self.e_added_classes
        input_dims["y"] += self.y_added_classes

    def update_dataset_infos(self, dataset_infos):
        if hasattr(dataset_infos, "atom_decoder"):
            dataset_infos.atom_decoder = (
                dataset_infos.atom_decoder + ["Y"] * self.x_added_classes
            )

    def get_limit_dist(self):
        return self.limit_dist

    def get_noise_dims(self):
        return {
            "X": len(self.limit_dist.X),
            "E": len(self.limit_dist.E),
            "y": len(self.limit_dist.E),
        }

    def ignore_virtual_classes(self, X, E, y=None):
        if self.transition == "absorbing":
            new_X = X[..., : -self.x_added_classes]
            new_E = E[..., : -self.e_added_classes]
            new_y = y[..., : -self.y_added_classes] if y is not None else None
            return new_X, new_E, new_y
        else:
            return X, E, y

    def add_virtual_classes(self, X, E, y=None):
        x_virtual = torch.zeros_like(X[..., :1]).repeat(1, 1, self.x_added_classes)
        new_X = torch.cat([X, x_virtual], dim=-1)

        e_virtual = torch.zeros_like(E[..., :1]).repeat(1, 1, 1, self.e_added_classes)
        new_E = torch.cat([E, e_virtual], dim=-1)

        if y is not None:
            y_virtual = torch.zeros_like(y[..., :1]).repeat(1, self.y_added_classes)
            new_y = torch.cat([y, y_virtual], dim=-1)
        else:
            new_y = None

        return new_X, new_E, new_y
