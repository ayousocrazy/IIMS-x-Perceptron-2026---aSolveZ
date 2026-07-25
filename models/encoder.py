import torch
import torch.nn as nn
from torch_geometric.nn import HGTConv


def build_message_passing_edges(data, exclude=(("drug", "polypharmacy", "drug"),)):
    exclude = set(exclude)
    return {
        edge_type: data[edge_type].edge_index
        for edge_type in data.edge_types
        if edge_type not in exclude
    }


def split_polypharmacy_edges(data, val_frac=0.1, test_frac=0.1, seed=42, edge_key=("drug", "polypharmacy", "drug")):
    edge_index = data[edge_key].edge_index
    edge_type = data[edge_key].edge_type
    n = edge_index.shape[1]

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator)

    n_val = int(n * val_frac)
    n_test = int(n * test_frac)
    n_train = n - n_val - n_test

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    def _slice(idx):
        return {"edge_index": edge_index[:, idx], "edge_type": edge_type[idx]}

    return {"train": _slice(train_idx), "val": _slice(val_idx), "test": _slice(test_idx)}


def validate_edge_index_dict(data, edge_index_dict):
    for edge_type, edge_index in edge_index_dict.items():
        if edge_index.numel() == 0:
            continue

        src_type, _, dst_type = edge_type
        n_src = data[src_type].num_nodes
        n_dst = data[dst_type].num_nodes

        max_src, min_src = edge_index[0].max().item(), edge_index[0].min().item()
        max_dst, min_dst = edge_index[1].max().item(), edge_index[1].min().item()

        if max_src >= n_src or min_src < 0:
            raise ValueError(
                f"{edge_type}: src index range [{min_src}, {max_src}] out of "
                f"bounds for '{src_type}' with num_nodes={n_src}. This usually "
                f"means edge_index_dict was built against a different/stale "
                f"version of the graph than `data` currently has."
            )
        if max_dst >= n_dst or min_dst < 0:
            raise ValueError(
                f"{edge_type}: dst index range [{min_dst}, {max_dst}] out of "
                f"bounds for '{dst_type}' with num_nodes={n_dst}. This usually "
                f"means edge_index_dict was built against a different/stale "
                f"version of the graph than `data` currently has."
            )


class HGTEncoder(nn.Module):
    def __init__(self, data, hidden_dim=128, num_heads=4, num_layers=2):
        super().__init__()
        assert hidden_dim % num_heads == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads "
            f"({num_heads}) -- HGTConv splits hidden_dim evenly across heads"
        )

        self.node_types = set(data.node_types)

        drug_input_dim = data["drug"].x.shape[1]
        self.drug_projection = nn.Linear(drug_input_dim, hidden_dim)
        self.gene_embedding = nn.Embedding(data["gene"].num_nodes, hidden_dim)
        self.side_effect_embedding = nn.Embedding(data["side_effect"].num_nodes, hidden_dim)

        self.convs = nn.ModuleList([
            HGTConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                metadata=data.metadata(),
                heads=num_heads,
            )
            for _ in range(num_layers)
        ])

    def input_features(self, data):
        """Build the initial per-node-type feature dict, before any message passing."""
        drug_x = self.drug_projection(data["drug"].x)
        device = drug_x.device
        gene_x = self.gene_embedding(torch.arange(data["gene"].num_nodes, device=device))
        side_x = self.side_effect_embedding(
            torch.arange(data["side_effect"].num_nodes, device=device)
        )
        return {"drug": drug_x, "gene": gene_x, "side_effect": side_x}

    def forward(self, data, edge_index_dict=None):
        if edge_index_dict is None:
            edge_index_dict = build_message_passing_edges(data)

        validate_edge_index_dict(data, edge_index_dict)

        x_dict = self.input_features(data)

        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            missing = self.node_types - set(x_dict.keys())
            if missing:
                raise ValueError(
                    f"After conv layer {i}, node type(s) {missing} produced no "
                    f"output -- they never appear as the destination of any "
                    f"edge type reachable from the current edge_index_dict. "
                    f"Add at least one incoming edge type for these node "
                    f"types, or exclude them from this encoder's node types."
                )

        return x_dict


if __name__ == "__main__":
    from pathlib import Path

    GRAPH_DIR = Path("../graph")

    data = torch.load(GRAPH_DIR / "heterodata_with_features.pt", weights_only=False)

    encoder = HGTEncoder(data)
    embeddings = encoder(data)

    print(embeddings["drug"].shape)
    print(embeddings["gene"].shape)
    print(embeddings["side_effect"].shape)