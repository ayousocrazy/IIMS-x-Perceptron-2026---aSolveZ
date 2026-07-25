import torch.nn as nn

from models.encoder import HGTEncoder
from models.decoder import DistMultDecoder


class DrugSafetyHGNN(nn.Module):

    def __init__(
        self,
        data,
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        num_relations=None,
    ):

        super().__init__()

        assert num_relations is not None, (
            "num_relations must be provided (e.g. len(relation_vocab) or "
            "via DistMultDecoder.from_relation_vocab)"
        )

        self.encoder = HGTEncoder(
            data,
            hidden_dim,
            num_heads,
            num_layers,
        )

        self.decoder = DistMultDecoder(
            num_relations,
            hidden_dim,
        )

    def forward(self, data, edge_index, edge_type, message_passing_edges=None):
        embeddings = self.encoder(data, edge_index_dict=message_passing_edges)

        drug_embeddings = embeddings["drug"]

        scores = self.decoder(
            drug_embeddings,
            edge_index,
            edge_type,
        )

        return scores