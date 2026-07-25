import json
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn


class DistMultDecoder(nn.Module):
    def __init__(self, num_relations, hidden_dim=128):
        super().__init__()
        self.num_relations = num_relations
        self.relation_embedding = nn.Embedding(num_relations, hidden_dim)

    @classmethod
    def from_relation_vocab(cls, relation_vocab_path, hidden_dim=128):
        path = Path(relation_vocab_path)
        if path.suffix == ".csv":
            num_relations = len(pd.read_csv(path))
        else:
            num_relations = len(json.loads(path.read_text()))
        return cls(num_relations=num_relations, hidden_dim=hidden_dim)

    def forward(self, drug_embeddings, edge_index, edge_type):
        max_relation_id = edge_type.max().item()
        assert max_relation_id < self.num_relations, (
            f"edge_type contains relation id {max_relation_id}, but this "
            f"decoder was constructed with num_relations={self.num_relations}. "
            f"num_relations must exactly match len(relation_vocab.json) from "
            f"Step 1 -- use DistMultDecoder.from_relation_vocab(...) to avoid "
            f"this drifting out of sync."
        )

        src, dst = edge_index
        src_emb = drug_embeddings[src]
        dst_emb = drug_embeddings[dst]
        rel_emb = self.relation_embedding(edge_type)

        score = (src_emb * rel_emb * dst_emb).sum(dim=1)
        return score