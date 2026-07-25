import argparse
import json
from pathlib import Path

import torch

from models.hgnn import DrugSafetyHGNN

DDI_KEY = ("drug", "polypharmacy", "drug")

DEFAULT_EDGE_TYPE = 0


class DrugSafetyPredictor:
    def __init__(self, data_path, checkpoint_path, hidden_dim=64,
                 num_heads=2, num_layers=2, device=None,
                 drug_nodes_csv=None, relation_lookup_csv=None,
                 drug_names_json=None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.data = torch.load(data_path, weights_only=False).to(self.device)
        num_relations = int(self.data[DDI_KEY].edge_type.max().item()) + 1

        self.model = DrugSafetyHGNN(
            self.data, hidden_dim=hidden_dim, num_heads=num_heads,
            num_layers=num_layers, num_relations=num_relations,
        ).to(self.device)

        state_dict = torch.load(checkpoint_path, weights_only=True, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        with torch.no_grad():
            embeddings = self.model.encoder(self.data)
        self.drug_embeddings = embeddings["drug"]

        self.cid_to_index = {}
        self.index_to_cid = {}
        if drug_nodes_csv is not None:
            self.load_drug_id_map(drug_nodes_csv)

        self.relation_names = {}
        if relation_lookup_csv is not None:
            self.load_relation_lookup(relation_lookup_csv)

        self.drug_names = {}
        if drug_names_json is not None:
            self.load_drug_names(drug_names_json)

    def load_drug_names(self, path):
        self.drug_names = json.loads(Path(path).read_text())

    def name_for_cid(self, cid):
        entry = self.drug_names.get(cid)
        if not entry:
            return cid
        return entry[0] if isinstance(entry, list) else entry

    def aliases_for_cid(self, cid):
        entry = self.drug_names.get(cid)
        if not entry:
            return [cid]
        return entry if isinstance(entry, list) else [entry]

    def load_relation_lookup(self, path):
        import csv
        path = Path(path)
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            has_names = reader.fieldnames and "side_effect_name" in reader.fieldnames
            for row in reader:
                label = row["side_effect_name"] if has_names and row.get("side_effect_name") else row["side_effect"]
                self.relation_names[int(row["relation_id"])] = label

    def load_drug_id_map(self, path):
        import csv
        path = Path(path)
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                cid = row["drug_id"]
                self.cid_to_index[cid] = idx
                self.index_to_cid[idx] = cid

        n_nodes = self.data["drug"].num_nodes
        if len(self.cid_to_index) != n_nodes:
            raise ValueError(
                f"drug_nodes.csv has {len(self.cid_to_index)} rows but the "
                f"graph has {n_nodes} drug nodes -- wrong CSV, or the CSV "
                f"was modified/reordered since the graph was built. Do not "
                f"trust predictions until this matches."
            )

    def resolve_index(self, drug):
        """Accepts either an int node index or a PubChem CID string (e.g. 'CID4168')."""
        if isinstance(drug, int):
            return drug
        if drug in self.cid_to_index:
            return self.cid_to_index[drug]
        raise ValueError(
            f"Unknown drug identifier: {drug!r}. Pass a node index (int) "
            f"or a PubChem CID string present in drug_nodes.csv (e.g. 'CID4168'). "
            f"Did you forget to pass drug_nodes_csv= when constructing DrugSafetyPredictor?"
        )

    @torch.no_grad()
    def predict_pairs(self, pairs, edge_type=DEFAULT_EDGE_TYPE):
        idx_a = [self.resolve_index(a) for a, _ in pairs]
        idx_b = [self.resolve_index(b) for _, b in pairs]

        edge_index = torch.tensor([idx_a, idx_b], dtype=torch.long, device=self.device)

        if isinstance(edge_type, int):
            edge_type_t = torch.full((len(pairs),), edge_type, dtype=torch.long, device=self.device)
        else:
            edge_type_t = torch.tensor(edge_type, dtype=torch.long, device=self.device)

        raw_scores = self.model.decoder(self.drug_embeddings, edge_index, edge_type_t)
        probs = torch.sigmoid(raw_scores)
        return probs.cpu().tolist()

    def predict_pair(self, drug_a, drug_b, edge_type=DEFAULT_EDGE_TYPE):
        return self.predict_pairs([(drug_a, drug_b)], edge_type=edge_type)[0]

    @torch.no_grad()
    def predict_pair_risk(self, drug_a, drug_b, top_k=5, relation_names=None):
        idx_a = self.resolve_index(drug_a)
        idx_b = self.resolve_index(drug_b)

        if relation_names is None:
            relation_names = self.relation_names

        num_relations = self.model.decoder.num_relations if hasattr(self.model.decoder, "num_relations") \
            else int(self.data[DDI_KEY].edge_type.max().item()) + 1

        edge_index = torch.tensor(
            [[idx_a] * num_relations, [idx_b] * num_relations],
            dtype=torch.long, device=self.device,
        )
        edge_type_t = torch.arange(num_relations, dtype=torch.long, device=self.device)

        raw_scores = self.model.decoder(self.drug_embeddings, edge_index, edge_type_t)
        probs = torch.sigmoid(raw_scores).cpu()

        top_indices = torch.topk(probs, k=min(top_k, num_relations)).indices.tolist()
        top_relations = [
            (relation_names[i] if relation_names and i in relation_names else i, probs[i].item())
            for i in top_indices
        ]

        return {
            "max_prob": probs.max().item(),
            "mean_prob": probs.mean().item(),
            "top_relations": top_relations,
        }


def main():
    parser = argparse.ArgumentParser(description="Score drug pair(s) for interaction risk")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--drug-nodes-csv", required=True,
                         help="path to drug_nodes.csv (or drug_nodes_with_smiles.csv) "
                              "-- must be the exact file/row-order used when the graph was built")
    parser.add_argument("--relation-lookup-csv", default=None,
                         help="path to data/processed/polypharmacy_relation_lookup.csv -- "
                              "if provided, risk summaries show side-effect names instead of "
                              "raw relation indices")
    parser.add_argument("--drug-a", required=True, help="PubChem CID (e.g. CID4168) or raw node index")
    parser.add_argument("--drug-b", required=True, help="PubChem CID (e.g. CID4168) or raw node index")
    parser.add_argument("--edge-type", type=int, default=None,
                         help="score ONE specific relation index instead of the full risk "
                              "summary across all 1301 relations")
    parser.add_argument("--top-k", type=int, default=5,
                         help="how many top relations to show in the risk summary (default 5)")
    args = parser.parse_args()

    predictor = DrugSafetyPredictor(
        data_path=args.data_path,
        checkpoint_path=args.checkpoint,
        drug_nodes_csv=args.drug_nodes_csv,
        relation_lookup_csv=args.relation_lookup_csv,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
    )

    # allow raw integer node indices too, for debugging
    drug_a = int(args.drug_a) if args.drug_a.isdigit() else args.drug_a
    drug_b = int(args.drug_b) if args.drug_b.isdigit() else args.drug_b

    if args.edge_type is not None:
        prob = predictor.predict_pair(drug_a, drug_b, edge_type=args.edge_type)
        print(f"Drug pair ({args.drug_a}, {args.drug_b}), relation={args.edge_type}: "
              f"predicted probability = {prob:.4f}")
    else:
        risk = predictor.predict_pair_risk(drug_a, drug_b, top_k=args.top_k)
        print(f"Drug pair ({args.drug_a}, {args.drug_b}) -- overall risk summary:")
        print(f"  max probability (any relation):  {risk['max_prob']:.4f}")
        print(f"  mean probability (all relations): {risk['mean_prob']:.4f}")
        print(f"  top {args.top_k} relations by probability:")
        for relation, prob in risk["top_relations"]:
            print(f"    {relation}: {prob:.4f}")


if __name__ == "__main__":
    main()