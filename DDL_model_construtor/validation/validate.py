import argparse
import csv
import random

import pandas as pd
import torch

from inference.inference import DrugSafetyPredictor


def main():
    parser = argparse.ArgumentParser(description="Validate inference against known DDI triples")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--drug-nodes-csv", required=True)
    parser.add_argument("--relation-lookup-csv", required=True)
    parser.add_argument("--drug-combo-csv", required=True,
                         help="data/processed/drug-combo-normalized.csv -- normalized STITCH IDs, "
                              "must match drug_nodes.csv's node ordering")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--n-samples", type=int, default=200,
                         help="number of positive triples to sample (negatives match this count)")
    parser.add_argument("--n-examples", type=int, default=8,
                         help="how many individual example rows to print")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    predictor = DrugSafetyPredictor(
        data_path=args.data_path,
        checkpoint_path=args.checkpoint,
        drug_nodes_csv=args.drug_nodes_csv,
        relation_lookup_csv=args.relation_lookup_csv,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
    )

    # code -> relation_id, from the same lookup used by the predictor
    code_to_relid = {}
    with open(args.relation_lookup_csv, newline="") as f:
        for row in csv.DictReader(f):
            code_to_relid[row["side_effect"]] = int(row["relation_id"])

    combo = pd.read_csv(args.drug_combo_csv)
    combo = combo[combo["STITCH 1"].isin(predictor.cid_to_index) &
                  combo["STITCH 2"].isin(predictor.cid_to_index) &
                  combo["Polypharmacy Side Effect"].isin(code_to_relid)]

    if len(combo) == 0:
        raise RuntimeError(
            "No rows in drug-combo.csv matched both the drug_nodes.csv CIDs and the "
            "relation lookup codes -- check that --drug-combo-csv is the NORMALIZED "
            "version (data/processed/drug-combo-normalized.csv), not the raw one."
        )

    # group known relations per (d1, d2) pair, for building "same pair, wrong relation" negatives
    pair_known_relations = {}
    for _, row in combo.iterrows():
        key = (row["STITCH 1"], row["STITCH 2"])
        pair_known_relations.setdefault(key, set()).add(code_to_relid[row["Polypharmacy Side Effect"]])

    n = min(args.n_samples, len(combo))
    sample = combo.sample(n=n, random_state=args.seed)

    all_relation_ids = list(code_to_relid.values())
    all_cids = list(predictor.cid_to_index.keys())

    pos_scores, same_pair_neg_scores, random_neg_scores = [], [], []
    examples = []

    for _, row in sample.iterrows():
        d1, d2 = row["STITCH 1"], row["STITCH 2"]
        rel_id = code_to_relid[row["Polypharmacy Side Effect"]]
        known = pair_known_relations[(d1, d2)]

        pos_score = predictor.predict_pair(d1, d2, edge_type=rel_id)
        pos_scores.append(pos_score)

        # same pair, wrong relation
        wrong_rel = rng.choice(all_relation_ids)
        tries = 0
        while wrong_rel in known and tries < 10:
            wrong_rel = rng.choice(all_relation_ids)
            tries += 1
        same_pair_score = predictor.predict_pair(d1, d2, edge_type=wrong_rel)
        same_pair_neg_scores.append(same_pair_score)

        # random pair, random relation (best-effort: not guaranteed unseen, but very likely
        # given 1301 relations x many drug pairs)
        rand_d1, rand_d2 = rng.choice(all_cids), rng.choice(all_cids)
        rand_rel = rng.choice(all_relation_ids)
        random_score = predictor.predict_pair(rand_d1, rand_d2, edge_type=rand_rel)
        random_neg_scores.append(random_score)

        if len(examples) < args.n_examples:
            examples.append({
                "drug_a": d1, "drug_b": d2,
                "true_relation": predictor.relation_names.get(rel_id, row["Polypharmacy Side Effect"]),
                "true_relation_score": pos_score,
                "wrong_relation": predictor.relation_names.get(wrong_rel, wrong_rel),
                "wrong_relation_score": same_pair_score,
            })

    def summarize(name, scores):
        s = torch.tensor(scores)
        print(f"{name}: mean={s.mean():.4f}  median={s.median():.4f}  "
              f"min={s.min():.4f}  max={s.max():.4f}  n={len(scores)}")

    print(f"\n=== Validation over {n} known triples ===\n")
    summarize("Known (positive) triples       ", pos_scores)
    summarize("Same pair, wrong relation (neg)", same_pair_neg_scores)
    summarize("Random pair, random relation   ", random_neg_scores)

    pos_mean = sum(pos_scores) / len(pos_scores)
    same_pair_mean = sum(same_pair_neg_scores) / len(same_pair_neg_scores)
    random_mean = sum(random_neg_scores) / len(random_neg_scores)

    print("\n=== Verdict ===")
    if pos_mean > same_pair_mean and pos_mean > random_mean:
        print("PASS: known triples score higher on average than both negative conditions. "
              "Inference wiring looks correct.")
    else:
        print("WARNING: known triples do NOT clearly outscore negatives on average. "
              "This suggests a possible bug in ID mapping, relation indexing, or the "
              "checkpoint/config used -- investigate before demoing.")

    print(f"\n=== {min(args.n_examples, len(examples))} example rows ===")
    for ex in examples:
        print(f"  {ex['drug_a']} + {ex['drug_b']}:")
        print(f"    true relation  '{ex['true_relation']}'  -> {ex['true_relation_score']:.4f}")
        print(f"    wrong relation '{ex['wrong_relation']}' -> {ex['wrong_relation_score']:.4f}")


if __name__ == "__main__":
    main()