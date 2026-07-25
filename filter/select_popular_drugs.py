import argparse
from pathlib import Path

import pandas as pd


def compute_drug_degree(ddi_edges: pd.DataFrame) -> pd.Series:
    return ddi_edges["source"].value_counts()


def select_top_drugs(drug_nodes: pd.DataFrame, ddi_edges: pd.DataFrame, top_n: int) -> pd.DataFrame:
    degree = compute_drug_degree(ddi_edges)

    drug_nodes = drug_nodes.copy()
    drug_nodes["degree"] = drug_nodes["node_index"].map(degree).fillna(0).astype(int)

    n_zero_degree = (drug_nodes["degree"] == 0).sum()
    if n_zero_degree:
        print(f"Note: {n_zero_degree} drug(s) have zero polypharmacy edges "
              f"(never selected unless top_n exceeds the number with edges)")

    selected = drug_nodes.sort_values("degree", ascending=False).head(top_n)
    return selected.sort_values("node_index").reset_index(drop=True)


def run(node_dir: str, edge_dir: str, output_dir: str, top_n: int = 100):
    node_dir = Path(node_dir)
    edge_dir = Path(edge_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    drug_nodes = pd.read_csv(node_dir / "drug_nodes.csv")
    ddi_edges = pd.read_csv(edge_dir / "drug_drug_polypharmacy_edges.csv")

    selected = select_top_drugs(drug_nodes, ddi_edges, top_n)

    print(f"Selected {len(selected)} / {len(drug_nodes)} drugs "
          f"(degree range: {selected['degree'].min()}-{selected['degree'].max()})")

    selected.to_csv(output_dir / "selected_drug_nodes.csv", index=False)
    print(f"Saved to {(output_dir / 'selected_drug_nodes.csv').resolve()}")

    return selected


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Select top-N drugs by polypharmacy edge degree")
    parser.add_argument("--node-dir", required=True, help="Directory containing drug_nodes.csv")
    parser.add_argument("--edge-dir", required=True, help="Directory containing drug_drug_polypharmacy_edges.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-n", type=int, default=200)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run(args.node_dir, args.edge_dir, args.output_dir, args.top_n)