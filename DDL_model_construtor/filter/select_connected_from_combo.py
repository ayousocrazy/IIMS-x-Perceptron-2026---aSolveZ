import argparse
from pathlib import Path
from collections import defaultdict

import pandas as pd


def build_adjacency_from_combo(combo_df: pd.DataFrame, col1: str, col2: str) -> dict:
    """drug_id (string) -> set of neighboring drug_id (string), deduplicated pairs."""
    adj = defaultdict(set)
    pairs = combo_df[[col1, col2]].drop_duplicates().itertuples(index=False, name=None)
    for a, b in pairs:
        if a == b:
            continue
        adj[a].add(b)
        adj[b].add(a)
    return adj


def select_connected_core(adj: dict, top_n: int) -> list:
    degree = {node: len(neighbors) for node, neighbors in adj.items()}
    unvisited_by_degree = sorted(degree, key=degree.get, reverse=True)

    selected = []
    selected_set = set()
    frontier = set()

    def start_new_component():
        for node in unvisited_by_degree:
            if node not in selected_set:
                return node
        return None

    while len(selected) < top_n:
        if not frontier:
            start = start_new_component()
            if start is None:
                print(f"Warning: ran out of connected drugs; only found {len(selected)} "
                      f"(requested {top_n}).")
                break
            selected.append(start)
            selected_set.add(start)
            frontier = set(adj.get(start, ())) - selected_set
            continue

        best_node, best_score = None, -1
        for node in frontier:
            score = len(adj[node] & selected_set)
            if score > best_score:
                best_node, best_score = node, score

        selected.append(best_node)
        selected_set.add(best_node)
        frontier.discard(best_node)
        frontier |= (adj.get(best_node, set()) - selected_set)

    return selected


def run(combo_csv: str, node_dir: str, output_dir: str, top_n: int = 57):
    combo_csv = Path(combo_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    combo_df = pd.read_csv(combo_csv, dtype=str)
    col1, col2 = combo_df.columns[0], combo_df.columns[1]
    print(f"Using columns as the drug pair: '{col1}', '{col2}'")

    adj = build_adjacency_from_combo(combo_df, col1, col2)
    print(f"Combo graph has {len(adj)} unique drugs and "
          f"{sum(len(v) for v in adj.values()) // 2} unique undirected pairs")

    selected_ids = select_connected_core(adj, top_n)
    degree = {node: len(neighbors) for node, neighbors in adj.items()}

    result = pd.DataFrame({
        "drug_id": selected_ids,
        "degree": [degree.get(d, 0) for d in selected_ids],
    })

    node_path = Path(node_dir) / "drug_nodes.csv" if node_dir else None
    if node_path and node_path.exists():
        drug_nodes = pd.read_csv(node_path, dtype=str)
        id_col = "drug_id" if "drug_id" in drug_nodes.columns else drug_nodes.columns[0]
        lookup = drug_nodes.set_index(id_col)["node_index"] if "node_index" in drug_nodes.columns else None
        if lookup is not None:
            result["node_index"] = result["drug_id"].map(lookup)
            result = result[["node_index", "drug_id", "degree"]]

    result = result.sort_values("degree", ascending=False).reset_index(drop=True)

    print(f"\nSelected {len(result)} / {len(adj)} drugs "
          f"(degree range: {result['degree'].min()}-{result['degree'].max()})")

    sel_set = set(selected_ids)
    surviving = combo_df[combo_df[col1].isin(sel_set) & combo_df[col2].isin(sel_set)]
    print(f"Combo rows (in {combo_csv.name}) where BOTH drugs are in this selection: {len(surviving)}")

    out_path = output_dir / "selected_drug_nodes.csv"
    result.to_csv(out_path, index=False)
    print(f"Saved to {out_path.resolve()}")

    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Select a connected core of N drugs directly from "
                    "drug-combo-normalized.csv so combo rows are guaranteed to survive"
    )
    parser.add_argument("--combo-csv", required=True, help="path to drug-combo-normalized.csv")
    parser.add_argument("--node-dir", default=None,
                         help="optional: directory containing drug_nodes.csv, to attach node_index")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-n", type=int, default=57)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run(args.combo_csv, args.node_dir, args.output_dir, args.top_n)
