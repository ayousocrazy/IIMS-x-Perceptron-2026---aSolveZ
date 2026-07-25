import argparse
import json
from pathlib import Path

import pandas as pd


def build_reindex_map(selected_drug_nodes: pd.DataFrame) -> dict:
    """old node_index -> new, contiguous 0..N-1 node_index"""
    old_indices = selected_drug_nodes["node_index"].tolist()
    return {old: new for new, old in enumerate(old_indices)}


def filter_and_reindex_ddi_edges(ddi_edges: pd.DataFrame, reindex_map: dict) -> pd.DataFrame:
    kept = ddi_edges[
        ddi_edges["source"].isin(reindex_map) & ddi_edges["target"].isin(reindex_map)
    ].copy()
    kept["source"] = kept["source"].map(reindex_map)
    kept["target"] = kept["target"].map(reindex_map)
    return kept.reset_index(drop=True)


def filter_and_reindex_drug_edges(edges: pd.DataFrame, reindex_map: dict, drug_col: str) -> pd.DataFrame:
    kept = edges[edges[drug_col].isin(reindex_map)].copy()
    kept[drug_col] = kept[drug_col].map(reindex_map)
    return kept.reset_index(drop=True)


def prune_orphaned_nodes(node_table: pd.DataFrame, edge_tables: list, id_col: str = "node_index",
                          edge_cols: list = None) -> tuple:
    touched = set()
    for edges, cols in zip(edge_tables, edge_cols):
        for col in cols:
            touched.update(edges[col].unique().tolist())

    n_before = len(node_table)
    kept = node_table[node_table[id_col].isin(touched)].copy()
    n_dropped = n_before - len(kept)
    if n_dropped:
        print(f"  dropped {n_dropped} orphaned node(s) with zero remaining edges")

    reindex_map = {old: new for new, old in enumerate(kept[id_col].tolist())}
    kept[id_col] = kept[id_col].map(reindex_map)
    return kept.reset_index(drop=True), reindex_map


def run(node_dir: str, edge_dir: str, selected_drugs_path: str, output_dir: str):
    node_dir = Path(node_dir)
    edge_dir = Path(edge_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_drug_nodes = pd.read_csv(selected_drugs_path)
    drug_reindex = build_reindex_map(selected_drug_nodes)

    print(f"Keeping {len(drug_reindex)} drugs")

    # --- drug-drug polypharmacy edges ---
    ddi_edges = pd.read_csv(edge_dir / "drug_drug_polypharmacy_edges.csv")
    ddi_filtered = filter_and_reindex_ddi_edges(ddi_edges, drug_reindex)
    print(f"drug-drug polypharmacy edges: {len(ddi_edges)} -> {len(ddi_filtered)}")

    n_relations_before = ddi_edges["relation_id"].nunique()
    n_relations_after = ddi_filtered["relation_id"].nunique()
    if n_relations_after < n_relations_before:
        print(f"  note: {n_relations_before - n_relations_after} relation type(s) "
              f"have zero edges left in the reduced graph (still valid, just unused)")

    # --- drug-gene edges ---
    dg_edges = pd.read_csv(edge_dir / "drug_gene_edges.csv")
    dg_filtered = filter_and_reindex_drug_edges(dg_edges, drug_reindex, drug_col="drug")
    print(f"drug-gene edges: {len(dg_edges)} -> {len(dg_filtered)}")

    # --- drug-side_effect edges ---
    dse_edges = pd.read_csv(edge_dir / "drug_sideeffect_edges.csv")
    dse_filtered = filter_and_reindex_drug_edges(dse_edges, drug_reindex, drug_col="drug")
    print(f"drug-side_effect edges: {len(dse_edges)} -> {len(dse_filtered)}")

    # --- gene-gene edges: prune genes with zero remaining drug-gene edges ---
    gg_edges = pd.read_csv(edge_dir / "gene_gene_edges.csv")
    gene_nodes = pd.read_csv(node_dir / "gene_nodes.csv")

    gene_nodes_kept, gene_reindex = prune_orphaned_nodes(
        gene_nodes, [dg_filtered], id_col="node_index", edge_cols=[["gene"]]
    )
    gg_filtered = gg_edges[
        gg_edges["source"].isin(gene_reindex) & gg_edges["target"].isin(gene_reindex)
    ].copy()
    gg_filtered["source"] = gg_filtered["source"].map(gene_reindex)
    gg_filtered["target"] = gg_filtered["target"].map(gene_reindex)
    dg_filtered["gene"] = dg_filtered["gene"].map(gene_reindex)
    print(f"genes: {len(gene_nodes)} -> {len(gene_nodes_kept)}")
    print(f"gene-gene edges: {len(gg_edges)} -> {len(gg_filtered)}")

    # --- side_effect nodes: prune SEs with zero remaining drug-side_effect edges ---
    se_nodes = pd.read_csv(node_dir / "sideeffect_nodes.csv")
    se_nodes_kept, se_reindex = prune_orphaned_nodes(
        se_nodes, [dse_filtered], id_col="node_index", edge_cols=[["side_effect"]]
    )
    dse_filtered["side_effect"] = dse_filtered["side_effect"].map(se_reindex)
    print(f"side_effect nodes: {len(se_nodes)} -> {len(se_nodes_kept)}")

    # --- save everything, re-indexed and consistent ---
    selected_drug_nodes_out = selected_drug_nodes.copy()
    selected_drug_nodes_out["node_index"] = selected_drug_nodes_out["node_index"].map(drug_reindex)
    selected_drug_nodes_out = selected_drug_nodes_out.sort_values("node_index").reset_index(drop=True)

    node_out = output_dir / "nodes"
    edge_out = output_dir / "edges"
    node_out.mkdir(parents=True, exist_ok=True)
    edge_out.mkdir(parents=True, exist_ok=True)

    selected_drug_nodes_out.to_csv(node_out / "drug_nodes.csv", index=False)
    gene_nodes_kept.to_csv(node_out / "gene_nodes.csv", index=False)
    se_nodes_kept.to_csv(node_out / "sideeffect_nodes.csv", index=False)

    ddi_filtered.to_csv(edge_out / "drug_drug_polypharmacy_edges.csv", index=False)
    dg_filtered.to_csv(edge_out / "drug_gene_edges.csv", index=False)
    dse_filtered.to_csv(edge_out / "drug_sideeffect_edges.csv", index=False)
    gg_filtered.to_csv(edge_out / "gene_gene_edges.csv", index=False)

    (output_dir / "drug_reindex_map.json").write_text(
        json.dumps({str(k): v for k, v in drug_reindex.items()}, indent=2)
    )

    print(f"\\nSaved reduced graph to {output_dir.resolve()}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Filter processed graph down to a drug subset")
    parser.add_argument("--node-dir", required=True)
    parser.add_argument("--edge-dir", required=True)
    parser.add_argument("--selected-drugs", required=True, help="Path to selected_drug_nodes.csv")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run(args.node_dir, args.edge_dir, args.selected_drugs, args.output_dir)