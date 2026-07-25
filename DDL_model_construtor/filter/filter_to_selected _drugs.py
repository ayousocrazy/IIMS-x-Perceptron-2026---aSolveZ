import os
import json
import pandas as pd

PROJECT_ROOT = "."                       # run script from your project root
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")   # edges/, nodes/, and the normalized+lookup csvs + smiles_cache.json all live here
UPDATED_DIR = os.path.join(PROJECT_ROOT, "updated_data")     # fresh filtered output goes here
SELECTED_FILE = os.path.join(PROJECT_ROOT, "update_data", "selected_drug_nodes.csv")  # <- note: reading from "update_data" (no 's'), per your project tree
SELECTED_DRUG_ID_COL = "drug_id"          # confirmed from your file

# Force a column name instead of auto-detecting, e.g. "drug_col": "drugbank_id"
OVERRIDES = {
    "drug_nodes.csv": {"drug_col": None},
    "drug_nodes_with_smiles.csv": {"drug_col": None},
    "drug_drug_polypharmacy_edges.csv": {"drug1_col": None, "drug2_col": None},
    "drug_gene_edges.csv": {"drug_col": None, "gene_col": None},
    "drug_sideeffect_edges.csv": {"drug_col": None, "sideeffect_col": None},
    "gene_gene_edges.csv": {"gene1_col": None, "gene2_col": None},
    "gene_nodes.csv": {"gene_col": None},
    "sideeffect_nodes.csv": {"sideeffect_col": None},
}


def log(msg):
    print(msg, flush=True)


def detect_id_column(df, id_set, exclude=(), min_overlap=0.3):
    """Return the column whose values overlap the given id_set the most."""
    best_col, best_score = None, 0.0
    for col in df.columns:
        if col in exclude:
            continue
        vals = df[col].astype(str).str.strip()
        if len(vals) == 0:
            continue
        overlap = vals.isin(id_set).mean()
        if overlap > best_score:
            best_col, best_score = col, overlap
    if best_score >= min_overlap:
        return best_col, best_score
    return None, best_score


def detect_pair_columns(df, id_set, min_overlap=0.3):
    """Find the two columns most likely to hold a pair of ids from id_set."""
    scores = []
    for col in df.columns:
        vals = df[col].astype(str).str.strip()
        overlap = vals.isin(id_set).mean()
        if overlap >= min_overlap:
            scores.append((col, overlap))
    scores.sort(key=lambda x: -x[1])
    if len(scores) >= 2:
        return scores[0][0], scores[1][0]
    return None, None


def read_csv_safe(path):
    if not os.path.exists(path):
        log(f"  [skip] not found: {path}")
        return None
    df = pd.read_csv(path, dtype=str, low_memory=False)
    return df


def write_csv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    log(f"  -> wrote {path}  ({len(df)} rows)")


def main():
    # ---- 1. load selected drugs -------------------------------------------------
    if not os.path.exists(SELECTED_FILE):
        raise FileNotFoundError(f"Could not find {SELECTED_FILE}")
    selected_df = pd.read_csv(SELECTED_FILE, dtype=str)
    selected_ids = set(selected_df[SELECTED_DRUG_ID_COL].astype(str).str.strip())
    log(f"Loaded {len(selected_ids)} selected drug ids from {SELECTED_FILE}\n")

    connected_genes = set()
    connected_sideeffects = set()

    for fname in ["drug_nodes.csv", "drug_nodes_with_smiles.csv"]:
        src = os.path.join(DATA_DIR, "nodes", fname)
        log(f"Processing {fname} ...")
        df = read_csv_safe(src)
        if df is None:
            continue
        drug_col = OVERRIDES[fname]["drug_col"]
        if drug_col is None:
            drug_col, score = detect_id_column(df, selected_ids)
            log(f"  detected drug id column: '{drug_col}' (overlap={score:.2f})")
        if drug_col is None:
            log(f"  [WARNING] couldn't detect drug id column in {fname}; edit OVERRIDES and re-run. Skipping.")
            continue
        filtered = df[df[drug_col].astype(str).str.strip().isin(selected_ids)]
        write_csv(filtered, os.path.join(UPDATED_DIR, "processed", "nodes", fname))

    # ---- 3. drug-drug polypharmacy edges (both sides must be selected) ----------
    fname = "drug_drug_polypharmacy_edges.csv"
    src = os.path.join(DATA_DIR, "edges", fname)
    log(f"\nProcessing {fname} ...")
    df = read_csv_safe(src)
    if df is not None:
        c1 = OVERRIDES[fname]["drug1_col"]
        c2 = OVERRIDES[fname]["drug2_col"]
        if c1 is None or c2 is None:
            c1, c2 = detect_pair_columns(df, selected_ids)
            log(f"  detected drug columns: '{c1}', '{c2}'")
        if c1 and c2:
            mask = (df[c1].astype(str).str.strip().isin(selected_ids) &
                    df[c2].astype(str).str.strip().isin(selected_ids))
            filtered = df[mask]
            write_csv(filtered, os.path.join(UPDATED_DIR, "processed", "edges", fname))
        else:
            log(f"  [WARNING] couldn't detect both drug columns in {fname}; edit OVERRIDES and re-run. Skipping.")

    # ---- 4. drug-gene edges (drug side must be selected) -------------------------
    fname = "drug_gene_edges.csv"
    src = os.path.join(DATA_DIR, "edges", fname)
    log(f"\nProcessing {fname} ...")
    df = read_csv_safe(src)
    if df is not None:
        drug_col = OVERRIDES[fname]["drug_col"]
        gene_col = OVERRIDES[fname]["gene_col"]
        if drug_col is None:
            drug_col, score = detect_id_column(df, selected_ids)
            log(f"  detected drug id column: '{drug_col}' (overlap={score:.2f})")
        if drug_col:
            filtered = df[df[drug_col].astype(str).str.strip().isin(selected_ids)]
            write_csv(filtered, os.path.join(UPDATED_DIR, "processed", "edges", fname))
            if gene_col is None:
                # gene column = the other id-like column (not the drug column)
                other_cols = [c for c in df.columns if c != drug_col]
                gene_col = other_cols[0] if other_cols else None
            if gene_col:
                connected_genes.update(filtered[gene_col].astype(str).str.strip().unique())
                log(f"  gene column assumed: '{gene_col}' -> {len(connected_genes)} unique genes connected")
        else:
            log(f"  [WARNING] couldn't detect drug id column in {fname}; edit OVERRIDES and re-run. Skipping.")

    # ---- 5. drug-sideeffect edges (drug side must be selected) -------------------
    fname = "drug_sideeffect_edges.csv"
    src = os.path.join(DATA_DIR, "edges", fname)
    log(f"\nProcessing {fname} ...")
    df = read_csv_safe(src)
    if df is not None:
        drug_col = OVERRIDES[fname]["drug_col"]
        se_col = OVERRIDES[fname]["sideeffect_col"]
        if drug_col is None:
            drug_col, score = detect_id_column(df, selected_ids)
            log(f"  detected drug id column: '{drug_col}' (overlap={score:.2f})")
        if drug_col:
            filtered = df[df[drug_col].astype(str).str.strip().isin(selected_ids)]
            write_csv(filtered, os.path.join(UPDATED_DIR, "processed", "edges", fname))
            if se_col is None:
                other_cols = [c for c in df.columns if c != drug_col]
                se_col = other_cols[0] if other_cols else None
            if se_col:
                connected_sideeffects.update(filtered[se_col].astype(str).str.strip().unique())
                log(f"  sideeffect column assumed: '{se_col}' -> {len(connected_sideeffects)} unique side effects connected")
        else:
            log(f"  [WARNING] couldn't detect drug id column in {fname}; edit OVERRIDES and re-run. Skipping.")

    # ---- 6. gene nodes (only genes touched by filtered drug-gene edges) ---------
    fname = "gene_nodes.csv"
    src = os.path.join(DATA_DIR, "nodes", fname)
    log(f"\nProcessing {fname} ...")
    df = read_csv_safe(src)
    if df is not None and connected_genes:
        gene_col = OVERRIDES[fname]["gene_col"]
        if gene_col is None:
            gene_col, score = detect_id_column(df, connected_genes)
            log(f"  detected gene id column: '{gene_col}' (overlap={score:.2f})")
        if gene_col:
            filtered = df[df[gene_col].astype(str).str.strip().isin(connected_genes)]
            write_csv(filtered, os.path.join(UPDATED_DIR, "processed", "nodes", fname))
        else:
            log(f"  [WARNING] couldn't detect gene id column in {fname}; edit OVERRIDES and re-run. Skipping.")
    elif df is not None:
        log("  [skip] no connected genes found upstream (check drug_gene_edges step above)")

    # ---- 7. sideeffect nodes ------------------------------------------------------
    fname = "sideeffect_nodes.csv"
    src = os.path.join(DATA_DIR, "nodes", fname)
    log(f"\nProcessing {fname} ...")
    df = read_csv_safe(src)
    if df is not None and connected_sideeffects:
        se_col = OVERRIDES[fname]["sideeffect_col"]
        if se_col is None:
            se_col, score = detect_id_column(df, connected_sideeffects)
            log(f"  detected sideeffect id column: '{se_col}' (overlap={score:.2f})")
        if se_col:
            filtered = df[df[se_col].astype(str).str.strip().isin(connected_sideeffects)]
            write_csv(filtered, os.path.join(UPDATED_DIR, "processed", "nodes", fname))
        else:
            log(f"  [WARNING] couldn't detect sideeffect id column in {fname}; edit OVERRIDES and re-run. Skipping.")
    elif df is not None:
        log("  [skip] no connected side effects found upstream (check drug_sideeffect_edges step above)")

    # ---- 8. gene-gene edges (both genes must have survived) -----------------------
    fname = "gene_gene_edges.csv"
    src = os.path.join(DATA_DIR, "edges", fname)
    log(f"\nProcessing {fname} ...")
    df = read_csv_safe(src)
    if df is not None and connected_genes:
        c1 = OVERRIDES[fname]["gene1_col"]
        c2 = OVERRIDES[fname]["gene2_col"]
        if c1 is None or c2 is None:
            c1, c2 = detect_pair_columns(df, connected_genes)
            log(f"  detected gene columns: '{c1}', '{c2}'")
        if c1 and c2:
            mask = (df[c1].astype(str).str.strip().isin(connected_genes) &
                    df[c2].astype(str).str.strip().isin(connected_genes))
            filtered = df[mask]
            write_csv(filtered, os.path.join(UPDATED_DIR, "processed", "edges", fname))
        else:
            log(f"  [WARNING] couldn't detect both gene columns in {fname}; edit OVERRIDES and re-run. Skipping.")
    elif df is not None:
        log("  [skip] no connected genes found upstream")

    # ---- 9a. drug-combo-normalized.csv (a DRUG-DRUG pair file -> BOTH must be selected) ----
    fname = "drug-combo-normalized.csv"
    src = os.path.join(DATA_DIR, fname)
    log(f"\nProcessing {fname} ...")
    df = read_csv_safe(src)
    if df is not None:
        c1, c2 = detect_pair_columns(df, selected_ids)
        log(f"  detected drug pair columns: '{c1}', '{c2}'")
        if c1 and c2:
            mask = (df[c1].astype(str).str.strip().isin(selected_ids) &
                    df[c2].astype(str).str.strip().isin(selected_ids))
            filtered = df[mask]
            write_csv(filtered, os.path.join(UPDATED_DIR, fname))
        else:
            log(f"  [WARNING] couldn't detect both drug columns in {fname}; edit manually. Skipping.")

    # ---- 9b. single-drug-referencing normalized / lookup csvs (row kept if ANY column matches) --
    for fname in ["drug-gene-normalized.csv", "drug-mono-normalized.csv",
                  "polypharmacy_relation_lookup.csv"]:
        src = os.path.join(DATA_DIR, fname)
        log(f"\nProcessing {fname} ...")
        df = read_csv_safe(src)
        if df is None:
            continue
        # keep any row that references a selected drug in ANY column
        mask = pd.Series(False, index=df.index)
        hit_cols = []
        for col in df.columns:
            vals = df[col].astype(str).str.strip()
            overlap = vals.isin(selected_ids).mean()
            if overlap > 0.05:  # column plausibly holds drug ids
                mask = mask | vals.isin(selected_ids)
                hit_cols.append(col)
        if hit_cols:
            log(f"  drug-id-like columns used for filtering: {hit_cols}")
            filtered = df[mask]
            write_csv(filtered, os.path.join(UPDATED_DIR, fname))
        else:
            log(f"  [WARNING] no column in {fname} matched selected drug ids; file skipped. Check manually if needed.")

    # ---- 10. smiles_cache.json ------------------------------------------------
    fname = "smiles_cache.json"
    src = os.path.join(DATA_DIR, fname)
    log(f"\nProcessing {fname} ...")
    if os.path.exists(src):
        with open(src) as f:
            cache = json.load(f)
        filtered_cache = {k: v for k, v in cache.items() if str(k).strip() in selected_ids}
        os.makedirs(UPDATED_DIR, exist_ok=True)
        out_path = os.path.join(UPDATED_DIR, fname)
        with open(out_path, "w") as f:
            json.dump(filtered_cache, f, indent=2)
        log(f"  -> wrote {out_path}  ({len(filtered_cache)} / {len(cache)} keys kept)")
    else:
        log(f"  [skip] not found: {src}")

    log("\nDone. Review any [WARNING] lines above — those files need a column")
    log("name hardcoded in OVERRIDES at the top of this script, then re-run.")


if __name__ == "__main__":
    main()