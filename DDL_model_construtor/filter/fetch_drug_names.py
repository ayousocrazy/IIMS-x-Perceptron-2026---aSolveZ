import argparse
import json
import time
from pathlib import Path

import pandas as pd
import pubchempy as pcp


def load_cache(cache_path: Path) -> dict:
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    return {}


def save_cache(cache_path: Path, cache: dict) -> None:
    cache_path.write_text(json.dumps(cache, indent=2))


def fetch_name(cid: str, cache: dict, max_retries: int = 3) -> str | None:
    if cid in cache:
        return cache[cid]

    numeric_cid = int(cid.replace("CID", ""))
    resolved = None

    for attempt in range(max_retries):
        try:
            compound = pcp.Compound.from_cid(numeric_cid)
            synonyms = compound.synonyms
            resolved = synonyms[0] if synonyms else compound.iupac_name
            break
        except Exception as e:
            print(f"{cid}: attempt {attempt + 1}/{max_retries} failed ({e})")
            time.sleep(0.5 * (attempt + 1))

    cache[cid] = resolved
    save_cache_ref[0](cache)  # persist immediately, same as Step 2's pattern
    time.sleep(0.2)
    return resolved


save_cache_ref = [lambda cache: None]  # patched in run() to close over cache_path


def run(drug_nodes_path: str, output_dir: str, cache_path: str = None):
    drug_nodes_path = Path(drug_nodes_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_path = Path(cache_path) if cache_path else output_dir / "drug_name_cache.json"
    cache = load_cache(cache_path)
    save_cache_ref[0] = lambda c: save_cache(cache_path, c)

    drug_nodes = pd.read_csv(drug_nodes_path)

    names = []
    for cid in drug_nodes["drug_id"]:
        name = fetch_name(cid, cache)
        names.append(name.lower() if name else None)

    drug_nodes = drug_nodes.copy()
    drug_nodes["name"] = names

    n_missing = drug_nodes["name"].isna().sum()
    if n_missing:
        print(f"{n_missing} / {len(drug_nodes)} drug(s) have no resolvable name")

    out_csv = drug_nodes[["drug_id", "name"]]
    out_csv.to_csv(output_dir / "drug_names.csv", index=False)

    out_json = {
        cid: (name if isinstance(name, str) else None)
        for cid, name in zip(drug_nodes["drug_id"], drug_nodes["name"])
    }
    (output_dir / "drug_names.json").write_text(json.dumps(out_json, indent=2))

    print(f"Saved {output_dir / 'drug_names.csv'}")
    print(f"Saved {output_dir / 'drug_names.json'}")

    return drug_nodes


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Fetch drug names from PubChem, lowercased")
    parser.add_argument("--drug-nodes", required=True, help="Path to drug_nodes.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-path", default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run(args.drug_nodes, args.output_dir, args.cache_path)