from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(".")
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

drug_combo = pd.read_csv(RAW_DIR / "drug-combo.csv")
relation_lookup = pd.read_csv(PROCESSED_DIR / "polypharmacy_relation_lookup.csv")

# code -> name, deduplicated (a given CUI code should map to one consistent name)
code_to_name = (
    drug_combo[["Polypharmacy Side Effect", "Side Effect Name"]]
    .drop_duplicates(subset="Polypharmacy Side Effect")
    .set_index("Polypharmacy Side Effect")["Side Effect Name"]
    .to_dict()
)

relation_lookup["side_effect_name"] = relation_lookup["side_effect"].map(code_to_name)

missing = relation_lookup["side_effect_name"].isna().sum()
if missing:
    print(f"WARNING: {missing} relation codes had no matching name in drug-combo.csv "
          f"-- these will fall back to their raw code in inference.py")

out_path = PROCESSED_DIR / "polypharmacy_relation_lookup.csv"
relation_lookup.to_csv(out_path, index=False)
print(f"Wrote {out_path} with columns: {list(relation_lookup.columns)}")
print(relation_lookup.head())