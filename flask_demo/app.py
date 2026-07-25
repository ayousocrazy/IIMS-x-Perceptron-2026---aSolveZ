import sys
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.inference import DrugSafetyPredictor

DATA_PATH = str(PROJECT_ROOT / "graph/heterodata_with_features.pt")
CHECKPOINT_PATH = str(PROJECT_ROOT / "checkpoints/search_v3/final/best_model.pt")
DRUG_NODES_CSV = str(PROJECT_ROOT / "data/processed/nodes/drug_nodes.csv")
DRUG_NODES_SMILES_CSV = str(PROJECT_ROOT / "data/processed/nodes/drug_nodes_with_smiles.csv")
RELATION_LOOKUP_CSV = str(PROJECT_ROOT / "data/processed/polypharmacy_relation_lookup.csv")
DRUG_NAMES_JSON = str(PROJECT_ROOT / "data/drug_stitch_to_name.json")
HIDDEN_DIM, NUM_HEADS, NUM_LAYERS = 64, 2, 2

BASELINE_KNOWN_MEAN = 0.545
BASELINE_RANDOM_MEAN = 0.337
TEST_AUROC = 0.6327

app = Flask(__name__)

print("Loading model and graph (one-time)...")
names_path_obj = Path(DRUG_NAMES_JSON)
if names_path_obj.exists():
    print(f"Found drug names file: {DRUG_NAMES_JSON}")
else:
    print(f"WARNING: drug names file NOT found at {DRUG_NAMES_JSON} "
          f"-- dropdown will show raw CIDs instead of names. "
          f"Check the path/filename match exactly.")

predictor = DrugSafetyPredictor(
    data_path=DATA_PATH,
    checkpoint_path=CHECKPOINT_PATH,
    drug_nodes_csv=DRUG_NODES_CSV,
    relation_lookup_csv=RELATION_LOOKUP_CSV,
    drug_names_json=DRUG_NAMES_JSON if names_path_obj.exists() else None,
    hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
)
print(f"Model loaded. {len(predictor.drug_names)} drug names loaded. Ready.")

smiles_path = Path(DRUG_NODES_SMILES_CSV)
if smiles_path.exists():
    drug_options_df = pd.read_csv(smiles_path)
else:
    drug_options_df = pd.read_csv(DRUG_NODES_CSV)
    drug_options_df["smiles"] = None


def risk_tier(max_prob):
    """Frames the score relative to validate.py's measured baselines."""
    if max_prob >= BASELINE_KNOWN_MEAN:
        return {
            "tier": "elevated",
            "label": "Elevated risk signal",
            "detail": "Scores at or above the average for KNOWN documented interactions in the training data.",
        }
    elif max_prob >= BASELINE_RANDOM_MEAN:
        return {
            "tier": "moderate",
            "label": "Moderate / uncertain",
            "detail": "Between the random-pair baseline and the known-interaction baseline -- inconclusive.",
        }
    else:
        return {
            "tier": "low",
            "label": "Low risk signal",
            "detail": "At or below the average for RANDOM, non-interacting pairs.",
        }


@app.route("/")
def index():
    return render_template(
        "index.html",
        test_auroc=f"{TEST_AUROC:.4f}",
        baseline_known=f"{BASELINE_KNOWN_MEAN:.3f}",
        baseline_random=f"{BASELINE_RANDOM_MEAN:.3f}",
    )


@app.route("/api/drugs")
def api_drugs():
    drugs = []
    for _, row in drug_options_df.iterrows():
        smiles = row.get("smiles")
        preview = None
        if pd.notna(smiles):
            s = str(smiles)
            preview = s[:24] + ("..." if len(s) > 24 else "")
        cid = row["drug_id"]

        for alias in predictor.aliases_for_cid(cid):
            drugs.append({
                "cid": cid,
                "name": alias,
                "smiles_preview": preview,
            })
    return jsonify(drugs)


@app.route("/api/predict")
def api_predict():
    cid_a = request.args.get("drug_a")
    cid_b = request.args.get("drug_b")
    top_k = int(request.args.get("top_k", 8))

    if not cid_a or not cid_b:
        return jsonify({"error": "drug_a and drug_b are required"}), 400
    if cid_a == cid_b:
        return jsonify({"error": "Please select two different drugs."}), 400
    if cid_a not in predictor.cid_to_index or cid_b not in predictor.cid_to_index:
        return jsonify({"error": "Unknown drug CID."}), 400

    risk = predictor.predict_pair_risk(cid_a, cid_b, top_k=top_k)
    tier = risk_tier(risk["max_prob"])

    return jsonify({
        "drug_a": cid_a,
        "drug_b": cid_b,
        "drug_a_name": predictor.name_for_cid(cid_a),
        "drug_b_name": predictor.name_for_cid(cid_b),
        "max_prob": risk["max_prob"],
        "mean_prob": risk["mean_prob"],
        "top_relations": [
            {"name": name, "probability": prob} for name, prob in risk["top_relations"]
        ],
        "tier": tier,
        "baselines": {"known": BASELINE_KNOWN_MEAN, "random": BASELINE_RANDOM_MEAN},
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)