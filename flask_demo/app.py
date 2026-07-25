import re
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
from flask import Flask, jsonify, render_template, request

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.inference import DrugSafetyPredictor
from tablet_classifier import TabletClassifier

DATA_PATH = str(PROJECT_ROOT / "graph/heterodata_with_features.pt")
CHECKPOINT_PATH = str(PROJECT_ROOT / "checkpoints/search_v3/final/best_model.pt")
DRUG_NODES_CSV = str(PROJECT_ROOT / "data/processed/nodes/drug_nodes.csv")
DRUG_NODES_SMILES_CSV = str(PROJECT_ROOT / "data/processed/nodes/drug_nodes_with_smiles.csv")
RELATION_LOOKUP_CSV = str(PROJECT_ROOT / "data/processed/polypharmacy_relation_lookup.csv")
DRUG_NAMES_JSON = str(PROJECT_ROOT / "data/drug_stitch_to_name.json")
HIDDEN_DIM, NUM_HEADS, NUM_LAYERS = 64, 2, 2

# Tablet-image classifier (separate model from the interaction graph).
# Adjust this if model.pth / classes.json live somewhere else.
CLASSIFIER_DIR = PROJECT_ROOT / "classifier"
CLASSIFIER_MODEL_PATH = str(CLASSIFIER_DIR / "model.pth")
CLASSIFIER_CLASSES_PATH = str(CLASSIFIER_DIR / "classes.json")

BASELINE_KNOWN_MEAN = 0.545
BASELINE_RANDOM_MEAN = 0.337
TEST_AUROC = 0.6327

# predict_multi scores every pairwise combination (O(n^2)), so cap the
# number of drugs (or uploaded images) a single request can compare to
# keep response times sane.
MAX_DRUGS = 6

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

print("Loading tablet-image classifier (one-time)...")
tablet_classifier = TabletClassifier(CLASSIFIER_MODEL_PATH, CLASSIFIER_CLASSES_PATH)
if tablet_classifier.ready:
    print(f"Classifier ready. {len(tablet_classifier.classes)} tablet classes loaded.")
else:
    print("WARNING: tablet image classifier failed to load -- the photo-upload "
          "mode will return an error until model.pth/classes.json are in place at "
          f"'{CLASSIFIER_MODEL_PATH}' / '{CLASSIFIER_CLASSES_PATH}'.")


def _normalize_name(name: str) -> str:
    """Loosely normalize a drug/medicine name so classifier labels (e.g.
    'Paracetamol 500mg Tablet') can be matched against graph drug names
    (e.g. 'Acetaminophen') as reliably as possible without any manual
    mapping file. This is a best-effort string match, not a clinical lookup."""
    name = name.lower()
    name = re.sub(r"\b\d+\s*(mg|mcg|g|ml)\b", " ", name)   # strip dosages
    name = re.sub(r"\btablet(s)?\b|\bcapsule(s)?\b", " ", name)
    name = re.sub(r"[^a-z0-9]+", " ", name)
    return name.strip()


# Reverse index: normalized drug name/alias -> CID, built from the same
# drug_stitch_to_name.json the interaction dropdown already uses. This is
# how a classifier prediction like "Paracetamol" gets turned into a CID
# the DrugSafetyHGNN graph actually knows about.
_name_to_cid = {}
for _cid in predictor.cid_to_index:
    for _alias in predictor.aliases_for_cid(_cid):
        _key = _normalize_name(_alias)
        if _key and _key not in _name_to_cid:
            _name_to_cid[_key] = _cid


def resolve_cid_for_classified_name(name: str):
    """Best-effort match from a classifier class label to a graph drug CID.
    Returns None if no reasonable match is found -- callers should treat
    that as 'this image's medicine isn't in the interaction graph' rather
    than guessing."""
    if not name:
        return None
    key = _normalize_name(name)
    if key in _name_to_cid:
        return _name_to_cid[key]
    # fallback: substring match either direction (e.g. "paracetamol" vs
    # "paracetamol ip"), picking the longest matching key as the safest bet
    best_key, best_cid = None, None
    for known_key, cid in _name_to_cid.items():
        if key in known_key or known_key in key:
            if best_key is None or len(known_key) > len(best_key):
                best_key, best_cid = known_key, cid
    return best_cid


def risk_tier(mean_prob):
    """Frames the score relative to validate.py's measured baselines.

    Severity is driven by mean_prob (the average across all 1301 relations)
    rather than max_prob, since a single spiking relation is a weaker signal
    of overall risk than a pair whose scores are elevated across the board.
    """
    if mean_prob >= BASELINE_KNOWN_MEAN:
        return {
            "tier": "elevated",
            "label": "Elevated risk signal",
            "detail": "Scores at or above the average for KNOWN documented interactions in the training data.",
        }
    elif mean_prob >= BASELINE_RANDOM_MEAN:
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
        max_drugs=MAX_DRUGS,
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
    tier = risk_tier(risk["mean_prob"])

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


def _score_multi(cids: List[str], top_k: int = 5) -> Dict:
    """Runs predict_multi over a list of CIDs and shapes the response the
    frontend expects. Shared by /api/predict_multi (drug-name mode) and
    /api/predict_from_images (photo mode) so both modes render identically."""
    pairwise = predictor.predict_multi(cids, top_k=top_k)

    pairs_out = []
    tier_counts = {"elevated": 0, "moderate": 0, "low": 0}
    riskiest = None

    for p in pairwise:
        tier = risk_tier(p["mean_prob"])
        tier_counts[tier["tier"]] += 1
        entry = {
            "drug_a": p["drug_a"],
            "drug_b": p["drug_b"],
            "drug_a_name": predictor.name_for_cid(p["drug_a"]),
            "drug_b_name": predictor.name_for_cid(p["drug_b"]),
            "max_prob": p["max_prob"],
            "mean_prob": p["mean_prob"],
            "top_relations": [
                {"name": name, "probability": prob} for name, prob in p["top_relations"]
            ],
            "tier": tier,
        }
        pairs_out.append(entry)
        if riskiest is None or entry["mean_prob"] > riskiest["mean_prob"]:
            riskiest = entry

    # sort pairs so the riskiest ones (by mean_prob) surface first in the UI
    pairs_out.sort(key=lambda e: e["mean_prob"], reverse=True)
    overall_mean_of_mean = sum(p["mean_prob"] for p in pairs_out) / len(pairs_out)

    summary = {
        "num_drugs": len(cids),
        "num_pairs": len(pairs_out),
        "tier_counts": tier_counts,
        "overall_mean_of_mean": overall_mean_of_mean,
        "riskiest_pair": {
            "drug_a_name": riskiest["drug_a_name"],
            "drug_b_name": riskiest["drug_b_name"],
            "max_prob": riskiest["max_prob"],
            "mean_prob": riskiest["mean_prob"],
            "tier": riskiest["tier"],
        } if riskiest else None,
    }

    return {
        "drugs": cids,
        "drug_names": [predictor.name_for_cid(c) for c in cids],
        "pairs": pairs_out,
        "summary": summary,
        "baselines": {"known": BASELINE_KNOWN_MEAN, "random": BASELINE_RANDOM_MEAN},
    }


@app.route("/api/predict_multi")
def api_predict_multi():
    """Score every pairwise combination among 2+ drugs, plus a rolled-up summary.

    Query params:
        drugs  -- comma-separated CIDs, e.g. "CID4168,CID3345,CID5090"
        top_k  -- top relations to return per pair (default 5)
    """
    raw = request.args.get("drugs", "")
    top_k = int(request.args.get("top_k", 5))
    cids = [c.strip() for c in raw.split(",") if c.strip()]

    if len(cids) < 2:
        return jsonify({"error": "Select at least two drugs."}), 400
    if len(cids) > MAX_DRUGS:
        return jsonify({"error": f"Please select at most {MAX_DRUGS} drugs at once."}), 400
    if len(set(cids)) != len(cids):
        return jsonify({"error": "Please select each drug only once."}), 400

    unknown = [c for c in cids if c not in predictor.cid_to_index]
    if unknown:
        return jsonify({"error": f"Unknown drug CID(s): {', '.join(unknown)}"}), 400

    return jsonify(_score_multi(cids, top_k=top_k))


def _classify_upload(file_storage, top_k: int = 3) -> Dict:
    """Classifies a single uploaded image and resolves it to a graph CID."""
    image_bytes = file_storage.read()
    top = tablet_classifier.classify(image_bytes, top_k=top_k)
    best = top[0] if top else None
    cid = resolve_cid_for_classified_name(best["class"]) if best else None
    return {
        "filename": file_storage.filename,
        "top_predictions": top,
        "predicted_class": best["class"] if best else None,
        "confidence": best["probability"] if best else None,
        "resolved_cid": cid,
        "resolved_name": predictor.name_for_cid(cid) if cid else None,
    }


@app.route("/api/classify", methods=["POST"])
def api_classify():
    """Classify one or more tablet photos without running the interaction panel.

    Form data:
        images -- one or more image files (field name 'images', repeated)
    """
    if not tablet_classifier.ready:
        return jsonify({"error": "The image classifier is not available on the server."}), 500

    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "Please upload at least one photo."}), 400
    if len(files) > MAX_DRUGS:
        return jsonify({"error": f"Please upload at most {MAX_DRUGS} photos at once."}), 400

    predictions = []
    for f in files:
        try:
            predictions.append(_classify_upload(f))
        except Exception as e:
            predictions.append({"filename": f.filename, "error": f"Could not read this image: {e}"})

    return jsonify({"predictions": predictions})


@app.route("/api/predict_from_images", methods=["POST"])
def api_predict_from_images():
    """Primary photo-mode endpoint: classify each uploaded photo into a drug,
    then run the same pairwise interaction scoring used by /api/predict_multi.

    Form data:
        images -- 2+ image files (field name 'images', repeated)
        top_k  -- top relations to return per pair (default 5)
    """
    if not tablet_classifier.ready:
        return jsonify({"error": "The image classifier is not available on the server."}), 500

    files = request.files.getlist("images")
    top_k = int(request.form.get("top_k", 5))

    if len(files) < 2:
        return jsonify({"error": "Upload at least two photos to check interactions."}), 400
    if len(files) > MAX_DRUGS:
        return jsonify({"error": f"Please upload at most {MAX_DRUGS} photos at once."}), 400

    classifications = []
    for f in files:
        try:
            classifications.append(_classify_upload(f))
        except Exception as e:
            return jsonify({"error": f"Could not read image '{f.filename}': {e}"}), 400

    unresolved = [c["filename"] for c in classifications if not c.get("resolved_cid")]
    if unresolved:
        return jsonify({
            "error": (f"Couldn't match {len(unresolved)} photo(s) to a known drug in the "
                      f"interaction graph: {', '.join(unresolved)}. Try naming the drug "
                      f"manually in Drug-name mode instead."),
            "classifications": classifications,
        }), 422

    cids = [c["resolved_cid"] for c in classifications]
    if len(set(cids)) != len(cids):
        return jsonify({
            "error": "Two or more of the uploaded photos were classified as the same drug. "
                     "Please upload photos of different medicines.",
            "classifications": classifications,
        }), 400

    result = _score_multi(cids, top_k=top_k)
    result["classifications"] = classifications
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5000)