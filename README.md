# AI-Powered Drug Safety System

Multimodal system for predicting drug-drug interactions and identifying medications, built for IIMS x Perceptron 2026.

Two tracks feed one system:

1. **Graph track (`DDL_model_construtor/`, `models/`, `graph/`)** — a Heterogeneous Graph Neural Network (HGNN), following the Decagon architecture, predicts *which specific side effect* a pair of drugs is likely to cause — not just a binary "do they interact?". The graph has three node types (drug, gene, side_effect) and treats each polypharmacy side effect as its own relation type on drug-drug edges, so the model learns relation-specific interaction patterns instead of one generic "interacts" signal.
2. **Vision track (`Image_model_constructor/`, `classifier/`)** — an image classifier identifies a medication from a photo of the tablet/pill, so a user can get from "photo of two pills" to "predicted interaction" without needing to already know the drug names.

`flask_demo/` wires both tracks together behind a web UI, with an LLM explainer layer for human-readable output.

> **Time-boxed for submission.** This README documents the fast path to a working, presentable state — not the full research pipeline (hyperparameter search, large-scale training) that there usually isn't time for before a deadline. Where a step has a "fast path" and a "full path," take the fast path unless you have hours to spare.

---

## 1. Setup

```bash
python -m venv env
env\Scripts\activate          # Windows
# source env/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

### ⚠️ Known import path issue — fix this first

Notebooks in `DDL_model_construtor/notebook/` and `DDL_model_construtor/testing/` add only `DDL_model_construtor/` to `sys.path`:
```python
sys.path.append(str(Path().resolve().parent))
```
That reaches `DDL_model_construtor/training` and `DDL_model_construtor/filter` fine, but **`models/` lives at the repo root**, one level further up — so `from models.hgnn import ...` will raise `ModuleNotFoundError` as-is. Pick one fix before running anything below:

- **Quick fix (no file moves):** change that line in every notebook to add the repo root instead:
  ```python
  sys.path.append(str(Path().resolve().parent.parent))
  ```
- **Structural fix:** move `models/` inside `DDL_model_construtor/` to match where `training/` and `filter/` already live, and update imports to `from DDL_model_construtor.models.hgnn import ...` accordingly.

The rest of this README assumes the quick fix (path to repo root).

---

## 2. Pipeline — step by step

All commands below assume you're running from the **repo root** unless a step says otherwise. Paths for scripts run with `python -m` use dots; paths for notebooks are relative to wherever you launch Jupyter.

### Step 1 — Build the full graph from raw data

Notebook: `DDL_model_construtor/notebook/01_graph_construction.ipynb`

Reads `data/raw/*.csv` (drug-combo, drug-gene, drug-mono, gene-gene, effectcategories), normalizes drug IDs (collapsing stereo-isomer CIDs to their flat form), builds node/edge tables, and assembles the full `HeteroData` graph.

Produces:
- `data/processed/nodes/*.csv`, `data/processed/edges/*.csv`
- `graph/heterodata.pt`

Run once — this is the full ~645-drug dataset and doesn't need to be rebuilt unless the raw CSVs change.

### Step 2 — Node features

Notebook: `DDL_model_construtor/notebook/02_node_feature_construction.ipynb`

- **Drugs:** fetches SMILES from PubChem (cached to `data/processed/smiles_cache.json` — re-running this notebook reuses the cache instead of re-fetching), computes 2048-bit Morgan (ECFP4) fingerprints via RDKit.
- **Genes / side effects:** left as learned embeddings, created inside the model itself (`models/encoder.py`), not precomputed here.

Produces: `graph/heterodata_with_features.pt`, `graph/drug_features.npy`.

### Step 3 — Reduce to a laptop-trainable drug subset

The full graph (~645 drugs, ~9.3M drug-drug edges) is too large to train on a laptop in the time available. `DDL_model_construtor/filter/` reduces it to a smaller, well-connected subset.

```bash
cd DDL_model_construtor

python -m filter.select_popular_drugs --node-dir ../data/processed/nodes --edge-dir ../data/processed/edges --output-dir ../update_data --top-n 200

python -m filter.filter_graph --node-dir ../data/processed/nodes --edge-dir ../data/processed/edges --selected-drugs ../update_data/selected_drug_nodes.csv --output-dir ../update_data

python -m filter.fetch_drug_names --drug-nodes ../update_data/nodes/drug_nodes.csv --output-dir ../update_data
```

> Your `filter/` folder also has `select_connected_from_combo.py`, `filter_to_selected _drugs.py` (note the space in that filename — check it's intentional), and `fix_relation_names.py`, which look like newer/alternate selection strategies beyond plain top-N-by-degree. If you've since standardized on one of those instead of `select_popular_drugs.py`, use that command in its place — same general shape (input node/edge dir → output dir).

This produces `update_data/nodes/`, `update_data/edges/`, `update_data/selected_drug_nodes.csv`, `update_data/drug_names.csv` / `.json`, `update_data/drug_reindex_map.json`.

**You still need to rebuild the `HeteroData` object and node features for this reduced set** (assemble a `HeteroData` from `update_data/nodes` + `update_data/edges`, matching Step 1's graph-assembly logic, then re-run Step 2's feature step pointed at `update_data/` instead of `data/processed/` — the `smiles_cache.json` from Step 2 will already cover these 200 drugs, so no new PubChem calls are needed). Save the result as `update_data/graph/heterodata_with_features.pt`.

### Step 4 — Model

`models/encoder.py` (HGT encoder), `models/decoder.py` (DistMult relation-aware decoder), `models/hgnn.py` (glue: graph → encoder → drug embeddings → decoder → scores). No changes needed here to run training — these are already wired together.

### Step 5 — Train (fast path — do this one)

```bash
cd DDL_model_construtor
python -c "
from training.train import run

model, result = run(
    data_path='../update_data/graph/heterodata_with_features.pt',
    output_dir='../checkpoints/submission_run',
    epochs=30,
    batch_size=100_000,
    lr=1e-3,
    hidden_dim=64,
    num_heads=2,
    num_layers=1,
    patience=6,
    save_artifacts=True,
    verbose=True,
)
print(f\"Test AUROC: {result['test_auroc']:.4f} | Test AUPRC: {result['test_auprc']:.4f}\")
"
```

No hyperparameter search — one fixed, deliberately cheap config (small `hidden_dim`, single layer, large `batch_size` to minimize the number of full-encoder-recompute steps per epoch). Produces `checkpoints/submission_run/{best_model.pt, final_model.pt, training_history.json}`.

### Step 5b — Full search (only if you have hours to spare)

```bash
cd DDL_model_construtor
python -m training.hp_search --data-path ../update_data/graph/heterodata_with_features.pt --output-dir ../checkpoints/full_run --n-trials 20 --search-epochs 15 --final-epochs 60
```
Or the equivalent notebook: `DDL_model_construtor/testing/full_training.ipynb` — **update its `DATA_PATH`** to `../../update_data/graph/heterodata_with_features.pt` first (it currently points at the full, unreduced graph, which is why trials were taking 300+ minutes each).

### Step 6 — Tests

Three notebooks in `DDL_model_construtor/testing/`, each checking a different layer — run whenever `models/` or `training/` changes:

| Notebook | Checks |
|---|---|
| `encoder_decoder_test.ipynb` | Encoder/decoder shapes, no message-passing leakage, `hidden_dim`/`num_heads` validation |
| `hgnn_test.ipynb` | Full model glue: leakage-free end-to-end, gradients reach both encoder and decoder |
| `test_training.ipynb` | Negative sampling correctness, one training step, loss actually decreases, checkpoints save/reload |

### Step 7 — Inference

`inference/inference.py` — takes two drugs, returns predicted side effects using the trained checkpoint. *(Exact CLI/function signature not reflected here — fill in once finalized, since this file wasn't reviewed as part of this README.)*

### Step 8 — Demo app

`flask_demo/app.py` — web UI wiring together the graph model (`inference/`), the tablet image classifier (`classifier/`), and `llm_explainer.py` for human-readable output. Run with:
```bash
python flask_demo/app.py
```
*(Confirm host/port and any required env vars in `.env`.)*

---

## 3. Project structure

```
.
├── data/                        raw + processed graph tables (full dataset)
├── update_data/                 reduced (~200 drug) graph tables, for laptop-scale training
├── graph/                       full-scale HeteroData + drug fingerprints
├── models/                      encoder.py, decoder.py, hgnn.py — model architecture
├── DDL_model_construtor/
│   ├── notebook/                01_graph_construction, 02_node_feature_construction, 03_model
│   ├── testing/                 encoder_decoder_test, hgnn_test, test_training, full_training
│   ├── training/                train.py, hp_search.py, negative_sampling.py
│   ├── filter/                  drug-subset selection + reduction scripts
│   └── validation/              validate.py
├── inference/                   trained-model → prediction wrapper
├── classifier/                  tablet image classifier (model.pth, classes.json)
├── Image_model_constructor/     training code for the image classifier
├── flask_demo/                  web UI tying both tracks together
└── checkpoints/                 saved training runs
```

---

## 4. What's done vs. what's left

**Done:** full graph construction + validation, real molecular features (PubChem + RDKit), HGT encoder / DistMult decoder architecture (tested for the message-passing leakage bug specifically — drug-drug edges are excluded from message passing by default so the model can't see the edge it's scoring), mini-batch training loop (negative sampling + BCE loss, memory-bounded by design), a 200-drug reduction pipeline for laptop-scale training, a three-tier automated test suite.

**Left, in priority order for submission:** (1) finish Step 3's reduced-graph rebuild if not done yet, (2) run Step 5's fast-path training and record the test AUROC/AUPRC, (3) confirm `inference/inference.py` and `flask_demo/app.py` run end-to-end against the trained checkpoint, (4) if time remains, Step 5b's search for a better config.

---

## 5. Troubleshooting

- **`ModuleNotFoundError: No module named 'models'`** — see the sys.path note in Setup above.
- **Training much slower than expected** — check `data_path` isn't accidentally pointing at `graph/heterodata_with_features.pt` (full 645-drug graph) instead of `update_data/graph/heterodata_with_features.pt` (reduced). This was the cause of 300+ min/trial in an earlier run.
- **Out of memory during training** — lower `batch_size` in Step 5's call; the encoder itself never sees the (huge) drug-drug edge set, so OOM at this stage means the *decoder*/loss batch is too large, not the graph.