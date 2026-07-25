import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score

from models.hgnn import DrugSafetyHGNN
from models.encoder import split_polypharmacy_edges
from training.negative_sampling import corrupt_edges

torch.set_num_threads(os.cpu_count())


def train_one_epoch(model, data, train_edge_index, train_edge_type,
                     optimizer, num_drugs, batch_size, generator,
                     grad_clip_norm=1.0, train_edge_frac=1.0):
    model.train()
    criterion = nn.BCEWithLogitsLoss()

    n_total = train_edge_index.shape[1]

    # Optional: subsample which train edges get scored this epoch (unchanged
    # from before -- still a cheap way to cut epoch time on CPU).
    if train_edge_frac < 1.0:
        n = max(1, int(n_total * train_edge_frac))
        subset = torch.randperm(n_total, generator=generator)[:n]
        train_edge_index = train_edge_index[:, subset]
        train_edge_type = train_edge_type[subset]
    n = train_edge_index.shape[1]

    perm = torch.randperm(n, generator=generator)

    total_loss = 0.0
    n_batches = 0
    
    for start in range(0, n, batch_size):
        batch_idx = perm[start:start + batch_size]
        pos_edge_index = train_edge_index[:, batch_idx]
        pos_edge_type = train_edge_type[batch_idx]

        neg_edge_index, neg_edge_type = corrupt_edges(
            pos_edge_index, pos_edge_type, num_drugs, generator=generator
        )

        optimizer.zero_grad()

        embeddings = model.encoder(data)
        drug_embeddings = embeddings["drug"]

        pos_scores = model.decoder(drug_embeddings, pos_edge_index, pos_edge_type)
        neg_scores = model.decoder(drug_embeddings, neg_edge_index, neg_edge_type)

        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])

        loss = criterion(scores, labels)
        loss.backward()

        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, data, edge_index, edge_type, num_drugs, generator):
    model.eval()

    neg_edge_index, neg_edge_type = corrupt_edges(
        edge_index, edge_type, num_drugs, generator=generator
    )

    embeddings = model.encoder(data)
    drug_embeddings = embeddings["drug"]

    pos_scores = model.decoder(drug_embeddings, edge_index, edge_type)
    neg_scores = model.decoder(drug_embeddings, neg_edge_index, neg_edge_type)

    scores = torch.cat([pos_scores, neg_scores]).cpu().numpy()
    labels = torch.cat(
        [torch.ones_like(pos_scores), torch.zeros_like(neg_scores)]
    ).cpu().numpy()

    auroc = roc_auc_score(labels, scores)
    auprc = average_precision_score(labels, scores)
    return auroc, auprc


def run(data_path, output_dir, epochs=10, batch_size=100_000, lr=1e-3,
        hidden_dim=128, num_heads=4, num_layers=2, val_frac=0.1, test_frac=0.1,
        seed=42, eval_every=1, patience=5, grad_clip_norm=1.0,
        lr_patience=2, lr_factor=0.5, min_lr=1e-6, save_artifacts=True,
        verbose=True, train_edge_frac=1.0):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_generator = torch.Generator().manual_seed(seed)
    val_generator = torch.Generator().manual_seed(seed + 1)
    test_generator = torch.Generator().manual_seed(seed + 2)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Using device: {device}")

    data = torch.load(data_path, weights_only=False)
    ddi_key = ("drug", "polypharmacy", "drug")
    num_drugs = data["drug"].num_nodes
    num_relations = int(data[ddi_key].edge_type.max().item()) + 1

    splits = split_polypharmacy_edges(data, val_frac=val_frac, test_frac=test_frac, seed=seed)

    data = data.to(device)
    for split_name in splits:
        splits[split_name]["edge_index"] = splits[split_name]["edge_index"].to(device)
        splits[split_name]["edge_type"] = splits[split_name]["edge_type"].to(device)

    model = DrugSafetyHGNN(
        data, hidden_dim=hidden_dim, num_heads=num_heads,
        num_layers=num_layers, num_relations=num_relations,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=lr_factor, patience=lr_patience, min_lr=min_lr
    )

    history = []
    best_val_auroc = -1.0
    epochs_since_improvement = 0
    stopped_early_at = None

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, data,
            splits["train"]["edge_index"], splits["train"]["edge_type"],
            optimizer, num_drugs, batch_size, train_generator,
            grad_clip_norm=grad_clip_norm, train_edge_frac=train_edge_frac,
        )

        log = {"epoch": epoch, "train_loss": train_loss,
               "lr": optimizer.param_groups[0]["lr"]}

        if epoch % eval_every == 0:
            val_auroc, val_auprc = evaluate(
                model, data, splits["val"]["edge_index"], splits["val"]["edge_type"],
                num_drugs, val_generator,
            )
            log["val_auroc"] = val_auroc
            log["val_auprc"] = val_auprc

            scheduler.step(val_auroc)

            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                epochs_since_improvement = 0
                if save_artifacts:
                    torch.save(model.state_dict(), output_dir / "best_model.pt")
            else:
                epochs_since_improvement += eval_every

            if epochs_since_improvement >= patience:
                stopped_early_at = epoch
                if verbose:
                    print(f"Early stopping at epoch {epoch} "
                          f"(no val_auroc improvement for {patience} epochs)")
                history.append(log)
                break

        if verbose:
            print(log)
        history.append(log)

    if save_artifacts:
        (output_dir / "training_history.json").write_text(json.dumps(history, indent=2))
        torch.save(model.state_dict(), output_dir / "final_model.pt")

        # reload best checkpoint (by val AUROC) before final test evaluation
        best_ckpt = output_dir / "best_model.pt"
        if best_ckpt.exists():
            model.load_state_dict(torch.load(best_ckpt, weights_only=True, map_location=device))

    test_auroc, test_auprc = evaluate(
        model, data, splits["test"]["edge_index"], splits["test"]["edge_type"],
        num_drugs, test_generator,
    )
    if verbose:
        print(f"Final test AUROC: {test_auroc:.4f}, AUPRC: {test_auprc:.4f}")

    result = {
        "best_val_auroc": best_val_auroc,
        "test_auroc": test_auroc,
        "test_auprc": test_auprc,
        "stopped_early_at": stopped_early_at,
        "history": history,
    }
    return model, result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train DrugSafetyHGNN")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-edge-frac", type=float, default=1.0,
                         help="fraction of train edges to sample per epoch (CPU speedup, default 1.0=all)")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run(
        data_path=args.data_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        eval_every=args.eval_every,
        patience=args.patience,
        grad_clip_norm=args.grad_clip_norm,
        seed=args.seed,
        train_edge_frac=args.train_edge_frac,
    )