"""
Hyperparameter search for DrugSafetyHGNN.
"""
import argparse
import json
import random
from pathlib import Path

from .train import run


SEARCH_SPACE = {
    "hidden_dim": [64, 128],
    "num_heads": [2, 4],
    "num_layers": [1, 2],
    "lr": [1e-2, 5e-3, 1e-3, 5e-4],
    "batch_size": [50_000, 100_000],
}


def sample_config(rng):
    return {k: rng.choice(v) for k, v in SEARCH_SPACE.items()}


def main():
    parser = argparse.ArgumentParser(description="Random hyperparameter search for DrugSafetyHGNN")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-trials", type=int, default=8,
                         help="lowered default for CPU-only time budgets; raise if you have time")
    parser.add_argument("--search-epochs", type=int, default=8,
                         help="epoch budget per trial during search (kept short)")
    parser.add_argument("--search-patience", type=int, default=3)
    parser.add_argument("--final-epochs", type=int, default=60,
                         help="epoch budget for the final retrain of the winning config")
    parser.add_argument("--final-patience", type=int, default=8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--search-train-edge-frac", type=float, default=1.0,
                         help="subsample this fraction of train edges per epoch DURING SEARCH ONLY "
                              "(e.g. 0.4) to cut search-phase epoch time on CPU; final retrain always "
                              "uses the full train set")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    trial_results = []

    for trial_idx in range(1, args.n_trials + 1):
        config = sample_config(rng)
        trial_dir = output_dir / f"trial_{trial_idx:03d}"
        print(f"\n=== Trial {trial_idx}/{args.n_trials}: {config} ===")

        try:
            _, result = run(
                data_path=args.data_path,
                output_dir=trial_dir,
                epochs=args.search_epochs,
                batch_size=config["batch_size"],
                lr=config["lr"],
                hidden_dim=config["hidden_dim"],
                num_heads=config["num_heads"],
                num_layers=config["num_layers"],
                val_frac=args.val_frac,
                test_frac=args.test_frac,
                seed=args.seed,
                patience=args.search_patience,
                save_artifacts=False,   # cheap trials: skip checkpoint I/O
                verbose=False,
                train_edge_frac=args.search_train_edge_frac,
            )
        except Exception as e:
            print(f"Trial {trial_idx} failed: {e}")
            continue

        trial_results.append({
            "trial": trial_idx,
            "config": config,
            "best_val_auroc": result["best_val_auroc"],
        })
        print(f"  -> best_val_auroc = {result['best_val_auroc']:.4f}")

    if not trial_results:
        raise RuntimeError("All trials failed; check data path and model imports.")

    trial_results.sort(key=lambda r: r["best_val_auroc"], reverse=True)
    (output_dir / "search_results.json").write_text(json.dumps(trial_results, indent=2))

    best = trial_results[0]
    print(f"\nBest config from search: {best['config']} "
          f"(val_auroc={best['best_val_auroc']:.4f})")

    print("\n=== Retraining winning config to full epoch budget ===")
    final_dir = output_dir / "final"
    model, final_result = run(
        data_path=args.data_path,
        output_dir=final_dir,
        epochs=args.final_epochs,
        batch_size=best["config"]["batch_size"],
        lr=best["config"]["lr"],
        hidden_dim=best["config"]["hidden_dim"],
        num_heads=best["config"]["num_heads"],
        num_layers=best["config"]["num_layers"],
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
        patience=args.final_patience,
        save_artifacts=True,
        verbose=True,
    )

    summary = {
        "winning_config": best["config"],
        "search_val_auroc": best["best_val_auroc"],
        "final_val_auroc": final_result["best_val_auroc"],
        "final_test_auroc": final_result["test_auroc"],
        "final_test_auprc": final_result["test_auprc"],
        "stopped_early_at": final_result["stopped_early_at"],
    }
    (output_dir / "final_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== Phase 1 complete ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()