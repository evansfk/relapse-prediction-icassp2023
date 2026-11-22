r"""Run the Person Identification proxy task (Table 2) on SSL or MiniRocket features.

Example:
    uv run python scripts/evaluate_person_id.py --data-dir outputs/synthetic_4h \
        --embeddings outputs/runs/tfc_4h/downstream_embeddings.npy --min-samples-per-user 10
"""

from __future__ import annotations

import argparse

from pathlib import Path

import numpy as np
import torch

from relapse_prediction.data.synthetic import generate_synthetic_wearable_dataset
from relapse_prediction.models.minirocket import MiniRocketExtractor
from relapse_prediction.training.evaluator import run_person_id_evaluation


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Person Identification evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None, help="Directory with dataset.npz.")
    parser.add_argument(
        "--embeddings",
        type=Path,
        nargs="+",
        default=None,
        help="One or n_splits per-seed .npy embedding files (omit to run 5-seed MiniRocket baseline).",
    )
    parser.add_argument(
        "--minirocket-features",
        type=int,
        default=9996,
        help="MiniRocket feature count when --embeddings is omitted (9996 in paper).",
    )
    parser.add_argument(
        "--min-samples-per-user",
        type=int,
        default=None,
        help="Min samples to retain a user (defaults to 100 for --data-dir, 10 for synthetic).",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    """Evaluate SSL or MiniRocket embeddings on the Person Identification proxy task."""
    args = parse_args()
    if args.data_dir is not None:
        loaded = np.load(args.data_dir / "dataset.npz")
        pretext_x = loaded["pretext_x"]
        downstream_x = loaded["downstream_x"]
        user_ids = loaded["downstream_user_ids"]
        min_samples = args.min_samples_per_user if args.min_samples_per_user is not None else 100
    else:
        bundle = generate_synthetic_wearable_dataset(window_hours=4, samples_per_user=30, seed=args.seed)
        pretext_x = bundle.pretext_x
        downstream_x = bundle.downstream_x
        user_ids = bundle.downstream_user_ids
        min_samples = args.min_samples_per_user if args.min_samples_per_user is not None else 10

    emb_paths: list[Path] | None = args.embeddings
    if emb_paths is not None:
        if len(emb_paths) == 1:
            emb_input: np.ndarray | list[np.ndarray] = np.load(emb_paths[0])
            source_name = emb_paths[0].name
        else:
            emb_input = [np.load(p) for p in emb_paths]
            source_name = f"{len(emb_paths)} per-seed embeddings"
    else:
        # Fit MiniRocket strictly on unlabeled pretext data (Section 4) with n_splits distinct seeds.
        emb_list = [
            MiniRocketExtractor(num_features=args.minirocket_features, device=args.device, seed=args.seed + i)
            .fit(pretext_x)
            .transform(downstream_x)
            for i in range(args.n_splits)
        ]
        emb_input = emb_list
        source_name = f"MiniRocket ({emb_list[0].shape[1]} dims, {args.n_splits} seeds)"

    summary = run_person_id_evaluation(
        emb_input,
        user_ids,
        min_samples_per_user=min_samples,
        n_splits=args.n_splits,
        epochs=args.epochs,
        device=args.device,
        seed=args.seed,
    )

    print(f"Person Identification Proxy Task — {source_name}")
    print(f"  Retained users : {summary.n_users_retained} ({summary.n_samples_retained} samples)")
    print(f"  Macro-F1       : {summary.macro_f1_mean:.4f} (+/- {summary.macro_f1_std:.4f})")
    print(f"  Macro-AUPRC    : {summary.macro_auprc_mean:.4f} (+/- {summary.macro_auprc_std:.4f})")


if __name__ == "__main__":
    main()
