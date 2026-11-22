"""Generate a synthetic wearable time-series + right-censored survival dataset.

Example:
    uv run python scripts/generate_synthetic_data.py --window-hours 4 --output-dir outputs/synthetic_4h
"""

from __future__ import annotations

import argparse

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from relapse_prediction.data.synthetic import generate_synthetic_wearable_dataset

if TYPE_CHECKING:
    from relapse_prediction.config import WindowHours


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for synthetic wearable dataset generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-hours", type=int, choices=[4, 8, 12], default=4)
    parser.add_argument("--seq-len", type=int, default=None, help="Optional sequence length override.")
    parser.add_argument("--n-pretext-users", type=int, default=20)
    parser.add_argument("--n-downstream-users", type=int, default=14)
    parser.add_argument("--samples-per-user", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/synthetic_4h"))
    return parser.parse_args()


def _resolve_window_hours(raw_hours: int, /) -> WindowHours:
    match raw_hours:
        case 4 | 8 | 12 as valid_hours:
            return valid_hours
        case _:
            msg = f"window_hours must be 4, 8, or 12, got {raw_hours}"
            raise ValueError(msg)


def main() -> None:
    """Generate and persist a synthetic wearable + survival dataset bundle."""
    args = parse_args()
    window_hours = _resolve_window_hours(args.window_hours)
    bundle = generate_synthetic_wearable_dataset(
        window_hours=window_hours,
        seq_len=args.seq_len,
        n_pretext_users=args.n_pretext_users,
        n_downstream_users=args.n_downstream_users,
        samples_per_user=args.samples_per_user,
        seed=args.seed,
    )

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_dir / "dataset.npz",
        pretext_x=bundle.pretext_x,
        pretext_user_ids=bundle.pretext_user_ids,
        downstream_x=bundle.downstream_x,
        downstream_user_ids=bundle.downstream_user_ids,
        event=bundle.event,
        time_days=bundle.time_days,
    )
    bundle.metadata.to_csv(out_dir / "metadata.csv", index=False)
    bundle.static_features.to_csv(out_dir / "static_features.csv")
    bundle.relapse_episodes.to_csv(out_dir / "relapse_episodes.csv", index=False)

    print(f"Saved synthetic dataset to {out_dir}")
    print(f"  Pretext shape    : {bundle.pretext_x.shape} ({args.n_pretext_users} users)")
    print(f"  Downstream shape : {bundle.downstream_x.shape} ({args.n_downstream_users} users)")
    print(f"  Event rate       : {bundle.event.mean():.1%} uncensored ({int(bundle.event.sum())}/{len(bundle.event)})")


if __name__ == "__main__":
    main()
