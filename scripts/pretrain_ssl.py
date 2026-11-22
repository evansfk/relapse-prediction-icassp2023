r"""Pretrain an SSL encoder (TFC, TS-TCC, or Mixing-Up) and export frozen embeddings.

Example:
    uv run python scripts/pretrain_ssl.py --method tfc --config configs/ssl/tfc_4h.yaml \
        --data-dir outputs/synthetic_4h --epochs 5 --output-dir outputs/runs/tfc_4h
"""

from __future__ import annotations

import argparse

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from relapse_prediction.config import MixingUpConfig, TFCConfig, TSTCCConfig, select_channel_indices
from relapse_prediction.data.synthetic import generate_synthetic_wearable_dataset
from relapse_prediction.training.ssl_trainer import (
    SSLPretrainResult,
    extract_embeddings,
    pretrain_mixing_up,
    pretrain_tfc,
    pretrain_tstcc,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from relapse_prediction.models.base import BaseSSLEncoder

_EXPECTED_NDIM = 3


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for SSL pretraining."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["tfc", "tstcc", "mixing_up"], required=True)
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML config path.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Directory with dataset.npz.")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic data on the fly.")
    parser.add_argument("--exclude-groups", nargs="*", default=[], help="Modality groups to exclude (Table 3).")
    parser.add_argument("--epochs", type=int, default=None, help="Override config epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override config batch size.")
    parser.add_argument("--seed", type=int, default=None, help="Override config random seed.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ssl_run"))
    return parser.parse_args()


def _load_method_config(args: argparse.Namespace, /) -> TFCConfig | TSTCCConfig | MixingUpConfig:
    config_cls: type[TFCConfig | TSTCCConfig | MixingUpConfig] = {
        "tfc": TFCConfig,
        "tstcc": TSTCCConfig,
        "mixing_up": MixingUpConfig,
    }[args.method]
    cfg = config_cls.from_yaml(args.config) if args.config is not None else config_cls()
    if args.seed is not None:
        cfg.seed = args.seed
    if args.epochs is not None:
        if args.epochs < 1:
            msg = f"--epochs must be >= 1, got {args.epochs}"
            raise ValueError(msg)
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    return cfg


def _run_method(
    cfg: TFCConfig | TSTCCConfig | MixingUpConfig, pretext_x: NDArray[np.float32], /, *, device: str
) -> SSLPretrainResult[BaseSSLEncoder]:
    cfg.input_channels = int(pretext_x.shape[1])
    cfg.seq_len = int(pretext_x.shape[2])
    if isinstance(cfg, TFCConfig):
        res_t = pretrain_tfc(pretext_x, cfg, device=device)
        return SSLPretrainResult(encoder=res_t.encoder, epoch_losses=res_t.epoch_losses)
    if isinstance(cfg, TSTCCConfig):
        res_ts = pretrain_tstcc(pretext_x, cfg, device=device)
        return SSLPretrainResult(encoder=res_ts.encoder, epoch_losses=res_ts.epoch_losses)
    res_m = pretrain_mixing_up(pretext_x, cfg, device=device)
    return SSLPretrainResult(encoder=res_m.encoder, epoch_losses=res_m.epoch_losses)


def main() -> None:
    """Run SSL pretraining on pretext data and export downstream embeddings."""
    args = parse_args()
    cfg = _load_method_config(args)

    if args.data_dir is not None:
        loaded = np.load(args.data_dir / "dataset.npz")
        pretext_x = loaded["pretext_x"]
        downstream_x = loaded["downstream_x"]
    elif args.synthetic:
        seed = args.seed if args.seed is not None else cfg.seed
        bundle = generate_synthetic_wearable_dataset(
            window_hours=4, seq_len=cfg.seq_len, samples_per_user=30, seed=seed
        )
        pretext_x = bundle.pretext_x
        downstream_x = bundle.downstream_x
    else:
        msg = "Specify either --data-dir <path> or --synthetic"
        raise SystemExit(msg)

    if (
        pretext_x.ndim != _EXPECTED_NDIM
        or downstream_x.ndim != _EXPECTED_NDIM
        or pretext_x.shape[1:] != downstream_x.shape[1:]
    ):
        msg = f"Mismatched pretext_x {pretext_x.shape} and downstream_x {downstream_x.shape} shapes"
        raise ValueError(msg)

    if args.exclude_groups:
        keep_channels = select_channel_indices(args.exclude_groups)
        pretext_x = pretext_x[:, keep_channels, :]
        downstream_x = downstream_x[:, keep_channels, :]

    result = _run_method(cfg, pretext_x, device=args.device)
    embeddings = extract_embeddings(result.encoder, downstream_x, device=args.device)

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(result.encoder.state_dict(), out_dir / f"{args.method}_encoder.pt")
    np.save(out_dir / "downstream_embeddings.npy", embeddings)

    print(f"Finished {args.method.upper()} pretraining ({len(result.epoch_losses)} epochs)")
    print(f"  Final epoch loss      : {result.epoch_losses[-1]:.4f}")
    print(f"  Downstream embeddings : {embeddings.shape} -> {out_dir / 'downstream_embeddings.npy'}")


if __name__ == "__main__":
    main()
