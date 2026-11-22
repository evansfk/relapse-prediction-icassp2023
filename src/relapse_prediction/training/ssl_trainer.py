"""PyTorch pretraining loops and embedding extraction for TFC, TS-TCC, and Mixing-Up."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

from relapse_prediction.config import MixingUpConfig, TFCConfig, TSTCCConfig
from relapse_prediction.data.datasets import TFCDataset, TSTCCDataset, WearableTensorDataset
from relapse_prediction.losses.mixup import MixUpLoss
from relapse_prediction.losses.ntxent import NTXentLoss, NTXentLossPoly
from relapse_prediction.models.mixing_up import MixingUpFCN
from relapse_prediction.models.tfc import TFCModel
from relapse_prediction.models.tstcc import TemporalContrasting, TSTCCEncoder
from relapse_prediction.utils.seed import seed_everything

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from torch import Tensor

    from relapse_prediction.models.base import BaseSSLEncoder

_MIN_CONTRASTIVE_BATCH = 2


@dataclass(frozen=True)
class SSLPretrainResult[E: BaseSSLEncoder]:
    """Typed output of an SSL pretraining run preserving the concrete encoder class."""

    encoder: E
    epoch_losses: list[float]


def _effective_batch_size(requested: int, dataset_len: int, /) -> int:
    """Compute a valid contrastive batch size (at least 2, at most ``dataset_len``)."""
    if dataset_len < _MIN_CONTRASTIVE_BATCH:
        msg = f"SSL pretraining requires at least {_MIN_CONTRASTIVE_BATCH} samples, got {dataset_len}"
        raise ValueError(msg)
    return max(_MIN_CONTRASTIVE_BATCH, min(requested, dataset_len))


def pretrain_tfc(
    samples: NDArray[np.floating[Any]] | Tensor,
    config: TFCConfig | None = None,
    /,
    *,
    device: str | torch.device = "cpu",
) -> SSLPretrainResult[TFCModel]:
    """Pretrain :class:`TFCModel` with Time-Frequency Consistency loss (Zhang et al., 2022).

    Note on consistency loss fix:
        Early upstream TFC code used unclipped ``(1 + l_tf - l_i)`` terms that actively
        pushed cross-domain augmented views apart (collapsing ``cos(z_t, z_f) < 0``).
        By default (``config.consistency_mode == "tf"``), this trainer uses the corrected
        positive alignment objective ``lam * (loss_t + loss_f) + (1 - lam) * l_tf``;
        setting ``consistency_mode="clamped_margin"`` applies detached hinge margins
        ``F.relu(1.0 + l_tf - l_i.detach())``.

    Args:
        samples: Unlabeled wearable time series of shape ``(S, C, L)``.
        config: :class:`TFCConfig` hyperparameters.
        device: Target PyTorch device.

    Returns:
        :class:`SSLPretrainResult` containing the trained ``TFCModel`` and per-epoch losses.
    """
    cfg = config or TFCConfig()
    seed_everything(cfg.seed)
    dev = torch.device(device)

    dataset = TFCDataset(
        samples,
        jitter_ratio=cfg.jitter_ratio,
        jitter_scale_ratio=cfg.jitter_scale_ratio,
        max_seg=cfg.max_seg,
        scale_loc=cfg.scale_loc,
        freq_mask_ratio=cfg.freq_mask_ratio,
        freq_add_ratio=cfg.freq_add_ratio,
        seed=cfg.seed,
    )
    batch_size = _effective_batch_size(cfg.batch_size, len(dataset))
    gen = torch.Generator()
    gen.manual_seed(cfg.seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, generator=gen)

    model = TFCModel(cfg).to(dev)
    criterion = NTXentLossPoly(temperature=cfg.temperature, use_cosine_similarity=cfg.use_cosine_similarity)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay)
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10)
        if cfg.use_lr_scheduler
        else None
    )

    epoch_losses: list[float] = []
    lam = cfg.consistency_weight

    for _ in range(cfg.epochs):
        model.train()
        batch_losses: list[float] = []
        for x_t, x_t_aug, x_f, x_f_aug in loader:
            x_t_d = x_t.to(dev)
            x_t_aug_d = x_t_aug.to(dev)
            x_f_d = x_f.to(dev)
            x_f_aug_d = x_f_aug.to(dev)

            optimizer.zero_grad()
            h_t, z_t, h_f, z_f = model(x_t_d, x_f_d)
            h_t_aug, z_t_aug, h_f_aug, z_f_aug = model(x_t_aug_d, x_f_aug_d)

            loss_t: Tensor = criterion(h_t, h_t_aug)
            loss_f: Tensor = criterion(h_f, h_f_aug)
            l_tf: Tensor = criterion(z_t, z_f)

            if cfg.consistency_mode == "clamped_margin":
                l_1: Tensor = criterion(z_t, z_f_aug)
                l_2: Tensor = criterion(z_t_aug, z_f)
                l_3: Tensor = criterion(z_t_aug, z_f_aug)
                loss_c: Tensor = (
                    F.relu(1.0 + l_tf - l_1.detach())
                    + F.relu(1.0 + l_tf - l_2.detach())
                    + F.relu(1.0 + l_tf - l_3.detach())
                )
            else:
                loss_c = l_tf

            loss: Tensor = lam * (loss_t + loss_f) + (1.0 - lam) * loss_c
            torch.autograd.backward(loss)
            optimizer.step()
            batch_losses.append(float(loss.item()))

        mean_epoch_loss = float(np.mean(batch_losses))
        epoch_losses.append(mean_epoch_loss)
        if scheduler is not None:
            scheduler.step(mean_epoch_loss)

    return SSLPretrainResult(encoder=model, epoch_losses=epoch_losses)


def pretrain_tstcc(
    samples: NDArray[np.floating[Any]] | Tensor,
    config: TSTCCConfig | None = None,
    /,
    *,
    device: str | torch.device = "cpu",
) -> SSLPretrainResult[TSTCCEncoder]:
    """Pretrain :class:`TSTCCEncoder` with Temporal and Contextual Contrasting (Eldele et al., 2021).

    Samples a single shared temporal cutoff ``t_sample`` per step so weak and strong
    context vectors summarize identical prefix lengths prior to contextual contrasting.

    Args:
        samples: Unlabeled wearable time series of shape ``(S, C, L)``.
        config: :class:`TSTCCConfig` hyperparameters.
        device: Target PyTorch device.

    Returns:
        :class:`SSLPretrainResult` containing the trained ``TSTCCEncoder`` and per-epoch losses.
    """
    cfg = config or TSTCCConfig()
    seed_everything(cfg.seed)
    dev = torch.device(device)

    dataset = TSTCCDataset(
        samples,
        jitter_scale_ratio=cfg.jitter_scale_ratio,
        jitter_ratio=cfg.jitter_ratio,
        max_seg=cfg.max_seg,
        scale_loc=cfg.scale_loc,
        seed=cfg.seed,
    )
    batch_size = _effective_batch_size(cfg.batch_size, len(dataset))
    gen = torch.Generator()
    gen.manual_seed(cfg.seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, generator=gen)

    encoder = TSTCCEncoder(cfg).to(dev)
    tc_model = TemporalContrasting(cfg).to(dev)
    ntxent = NTXentLoss(temperature=cfg.temperature, use_cosine_similarity=cfg.use_cosine_similarity)

    enc_opt = torch.optim.Adam(encoder.parameters(), lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay)
    tc_opt = torch.optim.Adam(tc_model.parameters(), lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay)

    epoch_losses: list[float] = []
    for _ in range(cfg.epochs):
        encoder.train()
        tc_model.train()
        batch_losses: list[float] = []
        for _, aug1, aug2 in loader:
            aug1_d = aug1.to(dev)
            aug2_d = aug2.to(dev)

            enc_opt.zero_grad()
            tc_opt.zero_grad()

            raw_feat1: Tensor = encoder(aug1_d)
            raw_feat2: Tensor = encoder(aug2_d)
            feat1 = F.normalize(raw_feat1, dim=1)
            feat2 = F.normalize(raw_feat2, dim=1)

            t_sample, _ = tc_model.sample_cutoff(feat1.shape[2], device=dev)
            tc_loss1, zis = tc_model(feat1, feat2, t_sample=t_sample)
            tc_loss2, zjs = tc_model(feat2, feat1, t_sample=t_sample)

            ctx_loss: Tensor = ntxent(zis, zjs)
            loss: Tensor = (tc_loss1 + tc_loss2) * cfg.lambda1 + ctx_loss * cfg.lambda2
            torch.autograd.backward(loss)
            enc_opt.step()
            tc_opt.step()
            batch_losses.append(float(loss.item()))

        epoch_losses.append(float(np.mean(batch_losses)))

    return SSLPretrainResult(encoder=encoder, epoch_losses=epoch_losses)


def pretrain_mixing_up(
    samples: NDArray[np.floating[Any]] | Tensor,
    config: MixingUpConfig | None = None,
    /,
    *,
    device: str | torch.device = "cpu",
) -> SSLPretrainResult[MixingUpFCN]:
    """Pretrain :class:`MixingUpFCN` with Mixing-Up contrastive learning (Wickstrøm et al., 2022).

    Uses same-batch permutation mixing without anchor duplication: computes ``z_1`` once per
    batch and passes ``perm`` directly into :class:`MixUpLoss` so every sample appears once
    in the logit bank with exact soft targets ``lam`` and ``1 - lam``.

    Args:
        samples: Unlabeled wearable time series of shape ``(S, C, L)``.
        config: :class:`MixingUpConfig` hyperparameters.
        device: Target PyTorch device.

    Returns:
        :class:`SSLPretrainResult` containing the trained ``MixingUpFCN`` and per-epoch losses.
    """
    cfg = config or MixingUpConfig()
    seed_everything(cfg.seed)
    dev = torch.device(device)
    rng = np.random.default_rng(cfg.seed)

    dataset = WearableTensorDataset(samples)
    batch_size = _effective_batch_size(cfg.batch_size, len(dataset))
    gen = torch.Generator()
    gen.manual_seed(cfg.seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, generator=gen)

    model = MixingUpFCN(cfg).to(dev)
    criterion = MixUpLoss(temperature=cfg.temperature)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    epoch_losses: list[float] = []
    for _ in range(cfg.epochs):
        model.train()
        batch_losses: list[float] = []
        for x, _ in loader:
            x_1 = x.to(dev)
            perm = torch.randperm(x_1.shape[0], device=dev)
            lam = float(rng.beta(cfg.alpha, cfg.alpha))
            x_aug = lam * x_1 + (1.0 - lam) * x_1[perm]

            optimizer.zero_grad()
            z_1, _ = model(x_1)
            z_aug, _ = model(x_aug)

            loss: Tensor = criterion(z_aug, z_1, lam=lam, perm=perm)
            torch.autograd.backward(loss)
            optimizer.step()
            batch_losses.append(float(loss.item()))

        epoch_losses.append(float(np.mean(batch_losses)))

    return SSLPretrainResult(encoder=model, epoch_losses=epoch_losses)


@torch.no_grad()
def extract_embeddings(
    encoder: BaseSSLEncoder,
    samples: NDArray[np.floating[Any]] | Tensor,
    /,
    *,
    batch_size: int = 128,
    device: str | torch.device = "cpu",
) -> NDArray[np.float32]:
    """Extract frozen ``(S, D)`` embeddings from any :class:`BaseSSLEncoder`.

    Args:
        encoder: Trained SSL encoder.
        samples: Input time series of shape ``(S, C, L)``.
        batch_size: Inference mini-batch size.
        device: Target PyTorch device.

    Returns:
        ``float32`` NumPy array of shape ``(S, encoder.embedding_dim)``.
    """
    dev = torch.device(device)
    encoder = encoder.to(dev)
    encoder.eval()

    dataset = WearableTensorDataset(samples)
    if len(dataset) == 0:
        return np.empty((0, encoder.embedding_dim), dtype=np.float32)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    outputs: list[NDArray[np.float32]] = []
    for x, _ in loader:
        emb = encoder.encode(x.to(dev))
        outputs.append(np.asarray(emb.detach().cpu().numpy(), dtype=np.float32))
    return np.concatenate(outputs, axis=0)
