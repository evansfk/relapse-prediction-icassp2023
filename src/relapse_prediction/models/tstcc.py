"""Temporal and Contextual Contrasting (TS-TCC) architecture (Eldele et al., IJCAI 2021)."""

from __future__ import annotations

from typing import override

import torch
import torch.nn.functional as F

from torch import Tensor, nn

from relapse_prediction.config import TSTCCConfig

from .base import BaseSSLEncoder, build_conv_encoder_blocks

_MIN_TC_SEQ_LEN = 2
_EXPECTED_NDIM = 3


class _FeedForward(nn.Module):
    """Position-wise feed-forward block inside ``SeqTransformer``."""

    def __init__(self, *, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, dim), nn.Dropout(dropout)
        )

    @override
    def forward(self, x: Tensor, /) -> Tensor:
        out: Tensor = self.net(x)
        return out


class _MultiHeadAttention(nn.Module):
    """Pure-PyTorch multi-head self-attention matching TS-TCC's ``attention.py``."""

    def __init__(self, *, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        if dim % heads != 0:
            msg = f"Attention dim ({dim}) must be divisible by heads ({heads})"
            raise ValueError(msg)
        self.heads = heads
        self.scale = dim**-0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))

    @override
    def forward(self, x: Tensor, /) -> Tensor:
        b, n, d = x.shape
        head_dim = d // self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = q.reshape(b, n, self.heads, head_dim).transpose(1, 2)
        k = k.reshape(b, n, self.heads, head_dim).transpose(1, 2)
        v = v.reshape(b, n, self.heads, head_dim).transpose(1, 2)

        dots = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(b, n, d)
        projected: Tensor = self.to_out(out)
        return projected


class _TransformerBlock(nn.Module):
    """Pre-norm residual Transformer layer."""

    def __init__(self, *, dim: int, heads: int, mlp_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _MultiHeadAttention(dim=dim, heads=heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = _FeedForward(dim=dim, hidden_dim=mlp_dim, dropout=dropout)

    @override
    def forward(self, x: Tensor, /) -> Tensor:
        x = x + self.attn(self.norm1(x))
        out: Tensor = x + self.ff(self.norm2(x))
        return out


class SeqTransformer(nn.Module):
    """Autoregressive sequence summarizer producing context vector ``c_t`` via a learned CLS token.

    Args:
        patch_size: Input feature channel dimension per timestep.
        dim: Transformer hidden dimension.
        depth: Number of Transformer layers.
        heads: Number of attention heads.
        mlp_dim: Hidden dimension inside the feed-forward block.
        dropout: Dropout rate.
    """

    def __init__(self, *, patch_size: int, dim: int, depth: int, heads: int, mlp_dim: int, dropout: float) -> None:
        super().__init__()
        self.patch_to_embedding = nn.Linear(patch_size, dim)
        self.c_token = nn.Parameter(torch.randn(1, 1, dim))
        self.layers = nn.ModuleList(
            [_TransformerBlock(dim=dim, heads=heads, mlp_dim=mlp_dim, dropout=dropout) for _ in range(depth)]
        )

    @override
    def forward(self, forward_seq: Tensor, /) -> Tensor:
        """Summarize ``forward_seq`` of shape ``(B, T, patch_size)`` into ``(B, dim)``."""
        x: Tensor = self.patch_to_embedding(forward_seq)
        c_tokens = self.c_token.expand(x.shape[0], -1, -1)
        x = torch.cat([c_tokens, x], dim=1)
        for layer in self.layers:
            x = layer(x)
        cls_token: Tensor = x[:, 0]
        return cls_token


class TSTCCEncoder(BaseSSLEncoder):
    """Three-block 1-D CNN backbone for TS-TCC.

    On the 8-hour dataset (``L = 480``), the backbone outputs feature maps of shape
    ``(B, 64, 22)``, flattening to the **1408-dimensional** embedding reported in Section 4
    of Fekas et al. (IEEE ICASSP 2023) (``768`` on 4h and ``2048`` on 12h).

    Args:
        config: :class:`TSTCCConfig` specifying CNN architecture and sequence length.
    """

    def __init__(self, config: TSTCCConfig | None = None, /) -> None:
        super().__init__()
        cfg = config or TSTCCConfig()
        self.config = cfg

        self.conv_encoder = build_conv_encoder_blocks(
            input_channels=cfg.input_channels,
            kernel_size=cfg.kernel_size,
            stride=cfg.stride,
            final_out_channels=cfg.final_out_channels,
            dropout=cfg.dropout,
        )

        # Infer feature map dimensions in eval mode so BatchNorm running stats stay untouched.
        self.conv_encoder.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, cfg.input_channels, cfg.seq_len)
            feat = self.forward(dummy)
            self._feature_seq_len = int(feat.shape[2])
            self._embedding_dim = int(feat.shape[1] * feat.shape[2])
        self.conv_encoder.train()

    @property
    def feature_seq_len(self) -> int:
        """Temporal length ``L_enc`` of the convolutional feature map."""
        return self._feature_seq_len

    @property
    @override
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def _validate_input_shape(self, x: Tensor, /) -> None:
        expected_tail = (self.config.input_channels, self.config.seq_len)
        if x.ndim != _EXPECTED_NDIM or x.shape[1:] != expected_tail:
            msg = (
                f"TSTCCEncoder expected shape (B, {self.config.input_channels}, {self.config.seq_len}), "
                f"got {tuple(x.shape)}"
            )
            raise ValueError(msg)

    @override
    def forward(self, x: Tensor, /) -> Tensor:
        """Return convolutional feature map of shape ``(B, final_out_channels, L_enc)``."""
        self._validate_input_shape(x)
        out: Tensor = self.conv_encoder(x)
        return out

    @override
    def encode(self, x: Tensor, /) -> Tensor:
        """Extract flattened convolutional embeddings of shape ``(B, final_out_channels * L_enc)``."""
        if x.shape[0] == 0:
            self._validate_input_shape(x)
            return x.new_empty((0, self._embedding_dim))
        feat = self.forward(x)
        return feat.reshape(feat.shape[0], -1)


class TemporalContrasting(nn.Module):
    """Cross-view autoregressive temporal contrasting module for TS-TCC.

    Summarizes a prefix of ``features_aug1`` with :class:`SeqTransformer` and predicts
    ``k = 1..timesteps`` future steps of ``features_aug2`` via InfoNCE, while returning
    a projected context vector for contextual contrasting.

    Args:
        config: :class:`TSTCCConfig` specifying hidden dimensions and prediction horizon.
    """

    def __init__(self, config: TSTCCConfig, /) -> None:
        super().__init__()
        self.num_channels = config.final_out_channels
        self.timesteps = config.tc_timesteps

        self.wk = nn.ModuleList([nn.Linear(config.tc_hidden_dim, self.num_channels) for _ in range(self.timesteps)])
        self.projection_head = nn.Sequential(
            nn.Linear(config.tc_hidden_dim, self.num_channels // 2),
            nn.BatchNorm1d(self.num_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.num_channels // 2, self.num_channels // 4),
        )
        self.seq_transformer = SeqTransformer(
            patch_size=self.num_channels,
            dim=config.tc_hidden_dim,
            depth=config.transformer_depth,
            heads=config.transformer_heads,
            mlp_dim=config.transformer_mlp_dim,
            dropout=config.transformer_dropout,
        )

    def sample_cutoff(self, seq_len: int, /, *, device: torch.device) -> tuple[int, int]:
        """Sample a valid ``(t_sample, horizon)`` pair for an encoded sequence of length ``seq_len``."""
        if seq_len < _MIN_TC_SEQ_LEN:
            msg = f"TemporalContrasting requires encoded seq_len >= {_MIN_TC_SEQ_LEN}, got {seq_len}"
            raise ValueError(msg)
        horizon = min(self.timesteps, seq_len - 1)
        t_sample = int(torch.randint(0, seq_len - horizon, (1,), device=device).item())
        return t_sample, horizon

    @override
    def forward(
        self, features_aug1: Tensor, features_aug2: Tensor, /, *, t_sample: int | None = None
    ) -> tuple[Tensor, Tensor]:
        """Compute cross-view temporal contrasting loss and projected context embedding.

        Args:
            features_aug1: Normalized encoder features ``(B, C, L_enc)`` from view 1.
            features_aug2: Normalized encoder features ``(B, C, L_enc)`` from view 2.
            t_sample: Optional shared cutoff index so both views summarize identical temporal
                prefixes before contextual contrasting.

        Returns:
            Tuple ``(nce_loss, projected_context)``.
        """
        z_aug1 = features_aug1.transpose(1, 2)
        z_aug2 = features_aug2.transpose(1, 2)

        batch_size, seq_len, _ = z_aug1.shape
        if t_sample is None:
            cutoff, horizon = self.sample_cutoff(seq_len, device=z_aug1.device)
        else:
            if seq_len < _MIN_TC_SEQ_LEN:
                msg = f"TemporalContrasting requires encoded seq_len >= {_MIN_TC_SEQ_LEN}, got {seq_len}"
                raise ValueError(msg)
            horizon = min(self.timesteps, seq_len - 1)
            cutoff = min(max(t_sample, 0), seq_len - horizon - 1)

        forward_seq = z_aug1[:, : cutoff + 1, :]
        c_t: Tensor = self.seq_transformer(forward_seq)

        nce = z_aug1.new_tensor(0.0)
        for step in range(horizon):
            target_step = z_aug2[:, cutoff + step + 1, :]
            pred_step: Tensor = self.wk[step](c_t)
            logits = torch.mm(target_step, pred_step.T)
            nce = nce + torch.diag(F.log_softmax(logits, dim=-1)).sum()

        nce = nce / (-float(batch_size * horizon))
        proj: Tensor = self.projection_head(c_t)
        return nce, proj
