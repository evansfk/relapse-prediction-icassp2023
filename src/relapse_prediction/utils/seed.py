"""Deterministic seeding utilities."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int, /) -> None:
    """Seed Python, NumPy, and PyTorch random number generators.

    Args:
        seed: Non-negative integer seed.
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
