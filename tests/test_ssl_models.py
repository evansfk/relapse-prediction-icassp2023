"""Unit tests for SSL encoders (TFC, TS-TCC, Mixing-Up), MiniRocket, and YAML configs."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from torch import nn

from relapse_prediction.config import MixingUpConfig, SurvivalConfig, TFCConfig, TSTCCConfig
from relapse_prediction.models.minirocket import MiniRocketExtractor, _compute_dilations
from relapse_prediction.models.mixing_up import MixingUpFCN
from relapse_prediction.models.tfc import TFCModel
from relapse_prediction.models.tstcc import TemporalContrasting, TSTCCEncoder

if TYPE_CHECKING:
    from relapse_prediction.models.base import BaseSSLEncoder

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


@pytest.mark.parametrize(
    ("seq_len", "expected_tstcc_dim"),
    [
        (240, 768),
        (480, 1408),  # Exact embedding dimension reported in ICASSP 2023 Section 4
        (720, 2048),
    ],
)
def test_paper_embedding_dimensions_and_clean_batchnorm_init(seq_len: int, expected_tstcc_dim: int) -> None:
    x = torch.randn(4, 10, seq_len)

    mixup = MixingUpFCN(MixingUpConfig(input_channels=10, seq_len=seq_len))
    mixup.eval()
    assert mixup.embedding_dim == 128
    assert mixup.encode(x).shape == (4, 128)

    tfc = TFCModel(TFCConfig(input_channels=10, seq_len=seq_len))
    # Regression C16: dummy shape inference in __init__ must not mutate BatchNorm running stats.
    for module in tfc.modules():
        if isinstance(module, nn.BatchNorm1d) and module.num_batches_tracked is not None:
            assert int(module.num_batches_tracked.item()) == 0
    tfc.eval()
    assert tfc.embedding_dim == 256
    assert tfc.encode(x).shape == (4, 256)

    tstcc = TSTCCEncoder(TSTCCConfig(input_channels=10, seq_len=seq_len))
    for module in tstcc.modules():
        if isinstance(module, nn.BatchNorm1d) and module.num_batches_tracked is not None:
            assert int(module.num_batches_tracked.item()) == 0
    tstcc.eval()
    assert tstcc.embedding_dim == expected_tstcc_dim
    assert tstcc.encode(x).shape == (4, expected_tstcc_dim)


@pytest.mark.parametrize("model_name", ["mixing_up", "tfc", "tstcc"])
def test_ssl_encoder_batch_independence_and_gradients(model_name: str) -> None:
    torch.manual_seed(0)
    x = torch.randn(6, 10, 240, requires_grad=True)

    model: BaseSSLEncoder
    if model_name == "mixing_up":
        model = MixingUpFCN(MixingUpConfig(input_channels=10, seq_len=240))
    elif model_name == "tfc":
        model = TFCModel(TFCConfig(input_channels=10, seq_len=240))
    else:
        model = TSTCCEncoder(TSTCCConfig(input_channels=10, seq_len=240))

    model.train()
    emb = model.encode(x)
    torch.autograd.backward(emb.sum())
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for name, param in model.named_parameters():
        if model_name == "mixing_up" and name.startswith("proj_head"):
            continue
        assert param.grad is not None, f"Missing gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"

    model.eval()
    x_eval = x.detach()
    with torch.no_grad():
        batch_out = model.encode(x_eval)
        batch_out_2 = model.encode(x_eval)
        single_outs = torch.cat([model.encode(x_eval[i : i + 1]) for i in range(len(x_eval))], dim=0)

    torch.testing.assert_close(batch_out, batch_out_2)
    torch.testing.assert_close(batch_out, single_outs, rtol=1e-4, atol=1e-4)


def test_tstcc_temporal_contrasting_shared_cutoff_and_guard() -> None:
    cfg = TSTCCConfig(input_channels=10, seq_len=240)
    enc = TSTCCEncoder(cfg)
    tc = TemporalContrasting(cfg)

    x1 = torch.randn(8, 10, 240)
    x2 = torch.randn(8, 10, 240)
    f1 = enc(x1)
    f2 = enc(x2)
    cutoff, _ = tc.sample_cutoff(f1.shape[2], device=f1.device)
    nce, proj = tc(f1, f2, t_sample=cutoff)
    assert torch.isfinite(nce)
    assert proj.shape == (8, cfg.final_out_channels // 4)
    torch.autograd.backward(nce + proj.sum())
    for p in tc.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()

    with pytest.raises(ValueError, match="seq_len >= 2"):
        tc(torch.randn(4, cfg.final_out_channels, 1), torch.randn(4, cfg.final_out_channels, 1))


def test_minirocket_extractor_deterministic_bounded_and_empty_safe() -> None:
    rng = np.random.default_rng(123)
    x = rng.standard_normal((24, 10, 120)).astype(np.float32)

    # Regression MR1: log-space dilation multiplicity preserves fine-scale density (d=1 gets 27/119 @ L=240).
    dils_4h, counts_4h = _compute_dilations(seq_len=240, num_features=9996, max_dilations_per_kernel=32)
    assert dils_4h[0] == 1
    assert counts_4h[0] == 27
    assert sum(counts_4h) == 119

    mrf1 = MiniRocketExtractor(num_features=336, seed=7)
    f1 = mrf1.fit_transform(x)
    assert f1.shape == (24, 336)
    assert np.all((f1 >= 0.0) & (f1 <= 1.0))

    mrf2 = MiniRocketExtractor(num_features=336, seed=7)
    f2 = mrf2.fit_transform(x)
    np.testing.assert_allclose(f1, f2)

    empty_out = mrf1.transform(np.empty((0, 10, 120), dtype=np.float32))
    assert empty_out.shape == (0, 336)

    with pytest.raises(ValueError, match="seq_len >= 9"):
        MiniRocketExtractor(num_features=84).fit(np.ones((4, 10, 8), dtype=np.float32))
    with pytest.raises(ValueError, match=r"Expected shape \(S, 10, 120\)"):
        mrf1.transform(np.ones((4, 10, 96), dtype=np.float32))


def test_all_yaml_configs_load_cleanly_and_coerce_tuples(tmp_path: Path) -> None:
    for name in ["tfc_4h.yaml", "tfc_8h.yaml", "tfc_12h.yaml"]:
        cfg = TFCConfig.from_yaml(CONFIGS_DIR / "ssl" / name)
        assert cfg.input_channels == 10
        assert isinstance(cfg.betas, tuple)
    for name in ["tstcc_4h.yaml", "tstcc_8h.yaml", "tstcc_12h.yaml"]:
        cfg_t = TSTCCConfig.from_yaml(CONFIGS_DIR / "ssl" / name)
        assert cfg_t.input_channels == 10
    cfg_m = MixingUpConfig.from_yaml(CONFIGS_DIR / "ssl" / "mixing_up.yaml")
    assert cfg_m.projection_dim == 128
    cfg_s = SurvivalConfig.from_yaml(CONFIGS_DIR / "survival" / "default.yaml")
    assert cfg_s.n_trees == 200

    # Regression C19: risk_quantiles list in YAML is coerced to tuple, and non-mapping YAML raises ValueError.
    custom_yaml = tmp_path / "custom.yaml"
    custom_yaml.write_text("risk_quantiles: [0.25, 0.75]\n", encoding="utf-8")
    loaded = SurvivalConfig.from_yaml(custom_yaml)
    assert loaded.risk_quantiles == (0.25, 0.75)

    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("- item1\n- item2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Expected YAML mapping"):
        SurvivalConfig.from_yaml(bad_yaml)
