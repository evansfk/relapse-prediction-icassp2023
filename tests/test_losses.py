"""Unit tests for contrastive and survival loss functions."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from pytorch_metric_learning.losses import NTXentLoss as PMLNTXentLoss
from pytorch_metric_learning.losses import SelfSupervisedLoss
from torch import Tensor

from relapse_prediction.losses.mixup import MixUpLoss
from relapse_prediction.losses.ntxent import NTXentLoss, NTXentLossPoly
from relapse_prediction.losses.survival import CoxPHLoss, MTLRLoss


@pytest.fixture
def normalized_embeddings(request: pytest.FixtureRequest) -> tuple[Tensor, Tensor]:
    torch.manual_seed(42)
    batch_size = int(request.param)
    z1 = F.normalize(torch.randn(batch_size, 128), dim=1)
    z2 = F.normalize(torch.randn(batch_size, 128), dim=1)
    return z1, z2


@pytest.mark.parametrize("temperature", [0.1, 0.5, 1.0, 2.0])
@pytest.mark.parametrize("normalized_embeddings", [2, 8, 32, 64], indirect=True)
def test_ntxent_matches_pytorch_metric_learning(
    normalized_embeddings: tuple[Tensor, Tensor], temperature: float
) -> None:
    z1, z2 = normalized_embeddings
    loss_fn = NTXentLoss(temperature=temperature)
    our_loss: Tensor = loss_fn(z1, z2)
    our_loss_2, sim = loss_fn.forward_with_logits(z1, z2)
    ref_loss: Tensor = SelfSupervisedLoss(PMLNTXentLoss(temperature=temperature))(z1, z2)
    torch.testing.assert_close(our_loss, ref_loss, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(our_loss, our_loss_2)
    assert sim.shape == (2 * len(z1), 2 * len(z1))


@pytest.mark.parametrize("normalized_embeddings", [32], indirect=True)
def test_ntxent_gradients_match_reference(normalized_embeddings: tuple[Tensor, Tensor]) -> None:
    z1, z2 = normalized_embeddings
    z1_ours = z1.clone().requires_grad_()
    z2_ours = z2.clone().requires_grad_()
    loss_ours: Tensor = NTXentLoss(temperature=0.5)(z1_ours, z2_ours)
    torch.autograd.backward(loss_ours)

    z1_ref = z1.clone().requires_grad_()
    z2_ref = z2.clone().requires_grad_()
    loss_ref: Tensor = SelfSupervisedLoss(PMLNTXentLoss(temperature=0.5))(z1_ref, z2_ref)
    torch.autograd.backward(loss_ref)

    assert z1_ours.grad is not None
    assert z2_ours.grad is not None
    assert z1_ref.grad is not None
    assert z2_ref.grad is not None
    torch.testing.assert_close(z1_ours.grad, z1_ref.grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(z2_ours.grad, z2_ref.grad, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("batch_size", [2, 16, 37])
def test_ntxent_poly_and_mixup_loss_dynamic_batch_sizes(batch_size: int) -> None:
    torch.manual_seed(7)
    z1 = torch.randn(batch_size, 64, requires_grad=True)
    z2 = torch.randn(batch_size, 64, requires_grad=True)
    z_aug = torch.randn(batch_size, 64, requires_grad=True)

    poly_loss: Tensor = NTXentLossPoly(temperature=0.2)(z1, z2)
    assert torch.isfinite(poly_loss)
    torch.autograd.backward(poly_loss)
    assert z1.grad is not None
    assert torch.isfinite(z1.grad).all()

    # Disjoint-batch Mixing-Up mode.
    mix_disjoint: Tensor = MixUpLoss(temperature=0.5)(z_aug, z1.detach(), z2.detach(), lam=0.35)
    assert torch.isfinite(mix_disjoint)
    torch.autograd.backward(mix_disjoint)
    assert z_aug.grad is not None
    assert torch.isfinite(z_aug.grad).all()

    # Regression C2: same-batch permutation mode avoids duplicate anchors & log(2) loss floor.
    perm = torch.roll(torch.arange(batch_size), shifts=1)
    orth = F.normalize(torch.eye(batch_size, 64), dim=1) if batch_size <= 64 else F.normalize(z1.detach(), dim=1)
    mix_same: Tensor = MixUpLoss(temperature=0.05)(orth, orth, lam=1.0, perm=perm)
    if batch_size <= 64:
        # With lam=1.0 and orthogonal anchors, loss approaches 0 (never trapped at log(2) = 0.693).
        assert mix_same.item() < 0.05


def test_coxph_loss_orders_concordant_risk_and_handles_breslow_ties() -> None:
    times = torch.tensor([5.0, 15.0, 30.0, 60.0, 90.0])
    events = torch.tensor([1.0, 1.0, 1.0, 0.0, 1.0])
    good_risk = torch.tensor([2.0, 1.0, 0.5, -0.5, -1.5], requires_grad=True)
    bad_risk = (-good_risk.detach()).clone().requires_grad_()

    criterion = CoxPHLoss()
    good_loss: Tensor = criterion(good_risk, time=times, event=events)
    bad_loss: Tensor = criterion(bad_risk, time=times, event=events)
    assert good_loss.item() < bad_loss.item()

    torch.autograd.backward(good_loss)
    assert good_risk.grad is not None
    assert torch.isfinite(good_risk.grad).all()

    # Zero-event batch returns finite zero loss without NaN.
    zero_loss: Tensor = criterion(good_risk.detach(), time=times, event=torch.zeros_like(events))
    assert zero_loss.item() == 0.0

    # Regression C3: exact Breslow tie handling is invariant to permutation of tied samples.
    t_tied = torch.tensor([10.0, 10.0, 10.0, 20.0])
    e_tied = torch.tensor([1.0, 1.0, 0.0, 1.0])
    r_tied = torch.tensor([1.2, 0.2, 0.7, -0.5])

    # Hand-computed exact Breslow NLL:
    # At t=20: risk set {3} -> -(r[3] - log(exp(r[3]))) = 0
    # At t=10: risk set {0, 1, 2, 3} -> -(r[0] - logsumexp(r)) - (r[1] - logsumexp(r))
    log_denom_10 = torch.logsumexp(r_tied, dim=0)
    expected_breslow = -((r_tied[0] - log_denom_10) + (r_tied[1] - log_denom_10)) / 3.0

    loss_order_a: Tensor = criterion(r_tied, time=t_tied, event=e_tied)
    perm = torch.tensor([2, 1, 0, 3])
    loss_order_b: Tensor = criterion(r_tied[perm], time=t_tied[perm], event=e_tied[perm])

    torch.testing.assert_close(loss_order_a, expected_breslow, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(loss_order_a, loss_order_b, rtol=1e-6, atol=1e-6)


def test_mtlr_loss_mixed_event_and_censored_gradients() -> None:
    torch.manual_seed(3)
    logits = torch.randn(16, 20, requires_grad=True)
    bins = torch.randint(0, 21, (16,))
    events = torch.tensor([True, False] * 8)

    loss: Tensor = MTLRLoss()(logits, bin_indices=bins, event=events)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0
    torch.autograd.backward(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
