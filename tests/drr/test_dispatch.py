"""How `render` chooses between the torch path and the fused backends (Triton on CUDA)."""

import pytest
import torch
from test_fused import fused_setup, make_camera, make_random_subject

from nanodrr.drr._fused import load_backend, pick_fused_backend
from nanodrr.drr.renderer import render


@pytest.mark.parametrize("soft_labels", [False, True])
def test_auto_picks_the_fused_backend(fused, soft_labels):
    subject, k_inv, rt_inv, sdd, h, w = fused_setup(fused, n_classes=3)
    kwargs = {"n_samples": 64, "soft_labels": soft_labels}
    explicit = render(subject, k_inv, rt_inv, sdd, h, w, backend=fused.backend, **kwargs)
    auto = render(subject, k_inv, rt_inv, sdd, h, w, backend="auto", **kwargs)
    torch_path = render(subject, k_inv, rt_inv, sdd, h, w, backend="torch", **kwargs)

    torch.testing.assert_close(auto, explicit, rtol=0, atol=0)  # the same kernel ran
    assert not torch.equal(auto, torch_path)  # and it was not the grid_sample path


def test_auto_compiles_with_gradients(fused):
    """`backend="auto"` (the default) must trace under `torch.compile(fullgraph=True)`.

    Choosing the backend must not call anything dynamo cannot trace, not even behind a cache wrapper.
    """
    subject, k_inv, rt_inv, sdd, h, w = fused_setup(fused, n_classes=3)
    weights = torch.randn(1, 3, h, w, device=fused.device, generator=torch.Generator(fused.device).manual_seed(0))

    def f(rt):
        return render(subject, k_inv, rt, sdd, h, w, n_samples=100, soft_labels=True)  # backend="auto"

    graphs = []

    def capture(gm, example_inputs):  # a dynamo backend that records the traced graph, then runs it
        graphs.append(gm)
        return gm.forward

    rt = rt_inv.clone().requires_grad_(True)
    (f(rt) * weights).sum().backward()

    torch._dynamo.reset()
    rt_c = rt_inv.clone().requires_grad_(True)
    out = torch.compile(f, backend=capture, fullgraph=True)(rt_c)
    (out * weights).sum().backward()

    op_name = load_backend(fused.backend).op_name
    assert any(op_name in str(node.target) for node in graphs[0].graph.nodes)  # auto chose the fused backend
    torch.testing.assert_close(out, f(rt_inv), rtol=1e-5, atol=1e-6)
    ref = rt.grad[0, :3]
    assert ((ref - rt_c.grad[0, :3]).abs().max() / ref.abs().max()).item() < 1e-3


def _auto_matches_torch(fused):
    subject, k_inv, rt_inv, sdd, h, w = fused_setup(fused)
    auto = render(subject, k_inv, rt_inv, sdd, h, w, n_samples=64, backend="auto")
    ref = render(subject, k_inv, rt_inv, sdd, h, w, n_samples=64, backend="torch")
    torch.testing.assert_close(auto, ref, rtol=0, atol=0)


def test_auto_falls_back_to_torch_for_deterministic_volume_gradients(fused):
    """The volume gradient uses atomics, so deterministic mode must route to torch."""
    subject, k_inv, rt_inv, sdd, h, w = fused_setup(fused)
    subject.convert_to_mu = False
    subject._image_hu.requires_grad_(True)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        auto = render(subject, k_inv, rt_inv, sdd, h, w, n_samples=64, backend="auto")
        ref = render(subject, k_inv, rt_inv, sdd, h, w, n_samples=64, backend="torch")
    finally:
        torch.use_deterministic_algorithms(False)

    torch.testing.assert_close(auto, ref, rtol=0, atol=0)


def test_auto_falls_back_to_torch_when_inputs_exceed_limits(fused, monkeypatch):
    monkeypatch.setattr("nanodrr.drr._fused.fused_supported", lambda *args: False)
    _auto_matches_torch(fused)


def test_auto_falls_back_to_torch_when_the_backend_is_unavailable(fused, monkeypatch):
    monkeypatch.setattr("nanodrr.drr._fused.available", lambda *args: False)
    _auto_matches_torch(fused)


def test_explicit_backend_rejects_oversized_inputs(fused, monkeypatch):
    monkeypatch.setattr("nanodrr.drr._fused.fused_supported", lambda *args: False)
    subject, k_inv, rt_inv, sdd, h, w = fused_setup(fused)

    with pytest.raises(ValueError, match="limits"):
        render(subject, k_inv, rt_inv, sdd, h, w, n_samples=64, backend=fused.backend)


def test_explicit_backend_rejects_other_devices(fused):
    subject = make_random_subject()
    k_inv, rt_inv, sdd, h, w = make_camera(torch.device("cpu"))

    with pytest.raises(ValueError, match="tensors"):
        render(subject, k_inv, rt_inv, sdd, h, w, n_samples=64, backend=fused.backend)


def test_pick_fused_backend_is_none_without_a_gpu_backend():
    subject = make_random_subject()
    _, rt_inv, _, h, w = make_camera(torch.device("cpu"))

    assert pick_fused_backend(subject, rt_inv, rt_inv.new_zeros(1, h * w, 3), h, w) is None
