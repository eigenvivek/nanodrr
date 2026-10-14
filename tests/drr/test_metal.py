"""Metal-specific behavior: reduced-precision volumes, the bfloat16 build fallback and the kernel's
argument layout. Everything that holds for every fused backend (parity with torch, gradients, soft labels,
clipping, `torch.compile`, `auto` dispatch) is in `test_fused.py` and `test_dispatch.py`."""

import re

import pytest
import torch
from test_fused import make_camera, make_random_subject

from nanodrr.drr import _metal
from nanodrr.drr.renderer import render

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")


def _setup(n_classes=1, size=32):
    device = torch.device("mps")
    subject = make_random_subject(size=size, n_classes=n_classes).to(device)
    return (subject, *make_camera(device))


def _reduced(n_classes, dtype, device):
    """A subject with only the volume and labelmap stored in `dtype`, plus its float32 twin
    holding the same (quantized) values. Geometry stays float32, as `Subject.to` would not."""
    ref = make_random_subject(n_classes=n_classes)
    ref._image_hu = ref._image_hu.to(dtype).float()  # quantize, so both see identical samples
    low = make_random_subject(n_classes=n_classes)
    low._image_hu = ref._image_hu.to(dtype)
    low.label = low.label.to(dtype)
    return ref.to(device), low.to(device, dtype=None)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("n_classes", [1, 3])
def test_metal_reduced_precision_volume_matches_quantized_float32(dtype, n_classes):
    device = torch.device("mps")
    ref, low = _reduced(n_classes, dtype, device)
    k_inv, rt_inv, sdd, h, w = make_camera(device)

    assert low.image.dtype == dtype
    expected = render(ref, k_inv, rt_inv, sdd, h, w, n_samples=100, backend="metal")
    out = render(low, k_inv, rt_inv, sdd, h, w, n_samples=100, backend="metal")

    assert out.dtype == dtype
    # Same samples, same float32 accumulation: only the final rounding to `dtype` differs
    tol = {torch.float16: 2e-3, torch.bfloat16: 1.6e-2}[dtype]
    assert ((expected - out.float()).abs().max() / expected.abs().max()).item() < tol


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_metal_reduced_precision_volume_gradient(dtype):
    device = torch.device("mps")
    ref, low = _reduced(1, dtype, device)
    k_inv, rt_inv, sdd, h, w = make_camera(device)
    grads = []
    for subject in (ref, low):
        subject.convert_to_mu = False
        subject._image_hu.requires_grad_(True)
        render(subject, k_inv, rt_inv, sdd, h, w, n_samples=100, backend="metal").float().sum().backward()
        grads.append(subject._image_hu.grad.float())

    assert low._image_hu.grad.dtype == dtype
    assert ((grads[0] - grads[1]).abs().max() / grads[0].abs().max()).item() < 2e-2


def test_metal_reduced_precision_halves_volume_footprint():
    """Reduced-precision volumes are read in place, with no float32 copy."""
    device = torch.device("mps")
    _, low = _reduced(1, torch.float16, device)
    k_inv, rt_inv, sdd, h, w = make_camera(device)
    before = torch.mps.current_allocated_memory()
    render(low, k_inv, rt_inv, sdd, h, w, n_samples=100, backend="metal")
    torch.mps.synchronize()

    assert torch.mps.current_allocated_memory() - before < low.image.numel() * 4


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_metal_pose_gradient_with_reduced_precision_volume(dtype):
    """The kernel accumulates pose gradients in float32, so a float32 pose stays finite and accurate
    with a low-precision volume. (A float16 *pose* tensor can overflow: gradients reach ~1e5.)"""
    device = torch.device("mps")
    ref, low = _reduced(1, dtype, device)
    k_inv, rt_inv, sdd, h, w = make_camera(device)
    grads = []
    for subject in (ref, low):
        rt = rt_inv.clone().requires_grad_(True)
        render(subject, k_inv, rt, sdd, h, w, n_samples=100, backend="metal").float().sum().backward()
        grads.append(rt.grad[0, :3])

    assert torch.isfinite(grads[1]).all()
    assert ((grads[0] - grads[1]).abs().max() / grads[0].abs().max()).item() < 2e-2


def test_auto_backend_falls_back_to_torch_when_metal_lacks_dtype(monkeypatch):
    monkeypatch.setattr("nanodrr.drr._metal._SUPPORTED", ())
    subject, k_inv, rt_inv, sdd, h, w = _setup()
    auto = render(subject, k_inv, rt_inv, sdd, h, w, n_samples=64, backend="auto")
    ref = render(subject, k_inv, rt_inv, sdd, h, w, n_samples=64, backend="torch")
    torch.testing.assert_close(auto, ref, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_metal_soft_labels_reduced_precision_volume(dtype):
    device = torch.device("mps")
    ref, low = _reduced(3, dtype, device)
    k_inv, rt_inv, sdd, h, w = make_camera(device)
    expected = render(ref, k_inv, rt_inv, sdd, h, w, n_samples=100, backend="metal", soft_labels=True)
    out = render(low, k_inv, rt_inv, sdd, h, w, n_samples=100, backend="metal", soft_labels=True)

    tol = {torch.float16: 2e-3, torch.bfloat16: 1.6e-2}[dtype]
    assert ((expected - out.float()).abs().max() / expected.abs().max()).item() < tol


def test_metal_bfloat16_build_failure_keeps_float32_and_float16(monkeypatch):
    """If the OS rejects the bfloat16 kernels despite the version check, the others must still build."""
    from nanodrr.drr import _metal

    real = torch.mps.compile_shader

    def reject_bfloat16(source):
        if "bfloat" in source:
            raise RuntimeError("bfloat is not supported by this Metal language version")
        return real(source)

    monkeypatch.setattr(torch.mps, "compile_shader", reject_bfloat16)
    _metal._compiled.cache_clear()
    try:
        _, built = _metal._compiled()
        assert built == (torch.float32, torch.float16)

        with pytest.raises(TypeError, match="bfloat16"):
            _metal._kernel("raymarch_fwd", torch.zeros(1, 1, 1, dtype=torch.bfloat16, device="mps"))

        subject, k_inv, rt_inv, sdd, h, w = _setup()
        out = render(subject.half(), k_inv, rt_inv, sdd, h, w, n_samples=64, backend="metal")
        assert torch.isfinite(out).all()
    finally:
        monkeypatch.undo()
        _metal._compiled.cache_clear()  # later tests get the real library back


def test_kernel_argument_layout_matches_the_launches(monkeypatch):
    """The MSL `[[buffer(i)]]` indices must be 0..n-1 and as many as the Python launch passes."""
    launched = {}

    def fake_kernel(name, vol):
        return lambda *args, threads: launched.setdefault(name, len(args))

    monkeypatch.setattr(_metal, "_kernel", fake_kernel)
    dev = torch.device("mps")
    vol, M = torch.zeros(2, 2, 2, device=dev), torch.zeros(1, 3, 4, device=dev)
    src, tgt, step = torch.zeros(1, 1, 3, device=dev), torch.zeros(1, 4, 3, device=dev), torch.zeros(1, 4, device=dev)
    _metal._forward(vol, vol, M, src, tgt, step, 4, 1, 2, False)
    _metal._backward(torch.zeros(1, 1, 4, device=dev), vol, vol, M, src, tgt, step, 4, 1, 2, True, True, False)

    for name in ("raymarch_fwd", "raymarch_bwd"):
        signature = re.search(rf"kernel void {name}\((.*?)\)\s*\{{", _metal._KERNELS, re.DOTALL).group(1)
        indices = [int(i) for i in re.findall(r"\[\[buffer\((\d+)\)\]\]", signature)]
        assert indices == list(range(len(indices)))
        assert launched[name] == len(indices)


def test_bfloat16_volume_rejects_class_labels_it_cannot_store():
    """Labels are stored in the volume's dtype, and bfloat16 integers are exact only up to 256."""
    subject, k_inv, rt_inv, sdd, h, w = _setup(n_classes=3)
    subject._image_hu = subject._image_hu.bfloat16()
    subject.n_classes = 300

    with pytest.raises(ValueError, match="not exact"):
        render(subject, k_inv, rt_inv, sdd, h, w, n_samples=16, backend="metal")
