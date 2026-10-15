"""Metal-specific behavior: reduced-precision volumes and labelmaps, the simdgroup reduction of the pose
gradient and the kernel's argument layout. Everything that holds for every fused backend (parity with
torch, gradients, soft labels, clipping, `torch.compile`, `auto` dispatch) is in `test_fused.py` and
`test_dispatch.py`."""

import re

import pytest
import torch
from test_fused import make_camera, make_random_subject

from nanodrr.camera import make_rt_inv
from nanodrr.drr import _metal
from nanodrr.drr._fused import pick_fused_backend
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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_metal_soft_labels_reduced_precision_volume(dtype):
    device = torch.device("mps")
    ref, low = _reduced(3, dtype, device)
    k_inv, rt_inv, sdd, h, w = make_camera(device)
    expected = render(ref, k_inv, rt_inv, sdd, h, w, n_samples=100, backend="metal", soft_labels=True)
    out = render(low, k_inv, rt_inv, sdd, h, w, n_samples=100, backend="metal", soft_labels=True)

    tol = {torch.float16: 2e-3, torch.bfloat16: 1.6e-2}[dtype]
    assert ((expected - out.float()).abs().max() / expected.abs().max()).item() < tol


def test_kernel_argument_layout_matches_the_launches(monkeypatch):
    """The MSL `[[buffer(i)]]` indices must be 0..n-1 and as many as the Python launch passes."""
    launched = {}

    class FakeKernel:
        thread_execution_width, max_threads_per_threadgroup = 32, 1024

        def __init__(self, name):
            self.name = name

        def __call__(self, *args, threads, group_size=None):
            launched.setdefault(self.name, len(args))

    def fake_kernel(name, vol, lab):
        return FakeKernel(name)

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


def test_bfloat16_volume_keeps_float32_labels_beyond_its_integer_range():
    """bfloat16 integers are exact only up to 256, so float32 labels are read as float32, not converted.
    `auto` must pick metal for such a subject, and label 299 must collect exactly what label 2 did."""
    subject, k_inv, rt_inv, sdd, h, w = _setup(n_classes=3)
    subject._image_hu = subject._image_hu.bfloat16()
    subject.label = subject.label.float()
    three = render(subject, k_inv, rt_inv, sdd, h, w, n_samples=64, backend="metal")

    subject.label[subject.label == 2] = 299
    subject.n_classes = 300
    assert pick_fused_backend(subject, rt_inv, rt_inv.new_zeros(1, h * w, 3), h, w) == "metal"
    out = render(subject, k_inv, rt_inv, sdd, h, w, n_samples=64, backend="auto")

    assert three[:, 2].abs().max() > 0
    torch.testing.assert_close(out[:, [0, 1, 299]], three, rtol=0, atol=0)
    assert out[:, 2:299].abs().max() == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_metal_float32_labels_match_labels_in_the_volume_dtype(dtype):
    """A reduced-precision volume reads float32 labels (the `Subject` default) as is, with the same result
    as labels stored in the volume's dtype (`Subject.half()`)."""
    _, low = _reduced(3, dtype, torch.device("mps"))
    k_inv, rt_inv, sdd, h, w = make_camera(torch.device("mps"))
    same = render(low, k_inv, rt_inv, sdd, h, w, n_samples=64, backend="metal")
    low.label = low.label.float()
    f32 = render(low, k_inv, rt_inv, sdd, h, w, n_samples=64, backend="metal")
    torch.testing.assert_close(same, f32, rtol=0, atol=0)


def test_metal_pose_gradient_with_rays_that_do_not_fill_simdgroups():
    """The pose gradient is summed per simdgroup over a padded [B, Np] grid. With N = 15 * 13 rays per
    pose (not a multiple of the simdgroup width) and two poses, padding lanes must add nothing and no
    simdgroup may mix the poses."""
    device = torch.device("mps")
    grads = []
    for backend in ("torch", "metal"):
        subject = make_random_subject().to(device)
        k_inv, _, sdd, h, w = make_camera(device, height=15, width=13)
        rt = make_rt_inv(
            torch.tensor([[5.0, -3.0, 8.0], [25.0, 4.0, -6.0]]),
            torch.tensor([[1.0, 80.0, -2.0], [-3.0, 78.0, 5.0]]),
            orientation="AP",
            isocenter=torch.zeros(3),
        ).to(device)
        rt.requires_grad_(True)
        render(subject, k_inv, rt, sdd, h, w, n_samples=200, backend=backend).sum().backward()
        grads.append(rt.grad[:, :3].cpu())

    ref, out = grads
    assert ((ref - out).abs().max() / ref.abs().max()).item() < 1e-3
