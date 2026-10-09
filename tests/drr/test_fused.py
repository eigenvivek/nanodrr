from dataclasses import dataclass

import pytest
import torch

from nanodrr.camera import make_k_inv, make_rt_inv
from nanodrr.data.subject import Subject
from nanodrr.drr.renderer import render

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@dataclass(frozen=True)
class Parity:
    """A backend/device under test and the torch reference it must match."""

    backend: str
    device: torch.device
    ref_backend: str
    ref_device: torch.device

    def runs(self):
        """(backend, device) of the reference, then of the implementation under test."""
        return [(self.ref_backend, self.ref_device), (self.backend, self.device)]


# Triton is compared with the torch backend on the same GPU; torch on a GPU with torch on the CPU
PARITY = {
    "triton-cuda": ("triton", "cuda", "torch", "cuda"),
    "torch-cuda": ("torch", "cuda", "torch", "cpu"),
    "torch-mps": ("torch", "mps", "torch", "cpu"),
}


@pytest.fixture(params=list(PARITY))
def parity(request) -> Parity:
    backend, device, ref_backend, ref_device = PARITY[request.param]
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("requires MPS")
    return Parity(backend, torch.device(device), ref_backend, torch.device(ref_device))


def make_random_subject(size: int = 32, n_classes: int = 1, seed: int = 0) -> Subject:
    """A random volume centred at the origin with 1 mm isotropic voxels."""
    torch.manual_seed(seed)
    image = torch.rand(1, 1, size, size, size, dtype=torch.float32)
    if n_classes > 1:
        label = (torch.rand_like(image) * n_classes).floor().clamp(max=n_classes - 1)
    else:
        label = torch.zeros_like(image)

    voxel_to_world = torch.eye(4, dtype=torch.float32)
    voxel_to_world[:3, 3] = -(size - 1) / 2
    world_to_voxel = torch.linalg.inv(voxel_to_world)
    voxel_to_grid = Subject._make_voxel_to_grid(image.shape)

    return Subject(
        imagedata=image,
        labeldata=label,
        voxel_to_world=voxel_to_world,
        world_to_voxel=world_to_voxel,
        voxel_to_grid=voxel_to_grid,
        isocenter=torch.zeros(3, dtype=torch.float32),
        max_label=n_classes - 1,
        convert_to_mu=False,
    )


def make_camera(device: torch.device, height: int = 16, width: int = 16):
    sdd = 100.0
    k_inv = make_k_inv(sdd, 1.0, 1.0, 0.0, 0.0, height, width, device=device)
    rt_inv = make_rt_inv(
        torch.tensor([[5.0, -3.0, 8.0]]),
        torch.tensor([[1.0, 80.0, -2.0]]),
        orientation="AP",
        isocenter=torch.zeros(3),
    ).to(device)
    sdd_t = torch.tensor([sdd], device=device)
    return k_inv, rt_inv, sdd_t, height, width


def _render_both(parity, n_classes=1, orthographic=False, batched_pose=False):
    """Render with the reference and with the implementation under test."""
    outs = []
    for backend, device in parity.runs():
        subject = make_random_subject(n_classes=n_classes).to(device)
        k_inv, rt_inv, sdd, height, width = make_camera(device)
        if batched_pose:  # two poses against a single shared detector (batch-1 k_inv)
            rt_inv = make_rt_inv(
                torch.tensor([[5.0, -3.0, 8.0], [25.0, 4.0, -6.0]]),
                torch.tensor([[1.0, 80.0, -2.0], [-3.0, 78.0, 5.0]]),
                orientation="AP",
                isocenter=torch.zeros(3),
            ).to(device)
        outs.append(
            render(
                subject, k_inv, rt_inv, sdd, height, width, n_samples=200, orthographic=orthographic, backend=backend
            ).cpu()
        )
    return outs


def _grads(backend, device, kind, n_classes=1, orthographic=False):
    """Gradients of a randomly weighted render w.r.t. the pose, volume, or intrinsics."""
    subject = make_random_subject(n_classes=n_classes).to(device)
    k_inv, rt_inv, sdd, height, width = make_camera(device)
    seed = {"pose": 1, "volume": 2, "intrinsics": 3}[kind]
    w = torch.randn(1, n_classes, height, width, generator=torch.Generator().manual_seed(seed)).to(device)

    rt = rt_inv.clone().requires_grad_(kind == "pose")
    k, s = k_inv.clone().requires_grad_(kind == "intrinsics"), sdd.clone().requires_grad_(kind == "intrinsics")
    if kind == "volume":
        subject.convert_to_mu = False
        subject._image_hu = subject._image_hu.detach().requires_grad_(True)
    out = render(subject, k, rt, s, height, width, n_samples=200, orthographic=orthographic, backend=backend)
    (out * w).sum().backward()

    if kind == "pose":
        return [rt.grad[0, :3].cpu()]  # SE(3) rows; the homogeneous row's phantom grad differs by design
    if kind == "volume":
        return [subject._image_hu.grad.cpu()]
    return [k.grad.cpu(), s.grad.cpu()]


def _assert_grads_match(parity, kind, **kwargs):
    ref = _grads(parity.ref_backend, parity.ref_device, kind, **kwargs)
    out = _grads(parity.backend, parity.device, kind, **kwargs)
    for r, o in zip(ref, out):
        assert ((r - o).abs().max() / r.abs().max()).item() < 1e-3


def _assert_render_parity(ref, out, n_classes):
    """Class routing is discontinuous at label boundaries, so ~1e-6 coordinate
    differences between backends may flip single samples between channels."""
    scale = ref.abs().max()
    if n_classes == 1:
        assert ((ref - out).abs().max() / scale).item() < 1e-4
    else:
        assert ((ref.sum(dim=1) - out.sum(dim=1)).abs().max() / scale).item() < 1e-4
        bad = ((ref - out).abs().amax(dim=1) / scale) > 1e-4
        assert bad.float().mean().item() < 1e-3


@pytest.mark.parametrize("orthographic", [False, True])
@pytest.mark.parametrize("n_classes", [1, 3])
def test_backend_matches_torch_forward(parity, orthographic, n_classes):
    ref, out = _render_both(parity, n_classes, orthographic)

    assert out.shape == ref.shape == (1, n_classes, 16, 16)
    _assert_render_parity(ref, out, n_classes)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("orthographic", [False, True])
def test_backend_matches_torch_pose_gradients(parity, orthographic):
    _assert_grads_match(parity, "pose", orthographic=orthographic)


@pytest.mark.parametrize("n_classes", [1, 3])
def test_backend_matches_torch_volume_gradients(parity, n_classes):
    _assert_grads_match(parity, "volume", n_classes=n_classes)


def test_backend_matches_torch_intrinsics_gradients(parity):
    _assert_grads_match(parity, "intrinsics")


@cuda
def test_triton_backend_compiles_with_gradients():
    """`torch.compile` must trace both kernel launches as opaque custom ops."""
    device = torch.device("cuda")
    subject = make_random_subject().to(device)
    k_inv, rt_inv, sdd, height, width = make_camera(device)

    torch.manual_seed(4)
    w = torch.randn(1, 1, height, width, device=device)

    def f(rt):
        return render(subject, k_inv, rt, sdd, height, width, n_samples=200, backend="triton")

    rt = rt_inv.clone().requires_grad_(True)
    (f(rt) * w).sum().backward()

    torch._dynamo.reset()
    rt_c = rt_inv.clone().requires_grad_(True)
    (torch.compile(f, fullgraph=True)(rt_c) * w).sum().backward()

    ref = rt.grad[0, :3]
    scale = ref.abs().max()
    assert ((ref - rt_c.grad[0, :3]).abs().max() / scale).item() < 1e-3


@cuda
def test_triton_pose_gradients_deterministic():
    """`gM` is reduced from per-program partials, so pose gradients are bitwise stable."""
    device = torch.device("cuda")
    subject = make_random_subject().to(device)
    k_inv, rt_inv, sdd, height, width = make_camera(device)

    torch.manual_seed(5)
    w = torch.randn(1, 1, height, width, device=device)

    grads = []
    for _ in range(2):
        rt = rt_inv.clone().requires_grad_(True)
        out = render(subject, k_inv, rt, sdd, height, width, n_samples=200, backend="triton")
        (out * w).sum().backward()
        grads.append(rt.grad.clone())

    torch.testing.assert_close(grads[0], grads[1], rtol=0, atol=0)


@pytest.mark.parametrize("n_classes", [1, 3])
def test_backend_broadcasts_pose_batch(parity, n_classes):
    """Batched rt_inv against a single shared detector (batch-1 k_inv)."""
    ref, out = _render_both(parity, n_classes, batched_pose=True)

    assert out.shape == ref.shape == (2, n_classes, 16, 16)
    _assert_render_parity(ref, out, n_classes)


@cuda
def test_triton_out_of_range_labels_are_safe():
    """Labels >= n_classes are dropped in forward and masked in backward."""
    device = torch.device("cuda")
    subject = make_random_subject(n_classes=5).to(device)
    subject.n_classes = 3  # fewer channels than the labelmap contains
    k_inv, rt_inv, sdd, height, width = make_camera(device)

    rt = rt_inv.clone().requires_grad_(True)
    out = render(subject, k_inv, rt, sdd, height, width, n_samples=200, backend="triton")
    out.sum().backward()

    assert out.shape[1] == 3
    assert torch.isfinite(out).all()
    assert torch.isfinite(rt.grad).all()


def test_invalid_n_samples_raises(device):
    subject = make_random_subject().to(device)
    k_inv, rt_inv, sdd, height, width = make_camera(device)

    with pytest.raises(ValueError, match="n_samples"):
        render(subject, k_inv, rt_inv, sdd, height, width, n_samples=1)


def test_auto_backend_falls_back_to_torch(device):
    if device.type == "cuda":
        pytest.skip("auto selects the triton backend on CUDA")
    subject = make_random_subject().to(device)
    k_inv, rt_inv, sdd, height, width = make_camera(device)

    ref = render(subject, k_inv, rt_inv, sdd, height, width, n_samples=64, backend="torch")
    out = render(subject, k_inv, rt_inv, sdd, height, width, n_samples=64, backend="auto")
    torch.testing.assert_close(ref, out, rtol=0, atol=0)


def test_unknown_backend_raises(device):
    subject = make_random_subject().to(device)
    k_inv, rt_inv, sdd, height, width = make_camera(device)

    with pytest.raises(ValueError, match="backend"):
        render(subject, k_inv, rt_inv, sdd, height, width, backend="cuda")
