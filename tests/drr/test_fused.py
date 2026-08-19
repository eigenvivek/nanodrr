import pytest
import torch

from nanodrr.camera import make_k_inv, make_rt_inv
from nanodrr.data.subject import Subject
from nanodrr.drr.renderer import render

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


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


def _both_backends(subject, k_inv, rt_inv, sdd, height, width, **kwargs):
    ref = render(subject, k_inv, rt_inv, sdd, height, width, backend="torch", **kwargs)
    out = render(subject, k_inv, rt_inv, sdd, height, width, backend="triton", **kwargs)
    return ref, out


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


@cuda
@pytest.mark.parametrize("orthographic", [False, True])
@pytest.mark.parametrize("n_classes", [1, 3])
def test_triton_backend_matches_torch_forward(orthographic, n_classes):
    device = torch.device("cuda")
    subject = make_random_subject(n_classes=n_classes).to(device)
    k_inv, rt_inv, sdd, height, width = make_camera(device)

    ref, out = _both_backends(subject, k_inv, rt_inv, sdd, height, width, n_samples=200, orthographic=orthographic)

    assert out.shape == ref.shape == (1, n_classes, height, width)
    _assert_render_parity(ref, out, n_classes)


@cuda
@pytest.mark.parametrize("orthographic", [False, True])
def test_triton_backend_matches_torch_pose_gradients(orthographic):
    device = torch.device("cuda")
    subject = make_random_subject().to(device)
    k_inv, rt_inv, sdd, height, width = make_camera(device)

    torch.manual_seed(1)
    w = torch.randn(1, 1, height, width, device=device)

    grads = []
    for backend in ("torch", "triton"):
        rt = rt_inv.clone().requires_grad_(True)
        out = render(subject, k_inv, rt, sdd, height, width, n_samples=200, orthographic=orthographic, backend=backend)
        (out * w).sum().backward()
        grads.append(rt.grad[0, :3])  # SE(3) rows; the homogeneous row's phantom grad differs by design

    scale = grads[0].abs().max()
    assert ((grads[0] - grads[1]).abs().max() / scale).item() < 1e-3


@cuda
@pytest.mark.parametrize("n_classes", [1, 3])
def test_triton_backend_matches_torch_volume_gradients(n_classes):
    device = torch.device("cuda")
    subject = make_random_subject(n_classes=n_classes).to(device)
    k_inv, rt_inv, sdd, height, width = make_camera(device)

    torch.manual_seed(2)
    w = torch.randn(1, n_classes, height, width, device=device)

    grads = []
    for backend in ("torch", "triton"):
        subject.convert_to_mu = False
        subject._image_hu = subject._image_hu.detach().requires_grad_(True)
        out = render(subject, k_inv, rt_inv, sdd, height, width, n_samples=200, backend=backend)
        (out * w).sum().backward()
        grads.append(subject._image_hu.grad.clone())

    scale = grads[0].abs().max()
    assert ((grads[0] - grads[1]).abs().max() / scale).item() < 1e-3


@cuda
def test_triton_backend_matches_torch_intrinsics_gradients():
    device = torch.device("cuda")
    subject = make_random_subject().to(device)
    k_inv, rt_inv, sdd, height, width = make_camera(device)

    torch.manual_seed(3)
    w = torch.randn(1, 1, height, width, device=device)

    grads = []
    for backend in ("torch", "triton"):
        k = k_inv.clone().requires_grad_(True)
        s = sdd.clone().requires_grad_(True)
        out = render(subject, k, rt_inv, s, height, width, n_samples=200, backend=backend)
        (out * w).sum().backward()
        grads.append((k.grad.clone(), s.grad.clone()))

    for ref, out in zip(grads[0], grads[1]):
        scale = ref.abs().max()
        assert ((ref - out).abs().max() / scale).item() < 1e-3


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


@cuda
@pytest.mark.parametrize("n_classes", [1, 3])
def test_triton_backend_broadcasts_pose_batch(n_classes):
    """Batched rt_inv against a single shared detector (batch-1 k_inv)."""
    device = torch.device("cuda")
    subject = make_random_subject(n_classes=n_classes).to(device)
    k_inv, _, sdd, height, width = make_camera(device)
    rt_inv = make_rt_inv(
        torch.tensor([[5.0, -3.0, 8.0], [25.0, 4.0, -6.0]]),
        torch.tensor([[1.0, 80.0, -2.0], [-3.0, 78.0, 5.0]]),
        orientation="AP",
        isocenter=torch.zeros(3),
    ).to(device)

    ref, out = _both_backends(subject, k_inv, rt_inv, sdd, height, width, n_samples=200)

    assert out.shape == ref.shape == (2, n_classes, height, width)
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


def test_invalid_n_samples_raises():
    subject = make_random_subject()
    k_inv, rt_inv, sdd, height, width = make_camera(torch.device("cpu"))

    with pytest.raises(ValueError, match="n_samples"):
        render(subject, k_inv, rt_inv, sdd, height, width, n_samples=1)


def test_auto_backend_works_on_cpu():
    subject = make_random_subject()
    k_inv, rt_inv, sdd, height, width = make_camera(torch.device("cpu"))

    ref = render(subject, k_inv, rt_inv, sdd, height, width, n_samples=64, backend="torch")
    out = render(subject, k_inv, rt_inv, sdd, height, width, n_samples=64, backend="auto")
    torch.testing.assert_close(ref, out, rtol=0, atol=0)


def test_unknown_backend_raises():
    subject = make_random_subject()
    k_inv, rt_inv, sdd, height, width = make_camera(torch.device("cpu"))

    with pytest.raises(ValueError, match="backend"):
        render(subject, k_inv, rt_inv, sdd, height, width, backend="cuda")
