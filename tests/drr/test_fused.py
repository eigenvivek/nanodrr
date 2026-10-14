from dataclasses import dataclass

import pytest
import torch

from nanodrr.camera import make_k_inv, make_rt_inv
from nanodrr.data.subject import Subject
from nanodrr.drr.renderer import render


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


# Triton is compared with the torch backend on the same GPU; Metal and torch on a GPU with torch on the CPU
PARITY = {
    "triton-cuda": ("triton", "cuda", "torch", "cuda"),
    "metal-mps": ("metal", "mps", "torch", "cpu"),
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


def _render_both(parity, n_classes=1, orthographic=False, batched_pose=False, soft_labels=False):
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
                subject,
                k_inv,
                rt_inv,
                sdd,
                height,
                width,
                n_samples=200,
                orthographic=orthographic,
                backend=backend,
                soft_labels=soft_labels,
            ).cpu()
        )
    return outs


def _grads(backend, device, kind, n_classes=1, orthographic=False, soft_labels=False):
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
    out = render(
        subject,
        k,
        rt,
        s,
        height,
        width,
        n_samples=200,
        orthographic=orthographic,
        backend=backend,
        soft_labels=soft_labels,
    )
    (out * w).sum().backward()

    if kind == "pose":
        return [rt.grad[0, :3].cpu()]  # SE(3) rows; the homogeneous row's phantom grad differs by design
    if kind == "volume":
        return [subject._image_hu.grad.cpu()]
    return [k.grad.cpu(), s.grad.cpu()]


def _assert_grads_match(parity, kind, tol=1e-3, **kwargs):
    ref = _grads(parity.ref_backend, parity.ref_device, kind, **kwargs)
    out = _grads(parity.backend, parity.device, kind, **kwargs)
    # One scale for all inputs: clipped rays do not depend on the SDD, whose gradient is zero
    scale = max(r.abs().max() for r in ref)
    for r, o in zip(ref, out):
        assert ((r - o).abs().max() / scale).item() < tol


def _assert_render_parity(ref, out, n_classes):
    """Class routing is discontinuous at label boundaries, so ~1e-6 coordinate differences between
    backends may flip single samples between channels. The channel sum, which routing cannot change,
    must match tightly; a few single-pixel flips are tolerated (on a 16x16 render, 1 pixel is 0.4%)."""
    scale = ref.abs().max()
    if n_classes == 1:
        assert ((ref - out).abs().max() / scale).item() < 1e-4
    else:
        assert ((ref.sum(dim=1) - out.sum(dim=1)).abs().max() / scale).item() < 1e-4
        bad = ((ref - out).abs().amax(dim=1) / scale) > 1e-4
        assert bad.float().mean().item() < 2e-2


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
    # With nearest-neighbor labels, a flipped sample reroutes its gradient to another channel
    _assert_grads_match(parity, "volume", tol=1e-3 if n_classes == 1 else 3e-2, n_classes=n_classes)


def test_backend_matches_torch_intrinsics_gradients(parity):
    _assert_grads_match(parity, "intrinsics")


@pytest.mark.parametrize("n_classes", [1, 3])
def test_backend_broadcasts_pose_batch(parity, n_classes):
    """Batched rt_inv against a single shared detector (batch-1 k_inv)."""
    ref, out = _render_both(parity, n_classes, batched_pose=True)

    assert out.shape == ref.shape == (2, n_classes, 16, 16)
    _assert_render_parity(ref, out, n_classes)


@pytest.mark.parametrize("orthographic", [False, True])
def test_rays_are_clipped_to_the_volume(parity, orthographic):
    """Rays that miss the volume render exactly zero; the rest match a much denser render."""
    outs, grads = [], []
    for backend, device in parity.runs():
        subject = make_random_subject().to(device)
        _, rt_inv, sdd, height, width = make_camera(device)
        k_inv = make_k_inv(100.0, 8.0, 8.0, 0.0, 0.0, height, width, device=device)  # 128 mm detector, 32 mm volume
        rt = rt_inv.clone().requires_grad_(True)
        kwargs = {"orthographic": orthographic, "backend": backend}
        out = render(subject, k_inv, rt, sdd, height, width, n_samples=50, **kwargs)
        out.square().sum().backward()
        dense = render(subject, k_inv, rt_inv, sdd, height, width, n_samples=2000, **kwargs)

        assert (out[..., 0, 0] == 0).all() and (out == 0).float().mean() > 0.25
        assert ((out - dense).abs().max() / dense.abs().max()).item() < 2e-2
        assert torch.isfinite(rt.grad).all()
        outs.append(out.detach().cpu())
        grads.append(rt.grad[0, :3].cpu())

    _assert_render_parity(outs[0], outs[1], n_classes=1)
    assert ((grads[0] - grads[1]).abs().max() / grads[0].abs().max()).item() < 1e-3


def test_invalid_n_samples_raises(device):
    subject = make_random_subject().to(device)
    k_inv, rt_inv, sdd, height, width = make_camera(device)

    with pytest.raises(ValueError, match="n_samples"):
        render(subject, k_inv, rt_inv, sdd, height, width, n_samples=1)


def test_auto_backend_falls_back_to_torch(device):
    if device.type in ("cuda", "mps"):
        pytest.skip("auto selects a fused backend on GPUs; see test_dispatch.py")
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


@pytest.mark.parametrize("orthographic", [False, True])
@pytest.mark.parametrize("batched_pose", [False, True])
def test_backend_matches_torch_forward_soft_labels(parity, orthographic, batched_pose):
    """Soft-label routing is continuous, so every channel must match (no flipped samples)."""
    ref, out = _render_both(parity, 3, orthographic, batched_pose, soft_labels=True)
    assert ((ref - out).abs().max() / ref.abs().max()).item() < 1e-4


@pytest.mark.parametrize("kind", ["pose", "volume", "intrinsics"])
def test_backend_matches_torch_gradients_soft_labels(parity, kind):
    _assert_grads_match(parity, kind, n_classes=3, soft_labels=True)


def test_backend_batched_pose_gradients_match_torch(parity):
    """Batched poses against a shared detector reduce gradients over the broadcast dimensions."""
    grads = []
    for backend, device in parity.runs():
        subject = make_random_subject().to(device)
        k_inv, _, sdd, height, width = make_camera(device)
        rt = make_rt_inv(
            torch.tensor([[5.0, -3.0, 8.0], [25.0, 4.0, -6.0]]),
            torch.tensor([[1.0, 80.0, -2.0], [-3.0, 78.0, 5.0]]),
            orientation="AP",
            isocenter=torch.zeros(3),
        ).to(device)
        rt.requires_grad_(True)
        render(subject, k_inv, rt, sdd, height, width, n_samples=200, backend=backend).sum().backward()
        grads.append(rt.grad[:, :3].cpu())

    ref, out = grads
    assert ((ref - out).abs().max() / ref.abs().max()).item() < 1e-3


# Properties of the fused backends themselves (Triton on CUDA, Metal on MPS), whatever the reference does


def fused_setup(fused, n_classes=1, size=32):
    subject = make_random_subject(size=size, n_classes=n_classes).to(fused.device)
    return (subject, *make_camera(fused.device))


def test_fused_pose_gradients_deterministic(fused):
    """Pose gradients are reduced from per-program (or per-ray) partials, so they are bitwise stable."""
    subject, k_inv, rt_inv, sdd, h, w = fused_setup(fused)
    weights = torch.randn(1, 1, h, w, device=fused.device, generator=torch.Generator(fused.device).manual_seed(5))

    grads = []
    for _ in range(2):
        rt = rt_inv.clone().requires_grad_(True)
        out = render(subject, k_inv, rt, sdd, h, w, n_samples=200, backend=fused.backend)
        (out * weights).sum().backward()
        grads.append(rt.grad.clone())

    torch.testing.assert_close(grads[0], grads[1], rtol=0, atol=0)


@pytest.mark.parametrize("soft_labels", [False, True])
def test_fused_out_of_range_labels_are_safe(fused, soft_labels):
    """Labels >= n_classes are dropped in forward and masked in backward."""
    subject, k_inv, rt_inv, sdd, h, w = fused_setup(fused, n_classes=5)
    subject.n_classes = 3  # fewer channels than the labelmap contains
    rt = rt_inv.clone().requires_grad_(True)
    out = render(subject, k_inv, rt, sdd, h, w, n_samples=100, backend=fused.backend, soft_labels=soft_labels)
    out.sum().backward()

    assert out.shape[1] == 3
    assert torch.isfinite(out).all()
    assert torch.isfinite(rt.grad).all()


@pytest.mark.parametrize("soft_labels", [False, True])
def test_fused_backward_compiles(fused, soft_labels):
    """`torch.compile` must trace both kernel launches as opaque custom ops."""
    subject, k_inv, rt_inv, sdd, h, w = fused_setup(fused, n_classes=3)
    weights = torch.randn(1, 3, h, w, device=fused.device, generator=torch.Generator(fused.device).manual_seed(4))

    def f(rt):
        return render(subject, k_inv, rt, sdd, h, w, n_samples=100, backend=fused.backend, soft_labels=soft_labels)

    rt = rt_inv.clone().requires_grad_(True)
    (f(rt) * weights).sum().backward()

    torch._dynamo.reset()
    rt_c = rt_inv.clone().requires_grad_(True)
    (torch.compile(f, fullgraph=True)(rt_c) * weights).sum().backward()

    ref = rt.grad[0, :3]
    assert ((ref - rt_c.grad[0, :3]).abs().max() / ref.abs().max()).item() < 1e-3


def test_fused_multiclass_channels_sum_to_single_class(fused):
    """Routing only splits the integral across channels; it must not change the total."""
    subject, k_inv, rt_inv, sdd, h, w = fused_setup(fused, n_classes=3)
    multi = render(subject, k_inv, rt_inv, sdd, h, w, n_samples=100, backend=fused.backend)
    subject.n_classes = 1
    single = render(subject, k_inv, rt_inv, sdd, h, w, n_samples=100, backend=fused.backend)

    torch.testing.assert_close(multi.sum(dim=1, keepdim=True), single, rtol=1e-4, atol=1e-4)


def test_fused_soft_labels_equal_masked_volume_renders(fused):
    """Channel c of a soft-label render is the plain render of the volume masked to class c."""
    subject, k_inv, rt_inv, sdd, h, w = fused_setup(fused, n_classes=3)
    soft = render(subject, k_inv, rt_inv, sdd, h, w, n_samples=100, backend=fused.backend, soft_labels=True)

    for c in range(3):
        masked = Subject(
            imagedata=subject.image * (subject.label == c),
            labeldata=torch.zeros_like(subject.label),
            voxel_to_world=subject.voxel_to_world,
            world_to_voxel=subject.world_to_voxel,
            voxel_to_grid=subject.voxel_to_grid,
            isocenter=subject.isocenter,
            max_label=0,
            convert_to_mu=False,
        )
        ref = render(masked, k_inv, rt_inv, sdd, h, w, n_samples=100, backend=fused.backend)
        assert (soft[:, c : c + 1] - ref).abs().max().item() < 1e-4 * ref.abs().max().item()


def test_fused_soft_labels_are_ignored_for_a_single_class(fused):
    subject, k_inv, rt_inv, sdd, h, w = fused_setup(fused, n_classes=1)
    hard = render(subject, k_inv, rt_inv, sdd, h, w, n_samples=64, backend=fused.backend)
    soft = render(subject, k_inv, rt_inv, sdd, h, w, n_samples=64, backend=fused.backend, soft_labels=True)

    torch.testing.assert_close(soft, hard, rtol=0, atol=0)
