import pytest
import torch
import torch.nn.functional as F
from pytest import approx
from test_fused import make_camera, make_random_subject

from nanodrr.data.subject import Subject
from nanodrr.drr import backends
from nanodrr.drr.backends import _grid_sample_nearest
from nanodrr.drr.renderer import render


def make_unit_impulse_subject() -> Subject:
    """Create a 3×3×3 volume with a single central voxel set to 1.

    The voxel spacing is 1 mm and the volume is centred at the origin so that
    the central voxel has world coordinates (0, 0, 0).
    """

    # Volume: (B=1, C=1, D=3, H=3, W=3)
    image = torch.zeros(1, 1, 3, 3, 3, dtype=torch.float32)
    image[0, 0, 1, 1, 1] = 1.0

    # Dummy label map with a single class
    label = torch.zeros_like(image)

    # Map voxel indices to world coordinates (mm).
    #
    # Voxel indices i ∈ {0, 1, 2} are mapped to world coordinates:
    #     x_world = i - 1
    # so that the central voxel (i = 1) sits at the origin.
    voxel_to_world = torch.eye(4, dtype=torch.float32)
    voxel_to_world[0, 3] = -1.0
    voxel_to_world[1, 3] = -1.0
    voxel_to_world[2, 3] = -1.0

    world_to_voxel = torch.linalg.inv(voxel_to_world)

    # Use the same voxel-to-grid mapping as the main code path.
    voxel_to_grid = Subject._make_voxel_to_grid(image.shape)

    # Isocenter at the origin (centre of the volume in world space)
    isocenter = torch.zeros(3, dtype=torch.float32)

    return Subject(
        imagedata=image,
        labeldata=label,
        voxel_to_world=voxel_to_world,
        world_to_voxel=world_to_voxel,
        voxel_to_grid=voxel_to_grid,
        isocenter=isocenter,
        max_label=0,
        convert_to_mu=False,
    )


def test_single_ray_integral_equals_one(device):
    """A single ray through the central voxel should integrate to 1."""

    subject = make_unit_impulse_subject().to(device)

    # Batch size 1, single detector pixel (H=W=1 → N=1)
    B, H, W = 1, 1, 1

    # Identity intrinsics/extrinsics: camera space == world space.
    k_inv = torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0)  # (1, 3, 3)
    rt_inv = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0)  # (1, 4, 4)
    sdd = torch.tensor([1.0], dtype=torch.float32, device=device)  # Unused when src/tgt are provided

    # Cast a single ray along the x-axis from x = -1.5 mm to x = +1.5 mm,
    # passing through the central voxel at the origin.
    src = torch.tensor([[[-1.5, 0.0, 0.0]]], dtype=torch.float32, device=device)  # (1, 1, 3)
    tgt = torch.tensor([[[1.5, 0.0, 0.0]]], dtype=torch.float32, device=device)  # (1, 1, 3)

    # Use many samples so that the Riemann sum closely approximates the
    # continuous line integral through the central voxel.
    n_samples = 1001

    rendered = render(
        subject=subject,
        k_inv=k_inv,
        rt_inv=rt_inv,
        sdd=sdd,
        height=H,
        width=W,
        n_samples=n_samples,
        src=src,
        tgt=tgt,
    )

    # Output shape: (B, C, H, W) with C=1.
    value = float(rendered.squeeze())

    # The integral of the unit impulse along this ray should be 1 (within a
    # small numerical tolerance due to discretisation).
    assert value == approx(1.0, rel=1e-3, abs=1e-3)


@pytest.mark.parametrize("orthographic", [False, True])
@pytest.mark.parametrize("n_classes", [1, 3])
def test_forward_render_matches_cpu(device, n_classes, orthographic):
    """The torch backend gives the same radiograph on every device as on the CPU."""
    outs = []
    for dev in (torch.device("cpu"), device):
        subject = make_random_subject(n_classes=n_classes).to(dev)
        k_inv, rt_inv, sdd, height, width = make_camera(dev)
        outs.append(
            render(
                subject, k_inv, rt_inv, sdd, height, width, n_samples=64, orthographic=orthographic, backend="torch"
            ).cpu()
        )
    ref, out = outs
    assert out.shape == (1, n_classes, 16, 16)
    # Class routing flips at label boundaries under ~1e-6 coordinate differences
    scale = ref.abs().max()
    assert ((ref.sum(dim=1) - out.sum(dim=1)).abs().max() / scale).item() < 1e-4
    bad = ((ref - out).abs().amax(dim=1) / scale) > 1e-4
    assert bad.float().mean().item() < 1e-3


@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("padding_mode", ["zeros", "border", "reflection"])
@pytest.mark.parametrize("shape", [(5, 6, 7), (6, 7)])
def test_grid_sample_nearest_matches_torch(device, shape, padding_mode, align_corners):
    """The manual nearest sampler reproduces `F.grid_sample(mode="nearest")` in 2D and 3D."""
    torch.manual_seed(0)
    vol = torch.randint(1, 6, (2, 3, *shape)).float()
    grid = torch.rand(2, 4, 9, len(shape)) * 3.2 - 1.6  # extends past [-1, 1]
    if len(shape) == 3:
        grid = grid[:, :, :, None]
    kw = {"padding_mode": padding_mode, "align_corners": align_corners}

    # A batch-expanded volume must give the same result without B copies
    for v in (vol, vol[:1].expand(2, -1, *[-1] * len(shape))):
        ref = F.grid_sample(v, grid, mode="nearest", **kw)  # CPU reference
        out = _grid_sample_nearest(v.to(device), grid.to(device), **kw)
        torch.testing.assert_close(out.cpu(), ref, rtol=0, atol=0)


def test_grid_sample_nearest_invalid_args():
    vol = torch.zeros(1, 1, 3, 3, 3)
    with pytest.raises(ValueError, match="padding_mode"):
        _grid_sample_nearest(vol, torch.zeros(1, 2, 2, 2, 3), padding_mode="wrap")
    with pytest.raises(ValueError, match="grid"):
        _grid_sample_nearest(vol, torch.zeros(1, 2, 2, 2, 2))


def test_grid_sample_dispatch(device, monkeypatch):
    """Nearest-mode sampling is routed to the manual sampler on MPS only; bilinear never is."""
    calls = []
    real = backends._grid_sample_nearest
    monkeypatch.setattr(backends, "_grid_sample_nearest", lambda *a, **k: calls.append(1) or real(*a, **k))

    vol = torch.rand(1, 1, 4, 5, 6, device=device)
    grid = torch.rand(1, 3, 3, 3, 3, device=device) * 2 - 1
    backends._grid_sample(vol, grid, mode="bilinear")
    assert not calls
    out = backends._grid_sample(vol, grid, mode="nearest")
    assert len(calls) == (device.type == "mps")
    torch.testing.assert_close(out.cpu(), F.grid_sample(vol.cpu(), grid.cpu(), mode="nearest", align_corners=False))


def _grads(device, kind, n_classes):
    """Gradients of a weighted render w.r.t. the pose (`pose`) or the volume (`volume`)."""
    subject = make_random_subject(n_classes=n_classes).to(device)
    k_inv, rt_inv, sdd, height, width = make_camera(device)
    w = torch.randn(1, n_classes, height, width, generator=torch.Generator().manual_seed(3)).to(device)
    rt = rt_inv.clone().requires_grad_(kind == "pose")
    if kind == "volume":
        subject.convert_to_mu = False
        subject._image_hu = subject._image_hu.detach().requires_grad_(True)
    out = render(subject, k_inv, rt, sdd, height, width, n_samples=64, backend="torch")
    (out * w).sum().backward()
    # SE(3) rows only; the homogeneous row's phantom gradient is not meaningful
    return (rt.grad[0, :3] if kind == "pose" else subject._image_hu.grad).cpu()


@pytest.mark.parametrize("kind,n_classes", [("pose", 1), ("pose", 3), ("volume", 1), ("volume", 3)])
def test_torch_backend_gradients_match_cpu(request, device, kind, n_classes):
    """The torch backend is differentiable w.r.t. pose and volume, with the same gradients on every device."""
    if device.type == "mps":
        request.applymarker(
            pytest.mark.xfail(
                raises=NotImplementedError,
                strict=True,
                reason="aten::grid_sampler_3d_backward is not implemented on MPS",
            )
        )
    ref = _grads(torch.device("cpu"), kind, n_classes)
    out = _grads(device, kind, n_classes)
    assert torch.isfinite(out).all() and out.abs().max() > 0
    assert ((ref - out).abs().max() / ref.abs().max()).item() < 1e-3
