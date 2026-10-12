import pytest
import torch
from pytest import approx

from nanodrr.data.subject import Subject
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
    _, H, W = 1, 1, 1

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


def make_labeled_subject(size: int = 24, n_classes: int = 4, seed: int = 0) -> Subject:
    """A random volume with random blocky labels, centred at the origin with 1 mm voxels."""
    g = torch.Generator().manual_seed(seed)
    image = torch.rand(1, 1, size, size, size, generator=g)
    blocks = torch.randint(0, n_classes, (1, 1, size // 4, size // 4, size // 4), generator=g).float()
    label = torch.nn.functional.interpolate(blocks, size=(size, size, size), mode="nearest")
    return _wrap(image, label, n_classes - 1)


def _wrap(image: torch.Tensor, label: torch.Tensor, max_label: int) -> Subject:
    size = image.shape[-1]
    voxel_to_world = torch.eye(4, dtype=torch.float32)
    voxel_to_world[:3, 3] = -(size - 1) / 2
    return Subject(
        imagedata=image,
        labeldata=label,
        voxel_to_world=voxel_to_world,
        world_to_voxel=torch.linalg.inv(voxel_to_world),
        voxel_to_grid=Subject._make_voxel_to_grid(image.shape),
        isocenter=torch.zeros(3, dtype=torch.float32),
        max_label=max_label,
        convert_to_mu=False,
    )


def _camera(device, batch: int = 1):
    from nanodrr.camera import make_k_inv, make_rt_inv

    sdd, height, width = 100.0, 16, 12
    k_inv = make_k_inv(sdd, 1.0, 1.0, 0.0, 0.0, height, width, device=device)
    rot = torch.tensor([[5.0, -3.0, 8.0], [25.0, 4.0, -6.0]])[:batch]
    xyz = torch.tensor([[1.0, 80.0, -2.0], [-3.0, 78.0, 5.0]])[:batch]
    rt_inv = make_rt_inv(rot, xyz, orientation="AP", isocenter=torch.zeros(3)).to(device)
    return k_inv, rt_inv, torch.tensor([sdd], device=device), height, width


@pytest.mark.parametrize("orthographic", [False, True])
@pytest.mark.parametrize("batch", [1, 2])
def test_soft_labels_equal_masked_volume_renders(device, orthographic, batch):
    """Channel c of a soft-label render is the plain render of the volume masked to class c."""
    n_classes = 4
    subject = make_labeled_subject(n_classes=n_classes)
    k_inv, rt_inv, sdd, height, width = _camera(device, batch)
    kwargs = {"n_samples": 200, "orthographic": orthographic, "backend": "torch"}

    soft = render(subject.to(device), k_inv, rt_inv, sdd, height, width, soft_labels=True, **kwargs)
    assert soft.shape == (batch, n_classes, height, width)

    for c in range(n_classes):
        masked = _wrap(subject.image * (subject.label == c), torch.zeros_like(subject.label), 0).to(device)
        ref = render(masked, k_inv, rt_inv, sdd, height, width, **kwargs)
        assert (soft[:, c : c + 1] - ref).abs().max().item() < 1e-4 * ref.abs().max().item()


def test_soft_labels_pose_gradients_match_masked_volume(device):
    """Per-structure channels are differentiable w.r.t. the pose, with the masked render's gradient."""
    subject = make_labeled_subject()
    k_inv, rt_inv, sdd, height, width = _camera(device)
    c = 2
    masked = _wrap(subject.image * (subject.label == c), torch.zeros_like(subject.label), 0).to(device)

    grads = []
    for subj, soft_labels, channel in ((subject.to(device), True, c), (masked, False, 0)):
        rt = rt_inv.clone().requires_grad_()
        out = render(subj, k_inv, rt, sdd, height, width, n_samples=200, backend="torch", soft_labels=soft_labels)
        out[:, channel].square().sum().backward()
        grads.append(rt.grad[0, :3])
    assert ((grads[0] - grads[1]).abs().max() / grads[1].abs().max()).item() < 1e-3


def test_soft_labels_volume_gradients_match_nearest_channel_sum(device):
    """The channel sum is the whole-volume render, so its volume gradient ignores the label mode."""
    k_inv, rt_inv, sdd, height, width = _camera(device)
    grads = []
    for soft_labels in (True, False):
        subject = make_labeled_subject().to(device)
        subject._image_hu = subject._image_hu.detach().requires_grad_(True)
        out = render(
            subject, k_inv, rt_inv, sdd, height, width, n_samples=200, backend="torch", soft_labels=soft_labels
        )
        out.sum().backward()
        grads.append(subject._image_hu.grad)
    assert ((grads[0] - grads[1]).abs().max() / grads[1].abs().max()).item() < 1e-4
