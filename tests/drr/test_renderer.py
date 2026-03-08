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


def test_single_ray_integral_equals_one():
    """A single ray through the central voxel should integrate to 1."""

    subject = make_unit_impulse_subject()

    # Batch size 1, single detector pixel (H=W=1 → N=1)
    B, H, W = 1, 1, 1

    # Identity intrinsics/extrinsics: camera space == world space.
    k_inv = torch.eye(3, dtype=torch.float32).unsqueeze(0)  # (1, 3, 3)
    rt_inv = torch.eye(4, dtype=torch.float32).unsqueeze(0)  # (1, 4, 4)
    sdd = torch.tensor([1.0], dtype=torch.float32)  # Unused when src/tgt are provided

    # Cast a single ray along the x-axis from x = -1.5 mm to x = +1.5 mm,
    # passing through the central voxel at the origin.
    src = torch.tensor([[[-1.5, 0.0, 0.0]]], dtype=torch.float32)  # (1, 1, 3)
    tgt = torch.tensor([[[1.5, 0.0, 0.0]]], dtype=torch.float32)   # (1, 1, 3)

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

