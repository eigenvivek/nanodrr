import math

import pytest
import torch

from nanodrr.camera import make_k_inv, make_rt_inv
from nanodrr.data.subject import Subject
from nanodrr.drr import render
from nanodrr.recon import fdk

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def make_phantom(size: int = 32) -> Subject:
    """Two smooth, asymmetric Gaussian blobs (catches axis flips and scale errors)."""
    coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    z, y, x = torch.meshgrid(coords, coords, coords, indexing="ij")
    blob1 = 0.030 * torch.exp(-((x - 3.0) ** 2 + (y + 2.0) ** 2 + (z - 1.0) ** 2) / (2 * 5.0**2))
    blob2 = 0.015 * torch.exp(-((x + 6.0) ** 2 + (y - 4.0) ** 2 + (z + 3.0) ** 2) / (2 * 3.0**2))
    image = (blob1 + blob2).reshape(1, 1, size, size, size)

    voxel_to_world = torch.eye(4, dtype=torch.float32)
    voxel_to_world[:3, 3] = -(size - 1) / 2
    return Subject(
        imagedata=image,
        labeldata=torch.zeros_like(image),
        voxel_to_world=voxel_to_world,
        world_to_voxel=torch.linalg.inv(voxel_to_world),
        voxel_to_grid=Subject._make_voxel_to_grid(image.shape),
        isocenter=torch.zeros(3, dtype=torch.float32),
        convert_to_mu=False,
    )


def make_scan(
    n_views: int,
    device=None,
    d_si: float = 200.0,
    sdd: float = 400.0,
    det: int = 64,
    delx: float = 1.5,
    span: float = 360.0,
):
    """Circular scan about the world z-axis (detector v-axis || rotation axis)."""
    step = span / n_views
    angles = torch.arange(n_views, dtype=torch.float32) * step
    rotation = torch.stack([angles, torch.zeros(n_views), torch.zeros(n_views)], dim=-1)
    translation = torch.tensor([[0.0, d_si, 0.0]]).expand(n_views, -1)
    rt_inv = make_rt_inv(rotation, translation, orientation="AP", isocenter=torch.zeros(3)).to(device)
    k_inv = make_k_inv(sdd, delx, delx, 0.0, 0.0, det, det, device=device)
    return k_inv, rt_inv, torch.tensor([sdd], device=device), det, det, torch.deg2rad(angles).to(device)


def render_views(subject, k_inv, rt_inv, sdd, height, width, orthographic, chunk: int = 20):
    views = []
    for i in range(0, rt_inv.shape[0], chunk):
        views.append(
            render(
                subject,
                k_inv,
                rt_inv[i : i + chunk],
                sdd,
                height,
                width,
                n_samples=400,
                orthographic=orthographic,
                backend="torch",
            )
        )
    return torch.cat(views)


def _phantom_error(recon, truth, radius: float = 12.0):
    """Relative RMSE and total-intensity ratio inside a central sphere."""
    size = truth.shape[-1]
    coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    z, y, x = torch.meshgrid(coords, coords, coords, indexing="ij")
    mask = x**2 + y**2 + z**2 < radius**2
    err = (recon - truth)[0, 0][mask]
    rel_rmse = err.square().mean().sqrt() / truth[0, 0][mask].square().mean().sqrt()
    scale = recon[0, 0][mask].sum() / truth[0, 0][mask].sum()
    return rel_rmse.item(), scale.item()


@pytest.mark.parametrize("orthographic", [False, True])
def test_fdk_reconstructs_phantom(orthographic):
    subject = make_phantom()
    k_inv, rt_inv, sdd, height, width, _ = make_scan(n_views=120)
    proj = render_views(subject, k_inv, rt_inv, sdd, height, width, orthographic)

    recon = fdk(subject, proj, k_inv, rt_inv, sdd, orthographic=orthographic, backend="torch")

    truth = subject.image
    assert recon.shape == truth.shape
    rel_rmse, scale = _phantom_error(recon, truth)
    assert rel_rmse < 0.1
    assert abs(scale - 1.0) < 0.04


@pytest.mark.parametrize("direction", [1.0, -1.0])
def test_fdk_short_scan_parker(direction):
    """A 210-degree short scan reconstructs correctly once `beta` is provided,
    in either rotation direction (exercises the automatic fan-angle sign)."""
    subject = make_phantom()
    k_inv, rt_inv, sdd, height, width, beta = make_scan(n_views=70, span=210.0)
    if direction < 0:  # reverse the scan order
        rt_inv, beta = rt_inv.flip(0), beta.flip(0)
    proj = render_views(subject, k_inv, rt_inv, sdd, height, width, orthographic=False)

    recon = fdk(subject, proj, k_inv, rt_inv, sdd, beta=direction * beta, backend="torch")
    rel_rmse, scale = _phantom_error(recon, subject.image)
    assert rel_rmse < 0.04  # a flipped fan-angle sign lands at ~0.055
    assert abs(scale - 1.0) < 0.05

    # Without Parker handling the doubly-covered wedge is double-counted
    naive = fdk(subject, proj, k_inv, rt_inv, sdd, dbeta=2 * math.pi / 70, backend="torch")
    naive_rmse, _ = _phantom_error(naive, subject.image)
    assert naive_rmse > 1.5 * rel_rmse


def test_fdk_orthographic_half_scan():
    """Parallel-beam FBP over 180 degrees is complete; `beta` drops the 1/2."""
    subject = make_phantom()
    k_inv, rt_inv, sdd, height, width, beta = make_scan(n_views=90, span=180.0)
    proj = render_views(subject, k_inv, rt_inv, sdd, height, width, orthographic=True)

    recon = fdk(subject, proj, k_inv, rt_inv, sdd, beta=beta, orthographic=True, backend="torch")
    rel_rmse, scale = _phantom_error(recon, subject.image)
    assert rel_rmse < 0.1
    assert abs(scale - 1.0) < 0.04


def test_fdk_truncation_padding():
    """An object much wider than the detector FOV reconstructs its interior only
    with lateral padding; without it, truncation bias corrupts the volume."""
    size = 32
    coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    z, y, x = torch.meshgrid(coords, coords, coords, indexing="ij")
    image = (0.03 * torch.exp(-(x**2 + y**2 + z**2) / (2 * 10.0**2))).reshape(1, 1, size, size, size)
    voxel_to_world = torch.eye(4, dtype=torch.float32)
    voxel_to_world[:3, 3] = -(size - 1) / 2
    subject = Subject(
        imagedata=image,
        labeldata=torch.zeros_like(image),
        voxel_to_world=voxel_to_world,
        world_to_voxel=torch.linalg.inv(voxel_to_world),
        voxel_to_grid=Subject._make_voxel_to_grid(image.shape),
        isocenter=torch.zeros(3, dtype=torch.float32),
        convert_to_mu=False,
    )

    # 24-px detector at delx=1.5: FOV at iso is +-9 mm, the blob extends well past it
    k_inv, rt_inv, sdd, height, width, _ = make_scan(n_views=120, det=24, delx=1.5)
    proj = render_views(subject, k_inv, rt_inv, sdd, height, width, orthographic=False)

    padded = fdk(subject, proj, k_inv, rt_inv, sdd, padding=12, backend="torch")
    naive = fdk(subject, proj, k_inv, rt_inv, sdd, backend="torch")
    rel_padded, scale_padded = _phantom_error(padded, subject.image, radius=6.0)
    rel_naive, _ = _phantom_error(naive, subject.image, radius=6.0)
    assert rel_padded < 0.3 * rel_naive
    assert abs(scale_padded - 1.0) < 0.05


@cuda
def test_fdk_backends_agree():
    device = torch.device("cuda")
    subject = make_phantom().to(device)
    k_inv, rt_inv, sdd, height, width, _ = make_scan(n_views=60, device=device)
    proj = render_views(subject, k_inv, rt_inv, sdd, height, width, orthographic=False)

    ref = fdk(subject, proj, k_inv, rt_inv, sdd, backend="torch")
    out = fdk(subject, proj, k_inv, rt_inv, sdd, backend="triton")
    assert ((ref - out).abs().max() / ref.abs().max()).item() < 1e-4


def test_fdk_offset_detector():
    """A half-fan scan: the panel is displaced so a full rotation covers a FOV
    wider than the panel. Requires Wang weighting *and* symmetrization."""
    subject = make_phantom()
    n_views, det, delx, sdd, d_si = 180, 48, 1.5, 400.0, 200.0
    angles = torch.arange(n_views, dtype=torch.float32) * (360.0 / n_views)
    rotation = torch.stack([angles, torch.zeros(n_views), torch.zeros(n_views)], dim=-1)
    translation = torch.tensor([[0.0, d_si, 0.0]]).expand(n_views, -1)
    rt_inv = make_rt_inv(rotation, translation, orientation="AP", isocenter=torch.zeros(3))
    k_inv = make_k_inv(sdd, delx, delx, delx * 16.0, 0.0, det, det)  # principal ray off-center
    sdd_t = torch.tensor([sdd])

    proj = render_views(subject, k_inv, rt_inv, sdd_t, det, det, orthographic=False)
    good = fdk(subject, proj, k_inv, rt_inv, sdd_t, offset_detector=True, backend="torch")
    naive = fdk(subject, proj, k_inv, rt_inv, sdd_t, backend="torch")

    rel_good, scale_good = _phantom_error(good, subject.image, radius=10.0)
    rel_naive, _ = _phantom_error(naive, subject.image, radius=10.0)
    assert rel_good < 0.12
    assert abs(scale_good - 1.0) < 0.06
    assert rel_good < 0.5 * rel_naive
