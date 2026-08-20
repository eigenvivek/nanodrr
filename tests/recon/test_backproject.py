import pytest
import torch

from nanodrr.camera import make_k_inv, make_rt_inv
from nanodrr.data.subject import Subject
from nanodrr.drr.renderer import _make_tgt
from nanodrr.geometry import transform_point
from nanodrr.recon import backproject

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def make_subject(size: int = 32) -> Subject:
    """An empty volume centred at the origin with 1 mm isotropic voxels."""
    image = torch.zeros(1, 1, size, size, size, dtype=torch.float32)
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


def make_camera(device=None, height: int = 16, width: int = 16, sdd: float = 100.0):
    k_inv = make_k_inv(sdd, 1.0, 1.0, 0.0, 0.0, height, width, device=device)
    rt_inv = make_rt_inv(
        torch.tensor([[5.0, -3.0, 8.0]]),
        torch.tensor([[1.0, 80.0, -2.0]]),
        orientation="AP",
        isocenter=torch.zeros(3),
    ).to(device)
    return k_inv, rt_inv, torch.tensor([sdd], device=device), height, width


def test_one_hot_pixel_backprojects_along_its_ray():
    subject = make_subject()
    k_inv, rt_inv, sdd, height, width = make_camera()
    row, col = 10, 4
    proj = torch.zeros(1, 1, height, width)
    proj[0, 0, row, col] = 1.0

    vol = backproject(subject, proj, k_inv, rt_inv, sdd, backend="torch")[0, 0]

    # World-space ray through that pixel
    src = transform_point(rt_inv, torch.zeros(1, 1, 3))[0, 0]
    tgt = _make_tgt(k_inv, sdd, height, width, torch.device("cpu"), torch.float32)
    tgt = transform_point(rt_inv, tgt)[0, row * width + col]
    direction = (tgt - src) / (tgt - src).norm()

    hits = (vol > 0.05 * vol.max()).nonzero().flip(-1).float()  # (z, y, x) -> (x, y, z) voxel indices
    world = transform_point(subject.voxel_to_world[None], hits[None])[0]
    dist = torch.linalg.cross((world - src).expand_as(world), direction.expand_as(world)).norm(dim=-1)
    assert hits.numel() > 0
    assert dist.max() < 2.0  # all significant voxels lie within ~1 detector-pixel footprint of the ray


@pytest.mark.parametrize("orthographic", [False, True])
def test_constant_image_recovers_distance_weight(orthographic):
    """Backprojecting ones gives (SDD / z)^2 for perspective and 1 for orthographic."""
    subject = make_subject()
    k_inv, rt_inv, sdd, height, width = make_camera(height=32, width=32)
    proj = torch.ones(1, 1, height, width)

    vol = backproject(subject, proj, k_inv, rt_inv, sdd, orthographic=orthographic, backend="torch")[0, 0]

    # Depth of each voxel centre in camera space, computed independently
    lo, hi = 12, 20  # central voxels, well inside the detector's shadow
    idx = torch.stack(
        torch.meshgrid(torch.arange(lo, hi), torch.arange(lo, hi), torch.arange(lo, hi), indexing="ij"),
        dim=-1,
    ).flip(-1)  # (x, y, z)
    world = transform_point(subject.voxel_to_world[None], idx.reshape(1, -1, 3).float())
    cam = transform_point(torch.linalg.inv(rt_inv.double()), world.double())
    z = cam[0, :, 2].reshape(hi - lo, hi - lo, hi - lo).float()

    expected = torch.ones_like(z) if orthographic else (sdd / z) ** 2
    assert torch.allclose(vol[lo:hi, lo:hi, lo:hi], expected, rtol=1e-4)


def test_gradcheck_torch_backend():
    subject = make_subject(size=6)
    k_inv, rt_inv, sdd, height, width = make_camera(height=7, width=5, sdd=90.0)
    proj = torch.rand(1, 1, height, width, dtype=torch.float64, requires_grad=True)

    def fn(p):
        return backproject(subject, p, k_inv.double(), rt_inv.double(), sdd.double(), backend="torch")

    assert torch.autograd.gradcheck(fn, (proj,))


@cuda
@pytest.mark.parametrize("orthographic", [False, True])
def test_triton_backend_matches_torch(orthographic):
    torch.manual_seed(0)
    device = torch.device("cuda")
    subject = make_subject().to(device)
    height, width = 24, 20
    k_inv = make_k_inv(100.0, 1.0, 1.2, 2.0, -1.0, height, width, device=device)
    rt_inv = make_rt_inv(
        torch.tensor([[5.0, -3.0, 8.0], [120.0, 4.0, -6.0], [250.0, 0.0, 2.0]]),
        torch.tensor([[1.0, 80.0, -2.0]]).expand(3, -1),
        orientation="AP",
        isocenter=torch.zeros(3),
    ).to(device)
    sdd = torch.tensor([100.0], device=device)
    proj = torch.rand(3, 1, height, width, device=device)

    ref = backproject(subject, proj, k_inv, rt_inv, sdd, orthographic=orthographic, backend="torch")
    out = backproject(subject, proj, k_inv, rt_inv, sdd, orthographic=orthographic, backend="triton")
    assert out.shape == ref.shape == (1, 1, 32, 32, 32)
    assert ((ref - out).abs().max() / ref.abs().max()).item() < 1e-4
