"""Tests for the epipolar-consistency module.

A small multi-blob phantom is rendered from a few cone-beam views (the volume is
fully inside the field of view, as Grangeat's theorem requires) and we check:

  * the projection matrix matches nanodrr's camera (source == ker(P), isocenter
    projects to the principal point);
  * the redundant Radon-intermediate signals agree along corresponding epipolar
    lines (the core consistency identity);
  * the metric is minimized at the true geometry and rises sharply for an
    observable (vertical) detector shift but barely for an unobservable
    (horizontal) one.
"""

import math

import torch

from nanodrr.camera import make_k_inv, make_rt_inv
from nanodrr.data.subject import Subject
from nanodrr.drr.renderer import render
from nanodrr.ecc import epipolar_consistency, projection_matrix, radon_intermediate, sample_lines
from nanodrr.ecc.consistency import _anchor, _camera_center, _compute_k01, _pseudo_inverse_T

SIZE, SDD, SID = 160, 1020.0, 850.0


def _phantom_subject(n: int = 64, spacing: float = 2.0) -> Subject:
    """A few offset ellipsoids in a volume centered at the world origin."""
    g = (torch.arange(n, dtype=torch.float32) - (n - 1) / 2) * spacing
    X, Y, Z = torch.meshgrid(g, g, g, indexing="ij")
    R = n * spacing / 2
    vol = torch.zeros(n, n, n)
    blobs = [
        ((0.0, 0.0, 0.0), (0.6 * R, 0.5 * R, 0.7 * R), 0.2),
        ((0.2 * R, 0.0, 0.1 * R), (0.22 * R, 0.18 * R, 0.2 * R), 0.9),
        ((-0.2 * R, 0.2 * R, -0.1 * R), (0.16 * R, 0.24 * R, 0.16 * R), 0.7),
    ]
    for (cx, cy, cz), (ax, ay, az), v in blobs:
        vol += v * (((X - cx) / ax) ** 2 + ((Y - cy) / ay) ** 2 + ((Z - cz) / az) ** 2 <= 1)

    image = vol[None, None]
    voxel_to_world = torch.eye(4)
    voxel_to_world[0, 0] = voxel_to_world[1, 1] = voxel_to_world[2, 2] = spacing
    voxel_to_world[:3, 3] = -(n - 1) / 2 * spacing
    return Subject(
        imagedata=image,
        labeldata=torch.zeros_like(image),
        voxel_to_world=voxel_to_world,
        world_to_voxel=torch.linalg.inv(voxel_to_world),
        voxel_to_grid=Subject._make_voxel_to_grid(image.shape),
        isocenter=torch.zeros(3),
        max_label=0,
        convert_to_mu=False,
    )


def _render_views(rotations):
    subject = _phantom_subject()
    iso = subject.isocenter
    k_inv = make_k_inv(SDD, 1.8, 1.8, 0.0, 0.0, SIZE, SIZE)
    rots = torch.tensor(rotations)
    trans = torch.zeros(len(rotations), 3)
    trans[:, 1] = SID
    rt_inv = make_rt_inv(rots, trans, orientation="AP", isocenter=iso)
    imgs = torch.cat(
        [render(subject, k_inv, rt_inv[i : i + 1], torch.tensor([SDD]), SIZE, SIZE) for i in range(len(rotations))]
    )
    Ps = projection_matrix(k_inv.expand(len(rotations), -1, -1), rt_inv)
    return imgs, Ps, rt_inv, iso


def test_projection_matrix_matches_nanodrr_camera():
    _, Ps, rt_inv, iso = _render_views([[0.0, 0.0, 0.0]])
    centers = _camera_center(Ps)
    torch.testing.assert_close(centers[0, :3], rt_inv[0, :3, 3], atol=1e-2, rtol=0)
    # det(M) > 0 and ||m3|| == 1
    assert torch.det(Ps[0, :3, :3]) > 0
    torch.testing.assert_close(Ps[0, 2, :3].norm(), torch.tensor(1.0))
    # isocenter (origin) projects to the principal point (image center)
    uv = Ps[0] @ torch.tensor([0.0, 0.0, 0.0, 1.0])
    torch.testing.assert_close(uv[:2] / uv[2], torch.tensor([SIZE / 2, SIZE / 2]), atol=1e-2, rtol=0)


def test_redundant_signals_agree():
    imgs, Ps, _, iso = _render_views([[-24.0, 0, 0], [24.0, 0, 0]])
    radon = radon_intermediate(imgs, n_angles=256)
    Pa = _anchor(Ps, iso)
    K0, K1 = _compute_k01(
        _camera_center(Pa)[0],
        _camera_center(Pa)[1],
        _pseudo_inverse_T(Pa)[0],
        _pseudo_inverse_T(Pa)[1],
        SIZE / 2,
        SIZE / 2,
    )
    kap = torch.arange(0.004, math.pi, 0.008)
    x = torch.stack([kap.cos(), kap.sin()])
    r0 = sample_lines(radon, 0, K0 @ x)
    r1 = sample_lines(radon, 1, K1 @ x)
    nz = (r0 != 0) & (r1 != 0)
    a, b = r0[nz], r1[nz]
    assert nz.sum() > 10
    corr = torch.corrcoef(torch.stack([a, b]))[0, 1]
    assert corr > 0.99
    assert (a - b).norm() / a.norm() < 0.06


def test_metric_minimized_at_truth_and_anisotropic():
    imgs, Ps, _, iso = _render_views([[-20.0, 0, 0], [0.0, 0, 0], [20.0, 0, 0]])
    radon = radon_intermediate(imgs, n_angles=256)
    sz = (SIZE, SIZE)

    def shift(du, dv):
        H = torch.eye(3)
        H[0, 2], H[1, 2] = du, dv
        Pp = Ps.clone()
        Pp[0] = H @ Ps[0]
        return float(epipolar_consistency(Pp, radon, sz, reference=0, reference_point=iso))

    m_true = shift(0.0, 0.0)
    m_v = shift(0.0, 3.0)  # observable: perpendicular to ~horizontal epipolar lines
    m_u = shift(3.0, 0.0)  # unobservable: parallel to them
    assert m_v > 20 * m_true  # sharp bowl in the observable direction
    assert m_v > 20 * m_u  # strong observability anisotropy
    # symmetric minimum at the true geometry
    assert m_true < shift(0.0, 1.0) < shift(0.0, 2.0) < m_v
