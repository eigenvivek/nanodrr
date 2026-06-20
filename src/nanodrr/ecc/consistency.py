import math

import torch
from jaxtyping import Float
from torch import Tensor

from .radon import RadonIntermediate, sample_lines

# Default angle between sampled epipolar planes: 0.1 degrees.
DEFAULT_DKAPPA = math.radians(0.1)


def projection_matrix(
    k_inv: Float[Tensor, "B 3 3"],
    rt_inv: Float[Tensor, "B 4 4"],
) -> Float[Tensor, "B 3 4"]:
    """Build oriented world-to-image projection matrices from nanodrr camera params.

    nanodrr describes a camera by the inverse intrinsics `k_inv` (pixel -> camera
    ray) and the camera-to-world matrix `rt_inv`. The projection is therefore
    `P ~ K [R | t]` with `K = k_inv^{-1}` and `[R | t] = rt_inv^{-1}[:3, :4]`.
    The result is normalized so that `det(M) > 0` and `||m3|| = 1` (the oriented
    pinhole convention), which fixes the orientation used by the consistency
    metric; `source position = ker(P) = rt_inv[:3, 3]`.
    """
    K = torch.linalg.inv(k_inv)
    world_to_cam = torch.linalg.inv(rt_inv)
    P = K @ world_to_cam[..., :3, :4]
    return _normalize(P)


def epipolar_consistency(
    Ps: Float[Tensor, "B 3 4"],
    radon: RadonIntermediate,
    image_size: tuple[int, int],
    dkappa: float = DEFAULT_DKAPPA,
    reference: int | None = None,
    reference_point: Float[Tensor, "3"] | None = None,
) -> Tensor:
    """Mean epipolar-consistency metric over a set of views (Aichert et al., 2015).

    For each pair of views the t-derivative of the 2D Radon transform is sampled
    along corresponding epipolar lines; by Grangeat's theorem these are redundant
    measurements of the same plane integral, so the metric integrates their
    squared difference over the pencil of epipolar planes. It is ~0 for a
    geometrically consistent set of noise-free projections and grows as the
    projection geometry is perturbed.

    Args:
        Ps: Normalized projection matrices, one per view (see `projection_matrix`).
        radon: Radon intermediates for the matching images (same batch order).
        image_size: `(width, height)` of the projections, in pixels.
        dkappa: Angular sampling of the epipolar-plane pencil, in radians.
        reference: If given, average only over pairs containing this view index
            (the consistency of one view, used for calibration / motion estimation);
            otherwise average over all unordered pairs.
        reference_point: Optional world point to anchor the epipolar-plane pencil
            (the object isocenter; paper Algorithm 1). Defaults to the world origin.

    Returns:
        A scalar tensor: the mean squared inconsistency.
    """
    n = Ps.shape[0]
    if radon.data.shape[0] != n:
        raise ValueError("Ps and radon must have the same number of views")
    if n < 2:
        raise ValueError("need at least two views")

    Pa = _anchor(Ps, reference_point)
    centers = _camera_center(Pa)
    pinvT = _pseudo_inverse_T(Pa)
    w2, h2 = image_size[0] / 2.0, image_size[1] / 2.0

    kappa = torch.arange(0.5 * dkappa, math.pi, dkappa, device=Ps.device, dtype=Ps.dtype)
    x = torch.stack([kappa.cos(), kappa.sin()])  # (2, K)
    xm = torch.stack([-kappa.cos(), kappa.sin()])

    pairs = (
        [(reference, j) for j in range(n) if j != reference]
        if reference is not None
        else [(i, j) for i in range(n) for j in range(i + 1, n)]
    )

    total = Ps.new_zeros(())
    for i, j in pairs:
        K0, K1 = _compute_k01(centers[i], centers[j], pinvT[i], pinvT[j], w2, h2)
        vp0 = sample_lines(radon, i, K0 @ x)
        vp1 = sample_lines(radon, j, K1 @ x)
        vm0 = sample_lines(radon, i, K0 @ xm)
        vm1 = sample_lines(radon, j, K1 @ xm)
        consistency = (vp0 - vp1).square() + (vm0 - vm1).square()
        total = total + consistency.sum() * dkappa
    return total / len(pairs)


# --------------------------------------------------------------------------- internals
def _normalize(P: Float[Tensor, "... 3 4"]) -> Float[Tensor, "... 3 4"]:
    norm_m3 = torch.linalg.norm(P[..., 2, :3], dim=-1)
    sign = torch.sign(torch.linalg.det(P[..., :3, :3]))
    return P / (sign * norm_m3)[..., None, None]


def _camera_center(P: Float[Tensor, "... 3 4"]) -> Float[Tensor, "... 4"]:
    """Source position `ker(P)` as a homogeneous 4-vector with `w = +1`."""
    Vh = torch.linalg.svd(P).Vh
    C = Vh[..., -1, :]
    return C / C[..., 3:4]


def _pseudo_inverse_T(P: Float[Tensor, "... 3 4"]) -> Float[Tensor, "... 3 4"]:
    """`P^{+T} = (P P^T)^{-1} P`, which back-projects a plane to its epipolar line."""
    return torch.linalg.solve(P @ P.transpose(-1, -2), P)


def _anchor(P: Float[Tensor, "... 3 4"], reference_point) -> Float[Tensor, "... 3 4"]:
    """Move the pencil's kappa=0 plane to pass through `reference_point` (Algorithm 1)."""
    if reference_point is None:
        return P
    X0 = torch.as_tensor(reference_point, device=P.device, dtype=P.dtype).reshape(3)
    Pt = P.clone()
    Pt[..., 3] = P[..., :3] @ X0 + P[..., 3]
    return Pt


def _compute_k01(C0, C1, P0invT, P1invT, n_x2, n_y2):
    """Maps `[cos kappa, sin kappa]` to the corresponding epipolar lines in each view.

    Builds the two oriented epipolar planes spanning the pencil through the
    baseline (the plane through the origin and the plane of maximal distance to
    it), maps each to its image line via `l = P^{+T} E`, then expresses the lines
    relative to the image center. Returns two `(3, 2)` matrices `K0, K1`.
    """
    B01 = C0[0] * C1[1] - C0[1] * C1[0]
    B02 = C0[0] * C1[2] - C0[2] * C1[0]
    B03 = C0[0] * C1[3] - C0[3] * C1[0]
    B12 = C0[1] * C1[2] - C0[2] * C1[1]
    B13 = C0[1] * C1[3] - C0[3] * C1[1]
    B23 = C0[2] * C1[3] - C0[3] * C1[2]
    s2 = torch.sqrt(B12 * B12 + B02 * B02 + B01 * B01)
    s3 = torch.sqrt(B03 * B03 + B13 * B13 + B23 * B23)

    K = torch.stack(
        [
            torch.stack([B12 / s2, (-B01 * B13 - B02 * B23) / (s2 * s3)]),
            torch.stack([-B02 / s2, (B01 * B03 - B12 * B23) / (s2 * s3)]),
            torch.stack([B01 / s2, (B02 * B03 + B12 * B13) / (s2 * s3)]),
            torch.stack([torch.zeros_like(s2), -s2 / s3]),
        ]
    )  # (4, 2)

    K0 = _shift_origin_and_normalize(P0invT @ K, n_x2, n_y2)
    K1 = _shift_origin_and_normalize(P1invT @ K, n_x2, n_y2)
    return K0, K1


def _shift_origin_and_normalize(L: Float[Tensor, "3 2"], x: float, y: float) -> Float[Tensor, "3 2"]:
    """Translate lines to be relative to the image center and fix a positive scale."""
    L = L.clone()
    L[2] = L[2] + x * L[0] + y * L[1]
    s0 = torch.hypot(L[0, 0], L[1, 0])
    return L / s0
