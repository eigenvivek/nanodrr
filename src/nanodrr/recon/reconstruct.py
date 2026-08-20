import math

import torch
from jaxtyping import Float

from ..data import Subject
from .backprojector import backproject
from .filtering import displaced_detector, fdk_filter, parker_weights, unwrap_angles


def fdk(
    subject: Subject,
    proj: Float[torch.Tensor, "B C H W"],
    k_inv: Float[torch.Tensor, "B 3 3"],
    rt_inv: Float[torch.Tensor, "B 4 4"],
    sdd: Float[torch.Tensor, "B"],
    dbeta: float | Float[torch.Tensor, "B"] | None = None,
    beta: Float[torch.Tensor, "B"] | None = None,
    sid: float | Float[torch.Tensor, "B"] | None = None,
    window: str = "hann",
    padding: int = 0,
    offset_detector: bool = False,
    orthographic: bool = False,
    backend: str = "auto",
    views_per_chunk: int = 4,
) -> Float[torch.Tensor, "1 C D H W"]:
    r"""Feldkamp-Davis-Kress (FDK) reconstruction from cone-beam projections.

    Analytic filtered backprojection for a circular scan: cosine-weight and
    ramp-filter each projection (`fdk_filter`), then accumulate the
    distance-weighted voxel-driven backprojection (`backproject`),

    $$
    f(\mathbf x) = \frac{1}{2} \int_0^{2\pi}
        \frac{\mathrm{SID} \cdot \mathrm{SDD}}{z(\mathbf x, \beta)^2} \,
        q_\beta\big(u(\mathbf x), v(\mathbf x)\big) \, \mathrm d\beta \,,
    $$

    discretized with angular spacing `dbeta`. The source-to-isocenter distance
    $\mathrm{SID}$ enters through the Jacobian of the fan-to-parallel rebinning
    ($\mathrm{SID} \, \mathrm d\beta$ is the source arc length), while
    $\mathrm{SDD}$ comes from filtering in physical detector coordinates. For
    `orthographic` projections this reduces to parallel-beam filtered
    backprojection (unit weights).

    **Short scans**: pass the per-view gantry angles via `beta`. If they cover
    less than a full circle, redundancy (Parker) weights are folded into the
    projections automatically and the $\tfrac12$ above is dropped — a short
    scan measures each ray once instead of twice. With `beta` given, `dbeta`
    also defaults to the actual per-view angular spacing instead of a uniform
    full circle.

    **Truncated projections**: if the subject extends laterally beyond the
    detector, the ramp filter sees a sharp cutoff at the detector edge, which
    produces a bright rim at the FOV boundary and low-frequency cupping. Set
    `padding` to extend each detector row by that many pixels per side
    (cosine-tapered edge extension) before filtering; intrinsics are widened
    to match, so the backprojection geometry is unchanged. The taper length
    implicitly models how far the object continues past the FOV — a quarter
    to half the detector width is a good default for heavily truncated scans,
    while unnecessary padding of untruncated data biases the DC term.

    **Offset ("half-fan") detectors**: set `offset_detector=True` when the
    panel is displaced laterally so a full rotation covers a field of view
    wider than the panel. `displaced_detector` then applies Wang redundancy
    weights and symmetrizes the panel about the principal ray, both of which
    are required — weighting alone leaves everything outside the overlap
    cylinder biased high.

    Assumes a circular scan whose rotation axis is parallel to the detector's
    height axis (rays are filtered along width).

    Args:
        subject: Geometry template; the reconstruction is produced on the
            voxel grid of `subject.image` (its intensities are not read).
        proj: Cone-beam projections, e.g. the output of `render`. Units of
            [mm x volume intensity], so a volume in linear attenuation
            coefficients reconstructs back to the same units.
        k_inv: Inverse intrinsic camera matrix.
        rt_inv: Camera-to-world (inverse extrinsic) matrices, one per view,
            in scan order.
        sdd: Source-to-detector distance (mm).
        dbeta: Angular spacing between views (radians). Defaults to the
            spacing implied by `beta`, or to a uniform full scan `2 * pi / B`.
        beta: Gantry angle of each view (radians, any offset; wrapping is
            fine). Enables short-scan (Parker) handling and per-view spacing.
        sid: Source-to-isocenter distance (mm), i.e., the radius of the source
            orbit. Defaults to the per-view distance between the camera center
            and `subject.isocenter`.
        window: Ramp-filter apodization — one of `"ram-lak"` (none),
            `"shepp-logan"`, or `"hann"`.
        padding: Lateral truncation-correction padding in pixels per side.
        offset_detector: Apply half-fan (displaced-detector) handling.
        orthographic: Reconstruct from parallel-beam projections.
        backend: Backprojection backend — `"auto"`, `"torch"`, or `"triton"`.
        views_per_chunk: Number of views the torch backend processes per
            `grid_sample` call; lower values reduce peak memory.

    Returns:
        Reconstructed volume on `subject`'s voxel grid, shape `(1, C, D, H, W)`.
    """
    B, _, _, width = proj.shape
    device = proj.device

    redundancy = 0.5  # full circle measures every ray twice
    if beta is not None:
        beta = torch.as_tensor(beta, device=device, dtype=torch.float32)
        beta_u = unwrap_angles(beta.double())
        if dbeta is None:
            dbeta = torch.gradient(beta_u)[0].abs().to(torch.float32)
        span = (beta_u.max() - beta_u.min()).item()
        mean_step = float(torch.as_tensor(dbeta, dtype=torch.float64).mean())
        if span < 2.0 * math.pi - mean_step / 2.0:
            redundancy = 1.0  # short scan: each ray measured once...
            if not orthographic:
                # ...except in the overscan wedges, which Parker weights resolve
                w = parker_weights(k_inv, rt_inv, sdd, beta, width)
                proj = proj * w.to(proj.dtype)[:, None, None, :]
    elif dbeta is None:
        dbeta = 2.0 * math.pi / B
    dbeta = torch.as_tensor(dbeta, device=device, dtype=torch.float32)

    if offset_detector and not orthographic:
        proj, k_inv = displaced_detector(proj, k_inv)

    if padding > 0:
        taper = torch.cos(torch.linspace(math.pi / 2.0, 0.0, padding, device=device)) ** 2
        taper = taper.to(proj.dtype)
        left = proj[..., :1] * taper[None, None, None, :]
        right = proj[..., -1:] * taper.flip(0)[None, None, None, :]
        proj = torch.cat([left, proj, right], dim=-1)
        # Same physical principal point on a wider detector: cx -> cx + padding
        k_inv = k_inv.clone()
        k_inv[..., 0, 2] = k_inv[..., 0, 2] - padding * k_inv[..., 0, 0]

    filtered = fdk_filter(proj, k_inv, sdd, window, orthographic)

    if orthographic:
        base = torch.ones(B, device=device, dtype=torch.float32)
    else:
        if sid is None:
            camera_center = rt_inv[..., :3, 3]
            sid = (camera_center - subject.isocenter).norm(dim=-1)
        sid = torch.as_tensor(sid, device=device, dtype=torch.float32)
        base = (sid * sdd.float()).expand(B)
    view_weights = base * dbeta * redundancy

    return backproject(subject, filtered, k_inv, rt_inv, sdd, view_weights, orthographic, backend, views_per_chunk)
