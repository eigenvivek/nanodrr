import math

import torch
import torch.nn.functional as F
from jaxtyping import Float

from ..drr.renderer import _make_tgt

_WINDOWS = ("ram-lak", "shepp-logan", "hann")


def unwrap_angles(beta: Float[torch.Tensor, "B"]) -> Float[torch.Tensor, "B"]:
    """Unwrap a sequence of angles (radians) so consecutive views differ by < pi."""
    d = torch.diff(beta)
    d = d - 2.0 * math.pi * torch.round(d / (2.0 * math.pi))
    return torch.cat([beta[:1], beta[0] + torch.cumsum(d, dim=0)])


def displaced_detector(
    proj: Float[torch.Tensor, "B C H W"],
    k_inv: Float[torch.Tensor, "B 3 3"],
    tol: float = 0.1,
) -> tuple[Float[torch.Tensor, "B C H W2"], Float[torch.Tensor, "B 3 3"]]:
    r"""Redundancy-weight and symmetrize projections from an offset detector.

    In a "half-fan" acquisition the detector is displaced laterally so that a
    full circular scan covers a much wider field of view than the panel alone.
    Rays near the principal ray are then measured twice per rotation (once on
    each side of the panel) while rays out on the wide side are measured once,
    so the doubly-measured strip must be down-weighted (Wang, Med. Phys. 2002):

    $$
    w(\gamma) = 1 \mp \sin\left(\frac{\pi}{2} \frac{\gamma}{\Gamma}\right)
        \quad \text{clamped to } [0, 2] \,,
    $$

    where $\gamma = \arctan(a / \mathrm{SDD})$ is the fan angle of a detector
    column and $\Gamma$ is the fan angle of the *short* side, i.e. the half-
    width of the overlap strip. The sign follows which side is short. Weights
    are in $[0, 2]$, so the usual full-scan $\tfrac12$ still applies.

    Weighting alone is not enough. The ramp filter is a long-tailed convolution,
    so filtering data that stops at the panel edge scatters signal *past* that
    edge, and those (mostly negative) tails still belong in the backprojection.
    This function therefore also zero-extends the detector so it is symmetric
    about the principal ray, matching RTK's `DisplacedDetectorImageFilter`,
    which doubles the panel width for the same reason. Dropping the extension
    biases everything outside the overlap cylinder high by 10-20%.

    Args:
        proj: Projections from an offset detector.
        k_inv: Inverse intrinsic matrix (per view or broadcast).
        tol: Treat the detector as centered — and return the inputs unchanged —
            when the principal ray sits within this fraction of the panel width
            of its center.

    Returns:
        proj: Weighted projections on the symmetrized (wider) detector.
        k_inv: Intrinsics for the symmetrized detector.
    """
    width = proj.shape[-1]
    device, work_dtype = proj.device, torch.promote_types(proj.dtype, torch.float32)
    k64 = k_inv.double()

    # a / SDD = K⁻¹₀₀ u + K⁻¹₀₂, so the fan angle needs no explicit spacing
    edges = torch.tensor([0.0, float(width)], device=device, dtype=torch.float64)
    g_edge = torch.atan(k64[..., 0, 0, None] * edges + k64[..., 0, 2, None])
    g_lo, g_hi = g_edge[..., 0].min(), g_edge[..., 1].max()
    if (g_lo + g_hi).abs() < tol * (g_hi - g_lo).abs():
        return proj, k_inv  # centered panel: nothing to do

    overlap = torch.minimum(-g_lo, g_hi)
    u = torch.arange(width, device=device, dtype=torch.float64) + 0.5
    gamma = torch.atan(k64[..., 0, 0, None] * u + k64[..., 0, 2, None])
    ramp = torch.sin((torch.pi / 2.0) * (gamma / overlap).clamp(-1.0, 1.0))
    w = 1.0 - ramp if -g_lo > g_hi else 1.0 + ramp
    proj = proj * w.to(work_dtype).reshape(-1, 1, 1, width).to(proj.dtype)

    # Zero-extend the short side until the panel is symmetric about the principal ray
    cx = (-k64[..., 0, 2] / k64[..., 0, 0]).mean().item()
    reach = max(cx, width - cx)
    left, right = int(math.ceil(reach - cx)), int(math.ceil(reach - (width - cx)))
    if left or right:
        proj = F.pad(proj, (left, right))
        if left:
            k_inv = k_inv.clone()
            k_inv[..., 0, 2] = k_inv[..., 0, 2] - left * k_inv[..., 0, 0]
    return proj, k_inv


def parker_weights(
    k_inv: Float[torch.Tensor, "B 3 3"],
    rt_inv: Float[torch.Tensor, "B 4 4"],
    sdd: Float[torch.Tensor, "B"],
    beta: Float[torch.Tensor, "B"],
    width: int,
) -> Float[torch.Tensor, "B W"]:
    r"""Short-scan redundancy weights (Parker, Med. Phys. 1982) for a flat detector.

    A short scan spans $\pi + 2\Gamma$ where $\Gamma$ is the overscan half-angle:
    fan rays in the doubly-measured wedges are smoothly down-weighted so each
    line integral has total weight one,

    $$
    w(\beta, \gamma) = \begin{cases}
        \sin^2\!\left(\frac{\pi}{4} \frac{\beta}{\Gamma - \gamma}\right)
            & 0 \le \beta \le 2(\Gamma - \gamma) \\
        1 & 2(\Gamma - \gamma) \le \beta \le \pi - 2\gamma \\
        \sin^2\!\left(\frac{\pi}{4} \frac{\pi + 2\Gamma - \beta}{\Gamma + \gamma}\right)
            & \pi - 2\gamma \le \beta \le \pi + 2\Gamma \,,
    \end{cases}
    $$

    with $\gamma$ the signed fan angle in the convention the formula assumes
    (parallel angle $\theta = \beta + \gamma$, conjugate rays at
    $\beta' = \beta + \pi + 2\gamma$). Working through the geometry, a ray at
    detector coordinate $a$ has parallel angle $\theta = \beta + \mathrm{const}
    - \arctan(a / \mathrm{SDD})$ when the detector $u$-axis points along the
    source's direction of motion, so
    $\gamma = -\mathrm{sign}(\hat{\mathbf x}_\mathrm{cam} \cdot \mathrm d
    \mathbf c / \mathrm d\beta) \arctan(a / \mathrm{SDD})$, with $\mathbf c$
    the camera center — the sign is derived from the poses, not guessed.

    Args:
        k_inv: Inverse intrinsic camera matrix (per view or broadcast).
        rt_inv: Camera-to-world matrices, one per view, in scan order.
        sdd: Source-to-detector distance (mm).
        beta: Gantry angle of each view (radians, any offset; wrapping is fine).
        width: Detector width in pixels.

    Returns:
        Per-view, per-column weights; all ones if the scan covers a full circle.
    """
    beta64 = unwrap_angles(beta.double())
    beta64 = (beta64 - beta64[0]).abs()  # scan-relative angle, direction-agnostic
    span = beta64.max()
    overscan = (span - math.pi) / 2.0
    if overscan <= 0 or span >= 2.0 * math.pi - 1e-3:
        return torch.ones(rt_inv.shape[0], width, device=beta.device, dtype=beta.dtype)

    # Signed fan angle of each detector column (physical coords via K^-1)
    u = torch.arange(width, device=beta.device, dtype=torch.float64) + 0.5
    a = sdd.double()[..., None] * (k_inv.double()[..., 0, 0, None] * u + k_inv.double()[..., 0, 2, None])
    gamma = torch.atan(a / sdd.double()[..., None])  # (B or 1, W)

    # Orient gamma: the parallel angle decreases as `a` advances along the
    # source's direction of motion, so gamma = -sign(x_cam . dc/dbeta) * atan(a/sdd)
    src = rt_inv[..., :3, 3].double()
    tangent = src.roll(-1, dims=0) - src.roll(1, dims=0)
    dots = (rt_inv[..., :3, 0].double() * tangent)[1:-1].sum(-1)
    gamma = -(dots.mean().sign() if dots.numel() else torch.tensor(1.0)) * gamma

    b = beta64[:, None]
    ramp_in = (math.pi / 4.0) * b / (overscan - gamma).clamp(min=1e-6)
    ramp_out = (math.pi / 4.0) * (math.pi + 2.0 * overscan - b) / (overscan + gamma).clamp(min=1e-6)
    w = torch.ones(rt_inv.shape[0], width, dtype=torch.float64, device=beta.device)
    w = torch.where(b <= 2.0 * (overscan - gamma), torch.sin(ramp_in.clamp(0.0, math.pi / 2.0)) ** 2, w)
    w = torch.where(b >= math.pi - 2.0 * gamma, torch.sin(ramp_out.clamp(0.0, math.pi / 2.0)) ** 2, w)
    return w.to(beta.dtype)


def _ramp_transfer(width: int, window: str, device: torch.device) -> tuple[Float[torch.Tensor, "F"], int]:
    r"""Frequency response of the band-limited ramp filter at unit detector spacing.

    The spatial-domain (Ram-Lak) kernel sampled at unit spacing is

    $$
    h[n] = \begin{cases}
        1/4 & n = 0 \\
        0 & n \text{ even} \\
        -1/(\pi^2 n^2) & n \text{ odd}
    \end{cases}
    $$

    Its DFT is taken over a zero-padded window of `pad >= 2 * width` samples so
    that multiplication in the frequency domain realizes *linear* (not circular)
    convolution over the detector row. Optional apodization windows trade
    resolution for noise suppression.

    Returns:
        transfer: Real `rfft` transfer function of the padded kernel.
        pad: The padded FFT length.
    """
    if window not in _WINDOWS:
        raise ValueError(f"Unknown window {window!r}; expected one of {_WINDOWS}")

    pad = 1 << (2 * width - 1).bit_length()
    n = torch.arange(pad, device=device)
    m = torch.where(n <= pad // 2, n, n - pad)  # signed offsets in FFT wrap-around layout
    h = torch.zeros(pad, dtype=torch.float64, device=device)
    h[0] = 0.25
    odd = m % 2 == 1
    h[odd] = -1.0 / (torch.pi**2 * m[odd].double() ** 2)

    transfer = torch.fft.rfft(h).real  # h is even under wrap-around, so the DFT is real
    k = torch.arange(pad // 2 + 1, device=device, dtype=torch.float64)
    if window == "shepp-logan":
        transfer = transfer * torch.sinc(k / pad)
    elif window == "hann":
        transfer = transfer * (0.5 * (1.0 + torch.cos(2.0 * torch.pi * k / pad)))
    return transfer, pad


def fdk_filter(
    proj: Float[torch.Tensor, "B C H W"],
    k_inv: Float[torch.Tensor, "B 3 3"],
    sdd: Float[torch.Tensor, "B"],
    window: str = "hann",
    orthographic: bool = False,
) -> Float[torch.Tensor, "B C H W"]:
    r"""Cosine-weight and ramp-filter projections for FDK reconstruction.

    Implements the filtering half of the FDK algorithm. Each projection is
    first weighted by the cosine of the cone angle,

    $$
    \bar p_\beta(a, b) = p_\beta(a, b) \, \frac{\mathrm{SDD}}{\sqrt{\mathrm{SDD}^2 + a^2 + b^2}} \,,
    $$

    where $(a, b)$ are physical detector coordinates (mm) relative to the
    principal point, then convolved row-wise (along the width axis) with the
    band-limited ramp kernel $h_{\Delta a}$:

    $$
    q_\beta(a, b) = \Delta a \sum_{a'} \bar p_\beta(a', b) \, h_{\Delta a}(a - a') \,.
    $$

    Since $h_{\Delta a}[n] = h_1[n] / \Delta a^2$, the convolution is evaluated
    with the unit-spacing kernel and rescaled by $1 / \Delta a$. For
    `orthographic` (parallel-beam) projections the cosine weight is identically
    one and only the ramp filter is applied.

    FDK filters along the detector axis perpendicular to the rotation axis:
    this function assumes the scan rotates about the detector's *height* axis
    and therefore filters along *width*.

    Args:
        proj: Projections to filter, e.g. the output of `render`.
        k_inv: Inverse intrinsic matrix of the acquiring camera.
        sdd: Source-to-detector distance (mm).
        window: Ramp apodization — one of `"ram-lak"` (none), `"shepp-logan"`,
            or `"hann"`.
        orthographic: If True, skip the cone-angle cosine weighting.

    Returns:
        Filtered projections with the same shape and dtype as `proj`.
    """
    *_, height, width = proj.shape
    device = proj.device
    work_dtype = torch.promote_types(proj.dtype, torch.float32)

    p = proj.to(work_dtype)
    if not orthographic:
        tgt = _make_tgt(k_inv, sdd, height, width, device, work_dtype)
        cosine = tgt[..., 2] / tgt.norm(dim=-1)
        p = p * cosine.reshape(-1, 1, height, width)

    transfer, pad = _ramp_transfer(width, window, device)
    q = torch.fft.irfft(torch.fft.rfft(p, n=pad, dim=-1) * transfer.to(work_dtype), n=pad, dim=-1)[..., :width]

    delta_a = (sdd * k_inv[..., 0, 0]).reshape(-1, 1, 1, 1).to(work_dtype)  # pixel spacing [mm], since K⁻¹₀₀ = Δa / SDD
    return (q / delta_a).to(proj.dtype)
