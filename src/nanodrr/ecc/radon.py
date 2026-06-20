import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor


@dataclass
class RadonIntermediate:
    """Derivative of the 2D Radon transform of a batch of projection images.

    This is the quantity compared along corresponding epipolar lines in
    epipolar-consistency conditions (Grangeat's theorem). For each image the
    `data` tensor stores `d/dt rho(alpha, t)`, where `alpha in [0, pi)` is the
    angle of the line normal and `t in [-t_max, t_max]` is the signed distance of
    the line to the image center.

    Attributes:
        data: `(B, n_angles, n_bins)` Radon intermediate, one slice per image.
        t_max: Half the image diagonal in pixels (the range of `t`).
        is_derivative: Whether the t-derivative (Grangeat) filter was applied. The
            derivative is odd, `dtr(alpha + pi, t) = -dtr(alpha, -t)`, which fixes
            the sign used when folding a line's angle into `[0, pi)`.
    """

    data: Float[Tensor, "B A T"]
    t_max: float
    is_derivative: bool = True

    def to(self, *args, **kwargs) -> "RadonIntermediate":
        return RadonIntermediate(self.data.to(*args, **kwargs), self.t_max, self.is_derivative)


def radon_intermediate(
    images: Float[Tensor, "B 1 H W"],
    n_angles: int = 360,
    n_bins: int | None = None,
    n_rays: int | None = None,
    derivative: bool = True,
) -> RadonIntermediate:
    """Compute the t-derivative of the 2D Radon transform of each image.

    The Radon transform is evaluated by rotate-and-sum (sampling each line with
    bilinear interpolation, then integrating), and differentiated along `t` by
    central differences. Runs on the images' device.

    Args:
        images: Projection images, shape `(B, 1, H, W)`, `(B, H, W)` or `(H, W)`.
        n_angles: Number of angular bins over `[0, pi)`.
        n_bins: Number of signed-distance bins (default ~ image diagonal).
        n_rays: Samples taken along each line when integrating (default ~ diagonal).
        derivative: If True apply the Grangeat derivative filter in `t`.

    Returns:
        A `RadonIntermediate` holding the `(B, n_angles, n_bins)` result.
    """
    x = images
    if x.ndim == 2:
        x = x[None]
    elif x.ndim == 4:
        x = x[:, 0]
    if x.ndim != 3:
        raise ValueError(f"expected images of shape (B,1,H,W), (B,H,W) or (H,W); got {tuple(images.shape)}")
    B, H, W = x.shape
    device, dtype = x.device, x.dtype

    t_max = 0.5 * math.hypot(float(W), float(H))
    n_bins = n_bins or math.ceil(2 * t_max)
    n_rays = n_rays or math.ceil(2 * t_max)

    x4 = x[:, None]  # (B, 1, H, W)
    alphas = torch.arange(n_angles, device=device, dtype=dtype) * (math.pi / n_angles)
    ts = torch.linspace(-t_max, t_max, n_bins, device=device, dtype=dtype)
    rs = torch.linspace(-t_max, t_max, n_rays, device=device, dtype=dtype)
    dr = (2 * t_max) / (n_rays - 1)

    tt, rr = ts[:, None], rs[None, :]  # (n_bins, 1), (1, n_rays)
    sino = torch.empty(B, n_angles, n_bins, device=device, dtype=dtype)
    for k in range(n_angles):
        ca, sa = torch.cos(alphas[k]), torch.sin(alphas[k])
        # Point on the line (centered coords): p = t * n + r * d, normal n = (cos a, sin a).
        px = tt * ca - rr * sa
        py = tt * sa + rr * ca
        gx = (px + W / 2.0 - 0.5) * 2.0 / (W - 1) - 1.0
        gy = (py + H / 2.0 - 0.5) * 2.0 / (H - 1) - 1.0
        grid = torch.stack([gx, gy], dim=-1)[None].expand(B, -1, -1, -1)
        samp = F.grid_sample(x4, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        sino[:, k] = samp[:, 0].sum(dim=-1) * dr

    if derivative:
        dt = float(ts[1] - ts[0])
        sino = torch.gradient(sino, spacing=dt, dim=-1)[0]
    return RadonIntermediate(sino, float(t_max), bool(derivative))


def sample_lines(
    radon: RadonIntermediate,
    index: int,
    lines: Float[Tensor, "3 N"],
) -> Float[Tensor, "N"]:
    """Sample image `index`'s Radon intermediate at lines `l0*x + l1*y + l2 = 0`.

    Coordinates are relative to the image center. Lines whose distance leaves the
    stored range (i.e. that miss the image) return 0. The angle is folded into
    `[0, pi)` with the derivative sign flip, so the returned value corresponds to
    the line's actual orientation.
    """
    data = radon.data[index]
    n_angles, n_bins = data.shape
    l0, l1, l2 = lines

    length = torch.hypot(l0, l1)
    safe = length > 1e-12
    length_safe = torch.where(safe, length, torch.ones_like(length))

    theta = torch.atan2(l1, l0)  # (-pi, pi]
    t = -l2 / length_safe
    sign = torch.ones_like(theta)

    flip = theta < 0.0
    theta = torch.where(flip, theta + math.pi, theta)
    t = torch.where(flip, -t, t)
    if radon.is_derivative:
        sign = torch.where(flip, -sign, sign)
    at_pi = theta >= math.pi - 1e-6
    theta = torch.where(at_pi, theta - math.pi, theta)
    t = torch.where(at_pi, -t, t)
    if radon.is_derivative:
        sign = torch.where(at_pi, -sign, sign)

    fa = theta / math.pi * n_angles
    a0 = torch.floor(fa).long()
    wa = fa - a0
    a0m = a0 % n_angles
    a1m = (a0 + 1) % n_angles

    ft = (t + radon.t_max) / (2.0 * radon.t_max) * (n_bins - 1)
    inside = (ft >= 0.0) & (ft <= n_bins - 1)
    ftc = ft.clamp(0.0, n_bins - 1)
    t0 = torch.floor(ftc).long()
    t1 = torch.clamp(t0 + 1, max=n_bins - 1)
    wt = ftc - t0

    v0 = data[a0m, t0] * (1 - wt) + data[a0m, t1] * wt
    v1 = data[a1m, t0] * (1 - wt) + data[a1m, t1] * wt
    val = sign * (v0 * (1 - wa) + v1 * wa)
    return torch.where(safe & inside, val, torch.zeros_like(val))
