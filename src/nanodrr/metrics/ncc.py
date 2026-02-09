import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from torch import Tensor


class NormalizedCrossCorrelation2d(torch.nn.Module):
    """Compute Normalized Cross Correlation between two batches of images."""

    def __init__(self, patch_size: int | None = None, eps: float = 1e-4):
        super().__init__()
        self.patch_size = patch_size
        self.eps = eps

    def forward(
        self,
        x1: Float[Tensor, "B 1 H W"],
        x2: Float[Tensor, "B 1 H W"],
    ) -> Float[Tensor, "B"]:
        if self.patch_size is not None:
            x1 = _to_patches(x1, self.patch_size)
            x2 = _to_patches(x2, self.patch_size)
        assert x1.shape == x2.shape, "Input images must be the same size"
        _, c, h, w = x1.shape
        x1, x2 = self._norm(x1), self._norm(x2)
        score = (x1 * x2).sum(dim=(1, 2, 3))
        score /= c * h * w
        return score

    def _norm(
        self,
        x: Float[Tensor, "B C H W"],
    ) -> Float[Tensor, "B C H W"]:
        mu = x.mean(dim=(-1, -2), keepdim=True)
        var = x.var(dim=(-1, -2), keepdim=True, correction=0) + self.eps
        return (x - mu) / var.sqrt()


class MultiscaleNormalizedCrossCorrelation2d(torch.nn.Module):
    """Compute Normalized Cross Correlation between two batches of images at multiple scales."""

    def __init__(
        self,
        patch_sizes: list[int | None] = [None],
        patch_weights: list[float] = [1.0],
        eps: float = 1e-4,
    ):
        super().__init__()
        assert len(patch_sizes) == len(patch_weights), "Each scale must have a weight"
        self.nccs = torch.nn.ModuleList([NormalizedCrossCorrelation2d(patch_size, eps) for patch_size in patch_sizes])
        self.patch_weights = patch_weights

    def forward(
        self,
        x1: Float[Tensor, "B 1 H W"],
        x2: Float[Tensor, "B 1 H W"],
    ) -> Float[Tensor, "B"]:
        scores = []
        for weight, ncc in zip(self.patch_weights, self.nccs):
            scores.append(weight * ncc(x1, x2))
        return torch.stack(scores, dim=0).sum(dim=0)


class GradientNormalizedCrossCorrelation2d(NormalizedCrossCorrelation2d):
    """Compute Normalized Cross Correlation between the image gradients of two batches of images."""

    def __init__(
        self,
        patch_size: int | None = None,
        sigma: float = 1.0,
        **kwargs,
    ):
        super().__init__(patch_size, **kwargs)
        self.sobel = _Sobel(sigma)

    def forward(
        self,
        x1: Float[Tensor, "B 1 H W"],
        x2: Float[Tensor, "B 1 H W"],
    ) -> Float[Tensor, "B"]:
        return super().forward(self.sobel(x1), self.sobel(x2))


def _to_patches(
    x: Float[Tensor, "B 1 H W"],
    patch_size: int,
) -> Float[Tensor, "B patch_size_sq H_out W_out"]:
    x = x.unfold(2, patch_size, step=1).unfold(3, patch_size, step=1).contiguous()
    return rearrange(x, "b c p1 p2 h w -> b (c p1 p2) h w")


def _make_gaussian_kernel_1d(
    sigma: float,
    kernel_size: int,
) -> Float[Tensor, "K"]:
    """Create a 1D Gaussian kernel."""
    x = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


class _Sobel(torch.nn.Module):
    def __init__(self, sigma: float):
        super().__init__()
        self.sigma = sigma

        Gx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
        Gy = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)
        self.register_buffer("sobel_weight", torch.stack([Gx, Gy]).unsqueeze(1))

        if self.sigma > 0:
            kernel_size: int = int(6 * sigma + 1) | 1
            kernel_1d = _make_gaussian_kernel_1d(sigma, kernel_size)
            self.register_buffer("gauss_h", kernel_1d.reshape(1, 1, 1, -1))
            self.register_buffer("gauss_v", kernel_1d.reshape(1, 1, -1, 1))
            self.gauss_pad = kernel_size // 2
        else:
            self.gauss_pad = 0

    def forward(
        self,
        img: Float[Tensor, "B 1 H W"],
    ) -> Float[Tensor, "B 2 H W"]:
        x = img.float()

        if self.sigma > 0:
            x = F.pad(x, [self.gauss_pad] * 4, mode="reflect")
            x = F.conv2d(x, self.gauss_h)
            x = F.conv2d(x, self.gauss_v)

        x = F.conv2d(x, self.sobel_weight, padding=1)
        return x
