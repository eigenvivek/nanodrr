import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from jaxtyping import Float


def plot_drr(
    img: Float[torch.Tensor, "B C H W"],
    title: str | None = None,
    ticks: bool = True,
    axs: matplotlib.axes.Axes | None = None,
    cmap: str = "gray",
    mask_cmap: str | matplotlib.colors.Colormap = "Set2",
    mask_n_colors: int = 7,
    interior_alpha: float = 0.3,
    edge_alpha: float = 1.0,
    edge_width: int = 1,
    **imshow_kwargs,
) -> list[matplotlib.axes.Axes]:
    """Plot a DRR image, optionally with a segmentation mask overlay."""
    axs = _plot_img(img.sum(dim=1, keepdim=True), title, ticks, axs, cmap, **imshow_kwargs)

    if img.shape[1] > 1:
        _plot_mask(
            img[:, 1:],
            axs=axs[0] if len(axs) == 1 else axs,
            mask_cmap=mask_cmap,
            mask_n_colors=mask_n_colors,
            interior_alpha=interior_alpha,
            edge_alpha=edge_alpha,
            edge_width=edge_width,
        )

    return axs


def _plot_img(
    img: Float[torch.Tensor, "B C H W"],
    title: str | None,
    ticks: bool,
    axs: matplotlib.axes.Axes | None,
    cmap: str,
    **imshow_kwargs,
) -> list[matplotlib.axes.Axes]:
    """Plot a single-channel image tensor."""
    n_imgs = len(img)
    if axs is None:
        _, axs = plt.subplots(ncols=n_imgs, figsize=(10, 5))
    if n_imgs == 1:
        axs = [axs]
    if title is None or isinstance(title, str):
        title = [title] * n_imgs

    for single_img, ax, t in zip(img, axs, title):
        ax.imshow(single_img.squeeze().cpu().detach(), cmap=cmap, **imshow_kwargs)
        _, height, width = single_img.shape
        ax.xaxis.tick_top()
        ax.set(
            xlabel=t,
            xticks=[0, width - 1],
            xticklabels=[1, width],
            yticks=[0, height - 1],
            yticklabels=[1, height],
        )
        if not ticks:
            ax.set_xticks([])
            ax.set_yticks([])

    return axs


def _plot_mask(
    img: Float[torch.Tensor, "B C H W"],
    axs: matplotlib.axes.Axes | list[matplotlib.axes.Axes],
    mask_cmap: str | matplotlib.colors.Colormap,
    mask_n_colors: int,
    interior_alpha: float,
    edge_alpha: float,
    edge_width: int,
) -> None:
    """Plot a segmentation mask overlay onto existing axes."""
    if len(img) == 1:
        axs = [axs]

    n_channels = img.shape[1]
    colors = _sample_colors(mask_cmap, n_channels, mask_n_colors)

    binary = (img > 0).float()
    kernel_size = edge_width * 2 + 1
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=img.device)

    for idx, ax in enumerate(axs):
        for jdx in range(n_channels):
            mask = binary[idx, jdx]
            if mask.sum() == 0:
                continue

            eroded = F.conv2d(mask[None, None], kernel, padding=edge_width)
            eroded = (eroded == kernel.numel()).float().squeeze()
            edge = mask - eroded

            rgba = torch.zeros(*mask.shape, 4, dtype=torch.uint8)
            rgba[..., :3] = torch.tensor(colors[jdx], dtype=torch.uint8)
            rgba[..., 3] = (
                (mask * interior_alpha * 255 + edge * (edge_alpha - interior_alpha) * 255).clamp(0, 255).to(torch.uint8)
            )
            ax.imshow(rgba.cpu())


def _sample_colors(
    cmap: str | matplotlib.colors.Colormap,
    n: int,
    n_base: int = 7,
) -> list[tuple[int, int, int]]:
    """Sample `n` RGB colors from a colormap, cycling through a base palette.

    Samples `n_base` evenly spaced colors, then assigns them round-robin
    so adjacent channels get maximally distinct colors.
    """
    colormap = matplotlib.colormaps.get_cmap(cmap)
    base = [tuple(int(c * 255) for c in colormap(i / max(n_base - 1, 1))[:3]) for i in range(n_base)]
    return [base[i % n_base] for i in range(n)]
