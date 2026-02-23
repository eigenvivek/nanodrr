import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float


def plot_drr(
    img: Float[torch.Tensor, "B C H W"],
    mask: Bool[torch.Tensor, "B C H W"] | None = None,
    title: list[str] | None = None,
    ticks: bool = True,
    axs: list[matplotlib.axes.Axes] | None = None,
    cmap: str = "gray",
    mask_cmap: str | matplotlib.colors.Colormap = "Set2",
    mask_n_colors: int = 7,
    interior_alpha: float = 0.3,
    edge_alpha: float = 1.0,
    edge_width: int = 1,
    **imshow_kwargs,
) -> list[matplotlib.axes.Axes]:
    """Plot a batch of DRR images, optionally with a segmentation mask overlay.

    Renders each image by summing across channels, simulating X-ray intensity
    accumulation along a ray. A segmentation mask can be overlaid in two ways:
    passed explicitly via ``mask``, or derived automatically when ``img`` has
    more than one channel (where channel 0 is background and channels 1+ are
    labeled structures). These two modes are mutually exclusive.

    When a mask is rendered, channel 0 is always dropped. It is assumed to
    represent background. Each remaining channel is drawn with a distinct
    color, a translucent interior fill, and an opaque boundary edge detected
    via morphological erosion.

    Args:
        img: Batch of DRR images with shape ``(B, C, H, W)``. If ``C > 1``,
            channels 1+ are treated as binary segmentation labels and a mask
            is derived as ``img > 0``. Channel intensities are summed across
            ``C`` for display.
        mask: Explicit segmentation mask with shape ``(B, C, H, W)``, where
            channel 0 is background and channels 1+ are labeled structures.
            Mutually exclusive with a multi-channel ``img``.
        title: Per-image labels of length ``B``, rendered as x-axis titles.
            If ``None``, no labels are shown.
        ticks: Whether to display 1-indexed pixel coordinate ticks. If
            ``False``, all tick marks are hidden. Defaults to ``True``.
        axs: Pre-existing axes to plot into. Must have length ``B``. If
            ``None``, a new figure with ``B`` subplots is created.
        cmap: Colormap for the DRR image. Defaults to ``"gray"``.
        mask_cmap: Colormap used to assign colors to segmentation channels.
            Colors are sampled evenly and cycled if the number of channels
            exceeds ``mask_n_colors``. Defaults to ``"Set2"``.
        mask_n_colors: Number of evenly spaced colors to sample from
            ``mask_cmap`` before cycling. Defaults to ``7``.
        interior_alpha: Opacity of the filled mask interior, in ``[0, 1]``.
            Defaults to ``0.3``.
        edge_alpha: Opacity of the mask boundary, in ``[0, 1]``.
            Defaults to ``1.0``.
        edge_width: Boundary thickness in pixels. Controls the erosion kernel
            size as ``2 * edge_width + 1``. Defaults to ``1``.
        **imshow_kwargs: Additional keyword arguments forwarded to
            ``ax.imshow`` for the DRR image only, not the mask.

    Returns:
        List of ``Axes`` of length ``B``, one per image in the batch.

    Raises:
        ValueError: If ``img`` has more than one channel and ``mask`` is
            also provided.
    """
    if img.shape[1] > 1 and mask is not None:
        raise ValueError("Pass either a multi-channel img or an explicit mask, not both.")

    axs = _plot_img(img.sum(dim=1, keepdim=True), title, ticks, axs, cmap, **imshow_kwargs)

    if img.shape[1] > 1:
        mask = img > 0
    elif mask is None:
        return axs

    _plot_mask(
        mask[:, 1:].float(),
        axs=axs,
        mask_cmap=mask_cmap,
        mask_n_colors=mask_n_colors,
        interior_alpha=interior_alpha,
        edge_alpha=edge_alpha,
        edge_width=edge_width,
    )
    return axs


def _plot_img(
    img: Float[torch.Tensor, "B C H W"],
    title: list[str] | None,
    ticks: bool,
    axs: list[matplotlib.axes.Axes] | None,
    cmap: str,
    **imshow_kwargs,
) -> list[matplotlib.axes.Axes]:
    """Plot a single-channel image tensor, returning a list of axes."""
    B = len(img)
    if axs is None:
        _, axs = plt.subplots(ncols=B, figsize=(10, 5))
    axs = [axs] if B == 1 else list(axs)
    titles = title if title is not None else [None] * B

    for single_img, ax, t in zip(img, axs, titles):
        ax.imshow(single_img.squeeze().cpu().detach(), cmap=cmap, **imshow_kwargs)
        _, H, W = single_img.shape
        ax.xaxis.tick_top()
        if ticks:
            ax.set(
                xlabel=t,
                xticks=[0, W - 1],
                xticklabels=[1, W],
                yticks=[0, H - 1],
                yticklabels=[1, H],
            )
        else:
            ax.set(xlabel=t)
            ax.set_xticks([])
            ax.set_yticks([])

    return axs


def _plot_mask(
    binary: Float[torch.Tensor, "B C H W"],
    axs: list[matplotlib.axes.Axes],
    mask_cmap: str | matplotlib.colors.Colormap,
    mask_n_colors: int,
    interior_alpha: float,
    edge_alpha: float,
    edge_width: int,
) -> None:
    """Plot a segmentation mask overlay onto existing axes."""
    n_channels = binary.shape[1]
    colors = _sample_colors(mask_cmap, n_channels, mask_n_colors)

    kernel_size = edge_width * 2 + 1
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=binary.device)

    for ax, channels in zip(axs, binary):
        for color, channel in zip(colors, channels):
            if channel.sum() == 0:
                continue

            eroded = F.conv2d(channel[None, None], kernel, padding=edge_width)
            eroded = (eroded == kernel.numel()).float().squeeze()
            edge = channel - eroded

            rgba = torch.zeros(*channel.shape, 4, dtype=torch.uint8)
            rgba[..., :3] = torch.tensor(color, dtype=torch.uint8)
            rgba[..., 3] = ((channel * interior_alpha + edge * edge_alpha) * 255).clamp(0, 255).to(torch.uint8)
            ax.imshow(rgba.cpu())


def _sample_colors(
    cmap: str | matplotlib.colors.Colormap,
    n: int,
    n_base: int = 7,
) -> list[tuple[int, int, int]]:
    """Sample `n` RGB colors from a colormap, cycling through a base palette of `n_base`."""
    colormap = matplotlib.colormaps.get_cmap(cmap)
    base = [tuple(int(c * 255) for c in colormap(i / max(n_base - 1, 1))[:3]) for i in range(n_base)]
    return [base[i % n_base] for i in range(n)]
