import cv2
import matplotlib.axes
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
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
    passed explicitly via `mask`, or derived automatically when `img` has
    more than one channel (where channel 0 is background and channels 1+ are
    labeled structures). These two modes are mutually exclusive.

    When a mask is rendered, channel 0 is always dropped. It is assumed to
    represent background. Each remaining channel is drawn with a distinct
    color, a translucent interior fill, and an opaque boundary edge detected
    via morphological erosion.

    Args:
        img: Batch of DRR images with shape `(B, C, H, W)`. If `C > 1`,
            channels 1+ are treated as binary segmentation labels and a mask
            is derived as `img > 0`. Channel intensities are summed across
            `C` for display.
        mask: Explicit segmentation mask with shape `(B, C, H, W)`, where
            channel 0 is background and channels 1+ are labeled structures.
            Mutually exclusive with a multi-channel `img`.
        title: Per-image labels of length `B`, rendered as x-axis titles.
            If `None`, no labels are shown.
        ticks: Whether to display 1-indexed pixel coordinate ticks. If
            `False`, all tick marks are hidden. Defaults to `True`.
        axs: Pre-existing axes to plot into. Must have length `B`. If
            `None`, a new figure with `B` subplots is created.
        cmap: Colormap for the DRR image. Defaults to `"gray"`.
        mask_cmap: Colormap used to assign colors to segmentation channels.
            Colors are sampled evenly and cycled if the number of channels
            exceeds `mask_n_colors`. Defaults to `"Set2"`.
        mask_n_colors: Number of evenly spaced colors to sample from
            `mask_cmap` before cycling. Defaults to `7`.
        interior_alpha: Opacity of the filled mask interior, in `[0, 1]`.
            Defaults to `0.3`.
        edge_alpha: Opacity of the mask boundary, in `[0, 1]`.
            Defaults to `1.0`.
        edge_width: Boundary thickness in pixels. Controls the erosion kernel
            size as `2 * edge_width + 1`. Defaults to `1`.
        **imshow_kwargs: Additional keyword arguments forwarded to
            `ax.imshow` for the DRR image only, not the mask.

    Returns:
        List of `Axes` of length `B`, one per image in the batch.

    Raises:
        ValueError: If `img` has more than one channel and `mask` is
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


def overlay(
    moving: Float[torch.Tensor, "B C H W"],
    fixed: Float[torch.Tensor, "B C H W"],
    title: list[str] | None = None,
    ticks: bool = True,
    axs: list[matplotlib.axes.Axes] | None = None,
    blur_kernel: int = 3,
    canny_low: int = 0,
    canny_high: int = 100,
    edge_color: tuple[float, float, float] = (1.0, 0.0, 0.0),
    edge_alpha: float = 1.0,
    edge_detection_size: int = 200,
) -> list[matplotlib.axes.Axes]:
    """Overlay moving image edges on fixed images for registration assessment.

    Edges are detected using Canny at a fixed resolution for threshold consistency,
    then upscaled with bilinear interpolation for anti-aliased rendering.

    Args:
        moving: Moving images, shape (B, C, H, W)
        fixed: Fixed images, shape (B, C, H, W)
        title: Optional titles for each image in batch
        ticks: Whether to show pixel coordinate ticks
        axs: Optional pre-existing axes to plot on
        blur_kernel: Gaussian blur kernel size (must be odd)
        canny_low: Canny lower threshold
        canny_high: Canny upper threshold
        edge_color: RGB color tuple for edges, values in [0, 1]
        edge_alpha: Edge opacity in [0, 1]
        edge_detection_size: Resolution for Canny detection

    Returns:
        List of matplotlib Axes objects

    Raises:
        ValueError: If input shapes don't match or parameters are invalid
    """
    if moving.ndim != 4:
        raise ValueError(f"Expected 4D tensors (B, C, H, W), got {moving.ndim}D")
    if title is not None and len(title) != moving.shape[0]:
        raise ValueError(f"Title length {len(title)} != batch size {moving.shape[0]}")
    if blur_kernel % 2 == 0 or blur_kernel < 1:
        raise ValueError(f"blur_kernel must be positive odd integer, got {blur_kernel}")
    if not 0 <= edge_alpha <= 1:
        raise ValueError(f"edge_alpha must be in [0, 1], got {edge_alpha}")

    fixed_gray = fixed.sum(dim=1, keepdim=True)
    moving_gray = moving.sum(dim=1)

    axs = _plot_img(fixed_gray, title, ticks, axs, cmap="gray")

    H, W = fixed_gray.shape[-2:]
    for moving_img, ax in zip(moving_gray, axs):
        img_np = moving_img.cpu().detach().numpy()
        img_uint8 = cv2.normalize(
            img_np, dst=np.empty_like(img_np), alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
        ).astype(np.uint8)

        img_small = cv2.resize(img_uint8, (edge_detection_size, edge_detection_size), interpolation=cv2.INTER_AREA)
        img_blurred = cv2.GaussianBlur(img_small, (blur_kernel, blur_kernel), 0)
        edges = cv2.Canny(img_blurred, canny_low, canny_high)

        edge_weights = cv2.resize(edges.astype(np.float32) / 255.0, (W, H), interpolation=cv2.INTER_LINEAR)

        rgba = np.zeros((H, W, 4), dtype=np.float32)
        rgba[..., :3] = edge_color
        rgba[..., 3] = edge_weights * edge_alpha

        ax.imshow(rgba)

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
    axs = axs if isinstance(axs, (list, np.ndarray)) else [axs]
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
    base = [
        (int(r * 255), int(g * 255), int(b * 255))
        for i in range(n_base)
        for r, g, b, *_ in [colormap(i / max(n_base - 1, 1))]
    ]
    return [base[i % n_base] for i in range(n)]
