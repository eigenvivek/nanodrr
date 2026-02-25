import pathlib
from base64 import b64encode

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import HTML
from IPython.display import display as ipython_display
from jaxtyping import Bool, Float
from tqdm import tqdm

from .imshow import overlay, plot_drr


def animate(
    moving_img: Float[torch.Tensor, "B C H W"],
    moving_mask: Bool[torch.Tensor, "B C H W"] | None = None,
    out: str | pathlib.Path | None = None,
    fixed_img: Float[torch.Tensor, "1 C H W"] | None = None,
    fixed_mask: Bool[torch.Tensor, "1 C H W"] | None = None,
    titles: list[str] | None = None,
    fps: int = 20,
    pause: float = 1.0,
    verbose: bool = True,
    blur_kernel: int = 3,
    canny_low: int = 0,
    canny_high: int = 100,
    edge_color: tuple[float, float, float] = (1.0, 0.0, 0.0),
    edge_alpha: float = 1.0,
    **kwargs,
) -> pathlib.Path | None:
    """Create an animated GIF from a batch of DRR images.

    Renders a sequence of DRR images as an animated GIF, with optional
    side-by-side comparison against a fixed reference image. When `out` is
    `None`, displays the animation inline in Jupyter notebooks.

    Multi-channel images are automatically converted to single-channel with
    segmentation masks extracted from channels 1+ (channel 0 is background).

    When `fixed_img` is provided, a third column is rendered showing the
    moving image edges overlaid on the fixed image via `overlay`.

    Args:
        moving_img: Batch of moving DRR images.
        moving_mask: Optional segmentation mask for moving images.
        out: Output file path, or `None` for inline display.
        fixed_img: Optional fixed reference image for comparison.
        fixed_mask: Optional segmentation mask for fixed image.
        titles: Optional per-frame titles of length `B`.
        fps: Frames per second for playback.
        pause: Pause duration in seconds at the end of the loop.
        verbose: Whether to display rendering progress.
        blur_kernel: Gaussian blur kernel size applied before Canny edge detection.
        canny_low: Lower hysteresis threshold for Canny edge detection.
        canny_high: Upper hysteresis threshold for Canny edge detection.
        edge_color: RGB color of the overlaid edges.
        edge_alpha: Opacity of the overlaid edges, in `[0, 1]`.
        **kwargs: Additional arguments forwarded to `imageio.v3.imwrite` or `plot_drr`.

    Returns:
        Path to saved file if `out` is provided, otherwise `None`.

    Raises:
        ValueError: If `titles` length does not match batch size.
    """
    B = len(moving_img)
    if titles is not None and len(titles) != B:
        raise ValueError(f"titles length ({len(titles)}) must match batch size ({B})")

    moving_img, moving_mask = _normalize(moving_img, moving_mask)
    if fixed_img is not None:
        fixed_img, fixed_mask = _normalize(fixed_img, fixed_mask)

    iio_keys = {"duration", "loop", "quality", "quantizer", "palettesize"}
    iio_kwargs = {k: v for k, v in kwargs.items() if k in iio_keys}
    iio_kwargs.setdefault("fps", fps)
    iio_kwargs.setdefault("loop", 0)
    plot_kwargs = {k: v for k, v in kwargs.items() if k not in iio_keys}

    has_fixed = fixed_img is not None
    n_cols = 3 if has_fixed else 1
    figsize = (3 * n_cols, 3)

    overlay_kwargs = dict(
        blur_kernel=blur_kernel,
        canny_low=canny_low,
        canny_high=canny_high,
        edge_color=edge_color,
        edge_alpha=edge_alpha,
    )

    iterator = tqdm(range(B), desc="Rendering frames", ncols=75) if verbose else range(B)
    frames = []

    for i in iterator:
        fig, axs = plt.subplots(ncols=n_cols, figsize=figsize, constrained_layout=True)
        axs = [axs] if n_cols == 1 else list(axs)

        if has_fixed:
            frame_img = torch.cat([fixed_img, moving_img[i : i + 1]])
            frame_mask = _concat_masks(fixed_mask, moving_mask[i : i + 1] if moving_mask is not None else None)
            frame_titles = ["Fixed", titles[i] if titles else "Moving", "Overlay"]
            plot_drr(frame_img, frame_mask, title=frame_titles[:2], axs=axs[:2], **plot_kwargs)
            overlay(fixed_img, moving_img[i : i + 1], title=[frame_titles[2]], axs=axs[2], **overlay_kwargs)
        else:
            frame_img = moving_img[i : i + 1]
            frame_mask = moving_mask[i : i + 1] if moving_mask is not None else None
            frame_titles = [titles[i]] if titles else None
            plot_drr(frame_img, frame_mask, title=frame_titles, axs=axs, **plot_kwargs)

        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3])
        plt.close(fig)

    if pause > 0:
        frames.extend([frames[-1]] * int(pause * fps))

    frames_array = np.stack(frames)
    if out is None:
        gif_bytes = iio.imwrite("<bytes>", frames_array, extension=".gif", **iio_kwargs)
        ipython_display(HTML(f"<img src='data:image/gif;base64,{b64encode(gif_bytes).decode()}'>"))
        return None
    else:
        out_path = pathlib.Path(out)
        iio.imwrite(out_path, frames_array, **iio_kwargs)
        return out_path


def _normalize(
    img: Float[torch.Tensor, "B C H W"],
    mask: Bool[torch.Tensor, "B C H W"] | None,
) -> tuple[Float[torch.Tensor, "B 1 H W"], Bool[torch.Tensor, "B C H W"] | None]:
    """Normalize multi-channel images to single-channel with separate masks."""
    if img.shape[1] > 1:
        if mask is None:
            mask = img > 0
        img = img.sum(dim=1, keepdim=True)
    return img, mask


def _concat_masks(
    mask1: torch.Tensor | None,
    mask2: torch.Tensor | None,
) -> torch.Tensor | None:
    """Concatenate two masks, padding channels as needed."""
    if mask1 is None and mask2 is None:
        return None

    if mask1 is None:
        return torch.cat([torch.zeros_like(mask2), mask2])
    if mask2 is None:
        return torch.cat([mask1, torch.zeros_like(mask1)])

    # Pad to matching channel dimensions
    max_ch = max(mask1.shape[1], mask2.shape[1])
    if mask1.shape[1] < max_ch:
        pad = torch.zeros(
            *mask1.shape[:1], max_ch - mask1.shape[1], *mask1.shape[2:], dtype=mask1.dtype, device=mask1.device
        )
        mask1 = torch.cat([mask1, pad], dim=1)
    if mask2.shape[1] < max_ch:
        pad = torch.zeros(
            *mask2.shape[:1], max_ch - mask2.shape[1], *mask2.shape[2:], dtype=mask2.dtype, device=mask2.device
        )
        mask2 = torch.cat([mask2, pad], dim=1)

    return torch.cat([mask1, mask2])
