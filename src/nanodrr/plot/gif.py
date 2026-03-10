from base64 import b64encode
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from imageio.v3 import imwrite
from IPython.display import HTML, display
from jaxtyping import Bool, Float
from matplotlib.backends.backend_agg import FigureCanvasAgg
from tqdm import tqdm

from .imshow import overlay, plot_drr


def animate(
    moving_img: Float[torch.Tensor, "B C H W"],
    moving_mask: Bool[torch.Tensor, "B C H W"] | None = None,
    out: str | Path | None = None,
    fixed_img: Float[torch.Tensor, "1 C H W"] | None = None,
    fixed_mask: Bool[torch.Tensor, "1 C H W"] | None = None,
    titles: list[str] | None = None,
    ticks: bool = True,
    fps: int = 20,
    pause: float = 1.0,
    blur_kernel: int = 3,
    canny_low: int = 0,
    canny_high: int = 100,
    edge_color: tuple[float, float, float] = (1.0, 0.0, 0.0),
    edge_alpha: float = 1.0,
    edge_detection_size: int = 200,
    verbose: bool = True,
    **kwargs,
) -> Path | None:
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
        ticks: Whether to show pixel coordinate ticks.
        fps: Frames per second for playback.
        pause: Pause duration in seconds at the end of the loop.
        blur_kernel: Gaussian blur kernel size applied before Canny edge detection.
        canny_low: Lower hysteresis threshold for Canny edge detection.
        canny_high: Upper hysteresis threshold for Canny edge detection.
        edge_color: RGB color of the overlaid edges.
        edge_alpha: Opacity of the overlaid edges, in `[0, 1]`.
        verbose: Whether to display rendering progress.
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

    n_cols = 3 if fixed_img is not None else 1
    figsize = (3 * n_cols, 3)

    iterator = tqdm(range(B), desc="Rendering frames", ncols=75) if verbose else range(B)
    frames = []

    for i in iterator:
        fig, axs = plt.subplots(ncols=n_cols, figsize=figsize, constrained_layout=True)
        axs = [axs] if n_cols == 1 else list(axs)

        if fixed_img is not None:
            frame_img = torch.cat([fixed_img, moving_img[i : i + 1]])
            frame_mask = _concat_masks(fixed_mask, moving_mask[i : i + 1] if moving_mask is not None else None)
            frame_titles = ["Fixed", titles[i] if titles else "Moving", "Overlay"]
            plot_drr(frame_img, frame_mask, title=frame_titles[:2], axs=axs[:2], ticks=ticks, **plot_kwargs)
            overlay(
                moving_img[i : i + 1],
                fixed_img,
                title=[frame_titles[2]],
                ticks=ticks,
                axs=axs[2],
                blur_kernel=blur_kernel,
                canny_low=canny_low,
                canny_high=canny_high,
                edge_color=edge_color,
                edge_alpha=edge_alpha,
                edge_detection_size=edge_detection_size,
            )
        else:
            frame_img = moving_img[i : i + 1]
            frame_mask = moving_mask[i : i + 1] if moving_mask is not None else None
            frame_titles = [titles[i]] if titles else None
            plot_drr(frame_img, frame_mask, title=frame_titles, ticks=ticks, axs=axs, **plot_kwargs)

        fig.canvas.draw()
        frames.append(np.asarray(cast(FigureCanvasAgg, fig.canvas).buffer_rgba())[..., :3])
        plt.close(fig)

    if pause > 0:
        frames.extend([frames[-1]] * int(pause * fps))

    frames_array = np.stack(frames)
    if out is None:
        gif_bytes = imwrite("<bytes>", frames_array, extension=".gif", **iio_kwargs)
        display(HTML(f"<img src='data:image/gif;base64,{b64encode(gif_bytes).decode()}'>"))
        return None
    else:
        out_path = Path(out)
        imwrite(out_path, frames_array, **iio_kwargs)
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
    if mask1 is None and mask2 is None:
        return None
    if mask1 is None:
        assert mask2 is not None
        return torch.cat([torch.zeros_like(mask2), mask2])
    if mask2 is None:
        return torch.cat([mask1, torch.zeros_like(mask1)])

    max_ch = max(mask1.shape[1], mask2.shape[1])
    return torch.cat([_pad_channels(mask1, max_ch), _pad_channels(mask2, max_ch)])


def _pad_channels(mask: torch.Tensor, n: int) -> torch.Tensor:
    if mask.shape[1] >= n:
        return mask
    pad = torch.zeros(*mask.shape[:1], n - mask.shape[1], *mask.shape[2:], dtype=mask.dtype, device=mask.device)
    return torch.cat([mask, pad], dim=1)
