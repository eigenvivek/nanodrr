from typing import Literal

import pyvista as pv
import torch
from jaxtyping import Float

from ..data import Subject
from ..drr import render
from .camera import make_cameras
from .surface import subject_to_mesh


CullingMode = Literal["front", "back", "frontface", "backface", "f", "b"]


def visualize_scene(
    subject: Subject,
    k_inv: Float[torch.Tensor, "B 3 3"],
    rt_inv: Float[torch.Tensor, "B 4 4"],
    sdd: Float[torch.Tensor, "B"],
    height: int,
    width: int,
    render_mesh: bool = True,
    render_imgs: bool = True,
    single_channel: bool = False,
    culling: CullingMode | None = "back",
    use_label: bool = False,
    cutoff: float | None = 0.01,
    plotter: pv.Plotter | None = None,
    verbose: bool = False,
    **kwargs,
) -> pv.Plotter:
    """Render a DRR and return a 3D scene with camera frustums and anatomy.

    Args:
        subject: The subject containing the CT volume and labelmap.
        k_inv: Inverse intrinsic matrices.
        rt_inv: Camera-to-world transforms.
        sdd: Source-to-detector distances.
        height: Detector height in pixels.
        width: Detector width in pixels.
        render_mesh: If True, extract a surface mesh and add it to the plot.
        render_imgs: If True, render DRRs before plotting.
        single_channel: If True, sum channels before texturing the detector.
        culling: Face culling mode passed to each mesh (e.g. `"back"`).
        use_label: If `True`, use the labelmap; if `False`, use the image volume.
        cutoff: Threshold for binarizing the image. Ignored when `use_label=True`.
            Set to `None` to use raw values without binarization.
        plotter: A PyVista plotter to add to.
        verbose: If True, print progress during mesh extraction.
        **kwargs: Additional arguments forwarded to :func:`render`.

    Returns:
        A PyVista plotter with the anatomy mesh and camera frustums added.
    """
    # Create a mesh from the subject
    if render_mesh:
        mesh = subject_to_mesh(subject, use_label, cutoff, verbose)

    # Render the DRR
    img = None
    if render_imgs:
        img = render(subject, k_inv, rt_inv, sdd, height, width, **kwargs)
        img = img.sum(dim=1, keepdim=True) if single_channel else img

    # Make the cameras
    cameras = make_cameras(k_inv, rt_inv, sdd, height, width, img)

    # Make the scene
    if plotter is None:
        plotter = pv.Plotter()
    if render_mesh:
        plotter.add_mesh(mesh)
    for cam in cameras:
        if render_imgs:
            plotter.add_mesh(cam["detector"], texture=cam["texture"], lighting=False, culling=culling)
        plotter.add_mesh(cam["camera"], show_edges=True, line_width=3, culling=culling)
        plotter.add_mesh(cam["principal_ray"], line_width=3, color="lime", culling=culling)
    return plotter
