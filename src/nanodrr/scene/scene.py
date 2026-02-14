import pyvista as pv
import torch
from jaxtyping import Float

from ..data import Subject
from ..drr import render
from .camera import make_cameras
from .surface import label_to_mesh


def visualize_scene(
    subject: Subject,
    k_inv: Float[torch.Tensor, "B 3 3"],
    rt_inv: Float[torch.Tensor, "B 4 4"],
    sdd: Float[torch.Tensor, "B"],
    height: int,
    width: int,
    single_channel: bool = False,
    culling: str | None = "back",
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
        single_channel: If True, sum channels before texturing the detector.
        culling: Face culling mode passed to each mesh (e.g. ``"back"``).
        verbose: If True, print progress during mesh extraction.
        **kwargs: Additional arguments forwarded to :func:`render`.

    Returns:
        A PyVista plotter with the anatomy mesh and camera frustums added.
    """
    # Get a mesh from the subject's labelmap
    mesh = label_to_mesh(subject, verbose)

    # Render the DRR
    img = render(subject, k_inv, rt_inv, sdd, height, width, **kwargs)
    if single_channel:
        img = img.sum(dim=1, keepdim=True)

    # Make the cameras
    cameras = make_cameras(img, k_inv, rt_inv, sdd, height, width)

    # Make the scene
    pl = pv.Plotter()
    pl.add_mesh(mesh)
    for cam in cameras:
        pl.add_mesh(cam["detector"], texture=cam["texture"], lighting=False, culling=culling)
        pl.add_mesh(cam["camera"], show_edges=True, line_width=3, culling=culling)
        pl.add_mesh(cam["principal_ray"], line_width=3, color="lime", culling=culling)
    return pl
