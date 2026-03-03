import pyvista as pv

from ..data import Subject
from .utils import subject_to_imagedata


def subject_to_mesh(
    subject: Subject,
    use_label: bool,
    cutoff: float | None,
    verbose: bool = False,
) -> pv.PolyData:
    """Convert a subject's image or labelmap into a PyVista surface mesh.

    Extracts isosurfaces using marching cubes (for images) or SurfaceNets
    (for labels), and optionally flips face normals to ensure consistent
    outward orientation.

    Args:
        subject: Subject containing data in `subject.image` or `subject.label`.
        use_label: If `True`, use the labelmap; if `False`, use the image volume.
        cutoff: Threshold for binarizing the image. Ignored when `use_label=True`.
            Set to `None` to use raw values without binarization.
        verbose: If `True`, display a progress bar during meshing.

    Returns:
        Surface mesh from the subject data.
    """
    # Don't apply cutoff to labelmaps
    cutoff_value = None if use_label else cutoff

    imagedata, invert = subject_to_imagedata(subject, cutoff=cutoff_value, use_label=use_label)
    mesh = imagedata.contour_labels(progress_bar=verbose)

    if invert:
        mesh = mesh.flip_faces(progress_bar=verbose)

    return mesh
