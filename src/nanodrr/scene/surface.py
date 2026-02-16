import pyvista as pv

from ..data import Subject
from .utils import subject_to_imagedata


def label_to_mesh(subject: Subject, verbose: bool = False) -> pv.PolyData:
    """Convert a subject's labelmap into a PyVista surface mesh.

    Extracts isosurfaces for each label class using marching cubes and
    optionally flips face normals to ensure consistent outward orientation.

    Args:
        subject: Subject containing a multi-class labelmap in `subject.label`.
        verbose: If `True`, display a progress bar during meshing.

    Returns:
        Surface mesh with one connected region per label class.
    """
    label, invert = subject_to_imagedata(subject, use_label=True)
    mesh = label.contour_labels(progress_bar=verbose)
    if invert:
        mesh = mesh.flip_faces(progress_bar=verbose)
    return mesh
