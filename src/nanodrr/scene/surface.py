import pyvista as pv

from ..data import Subject
from .utils import subject_to_imagedata


def label_to_mesh(subject: Subject, verbose: bool = False) -> pv.PolyData:
    label, invert = subject_to_imagedata(subject, use_label=True)
    mesh = label.contour_labels(progress_bar=verbose)
    if invert:
        mesh = mesh.flip_faces(progress_bar=verbose)
    return mesh
