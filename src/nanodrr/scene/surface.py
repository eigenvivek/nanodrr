import numpy as np
import pyvista as pv
from jaxtyping import Float
from trimesh import Trimesh

from ..data import Subject


def image_to_mesh(subject: Subject, threshold: float, verbose: bool = False) -> pv.PolyData:
    data = subject.image.squeeze().cpu().permute(2, 1, 0).numpy() > threshold
    affine = subject.voxel_to_world.cpu().numpy()
    return surface_nets(data.astype(np.float32), affine, verbose)


def label_to_mesh(subject: Subject, verbose: bool = False) -> pv.PolyData:
    data = subject.label.squeeze().cpu().permute(2, 1, 0).numpy()
    affine = subject.voxel_to_world.cpu().numpy()
    return surface_nets(data, affine, verbose)


def surface_nets(
    data: Float[np.ndarray, "W H D"],
    affine: Float[np.ndarray, "4 4"],
    verbose: bool,
) -> Trimesh:

    grid = pv.ImageData(dimensions=data.shape, spacing=(1, 1, 1), origin=(0, 0, 0))
    grid.point_data["values"] = data.flatten(order="F")

    mesh = grid.contour_labels(progress_bar=verbose)

    mesh = mesh.transform(affine, inplace=False, progress_bar=verbose)
    if np.linalg.det(affine) < 0:
        mesh = mesh.flip_faces(progress_bar=verbose)

    return pv.to_trimesh(mesh, triangulate=True)
