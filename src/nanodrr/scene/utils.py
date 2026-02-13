import numpy as np
import pyvista as pv

from ..data import Subject


def subject_to_imagedata(
    subject: Subject,
    use_label: bool = False,
) -> tuple[pv.ImageData, bool]:
    data = subject.label if use_label else subject.image
    data = data.squeeze().cpu().permute(2, 1, 0).numpy()
    affine = subject.voxel_to_world.cpu().numpy()
    invert = np.linalg.det(affine) < 0

    grid = pv.ImageData(
        dimensions=data.shape,
        spacing=(1, 1, 1),
        origin=(0, 0, 0),
    )
    grid.point_data["values"] = data.flatten(order="F")
    return grid.transform(affine, inplace=False), invert
