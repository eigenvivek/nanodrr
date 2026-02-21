import numpy as np
import pyvista as pv

from ..data import Subject


def subject_to_imagedata(
    subject: Subject,
    use_label: bool = False,
) -> tuple[pv.ImageData, bool]:
    """Convert a subject's volume or labelmap into a PyVista `ImageData` grid.

    Reorders the volume axes for VTK convention, applies the voxel-to-world
    affine transform, and detects whether the affine has a negative
    determinant (indicating a left-handed coordinate system that requires
    face flipping for correct surface normals).

    Args:
        subject: Subject containing `subject.image` and optionally
            `subject.label`.
        use_label: If `True`, convert the labelmap instead of the density
            volume.

    Returns:
        A tuple of:
            - The transformed `ImageData` grid in world coordinates.
            - `True` if the affine has a negative determinant (faces should
              be flipped), `False` otherwise.
    """
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
