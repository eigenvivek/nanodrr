"""Epipolar consistency conditions (ECC) for transmission imaging.

Implements the geometric-consistency metric of

    A. Aichert et al., "Epipolar Consistency in Transmission Imaging,"
    IEEE Transactions on Medical Imaging, 2015,

which measures, without any 3D reconstruction, how geometrically consistent a set
of X-ray projections and their projection matrices are. It is ~0 for noise-free
projections at the true geometry and grows with geometric error, so it can be used
as an objective for projection-matrix calibration and motion estimation.
"""

from .consistency import DEFAULT_DKAPPA, epipolar_consistency, projection_matrix
from .radon import RadonIntermediate, radon_intermediate, sample_lines

__all__ = [
    "projection_matrix",
    "radon_intermediate",
    "RadonIntermediate",
    "sample_lines",
    "epipolar_consistency",
    "DEFAULT_DKAPPA",
]
