from enum import Enum

import torch
import roma
from jaxtyping import Float


class Parameterization(str, Enum):
    EULER = "euler"
    QUATERNION = "quaternion"
    QUATERNION_ADJUGATE = "quaternion_adjugate"
    ROTATION_9D = "rotation_9d"
    SO3_LOG = "so3_log"
    SE3_LOG = "se3_log"

    @property
    def dim(self) -> int:
        return {
            Parameterization.EULER: 3,
            Parameterization.QUATERNION: 4,
            Parameterization.QUATERNION_ADJUGATE: 10,
            Parameterization.ROTATION_9D: 9,
            Parameterization.SO3_LOG: 3,
            Parameterization.SE3_LOG: 3,
        }[self]


def convert(
    rotation: Float[torch.Tensor, "B D"],
    translation: Float[torch.Tensor, "B 3"],
    parameterization: Parameterization,
    convention: str = None,
    degrees: bool = True,
    isocenter: Float[torch.Tensor, "3"] | None = None,
) -> Float[torch.Tensor, "B 4 4"]:
    """Convert a rotation parameterization + camera center into a (B, 4, 4) SE(3) matrix.

    The translation is interpreted as the camera center in world coordinates,
    i.e. the resulting matrix stores t = R @ translation.

    Args:
        rotation: Rotation parameters, shape depends on parameterization:
            - EULER:                (B, 3) Euler angles
            - QUATERNION:           (B, 4) unit quaternion (XYZW)
            - QUATERNION_ADJUGATE:  (B, 10) upper-tri of 4x4 symmetric matrix
            - ROTATION_9D:          (B, 9) flattened 3x3 matrix (projected via SVD)
            - SE3_LOG:              (B, 3) rotation part of se(3) logarithm
        translation: Camera center in world coordinates, shape (B, 3).
            For SE3_LOG this is the log-translation (coupled via the V-matrix).
        parameterization: Which rotation representation to use.
        convention: Required for EULER only. 3-letter string from {X, Y, Z},
            e.g. "XYZ".
        degrees: If True and parameterization is EULER, interpret angles in degrees.

    Returns:
        Batched SE(3) matrices of shape (B, 4, 4).
    """
    parameterization = Parameterization(parameterization)

    if parameterization == Parameterization.SE3_LOG:
        return _se3_exp_map(rotation, translation)

    R = rotation_to_matrix(rotation, parameterization, convention, degrees)
    t = torch.einsum("bij, bj -> bi", R, translation)
    if isocenter is not None:
        t = t + isocenter
    return make_se3(R, t)


def rotation_to_matrix(
    rotation: Float[torch.Tensor, "B D"],
    parameterization: Parameterization,
    convention: str = None,
    degrees: bool = False,
) -> Float[torch.Tensor, "B 3 3"]:
    """Convert rotation parameters into a (B, 3, 3) rotation matrix.

    Args:
        rotation: Rotation parameters (shape depends on parameterization).
        parameterization: Which rotation representation to use.
        convention: Required for EULER. 3-letter string from {X, Y, Z}.
        degrees: If True and EULER, interpret angles in degrees.

    Returns:
        Batched rotation matrices of shape (B, 3, 3).
    """
    parameterization = Parameterization(parameterization)

    if parameterization == Parameterization.EULER:
        if convention is None:
            raise ValueError("convention must be specified for Euler angles (e.g., 'XYZ', 'ZYX')")
        if degrees:
            rotation = rotation * (torch.pi / 180.0)
        return roma.euler_to_rotmat(convention, rotation)

    if parameterization == Parameterization.QUATERNION:
        return roma.unitquat_to_rotmat(rotation)

    if parameterization == Parameterization.QUATERNION_ADJUGATE:
        q = quaternion_adjugate_to_quaternion(rotation)
        q = roma.quat_wxyz_to_xyzw(q)
        return roma.unitquat_to_rotmat(q)

    if parameterization == Parameterization.ROTATION_9D:
        return roma.special_procrustes(rotation.reshape(-1, 3, 3))

    if parameterization == Parameterization.SO3_LOG:
        return roma.rotvec_to_rotmat(rotation)

    if parameterization == Parameterization.SE3_LOG:
        return roma.rotvec_to_rotmat(rotation)

    raise ValueError(f"Unknown parameterization: {parameterization}")


def make_se3(R: Float[torch.Tensor, "B 3 3"], t: Float[torch.Tensor, "B 3"]) -> Float[torch.Tensor, "B 4 4"]:
    """Assemble a (B, 4, 4) SE(3) matrix from rotation and translation.

    Args:
        R: Rotation matrices of shape (B, 3, 3).
        t: Translation vectors of shape (B, 3).

    Returns:
        Homogeneous transformation matrices of shape (B, 4, 4).
    """
    B = R.shape[0]
    T = torch.zeros(B, 4, 4, dtype=R.dtype, device=R.device)
    T[:, :3, :3] = R
    T[:, :3, 3] = t
    T[:, 3, 3] = 1.0
    return T


def quaternion_adjugate_to_quaternion(rotation: Float[torch.Tensor, "B 10"]) -> Float[torch.Tensor, "B 4"]:
    """Convert a 10D quaternion-adjugate vector to a unit quaternion.

    The 10D vector encodes the upper triangle of a symmetric 4x4 matrix (the
    quaternion adjugate). The quaternion is recovered as the column with largest
    norm, avoiding an explicit eigendecomposition via a fast method.

    Reference: https://arxiv.org/abs/2205.09116

    Args:
        rotation: (B, 10) upper-triangular entries of the 4x4 symmetric matrix.

    Returns:
        Unit quaternions of shape (B, 4).
    """
    A = _vec10_to_symmetric4x4(rotation)
    norms = A.norm(dim=1).amax(dim=1, keepdim=True)
    max_eigenvectors = torch.argmax(A.norm(dim=1), dim=1)
    return A[range(len(A)), max_eigenvectors] / norms


def se3_log_map(matrix: Float[torch.Tensor, "B 4 4"]) -> tuple[Float[torch.Tensor, "B 3"], Float[torch.Tensor, "B 3"]]:
    """Compute the SE(3) logarithm of a batch of 4x4 transformation matrices.

    Args:
        matrix: (B, 4, 4) SE(3) matrices.

    Returns:
        Tuple of (log_rotation, log_translation), each (B, 3).
    """
    R = matrix[:, :3, :3]
    t = matrix[:, :3, 3]

    log_rotation = roma.rotmat_to_rotvec(R)
    V = _se3_V_matrix(log_rotation)
    log_translation = torch.linalg.solve(V, t.unsqueeze(-1)).squeeze(-1)

    return log_rotation, log_translation


def _se3_exp_map(
    log_rotation: Float[torch.Tensor, "B 3"], log_translation: Float[torch.Tensor, "B 3"]
) -> Float[torch.Tensor, "B 4 4"]:
    """Compute the SE(3) exponential map.

    Args:
        log_rotation: (B, 3) rotation part of the se(3) logarithm (rotation vector).
        log_translation: (B, 3) translation part of the se(3) logarithm.

    Returns:
        (B, 4, 4) SE(3) matrices.
    """
    R = roma.rotvec_to_rotmat(log_rotation)
    V = _se3_V_matrix(log_rotation)
    t = torch.einsum("bij, bj -> bi", V, log_translation)
    return make_se3(R, t)


def _se3_V_matrix(log_rotation: Float[torch.Tensor, "B 3"], eps: float = 1e-4) -> Float[torch.Tensor, "B 3 3"]:
    """Compute the V matrix for the SE(3) exp/log maps.

    V = I + ((1 - cos θ) / θ²) [ω]× + ((θ - sin θ) / θ³) [ω]×²

    where θ = ‖log_rotation‖ and [ω]× is the skew-symmetric (hat) matrix.

    Args:
        log_rotation: (B, 3) rotation vectors.
        eps: Clamping threshold to avoid division by zero near θ=0.

    Returns:
        (B, 3, 3) V matrices.
    """
    theta_sq = (log_rotation * log_rotation).sum(dim=-1)  # (B,)
    theta = theta_sq.clamp(min=eps).sqrt()  # (B,)

    skew = _hat(log_rotation)  # (B, 3, 3)
    skew_sq = skew @ skew  # (B, 3, 3)

    I = torch.eye(3, dtype=log_rotation.dtype, device=log_rotation.device)  # noqa: E741
    a = ((1.0 - torch.cos(theta)) / theta_sq).unsqueeze(-1).unsqueeze(-1)
    b = ((theta - torch.sin(theta)) / (theta_sq * theta)).unsqueeze(-1).unsqueeze(-1)

    return I + a * skew + b * skew_sq


def _hat(v: Float[torch.Tensor, "B 3"]) -> Float[torch.Tensor, "B 3 3"]:
    """Compute the skew-symmetric (hat) matrix of a batch of 3D vectors.

    Args:
        v: (B, 3) vectors.

    Returns:
        (B, 3, 3) skew-symmetric matrices.
    """
    x, y, z = v.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        [
            *(zero, -z, y),
            *(z, zero, -x),
            *(-y, x, zero),
        ],
        dim=-1,
    ).reshape(v.shape[0], 3, 3)


def _vec10_to_symmetric4x4(vec: Float[torch.Tensor, "B 10"]) -> Float[torch.Tensor, "B 4 4"]:
    """Convert a 10D vector to a symmetric 4x4 matrix.

    Args:
        vec: (B, 10) upper-triangular entries.

    Returns:
        (B, 4, 4) symmetric matrices.
    """
    B = vec.shape[0]
    A = torch.zeros(B, 4, 4, dtype=vec.dtype, device=vec.device)
    idx, jdx = torch.triu_indices(4, 4)
    A[:, idx, jdx] = vec
    A[:, jdx, idx] = vec
    return A
