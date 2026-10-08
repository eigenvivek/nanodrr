import torch

from nanodrr.geometry.se3 import (
    Parameterization,
    convert,
    make_se3,
    quaternion_adjugate_to_quaternion,
    rotation_to_matrix,
    se3_log_map,
)


def test_make_se3_and_se3_log_roundtrip(device):
    """SE(3) matrices should round-trip through log/exp."""
    B = 4
    R = torch.eye(3, device=device).unsqueeze(0).repeat(B, 1, 1)
    # Add small random rotations
    rotvec = torch.randn(B, 3, device=device) * 0.1
    # matrix_exp is not implemented on MPS, so build the rotations on the CPU
    R = (
        torch.linalg.matrix_exp(
            torch.stack(
                [
                    torch.tensor(
                        [
                            [0.0, -z, y],
                            [z, 0.0, -x],
                            [-y, x, 0.0],
                        ],
                    )
                    for x, y, z in rotvec.cpu()
                ],
                dim=0,
            )
        ).to(device)
        @ R
    )

    t = torch.randn(B, 3, device=device)
    T = make_se3(R, t)

    log_R, log_t = se3_log_map(T)
    T_roundtrip = convert(
        rotation=log_R,
        translation=log_t,
        parameterization=Parameterization.SE3_LOG,
        convention=None,
    )

    torch.testing.assert_close(T_roundtrip, T, atol=1e-4, rtol=1e-4)


def test_rotation_to_matrix_euler_roundtrip(device):
    """Euler parameterization should be invertible via roma (up to numerical tolerance)."""
    B = 8
    angles_deg = torch.randn(B, 3, device=device) * 45.0
    R = rotation_to_matrix(
        rotation=angles_deg,
        parameterization=Parameterization.EULER,
        convention="ZYX",
        degrees=True,
    )
    # Ensure rotation matrices are orthonormal
    I = torch.eye(3, device=device)
    RtR = R.transpose(-1, -2) @ R
    torch.testing.assert_close(RtR, I.expand_as(RtR), atol=1e-5, rtol=1e-5)


def test_quaternion_adjugate_to_quaternion_unit_norm(device):
    """Quaternion recovered from 10D adjugate should be unit-normalized."""
    B = 5
    # Start from random quaternions
    q = torch.randn(B, 4, device=device)
    q = q / q.norm(dim=-1, keepdim=True)

    # Build symmetric 4x4 matrices whose columns are scaled versions of q.
    A = torch.zeros(B, 4, 4, device=device)
    scales = torch.linspace(1.0, 2.0, steps=4, device=device)
    for i in range(4):
        A[:, :, i] = q * scales[i]
    A = 0.5 * (A + A.transpose(-1, -2))

    # Encode upper triangle into 10D representation.
    idx, jdx = torch.triu_indices(4, 4, device=device)
    vec10 = A[:, idx, jdx]

    q_rec = quaternion_adjugate_to_quaternion(vec10)
    norms = q_rec.norm(dim=-1)
    torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5)
