import torch
from jaxtyping import Float


def skew_symmetric(v: Float[torch.Tensor, "... 3"]) -> Float[torch.Tensor, "... 3 3"]:
    batch_shape = v.shape[:-1]
    K = torch.zeros(*batch_shape, 3, 3, device=v.device, dtype=v.dtype)
    K[..., 0, 1] = -v[..., 2]
    K[..., 0, 2] = v[..., 1]
    K[..., 1, 0] = v[..., 2]
    K[..., 1, 2] = -v[..., 0]
    K[..., 2, 0] = -v[..., 1]
    K[..., 2, 1] = v[..., 0]
    return K


def so3_exp_map(omega: Float[torch.Tensor, "... 3"]) -> Float[torch.Tensor, "... 3 3"]:
    """Rodrigues' formula: axis-angle to rotation matrix."""
    theta_sq = torch.sum(omega * omega, dim=-1, keepdim=True)
    theta = torch.sqrt(theta_sq)

    small_angle = theta_sq < 1e-8

    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)

    sin_theta_over_theta = torch.where(small_angle, 1.0 - theta_sq / 6.0, sin_theta / theta)
    one_minus_cos_over_theta_sq = torch.where(small_angle, 0.5 - theta_sq / 24.0, (1.0 - cos_theta) / theta_sq)

    K = skew_symmetric(omega)
    K_sq = torch.matmul(K, K)

    R = (
        torch.eye(3, device=omega.device, dtype=omega.dtype)
        + sin_theta_over_theta.unsqueeze(-1) * K
        + one_minus_cos_over_theta_sq.unsqueeze(-1) * K_sq
    )

    return R


def so3_log_map(R: Float[torch.Tensor, "... 3 3"]) -> Float[torch.Tensor, "... 3"]:
    """Rotation matrix to axis-angle via the matrix logarithm.

    Handles three regimes branchlessly (small angle, general, near-pi)
    so that torch.compile can trace through the entire function.
    """
    batch_shape = R.shape[:-2]

    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)

    skew = 0.5 * (R - R.transpose(-1, -2))
    omega_raw = torch.stack(
        [skew[..., 2, 1], skew[..., 0, 2], skew[..., 1, 0]],
        dim=-1,
    )

    theta_unsq = theta.unsqueeze(-1)
    sin_theta = torch.sin(theta).unsqueeze(-1)
    theta_sq = theta_unsq * theta_unsq

    # Small angle: Taylor expansion of theta / sin(theta)
    small_angle = theta_unsq.abs() < 1e-4
    scale_small = 1.0 + theta_sq / 6.0
    scale_general = theta_unsq / sin_theta.clamp(min=1e-7)

    omega = torch.where(small_angle, scale_small, scale_general) * omega_raw

    # Near pi: extract eigenvector from R + I
    near_pi = (theta_unsq.abs() > 3.1) & ~small_angle
    RpI = R + torch.eye(3, device=R.device, dtype=R.dtype)
    col_norms = RpI.norm(dim=-2)
    best_col = col_norms.argmax(dim=-1)
    idx = best_col.unsqueeze(-1).unsqueeze(-1).expand(*batch_shape, 3, 1)
    v = torch.gather(RpI, -1, idx).squeeze(-1)
    v = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    omega_pi = v * theta_unsq
    dot = (omega_pi * omega_raw).sum(dim=-1, keepdim=True)
    omega_pi = torch.where(dot < 0, -omega_pi, omega_pi)

    omega = torch.where(near_pi, omega_pi, omega)

    return omega
