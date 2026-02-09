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
