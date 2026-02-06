import torch
from jaxtyping import Float


def _homo(v: Float[torch.Tensor, "*batch N"]) -> Float[torch.Tensor, "*batch N+1"]:
    """Convert to homogeneous coordinates by appending ones."""
    return torch.cat([v, torch.ones_like(v[..., :1])], dim=-1)


def _dehomo(v: Float[torch.Tensor, "*batch N+1"]) -> Float[torch.Tensor, "*batch N"]:
    """Convert from homogeneous coordinates by dividing by last component."""
    return v[..., :-1] / v[..., -1:]


def transform_point(
    xform: Float[torch.Tensor, "*batch N+1 N+1"], v: Float[torch.Tensor, "*batch N"]
) -> Float[torch.Tensor, "*batch N"]:
    """
    Apply homogeneous transformation to points.

    Args:
        xform: Transformation matrix of shape (..., N+1, N+1)
        v: Points of shape (..., N)

    Returns:
        Transformed points of shape (..., N)
    """
    return _dehomo(torch.einsum("...ij,...j->...i", xform, _homo(v)))
