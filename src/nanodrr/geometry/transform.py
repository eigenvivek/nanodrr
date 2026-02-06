import torch
from jaxtyping import Float


def _homo(v: Float[torch.Tensor, "*B N"]) -> Float[torch.Tensor, "*B N+1"]:
    """Convert to homogeneous coordinates by appending ones."""
    return torch.cat([v, torch.ones_like(v[..., :1])], dim=-1)


def _dehomo(v: Float[torch.Tensor, "*B N+1"]) -> Float[torch.Tensor, "*B N"]:
    """Convert from homogeneous coordinates by dividing by last component."""
    return v[..., :-1] / v[..., -1:]


def transform_point(
    xform: Float[torch.Tensor, "B *_ N+1 N+1"],
    v: Float[torch.Tensor, "B *_ N"],
) -> Float[torch.Tensor, "B *_ N"]:
    """Apply homogeneous transformation to points.

    Args:
        xform: Transformation matrix of shape (B, ..., N+1, N+1).
        v: Points of shape (B, ..., N).

    Returns:
        Transformed points of shape (B, ..., N).
    """
    return _dehomo(torch.einsum("b...ij,b...j->b...i", xform, _homo(v)))
