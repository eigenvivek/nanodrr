import torch
from jaxtyping import Float


def hu_to_mu(
    data: Float[torch.Tensor, "1 1 D H W"],
    mu_water: float,
) -> Float[torch.Tensor, "1 1 D H W"]:
    """Convert clamped Hounsfield units to linear attenuation coefficients."""
    return mu_water * (1.0 + data.clamp(min=-1000) / 1000.0)
