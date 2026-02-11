import torch
from jaxtyping import Float


def hu_to_mu(
    data: Float[torch.Tensor, "1 1 D H W"],
    mu_water: float = 0.0192,
    mu_bone: float = 0.0573,
    hu_bone: float = 1000.0,
) -> Float[torch.Tensor, "1 1 D H W"]:
    """Convert Hounsfield units to linear attenuation coefficients.

    Uses bilinear scaling with air-water model for HU <= 0 and
    water-bone model for HU > 0.

    Args:
        data: CT volume in Hounsfield Units with shape (1, 1, D, H, W).
        mu_water: Linear attenuation coefficient of water [1/mm] at target
            energy. Default 0.0192 corresponds to ~70 keV (typical CT
            effective energy).
        mu_bone: Linear attenuation coefficient of cortical bone [1/mm] at
            target energy. Default 0.0573 corresponds to ~70 keV.
        hu_bone: HU value corresponding to pure cortical bone. Default 1000.
            Typical range is 1000-2000 depending on bone type and scanner.

    Returns:
        Linear attenuation coefficients [1/mm] with same shape as input.

    References:
        NIST XCOM database for mass attenuation coefficients.
        Water density: 1.0 g/cm³, cortical bone density: 1.92 g/cm³.
    """
    hu_clamped = data.clamp(min=-1000.0)

    mu_low = mu_water * (1.0 + hu_clamped / 1000.0)
    mu_high = mu_water + (hu_clamped / hu_bone) * (mu_bone - mu_water)
    mu = torch.where(hu_clamped <= 0, mu_low, mu_high)

    return mu.clamp(min=0.0)
