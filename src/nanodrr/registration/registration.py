import torch
from jaxtyping import Float

from ..data import Subject
from ..drr import render
from ..geometry import convert


class Registration(torch.nn.Module):
    """Differentiable 2D/3D registration module.

    Optimize a pose in SE(3) by aligning a rendered X-ray with a real target
    X-ray image. The initial intrinsic and extrinsic matrices are fixed at
    construction. Optimizable parameters are parameterized as perturbations.

    Args:
        subject: The volume to render DRRs from during optimization.
        rt_inv: Initial inverse extrinsic (camera-to-world) matrix.
        k_inv: Inverse intrinsic camera matrix.
        sdd: Source-to-detector distance.
        height: Output image height in pixels.
        width: Output image width in pixels.
        eps: Small constant for numerical stability.
    """

    def __init__(
        self,
        subject: Subject,
        rt_inv: Float[torch.Tensor, "1 4 4"],
        k_inv: Float[torch.Tensor, "1 3 3"],
        sdd: Float[torch.Tensor, "1"],
        height: int,
        width: int,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.subject = subject
        self.rt_inv = rt_inv
        self.k_inv = k_inv
        self.sdd = sdd
        self.height = height
        self.width = width

        # Parameterization of the perturbation
        self._rot = torch.nn.Parameter(eps * torch.randn(1, 3, device=c.device))
        self._xyz = torch.nn.Parameter(eps * torch.randn(1, 3, device=c.device))

    def forward(self) -> Float[torch.Tensor, "1 C H W"]:
        return render(
            self.subject,
            self.k_inv,
            self.pose @ self.rt_inv,
            self.sdd,
            self.height,
            self.width,
        )

    @property
    def pose(self) -> Float[torch.Tensor, "1 4 4"]:
        return convert(self._rot, self._xyz, "so3_log")
