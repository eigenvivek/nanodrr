import torch
from jaxtyping import Float

from ..data import Subject
from ..drr import render
from ..geometry import convert, transform_point


class Registration(torch.nn.Module):
    """Differentiable 2D/3D registration module.

    Optimize poses in SE(3) by aligning rendered X-rays with real target
    X-ray images. Initial intrinsic and extrinsic matrices are fixed at
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
        rt_inv: Float[torch.Tensor, "B 4 4"],
        k_inv: Float[torch.Tensor, "B 3 3"],
        sdd: Float[torch.Tensor, "B"],
        height: int,
        width: int,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.subject = subject
        self.register_buffer("rt_inv", rt_inv)
        self.register_buffer("k_inv", k_inv)
        self.register_buffer("sdd", sdd)
        self.height = height
        self.width = width

        # Parameterize the perturbation about the isocenter rather than the world origin
        c = transform_point(rt_inv.inverse(), subject.isocenter[None])
        self.register_buffer("pivot", self._make_translation(c))
        self.register_buffer("pivot_inv", self._make_translation(-c))

        # Parameterization of the perturbation
        self._rot = torch.nn.Parameter(eps * torch.randn(1, 3, device=c.device))
        self._xyz = torch.nn.Parameter(eps * torch.randn(1, 3, device=c.device))

    def forward(self) -> Float[torch.Tensor, "B C H W"]:
        return render(
            self.subject,
            self.k_inv,
            self.rt_inv @ self.pose,
            self.sdd,
            self.height,
            self.width,
        )

    @property
    def pose(self) -> Float[torch.Tensor, "B 4 4"]:
        T = convert(self._rot, self._xyz, "so3_log")
        return self.pivot @ T @ self.pivot_inv

    @torch.no_grad()
    def rescale_(self, scale: float):
        """Change the rendering resolution by rescaling the camera's intrinsics."""
        self.k_inv[..., 0, 0] /= scale
        self.k_inv[..., 1, 1] /= scale
        self.height = int(round(self.height * scale))
        self.width = int(round(self.width * scale))

    @staticmethod
    def _make_translation(translation: Float[torch.Tensor, "B 3"]) -> Float[torch.Tensor, "B 4 4"]:
        """Make a 4x4 matrix representing a translation."""
        B = len(translation)
        T = torch.eye(4, device=translation.device, dtype=translation.dtype)[None].repeat(B, 1, 1)
        T[:, :3, 3] = translation
        return T
