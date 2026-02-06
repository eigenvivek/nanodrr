import torch

from ..data import Subject
from ..drr import render
from ..geometry import transform_point, so3_exp_map


class Registration(torch.nn.Module):
    def __init__(
        self,
        subject: Subject,
        rt_inv: torch.Tensor,
        k_inv: torch.Tensor,
        sdd: torch.Tensor,
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

        # Rotation pivot: isocenter projected into camera frame
        c = transform_point(rt_inv.inverse(), subject.isocenter)
        self.pivot = torch.eye(4, device=c.device, dtype=c.dtype)[None]
        self.pivot_inv = torch.eye(4, device=c.device, dtype=c.dtype)[None]
        self.pivot[:, :3, 3] = c
        self.pivot_inv[:, :3, 3] = -c

        # Parameterization of the perturbation
        self._rot = torch.nn.Parameter(eps * torch.randn(1, 3, device=c.device))
        self._xyz = torch.nn.Parameter(eps * torch.randn(1, 3, device=c.device))

    def forward(self):
        pose = self.rt_inv @ self.pose
        img = render(self.subject, self.k_inv, pose, self.sdd, self.height, self.width)
        return img.reshape(1, 1, self.height, self.width)

    @property
    def pose(self):
        R = so3_exp_map(self._rot)
        t = torch.einsum("bij,bj->bi", R, self._xyz)
        T = torch.eye(4, device=self._rot.device)[None]
        T[:, :3, :3] = R
        T[:, :3, 3] = t
        return self.pivot @ T @ self.pivot_inv
