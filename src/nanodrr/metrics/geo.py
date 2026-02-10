import torch
from jaxtyping import Float
from roma import rotmat_geodesic_distance
from torch import Tensor


class DoubleGeodesicSE3(torch.nn.Module):
    """Calculate the angular and translational geodesics between two SE(3) transformation matrices."""

    def __init__(
        self,
        sdd: float,
        eps: float = 1e-4,
    ):
        super().__init__()
        self.sdr = sdd / 2
        self.eps = eps

    def forward(
        self,
        pose_1: Float[Tensor, "B 4 4"],
        pose_2: Float[Tensor, "B 4 4"],
    ) -> tuple[Float[Tensor, "B"], Float[Tensor, "B"], Float[Tensor, "B"]]:
        r1, t1 = pose_1[:, :3, :3], pose_1[:, :3, 3]
        r2, t2 = pose_2[:, :3, :3], pose_2[:, :3, 3]
        t1 = torch.einsum("bij,bj->bi", r1.transpose(-1, -2), t1)
        t2 = torch.einsum("bij,bj->bi", r2.transpose(-1, -2), t2)
        rot = self._rot_geodesic(r1, r2)
        xyz = self._xyz_geodesic(t1, t2)
        dou = (rot.square() + xyz.square() + self.eps).sqrt()
        return rot, xyz, dou

    def _rot_geodesic(
        self,
        r1: Float[Tensor, "B 3 3"],
        r2: Float[Tensor, "B 3 3"],
    ) -> Float[Tensor, "B"]:
        return self.sdr * rotmat_geodesic_distance(r1, r2)

    def _xyz_geodesic(
        self,
        t1: Float[Tensor, "B 3"],
        t2: Float[Tensor, "B 3"],
    ) -> Float[Tensor, "B"]:
        return (t1 - t2).norm(dim=-1)
