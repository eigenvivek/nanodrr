from pathlib import Path

import torch
from torchio import ScalarImage


class Subject(torch.nn.Module):
    def __init__(
        self,
        imagedata: torch.Tensor,
        # labeldata: torch.Tensor,
        world_to_voxel: torch.Tensor,
        voxel_to_world: torch.Tensor,
        isocenter: torch.Tensor,
        dims: torch.Tensor,
    ):
        super().__init__()
        self.register_buffer("image", imagedata)
        # self.register_buffer("label", labeldata)
        self.register_buffer("world_to_voxel", world_to_voxel)
        self.register_buffer("voxel_to_world", voxel_to_world)
        self.register_buffer("isocenter", isocenter)
        self.register_buffer("dims", dims)

    @classmethod
    def from_filepath(
        cls,
        imagepath: str | Path,
        # labelpath: str | Path = None,
        convert_to_mu: bool = True,
        mu_water: float = 0.019,
    ):
        image = ScalarImage(imagepath)

        # Load the affine matrices
        voxel_to_world = torch.from_numpy(image.affine)
        world_to_voxel = torch.inverse(voxel_to_world)
        voxel_to_world = voxel_to_world.to(dtype=torch.float32)
        world_to_voxel = world_to_voxel.to(dtype=torch.float32)

        # Load the data
        data = cls.fixdim(image.data)
        if convert_to_mu:
            data = cls.hu_to_mu(data, mu_water)

        # Get the isocenter
        isocenter = torch.tensor(image.get_center()).to(dtype=torch.float32)

        # Get the dims
        *_, D, H, W = data.shape
        dims = torch.tensor([W - 1, H - 1, D - 1], dtype=torch.float32)

        return cls(data, world_to_voxel, voxel_to_world, isocenter, dims)

    @staticmethod
    def hu_to_mu(data: torch.Tensor, mu_water: float) -> torch.Tensor:
        data = data.clamp(-1000)
        return mu_water * (1 + data / 1000)

    @staticmethod
    def fixdim(data: torch.Tensor) -> torch.Tensor:
        return data.unsqueeze(0).permute(0, 1, -1, -2, -3).contiguous()
