from pathlib import Path

import torch
from torchio import ScalarImage, LabelMap


class Subject(torch.nn.Module):
    def __init__(
        self,
        imagedata: torch.Tensor,
        labeldata: torch.Tensor,
        world_to_voxel: torch.Tensor,
        voxel_to_grid: torch.Tensor,
        isocenter: torch.Tensor,
    ):
        super().__init__()
        self.register_buffer("image", imagedata)
        self.register_buffer("label", labeldata)
        self.register_buffer("world_to_voxel", world_to_voxel)
        self.register_buffer("voxel_to_grid", voxel_to_grid)
        self.register_buffer("isocenter", isocenter)

        self.n_classes = int(self.label.max().item()) + 1

    @classmethod
    def from_filepath(
        cls,
        imagepath: str | Path,
        labelpath: str | Path = None,
        convert_to_mu: bool = True,
        mu_water: float = 0.019,
    ):
        image = ScalarImage(imagepath)
        label = LabelMap(labelpath) if labelpath is not None else None

        # Load the affine matrices
        world_to_voxel = torch.linalg.inv(torch.from_numpy(image.affine)).to(dtype=torch.float32)

        # Load the image data
        imagedata = cls.fixdim(image.data)
        if convert_to_mu:
            imagedata = cls.hu_to_mu(imagedata, mu_water)

        # Load the label data
        labeldata = cls.fixdim(label.data.to(imagedata)) if label is not None else torch.zeros_like(imagedata)

        # Get the isocenter
        isocenter = torch.tensor(image.get_center()).to(dtype=torch.float32)

        # Precompute voxel-to-grid transform for grid_sample
        *_, D, H, W = imagedata.shape
        dims = torch.tensor([W - 1, H - 1, D - 1], dtype=torch.float32)
        scale = 2.0 / dims
        voxel_to_grid = torch.eye(4, dtype=torch.float32)
        voxel_to_grid[0, 0] = scale[0]
        voxel_to_grid[1, 1] = scale[1]
        voxel_to_grid[2, 2] = scale[2]
        voxel_to_grid[:3, 3] = -1.0

        return cls(imagedata, labeldata, world_to_voxel, voxel_to_grid, isocenter)

    @staticmethod
    def hu_to_mu(data: torch.Tensor, mu_water: float) -> torch.Tensor:
        data = data.clamp(-1000)
        return mu_water * (1 + data / 1000)

    @staticmethod
    def fixdim(data: torch.Tensor) -> torch.Tensor:
        return data.unsqueeze(0).permute(0, 1, -1, -2, -3).contiguous()
