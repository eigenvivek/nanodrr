from pathlib import Path

import torch
from jaxtyping import Float
from torchio import ScalarImage, LabelMap


class Subject(torch.nn.Module):
    def __init__(
        self,
        imagedata: Float[torch.Tensor, "1 1 D H W"],
        labeldata: Float[torch.Tensor, "1 1 D H W"],
        voxel_to_world: Float[torch.Tensor, "4 4"],
        world_to_voxel: Float[torch.Tensor, "4 4"],
        voxel_to_grid: Float[torch.Tensor, "4 4"],
        isocenter: Float[torch.Tensor, "3"],
    ) -> None:
        super().__init__()
        self.register_buffer("image", imagedata)
        self.register_buffer("label", labeldata)
        self.register_buffer("world_to_grid", voxel_to_grid @ world_to_voxel)
        self.register_buffer("isocenter", isocenter)

        self.register_buffer("voxel_to_world", voxel_to_world)
        self.register_buffer("world_to_voxel", world_to_voxel)
        self.register_buffer("voxel_to_grid", voxel_to_grid)

        self.n_classes = int(self.label.max().item()) + 1

    @classmethod
    def from_filepath(
        cls,
        imagepath: str | Path,
        labelpath: str | Path | None = None,
        convert_to_mu: bool = True,
        mu_water: float = 0.019,
    ):
        image = ScalarImage(imagepath)
        label = LabelMap(labelpath) if labelpath is not None else None
        return cls.from_images(image, label, convert_to_mu, mu_water)

    @classmethod
    def from_images(
        cls,
        image: ScalarImage,
        label: LabelMap | None = None,
        convert_to_mu: bool = True,
        mu_water: float = 0.019,
    ):
        # Load the affine matrix
        voxel_to_world = torch.from_numpy(image.affine)
        world_to_voxel = voxel_to_world.inverse().to(dtype=torch.float32)
        voxel_to_world = voxel_to_world.to(dtype=torch.float32)

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
        voxel_to_grid[:3, :3] = torch.diag(scale)
        voxel_to_grid[:3, 3] = -1.0

        return cls(imagedata, labeldata, voxel_to_world, world_to_voxel, voxel_to_grid, isocenter)

    @staticmethod
    def hu_to_mu(data: Float[torch.Tensor, "1 1 D H W"], mu_water: float) -> Float[torch.Tensor, "1 1 D H W"]:
        data = data.clamp(-1000)
        return mu_water * (1 + data / 1000)

    @staticmethod
    def fixdim(data: Float[torch.Tensor, "1 W H D"]) -> Float[torch.Tensor, "1 1 D H W"]:
        return data.unsqueeze(0).permute(0, 1, -1, -2, -3).contiguous()
