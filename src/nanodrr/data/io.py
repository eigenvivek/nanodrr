from pathlib import Path

import torch
from jaxtyping import Float
from torchio import LabelMap, ScalarImage

from .preprocess import hu_to_mu


class Subject(torch.nn.Module):
    """CT volume and (optional) labelmap compatible with [`torch.nn.functional.grid_sample`](https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html).

    Fuses all spatial transforms for sampling (world → voxel → grid) so downstream
    rendering only needs cheap matmuls.
    """

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
    ) -> "Subject":
        """Load a subject from NIfTI (or any TorchIO-supported) file paths.

        Args:
            imagepath: Path to the CT volume.
            labelpath: Optional path to a label map.
            convert_to_mu: Convert Hounsfield units to linear attenuation.
            mu_water: Linear attenuation coefficient of water (mm⁻¹).
        """
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
    ) -> "Subject":
        """Construct a subject from TorchIO image objects.

        Args:
            image: CT volume as a ``ScalarImage``.
            label: Optional segmentation as a ``LabelMap``.
            convert_to_mu: Convert Hounsfield units to linear attenuation.
            mu_water: Linear attenuation coefficient of water (mm⁻¹).
        """
        # Affine: invert in float64 for numerical accuracy, then downcast
        voxel_to_world_f64 = torch.from_numpy(image.affine).to(torch.float64)
        voxel_to_world = voxel_to_world_f64.to(torch.float32)
        world_to_voxel = voxel_to_world_f64.inverse().to(torch.float32)

        # Image data
        imagedata = cls._to_bcdhw(image.data).to(torch.float32)
        if convert_to_mu:
            imagedata = hu_to_mu(imagedata, mu_water)

        # Label data
        if label is not None:
            labeldata = cls._to_bcdhw(label.data).to(torch.float32)
        else:
            labeldata = torch.zeros_like(imagedata)

        isocenter = torch.tensor(image.get_center(), dtype=torch.float32)
        voxel_to_grid = cls._make_voxel_to_grid(imagedata.shape)

        return cls(
            imagedata,
            labeldata,
            voxel_to_world,
            world_to_voxel,
            voxel_to_grid,
            isocenter,
        )

    @staticmethod
    def _to_bcdhw(
        data: Float[torch.Tensor, "1 W H D"],
    ) -> Float[torch.Tensor, "1 1 D H W"]:
        """Reshape TorchIO's (1, W, H, D) layout to (1, 1, D, H, W)."""
        return data.unsqueeze(0).permute(0, 1, 4, 3, 2).contiguous()

    @staticmethod
    def _make_voxel_to_grid(shape: torch.Size) -> Float[torch.Tensor, "4 4"]:
        """Build the voxel → [-1, 1] grid transform used by ``grid_sample``.

        Args:
            shape: (1, 1, D, H, W) volume shape.
        """
        *_, D, H, W = shape
        scale = 2.0 / torch.tensor([W - 1, H - 1, D - 1], dtype=torch.float32)
        mat = torch.eye(4, dtype=torch.float32)
        mat[:3, :3] = torch.diag(scale)
        mat[:3, 3] = -1.0
        return mat
