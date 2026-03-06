from pathlib import Path

import torch
from jaxtyping import Float
from torchio import LabelMap, ScalarImage

from .preprocess import HU_BONE, MU_BONE, MU_WATER, hu_to_mu

_MU_FIELDS = frozenset({"convert_to_mu", "mu_water", "mu_bone", "hu_bone"})


def _validate_mu_kwargs(mu: dict) -> None:
    unknown = mu.keys() - _MU_FIELDS
    if unknown:
        raise TypeError(f"Unknown mu param(s): {', '.join(repr(k) for k in unknown)}")


class _CachedParam:
    """Descriptor for scalar params that invalidates `_image_mu_cache` on change.

    Accessing this descriptor on the class itself returns the descriptor object;
    accessing it on an instance returns the stored value (or the default).
    """

    def __init__(self, default) -> None:
        self.default = default

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = f"_{name}"

    def __get__(self, obj: "Subject | None", objtype: type | None = None):
        if obj is None:
            return self
        return getattr(obj, self.name, self.default)

    def __set__(self, obj: "Subject", value) -> None:
        if getattr(obj, self.name, self.default) != value:
            setattr(obj, self.name, value)
            if hasattr(obj, "_image_mu_cache"):
                obj._image_mu_cache = None


class Subject(torch.nn.Module):
    """Wrapper for a CT volume and (optional) labelmap that is compatible with torch's
    [`grid_sample`](https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html).

    Fuses all spatial transforms required for sampling (`world → voxel → grid`) so
    that rendering only needs to perform a single matmul.

    Updating any of `convert_to_mu`, `mu_water`, `mu_bone`, or `hu_bone` automatically
    recomputes the HU to linear attenuation coefficient (LAC) conversion.
    """

    convert_to_mu: bool = _CachedParam(True)
    mu_water: float = _CachedParam(MU_WATER)
    mu_bone: float = _CachedParam(MU_BONE)
    hu_bone: float = _CachedParam(HU_BONE)

    def __init__(
        self,
        imagedata: Float[torch.Tensor, "1 1 D H W"],
        labeldata: Float[torch.Tensor, "1 1 D H W"],
        voxel_to_world: Float[torch.Tensor, "4 4"],
        world_to_voxel: Float[torch.Tensor, "4 4"],
        voxel_to_grid: Float[torch.Tensor, "4 4"],
        isocenter: Float[torch.Tensor, "3"],
        max_label: int | None = None,
        **mu,
    ) -> None:
        _validate_mu_kwargs(mu)
        super().__init__()
        self.register_buffer("_image_hu", imagedata)
        self.register_buffer("_image_mu_cache", None, persistent=False)
        self.register_buffer("label", labeldata)
        self.register_buffer("world_to_grid", voxel_to_grid @ world_to_voxel)
        self.register_buffer("isocenter", isocenter)
        self.register_buffer("voxel_to_world", voxel_to_world)
        self.register_buffer("world_to_voxel", world_to_voxel)
        self.register_buffer("voxel_to_grid", voxel_to_grid)

        for k, v in mu.items():
            setattr(self, k, v)

        if max_label is not None:
            self.n_classes = int(max_label + 1)
        else:
            self.n_classes = int(self.label.max().item()) + 1

    @property
    def image(self) -> Float[torch.Tensor, "1 1 D H W"]:
        """Volume in units of LACs (or raw values if `convert_to_mu` is False).
        Result is cached and recomputed only when conversion params change.
        """
        if not self.convert_to_mu:
            return self._image_hu
        if self._image_mu_cache is None:
            self._image_mu_cache = hu_to_mu(self._image_hu, self.mu_water, self.mu_bone, self.hu_bone)
        return self._image_mu_cache

    def to(self, *args, **kwargs) -> "Subject":
        # Preserve the cache across device/dtype transfers by recomputing on the
        # new device rather than discarding it — but only if one existed before.
        had_cache = self._image_mu_cache is not None
        result = super().to(*args, **kwargs)
        if had_cache:
            result._image_mu_cache = hu_to_mu(result._image_hu, result.mu_water, result.mu_bone, result.hu_bone)
        return result

    def set_mu(
        self,
        mu_water: float | torch.Tensor | None = None,
        mu_bone: float | torch.Tensor | None = None,
        hu_bone: float | torch.Tensor | None = None,
    ) -> None:
        """Recompute the LAC cache from new conversion parameters.

        Prefer this over setting `mu_water`, `mu_bone`, and `hu_bone` individually
        when calling under `torch.compile`, as it performs a single tensor op and
        avoids Python-side attribute mutations that would cause graph breaks.
        The stored scalar params (`self.mu_water` etc.) are intentionally *not*
        updated here; they remain as the baseline defaults.
        """
        self._image_mu_cache = hu_to_mu(
            self._image_hu,
            mu_water if mu_water is not None else self.mu_water,
            mu_bone if mu_bone is not None else self.mu_bone,
            hu_bone if hu_bone is not None else self.hu_bone,
        )

    @classmethod
    def from_filepath(
        cls,
        imagepath: str | Path,
        labelpath: str | Path | None = None,
        max_label: int | None = None,
        **mu,
    ) -> "Subject":
        """Load a subject from any TorchIO-supported file path.

        Args:
            imagepath: Path to the CT volume.
            labelpath: Optional path to a label map.
            max_label: Override the maximum label index. If provided, `n_classes`
                is set to `max_label + 1` instead of being inferred from the data.
            **mu: HU → μ conversion params. See `from_images` for details.
        """
        _validate_mu_kwargs(mu)
        image = ScalarImage(imagepath)
        label = LabelMap(labelpath) if labelpath is not None else None
        return cls.from_images(image, label, max_label, **mu)

    @classmethod
    def from_images(
        cls,
        image: ScalarImage,
        label: LabelMap | None = None,
        max_label: int | None = None,
        **mu,
    ) -> "Subject":
        """Construct a subject from TorchIO image objects.

        Args:
            image: CT volume as a `ScalarImage`.
            label: Optional segmentation as a `LabelMap`.
            max_label: Override the maximum label index. If provided, `n_classes`
                is set to `max_label + 1` instead of being inferred from the data.
            **mu: HU → μ conversion params (`convert_to_mu`, `mu_water`, `mu_bone`,
                `hu_bone`). Unspecified keys fall back to class-level defaults.

        Raises:
            TypeError: If any unrecognised key is passed via `**mu`.
        """
        _validate_mu_kwargs(mu)

        # Affine: invert in float64 for numerical accuracy, then downcast
        voxel_to_world_f64 = torch.from_numpy(image.affine).to(torch.float64)
        voxel_to_world = voxel_to_world_f64.to(torch.float32)
        world_to_voxel = voxel_to_world_f64.inverse().to(torch.float32)

        # Store raw HU — conversion is applied lazily via .image
        imagedata = cls._to_bcdhw(image.data).to(torch.float32)

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
            max_label,
            **mu,
        )

    @staticmethod
    def _to_bcdhw(
        data: Float[torch.Tensor, "1 W H D"],
    ) -> Float[torch.Tensor, "1 1 D H W"]:
        """Reshape TorchIO's (1, W, H, D) layout to (1, 1, D, H, W)."""
        return data.unsqueeze(0).permute(0, 1, 4, 3, 2).contiguous()

    @staticmethod
    def _make_voxel_to_grid(shape: torch.Size) -> Float[torch.Tensor, "4 4"]:
        """Build the voxel → [-1, 1] grid transform used by `grid_sample`.

        Args:
            shape: (1, 1, D, H, W) volume shape.
        """
        *_, D, H, W = shape
        scale = 2.0 / torch.tensor([W - 1, H - 1, D - 1], dtype=torch.float32)
        mat = torch.eye(4, dtype=torch.float32)
        mat[:3, :3] = torch.diag(scale)
        mat[:3, 3] = -1.0
        return mat
