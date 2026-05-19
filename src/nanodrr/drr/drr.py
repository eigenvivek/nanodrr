import torch
from jaxtyping import Float

from .renderer import render, _make_tgt, _make_src
from ..camera import make_k_inv
from ..data import Subject


class DRR(torch.nn.Module):
    """Digitally reconstructed radiograph (DRR) generator module.

    Encapsulates the intrinsic camera parameters needed to cast rays from an
    X-ray source through a 3D volume. Once initialized, call `forward` with a
    `Subject` and extrinsic pose to produce a synthetic radiograph.

    The intrinsic parameters (`k_inv`, `sdd`, `height`, `width`) are stored as
    buffers or attributes so they travel with the module across devices and are
    included in `state_dict`.

    Attributes:
        _intrinsic_params: Set of parameter names that define the camera
            intrinsics.

    Args:
        k_inv: Inverse intrinsic camera matrix. Maps pixel coordinates to
            camera-space ray directions.
        sdd: Source-to-detector distance, i.e., the distance from the X-ray
            source to the imaging plane.
        height: Output image height in pixels.
        width: Output image width in pixels.
        orthographic: If True, use orthographic projection (parallel rays).
            If False (default), use perspective projection (point source).
    """

    k_inv: Float[torch.Tensor, "B 3 3"]
    sdd: Float[torch.Tensor, "B"]
    src: Float[torch.Tensor, "B (H W) 3"] | Float[torch.Tensor, "B 1 3"]
    tgt: Float[torch.Tensor, "B (H W) 3"]

    _intrinsic_params = {"k_inv", "sdd", "height", "width", "orthographic"}

    def __init__(
        self,
        k_inv: Float[torch.Tensor, "B 3 3"],
        sdd: Float[torch.Tensor, "B"],
        height: int,
        width: int,
        orthographic: bool = False,
    ):
        super().__init__()
        self.register_buffer("k_inv", k_inv)
        self.register_buffer("sdd", sdd)
        self.height = height
        self.width = width
        self.orthographic = orthographic
        self._compute_src_tgt()

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name in self._intrinsic_params and all(hasattr(self, p) for p in self._intrinsic_params):
            self._compute_src_tgt()

    def _compute_src_tgt(self):
        tgt = _make_tgt(self.k_inv, self.sdd, self.height, self.width, self.k_inv.device, self.k_inv.dtype)
        src = _make_src(self.orthographic, tgt, self.sdd)
        self.register_buffer("tgt", tgt, persistent=False)
        self.register_buffer("src", src, persistent=False)

    def forward(
        self,
        subject: Subject,
        rt_inv: Float[torch.Tensor, "B 4 4"],
        n_samples: int = 500,
    ):
        return render(
            subject,
            self.k_inv,
            rt_inv,
            self.sdd,
            self.height,
            self.width,
            n_samples,
            self.src,
            self.tgt,
        )

    @classmethod
    def from_carm_intrinsics(
        cls,
        sdd: float,
        delx: float,
        dely: float,
        x0: float,
        y0: float,
        height: int,
        width: int,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        orthographic: bool = False,
    ):
        k_inv = make_k_inv(sdd, delx, dely, x0, y0, height, width).to(dtype=dtype, device=device)
        sdd_tensor = torch.tensor([sdd], dtype=dtype, device=device)
        return cls(k_inv, sdd_tensor, height, width, orthographic)
