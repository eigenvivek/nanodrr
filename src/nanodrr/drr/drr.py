import torch
from jaxtyping import Float

from .render import render, _make_tgt
from ..camera import make_k_inv
from ..data import Subject


class DRR(torch.nn.Module):
    _intrinsic_params = {"k_inv", "sdd", "height", "width"}

    def __init__(
        self,
        k_inv: Float[torch.Tensor, "B 3 3"],
        sdd: Float[torch.Tensor, "B"],
        height: int,
        width: int,
    ):
        super().__init__()
        self.register_buffer("k_inv", k_inv)
        self.register_buffer("sdd", sdd)
        self.height = height
        self.width = width
        self._compute_tgt()

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name in self._intrinsic_params and all(hasattr(self, p) for p in self._intrinsic_params):
            self._compute_tgt()

    def _compute_tgt(self):
        tgt = _make_tgt(self.k_inv, self.sdd, self.height, self.width, self.k_inv.device, self.k_inv.dtype)
        self.register_buffer("tgt", tgt, persistent=False)

    def forward(
        self,
        subject: Subject,
        rt_inv: Float[torch.Tensor, "B 4 4"],
        n_samples: int = 500,
        align_corners: bool = True,
    ):
        return render(
            subject,
            self.k_inv,
            rt_inv,
            self.sdd,
            self.height,
            self.width,
            n_samples,
            align_corners,
            None,
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
    ):
        k_inv = make_k_inv(sdd, delx, dely, x0, y0, height, width).to(dtype=dtype, device=device)
        sdd = torch.tensor([sdd], dtype=dtype, device=device)
        return cls(k_inv, sdd, height, width)
