from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import torch
from jaxtyping import Float
from PIL import Image

from ..drr.render import _make_tgt
from ..geometry import transform_point
from ..plot import plot_drr


def make_cameras(
    k_inv: Float[torch.Tensor, "B 3 3"],
    rt_inv: Float[torch.Tensor, "B 4 4"],
    sdd: Float[torch.Tensor, "B"],
    height: int,
    width: int,
    img: Float[torch.Tensor, "B C H W"] | None = None,
    frustum_size: float = 0.125,
) -> list[dict]:
    B = rt_inv.shape[0]
    device, dtype = rt_inv.device, rt_inv.dtype

    src_cam = torch.zeros(B, 1, 3, device=device, dtype=dtype)
    tgt_cam = _make_tgt(k_inv, sdd, height, width, device, dtype)

    src = transform_point(rt_inv, src_cam).squeeze(1).cpu().detach().numpy()
    tgt = transform_point(rt_inv, tgt_cam).reshape(B, height, width, 3).cpu().detach().numpy()

    return [
        {
            "detector": _detector(tgt[b]) if img is not None else None,
            "texture": _texture(img[b : b + 1]) if img is not None else None,
            "camera": _frustum(src[b], tgt[b], frustum_size),
            "principal_ray": pv.Line(src[b], tgt[b].mean(axis=(0, 1))),
        }
        for b in range(B)
    ]


def _texture(img: Float[torch.Tensor, "1 C H W"]) -> pv.Texture:
    ax = plot_drr(img, ticks=False)[0]
    ax.set_axis_off()
    ax.figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
    buf = BytesIO()
    ax.figure.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(ax.figure)
    return pv.numpy_to_texture(np.array(Image.open(buf).convert("RGB")))


def _detector(target: np.ndarray) -> pv.PolyData:
    det = pv.StructuredGrid(target[..., 0], target[..., 1], target[..., 2])
    det.texture_map_to_plane(origin=target[-1, 0], point_u=target[-1, -1], point_v=target[0, 0], inplace=True)
    surf = det.extract_surface(algorithm="dataset_surface")
    tcoords = surf.GetPointData().GetTCoords()  # preserve UVs across merge
    merged = surf.merge(surf.flip_faces())
    merged.GetPointData().SetTCoords(tcoords)
    return merged


def _frustum(source: np.ndarray, target: np.ndarray, size: float) -> pv.PolyData:
    vertices = np.stack(
        [
            source + size * (target[0, 0] - source),
            source + size * (target[-1, 0] - source),
            source + size * (target[-1, -1] - source),
            source + size * (target[0, -1] - source),
            source,
        ]
    )
    faces = np.hstack([[4, 0, 3, 2, 1], [3, 0, 4, 1], [3, 1, 4, 2], [3, 0, 3, 4], [3, 2, 4, 3]])
    mesh = pv.PolyData(vertices, faces)
    return mesh.merge(mesh.flip_faces())
