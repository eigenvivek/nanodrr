---
icon: lucide/git-compare-arrows
---

# Changes from DiffDRR

## Modularity

`nanodrr` explicitly disentangles all rendering operations from any specific subject, which is the current paradigm in `DiffDRR`. It does this in both the traditional object-oriented rendering interface, as well as a new functional interface.

### Functional interface

`nanodrr` prioritizes functional DRR rendering ([`nanodrr.drr.render`](api/drr.md#nanodrr.drr.render)):

``` python hl_lines="2-5"
def render(
    subject: Subject,
    k_inv: Float[Tensor, "B 3 3"],
    rt_inv: Float[Tensor, "B 4 4"],
    sdd: Float[Tensor, "B"],
    height: int,
    width: int,
    n_samples: int = 500,
    align_corners: bool = True,
    src: Float[Tensor, "B (H W) 3"] | None = None,
    tgt: Float[Tensor, "B (H W) 3"] | None = None,
) -> Float[Tensor, "B C H W"]
```

This function lets you freely exchange the [`nanodrr.data.Subject`](api/data.md#nanodrr.data.Subject), (inverse) [intrinsic matrices](api/camera.md#intrinsics), and (inverse) [extrinsic matrices](api/camera.md#extrinsics) at runtime.

### Object-oriented interface

`nanodrr` also provides a class object for DRR rendering with fixed intrinsic parameters ([`nanodrr.drr.DRR`](api/drr.md#nanodrr.drr.DRR)):

``` python
class DRR(torch.nn.Module):
    def __init__(
        self,
        k_inv: Float[torch.Tensor, "B 3 3"],
        sdd: Float[torch.Tensor, "B"],
        height: int,
        width: int,
    ) -> None:
```

As highlighted in the [basic usage tutorial](tutorials/demo.md), different [`nanodrr.data.Subject`](api/data.md#nanodrr.data.Subject)s can be passed to this object at runtime:

```python
drr = DRR(k_inv, sdd, height, width)
drr(subject, rt_inv)
```

## Subjects

### HU conversion

To convert Hounsfield Units (HUs) to linear attenuation coefficients (LAC), often denoted as $\mu$, [`nanodrr.data.preprocess.hu_to_mu`](api/data.md#nanodrr.data.preprocess) implements bilinear scaling with air-water model for HU ≤ 0 and water-bone model for HU > 0:

$$
\mu = \begin{cases}
    \mu_{\mathrm{water}} \cdot \left(\frac{\mathrm{HU}}{1000} + 1\right) & \text{if } \mathrm{HU} \leq 0 \\
    \mu_{\mathrm{water}} + (\mu_{\mathrm{bone}} - \mu_{\mathrm{water}}) \cdot \frac{\mathrm{HU}}{\mathrm{HU}_{\mathrm{bone}}} & \text{if } \mathrm{HU} > 0
\end{cases}
$$

### Coordinate system

Unlike `DiffDRR`, [`Subject`](api/data.md#nanodrr.data.Subject)s in `nanodrr` *are not centered at the world origin by default*. Instead, we use the volume's affine matrix (`voxel_to_world`) to place each volume in world coordinates. 

``` python hl_lines="6"
class Subject(torch.nn.Module):
    def __init__(
        self,
        imagedata: Float[torch.Tensor, "1 1 D H W"],
        labeldata: Float[torch.Tensor, "1 1 D H W"],
        voxel_to_world: Float[torch.Tensor, "4 4"],
        world_to_voxel: Float[torch.Tensor, "4 4"],
        voxel_to_grid: Float[torch.Tensor, "4 4"],
        isocenter: Float[torch.Tensor, "3"],
        max_label: int | None = None,
    ) -> None:
```

## Fused rendering transforms

The function that samples points in the volume, [`torch.nn.functional.grid_sample`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html), requires normalized coordinates in `[-1, 1]`. In `DiffDRR`, we perform this with many division ops, which are slow. In `nanodrr`, we parameterize this coordinate transform as an affine matrix:

``` python
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
```

Note that this matrix is constant for every subject (voxel dimensions don't change). This lets us fuse many of the coordinate transforms needed for rendering into a single $4\times4$ matrix (`world → voxel → grid sample`), where `world → voxel` is given by the inverse affine matrix (`world_to_voxel`).

## Traditional camera geometry

All projective geometry is implemented internally using the standard [Hartley and Zisserman](https://www.cambridge.org/core/books/multiple-view-geometry-in-computer-vision/0B6F289C78B2B23F596CAA76D3D43F7A) pinhole camera formulation (see the documentation for how we handle [intrinsics](api/camera.md#intrinsics) and [extrinsics](api/camera.md#extrinsics)).

## Miscellaneous

- `nanodrr` works with `torch.compile` and automatic mixed precision
- Parameters and returns are extensively annotated
    - Tensor types and shapes are annotated with `jaxtyping`
