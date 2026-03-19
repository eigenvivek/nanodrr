import numpy as np
import torch
import warp as wp
from jaxtyping import Float

from nanodrr.data import Subject

wp.init()


@wp.struct
class WarpSubject:
    image: wp.Texture3D
    label: wp.Texture3D
    world2voxel: wp.mat44
    nx: int
    ny: int
    nz: int


@wp.func
def _make_tgt(
    k_inv: wp.mat33,
    sdd: float,
    ray_idx: int,
    width: int,
) -> wp.vec3:
    u = float(ray_idx % width) + 0.5
    v = float(ray_idx // width) + 0.5
    uv = wp.vec3(u, v, 1.0)
    return sdd * wp.mul(k_inv, uv)


@wp.kernel
def _render(
    subject: WarpSubject,
    k_inv: wp.array(dtype=wp.mat33),
    sdd: wp.array(dtype=float),
    height: int,
    width: int,
    cam2voxel: wp.array(dtype=wp.mat44),
    n_samples: int,
    output: wp.array(dtype=float, ndim=3),  # [B, n_labels, n_pixels]
):
    tid = wp.tid()
    n_pixels = height * width
    b = tid // n_pixels
    ray_idx = tid % n_pixels

    source_cam = wp.vec3(0.0, 0.0, 0.0)
    target_cam = _make_tgt(k_inv[b], sdd[b], ray_idx, width)

    n_intervals = float(n_samples - 1)
    step_size = wp.length(target_cam * (1.0 / n_intervals))

    source = wp.transform_point(cam2voxel[b], source_cam)
    target = wp.transform_point(cam2voxel[b], target_cam)

    delta = (target - source) * (1.0 / n_intervals)

    t_min = float(0.0)
    t_max = float(1.0)
    for axis in range(3):
        d = target[axis] - source[axis]
        bound = float(subject.nx) if axis == 0 else (float(subject.ny) if axis == 1 else float(subject.nz))
        if wp.abs(d) > 1e-6:
            ta = (float(0.0) - source[axis]) / d
            tb = (bound - source[axis]) / d
            t_min = wp.max(t_min, wp.min(ta, tb))
            t_max = wp.min(t_max, wp.max(ta, tb))

    if t_min >= t_max:
        return

    i_min = int(wp.max(wp.floor(t_min * n_intervals), float(0)))
    i_max = int(wp.min(wp.ceil(t_max * n_intervals), n_intervals))

    for i in range(i_min, i_max):
        pos = source + delta * float(i)  # Direct computation (fixes gradient bug)
        intensity = wp.texture_sample(subject.image, pos, dtype=float) * step_size
        label = int(wp.texture_sample(subject.label, pos, dtype=float))
        output[b, label, ray_idx] += intensity


# --- Subject conversion ---

_wp_subject_cache: dict[int, tuple[Subject, WarpSubject]] = {}


def _to_wp_subject(s: Subject) -> WarpSubject:
    key = id(s)
    if key in _wp_subject_cache:
        return _wp_subject_cache[key][1]

    image_np = s.image.cpu().squeeze().numpy()
    label_np = s.label.cpu().squeeze().numpy()
    D, H, W = image_np.shape

    wp_subject = WarpSubject()
    wp_subject.image = wp.Texture3D(
        image_np,
        normalized_coords=False,
        address_mode=wp.TextureAddressMode.BORDER,
        filter_mode=wp.TextureFilterMode.LINEAR,
    )
    wp_subject.label = wp.Texture3D(
        label_np,
        normalized_coords=False,
        address_mode=wp.TextureAddressMode.BORDER,
        filter_mode=wp.TextureFilterMode.CLOSEST,
    )
    wp_subject.world2voxel = wp.mat44(s.world_to_voxel.cpu().numpy())
    wp_subject.nz = D
    wp_subject.ny = H
    wp_subject.nx = W

    _wp_subject_cache[key] = (s, wp_subject)
    return wp_subject


# --- Caches ---

_subject_registry: dict[int, WarpSubject] = {}
_k_inv_cache: dict[tuple, wp.array] = {}
_sdd_cache: dict[tuple, wp.array] = {}
_output_cache: dict[tuple, tuple[wp.array, torch.Tensor]] = {}
_tape_cache: dict[int, tuple] = {}


def _get_wp_k_inv(k_inv_t: torch.Tensor) -> wp.array:
    key = (k_inv_t.data_ptr(), k_inv_t.shape[0])
    if key not in _k_inv_cache:
        B = k_inv_t.shape[0]
        _k_inv_cache[key] = wp.array(
            [wp.mat33(k_inv_t[b].cpu().numpy()) for b in range(B)],
            dtype=wp.mat33,
        )
    return _k_inv_cache[key]


def _get_wp_sdd(sdd_t: torch.Tensor) -> wp.array:
    key = (sdd_t.data_ptr(), sdd_t.shape[0])
    if key not in _sdd_cache:
        B = sdd_t.shape[0]
        _sdd_cache[key] = wp.array(
            [float(sdd_t[b].item()) for b in range(B)],
            dtype=float,
        )
    return _sdd_cache[key]


def _get_wp_output(subject_ptr: int, B: int, height: int, width: int, n_labels: int) -> tuple[wp.array, torch.Tensor]:
    cache_key = (subject_ptr, B, height, width, n_labels)
    if cache_key not in _output_cache:
        wp_output = wp.zeros((B, n_labels, height * width), dtype=float)
        torch_output = wp.to_torch(wp_output)
        _output_cache[cache_key] = (wp_output, torch_output)
    wp_output, torch_output = _output_cache[cache_key]
    wp_output.zero_()
    return wp_output, torch_output


def _get_world2voxel_t(subject: WarpSubject, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(
        np.array(subject.world2voxel).reshape(4, 4),
        dtype=dtype,
        device=device,
    )


# --- Custom ops ---


@torch.library.custom_op("drr::render_forward", mutates_args=())
def _render_forward(
    cam2world_t: torch.Tensor,  # [B, 4, 4]
    k_inv_t: torch.Tensor,  # [B, 3, 3]
    sdd_t: torch.Tensor,  # [B]
    height: int,
    width: int,
    n_samples: int,
    n_labels: int,
    subject_ptr: int,
) -> tuple[torch.Tensor, int]:  # Return (output, tape_ptr)
    subject = _subject_registry[subject_ptr]
    B = cam2world_t.shape[0]

    world2voxel_t = _get_world2voxel_t(subject, cam2world_t.dtype, cam2world_t.device)
    cam2voxel_t = (world2voxel_t.unsqueeze(0).expand(B, -1, -1) @ cam2world_t).contiguous()

    wp_cam2voxel = wp.from_torch(cam2voxel_t, dtype=wp.mat44, requires_grad=True)
    wp_k_inv = _get_wp_k_inv(k_inv_t)
    wp_sdd = _get_wp_sdd(sdd_t)
    wp_output, torch_output = _get_wp_output(subject_ptr, B, height, width, n_labels)

    # Record tape in forward pass
    tape = wp.Tape()
    with tape:
        wp.launch(
            kernel=_render,
            dim=B * height * width,
            inputs=[subject, wp_k_inv, wp_sdd, height, width, wp_cam2voxel, n_samples],
            outputs=[wp_output],
        )

    # Store tape and arrays for backward
    tape_ptr = id(tape)
    _tape_cache[tape_ptr] = (tape, wp_cam2voxel, wp_output, world2voxel_t)

    return torch_output, tape_ptr


@_render_forward.register_fake
def _(cam2world_t, k_inv_t, sdd_t, height, width, n_samples, n_labels, subject_ptr):
    B = cam2world_t.shape[0]
    return torch.empty(B, n_labels, height * width, dtype=torch.float32), 0


@torch.library.custom_op("drr::render_backward", mutates_args=())
def _render_backward(
    tape_ptr: int,
    adj_output: torch.Tensor,
) -> torch.Tensor:
    # Retrieve cached tape and arrays (no forward re-execution!)
    tape, wp_cam2voxel, wp_output, world2voxel_t = _tape_cache[tape_ptr]

    # Set output gradient and run backward
    wp_output.grad = wp.from_torch(adj_output.contiguous(), requires_grad=False)
    tape.backward()

    # Transform gradient to cam2world space
    adj_cam2voxel = wp.to_torch(wp_cam2voxel.grad)
    adj_cam2world = world2voxel_t.T.unsqueeze(0) @ adj_cam2voxel

    # Clean up
    del _tape_cache[tape_ptr]
    tape.reset()

    return adj_cam2world


@_render_backward.register_fake
def _(tape_ptr, adj_output):
    B = adj_output.shape[0]
    return torch.empty(B, 4, 4, dtype=torch.float32)


def _backward(ctx, adj_output, _):  # Two gradients: for output and tape_ptr
    adj_cam2world = _render_backward(ctx.tape_ptr, adj_output)
    # Return gradients for: cam2world_t, k_inv_t, sdd_t, height, width, n_samples, n_labels, subject_ptr
    return adj_cam2world, None, None, None, None, None, None, None


def _setup_context(ctx, inputs, output):  # 'output' receives the tuple
    ctx.cam2world_t, ctx.k_inv_t, ctx.sdd_t, ctx.height, ctx.width, ctx.n_samples, ctx.n_labels, ctx.subject_ptr = (
        inputs
    )
    out_tensor, ctx.tape_ptr = output  # Unpack the tuple (output, tape_ptr)


_render_forward.register_autograd(_backward, setup_context=_setup_context)


# --- Public API ---


def render(
    subject: Subject,
    k_inv: Float[torch.Tensor, "B 3 3"],
    rt_inv: Float[torch.Tensor, "B 4 4"],
    sdd: Float[torch.Tensor, "B"],
    height: int,
    width: int,
    n_samples: int = 500,
    src: Float[torch.Tensor, "B (H W) 3"] | None = None,
    tgt: Float[torch.Tensor, "B (H W) 3"] | None = None,
    n_labels: int | None = None,
) -> Float[torch.Tensor, "B n_labels H W"]:
    """Differentiable ray marching through a volume and optional labelmap.

    Casts rays from an X-ray source through a 3D volume and integrates sampled
    intensities along each ray to produce a synthetic radiograph. The integration
    is performed per-structure, yielding one channel per label class.

    Args:
        subject: The volume to render.
        k_inv: Inverse intrinsic camera matrix. Shape ``(B, 3, 3)``.
        rt_inv: Camera-to-world extrinsic matrix. Shape ``(B, 4, 4)``.
        sdd: Source-to-detector distance in mm. Shape ``(B,)``.
        height: Output image height in pixels.
        width: Output image width in pixels.
        n_samples: Number of samples along each ray.
        src: Ignored. Accepted for API compatibility with the PyTorch backend.
        tgt: Ignored. Accepted for API compatibility with the PyTorch backend.
            Ray endpoints are computed per-thread inside the kernel via ``_make_tgt``.
        n_labels: Number of label classes. Defaults to ``subject.n_classes``.

    Returns:
        Rendered radiograph of shape ``(B, n_labels, H, W)``.
    """
    if n_labels is None:
        n_labels = subject.n_classes

    wp_subject = _to_wp_subject(subject)
    subject_ptr = id(wp_subject)
    _subject_registry[subject_ptr] = wp_subject

    B = rt_inv.shape[0]
    k_inv = k_inv.expand(B, -1, -1).contiguous()
    sdd = sdd.expand(B).contiguous()

    out, _ = _render_forward(  # Unpack tuple (output, tape_ptr)
        rt_inv,
        k_inv,
        sdd,
        height,
        width,
        n_samples,
        n_labels,
        subject_ptr,
    )

    return out.reshape(-1, n_labels, height, width)
