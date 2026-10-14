"""Fused Metal ray-marching kernels, compiled at runtime with `torch.mps.compile_shader`.

One thread marches one ray, so the sample grid and per-sample intensities of the `grid_sample` path are
never materialized. The semantics are those of the Triton kernel (see `_triton.py`): rays are clipped to
the volume and sampled at midpoints, with trilinear intensities (zero padded), nearest or per-neighbor
(soft) label routing, and analytic gradients for the camera-to-voxel transform, the ray endpoints, the
step size and the volume. The volume and labelmap may be float32, float16 or bfloat16; they are read in
place and promoted per sample, while geometry and accumulation stay float32.

The kernels are written once and instantiated per volume dtype (`raymarch_fwd_f16`, ...).
"""

import functools
import platform

import torch

from ._fused import FusedBackend, register_fused_op, src_strides

_HEADER = """
#include <metal_stdlib>
using namespace metal;

template <typename T>
inline float fetch(device const T* v, int ix, int iy, int iz, int D, int H, int W) {
    if (ix < 0 || ix >= W || iy < 0 || iy >= H || iz < 0 || iz >= D) return 0.0f;
    return (float)v[(iz * H + iy) * W + ix];
}

// Add `val` to a per-thread running sum for class `cls`, flushing the sum to `o` when the
// class changes: rays cross few class boundaries, so most samples never touch memory
inline void accumulate(device float* o, int N, int C, float step, thread int& cur, thread float& run, int cls, float val) {
    if (cls < 0 || cls >= C || val == 0.0f) return;
    if (cls != cur) {
        o[(long)cur * N] += run * step;
        run = 0.0f;
        cur = cls;
    }
    run += val;
}

// The 8 trilinear neighbors of the cell at (ix, iy, iz), indexed by the bits (dz, dy, dx): c = 4 dz + 2 dy + dx
template <typename T>
inline void fetch8(device const T* v, int ix, int iy, int iz, int D, int H, int W, thread float* out) {
    for (int c = 0; c < 8; ++c) out[c] = fetch(v, ix + (c & 1), iy + ((c >> 1) & 1), iz + (c >> 2), D, H, W);
}

// Trilinear weight of neighbor c at the fractional position f
inline float corner_weight(int c, float3 f) {
    return ((c & 1) ? f.x : 1 - f.x) * ((c & 2) ? f.y : 1 - f.y) * ((c & 4) ? f.z : 1 - f.z);
}

inline float trilerp(thread const float* v, float3 f) {
    float c00 = mix(v[0], v[1], f.x), c01 = mix(v[2], v[3], f.x);
    float c10 = mix(v[4], v[5], f.x), c11 = mix(v[6], v[7], f.x);
    return mix(mix(c00, c01, f.y), mix(c10, c11, f.y), f.z);
}

// Gradient of `trilerp` with respect to the fractional position
inline float3 trilerp_grad(thread const float* v, float3 f) {
    float3 d;
    d.x = ((v[1] - v[0]) * (1 - f.y) + (v[3] - v[2]) * f.y) * (1 - f.z)
        + ((v[5] - v[4]) * (1 - f.y) + (v[7] - v[6]) * f.y) * f.z;
    d.y = ((v[2] - v[0]) * (1 - f.x) + (v[3] - v[1]) * f.x) * (1 - f.z)
        + ((v[6] - v[4]) * (1 - f.x) + (v[7] - v[5]) * f.x) * f.z;
    d.z = ((v[4] - v[0]) * (1 - f.x) + (v[5] - v[1]) * f.x) * (1 - f.y)
        + ((v[6] - v[2]) * (1 - f.x) + (v[7] - v[3]) * f.x) * f.y;
    return d;
}

struct Ray {
    float3 s, e, p0, dp;     // camera-space source and direction; voxel-space origin and direction
    float3 tmin, tmax;       // per-axis ray parameters of the two bounding planes
    float t_lo, t_hi;        // unclamped bounds of the ray inside the box
    float t0, t1;            // the clipped interval [t0, t1] that is sampled
    bool hit;
};

// Load ray `gid` and clip it to the box outside of which the trilinear footprint is identically
// zero. Rays that miss it collapse to a point (`hit` is false)
inline Ray setup_ray(
    device const float* M, device const float* src, device const float* tgt,
    int gid, int b, int n, int D, int H, int W, int src_ray_stride, int src_batch_stride)
{
    Ray r;
    device const float* m = M + b * 12;
    int so = b * src_batch_stride + n * src_ray_stride;
    r.s = float3(src[so], src[so + 1], src[so + 2]);
    r.e = float3(tgt[gid * 3], tgt[gid * 3 + 1], tgt[gid * 3 + 2]) - r.s;
    float3 m0 = float3(m[0], m[1], m[2]), m1 = float3(m[4], m[5], m[6]), m2 = float3(m[8], m[9], m[10]);
    r.p0 = float3(dot(m0, r.s) + m[3], dot(m1, r.s) + m[7], dot(m2, r.s) + m[11]);
    r.dp = float3(dot(m0, r.e), dot(m1, r.e), dot(m2, r.e));

    float3 lo = float3(-1.0f), hi = float3((float)W, (float)H, (float)D);
    float3 safe = select(r.dp, float3(1e-12f), abs(r.dp) < 1e-12f);
    float3 ta = (lo - r.p0) / safe, tb = (hi - r.p0) / safe;
    r.tmin = min(ta, tb);
    r.tmax = max(ta, tb);
    r.t_lo = max(max(r.tmin.x, r.tmin.y), r.tmin.z);
    r.t_hi = min(min(r.tmax.x, r.tmax.y), r.tmax.z);
    r.t0 = clamp(r.t_lo, 0.0f, 1.0f);
    r.t1 = max(clamp(r.t_hi, 0.0f, 1.0f), r.t0);
    r.hit = r.t0 < r.t1;
    return r;
}

"""

_KERNELS = """
// The argument order must match `_forward` below
kernel void raymarch_fwd(
    device const VT* vol     [[buffer(0)]],
    device const VT* lab     [[buffer(1)]],
    device const float* M    [[buffer(2)]],   // [B, 3, 4] camera -> voxel pixel coords
    device const float* src  [[buffer(3)]],   // [B, N or 1, 3]
    device const float* tgt  [[buffer(4)]],   // [B, N, 3]
    device const float* stp  [[buffer(5)]],   // [B, N]
    device float* out        [[buffer(6)]],   // [B, C, N], zero-initialized
    constant int& N          [[buffer(7)]],
    constant int& D          [[buffer(8)]],
    constant int& H          [[buffer(9)]],
    constant int& W          [[buffer(10)]],
    constant int& S          [[buffer(11)]],
    constant int& C          [[buffer(12)]],
    constant int& src_ray_stride   [[buffer(13)]],
    constant int& src_batch_stride [[buffer(14)]],
    constant int& soft       [[buffer(15)]],   // route each trilinear neighbor to its own label (C > 1)
    uint tid [[thread_position_in_grid]])
{
    int gid = (int)tid;
    int b = gid / N, n = gid - b * N;
    Ray r = setup_ray(M, src, tgt, gid, b, n, D, H, W, src_ray_stride, src_batch_stride);
    if (!r.hit) return;

    float inv_s = 1.0f / (float)S;
    device float* o = out + (long)b * C * N + n;
    float step = stp[gid] * (r.t1 - r.t0);  // `stp` is the step [mm] of the unclipped ray
    int cur = 0;
    float run = 0.0f;

    // Sample the midpoints of S equal parts of the clipped interval
    for (int k = 0; k < S; ++k) {
        float t = r.t0 + (((float)k + 0.5f) * inv_s) * (r.t1 - r.t0);
        float3 p = r.p0 + t * r.dp;
        float3 f0 = floor(p);
        float3 f = p - f0;
        int ix = (int)f0.x, iy = (int)f0.y, iz = (int)f0.z;

        float v[8];
        fetch8(vol, ix, iy, iz, D, H, W, v);

        if (soft) {  // each neighbor goes to the channel of its own label
            float l[8];
            fetch8(lab, ix, iy, iz, D, H, W, l);
            for (int c = 0; c < 8; ++c) accumulate(o, N, C, step, cur, run, (int)l[c], v[c] * corner_weight(c, f));
            continue;
        }
        float val = trilerp(v, f);

        int cls = 0;
        if (C > 1) cls = (int)fetch(lab, (int)rint(p.x), (int)rint(p.y), (int)rint(p.z), D, H, W);
        accumulate(o, N, C, step, cur, run, cls, val);
    }
    o[(long)cur * N] += run * step;
}

// The argument order must match `_backward` below
kernel void raymarch_bwd(
    device const float* go   [[buffer(0)]],   // [B, C, N]
    device const VT* vol     [[buffer(1)]],
    device const VT* lab     [[buffer(2)]],
    device const float* M    [[buffer(3)]],
    device const float* src  [[buffer(4)]],
    device const float* tgt  [[buffer(5)]],
    device const float* stp  [[buffer(6)]],
    device float* gM         [[buffer(7)]],   // [B*N, 12] per-ray partials, reduced in torch
    device float* gsrc       [[buffer(8)]],   // [B*N, 3]
    device float* gtgt       [[buffer(9)]],   // [B*N, 3]
    device float* gstep      [[buffer(10)]],  // [B*N]
    device atomic_float* gvol [[buffer(11)]], // [D, H, W], zero-initialized
    constant int& N          [[buffer(12)]],
    constant int& D          [[buffer(13)]],
    constant int& H          [[buffer(14)]],
    constant int& W          [[buffer(15)]],
    constant int& S          [[buffer(16)]],
    constant int& C          [[buffer(17)]],
    constant int& src_ray_stride   [[buffer(18)]],
    constant int& src_batch_stride [[buffer(19)]],
    constant int& need_gvol  [[buffer(20)]],
    constant int& need_gcam  [[buffer(21)]],
    constant int& soft       [[buffer(22)]],
    uint tid [[thread_position_in_grid]])
{
    int gid = (int)tid;
    int b = gid / N, n = gid - b * N;
    Ray r = setup_ray(M, src, tgt, gid, b, n, D, H, W, src_ray_stride, src_batch_stride);
    device const float* m = M + b * 12;
    float3 m0 = float3(m[0], m[1], m[2]), m1 = float3(m[4], m[5], m[6]), m2 = float3(m[8], m[9], m[10]);

    float step0 = stp[gid];  // step [mm] of the unclipped ray
    float step = step0 * (r.t1 - r.t0);
    float3 aM0 = 0.0f, aM1 = 0.0f, aM2 = 0.0f;  // dL/dM rows (3 weights each)
    float3 aT = 0.0f;                           // dL/dM translation column
    float3 gs = 0.0f, gt = 0.0f;
    float gst = 0.0f;                           // sum of gg * val: d(loss)/d(step) up to the clip scale
    float adir = 0.0f, audir = 0.0f;            // sensitivity to sliding the samples along the ray
    float inv_s = 1.0f / (float)S;

    if (r.hit) for (int k = 0; k < S; ++k) {
        float u = ((float)k + 0.5f) * inv_s;
        float tt = r.t0 + u * (r.t1 - r.t0);
        float3 p = r.p0 + tt * r.dp;
        float3 f0 = floor(p);
        float3 f = p - f0;
        int ix = (int)f0.x, iy = (int)f0.y, iz = (int)f0.z;

        float v[8], gj[8];
        fetch8(vol, ix, iy, iz, D, H, W, v);

        // Soft labels: each neighbor feeds the channel of its own label, so weight it by that
        // channel's upstream gradient and the single-channel derivatives below apply as is
        for (int c = 0; c < 8; ++c) gj[c] = 1.0f;
        if (soft) {
            float l[8];
            fetch8(lab, ix, iy, iz, D, H, W, l);
            for (int c = 0; c < 8; ++c) {
                int lj = (int)l[c];
                gj[c] = (lj >= 0 && lj < C) ? go[((long)b * C + lj) * N + n] : 0.0f;
                v[c] *= gj[c];
            }
        }
        float val = trilerp(v, f);

        float gg = 1.0f;  // soft labels fold the upstream gradient into the neighbors
        if (!soft) {
            int cls = 0;
            if (C > 1) cls = (int)fetch(lab, (int)rint(p.x), (int)rint(p.y), (int)rint(p.z), D, H, W);
            if (cls < 0 || cls >= C) continue;  // dropped in the forward pass
            gg = go[((long)b * C + cls) * N + n];
        }
        float g = gg * step;

        // Analytic trilinear derivatives w.r.t. the voxel-space position
        float3 d = trilerp_grad(v, f);
        float3 gp = g * d;

        // Sensitivity to sliding the sample along the ray, for the clipping bounds
        float gdir = dot(gp, r.dp);
        adir += gdir;
        audir += u * gdir;
        gst += gg * val;

        // dL/dM[i, j] = gpos_i * q_j, q = lerp(src, tgt, t) in camera space
        float3 q = r.s + tt * r.e;
        aM0 += gp.x * q; aM1 += gp.y * q; aM2 += gp.z * q;
        aT += gp;

        if (need_gcam) {
            float3 gq = m0 * gp.x + m1 * gp.y + m2 * gp.z;  // M[:, :3]^T @ gpos
            gs += (1.0f - tt) * gq;
            gt += tt * gq;
        }

        if (need_gvol) {
            for (int c = 0; c < 8; ++c) {
                int jx = ix + (c & 1), jy = iy + ((c >> 1) & 1), jz = iz + (c >> 2);
                if (jx < 0 || jx >= W || jy < 0 || jy >= H || jz < 0 || jz >= D) continue;
                atomic_fetch_add_explicit(&gvol[(jz * H + jy) * W + jx], g * corner_weight(c, f) * gj[c], memory_order_relaxed);
            }
        }
    }

    if (r.hit) {
        // Chain through the clipping bounds: t_s = (1 - u_s) t0 + u_s t1 and step = step0 (t1 - t0).
        // An unclamped bound is t = (plane - p0_a) / delta_a for the axis a that sets it, so
        // dt/dM[a] = -[q(t), 1] / delta_a
        float g0 = (r.t_lo > 0.0f && r.t_lo < 1.0f) ? adir - audir - step0 * gst : 0.0f;
        float g1 = (r.t_hi > 0.0f && r.t_hi < 1.0f) ? audir + step0 * gst : 0.0f;
        float3 safe = select(r.dp, float3(1e-12f), abs(r.dp) < 1e-12f);
        float3 rr = -1.0f / safe;
        float3 k0 = float3(r.t_lo == r.tmin.x ? g0 * rr.x : 0.0f,
                           (r.t_lo == r.tmin.y && r.t_lo != r.tmin.x) ? g0 * rr.y : 0.0f,
                           (r.t_lo != r.tmin.y && r.t_lo != r.tmin.x) ? g0 * rr.z : 0.0f);
        float3 k1 = float3(r.t_hi == r.tmax.x ? g1 * rr.x : 0.0f,
                           (r.t_hi == r.tmax.y && r.t_hi != r.tmax.x) ? g1 * rr.y : 0.0f,
                           (r.t_hi != r.tmax.y && r.t_hi != r.tmax.x) ? g1 * rr.z : 0.0f);
        float3 q0 = r.s + r.t0 * r.e, q1 = r.s + r.t1 * r.e;
        aM0 += k0.x * q0 + k1.x * q1;
        aM1 += k0.y * q0 + k1.y * q1;
        aM2 += k0.z * q0 + k1.z * q1;
        aT += k0 + k1;
        if (need_gcam) {
            // The bounds also move with the endpoints: dt/dsrc = (1 - t) dt/dp0, dt/dtgt = t dt/dp0
            float3 ws = k0 * (1.0f - r.t0) + k1 * (1.0f - r.t1);
            float3 wt = k0 * r.t0 + k1 * r.t1;
            gs += m0 * ws.x + m1 * ws.y + m2 * ws.z;
            gt += m0 * wt.x + m1 * wt.y + m2 * wt.z;
            gst *= (r.t1 - r.t0);
        }
    }

    device float* gm = gM + (long)gid * 12;
    gm[0] = aM0.x; gm[1] = aM0.y; gm[2] = aM0.z; gm[3] = aT.x;
    gm[4] = aM1.x; gm[5] = aM1.y; gm[6] = aM1.z; gm[7] = aT.y;
    gm[8] = aM2.x; gm[9] = aM2.y; gm[10] = aM2.z; gm[11] = aT.z;
    if (need_gcam) {
        gsrc[gid * 3] = gs.x; gsrc[gid * 3 + 1] = gs.y; gsrc[gid * 3 + 2] = gs.z;
        gtgt[gid * 3] = gt.x; gtgt[gid * 3 + 1] = gt.y; gtgt[gid * 3 + 2] = gt.z;
        gstep[gid] = gst;
    }
}
"""

# Volume and labelmap element types the kernels are instantiated for; geometry stays float32
_DTYPES = {torch.float32: ("float", "f32"), torch.float16: ("half", "f16"), torch.bfloat16: ("bfloat", "bf16")}


def _build_source(dtypes) -> str:
    """The shared helpers plus one copy of the kernels per volume dtype (`raymarch_fwd_f16`, ...)."""
    out = [_HEADER]
    for ctype, suffix in (_DTYPES[dt] for dt in dtypes):
        body = _KERNELS.replace("raymarch_fwd", f"raymarch_fwd_{suffix}").replace(
            "raymarch_bwd", f"raymarch_bwd_{suffix}"
        )
        out.append(f"#define VT {ctype}\n{body}\n#undef VT\n")
    return "\n".join(out)


def _expected_dtypes() -> tuple[torch.dtype, ...]:
    """Volume dtypes the kernels are expected to build for, from facts that need no compilation.

    bfloat16 needs the Metal 3.1 language version (macOS 14). This is a constant, read once at import,
    so that `supports` can be evaluated inside `torch.compile`: dynamo cannot trace `compile_shader`,
    nor a cached wrapper around it.
    """
    if not (torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")):
        return ()
    try:
        major = int(platform.mac_ver()[0].split(".")[0])
    except ValueError:
        major = 0
    return (torch.float32, torch.float16, torch.bfloat16) if major >= 14 else (torch.float32, torch.float16)


_SUPPORTED = _expected_dtypes()


@functools.cache
def _compiled():
    """`(library, built dtypes)`, compiled on first use by an eager kernel launch (never while tracing).

    If the OS rejects the bfloat16 kernels despite the version check, float32 and float16 still build."""
    if not _SUPPORTED:
        raise RuntimeError("the metal backend needs an MPS device and torch.mps.compile_shader")
    candidates = [list(_SUPPORTED)]
    if torch.bfloat16 in _SUPPORTED:
        candidates.append([dt for dt in _SUPPORTED if dt != torch.bfloat16])
    error = None
    for dtypes in candidates:
        try:
            return torch.mps.compile_shader(_build_source(dtypes)), tuple(dtypes)
        except Exception as e:  # noqa: BLE001  # shader compile errors surface as RuntimeError
            error = e
    raise RuntimeError(f"Metal shader compilation failed: {error}")


def supports(dtype: torch.dtype) -> bool:
    """Whether the fused kernels are expected to run on this machine for a volume of `dtype`.

    A constant lookup, safe under `torch.compile`; a kernel launch raises if a build still fails."""
    return dtype in _SUPPORTED


def _kernel(name: str, vol: torch.Tensor):
    """The `name` kernel instantiated for the volume's dtype."""
    library, built = _compiled()
    if vol.dtype not in built:
        raise TypeError(f"metal backend does not support {vol.dtype} volumes on this machine")
    return getattr(library, f"{name}_{_DTYPES[vol.dtype][1]}")


def _forward(vol, lab, M, src, tgt, step, n_samples, n_classes, width, soft_labels):
    """Forward launch of the fused kernel: the float32 render `[B, C, N]`.

    `vol`/`lab` may be float32, float16 or bfloat16 (the same dtype), geometry is float32.
    `width` is unused: one thread marches one ray, in linear order.
    """
    B, N, _ = tgt.shape
    D, H, W = vol.shape
    out = torch.zeros(B, n_classes, N, device=vol.device, dtype=torch.float32)
    _kernel("raymarch_fwd", vol)(
        vol, lab, M, src, tgt, step, out,
        N, D, H, W, n_samples, n_classes, *src_strides(src), int(soft_labels),
        threads=B * N,
    )  # fmt: skip
    return out


def _backward(go, vol, lab, M, src, tgt, step, n_samples, n_classes, width, need_gvol, need_gcam, soft_labels):
    """Backward launch of the fused kernel: `(gvol, gM, gsrc, gtgt, gstep)`, unrequested ones empty."""
    B, N, _ = tgt.shape
    D, H, W = vol.shape
    dev, f32 = vol.device, torch.float32
    gM_part = torch.empty(B * N, 12, device=dev, dtype=f32)
    gvol = torch.zeros(D, H, W, device=dev, dtype=f32) if need_gvol else gM_part.new_empty(1)
    gsrc = torch.empty(B * N, 3, device=dev, dtype=f32) if need_gcam else gM_part.new_empty(1)
    gtgt = torch.empty(B * N, 3, device=dev, dtype=f32) if need_gcam else gM_part.new_empty(1)
    gstep = torch.empty(B * N, device=dev, dtype=f32) if need_gcam else gM_part.new_empty(1)

    _kernel("raymarch_bwd", vol)(
        go.float().contiguous(), vol, lab, M, src, tgt, step,
        gM_part, gsrc, gtgt, gstep, gvol,
        N, D, H, W, n_samples, n_classes, *src_strides(src), int(need_gvol), int(need_gcam), int(soft_labels),
        threads=B * N,
    )  # fmt: skip
    # Per-ray partials are reduced here, so the pose gradient is deterministic
    gM = gM_part.view(B, N, 12).sum(dim=1).view(B, 3, 4)
    if not need_gvol:
        gvol = gvol.new_empty(0)
    if need_gcam:
        gsrc, gtgt, gstep = gsrc.view(B, N, 3), gtgt.view(B, N, 3), gstep.view(B, N)
    else:
        gsrc, gtgt, gstep = (gvol.new_empty(0) for _ in range(3))
    return gvol, gM, gsrc, gtgt, gstep


BACKEND = register_fused_op(
    FusedBackend(
        name="metal",
        op_name="metal_raymarch",
        forward=_forward,
        backward=_backward,
        supports=supports,
        matching_label_dtype=True,
    )
)
