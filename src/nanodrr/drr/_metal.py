"""Fused Metal ray-marching kernels, compiled at runtime with `torch.mps.compile_shader`.

One thread marches one ray, so the sample grid and per-sample intensities of the `grid_sample` path are
never materialized. The semantics are those of the Triton kernel (see `_triton.py`): rays are clipped to
the volume and sampled at midpoints, with trilinear intensities (zero padded), nearest or per-neighbor
(soft) label routing, and analytic gradients for the camera-to-voxel transform, the ray endpoints, the
step size and the volume. The volume and labelmap may be float32, float16 or bfloat16; they are read in
place and promoted per sample, while geometry and accumulation stay float32. The labelmap is read in the
volume's dtype or in float32, so neither the float32 labelmap of `Subject` nor a `Subject.half()` is copied.

The kernels are written once and instantiated per (volume, labelmap) dtype pair (`raymarch_fwd_f16_f32`, ...).
"""

import functools

import torch

from .backends import register_fused_op

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
    device const LT* lab     [[buffer(1)]],
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
    device const LT* lab     [[buffer(2)]],
    device const float* M    [[buffer(3)]],
    device const float* src  [[buffer(4)]],
    device const float* tgt  [[buffer(5)]],
    device const float* stp  [[buffer(6)]],
    device float* gM         [[buffer(7)]],   // [B*Np/w, 12] per-simdgroup partials, reduced in torch
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
    constant int& Np         [[buffer(23)]],   // N padded to a multiple of the simdgroup width w
    uint tid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint w [[threads_per_simdgroup]])
{
    // The grid is [B, Np], so no simdgroup straddles two poses. Padding threads march a real ray
    // with `hit` cleared: they contribute zeros but take part in the simdgroup reduction of gM
    int b = (int)tid / Np, n = (int)tid - b * Np;
    bool active = n < N;
    n = min(n, N - 1);
    int gid = b * N + n;
    Ray r = setup_ray(M, src, tgt, gid, b, n, D, H, W, src_ray_stride, src_batch_stride);
    r.hit = r.hit && active;
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

    // Every lane reaches this point (no early returns), so the simdgroup sums are over all w rays
    aM0 = simd_sum(aM0); aM1 = simd_sum(aM1); aM2 = simd_sum(aM2); aT = simd_sum(aT);
    if (lane == 0) {
        device float* gm = gM + (long)(tid / w) * 12;
        gm[0] = aM0.x; gm[1] = aM0.y; gm[2] = aM0.z; gm[3] = aT.x;
        gm[4] = aM1.x; gm[5] = aM1.y; gm[6] = aM1.z; gm[7] = aT.y;
        gm[8] = aM2.x; gm[9] = aM2.y; gm[10] = aM2.z; gm[11] = aT.z;
    }
    if (active && need_gcam) {
        gsrc[gid * 3] = gs.x; gsrc[gid * 3 + 1] = gs.y; gsrc[gid * 3 + 2] = gs.z;
        gtgt[gid * 3] = gt.x; gtgt[gid * 3 + 1] = gt.y; gtgt[gid * 3 + 2] = gt.z;
        gstep[gid] = gst;
    }
}
"""

# Volume and labelmap element types the kernels are instantiated for; geometry stays float32
_DTYPES = {torch.float32: ("float", "f32"), torch.float16: ("half", "f16"), torch.bfloat16: ("bfloat", "bf16")}


def _src_strides(src: torch.Tensor) -> tuple[int, int]:
    """`(ray, batch)` strides of the ray sources `[B, N, 3]` or `[B, 1, 3]` (a broadcast source has ray stride 0)."""
    return (0 if src.shape[1] == 1 else 3), src.shape[1] * 3


def _label_dtypes(vol_dtype: torch.dtype) -> tuple[torch.dtype, ...]:
    """Labelmap dtypes the kernels read for a volume of `vol_dtype`: the volume's own, or float32."""
    return tuple(dict.fromkeys((vol_dtype, torch.float32)))


def _build_source() -> str:
    """The shared helpers plus one copy of the kernels per (volume, labelmap) dtype pair (`raymarch_fwd_f16_f32`, ...)."""
    out = [_HEADER]
    for vt in _DTYPES:
        for lt in _label_dtypes(vt):
            suffix = f"{_DTYPES[vt][1]}_{_DTYPES[lt][1]}"
            body = _KERNELS.replace("raymarch_fwd", f"raymarch_fwd_{suffix}").replace(
                "raymarch_bwd", f"raymarch_bwd_{suffix}"
            )
            out.append(f"#define VT {_DTYPES[vt][0]}\n#define LT {_DTYPES[lt][0]}\n{body}\n#undef VT\n#undef LT\n")
    return "\n".join(out)


@functools.cache
def _compiled():
    """The shader library, compiled on first use by an eager kernel launch (never while tracing).

    bfloat16 needs Metal 3.1 (macOS 14), which every torch build with MPS support for this package has."""
    return torch.mps.compile_shader(_build_source())


def _as_label(vol: torch.Tensor, lab: torch.Tensor) -> torch.Tensor:
    """The labelmap in a dtype the kernels read: as stored if that is the volume's dtype or float32
    (no copy), else float32, which holds integer labels exactly up to 2**24."""
    return lab if lab.dtype in _label_dtypes(vol.dtype) else lab.float().contiguous()


def _kernel(name: str, vol: torch.Tensor, lab: torch.Tensor):
    """The `name` kernel instantiated for the volume's and labelmap's dtypes."""
    return getattr(_compiled(), f"{name}_{_DTYPES[vol.dtype][1]}_{_DTYPES[lab.dtype][1]}")


def _forward(vol, lab, M, src, tgt, step, n_samples, n_classes, width, soft_labels):
    """Forward launch of the fused kernel: the float32 render `[B, C, N]`.

    `vol` may be float32, float16 or bfloat16, `lab` any dtype (see `_as_label`), geometry is float32.
    `width` is unused: one thread marches one ray, in linear order.
    """
    B, N, _ = tgt.shape
    D, H, W = vol.shape
    lab = _as_label(vol, lab)
    out = torch.zeros(B, n_classes, N, device=vol.device, dtype=torch.float32)
    _kernel("raymarch_fwd", vol, lab)(
        vol, lab, M, src, tgt, step, out,
        N, D, H, W, n_samples, n_classes, *_src_strides(src), int(soft_labels),
        threads=B * N,
    )  # fmt: skip
    return out


def _backward(go, vol, lab, M, src, tgt, step, n_samples, n_classes, width, need_gvol, need_gcam, soft_labels):
    """Backward launch of the fused kernel: `(gvol, gM, gsrc, gtgt, gstep)`, unrequested ones empty.

    The pose gradient is summed per simdgroup in the kernel and over simdgroups here, so its scratch is
    `B * N / w` rows rather than one per ray, and the result is deterministic.
    """
    B, N, _ = tgt.shape
    D, H, W = vol.shape
    dev, f32 = vol.device, torch.float32
    lab = _as_label(vol, lab)
    kernel = _kernel("raymarch_bwd", vol, lab)

    # Each pose's rays fill whole simdgroups (padded to Np), and threadgroups hold whole simdgroups
    w = kernel.thread_execution_width
    Np = -(-N // w) * w
    group_size = kernel.max_threads_per_threadgroup // w * w

    gM_part = torch.empty(B * Np // w, 12, device=dev, dtype=f32)
    unused = gM_part.new_empty(1)  # bound to the buffers of unrequested gradients, never written
    gvol = torch.zeros(D, H, W, device=dev, dtype=f32) if need_gvol else unused
    gsrc = torch.empty(B, N, 3, device=dev, dtype=f32) if need_gcam else unused
    gtgt = torch.empty(B, N, 3, device=dev, dtype=f32) if need_gcam else unused
    gstep = torch.empty(B, N, device=dev, dtype=f32) if need_gcam else unused

    kernel(
        go.float().contiguous(), vol, lab, M, src, tgt, step,
        gM_part, gsrc, gtgt, gstep, gvol,
        N, D, H, W, n_samples, n_classes, *_src_strides(src), int(need_gvol), int(need_gcam), int(soft_labels), Np,
        threads=B * Np, group_size=group_size,
    )  # fmt: skip
    gM = gM_part.view(B, Np // w, 12).sum(dim=1).view(B, 3, 4)
    if not need_gvol:
        gvol = gM.new_empty(0)
    if not need_gcam:
        gsrc, gtgt, gstep = (gM.new_empty(0) for _ in range(3))  # distinct: a custom op's outputs must not alias
    return gvol, gM, gsrc, gtgt, gstep


metal_raymarch = register_fused_op("metal", "metal_raymarch", _forward, _backward)
