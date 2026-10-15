import torch
import triton
import triton.language as tl

from ._fused import register_fused_op

_BLOCK = 128
_WARPS = 4
_TILE_W = 16


@triton.jit
def _sample(vol_ptr, ix, iy, iz, D, H, W, m):
    """Load vol[iz, iy, ix] with zero padding outside bounds."""
    inb = m & (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H) & (iz >= 0) & (iz < D)
    idx = (iz * H + iy) * W + ix
    return tl.load(vol_ptr + idx, mask=inb, other=0.0)


@triton.jit
def _label(lab_ptr, ix, iy, iz, D, H, W, m):
    """Load the integer label at voxel [iz, iy, ix] (0 outside bounds)."""
    return _sample(lab_ptr, ix, iy, iz, D, H, W, m).to(tl.int32)


@triton.jit
def _label_go(go_ptr, lab_ptr, ix, iy, iz, D, H, W, m, bCN, N, offs, C):
    """Upstream gradient go[b, label[iz, iy, ix], n] of the channel that a voxel is routed to."""
    cls = _label(lab_ptr, ix, iy, iz, D, H, W, m)
    return tl.load(go_ptr + bCN + cls * N + offs, mask=m & (cls >= 0) & (cls < C), other=0.0).to(tl.float32)


@triton.jit
def _scatter(out_ptr, out_base, cls, val, N, offs, m, C):
    """out[cls, n] += val for each lane. Lanes own distinct n, so no atomics are needed."""
    ok = m & (cls >= 0) & (cls < C) & (val != 0.0)
    p = out_ptr + out_base + cls * N + offs
    tl.store(p, tl.load(p, mask=ok, other=0.0) + val, mask=ok)


@triton.jit
def _run_length(out_ptr, out_base, cur_cls, cur_sum, cls, val, step, N, offs, m, C):
    """Accumulate `val` into a per-lane running sum, flushing it to `out` when the class changes."""
    active = m & (cls >= 0) & (cls < C) & (val != 0.0)
    change = active & (cls != cur_cls)
    _scatter(out_ptr, out_base, cur_cls, cur_sum * step, N, offs, change, C)
    cur_sum = tl.where(change, 0.0, cur_sum) + tl.where(active, val, 0.0)
    cur_cls = tl.where(change, cls, cur_cls)
    return cur_cls, cur_sum


@triton.jit
def _axis_range(p0, dp, lo, hi):
    """Ray-parameter interval over which `lo <= p0 + t * dp <= hi` (per lane)."""
    safe = tl.where(tl.abs(dp) < 1e-12, 1e-12, dp)
    t0 = (lo - p0) / safe
    t1 = (hi - p0) / safe
    return tl.minimum(t0, t1), tl.maximum(t0, t1)


@triton.jit
def _round_half_even(x):
    """Round to nearest, ties to even — matches `grid_sample`'s nearest mode."""
    big = 12582912.0  # 1.5 * 2**23; fp32 addition rounds ties to even for |x| < 2**22
    return (x.to(tl.float32) + big) - big


@triton.jit
def _fused_kernel(
    vol_ptr,
    lab_ptr,
    M_ptr,
    src_ptr,
    tgt_ptr,
    step_ptr,
    out_ptr,
    go_ptr,
    gvol_ptr,
    gM_ptr,
    gsrc_ptr,
    gtgt_ptr,
    gstep_ptr,
    N,
    D,
    H,
    W,
    S,
    inv_s,
    src_ray_stride,
    src_batch_stride,
    IW,
    BCN,
    BN,
    C: tl.constexpr,
    CP2: tl.constexpr,
    BACKWARD: tl.constexpr,
    NEED_GVOL: tl.constexpr,
    NEED_GCAM: tl.constexpr,
    SOFT: tl.constexpr,
    SPLIT: tl.constexpr,
    TW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Fused forward (BACKWARD=0) or backward (BACKWARD=1) ray march.

    Rays go from `src` to `tgt` (camera space); `M` maps camera space to voxel
    pixel coordinates, so sample `s` sits at `M @ [lerp(src, tgt, t_s), 1]`.
    Each ray is clipped to the interval `[t0, t1]` over which its trilinear
    footprint can touch the volume, and `t_s` is the midpoint of the `s`-th of
    S equal parts of it. The backward pass differentiates through `t0` and `t1`.
    With a labelmap (C > 1), each sample's intensity is routed to the channel
    given by a nearest-neighbor label lookup at the same position. With SOFT,
    each of the 8 trilinear neighbors is instead routed to its own label.

    Each program marches BLOCK rays — a (BLOCK // TW) x TW pixel tile of an
    IW-wide image when TW > 0, else BLOCK consecutive rays — over the k-th of
    SPLIT chunks of the sample range, writing forward partials to slice k of
    `out`.
    """
    pid = tl.program_id(0)
    k = pid % SPLIT
    blk = pid // SPLIT
    if TW > 0:
        TH: tl.constexpr = BLOCK // TW
        tiles_w = tl.cdiv(IW, TW)
        IH = N // IW
        nblocks = tiles_w * tl.cdiv(IH, TH)
        b = blk // nblocks
        tile = blk % nblocks
        r = (tile // tiles_w) * TH + tl.arange(0, BLOCK) // TW
        c = (tile % tiles_w) * TW + tl.arange(0, BLOCK) % TW
        offs = r * IW + c
        m = (r < IH) & (c < IW)
    else:
        nblocks = tl.cdiv(N, BLOCK)
        b = blk // nblocks
        offs = (blk % nblocks) * BLOCK + tl.arange(0, BLOCK)
        m = offs < N
    base = b * N + offs
    chunk = tl.cdiv(S, SPLIT)
    s_lo = k * chunk
    s_hi = tl.minimum(s_lo + chunk, S)

    # M is [B, 3, 4] row-major: fused camera → voxel-pixel affine transform
    Mb = M_ptr + b * 12
    m00 = tl.load(Mb + 0)
    m01 = tl.load(Mb + 1)
    m02 = tl.load(Mb + 2)
    m03 = tl.load(Mb + 3)
    m10 = tl.load(Mb + 4)
    m11 = tl.load(Mb + 5)
    m12 = tl.load(Mb + 6)
    m13 = tl.load(Mb + 7)
    m20 = tl.load(Mb + 8)
    m21 = tl.load(Mb + 9)
    m22 = tl.load(Mb + 10)
    m23 = tl.load(Mb + 11)

    # Camera-space ray endpoints; src may be per-ray or broadcast (stride 0)
    soff = b * src_batch_stride + offs * src_ray_stride
    sx = tl.load(src_ptr + soff + 0, mask=m, other=0.0)
    sy = tl.load(src_ptr + soff + 1, mask=m, other=0.0)
    sz = tl.load(src_ptr + soff + 2, mask=m, other=0.0)
    tx = tl.load(tgt_ptr + base * 3 + 0, mask=m, other=0.0)
    ty = tl.load(tgt_ptr + base * 3 + 1, mask=m, other=0.0)
    tz = tl.load(tgt_ptr + base * 3 + 2, mask=m, other=0.0)

    # Pixel-space ray: p0 = M @ [src, 1], delta = M[:, :3] @ (tgt - src)
    ex = tx - sx
    ey = ty - sy
    ez = tz - sz
    p0x = m00 * sx + m01 * sy + m02 * sz + m03
    p0y = m10 * sx + m11 * sy + m12 * sz + m13
    p0z = m20 * sx + m21 * sy + m22 * sz + m23
    dxp = m00 * ex + m01 * ey + m02 * ez
    dyp = m10 * ex + m11 * ey + m12 * ez
    dzp = m20 * ex + m21 * ey + m22 * ez

    # Clip each ray to the box outside of which the trilinear footprint is
    # identically zero. Rays that miss it collapse to a point with zero step
    tx0, tx1 = _axis_range(p0x, dxp, -1.0, W * 1.0)
    ty0, ty1 = _axis_range(p0y, dyp, -1.0, H * 1.0)
    tz0, tz1 = _axis_range(p0z, dzp, -1.0, D * 1.0)
    t_lo = tl.maximum(tl.maximum(tx0, ty0), tz0)
    t_hi = tl.minimum(tl.minimum(tx1, ty1), tz1)
    t0 = tl.minimum(tl.maximum(t_lo, 0.0), 1.0)
    t1 = tl.maximum(tl.minimum(tl.maximum(t_hi, 0.0), 1.0), t0)
    hit = m & (t0 < t1)
    s_hi = s_lo + (s_hi - s_lo) * tl.max(hit.to(tl.int32), axis=0)  # skip tiles that miss

    # `step` arrives as the step size [mm] of the unclipped ray
    step0 = tl.load(step_ptr + base, mask=m, other=0.0)
    step = step0 * (t1 - t0)
    classes = tl.arange(0, CP2)
    cmask = classes < C

    if BACKWARD:
        if C == 1:
            go_tile = tl.load(
                go_ptr + b * C * N + classes[None, :] * N + offs[:, None],
                mask=m[:, None] & cmask[None, :],
                other=0.0,
            )
        a00 = tl.zeros([BLOCK], dtype=tl.float32)
        a01 = tl.zeros([BLOCK], dtype=tl.float32)
        a02 = tl.zeros([BLOCK], dtype=tl.float32)
        a03 = tl.zeros([BLOCK], dtype=tl.float32)
        a10 = tl.zeros([BLOCK], dtype=tl.float32)
        a11 = tl.zeros([BLOCK], dtype=tl.float32)
        a12 = tl.zeros([BLOCK], dtype=tl.float32)
        a13 = tl.zeros([BLOCK], dtype=tl.float32)
        a20 = tl.zeros([BLOCK], dtype=tl.float32)
        a21 = tl.zeros([BLOCK], dtype=tl.float32)
        a22 = tl.zeros([BLOCK], dtype=tl.float32)
        a23 = tl.zeros([BLOCK], dtype=tl.float32)
        gsx = tl.zeros([BLOCK], dtype=tl.float32)
        gsy = tl.zeros([BLOCK], dtype=tl.float32)
        gsz = tl.zeros([BLOCK], dtype=tl.float32)
        gtx = tl.zeros([BLOCK], dtype=tl.float32)
        gty = tl.zeros([BLOCK], dtype=tl.float32)
        gtz = tl.zeros([BLOCK], dtype=tl.float32)
        astep = tl.zeros([BLOCK], dtype=tl.float32)
        adir = tl.zeros([BLOCK], dtype=tl.float32)
        audir = tl.zeros([BLOCK], dtype=tl.float32)
    else:
        # With a labelmap, each lane keeps the running sum of one class in
        # registers and adds it to `out` when the class changes. A dense
        # [BLOCK, C] tile would cost O(C) per sample, but rays cross few class
        # boundaries
        out_base = k * BCN + b * C * N
        cur_cls = tl.zeros([BLOCK], dtype=tl.int32)
        cur_sum = tl.zeros([BLOCK], dtype=tl.float32)

    for s in range(s_lo, s_hi):
        u = (s + 0.5) * inv_s
        t = t0 + u * (t1 - t0)
        x = p0x + t * dxp
        y = p0y + t * dyp
        z = p0z + t * dzp
        x0 = tl.floor(x)
        y0 = tl.floor(y)
        z0 = tl.floor(z)
        fx = x - x0
        fy = y - y0
        fz = z - z0
        ix = x0.to(tl.int32)
        iy = y0.to(tl.int32)
        iz = z0.to(tl.int32)

        v000 = _sample(vol_ptr, ix, iy, iz, D, H, W, m)
        v001 = _sample(vol_ptr, ix + 1, iy, iz, D, H, W, m)
        v010 = _sample(vol_ptr, ix, iy + 1, iz, D, H, W, m)
        v011 = _sample(vol_ptr, ix + 1, iy + 1, iz, D, H, W, m)
        v100 = _sample(vol_ptr, ix, iy, iz + 1, D, H, W, m)
        v101 = _sample(vol_ptr, ix + 1, iy, iz + 1, D, H, W, m)
        v110 = _sample(vol_ptr, ix, iy + 1, iz + 1, D, H, W, m)
        v111 = _sample(vol_ptr, ix + 1, iy + 1, iz + 1, D, H, W, m)

        if SOFT:
            if BACKWARD:
                # Weight each neighbor by the upstream gradient of its own channel,
                # so the single-channel derivatives below apply to the products
                bCN = b * C * N
                v000 = v000 * _label_go(go_ptr, lab_ptr, ix, iy, iz, D, H, W, m, bCN, N, offs, C)
                v001 = v001 * _label_go(go_ptr, lab_ptr, ix + 1, iy, iz, D, H, W, m, bCN, N, offs, C)
                v010 = v010 * _label_go(go_ptr, lab_ptr, ix, iy + 1, iz, D, H, W, m, bCN, N, offs, C)
                v011 = v011 * _label_go(go_ptr, lab_ptr, ix + 1, iy + 1, iz, D, H, W, m, bCN, N, offs, C)
                v100 = v100 * _label_go(go_ptr, lab_ptr, ix, iy, iz + 1, D, H, W, m, bCN, N, offs, C)
                v101 = v101 * _label_go(go_ptr, lab_ptr, ix + 1, iy, iz + 1, D, H, W, m, bCN, N, offs, C)
                v110 = v110 * _label_go(go_ptr, lab_ptr, ix, iy + 1, iz + 1, D, H, W, m, bCN, N, offs, C)
                v111 = v111 * _label_go(go_ptr, lab_ptr, ix + 1, iy + 1, iz + 1, D, H, W, m, bCN, N, offs, C)

        c00 = v000 * (1 - fx) + v001 * fx
        c01 = v010 * (1 - fx) + v011 * fx
        c10 = v100 * (1 - fx) + v101 * fx
        c11 = v110 * (1 - fx) + v111 * fx
        c0 = c00 * (1 - fy) + c01 * fy
        c1 = c10 * (1 - fy) + c11 * fy
        val = c0 * (1 - fz) + c1 * fz

        if C == 1:
            cls = tl.zeros([BLOCK], dtype=tl.int32)
        else:
            if not SOFT:
                rx = _round_half_even(x).to(tl.int32)
                ry = _round_half_even(y).to(tl.int32)
                rz = _round_half_even(z).to(tl.int32)
                cls = _sample(lab_ptr, rx, ry, rz, D, H, W, m).to(tl.int32)

        if BACKWARD:
            if SOFT:
                gg = tl.full([BLOCK], 1.0, dtype=tl.float32)  # already folded into the neighbors
            elif C == 1:
                gg = tl.sum(go_tile, axis=1)
            else:
                # Gather go[b, cls, n] per sample: flat in C, unlike a per-lane
                # class tile, whose register cost grows with CP2. Out-of-range
                # labels are masked, matching the forward's silent drop
                gg = tl.load(go_ptr + b * C * N + cls * N + offs, mask=m & (cls >= 0) & (cls < C), other=0.0)
            g = tl.where(m, gg * step, 0.0).to(tl.float32)

            # Analytic trilinear derivatives wrt the pixel-space position
            dvx = ((v001 - v000) * (1 - fy) + (v011 - v010) * fy) * (1 - fz) + (
                (v101 - v100) * (1 - fy) + (v111 - v110) * fy
            ) * fz
            dvy = ((v010 - v000) * (1 - fx) + (v011 - v001) * fx) * (1 - fz) + (
                (v110 - v100) * (1 - fx) + (v111 - v101) * fx
            ) * fz
            dvz = ((v100 - v000) * (1 - fx) + (v101 - v001) * fx) * (1 - fy) + (
                (v110 - v010) * (1 - fx) + (v111 - v011) * fx
            ) * fy
            gx = g * dvx
            gy = g * dvy
            gz = g * dvz

            # Sensitivity to sliding the sample along the ray, for the clipping bounds
            gdir = gx * dxp + gy * dyp + gz * dzp
            adir += gdir
            audir += u * gdir
            astep += gg * val

            # dL/dM[i, j] = Σ_s gpos_i * q_j with q = lerp(src, tgt, t) (camera space)
            qx = sx + t * ex
            qy = sy + t * ey
            qz = sz + t * ez
            a00 += gx * qx
            a01 += gx * qy
            a02 += gx * qz
            a03 += gx
            a10 += gy * qx
            a11 += gy * qy
            a12 += gy * qz
            a13 += gy
            a20 += gz * qx
            a21 += gz * qy
            a22 += gz * qz
            a23 += gz

            if NEED_GCAM:
                # Chain gpos back to camera space: gq = M[:, :3]^T @ gpos
                gqx = m00 * gx + m10 * gy + m20 * gz
                gqy = m01 * gx + m11 * gy + m21 * gz
                gqz = m02 * gx + m12 * gy + m22 * gz
                gsx += (1 - t) * gqx
                gsy += (1 - t) * gqy
                gsz += (1 - t) * gqz
                gtx += t * gqx
                gty += t * gqy
                gtz += t * gqz

            if NEED_GVOL:
                wx0 = 1 - fx
                wy0 = 1 - fy
                wz0 = 1 - fz
                for corner in tl.static_range(8):
                    cz = corner // 4
                    cy = (corner // 2) % 2
                    cx = corner % 2
                    jx = ix + cx
                    jy = iy + cy
                    jz = iz + cz
                    inb = m & (jx >= 0) & (jx < W) & (jy >= 0) & (jy < H) & (jz >= 0) & (jz < D)
                    w = tl.where(cx == 1, fx, wx0) * tl.where(cy == 1, fy, wy0) * tl.where(cz == 1, fz, wz0)
                    idx = (jz * H + jy) * W + jx
                    if SOFT:
                        w = w * _label_go(go_ptr, lab_ptr, jx, jy, jz, D, H, W, m, b * C * N, N, offs, C)
                    tl.atomic_add(gvol_ptr + idx, g * w, mask=inb)
        else:
            if SOFT:
                for corner in tl.static_range(8):
                    cz = corner // 4
                    cy = (corner // 2) % 2
                    cx = corner % 2
                    w = tl.where(cx == 1, fx, 1 - fx) * tl.where(cy == 1, fy, 1 - fy) * tl.where(cz == 1, fz, 1 - fz)
                    vj = _sample(vol_ptr, ix + cx, iy + cy, iz + cz, D, H, W, m)
                    lj = _label(lab_ptr, ix + cx, iy + cy, iz + cz, D, H, W, m)
                    cur_cls, cur_sum = _run_length(out_ptr, out_base, cur_cls, cur_sum, lj, vj * w, step, N, offs, m, C)
            elif C == 1:
                cur_sum += val
            else:
                cur_cls, cur_sum = _run_length(out_ptr, out_base, cur_cls, cur_sum, cls, val, step, N, offs, m, C)

    if BACKWARD:
        # Chain through the clipping bounds: t_s = (1 - u_s) t0 + u_s t1 and
        # step = step0 (t1 - t0). An unclamped bound is t = (plane - p0_a) / delta_a
        # for the axis a that sets it, so dt/dM[a] = -[q(t), 1] / delta_a
        g0 = tl.where(hit & (t_lo > 0.0) & (t_lo < 1.0), adir - audir - step0 * astep, 0.0)
        g1 = tl.where(hit & (t_hi > 0.0) & (t_hi < 1.0), audir + step0 * astep, 0.0)
        rx = -1.0 / tl.where(tl.abs(dxp) < 1e-12, 1e-12, dxp)
        ry = -1.0 / tl.where(tl.abs(dyp) < 1e-12, 1e-12, dyp)
        rz = -1.0 / tl.where(tl.abs(dzp) < 1e-12, 1e-12, dzp)
        k0x = tl.where(t_lo == tx0, g0 * rx, 0.0)
        k0y = tl.where((t_lo == ty0) & (t_lo != tx0), g0 * ry, 0.0)
        k0z = tl.where((t_lo != ty0) & (t_lo != tx0), g0 * rz, 0.0)
        k1x = tl.where(t_hi == tx1, g1 * rx, 0.0)
        k1y = tl.where((t_hi == ty1) & (t_hi != tx1), g1 * ry, 0.0)
        k1z = tl.where((t_hi != ty1) & (t_hi != tx1), g1 * rz, 0.0)
        q0x = sx + t0 * ex
        q0y = sy + t0 * ey
        q0z = sz + t0 * ez
        q1x = sx + t1 * ex
        q1y = sy + t1 * ey
        q1z = sz + t1 * ez
        a00 += k0x * q0x + k1x * q1x
        a01 += k0x * q0y + k1x * q1y
        a02 += k0x * q0z + k1x * q1z
        a03 += k0x + k1x
        a10 += k0y * q0x + k1y * q1x
        a11 += k0y * q0y + k1y * q1y
        a12 += k0y * q0z + k1y * q1z
        a13 += k0y + k1y
        a20 += k0z * q0x + k1z * q1x
        a21 += k0z * q0y + k1z * q1y
        a22 += k0z * q0z + k1z * q1z
        a23 += k0z + k1z
        if NEED_GCAM:
            # The bounds also move with the endpoints: dt/dsrc = (1 - t) dt/dp0, dt/dtgt = t dt/dp0
            wsx = k0x * (1 - t0) + k1x * (1 - t1)
            wsy = k0y * (1 - t0) + k1y * (1 - t1)
            wsz = k0z * (1 - t0) + k1z * (1 - t1)
            wtx = k0x * t0 + k1x * t1
            wty = k0y * t0 + k1y * t1
            wtz = k0z * t0 + k1z * t1
            gsx += m00 * wsx + m10 * wsy + m20 * wsz
            gsy += m01 * wsx + m11 * wsy + m21 * wsz
            gsz += m02 * wsx + m12 * wsy + m22 * wsz
            gtx += m00 * wtx + m10 * wty + m20 * wtz
            gty += m01 * wtx + m11 * wty + m21 * wtz
            gtz += m02 * wtx + m12 * wty + m22 * wtz
            astep = astep * (t1 - t0)

        # Per-program partials, reduced in torch, so the pose gradient is deterministic
        gMb = gM_ptr + pid * 12
        tl.store(gMb + 0, tl.sum(a00))
        tl.store(gMb + 1, tl.sum(a01))
        tl.store(gMb + 2, tl.sum(a02))
        tl.store(gMb + 3, tl.sum(a03))
        tl.store(gMb + 4, tl.sum(a10))
        tl.store(gMb + 5, tl.sum(a11))
        tl.store(gMb + 6, tl.sum(a12))
        tl.store(gMb + 7, tl.sum(a13))
        tl.store(gMb + 8, tl.sum(a20))
        tl.store(gMb + 9, tl.sum(a21))
        tl.store(gMb + 10, tl.sum(a22))
        tl.store(gMb + 11, tl.sum(a23))
        if NEED_GCAM:
            gc = k * BN * 3 + base * 3
            tl.store(gsrc_ptr + gc + 0, gsx, mask=m)
            tl.store(gsrc_ptr + gc + 1, gsy, mask=m)
            tl.store(gsrc_ptr + gc + 2, gsz, mask=m)
            tl.store(gtgt_ptr + gc + 0, gtx, mask=m)
            tl.store(gtgt_ptr + gc + 1, gty, mask=m)
            tl.store(gtgt_ptr + gc + 2, gtz, mask=m)
            tl.store(gstep_ptr + k * BN + base, astep, mask=m)
    else:
        if C == 1:
            tl.store(out_ptr + out_base + offs, cur_sum * step, mask=m)
        else:
            _scatter(out_ptr, out_base, cur_cls, cur_sum * step, N, offs, m, C)


def _config(N, width):
    """Program count and tile width for the ray → program mapping."""
    if width > 0:
        return triton.cdiv(width, _TILE_W) * triton.cdiv(N // width, _BLOCK // _TILE_W), _TILE_W
    return triton.cdiv(N, _BLOCK), 0


def _split(B, nblocks, n_samples):
    """Sample-range split factor: small launches underfill the GPU, so trade
    partial-sum traffic for parallelism; larger launches already saturate it."""
    if B * nblocks > 384:
        return 1
    return max(1, min(16 if nblocks < 64 else 8, n_samples // 8))


def _launch(
    vol,
    lab,
    M,
    src,
    tgt,
    step,
    out,
    go,
    gvol,
    gM,
    gsrc,
    gtgt,
    gstep,
    n_samples,
    n_classes,
    width,
    split,
    backward,
    need_gvol,
    need_gcam,
    soft,
):
    """Launch the fused kernel in forward (backward=False) or backward mode."""
    B, N, _ = tgt.shape
    D, H, W = vol.shape
    nblocks, tw = _config(N, width)
    broadcast_src = src.shape[1] == 1
    grid = (B * nblocks * split,)
    _fused_kernel[grid](
        vol,
        lab,
        M,
        src,
        tgt,
        step,
        out,
        go,
        gvol,
        gM,
        gsrc,
        gtgt,
        gstep,
        N,
        D,
        H,
        W,
        n_samples,
        1.0 / n_samples,
        0 if broadcast_src else 3,
        3 if broadcast_src else N * 3,
        width,
        B * n_classes * N,
        B * N,
        C=n_classes,
        CP2=triton.next_power_of_2(n_classes),
        BACKWARD=backward,
        NEED_GVOL=need_gvol,
        NEED_GCAM=need_gcam,
        SOFT=soft,
        SPLIT=split,
        TW=tw,
        BLOCK=_BLOCK,  # ty: ignore[invalid-argument-type]
        num_warps=_WARPS,  # ty: ignore[unknown-argument]
    )


def _forward(vol, lab, M, src, tgt, step, n_samples, n_classes, width, soft_labels):
    """Forward launch of the fused kernel: the float32 render `[B, C, N]`."""
    B, N, _ = tgt.shape
    split = _split(B, _config(N, width)[0], n_samples)
    alloc = torch.empty if n_classes == 1 else torch.zeros  # with a labelmap, the kernel adds into `out`
    out = alloc(split, B, n_classes, N, device=vol.device, dtype=torch.float32)
    _launch(
        vol,
        lab,
        M,
        src,
        tgt,
        step,
        out,
        out,
        out,
        out,
        out,
        out,
        out,  # unused go/grad pointers in the forward pass
        n_samples,
        n_classes,
        width,
        split,
        backward=False,
        need_gvol=False,
        need_gcam=False,
        soft=soft_labels,
    )
    return out[0] if split == 1 else out.sum(dim=0)


def _backward(go, vol, lab, M, src, tgt, step, n_samples, n_classes, width, need_gvol, need_gcam, soft_labels):
    """Backward launch of the fused kernel: `(gvol, gM, gsrc, gtgt, gstep)`, reduced over programs."""
    B, N, _ = tgt.shape
    split = _split(B, _config(N, width)[0], n_samples)
    gM_part = torch.empty(B * _config(N, width)[0] * split, 12, device=M.device, dtype=torch.float32)
    gvol = torch.zeros_like(vol, dtype=torch.float32) if need_gvol else gM_part.new_empty(0)
    gsrc = torch.empty(split, B, N, 3, device=M.device, dtype=torch.float32) if need_gcam else gM_part.new_empty(0)
    gtgt = torch.empty(split, B, N, 3, device=M.device, dtype=torch.float32) if need_gcam else gM_part.new_empty(0)
    gstep = torch.empty(split, B, N, device=M.device, dtype=torch.float32) if need_gcam else gM_part.new_empty(0)

    _launch(
        vol,
        lab,
        M,
        src,
        tgt,
        step,
        gM_part,  # unused out pointer in the backward pass
        go.contiguous(),
        gvol if need_gvol else gM_part,
        gM_part,
        gsrc if need_gcam else gM_part,
        gtgt if need_gcam else gM_part,
        gstep if need_gcam else gM_part,
        n_samples,
        n_classes,
        width,
        split,
        backward=True,
        need_gvol=need_gvol,
        need_gcam=need_gcam,
        soft=soft_labels,
    )
    gM = gM_part.view(B, -1, 12).sum(dim=1).view(B, 3, 4)
    if need_gcam:
        gsrc, gtgt, gstep = gsrc.sum(dim=0), gtgt.sum(dim=0), gstep.sum(dim=0)
    return gvol, gM, gsrc, gtgt, gstep


triton_raymarch = register_fused_op("triton", "triton_raymarch", _forward, _backward)
