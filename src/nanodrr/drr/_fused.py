import torch
import triton
import triton.language as tl

_BLOCK = 128
_WARPS = 4


@triton.jit
def _sample(vol_ptr, ix, iy, iz, D, H, W, m):
    """Load vol[iz, iy, ix] with zero padding outside bounds."""
    inb = m & (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H) & (iz >= 0) & (iz < D)
    idx = (iz * H + iy) * W + ix
    return tl.load(vol_ptr + idx, mask=inb, other=0.0)


@triton.jit
def _round_half_away(x):
    """Round to nearest, ties away from zero — matches CUDA grid_sampler's ::round."""
    return tl.where(x >= 0, tl.floor(x + 0.5), -tl.floor(0.5 - x))


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
    C: tl.constexpr,
    CP2: tl.constexpr,
    BACKWARD: tl.constexpr,
    NEED_GVOL: tl.constexpr,
    NEED_GCAM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Fused forward (BACKWARD=0) or backward (BACKWARD=1) ray march.

    Rays go from `src` to `tgt` (camera space); `M` maps camera space to voxel
    pixel coordinates, so sample `s` sits at `M @ [lerp(src, tgt, t_s), 1]`.
    With a labelmap (C > 1), each sample's intensity is routed to the channel
    given by a nearest-neighbor label lookup at the same position.
    """
    pid = tl.program_id(0)
    nblocks = tl.cdiv(N, BLOCK)
    b = pid // nblocks
    offs = (pid % nblocks) * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    base = b * N + offs

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

    step = tl.load(step_ptr + base, mask=m, other=0.0)
    classes = tl.arange(0, CP2)
    cmask = classes < C

    if BACKWARD:
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
    else:
        acc = tl.zeros([BLOCK, CP2], dtype=tl.float32)

    for s in range(S):
        t = s * inv_s
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
            rx = _round_half_away(x).to(tl.int32)
            ry = _round_half_away(y).to(tl.int32)
            rz = _round_half_away(z).to(tl.int32)
            cls = _sample(lab_ptr, rx, ry, rz, D, H, W, m).to(tl.int32)

        if BACKWARD:
            if C == 1:
                gg = tl.sum(go_tile, axis=1)
            else:
                gg = tl.sum(go_tile * (cls[:, None] == classes[None, :]), axis=1)
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
                astep += gg * val

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
                    tl.atomic_add(gvol_ptr + idx, g * w, mask=inb)
        else:
            acc += val[:, None] * (cls[:, None] == classes[None, :])

    if BACKWARD:
        gMb = gM_ptr + b * 12
        tl.atomic_add(gMb + 0, tl.sum(a00))
        tl.atomic_add(gMb + 1, tl.sum(a01))
        tl.atomic_add(gMb + 2, tl.sum(a02))
        tl.atomic_add(gMb + 3, tl.sum(a03))
        tl.atomic_add(gMb + 4, tl.sum(a10))
        tl.atomic_add(gMb + 5, tl.sum(a11))
        tl.atomic_add(gMb + 6, tl.sum(a12))
        tl.atomic_add(gMb + 7, tl.sum(a13))
        tl.atomic_add(gMb + 8, tl.sum(a20))
        tl.atomic_add(gMb + 9, tl.sum(a21))
        tl.atomic_add(gMb + 10, tl.sum(a22))
        tl.atomic_add(gMb + 11, tl.sum(a23))
        if NEED_GCAM:
            tl.store(gsrc_ptr + base * 3 + 0, gsx, mask=m)
            tl.store(gsrc_ptr + base * 3 + 1, gsy, mask=m)
            tl.store(gsrc_ptr + base * 3 + 2, gsz, mask=m)
            tl.store(gtgt_ptr + base * 3 + 0, gtx, mask=m)
            tl.store(gtgt_ptr + base * 3 + 1, gty, mask=m)
            tl.store(gtgt_ptr + base * 3 + 2, gtz, mask=m)
            tl.store(gstep_ptr + base, astep, mask=m)
    else:
        tl.store(
            out_ptr + b * C * N + classes[None, :] * N + offs[:, None],
            acc * step[:, None],
            mask=m[:, None] & cmask[None, :],
        )


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
    backward,
    need_gvol,
    need_gcam,
):
    """Launch the fused kernel in forward (backward=False) or backward mode."""
    B, N, _ = tgt.shape
    D, H, W = vol.shape
    broadcast_src = src.shape[1] == 1
    grid = (B * triton.cdiv(N, _BLOCK),)
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
        1.0 / (n_samples - 1),
        0 if broadcast_src else 3,
        3 if broadcast_src else N * 3,
        C=n_classes,
        CP2=triton.next_power_of_2(n_classes),
        BACKWARD=backward,
        NEED_GVOL=need_gvol,
        NEED_GCAM=need_gcam,
        BLOCK=_BLOCK,  # ty: ignore[invalid-argument-type]
        num_warps=_WARPS,  # ty: ignore[unknown-argument]
    )


@torch.library.custom_op("nanodrr::fused_raymarch", mutates_args=())
def fused_raymarch(
    vol: torch.Tensor,
    lab: torch.Tensor,
    M: torch.Tensor,
    src: torch.Tensor,
    tgt: torch.Tensor,
    step: torch.Tensor,
    n_samples: int,
    n_classes: int,
) -> torch.Tensor:
    """Fused differentiable ray marching.

    Registered as a custom op so `torch.compile` treats the kernel as opaque:
    inductor takes shapes from the fake impl below rather than re-compiling
    the Triton source, which would break its constexpr dead-branch pruning.
    """
    B, N, _ = tgt.shape
    M = M.contiguous()
    out = torch.empty(B, n_classes, N, device=vol.device, dtype=vol.dtype)
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
        backward=False,
        need_gvol=False,
        need_gcam=False,
    )
    return out


@fused_raymarch.register_fake
def _(vol, lab, M, src, tgt, step, n_samples, n_classes):
    B, N, _ = tgt.shape
    return torch.empty(B, n_classes, N, device=vol.device, dtype=vol.dtype)


def _setup_context(ctx, inputs, output):
    vol, lab, M, src, tgt, step, n_samples, n_classes = inputs
    ctx.save_for_backward(vol, lab, M.contiguous(), src, tgt, step)
    ctx.n_samples = n_samples
    ctx.n_classes = n_classes


@torch.library.custom_op("nanodrr::fused_raymarch_bwd", mutates_args=())
def _fused_raymarch_bwd(
    go: torch.Tensor,
    vol: torch.Tensor,
    lab: torch.Tensor,
    M: torch.Tensor,
    src: torch.Tensor,
    tgt: torch.Tensor,
    step: torch.Tensor,
    n_samples: int,
    n_classes: int,
    need_gvol: bool,
    need_gcam: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward kernel launch, opaque to `torch.compile` like the forward."""
    B, N, _ = tgt.shape
    gM = torch.zeros(B, 3, 4, device=M.device, dtype=torch.float32)
    gvol = torch.zeros_like(vol, dtype=torch.float32) if need_gvol else gM.new_empty(0)
    gsrc = torch.empty(B, N, 3, device=M.device, dtype=torch.float32) if need_gcam else gM.new_empty(0)
    gtgt = torch.empty(B, N, 3, device=M.device, dtype=torch.float32) if need_gcam else gM.new_empty(0)
    gstep = torch.empty(B, N, device=M.device, dtype=torch.float32) if need_gcam else gM.new_empty(0)

    _launch(
        vol,
        lab,
        M,
        src,
        tgt,
        step,
        gM,  # unused out pointer in the backward pass
        go.contiguous(),
        gvol if need_gvol else gM,
        gM,
        gsrc if need_gcam else gM,
        gtgt if need_gcam else gM,
        gstep if need_gcam else gM,
        n_samples,
        n_classes,
        backward=True,
        need_gvol=need_gvol,
        need_gcam=need_gcam,
    )
    return gvol, gM, gsrc, gtgt, gstep


@_fused_raymarch_bwd.register_fake
def _(go, vol, lab, M, src, tgt, step, n_samples, n_classes, need_gvol, need_gcam):
    B, N, _ = tgt.shape
    gM = torch.empty(B, 3, 4, device=M.device, dtype=torch.float32)
    gvol = torch.empty(vol.shape, device=M.device, dtype=torch.float32) if need_gvol else gM.new_empty(0)
    gsrc = torch.empty(B, N, 3, device=M.device, dtype=torch.float32) if need_gcam else gM.new_empty(0)
    gtgt = torch.empty(B, N, 3, device=M.device, dtype=torch.float32) if need_gcam else gM.new_empty(0)
    gstep = torch.empty(B, N, device=M.device, dtype=torch.float32) if need_gcam else gM.new_empty(0)
    return gvol, gM, gsrc, gtgt, gstep


def _backward(ctx, go):
    vol, lab, M, src, tgt, step = ctx.saved_tensors
    need_gvol = ctx.needs_input_grad[0]
    need_gcam = any(ctx.needs_input_grad[i] for i in (3, 4, 5))
    gvol, gM, gsrc, gtgt, gstep = _fused_raymarch_bwd(
        go, vol, lab, M, src, tgt, step, ctx.n_samples, ctx.n_classes, need_gvol, need_gcam
    )
    if need_gcam and src.shape[1] == 1:
        gsrc = gsrc.sum(dim=1, keepdim=True)
    return (
        gvol.to(vol.dtype) if need_gvol else None,
        None,  # lab: nearest-neighbor lookup, not differentiable
        gM.to(M.dtype),
        gsrc.to(src.dtype) if need_gcam else None,
        gtgt.to(tgt.dtype) if need_gcam else None,
        gstep.to(step.dtype) if need_gcam else None,
        None,  # n_samples
        None,  # n_classes
    )


torch.library.register_autograd("nanodrr::fused_raymarch", _backward, setup_context=_setup_context)
