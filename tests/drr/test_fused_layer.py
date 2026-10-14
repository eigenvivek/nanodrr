"""The shared layer of the fused backends (in `backends.py`), on the CPU with a pure-torch reference backend.

The Triton and Metal kernels need a GPU, but everything around them — the differentiable ops, their
fakes, the autograd wiring and the broadcast reductions — is common code. This file exercises it
with a backend whose `forward` is plain torch and whose `backward` is autograd of that forward.
"""

import pytest
import torch
import torch.nn.functional as F

from nanodrr.drr.backends import register_fused_op

B, N, S, D, H, W = 2, 6, 8, 5, 6, 7


def _forward(vol, lab, M, src, tgt, step, n_samples, n_classes, width, soft_labels):
    """Unclipped midpoint trilinear march, one class: `[B, 1, N]`."""
    t = (torch.arange(n_samples, dtype=src.dtype) + 0.5) / n_samples
    q = src[:, :, None] + t[None, None, :, None] * (tgt - src)[:, :, None]  # [B, N, S, 3] camera space
    p = torch.einsum("bij,bnsj->bnsi", M[:, :, :3], q) + M[:, None, None, :, 3]  # voxel coordinates
    size = torch.tensor([W, H, D], dtype=src.dtype)
    grid = (2 * (p + 0.5) / size - 1)[:, :, :, None]  # [B, N, S, 1, 3]
    img = F.grid_sample(vol[None, None].expand(M.shape[0], -1, -1, -1, -1), grid, align_corners=False)
    return (img[:, 0, :, :, 0].sum(dim=-1) * step)[:, None]


def _backward(go, vol, lab, M, src, tgt, step, n_samples, n_classes, width, need_gvol, need_gcam, soft_labels):
    # `torch.func.vjp`, because autograd is disabled inside a custom op's implementation
    def f(vol, M, src, tgt, step):
        return _forward(vol, lab, M, src, tgt, step, n_samples, n_classes, width, False)

    _, vjp = torch.func.vjp(f, vol, M, src.expand_as(tgt), tgt, step)
    gvol, gM, gsrc, gtgt, gstep = vjp(go.float())
    gvol = gvol if need_gvol else gM.new_empty(0)
    gcam = (gsrc, gtgt, gstep) if need_gcam else [gM.new_empty(0) for _ in range(3)]  # distinct: no aliasing
    return gvol, gM, *gcam


raymarch = register_fused_op("reference", "test_reference_raymarch", _forward, _backward)


def _inputs(src_rays=N, seed=0):
    g = torch.Generator().manual_seed(seed)
    vol = torch.rand(D, H, W, generator=g)
    M = torch.eye(4)[:3].repeat(B, 1, 1) + 0.05 * torch.randn(B, 3, 4, generator=g)
    M[:, :, 3] += torch.tensor([W / 2, H / 2, D / 2])
    src = torch.randn(B, src_rays, 3, generator=g)
    tgt = torch.randn(B, N, 3, generator=g)
    step = torch.rand(B, N, generator=g) + 0.5
    return vol, vol, M, src, tgt, step


def _op(vol, lab, M, src, tgt, step):
    return raymarch(vol, lab, M, src, tgt, step, S, 1, W, False)


@pytest.mark.parametrize("src_rays", [N, 1])
def test_op_gradients_match_autograd_of_the_forward(src_rays):
    """Includes the reduction of per-ray source gradients onto a broadcast `[B, 1, 3]` source."""
    leaves = [x.requires_grad_() for x in (_inputs(src_rays)[i].clone() for i in (0, 2, 3, 4, 5))]
    vol, M, src, tgt, step = leaves
    lab = vol.detach()
    w = torch.randn(B, 1, N, generator=torch.Generator().manual_seed(1))

    (_op(vol, lab, M, src, tgt, step) * w).sum().backward()
    op_grads = [x.grad.clone() for x in leaves]
    for x in leaves:
        x.grad = None
    (_forward(vol, lab, M, src.expand_as(tgt), tgt, step, S, 1, W, False) * w).sum().backward()

    for got, want in zip(op_grads, [x.grad for x in leaves]):
        assert got.shape == want.shape
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)


def test_op_only_differentiates_what_requires_grad():
    vol, lab, M, src, tgt, step = _inputs()
    M = M.requires_grad_()
    _op(vol, lab, M, src, tgt, step).sum().backward()

    assert M.grad is not None and M.grad.shape == (B, 3, 4)
    assert all(x.grad is None for x in (vol, src, tgt, step))


def test_op_output_has_the_volume_dtype():
    vol, lab, M, src, tgt, step = _inputs()
    out = _op(vol.double(), lab.double(), M.double(), src.double(), tgt.double(), step.double())

    assert out.dtype == torch.float64 and out.shape == (B, 1, N)


def test_volume_gradient_refuses_deterministic_mode():
    vol, lab, M, src, tgt, step = _inputs()
    vol.requires_grad_()
    out = _op(vol, lab, M, src, tgt, step).sum()
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        with pytest.raises(RuntimeError, match="nondeterministic atomics"):
            out.backward()
    finally:
        torch.use_deterministic_algorithms(False)


def test_fake_implementations_match_the_real_ones():
    """`torch.compile` relies on the fakes for shapes and dtypes."""
    vol, lab, M, src, tgt, step = _inputs()
    for x in (vol, M, src, tgt, step):
        x.requires_grad_()
    torch.library.opcheck(raymarch, (vol, lab, M, src, tgt, step, S, 1, W, False))
