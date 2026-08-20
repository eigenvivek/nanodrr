import math

import pytest
import torch

from nanodrr.camera import make_k_inv
from nanodrr.recon import displaced_detector, fdk_filter


def test_ramp_impulse_response():
    """An impulse row filters to the Ram-Lak kernel scaled by the pixel spacing."""
    sdd, delx = 500.0, 2.0
    height, width = 8, 64
    proj = torch.zeros(1, 1, height, width)
    proj[0, 0, 4, 32] = 1.0
    k_inv = make_k_inv(sdd, delx, delx, 0.0, 0.0, height, width)

    out = fdk_filter(proj, k_inv, torch.tensor([sdd]), window="ram-lak", orthographic=True)

    row = out[0, 0, 4]
    assert torch.allclose(row[32], torch.tensor(0.25 / delx), atol=1e-6)
    assert torch.allclose(row[31], torch.tensor(-1.0 / (math.pi**2 * delx)), atol=1e-6)
    assert torch.allclose(row[33], torch.tensor(-1.0 / (math.pi**2 * delx)), atol=1e-6)
    assert row[30].abs() < 1e-4 * row[32]  # even taps are zero up to padding wrap
    assert out[0, 0, 5].abs().max() == 0.0  # filtering is row-wise


def test_cosine_weighting():
    """Perspective filtering equals orthographic filtering of cosine-weighted projections."""
    torch.manual_seed(0)
    sdd, delx, dely, x0, y0 = 300.0, 1.5, 2.5, 3.0, -2.0
    height, width = 16, 32
    proj = torch.rand(2, 1, height, width)
    k_inv = make_k_inv(sdd, delx, dely, x0, y0, height, width)
    sdd_t = torch.tensor([sdd])

    # Physical detector coordinates relative to the principal point
    u = torch.arange(width) + 0.5
    v = torch.arange(height) + 0.5
    a = delx * (u - (x0 / delx + width / 2.0))
    b = dely * (v - (y0 / dely + height / 2.0))
    cosine = sdd / torch.sqrt(sdd**2 + a[None, :] ** 2 + b[:, None] ** 2)

    persp = fdk_filter(proj, k_inv, sdd_t, window="ram-lak")
    ortho = fdk_filter(proj * cosine, k_inv, sdd_t, window="ram-lak", orthographic=True)
    assert torch.allclose(persp, ortho, atol=1e-6)


def test_window_apodizes():
    """Apodized windows shrink the impulse response peak relative to Ram-Lak."""
    height, width = 4, 32
    proj = torch.zeros(1, 1, height, width)
    proj[0, 0, 2, 16] = 1.0
    k_inv = make_k_inv(100.0, 1.0, 1.0, 0.0, 0.0, height, width)
    sdd = torch.tensor([100.0])

    peaks = {
        w: fdk_filter(proj, k_inv, sdd, window=w, orthographic=True)[0, 0, 2, 16]
        for w in ("ram-lak", "shepp-logan", "hann")
    }
    assert peaks["hann"] < peaks["shepp-logan"] < peaks["ram-lak"]


def test_unknown_window_raises():
    proj = torch.zeros(1, 1, 4, 8)
    k_inv = make_k_inv(100.0, 1.0, 1.0, 0.0, 0.0, 4, 8)
    with pytest.raises(ValueError, match="window"):
        fdk_filter(proj, k_inv, torch.tensor([100.0]), window="hamming")


def test_displaced_detector_leaves_centered_panel_alone():
    proj = torch.rand(2, 1, 8, 32)
    k_inv = make_k_inv(500.0, 1.0, 1.0, 0.0, 0.0, 8, 32)
    out, k_out = displaced_detector(proj, k_inv)
    assert torch.equal(out, proj)
    assert torch.equal(k_out, k_inv)


def test_displaced_detector_weights_and_symmetry():
    """Conjugate rays in the overlap strip must sum to 2, and the returned
    panel must be symmetric about the principal ray."""
    sdd, delx, width, height = 500.0, 1.0, 64, 4
    x0 = delx * 20.0  # principal point 20 px right of center -> half-fan
    k_inv = make_k_inv(sdd, delx, delx, x0, 0.0, height, width)
    ones = torch.ones(1, 1, height, width)

    w, k_out = displaced_detector(ones, k_inv)
    assert w.shape[-1] > width  # panel was extended

    # Principal ray is centered on the returned panel
    cx = (-k_out[0, 0, 2] / k_out[0, 0, 0]).item()
    assert abs(cx - w.shape[-1] / 2) <= 1.0

    # Conjugate columns (mirrored about the principal ray) sum to 2, so every
    # doubly-measured ray ends up with total weight 1 after the full-scan 1/2.
    row = w[0, 0, 0]
    n = row.shape[0]
    for i in (0, 10, 30, 51, 60):
        assert abs((row[i] + row[n - 1 - i]).item() - 2.0) < 1e-4
    assert row.max().item() <= 2.0 + 1e-6 and row.min().item() >= 0.0

    # The two columns straddling the principal ray average to 1
    c = int(cx)
    assert abs(0.5 * (row[c - 1] + row[c]).item() - 1.0) < 1e-3

    # Wide side saturates at 2, short side runs out to 0
    assert row[0].item() == pytest.approx(2.0)
    assert row[-1].item() == pytest.approx(0.0)
