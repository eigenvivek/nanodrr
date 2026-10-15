from dataclasses import dataclass

import pytest
import torch

from nanodrr.drr._fused import DEVICE_BACKENDS, available


@dataclass(frozen=True)
class Fused:
    """A fused backend and the device it runs on."""

    backend: str
    device: torch.device


@pytest.fixture(params=list(DEVICE_BACKENDS.items()), ids=lambda p: f"{p[1]}-{p[0]}")
def fused(request) -> Fused:
    """Each fused backend (Triton on CUDA, Metal on MPS) that this machine can run."""
    device, backend = request.param
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("requires MPS")
    if not available(backend):
        pytest.skip(f"the {backend} backend is unavailable")
    return Fused(backend, torch.device(device))
