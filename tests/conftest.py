import pytest
import torch


def _available(name: str) -> bool:
    if name == "cuda":
        return torch.cuda.is_available()
    if name == "mps":
        return torch.backends.mps.is_available()
    return True


@pytest.fixture(
    params=[
        pytest.param(name, marks=pytest.mark.skipif(not _available(name), reason=f"requires {name.upper()}"))
        for name in ("cpu", "mps", "cuda")
    ]
)
def device(request) -> torch.device:
    return torch.device(request.param)
