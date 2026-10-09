---
icon: lucide/rocket
---

# nanodrr

[![tests](https://github.com/eigenvivek/nanodrr/actions/workflows/tests.yml/badge.svg)](https://github.com/eigenvivek/nanodrr/actions/workflows/tests.yml)
[![docs](https://github.com/eigenvivek/nanodrr/actions/workflows/docs.yml/badge.svg)](https://github.com/eigenvivek/nanodrr/actions/workflows/docs.yml)
[![pypi](https://img.shields.io/pypi/v/nanodrr?label=PyPI%20version&logo=python&logoColor=white)](https://pypi.org/project/nanodrr)

A performance-oriented reimplementation of [`DiffDRR`](https://github.com/eigenvivek/DiffDRR) with the following improvements:

- Optimized, pure PyTorch implementation (**~5× faster than `DiffDRR` at baseline**)
- Fused Triton rendering kernel, used by default on CUDA (**up to ~80× faster than `DiffDRR`**)
- Modular design (freely swap subjects, extrinsics, and intrinsics during rendering)
- Compatibility with `torch.compile` and mixed precision
- Extensive type hints with `jaxtyping`
- Runs on CUDA, Apple silicon (MPS), and CPU
- Standard Python package structure managed with `uv`

All projective geometry is implemented internally using the standard [Hartley and Zisserman](https://www.cambridge.org/core/books/multiple-view-geometry-in-computer-vision/0B6F289C78B2B23F596CAA76D3D43F7A) pinhole camera formulation.

All changes to `DiffDRR` are summarized [here](changes.md).

## Installation

!!! note "PyTorch version"
    On `pytorch<2.9`, `torch.compile` with `bfloat16` is slower than eager for the pure PyTorch backend due to a CUDA graph capture issue (see [Benchmarks](#benchmarks)). The fused Triton backend is unaffected.

    MPS-accelerated differentiable rendering requires `pytorch>=2.13`, which is the first release with native MPS kernels for `grid_sample` backwards and 3D nearest-neighbor sampling.

To strictly install the renderer:
```
pip install nanodrr
```

To install the optional [plotting](https://vivekg.dev/nanodrr/api/plot/) or [3D visualization module](https://vivekg.dev/nanodrr/api/scene/):
```
pip install "nanodrr[plot]"   # 2D visualization (matplotlib, opencv)
pip install "nanodrr[scene]"  # 3D visualization (VTK, PyVista)
pip install "nanodrr[all]"    # All extras
```

## Benchmarks

!!! tip "Highlights"
    - **~5× faster** than [`DiffDRR`](https://github.com/eigenvivek/DiffDRR) with the pure PyTorch backend (1,093 FPS vs 223 FPS)
    - **~57× faster** with the fused Triton kernel, without compilation (12,600 FPS vs 223 FPS)
    - **~80× faster** with `torch.compile` and `bfloat16` (17,820 FPS vs 223 FPS)
    - **~3.5× less memory** than `DiffDRR` (322 MB vs 1,170 MB peak reserved with `bfloat16` + compile)

![Benchmarking runtime, FPS, and memory usage.](assets/images/benchmark.png#only-light)
![Benchmarking runtime, FPS, and memory usage.](assets/images/benchmark_dark.png#only-dark)

!!! abstract "*FPS computed from per-frame GPU kernel time (`torch.profiler`).*"
    Median wall-clock timings are also recorded in [`tests/benchmark/benchmark.csv`](https://github.com/eigenvivek/nanodrr/blob/main/tests/benchmark/benchmark.csv). Benchmarked by rendering 200×200 DRRs on an NVIDIA RTX 6000 Ada (48 GB) with Python 3.12. Compile represents `torch.compile(mode="reduce-overhead", fullgraph=True)`. Full experiment at [`tests/benchmark/`](https://github.com/eigenvivek/nanodrr/tree/main/tests/benchmark).

## Roadmap

- [x] Implement a fully optimized renderer
- [x] Port strictly necessary modules from `DiffDRR` (e.g., SE(3) utilities, loss functions, and 2D plotting)
- [x] Migrate 3D plotting functions to an optional module
- [ ] Integrate with [`xvr`](https://github.com/eigenvivek/xvr) to speed up network training and registration
- [ ] Integrate with [`polypose`](https://github.com/eigenvivek/polypose) to speed up registration
- [ ] Release as `v1.0.0` of `DiffDRR`!
