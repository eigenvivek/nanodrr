# nanodrr

A performance-oriented reimplementation of [`DiffDRR`](https://github.com/eigenvivek/DiffDRR).

## Goals

- [x] As-optimized-as-possible pure PyTorch code
- [x] As-modular-as-possible implementation (e.g., freely swap subjects/extrinsics/intrinsics)
- [x] Compatibility with `torch.compile`
- [x] Compatibility with mixed precision
- [x] Compatibility with [`torchio` dataloaders](https://docs.torchio.org/data/dataset.html)
- [x] Extensive typehints with `jaxtyping`
- [x] Traditional python package structure with `uv`

## Roadmap

- [x] Implement a fully optimized renderer
- [x] Port strictly necessary modules from `DiffDRR` (e.g., SE(3) utilities, loss functions, and 2D plotting)
- [ ] Integrate with [`xvr`](https://github.com/eigenvivek/xvr) to speed up network training and registration
- [ ] Integrate with [`polypose`](https://github.com/eigenvivek/polypose) to speed up registration
- [ ] Migrate 3D plotting functions to a new, standalone library
- [ ] Release as `v1.0.0` of `DiffDRR`!

## Benchmarks

> [!IMPORTANT]
> - **~5× faster** than [`DiffDRR`](https://github.com/eigenvivek/DiffDRR) out of the box, without compilation
> - **~8× faster** with `torch.compile` and `bfloat16` on `pytorch>=2.9`
> - **~2.5× less memory** than `DiffDRR` (516 MB vs 1,344 MB peak reserved with `bfloat16` + compile)

![Benchmarking runtime, FPS, and memory usage.](tests/benchmark/benchmark.png "benchmark")

> *Mean ± std. dev. of 10 runs, 100 loops each. Benchmarked by rendering 200×200 DRRs on an NVIDIA RTX 6000 Ada (48 GB) with Python 3.12. Compile represents `torch.compile(mode="reduce-overhead", fullgraph=True)`. Full experiment at [`tests/benchmark/`](tests/benchmark/).*

## Installation

> [!NOTE]
>
> On `pytorch<2.9`, `torch.compile` with `bfloat16` is slower than eager due to a CUDA graph capture issue. Use `pytorch>=2.9` (Triton ≥3.5) for best results.
