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

## Benchmark

Even without compilation, **`nanodrr` is ~5× faster than [`DiffDRR`](https://github.com/eigenvivek/DiffDRR).** With `pytorch>=2.9`, `torch.compile` with `bfloat16` works correctly and achieves ~8× speedup.

> [!NOTE]
>
> On `pytorch<2.9`, `torch.compile` with `bfloat16` is slower than eager due to a CUDA graph capture issue. Use `pytorch>=2.9` (Triton ≥3.5) for best results.

Experimental Setup: 
- Python 3.12.12
- NVIDIA RTX 6000 Ada (48 GB VRAM)
- 200×200 DRR
- Compile `mode="reduce-overhead"` with `fullgraph=True`
- Benchmark script: [`tests/benchmark/benchmark.py`](tests/benchmark/benchmark.py)

![Benchmarking runtime, FPS, and memory usage.](tests/benchmark/benchmark.png "benchmark")

> *Mean ± std. dev. of 10 runs, 100 loops each.*
