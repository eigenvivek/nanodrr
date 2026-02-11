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

`nanodrr` is compatible with `pytorch>=2.4`. Even without compilation, it is ~5× faster than [`DiffDRR`](https://github.com/eigenvivek/DiffDRR). With `pytorch>=2.9`, `torch.compile` with `bfloat16` works correctly and achieves ~8× speedup.

> [!NOTE]
>
> On `pytorch<2.9`, `torch.compile` with `bfloat16` is slower than eager due to a CUDA graph capture issue. Use `pytorch>=2.9` (Triton ≥3.5) for best results.

### Runtime (lower is better)

| PyTorch | Triton | DiffDRR (f32) | nanodrr (f32) | nanodrr + compile (f32) | nanodrr (bf16) | nanodrr + compile (bf16) |
|---------|--------|-------------:|-------------:|------------------------:|---------------:|-------------------------:|
| 2.4.1   | 3.0.0  | 5,271 ± 6 μs | 1,053 ± 1 μs | 877 ± 1 μs | 642 ± 15 μs | 1,052 ± 3 μs |
| 2.5.1   | 3.1.0  | 5,269 ± 5 μs | 1,051 ± 1 μs | 878 ± 0 μs | 636 ± 35 μs | 1,053 ± 3 μs |
| 2.6.0   | 3.2.0  | 5,304 ± 12 μs | 1,052 ± 1 μs | 887 ± 1 μs | 648 ± 15 μs | 1,059 ± 3 μs |
| 2.7.1   | 3.3.1  | 5,454 ± 255 μs | 1,052 ± 1 μs | 875 ± 1 μs | 640 ± 20 μs | 1,053 ± 3 μs |
| 2.8.0   | 3.4.0  | 5,275 ± 5 μs | 1,052 ± 1 μs | 875 ± 1 μs | 641 ± 16 μs | 1,059 ± 4 μs |
| 2.9.1   | 3.5.1  | 4,689 ± 6 μs | 1,056 ± 1 μs | 880 ± 1 μs | 668 ± 18 μs | 608 ± 19 μs |
| 2.10.0  | 3.6.0  | 4,711 ± 5 μs | 1,057 ± 1 μs | 882 ± 1 μs | 666 ± 23 μs | 598 ± 23 μs |

> *Mean ± std. dev. of 10 runs, 100 loops each.*

### FPS (higher is better)

| PyTorch | Triton | DiffDRR (f32) | nanodrr (f32) | nanodrr + compile (f32) | nanodrr (bf16) | nanodrr + compile (bf16) |
|---------|--------|-------------:|-------------:|------------------------:|---------------:|-------------------------:|
| 2.4.1   | 3.0.0  | 190 | 950 | 1,140 | 1,558 | 951 |
| 2.5.1   | 3.1.0  | 190 | 951 | 1,139 | 1,572 | 950 |
| 2.6.0   | 3.2.0  | 189 | 951 | 1,127 | 1,543 | 944 |
| 2.7.1   | 3.3.1  | 183 | 951 | 1,143 | 1,563 | 950 |
| 2.8.0   | 3.4.0  | 190 | 950 | 1,143 | 1,560 | 944 |
| 2.9.1   | 3.5.1  | 213 | 947 | 1,136 | 1,497 | 1,645 |
| 2.10.0  | 3.6.0  | 212 | 946 | 1,134 | 1,502 | 1,672 |

> *Mean ± std. dev. of 10 runs, 100 loops each.*

Experimental Setup: 
- Python 3.12.12
- 200×200 DRR
- Compile `mode="reduce-overhead"` with `fullgraph=True`
- Benchmark script: [`tests/benchmark.py`](tests/bench_render.py), runner: [`tests/benchmark.sh`](tests/run_benchmarks.sh)
