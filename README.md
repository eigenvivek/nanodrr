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
- 200×200 DRR
- Compile `mode="reduce-overhead"` with `fullgraph=True`
- Benchmark script: [`tests/benchmark.py`](tests/benchmark.py), runner: [`tests/benchmark.sh`](tests/benchmark.sh)

### Runtime (lower is better)

> *Mean ± std. dev. of 10 runs, 100 loops each.*

<table>
  <tr>
    <th rowspan="2" align="left">PyTorch</th>
    <th rowspan="2" align="left">Triton</th>
    <th align="center">DiffDRR</th>
    <th colspan="4" align="center">nanodrr</th>
  </tr>
  <tr>
    <th align="center">f32</th>
    <th align="center">f32</th>
    <th align="center">+ compile (f32)</th>
    <th align="center">bf16</th>
    <th align="center">bf16 + compile</th>
  </tr>
  <tr>
    <td align="left">2.4.1</td><td align="left">3.0.0</td>
    <td align="center">5,271 ± 6 μs</td>
    <td align="center">1,053 ± 1 μs</td><td align="center">877 ± 1 μs</td>
    <td align="center">642 ± 15 μs</td><td align="center">1,052 ± 3 μs</td>
  </tr>
  <tr>
    <td align="left">2.5.1</td><td align="left">3.1.0</td>
    <td align="center">5,269 ± 5 μs</td>
    <td align="center">1,051 ± 1 μs</td><td align="center">878 ± 0 μs</td>
    <td align="center">636 ± 35 μs</td><td align="center">1,053 ± 3 μs</td>
  </tr>
  <tr>
    <td align="left">2.6.0</td><td align="left">3.2.0</td>
    <td align="center">5,304 ± 12 μs</td>
    <td align="center">1,052 ± 1 μs</td><td align="center">887 ± 1 μs</td>
    <td align="center">648 ± 15 μs</td><td align="center">1,059 ± 3 μs</td>
  </tr>
  <tr>
    <td align="left">2.7.1</td><td align="left">3.3.1</td>
    <td align="center">5,454 ± 255 μs</td>
    <td align="center">1,052 ± 1 μs</td><td align="center">875 ± 1 μs</td>
    <td align="center">640 ± 20 μs</td><td align="center">1,053 ± 3 μs</td>
  </tr>
  <tr>
    <td align="left">2.8.0</td><td align="left">3.4.0</td>
    <td align="center">5,275 ± 5 μs</td>
    <td align="center">1,052 ± 1 μs</td><td align="center">875 ± 1 μs</td>
    <td align="center">641 ± 16 μs</td><td align="center">1,059 ± 4 μs</td>
  </tr>
  <tr>
    <td align="left">2.9.1</td><td align="left">3.5.1</td>
    <td align="center">4,689 ± 6 μs</td>
    <td align="center">1,056 ± 1 μs</td><td align="center">880 ± 1 μs</td>
    <td align="center">668 ± 18 μs</td><td align="center"><b>608 ± 19 μs</b></td>
  </tr>
  <tr>
    <td align="left">2.10.0</td><td align="left">3.6.0</td>
    <td align="center">4,711 ± 5 μs</td>
    <td align="center">1,057 ± 1 μs</td><td align="center">882 ± 1 μs</td>
    <td align="center">666 ± 23 μs</td><td align="center"><b>598 ± 23 μs</b></td>
  </tr>
</table>

### FPS (higher is better)

> *Mean ± std. dev. of 10 runs, 100 loops each.*

<table>
  <tr>
    <th rowspan="2" align="left">PyTorch</th>
    <th rowspan="2" align="left">Triton</th>
    <th align="center">DiffDRR</th>
    <th colspan="4" align="center">nanodrr</th>
  </tr>
  <tr>
    <th align="center">f32</th>
    <th align="center">f32</th>
    <th align="center">+ compile (f32)</th>
    <th align="center">bf16</th>
    <th align="center">bf16 + compile</th>
  </tr>
  <tr>
    <td align="left">2.4.1</td><td align="left">3.0.0</td>
    <td align="center">190 ± 0</td>
    <td align="center">950 ± 1</td><td align="center">1,140 ± 1</td>
    <td align="center">1,558 ± 36</td><td align="center">951 ± 3</td>
  </tr>
  <tr>
    <td align="left">2.5.1</td><td align="left">3.1.0</td>
    <td align="center">190 ± 0</td>
    <td align="center">951 ± 1</td><td align="center">1,139 ± 0</td>
    <td align="center">1,572 ± 87</td><td align="center">950 ± 3</td>
  </tr>
  <tr>
    <td align="left">2.6.0</td><td align="left">3.2.0</td>
    <td align="center">189 ± 0</td>
    <td align="center">951 ± 1</td><td align="center">1,127 ± 1</td>
    <td align="center">1,543 ± 36</td><td align="center">944 ± 3</td>
  </tr>
  <tr>
    <td align="left">2.7.1</td><td align="left">3.3.1</td>
    <td align="center">183 ± 9</td>
    <td align="center">951 ± 1</td><td align="center">1,143 ± 1</td>
    <td align="center">1,562 ± 49</td><td align="center">950 ± 3</td>
  </tr>
  <tr>
    <td align="left">2.8.0</td><td align="left">3.4.0</td>
    <td align="center">190 ± 0</td>
    <td align="center">951 ± 1</td><td align="center">1,143 ± 1</td>
    <td align="center">1,560 ± 39</td><td align="center">944 ± 4</td>
  </tr>
  <tr>
    <td align="left">2.9.1</td><td align="left">3.5.1</td>
    <td align="center">213 ± 0</td>
    <td align="center">947 ± 1</td><td align="center">1,136 ± 1</td>
    <td align="center">1,497 ± 40</td><td align="center"><b>1,645 ± 51</b></td>
  </tr>
  <tr>
    <td align="left">2.10.0</td><td align="left">3.6.0</td>
    <td align="center">212 ± 0</td>
    <td align="center">946 ± 1</td><td align="center">1,134 ± 1</td>
    <td align="center">1,502 ± 52</td><td align="center"><b>1,672 ± 64</b></td>
  </tr>
</table>
