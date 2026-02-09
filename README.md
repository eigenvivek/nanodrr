# nanodrr

A performance-oriented reimplementation of [`DiffDRR`](https://github.com/eigenvivek/DiffDRR).

## Goals

- [x] As-optimized-as-possible pure PyTorch code
- [x] As-modular-as-possible implementation (e.g., freely swap subjects/extrinsics/intrinsics)
- [x] Compatibility with `torch.compile`
- [x] Compatibility with mixed precision
- [ ] Compatibility with [`torchio` dataloaders](https://docs.torchio.org/data/dataset.html)
- [x] Extensive typehints with `jaxtyping`
- [x] Traditional python package structure with `uv`

## Roadmap

- [x] Implement a fully optimized renderer
- [ ] Port strictly necessary modules from `DiffDRR` (e.g., SE(3) utilities, loss functions, and 2D plotting)
- [ ] Integrate with [`xvr`](https://github.com/eigenvivek/xvr) to speed up network training and registration
- [ ] Integrate with [`polypose`](https://github.com/eigenvivek/polypose) to speed up registration
- [ ] Migrate 3D plotting functions to a new, standalone library
- [ ] Release as `v1.0.0` of `DiffDRR`!
