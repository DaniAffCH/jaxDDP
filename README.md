# jaxDDP

A JAX-based differentiable dynamic programming implementation. It solves batched problems on GPUs.

Benchmark on NVIDIA GeForce RTX 4060 and Intel Core i9-14900HX batched CPU vs batched GPU:

![benchmark_jax_gpu](imgs/benchmark_jax_gpu.png)

| B | CPU (s) | GPU (s) | GPU speedup |
|---|---|---|---|
| 1 | 0.011 | 0.232 | 0.05× |
| 2 | 0.016 | 0.319 | 0.05× |
| 4 | 0.025 | 0.247 | 0.10× |
| 8 | 0.082 | 0.363 | 0.23× |
| 16 | 0.150 | 0.283 | 0.53× |
| 32 | 0.232 | 0.411 | 0.56× |
| 64 | 0.327 | 0.408 | 0.80× |
| 128 | 0.901 | 0.496 | 1.82× |
| 256 | 1.860 | 0.629 | 2.96× |
| 512 | 2.632 | 0.864 | 3.05× |
| 1024 | 4.688 | 1.461 | 3.21× |
| 2048 | 11.992 | 2.287 | 5.24× |
| 4096 | 27.049 | 4.924 | 5.50× |