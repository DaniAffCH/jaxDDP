# torchDDP

A PyTorch-based differentiable dynamic programming implementation. It solves batched problems on GPUs.

PyTorch Benchmark on AMD Ryzen 9 9950X batched vs unbatched:

![benchmark_torch](imgs/benchmark_torch.png)

JAX Benchmark on NVIDIA GeForce RTX 4060 and Intel Core i9-14900HX batched CPU vs batched GPU:

![benchmark_jax_gpu](imgs/benchmark_jax_gpu.png)
