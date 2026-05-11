# jaxDDP

A JAX-based differentiable dynamic programming implementation. It solves batched problems on GPUs.

## Installation

Install the CUDA-enabled JAX for your system first:
```bash
pip install "jax[cuda12]"   # CUDA 12
pip install "jax[cuda11]"   # CUDA 11
pip install jax             # CPU only
```
Then install jaxddp:

```bash
pip install -e .
```

## Usage

Define your dynamics and cost functions, then solve:

```python
import jax.numpy as jnp
from jaxddp import JaxDDP

# scalar-in, scalar-out functions operating on a single (x, u) pair
def dynamics(x, u):
    A = jnp.array([[1.0, 0.05], [0.0, 1.0]])
    B = jnp.array([[0.0], [0.05]])
    return A @ x + B @ u

def running_cost(x, u):
    return 0.5 * (x @ x + 0.1 * (u @ u))

def terminal_cost(x):
    return 5.0 * (x @ x)

solver = JaxDDP(dynamics, running_cost, terminal_cost, nx=2, nu=1)

# x0: (B, nx)  us_init: (B, T, nu)  — batched over B independent problems
x0      = jnp.array([[1.0, 0.0]])
us_init = jnp.zeros((1, 50, 1))

xs, us = solver.solve(x0, us_init, n_iter=100)
```

See `test/` for cartpole and LQR examples.

## Benchmark

Results on NVIDIA GeForce RTX 5090 / AMD Ryzen 9-9950X.
To reproduce on your hardware:
```bash
python test/jax/benchmark.py
```

Batched CPU vs batched GPU:

![benchmark_jax_gpu](imgs/benchmark_jax_gpu.png)

| B | CPU (s) | GPU (s) | GPU speedup |
|---|---|---|---|
| 1 | 0.006 | 0.225 | 0.03× |
| 2 | 0.010 | 0.217 | 0.05× |
| 4 | 0.014 | 0.230 | 0.06× |
| 8 | 0.027 | 0.265 | 0.10× |
| 16 | 0.047 | 0.272 | 0.17× |
| 32 | 0.100 | 0.328 | 0.30× |
| 64 | 0.177 | 0.323 | 0.55× |
| 128 | 0.325 | 0.349 | 0.93× |
| 256 | 0.625 | 0.381 | 1.64× |
| 512 | 1.356 | 0.444 | 3.05× |
| 1024 | 2.786 | 0.524 | 5.32× |
| 2048 | 5.932 | 0.672 | 8.83× |
| 4096 | 14.622 | 1.278 | 11.44× |