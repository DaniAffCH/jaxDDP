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

result = solver.solve(x0, us_init, n_iter=100)
xs   = result["xs"]     # (B, T+1, nx) state trajectory
us   = result["us"]     # (B, T, nu)   optimised actions
cost = result["cost"]   # (B,)         total cost per problem
it   = result["n_iter"] # scalar       iterations until convergence
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
| 1 | 0.009 | 0.195 | 0.05× |
| 2 | 0.013 | 0.182 | 0.07× |
| 4 | 0.022 | 0.188 | 0.12× |
| 8 | 0.041 | 0.182 | 0.23× |
| 16 | 0.077 | 0.171 | 0.45× |
| 32 | 0.156 | 0.177 | 0.88× |
| 64 | 0.287 | 0.181 | 1.59× |
| 128 | 0.514 | 0.171 | 3.01× |
| 256 | 0.967 | 0.193 | 5.01× |
| 512 | 1.670 | 0.234 | 7.14× |
| 1024 | 2.929 | 0.310 | 9.45× |
| 2048 | 5.758 | 0.452 | 12.74× |
| 4096 | 12.124 | 0.741 | 16.36× |