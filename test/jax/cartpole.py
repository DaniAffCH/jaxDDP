import jax
import jax.numpy as jnp
import numpy as np
from jaxddp import JaxDDP
from ..utils import gif_cartpole
import time

jax.config.update("jax_enable_x64", True)

nx, nu = 4, 1
dt     = 0.05
T      = 50

mc = 1.0
mp = 0.1
l  = 0.5
g  = 9.81

def dynamics(x, u):
    def f(x):
        p, theta, pdot, thetadot = x[0], x[1], x[2], x[3]
        sin_t = jnp.sin(theta)
        cos_t = jnp.cos(theta)
        denom = mc + mp * sin_t**2
        p_ddot     = (u[0] + mp * sin_t * (l * thetadot**2 - g * cos_t)) / denom
        theta_ddot = (g * sin_t * (mc + mp) - cos_t * (u[0] + mp * l * thetadot**2 * sin_t)) / (l * denom)
        return jnp.stack([pdot, thetadot, p_ddot, theta_ddot])

    k1 = f(x)
    k2 = f(x + dt/2 * k1)
    k3 = f(x + dt/2 * k2)
    k4 = f(x + dt   * k3)
    return x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

def running_cost(x, u):
    return 0.5 * (x[0]**2 + 10.0 * x[1]**2 + 1 * (u @ u))

def terminal_cost(x):
    return 50.0 * (x[0]**2 + 10.0 * x[1]**2 + 3 * x[2]**2 + 3 * x[3]**2)

x0      = jnp.array([[0.0, -3.14, 0.0, 0.0]])
us_init = jnp.zeros((1, T, nu))

t0 = time.time()
solver = JaxDDP(dynamics, running_cost, terminal_cost, nx=nx, nu=nu)
print(f"Solver setup time: {time.time() - t0}")

t0 = time.time()
result = solver.solve(x0, us_init, 100, 0.01)
jax.block_until_ready(result)
print(f"Solver solve time (includes JIT compilation): {time.time() - t0}")

t0 = time.time()
result = solver.solve(x0, us_init, 100, 0.01)
jax.block_until_ready(result)
print(f"Solver solve time (compiled): {time.time() - t0}")

xs, us = result["xs"], result["us"]
print("Final state:", xs[0, -1])
print(f"Final cost:  {float(result['cost'][0]):.4f}  (iters: {int(result['n_iter'])})")

gif_cartpole(np.array(xs[0]), dt=dt, l=l, path="cartpole.gif", fps=15)
