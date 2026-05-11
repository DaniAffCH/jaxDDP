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
B      = 32

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

thetas  = jnp.linspace(-3.14, 3.14, B)
x0      = jnp.zeros((B, nx))
x0      = x0.at[:, 1].set(thetas)
us_init = jnp.zeros((B, T, nu))

t0 = time.time()
solver = JaxDDP(dynamics, running_cost, terminal_cost, nx=nx, nu=nu)
print(f"Solver setup time: {time.time() - t0}")

t0 = time.time()
xs, us = solver.solve(x0, us_init, 100, 0.01)
jax.block_until_ready((xs, us))
print(f"Solver solve time (includes JIT compilation): {time.time() - t0}")

t0 = time.time()
xs, us = solver.solve(x0, us_init, 100, 0.01)
jax.block_until_ready((xs, us))
print(f"Solver solve time (compiled): {time.time() - t0}")

costs = solver.total_cost(xs, us)
print(f"Final costs — mean: {float(costs.mean()):.4f}  "
      f"min: {float(costs.min()):.4f}  "
      f"max: {float(costs.max()):.4f}")

best  = int(costs.argmin())
worst = int(costs.argmax())

gif_cartpole(np.array(xs[best]), dt=dt, l=l, path="cartpole_best.gif", fps=15)
print(f"Saved best problem {best} (theta0={float(thetas[best]):.2f}, cost={float(costs[best]):.4f})")

gif_cartpole(np.array(xs[worst]), dt=dt, l=l, path="cartpole_worst.gif", fps=15)
print(f"Saved worst problem {worst} (theta0={float(thetas[worst]):.2f}, cost={float(costs[worst]):.4f})")
