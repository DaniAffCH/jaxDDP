import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import solve_discrete_are
from jaxddp import JaxDDP

jax.config.update("jax_enable_x64", True)

nx, nu = 2, 1
dt     = 0.05
T      = 50

def dynamics(x, u):
    A = jnp.array([[1.0, dt], [0.0, 1.0]])
    B = jnp.array([[0.0], [dt]])
    return A @ x + B @ u

def running_cost(x, u):
    return 0.5 * (x @ x + 0.1 * (u @ u))

def terminal_cost(x):
    return 5.0 * (x @ x)

x0      = jnp.array([[1.0, 0.0]])
us_init = jnp.zeros((1, T, nu))

solver = JaxDDP(dynamics, running_cost, terminal_cost, nx=nx, nu=nu)
result = solver.solve(x0, us_init, 50)
xs, us = result["xs"], result["us"]

print("DDP final state:", xs[0, -1])
print(f"DDP final cost:  {float(result['cost'][0]):.4f}  (iters: {int(result['n_iter'])})")

A = np.array([[1.0, dt], [0.0, 1.0]])
B = np.array([[0.0], [dt]])
Q = np.eye(nx)
R = np.array([[0.1]])

P = solve_discrete_are(A, B, Q, R)
cost = 0.5 * np.array([1.0, 0.0]) @ P @ np.array([1.0, 0.0])
print("LQR optimal cost:", cost)
