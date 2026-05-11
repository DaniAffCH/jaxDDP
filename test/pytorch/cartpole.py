import torch
import numpy as np
from jaxddp import TorchDDP
from ..utils import gif_cartpole
import time

dtype  = torch.float64
device = "cpu"

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
        sin_t = torch.sin(theta)
        cos_t = torch.cos(theta)
        denom = mc + mp * sin_t**2
        p_ddot     = (u[0] + mp * sin_t * (l * thetadot**2 - g * cos_t)) / denom
        theta_ddot = (g * sin_t * (mc + mp) - cos_t * (u[0] + mp * l * thetadot**2 * sin_t)) / (l * denom)
        return torch.stack([pdot, thetadot, p_ddot, theta_ddot])

    k1 = f(x)
    k2 = f(x + dt/2 * k1)
    k3 = f(x + dt/2 * k2)
    k4 = f(x + dt   * k3)
    return x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

def running_cost(x, u):
    return 0.5 * (x[0]**2 + 10.0 * x[1]**2 + 1 * (u @ u))

def terminal_cost(x):
    return 50.0 * (x[0]**2 + 10.0 * x[1]**2 + 3 * x[2]**2 + 3 * x[3]**2)

x0      = torch.tensor([0.0, -3.14, 0.0, 0.0], dtype=dtype).unsqueeze(0)
us_init = torch.zeros(1, T, nu, dtype=dtype)

t0 = time.time()
solver = TorchDDP(dynamics, running_cost, terminal_cost, nx=nx, nu=nu)
print(f"Solver setup time: {time.time() - t0}")
t0 = time.time()
xs, us = solver.solve(x0, us_init, n_iter=100, reg=0.01)
print(f"Solver solve time: {time.time() - t0}")

print("Final state:", xs[0, -1])
print("Final cost: ", solver.total_cost(xs, us)[0].item())

gif_cartpole(xs[0], dt=dt, l=l, path="cartpole.gif", fps=15)
