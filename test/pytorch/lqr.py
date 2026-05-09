import torch
import numpy as np
from scipy.linalg import solve_discrete_are
from ...src import TorchDDP

torch.manual_seed(0)
dtype  = torch.float64
device = "cpu"

nx, nu = 2, 1
dt     = 0.05
T      = 50

def dynamics(x, u):
    A = torch.tensor([[1.0, dt], [0.0, 1.0]], dtype=dtype)
    B = torch.tensor([[0.0], [dt]],            dtype=dtype)
    return A @ x + B @ u

def running_cost(x, u):
    return 0.5 * (x @ x + 0.1 * (u @ u))

def terminal_cost(x):
    return 5.0 * (x @ x)

x0      = torch.tensor([1.0, 0.0], dtype=dtype).unsqueeze(0)
us_init = torch.zeros(1, T, nu, dtype=dtype)

solver = TorchDDP(dynamics, running_cost, terminal_cost, nx=nx, nu=nu)
xs, us = solver.solve(x0, us_init, n_iter=50)

print("DDP final state:", xs[0, -1])
print("DDP final cost: ", solver.total_cost(xs, us)[0].item())

A = np.array([[1.0, dt], [0.0, 1.0]])
B = np.array([[0.0], [dt]])
Q = np.eye(nx)
R = np.array([[0.1]])

P = solve_discrete_are(A, B, Q, R)
cost = 0.5 * x0.squeeze().numpy() @ P @ x0.squeeze().numpy()
print("LQR optimal cost:", cost)
