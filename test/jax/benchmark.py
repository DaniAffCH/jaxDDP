import jax
import jax.numpy as jnp
import numpy as np
import time
import matplotlib.pyplot as plt
from jaxddp import JaxDDP

jax.config.update("jax_enable_x64", True)

nx, nu = 4, 1
dt     = 0.05
T      = 50
mc, mp, l, g = 1.0, 0.1, 0.5, 9.81

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
    return 0.5 * (x[0]**2 + 10.0 * x[1]**2 + (u @ u))

def terminal_cost(x):
    return 50.0 * (x[0]**2 + 10.0 * x[1]**2 + 3*x[2]**2 + 3*x[3]**2)

N_ITER = 50
REG    = 0.01

LARGE_BATCH_MODE = True

BATCH_SIZES = (
    [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    if LARGE_BATCH_MODE else
    [1, 2, 4, 8, 16, 32, 64]
)

try:
    gpu_device = jax.devices('gpu')[0]
    has_gpu = True
except RuntimeError:
    has_gpu = False

cpu_device = jax.devices('cpu')[0]

if has_gpu:
    print("GPU detected:", gpu_device)
else:
    print("No GPU detected, running CPU only.")

def make_problem(B, device=None):
    thetas  = jnp.linspace(-3.14, 3.14, B)
    x0      = jnp.zeros((B, nx))
    x0      = x0.at[:, 1].set(thetas)
    us_init = jnp.zeros((B, T, nu))
    if device is not None:
        x0      = jax.device_put(x0, device)
        us_init = jax.device_put(us_init, device)
    return x0, us_init

def time_sequential(solver, B, device=None, n_repeat=3):
    # Warmup: compile for batch size 1
    x0_w, us_w = make_problem(1, device)
    jax.block_until_ready(solver.solve(x0_w, us_w, N_ITER, REG))

    times = []
    for _ in range(n_repeat):
        x0, us_init = make_problem(B, device)
        t0 = time.time()
        for b in range(B):
            result = solver.solve(x0[b:b+1], us_init[b:b+1], N_ITER, REG)
            jax.block_until_ready(result)
        times.append(time.time() - t0)
    return np.mean(times)

def time_batched(solver, B, device=None, n_repeat=3):
    # Warmup: compile for this batch size
    x0_w, us_w = make_problem(B, device)
    jax.block_until_ready(solver.solve(x0_w, us_w, N_ITER, REG))

    times = []
    for _ in range(n_repeat):
        x0, us_init = make_problem(B, device)
        t0 = time.time()
        result = solver.solve(x0, us_init, N_ITER, REG)
        jax.block_until_ready(result)
        times.append(time.time() - t0)
    return np.mean(times)

solver = JaxDDP(dynamics, running_cost, terminal_cost, nx=nx, nu=nu)

results = {"B": BATCH_SIZES, "batched_cpu": [], "speedup_cpu": []}
if not LARGE_BATCH_MODE:
    results["sequential"] = []
if has_gpu:
    results["batched_gpu"] = []
    results["speedup_gpu"] = []

header = f"{'B':>6}  {'batched CPU (s)':>16}"
if not LARGE_BATCH_MODE:
    header = f"{'B':>6}  {'sequential (s)':>16}  {'batched CPU (s)':>16}  {'speedup CPU':>12}"
if has_gpu:
    header += f"  {'batched GPU (s)':>16}  {'speedup GPU':>12}"
print("\n" + header)

for B in BATCH_SIZES:
    t_batch_cpu = time_batched(solver, B, cpu_device)
    results["batched_cpu"].append(t_batch_cpu)

    if LARGE_BATCH_MODE:
        row = f"{B:>6}  {t_batch_cpu:>16.3f}"
    else:
        t_seq = time_sequential(solver, B, cpu_device)
        speedup_cpu = t_seq / t_batch_cpu
        results["sequential"].append(t_seq)
        results["speedup_cpu"].append(speedup_cpu)
        row = f"{B:>6}  {t_seq:>16.3f}  {t_batch_cpu:>16.3f}  {speedup_cpu:>12.2f}x"

    if has_gpu:
        t_batch_gpu = time_batched(solver, B, gpu_device)
        results["batched_gpu"].append(t_batch_gpu)
        if not LARGE_BATCH_MODE:
            speedup_gpu = t_seq / t_batch_gpu
            results["speedup_gpu"].append(speedup_gpu)
            row += f"  {t_batch_gpu:>16.3f}  {speedup_gpu:>12.2f}x"
        else:
            row += f"  {t_batch_gpu:>16.3f}"

    print(row)

B_arr  = np.array(BATCH_SIZES)
colors = {"sequential": "#e07b54", "batched_cpu": "#4c8eda", "batched_gpu": "#4cbe7a"}

if LARGE_BATCH_MODE:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(B_arr, results["batched_cpu"], "s-", color=colors["batched_cpu"], label="Batched CPU", lw=2)
    if has_gpu:
        ax.plot(B_arr, results["batched_gpu"], "^-", color=colors["batched_gpu"], label="Batched GPU", lw=2)
    ax.set_xlabel("Batch size B")
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title("Batched JAX DDP — wall-clock time vs batch size")
    ax.set_xscale("log", base=2)
    ax.set_xticks(B_arr)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.legend()
    ax.grid(True, alpha=0.3)
else:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(B_arr, results["sequential"],  "o-", color=colors["sequential"],  label="Sequential (CPU)", lw=2)
    ax1.plot(B_arr, results["batched_cpu"], "s-", color=colors["batched_cpu"], label="Batched (CPU)",     lw=2)
    if has_gpu:
        ax1.plot(B_arr, results["batched_gpu"], "^-", color=colors["batched_gpu"], label="Batched (GPU)", lw=2)
    ax1.set_xlabel("Batch size B")
    ax1.set_ylabel("Wall-clock time (s)")
    ax1.set_title("Wall-clock time vs batch size")
    ax1.legend()
    ax1.set_xticks(B_arr)
    ax1.grid(True, alpha=0.3)

    ax2.plot(B_arr, results["speedup_cpu"], "s-", color=colors["batched_cpu"], label="Batched CPU / Sequential", lw=2)
    if has_gpu:
        ax2.plot(B_arr, results["speedup_gpu"], "^-", color=colors["batched_gpu"], label="Batched GPU / Sequential", lw=2)
    ax2.axhline(1.0, color="gray", ls="--", lw=1)
    ax2.set_xlabel("Batch size B")
    ax2.set_ylabel("Speedup (×)")
    ax2.set_title("Speedup over sequential")
    ax2.legend()
    ax2.set_xticks(B_arr)
    ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("benchmark_jax.png", dpi=150)
plt.show()
print("\nSaved benchmark_jax.png")
