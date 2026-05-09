import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter

def gif_cartpole(xs, dt: float, l: float, path: str = "cartpole.gif", fps: int = 30):
    xs_np = xs.detach().cpu().numpy() if hasattr(xs, 'detach') else np.asarray(xs)
    T     = len(xs_np) - 1
 
    cart_w, cart_h = 0.4, 0.2
    pole_len        = 2 * l
 
    p_min = xs_np[:, 0].min() - 1.0
    p_max = xs_np[:, 0].max() + 1.0
 
    fig, (ax_main, ax_theta) = plt.subplots(
        2, 1, figsize=(8, 6),
        gridspec_kw={"height_ratios": [3, 1]}
    )
    fig.tight_layout(pad=2.0)
 
    # --- main cartpole axes ---
    ax_main.set_xlim(p_min, p_max)
    ax_main.set_ylim(-pole_len - 0.3, pole_len + 0.3)
    ax_main.set_aspect("equal")
    ax_main.axhline(0, color="gray", lw=1, ls="--")
    ax_main.set_title("Cartpole trajectory")
 
    cart_rect = patches.Rectangle(
        (0, 0), cart_w, cart_h,
        linewidth=1.5, edgecolor="steelblue", facecolor="steelblue", alpha=0.8
    )
    ax_main.add_patch(cart_rect)
    pole_line, = ax_main.plot([], [], "o-", lw=3, color="tomato", markersize=8)
    time_text   = ax_main.text(0.02, 0.95, "", transform=ax_main.transAxes, fontsize=10)
 
    t_arr = np.linspace(0, T * dt, T + 1)
    ax_theta.plot(t_arr, xs_np[:, 1], color="tomato", lw=1.5)
    ax_theta.axhline(0, color="gray", lw=1, ls="--")
    ax_theta.set_xlabel("time [s]")
    ax_theta.set_ylabel("θ [rad]")
    ax_theta.set_xlim(0, T * dt)
    vline = ax_theta.axvline(0, color="steelblue", lw=1.5, ls=":")
 
    def init():
        cart_rect.set_xy((-cart_w / 2, -cart_h / 2))
        pole_line.set_data([], [])
        time_text.set_text("")
        vline.set_xdata([0])
        return cart_rect, pole_line, time_text, vline
 
    def update(frame):
        p, theta = xs_np[frame, 0], xs_np[frame, 1]
 
        cart_rect.set_xy((p - cart_w / 2, -cart_h / 2))
 
        tip_x = p + pole_len * np.sin(theta)
        tip_y =     pole_len * np.cos(theta)
        pole_line.set_data([p, tip_x], [0, tip_y])
 
        time_text.set_text(f"t = {frame * dt:.2f}s   θ = {theta:.3f} rad")
        vline.set_xdata([frame * dt])
        return cart_rect, pole_line, time_text, vline
 
    step   = max(1, int(1.0 / (fps * dt)))
    frames = range(0, T + 1, step)
 
    anim = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Saved: {path}")
