import numpy as np
from scipy.special import erfc
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def update(grid, N, Dt):
    grid2 = grid.copy()
    dx2 = (1/N)**2

    #makin it wrap around
    left  = np.roll(grid,1,axis=1)
    right = np.roll(grid,-1,axis=1)

    #laplace
    lap = left[1:-1, :] +right[1:-1, :] + grid[:-2,:] + grid[2:,:] - 4*grid[1:-1, :]
    grid2[1:-1, :] = grid[1:-1, :] + (Dt/dx2) * lap

    return grid2


def get_dt(n, lim = 1.0):
    #gets a stable dt for the chosen N
    dx = 1.0 / n
    dx2 = dx * dx
    return lim * dx2 / (4.0)

def c_analytic(y, t, n_terms=2000):

    y = np.asarray(y, dtype=float)

    if t == 0:
        return np.zeros_like(y)

    denom = 2.0 * np.sqrt(t)
    res = np.zeros_like(y)
    for i in range(n_terms):
        a = (1 - y + 2.0*i)/denom
        b = (1 + y + 2.0*i)/denom
        res += erfc(a) - erfc(b)

    return res

def get_snapshots(n, dt, t_end, target_times):

    grid = np.zeros((n, n))
    grid[n - 1, :] = 1.0

    snapshots = {}
    t = 0.0

    while t <= t_end:
        t_old = t
        grid = update(grid, n, dt)
        t += dt

        for target in target_times:
            if target not in snapshots and (t_old < target <= t):
                snapshots[target] = grid.copy()

    return snapshots

def plot_comparison(snapshots, target_times, n):
    y = np.linspace(0, 1, n)

    plt.figure(figsize=(7, 5))
    for t in target_times:
        grid_snapshot = snapshots[t]
        c_num = grid_snapshot.mean(axis=1)
        c_exact = c_analytic(y,t)

        plt.plot(y, c_exact, linewidth=2)
        plt.plot(y, c_num, linestyle="--", linewidth=1.5)

        # label time
        idx = int(0.8 * len(y))
        plt.text(y[idx], c_exact[idx], f"{t}", fontsize=10, va="center")

    plt.xlabel("y")
    plt.ylabel("c(y,t)")
    plt.title("anlytic vs simulation at different times")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("comparison_all_times.png", dpi=200)
    plt.close()

def plot_heatmaps(snapshots, target_times):
    for t in target_times:
        grid_snapshot = snapshots[t]

        plt.figure()
        plt.imshow(grid_snapshot, origin="lower", cmap="viridis", vmin=0, vmax=1, aspect="auto")
        plt.colorbar(label="Concentration")
        plt.title(f"2D Diffusion at t = {t}")
        plt.xlabel("i")
        plt.ylabel("j")
        plt.tight_layout()
        plt.savefig(f"heatmap_t_{t}.png", dpi=200)
        plt.close()


def animate(_frame_idx, state, t_end, skip, n, dt, im, title):
    if state["t"] >= t_end:
        return (im, title)

    for _ in range(skip):
        if state["t"] >= t_end:
            break
        state["grid"] = update(state["grid"], n, dt)
        state["t"] += dt

    im.set_data(state["grid"])
    title.set_text(f"2D Diffusion, t = {state['t']:.5f}")
    return (im, title)


def make_animation(n, t_end, dt, skip= 5, outfile="diffusion.gif"):
    dx = 1.0 / n
    grid = np.zeros((n, n))
    grid[n-1, :]=1.0
    fig, ax = plt.subplots()
    im = ax.imshow(grid, origin="lower", vmin=0, vmax=1, aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Concentration")

    title = ax.set_title("2D Diffusion, t = 0.0000")
    ax.set_xlabel("i")
    ax.set_ylabel("j")

    state = {"t": 0.0, "grid": grid}

    frames = int(np.ceil(t_end / (skip * dt)))
    anim = FuncAnimation(fig, animate,
        fargs=(state, t_end, skip, n, dt, im, title),
        frames=frames,interval=30,blit=False,)
    
    anim.save(outfile, writer=PillowWriter(fps=30))
    plt.close(fig)


#MAIN CODE
n = 100
target_times = [0.001, 0.01, 0.1, 1.0]
dt = get_dt(n)
print(dt)

snapshots = get_snapshots(n, dt, t_end=1.0, target_times=target_times)
plot_comparison(snapshots, target_times, n)
plot_heatmaps(snapshots, target_times)
make_animation(n, t_end=0.5, dt=dt)