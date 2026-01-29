import numpy as np
import matplotlib.pyplot as plt
from jax import numpy as jnp
import matplotlib.cm as cm

# ----------------------------
# Single Pendulum Positions
# ----------------------------
def single_pendulum_positions(theta, l=1.0):
    x = l * np.sin(theta)
    y = -l * np.cos(theta)
    return x, y

# ----------------------------
# Condensed trajectory plot
# ----------------------------
def plot_single_pendulum_time_condensed(
    theta,
    timespan=None,          # Pass actual time array
    sample=20,
    l=1.0,
    total_width=None,       # Optional: if None, use real timespan
    ground_color='#7fbf7f',
    sampling_strategy='uniform',  # 'uniform', 'sqrt', 'log'
    colormap='viridis',
    filename=None
):
    """
    Condensed single pendulum trajectory:
    - Links: gradient colormap along time
    - Joint: tip green
    """
    x, y = single_pendulum_positions(theta, l)

    # Determine total width
    if total_width is None:
        total_width = timespan[-1] if timespan is not None else 10.0

    # Sampling strategy
    if sampling_strategy == 'sqrt':
        t_normalized = np.linspace(0, 1, sample) ** 2
        idx = (t_normalized * (len(theta) - 1)).astype(int)
    elif sampling_strategy == 'log':
        t_normalized = np.logspace(0, 1, sample) - 1
        t_normalized = t_normalized / t_normalized[-1]
        idx = (t_normalized * (len(theta) - 1)).astype(int)
    else:
        idx = np.linspace(0, len(theta) - 1, sample).astype(int)

    # Colormap for links
    cmap = cm.get_cmap(colormap)

    fig, ax = plt.subplots(figsize=(6.8, 2.2))

    # Determine ground level
    all_y = list(y[idx])
    ground_y = min(all_y) - 0.3

    for k, i in enumerate(idx):
        # Map sampled index to horizontal position
        if timespan is not None:
            x_offset = (timespan[i] / timespan[idx[-1]]) * total_width
        else:
            x_offset = (k / (sample - 1)) * total_width

        # Gradient color along time
        color = cmap(k / (sample - 1))

        # Pendulum link
        ax.plot([x_offset, x_offset + x[i]],
                [0, y[i]],
                lw=1.5,
                color=color,
                alpha=0.9)

        # Tip joint (green)
        ax.scatter([x_offset + x[i]],
                   [y[i]],
                   s=5,
                   color='black',
                   zorder=4)

    # Ground line
    ax.plot([-0.3, total_width + 0.3],
            [ground_y, ground_y],
            color=ground_color,
            lw=4)

    ax.set_aspect('equal')
    ax.yaxis.set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if timespan is not None:
        n_ticks = 11
        tick_positions = np.linspace(0, total_width, n_ticks)
        tick_labels = [str(i) for i in range(0, n_ticks)]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_xlim(-0.3, total_width + 0.3)
        ax.set_ylim(ground_y - 0.2, l + 0.5)
        ax.set_xlabel('Time (s)')

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()



# ----------------------------
# Example usage
# ----------------------------
name = 'SINGLE_PENDULUM-h=600-gamma=1.0-shots=512-iter=1-elite=0.1-smooth=0.1-alpha=1.0-dt=0.01'

# Load trajectory data
X = jnp.load(name + '-traj.npy')
T_swing_up = 1000  # <-- can change freely
dt = 0.01
t = jnp.arange(T_swing_up) * dt
theta = X[:T_swing_up, 0]  # single angle

hist = jnp.load(name + '-hist.npy')[:T_swing_up]

# Plot trajectory synced to real time
plot_single_pendulum_time_condensed(
    theta,
    timespan=t,
    sample=150,
    total_width=None,  # will automatically match real time
    sampling_strategy='uniform',
    filename='single_pendulum_traj.png'
)

# Plot bits/s over time (automatically matches trajectory)
fig, ax = plt.subplots(1, 1, figsize=(6, 3))
ax.set_xlabel('Time (s)')
ax.set_ylabel('nats/s')
ax.plot(t, hist)
fig.tight_layout()
fig.savefig('single_pendulum_hist.png', dpi=300)
plt.show()
