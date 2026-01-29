import numpy as np
import matplotlib.pyplot as plt
from jax import numpy as jnp
import matplotlib.cm as cm

def double_pendulum_positions(theta0, theta1, l0=1.0, l1=1.0):
    x0 = l0 * np.sin(theta0)
    y0 = -l0 * np.cos(theta0)
    x1 = x0 + l1 * np.sin(theta0 + theta1)
    y1 = y0 - l1 * np.cos(theta0 + theta1)
    return x0, y0, x1, y1

def plot_trajectory_time_condensed(
    theta0,
    theta1,
    sample=20,
    l0=1.0,
    l1=1.0,
    total_width=10.0,
    ground_color='#7fbf7f',
    sampling_strategy='uniform',
    colormap='viridis',
    filename=None,
    timespan=None
):
    """
    Condensed double pendulum trajectory:
    - Links: gradient colormap (viridis)
    - Joints: first joint (between links) black
              second joint (tip) green
    """
    x0, y0, x1, y1 = double_pendulum_positions(theta0, theta1, l0, l1)
    
    # Sampling strategy
    if sampling_strategy == 'sqrt':
        t_normalized = np.linspace(0, 1, sample) ** 2
        idx = (t_normalized * (len(theta0) - 1)).astype(int)
    elif sampling_strategy == 'log':
        t_normalized = np.logspace(0, 1, sample) - 1
        t_normalized = t_normalized / t_normalized[-1]
        idx = (t_normalized * (len(theta0) - 1)).astype(int)
    else:
        idx = np.linspace(0, len(theta0) - 1, sample).astype(int)
    
    # Colormap for links
    cmap = cm.get_cmap(colormap)

    fig, ax = plt.subplots(figsize=(6.8, 2.2))
    
    # Determine ground level (for spacing)
    all_y = []
    for i in idx:
        all_y.extend([0, y0[i], y1[i]])
    ground_y = min(all_y) - 0.3

    for k, i in enumerate(idx):
        x_offset = (k / (sample - 1)) * total_width
        color = cmap(k / (sample - 1))  # Gradient along time

        # Draw links
        ax.plot([x_offset, x_offset + x0[i]],
                [0, y0[i]],
                lw=1.5,
                color=color,
                alpha=0.9)
        
        ax.plot([x_offset + x0[i], x_offset + x1[i]],
                [y0[i], y1[i]],
                lw=1.5,
                color=color,
                alpha=0.9)

        # Draw joints: first joint (between links) black, tip green
        joint_x = [x_offset + x0[i], x_offset + x1[i]]
        joint_y = [y0[i], y1[i]]
        joint_colors = ['black', 'black']
        ax.scatter(joint_x, joint_y,
                   s=5,
                   color=joint_colors,
                   zorder=4)

    # Draw ground line
    ax.plot([-0.3, total_width + 0.3],
            [ground_y, ground_y],
            color=ground_color,
            lw=4)

    ax.set_aspect('equal')
    ax.yaxis.set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if timespan is not None:
        n_ticks = 13
        tick_positions = np.linspace(0, total_width, num=n_ticks)
        tick_labels = [str(i) for i in range(0, n_ticks)]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_xlim(-0.3, total_width + 0.3)
        ax.set_ylim(ground_y - 0.2, 2.2)
        ax.set_xlabel('Time (s)')

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=500, bbox_inches='tight')
    plt.show()


# # Condensed, publication-safe version:
# # Time is mapped to a fixed horizontal span (single-column friendly)

# import numpy as np
# import matplotlib.pyplot as plt
# from jax import numpy as jnp

# def double_pendulum_positions(theta0, theta1, l0=1.0, l1=1.0):
#     x0 = l0 * np.sin(theta0)
#     y0 = -l0 * np.cos(theta0)
#     x1 = x0 + l1 * np.sin(theta0 + theta1)
#     y1 = y0 - l1 * np.cos(theta0 + theta1)
#     return x0, y0, x1, y1

# def plot_trajectory_time_condensed(
#     theta0,
#     theta1,
#     sample=20,
#     l0=1.0,
#     l1=1.0,
#     total_width=10.0,
#     link_color='#c49a6c',
#     joint_color='#8b5a2b',
#     ground_color='#7fbf7f',
#     sampling_strategy='uniform',  # 'uniform', 'sqrt', or 'log'
#     filename = None,
#     timespan = None
# ):
#     '''
#     DIAYN-style condensed trajectory:
#     time mapped to a fixed horizontal span.
    
#     sampling_strategy:
#         - 'uniform': evenly spaced samples
#         - 'sqrt': denser at beginning (square root spacing)
#         - 'log': very dense at beginning (logarithmic spacing)
#     '''
#     x0, y0, x1, y1 = double_pendulum_positions(theta0, theta1, l0, l1)
    
#     # Choose sampling strategy
#     if sampling_strategy == 'sqrt':
#         # Square root spacing - denser at beginning
#         t_normalized = np.linspace(0, 1, sample) ** 2
#         idx = (t_normalized * (len(theta0) - 1)).astype(int)
#     elif sampling_strategy == 'log':
#         # Logarithmic spacing - very dense at beginning
#         t_normalized = np.logspace(0, 1, sample) - 1
#         t_normalized = t_normalized / t_normalized[-1]
#         idx = (t_normalized * (len(theta0) - 1)).astype(int)
#     else:
#         # Uniform spacing (original)
#         idx = np.linspace(0, len(theta0) - 1, sample).astype(int)

#     fig, ax = plt.subplots(figsize=(6.8, 2.2))

#     # Collect all y positions to determine appropriate ground level
#     all_y = []
#     for i in idx:
#         all_y.extend([0, y0[i], y1[i]])
    
#     ground_y = min(all_y) - 0.3

#     for k, i in enumerate(idx):
#         x_offset = (k / (sample - 1)) * total_width

#         # first link
#         ax.plot(
#             [x_offset, x_offset + x0[i]],
#             [0, y0[i]],
#             lw=1.5,
#             color=link_color,
#             alpha=0.9,
#         )

#         # second link
#         ax.plot(
#             [x_offset + x0[i], x_offset + x1[i]],
#             [y0[i], y1[i]],
#             lw=1.5,
#             color=link_color,
#             alpha=0.9,
#         )

#         # joints
#         ax.scatter(
#             [x_offset, x_offset + x0[i], x_offset + x1[i]],
#             [0, y0[i], y1[i]],
#             s=10,
#             color=joint_color,
#             zorder=3,
#         )

#     # ground line
#     ax.plot(
#         [-0.3, total_width + 0.3],
#         [ground_y, ground_y],
#         color=ground_color,
#         lw=4,
#     )

#     ax.set_aspect('equal')
#     # ax.axis('off')
#     ax.yaxis.set_visible(False)
#     # Turn off the frame
#     for spine in ax.spines.values():
#         spine.set_visible(False)

#     if timespan is not None:
#         n_ticks = 13
#         tick_positions = np.linspace(0, total_width, num=n_ticks)  # positions along the x-axis
#         tick_labels = [str(i) for i in range(0, n_ticks)]

#         ax.set_xticks(tick_positions)
#         ax.set_xticklabels(tick_labels)

#         ax.set_xlim(-0.3, total_width + 0.3)
#         ax.set_ylim(ground_y - 0.2, 2.2)
#         ax.set_xlabel('Time (s)')

#     plt.tight_layout()
#     if filename:
#         plt.savefig(filename, dpi=300, bbox_inches='tight')
#     plt.show()

# name = 'DOUBLE_PENDULUM-h=512-gamma=1.0-shots=512-iter=10-elite=0.1-smooth=0.1-alpha=1.0-dt=0.01'
name = 'DOUBLE_PENDULUM-gear=6.0-h=512-gamma=1.0-shots=512-iter=10-elite=0.1-smooth=0.1-alpha=1.0-dt=0.01'

# ---- Load and process data ----
X = jnp.load(name + '-traj.npy')

# Focus on swing-up stage (before timestep 800)
T_swing_up = 1200
t = jnp.arange(T_swing_up) * 0.01
theta0 = X[:T_swing_up, 0]
theta1 = X[:T_swing_up, 1]

hist = jnp.load(name + '-hist.npy')[:T_swing_up]

# Visualize the trajectory with denser sampling at the beginning
plot_trajectory_time_condensed(
    theta0,
    theta1,
    sample = 150,
    total_width = 15,
    sampling_strategy = 'uniform',  # Try 'sqrt' or 'log' for denser early sampling
    filename = 'double_pendulum_traj.png',
    timespan = t
)

fig, ax = plt.subplots(1, 1, figsize = (6, 3))
ax.set_xlabel('Time (s)')
ax.set_ylabel('nats/s')
ax.plot(t, hist)
fig.tight_layout()
fig.savefig('double_pendulum_hist.png', dpi = 300)
plt.show()