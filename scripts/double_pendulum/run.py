import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['MUJOCO_GL'] = 'egl'

import jax
from jax import numpy as jnp
import matplotlib.pyplot as plt

from val import Dynamics, make_step, make_unroll
from val.cem import CEM
from val.info import make_compute_rate

if __name__ == '__main__':
    seed = 0
    key = jax.random.PRNGKey(seed)

    dt = 0.01
    horizon = 512
    shots = 512
    steps = 2000

    iterations = 10
    elite_frac = 0.1
    smoothing = 0.1
    alpha = 1.0
    rho = 0.9
    gamma = 1.0

    name = f'DOUBLE_PENDULUM-gear=6.0-h={horizon}-gamma={gamma}-shots={shots}-iter={iterations}-elite={elite_frac}-smooth={smoothing}-alpha={alpha}-dt={dt}'

    dyn = Dynamics('xml/double_pendulum.xml', dt = dt)
    step = make_step(dyn)
    print(dyn.state_dim, dyn.control_dim)

    mpc = CEM(
        dyn, 
        make_compute_rate(dyn, alpha, gamma),
        shots, 
        horizon, 
        iterations, 
        elite_frac,
        smoothing,
        rho)
    
    ## initial state
    xt = jnp.zeros(dyn.state_dim)

    X = jnp.zeros((steps + 1, dyn.state_dim))
    X = X.at[0].set(xt)

    hist = jnp.zeros(steps)

    for t in range(steps):
        key, subkey = jax.random.split(key)
        ut, J = mpc(xt, subkey)
        xt = step(xt, ut)
        print(t, xt, ut, J)

        X = X.at[t+1].set(xt)
        hist = hist.at[t].set(J)

    jnp.save(name + '-hist.npy', hist)
    jnp.save(name + '-traj.npy', X)

    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('nats / s')
    T = jnp.arange(0, steps)
    ax.plot(T * dt, hist)
    fig.tight_layout()
    fig.savefig(name + '.png', dpi = 300)
    plt.show()

    dyn.render(X, path = name + '.mp4', skip = 1, distance = 4)