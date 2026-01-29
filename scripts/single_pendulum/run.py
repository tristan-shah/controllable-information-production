# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ['MUJOCO_GL'] = 'egl'

import jax
from jax import numpy as jnp
from jax import Array
from einops import einsum
import matplotlib.pyplot as plt

from val import Dynamics, make_step, make_unroll
from val.cem import CEM
# from val.info import make_compute_rate

@jax.jit
def compute_volume(fx: Array, fu: Array, alpha: float, gamma: float):
    
    dx = fx.shape[-1]
    du = fu.shape[-1]

    Q = jnp.eye(dx)
    R = jnp.eye(du) * alpha

    def scan_fn(carry: tuple[Array, Array, Array], inputs: tuple[Array, Array]):
        
        logdet, P_t, Y_t, V_t, W_t = carry
        fx_t, fu_t = inputs

        logdet = logdet + jnp.linalg.slogdet(P_t + fx_t.T @ fx_t).logabsdet
        S_inv = jnp.linalg.inv(R + gamma * fu_t.T @ V_t @ fu_t)
        ## feedback gain
        K_t = - gamma * S_inv @ fu_t.T @ V_t @ fx_t
        ## open loop entropy
        Y_t = Q + gamma * fx_t.T @ Y_t @ fx_t
        P_t = Q - fx_t.T @ jnp.linalg.inv(P_t + fx_t @ fx_t.T) @ fx_t
        ## riccati equation
        V_t = Q + gamma * fx_t.T @ V_t @ fx_t - gamma ** 2 * fx_t.T @ V_t @ fu_t @ S_inv @ fu_t.T @ V_t @ fx_t
        ## closed loop entropy
        W_t = Q + gamma * (fx_t + fu_t @ K_t).T @ W_t @ (fx_t + fu_t @ K_t)
        
        carry = (logdet, P_t, Y_t, V_t, W_t)
        return carry, carry
    
    init_logdet = jnp.linalg.slogdet(Q).logabsdet
    _, (logdet, P, Y, V, W) = jax.lax.scan(scan_fn, init = (init_logdet, Q, Q, Q, Q), xs = (fx, fu), reverse = True)
    return logdet, P, Y, V, W

def make_compute_rate(dyn: Dynamics, alpha: float = 1.0, gamma: float = 1.0):

    ## helper functions
    step = make_step(dyn)
    traj_linerize = jax.jit(jax.vmap(jax.jacfwd(step, argnums = (0, 1))))
    batch_traj_linearize = jax.jit(jax.vmap(traj_linerize))
    unroll = make_unroll(step)
    batch_unroll = jax.jit(jax.vmap(unroll, in_axes = (None, 0)))
    batch_compute_volume = jax.jit(jax.vmap(compute_volume, in_axes = (0, 0, None, None)))

    dt = dyn.mjx_model.opt.timestep

    def compute_rate(xt: Array, U_batch: Array):

        ## unroll trajectories
        X_batch = batch_unroll(xt, U_batch)
        ## linearize the batch of trajectories
        fx_batch, fu_batch = batch_traj_linearize(X_batch[:, :-1, :], U_batch)
        # ## compute entropy of each trajectory (manually setting alpha = 1.0)
        logdet, P, Y, V, W = batch_compute_volume(fx_batch, fu_batch, alpha, gamma)

        horizon = U_batch.shape[1]
        T = jnp.arange(1, horizon + 1)

        ## compute entropy
        ol_entropy = logdet / (2 * jnp.flip(T) * dt)
        cl_entropy = jnp.linalg.slogdet(W).logabsdet / (2 * jnp.flip(T) * dt) ## closed loop
        ## convert entropy into information
        information = ol_entropy - cl_entropy
        
        rate = information

        return rate[:, 0]
    
    return jax.jit(compute_rate)


if __name__ == '__main__':
    seed = 0
    key = jax.random.PRNGKey(seed)

    # dt = 0.01
    dt = 0.05
    # horizon = 512
    horizon = 512
    shots = 512
    # steps = 1200
    steps = 600 

    iterations = 1
    elite_frac = 0.1
    smoothing = 0.1
    alpha = 1.0
    rho = 0.9
    gamma = 1.0

    name = f'woodbury-SINGLE_PENDULUM-h={horizon}-gamma={gamma}-shots={shots}-iter={iterations}-elite={elite_frac}-smooth={smoothing}-alpha={alpha}-dt={dt}'

    ## initialize dynamics
    dyn = Dynamics('xml/pendulum.xml', dt = dt)
    step = make_step(dyn)
    unroll = make_unroll(step)

    ## initialize agent
    mpc = CEM(
        dyn,
        make_compute_rate(dyn, alpha, gamma),
        shots, 
        horizon, 
        iterations, 
        elite_frac,
        smoothing,
        rho)
    
    ## get initial state
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

    dyn.render(X, path = name + '.mp4', skip = 2, distance = 4)