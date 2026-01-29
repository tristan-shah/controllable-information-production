import jax
from jax import Array
from jax import numpy as jnp

from cip import Dynamics, make_step, make_unroll

@jax.jit
def compute_volume(fx: Array, fu: Array, alpha: float, gamma: float):
    
    dx = fx.shape[-1]
    du = fu.shape[-1]

    Q = jnp.eye(dx) * 1.0
    R = jnp.eye(du) * alpha

    def scan_fn(carry: tuple[Array, Array, Array], inputs: tuple[Array, Array]):
        
        Y_t, V_t, W_t = carry
        fx_t, fu_t = inputs

        S_inv = jnp.linalg.inv(R + gamma * fu_t.T @ V_t @ fu_t)
        ## feedback gain
        K_t = - gamma * S_inv @ fu_t.T @ V_t @ fx_t
        ## open loop entropy
        Y_t = Q + gamma * fx_t.T @ Y_t @ fx_t
        ## riccati equation
        V_t = Q + gamma * fx_t.T @ V_t @ fx_t - gamma ** 2 * fx_t.T @ V_t @ fu_t @ S_inv @ fu_t.T @ V_t @ fx_t
        ## closed loop entropy
        W_t = Q + gamma * (fx_t + fu_t @ K_t).T @ W_t @ (fx_t + fu_t @ K_t)

        carry = (Y_t, V_t, W_t)

        return carry, (Y_t, V_t, W_t, K_t)
    
    _, (Y, V, W, K) = jax.lax.scan(scan_fn, init = (Q, Q, Q), xs = (fx, fu), reverse = True)
    return Y, V, W, K

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
        Y, _, W, _ = batch_compute_volume(fx_batch, fu_batch, alpha, gamma)

        horizon = U_batch.shape[1]
        T = jnp.arange(1, horizon + 1)

        ## compute entropy
        ol_entropy = jnp.linalg.slogdet(Y).logabsdet / (2 * jnp.flip(T) * dt) ## open loop
        cl_entropy = jnp.linalg.slogdet(W).logabsdet / (2 * jnp.flip(T) * dt) ## closed loop
        ## convert entropy into information
        information = ol_entropy - cl_entropy
        
        rate = information

        return rate[:, 0]
    
    return jax.jit(compute_rate)