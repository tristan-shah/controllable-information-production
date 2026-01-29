import jax
from jax import Array
from jax import numpy as jnp

from val import Dynamics

def ar1_noise(key, shots: int, horizon: int, control_dim: int, rho: float = 0.9):
    '''
    Returns noise of shape (shots, horizon, control_dim)
    with AR(1) time correlation.
    '''
    key, subkey = jax.random.split(key)
    eps = jax.random.normal(subkey, (horizon, shots, control_dim))

    alpha = jnp.sqrt(1.0 - rho ** 2)

    def step(prev, curr):
        out = rho * prev + alpha * curr
        return out, out

    init = eps[0]                      # (shots, control_dim)
    _, ys = jax.lax.scan(step, init, eps[1:])

    noise = jnp.concatenate(
        [init[None, ...], ys], axis=0
    )                                   # (horizon, shots, control_dim)

    return jnp.transpose(noise, (1, 0, 2))


class CEM:
    def __init__(
            self, 
            dyn: Dynamics, 
            objective: callable,
            shots: int, 
            horizon: int, 
            iterations: int, 
            elite_frac: float,
            # keep_frac: float,
            smoothing: float,
            rho: float = 0.9):
        
        assert 0.0 < elite_frac <= 1.0

        ## actuator ranges
        self.low = dyn.mjx_model.actuator_ctrlrange[:, 0]
        self.high = dyn.mjx_model.actuator_ctrlrange[:, 1]

        ## objective function
        self.objective = objective

        ## planning parameters
        self.shots = shots
        self.horizon = horizon
        self.control_dim = dyn.control_dim
        self.iterations = iterations
        self.n_elite = max(1, int(elite_frac * shots))
        # self.n_keep = max(1, int(keep_frac * self.n_elite))
        self.smoothing = smoothing
        self.rho = rho

        ## set a minimum exploration amount
        self.min_std = 0.05 * (self.high - self.low) ## default

        ## initial mean and std
        self.mean = jnp.zeros((self.horizon, self.control_dim))
        self.std = jnp.ones((self.horizon, self.control_dim))

        self.elites = None

    def roll(self, key):

        ## roll backward
        self.mean = jnp.roll(self.mean, shift = -1, axis = 0)
        self.std = jnp.roll(self.std, shift = -1, axis = 0)
        self.elites = jnp.roll(self.elites, shift = -1, axis = 1)

        ## set last element
        self.mean = self.mean.at[-1].set(self.mean[-2])
        self.std = self.std.at[-1].set(self.std[-2])

        if self.elites is not None:
            ## add a random last action to the shifted elites
            key, subkey = jax.random.split(key)
            random_last = jax.random.uniform(subkey, (self.elites.shape[0], self.control_dim), minval = self.low, maxval = self.high)
            self.elites = self.elites.at[:, -1, :].set(random_last)
        return None

    def __call__(self, xt: Array, key):

        for i in range(self.iterations):

            key, subkey = jax.random.split(key)
            ## generate correlated noise
            noise = ar1_noise(subkey, self.shots, self.horizon, self.control_dim, self.rho)
            ## generate a batch of random control signals
            U_batch = self.mean[None, :, :] + self.std[None, :, :] * noise
            U_batch = U_batch.clip(self.low[None, None, :], self.high[None, None, :])

            ## evaluate control signals in parallel
            J = self.objective(xt, U_batch)

            ## select top performing control sequences
            elite_idx = jnp.argsort(J, descending = True)[:self.n_elite]
            U_elite = U_batch[elite_idx]
            ## store elites from previous iteration
            self.elites = U_elite

            ## fit gaussian
            new_mean = jnp.mean(U_elite, axis = 0)
            new_std = jnp.std(U_elite, axis = 0)

            ## smooth update
            self.mean = (1 - self.smoothing) * new_mean + self.smoothing * self.mean
            self.std  = (1 - self.smoothing) * new_std  + self.smoothing * self.std
            ## maintain minimum exploration
            self.std = self.std.clip(min = self.min_std)

        ## return best action
        best_idx = jnp.argmax(J)
        U_best = U_batch[best_idx]
        ut = U_best[0]

        ## rolls over time dependent sequences one step
        self.roll(key)

        return ut, J[elite_idx].mean()
