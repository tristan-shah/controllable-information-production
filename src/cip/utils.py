import jax
from jax import Array
from jax import numpy as jnp
from einops import einsum
from mujoco.mjx import Data

def split_state(xt: Array, nq: int):
    return xt[:nq], xt[nq:]

def get_state(data: Data):
    return jnp.concatenate([data.qpos, data.qvel])
