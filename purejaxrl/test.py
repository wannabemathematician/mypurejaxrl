import jax
from jax.tree_util import tree_map
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import functools
from typing import Sequence, NamedTuple, Any, Dict
import distrax
from wrappers import (
    LogWrapper,
    BraxGymnaxWrapper,
    VecEnv,
    NormalizeVecObservation,
    NormalizeVecReward,
    ClipAction,
)

tree1 = {'params':{'p1': jnp.array([1., 2.]), 'p2': jnp.array([3., 4.])}}
tree2 = {'params':{'p1': jnp.array([1., 1.]), 'p2': jnp.array([1., 1.])}}

def tree_sub(a, b):
  return jnp.sum((a - b)**2)

tree3 = tree_map(tree_sub, tree1, tree2)
print(tree3)
print(sum(jax.tree.leaves(tree3)))




class LSTM(nn.Module):

  @nn.compact
  def __call__(self, x):
    ScanLSTM = nn.scan(
      nn.LSTMCell, variable_broadcast="params",
      split_rngs={"params": False}, in_axes=1, out_axes=1)

    lstm = ScanLSTM(128)
    input_shape =  x[:, 0].shape
    carry = lstm.initialize_carry(jax.random.key(0), input_shape)
    carry, x = lstm(carry, x)
    x = x[:, -1]
    x = nn.Dense(
            1, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(x)
    x = nn.relu(x)
    return x

'''x = jnp.ones((128, 20, 3))
module = LSTM(features=5)
y, variables = module.init_with_output(jax.random.key(0), x)
print(y)



# In this example we are going to loop over 3 history states in our lstm 
def clump_stuff(state, action, reward, next_state):
    return jnp.hstack([state.reshape((1, -1)), action.reshape((1, -1)),
                                    reward.reshape((1, -1)), next_state.reshape((1, -1))])
state1 = jnp.array([[1.0]])
action1 = jnp.array([[0.2]])
reward1 = jnp.array([[0.0]])

state2 = jnp.array([[2.0]])
action2 = jnp.array([[0.1]])
reward2 = jnp.array([[0.0]])

state3 = jnp.array([[3.0]])
action3 = jnp.array([[0.4]])
reward3 = jnp.array([[1.0]])

state4 = jnp.array([[4.0]])

h1 = clump_stuff(state1, action1, reward1, state2)
h2 = clump_stuff(state2, action2, reward2, state3)
h3 = clump_stuff(state3, action3, reward3, state4) 

history = jnp.vstack([h1, h2, h3]).reshape(1, 3, -1)
print('Generated history:\n', history)

rng = jax.random.key(0)
network = LSTM()
rng, _rng = jax.random.split(rng)
init_x = jnp.zeros((1, 1, 3, 4))
network_params = network.init(_rng, init_x)

tx = optax.chain(
    optax.clip_by_global_norm(1),
    optax.adam(1e-3, eps=1e-5),
)
train_state = TrainState.create(
    apply_fn=network.apply,
    params=network_params,
    tx=tx,
)

print(network.apply(network_params, history).shape)'''
