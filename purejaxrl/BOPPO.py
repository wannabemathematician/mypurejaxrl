import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax
from wrappers import (
    LogWrapper,
    BraxGymnaxWrapper,
    VecEnv,
    NormalizeVecObservation,
    NormalizeVecReward,
    ClipAction,
)


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, h):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        if h.shape[0] == config['HISTORY_LENGTH']:
            h = h[jnp.newaxis, :, : ]

        # LSTM part
        ScanLSTM = nn.scan(
        nn.LSTMCell, variable_broadcast="params",
        split_rngs={"params": False}, in_axes=1, out_axes=1)

        lstm = ScanLSTM(128)
        input_shape =  h[:, 0].shape
        carry = lstm.initialize_carry(jax.random.key(0), input_shape)

        carry, h = lstm(carry, h)
        h = h[:, -1]    
        # End LSTM part 
        x = h 

        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        critic = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)
 

    
class BBO(nn.Module):
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        x = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        x = activation(x)
        x = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        x = activation(x)
        x = nn.Dense(
            1, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(x)

        return x


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    history: jnp.ndarray
    next_history: jnp.ndarray
    info: jnp.ndarray


def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
       config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    env, env_params = BraxGymnaxWrapper(config["ENV_NAME"]), None
    env = LogWrapper(env)
    env = ClipAction(env)
    env = VecEnv(env)
    if config["NORMALIZE_ENV"]:
        env = NormalizeVecObservation(env)
        env = NormalizeVecReward(env, config["GAMMA"])

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):

        # INIT History buffer 
        state_n = env.observation_space(env_params).shape[0]
        action_n = env.action_space(env_params).shape[0]
        reward_n = 1
        print('infos', state_n, action_n, reward_n)
        history = jnp.zeros((config['NUM_ENVS'], config['HISTORY_LENGTH'], 2 * state_n + action_n + reward_n)) # h <- h \union {(s, a, r, s')}
        init_history = jnp.zeros((config['HISTORY_LENGTH'], 2 * state_n + action_n + reward_n))
        # INIT NETWORK
        bbo_network = BBO(activation=config["ACTIVATION"])
        rng, _rng = jax.random.split(rng)
        init_q = jnp.zeros((1,))
        bbo_network_params = bbo_network.init(_rng, init_q)
        bbo_params_zero = bbo_network_params
        if config["ANNEAL_LR"]:
            bbo_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            bbo_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        bbo_train_state = TrainState.create(
            apply_fn=bbo_network.apply,
            params=bbo_network_params,
            tx=bbo_tx,
        )

        network = ActorCritic(
            env.action_space(env_params).shape[0], activation=config["ACTIVATION"]
        )
        
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((config['NUM_ENVS'], env.observation_space(env_params).shape[0]))
        print('init his shape', init_history.shape)
        network_params = network.init(_rng, init_history)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = env.reset(reset_rng, env_params)

    
        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng, bbo_train_state, bbo_params_zero, prev_history = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                print('prev his shape innit', prev_history.shape)
                pi, value = network.apply(train_state.params, prev_history)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = env.step(
                    rng_step, env_state, action, env_params
                )
                # Adding to the history buffer
                clump = jnp.hstack([last_obs.reshape((config['NUM_ENVS'], -1)), action.reshape((config['NUM_ENVS'], -1)),
                                    reward.reshape((config['NUM_ENVS'], -1)), obsv.reshape((config['NUM_ENVS'], -1))])
                print('clump shape innit', clump.shape)
                history = jnp.zeros_like(prev_history)
                history.at[:, :-1, :].set(prev_history[:, 1:, :])
                history.at[:, -1, :].set(clump)
                # End adding to the history buffer

                # Adding history to the transition 
                transition = Transition(
                    done, action, value, reward, log_prob, last_obs, obsv, prev_history, history,  info
                )


                runner_state = (train_state, env_state, obsv, rng, bbo_train_state, bbo_params_zero, history)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

          

            # CALCULATE ADVANTAGE WE NO LONGER USE THIS
            train_state, env_state, last_obs, rng, bbo_train_state, bbo_params_zero, history = runner_state
            _, last_val = network.apply(train_state.params, history)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # WE NO LONGER USE THE ADVANTAGES AND TARGETS ^

            # UPDATE NETWORK
            def bbo_loss_fn(params, traj_batch, gae, targets):
                def mse(pred, target):
                    return jnp.inner(pred - target, pred - target)
                batch_size = traj_batch.value.shape[0]
                bbo_value = bbo_network.apply(bbo_train_state.params, traj_batch.value.reshape((batch_size, 1)))
                _, value = network.apply(train_state.params, traj_batch.next_history)
                
                r_plus_q = traj_batch.reward + value 

                loss_mse = jnp.mean(mse(bbo_value, r_plus_q)) 

                # l2 regularisation part 
                def param_sub(a, b):
                    return jnp.sum((a - b)**2)
                loss_tree = tree_map(param_sub, bbo_train_state.params, bbo_params_zero)
                l2_loss = sum(jax.tree.leaves(loss_tree))
                # end l2 regularisation part 

                loss1 = loss_mse / (2 * config['VAR'])
                loss2 = l2_loss / (2 * config['VAR_0'])

                return loss1 + loss2, (loss1, loss2)

            def ppo_loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(params, traj_batch.history)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        #value_pred_clipped = traj_batch.value + (
                        #    value - traj_batch.value
                        #).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        #value_losses = jnp.square(value - targets)
                        #value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        #value_loss = (
                        #    0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        #)
                        print('taget batch value', traj_batch.value.shape)
                        target = bbo_network.apply(bbo_train_state.params, traj_batch.value.reshape((-1, 1))).reshape((-1))
                        value_loss = jnp.mean(jnp.square(value - target))
                        print('herrrrreee weeee gooo', value_loss.shape)

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

            def _update_epoch(update_state, unused, loss_fn):
                def _update_minbatch(train_state, batch_info, loss_fn):
                    traj_batch, advantages, targets = batch_info

                    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                print('batch size la', config['MINIBATCH_SIZE'])
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    lambda train_state, batch_info: _update_minbatch(train_state, batch_info, loss_fn), train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss
            
            # BBO update
            update_state = (bbo_train_state, traj_batch, advantages, targets, rng)
            update_bbo_epoch = lambda update_state, unused: _update_epoch(update_state, unused, bbo_loss_fn)
            update_state, loss_info = jax.lax.scan(
                update_bbo_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            bbo_train_state = update_state[0]
            #metric = traj_batch.info
            rng = update_state[-1]
            # End BBO update

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_ppo_epoch = lambda update_state, unused: _update_epoch(update_state, unused, ppo_loss_fn)
            update_state, loss_info = jax.lax.scan(
                update_ppo_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]


            if config.get("DEBUG"):

                def callback(info):
                    return_values = info["returned_episode_returns"][
                        info["returned_episode"]
                    ]
                    timesteps = (
                        info["timestep"][info["returned_episode"]] * config["NUM_ENVS"]
                    )
                    for t in range(len(timesteps)):
                        print(
                            f"global step={timesteps[t]}, episodic return={return_values[t]}"
                        )

                jax.debug.callback(callback, metric)

            runner_state = (train_state, env_state, last_obs, rng, bbo_train_state, bbo_params_zero, history)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng, bbo_train_state, bbo_params_zero, history)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


if __name__ == "__main__":
    config = {
        "LR": 3e-4,
        "NUM_ENVS": 128,
        "NUM_STEPS": 10,
        "TOTAL_TIMESTEPS": 5e7,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 32,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.0,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh",
        "ENV_NAME": "hopper",
        "ANNEAL_LR": False,
        "NORMALIZE_ENV": True,
        "DEBUG": False,
        # BOPPO Params
        'HISTORY_LENGTH': 1000,
        "VAR": 1,
        "VAR_0": 1,
    }
    rng = jax.random.PRNGKey(30)
    train_jit = jax.jit(make_train(config))
    out = train_jit(rng)
