import numpy as np
import os
from ml_collections import config_dict
os.environ['__NV_PRIME_RENDER_OFFLOAD'] = '1'
os.environ['__GLX_VENDOR_LIBRARY_NAME'] = 'nvidia'
os.environ['MUJOCO_GL'] = 'egl'

# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

from functools import partial
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
import wandb


import jax
import mediapy as media
from randomize import domain_randomize
import balance 
from datetime import datetime

np.set_printoptions(precision=3, suppress=True, linewidth=100)

# Define the reward scale override
reward_overrides = {"reward_config": {"scales": {"height": -1.0}}}

env = balance.G1Env(config_overrides=reward_overrides)
eval_env = balance.G1Env(config_overrides=reward_overrides)
env_cfg = balance.default_config()


env_name = "g1_balance"
now = datetime.now()
timestamp = now.strftime("%Y%m%d-%H%M%S")
exp_name = f"{env_name}-{timestamp}"


ckpt_path = os.path.abspath(os.path.join(".", "checkpoints", exp_name))
os.makedirs(ckpt_path, exist_ok=True)
print(f"Checkpoint path: {ckpt_path}")# media.show_video(frames, fps=1.0 / env.dt)


wandb.init(project="mjxrl", config=env_cfg)
wandb.config.update({
    "env_name": env_name,
})

from ml_collections import config_dict


# --- Original PPO Parameters ---
original_ppo_params = config_dict.create(
    num_timesteps=400_000_000,
    reward_scaling=0.1,
    episode_length=env_cfg.episode_length,
    normalize_observations=True,
    action_repeat=1,
    unroll_length=32,
    num_minibatches=32,
    num_updates_per_batch=5,
    discounting=0.98,
    learning_rate=1e-4,
    entropy_cost=0,
    num_envs=32768,
    batch_size=1024,
    num_evals=16,
    log_training_metrics=True,
    network_factory=config_dict.create(
        policy_hidden_layer_sizes=(512, 256, 64),
        value_hidden_layer_sizes=(256, 256, 256, 256),
        # policy_obs_key="state",
        value_obs_key="privileged_state"
    )
)

# --- Debug PPO Parameters ---
debug_ppo_params = config_dict.create(
    num_timesteps=10_000,
    reward_scaling=0.1,
    episode_length=env_cfg.episode_length,
    normalize_observations=True,
    action_repeat=1,
    unroll_length=16,
    num_minibatches=4,
    num_updates_per_batch=1,
    discounting=0.98,
    learning_rate=1e-4,
    entropy_cost=0,
    num_envs=64,
    batch_size=256, # (64 * 16) / 4
    num_evals=2,
    log_training_metrics=True,
    network_factory=config_dict.create(
        policy_hidden_layer_sizes=(64,),
        value_hidden_layer_sizes=(64, 64),
        # policy_obs_key="state",
        value_obs_key="privileged_state"
    )
)

# Select the parameters to use
ppo_params = debug_ppo_params #original_ppo_params

x_data, y_data, y_dataerr = [], [], []
times = [datetime.now()]


def progress_cli(num_steps, metrics):
  """Prints progress metrics to the console, including all available metrics."""

  wandb.log(metrics, step=num_steps)
  print(".", end="", flush=True)


ppo_training_params = dict(ppo_params)
network_factory = ppo_networks.make_ppo_networks
if "network_factory" in ppo_params:
  del ppo_training_params["network_factory"]
  network_factory = partial(
      ppo_networks.make_ppo_networks,
      **ppo_params.network_factory
  )
from mujoco_playground import wrapper

train_fn = partial(
    ppo.train, **dict(ppo_training_params),
    network_factory=network_factory,
    progress_fn=progress_cli,
    # policy_params_fn=policy_params_fn,
    randomization_fn=domain_randomize,
    wrap_env_fn=wrapper.wrap_for_brax_training,
    save_checkpoint_path=ckpt_path  
)

make_inference_fn, params, metrics = train_fn(
    environment=env,
    eval_env=eval_env
)

eval_env_2 = balance.G1Env(config_overrides=reward_overrides)

jit_reset = jax.jit(eval_env_2.reset)
jit_step = jax.jit(eval_env_2.step)
jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))

rng = jax.random.PRNGKey(42)
rollout = []
n_episodes = 1

for _ in range(n_episodes):
  state = jit_reset(rng)
  rollout.append(state)
  for i in range(env_cfg.episode_length):
    act_rng, rng = jax.random.split(rng)
    ctrl, _ = jit_inference_fn(state.obs, act_rng)
    state = jit_step(state, ctrl)
    rollout.append(state)


frames = eval_env_2.render(rollout, camera="track")
frames_np = np.array(frames)
frames_np_rearranged = np.transpose(frames_np, (0, 3, 1, 2))
wandb.log({"video": wandb.Video(frames_np_rearranged, fps=1.0 / env.dt, format="gif")})