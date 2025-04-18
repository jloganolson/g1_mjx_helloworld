import numpy as np
import os
from ml_collections import config_dict
os.environ['__NV_PRIME_RENDER_OFFLOAD'] = '1'
os.environ['__GLX_VENDOR_LIBRARY_NAME'] = 'nvidia'
os.environ['MUJOCO_GL'] = 'egl'
os.environ['JAX_DEFAULT_MATMUL_PRECISION'] = 'highest'

# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

from functools import partial
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
import wandb


import jax
# import mediapy as media # media import removed as it wasn't used after refactor
from randomize import domain_randomize
import balance # Import balance to access default_config
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from mujoco_playground import wrapper
from ml_collections import config_dict


np.set_printoptions(precision=3, suppress=True, linewidth=100)


def train_run(config=None):
    """Trains the G1 balance task with PPO, configurable via wandb sweep."""
    run = wandb.init(config=config)
    cfg = wandb.config

    # --- Environment Setup ---
    # Start with the default environment config
    env_cfg = balance.default_config()

    # Get the reward scale set directly from the sweep config
    if hasattr(cfg, 'reward_scale_set') and isinstance(cfg.reward_scale_set, dict):
        reward_scales = cfg.reward_scale_set
        # Create the correct overrides: Update ONLY the scales within the default reward_config
        # Make a copy to avoid modifying the original default_config object if it's reused
        merged_reward_config = env_cfg.reward_config.copy_and_resolve_references()
        merged_reward_config.scales.update(reward_scales) # Update the scales part
        reward_overrides = {"reward_config": merged_reward_config}
    else:
        print("Warning: 'reward_scale_set' not found in wandb.config or is not a dict. Using default reward config.")
        reward_overrides = {}

    print(f"Using reward overrides (showing only scales): {reward_overrides.get('reward_config', {}).get('scales', {})}")

    # Initialize environment with potentially updated reward config
    env = balance.G1Env(config_overrides=reward_overrides)
    eval_env = balance.G1Env(config_overrides=reward_overrides)

    env_name = "g1_balance"
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    run_name = wandb.run.name if wandb.run else timestamp
    exp_name = f"{env_name}-{run_name}"

    ckpt_path = os.path.abspath(os.path.join(".", "checkpoints", exp_name))
    os.makedirs(ckpt_path, exist_ok=True)
    print(f"Checkpoint path: {ckpt_path}")

    # --- PPO Parameters ---
    # Use sweep config for hyperparameters if available, otherwise use defaults
    ppo_params = config_dict.create(
        num_timesteps=cfg.get('num_timesteps', 400_000_000), # Reverted to original default
        reward_scaling=cfg.get('reward_scaling', 0.1),
        episode_length=env_cfg.episode_length, # Use episode length from env_cfg
        normalize_observations=True,
        num_resets_per_eval=1,
        action_repeat=1,
        unroll_length=cfg.get('unroll_length', 32), # Reverted to original default
        num_minibatches=cfg.get('num_minibatches', 32), # Reverted to original default
        num_updates_per_batch=cfg.get('num_updates_per_batch', 5), # Reverted to original default
        discounting=cfg.get('discounting', 0.98),
        learning_rate=cfg.get('learning_rate', 1e-4), # Get LR from sweep config
        entropy_cost=cfg.get('entropy_cost', 0),
        num_envs=cfg.get('num_envs', 32768), # Reverted to original default
        batch_size=cfg.get('batch_size', 1024), # Reverted to original default
        num_evals=cfg.get('num_evals', 16), # Reverted to original default
        clipping_epsilon=cfg.get('clipping_epsilon', 0.2),
        log_training_metrics=True,
        network_factory=config_dict.create(
            policy_hidden_layer_sizes=cfg.get('policy_hidden_layer_sizes', (512, 256, 64)), # Reverted to original default
            value_hidden_layer_sizes=cfg.get('value_hidden_layer_sizes', (256, 256, 256, 256)), # Reverted to original default
            value_obs_key="privileged_state"
        )
    )
    # Log the actual PPO params being used
    wandb.config.update(ppo_params, allow_val_change=True)

    # --- Progress Callback ---
    def progress_cli(num_steps, metrics):
        wandb.log(metrics, step=num_steps)
        print(".", end="", flush=True)

    # --- Training Setup ---
    ppo_training_params = dict(ppo_params)
    network_factory_config = ppo_training_params.pop("network_factory")
    network_factory = partial(
        ppo_networks.make_ppo_networks,
        **network_factory_config
    )

    train_fn = partial(
        ppo.train, **ppo_training_params,
        network_factory=network_factory,
        progress_fn=progress_cli,
        randomization_fn=domain_randomize,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        save_checkpoint_path=ckpt_path
    )

    # --- Run Training ---
    print("Starting training...")
    make_inference_fn, params, metrics = train_fn(
        environment=env,
        eval_env=eval_env
    )
    print("\nTraining finished.")

    # --- Evaluation & Logging ---
    print("Starting evaluation...")
    # Use a fresh eval env instance with the same overrides
    eval_env_2 = balance.G1Env(config_overrides=reward_overrides)

    jit_reset = jax.jit(eval_env_2.reset)
    jit_step = jax.jit(eval_env_2.step)
    jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))

    rng = jax.random.PRNGKey(cfg.get('eval_seed', 42))
    rollout_states = [] # Store full State objects for rendering
    n_episodes = 1

    episode_rewards = []
    state = jit_reset(rng)
    rollout_states.append(state) # Store initial state

    current_episode_reward = 0
    for _ in range(env_cfg.episode_length):
        act_rng, rng = jax.random.split(rng)
        ctrl, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_step(state, ctrl)
        rollout_states.append(state)
        current_episode_reward += state.reward

    episode_rewards.append(current_episode_reward)

    avg_eval_reward = np.mean(episode_rewards)


    # Log video of the episode
    try:
        # Pass pipeline states to render function if that's what it expects
        pipeline_states = [s.pipeline_state for s in rollout_states]
        frames = eval_env_2.render(pipeline_states, camera="track")
        frames_np = np.array(frames)
        if frames_np.ndim == 4 and frames_np.shape[-1] == 3:
            frames_np_rearranged = np.transpose(frames_np, (0, 3, 1, 2))
            # Log video without specifying a step
            wandb.log({"evaluation_video": wandb.Video(frames_np_rearranged, fps=1.0 / eval_env_2.dt, format="mp4")})
        else:
            print(f"Warning: Could not log video, unexpected frame format: {frames_np.shape}")
    except Exception as e:
        print(f"Warning: Rendering or video logging failed: {e}")

    print("Evaluation finished.")
    run.finish()

