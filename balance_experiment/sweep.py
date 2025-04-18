import wandb
import os
import sys
from test_sweep import train_run # Import the training function
# Import the reward scale sets to embed them in the config
from reward_configs import REWARD_SCALE_SETS

import numpy as np
import os
from dotenv import load_dotenv
load_dotenv()
# from ml_collections import config_dict
os.environ['__NV_PRIME_RENDER_OFFLOAD'] = '1'
os.environ['__GLX_VENDOR_LIBRARY_NAME'] = 'nvidia'
os.environ['MUJOCO_GL'] = 'egl'
os.environ['JAX_DEFAULT_MATMUL_PRECISION'] = 'highest'

# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags
np.set_printoptions(precision=3, suppress=True, linewidth=100)
# os.environ['__NV_PRIME_RENDER_OFFLOAD'] = '1'
# os.environ['__GLX_VENDOR_LIBRARY_NAME'] = 'nvidia'
# os.environ['MUJOCO_GL'] = 'egl'
# os.environ['JAX_DEFAULT_MATMUL_PRECISION'] = 'highest'

# # Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
# xla_flags = os.environ.get('XLA_FLAGS', '')
# xla_flags += ' --xla_gpu_triton_gemm_annvidiy=True'
# os.environ['XLA_FLAGS'] = xla_flags


# Make sure the script can find the 'test' module if run from root
sys.path.append(os.path.dirname(__file__))


# 1. Login to W&B (optional, can also be done via environment variables or cli)
# wandb.login()

# 2. Define the sweep configuration
sweep_configuration = {
    'method': 'grid',  # 'grid' will try each reward set exactly once
    'metric': {
        'name': 'final_eval_reward', # The metric to optimize (logged in train_run)
        'goal': 'maximize'          # Target direction ('minimize' or 'maximize')
    },
    'parameters': {
        # Define a parameter where each value is a complete reward scale dictionary
        'reward_scale_set': {
            'values': REWARD_SCALE_SETS # Embed the list of dictionaries here
        },

        # You can still sweep over other hyperparameters independently.
        # If you do, W&B will create runs for all combinations.
        # Example: Sweep over learning rates for each reward set
        'learning_rate': {
            'values': [1e-3, 1e-4]
        },

        # Note: If REWARD_SCALE_SETS becomes very large, embedding it directly
        # might make the sweep configuration less readable. The index method
        # might be preferred in such cases.
    }
}

# Calculate the number of runs based on the configuration
# For a grid search, this is the number of reward sets if it's the only parameter
num_runs = len(REWARD_SCALE_SETS)*2
# If you add other grid parameters like 'learning_rate', multiply the lengths:
# num_runs = len(REWARD_SCALE_SETS) * len(sweep_configuration['parameters']['learning_rate']['values'])

# 3. Initialize the sweep
sweep_id = wandb.sweep(
    sweep=sweep_configuration,
    project="mjxrl" # Specify your W&B project name
)

print(f"Sweep initialized with {num_runs} configurations. Sweep ID: {sweep_id}")
print("Each run will use one of the specified reward scale sets.")
print("Run the following command in your terminal to start the agent:")
print(f"wandb agent {sweep_id}")

# 4. (Optional) Start the agent programmatically
# agent_count = num_runs # Run the agent for each defined configuration
# print(f"Starting agent programmatically to run {agent_count} configurations...")
wandb.agent(sweep_id, function=train_run, count=num_runs) 