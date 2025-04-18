import wandb
import os
import sys
from test_sweep import train_run 
from reward_configs import REWARD_SCALE_SETS

import numpy as np
import os
from dotenv import load_dotenv
load_dotenv()

os.environ['__NV_PRIME_RENDER_OFFLOAD'] = '1'
os.environ['__GLX_VENDOR_LIBRARY_NAME'] = 'nvidia'
os.environ['MUJOCO_GL'] = 'egl'
os.environ['JAX_DEFAULT_MATMUL_PRECISION'] = 'highest'

# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags
np.set_printoptions(precision=3, suppress=True, linewidth=100)
sys.path.append(os.path.dirname(__file__))


sweep_configuration = {
    'method': 'grid',  # 'grid' will try each reward set exactly once
    'parameters': {
        'pose': {
            'values': [-0.1,-0.5,-1.0]
        },
    }
}

# Calculate the number of runs based on the configuration
# For a grid search, this is the number of reward sets if it's the only parameter
num_runs = len(REWARD_SCALE_SETS)*2

sweep_id = wandb.sweep(
    sweep=sweep_configuration,
    project="g1_balance" # Specify your W&B project name
)


wandb.agent(sweep_id, function=train_run) 