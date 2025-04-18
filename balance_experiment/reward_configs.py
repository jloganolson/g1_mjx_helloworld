"""Configuration file for predefined reward scale sets for the G1 balance task."""

# Define your sets of reward scales here.
# Each dictionary represents one complete set of scales to be used in a single training run.
# Make sure the keys (e.g., 'height', 'velocity', 'torso_upright') match the keys
# expected by the G1Env reward calculation.

REWARD_SCALE_SETS = [
    # Set 1: Example - Focus on height and uprightness
    {
        "height": -0.5,
        "orientation": 0.5,
        "dof_pos_limits": -0.5,
        "pose": -0.1,
        "alive": 0.,
        "termination": -100.0,
        "stay_still": -0.1,

        "torques": 0,
        "action_rate": -0.01,
        "energy": -0.003,
        "dof_acc": -2.5e-7,
    },
    # Set 2: Example - Focus on velocity and low action rate (assuming you have these)
    {
        "height": -1,
        "orientation": 1.,
        "dof_pos_limits": -0.5,
        "pose": -0.1,
        "alive": 0.,
        "termination": 0.0,
        "stay_still": -0,

        "torques": 0,
        "action_rate": 0,
        "energy": -0.003,
        "dof_acc": -2.5e-7,
    },
]

