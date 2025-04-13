
# from etils import epath

# from mujoco_playground._src import mjx_env


# ROOT_BODY = "torso_link"

# GRAVITY_SENSOR = "upvector"
# GLOBAL_LINVEL_SENSOR = "global_linvel"
# GLOBAL_ANGVEL_SENSOR = "global_angvel"
# LOCAL_LINVEL_SENSOR = "local_linvel"
# ACCELEROMETER_SENSOR = "accelerometer"
# GYRO_SENSOR = "gyro"

RESTRICTED_JOINT_RANGE = (
    # Left leg. 6
    (-2.5307, 2.8798),
    (-0.5236, 2.9671),
    (-2.7576, 2.7576),
    (-0.087267, 2.8798),
    (-0.87267, 0.5236),
    (-0.2618, 0.2618),
    # Right leg. 6
    (-2.5307, 2.8798),
    (-2.9671, 0.5236),
    (-2.7576, 2.7576),
    (-0.087267, 2.8798),
    (-0.87267, 0.5236),
    (-0.2618, 0.2618),
    # Waist. 1
    (-2.618, 2.618),
    # Left shoulder. 5
    (-3.0892, 2.6704),
    (-1.5882, 2.2515),
    (-2.618, 2.618),
    (-1.0472, 2.0944),
    (-1.97222, 1.97222),
    # Right shoulder. 5
    (-3.0892, 2.6704),
    (-2.2515, 1.5882),
    (-2.618, 2.618),
    (-1.0472, 2.0944),
    (-1.97222, 1.97222),
)
