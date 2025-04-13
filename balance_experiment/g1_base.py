
from typing import Any, Dict, Optional, Union

from etils import epath
import jax
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env
import g1_constants 

def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.002,
        episode_length=1000,
        action_repeat=1,
        action_scale=0.5,
        history_len=1,
        restricted_joint_range=True,
        soft_joint_pos_limit_factor=0.95,
        reward_config=config_dict.create(
          scales=config_dict.create(
              # Tracking related rewards.
              target_joint=-1,
              joint_deviation=-.1
          ),
        ),  
    )

class G1Env(mjx_env.MjxEnv):
  """Base class for G1 environments."""

  def __init__(
      self,
      config: config_dict.ConfigDict,
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ) -> None:
    super().__init__(config, config_overrides)

    self._mj_model =  mujoco.MjModel.from_xml_path("./g1_description/scene_mjx_alt.xml")

    self._mj_model.opt.timestep = self.sim_dt

    if self._config.restricted_joint_range:
      self._mj_model.jnt_range[1:] = g1_constants.RESTRICTED_JOINT_RANGE
      self._mj_model.actuator_ctrlrange[:] = g1_constants.RESTRICTED_JOINT_RANGE

    self._mj_model.vis.global_.offwidth = 3840
    self._mj_model.vis.global_.offheight = 2160

    self._mjx_model = mjx.put_model(self._mj_model)


  # def get_gravity(self, data: mjx.Data, frame: str) -> jax.Array:
  #   """Return the gravity vector in the world frame."""
  #   return mjx_env.get_sensor_data(
  #       self.mj_model, data, f"{consts.GRAVITY_SENSOR}_{frame}"
  #   )

  # def get_global_linvel(self, data: mjx.Data, frame: str) -> jax.Array:
  #   """Return the linear velocity of the robot in the world frame."""
  #   return mjx_env.get_sensor_data(
  #       self.mj_model, data, f"{consts.GLOBAL_LINVEL_SENSOR}_{frame}"
  #   )

  # def get_global_angvel(self, data: mjx.Data, frame: str) -> jax.Array:
  #   """Return the angular velocity of the robot in the world frame."""
  #   return mjx_env.get_sensor_data(
  #       self.mj_model, data, f"{consts.GLOBAL_ANGVEL_SENSOR}_{frame}"
  #   )

  # def get_local_linvel(self, data: mjx.Data, frame: str) -> jax.Array:
  #   """Return the linear velocity of the robot in the local frame."""
  #   return mjx_env.get_sensor_data(
  #       self.mj_model, data, f"{consts.LOCAL_LINVEL_SENSOR}_{frame}"
  #   )

  # def get_accelerometer(self, data: mjx.Data, frame: str) -> jax.Array:
  #   """Return the accelerometer readings in the local frame."""
  #   return mjx_env.get_sensor_data(
  #       self.mj_model, data, f"{consts.ACCELEROMETER_SENSOR}_{frame}"
  #   )

  # def get_gyro(self, data: mjx.Data, frame: str) -> jax.Array:
  #   """Return the gyroscope readings in the local frame."""
  #   return mjx_env.get_sensor_data(
  #       self.mj_model, data, f"{consts.GYRO_SENSOR}_{frame}"
  #   )

  # Accessors.

  # @property
  # def xml_path(self) -> str:
  #   return self._xml_path

  @property
  def action_size(self) -> int:
    return self._mjx_model.nu

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model
