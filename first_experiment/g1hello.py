from typing import Any, Dict, Optional, Union
import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from mujoco_playground._src import mjx_env
from mujoco_playground._src import reward
from mujoco_playground._src.dm_control_suite import common


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.002,
        episode_length=1000,
        action_repeat=1,
        action_scale=0.5,
        history_len=1,
        restricted_joint_range=False,
        soft_joint_pos_limit_factor=0.95,
        reward_config=config_dict.create(
          scales=config_dict.create(
              # Tracking related rewards.
              target_joint=-1,
              joint_deviation=-.1
          ),
        ), 
    )


class G1_23dof(mjx_env.MjxEnv):
    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(config, config_overrides=config_overrides)
        self._mj_model =  mujoco.MjModel.from_xml_path("./g1_description/scene_mjx_alt.xml")
        self._mj_model.opt.timestep = self.sim_dt
        self._mjx_model = mjx.put_model(self._mj_model)

        self.target_joints = jp.array(self._mj_model.joint(f"left_shoulder_roll_joint").qposadr)
        self.other_joints = jp.array([i for i in range(self._mj_model.nv) if i not in self.target_joints])


        self._post_init()

    def _post_init(self) -> None:
        self._init_q = jp.array(self._mj_model.keyframe("default_pose").qpos)
        self._default_pose = jp.array(self._mj_model.keyframe("default_pose").qpos)


    def reset(self, rng: jax.Array) -> mjx_env.State:
        qpos = self._init_q
        qvel = jp.zeros(self.mjx_model.nv)

        rng, key = jax.random.split(rng, 2)

            
        qpos = qpos * jax.random.uniform(key, (23,), minval=-0.5, maxval=.5)
    


        data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel, ctrl=qpos)

        # metrics = {
        #     "reward/target": jp.zeros(()),
        # }
        metrics = {}
        for k in self._config.reward_config.scales.keys():
            metrics[f"reward/{k}"] = jp.zeros(())
        info = {"rng": rng}

        reward, done = jp.zeros(2)  # pylint: disable=redefined-outer-name

        obs = self._get_obs(data, info)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        data = mjx_env.step(self.mjx_model, state.data, action, self.n_substeps)
        obs = self._get_obs(data, state.info)
        done = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        done = done.astype(float)


        rewards = self._get_reward(data)
        rewards = {
            k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
        }
        reward = sum(rewards.values()) * self.dt

        return mjx_env.State(data, obs, reward, done, state.metrics, state.info)

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        del info  # Unused.
        return jp.concatenate([
            data.qpos,
            data.qvel,
        ])

    def _get_reward(self, data: mjx.Data) -> dict[str, jax.Array]:
        return {
            "target_joint": self.joint_target_l2(data.qpos, self.target_joints, jp.ones_like(self.target_joints)),
            "joint_deviation": self.joint_deviation_l1(data.qpos, self.other_joints),
        }
    
    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self.mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    def wrap_to_pi(self, angle: jax.Array) -> jax.Array:
        return jp.mod(angle + jp.pi, 2 * jp.pi) - jp.pi

    
    def joint_target_l2(self, qpos: jax.Array, joint_indices: jax.Array, target: jax.Array) -> jax.Array:
        error = self.wrap_to_pi(qpos[joint_indices]) - target
        return jp.sum(jp.square(error))
    
    def joint_deviation_l1(self, qpos: jax.Array, joint_indices: jax.Array) -> jax.Array:
        error = self.wrap_to_pi(qpos[joint_indices]) - self._default_pose[joint_indices]
        return jp.sum(jp.abs(error))
