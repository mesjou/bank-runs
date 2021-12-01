import gym
import numpy as np
from gym import spaces


class RescaleObservation(gym.ObservationWrapper):
    """Rescales the continuous obs space of the environment to [0.0, 1.0].
    Everything outside the specified range [min_obs, max_obs] is clipped to the boundary values.
    Example::
        RescaleObservation(env, min_obs, max_obs).observation_space == Box(0, 1)
        True
    """

    def __init__(self, env, min_obs, max_obs):
        assert isinstance(
            env.observation_space, spaces.Box
        ), f"expected Box action space, got {type(env.observation_space)}"
        assert np.less_equal(min_obs, max_obs).all(), (min_obs, max_obs)
        super().__init__(env)
        self.min_obs = min_obs
        self.max_obs = max_obs
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=env.observation_space.shape, dtype=env.observation_space.dtype,
        )

    def observation(self, observation):
        low = self.observation_space.low
        high = self.observation_space.high
        observation = low + (high - low) * ((observation - self.min_obs) / (self.max_obs - self.min_obs))
        observation = np.clip(observation, low, high)
        return observation
