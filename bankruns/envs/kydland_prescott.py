import logging
import random
from abc import ABC
from typing import List

import gym
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils import override
from ray.rllib.utils.typing import MultiAgentDict

logger = logging.getLogger(__name__)


class KydlandPrescott(MultiAgentEnv, ABC):
    """Implements the Kydland and Prescott model as an agent based model with adaptive households."""

    def __init__(
        self,
        num_hh=100,
        fraction_believer=0.5,
        num_imitation=2,
        seed=None,
        env_name="KydlandPrescott",
        max_steps=5000,
        natural_unemployment=5.5,
    ):
        # gym api
        self.action_space = gym.spaces.Box(low=-10.0, high=15.0, shape=[2])
        self.observation_space = gym.spaces.Box(low=float("-inf"), high=float("inf"), shape=[5])

        # hyperparameter
        self.num_agents: int = 1
        self.num_hh: int = num_hh
        self.max_steps: int = max_steps
        assert 0.0 < natural_unemployment <= 100.0, "Natural unemployment must lie above 0% and max 100%"
        self.natural_unemployment: float = natural_unemployment
        assert isinstance(num_imitation, int)
        assert num_imitation % 2.0 == 0.0, "Num imitation must be a multiple of 2"
        assert num_imitation <= num_hh, "Number of imitators must not exceed number of household"
        self.num_imitation: int = num_imitation

        # storage
        self.agents: List = ["cb"]
        self.hh: List = []
        self.step_count_in_current_episode: int = 0
        self.fraction_believer: float = fraction_believer
        self.init_fraction_beliver: float = fraction_believer
        self.direction_believer: int = 0
        self.unemployment: float = natural_unemployment
        self.direction_unemployment: int = 0
        self.inflation: float = 0.0
        self.direction_inflation: int = 0
        self.burn_in: int = 100

        # misc
        if seed is not None:
            self.seed(seed)
        self.metadata = {"name": env_name}

    @staticmethod
    def seed(seed):
        """Sets the numpy and built-in random number generator seed.
        Args:
            seed (int, float): Seed value to use. Must be > 0. Converted to int
                internally if provided value is a float.
        """
        assert isinstance(seed, (int, float))
        seed = int(seed)
        assert seed > 0

        np.random.seed(seed)
        random.seed(seed)

    @override(MultiAgentEnv)
    def reset(self):
        print("Resettttting")
        self.step_count_in_current_episode = 0
        self._reset_hh()
        obs = self._observe()
        return {agent: obs for agent in self.agents}

    def _observe(self):
        """Define the observation of each agent.

        The central bank observes:
        1. Rate of unemployment: ut
        2. Direction of unemployment: ???ut = 1 if ut???1 < ut 0 otherwise
        3. Direction of inflation: ???yt = 1 if yt???1 < yt 0 otherwise
        4. Fraction of believer: ??t
        5. Direction of believer: ?????t = 1 if ??t???1 < ??t 0 otherwise

        """

        obs = {
            "unemployment": self.unemployment,
            "direction_unemployment": self.direction_unemployment,
            "direction_inflation": self.direction_inflation,
            "fraction_beliver": self.fraction_believer,
            "direction_believer": self.direction_believer,
        }
        return obs

    def _rewards(self, unemployment, inflation):
        """Calculate reward of central bank: the welfare of the economy."""
        return (np.square(unemployment) + np.square(inflation)) / -2.0

    @override(MultiAgentEnv)
    def step(self, action_dict):
        self.step_count_in_current_episode += 1

        # cb announce inflation and hh build expectation
        announced_inflation = action_dict["cb"][0]
        expected_inflation = self._hh_forecast(announced_inflation)

        # cb sets inflation rate
        inflation = action_dict["cb"][1]
        self.direction_inflation = 1 if self.inflation < inflation else 0
        self.inflation = inflation

        # hh earn utility and update forecast rule
        for hh, expectation in zip(self.hh, expected_inflation):
            hh.set_utility(expectation, inflation)
            if self.step_count_in_current_episode >= self.burn_in:
                hh.adapt_forecast(inflation, expectation)

        # hh imitate each other
        self._hh_imitate()
        fraction_beliver = np.mean([hh.type() == "Believer" for hh in self.hh])
        self.direction_believer = 1 if self.fraction_believer < fraction_beliver else 0
        self.fraction_believer = fraction_beliver

        # unemployment arises
        unemployment = self._augmented_philips_curve(inflation, np.mean(expected_inflation))
        self.direction_unemployment = 1 if self.unemployment < unemployment else 0
        self.unemployment = unemployment

        # create return
        rew = self._rewards(unemployment, inflation)
        obs = self._observe()
        done = self.step_count_in_current_episode >= self.max_steps
        info = {}

        return self._to_rllib_api(obs, rew, done, info)

    def _to_rllib_api(self, observation, reward, dones, info):
        observations = {agent: observation for agent in self.agents}
        rewards = {agent: reward for agent in self.agents}
        dones = {agent: dones for agent in self.agents + ["__all__"]}
        infos = {agent: info for agent in self.agents}
        return observations, rewards, dones, infos

    def from_rllib_api(self, action_dict: MultiAgentDict):
        actions = np.zeros(self.num_agents, dtype=np.int32)
        for i, agent in enumerate(self.agents):
            if agent in action_dict:
                actions[i] = action_dict[agent]
        return actions

    def _reset_hh(self):
        self.hh = np.random.choice(
            [Believer(), NonBeliever()], self.num_hh, p=[self.init_fraction_beliver, 1 - self.init_fraction_beliver]
        ).tolist()

    def _hh_imitate(self):
        """Draw number of hh from list, match them, then update their status"""
        hh_meeting_idx = np.random.choice(range(self.num_hh), self.num_imitation, replace=False)
        hh1_idx = hh_meeting_idx[: len(hh_meeting_idx) // 2]
        hh2_idx = hh_meeting_idx[len(hh_meeting_idx) // 2 :]
        for h1, h2 in zip(hh1_idx, hh2_idx):
            if self.hh[h1].__class__.__name__ == self.hh[h2].__class__.__name__:
                continue
            elif self.hh[h1].utility > self.hh[h2].utility:
                self.hh[h2] = self.hh[h1].__class__()
            else:
                self.hh[h1] = self.hh[h2].__class__()

    def _hh_forecast(self, announced_inflation):
        expected_inflation = [
            hh.forecast(announced_inflation, self.natural_unemployment, self.fraction_believer) for hh in self.hh
        ]
        return expected_inflation

    def _augmented_philips_curve(self, inflation, mean_expectations):
        """The augmented philips curve.

        :return unemployment
        """
        # return min(max(self.natural_unemployment - inflation + mean_expectations, 0.0), 100.0)
        return self.natural_unemployment - inflation + mean_expectations


class Believer(ABC):
    def __init__(self):
        self.utility = 0.0
        self.forecast_costs = 0.0

    def type(self):
        return self.__class__.__name__

    def forecast(self, announced_inflation, natural_unemployment, fraction_believer):
        return announced_inflation

    def adapt_forecast(self, inflation, expected_inflation):
        pass

    def set_utility(self, expectation: float, inflation: float):
        """Utility of the household agents."""
        self.utility = (np.square(inflation - expectation) + np.square(inflation)) / -2.0 - self.forecast_costs


class NonBeliever(Believer):
    def __init__(self):
        super().__init__()
        self.forecast_error = 0.0
        self.learning_rate = 0.005
        self.forecast_costs = 3.3

    def adapt_forecast(self, inflation, expected_inflation):
        self.forecast_error += self.learning_rate * (inflation - expected_inflation)

    def forecast(self, announced_inflation, natural_unemployment, fraction_believer):
        return (fraction_believer * announced_inflation + natural_unemployment) / (
            1 + fraction_believer
        ) + self.forecast_error
