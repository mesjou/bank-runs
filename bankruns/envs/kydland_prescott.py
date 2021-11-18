import logging
from abc import ABC

import gym
import numpy as np
from gym.utils import seeding
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
        self.num_agents = 1.0
        self.agents = ["cb"]
        self.num_hh = num_hh
        self.max_steps = max_steps
        self.step_count_in_current_episode = None
        self.action_space = gym.spaces.Discrete(2)  # todo change to two dimensional
        self.observation_space = gym.spaces.Discrete(3)

        assert 0.0 < natural_unemployment <= 1.0, "Natural unemployment must lie above 0 and max 1.0"
        self.natural_unemployment = natural_unemployment
        self.fraction_believer = fraction_believer
        self.unemployment = natural_unemployment
        self.direction_unemployment: int = 0
        self.inflation = 0.0

        assert isinstance(num_imitation, int)
        assert num_imitation % 2.0 == 0.0, "Num imitation must be a multiple of 2"
        assert num_imitation <= num_hh, "Number of imitators must not exceed number of household"
        self.num_imitation = num_imitation

        self.seed(seed)
        self.metadata = {"name": env_name}

    @override(MultiAgentEnv)
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @override(MultiAgentEnv)
    def reset(self):
        self.step_count_in_current_episode = 0
        self._reset_hh()
        obs = self._observe()
        return {agent: obs for agent in self.agents}

    def _reset_hh(self):
        self.hh = np.random.choice(
            [Believer(), NonBeliever()], self.num_hh, p=[self.fraction_believer, 1 - self.fraction_believer]
        )

    def _hh_imitate(self):
        """Draw number of hh from list, match them, then update their status"""
        hh_meeting = np.random.choice(self.hh, self.num_imitation, replace=False)
        hh1 = hh_meeting[: len(hh_meeting) // 2]
        hh2 = hh_meeting[len(hh_meeting) // 2 :]
        for h1, h2 in zip(hh1, hh2):
            if h1.utility > h2.utility:
                h1 = h2.__class__
            else:
                h2 = h1.__class__

    def _get_fractions(self):
        """Get fraction of belivers and non-belivers.
        Calculate new fraction of believers, direction of change and update fraction of belivers attribute.

        :returns
        fraction of non belivers
        direction of change: 1 if fraction belivers increase, else 0

        """

        fraction_beliver = np.mean([isinstance(hh, Believer) for hh in self.hh])
        direction = 1 if self.fraction_believer < fraction_beliver else 0
        self.fraction_believer = fraction_beliver

        return fraction_beliver, direction

    def _observe(self):
        """Define the observation of each agent.

        CB and hh observe:
        1. Past rate of unemployment: ut−1;
        2. Direction of inflation: ∆yt = 1 if yt−2 < yt−1 0 otherwise
        3. Direction of unemployment: ∆ut = 1 if ut−2 < ut−1 0 otherwise

        cb actions: inflation, announced_inflation
        """

        self._hh_imitate()
        fraction_beliver, direction_believer = self._get_fractions()
        obs = {
            "unemployment": self.natural_unemployment,
            "unemployment_direction": 1.0,
            "inflation_direction": 1.0,
            "fraction_beliver": fraction_beliver,
            "fraction_beliver_direction": direction_believer,
        }
        return [obs]

    def _rewards(self, actions: np.array):
        """Calculate reward of central bank."""
        # todo split into several functions -> too much going on here
        expected_inflation = self.hh_forecast(actions[0])
        unemployment = self.augmented_philips_curve(actions[1], np.mean(expected_inflation))
        self.direction_unemployment = 1 if self.unemployment < unemployment else 0
        for hh, expectation in zip(self.hh, expected_inflation):
            hh.set_utility(expectation, actions[1])
        reward = welfare(unemployment, actions[1])
        self.unemployment = unemployment
        return [reward]

    @override(MultiAgentEnv)
    def step(self, action_dict):
        self.step_count_in_current_episode += 1

        rew = self._rewards(action_dict["cb"])
        obs = self._observe()
        done = self.step_count_in_current_episode >= self.max_steps
        info = {}

        return self._to_rllib_api(obs, rew, done, info)

    def _to_rllib_api(self, observation, rewards, dones, info):
        observations = {agent: observation for agent in self.agents}
        rewards = {agent: rewards[i] for i, agent in enumerate(self.agents)}
        dones = {agent: dones for agent in self.agents + ["__all__"]}
        infos = {agent: info for agent in self.agents}
        return observations, rewards, dones, infos

    def from_rllib_api(self, action_dict: MultiAgentDict):
        actions = np.zeros(self.num_agents, dtype=np.int32)
        for i, agent in enumerate(self.agents):
            if agent in action_dict:
                actions[i] = action_dict[agent]
        return actions

    def hh_forecast(self, announced_inflation):
        expected_inflation = [self.hh.forecast(announced_inflation)]
        return expected_inflation

    def augmented_philips_curve(self, inflation, mean_expectations):
        """The augmented philips curve.

        :return unemployment
        """
        return self.natural_unemployment - inflation + mean_expectations


def welfare(unemployment, inflation):
    """Welfare of the economy observed by the cb."""
    return (np.square(unemployment) + np.square(inflation)) / -2.0


class Believer(ABC):
    def __init__(self):
        self.utility = 0.0

    def forecast(self, announced_inflation, natural_unemployment):
        return announced_inflation

    def adapt_forecast(self, inflation, expected_inflation):
        pass

    def set_utility(self, expectation: float, inflation: float):
        """Utility of the household agents."""
        self.utility = (np.square(inflation - expectation) + np.square(inflation)) / -2.0


class NonBeliever(Believer):
    def __init__(self):
        super().__init__()
        self.forecast_error = 0.0
        self.learning_rate = 0.1
        self.forecast_costs = 3.3

    def adapt_forecast(self, inflation, expected_inflation):
        self.forecast_error += self.learning_rate * (inflation - expected_inflation)

    def forecast(self, announced_inflation, natural_unemployment):
        return announced_inflation + natural_unemployment + self.forecast_error - self.forecast_costs
