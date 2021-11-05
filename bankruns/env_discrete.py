import logging
import numpy as np
from gym.utils import seeding
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from abc import ABC
from ray.rllib.utils.typing import MultiAgentDict
from ray.rllib.utils import override
import gym


logger = logging.getLogger(__name__)


class DiamondDiscrete(MultiAgentEnv, ABC):
    """Environment that implements Diamond game. Action=1 means early withdraw, action=0 means hold at bank."""
    def __init__(
        self,
        num_agents,
        seed=None,
        env_name="Diamond",
        max_steps=200,
        coop=0.3,
        R=2.0,
    ):
        self.num_agents = num_agents
        self.agents = [f"agent-{n}" for n in range(self.num_agents)]
        self.max_steps = max_steps
        self.step_count_in_current_episode = None
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Discrete(2)

        self.R = R

        assert 0.01 <= coop <= 1.0, "Incentives in returns must be fulfilled"
        self.r = self.coordination_parameter(coop)
        print("r:", self.r, "R:", self.R, "coop:", coop)
        assert 1.0 < self.r <= self.R, "Incentives in returns must be fulfilled"

        self.seed(seed)
        self.metadata = {'name': env_name}

    def coordination_parameter(self, coop):
        return 1.0 / (1.0 - coop * (self.R - 1.0) / self.R)

    @override(MultiAgentEnv)
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @override(MultiAgentEnv)
    def reset(self):
        self.step_count_in_current_episode = 0
        obs = self.observation_space.sample()
        return {agent: obs for agent in self.agents}

    def _observe(self, actions: np.array):
        return np.round(actions.mean(), decimals=0).astype(int)

    def _rewards(self, actions: np.array):
        e = actions.sum()
        if e == self.num_agents:
            rewards = np.full(self.num_agents, fill_value=1.0, dtype=np.float32)
        elif e == 0:
            rewards = np.full(self.num_agents, fill_value=self.R, dtype=np.float32)
        else:
            profit_waiter = np.max([0, (self.num_agents - e * self.r) / (self.num_agents - e) * self.R])
            profit_early_withdrawer = np.min([self.r, self.num_agents / e]) if e > 0.0 else self.r
            rewards = np.where(actions == 1.0, profit_early_withdrawer, profit_waiter)

        return rewards

    @override(MultiAgentEnv)
    def step(self, action_dict):
        self.step_count_in_current_episode += 1

        actions = self.from_rllib_api(action_dict)
        obs = self._observe(actions)
        rew = self._rewards(actions)
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
