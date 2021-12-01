import logging
from abc import ABC

import numpy as np
from bankruns.utils.interfaces import MultiplePlayersInfo
from gym.utils import seeding
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils import override
from ray.rllib.utils.typing import MultiAgentDict

logger = logging.getLogger(__name__)


class Diamond(MultiplePlayersInfo, MultiAgentEnv, ABC):
    """Environment that implements Diamond game. Action=1 means early withdraw, action=0 means hold at bank."""

    def __init__(
        self, num_agents, seed=None, env_name="Diamond", max_steps=100, r=1.1, R=2.0,
    ):
        self.num_agents = num_agents
        self.agents = [f"agent-{n}" for n in range(self.num_agents)]
        self.max_steps = max_steps
        self.step_count_in_current_episode = None

        assert 1.0 < r < R, "Incentives in returns must be fulfilled"
        self.r = r
        self.R = R

        self.seed(seed)
        self.metadata = {"name": env_name}

    @override(MultiAgentEnv)
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @override(MultiAgentEnv)
    def reset(self):
        self.step_count_in_current_episode = 0
        obs = np.random.uniform(0, 1, size=(1,))
        return {agent: obs for agent in self.agents}

    def _observe(self, actions: np.array):
        return np.array([actions.mean()])

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
