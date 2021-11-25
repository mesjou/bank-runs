from gym import Env
from ray.rllib.env.multi_agent_env import MultiAgentEnv


class MultiToSingle(Env):
    """Converts an MultiAgentEnv into a Gym style env with num_agents = 1.
    Consequently observations, actions and rewards are singleton tuples.
    The observation action spaces are singleton Tuple spaces.
    The info dict is nested inside an outer with key 0."""

    def __init__(self, env: MultiAgentEnv):
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.wrapped_env = env

    def step(self, action):
        print(action)
        agent_name = self.wrapped_env.agents[0]
        action_dict = {agent_name: action}
        observations, rewards, done, infos = self.wrapped_env.step(action_dict)
        rewards = rewards[agent_name]
        observations = list(observations[agent_name].values())
        done = done[agent_name]
        infos = infos[agent_name]
        print(observations, rewards)
        return observations, rewards, done, infos

    def reset(self):
        agent_name = self.wrapped_env.agents[0]
        observations = self.wrapped_env.reset()
        return list(observations[agent_name].values())

    def seed(self, seed=None):
        if seed is not None:
            self.wrapped_env.seed(seed)

    @property
    def unwrapped(self):
        return self.wrapped_env

    def render(self, mode="human"):
        pass
