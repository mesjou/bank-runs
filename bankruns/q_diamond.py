from bankruns.env_discrete import DiamondDiscrete
from bankruns.agents.q_learning import QLearner
import numpy as np


def decay_policy(init_value, end_value, step, end_after):
    return max([init_value - (init_value - end_value) / end_after * step, end_value])


if __name__ == "__main__":

    # init policies
    num_agents = 5
    env = DiamondDiscrete(num_agents)
    policies = {f"agent-{n}": QLearner(env.observation_space.n, env.action_space.n) for n in range(num_agents)}
    decay_schedule = True

    done = False
    obs = env.reset()
    while not done:

        if decay_schedule:
            for agent_id, agent in policies.items():
                agent.set_learning_rate(decay_policy(0.8, 0.2, env.step_count_in_current_episode, env.max_steps))
                agent.set_epsilon(decay_policy(0.8, 0.0, env.step_count_in_current_episode, int(env.max_steps / 1.5)))

        # acting
        actions = {agent_id: agent.act(obs[agent_id]) for agent_id, agent in policies.items()}
        next_obs, rew, done, info = env.step(actions)
        done = done["__all__"]

        # training
        for agent_id, agent in policies.items():
            agent.learn(obs[agent_id], actions[agent_id], rew[agent_id], next_obs[agent_id])

        print("\n Run or no run: ", np.array(list(actions.values())).mean())
        # print("Learning_rate: ", agent.get_learning_rate())
        # print("Epsilon: ", agent.get_epsilon())

        obs = next_obs

    print("Training finished.\n")
