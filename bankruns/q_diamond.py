import numpy as np
from bankruns.agents.q_learning import QLearner
from bankruns.env_discrete import DiamondDiscrete


def decay_policy(init_value, end_value, step, end_after):
    return max([init_value - (init_value - end_value) / end_after * step, end_value])


def run_diamond(num_agents=5, coop=0.01, max_steps=200, decay_schedule=True):

    # init policies
    env = DiamondDiscrete(num_agents, max_steps=max_steps, coop=coop)
    policies = {f"agent-{n}": QLearner(env.observation_space.n, env.action_space.n) for n in range(num_agents)}

    done = False
    action_history = []
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

        action_history.append([a for agent, a in sorted(actions.items())])
        print("Run or no run: ", np.array(list(actions.values())).mean())

        obs = next_obs

    print("Training finished.\n")
    return np.array(action_history)


if __name__ == "__main__":
    run_diamond(10)
