from bankruns.env_discrete import DiamondDiscrete
import pytest
import numpy as np


def test_env():
    env = DiamondDiscrete(5)
    env.r = 1.1
    env.reset()

    # everybody runs
    obs, rew, done, info = env.step({f"agent-{n}": 1.0 for n in range(5)})
    for agent, r in rew.items():
        assert r == 1.0
        assert obs[agent] == 1.0

    # nobody runs
    obs, rew, done, info = env.step({f"agent-{n}": 0.0 for n in range(5)})
    for agent, r in rew.items():
        assert r == env.R
        assert obs[agent] == 0.0

    # 2 of 5 run
    actions = [1.0, 1.0, 0.0, 0.0, 0.0]
    rewards = [1.1, 1.1, 28/15, 28/15, 28/15]
    obs, rew, done, info = env.step({f"agent-{n}": actions[n] for n in range(5)})
    i = 0
    for agent, r in rew.items():
        assert pytest.approx(r) == rewards[i]
        assert pytest.approx(obs[agent]) == 0.0
        i += 1

    # 4 of 5 run
    actions = [1.0, 1.0, 1.0, 0.0, 1.0]
    rewards = [1.1, 1.1, 1.1, 1.2, 1.1]
    obs, rew, done, info = env.step({f"agent-{n}": actions[n] for n in range(5)})
    i = 0
    for agent, r in rew.items():
        assert pytest.approx(r) == rewards[i]
        assert pytest.approx(obs[agent]) == 1.0
        i += 1


def test_coordination_parameter():
    N = 5
    for coop in np.linspace(0.01, 1.0, 15):
        env = DiamondDiscrete(N, coop=coop)
        e_star = (env.R - env.r) / env.r * (env.R - 1.0) * N
        assert pytest.approx(coop) == 1 - e_star / N
        assert pytest.approx(env.r) == (N - e_star * env.r) / (N - e_star) * env.R


if __name__ == "__main__":
    test_env()
    test_coordination_parameter()
