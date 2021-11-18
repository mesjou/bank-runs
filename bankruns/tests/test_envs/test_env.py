import pytest

from bankruns.envs.env import Diamond


def test_env():
    env = Diamond(5)
    env.reset()

    # everybody runs
    obs, rew, done, info = env.step({f"agent-{n}": 1.0 for n in range(5)})
    for agent, r in rew.items():
        assert r == 1.0
        assert obs[agent][0] == 1.0

    # nobody runs
    obs, rew, done, info = env.step({f"agent-{n}": 0.0 for n in range(5)})
    for agent, r in rew.items():
        assert r == env.R
        assert obs[agent][0] == 0.0

    # 2 of 5 run
    actions = [1.0, 1.0, 0.0, 0.0, 0.0]
    rewards = [1.1, 1.1, 28 / 15, 28 / 15, 28 / 15]
    obs, rew, done, info = env.step({f"agent-{n}": actions[n] for n in range(5)})
    i = 0
    for agent, r in rew.items():
        assert pytest.approx(r) == rewards[i]
        assert pytest.approx(obs[agent][0]) == 0.4
        i += 1

    # 4 of 5 run
    actions = [1.0, 1.0, 1.0, 0.0, 1.0]
    rewards = [1.1, 1.1, 1.1, 1.2, 1.1]
    obs, rew, done, info = env.step({f"agent-{n}": actions[n] for n in range(5)})
    i = 0
    for agent, r in rew.items():
        assert pytest.approx(r) == rewards[i]
        assert pytest.approx(obs[agent][0]) == 0.8
        i += 1


if __name__ == "__main__":
    test_env()
