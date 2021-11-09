import matplotlib.pyplot as plt
import numpy as np
from bankruns.q_diamond import run_diamond

cooperation_values = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
num_agents = 10
max_steps = 200


def run():
    results = {}
    for coop in cooperation_values:
        results[coop] = []
        for i in range(15):
            results[coop].append(run_diamond(num_agents=num_agents, coop=coop, max_steps=max_steps))
        results[coop] = np.array(results[coop])
    return results


def plot(results):
    plt.plot(cooperation_values, [results[coop].mean() for coop in cooperation_values], label="all rounds")
    plt.plot(
        cooperation_values,
        [results[coop][:, : int(max_steps / 1.5), :].mean() for coop in cooperation_values],
        label="after experimentation",
    )
    plt.legend()
    plt.show()


if __name__ == "__main__":
    results = run()
    plot(results)
