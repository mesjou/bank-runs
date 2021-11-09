from collections import Counter

import matplotlib.pyplot as plt
from bankruns.q_diamond import run_diamond

coop = 0.2
num_agents = 10
max_steps = 200


def run():
    results = run_diamond(num_agents=num_agents, coop=coop, max_steps=max_steps)
    for period in range(max_steps):
        fig, axs = plt.subplots(2)

        # upper plot
        fig.suptitle("Vertically stacked subplots")
        counts = Counter(results[period, :])
        if 0 not in counts.keys():
            counts[0] = 0
        elif 1 not in counts.keys():
            counts[1] = 0
        axs[0].bar(counts.keys(), counts.values(), color="lightblue")
        axs[0].set_ylim([0, num_agents])
        axs[0].set_xticks([0, 1])
        axs[0].set_xticklabels(["No Run", "Run"])

        # lower plot
        axs[1].plot(results[: period + 1, :].mean(axis=1) * num_agents, color="lightblue")
        axs[1].set_xlim([0, max_steps])
        axs[1].set_ylim([0, num_agents])
        plt.savefig("02/{:03d}.png".format(period))
        plt.close(fig)


if __name__ == "__main__":
    run()