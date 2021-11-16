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
        counts = Counter(results[period, :])
        if 0 not in counts.keys():
            counts[0] = 0
        elif 1 not in counts.keys():
            counts[1] = 0
        axs[0].bar(counts.keys(), counts.values(), color="#f7b596")
        axs[0].set_ylim([0, num_agents])
        axs[0].set_xticks([0, 1])
        axs[0].set_yticks([])
        axs[0].set_xticklabels(["Keine Panik", "Panik"])
        axs[0].spines["top"].set_visible(False)
        axs[0].spines["right"].set_visible(False)
        axs[0].spines["left"].set_visible(False)
        axs[0].spines["bottom"].set_visible(False)

        # lower plot
        axs[1].plot(results[: period + 1, :].mean(axis=1) * num_agents, color="#f7b596", linewidth=2.00)
        axs[1].set_xlim([0, max_steps])
        axs[1].set_ylim([0, num_agents])
        axs[1].spines["top"].set_visible(False)
        axs[1].spines["right"].set_visible(False)
        axs[1].set_xlabel("Runde")
        axs[1].set_ylabel("Anzahl Paniker")
        plt.savefig("02/{:03d}.png".format(period))
        plt.close(fig)


if __name__ == "__main__":
    run()
