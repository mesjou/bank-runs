import matplotlib.pyplot as plt
import numpy as np
from bankruns.q_diamond import run_diamond

cooperation_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
num_agents = 10
max_steps = 200


def run():
    results = {}
    for coop in cooperation_values:
        results[coop] = []
        for i in range(20):
            results[coop].append(run_diamond(num_agents=num_agents, coop=coop, max_steps=max_steps))
        results[coop] = np.array(results[coop])
    return results


def get_human(path):
    """Experimental data shows number of wait, 10 - data tells number of run participants"""
    data = np.genfromtxt(path, skip_header=1, dtype=np.float, delimiter=",")
    data = 10 - data
    return data


def plot(results, human):
    # plt.plot(cooperation_values, [results[coop].mean() * 10 for coop in cooperation_values], label="all rounds")
    mean_human = np.nanmean(human, axis=0)
    sd_human = np.nanstd(human, axis=0)
    mean_rl = [results[coop][:, int(max_steps / 1.5) :, :].mean() * 10 for coop in cooperation_values]
    sd_rl = [(results[coop][:, int(max_steps / 1.5) :, :] * 10).std() for coop in cooperation_values]
    # mean_rl = [results[coop].mean() * 10 for coop in cooperation_values]
    # sd_rl = [(results[coop] * 10).std() for coop in cooperation_values]

    fig, ax = plt.subplots()
    plt.gcf().set_size_inches(w=5.5, h=3.0, forward=True)
    ax.plot(cooperation_values, mean_human, linestyle="solid", color="#3b87be", linewidth=1.0, label="Menschen")
    ax.fill_between(cooperation_values, mean_human - sd_human, mean_human + sd_human, color="#d4e6f1", alpha=0.4)
    ax.plot(cooperation_values, mean_rl, linestyle="solid", color="#930c26", linewidth=1.0, label="KI")
    ax.fill_between(
        cooperation_values,
        np.array(mean_rl) - np.array(sd_rl),
        np.array(mean_rl) + np.array(sd_rl),
        color="#f7b596",
        alpha=0.4,
    )
    plt.ylabel(r"Anzahl Paniker")
    ax.set_ylim([0, 10])
    ax.set_xlabel("Kurzfristige Auszahlung")
    ax.set_xticklabels([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.legend()
    plt.savefig("comparison_human_ai.pdf", bbox_inches="tight")
    plt.close("all")

    fig, ax = plt.subplots()
    plt.gcf().set_size_inches(w=5.5, h=3.0, forward=True)
    ax.plot(cooperation_values, mean_human, linestyle="solid", color="#3b87be", linewidth=1.0, label="Menschen")
    ax.fill_between(cooperation_values, mean_human - sd_human, mean_human + sd_human, color="#d4e6f1", alpha=0.4)
    plt.ylabel(r"Anzahl Paniker")
    ax.set_ylim([0, 10])
    ax.set_xlabel("Kurzfristige Auszahlung")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.legend()
    plt.savefig("human.pdf", bbox_inches="tight")
    plt.close("all")


if __name__ == "__main__":
    human = get_human("arifovic_data.csv")
    results = run()
    plot(results, human)
