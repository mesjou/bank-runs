from abc import ABC

from ray.rllib.examples.env.utils.interfaces import InfoAccumulationInterface


class MultiplePlayersInfo(InfoAccumulationInterface, ABC):
    """
    Interface that adds logging capability in a multiple player discrete game.
    Logs the frequency of each state.
    """

    def _init_info(self):
        self.run_count = []
        self.no_run_count = []

    def _reset_info(self):
        self.run_count.clear()
        self.no_run_count.clear()

    def _get_episode_info(self):
        return {
            "run_freq": sum(self.run_count) / len(self.run_count),
            "no_run_freq": sum(self.no_run_count) / len(self.no_run_count),
        }

    def _accumulate_info(self, actions):
        for a in actions:
            self.run_count.append(a == 1.0)
            self.no_run_count.append(a == 0.0)
