import os

import gym
import numpy as np
import ray
from bankruns.envs.diamond_continous import Diamond
from bankruns.utils import log, miscellaneous
from bankruns.utils.callbacks import SimpleCallback
from ray import tune
from ray.rllib.agents.pg import PGTrainer
from ray.tune.registry import register_env


def env_creator(env_config):
    return Diamond(**env_config)


def main(debug, stop_iters=30000, tf=True):
    train_n_replicates = 1 if debug else 1
    seeds = miscellaneous.get_random_seeds(train_n_replicates)
    exp_name, path = log.log_in_current_day_dir("PG_IPD")

    ray.init(num_cpus=os.cpu_count(), num_gpus=0, local_mode=debug)

    rllib_config, stop_config = get_rllib_config(seeds, debug, stop_iters, tf)

    register_env("diamond", env_creator)

    tune_analysis = tune.run(
        PGTrainer,
        config=rllib_config,
        stop=stop_config,
        checkpoint_at_end=True,
        name=exp_name,
        log_to_file=True,
        local_dir=path,
    )
    ray.shutdown()
    return tune_analysis


def get_rllib_config(seeds, debug=False, stop_iters=50, tf=True):
    num_agents = 5
    ACTION_SPACE = gym.spaces.Discrete(2)
    OBSERVATION_SPACE = gym.spaces.Box(low=0, high=1, dtype=np.uint8, shape=(1,))

    stop_config = {
        "training_iteration": 2 if debug else stop_iters,
    }

    env_config = {
        "num_agents": num_agents,
    }
    policies = {
        "agent-0": (None, OBSERVATION_SPACE, ACTION_SPACE, {}),
        "agent-1": (None, OBSERVATION_SPACE, ACTION_SPACE, {}),
        "agent-2": (None, OBSERVATION_SPACE, ACTION_SPACE, {}),
        "agent-3": (None, OBSERVATION_SPACE, ACTION_SPACE, {}),
        "agent-4": (None, OBSERVATION_SPACE, ACTION_SPACE, {}),
    }

    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        assert agent_id in policies.keys()
        return str(agent_id)

    rllib_config = {
        "env": "diamond",
        "env_config": env_config,
        "multiagent": {
            "policies": policies,
            # "policies": {
            #    f"agent-{n}": (
            #        None,
            #        OBSERVATION_SPACE,
            #        ACTION_SPACE,
            #        {},
            #    ) for n in range(env_config["num_agents"])
            # },
            "policy_mapping_fn": policy_mapping_fn,
        },
        "seed": tune.grid_search(seeds),
        "callbacks": SimpleCallback,  # log.get_logging_callbacks_class(log_full_epi=True),
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "framework": "tf" if tf else "torch",
    }

    return rllib_config, stop_config


if __name__ == "__main__":
    debug_mode = False
    main(debug_mode)
