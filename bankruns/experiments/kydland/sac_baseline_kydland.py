import os

import ray
from bankruns.envs.kydland_prescott import KydlandPrescott
from bankruns.utils import log, miscellaneous
from bankruns.utils.callbacks import SimpleCallback
from ray import tune
from ray.rllib.agents.sac import SACTrainer
from ray.tune.registry import register_env


def env_creator(env_config):
    return KydlandPrescott(**env_config)


def main(debug, stop_iters=30000, tf=True):
    train_n_replicates = 1 if debug else 1
    seeds = miscellaneous.get_random_seeds(train_n_replicates)
    exp_name, path = log.log_in_current_day_dir("PG_IPD")

    ray.init(num_cpus=os.cpu_count(), num_gpus=0, local_mode=debug)

    rllib_config, stop_config = get_rllib_config(seeds, debug, stop_iters, tf)

    register_env("kydland", env_creator)

    tune_analysis = tune.run(
        SACTrainer,
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
    ACTION_SPACE = KydlandPrescott().action_space
    OBSERVATION_SPACE = KydlandPrescott().observation_space

    stop_config = {
        "training_iteration": 2 if debug else stop_iters,
    }

    env_config = {
        "num_agents": num_agents,
    }
    policies = {
        "cb": (None, OBSERVATION_SPACE, ACTION_SPACE, {}),
    }

    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        assert agent_id in policies.keys()
        return str(agent_id)

    rllib_config = {
        "env": "kydland",
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
