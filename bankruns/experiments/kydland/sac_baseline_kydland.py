import os

import ray
from bankruns.envs.kydland_prescott import KydlandPrescott
from bankruns.utils import log, miscellaneous
from bankruns.utils.callbacks import SimpleCallback
from bankruns.utils.wrappers import MultiToSingle
from ray import tune
from ray.rllib.agents.sac import SACTrainer
from ray.tune.registry import register_env


def env_creator(env_config):
    return MultiToSingle(KydlandPrescott(**env_config))


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
    stop_config = {
        "training_iteration": 2 if debug else stop_iters,
    }

    env_config = {}
    rllib_config = {
        "env": "kydland",
        "env_config": env_config,
        "Q_model": {
            "fcnet_hiddens": [8, 8],
            "fcnet_activation": "relu",
            "post_fcnet_hiddens": [],
            "post_fcnet_activation": None,
            "custom_model": None,  # Use this to define custom Q-model(s).
            "custom_model_config": {},
        },
        # Model options for the policy function (see `Q_model` above for details).
        # The difference to `Q_model` above is that no action concat'ing is
        # performed before the post_fcnet stack.
        "policy_model": {
            "fcnet_hiddens": [8, 8],
            "fcnet_activation": "relu",
            "post_fcnet_hiddens": [],
            "post_fcnet_activation": None,
            "custom_model": None,  # Use this to define a custom policy model.
            "custom_model_config": {},
        },
        "seed": tune.grid_search(seeds),
        "callbacks": SimpleCallback,  # log.get_logging_callbacks_class(log_full_epi=True),
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "framework": "tf" if tf else "torch",
        "soft_horizon": False,
        "n_step": 3,
        "prioritized_replay": True,
        "initial_alpha": 0.2,
        "learning_starts": 256,
        "clip_actions": False,
        "timesteps_per_iteration": 0,
        "optimization": {"actor_learning_rate": 0.005, "critic_learning_rate": 0.005, "entropy_learning_rate": 0.0001},
    }

    return rllib_config, stop_config


if __name__ == "__main__":
    debug_mode = False
    main(debug_mode)
