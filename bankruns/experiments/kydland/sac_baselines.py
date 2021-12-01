import numpy as np
from bankruns.envs.kydland_prescott import KydlandPrescott
from bankruns.wrappers.env_converter import MultiToSingle
from bankruns.wrappers.rescaler import RescaleObservation
from gym.wrappers.rescale_action import RescaleAction
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env

# Instantiate the env and check compatibility
env = MultiToSingle(KydlandPrescott(max_steps=500, fraction_believer=0.02))
check_env(env)
policy_kwargs = dict(net_arch=[16, 16])


# env = NormalizeObservation(NormalizeReward(RescaleAction(env, -1.0, 1.0)))
observation_space = np.array([0.0, 0.0, 0.0, 0.0, 0.0]), np.array([20.0, 1.0, 1.0, 1.0, 1.0])
env = RescaleObservation(RescaleAction(env, -1.0, 1.0), observation_space[0], observation_space[1])


if __name__ == "__main__":
    model = SAC(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=3e-3,
        buffer_size=100000,  # 1e6
        learning_starts=100,
        batch_size=256,
        tau=0.005,
        gamma=0.95,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=1,
        target_entropy="auto",
        sde_sample_freq=-1,
        tensorboard_log="sac/",
    ).learn(total_timesteps=50000)

# todo plot values of action? read into algorithm
# low learning rate of agents: nash outcome
# fast learning rate of agetns: for now system explodes, but maybe ramsey possible if cb less variance?
# # or other explanation?
