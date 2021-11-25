from bankruns.envs.kydland_prescott import KydlandPrescott
from bankruns.utils.wrappers import MultiToSingle
from stable_baselines3 import SAC

# Instantiate the env and check compatibility
env = MultiToSingle(KydlandPrescott())
policy_kwargs = dict(net_arch=[16, 16])


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
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=1,
        target_entropy="auto",
        sde_sample_freq=-1,
        tensorboard_log="sac/",
    ).learn(total_timesteps=5000)

# todo add observation scaling wrapper
# todo plot values of action? read into algorithm
