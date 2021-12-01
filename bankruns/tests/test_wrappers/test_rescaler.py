import numpy as np
from bankruns.envs.kydland_prescott import KydlandPrescott
from bankruns.wrappers.env_converter import MultiToSingle
from bankruns.wrappers.rescaler import RescaleObservation


def test_wrapping():
    gym_env = MultiToSingle(KydlandPrescott())
    rescaled_env = RescaleObservation(
        gym_env, np.array([0.0, 0.0, 0.0, 0.0, 0.0]), np.array([20.0, 1.0, 1.0, 1.0, 1.0])
    )
    print(gym_env.reset())
    print(rescaled_env.reset())
    print(gym_env.step([0.1, 0.2]))


if __name__ == "__main__":
    test_wrapping()
