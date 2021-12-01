from bankruns.envs.kydland_prescott import KydlandPrescott
from bankruns.wrappers.env_converter import MultiToSingle


def test_wrapping():
    gym_env = MultiToSingle(KydlandPrescott())
    gym_env.reset()
    print(gym_env.step([0.1, 0.2]))


if __name__ == "__main__":
    test_wrapping()
