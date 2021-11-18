from bankruns.envs.kydland_prescott import Believer, KydlandPrescott, NonBeliever


def test_reset():
    env = KydlandPrescott()
    obs = env.reset()
    assert "cb" in obs.keys()
    assert len(env.hh) == env.num_hh
    for hh in env.hh:
        assert isinstance(hh, Believer) or isinstance(hh, NonBeliever)


def test_step():
    env = KydlandPrescott()
    env.reset()
    inflation = 0.0
    accouncement = 0.0
    action = {"cb": [accouncement, inflation]}
    obs, rew, done, info = env.step(action)
    assert env.inflation == inflation
    assert env.direction_inflation == 0


if __name__ == "__main__":
    test_step()
