import copy

from bankruns.envs.kydland_prescott import Believer, KydlandPrescott, NonBeliever


def test_reset():
    env = KydlandPrescott()
    obs = env.reset()
    assert "cb" in obs.keys()
    assert len(env.hh) == env.num_hh
    for hh in env.hh:
        assert isinstance(hh, Believer) or isinstance(hh, NonBeliever)


def test_inflation_calculation():
    env = KydlandPrescott()
    env.reset()

    inflation = 0.0
    accouncement = 0.0
    action = {"cb": [accouncement, inflation]}
    obs, rew, done, info = env.step(action)
    assert env.inflation == inflation
    assert env.direction_inflation == 0

    action = {"cb": [accouncement, 0.5]}
    obs, rew, done, info = env.step(action)
    assert env.inflation == 0.5
    assert env.direction_inflation == 1
    assert obs["cb"]["direction_inflation"] == 1


def test_imitation():
    env = KydlandPrescott(seed=1)
    env.reset()
    hh_original = copy.deepcopy(env.hh)
    assert (
        sum([h_old.__class__.__name__ == h_new.__class__.__name__ for h_old, h_new in zip(hh_original, env.hh)])
        == env.num_hh
    )

    # after one step one hh should have changed from non beliving to beliver or vice versa
    assert env.hh[34].__class__.__name__ == "NonBeliever"
    env.step({"cb": [0.5, 0.5]})
    assert (
        sum([h_old.__class__.__name__ == h_new.__class__.__name__ for h_old, h_new in zip(hh_original, env.hh)])
        == env.num_hh - env.num_imitation / 2
    )

    # the imitated household agent starts with zero utility
    assert sum(hh.utility == 0 for hh in env.hh) == env.num_imitation / 2

    # household 34 changed from non believer to believer
    assert env.hh[1].utility < env.hh[0].utility  # 1 is non believer 0 is believer
    assert env.hh[34].utility == 0.0
    assert env.hh[34].__class__.__name__ == "Believer"


if __name__ == "__main__":
    test_imitation()
