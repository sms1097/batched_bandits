from batched_bandits.agents import (
    BatchedBernoulliThompsonAgent,
    BatchedEpsGreedyAgent,
    ABAgent,
)
from batched_bandits.experiment import BatchNumberExperiment, ArmNumberExperiment

from batched_bandits.util import (
    make_arithmetic_grid,
    make_geometric_grid,
    make_minimax_grid,
    plot_regret,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from functools import partial


class Arm:
    def __init__(self, mean):
        self.mean = mean

    def sample(self):
        return np.random.binomial(1, self.mean)


def main():
    NUM_SIMS = 100
    M = 5
    T = 5000
    BATCH_SIZES_TEST = [2, 4, 6, 8]
    STATIC_ARM_DICT = {i: Arm(i) for i in (0.5, 0.51, 0.6)}
    ARM_MEAN_TEST = (0.6, 0.5, 0.51, 0.49, 0.48, 0.52)

    test_batch_config = {
        "arm_dict": STATIC_ARM_DICT,
        "num_sims": NUM_SIMS,
        "agents_kwargs": {
            "TS": {},
            "AB": {},
            "eps_greedy": {"epsilon": 0.9, "eps_decay": 0.9},
        },
        "agent_dict": {
            "TS": partial(BatchedBernoulliThompsonAgent),
            "AB": partial(ABAgent),
            "eps_greedy": partial(BatchedEpsGreedyAgent),
        },
        "grid_dict": {
            "minimax": partial(make_minimax_grid),
            "geometric": partial(make_geometric_grid),
            "arithmetic": partial(make_arithmetic_grid),
        },
    }

    test_arms_config = {
        "num_sims": NUM_SIMS,
        "agents_kwargs": {
            "TS": {},
            "AB": {},
            "eps_greedy": {"epsilon": 0.9, "eps_decay": 0.9},
        },
        "agent_dict": {
            "TS": partial(BatchedBernoulliThompsonAgent),
            "AB": partial(ABAgent),
            "eps_greedy": partial(BatchedEpsGreedyAgent),
        },
        "grid_dict": {
            "minimax": make_minimax_grid(T, M),
            "geometric": make_geometric_grid(T, M),
            "arithmetic": make_arithmetic_grid(T, M),
        },
        "arm_constructor": partial(Arm),
    }

    arm_exp = ArmNumberExperiment(
        T=T, M=M, arm_means=ARM_MEAN_TEST, config=test_arms_config
    )

    batch_exp = BatchNumberExperiment(
        m_list=BATCH_SIZES_TEST, T=T, config=test_batch_config
    )

    batch_exp.run()
    batch_exp.save_experiment()

    arm_exp.run()
    arm_exp.save_experiment()


if __name__ == "__main__":
    main()
