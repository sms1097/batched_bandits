import numpy as np
import pandas as pd
from .util import random_argmax, random_argmax_dict


class BaSEAgent:
    def __init__(self, arm_dict, grid, gamma):
        self.A = arm_dict.copy()
        self.M = len(grid)
        self.K = len(arm_dict.keys())
        self.T = np.sum(grid)
        self.grid = grid
        self.gamma = gamma
        self.reward_dict = {arm: [] for arm in self.A.keys()}
        self.history = []
        self.arm_dict_len = []

    def get_history(self):
        return self.history

    def get_arms(self):
        return list(self.A.keys())

    def get_action(self, greedy=False):
        arms = self.get_arms()

        if not greedy:
            a = np.random.choice(arms)
        else:
            a = random_argmax_dict(self.Y_i_)

        return a

    def update(self):
        # don't process if A is single action
        # roundoff errors can occur with comparison of Y_i
        # and Y_max
        if len(self.A) > 1:
            Y_i = {arm: np.mean(a_r) for arm, a_r in self.reward_dict.items()}
            Y_max = np.max(list(Y_i.values()))

            for arm in self.get_arms():
                if Y_max - Y_i[arm] >= np.sqrt(
                    self.gamma * np.log(self.T * self.K) / len(self.reward_dict[arm])
                ):
                    self.A.pop(arm)

    def simulate(self):
        # explore in earlier batches with se
        for batch_num, batch in enumerate(self.grid[:-1]):
            for i in range(batch):
                a = self.get_action()
                r = self.A[a].sample()

                self.reward_dict[a].append(r)
                self.history.append(r)
            self.update()
            self.arm_dict_len.append(len(self.get_arms()))

        # exploit last batch
        self.Y_i_ = {a: np.mean(self.reward_dict[a]) for a in self.get_arms()}
        for _ in range(self.grid[-1]):
            a = self.get_action(greedy=True)
            r = self.A[a].sample()
            self.reward_dict[a].append(r)
            self.history.append(r)

        return self


class BatchedEpsGreedyAgent:
    def __init__(self, arm_dict, grid, epsilon):
        self.T = np.sum(grid)
        self.grid = grid
        self.A = arm_dict.copy()
        self.reward_dict = {arm: [] for arm in self.A.keys()}
        self.epsilon = epsilon
        self.K = len(self.A.keys())
        self.history = []

    def get_arms(self):
        return list(self.A.keys())

    def take_action(self, curr_reward_dict, curr_history):
        # eps-greedy, but check if there's enough history first
        if (
            np.random.uniform(0, 1) > 1 - self.epsilon
            and len(curr_history) > self.K * 4
        ):
            action = random_argmax_dict(curr_reward_dict)
        else:
            arms = self.get_arms()
            action = np.random.choice(arms)
        return action

    def get_history(self):
        return self.history

    def simulate(self):
        # explore in earlier batches with se
        for batch_num, batch in enumerate(self.grid[:-1]):
            curr_reward_dict = {
                arm: np.mean(values) for arm, values in self.reward_dict.items()
            }
            curr_history = self.history.copy()
            for i in range(batch):
                a = self.take_action(curr_reward_dict, curr_history)
                r = self.A[a].sample()
                self.reward_dict[a].append(r)
                self.history.append(r)

            self.epsilon *= 1 / len(self.grid)

        self.epsilon = 1  # full exploit last batch
        for _ in range(self.grid[-1]):
            a = self.take_action(curr_reward_dict, curr_history)
            r = self.A[a].sample()
            self.reward_dict[a].append(r)
            self.history.append(r)
        return self


class BatchedBernoulliThompsonAgent:
    """
    Some of this code is loosely inspired by the Thompson Sampling Tutorial
    from Stanford. Credit for some of the functions goes to those authors here:
    https://github.com/iosband/ts_tutorial
    """

    def __init__(self, arm_dict, grid, alpha_init=1, beta_init=1):
        self.T = np.sum(grid)
        self.grid = grid
        self.A = arm_dict.copy()
        self.reward_dict = {arm: [] for arm in self.A.keys()}
        self.K = len(self.A.keys())
        self.history = []
        self.alphas = np.ones(self.K) * alpha_init
        self.betas = np.ones(self.K) * beta_init
        self.arm2idx = {arm: i for i, arm in enumerate(self.A.keys())}
        self.idx2arm = {i: arm for arm, i in self.arm2idx.items()}

    def get_arms(self):
        return list(self.A.keys())

    def get_posterior_mean(self, alphas, betas):
        return alphas / (alphas + betas)

    def sample_posterior(self, alphas, betas):
        return np.random.beta(alphas, betas)

    def take_action(self, curr_alphas, curr_betas, greedy=False):
        args = (curr_alphas, curr_betas)
        thetas = (
            self.sample_posterior(*args)
            if not greedy
            else self.get_posterior_mean(*args)
        )
        return self.idx2arm[random_argmax(thetas)]

    def update(self, a, r):
        a_idx = self.arm2idx[a]
        self.reward_dict[a].append(r)
        self.alphas[a_idx] += r
        self.betas[a_idx] += 1 - r

    def get_history(self):
        return self.history

    def simulate(self):
        # explore in earlier batches with se
        for batch_num, batch in enumerate(self.grid[:-1]):
            curr_alphas, curr_betas = self.alphas.copy(), self.betas.copy()
            for i in range(batch):
                a = self.take_action(curr_alphas, curr_betas, greedy=False)
                r = self.A[a].sample()
                self.update(a, r)
                self.history.append(r)

        curr_alphas, curr_betas = self.alphas.copy(), self.betas.copy()
        for _ in range(self.grid[-1]):
            a = self.take_action(curr_alphas, curr_betas, greedy=True)
            r = self.A[a].sample()
            self.update(a, r)
            self.history.append(r)

        return self


class ABAgent:
    def __init__(self, arm_dict, grid):
        self.T = np.sum(grid)
        self.grid = grid
        self.A = arm_dict.copy()
        self.reward_dict = {arm: [] for arm in self.A.keys()}
        self.K = len(self.A.keys())
        self.history = []

    def get_arms(self):
        return list(self.A.keys())

    def get_history(self):
        return self.history

    def simulate(self):
        # explore in earlier batches with se
        for batch_num, batch in enumerate(self.grid[:-1]):
            for i in range(batch):
                a = np.random.choice(self.get_arms())
                r = self.A[a].sample()
                self.reward_dict[a].append(r)
                self.history.append(r)

        value_dict = {
            arm: np.mean(rewards) for arm, rewards in self.reward_dict.items()
        }
        for _ in range(self.grid[-1]):
            a = random_argmax_dict(value_dict)
            r = self.A[a].sample()
            self.reward_dict[a].append(r)
            self.history.append(r)

        return self
