import numpy as np
import pandas as pd
import seaborn as sns


def random_argmax(x):
    x = np.array(x)
    return np.argmax(x)


def random_argmax_dict(d):
    keys = list(d.keys())
    values = list(d.values())
    key_idx = random_argmax(values)
    return keys[key_idx]


def average_dictionary_keys(d):
    return {key: np.mean(values) for key, values in d.items()}


def plot_regret(agent_list=None, exp=None, df=None):
    if df is None:
        df = pd.concat([x.get_regret() for x in exp.sims])
        df = df[df.agent_name.isin(agent_list)]
    sns.boxplot(x="exp_name", y="cumulative_regret", hue="agent_name", data=df)


def compare_agents(agent_list, exp_list):
    temp = pd.concat([x.get_regret() for exp in exp_list for x in exp.sims])
    temp = temp[temp.agent_name.isin(agent_list)]
    sns.boxplot(x="exp_name", y="cumulative_regret", hue="agent_name", data=temp)


def grid2steps(grid):
    out = []
    for a, b in zip(grid[1:], grid[:-1]):
        out.append(a - b)
    out.insert(0, grid[0])
    return out


def make_geometric_grid(T, M):
    step_size = T ** (1 / M)
    grid = [int(step_size)]
    for _ in range(M - 1):
        term = int(grid[-1] * step_size)
        grid.append(term)
    grid[-1] = T
    return grid2steps(grid)


def make_minimax_grid(T, M):
    step_size = T ** (1 / (2 - 2 ** (1 - M)))
    grid = [int(step_size)]
    for _ in range(M - 1):
        term = int(step_size * np.sqrt(grid[-1]))
        grid.append(term)

    grid[-1] = T
    return grid2steps(grid)


def make_arithmetic_grid(T, M):
    step_size = T / M
    return [int(step_size) for _ in range(M)]


def instant_grid(T, M):
    return [1 for _ in range(T)]

