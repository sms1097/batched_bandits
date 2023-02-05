import numpy as np
import pandas as pd

from .simulation import Simulation


class BatchNumberExperiment:
    def __init__(self, m_list, T, config):
        self.M_list = m_list
        self.T = T
        self.config = config

    def build_config(self, M):
        """Instantiate TBD variables in config"""
        config = self.config.copy()
        grid_dict = {}
        for grid_name, grid in self.config["grid_dict"].items():
            grid_dict[grid_name] = grid(self.T, M)

        config.pop("grid_dict")
        config["grid_dict"] = grid_dict
        config["M"] = M

        return config

    def save_experiment(self):
        out = pd.concat([x.get_regret() for x in self.sims])
        out.to_csv("batch_number_experiment.csv")

    def run(self):
        self.sims = []
        for M in self.M_list:
            config = self.build_config(M)

            # Nested dicts in kwargs don't seem to agree
            sim = Simulation(
                grid_dict=config["grid_dict"],
                agent_dict=config["agent_dict"],
                agents_kwargs=config["agents_kwargs"],
                arm_dict=config["arm_dict"],
                exp_name=f"M={M}",
                num_sims=config["num_sims"],
            )
            sim.run_sims()
            self.sims.append(sim)
        return self


class ArmNumberExperiment:
    def __init__(self, T, M, arm_means, config):
        self.T = T
        self.config = config
        self.M = M
        self.arm_means = arm_means
        self.sims = []

    def save_experiment(self):
        out = pd.concat([x.get_regret() for x in self.sims])
        out.to_csv("arm_number_experiment.csv")

    def run(self):
        max_mean = np.max(self.arm_means)
        arm_dict = {max_mean: self.config['arm_constructor'](max_mean)}

        # iterate through and add suboptimal arms
        for am in self.arm_means:
            if am == max_mean:
                continue

            arm_dict[am] = self.config['arm_constructor'](am)
            # Nested dicts in kwargs don't seem to agree
            sim = Simulation(
                grid_dict=self.config["grid_dict"],
                agent_dict=self.config["agent_dict"],
                agents_kwargs=self.config["agents_kwargs"],
                arm_dict=arm_dict,
                exp_name=f"K={len(arm_dict.keys())}",
                num_sims=self.config["num_sims"],
            )
            sim.run_sims()
            self.sims.append(sim)
        return self
