import numpy as np
import pandas as pd


class Simulation:
    """Holds info to simulate experiment"""

    def __init__(
        self, grid_dict, agent_dict, agents_kwargs, arm_dict, exp_name, num_sims=500
    ):
        self.grid_dict = grid_dict
        self.agent_dict = agent_dict
        self.agents_kwargs = agents_kwargs
        self.arm_dict = arm_dict
        self.exp_name = exp_name
        self.histories = {
            f"{agent_name} ({grid_name} grid)": []
            for agent_name in agent_dict.keys()
            for grid_name in grid_dict.keys()
        }
        self.max_reward_history = []
        self.num_sims = num_sims

    def get_max_reward(self, arm_dict):
        values = list([arm.mean for arm in arm_dict.values()])
        return np.max(values)

    def run_sims(self):
        for _ in range(self.num_sims):
            agent_instance_dict = {
                f"{agent_name} ({grid_name} grid)": agent_constructor(
                    arm_dict=self.arm_dict,
                    grid=grid,
                    **self.agents_kwargs[agent_name],
                )
                for agent_name, agent_constructor in self.agent_dict.items()
                for grid_name, grid in self.grid_dict.items()
            }

            for agent_name, agent in agent_instance_dict.items():
                agent.simulate()
                self.histories[agent_name].append(agent.get_history())

            self.max_reward_history.append(self.get_max_reward(self.arm_dict))

    def get_regret(self):
        regret = pd.DataFrame(
            {
                agent_name: self.max_reward_history
                - np.array(agent_history).mean(axis=1)
                for agent_name, agent_history in self.histories.items()
            }
        )
        regret = pd.melt(regret)
        regret.columns = ["agent_name", "cumulative_regret"]
        regret["exp_name"] = self.exp_name

        return regret

    def get_reward(self):
        cum_reward = pd.DataFrame(
            {
                agent_name: np.array(agent_history).mean(axis=1)
                for agent_name, agent_history in self.histories.items()
            }
        )
        cum_reward = pd.melt(cum_reward)
        cum_reward.columns = ["agent_name", "cumulative_reward"]
        cum_reward["exp_name"] = self.exp_name

        return cum_reward
