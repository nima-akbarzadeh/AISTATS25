### Problem Parameters
import numpy as np


# Define the reward values for each arm
def rewards(time_horizon, num_arms, num_states, function_type, num_actions=1):
    """
    Calculate rewards for each arm.
    """
    if num_actions == 1:
        vals = np.ones((num_states, num_arms))
        for a in range(num_arms):
            if function_type[a] > 0:
                vals[:, a] = (np.linspace(0, num_states-1, num=num_states)) ** function_type[a] / (num_states-1) ** function_type[a]
    else:
        vals = np.ones((num_states, num_actions, num_arms))
        for a in range(num_arms):
            for act in range(num_actions):
                if function_type[a] > 0:
                    vals[:, act, a] = (np.linspace(0, num_states-1, num=num_states)) ** function_type[a] / (num_states-1) ** function_type[a]
    vals = np.round(vals / time_horizon, 3)
    return vals


# Define the reward rewards for each arm
def rewards_utility(time_horizon, num_arms, num_states, function_type, thresholds, u_type, u_order, num_actions=1):
    """
    Calculate a utility function of the rewards for each arm.
    """

    def compute_risk(total_rewards, arm):
        objectives = np.ones(len(total_rewards))
        for s, tr in enumerate(total_rewards):
            if u_type == 1:
                objectives[s] = 1 if tr - thresholds[arm] >= 0 else 0
            elif u_type == 3:
                objectives[s] = (1 + np.exp(-u_order * (1 - thresholds[arm]))) / (1 + np.exp(-u_order * (tr - thresholds[arm])))
            else:
                objectives[s] = 1 - thresholds[arm]**(- 1/u_order) * (np.maximum(0, thresholds[arm] - tr))**(1/u_order)
                    
            return objectives

    if num_actions == 1:
        vals = np.ones((num_states, num_arms))
        for a in range(num_arms):
            if function_type[a] > 0:
                vals[:, a] = (np.linspace(0, num_states-1, num=num_states)) ** function_type[a] / (num_states-1) ** function_type[a]
                vals[:, a] = np.round(compute_risk(vals[:, a], a), 2)
    else:
        vals = np.ones((num_states, num_actions, num_arms))
        for a in range(num_arms):
            for act in range(num_actions):
                if function_type[a] > 0:
                    vals[:, act, a] = (np.linspace(0, num_states-1, num=num_states)) ** function_type[a] / (num_states-1) ** function_type[a]
                    vals[:, act, a] = np.round(compute_risk(vals[:, act, a], a), 2)

    vals = np.round(vals / time_horizon, 3)
    return vals


# Define the Markov dynamics for each arm
class MarkovDynamics:

    def __init__(self, num_arms: int, num_states: int, prob_remain, transition_type: int, increasing: bool):
        self.num_s = num_states
        self.num_a = num_arms
        self.increasing = increasing
        self.transitions = self.generate(prob_remain, transition_type)

    @staticmethod
    def ceil_to_decimals(arr, decimals):
        factor = 10 ** decimals
        return np.floor(arr * factor) / factor

    def generate(self, prob_remain, transition_type):
        transitions = np.zeros((self.num_s, self.num_s, 2, self.num_a))
        for a in range(self.num_a):
            if transition_type == 0:
                transitions[0, 0, 1, a] = 1
                transitions[0, 0, 0, a] = 1
                for s in range(1, self.num_s):
                    transitions[s, s, 1, a] = 1
                    transitions[s, 0, 0, a] = 1 - prob_remain[a]
                    transitions[s, s, 0, a] = prob_remain[a]
            elif transition_type == 1:
                transitions[0, 0, 1, a] = 1
                transitions[0, 0, 0, a] = 1
                for s in range(1, self.num_s):
                    transitions[s, 0, 1, a] = 1 - 2 * prob_remain[a]
                    transitions[s, s, 1, a] = 2 * prob_remain[a]
                    transitions[s, 0, 0, a] = 1 - prob_remain[a]
                    transitions[s, s, 0, a] = prob_remain[a]
            elif transition_type == 2:
                transitions[0, 0, 1, a] = 1
                transitions[0, 0, 0, a] = 1
                for s in range(1, self.num_s):
                    transitions[s, s-1, 1, a] = prob_remain[a]
                    transitions[s, s, 1, a] = 1 - prob_remain[a]
                    transitions[s, s-1, 0, a] = 1 - prob_remain[a]
                    transitions[s, s, 0, a] = prob_remain[a]
            elif transition_type == 3:
                for s in range(self.num_s-1):
                    transitions[s, s, 1, a] = (self.num_s - s - 1) * prob_remain[a]
                    transitions[s, self.num_s-1, 1, a] = 1 - (self.num_s - s - 1) * prob_remain[a]
                transitions[self.num_s-1, self.num_s-1, 1, a] = 1
                transitions[0, 0, 0, a] = 1
                transitions[1:, 0, 0, a] = (1 - (self.num_s - 1) * prob_remain[a]) * np.ones(self.num_s-1)
                transitions[1:, 1:, 0, a] = np.tril(np.full((self.num_s - 1, self.num_s - 1), prob_remain[a]))
                for s in range(1, self.num_s):
                    transitions[s, s, 0, a] = (self.num_s - s) * transitions[s, s, 0, a]
            elif transition_type == 4:
                transitions[:self.num_s-1, :self.num_s-1, 1, a] = prob_remain[a] * np.triu(np.ones((self.num_s-1, self.num_s-1)))
                transitions[:, self.num_s-1, 1, a] = 1 - (self.num_s - np.arange(self.num_s) - 1) * prob_remain[a]
                transitions[0, 0, 0, a] = 1
                transitions[1:, 0, 0, a] = (1 - (self.num_s - 1) * prob_remain[a]) * np.ones(self.num_s-1)
                transitions[1:, 1:, 0, a] = np.tril(np.full((self.num_s - 1, self.num_s - 1), prob_remain[a]))
                for s in range(1, self.num_s):
                    transitions[s, s, 0, a] = (self.num_s - s) * transitions[s, s, 0, a]
            elif transition_type == 5:
                transitions[0, 0, 0, a] = 1
                for s in range(1, self.num_s):
                    transitions[s, 1:s+1, 0, a] = np.round(prob_remain[a] * np.ones(s), 2)
                    transitions[s, 0, 0, a] = 1 - sum(transitions[s, 1:, 0, a])
                for s in range(self.num_s):
                    transitions[s, -1, 1, a] = np.round((s+1) * prob_remain[a], 2)
                    transitions[s, 0, 1, a] = 1 - transitions[s, -1, 1, a]
            elif transition_type == 6:
                transitions[:, :, 0, a] = np.round(((1 - prob_remain[a]) / (self.num_s-1)) * np.ones((self.num_s, self.num_s)), 2)
                for s in range(self.num_s):
                    transitions[s, s, 0, a] = 1 - sum(transitions[0, 1:, 0, a])
                transitions[:, :, 1, a] = np.round(((1 - prob_remain[a]) / (self.num_s-1)) * np.triu(np.ones((self.num_s, self.num_s))), 2)
                for s in range(self.num_s):
                    transitions[s, -1, 1, a] = 1 - sum(transitions[s, :self.num_s-1, 1, a])
            elif transition_type == 11:
                pr_ss_0 = prob_remain[0][a]
                pr_sr_0 = prob_remain[1][a]
                pr_sp_0 = prob_remain[2][a]
                if pr_ss_0 + pr_sr_0 + pr_sp_0 > 1:
                    sumprobs = pr_ss_0 + pr_sr_0 + pr_sp_0
                    pr_ss_0 = self.ceil_to_decimals(pr_ss_0 / sumprobs, 3)
                    pr_sr_0 = self.ceil_to_decimals(pr_sr_0 / sumprobs, 3)
                    pr_sp_0 = self.ceil_to_decimals(pr_sp_0 / sumprobs, 3)
                pr_rr_0 = prob_remain[3][a]
                pr_rp_0 = prob_remain[4][a]
                if pr_rr_0 + pr_rp_0 > 1:
                    sumprobs = pr_rr_0 + pr_rp_0
                    pr_rr_0 = self.ceil_to_decimals(pr_rr_0 / sumprobs, 3)
                    pr_rp_0 = self.ceil_to_decimals(pr_rp_0 / sumprobs, 3)
                pr_pp_0 = prob_remain[5][a]
                pr_ss_1 = prob_remain[6][a]
                pr_sr_1 = prob_remain[7][a]
                pr_sp_1 = prob_remain[8][a]
                if pr_ss_1 + pr_sr_1 + pr_sp_1 > 1:
                    sumprobs = pr_ss_1 + pr_sr_1 + pr_sp_1
                    pr_ss_1 = self.ceil_to_decimals(pr_ss_1 / sumprobs, 3)
                    pr_sr_1 = self.ceil_to_decimals(pr_sr_1 / sumprobs, 3)
                    pr_sp_1 = self.ceil_to_decimals(pr_sp_1 / sumprobs, 3)
                pr_rr_1 = prob_remain[3][a]
                pr_rp_1 = prob_remain[4][a]
                if pr_rr_1 + pr_rp_1 > 1:
                    sumprobs = pr_rr_1 + pr_rp_1
                    pr_rr_1 = self.ceil_to_decimals(pr_rr_1 / sumprobs, 3)
                    pr_rp_1 = self.ceil_to_decimals(pr_rp_1 / sumprobs, 3)
                pr_pp_1 = prob_remain[11][a]
                transitions[:, :, 0, a] = np.array([
                    [1, 0, 0, 0],
                    [1 - pr_pp_0, pr_pp_0, 0, 0],
                    [1 - (pr_rp_0 + pr_rr_0), pr_rp_0, pr_rr_0, 0],
                    [1 - (pr_sp_0 + pr_sr_0 + pr_ss_0), pr_sp_0, pr_sr_0, pr_ss_0]
                ])
                transitions[:, :, 1, a] = np.array([
                    [1, 0, 0, 0],
                    [1 - pr_pp_1, pr_pp_1, 0, 0],
                    [1 - (pr_rp_1 + pr_rr_1), pr_rp_1, pr_rr_1, 0],
                    [1 - (pr_sp_1 + pr_sr_1 + pr_ss_1), pr_sp_1, pr_sr_1, pr_ss_1]
                ])
            elif transition_type == 12:
                pr_ss_0 = prob_remain[0][a]
                pr_sr_0 = prob_remain[1][a]
                if pr_ss_0 + pr_sr_0 > 1:
                    sumprobs = pr_ss_0 + pr_sr_0
                    pr_ss_0 = self.ceil_to_decimals(pr_ss_0 / sumprobs, 3)
                    pr_sr_0 = self.ceil_to_decimals(pr_sr_0 / sumprobs, 3)
                pr_rr_0 = prob_remain[2][a]
                pr_pp_0 = prob_remain[3][a]
                pr_ss_1 = prob_remain[4][a]
                pr_sr_1 = prob_remain[5][a]
                if pr_ss_1 + pr_sr_1 > 1:
                    sumprobs = pr_ss_1 + pr_sr_1
                    pr_ss_1 = self.ceil_to_decimals(pr_ss_1 / sumprobs, 3)
                    pr_sr_1 = self.ceil_to_decimals(pr_sr_1 / sumprobs, 3)
                pr_rr_1 = prob_remain[6][a]
                pr_pp_1 = prob_remain[7][a]
                transitions[:, :, 0, a] = np.array([
                    [1, 0, 0, 0],
                    [1 - pr_pp_0, pr_pp_0, 0, 0],
                    [0, 1 - pr_rr_0, pr_rr_0, 0],
                    [0, 1 - (pr_sr_0 + pr_ss_0), pr_sr_0, pr_ss_0]
                ])
                transitions[:, :, 1, a] = np.array([
                    [1, 0, 0, 0],
                    [1 - pr_pp_1, pr_pp_1, 0, 0],
                    [0, 1 - pr_rr_1, pr_rr_1, 0],
                    [0, 1 - (pr_sr_1 + pr_ss_1), pr_sr_1, pr_ss_1]
                ])
            elif transition_type == 13:
                pr_ss_0 = prob_remain[0][a]
                pr_sp_0 = prob_remain[1][a]
                if pr_ss_0 + pr_sp_0 > 1:
                    sumprobs = pr_ss_0 + pr_sp_0
                    pr_ss_0 = self.ceil_to_decimals(pr_ss_0 / sumprobs, 3)
                    pr_sp_0 = self.ceil_to_decimals(pr_sp_0 / sumprobs, 3)
                pr_pp_0 = prob_remain[2][a]
                pr_ss_1 = prob_remain[3][a]
                pr_sp_1 = prob_remain[4][a]
                if pr_ss_1 + pr_sp_1 > 1:
                    sumprobs = pr_ss_1 + pr_sp_1
                    pr_ss_1 = self.ceil_to_decimals(pr_ss_1 / sumprobs, 3)
                    pr_sp_1 = self.ceil_to_decimals(pr_sp_1 / sumprobs, 3)
                pr_pp_1 = prob_remain[5][a]
                transitions[:, :, 0, a] = np.array([
                    [1, 0, 0],
                    [1 - pr_pp_0, pr_pp_0, 0],
                    [1 - (pr_sp_0 + pr_ss_0), pr_sp_0, pr_ss_0]
                ])
                transitions[:, :, 1, a] = np.array([
                    [1, 0, 0],
                    [1 - pr_pp_1, pr_pp_1, 0],
                    [1 - (pr_sp_1 + pr_ss_1), pr_sp_1, pr_ss_1]
                ])
            elif transition_type == 14:
                pr_ss_0 = prob_remain[0][a]
                pr_pp_0 = prob_remain[1][a]
                pr_ss_1 = prob_remain[2][a]
                pr_pp_1 = prob_remain[3][a]
                transitions[:, :, 0, a] = np.array([
                    [1, 0, 0],
                    [1-pr_pp_0, pr_pp_0, 0],
                    [0, 1-pr_ss_0, pr_ss_0]
                    ])
                transitions[:, :, 1, a] = np.array([
                    [1, 0, 0],
                    [1-pr_pp_1, pr_pp_1, 0],
                    [0, 1-pr_ss_1, pr_ss_1]
                    ])

        return transitions

