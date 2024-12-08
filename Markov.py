### Problem Parameters
import numpy as np
from processes import compute_utility


# Define the reward values for each arm
def rewards(time_horizon, num_arms, num_states):
    """
    Calculate rewards for each arm.
    """
    vals = np.ones((num_states, num_arms))
    for a in range(num_arms):
        vals[:, a] = np.linspace(0, num_states-1, num=num_states) / (num_states-1)
    return np.round(vals / time_horizon, 3)


# Define the reward rewards for each arm
def rewards_utility(time_horizon, num_arms, num_states, threshold, u_type, u_order):
    """
    Calculate a utility function of the rewards for each arm.
    """
    vals = np.ones((num_states, num_arms))
    total_rewards = np.linspace(0, num_states-1, num=num_states) / (num_states-1)
    for x, total_reward in enumerate(total_rewards):
        vals[x, :] = compute_utility(total_reward, threshold, u_type, u_order) * np.ones(num_arms)
    return np.round(vals / time_horizon, 3)


def ceil_to_decimals(arr, decimals):
    factor = 10 ** decimals
    return np.floor(arr * factor) / factor


# Define the Markov dynamics for each arm
def get_transitions(num_arms: int, num_states: int, prob_remain, transition_type):

    transitions = np.zeros((num_states, num_states, 2, num_arms))
    for a in range(num_arms):
        if transition_type == 'structured':
            for s in range(num_states-1):
                transitions[s, s, 1, a] = (num_states - s - 1) * prob_remain[a]
                transitions[s, num_states-1, 1, a] = 1 - (num_states - s - 1) * prob_remain[a]
            transitions[num_states-1, num_states-1, 1, a] = 1
            transitions[0, 0, 0, a] = 1
            transitions[1:, 0, 0, a] = (1 - (num_states - 1) * prob_remain[a]) * np.ones(num_states-1)
            transitions[1:, 1:, 0, a] = np.tril(np.full((num_states - 1, num_states - 1), prob_remain[a]))
            for s in range(1, num_states):
                transitions[s, s, 0, a] = (num_states - s) * transitions[s, s, 0, a]
        elif transition_type == 11:
            pr_ss_0 = prob_remain[0][a]
            pr_sr_0 = prob_remain[1][a]
            pr_sp_0 = prob_remain[2][a]
            if pr_ss_0 + pr_sr_0 + pr_sp_0 > 1:
                sumprobs = pr_ss_0 + pr_sr_0 + pr_sp_0
                pr_ss_0 = ceil_to_decimals(pr_ss_0 / sumprobs, 3)
                pr_sr_0 = ceil_to_decimals(pr_sr_0 / sumprobs, 3)
                pr_sp_0 = ceil_to_decimals(pr_sp_0 / sumprobs, 3)
            pr_rr_0 = prob_remain[3][a]
            pr_rp_0 = prob_remain[4][a]
            if pr_rr_0 + pr_rp_0 > 1:
                sumprobs = pr_rr_0 + pr_rp_0
                pr_rr_0 = ceil_to_decimals(pr_rr_0 / sumprobs, 3)
                pr_rp_0 = ceil_to_decimals(pr_rp_0 / sumprobs, 3)
            pr_pp_0 = prob_remain[5][a]
            pr_ss_1 = prob_remain[6][a]
            pr_sr_1 = prob_remain[7][a]
            pr_sp_1 = prob_remain[8][a]
            if pr_ss_1 + pr_sr_1 + pr_sp_1 > 1:
                sumprobs = pr_ss_1 + pr_sr_1 + pr_sp_1
                pr_ss_1 = ceil_to_decimals(pr_ss_1 / sumprobs, 3)
                pr_sr_1 = ceil_to_decimals(pr_sr_1 / sumprobs, 3)
                pr_sp_1 = ceil_to_decimals(pr_sp_1 / sumprobs, 3)
            pr_rr_1 = prob_remain[3][a]
            pr_rp_1 = prob_remain[4][a]
            if pr_rr_1 + pr_rp_1 > 1:
                sumprobs = pr_rr_1 + pr_rp_1
                pr_rr_1 = ceil_to_decimals(pr_rr_1 / sumprobs, 3)
                pr_rp_1 = ceil_to_decimals(pr_rp_1 / sumprobs, 3)
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
                pr_ss_0 = ceil_to_decimals(pr_ss_0 / sumprobs, 3)
                pr_sr_0 = ceil_to_decimals(pr_sr_0 / sumprobs, 3)
            pr_rr_0 = prob_remain[2][a]
            pr_pp_0 = prob_remain[3][a]
            pr_ss_1 = prob_remain[4][a]
            pr_sr_1 = prob_remain[5][a]
            if pr_ss_1 + pr_sr_1 > 1:
                sumprobs = pr_ss_1 + pr_sr_1
                pr_ss_1 = ceil_to_decimals(pr_ss_1 / sumprobs, 3)
                pr_sr_1 = ceil_to_decimals(pr_sr_1 / sumprobs, 3)
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

    return transitions
