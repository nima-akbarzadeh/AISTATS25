### Risk-Neutral & Risk-Aware Whittle Index
import numpy as np
from itertools import product
from processes import compute_utility


class Whittle:

    def __init__(self, num_states: int, num_arms: int, reward, transition, horizon):
        self.num_x = num_states
        self.num_a = num_arms
        self.reward = reward
        self.transition = transition
        self.horizon = horizon
        self.digits = 3
        self.whittle_indices = []

    def get_indices(self, index_range, n_trials):
        l_steps = index_range / n_trials
        self.binary_search(0, index_range, l_steps)

    def is_equal_mat(self, mat1, mat2, tol=1e-6):
        return np.all(np.abs(mat1 - mat2) < tol)

    def indexability_check(self, arm_indices, nxt_pol, ref_pol, penalty):
        for t in range(self.horizon):
            if np.any((ref_pol[:, t] == 0) & (nxt_pol[:, t] == 1)):
                return False, np.zeros((self.num_x, self.horizon))
            else:
                elements = np.argwhere((ref_pol[:, t] == 1) & (nxt_pol[:, t] == 0))
                for e in elements:
                    arm_indices[e, t] = penalty
        return True, arm_indices

    def binary_search(self, lower_bound, upper_bound, l_steps):
        for arm in range(self.num_a):
            arm_indices = np.zeros((self.num_x, self.horizon))
            penalty_ref = lower_bound
            ref_pol, _, _ = self.backward(arm, penalty_ref)
            ubp_pol, _, _ = self.backward(arm, upper_bound)
            while not self.is_equal_mat(ref_pol, ubp_pol):
                lb_temp = penalty_ref
                ub_temp = upper_bound
                penalty = 0.5 * (lb_temp + ub_temp)
                diff = np.abs(ub_temp - lb_temp)
                while l_steps < diff:
                    som_pol, _, _ = self.backward(arm, penalty)
                    if self.is_equal_mat(som_pol, ref_pol):
                        lb_temp = penalty
                    else:
                        ub_temp = penalty
                    penalty = 0.5 * (lb_temp + ub_temp)
                    diff = np.abs(ub_temp - lb_temp)
                penalty_ref = penalty + l_steps
                nxt_pol, _, _ = self.backward(arm, penalty_ref)
                flag, arm_indices = self.indexability_check(arm_indices, nxt_pol, ref_pol, penalty)
                if flag:
                    ref_pol = np.copy(nxt_pol)
                else:
                    break
            self.whittle_indices.append(arm_indices)

    def backward(self, arm, penalty):
        # Value function initialization
        V = np.zeros((self.num_x, self.horizon + 1), dtype=np.float32)

        # State-action value function
        Q = np.zeros((self.num_x, self.horizon, 2), dtype=np.float32)

        # Policy function
        pi = np.zeros((self.num_x, self.horizon), dtype=np.int32)

        # Backward induction timing
        for t in range(self.horizon - 1, -1, -1):
            for x in range(self.num_x):
                # Calculate Q-values for both actions
                Q[x, t, 0] = self.reward[x, arm] + np.dot(V[:, t + 1], self.transition[x, :, 0, arm])
                Q[x, t, 1] = self.reward[x, arm] - penalty / self.horizon + np.dot(V[:, t + 1], self.transition[x, :, 1, arm])

                # Optimal action and value
                pi[x, t] = np.argmax(Q[x, t, :])
                V[x, t] = np.max(Q[x, t, :])

        return pi, V, Q

    def take_action(self, n_choices, current_x, current_t):

        current_indices = np.zeros(self.num_a)
        count_positive = 0
        for arm in range(self.num_a):
            w_idx = self.whittle_indices[arm][current_x[arm], current_t]
            current_indices[arm] = w_idx
            if w_idx >= 0:
                count_positive += 1
        n_choices = np.minimum(n_choices, count_positive)

        max_index = np.max(current_indices)
        candidates = np.where(current_indices == max_index)[0]
        chosen = np.random.choice(candidates, size=min(n_choices, len(candidates)), replace=False)
        action_vector = np.zeros_like(current_indices, dtype=int)
        action_vector[chosen] = 1

        return action_vector


def possible_reward_sums(rewards, num_steps):
    reward_combinations = product(rewards, repeat=num_steps)
    sums = set()
    for combination in reward_combinations:
        sums.add(np.round(sum(combination), 3))
    return sorted(sums)


class RiskAwareWhittle:
    
    def __init__(self, num_states: int, num_arms: int, rewards, transition, horizon, u_type, u_order, threshold):
        self.num_x = num_states
        self.num_a = num_arms
        self.rewards = rewards
        self.transition = transition
        self.horizon = horizon
        self.u_type = u_type
        self.digits = 3
        self.n_realize = []
        self.n_augment = [0] * self.num_a
        self.all_rews = []
        self.all_utility_values = []

        for a in range(self.num_a):

            all_immediate_rew = self.rewards[:, a]
            arm_n_realize = []
            all_total_rewards = []
            for t in range(self.horizon):
                all_total_rewards_by_t = possible_reward_sums(all_immediate_rew.flatten(), t + 1)
                arm_n_realize.append(len(all_total_rewards_by_t))
                if t == self.horizon - 1:
                    all_total_rewards = all_total_rewards_by_t

            self.n_augment[a] = len(all_total_rewards)
            self.all_rews.append(all_total_rewards)
            self.n_realize.append(arm_n_realize)

            arm_utilities = []
            for total_reward in all_total_rewards:
                arm_utilities.append(compute_utility(total_reward, threshold, u_type, u_order))
            self.all_utility_values.append(arm_utilities)

        self.whittle_indices = []

    def get_indices(self, index_range, n_trials):
        l_steps = index_range / n_trials
        self.binary_search(0, index_range, l_steps)

    def is_equal_mat(self, mat1, mat2, realize_index):
        for t in range(self.horizon):
            mat1_new = mat1[:realize_index[t], :]
            mat2_new = mat2[:realize_index[t], :]
            if not np.array_equal(mat1_new, mat2_new):
                return False
        return True

    def indexability_check(self, arm, arm_indices, realize_index, nxt_pol, ref_pol, penalty):
        for t in range(self.horizon):
            ref_pol_new = ref_pol[:realize_index[t], :, t]
            nxt_pol_new = nxt_pol[:realize_index[t], :, t]
            if np.any((ref_pol_new == 0) & (nxt_pol_new == 1)):
                return False, np.zeros((self.n_augment[arm], self.num_x, self.horizon))
            else:
                elements = np.argwhere((ref_pol_new == 1) & (nxt_pol_new == 0))
                for e in elements:
                    arm_indices[e[0], e[1], t] = penalty
        return True, arm_indices

    def binary_search(self, lower_bound, upper_bound, l_steps):

        for arm in range(self.num_a):
            arm_indices = np.zeros((self.n_augment[arm], self.num_x, self.horizon))
            penalty_ref = lower_bound
            ref_pol, _, _ = self.backward_discreteliftedstate(arm, penalty_ref)
            ubp_pol, _, _ = self.backward_discreteliftedstate(arm, upper_bound)
            while not self.is_equal_mat(ref_pol, ubp_pol, self.n_realize[arm]):
                lb_temp = penalty_ref
                ub_temp = upper_bound
                penalty = np.round(0.5 * (lb_temp + ub_temp), self.digits)
                diff = np.abs(ub_temp - lb_temp)
                while l_steps < diff:
                    som_pol, _, _ = self.backward_discreteliftedstate(arm, penalty)
                    if self.is_equal_mat(som_pol, ref_pol, self.n_realize[arm]):
                        lb_temp = penalty
                    else:
                        ub_temp = penalty
                    penalty = np.round(0.5 * (lb_temp + ub_temp), self.digits)
                    diff = np.abs(ub_temp - lb_temp)
                penalty_ref = penalty + l_steps
                nxt_pol, _, _ = self.backward_discreteliftedstate(arm, penalty_ref)
                indexability_flag, arm_indices = self.indexability_check(arm, arm_indices, self.n_realize[arm], nxt_pol, ref_pol, penalty)
                if indexability_flag:
                    ref_pol = np.copy(nxt_pol)
                else:
                    break
            self.whittle_indices.append(arm_indices)

    def backward_discreteliftedstate(self, arm, penalty):

        # Value function initialization
        V = np.zeros((self.n_augment[arm], self.num_x, self.horizon + 1), dtype=np.float32)
        for l in range(self.n_augment[arm]):
            V[l, :, self.horizon] = self.all_utility_values[arm][l] * np.ones(self.num_x)

        # State-action value function
        Q = np.zeros((self.n_augment[arm], self.num_x, self.horizon, 2), dtype=np.float32)

        # Policy function
        pi = np.zeros((self.n_augment[arm], self.num_x, self.horizon), dtype=np.int32)

        # Backward induction timing
        t = self.horizon - 1

        # The value iteration loop
        while t >= 0:

            # Loop over the first dimension of the state space
            for x in range(self.num_x):

                # Loop over the second dimension of the state space
                for l in range(self.n_realize[arm][t]):

                    nxt_l = max(0, min(self.n_augment[arm] - 1, l + x))
                    
                    Q[l, x, t, 0] = np.dot(V[nxt_l, :, t + 1], self.transition[x, :, 0, arm])
                    Q[l, x, t, 1] = - penalty / self.horizon + np.dot(V[nxt_l, :, t + 1], self.transition[x, :, 1, arm])

                    # Get the value function and the policy
                    pi[l, x, t] = np.argmax(Q[l, x, t, :])
                    V[l, x, t] = np.max(Q[l, x, t, :])

            t = t - 1
        
        return pi, V, Q

    def take_action(self, n_choices, current_l, current_x, current_t):

        current_indices = np.zeros(self.num_a)
        count_positive = 0
        for arm in range(self.num_a):
            w_idx = self.whittle_indices[arm][current_l[arm], current_x[arm], current_t]
            current_indices[arm] = w_idx
            if w_idx >= 0:
                count_positive += 1
        n_choices = np.minimum(n_choices, count_positive)

        max_index = np.max(current_indices)
        candidates = np.where(current_indices == max_index)[0]
        chosen = np.random.choice(candidates, size=min(n_choices, len(candidates)), replace=False)
        action_vector = np.zeros_like(current_indices, dtype=int)
        action_vector[chosen] = 1

        return action_vector
