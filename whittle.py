### Risk-Neutral & Risk-Aware Whittle Index
import numpy as np
from processes import compute_risk


class Whittle:

    def __init__(self, num_states: int, num_arms: int, reward, transition, horizon):
        self.num_x = num_states
        self.num_a = num_arms
        self.reward = reward
        self.transition = transition
        self.horizon = horizon
        self.digits = 4
        self.w_indices = []

    def get_whittle_indices(self, computation_type, params, n_trials):
        if computation_type == 2:
            l_steps = params[1] / n_trials
            self.whittle_binary_search(params[0], params[1], l_steps)
        else:
            self.whittle_brute_force(params[0], params[1], n_trials)

    def is_equal_mat(self, mat1, mat2):
        for t in range(self.horizon):
            if not np.array_equal(mat1[:, t], mat2[:, t]):
                return False
        return True

    def indexability_check(self, arm_indices, nxt_pol, ref_pol, penalty):
        for t in range(self.horizon):
            if np.any((ref_pol[:, t] == 0) & (nxt_pol[:, t] == 1)):
                print("Not indexable!")
                return False, np.zeros((self.num_x, self.horizon))
            else:
                elements = np.argwhere((ref_pol[:, t] == 1) & (nxt_pol[:, t] == 0))
                for e in elements:
                    arm_indices[e, t] = penalty
        return True, arm_indices

    def whittle_brute_force(self, lower_bound, upper_bound, num_trials):
        for arm in range(self.num_a):
            arm_indices = np.zeros((self.num_x, self.horizon))
            penalty_ref = lower_bound
            ref_pol, _, _ = self.backward(arm, penalty_ref)
            upb_pol, _, _ = self.backward(arm, upper_bound)
            for penalty in np.linspace(lower_bound, upper_bound, num_trials):
                nxt_pol, _, _ = self.backward(arm, penalty)
                if self.is_equal_mat(nxt_pol, upb_pol):
                    flag, arm_indices = self.indexability_check(arm_indices, nxt_pol, ref_pol, penalty)
                    break
                else:
                    if not self.is_equal_mat(nxt_pol, ref_pol):
                        flag, arm_indices = self.indexability_check(arm_indices, nxt_pol, ref_pol, penalty)
                        if flag:
                            ref_pol = np.copy(nxt_pol)
                        else:
                            break
            self.w_indices.append(arm_indices)

    def whittle_binary_search(self, lower_bound, upper_bound, l_steps):
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
            self.w_indices.append(arm_indices)

    def backward(self, arm, penalty):

        # Value function initialization
        V = np.zeros((self.num_x, self.horizon + 1), dtype=np.float32)

        # State-action value function
        Q = np.zeros((self.num_x, self.horizon, 2), dtype=np.float32)

        # Policy function
        pi = np.zeros((self.num_x, self.horizon), dtype=np.int32)

        # Backward induction timing
        t = self.horizon - 1

        # The value iteration loop
        while t >= 0:

            # Loop over the first dimension of the state space
            for x in range(self.num_x):

                # Get the state-action value functions
                for act in range(2):
                    if len(self.reward.shape) == 3:
                        Q[x, t, act] = self.reward[x, act, arm] - penalty * act / self.horizon + np.dot(V[:, t + 1], self.transition[x, :, act, arm])
                    else:
                        Q[x, t, act] = self.reward[x, arm] - penalty * act / self.horizon + np.dot(V[:, t + 1], self.transition[x, :, act, arm])

                # Get the value function and the policy
                if Q[x, t, 1] < Q[x, t, 0]:
                    V[x, t] = Q[x, t, 0]
                    pi[x, t] = 0
                else:
                    V[x, t] = Q[x, t, 1]
                    pi[x, t] = 1

            t = t - 1

        return pi, V, Q

    @staticmethod
    def Whittle_policy(whittle_indices, n_selection, current_x, current_t):
        num_a = len(whittle_indices)

        current_indices = np.zeros(num_a)
        for arm in range(num_a):
            current_indices[arm] = whittle_indices[arm][current_x[arm], current_t]

        # Sort indices based on values and shuffle indices with same values
        sorted_indices = np.argsort(current_indices)[::-1]
        unique_indices, counts = np.unique(current_indices[sorted_indices], return_counts=True)
        top_indices = []
        top_len = 0
        for idx in range(len(unique_indices)):
            indices = np.where(current_indices == unique_indices[len(unique_indices) - idx - 1])[0]
            shuffled_indices = np.random.permutation(indices)
            if top_len + len(shuffled_indices) < n_selection:
                top_indices.extend(list(shuffled_indices))
                top_len += len(shuffled_indices)
            elif top_len + len(shuffled_indices) == n_selection:
                top_indices.extend(list(shuffled_indices))
                top_len += len(shuffled_indices)
                break
            else:
                top_indices.extend(list(shuffled_indices[:n_selection - top_len]))
                top_len += len(shuffled_indices[:n_selection - top_len])
                break

        # Create action vector
        action_vector = np.zeros_like(current_indices, dtype=np.int32)
        action_vector[top_indices] = 1

        return action_vector


class RiskAwareWhittle:
    
    def __init__(self, num_states: int, num_arms: int, rewards, transition, horizon, u_type, u_order, thresholds):
        self.num_x = num_states
        self.num_a = num_arms
        self.rewards = rewards
        self.transition = transition
        self.horizon = horizon
        self.u_type = u_type

        self.digits = 3
        self.n_realize = []
        self.n_augment = [0] * num_arms
        self.all_rews = []
        self.all_valus = []

        for a in range(num_arms):

            if len(self.rewards.shape) == 3:
                all_immediate_rew = self.rewards[:, :, a]
            else:
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

            arm_valus = []
            for total_rewards in all_total_rewards:
                if u_type == 2:
                    arm_valus.append(np.round(1 - thresholds[a] ** (- 1 / u_order) * (np.maximum(0, thresholds[a] - total_rewards)) ** (1 / u_order), 3))
                elif u_type == 3:
                    arm_valus.append(np.round((1 + np.exp(-u_order * (1 - thresholds[a]))) / (1 + np.exp(-u_order * (total_rewards - thresholds[a]))), 3))
                else:
                    arm_valus.append(1 if total_rewards >= thresholds[a] else 0)

            self.all_valus.append(arm_valus)

        self.w_indices = []

    def get_whittle_indices(self, computation_type, params, n_trials):
        if computation_type == 1:
            l_steps = params[1] / n_trials
            self.whittle_binary_search(params[0], params[1], l_steps)
        else:
            self.whittle_brute_force(params[0], params[1], n_trials)

    def is_equal_mat(self, mat1, mat2, realize_index):
        for t in range(self.horizon):
            mat1_new = mat1[:realize_index[t], :]
            mat2_new = mat2[:realize_index[t], :]
            if not np.array_equal(mat1_new, mat2_new):
                return False
        return True

    def indexability_check(self, arm, arm_indices, realize_index, nxt_pol, ref_pol, penalty, nxt_Q, ref_Q):
        for t in range(self.horizon):
            ref_pol_new = ref_pol[:realize_index[t], :, t]
            nxt_pol_new = nxt_pol[:realize_index[t], :, t]
            # ref_Q_new = ref_Q[:realize_index[t], :, t, :]
            # nxt_Q_new = nxt_Q[:realize_index[t], :, t, :]
            if np.any((ref_pol_new == 0) & (nxt_pol_new == 1)):
                print("Not indexable!")
                # elements = np.argwhere((ref_pol_new == 0) & (nxt_pol_new == 1))
                # for e in elements:
                #     print(f'element: {[e[0], e[1], t]}')
                #     print(f'penalty: {penalty}')
                #     print(f'ref policy: {ref_pol_new[:, e[1]]}')
                #     print(f'nxt policy: {nxt_pol_new[:, e[1]]}')
                #     print(f'ref Q0: {ref_Q_new[e[0], e[1], 0]}')
                #     print(f'ref Q1: {ref_Q_new[e[0], e[1], 1]}')
                #     print(f'nxt Q0: {nxt_Q_new[e[0], e[1], 0]}')
                #     print(f'nxt Q1: {nxt_Q_new[e[0], e[1], 1]}')
                return False, np.zeros((self.n_augment[arm], self.num_x, self.horizon))
            else:
                elements = np.argwhere((ref_pol_new == 1) & (nxt_pol_new == 0))
                for e in elements:
                    arm_indices[e[0], e[1], t] = penalty
        return True, arm_indices

    def whittle_brute_force(self, lower_bound, upper_bound, num_trials):

        for arm in range(self.num_a):
            arm_indices = np.zeros((self.n_augment[arm], self.num_x, self.horizon))
            penalty_ref = lower_bound
            ref_pol, _, ref_Q = self.backward_discreteliftedstate(arm, penalty_ref)
            upb_pol, _, _ = self.backward_discreteliftedstate(arm, upper_bound)
            for penalty in np.linspace(lower_bound, upper_bound, num_trials):
                penalty = np.round(penalty, self.digits)
                nxt_pol, _, nxt_Q = self.backward_discreteliftedstate(arm, penalty)
                if self.is_equal_mat(nxt_pol, upb_pol, self.n_realize[arm]):
                    flag, arm_indices = self.indexability_check(arm, arm_indices, self.n_realize[arm], nxt_pol, ref_pol, penalty, nxt_Q, ref_Q)
                    break
                else:
                    if not self.is_equal_mat(nxt_pol, ref_pol, self.n_realize[arm]):
                        flag, arm_indices = self.indexability_check(arm, arm_indices, self.n_realize[arm], nxt_pol, ref_pol, penalty, nxt_Q, ref_Q)
                        if flag:
                            ref_pol = np.copy(nxt_pol)
                            ref_Q = np.copy(nxt_Q)
                        else:
                            break
            self.w_indices.append(arm_indices)

    def whittle_binary_search(self, lower_bound, upper_bound, l_steps):

        for arm in range(self.num_a):
            arm_indices = np.zeros((self.n_augment[arm], self.num_x, self.horizon))
            penalty_ref = lower_bound
            ref_pol, _, ref_Q = self.backward_discreteliftedstate(arm, penalty_ref)
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
                nxt_pol, _, nxt_Q = self.backward_discreteliftedstate(arm, penalty_ref)
                indexability_flag, arm_indices = self.indexability_check(arm, arm_indices, self.n_realize[arm], nxt_pol, ref_pol, penalty, nxt_Q, ref_Q)
                if indexability_flag:
                    ref_pol = np.copy(nxt_pol)
                    ref_Q = np.copy(nxt_Q)
                else:
                    break
            self.w_indices.append(arm_indices)

    def backward_discreteliftedstate(self, arm, penalty):

        # Value function initialization
        V = np.zeros((self.n_augment[arm], self.num_x, self.horizon + 1), dtype=np.float32)
        for l in range(self.n_augment[arm]):
            V[l, :, self.horizon] = self.all_valus[arm][l] * np.ones(self.num_x)

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

                    for act in range(2):

                        nxt_l = max(0, min(self.n_augment[arm] - 1, l + x))
                        
                        Q[l, x, t, act] = np.round(- penalty * act / self.horizon + np.dot(V[nxt_l, :, t + 1], self.transition[x, :, act, arm]), self.digits + 1)

                    # Get the value function and the policy
                    if 0 < Q[l, x, t, 0] - Q[l, x, t, 1]:
                        V[l, x, t] = Q[l, x, t, 0]
                        pi[l, x, t] = 0
                    else:
                        V[l, x, t] = Q[l, x, t, 1]
                        pi[l, x, t] = 1

            t = t - 1
        
        return pi, V, Q

    @staticmethod
    def Whittle_policy(whittle_indices, n_selection, current_x, current_l, current_t):
        num_a = len(whittle_indices)

        current_indices = np.zeros(num_a)
        count_positive = 0
        for arm in range(num_a):
            w_idx = whittle_indices[arm][current_l[arm], current_x[arm], current_t]
            current_indices[arm] = w_idx
            if w_idx >= 0:
                count_positive += 1
        n_selection = np.minimum(n_selection, count_positive)

        # Sort indices based on values and shuffle indices with same values
        sorted_indices = np.argsort(current_indices)[::-1]
        unique_indices, counts = np.unique(current_indices[sorted_indices], return_counts=True)
        top_indices = []
        top_len = 0
        for idx in range(len(unique_indices)):
            indices = np.where(current_indices == unique_indices[len(unique_indices) - idx - 1])[0]
            shuffled_indices = np.random.permutation(indices)
            if top_len + len(shuffled_indices) < n_selection:
                top_indices.extend(list(shuffled_indices))
                top_len += len(shuffled_indices)
            elif top_len + len(shuffled_indices) == n_selection:
                top_indices.extend(list(shuffled_indices))
                top_len += len(shuffled_indices)
                break
            else:
                top_indices.extend(list(shuffled_indices[:n_selection - top_len]))
                top_len += len(shuffled_indices[:n_selection - top_len])
                break

        # Create action vector
        action_vector = np.zeros_like(current_indices, dtype=np.int32)
        action_vector[top_indices] = 1

        return action_vector


from itertools import product
def possible_reward_sums(rewards, num_steps):
    reward_combinations = product(rewards, repeat=num_steps)
    sums = set()
    for combination in reward_combinations:
        sums.add(np.round(sum(combination), 3))
    return sorted(sums)