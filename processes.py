from itertools import combinations
import random
import numpy as np


def compute_risk(totalrewards, thresholds, u_type, u_order, n_bandits):
    objectives = np.zeros(n_bandits)
    for a in range(n_bandits):
        if u_type == 1:
            objectives[a] = 1 if totalrewards[a] >= thresholds[a] else 0
        elif u_type == 3:
            objectives[a] = (1 + np.exp(-u_order * (1-thresholds[a]))) / (1 + np.exp(-u_order * (totalrewards[a]-thresholds[a])))
        else:
            objectives[a] = 1 - thresholds[a]**(- 1/u_order) * (np.maximum(0, thresholds[a] - totalrewards[a]))**(1/u_order)
            
    return objectives


def Process_WhtlRB(W, n_episodes, n_steps, n_states, n_bandits, n_choices, thresholds, rewards,
                   transitions, initial_states, u_type, u_order):

    ##################################################### Process
    totalrewards = np.zeros((n_bandits, n_episodes))
    objectives = np.zeros((n_bandits, n_episodes))
    counts = np.zeros((n_states, n_states, 2, n_bandits))
    for k in range(n_episodes):
        states = initial_states.copy()
        for t in range(n_steps):
            _states = np.copy(states)
            actions = W.Whittle_policy(W.w_indices, n_choices, _states, t)
            for a in range(n_bandits):
                if len(rewards.shape) == 3:
                    totalrewards[a, k] += rewards[_states[a], actions[a], a]
                else:
                    totalrewards[a, k] += rewards[_states[a], a]
                states[a] = np.random.choice(n_states, p=transitions[_states[a], :, actions[a], a])
                counts[_states[a], states[a], actions[a], a] += 1
        objectives[:, k] = compute_risk(totalrewards[:, k], thresholds, u_type, u_order, n_bandits)

    return np.around(totalrewards, 2), np.around(objectives, 2), counts


def Process_LearnWhtlRB(W, LearnW, n_episodes, n_steps, n_states, n_bandits, n_choices, thresholds, rewards, 
                        transitions, initial_states, u_type, u_order):
    totalrewards = np.zeros((n_bandits, n_episodes))
    objectives = np.zeros((n_bandits, n_episodes))
    learn_totalrewards = np.zeros((n_bandits, n_episodes))
    learn_objectives = np.zeros((n_bandits, n_episodes))
    counts = np.zeros((n_states, n_states, 2, n_bandits))
    for k in range(n_episodes):
        states = initial_states.copy()
        learn_states = initial_states.copy()
        for t in range(n_steps):
            _states = np.copy(states)
            _learn_states = np.copy(learn_states)
            actions = W.Whittle_policy(W.w_indices, n_choices, _states, t)
            learn_actions = LearnW.Whittle_policy(LearnW.w_indices, n_choices, _learn_states, t)
            for a in range(n_bandits):
                if len(rewards.shape) == 3:
                    totalrewards[a, k] += rewards[_states[a], actions[a], a]
                    learn_totalrewards[a, k] += rewards[_learn_states[a], learn_actions[a], a]
                else:
                    totalrewards[a, k] += rewards[_states[a], a]
                    learn_totalrewards[a, k] += rewards[_learn_states[a], a]
                states[a] = np.random.choice(n_states, p=transitions[_states[a], :, actions[a], a])
                if actions[a] == learn_actions[a] and _states[a] == _learn_states[a]:
                    learn_states[a] = np.copy(states[a])
                else:
                    learn_states[a] = np.random.choice(n_states, p=transitions[_learn_states[a], :, learn_actions[a], a])
                counts[_learn_states[a], learn_states[a], learn_actions[a], a] += 1
        objectives[:, k] = compute_risk(totalrewards[:, k], thresholds, u_type, u_order, n_bandits)
        learn_objectives[:, k] = compute_risk(learn_totalrewards[:, k], thresholds, u_type, u_order, n_bandits)

    return totalrewards, objectives, learn_totalrewards, learn_objectives, counts


def Process_SafeRB(SafeW, n_episodes, n_steps, n_states, n_bandits, n_choices, thresholds, rewards,
                   transitions, initial_states, u_type, u_order):

    ##################################################### Process
    totalrewards = np.zeros((n_bandits, n_episodes))
    objectives = np.zeros((n_bandits, n_episodes))
    counts = np.zeros((n_states, n_states, 2, n_bandits))
    for k in range(n_episodes):
        states = initial_states.copy()
        _lifted = np.zeros(n_bandits, dtype=np.int32)
        for t in range(n_steps):
            _states = np.copy(states)
            actions = SafeW.Whittle_policy(SafeW.w_indices, n_choices, _states, _lifted, t)
            for b in range(n_bandits):
                _lifted[b] = max(0, min(SafeW.n_augment[b]-1, _lifted[b] + _states[b]))
            for a in range(n_bandits):
                if len(rewards.shape) == 3:
                    totalrewards[a, k] += rewards[_states[a], actions[a], a]
                else:
                    totalrewards[a, k] += rewards[_states[a], a]
                states[a] = np.random.choice(n_states, p=transitions[_states[a], :, actions[a], a])
                counts[_states[a], states[a], actions[a], a] += 1
        objectives[:, k] = compute_risk(totalrewards[:, k], thresholds, u_type, u_order, n_bandits)

    return np.around(totalrewards, 2), np.around(objectives, 2), counts


def Process_LearnSafeRB(SafeW, LearnW, n_episodes, n_steps, n_states, n_bandits,
                        n_choices, thresholds, rewards, transitions, initial_states, u_type, u_order):
    totalrewards = np.zeros((n_bandits, n_episodes))
    objectives = np.zeros((n_bandits, n_episodes))
    learn_totalrewards = np.zeros((n_bandits, n_episodes))
    learn_objectives = np.zeros((n_bandits, n_episodes))
    counts = np.zeros((n_states, n_states, 2, n_bandits))
    for k in range(n_episodes):
        states = initial_states.copy()
        learn_states = initial_states.copy()
        _lifted = np.zeros(n_bandits, dtype=np.int32)
        _learn_lifted = np.zeros(n_bandits, dtype=np.int32)
        for t in range(n_steps):
            _states = np.copy(states)
            _learn_states = np.copy(learn_states)
            actions = SafeW.Whittle_policy(SafeW.w_indices, n_choices, _states, _lifted, t)
            learn_actions = LearnW.Whittle_policy(LearnW.w_indices, n_choices, _learn_states, _learn_lifted, t)
            for b in range(n_bandits):
                _lifted[b] = max(0, min(SafeW.n_augment[b]-1, _lifted[b] + _states[b]))
                _learn_lifted[b] = max(0, min(LearnW.n_augment[b]-1, _learn_lifted[b] + _learn_states[b]))
            for a in range(n_bandits):
                if len(rewards.shape) == 3:
                    totalrewards[a, k] += rewards[_states[a], actions[a], a]
                    learn_totalrewards[a, k] += rewards[_learn_states[a], learn_actions[a], a]
                else:
                    totalrewards[a, k] += rewards[_states[a], a]
                    learn_totalrewards[a, k] += rewards[_learn_states[a], a]
                states[a] = np.random.choice(n_states, p=transitions[_states[a], :, actions[a], a])
                if actions[a] == learn_actions[a] and _states[a] == _learn_states[a]:
                    learn_states[a] = np.copy(states[a])
                else:
                    learn_states[a] = np.random.choice(n_states, p=transitions[_learn_states[a], :, learn_actions[a], a])
                counts[_learn_states[a], learn_states[a], learn_actions[a], a] += 1
        objectives[:, k] = compute_risk(totalrewards[:, k], thresholds, u_type, u_order, n_bandits)
        learn_objectives[:, k] = compute_risk(learn_totalrewards[:, k], thresholds, u_type, u_order, n_bandits)

    return totalrewards, objectives, learn_totalrewards, learn_objectives, counts


def Process_Random(n_episodes, n_steps, n_states, n_bandits, n_choices, thresholds, rewards, transitions,
                   initial_states, u_type, u_order):

    ##################################################### Process
    totalrewards = np.zeros((n_bandits, n_episodes))
    objectives = np.zeros((n_bandits, n_episodes))
    counts = np.zeros((n_states, n_states, 2, n_bandits))
    for k in range(n_episodes):
        states = initial_states.copy()
        for t in range(n_steps):
            _states = np.copy(states)
            selected_indices = random.sample(range(n_bandits), n_choices)
            actions = [1 if i in selected_indices else 0 for i in range(n_bandits)]
            for a in range(n_bandits):
                if len(rewards.shape) == 3:
                    totalrewards[a, k] += rewards[_states[a], actions[a], a]
                else:
                    totalrewards[a, k] += rewards[_states[a], a]
                states[a] = np.random.choice(n_states, p=transitions[_states[a], :, actions[a], a])
                counts[_states[a], states[a], actions[a], a] += 1
        objectives[:, k] = compute_risk(totalrewards[:, k], thresholds, u_type, u_order, n_bandits)

    return np.around(totalrewards, 2), np.around(objectives, 2), counts


def Process_Greedy(n_episodes, n_steps, n_states, n_bandits, n_choices, thresholds, rewards, transitions,
                   initial_states, u_type, u_order):

    ##################################################### Process
    totalrewards = np.zeros((n_bandits, n_episodes))
    objectives = np.zeros((n_bandits, n_episodes))
    counts = np.zeros((n_states, n_states, 2, n_bandits))
    for k in range(n_episodes):
        states = initial_states.copy()
        for t in range(n_steps):
            rew_vec = np.zeros(n_bandits)
            for a2 in range(n_bandits):
                rew_vec[a2] = rewards[states[a2], a2]
            _states = np.copy(states)
            top_indices = np.argsort(rew_vec)[-n_choices:]
            actions = np.zeros_like(rew_vec, dtype=np.int32)
            actions[top_indices] = 1
            for a in range(n_bandits):
                if len(rewards.shape) == 3:
                    totalrewards[a, k] += rewards[_states[a], actions[a], a]
                else:
                    totalrewards[a, k] += rewards[_states[a], a]
                states[a] = np.random.choice(n_states, p=transitions[_states[a], :, actions[a], a])
                counts[_states[a], states[a], actions[a], a] += 1
        objectives[:, k] = compute_risk(totalrewards[:, k], thresholds, u_type, u_order, n_bandits)

    return np.around(totalrewards, 2), np.around(objectives, 2), counts


def Process_Myopic(n_episodes, n_steps, n_states, n_bandits, n_choices, thresholds, rewards, transitions,
                   initial_states, u_type, u_order):

    ##################################################### Process
    totalrewards = np.zeros((n_bandits, n_episodes))
    objectives = np.zeros((n_bandits, n_episodes))
    counts = np.zeros((n_states, n_states, 2, n_bandits))
    for k in range(n_episodes):
        states = initial_states.copy()
        for t in range(n_steps):
            combs = list(combinations(range(n_bandits), n_choices))
            imm_rews = np.zeros(len(combs))
            for i, comb in enumerate(combs):
                rew = 0
                for a in range(n_bandits):
                    if a in comb:
                        rew += rewards[states[a], 1, a]
                    else:
                        rew += rewards[states[a], 0, a]
                imm_rews[i] = rew
            max_index = np.argmax(imm_rews)
            best_comb = combs[max_index]
            actions = np.array([(1 if i in best_comb else 0) for i in range(n_bandits)])
            _states = np.copy(states)
            for a in range(n_bandits):
                totalrewards[a, k] += rewards[_states[a], actions[a], a]
                states[a] = np.random.choice(n_states, p=transitions[_states[a], :, actions[a], a])
                counts[_states[a], states[a], actions[a], a] += 1
        objectives[:, k] = compute_risk(totalrewards[:, k], thresholds, u_type, u_order, n_bandits)

    return np.around(totalrewards, 2), np.around(objectives, 2), counts
