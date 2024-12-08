import random
import numpy as np


def compute_utility(total_reward, threshold, u_type, u_order):
    if u_type == 1:
        if total_reward - threshold >= 0:
            return 1
        else:
            return 0
    elif u_type == 2:
        return 1 - threshold**(- 1/u_order) * (np.maximum(0, threshold - total_reward))**(1/u_order)
    else:
        return (1 + np.exp(-u_order * (1 - threshold))) / (1 + np.exp(-u_order * (total_reward - threshold)))


def process_neutral_whittle(nWhittle, n_iterations, n_steps, n_states, n_arms, n_choices, threshold, rewards, transitions, initial_states, u_type, u_order):

    ##################################################### Process
    totalrewards = np.zeros((n_arms, n_iterations))
    objectives = np.zeros((n_arms, n_iterations))
    for k in range(n_iterations):
        states = initial_states.copy()
        for t in range(n_steps):
            actions = nWhittle.take_action(n_choices, states, t)
            for a in range(n_arms):
                totalrewards[a, k] += rewards[states[a], a]
                states[a] = np.random.choice(n_states, p=transitions[states[a], :, actions[a], a])
        for a in range(n_arms):
            objectives[a, k] = compute_utility(totalrewards[a, k], threshold, u_type, u_order)

    return totalrewards, objectives


def process_neutral_whittle_learning(nWhittle, nWhittle_learn, n_iterations, n_steps, n_states, n_arms, n_choices, threshold, rewards, transitions, initial_states, u_type, u_order):
    totalrewards = np.zeros((n_arms, n_iterations))
    objectives = np.zeros((n_arms, n_iterations))
    learn_totalrewards = np.zeros((n_arms, n_iterations))
    learn_objectives = np.zeros((n_arms, n_iterations))
    counts = np.zeros((n_states, n_states, 2, n_arms))
    for k in range(n_iterations):
        states = initial_states.copy()
        learn_states = initial_states.copy()
        for t in range(n_steps):
            actions = nWhittle.take_action(n_choices, states, t)
            learn_actions = nWhittle_learn.take_action(n_choices, learn_states, t)
            _learn_states = np.copy(learn_states)
            for a in range(n_arms):
                totalrewards[a, k] += rewards[states[a], a]
                learn_totalrewards[a, k] += rewards[learn_states[a], a]
                states[a] = np.random.choice(n_states, p=transitions[states[a], :, actions[a], a])
                learn_states[a] = np.random.choice(n_states, p=transitions[learn_states[a], :, learn_actions[a], a])
                counts[_learn_states[a], learn_states[a], learn_actions[a], a] += 1
        for a in range(n_arms):
            objectives[a, k] = compute_utility(totalrewards[a, k], threshold, u_type, u_order)
            learn_objectives[a, k] = compute_utility(learn_totalrewards[a, k], threshold, u_type, u_order)

    return totalrewards, objectives, learn_totalrewards, learn_objectives, counts


def process_riskaware_whittle(rWhittle, n_iterations, n_steps, n_states, n_arms, n_choices, threshold, rewards, transitions, initial_states, u_type, u_order):

    ##################################################### Process
    totalrewards = np.zeros((n_arms, n_iterations))
    objectives = np.zeros((n_arms, n_iterations))
    for k in range(n_iterations):
        lifted = np.zeros(n_arms, dtype=np.int32)
        states = initial_states.copy()
        for t in range(n_steps):
            actions = rWhittle.take_action(n_choices, lifted, states, t)
            for a in range(n_arms):
                totalrewards[a, k] += rewards[states[a], a]
                lifted[a] = max(0, min(rWhittle.n_augment[a]-1, lifted[a] + states[a]))
                states[a] = np.random.choice(n_states, p=transitions[states[a], :, actions[a], a])
        for a in range(n_arms):
            objectives[a, k] = compute_utility(totalrewards[a, k], threshold, u_type, u_order)

    return totalrewards, objectives


def process_riskaware_whittle_learning(rWhittle, rWhittle_learn, n_iterations, n_steps, n_states, n_arms, n_choices, threshold, rewards, transitions, initial_states, u_type, u_order):
    totalrewards = np.zeros((n_arms, n_iterations))
    objectives = np.zeros((n_arms, n_iterations))
    learn_totalrewards = np.zeros((n_arms, n_iterations))
    learn_objectives = np.zeros((n_arms, n_iterations))
    counts = np.zeros((n_states, n_states, 2, n_arms))
    for k in range(n_iterations):
        lifted = np.zeros(n_arms, dtype=np.int32)
        states = initial_states.copy()
        learn_lifted = np.zeros(n_arms, dtype=np.int32)
        learn_states = initial_states.copy()
        for t in range(n_steps):
            actions = rWhittle.take_action(n_choices, lifted, states, t)
            learn_actions = rWhittle_learn.take_action(n_choices, learn_lifted, learn_states, t)
            _learn_states = np.copy(learn_states)
            for a in range(n_arms):
                totalrewards[a, k] += rewards[states[a], a]
                lifted[a] = max(0, min(rWhittle.n_augment[a]-1, lifted[a] + states[a]))
                states[a] = np.random.choice(n_states, p=transitions[states[a], :, actions[a], a])
                learn_totalrewards[a, k] += rewards[learn_states[a], a]
                learn_lifted[a] = max(0, min(rWhittle_learn.n_augment[a]-1, learn_lifted[a] + learn_states[a]))
                learn_states[a] = np.random.choice(n_states, p=transitions[learn_states[a], :, learn_actions[a], a])
                counts[_learn_states[a], learn_states[a], learn_actions[a], a] += 1
        for a in range(n_arms):
            objectives[a, k] = compute_utility(totalrewards[a, k], threshold, u_type, u_order)
            learn_objectives[a, k] = compute_utility(learn_totalrewards[a, k], threshold, u_type, u_order)

    return totalrewards, objectives, learn_totalrewards, learn_objectives, counts


def process_Random(n_iterations, n_steps, n_states, n_arms, n_choices, threshold, rewards, transitions, initial_states, u_type, u_order):

    ##################################################### Process
    totalrewards = np.zeros((n_arms, n_iterations))
    objectives = np.zeros((n_arms, n_iterations))
    for k in range(n_iterations):
        states = initial_states.copy()
        for _ in range(n_steps):
            selected_indices = random.sample(range(n_arms), n_choices)
            actions = [1 if i in selected_indices else 0 for i in range(n_arms)]
            for a in range(n_arms):
                totalrewards[a, k] += rewards[states[a], a]
                states[a] = np.random.choice(n_states, p=transitions[states[a], :, actions[a], a])
        for a in range(n_arms):
            objectives[a, k] = compute_utility(totalrewards[a, k], threshold, u_type, u_order)

    return totalrewards, objectives, _
