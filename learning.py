from scipy.stats import dirichlet
import joblib
from Markov import *
from whittle import *
from processes import *
from multiprocessing import Pool, cpu_count
import numpy as np
import joblib
import time


def Process_LearnRAWIP_iteration(i, l_episodes, n_episodes, n_steps, n_states, n_arms, n_choices, thresholds,
                      t_type, t_increasing, method, tru_rew, tru_dyn, initial_states, u_type, u_order, max_wi):
    n_trials_safety = n_states * n_steps

    # Initialization
    print(f"Iteration {i} starts ...")
    start_time = time.time()
    results = {
        "plan_rewards": np.zeros((l_episodes, n_arms)),
        "plan_objectives": np.zeros((l_episodes, n_arms)),
        "learn_rewards": np.zeros((l_episodes, n_arms)),
        "learn_objectives": np.zeros((l_episodes, n_arms)),
        "learn_indexerrors": np.zeros((l_episodes, n_arms)),
        "learn_transitionerrors": np.ones((l_episodes, n_arms)),
    }

    PlanW = RiskAwareWhittle(n_states, n_arms, tru_rew, tru_dyn, n_steps, u_type, u_order, thresholds)
    PlanW.get_whittle_indices(computation_type=method, params=[0, max_wi], n_trials=n_trials_safety)
    plan_indices = PlanW.w_indices

    # Set up learning dynamics
    if t_type < 10:
        probs = np.array([np.round(np.random.uniform(0.1 / n_states, 1 / n_states), 2) for _ in range(n_arms)])
        Mest = MarkovDynamics(n_arms, n_states, probs, t_type, t_increasing)
        est_transitions = Mest.transitions
        LearnW = RiskAwareWhittle(n_states, n_arms, tru_rew, Mest.transitions, n_steps, u_type, u_order, thresholds)
    else:
        est_transitions = np.zeros((n_states, n_states, 2, n_arms))
        for a in range(n_arms):
            for s1 in range(n_states):
                for act in range(2):
                    est_transitions[s1, :, act, a] = dirichlet.rvs(np.ones(n_states))
        LearnW = RiskAwareWhittle(n_states, n_arms, tru_rew, est_transitions, n_steps, u_type, u_order, thresholds)

    LearnW.get_whittle_indices(computation_type=method, params=[0, max_wi], n_trials=n_trials_safety)
    counts = np.ones((n_states, n_states, 2, n_arms))

    for l in range(l_episodes):
        plan_totalrewards, plan_objectives, learn_totalrewards, learn_objectives, cnts = \
            Process_LearnSafeRB(PlanW, LearnW, n_episodes, n_steps, n_states, n_arms,
                                n_choices, thresholds, tru_rew, tru_dyn, initial_states, u_type, u_order)
        counts += cnts

        # Update transitions
        est_transitions = np.zeros((n_states, n_states, 2, n_arms))
        for a in range(n_arms):
            for s1 in range(n_states):
                for act in range(2):
                    est_transitions[s1, :, act, a] = dirichlet.rvs(counts[s1, :, act, a])

        SafeW = RiskAwareWhittle(n_states, n_arms, tru_rew, est_transitions, n_steps, u_type, u_order, thresholds)
        SafeW.get_whittle_indices(computation_type=method, params=[0, max_wi], n_trials=n_trials_safety)
        sw_indices = SafeW.w_indices

        for a in range(n_arms):
            results["learn_transitionerrors"][l, a] = np.max(np.abs(est_transitions[:, :, :, a] - tru_dyn[:, :, :, a]))
            results["learn_indexerrors"][l, a] = np.max(np.abs(sw_indices[a] - plan_indices[a]))
            results["plan_rewards"][l, a] = np.round(np.mean(plan_totalrewards[a, :]), 2)
            results["plan_objectives"][l, a] = np.round(np.mean(plan_objectives[a, :]), 2)
            results["learn_rewards"][l, a] = np.round(np.mean(learn_totalrewards[a, :]), 2)
            results["learn_objectives"][l, a] = np.round(np.mean(learn_objectives[a, :]), 2)

    print(f"Iteration {i} end with duration: {time.time() - start_time}")
    return results


def Process_LearnRAWIP(n_iterations, l_episodes, n_episodes, n_steps, n_states, n_arms, n_choices, thresholds,
                          t_type, t_increasing, method, tru_rew, tru_dyn, initial_states, u_type, u_order, save_data, max_wi):
    """
    Processes multiple iterations of Safe Learning without multiprocessing.
    """
    # Storage for results
    all_plan_rewards = np.zeros((n_iterations, l_episodes, n_arms))
    all_plan_objectives = np.zeros((n_iterations, l_episodes, n_arms))
    all_learn_rewards = np.zeros((n_iterations, l_episodes, n_arms))
    all_learn_objectives = np.zeros((n_iterations, l_episodes, n_arms))
    all_learn_indexerrors = np.zeros((n_iterations, l_episodes, n_arms))
    all_learn_transitionerrors = np.ones((n_iterations, l_episodes, n_arms))

    print(f"Starting {n_iterations} iterations of learning process.")

    # Sequentially process each iteration
    for n in range(n_iterations):
        print(f"Processing iteration {n + 1} of {n_iterations}...")

        # Call the `process_iteration` function for this iteration
        results = Process_LearnRAWIP_iteration(
            n, l_episodes=l_episodes, n_episodes=n_episodes, n_steps=n_steps,
            n_states=n_states, n_arms=n_arms, n_choices=n_choices, thresholds=thresholds, t_type=t_type,
            t_increasing=t_increasing, method=method, tru_rew=tru_rew, tru_dyn=tru_dyn,
            initial_states=initial_states, u_type=u_type, u_order=u_order, max_wi=max_wi
        )

        # Store the results for this iteration
        all_plan_rewards[n] = results["plan_rewards"]
        all_plan_objectives[n] = results["plan_objectives"]
        all_learn_rewards[n] = results["learn_rewards"]
        all_learn_objectives[n] = results["learn_objectives"]
        all_learn_indexerrors[n] = results["learn_indexerrors"]
        all_learn_transitionerrors[n] = results["learn_transitionerrors"]

    # Save results if required
    if save_data:
        filename = f'./output-learn-finite/rawip_{n_episodes}{n_steps}{n_states}{t_type}{u_type}{n_choices}{thresholds[0]}{u_type}{u_order}.joblib'
        joblib.dump(
            [all_learn_transitionerrors, all_learn_indexerrors, all_learn_rewards, all_learn_objectives,
             all_plan_rewards, all_plan_objectives],
            filename
        )
        print(f"Data saved to {filename}")

    return all_learn_transitionerrors, all_learn_indexerrors, all_learn_rewards, all_learn_objectives, all_plan_rewards, all_plan_objectives


def ProcessMulti_LearnRAWIP(n_iterations, l_episodes, n_episodes, n_steps, n_states, n_arms, n_choices, thresholds,
                               t_type, t_increasing, method, tru_rew, tru_dyn, initial_states, u_type, u_order, save_data, max_wi):
    num_workers = cpu_count() - 1

    # Define arguments for each iteration
    args = [
        (i, l_episodes, n_episodes, n_steps, n_states, n_arms, n_choices, thresholds,
         t_type, t_increasing, method, tru_rew, tru_dyn, initial_states, u_type, u_order, max_wi)
        for i in range(n_iterations)
    ]

    # Use multiprocessing pool
    with Pool(num_workers) as pool:
        results = pool.starmap(Process_LearnRAWIP_iteration, args)

    # Aggregate results
    all_learn_transitionerrors = np.stack([res["learn_transitionerrors"] for res in results])
    all_learn_indexerrors = np.stack([res["learn_indexerrors"] for res in results])
    all_learn_rewards = np.stack([res["learn_rewards"] for res in results])
    all_learn_objectives = np.stack([res["learn_objectives"] for res in results])
    all_plan_rewards = np.stack([res["plan_rewards"] for res in results])
    all_plan_objectives = np.stack([res["plan_objectives"] for res in results])

    if save_data:
        joblib.dump([all_learn_transitionerrors, all_learn_indexerrors, all_learn_rewards, all_learn_objectives, all_plan_rewards, all_plan_objectives],
                    f'./output-learn-finite/rawip_{n_episodes}{n_steps}{n_states}{t_type}{u_type}{n_choices}{thresholds[0]}{u_type}{u_order}.joblib')

    return all_learn_transitionerrors, all_learn_indexerrors, all_learn_rewards, all_learn_objectives, all_plan_rewards, all_plan_objectives


def Process_LearnWIP_iteration(i, l_episodes, n_episodes, n_steps, n_states, n_arms, n_choices, thresholds,
                                t_type, t_increasing, method, tru_rew, tru_dyn, initial_states, u_type, u_order, max_wi):
    n_trials_safety = n_states * n_steps

    # Initialization
    print(f"Iteration {i} starts ...")
    start_time = time.time()
    results = {
        "plan_rewards": np.zeros((l_episodes, n_arms)),
        "plan_objectives": np.zeros((l_episodes, n_arms)),
        "learn_rewards": np.zeros((l_episodes, n_arms)),
        "learn_objectives": np.zeros((l_episodes, n_arms)),
        "learn_indexerrors": np.zeros((l_episodes, n_arms)),
        "learn_transitionerrors": np.ones((l_episodes, n_arms)),
        "learn_probs": np.ones((l_episodes, n_arms))
    }

    PlanW = Whittle(n_states, n_arms, tru_rew, tru_dyn, n_steps)
    PlanW.get_whittle_indices(computation_type=method, params=[0, max_wi], n_trials=n_trials_safety)
    plan_indices = PlanW.w_indices

    # Learning dynamics setup
    for l in range(l_episodes):
        if t_type < 10:
            results["learn_probs"][l, :] = np.array([np.round(np.random.uniform(0.1 / n_states, 1 / n_states), 2)
                                                     for _ in range(n_arms)])
            Mest = MarkovDynamics(n_arms, n_states, results["learn_probs"][l, :], t_type, t_increasing)
            est_transitions = Mest.transitions
            LearnW = Whittle(n_states, n_arms, tru_rew, Mest.transitions, n_steps)
        else:
            est_transitions = np.zeros((n_states, n_states, 2, n_arms))
            for a in range(n_arms):
                for s1 in range(n_states):
                    for act in range(2):
                        est_transitions[s1, :, act, a] = dirichlet.rvs(np.ones(n_states))
            LearnW = Whittle(n_states, n_arms, tru_rew, est_transitions, n_steps)

        LearnW.get_whittle_indices(computation_type=method, params=[0, max_wi], n_trials=n_trials_safety)
        w_indices = LearnW.w_indices
        plan_totalrewards, plan_objectives, learn_totalrewards, learn_objectives, cnts = \
            Process_LearnWhtlRB(PlanW, LearnW, n_episodes, n_steps, n_states, n_arms, n_choices, thresholds,
                                tru_rew, tru_dyn, initial_states, u_type, u_order)

        # Record results for each arm
        for a in range(n_arms):
            results["learn_transitionerrors"][l, a] = np.max(np.abs(est_transitions[:, :, :, a] - tru_dyn[:, :, :, a]))
            results["learn_indexerrors"][l, a] = np.max(np.abs(w_indices[a] - plan_indices[a]))
            results["plan_rewards"][l, a] = np.round(np.mean(plan_totalrewards[a, :]), 2)
            results["plan_objectives"][l, a] = np.round(np.mean(plan_objectives[a, :]), 2)
            results["learn_rewards"][l, a] = np.round(np.mean(learn_totalrewards[a, :]), 2)
            results["learn_objectives"][l, a] = np.round(np.mean(learn_objectives[a, :]), 2)

    print(f"Iteration {i} end with duration: {time.time() - start_time}")
    return results


def Process_LearnWIP(n_iterations, l_episodes, n_episodes, n_steps, n_states, n_arms, n_choices, thresholds,
                      t_type, t_increasing, method, tru_rew, tru_dyn, initial_states, u_type, u_order, save_data, max_wi):
    # Storage for aggregated results
    all_plan_rewards = np.zeros((n_iterations, l_episodes, n_arms))
    all_plan_objectives = np.zeros((n_iterations, l_episodes, n_arms))
    all_learn_rewards = np.zeros((n_iterations, l_episodes, n_arms))
    all_learn_objectives = np.zeros((n_iterations, l_episodes, n_arms))
    all_learn_indexerrors = np.zeros((n_iterations, l_episodes, n_arms))
    all_learn_probs = np.ones((n_iterations, l_episodes, n_arms))
    all_learn_transitionerrors = np.ones((n_iterations, l_episodes, n_arms))

    for n in range(n_iterations):
        print(f"Processing iteration {n + 1} of {n_iterations}...")
        results = Process_LearnWIP_iteration(
            n, l_episodes=l_episodes, n_episodes=n_episodes, n_steps=n_steps, n_states=n_states, n_arms=n_arms,
            n_choices=n_choices, thresholds=thresholds, t_type=t_type, t_increasing=t_increasing, method=method,
            tru_rew=tru_rew, tru_dyn=tru_dyn, initial_states=initial_states, u_type=u_type, u_order=u_order, max_wi=max_wi
        )

        all_plan_rewards[n] = results["plan_rewards"]
        all_plan_objectives[n] = results["plan_objectives"]
        all_learn_rewards[n] = results["learn_rewards"]
        all_learn_objectives[n] = results["learn_objectives"]
        all_learn_indexerrors[n] = results["learn_indexerrors"]
        all_learn_transitionerrors[n] = results["learn_transitionerrors"]
        all_learn_probs[n] = results["learn_probs"]

    if save_data:
        filename = f'./output-learn-finite/wip_{n_episodes}{n_steps}{n_states}{t_type}{u_type}{n_choices}{thresholds[0]}{u_type}{u_order}.joblib'
        joblib.dump([all_learn_transitionerrors, all_learn_indexerrors, all_learn_rewards, all_learn_objectives,
                     all_plan_rewards, all_plan_objectives],
                    filename)
        print(f"Data saved to {filename}")

    return all_learn_transitionerrors, all_learn_indexerrors, all_learn_rewards, all_learn_objectives, all_plan_rewards, all_plan_objectives


def ProcessMulti_LearnWIP(n_iterations, l_episodes, n_episodes, n_steps, n_states, n_arms, n_choices, thresholds,
                           t_type, t_increasing, method, tru_rew, tru_dyn, initial_states, u_type, u_order, save_data, max_wi):
    num_workers = cpu_count() - 1

    # Define arguments for each iteration
    args = [
        (i, l_episodes, n_episodes, n_steps, n_states, n_arms, n_choices, thresholds,
         t_type, t_increasing, method, tru_rew, tru_dyn, initial_states, u_type, u_order, max_wi)
        for i in range(n_iterations)
    ]

    # Use multiprocessing
    with Pool(num_workers) as pool:
        results = pool.starmap(Process_LearnWIP_iteration, args)

    # Aggregate results
    all_learn_transitionerrors = np.stack([res["learn_transitionerrors"] for res in results])
    all_learn_indexerrors = np.stack([res["learn_indexerrors"] for res in results])
    all_learn_rewards = np.stack([res["learn_rewards"] for res in results])
    all_learn_objectives = np.stack([res["learn_objectives"] for res in results])
    all_plan_rewards = np.stack([res["plan_rewards"] for res in results])
    all_plan_objectives = np.stack([res["plan_objectives"] for res in results])

    if save_data:
        filename = f'./output-learn-finite/wip_{n_episodes}{n_steps}{n_states}{t_type}{u_type}{n_choices}{thresholds[0]}{u_type}{u_order}.joblib'
        joblib.dump([all_learn_transitionerrors, all_learn_indexerrors, all_learn_rewards, all_learn_objectives,
                     all_plan_rewards, all_plan_objectives],
                    filename)

    return all_learn_transitionerrors, all_learn_indexerrors, all_learn_rewards, all_learn_objectives, all_plan_rewards, all_plan_objectives


def Process_LearnRUWIP(n_iterations, l_episodes, n_episodes, n_steps, n_states, n_arms, n_choices, thresholds,
                      t_type, t_increasing, method, tru_rew, tru_dyn, initial_states, u_type, u_order, save_data, max_wi):
    # Storage for aggregated results
    all_plan_rewards = np.zeros((n_iterations, l_episodes, n_arms))
    all_plan_objectives = np.zeros((n_iterations, l_episodes, n_arms))
    all_learn_rewards = np.zeros((n_iterations, l_episodes, n_arms))
    all_learn_objectives = np.zeros((n_iterations, l_episodes, n_arms))
    all_learn_indexerrors = np.zeros((n_iterations, l_episodes, n_arms))
    all_learn_probs = np.ones((n_iterations, l_episodes, n_arms))
    all_learn_transitionerrors = np.ones((n_iterations, l_episodes, n_arms))

    for n in range(n_iterations):
        print(f"Processing iteration {n + 1} of {n_iterations}...")
        results = Process_LearnWIP_iteration(
            n, l_episodes=l_episodes, n_episodes=n_episodes, n_steps=n_steps, n_states=n_states, n_arms=n_arms,
            n_choices=n_choices, thresholds=thresholds, t_type=t_type, t_increasing=t_increasing, method=method,
            tru_rew=tru_rew, tru_dyn=tru_dyn, initial_states=initial_states, u_type=u_type, u_order=u_order, max_wi=max_wi
        )

        all_plan_rewards[n] = results["plan_rewards"]
        all_plan_objectives[n] = results["plan_objectives"]
        all_learn_rewards[n] = results["learn_rewards"]
        all_learn_objectives[n] = results["learn_objectives"]
        all_learn_indexerrors[n] = results["learn_indexerrors"]
        all_learn_transitionerrors[n] = results["learn_transitionerrors"]
        all_learn_probs[n] = results["learn_probs"]

    if save_data:
        filename = f'./output-learn-finite/ruwip_{n_episodes}{n_steps}{n_states}{t_type}{u_type}{n_choices}{thresholds[0]}{u_type}{u_order}.joblib'
        joblib.dump([all_learn_transitionerrors, all_learn_indexerrors, all_learn_rewards, all_learn_objectives,
                     all_plan_rewards, all_plan_objectives],
                    filename)
        print(f"Data saved to {filename}")

    return all_learn_transitionerrors, all_learn_indexerrors, all_learn_rewards, all_learn_objectives, all_plan_rewards, all_plan_objectives


def ProcessMulti_LearnRUWIP(n_iterations, l_episodes, n_episodes, n_steps, n_states, n_arms, n_choices, thresholds,
                           t_type, t_increasing, method, tru_rew, tru_dyn, initial_states, u_type, u_order, save_data, max_wi):
    num_workers = cpu_count() - 1

    # Define arguments for each iteration
    args = [
        (i, l_episodes, n_episodes, n_steps, n_states, n_arms, n_choices, thresholds,
         t_type, t_increasing, method, tru_rew, tru_dyn, initial_states, u_type, u_order, max_wi)
        for i in range(n_iterations)
    ]

    # Use multiprocessing
    with Pool(num_workers) as pool:
        results = pool.starmap(Process_LearnWIP_iteration, args)

    # Aggregate results
    all_learn_transitionerrors = np.stack([res["learn_transitionerrors"] for res in results])
    all_learn_indexerrors = np.stack([res["learn_indexerrors"] for res in results])
    all_learn_rewards = np.stack([res["learn_rewards"] for res in results])
    all_learn_objectives = np.stack([res["learn_objectives"] for res in results])
    all_plan_rewards = np.stack([res["plan_rewards"] for res in results])
    all_plan_objectives = np.stack([res["plan_objectives"] for res in results])

    if save_data:
        filename = f'./output-learn-finite/ruwip_{n_episodes}{n_steps}{n_states}{t_type}{u_type}{n_choices}{thresholds[0]}{u_type}{u_order}.joblib'
        joblib.dump([all_learn_transitionerrors, all_learn_indexerrors, all_learn_rewards, all_learn_objectives,
                     all_plan_rewards, all_plan_objectives],
                    filename)

    return all_learn_transitionerrors, all_learn_indexerrors, all_learn_rewards, all_learn_objectives, all_plan_rewards, all_plan_objectives

