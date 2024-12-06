from scipy.stats import dirichlet
import joblib
from Markov import *
from whittle import *
from processes import *
from multiprocessing import Pool, cpu_count
import numpy as np
import joblib
import time


def Process_LearnSafeTSRB_iteration(i, l_episodes, n_episodes, n_steps, n_states, n_arms, n_choices, 
                                    thresholds, method, tru_rew, tru_dyn, initial_states, u_type, u_order, 
                                    wip_params, PlanW, n_trials_safety):

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

    # Set up learning dynamics
    est_transitions = np.zeros((n_states, n_states, 2, n_arms))
    for a in range(n_arms):
        for s1 in range(n_states):
            for act in range(2):
                est_transitions[s1, :, act, a] = dirichlet.rvs(np.ones(n_states))
    LearnW = RiskAwareWhittle(n_states, n_arms, tru_rew, est_transitions, n_steps, u_type, u_order, thresholds)
    LearnW.get_whittle_indices(computation_type=method, params=wip_params, n_trials=n_trials_safety)
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
        LearnW = RiskAwareWhittle(n_states, n_arms, tru_rew, est_transitions, n_steps, u_type, u_order, thresholds)
        LearnW.get_whittle_indices(computation_type=method, params=wip_params, n_trials=n_trials_safety)

        for a in range(n_arms):
            results["learn_transitionerrors"][l, a] = np.max(np.abs(est_transitions[:, :, :, a] - tru_dyn[:, :, :, a]))
            results["learn_indexerrors"][l, a] = np.max(np.abs(LearnW.w_indices[a] - PlanW.w_indices[a]))
            results["plan_rewards"][l, a] = np.round(np.mean(plan_totalrewards[a, :]), 2)
            results["plan_objectives"][l, a] = np.round(np.mean(plan_objectives[a, :]), 2)
            results["learn_rewards"][l, a] = np.round(np.mean(learn_totalrewards[a, :]), 2)
            results["learn_objectives"][l, a] = np.round(np.mean(learn_objectives[a, :]), 2)

    print(f"Iteration {i} end with duration: {time.time() - start_time}")
    return results


def Process_LearnSafeTSRB(n_iterations, l_episodes, n_episodes, n_steps, n_states, n_arms, n_choices, 
                          thresholds, method, tru_rew, tru_dyn, initial_states, u_type, 
                          u_order, save_data, wip_params, wip_trials):
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

    n_trials_safety = wip_trials
    PlanW = RiskAwareWhittle(n_states, n_arms, tru_rew, tru_dyn, n_steps, u_type, u_order, thresholds)
    PlanW.get_whittle_indices(computation_type=method, params=wip_params, n_trials=n_trials_safety)

    # Sequentially process each iteration
    for n in range(n_iterations):
        print(f"Iteration {n} starts ...")

        # Call the `process_iteration` function for this iteration
        results = Process_LearnSafeTSRB_iteration(
            n, l_episodes=l_episodes, n_episodes=n_episodes, n_steps=n_steps,
            n_states=n_states, n_arms=n_arms, n_choices=n_choices, thresholds=thresholds, 
            method=method, tru_rew=tru_rew, tru_dyn=tru_dyn,
            initial_states=initial_states, u_type=u_type, u_order=u_order, wip_params=wip_params, PlanW=PlanW,
            n_trials_safety=n_trials_safety
        )

        # Store the results for this iteration
        all_plan_rewards[n] = results["plan_rewards"]
        all_plan_objectives[n] = results["plan_objectives"]
        all_learn_rewards[n] = results["learn_rewards"]
        all_learn_objectives[n] = results["learn_objectives"]
        all_learn_indexerrors[n] = results["learn_indexerrors"]
        all_learn_transitionerrors[n] = results["learn_transitionerrors"]

        print(f"Iteration {n} ends ...")

    # Save results if required
    if save_data:
        filename = f'./output-learn-finite/safetsrb_ne{n_episodes}_nt{n_steps}_ns{n_states}_na{n_arms}_tt{t_type}_ut{u_type}_nc{n_choices}_th{thresholds[0]}_ut{u_type}_uo{u_order}.joblib'
        joblib.dump(
            [all_learn_transitionerrors, all_learn_indexerrors, all_learn_rewards, all_learn_objectives,
             all_plan_rewards, all_plan_objectives],
            filename
        )
        print(f"Data saved to {filename}")

    return all_learn_transitionerrors, all_learn_indexerrors, all_learn_rewards, all_learn_objectives, all_plan_rewards, all_plan_objectives


def ProcessMulti_LearnSafeTSRB(n_iterations, l_episodes, n_episodes, n_steps, n_states, n_arms, n_choices, 
                               thresholds, method, tru_rew, tru_dyn, initial_states, 
                               u_type, u_order, save_data, wip_params, wip_trials):
    num_workers = cpu_count() - 1

    PlanW = RiskAwareWhittle(n_states, n_arms, tru_rew, tru_dyn, n_steps, u_type, u_order, thresholds)
    PlanW.get_whittle_indices(computation_type=method, params=wip_params, n_trials=wip_trials)

    # Define arguments for each iteration
    args = [
        (i, l_episodes, n_episodes, n_steps, n_states, n_arms, n_choices, thresholds,
         method, tru_rew, tru_dyn, initial_states, u_type, u_order, wip_params,
         PlanW, wip_trials) 
         for i in range(n_iterations)
    ]

    # Use multiprocessing pool
    with Pool(num_workers) as pool:
        results = pool.starmap(Process_LearnSafeTSRB_iteration, args)

    # Aggregate results
    all_learn_transitionerrors = np.stack([res["learn_transitionerrors"] for res in results])
    all_learn_indexerrors = np.stack([res["learn_indexerrors"] for res in results])
    all_learn_rewards = np.stack([res["learn_rewards"] for res in results])
    all_learn_objectives = np.stack([res["learn_objectives"] for res in results])
    all_plan_rewards = np.stack([res["plan_rewards"] for res in results])
    all_plan_objectives = np.stack([res["plan_objectives"] for res in results])

    if save_data:
        joblib.dump([all_learn_transitionerrors, all_learn_indexerrors, all_learn_rewards, all_learn_objectives, all_plan_rewards, all_plan_objectives],
                    f'./output-learn-finite/safetsrb_ne{n_episodes}_nt{n_steps}_ns{n_states}_na{n_arms}_tt{t_type}_ut{u_type}_nc{n_choices}_th{thresholds[0]}_ut{u_type}_uo{u_order}.joblib')

    return all_learn_transitionerrors, all_learn_indexerrors, all_learn_rewards, all_learn_objectives, all_plan_rewards, all_plan_objectives


def Process_LearnTSRB_iteration(i, l_episodes, n_episodes, n_steps, n_states, n_arms, n_choices, thresholds,
                                t_type, t_increasing, method, tru_rew, tru_dyn, initial_states, u_type, 
                                u_order, wip_params, PlanW, n_trials_safety):

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

        LearnW.get_whittle_indices(computation_type=method, params=wip_params, n_trials=n_trials_safety)
        w_indices = LearnW.w_indices
        plan_totalrewards, plan_objectives, learn_totalrewards, learn_objectives, cnts = \
            Process_LearnWhtlRB(PlanW, LearnW, n_episodes, n_steps, n_states, n_arms, n_choices, thresholds,
                                tru_rew, tru_dyn, initial_states, u_type, u_order)

        # Record results for each arm
        for a in range(n_arms):
            results["learn_transitionerrors"][l, a] = np.max(np.abs(est_transitions[:, :, :, a] - tru_dyn[:, :, :, a]))
            results["learn_indexerrors"][l, a] = np.max(np.abs(w_indices[a] - PlanW.w_indices[a]))
            results["plan_rewards"][l, a] = np.round(np.mean(plan_totalrewards[a, :]), 2)
            results["plan_objectives"][l, a] = np.round(np.mean(plan_objectives[a, :]), 2)
            results["learn_rewards"][l, a] = np.round(np.mean(learn_totalrewards[a, :]), 2)
            results["learn_objectives"][l, a] = np.round(np.mean(learn_objectives[a, :]), 2)

    print(f"Iteration {i} end with duration: {time.time() - start_time}")
    return results


def Process_LearnTSRB(n_iterations, l_episodes, n_episodes, n_steps, n_states, n_arms, n_choices, thresholds,
                      t_type, t_increasing, method, tru_rew, tru_dyn, initial_states, u_type, u_order, 
                      save_data, wip_params, wip_trials):
    # Storage for aggregated results
    all_plan_rewards = np.zeros((n_iterations, l_episodes, n_arms))
    all_plan_objectives = np.zeros((n_iterations, l_episodes, n_arms))
    all_learn_rewards = np.zeros((n_iterations, l_episodes, n_arms))
    all_learn_objectives = np.zeros((n_iterations, l_episodes, n_arms))
    all_learn_indexerrors = np.zeros((n_iterations, l_episodes, n_arms))
    all_learn_probs = np.ones((n_iterations, l_episodes, n_arms))
    all_learn_transitionerrors = np.ones((n_iterations, l_episodes, n_arms))

    n_trials_safety = wip_trials
    PlanW = Whittle(n_states, n_arms, tru_rew, tru_dyn, n_steps)
    PlanW.get_whittle_indices(computation_type=method, params=wip_params, n_trials=n_trials_safety)

    for n in range(n_iterations):
        print(f"Iteration {n} starts ...")
        results = Process_LearnTSRB_iteration(
            n, l_episodes=l_episodes, n_episodes=n_episodes, n_steps=n_steps, n_states=n_states, n_arms=n_arms,
            n_choices=n_choices, thresholds=thresholds, t_type=t_type, t_increasing=t_increasing, method=method,
            tru_rew=tru_rew, tru_dyn=tru_dyn, initial_states=initial_states, u_type=u_type, u_order=u_order, wip_params=wip_params,
            PlanW=PlanW, n_trials_safety=n_trials_safety
        )

        all_plan_rewards[n] = results["plan_rewards"]
        all_plan_objectives[n] = results["plan_objectives"]
        all_learn_rewards[n] = results["learn_rewards"]
        all_learn_objectives[n] = results["learn_objectives"]
        all_learn_indexerrors[n] = results["learn_indexerrors"]
        all_learn_transitionerrors[n] = results["learn_transitionerrors"]
        all_learn_probs[n] = results["learn_probs"]
        print(f"Iteration {n} ends ...")

    if save_data:
        filename = f'./output-learn-finite/tsrb_ne{n_episodes}_nt{n_steps}_ns{n_states}_na{n_arms}_tt{t_type}_ut{u_type}_nc{n_choices}_th{thresholds[0]}_ut{u_type}_uo{u_order}.joblib'
        joblib.dump([all_learn_transitionerrors, all_learn_indexerrors, all_learn_rewards, all_learn_objectives,
                     all_plan_rewards, all_plan_objectives],
                    filename)
        print(f"Data saved to {filename}")

    return all_learn_transitionerrors, all_learn_indexerrors, all_learn_rewards, all_learn_objectives, all_plan_rewards, all_plan_objectives


def ProcessMulti_LearnTSRB(n_iterations, l_episodes, n_episodes, n_steps, n_states, n_arms, n_choices, thresholds,
                           t_type, t_increasing, method, tru_rew, tru_dyn, initial_states, u_type, u_order, save_data, wip_params, wip_trials):
    num_workers = cpu_count() - 1

    n_trials_safety = wip_trials
    PlanW = Whittle(n_states, n_arms, tru_rew, tru_dyn, n_steps)
    PlanW.get_whittle_indices(computation_type=method, params=wip_params, n_trials=n_trials_safety)

    # Define arguments for each iteration
    args = [
        (i, l_episodes, n_episodes, n_steps, n_states, n_arms, n_choices, thresholds,
         t_type, t_increasing, method, tru_rew, tru_dyn, initial_states, u_type, u_order, 
         wip_params, PlanW, n_trials_safety)
        for i in range(n_iterations)
    ]

    # Use multiprocessing
    with Pool(num_workers) as pool:
        results = pool.starmap(Process_LearnTSRB_iteration, args)

    # Aggregate results
    all_learn_transitionerrors = np.stack([res["learn_transitionerrors"] for res in results])
    all_learn_indexerrors = np.stack([res["learn_indexerrors"] for res in results])
    all_learn_rewards = np.stack([res["learn_rewards"] for res in results])
    all_learn_objectives = np.stack([res["learn_objectives"] for res in results])
    all_plan_rewards = np.stack([res["plan_rewards"] for res in results])
    all_plan_objectives = np.stack([res["plan_objectives"] for res in results])

    if save_data:
        filename = f'./output-learn-finite/tsrb_ne{n_episodes}_nt{n_steps}_ns{n_states}_na{n_arms}_tt{t_type}_ut{u_type}_nc{n_choices}_th{thresholds[0]}_ut{u_type}_uo{u_order}.joblib'
        joblib.dump([all_learn_transitionerrors, all_learn_indexerrors, all_learn_rewards, all_learn_objectives,
                     all_plan_rewards, all_plan_objectives],
                    filename)

    return all_learn_transitionerrors, all_learn_indexerrors, all_learn_rewards, all_learn_objectives, all_plan_rewards, all_plan_objectives


def Process_LearnNlTSRB(n_iterations, l_episodes, n_episodes, n_steps, n_states, n_arms, n_choices, thresholds,
                      t_type, t_increasing, method, tru_rew, tru_dyn, initial_states, u_type, u_order, save_data, wip_params, wip_trials):
    # Storage for aggregated results
    all_plan_rewards = np.zeros((n_iterations, l_episodes, n_arms))
    all_plan_objectives = np.zeros((n_iterations, l_episodes, n_arms))
    all_learn_rewards = np.zeros((n_iterations, l_episodes, n_arms))
    all_learn_objectives = np.zeros((n_iterations, l_episodes, n_arms))
    all_learn_indexerrors = np.zeros((n_iterations, l_episodes, n_arms))
    all_learn_probs = np.ones((n_iterations, l_episodes, n_arms))
    all_learn_transitionerrors = np.ones((n_iterations, l_episodes, n_arms))

    n_trials_safety = wip_trials
    PlanW = Whittle(n_states, n_arms, tru_rew, tru_dyn, n_steps)
    PlanW.get_whittle_indices(computation_type=method, params=wip_params, n_trials=n_trials_safety)

    for n in range(n_iterations):
        print(f"Iteration {n} starts ...")
        results = Process_LearnTSRB_iteration(
            n, l_episodes=l_episodes, n_episodes=n_episodes, n_steps=n_steps, n_states=n_states, n_arms=n_arms,
            n_choices=n_choices, thresholds=thresholds, t_type=t_type, t_increasing=t_increasing, method=method,
            tru_rew=tru_rew, tru_dyn=tru_dyn, initial_states=initial_states, u_type=u_type, u_order=u_order, wip_params=wip_params,
            PlanW=PlanW, n_trials_safety=n_trials_safety
        )

        all_plan_rewards[n] = results["plan_rewards"]
        all_plan_objectives[n] = results["plan_objectives"]
        all_learn_rewards[n] = results["learn_rewards"]
        all_learn_objectives[n] = results["learn_objectives"]
        all_learn_indexerrors[n] = results["learn_indexerrors"]
        all_learn_transitionerrors[n] = results["learn_transitionerrors"]
        all_learn_probs[n] = results["learn_probs"]
        print(f"Iteration {n} ends ...")

    if save_data:
        filename = f'./output-learn-finite/nltsrb_ne{n_episodes}_nt{n_steps}_ns{n_states}_na{n_arms}_tt{t_type}_ut{u_type}_nc{n_choices}_th{thresholds[0]}_ut{u_type}_uo{u_order}.joblib'
        joblib.dump([all_learn_transitionerrors, all_learn_indexerrors, all_learn_rewards, all_learn_objectives,
                     all_plan_rewards, all_plan_objectives],
                    filename)
        print(f"Data saved to {filename}")

    return all_learn_transitionerrors, all_learn_indexerrors, all_learn_rewards, all_learn_objectives, all_plan_rewards, all_plan_objectives


def ProcessMulti_LearnNlTSRB(n_iterations, l_episodes, n_episodes, n_steps, n_states, n_arms, n_choices, thresholds,
                           t_type, t_increasing, method, tru_rew, tru_dyn, initial_states, u_type, u_order, save_data, wip_params, wip_trials):
    num_workers = cpu_count() - 1

    n_trials_safety = wip_trials
    PlanW = Whittle(n_states, n_arms, tru_rew, tru_dyn, n_steps)
    PlanW.get_whittle_indices(computation_type=method, params=wip_params, n_trials=n_trials_safety)

    # Define arguments for each iteration
    args = [
        (i, l_episodes, n_episodes, n_steps, n_states, n_arms, n_choices, thresholds,
         t_type, t_increasing, method, tru_rew, tru_dyn, initial_states, u_type, u_order, 
         wip_params, PlanW, n_trials_safety)
        for i in range(n_iterations)
    ]

    # Use multiprocessing
    with Pool(num_workers) as pool:
        results = pool.starmap(Process_LearnTSRB_iteration, args)

    # Aggregate results
    all_learn_transitionerrors = np.stack([res["learn_transitionerrors"] for res in results])
    all_learn_indexerrors = np.stack([res["learn_indexerrors"] for res in results])
    all_learn_rewards = np.stack([res["learn_rewards"] for res in results])
    all_learn_objectives = np.stack([res["learn_objectives"] for res in results])
    all_plan_rewards = np.stack([res["plan_rewards"] for res in results])
    all_plan_objectives = np.stack([res["plan_objectives"] for res in results])

    if save_data:
        filename = f'./output-learn-finite/nltsrb_ne{n_episodes}_nt{n_steps}_ns{n_states}_na{n_arms}_tt{t_type}_ut{u_type}_nc{n_choices}_th{thresholds[0]}_ut{u_type}_uo{u_order}.joblib'
        joblib.dump([all_learn_transitionerrors, all_learn_indexerrors, all_learn_rewards, all_learn_objectives,
                     all_plan_rewards, all_plan_objectives],
                    filename)

    return all_learn_transitionerrors, all_learn_indexerrors, all_learn_rewards, all_learn_objectives, all_plan_rewards, all_plan_objectives

