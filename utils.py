import numpy
import joblib
from processes import *
from whittle import *
from Markov import *
from learning import *
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


def run_multiple_planning_combinations(param_list):

    # Determine the number of CPUs to use
    num_cpus = cpu_count()-1
    print(f"Using {num_cpus} CPUs")
    
    eval_keys = ['Neutral', 'RewUtility', 'RiskAware', 
                 'RI_RiskAware_to_Neutral', 'RI_RiskAware_to_RewUtility', 'RI_RewUtility_to_Neutral',
                 'DF_RiskAware_to_Neutral', 'DF_RiskAware_to_RewUtility', 'DF_RewUtility_to_Neutral',]
    results = {key: {} for key in eval_keys}
    averages = {key: {} for key in eval_keys}
    total = len(param_list)
    with Pool(num_cpus) as pool:
        # Use imap to get results as they complete
        for count, output in enumerate(pool.imap_unordered(run_a_planning_combination, param_list), 1):
            key_value, avg_n, avg_ru, avg_ra, improve_rn, improve_ru, improve_un, diff_rn, diff_ru, diff_un = output
            for i, value in enumerate([avg_n, avg_ru, avg_ra, improve_rn, improve_ru, improve_un, diff_rn, diff_ru, diff_un]):
                results[eval_keys[i]][key_value] = value

            print(f"{count} / {total}: {key_value} ---> MEAN-Rel-RN: {improve_rn}, MEAN-Rel-UN: {improve_un}")
            for _, value in zip(['nt', 'ns', 'nc', 'ut', 'th', 'fr'], output[0].split('_')):
                param_key = f'{value}'
                for i, avg_key in enumerate(eval_keys):
                    if param_key not in averages[avg_key]:
                        averages[avg_key][param_key] = []
                    averages[avg_key][param_key].append(results[eval_keys[i]][key_value])
    
    return results, averages


def run_a_planning_combination(params):
    nt, ns, nc, ut, th, fr, n_iterations, save_flag, PATH = params
    key_value = f'nt{nt}_ns{ns}_nc{nc}_ut{ut}_th{th}_fr{fr}'
    na = nc * ns

    rew_vals = rewards(nt, na, ns)
    rew_utility_vals = rewards_utility(nt, na, ns, th, ut[0], ut[1])
    prob_remain = numpy.round(numpy.linspace(0.1 / ns, 1 / ns, na), 2)
    markov_matrix = get_transitions(na, ns, prob_remain, 'structured')

    Neutral_Whittle = Whittle(ns, na, rew_vals, markov_matrix, nt)
    Neutral_Whittle.get_indices(2*nt, nt*ns*na)

    Utility_Whittle = Whittle(ns, na, rew_utility_vals, markov_matrix, nt)
    Utility_Whittle.get_indices(2*nt, nt*ns*na)

    RiskAware_Whittle = RiskAwareWhittle(ns, na, rew_vals, markov_matrix, nt, ut[0], ut[1], th)
    RiskAware_Whittle.get_indices(2*nt, nt*ns*na)

    nch = max(1, int(round(fr * na)))
    initial_states = (ns - 1) * numpy.ones(na, dtype=numpy.int32)

    processes = [
        ("Neutral", lambda *args: process_neutral_whittle(Neutral_Whittle, *args)),
        ("RewUtility", lambda *args: process_neutral_whittle(Utility_Whittle, *args)),
        ("RiskAware", lambda *args: process_riskaware_whittle(RiskAware_Whittle, *args))
    ]

    results = {}
    for name, process in processes:
        rew, obj = process(n_iterations, nt, ns, na, nch, th, rew_vals, markov_matrix, initial_states, ut[0], ut[1])
        if save_flag:
            joblib.dump([rew, obj], f"{PATH}{key_value}_{name}.joblib")
        results[name] = numpy.mean(obj)

    improve_rn = numpy.round(100 * (results['RiskAware'] - results['Neutral']) / results['Neutral'], 2)
    improve_ru = numpy.round(100 * (results['RiskAware'] - results['RewUtility']) / results['RewUtility'], 2)
    improve_un = numpy.round(100 * (results['RewUtility'] - results['Neutral']) / results['Neutral'], 2)

    diff_rn = na * np.round(results['RiskAware'] - results['Neutral'], 4)
    diff_ru = na * np.round(results['RiskAware'] - results['RewUtility'], 4)
    diff_un = na * np.round(results['RewUtility'] - results['Neutral'], 4)

    return key_value, results["Neutral"], results["RewUtility"], results["RiskAware"], improve_rn, improve_ru, improve_un, diff_rn, diff_ru, diff_un


def run_learning_combination(params):
    nt, ns, na, tt, ut, th, nc, l_episodes, n_episodes, n_iterations, save_data, PATH = params

    if tt == 'structured':
        prob_remain = numpy.round(numpy.linspace(0.1 / ns, 0.1 / ns, na), 2)
    elif tt == 11:
        pr_ss_0 = np.round(np.linspace(0.596, 0.690, na), 3)
        np.random.shuffle(pr_ss_0)
        pr_sr_0 = np.round(np.linspace(0.045, 0.061, na), 3)
        np.random.shuffle(pr_sr_0)
        pr_sp_0 = np.round(np.linspace(0.201, 0.287, na), 3)
        np.random.shuffle(pr_sp_0)
        pr_rr_0 = np.round(np.linspace(0.759, 0.822, na), 3)
        np.random.shuffle(pr_rr_0)
        pr_rp_0 = np.round(np.linspace(0.130, 0.169, na), 3)
        np.random.shuffle(pr_rp_0)
        pr_pp_0 = np.round(np.linspace(0.882, 0.922, na), 3)
        np.random.shuffle(pr_pp_0)
        pr_ss_1 = np.round(np.linspace(0.733, 0.801, na), 3)
        np.random.shuffle(pr_ss_1)
        pr_sr_1 = np.round(np.linspace(0.047, 0.078, na), 3)
        np.random.shuffle(pr_sr_1)
        pr_sp_1 = np.round(np.linspace(0.115, 0.171, na), 3)
        np.random.shuffle(pr_sp_1)
        pr_rr_1 = np.round(np.linspace(0.758, 0.847, na), 3)
        np.random.shuffle(pr_rr_1)
        pr_rp_1 = np.round(np.linspace(0.121, 0.193, na), 3)
        np.random.shuffle(pr_rp_1)
        pr_pp_1 = np.round(np.linspace(0.879, 0.921, na), 3)
        np.random.shuffle(pr_pp_1)
        prob_remain = [pr_ss_0, pr_sr_0, pr_sp_0, pr_rr_0, pr_rp_0, pr_pp_0, pr_ss_1, pr_sr_1, pr_sp_1, pr_rr_1, pr_rp_1, pr_pp_1]
        ns=4
    elif tt == 12:
        pr_ss_0 = np.round(np.linspace(0.668, 0.738, na), 3)
        np.random.shuffle(pr_ss_0)
        pr_sr_0 = np.round(np.linspace(0.045, 0.061, na), 3)
        np.random.shuffle(pr_sr_0)
        pr_rr_0 = np.round(np.linspace(0.831, 0.870, na), 3)
        np.random.shuffle(pr_rr_0)
        pr_pp_0 = np.round(np.linspace(0.882, 0.922, na), 3)
        np.random.shuffle(pr_pp_0)
        pr_ss_1 = np.round(np.linspace(0.782, 0.833, na), 3)
        np.random.shuffle(pr_ss_1)
        pr_sr_1 = np.round(np.linspace(0.047, 0.078, na), 3)
        np.random.shuffle(pr_sr_1)
        pr_rr_1 = np.round(np.linspace(0.807, 0.879, na), 3)
        np.random.shuffle(pr_rr_1)
        pr_pp_1 = np.round(np.linspace(0.879, 0.921, na), 3)
        np.random.shuffle(pr_pp_1)
        prob_remain = [pr_ss_0, pr_sr_0, pr_rr_0, pr_pp_0, pr_ss_1, pr_sr_1, pr_rr_1, pr_pp_1]
        ns=4
    elif tt == 13:
        pr_ss_0 = np.round(np.linspace(0.657, 0.762, na), 3)
        np.random.shuffle(pr_ss_0)
        pr_sp_0 = np.round(np.linspace(0.201, 0.287, na), 3)
        np.random.shuffle(pr_sp_0)
        pr_pp_0 = np.round(np.linspace(0.882, 0.922, na), 3)
        np.random.shuffle(pr_pp_0)
        pr_ss_1 = np.round(np.linspace(0.806, 0.869, na), 3)
        np.random.shuffle(pr_ss_1)
        pr_sp_1 = np.round(np.linspace(0.115, 0.171, na), 3)
        np.random.shuffle(pr_sp_1)
        pr_pp_1 = np.round(np.linspace(0.879, 0.921, na), 3)
        np.random.shuffle(pr_pp_1)
        prob_remain = [pr_ss_0, pr_sp_0, pr_pp_0, pr_ss_1, pr_sp_1, pr_pp_1]
        ns=3
    elif tt == 14:
        pr_ss_0 = np.round(np.linspace(0.713, 0.799, na), 3)
        np.random.shuffle(pr_ss_0)
        pr_pp_0 = np.round(np.linspace(0.882, 0.922, na), 3)
        np.random.shuffle(pr_pp_0)
        pr_ss_1 = np.round(np.linspace(0.829, 0.885, na), 3)
        np.random.shuffle(pr_ss_1)
        pr_pp_1 = np.round(np.linspace(0.879, 0.921, na), 3)
        np.random.shuffle(pr_pp_1)
        prob_remain = [pr_ss_0, pr_pp_0, pr_ss_1, pr_pp_1]
        ns=3
    else:
        prob_remain = numpy.round(numpy.linspace(0.1, 0.9, na), 2)

    key_value = f'nt{nt}_ns{ns}_na{na}_tt{tt}_ut{ut}_th{th}_nc{nc}'
    rew_vals = rewards(nt, na, ns)
    rew_utility_vals = rewards_utility(nt, na, ns, th, ut[0], ut[1])
    prob_remain = numpy.round(numpy.linspace(0.1 / ns, 1 / ns, na), 2)
    markov_matrix = get_transitions(na, ns, prob_remain, tt)
    initial_states = (ns - 1) * numpy.ones(na, dtype=numpy.int32)
    w_range = 2*nt
    w_trials = nt*ns*na

    prob_err_ln, indx_err_ln, _, obj_ln, _, obj_n = multiprocess_learn_LRNPTS(
        n_iterations, l_episodes, n_episodes, nt, ns, na, nc, th, rew_vals, markov_matrix, initial_states, ut[0], ut[1], 
        save_data, w_range, w_trials, f'{PATH}neutral_{key_value}.joblib'
    )
    prob_err_lu, indx_err_lu, _, obj_lu, _, obj_u = multiprocess_learn_LRNPTS(
        n_iterations, l_episodes, n_episodes, nt, ns, na, nc, th, rew_utility_vals, markov_matrix, initial_states, ut[0], ut[1], 
        save_data, w_range, w_trials, f'{PATH}rewutility_{key_value}.joblib'
    )
    prob_err_lr, indx_err_lr, _, obj_lr, _, obj_r = multiprocess_learn_LRAPTS(
        n_iterations, l_episodes, n_episodes, nt, ns, na, nc, th, rew_vals, markov_matrix, initial_states, ut[0], ut[1], 
        save_data, w_range, w_trials, f'{PATH}riskaware_{key_value}.joblib'
    )

    process_and_plot(prob_err_ln, indx_err_ln, obj_n, obj_ln, 'lw', PATH, key_value)
    process_and_plot(prob_err_lu, indx_err_lu, obj_u, obj_lu, 'ln', PATH, key_value)
    process_and_plot(prob_err_lr, indx_err_lr, obj_r, obj_lr, 'ls', PATH, key_value)

    reg_lu, creg_lu, bounds_lu = compute_bounds(obj_r, obj_lu)
    plot_data(creg_lu, 'Episodes', 'Regret', f'{PATH}cumreg_lu_{key_value}.png')
    plot_data(creg_lu, 'Episodes', 'Regret', f'{PATH}cumregbounds_lu_{key_value}.png', fill_bounds=bounds_lu)
    plot_data(reg_lu, 'Episodes', 'Regret/K', f'{PATH}reg_lu_{key_value}.png')

    reg_ln, creg_ln, bounds_ln = compute_bounds(obj_r, obj_ln)
    plot_data(creg_ln, 'Episodes', 'Regret', f'{PATH}cumreg_ln_{key_value}.png')
    plot_data(creg_ln, 'Episodes', 'Regret', f'{PATH}cumregbounds_ln_{key_value}.png', fill_bounds=bounds_ln)
    plot_data(reg_ln, 'Episodes', 'Regret/K', f'{PATH}reg_ln_{key_value}.png')


def plot_data(y_data, xlabel, ylabel, filename, x_data=None, ylim=None, linewidth=4, fill_bounds=None):
    """
    Generic plotting function to handle repetitive plotting tasks.
    """
    plt.figure(figsize=(8, 6))
    x_data = x_data if x_data is not None else range(len(y_data))
    plt.plot(x_data, y_data, linewidth=linewidth)
    if fill_bounds:
        lower_bound, upper_bound = fill_bounds
        plt.fill_between(x_data, lower_bound, upper_bound, color='blue', alpha=0.2)
    plt.xlabel(xlabel, fontsize=14, fontweight='bold')
    plt.ylabel(ylabel, fontsize=14, fontweight='bold')
    if ylim:
        plt.ylim(ylim)
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def compute_bounds(perf_ref, perf_lrn):
    """
    Computes regret and confidence bounds.
    """
    avg_creg = np.mean(np.cumsum(np.sum(perf_ref - perf_lrn, axis=2), axis=1), axis=0)
    std_creg = np.std(np.cumsum(np.sum(perf_ref - perf_lrn, axis=2), axis=1), axis=0)
    avg_reg = [avg_creg[k] / (k + 1) for k in range(len(avg_creg))]
    return avg_reg, avg_creg, (avg_creg - std_creg, avg_creg + std_creg)


def process_and_plot(prob_err, indx_err, perf_ref, perf_lrn, suffix, path, key_value):
    """
    Processes data and generates all required plots for a given suffix.
    """
    trn_err = np.mean(prob_err, axis=(0, 2))
    wis_err = np.mean(indx_err, axis=(0, 2))
    reg, creg, bounds = compute_bounds(perf_ref, perf_lrn)

    plot_data(trn_err, 'Episodes', 'Max Transition Error', f'{path}per_{suffix}_{key_value}.png')
    plot_data(wis_err, 'Episodes', 'Max WI Error', f'{path}wer_{suffix}_{key_value}.png')
    plot_data(creg, 'Episodes', 'Regret', f'{path}cumreg_{suffix}_{key_value}.png')
    plot_data(creg, 'Episodes', 'Regret', f'{path}cumregbounds_{suffix}_{key_value}.png', fill_bounds=bounds)
    plot_data(reg, 'Episodes', 'Regret/K', f'{path}reg_{suffix}_{key_value}.png')
