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


def run_multiple_planning_combinations(param_list, param_sets):

    # Determine the number of CPUs to use
    num_cpus = cpu_count()-1
    print(f"Using {num_cpus} CPUs")
    
    eval_keys = ['Neutral', 'RewUtility', 'RiskAware', 'RI_RiskAware_to_Neutral', 'RI_RiskAware_to_RewUtility', 'RI_RewUtility_to_Neutral']
    results = {key: {} for key in eval_keys}
    averages = {key: {} for key in eval_keys}
    total = len(param_list)
    # Create a Pool of workers
    with Pool(num_cpus) as pool:
        # Use imap to get results as they complete
        for count, result in enumerate(pool.imap_unordered(run_planning_combination, param_list), 1):
            key_value, avg_n, avg_ru, avg_ra, improve_rn, improve_ru, improve_un = result
            for i, value in enumerate([avg_n, avg_ru, avg_ra, improve_rn, improve_ru, improve_un]):
                results[eval_keys[i]][key_value] = value

            print(f"{count} / {total}: {key_value} ---> MEAN-Rel-RN: {improve_rn}, MEAN-Rel-UN: {improve_un}")

            for _, value in zip(['nt', 'ns', 'nc', 'ft', 'tt', 'ut', 'th', 'fr', 'meth'], result[0].split('_')):
                param_key = f'{value}'
                for i, avg_key in enumerate(eval_keys):
                    if param_key not in averages[avg_key]:
                        averages[avg_key][param_key] = []
                    averages[avg_key][param_key].append(results[eval_keys[i]][key_value])
    
    return results, averages


def run_planning_combination(params):
    nt, ns, nc, ft, tt, ut, th, fr, method, n_episodes, PATH = params
    key_value = f'nt{nt}_ns{ns}_nc{nc}_ft{ft}_tt{tt}_ut{ut}_th{th}_fr{fr}_meth{method}'
    # print(f'Running for {key_value}')
    
    na = nc * ns
    ftype = numpy.ones(na, dtype=numpy.int32) if ft == 'hom' else 1 + numpy.arange(na)

    prob_remain = numpy.round(numpy.linspace(0.1 / ns, 1 / ns, na), 2)
    numpy.random.shuffle(prob_remain)

    r_vals = rewards(nt, na, ns, ftype)
    r_vals_nl = rewards_utility(nt, na, ns, ftype, th * numpy.ones(na), ut[0], ut[1])
    M = MarkovDynamics(na, ns, prob_remain, tt, True)

    NeutralWhittleObj = Whittle(ns, na, r_vals, M.transitions, nt)
    NeutralWhittleObj.get_whittle_indices(computation_type=method, params=[0, nt], n_trials=1000)

    UtilityWhittleObj = Whittle(ns, na, r_vals_nl, M.transitions, nt)
    UtilityWhittleObj.get_whittle_indices(computation_type=method, params=[0, nt], n_trials=1000)

    RiskAwareWhittleObj = RiskAwareWhittle(ns, na, r_vals, M.transitions, nt, ut[0], ut[1], th * numpy.ones(na))
    RiskAwareWhittleObj.get_whittle_indices(computation_type=method, params=[0, nt], n_trials=1000)

    nch = max(1, int(round(fr * na)))
    initial_states = (ns - 1) * numpy.ones(na, dtype=numpy.int32)

    processes = [
        ("Neutral", lambda *args: Process_WhtlRB(NeutralWhittleObj, *args)),
        ("RewUtility", lambda *args: Process_WhtlRB(UtilityWhittleObj, *args)),
        ("RiskAware", lambda *args: Process_SafeRB(RiskAwareWhittleObj, *args))
    ]

    results = {}
    for name, process in processes:
        rew, obj, _ = process(n_episodes, nt, ns, na, nch, th * numpy.ones(na), r_vals, M.transitions,
                              initial_states, ut[0], ut[1])
        joblib.dump([rew, obj], f"{PATH}{key_value}_{name}.joblib")
        results[name] = numpy.round(numpy.mean(obj), 3)

    improve_rn = numpy.round(100 * (results['RiskAware'] - results['Neutral']) / results['Neutral'], 2)
    improve_ru = numpy.round(100 * (results['RiskAware'] - results['RewUtility']) / results['RewUtility'], 2)
    improve_un = numpy.round(100 * (results['RewUtility'] - results['Neutral']) / results['Neutral'], 2)

    # print(f'Ending for {key_value}')

    return key_value, results["Neutral"], results["RewUtility"], results["RiskAware"], improve_rn, improve_ru, improve_un


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


def run_learning_combination(params):
    nt, ns, na, ft, tt, ut, uo, th, nc, method, l_episodes, n_episodes, n_iterations, PATH = params
    key_value = f'nt{nt}_ns{ns}_na{na}_ft{ft}_tt{tt}_ut{ut}_uo{uo}_th{th}_nc{nc}_meth{method}'
    ftype = numpy.ones(na, dtype=numpy.int32) if ft == 'hom' else 1 + numpy.arange(na)

    if tt == 0:
        prob_remain = numpy.round(numpy.linspace(0.1, 0.9, na), 2)
    elif tt == 1 or tt == 2:
        prob_remain = numpy.round(numpy.linspace(0.05, 0.45, na), 2)
    elif tt == 3:
        prob_remain = numpy.round(numpy.linspace(0.1 / ns, 0.1 / ns, na), 2)
    elif tt == 4 or tt == 5:
        prob_remain = numpy.round(numpy.linspace(0.1 / ns, 1 / ns, na), 2)
    elif tt == 6:
        prob_remain = numpy.round(numpy.linspace(0.2, 0.8, na), 2)
    else:
        prob_remain = numpy.round(numpy.linspace(0.1, 0.9, na), 2)
    numpy.random.shuffle(prob_remain)

    r_vals = rewards(nt, na, ns, ftype)
    r_vals_nl = rewards_utility(nt, na, ns, ftype, th * numpy.ones(na), ut, uo)
    M = MarkovDynamics(na, ns, prob_remain, tt, True)
    thresh = th * numpy.ones(na)

    initial_states = (ns - 1) * numpy.ones(na, dtype=numpy.int32)

    prob_err_ln, indx_err_ln, _, obj_ln, _, obj_n = ProcessMulti_LearnTSRB(
        n_iterations, l_episodes, n_episodes, nt, ns, na, nc,
        thresh, tt, True, method, r_vals, M.transitions,
        initial_states, ut, uo, False, nt
    )
    prob_err_lu, indx_err_lu, _, obj_lu, _, obj_u = ProcessMulti_LearnNlTSRB(
        n_iterations, l_episodes, n_episodes, nt, ns, na, nc,
        thresh, tt, True, method, r_vals_nl, M.transitions,
        initial_states, ut, uo, False, nt
    )
    prob_err_lr, indx_err_lr, _, obj_lr, _, obj_r = ProcessMulti_LearnSafeTSRB(
        n_iterations, l_episodes, n_episodes, nt, ns, na, nc,
        thresh, tt, True, method, r_vals, M.transitions,
        initial_states, ut, uo, False, nt
    )

    process_and_plot(prob_err_ln, indx_err_ln, obj_n, obj_ln, 'lw', PATH, key_value)
    process_and_plot(prob_err_lu, indx_err_lu, obj_u, obj_lu, 'ln', PATH, key_value)
    process_and_plot(prob_err_lr, indx_err_lr, obj_r, obj_lr, 'ls', PATH, key_value)

    reg_rlu, creg_rlu, bounds_rlu = compute_bounds(obj_r, obj_lu)
    plot_data(creg_rlu, 'Episodes', 'Regret', f'{PATH}cumreg_rlu_{key_value}.png')
    plot_data(creg_rlu, 'Episodes', 'Regret', f'{PATH}cumregbounds_rlu_{key_value}.png', fill_bounds=bounds_rlu)
    plot_data(reg_rlu, 'Episodes', 'Regret/K', f'{PATH}reg_rlu_{key_value}.png')

    reg_rln, creg_rln, bounds_rln = compute_bounds(obj_r, obj_ln)
    plot_data(creg_rln, 'Episodes', 'Regret', f'{PATH}cumreg_rln_{key_value}.png')
    plot_data(creg_rln, 'Episodes', 'Regret', f'{PATH}cumregbounds_rln_{key_value}.png', fill_bounds=bounds_rln)
    plot_data(reg_rln, 'Episodes', 'Regret/K', f'{PATH}reg_rln_{key_value}.png')
