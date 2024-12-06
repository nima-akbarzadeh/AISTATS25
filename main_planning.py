import os
import numpy
import pandas as pd
from utils import *
import warnings
warnings.filterwarnings("ignore")

def main():

    param_sets = {
        'n_steps_set': [3, 4, 5],
        'n_states_set': [2, 3, 4, 5],
        'armcoef_set': [3, 4, 5],
        'f_type_set': ['hom'], # homogeneous reward functions
        't_type_set': [3], # 3 is for the third structured model, 13 is for the third model of the clinical example
        'u_type_set': [(1, 0), (2, 4), (2, 8), (2, 16), (3, 4), (3, 8), (3, 16)],
        'threshold_set': [np.round(0.1 * n, 1) for n in range(1, 10)],
        'fraction_set': [0.3, 0.4, 0.5]
    }

    # param_sets = {
    #     'n_steps_set': [3],
    #     'n_states_set': [3],
    #     'armcoef_set': [1],
    #     'f_type_set': ['hom'], # homogeneous reward functions
    #     't_type_set': [3], # 3 is for the third structured model, 13 is for the third model of the clinical example
    #     'u_type_set': [(2, 8)],
    #     'threshold_set': [np.round(0.1 * n, 1) for n in range(5, 6)],
    #     'fraction_set': [0.5]
    # }

    whittle_computation_method = 2
    n_episodes = 100
    
    save_flag=False
    PATH = f'./output-finite/method{whittle_computation_method}_ne{n_episodes}/'
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    param_list = [
        (nt, ns, nc, ft_type, tt, ut, th, fr, whittle_computation_method, n_episodes, save_flag, PATH)
        for nt in param_sets['n_steps_set']
        for ns in param_sets['n_states_set']
        for nc in param_sets['armcoef_set']
        for ft_type in param_sets['f_type_set']
        for tt in param_sets['t_type_set']
        for ut in param_sets['u_type_set']
        for th in param_sets['threshold_set']
        for fr in param_sets['fraction_set']
    ]

    results, averages = run_multiple_planning_combinations(param_list)

    # Save results to Excel
    df1 = pd.DataFrame({f'MEAN-{key.capitalize()}': value for key, value in results.items()})
    df1.index.name = 'Key'
    df1.to_excel(f'{PATH}Res_m{whittle_computation_method}.xlsx')

    df2 = pd.DataFrame({f'MEAN-{key.capitalize()}': {k: numpy.mean(v) if v else 0 for k, v in avg.items()}
                        for key, avg in averages.items()})
    df2.index.name = 'Key'
    df2.to_excel(f'{PATH}ResAvg_m{whittle_computation_method}.xlsx')


if __name__ == '__main__':
    main()