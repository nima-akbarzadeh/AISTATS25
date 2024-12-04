import os
from utils import *
import warnings
warnings.filterwarnings("ignore")


def main():
    param_sets = {
        'n_steps_set': [4],
        'n_states_set': [4],
        'armcoef_set': [1],
        'f_type_set': ['hom'],
        't_type_set': [3],
        'u_type_set': [1],
        'u_order_set': [0],
        'threshold_set': [0.5],
        'fraction_set': [0.1]
    }

    whittle_computation_method = 2
    learning_episodes = 500
    n_averaging_episodes = 10
    n_iterations = 100

    PATH = f'./output-learn-finite-{whittle_computation_method}-{learning_episodes}-{n_averaging_episodes}-{n_iterations}/'
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    param_list = [
        (nt, ns, nc, ft_type, tt, ut, uo, th, fr, whittle_computation_method, learning_episodes, n_averaging_episodes, n_iterations, PATH3)
        for nt in param_sets['n_steps_set']
        for ns in param_sets['n_states_set']
        for nc in param_sets['armcoef_set']
        for ft_type in param_sets['f_type_set']
        for tt in param_sets['t_type_set']
        for ut in param_sets['u_type_set']
        for uo in param_sets['u_order_set']
        for th in param_sets['threshold_set']
        for fr in param_sets['fraction_set']
    ]
    
    for params in param_list:
        run_learning_combination(params)

if __name__ == '__main__':
    main()
