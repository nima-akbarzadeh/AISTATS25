import os
from utils import *
import warnings
warnings.filterwarnings("ignore")


def main():
    param_sets = {
        'n_steps': [5],
        'n_states': [4],
        'n_arms': [5],
        'transition_type': [11],
        'utility_functions': [(1, 0), (2, 4), (2, 8), (2, 16), (3, 4), (3, 8), (3, 16)],
        'thresholds': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        'arm_choices': [1]
    }

    learning_episodes = 2
    n_averaging_episodes = 1
    n_iterations = 5

    save_data = True
    PATH = f'./learning-finite-{learning_episodes}-{n_averaging_episodes}-{n_iterations}/medical_examples/'
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    param_list = [
        (nt, ns, na, tt, ut, th, nc, learning_episodes, n_averaging_episodes, n_iterations, save_data, PATH)
        for nt in param_sets['n_steps']
        for ns in param_sets['n_states']
        for na in param_sets['n_arms']
        for tt in param_sets['transition_type']
        for ut in param_sets['utility_functions']
        for th in param_sets['thresholds']
        for nc in param_sets['arm_choices']
    ]
    
    for params in param_list:
        print('='*50)
        run_learning_combination(params)

if __name__ == '__main__':
    main()
