# pylint: skip-file

import argparse
import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from domains.suitcaselock import SuitcaseLock

def compute_cost_matrix(lock, n=6, v=4, k=1):
    actions = lock.actions()[:n]

    # Compute the adjacency matrix
    n_states = v**n
    A = np.eye(n_states,dtype=int)

    # row = from
    # column = to
    get_state_id = lambda lock: int(''.join(map(str,lock.state)), base=v)
    def get_successors(lock):
        return [copy.deepcopy(lock).apply_macro(diff=a) for a in actions]

    for state in lock.states():
        state_id = get_state_id(state)
        next_states = get_successors(state)
        for next_state in next_states:
            next_state_id = get_state_id(next_state)
            A[state_id,next_state_id] = 1

    # Compute whether there is a path from i to j of length <= L
    has_path = [np.eye(n_states), A]
    length = 1
    while np.any(has_path[-1] != np.ones((n_states,n_states))):
        has_path.append((np.matmul(has_path[-1],A)>0).astype(int))
        length += 1
        if length >= n_states:
            break

    # Compute the newly added paths at each length L
    path_length = [0*np.eye(n_states)]
    for i, _ in enumerate(has_path):
        path_length.append( (i+1)*(has_path[i]-has_path[i-1]) )

    # Compute the shortest path length from i to j
    distance_matrix = np.sum(np.stack(path_length), 0)
    return distance_matrix

def compute_heuristic_matrix(lock, n=6, v=2, k=1):
    n_states = v**n

    # Compute the goal-count heuristic from i to j
    heuristic = lambda start, goal: sum(goal.summarize_effects(baseline=start) > 0)

    heuristic_matrix = np.zeros((n_states, n_states), dtype=int)
    for row, start in enumerate(lock.states()):
        for col, goal in enumerate(lock.states()):
            heuristic_matrix[row,col] = heuristic(start,goal)

    return heuristic_matrix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=6, help='Number of dials')
    parser.add_argument('-v', type=int, default=2, help='Number of digits per dial')
    parser.add_argument('--n_trials', type=int, default=10, help='Number of trials per effect size')
    args = parser.parse_args()

    n = args.n
    v = args.v

    results_dir = 'results/heuristic/lock_{}x{}ary/'.format(n, v)
    os.makedirs(results_dir, exist_ok=True)

    n_trials = args.n_trials
    data = []
    for k in range(1,n):
        for seed in range(n_trials):
            np.random.seed(seed)
            lock = SuitcaseLock(n_vars=n, n_values=v, entanglement=k)
            D = compute_cost_matrix(lock, n, v, k)
            H = compute_heuristic_matrix(lock, n, v, k)
            data = pd.DataFrame([{'n_dials': n, 'n_values': v, 'k': k, 'distance': d, 'heuristic': h, 'seed': seed}
                         for d, h in zip(D.flatten(), H.flatten())])
            results_file = results_dir+'k-{:02d}_seed-{:03d}.csv'.format(k, seed)
            data.to_csv(results_file, index=False)
            print('Results saved to', results_file)
