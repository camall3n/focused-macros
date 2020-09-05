# pylint: skip-file

import argparse
import copy
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall
from tqdm import tqdm

from domains.suitcaselock import SuitcaseLock

class CPUTimer:
    def __enter__(self):
        self.start = time.time()
        self.end = self.start
        self.duration = 0.0
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.end = time.time()
            self.duration = self.end - self.start

get_state_id = lambda lock: int(''.join(map(str,lock.state)), base=v)

def compute_cost_matrix(lock, n=6, v=4, k=1):
    actions = lock.actions()[:n]

    print('Computing the adjacency matrix')
    n_states = v**n
    if n_states > 2**16-1:
        print('Warning: too many states for uint16: {} > {}'.format(n_states, 2**16-1))
    A = np.eye(n_states, dtype=np.uint16)

    # row = from
    # column = to
    def get_successors(lock):
        return [copy.deepcopy(lock).apply_macro(diff=a) for a in actions]

    for state in tqdm(lock.states(), total=n_states):
        state_id = get_state_id(state)
        next_states = get_successors(state)
        for next_state in next_states:
            next_state_id = get_state_id(next_state)
            A[state_id,next_state_id] = 1

    # Simultaneously compute:
    # - whether there is a path from i to j of length <= L
    # - the length of the shortest path length from i to j
    print('Computing length of shortest paths')
    min_path_length = 0*np.eye(n_states, dtype=np.uint16) # (zero cost to reach same state)
    length = 0
    has_path_of_length_n = np.eye(n_states, dtype=bool)
    has_path_of_length_n_plus_1 = A.astype(bool)
    min_path_length += (length+1) * (has_path_of_length_n_plus_1 ^ has_path_of_length_n).astype(np.uint16)
    has_path_of_length_n = has_path_of_length_n_plus_1
    length += 1
    with tqdm(total=n_states) as pbar:
        while np.any(has_path_of_length_n != np.ones((n_states,n_states))):
            has_path_of_length_n_plus_1 = np.matmul(has_path_of_length_n, A).astype(bool)
            min_path_length += (length+1) * (has_path_of_length_n_plus_1 ^ has_path_of_length_n).astype(np.uint16)
            has_path_of_length_n = has_path_of_length_n_plus_1
            length += 1
            pbar.update(1)
            if length >= n_states:
                break
        pbar.update(n_states)
        pbar.refresh()
        time.sleep(1)
    return min_path_length

def compute_apsp_floyd_warshall(lock, n=6, v=2, k=1):
    actions = lock.actions()[:n]

    print('Computing the adjacency matrix')
    n_states = v**n
    if n_states > 2**16-1:
        print('Warning: too many states for uint16: {} > {}'.format(n_states, 2**16-1))
    A = np.eye(n_states, dtype=np.uint16)

    # row = from
    # column = to
    def get_successors(lock):
        return [copy.deepcopy(lock).apply_macro(diff=a) for a in actions]

    for state in tqdm(lock.states(), total=n_states):
        state_id = get_state_id(state)
        next_states = get_successors(state)
        for next_state in next_states:
            next_state_id = get_state_id(next_state)
            A[state_id,next_state_id] = 1

    graph = csr_matrix(A)
    min_path_length = floyd_warshall(csgraph=graph, directed=True, return_predecessors=False)
    return min_path_length


def compute_heuristic_matrix(lock, n=6, v=2, k=1):
    n_states = v**n

    # Compute the goal-count heuristic from i to j
    heuristic_matrix = np.zeros((n_states, n_states), dtype=int)
    for row, start in enumerate(lock.states()):
        for col, goal in enumerate(lock.states()):
            heuristic_matrix[row,col] = heuristic(start,goal)

    return heuristic_matrix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=6, help='Number of dials')
    parser.add_argument('-v', type=int, default=2, help='Number of digits per dial')
    parser.add_argument('-k', type=int, default=1, help='Average effect size')
    parser.add_argument('--seed', '-s', type=int, default=1, help='Random seed')
    args = parser.parse_args()

    n = args.n
    v = args.v
    k = args.k
    assert 1 <= k < n
    seed = args.seed
    n_states = v**n

    results_dir = 'results/heuristic/lock_{}x{}ary/'.format(n, v)
    os.makedirs(results_dir, exist_ok=True)

    np.random.seed(seed)
    lock = SuitcaseLock(n_vars=n, n_values=v, entanglement=k)

    with CPUTimer() as timer:
        D = compute_apsp_floyd_warshall(lock, n, v, k)
    print('floyd_warshall:', timer.duration)
    assert False

    print('Comparing heuristic to true cost')
    heuristic = lambda start, goal: sum(goal.summarize_effects(baseline=start) > 0)
    data = []
    results_file = results_dir+'k-{:02d}_seed-{:03d}.csv'.format(k, seed)

    with open(results_file, 'w', newline='') as file:
        index = 0
        for row, start in enumerate(tqdm(lock.states(), total=n_states)):
            for col, goal in enumerate(lock.states()):
                h = heuristic(start, goal)
                d = D[get_state_id(start),get_state_id(goal)]
                data = {
                    'n_dials': n,
                    'n_values': v,
                    'k': k,
                    'distance': d,
                    'heuristic': h,
                    'seed': seed
                }
                pd.DataFrame([data]).to_csv(file, index=None, header=(index == 0))
                index += 1
    print('Results saved to', results_file)
