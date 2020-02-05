import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from domains.suitcaselock import SuitcaseLock, rank_mod2

def compute_cost_matrix(n=6, v=2, k=1):
    lock = SuitcaseLock(n_vars=n, n_values=v, entanglement=k)
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

def compute_heuristic_matrix(n=6, v=2, k=1):
    lock = SuitcaseLock(n_vars=n, n_values=v, entanglement=k)
    n_states = v**n

    # Compute the goal-count heuristic from i to j
    heuristic = lambda start, goal: sum(goal.summarize_effects(baseline=start) > 0)

    heuristic_matrix = np.zeros((n_states, n_states), dtype=int)
    for row, start in enumerate(lock.states()):
        for col, goal in enumerate(lock.states()):
            heuristic_matrix[row,col] = heuristic(start,goal)

    return heuristic_matrix

#%%
n = 6
v = 2
n_trials = 10
data = []
for k in tqdm(range(1,n)):
    for trial in range(n_trials):
        D = compute_cost_matrix(n, v, k)
        H = compute_heuristic_matrix(n, v, k)
        data.extend([{'distance': d, 'heuristic': h, 'k': k, 'trial': trial}
                     for d, h in zip(D.flatten(), H.flatten())])
data = pd.DataFrame(data)

#%%
fig, ax = plt.subplots(figsize=(8,6))
sns.pointplot(data=data, x='distance', y='heuristic', hue='k', ci='sd', dodge=0.25)
plt.savefig('results/plots/suitcaselock_heuristic_vs_true_distance.png')
plt.show()
