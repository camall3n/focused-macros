import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from notebooks.matmodn import mod_rank
from suitcaselock.suitcaselock import SuitcaseLock

def compute_D_and_H(n=6, v=2, k=1):
    lock = SuitcaseLock(n_vars=n, n_values=v, entanglement=k)
    skills = lock.actions()[:n]
    models = [copy.deepcopy(lock).apply_macro(diff=s).summarize_effects(baseline=lock) for s in skills]

    # Get the actions matrix
    M = np.asarray(skills[:n])
    mod_rank(M,2)

    # Compute the adjacency matrix
    n_states = v**n
    A = np.eye(n_states,dtype=int)

    # row = from
    # column = to
    get_state_id = lambda lock: int(''.join(map(str,lock.state)), base=v)
    def get_successors(lock):
        return [copy.deepcopy(lock).apply_macro(diff=m) for s,m in zip(skills, models)]

    for s in lock.states():
        id = get_state_id(s)
        next_states = get_successors(s)
        [get_state_id(s_prime) for s_prime in next_states]
        for s_prime in next_states:
            id_prime = get_state_id(s_prime)
            A[id, id_prime] = 1

    # Compute whether there is a path from i to j of length <= L
    HasPath = [np.eye(n_states), A]
    l = 1
    while np.any(HasPath[-1] != np.ones((n_states,n_states))):
        HasPath.append((np.matmul(HasPath[-1],A)>0).astype(int))
        l += 1
        if l >= n_states:
            break

    # Compute the newly added paths at each length L
    PathLength = [0*np.eye(n_states)]
    for i in range(len(HasPath)):
        PathLength.append( (i+1)*(HasPath[i]-HasPath[i-1]) )

    # Compute the shortest path length from i to j
    D = np.sum(np.stack(PathLength), 0)

    # Compute the goal-count heuristic from i to j
    heuristic = lambda start, goal: sum(goal.summarize_effects(baseline=start) > 0)

    H = np.zeros((n_states, n_states), dtype=int)
    for row, start in enumerate(lock.states()):
        for col, goal in enumerate(lock.states()):
            H[row, col] = heuristic(start,goal)

    return D, H

#%%
n = 6
v = 2
n_trials = 100
data = []
for k in tqdm(range(1,n)):
    for trial in range(n_trials):
        D, H = compute_D_and_H(n, v, k)
        data.extend([{'distance': d, 'heuristic': h, 'k': k, 'trial': trial} for d, h in zip(D.flatten(), H.flatten())])
    # plt.scatter(x=D.flatten(), y=H.flatten())
data = pd.DataFrame(data)

#%%
fig, ax = plt.subplots(figsize=(8,6))
sns.pointplot(data=data, x='distance', y='heuristic', hue='k', ci='sd', dodge=0.25)
plt.savefig('results/plots/heuristic_vs_true_distance.png')
plt.show()
