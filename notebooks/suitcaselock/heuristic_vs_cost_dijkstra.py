import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from suitcaselock.suitcaselock import SuitcaseLock
from notebooks import search

n = 6 # digits
v = 2 # values per digit
n_trials = 10
data = []
for k in tqdm(range(1,n)):
    for trial in tqdm(range(n_trials)):
        newlock = SuitcaseLock(n_vars=n, n_values=v, entanglement=k)
        skills = newlock.actions()

        models = [copy.deepcopy(newlock).apply_macro(diff=s).summarize_effects(baseline=newlock) for s in skills]

        def wrapped_dijkstra(start, debug_fn=None):
            is_goal = lambda node: node.state == newlock
            heuristic = lambda lock: 0
            step_cost = lambda skill: 1
            def get_successors(lock):
                return [(copy.deepcopy(lock).apply_macro(diff=m), s) for s,m in zip(skills, models)]
            results = search.dijkstra(start, is_goal, step_cost, heuristic, get_successors, max_transitions=1e9, debug_fn=debug_fn, quiet=True)
            states, actions, n_expanded, n_transitions, candidates = results
            if len(actions)==0 and start != newlock:
                cost = np.inf
            else:
                cost = len(actions)
            return cost

        for start in newlock.states():
            h_score = sum(start.summarize_effects(baseline=newlock) > 0)
            cost = wrapped_dijkstra(start)
            if cost == np.inf:
                break

            data.append({'id': int(''.join(map(str,start.state)), base=v), 'h_score': h_score, 'cost': cost, 'entanglement': k, 'trial': trial})
        if cost == np.inf:
            break
    if cost == np.inf:
        break
data = pd.DataFrame(data)
#%%

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    data.query('entanglement==2')
    print(data[data['entanglement']==2])
#%%
sns.pointplot(data=data,x='cost', y='h_score', estimator=np.mean, hue='entanglement', units='trial')
plt.title('Heuristic vs. Cost (by entanglement value)')
plt.legend(ncol=5)
