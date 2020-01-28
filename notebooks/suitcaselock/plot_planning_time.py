import copy
import glob
from itertools import groupby, count
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import pandas as pd
import seaborn as sns
import notebooks.picklefix
from notebooks.rrank import rrank, reduce
from notebooks.search import dijkstra

alg = 'gbfs'
n_vars = 20
results_dir = 'results/suitcaselock/{}/'.format(alg)

if alg == 'astar':
    if n_vars == 20:
        n_values = 2
        transition_cap = 5e7
    else:
        n_values = 4
        transition_cap = 1e8
elif alg == 'gbfs':
    if n_vars == 20:
        n_values = 2
        transition_cap = 2.2e7
    else:
        n_values = 4
        transition_cap = 2e7

result_files = sorted(glob.glob(results_dir+'n_vars-{}/n_values-{}/entanglement-*/*.pickle'.format(n_vars, n_values)))
result_files = [r for r in result_files if 'entanglement-17' not in r]

curve_data = []
for filename in result_files:
# def generate_plot(filename, ax, color=None, label=None):
    with open(filename, 'rb') as f:
        try:
            search_results = pickle.load(f)
        except EOFError:
            continue
    states, actions, n_expanded, n_transitions, candidates = search_results[:5]
    seed = int(filename.split('/')[-1].split('.')[0].split('-')[-1])
    entanglement = int(filename.split('/')[-2].split('-')[-1])
    goal = states[0].reset().scramble(seed=seed+1000)
    n_errors = sum(states[-1].summarize_effects(baseline=goal)>0)
    x = [c for c,n in candidates]
    y = [n.h_score for c,n in candidates]

    # Extend final value to end of plot
    if n_errors > 0:
        x += [n_transitions]
        y += [y[-1]]
    if n_values == 2:
        # divide steps by 2, since we can ignore decrement actions
        x = [steps/2 for steps in x]

    [curve_data.append({'transitions': t, 'n_errors': err, 'seed': seed, 'entanglement': entanglement}) for t, err in zip(x,y)]

#%%
curve_data = pd.DataFrame(curve_data)
fig, ax = plt.subplots(figsize=(8,6))
lines = []
for i, k in enumerate(np.unique(curve_data['entanglement'])):
    sns.lineplot(data=curve_data.query('entanglement==@k'), x='transitions', y='n_errors', legend=False, estimator=None, units='seed', ax=ax, zorder=50-5*i)
    lines.append(ax.get_lines()[-1])
ax.legend(lines,np.unique(curve_data['entanglement']))
ax.set_title('Planning performance by entanglement ({} vars, {} values)'.format(n_vars, n_values))
ax.set_ylim([0,n_vars//2])
ax.set_xlim([0,transition_cap])
ax.set_xticklabels(np.asarray(ax.get_xticks())/1e6)
ax.hlines(n_vars,0,transition_cap,linestyles='dashed',linewidths=1)
ax.set_ylabel('Number of errors remaining')
ax.set_xlabel('Number of transitions considered (millions)')
plt.savefig('results/plots/suitcaselock/suitcaselock_planning_time.png')
plt.show()

#%%
solves = []
data = []
failures = []
len(states[0].actions())*n_vars
len(states[0].actions())*n_values**n_vars
max_transitions

for i,filename in enumerate(result_files):
    seed = int(filename.split('/')[-1].split('.')[0].split('-')[-1])
    k = int(filename.split('/')[-2].split('-')[-1])
    with open(filename, 'rb') as f:
        try:
            search_results = pickle.load(f)
        except EOFError:
            print('error reading', filename)
            failures.append((k, seed))
            continue
    states, actions, n_expanded, n_transitions, candidates = search_results
    goal = states[0].reset().scramble(seed=seed+1000)
    mean_entanglement = np.mean(np.sum(np.stack(states[0].actions()[:n_vars]),axis=-1))
    plan_length = len(actions)

    n_errors = sum(states[-1].summarize_effects(baseline=goal)>0)
    if n_errors == 0:
        solves.append(i)
    data.append({
        'transitions': n_transitions/2,
        'plan_length': plan_length,
        'k': k,
        'entanglement': mean_entanglement,
        'seed': seed,
        'n_errors': n_errors,
    })
data = pd.DataFrame(data)

#%%
print('Solve Counts')
print()
all_k_values = np.unique([int(filename.split('/')[-2].split('-')[-1]) for filename in result_files])
total_solves = 0
total_attempts = 0
failures = []
successes = []
for k in all_k_values:
    n_solves = len(data.query('(k==@k) and (n_errors==0) and (transitions < @transition_cap)'))
    n_attempts = len(data.query('k==@k'))
    failures += [(k, s) for s in sorted(list(data.query('(k==@k) and (n_errors>0)')['seed']))]
    successes += [(k, s) for s in sorted(list(data.query('(k==@k) and (n_errors==0)')['seed']))]
    total_solves += n_solves
    total_attempts += n_attempts

    print('{:2d}: {:3d} out of {:3d}'.format(k, n_solves, n_attempts))
print()
print('{:4d} out of {:4d}'.format(total_solves, total_attempts))
print(failures)

#%%
def as_range(iterable): # not sure how to do this part elegantly
    l = list(iterable)
    if len(l) > 1:
        return '{0}-{1}'.format(l[0], l[-1])
    else:
        return '{0}'.format(l[0])

print('Missing:')
for k in all_k_values:
    missing = [x for x in range(1,301) if x not in list(data.query('k==@k')['seed'])]
    missing_str = ','.join(as_range(g) for _, g in groupby(missing, key=lambda n, c=count(): n-next(c)))
    print('{:2d}: {}'.format(k, missing_str))

#%%
plt.rcParams.update({'font.size': 24})
for yscale_mode in ['linear']:
    fig, ax = plt.subplots(figsize=(8,6))
    sns.boxplot(x='k',y='transitions', data=data, color='C0', ax=ax)

    # sns.scatterplot(x='entanglement',y='transitions', data=data.query('n_errors==0'), color='C0', ax=ax)

    plt.xlabel('Variables modified per action')
    plt.ylabel('Simulator steps {}'.format(' (millions)' if yscale_mode=='linear' else ''))
    ax.set_yscale(yscale_mode)
    ax.set_yticklabels(np.asarray(ax.get_yticks())/1e6)
    # ax.set_xticks(np.arange(1,n_vars))
    # plt.title(r'Planning time vs. entanglement ($N_V={}$, $M={}$)'.format(n_vars, n_values))
    plt.tight_layout()
    plt.savefig('results/plots/suitcaselock/suitcase_{}ary.png'.format(n_values))
    plt.show()
