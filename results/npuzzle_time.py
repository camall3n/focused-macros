import copy
import glob
from itertools import groupby, count
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import pandas as pd
import seaborn as sns

n_puzzle = 15
results_dir = 'results/npuzzle/{}-puzzle/default_goal/'.format(n_puzzle)

transition_cap = 2e6
# n_vars, n_values, transition_cap = 15, 4, 20e6
# n_vars, n_values, transition_cap = 12, 4, 40e6

result_files = sorted(glob.glob(results_dir+'*/*.pickle'))
all_tags = ['primitive', 'random', 'generated']

#%%
curve_data = []
for filename in result_files:
    with open(filename, 'rb') as f:
        search_results = pickle.load(f)
    states, actions, n_expanded, n_transitions, candidates = search_results[:5]
    seed = int(filename.split('/')[-1].split('.')[0].split('-')[-1])
    tag = filename.split('/')[-2]
    if 'default_goal' in filename:
        goal = states[0].reset()
    else:
        goal = states[0].reset().scramble(seed=seed+1000)
    n_errors = len(states[-1].summarize_effects(baseline=goal)[0])
    x = [c for c,n in candidates]
    y = [n.h_score for c,n in candidates]

    # Extend final value to end of plot
    if n_errors > 0:
        x += [n_transitions]
        y += [y[-1]]

    [curve_data.append({'transitions': t, 'n_errors': e, 'seed': seed, 'tag': tag}) for t, e in zip(x,y)]

curve_data = pd.DataFrame(curve_data)
fig, ax = plt.subplots(figsize=(8,6))
sns.lineplot(data=curve_data, x='transitions', y='n_errors', hue='tag', hue_order=all_tags, palette=['C0','C2','C3'], estimator=None, units='seed', alpha=0.6, ax=ax)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[1:],labels[1:])
ax.hlines(n_puzzle+1,0,transition_cap,linestyles='dashed',linewidths=1)
ax.set_ylim([0,ax.get_ylim()[1]])
ax.set_xlim([0,transition_cap])
ax.set_xticklabels(np.asarray(ax.get_xticks())/1e6)
ax.set_xlabel('Number of transitions considered (millions)')
ax.set_ylabel('Number of errors remaining')
ax.set_title('Planning performance ({}-Puzzle)'.format(n_puzzle))
plt.savefig('results/plots/npuzzle/npuzzle_planning_time.png')
plt.show()

#%%
solves = []
data = []
for i,filename in enumerate(result_files):
    with open(filename, 'rb') as f:
        search_results = pickle.load(f)
    seed = int(filename.split('/')[-1].split('.')[0].split('-')[-1])
    tag = filename.split('/')[-2]
    states, actions, n_expanded, n_transitions, candidates = search_results
    if 'default_goal' in filename:
        goal = states[0].reset()
    else:
        goal = states[0].reset().scramble(seed=seed+1000)

    n_errors = len(states[-1].summarize_effects(baseline=goal)[0])
    if n_errors == 0:
        solves.append(i)
    data.append({
        'transitions': n_transitions,
        'tag': tag,
        'seed': seed,
        'n_errors': n_errors,
    })
data = pd.DataFrame(data)

#%%
print('Solve Counts')
print()
total_solves = 0
total_attempts = 0
for tag in all_tags:
    n_solves = len(data.query('(tag==@tag) and (n_errors==0) and (transitions < @transition_cap)'))
    n_attempts = len(data.query('tag==@tag'))
    total_solves += n_solves
    total_attempts += n_attempts

    print('{:10s} {:3d} out of {:3d}'.format(tag+':', n_solves, n_attempts))
print()
print('{:4d} out of {:4d}'.format(total_solves, total_attempts))

#%%
def as_range(iterable): # not sure how to do this part elegantly
    l = list(iterable)
    if len(l) > 1:
        return '{0}-{1}'.format(l[0], l[-1])
    else:
        return '{0}'.format(l[0])

print('Missing:')
for tag in all_tags:
    missing = [x for x in range(1,301) if x not in list(data.query('tag==@tag')['seed'])]
    missing_str = ','.join(as_range(g) for _, g in groupby(missing, key=lambda n, c=count(): n-next(c)))
    print('{:10s} {}'.format(tag+':', missing_str))

#%%
fig, ax = plt.subplots(figsize=(8,6))
sns.scatterplot(x='transitions', y='n_errors', data=data.groupby('tag', as_index=False).mean(), hue='tag', palette={'primitive':'C0','random':'C2','generated':'C3'}, hue_order=['primitive','random','generated'], style='tag', style_order=['primitive','random','generated'], markers=['o','^','P'], ax=ax, s=70)
ax.hlines(n_puzzle+1,0,transition_cap,linestyles='dashed',linewidths=1)
ax.set_xlim([0,transition_cap])
ax.set_xticklabels(list(map(lambda x: x/1e6,ax.get_xticks())))
ax.set_title('Mean final planning performance (15-Puzzle)')
ax.set_ylabel('Number of errors remaining')
ax.set_xlabel('Number of transitions considered (millions)')

handles, labels = ax.get_legend_handles_labels()
handles = handles[1:]
labels = ['actions only','actions + random skills', 'actions + generated skills']
ax.legend(handles=handles, labels=labels, framealpha=1, borderpad=0.7)
plt.savefig('results/plots/npuzzle/npuzzle_mean_planning_performance.png')
plt.show()
