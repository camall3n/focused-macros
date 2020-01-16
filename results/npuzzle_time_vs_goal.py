import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import seaborn as sns

default_results = sorted(glob.glob('results/npuzzle/15-puzzle/default_goal/'+'generated/*.pickle'))
random_results = sorted(glob.glob('results/npuzzle/15-puzzle/random_goal/'+'generated/*.pickle'))

#%%
transition_cap = 1e6
def generate_plot(filename, ax, color=None, label=None):
    with open(filename, 'rb') as f:
        search_results = pickle.load(f)
    seed = int(filename.split('/')[-1].split('.')[0].split('-')[-1])
    states, actions, n_expanded, n_transitions, candidates = search_results[:5]
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

    ax.plot(x,y,c=color,alpha=0.6, label=label)
    return n_errors

fig, ax = plt.subplots(figsize=(8,6))
ax.set_title('Planning performance (15-Puzzle)')
ax.set_ylim([0,16])
ax.set_xlim([0,transition_cap])
ax.set_xticklabels(list(map(lambda x: x/1e6,ax.get_xticks())))
ax.hlines(16,0,transition_cap,linestyles='dashed',linewidths=1)
ax.set_ylabel('Number of errors remaining')
ax.set_xlabel('Number of transitions considered (millions)')
# for i,f in enumerate(random_results):
#     label = None if i > 0 else 'actions + fixed random skills'
#     generate_plot(f, ax, 'C2', label=label)
for i,f in enumerate(random_results):
    label = None if i > 0 else 'random goals'
    generate_plot(f, ax, 'C4', label=label)
for i,f in enumerate(default_results):
    label = None if i > 0 else 'default goal'
    generate_plot(f, ax, 'C3', label=label)

ax.legend()
plt.savefig('results/plots/npuzzle/npuzzle_planning_alt_goals.png')
plt.show()

#%%
data = []
all_tags = ['default', 'random']
all_results = [default_results, random_results]
for tag, results in zip(all_tags, all_results):
    for i,filename in enumerate(results):
        with open(filename, 'rb') as f:
            search_results = pickle.load(f)
        seed = int(filename.split('/')[-1].split('.')[0].split('-')[-1])
        states, actions, n_expanded, n_transitions, candidates = search_results
        if 'default_goal' in filename:
            goal = states[0].reset()
        else:
            goal = states[0].reset().scramble(seed=seed+1000)

        n_errors = len(states[-1].summarize_effects(baseline=goal)[0])
        data.append({
            'tag': tag,
            'transitions': n_transitions,
            'seed': seed,
            'n_errors': n_errors,
        })
data = pd.DataFrame(data)

#%%
print('Solve Counts')
print()
for tag in all_tags:
    transition_cap = 1e6
    n_solves = len(data.query('(tag==@tag) and (n_errors==0) and (transitions < @transition_cap)'))
    n_attempts = len(data.query('tag==@tag'))

    print('{}: {} out of {}'.format( tag, n_solves, n_attempts))

#%%
with open(default_results[0], 'rb') as f:
    search_results = pickle.load(f)
    states, actions, n_expanded, n_transitions, candidates = search_results
    print(states[-1])

#%%
with open(random_results[0], 'rb') as f:
    search_results = pickle.load(f)
    states, actions, n_expanded, n_transitions, candidates = search_results
    print(states[-1])