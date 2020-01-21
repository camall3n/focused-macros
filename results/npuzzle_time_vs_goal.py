import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import seaborn as sns
import notebooks.picklefix

default_results = sorted(glob.glob('results/npuzzle/gbfs/15-puzzle/default_goal/'+'generated/*.pickle'))
random_results = sorted(glob.glob('results/npuzzle/gbfs/15-puzzle/random_goal/'+'generated/*.pickle'))

#%%
transition_cap = 1.5e4
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
ax.set_title('Planning performance by goal type (15-Puzzle)')
ax.hlines(16,0,transition_cap,linestyles='dashed',linewidths=1)
ax.set_ylim([0,ax.get_ylim()[1]])
ax.set_xlim([0,transition_cap])
ax.set_ylabel('Number of errors remaining')
ax.set_xlabel('Number of transitions considered')
lines, names = [], []
for i,f in enumerate(random_results):
    label = None if i > 0 else 'random goals'
    generate_plot(f, ax, 'C4', label=label)
lines.append(ax.get_lines()[-1])
names.append('random goals')
for i,f in enumerate(default_results):
    label = None if i > 0 else 'default goal'
    generate_plot(f, ax, 'C3', label=label)
lines.append(ax.get_lines()[-1])
names.append('default goal')
ax.legend(lines[::-1],names[::-1],framealpha=1, borderpad=0.7)
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
    med_transitions = data.query('(tag==@tag) and (n_errors==0)')['transitions']
    std_transitions = data.query('(tag==@tag) and (n_errors==0)').std()['transitions']

    print('{}: {} out of {} :: {}'.format( tag, n_solves, n_attempts, med_transitions))
#%%

sns.boxenplot(data=data.query('n_errors==0'), y='tag', x='transitions', palette=['C3','C4'], orient='h')
plt.title('Planning time by goal type (15-puzzle)')
plt.ylabel('')
plt.xlabel('Number of transitions (in thousands)')
plt.xlim([0,16000])
ax = plt.gca()
ax.set_xticklabels(list(map(lambda x: int(x/1e3),ax.get_xticks())))
ax.set_yticklabels(['default goal', 'random goals'])
plt.savefig('results/plots/npuzzle/npuzzle_planning_alt_goals_boxplot.png')
plt.show()

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
