import glob
from itertools import groupby, count
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import seaborn as sns

from domains import cube
from domains.cube import pattern
import notebooks.picklefix

default_results = sorted(glob.glob('results/cube/gbfs/default_goal/'+'generated-v0.4/*.pickle'))
random_results = sorted(glob.glob('results/cube/gbfs/random_goal/'+'generated-v0.4/*.pickle'))

#%%
def generate_plot(filename, ax, color=None, label=None):
    with open(filename, 'rb') as f:
        search_results = pickle.load(f)
    seed = int(filename.split('/')[-1].split('.')[0].split('-')[-1])
    states, actions, n_expanded, n_transitions, candidates = search_results[:5]
    if 'default_goal' in filename:
        goal = cube.Cube()
    else:
        goal = cube.Cube().apply(sequence=pattern.scramble(seed=seed+1000))

    n_errors = len(states[-1].summarize_effects(baseline=goal))
    x = [c for c,n in candidates]
    y = [n.h_score for c,n in candidates]

    # Extend final value to end of plot
    if n_errors > 0:
        x += [n_transitions]
        y += [y[-1]]

    ax.plot(x,y,c=color,alpha=0.6, label=label)
    return n_errors

fig, ax = plt.subplots(figsize=(8,6))
ax.set_title('Planning performance by goal type (Rubik\'s cube)')
ax.set_ylim([0,50])
ax.set_xlim([0,2e6])
ax.set_xticklabels(list(map(lambda x: x/1e6,ax.get_xticks())))
ax.hlines(48,0,2e6,linestyles='dashed',linewidths=1)
ax.set_ylabel('Number of errors remaining')
ax.set_xlabel('Number of simulation steps (in millions)')

for i,f in enumerate(default_results):
    label = None if i > 0 else 'default goal'
    generate_plot(f, ax, 'C3', label=label)
for i,f in enumerate(random_results):
    label = None if i > 0 else 'random goals'
    generate_plot(f, ax, 'C4', label=label)

ax.legend(framealpha=1)
plt.savefig('results/plots/cube/cube_planning_alt_goals.png')
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
            goal = cube.Cube()
        else:
            goal = cube.Cube().apply(sequence=pattern.scramble(seed=seed+1000))

        n_errors = len(states[-1].summarize_effects(baseline=goal))
        data.append({
            'tag': tag,
            'transitions': n_transitions,
            'seed': seed,
            'n_errors': n_errors,
        })
data = pd.DataFrame(data)
data.groupby('tag', as_index=False).mean()
#%%
sns.boxenplot(data=data.query('n_errors==0'), y='tag', x='transitions', palette=['C3','C4'], orient='h')
plt.ylabel('')
plt.xlim([-2000,1000000])
# plt.gca().invert_yaxis()
plt.title('Planning time by goal type (Rubik\'s cube)')
plt.ylabel('')
ax = plt.gca()
ax.set_xticklabels(list(map(lambda x: x/1e6,ax.get_xticks())))
plt.xlabel('Number of simulation steps (in millions)')
ax.set_yticklabels(['default goal', 'random goals'])
plt.savefig('results/plots/cube/cube_planning_time_boxplot.png')
plt.show()

#%%
print('Solve Counts')
print()
for tag in all_tags:
    transition_cap = 1e6
    n_solves = len(data.query('(tag==@tag) and (n_errors==0) and (transitions < @transition_cap)'))
    n_attempts = len(data.query('tag==@tag'))

    print('{}: {} out of {}'.format( tag, n_solves, n_attempts))

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
