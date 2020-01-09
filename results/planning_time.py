import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import seaborn as sns

from cube import cube
from cube import pattern
from util import rsync

# rsync(source='brown:~/dev/skills-for-planning/results/planning',
      # dest='results/')
results_dir = 'results/planning/default_goal/'
primitive_results = glob.glob(results_dir+'primitive/*.pickle')
expert_results = glob.glob(results_dir+'expert/*.pickle')
random_results = glob.glob(results_dir+'random/*.pickle')
full_random_results = glob.glob(results_dir+'full_random/*.pickle')
gen_version = '0.4'
generated_results = glob.glob(results_dir+'generated-v'+gen_version+'/*.pickle')

#%%
def generate_plot(filename, ax, color=None, label=None, shifted_by=0):
    with open(filename, 'rb') as f:
        search_results = pickle.load(f)
    states, actions, n_expanded, n_transitions, candidates = search_results[:5]

    n_errors = len(states[-1].summarize_effects())
    x = [c+shifted_by for c,n in candidates]
    y = [n.h_score for c,n in candidates]
    if shifted_by > 0:
        x = [0] + x
        y = [y[0]] + y

    # Extend final value to end of plot
    if n_errors > 0:
        x += [n_transitions]
        y += [y[-1]]

    ax.plot(x,y,c=color,alpha=0.6, linewidth=2, label=label)
    return n_errors

fig, ax = plt.subplots(figsize=(8,6))
ax.set_title('Planning performance')
ax.set_ylim([0,50])
ax.set_xlim([0,2e6])
# labels = ax.get_xticks()
ax.set_xticklabels(list(map(lambda x: x/1e6,ax.get_xticks())))
ax.hlines(48,0,2e6,linestyles='dashed',linewidths=1)
ax.set_ylabel('Number of errors remaining')
ax.set_xlabel('Number of transitions considered (millions)')
# for i,f in enumerate(random_results):
#     label = None if i > 0 else 'actions + fixed random skills'
#     generate_plot(f, ax, 'C2', label=label)
for i,f in enumerate(full_random_results):
    label = None if i > 0 else 'actions + random skills'
    generate_plot(f, ax, 'C2', label=label)
for i,f in enumerate(primitive_results):
    label = None if i > 0 else 'actions only'
    generate_plot(f, ax, 'C0', label=label)
for i,f in enumerate(generated_results):
    label = None if i > 0 else 'actions + generated skills'
    generate_plot(f, ax, 'C3', label=label)
for i,f in enumerate(expert_results):
    label = None if i > 0 else 'actions + expert skills'
    generate_plot(f, ax, 'C1', label=label)

handles,labels = ax.get_legend_handles_labels()
handles = [handles[1], handles[3], handles[0], handles[2]]
labels = [labels[1], labels[3], labels[0], labels[2]]
ax.legend(handles, labels, framealpha=1, borderpad=0.7)
plt.savefig('results/plots/planning_time_actions.png')
plt.show()

#%%
data = []
all_tags = ['primitive', 'expert', 'random', 'generated']
all_results = [primitive_results, expert_results, full_random_results, generated_results]
for tag, results in zip(all_tags, all_results):
    for i,filename in enumerate(results):
        with open(filename, 'rb') as f:
            search_results = pickle.load(f)
        seed = int(filename.split('/')[-1].split('.')[0].split('-')[-1])
        states, actions, n_expanded, n_transitions, candidates = search_results
        if 'default_goal' in filename:
            goal = cube.Cube()
        else:
            goal = cube.Cube().apply(pattern.scramble(seed=seed+1000))

        n_errors = len(states[-1].summarize_effects(baseline=goal))
        if n_errors == 0:
            solves.append(i)
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
fig, ax = plt.subplots(figsize=(8,6))
sns.violinplot(x='tag',y='n_errors', data=data, units='seed', cut=0, inner=None, ax=ax)
ax.set_ylim([0,50])
ax.set_xlim(-.5, len(all_tags)-0.5)
ax.hlines(48,-1,10,linestyles='dashed',linewidths=1)
ax.set_xlabel('')
plt.show()

#%%
fig, ax = plt.subplots(figsize=(8,6))
sns.violinplot(x='transitions',y='tag', data=data, ax=ax, scale='width', cut=0, inner=None)
# ax.set_title('Planning performance')
# ax.set_ylim([0,50])
# ax.set_xlim([0,2e6])
# labels = ax.get_xticks()
ax.set_xticklabels(list(map(lambda x: x/1e6,ax.get_xticks())))
# ax.hlines(48,0,2e6,linestyles='dashed',linewidths=1)
# ax.set_ylabel('Number of errors remaining')
# ax.set_xlabel('Number of transitions considered (millions)')
plt.ylabel('')
plt.show()

#%%
fig, ax = plt.subplots(figsize=(8,6))
sns.scatterplot(x='transitions', y='n_errors', data=data.groupby('tag', as_index=False).median(), hue='tag', hue_order=['primitive','expert','random','generated'], style='tag', style_order=['primitive','expert','random','generated'], markers=['o','X','^','P'], ax=ax, s=150)
ax.set_xticklabels(list(map(lambda x: x/1e6,ax.get_xticks())))
ax.hlines(48,-0.05e6,2.05e6,linestyles='dashed',linewidths=1)
ax.set_xlim([-0.05e6,2.05e6])
ax.set_xticklabels(list(map(lambda x: x/1e6,ax.get_xticks())))
ax.set_title('Median final planning performance')
ax.set_ylabel('Number of errors remaining')
ax.set_xlabel('Number of transitions considered (millions)')
handles, labels = ax.get_legend_handles_labels()
handles = handles[1:]
labels = ['actions only','actions + expert skills', 'actions + random skills', 'actions + generated skills']
ax.legend(handles=handles, labels=labels, framealpha=1, borderpad=0.7)
plt.show()

#%%
fig, ax = plt.subplots()
sns.scatterplot(x='transitions', y='n_errors', data=data.groupby('tag', as_index=False).mean(), hue='tag', hue_order=['primitive','expert','random','generated'], style='tag', style_order=['primitive','expert','random','generated'], markers=['o','X','^','P'], ax=ax, s=70)
ax.hlines(48,-0.05e6,2.05e6,linestyles='dashed',linewidths=1)
ax.set_xlim([-0.05e6,2.05e6])
ax.set_xticklabels(list(map(lambda x: x/1e6,ax.get_xticks())))
ax.set_title('Mean final planning performance')
ax.set_ylabel('Number of errors remaining')
ax.set_xlabel('Number of transitions considered (millions)')

handles, labels = ax.get_legend_handles_labels()
handles = handles[1:]
labels = ['actions only','actions + expert skills', 'actions + random skills', 'actions + generated skills']
ax.legend(handles=handles, labels=labels, framealpha=1, borderpad=0.7)
plt.savefig('results/plots/mean_planning_performance.png')
plt.show()

#%%
# render the cubes where expert skills failed to solve
for i,filename in enumerate(generated_results):
    seed = int(filename.split('/')[-1].split('.')[0].split('-')[-1])
    if seed not in list(data.query('(tag == "generated") and (n_errors > 0 )')['seed']):
        continue
    with open(filename, 'rb') as f:
        search_results = pickle.load(f)
    states, actions, n_expanded, n_transitions, candidates = search_results

    states[-1].render()
    # results_dir = 'results/cube_deadends'
    # os.makedirs(results_dir, exist_ok=True)
    # with open(results_dir+'/seed-{:03d}.pickle'.format(seed), 'wb') as f:
    #     pickle.dump(states[-1], f)

#%%
sorted(list(data.query('(tag=="generated") and (n_errors>0)')['seed']))
