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

results_dir = 'results/cube/gbfs/default_goal/'
version = 'v0.4'
result_files = sorted(glob.glob(results_dir+'*/*.pickle'))
transition_cap = 2e6

curve_data = []
for filename in result_files:
# def generate_plot(filename, ax, color=None, label=None):
    with open(filename, 'rb') as f:
        try:
            search_results = pickle.load(f)
        except EOFError:
            print('error reading', filename)
            continue
    states, actions, n_expanded, n_transitions, candidates = search_results[:5]
    seed = int(filename.split('/')[-1].split('.')[0].split('-')[-1])
    tag = filename.split('/')[-2].split('-')[0]
    if tag == 'full_random':
        tag = 'random'
    v = filename.split('/')[-2].split('-')[-1] if tag == 'generated' else version
    if 'default_goal' in filename:
        goal = cube.Cube()
    else:
        goal = cube.Cube().apply(pattern.scramble(seed=seed+1000))
    n_errors = len(states[-1].summarize_effects(baseline=goal))
    x = [transitions for transitions, node in candidates]
    y = [node.h_score for transitions, node in candidates]

    # Extend final value to end of plot
    if n_errors > 0:
        x += [n_transitions]
        y += [y[-1]]
    [curve_data.append({'transitions': t, 'n_errors': err, 'seed': seed, 'tag': tag, 'version': v}) for t, err in zip(x,y)]
curve_data = pd.DataFrame(curve_data).query("version==@version")
#%%
plt.rcParams.update({'font.size': 18})
fig, ax = plt.subplots(figsize=(8,6))
lines = []
names = []

plot_vars = [
    {'tag':'random', 'desc':'random', 'color': 'C2', 'zorder': 5},
    {'tag':'primitive', 'desc':'primitive', 'color': 'C0', 'zorder': 15},
    {'tag':'generated', 'desc':'learned', 'color': 'C3', 'zorder': 10},
    {'tag':'expert', 'desc':'expert', 'color': 'C1', 'zorder': 20},
]
for plot_dict in plot_vars:
    tag = plot_dict['tag']
    desc = plot_dict['desc']
    c = plot_dict['color']
    z = plot_dict['zorder']
    if len(curve_data.query('tag==@tag')) > 0:
        sns.lineplot(data=curve_data.query('tag==@tag'), x='transitions', y='n_errors', legend=False, estimator=None, units='seed', ax=ax, linewidth=2, alpha=.6, color=c, zorder=z)
        lines.append(ax.get_lines()[-1])
        names.append(desc)
# lines, names = zip(*[(l, d['desc']) for d,l in zip(plot_vars,lines)])
ax.legend(lines,names,framealpha=1, borderpad=0.7)
# ax.set_title('Planning performance by action/macro-action type (Rubik\'s cube)')
ax.set_ylim([0,50])
ax.set_xlim([0,transition_cap])
ax.set_xticklabels(np.asarray(ax.get_xticks())/1e6)
ax.set_axisbelow(False)
# [i.set_linewidth(1) for i in ax.spines.values()]
ax.hlines(48,0,transition_cap,linestyles='dashed',linewidths=2)
ax.set_ylabel('Number of errors remaining')
ax.set_xlabel('Simulator steps (in millions)')
plt.savefig('results/plots/cube/cube_planning.png')
plt.show()

#%%
data = []
for filename in result_files:
    with open(filename, 'rb') as f:
        try:
            search_results = pickle.load(f)
        except EOFError:
            print('error reading', filename)
            continue
    states, actions, n_expanded, n_transitions, candidates = search_results[:5]
    seed = int(filename.split('/')[-1].split('.')[0].split('-')[-1])
    tag = filename.split('/')[-2].split('-')[0]
    if tag == 'full_random':
        tag = 'random'
    v = filename.split('/')[-2].split('-')[-1] if tag == 'generated' else 'v'+version
    if 'default_goal' in filename:
        goal = cube.Cube()
    else:
        goal = cube.Cube().apply(pattern.scramble(seed=seed+1000))
    n_errors = len(states[-1].summarize_effects(baseline=goal))
    n_action_steps = len(np.concatenate(actions))
    n_macro_steps = len(actions)

    data.append({'transitions': n_transitions, 'n_errors': n_errors, 'n_action_steps': n_action_steps, 'n_macro_steps': n_macro_steps, 'seed': seed, 'tag': tag, 'version': v})

data = pd.DataFrame(data).query("version!='v0.2'")
data.groupby('tag', as_index=False).mean()
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

#%%
fig, ax = plt.subplots(figsize=(8,6))
plot_vars = [
    # {'tag':'primitive', 'desc':'actions only', 'color': 'C0', 'marker': 'o', 'zorder': 15},
    # {'tag':'random', 'desc':'actions + random macros', 'color': 'C2', 'marker': '^', 'zorder': 5},
    {'tag':'expert', 'desc':'actions + expert macros', 'color': 'C1', 'marker': 'X', 'zorder': 20},
    {'tag':'generated', 'desc':'actions + learned macros', 'color': 'C3', 'marker': 'P', 'zorder': 10},
]
for plot_dict in plot_vars:
    tag = plot_dict['tag']
    desc = plot_dict['desc']
    c = plot_dict['color']
    z = plot_dict['zorder']
    marker = plot_dict['marker']
    if len(data.query('tag==@tag')) > 0:
        sns.scatterplot(data=data.query('tag==@tag'), x='n_macro_steps', y='transitions', label=desc, estimator=None, units='seed', ax=ax, color=c, marker=marker, s=150, zorder=z)

ax.set_ylim([0,8e5])
ax.set_xlim([0,120])
ax.set_title('Solution speed vs. length (Rubik\'s cube)')
ax.set_ylabel('Number of simulation steps')
ax.set_xlabel('Plan length (macro-action steps)')
plt.savefig('results/plots/cube/cube_plan_length_macros.png')
plt.show()

data.query("tag=='generated'").mean()['n_action_steps']
data.query("tag=='expert'").mean()['n_action_steps']
#%%
fig, ax = plt.subplots(figsize=(8,6))
plot_vars = [
    # {'tag':'primitive', 'desc':'actions only', 'color': 'C0', 'marker': 'o', 'zorder': 15},
    # {'tag':'random', 'desc':'actions + random macros', 'color': 'C2', 'marker': '^', 'zorder': 5},
    {'tag':'expert', 'desc':'actions + expert macros', 'color': 'C1', 'marker': 'X', 'zorder': 20},
    {'tag':'generated', 'desc':'actions + learned macros', 'color': 'C3', 'marker': 'P', 'zorder': 10},
]
for plot_dict in plot_vars:
    tag = plot_dict['tag']
    desc = plot_dict['desc']
    c = plot_dict['color']
    z = plot_dict['zorder']
    marker = plot_dict['marker']
    if len(data.query('tag==@tag')) > 0:
        sns.scatterplot(data=data.query('tag==@tag'), x='transitions', y='n_action_steps', label=desc, estimator=None, units='seed', ax=ax, color=c, marker=marker, s=150, zorder=z)
ax.set_ylim([0,1200])
ax.set_xlim([0,1e6])
ax.legend(loc='upper left')
ax.set_title('Solution length vs. planning time (Rubik\'s cube)')
ax.set_xlabel('Number of simulation steps')
ax.set_ylabel('Plan length (primitive action steps)')
plt.savefig('results/plots/cube/cube_plan_length_actions.png')
plt.show()

#%%
data = []
for filename in result_files:
# def generate_plot(filename, ax, color=None, label=None):
    with open(filename, 'rb') as f:
        try:
            search_results = pickle.load(f)
        except EOFError:
            continue
    states, actions, n_expanded, n_transitions, candidates = search_results[:5]
    seed = int(filename.split('/')[-1].split('.')[0].split('-')[-1])
    tag = filename.split('/')[-2].split('-')[0]
    if tag == 'full_random':
        tag = 'random'
    v = filename.split('/')[-2].split('-')[-1] if tag == 'generated' else 'v'+version
    if 'default_goal' in filename:
        goal = cube.Cube()
    else:
        goal = cube.Cube().apply(pattern.scramble(seed=seed+1000))
    n_errors = len(states[-1].summarize_effects(baseline=goal))
    n_action_steps = len(np.concatenate(actions))
    n_macro_steps = len(actions)
    macro_lengths = list(map(len,actions))
    x = [c for c,n in candidates]
    y = [n.h_score for c,n in candidates]

    # Extend final value to end of plot
    if n_errors > 0:
        x += [n_transitions]
        y += [y[-1]]
    [data.append({'transitions': n_transitions, 'n_errors': n_errors, 'n_action_steps': n_action_steps, 'n_macro_steps': n_macro_steps, 'macro_length': l, 'seed': seed, 'tag': tag, 'version': v}) for l in macro_lengths]

data = pd.DataFrame(data).query("version!='v0.2'")

fig, ax = plt.subplots(figsize=(8,6))
sns.violinplot(x='tag', y='macro_length', data=data, hue='tag', palette={'primitive':'C0','expert':'C1','random':'C2','generated':'C3'}, hue_order=['primitive','expert','random','generated'], style='tag', style_order=['primitive','expert','random','generated'], ax=ax, cut=0, inner=None, dodge=False)

ax.set_title('Macro length distribution (Rubik\'s cube)')
ax.set_ylabel('Length of macro (number of primitive actions)')
ax.set_xlabel('Macro type')

plt.savefig('results/plots/cube/cube_macro_length.png')
plt.show()
