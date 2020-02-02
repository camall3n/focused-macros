import copy
import glob
from itertools import groupby, count
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import pandas as pd
import seaborn as sns
import notebooks.picklefix  # pylint: disable=W0611

n_puzzle = 15
alg = 'gbfs'
results_dir = 'results/npuzzle/{}/{}-puzzle/default_goal/'.format(alg,n_puzzle)

if 'astar' in alg:
    transition_cap = 2e6
elif alg == 'gbfs':
    transition_cap = 5e5
# n_vars, n_values, transition_cap = 15, 4, 20e6
# n_vars, n_values, transition_cap = 12, 4, 40e6

result_files = sorted(glob.glob(results_dir+'*/*.pickle'))
all_tags = ['primitive', 'random', 'learned']

#%%
curve_data = []
for filename in result_files:
    with open(filename, 'rb') as f:
        try:
            search_results = pickle.load(f)
        except EOFError:
            continue
    states, actions, n_expanded, n_transitions, candidates = search_results[:5]
    seed = int(filename.split('/')[-1].split('.')[0].split('-')[-1])
    tag = filename.split('/')[-2]
    if 'default_goal' in filename:
        goal = states[0].reset()
    else:
        goal = states[0].reset().scramble(seed=seed+1000)
    n_errors = len(states[-1].summarize_effects(baseline=goal)[0])
    x = [c for c, n in candidates]
    y = [n.h_score for c, n in candidates]

    # Extend final value to end of plot
    if n_errors > 0:
        x += [n_transitions]
        y += [y[-1]]

    [curve_data.append({'transitions': t, 'n_errors': e, 'seed': seed, 'tag': tag}) for t, e in zip(x,y)]
curve_data = pd.DataFrame(curve_data)
#%%
fig, ax = plt.subplots(figsize=(8,6))
lines = []
names = []

plot_vars = [
    {'tag':'primitive', 'desc':'actions only', 'color': 'C0', 'zorder': 10},
    {'tag':'random', 'desc':'actions + random macros', 'color': 'C2', 'zorder': 5},
    {'tag':'learned', 'desc':'actions + learned macros', 'color': 'C3', 'zorder': 15},
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
# lines, names = zip(*[(l, d['desc']) for d, l in zip(plot_vars,lines)])
ax.legend(lines,names,framealpha=1, borderpad=0.7)
ax.hlines(n_puzzle+1,0,transition_cap,linestyles='dashed',linewidths=1)
ax.set_ylim([0,ax.get_ylim()[1]])
ax.set_xlim([0,transition_cap])
# ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
# ax.set_xticklabels(np.asarray(ax.get_xticks()))
ax.set_ylabel('Number of errors remaining')
ax.set_xlabel('Number of simulation steps')
ax.set_title('Planning performance ({}-Puzzle)'.format(n_puzzle))
plt.savefig('results/plots/npuzzle/npuzzle_planning_time.png')
plt.show()
#%%
solves = []
data = []
for i, filename in enumerate(result_files):
    with open(filename, 'rb') as f:
        try:
            search_results = pickle.load(f)
        except EOFError:
            continue
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
plt.figure()
plt.plot(0,0,c='C2',label='random', lw=3)
plt.plot(0,0,c='C0',label='primitive', lw=3)
plt.plot(0,0,c='C3',label='learned', lw=3)
plt.legend()
handles, labels = plt.gca().get_legend_handles_labels()
plt.show()
plt.close()

plt.rcParams.update({'font.size': 18, 'figure.figsize': (8,6)})

g = sns.catplot(data=data.query('n_errors==0'), y='tag', x='transitions', kind='boxen', palette=['C3','C0','C2'], orient='h', legend='True')
g.despine(right=False, top=False)
# plt.title('Planning time by action/macro-action type (15-puzzle)')
plt.ylabel('Macro-action type')
plt.xlabel('Simulator steps (in thousands)')
plt.gcf().set_size_inches(8,6)
plt.tight_layout()
plt.xlim([-100,500000])
ax = plt.gca()
ax.invert_yaxis()
ax.set_xticklabels(map(int,np.asarray(ax.get_xticks(),dtype=int)//1e3))
ax.set_yticklabels([])
plt.tight_layout()
# lines = ax.get_lines()
# for i, c in enumerate(['C2','C0','C3']):
#     lines[i].set_color(c)
#     lines[i].set_alpha(1.)
ax.legend(handles, labels,  loc='lower right')
plt.gcf().savefig('results/plots/npuzzle/npuzzle_planning_time.png')
plt.show()

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
data = []
for filename in result_files:
    with open(filename, 'rb') as f:
        try:
            search_results = pickle.load(f)
        except EOFError:
            continue
    states, actions, n_expanded, n_transitions, candidates = search_results[:5]
    seed = int(filename.split('/')[-1].split('.')[0].split('-')[-1])
    tag = filename.split('/')[-2]
    if 'default_goal' in filename:
        goal = states[0].reset()
    else:
        goal = states[0].reset().scramble(seed=seed+1000)
    n_errors = len(states[-1].summarize_effects(baseline=goal)[0])
    n_action_steps = len(np.concatenate(actions))
    n_macro_steps = len(actions)
    x = [c for c, n in candidates]
    y = [n.h_score for c, n in candidates]

    # Extend final value to end of plot
    if n_errors > 0:
        x += [n_transitions]
        y += [y[-1]]

    data.append({
        'n_action_steps': n_action_steps,
        'n_macro_steps': n_macro_steps,
        'n_errors': n_errors,
        'seed': seed,
        'tag': tag
    })

data = pd.DataFrame(data)

#%%
fig, ax = plt.subplots(figsize=(8,6))
sns.scatterplot(x='n_action_steps', y='n_errors', data=data, hue='tag', palette={'primitive':'C0','random':'C2','learned':'C3'}, hue_order=['primitive','random','learned'], style='tag', style_order=['primitive','random','learned'], markers=['o','^','P'], ax=ax, s=150)
ax.set_ylim([0,n_puzzle+1])
ax.set_title('Final plan quality vs. length (15-Puzzle)')
ax.set_ylabel('Number of errors remaining')
ax.set_xlabel('Plan length (primitive action steps)')

handles, labels = ax.get_legend_handles_labels()
handles = handles[1:]
labels = ['actions only','actions + random macros', 'actions + learned macros']
ax.legend(handles=handles, labels=labels, framealpha=1, borderpad=0.7)
plt.savefig('results/plots/npuzzle/npuzzle_plan_length_actions.png')
plt.show()

#%%
fig, ax = plt.subplots(figsize=(8,6))
sns.scatterplot(x='n_macro_steps', y='n_errors', data=data, hue='tag', palette={'primitive':'C0','random':'C2','learned':'C3'}, hue_order=['primitive','random','learned'], style='tag', style_order=['primitive','random','learned'], markers=['o','^','P'], ax=ax, s=150)
ax.set_ylim([0,n_puzzle+1])
ax.set_title('Final plan quality vs. length (15-Puzzle)')
ax.set_ylabel('Number of errors remaining')
ax.set_xlabel('Plan length (macro-action steps)')

handles, labels = ax.get_legend_handles_labels()
handles = handles[1:]
labels = ['actions only','actions + random macros', 'actions + learned macros']
ax.legend(handles=handles, labels=labels, framealpha=1, borderpad=0.7)
plt.savefig('results/plots/npuzzle/npuzzle_plan_length_macros.png')
plt.show()

#%%
data = []
for filename in result_files:
    with open(filename, 'rb') as f:
        try:
            search_results = pickle.load(f)
        except EOFError:
            continue
    states, actions, n_expanded, n_transitions, candidates = search_results[:5]
    seed = int(filename.split('/')[-1].split('.')[0].split('-')[-1])
    tag = filename.split('/')[-2]
    if 'default_goal' in filename:
        goal = states[0].reset()
    else:
        goal = states[0].reset().scramble(seed=seed+1000)
    n_errors = len(states[-1].summarize_effects(baseline=goal)[0])
    n_action_steps = len(np.concatenate(actions))
    n_macro_steps = len(actions)
    macro_lengths = list(map(len, actions))
    x = [c for c, n in candidates]
    y = [n.h_score for c, n in candidates]

    # Extend final value to end of plot
    if n_errors > 0:
        x += [n_transitions]
        y += [y[-1]]

    [data.append({
        'n_action_steps': n_action_steps,
        'n_macro_steps': n_macro_steps,
        'n_errors': n_errors,
        'macro_length': l,
        'seed': seed,
        'tag': tag
    }) for l in macro_lengths]

data = pd.DataFrame(data)

fig, ax = plt.subplots(figsize=(8,6))
sns.violinplot(x='tag', y='macro_length', data=data, hue='tag', palette={'primitive':'C0','random':'C2','learned':'C3'}, hue_order=['primitive','random','learned'], style='tag', style_order=['primitive','random','learned'], ax=ax, cut=0, inner=None, dodge=False)

ax.legend(loc='upper center')
ax.set_title('Macro length distribution (15-puzzle)')
ax.set_ylabel('Macro length (primitive actions)')
ax.set_xlabel('Macro type')

plt.savefig('results/plots/npuzzle/npuzzle_macro_length.png')
plt.show()
