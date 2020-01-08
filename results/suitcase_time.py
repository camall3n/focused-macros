import glob
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import pandas as pd
import seaborn as sns

from util import rsync

# rsync(source='brown:~/dev/skills-for-planning/results/planning',
      # dest='results/')
results_dir = 'results/suitcaselock/'
primitive_results = sorted(glob.glob(results_dir+'n_vars-15/n_values-4/max_vars-*/*.pickle'))
# expert_results = glob.glob(results_dir+'expert/*.pickle')
# random_results = glob.glob(results_dir+'random/*.pickle')
# full_random_results = glob.glob(results_dir+'full_random/*.pickle')
# gen_version = '0.2'
# generated_results = glob.glob(results_dir+'generated-v'+gen_version+'/*.pickle')


def generate_plot(filename, ax, color=None, label=None):
    with open(filename, 'rb') as f:
        search_results = pickle.load(f)
    states, actions, n_expanded, n_transitions, candidates = search_results[:5]
    seed = int(filename.split('/')[-1].split('.')[0].split('-')[-1])
    goal = states[0].reset().scramble(seed=seed+1000)
    n_errors = sum(states[-1].summarize_effects(baseline=goal)>0)
    x = [c for c,n in candidates]
    y = [n.h_score for c,n in candidates]

    # Extend final value to end of plot
    if n_errors > 0:
        x += [n_transitions]
        y += [y[-1]]

    ax.plot(x,y,c=color,alpha=0.6, label=label)
    return n_errors

fig, ax = plt.subplots(figsize=(8,6))
ax.set_title('Planning performance')
ax.set_ylim([0,8])
ax.set_xlim([0,20e6])
ax.hlines(8,0,1e5,linestyles='dashed',linewidths=1)
ax.set_ylabel('Number of errors remaining')
ax.set_xlabel('Number of transitions considered')
# for i,f in enumerate(random_results):
#     label = None if i > 0 else 'actions + fixed random skills'
#     generate_plot(f, ax, 'C2', label=label)
# for i,f in enumerate(full_random_results):
#     label = None if i > 0 else 'actions + random skills'
#     generate_plot(f, ax, 'C2', label=label)
for i,f in enumerate(primitive_results):
    seed = int(f.split('/')[-1].split('.')[0].split('-')[-1])
    max_vars = int(f.split('/')[4].split('-')[-1])
    label = None if seed > 1 else str(max_vars)
    color = 'C{}'.format(max_vars-1)
    generate_plot(f, ax, color, label=label)
# for i,f in enumerate(expert_results):
#     label = None if i > 0 else 'actions + expert skills'
#     generate_plot(f, ax, 'C1', label=label)
# for i,f in enumerate(generated_results):
#     label = None if i > 0 else 'actions + generated v{}'.format(gen_version)
#     generate_plot(f, ax, 'C3', label=label)
ax.legend()
# plt.savefig('results/plots/planning_time_v{}.png'.format(gen_version))
plt.show()

#%%
solves = []
data = []
for i,filename in enumerate(primitive_results):
    with open(filename, 'rb') as f:
        search_results = pickle.load(f)
    seed = int(filename.split('/')[-1].split('.')[0].split('-')[-1])
    max_vars = int(filename.split('/')[4].split('-')[-1])
    states, actions, n_expanded, n_transitions, candidates = search_results
    goal = states[0].reset().scramble(seed=seed+1000)

    n_errors = sum(states[-1].summarize_effects(baseline=goal)>0)
    if n_errors == 0:
        solves.append(i)
    data.append({
        'transitions': n_transitions,
        'max_vars': max_vars,
        'seed': seed,
        'n_errors': n_errors,
    })
data = pd.DataFrame(data)

#%%
print('Solve Counts')
print()
all_k_values = np.unique([int(filename.split('/')[-2].split('-')[-1]) for filename in primitive_results])
for k in all_k_values:
    transition_cap = 2e6
    n_solves = len(data.query('(max_vars==@k) and (n_errors==0) and (transitions < @transition_cap)'))
    n_attempts = len(data.query('max_vars==@k'))

    print('{:2d}: {} out of {}'.format(k, n_solves, n_attempts))

for k in all_k_values:
    missing = [x for x in range(1,101) if x not in list(data.query('max_vars==@k')['seed'])]
    print('{:2d}: {}'.format(k, missing))

#%%
fig, ax = plt.subplots()
sns.pointplot(x='max_vars',y='transitions', data=data, units='seed', join=False, estimator=np.mean, color='C0', ax=ax)
sns.pointplot(x='max_vars',y='transitions', data=data, units='seed', join=False, estimator=np.median, color='C1', ax=ax)
plt.legend(handles=ax.lines[::6],labels=['mean', 'median'], loc='lower right')
plt.xlabel('Max number of variables changed per action')
plt.ylabel('Number of transitions considered')
# plt.ylim([0,1e5])
plt.title('Planning Time vs. Effect Size')
# plt.yscale('log')
plt.tight_layout()
plt.savefig('results/plots/planning_time_vs_effect_size.png')
plt.show()
data['transitions'].max()
print(solves)
print(len(solves), 'out of', len(primitive_results))
#%%
sns.violinplot(x='max_vars',y='n_errors', data=data, units='seed', cut=0, inner=None, scale='count')
#%%
sns.violinplot(y='max_vars',x='transitions', data=data, orient='h', cut=0, inner=None, scale='count')
ax = plt.gca()
ax.set_xticklabels(list(map(lambda x: x/1e6,ax.get_xticks())))
ax.set_xlabel('transitions (millions)')
plt.show()

#%%

sns.lmplot(x='transitions', y='n_errors', data=data.groupby('max_vars', as_index=False).mean(), hue='max_vars', markers=["o", "x", "1", "+", "s"], fit_reg=False, legend=False, height=6, scatter_kws={"s": 70})
# fig = plt.gcf()
# fig.set_size_inches(8,6)
ax = plt.gca()
# ax.hlines(48,-0.05e6,2.05e6,linestyles='dashed',linewidths=1)
ax.set_ylim([0,ax.get_ylim()[1]])
ax.set_xlim([0,20e6])
ax.set_xticklabels(list(map(lambda x: int(x/1e3),ax.get_xticks())))
ax.set_title('Mean planning performance')
ax.set_ylabel('Number of errors')
ax.set_xlabel('Number of transitions (thousands)')
plt.legend(loc='lower right')
plt.show()
