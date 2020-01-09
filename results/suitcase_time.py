import glob
from itertools import groupby, count
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

n_vars, n_values, transition_cap = 20, 2, 2e6
# n_vars, n_values, transition_cap = 15, 4, 20e6
# n_vars, n_values, transition_cap = 12, 4, 40e6

primitive_results = sorted(glob.glob(results_dir+'n_vars-{}/n_values-{}/max_vars-*/*.pickle'.format(n_vars, n_values)))

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
ax.set_xlim([0,transition_cap])
ax.hlines(8,0,transition_cap,linestyles='dashed',linewidths=1)
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
total_solves = 0
total_attempts = 0
for k in all_k_values:
    n_solves = len(data.query('(max_vars==@k) and (n_errors==0) and (transitions < @transition_cap)'))
    n_attempts = len(data.query('max_vars==@k'))
    total_solves += n_solves
    total_attempts += n_attempts

    print('{:2d}: {:3d} out of {:3d}'.format(k, n_solves, n_attempts))
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
for k in all_k_values:
    missing = [x for x in range(1,301) if x not in list(data.query('max_vars==@k')['seed'])]
    missing_str = ','.join(as_range(g) for _, g in groupby(missing, key=lambda n, c=count(): n-next(c)))
    print('{:2d}: {}'.format(k, missing_str))

#%%
max_entanglement = list(data.groupby('max_vars',as_index=False).mean()['max_vars'])
transitions = data.groupby('max_vars',as_index=False).mean()['transitions']

mean_entanglement = [np.mean(list(range(1,k+1))+[k]*(n_vars-k)) for k in max_entanglement]

fig, ax = plt.subplots(figsize=(8,6))
# plt.scatter(transitions, max_entanglement, marker='x', label='max entanglement')
plt.scatter(mean_entanglement, transitions, marker='o')
for mean_e, trans, k in zip(mean_entanglement, transitions, max_entanglement):
    if k==1:
        k='1 = max_k'
    ax.annotate('%s' % k, xy=(mean_e, trans), xytext=(mean_e+.05,trans+6000), textcoords='data')
plt.ylabel('transitions')
plt.xlabel('mean number of variables modified')
plt.title('Planning Time vs. Mean Effect Size ({} vars, {} values)'.format(n_vars, n_values))
ax.set_ylim([0,ax.get_ylim()[1]])
ax.set_xlim([0,ax.get_xlim()[1]])
# plt.legend()
plt.tight_layout()
plt.savefig('results/plots/planning_time_vs_effect_size_{}-{}.png'.format(n_vars, n_values))
plt.show()

#%%
fig, ax = plt.subplots()
sns.pointplot(x='max_vars',y='transitions', data=data, units='seed', join=False, estimator=np.mean, color='C0', ax=ax)
sns.pointplot(x='max_vars',y='transitions', data=data, units='seed', join=False, estimator=np.median, color='C1', ax=ax)
plt.legend(handles=ax.lines[::len(all_k_values+1)],labels=['mean', 'median'], loc='best')
plt.xlabel('Max number of variables changed per action')
plt.ylabel('Number of transitions considered')
# plt.ylim([0,1e5])
plt.title('Planning Time vs. Max Effect Size ({} vars, {} values)'.format(n_vars, n_values))
# plt.yscale('log')
plt.tight_layout()
# plt.savefig('results/plots/planning_time_vs_effect_size_{}-{}.png'.format(n_vars, n_values))
plt.show()

#%%
# sns.violinplot(x='max_vars',y='n_errors', data=data, units='seed', cut=0, inner=None, scale='count')
#%%
# sns.violinplot(y='max_vars',x='transitions', data=data, orient='h', cut=0, inner=None, scale='count')
# ax = plt.gca()
# ax.set_xticklabels(list(map(lambda x: x/1e6,ax.get_xticks())))
# ax.set_xlabel('transitions (millions)')
# plt.show()
