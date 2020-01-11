import copy
import glob
from itertools import groupby, count
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import pandas as pd
import seaborn as sns

results_dir = 'results/npuzzle/8-puzzle/default_goal/'

transition_cap = 1e5
# n_vars, n_values, transition_cap = 15, 4, 20e6
# n_vars, n_values, transition_cap = 12, 4, 40e6

result_files = sorted(glob.glob(results_dir+'*/*.pickle'))

def generate_plot(filename, ax, color=None):
    filename = result_files[0]
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

    ax.plot(x,y,c=color,alpha=0.6, label=tag)
    return n_errors

fig, ax = plt.subplots(figsize=(8,6))
ax.set_title('Planning performance')
ax.set_ylim([0,8])
ax.set_xlim([0,transition_cap])
ax.hlines(8,0,transition_cap,linestyles='dashed',linewidths=1)
ax.set_ylabel('Number of errors remaining')
ax.set_xlabel('Number of transitions considered')
for i,f in enumerate(result_files):
    seed = int(f.split('/')[-1].split('.')[0].split('-')[-1])
    color = 'C{}'.format(i)
    generate_plot(f, ax, color)
ax.legend()
# plt.savefig('results/plots/fixed_suitcase_planning_time_v{}.png'.format(gen_version))
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
all_tags = list(map(str,np.unique([filename.split('/')[-2] for filename in result_files])))
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
entanglement = list(data.groupby('entanglement',as_index=False).mean()['entanglement'])
transitions = list(data.groupby('entanglement',as_index=False).mean()['transitions'])

fig, ax = plt.subplots(figsize=(8,6))
# plt.scatter(transitions, entanglement, marker='x', label='max entanglement')
plt.scatter(entanglement, transitions, marker='o')
# for mean_e, trans, k in zip(entanglement, transitions, entanglement):
#     if k==1:
#         k='k = 1'
#     ax.annotate('%s' % k, xy=(mean_e, trans), xytext=(mean_e+.05,trans+6000), textcoords='data')
plt.ylabel('Number of transitions considered')
plt.xlabel('Number of variables modified')
plt.title('Planning Time vs. Effect Size ({} vars, {} values)'.format(n_vars, n_values))
# ax.set_ylim([0,ax.get_ylim()[1]])
# ax.set_xlim([0,ax.get_xlim()[1]])
# plt.legend()
# ax.set_yscale('log')
plt.tight_layout()
plt.savefig('results/plots/fixed_suitcase_planning_time_vs_effect_size_{}-{}.png'.format(n_vars, n_values))
plt.show()

#%%
fig, ax = plt.subplots(figsize=(8,6))
sns.pointplot(x='entanglement',y='transitions', data=data.query('n_errors==0'), units='seed', join=False, estimator=np.mean, color='C0', ax=ax)
sns.pointplot(x='entanglement',y='transitions', data=data.query('n_errors==0'), units='seed', join=False, estimator=np.median, color='C1', ax=ax)
plt.legend(handles=ax.lines[::len(all_k_values+1)],labels=['mean', 'median'], loc='best')
plt.xlabel('Number of variables changed per action')
plt.ylabel('Number of transitions considered')
# plt.ylim([0,1e5])
plt.title('Planning Time vs. Effect Size ({} vars, {} values) -- [linear scale]'.format(n_vars, n_values))
plt.tight_layout()
plt.savefig('results/plots/fixed_suitcase_planning_time_vs_effect_size_linear_{}-{}.png'.format(n_vars, n_values))
plt.show()
#%%
fig, ax = plt.subplots(figsize=(8,6))
sns.pointplot(x='entanglement',y='transitions', data=data.query('n_errors==0'), units='seed', join=False, estimator=np.mean, color='C0', ax=ax)
sns.pointplot(x='entanglement',y='transitions', data=data.query('n_errors==0'), units='seed', join=False, estimator=np.median, color='C1', ax=ax)
plt.legend(handles=ax.lines[::len(all_k_values+1)],labels=['mean', 'median'], loc='best')
plt.xlabel('Number of variables changed per action')
plt.ylabel('Number of transitions considered')
# plt.ylim([0,1e5])
plt.title('Planning Time vs. Effect Size ({} vars, {} values) -- [log scale]'.format(n_vars, n_values))
plt.yscale('log')
plt.tight_layout()
plt.savefig('results/plots/fixed_suitcase_planning_time_vs_effect_size_log_{}-{}.png'.format(n_vars, n_values))
plt.show()

#%%
# sns.violinplot(x='entanglement',y='n_errors', data=data, units='seed', cut=0, inner=None, scale='area')
#%%
# sns.violinplot(y='entanglement',x='transitions', data=data, orient='h', cut=0, inner=None, scale='count')
# ax = plt.gca()
# ax.set_xticklabels(list(map(lambda x: x/1e6,ax.get_xticks())))
# ax.set_xlabel('transitions (millions)')
# plt.show()

#%%
# fig, ax = plt.subplots()
# sns.scatterplot(x='transitions', y='n_errors',data=data.groupby('entanglement', as_index=False).mean(), hue='entanglement', ax=ax, s=70, legend='full')
# # ax.hlines(n_vars,0,transition_cap,linestyles='dashed',linewidths=1)
# # ax.set_ylim([0,n_vars])
# # ax.set_xlim([0,transition_cap])
# # ax.set_xticklabels(list(map(lambda x: x/1e6,ax.get_xticks())))
# ax.set_title('Mean final planning performance')
# ax.set_ylabel('Number of errors remaining')
# ax.set_xlabel('Number of transitions considered')
#
# ax.set_xscale('log')
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles=handles, labels=labels, ncol=2)
# # handles = handles[1:]
# # labels = ['actions only','actions + expert skills', 'actions + random skills', 'actions + generated skills']
# # ax.legend(handles=handles, labels=labels, framealpha=1, borderpad=0.7)
# # plt.savefig('results/plots/mean_planning_performance.png')
# plt.show()
