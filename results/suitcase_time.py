import glob
from itertools import groupby, count
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import pandas as pd
import seaborn as sns
import notebooks.picklefix

results_dir = 'results/suitcaselock/'

n_vars, n_values, transition_cap = 20, 2, 5e7
# n_vars, n_values, transition_cap = 15, 4, 20e6
# n_vars, n_values, transition_cap = 12, 4, 40e6

result_files = sorted(glob.glob(results_dir+'n_vars-{}/n_values-{}/entanglement-*/*.pickle'.format(n_vars, n_values)))

curve_data = []
for filename in result_files:
# def generate_plot(filename, ax, color=None, label=None):
    with open(filename, 'rb') as f:
        search_results = pickle.load(f)
    states, actions, n_expanded, n_transitions, candidates = search_results[:5]
    seed = int(filename.split('/')[-1].split('.')[0].split('-')[-1])
    entanglement = int(filename.split('/')[4].split('-')[-1])
    goal = states[0].reset().scramble(seed=seed+1000)
    n_errors = sum(states[-1].summarize_effects(baseline=goal)>0)
    x = [c for c,n in candidates]
    y = [n.h_score for c,n in candidates]

    # Extend final value to end of plot
    if n_errors > 0:
        x += [n_transitions]
        y += [y[-1]]
    [curve_data.append({'transitions': t, 'n_errors': err, 'seed': seed, 'entanglement': entanglement}) for t, err in zip(x,y)]
    # ax.plot(x,y,c=color,alpha=0.6, label=label)
    # return n_errors

# fig, ax = plt.subplots(figsize=(8,6))
# ax.set_title('Planning performance')
# ax.set_ylim([0,8])
# ax.set_xlim([0,transition_cap])
# ax.hlines(8,0,transition_cap,linestyles='dashed',linewidths=1)
# ax.set_ylabel('Number of errors remaining')
# ax.set_xlabel('Number of transitions considered')
# for i,f in enumerate(result_files):
#     seed = int(f.split('/')[-1].split('.')[0].split('-')[-1])
#     entanglement = int(f.split('/')[4].split('-')[-1])
#     label = None if seed > 1 else str(entanglement)
#     color = 'C{}'.format(entanglement-1)
#     generate_plot(f, ax, color, label=label)

curve_data = pd.DataFrame(curve_data)
fig, ax = plt.subplots(figsize=(8,6))
lines = []
for k in np.unique(curve_data['entanglement'])[::-1]:
    sns.lineplot(data=curve_data.query('entanglement==@k'), x='transitions', y='n_errors', legend=False, estimator=None, units='seed', ax=ax)
    lines.append(ax.get_lines()[-1])
ax.legend(lines[::-1],np.unique(curve_data['entanglement']))
ax.set_title('Planning performance ({} vars, {} values)'.format(n_vars, n_values))
ax.set_ylim([0,8])
ax.set_xlim([0,transition_cap])
ax.set_xticklabels(np.asarray(ax.get_xticks())/1e6)
ax.hlines(8,0,transition_cap,linestyles='dashed',linewidths=1)
ax.set_ylabel('Number of errors remaining')
ax.set_xlabel('Number of transitions considered')
plt.savefig('results/plots/suitcaselock/suitcaselock_planning_time.png')
plt.show()

#%%
solves = []
data = []
for i,filename in enumerate(result_files):
    with open(filename, 'rb') as f:
        search_results = pickle.load(f)
    seed = int(filename.split('/')[-1].split('.')[0].split('-')[-1])
    entanglement = int(filename.split('/')[4].split('-')[-1])
    states, actions, n_expanded, n_transitions, candidates = search_results
    goal = states[0].reset().scramble(seed=seed+1000)

    n_errors = sum(states[-1].summarize_effects(baseline=goal)>0)
    if n_errors == 0:
        solves.append(i)
    data.append({
        'transitions': n_transitions,
        'entanglement': entanglement,
        'seed': seed,
        'n_errors': n_errors,
    })
data = pd.DataFrame(data)

#%%
print('Solve Counts')
print()
all_k_values = np.unique([int(filename.split('/')[-2].split('-')[-1]) for filename in result_files])
total_solves = 0
total_attempts = 0
for k in all_k_values:
    n_solves = len(data.query('(entanglement==@k) and (n_errors==0) and (transitions < @transition_cap)'))
    n_attempts = len(data.query('entanglement==@k'))
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
    missing = [x for x in range(1,301) if x not in list(data.query('entanglement==@k')['seed'])]
    missing_str = ','.join(as_range(g) for _, g in groupby(missing, key=lambda n, c=count(): n-next(c)))
    print('{:2d}: {}'.format(k, missing_str))

#%%
# entanglement = list(data.groupby('entanglement',as_index=False).mean()['entanglement'])
# transitions = list(data.groupby('entanglement',as_index=False).mean()['transitions'])
#
# fig, ax = plt.subplots(figsize=(8,6))
# # plt.scatter(transitions, entanglement, marker='x', label='max entanglement')
# plt.scatter(entanglement, transitions, marker='o')
# # for mean_e, trans, k in zip(entanglement, transitions, entanglement):
# #     if k==1:
# #         k='k = 1'
# #     ax.annotate('%s' % k, xy=(mean_e, trans), xytext=(mean_e+.05,trans+6000), textcoords='data')
# plt.ylabel('Number of transitions considered')
# plt.xlabel('Number of variables modified')
# plt.title('Planning Time vs. Effect Size ({} vars, {} values)'.format(n_vars, n_values))
# # ax.set_ylim([0,ax.get_ylim()[1]])
# # ax.set_xlim([0,ax.get_xlim()[1]])
# # plt.legend()
# # ax.set_yscale('log')
# plt.tight_layout()
# plt.savefig('results/plots/fixed_suitcase_planning_time_vs_effect_size_{}-{}.png'.format(n_vars, n_values))
# plt.show()

#%%
fig, ax = plt.subplots(figsize=(8,6))
sns.lineplot(x='entanglement',y='transitions', data=data, estimator=np.median, color='C1', ci=95, ax=ax, err_style='bars', markers=True, lw=0, sizes=150)
sns.scatterplot(x='entanglement',y='transitions', data=data.groupby(['entanglement'], as_index=False).median(), color='C1', ax=ax, markers='o', sizes=150, label='median')
sns.lineplot(x='entanglement',y='transitions', data=data, estimator=np.mean, color='C0', ci=95, ax=ax, err_style='bars', markers=True, lw=0, sizes=150)
sns.scatterplot(x='entanglement',y='transitions', data=data.groupby(['entanglement'], as_index=False).mean(), color='C0', ax=ax, markers='o', sizes=150, label='mean')

handles, labels = ax.get_legend_handles_labels()
plt.legend(handles=handles[::-1],labels=labels[::-1], loc='best')
plt.xlabel('Number of variables changed per action')
plt.ylabel('Number of transitions considered (millions)')
ax.set_yticklabels(np.asarray(ax.get_yticks())/1e6)
ax.set_xticks(np.arange(1,20))
plt.title('Planning time vs. entanglement ({} vars, {} values) -- [linear scale]'.format(n_vars, n_values))
plt.tight_layout()
plt.savefig('results/plots/suitcaselock/suitcaselock_planning_time_vs_entanglement_linear_{}-{}.png'.format(n_vars, n_values))
plt.show()
#%%
fig, ax = plt.subplots(figsize=(8,6))
sns.lineplot(x='entanglement',y='transitions', data=data, estimator=np.median, color='C1', ci=95, ax=ax, err_style='bars', markers=True, lw=0, sizes=150)
sns.scatterplot(x='entanglement',y='transitions', data=data.groupby(['entanglement'], as_index=False).median(), color='C1', ax=ax, markers='o', sizes=150, label='median')
sns.lineplot(x='entanglement',y='transitions', data=data, estimator=np.mean, color='C0', ci=95, ax=ax, err_style='bars', markers=True, lw=0, sizes=150)
sns.scatterplot(x='entanglement',y='transitions', data=data.groupby(['entanglement'], as_index=False).mean(), color='C0', ax=ax, markers='o', sizes=150, label='mean')

handles, labels = ax.get_legend_handles_labels()
plt.legend(handles=handles[::-1],labels=labels[::-1], loc='best')
plt.xlabel('Number of variables changed per action')
plt.ylabel('Number of transitions considered')
# plt.ylim([0,1e5])
ax.set_xticks(np.arange(1,20))
plt.title('Planning time vs. entanglement ({} vars, {} values) -- [log scale]'.format(n_vars, n_values))
plt.yscale('log')
plt.tight_layout()
plt.savefig('results/plots/suitcaselock/suitcaselock_planning_time_vs_entanglement_log_{}-{}.png'.format(n_vars, n_values))
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
