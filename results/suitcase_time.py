import copy
import glob
from itertools import groupby, count
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import pandas as pd
import seaborn as sns
import notebooks.picklefix
from notebooks.rrank import rrank, reduce
from notebooks.search import dijkstra

alg = 'gbfs'
n_vars = 20
results_dir = 'results/suitcaselock/{}/'.format(alg)

if alg == 'astar':
    if n_vars == 20:
        n_values = 2
        transition_cap = 5e7
    else:
        n_values = 4
        transition_cap = 1e8
elif alg == 'gbfs':
    if n_vars == 20:
        n_values = 2
        transition_cap = 2.2e7
    else:
        n_values = 4
        transition_cap = 2e7

# n_vars, n_values, transition_cap = 20, 2, 10e6
# n_vars, n_values, transition_cap = 10, 4, 2e7
# n_vars, n_values, transition_cap = 12, 4, 40e6

result_files = sorted(glob.glob(results_dir+'n_vars-{}/n_values-{}/entanglement-*/*.pickle'.format(n_vars, n_values)))
result_files = [r for r in result_files if 'entanglement-17' not in r]

curve_data = []
for filename in result_files:
# def generate_plot(filename, ax, color=None, label=None):
    with open(filename, 'rb') as f:
        try:
            search_results = pickle.load(f)
        except EOFError:
            continue
    states, actions, n_expanded, n_transitions, candidates = search_results[:5]
    seed = int(filename.split('/')[-1].split('.')[0].split('-')[-1])
    entanglement = int(filename.split('/')[-2].split('-')[-1])
    goal = states[0].reset().scramble(seed=seed+1000)
    n_errors = sum(states[-1].summarize_effects(baseline=goal)>0)
    x = [c/2 for c,n in candidates]
    y = [n.h_score for c,n in candidates]

    # Extend final value to end of plot
    if n_errors > 0:
        x += [n_transitions/2]
        y += [y[-1]]
    [curve_data.append({'transitions': t, 'n_errors': err, 'seed': seed, 'entanglement': entanglement}) for t, err in zip(x,y)]

#%%
curve_data = pd.DataFrame(curve_data)
fig, ax = plt.subplots(figsize=(8,6))
lines = []
for k in np.unique(curve_data['entanglement'])[::-1]:
    sns.lineplot(data=curve_data.query('entanglement==@k'), x='transitions', y='n_errors', legend=False, estimator=None, units='seed', ax=ax)
    lines.append(ax.get_lines()[-1])
ax.legend(lines[::-1],np.unique(curve_data['entanglement']))
ax.set_title('Planning performance ({} vars, {} values)'.format(n_vars, n_values))
ax.set_ylim([0,n_vars//2])
ax.set_xlim([0,transition_cap])
ax.set_xticklabels(np.asarray(ax.get_xticks())/1e6)
ax.hlines(n_vars,0,transition_cap,linestyles='dashed',linewidths=1)
ax.set_ylabel('Number of errors remaining')
ax.set_xlabel('Number of transitions considered (millions)')
plt.savefig('results/plots/suitcaselock/suitcaselock_planning_time.png')
plt.show()

#%%
solves = []
data = []
failures = []
len(states[0].actions())*n_vars
len(states[0].actions())*n_values**n_vars
max_transitions

for i,filename in enumerate(result_files):
    seed = int(filename.split('/')[-1].split('.')[0].split('-')[-1])
    k = int(filename.split('/')[-2].split('-')[-1])
    with open(filename, 'rb') as f:
        try:
            search_results = pickle.load(f)
        except EOFError:
            print('error reading', filename)
            failures.append((k, seed))
            continue
    states, actions, n_expanded, n_transitions, candidates = search_results
    goal = states[0].reset().scramble(seed=seed+1000)
    mean_entanglement = np.mean(np.sum(np.stack(states[0].actions()[:n_vars]),axis=-1))
    plan_length = len(actions)

    n_errors = sum(states[-1].summarize_effects(baseline=goal)>0)
    if n_errors == 0:
        solves.append(i)
    data.append({
        'transitions': n_transitions/2,
        'plan_length': plan_length,
        'k': k,
        'entanglement': mean_entanglement,
        'seed': seed,
        'n_errors': n_errors,
    })
data = pd.DataFrame(data)
#%%
sns.boxenplot(data=data, x='k', y='plan_length')
plt.yscale('log')
plt.show()


#%%
print('Solve Counts')
print()
all_k_values = np.unique([int(filename.split('/')[-2].split('-')[-1]) for filename in result_files])
total_solves = 0
total_attempts = 0
failures = []
successes = []
for k in all_k_values:
    n_solves = len(data.query('(k==@k) and (n_errors==0) and (transitions < @transition_cap)'))
    n_attempts = len(data.query('k==@k'))
    failures += [(k, s) for s in sorted(list(data.query('(k==@k) and (n_errors>0)')['seed']))]
    successes += [(k, s) for s in sorted(list(data.query('(k==@k) and (n_errors==0)')['seed']))]
    total_solves += n_solves
    total_attempts += n_attempts

    print('{:2d}: {:3d} out of {:3d}'.format(k, n_solves, n_attempts))
print()
print('{:4d} out of {:4d}'.format(total_solves, total_attempts))
print(failures)

#%%
def as_range(iterable): # not sure how to do this part elegantly
    l = list(iterable)
    if len(l) > 1:
        return '{0}-{1}'.format(l[0], l[-1])
    else:
        return '{0}'.format(l[0])

print('Missing:')
for k in all_k_values:
    missing = [x for x in range(1,301) if x not in list(data.query('k==@k')['seed'])]
    missing_str = ','.join(as_range(g) for _, g in groupby(missing, key=lambda n, c=count(): n-next(c)))
    print('{:2d}: {}'.format(k, missing_str))

#%%
fig, ax = plt.subplots(figsize=(8,6))
sns.lineplot(x='k',y='transitions', data=data, estimator=np.median, color='C1', ci=95, ax=ax, err_style='bars', markers=True, lw=0, sizes=150)
sns.scatterplot(x='k',y='transitions', data=data.groupby(['k'], as_index=False).median(), color='C1', ax=ax, markers='o', sizes=150, label='median')
sns.lineplot(x='k',y='transitions', data=data, estimator=np.mean, color='C0', ci=95, ax=ax, err_style='bars', markers=True, lw=0, sizes=150)
sns.scatterplot(x='k',y='transitions', data=data.groupby(['k'], as_index=False).mean(), color='C0', ax=ax, markers='o', sizes=150, label='mean')
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles=handles[::-1],labels=labels[::-1], loc='best')

# sns.scatterplot(x='entanglement',y='transitions', data=data.query('n_errors==0'), color='C0', ax=ax)
plt.xlabel('Number of variables changed per action')
plt.ylabel('Number of transitions considered (millions)')
ax.set_yticklabels(np.asarray(ax.get_yticks())/1e6)
ax.set_xticks(np.arange(1,n_vars))
plt.title('Planning time vs. entanglement ({} vars, {} values) -- [linear scale]'.format(n_vars, n_values))
plt.tight_layout()
plt.savefig('results/plots/suitcaselock/suitcaselock_planning_time_vs_entanglement_linear_{}-{}.png'.format(n_vars, n_values))
plt.show()
#%%
fig, ax = plt.subplots(figsize=(8,6))
sns.lineplot(x='k',y='transitions', data=data, estimator=np.median, color='C1', ci=95, ax=ax, err_style='bars', markers=True, lw=0, sizes=150)
sns.scatterplot(x='k',y='transitions', data=data.groupby(['k'], as_index=False).median(), color='C1', ax=ax, markers='o', sizes=150, label='median')
sns.lineplot(x='k',y='transitions', data=data, estimator=np.mean, color='C0', ci=95, ax=ax, err_style='bars', markers=True, lw=0, sizes=150)
sns.scatterplot(x='k',y='transitions', data=data.groupby(['k'], as_index=False).mean(), color='C0', ax=ax, markers='o', sizes=150, label='mean')
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles=handles[::-1],labels=labels[::-1], loc='best')

# sns.scatterplot(x='entanglement',y='transitions', data=data.query('n_errors==0'), color='C0', ax=ax)
plt.xlabel('Number of variables changed per action')
plt.ylabel('Number of transitions considered (log scale)')
# plt.ylim([0,1e5])
# ax.set_xticks(np.arange(1,n_vars))
plt.title('Planning time vs. entanglement ({} vars, {} values)'.format(n_vars, n_values))
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

#%%
print('( k,seed): R  RR errors?')
for k, s in successes:
    filename = 'results/suitcaselock/gbfs/n_vars-{}/n_values-{}/entanglement-{:d}/seed-{:03d}.pickle'.format(n_vars, n_values, k,s)
    with open(filename, 'rb') as f:
        try:
            search_results = pickle.load(f)
        except EOFError:
            pass
    states, actions, n_expanded, n_transitions, candidates = search_results[:5]
    goal = copy.deepcopy(states[0]).reset().scramble(seed=s+1000)
    n_errors = sum(states[-1].summarize_effects(baseline=goal)>0)
    M = np.stack(states[0].actions()[:n_vars])
    rank = np.linalg.matrix_rank(M)
    reduced_rank = rrank(M)
    reduce(M)
    if reduced_rank != n_vars:
        print('({:2d}, {:03d}): {:2d} {:2d} {}'.format(k, s, rank, reduced_rank, n_errors>0))
        break

    rrank(np.stack(states[0].actions()[:n_vars]))
    if k==10 and s==7:
        break

def get_successors(lock):
    return [(copy.deepcopy(lock).apply_macro(diff=m), s) for s,m in zip(goal.actions(), goal.actions())]

# dijkstra_results = dijkstra(start, is_goal=(lambda node: node.state==goal), step_cost=(lambda x:1), get_successors=get_successors, max_transitions=int(4**10+200))
