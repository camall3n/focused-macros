import glob
import matplotlib.pyplot as plt
import pickle
import os
import pandas as pd
import seaborn as sns

from util import rsync

# rsync(source='brown:~/dev/skills-for-planning/results/planning',
      # dest='results/')
results_dir = 'results/suitcaselock/'
primitive_results = glob.glob(results_dir+'*/primitive/*.pickle')
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

    sum(len(a) for a in actions)
    len(actions)
    n_expanded
    n_transitions
    n_errors
    ax.plot(x,y,c=color,alpha=0.6, label=label)
    return n_errors

fig, ax = plt.subplots(figsize=(8,6))
ax.set_title('Planning performance')
ax.set_ylim([0,8])
ax.set_xlim([0,1e5])
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
    max_vars = int(f.split('/')[2].split('_')[-1])
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
plt.savefig('results/plots/planning_time_v{}.png'.format(gen_version))
plt.show()

#%%
solves = []
data = []
for i,filename in enumerate(primitive_results):
    with open(filename, 'rb') as f:
        search_results = pickle.load(f)
    seed = int(filename.split('/')[-1].split('.')[0].split('-')[-1])
    max_vars = int(filename.split('/')[2].split('_')[-1])
    states, actions, n_expanded, n_transitions, candidates = search_results
    goal = states[0].reset().scramble(seed=seed+1000)

    n_errors = sum(states[-1].summarize_effects(baseline=goal)>0)
    if n_errors == 0:
        solves.append(i)
    data.append({
        'transitions': n_transitions,
        'max_vars': max_vars,
        'seed': seed,
    })
data = pd.DataFrame(data)
sns.pointplot(x='max_vars',y='transitions', data=data)
sns.pointplot(x='max_vars',y='transitions', data=data, estimator=None)
grid = sns.FacetGrid(data, col="seed", hue="seed",
                     col_wrap=4, height=1.5)
grid.map(plt.plot, 'max_vars', 'transitions')
grid.fig.tight_layout(w_pad=1)
print(len(solves), 'out of', len(primitive_results))

print(solves)
