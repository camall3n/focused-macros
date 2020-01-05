import glob
import matplotlib.pyplot as plt
import pickle
import os

from util import rsync
from cube import cube, pattern

# rsync(source='brown:~/dev/skills-for-planning/results/planning',
      # dest='results/')
gen_version = '0.2'
default_results = sorted(glob.glob('results/planning/default_goal/'+'generated-v'+gen_version+'/*.pickle'))
random_results = sorted(glob.glob('results/planning/random_goal/'+'generated-v'+gen_version+'/*.pickle'))

#%%
def generate_plot(filename, ax, color=None, label=None):
    with open(filename, 'rb') as f:
        search_results = pickle.load(f)
    states, actions, n_expanded, n_transitions, candidates = search_results[:5]

    n_errors = len(states[-1].summarize_effects())
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
ax.set_ylim([0,50])
ax.set_xlim([0,1e5])
ax.hlines(48,0,1e5,linestyles='dashed',linewidths=1)
ax.set_ylabel('Number of errors remaining')
ax.set_xlabel('Number of transitions considered')
# for i,f in enumerate(random_results):
#     label = None if i > 0 else 'actions + fixed random skills'
#     generate_plot(f, ax, 'C2', label=label)
for i,f in enumerate(default_results):
    label = None if i > 0 else 'default goal'
    generate_plot(f, ax, 'C3', label=label)
for i,f in enumerate(random_results):
    label = None if i > 0 else 'random goals'
    generate_plot(f, ax, 'C4', label=label)

ax.legend()
plt.savefig('results/plots/planning_alt_goals.png'.format(gen_version))
plt.show()

#%%
solves = []
for i,filename in enumerate(default_results):
    with open(filename, 'rb') as f:
        search_results = pickle.load(f)
    states, actions, n_expanded, n_transitions, candidates = search_results

    n_errors = len(states[-1].summarize_effects())
    if n_errors == 0:
        solves.append(i)
print(len(solves), 'out of', len(default_results))

print(solves)

#%%
solves = []
for i,filename in enumerate(random_results):
    with open(filename, 'rb') as f:
        search_results = pickle.load(f)
    seed = int(filename.split('/')[-1].split('.')[-2].split('-')[-1])
    goal = cube.Cube().apply(pattern.scramble(seed+1000))
    states, actions, n_expanded, n_transitions, candidates = search_results

    n_errors = len(states[-1].summarize_effects(baseline=goal))
    if n_errors == 0:
        solves.append(i)
print(len(solves), 'out of', len(random_results))

print(solves)

#%%
with open(default_results[0], 'rb') as f:
    search_results = pickle.load(f)
    states, actions, n_expanded, n_transitions, candidates = search_results
    states[-1].render()

#%%
with open(random_results[0], 'rb') as f:
    search_results = pickle.load(f)
    states, actions, n_expanded, n_transitions, candidates = search_results
    states[-1a].render()
