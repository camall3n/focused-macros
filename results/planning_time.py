import glob
import matplotlib.pyplot as plt
import pickle
import os

from util import rsync

rsync(source='brown:~/dev/skills-for-planning/results/planning',
      dest='results/')
results_dir = 'results/planning/'
primitive_results = glob.glob(results_dir+'*-primitive.pickle')
expert_results = glob.glob(results_dir+'*-expert.pickle')

#%%
def generate_plot(filename, ax, color=None, label=None):
    with open(filename, 'rb') as f:
        search_results = pickle.load(f)
    states, actions, n_expanded, n_transitions, candidates = search_results

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
    ax.plot(x,y,c=color, label=label)

fig, ax = plt.subplots()
ax.set_title('Planning performance')
ax.set_ylim([0,50])
ax.set_xlim([0,1e5])
ax.hlines(48,0,1e5,linestyles='dashed',linewidths=1)
ax.set_ylabel('Number of errors remaining')
ax.set_xlabel('Number of transitions considered')
for i,f in enumerate(primitive_results):
    label = None if i > 0 else 'actions only'
    generate_plot(f, ax, 'C0', label=label)
for i,f in enumerate(expert_results):
    label = None if i > 0 else 'actions + expert skills'
    generate_plot(f, ax, 'C1', label=label)
ax.legend()
# plt.savefig('results/plots/planning_time.png')
plt.show()
