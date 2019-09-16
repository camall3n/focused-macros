import glob
import matplotlib.pyplot as plt
import pickle
import os

from util import rsync

# results_dir = 'results/planning/'
# os.makedirs(results_dir, exist_ok=True)
rsync(source='brown:~/dev/skills-for-planning/results/planning',
      dest='results/')
primitive_results = glob.glob(results_dir+'*-primitive.pickle')
expert_results = glob.glob(results_dir+'*-expert.pickle')

#%%
def generate_plot(filename, ax, color=None):
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
    ax.plot(x,y,c=color)

fig, ax = plt.subplots()
ax.set_ylim([0,48])
ax.set_xlim([0,1e5])
for f in expert_results:
    generate_plot(f, ax, 'C0')
for f in expert_results:
    generate_plot(f, ax, 'C1')
plt.show()

# states[-1].render()
