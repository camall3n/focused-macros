import glob
import matplotlib.pyplot as plt
import pickle

results_dir = 'results/planning/'
results_files = glob.glob(results_dir+'*.pickle')
filename = results_files[0]
with open(filename, 'rb') as f:
    search_results = pickle.load(f)
states, actions, n_expanded, n_transitions, candidates = search_results

n_errors = len(states[-1].summarize_effects())
x = [c for c,n in candidates]
y = [n.h_score for c,n in candidates]

#%%
x
y
sum(len(a) for a in actions)
len(actions)
n_expanded
n_transitions
n_errors

fig, ax = plt.subplots()
ax.plot(x,y)
ax.set_ylim([0,48])

# states[-1].render()
