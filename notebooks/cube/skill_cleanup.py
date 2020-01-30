import glob
import matplotlib.pyplot as plt
import pickle
import os

from domains.cube import actions as primitive_actions
from domains.cube.macros import expert
import notebooks.picklefix

version = '0.4'
# 0.1 is unused. (I can't remember what this one was, but it was worse than 0.2)
# 0.2 is the top 480 macros found, by h-score, within 1M simulator steps
# 0.3 is the top 480 macros found, by f-score, within 1M simulator steps
# 0.4 is the top 576 macros found, by h-score, within 1M simulator steps
results_dir = 'results/macros/cube/'
filename = glob.glob(results_dir+'v'+version+'-results.pickle')[-1]
with open(filename, 'rb') as f:
    search_results = pickle.load(f)
best_n = search_results[-1]
best_n = [(score, [a[0] for a in macro]) for score, macro in best_n]

n_macros = len(expert.macros)
best_n[-n_macros:]

clean_macros = []
for score, macro in best_n:
    if macro != [] and ' '.join(macro) not in primitive_actions:
        clean_macros.append(macro)
clean_macros = clean_macros[-n_macros:]

#%% Save the results
os.makedirs('results/macros/cube', exist_ok=True)
with open('results/macros/cube/v{}-clean_skills.pickle'.format(version), 'wb') as f:
    pickle.dump(clean_macros, f)
