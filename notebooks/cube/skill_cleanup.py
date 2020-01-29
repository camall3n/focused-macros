import glob
import matplotlib.pyplot as plt
import pickle
import os

from domains.cube import actions as primitive_actions
from domains.cube.options import expert
import notebooks.picklefix

version = '0.4'
# 0.1 is unused. (I can't remember what this one was, but it was worse than 0.2)
# 0.2 is the top 480 skills found, by h-score, within 1M simulator steps
# 0.3 is the top 480 skills found, by f-score, within 1M simulator steps
# 0.4 is the top 576 skills found, by h-score, within 1M simulator steps
results_dir = 'results/skillsearch/cube/'
filename = glob.glob(results_dir+'v'+version+'-results.pickle')[-1]
with open(filename, 'rb') as f:
    search_results = pickle.load(f)
best_n = search_results[-1]
best_n = [(score, [a[0] for a in skill]) for score, skill in best_n]

n_options = len(expert.options)
best_n[-n_options:]

clean_skills = []
for score, skill in best_n:
    if skill != [] and ' '.join(skill) not in primitive_actions:
        clean_skills.append(skill)
clean_skills = clean_skills[-n_options:]

#%% Save the results
os.makedirs('results/skillsearch/cube', exist_ok=True)
with open('results/skillsearch/cube/v{}-clean_skills.pickle'.format(version), 'wb') as f:
    pickle.dump(clean_skills, f)
