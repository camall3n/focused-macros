import glob
import matplotlib.pyplot as plt
import pickle
import os

from cube.cube import actions as primitive_actions
from cube.options import expert

version = '0.4'# 0.4 is 0.2 with more skills saved
results_dir = 'results/skillsearch/'
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
os.makedirs('results/skillsearch', exist_ok=True)
with open('results/skillsearch/v{}-clean_skills.pickle'.format(version), 'wb') as f:
    pickle.dump(clean_skills, f)

# #%% Print the generated skills v0.3
# states, actions, n_expanded, n_transitions, candidates, best_n = search_results
# actions
# for score, skill in best_n:
#     print(str(score-len(skill)), str(score),'-', ' '.join([s[0] for s in skill]))
#
# #%% Print the generated skills v0.2
# states, actions, n_expanded, n_transitions, candidates, best_n = search_results
# actions
# for score, skill in best_n:
#     print(str(score), str(score+len(skill)),'-', ' '.join([s[0] for s in skill]))
#
# #%% Print the generated skills v0.1
# states, actions, n_expanded, n_transitions, candidates, best_n = search_results
# actions
# for node, skill in best_n:
#     print(len(node.state.summarize_effects()),'-', ' '.join([s[0] for s in skill]))
