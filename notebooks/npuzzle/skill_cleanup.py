import glob
import matplotlib.pyplot as plt
import pickle
import os


from cube.options import expert

version = '0.1'
results_dir = 'results/skillsearch/npuzzle/'
filenames = glob.glob(results_dir+'v'+version+'-*-results.pickle')
skills = {}
for filename in filenames:
    r = int(filename.split('/')[-1].split('-')[-3][1:])
    c = int(filename.split('/')[-1].split('-')[-2][1:])
    with open(filename, 'rb') as f:
        search_results = pickle.load(f)
    best_n = search_results[-1]
    best_n = [(score, [a[0] for a in skill]) for score, skill in best_n]

    clean_skills = []
    for score, skill in best_n:
        if skill != []:
            clean_skills.append(skill)

    n_options = 100
    clean_skills = clean_skills[-n_options:]

    skills[(r,c)] = clean_skills

#%% Save the results
os.makedirs('results/skillsearch/npuzzle/', exist_ok=True)
with open('results/skillsearch/npuzzle/v{}-clean_skills.pickle'.format(version), 'wb') as f:
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
