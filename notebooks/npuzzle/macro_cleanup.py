import glob
import matplotlib.pyplot as plt
import pickle
import os

version = '0.2'
results_dir = 'results/macros/npuzzle/'
filenames = glob.glob(results_dir+'v'+version+'-*-results.pickle')
macros = {}
for filename in filenames:
    r = int(filename.split('/')[-1].split('-')[-3][1:])
    c = int(filename.split('/')[-1].split('-')[-2][1:])
    with open(filename, 'rb') as f:
        search_results = pickle.load(f)
    best_n = search_results[-1]
    best_n = [(score, [a[0] for a in macro]) for score, macro in best_n]

    clean_macros = []
    for score, macro in best_n:
        if macro != []:
            clean_macros.append(macro)

    n_macros = 100
    clean_macros = clean_macros[-n_macros:]

    macros[(r,c)] = clean_macros

#%% Save the results
os.makedirs('results/macros/npuzzle/', exist_ok=True)
with open('results/macros/npuzzle/v{}-clean_macros.pickle'.format(version), 'wb') as f:
    pickle.dump(macros, f)
