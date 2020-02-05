import glob
import pickle
import os

from domains.cube import ACTIONS as primitive_actions
from domains.cube.macros import expert
import notebooks.picklefix  # pylint: disable=W0611


def main():
    """Clean up and store the learned macro-actions found with macro_search

    Load the discovered macro-actions, remove any primitive actions, and save
    the best N macros where N is equal to the number of expert macros.
    """
    results_dir = 'results/macros/cube/'
    filename = glob.glob(results_dir+'macro-results.pickle')[-1]
    with open(filename, 'rb') as file:
        search_results = pickle.load(file)
    best_n = search_results[-1]
    best_n = [(score, [a[0] for a in macro]) for score, macro in best_n]

    n_macros = len(expert.macros)

    clean_macros = []
    for _, macro in best_n:
        if macro != [] and ' '.join(macro) not in primitive_actions:
            clean_macros.append(macro)
    clean_macros = clean_macros[-n_macros:]

    #%% Save the results
    os.makedirs('results/macros/cube', exist_ok=True)
    with open('results/macros/cube/clean_macros.pickle', 'wb') as file:
        pickle.dump(clean_macros, file)


if __name__ == '__main__':
    main()
