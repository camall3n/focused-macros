from collections import OrderedDict
import glob
import pickle
import os

def main():
    """Combine N-Puzzle macros for different starting positions into a single file"""
    results_dir = 'results/macros/npuzzle/'
    set_suffix = '-set12'
    filenames = glob.glob(results_dir+'macro{}-n15-*-results.pickle'.format(set_suffix))
    macros = OrderedDict()
    for filename in filenames:
        row = int(filename.split('/')[-1].split('-')[-3][1:])
        col = int(filename.split('/')[-1].split('-')[-2][1:])
        with open(filename, 'rb') as file:
            search_results = pickle.load(file)
        best_n = search_results[-1]
        best_n = [(score, [a[0] for a in macro]) for score, macro in best_n]

        clean_macros = []
        for _, macro in best_n:
            if macro != []:
                clean_macros.append(macro)

        n_macros = 100
        clean_macros = clean_macros[-n_macros:]

        macros[(row,col)] = clean_macros

    #%% Save the results
    os.makedirs('results/macros/npuzzle/', exist_ok=True)
    with open('results/macros/npuzzle/clean_macros{}.pickle'.format(set_suffix), 'wb') as file:
        pickle.dump(macros, file)

if __name__ == '__main__':
    main()
