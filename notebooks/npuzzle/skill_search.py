from collections import defaultdict
import argparse
import copy
import pickle
import os
import sys
from npuzzle import npuzzle
from notebooks import search

if 'ipykernel' in sys.argv[0]:
    sys.argv = [sys.argv[0], '-v', '0.2']
parser = argparse.ArgumentParser()
parser.add_argument('-n', type=int, default=15, choices=[8, 15, 24, 35, 48, 63, 80],
                    help='Number of tiles')
parser.add_argument('-r', type=int, default=0,
                    help='Initial row for blank space')
parser.add_argument('-c', type=int, default=0,
                    help='Initial col for blank space')
parser.add_argument('-v', type=str, required=True,
                    help='Which version to use for generated macros')
parser.add_argument('--max_transitions', type=lambda x: int(float(x)), default=62500,
                    help='Maximum number of variables changed per primitive action')
parser.add_argument('--save_best_n', type=int, default=100,
                    help='Number of best macros to save')
args = parser.parse_args()

#%% Make n-puzzle and set initial blank location
puzzle = npuzzle.NPuzzle(n=15, start_blank=(args.r, args.c))
tag = 'n{}-r{}-c{}'.format(args.n, *puzzle.blank_idx)

#%% Configure the search
startpuz = copy.deepcopy(puzzle)
is_goal = lambda node: False
step_cost = lambda action: 1
def heuristic(puz):
    swap_list, starting_blank_idx = puz.summarize_effects(baseline=puzzle)
    if len(swap_list) == 0:
        return float('inf')
    else:
        return len(swap_list)

def get_successors(puz):
    return [(copy.deepcopy(puz).transition(a), [a]) for a in puz.actions()]

#%% Run the search
search_results = search.astar(startpuz, is_goal, step_cost, heuristic, get_successors, args.max_transitions, args.save_best_n)

#%% Save the results
results_dir = 'results/skillsearch/npuzzle/'
os.makedirs(results_dir, exist_ok=True)
with open(results_dir+'v{}-{}-results.pickle'.format(args.v, tag), 'wb') as f:
    pickle.dump(search_results, f)
