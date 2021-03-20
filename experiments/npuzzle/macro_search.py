import argparse
import copy
import os
import pickle
import sys

from domains.npuzzle import npuzzle
from experiments import search


def main():
    """Run A* to find macros for the specified N-Puzzle

    Use --help to see a pretty description of the arguments
    """
    if 'ipykernel' in sys.argv[0]:
        sys.argv = [sys.argv[0]]
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=15, choices=[8, 15, 24, 35, 48, 63, 80],
                        help='Number of tiles')
    parser.add_argument('-r', type=int, default=0,
                        help='Initial row for blank space')
    parser.add_argument('-c', type=int, default=0,
                        help='Initial col for blank space')
    parser.add_argument('--max_transitions', type=lambda x: int(float(x)), default=2000,
                        help='Maximum number of variables changed per primitive action')
    parser.add_argument('--save_best_n', type=int, default=12,
                        help='Number of best macros to save')
    args = parser.parse_args()

    #%% Make n-puzzle and set initial blank location
    puzzle = npuzzle.NPuzzle(n=15, start_blank=(args.r, args.c))
    tag = 'set{}-n{}-r{}-c{}'.format(args.save_best_n, args.n, *puzzle.blank_idx)

    #%% Configure the search
    def heuristic(puz):
        swap_list, _ = puz.summarize_effects(baseline=puzzle)
        if len(swap_list) == 0:
            return float('inf')
        return len(swap_list)

    def get_successors(puz):
        return [(copy.deepcopy(puz).transition(a), [a]) for a in puz.actions()]

    #%% Run the search
    search_results = search.astar(start = copy.deepcopy(puzzle),
                                  is_goal = lambda node: False,
                                  step_cost = lambda action: 1,
                                  heuristic = heuristic,
                                  get_successors = get_successors,
                                  max_transitions = args.max_transitions,
                                  save_best_n = args.save_best_n)

    #%% Save the results
    results_dir = 'results/macros/npuzzle/'
    os.makedirs(results_dir, exist_ok=True)
    with open(results_dir+'macro-{}-results.pickle'.format(tag), 'wb') as file:
        pickle.dump(search_results, file)


if __name__ == '__main__':
    main()
