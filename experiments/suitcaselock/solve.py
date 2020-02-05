import argparse
import copy
import os
import pickle
import random
import sys

import numpy as np

from domains.suitcaselock import SuitcaseLock
from experiments import search


def parse_args():
    """Parse input arguments

    Use --help to see a pretty description of the arguments
    """
    if 'ipykernel' in sys.argv[0]:
        sys.argv = [sys.argv[0]]
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed','-s', type=int, default=1,
                        help='Seed to use for RNGs')
    parser.add_argument('--n_vars', type=int, default=4,
                        help='Number of variables')
    parser.add_argument('--n_values', type=int, default=10,
                        help='Number of possible values for each variable')
    parser.add_argument('--entanglement', type=int, default=1,
                        help='Maximum number of variables changed per primitive action')
    parser.add_argument('--search_alg', type=str, default='gbfs',
                        choices = ['astar', 'gbfs', 'weighted-astar'],
                        help='Search algorithm to run')
    parser.add_argument('--g_weight', type=float, default=None,
                        help='Weight for g-score in weighted A*')
    parser.add_argument('--h_weight', type=float, default=None,
                        help='Weight for h-score in weighted A*')
    parser.add_argument('--max_transitions', type=lambda x: int(float(x)), default=1e5,
                        help='Maximum number of state transitions')
    return parser.parse_args()


def solve():
    """Instantiate a SuitcaseLock and solve with the specified search algorithm"""
    args = parse_args()

    seed = args.random_seed

    # Set up the scramble
    random.seed(seed)
    np.random.seed(seed)

    newlock = SuitcaseLock(n_vars=args.n_vars, n_values=args.n_values,
                           entanglement=args.entanglement)
    start = copy.deepcopy(newlock).scramble(seed=seed)
    goal = copy.deepcopy(newlock).scramble(seed=seed+1000)

    actions = newlock.actions()

    print('Using seed: {:03d}'.format(seed))
    print('Start:', start)
    print('Goal:', goal)

    # Set up the search problem
    search_fn = {
        'astar': search.astar,
        'gbfs': search.gbfs,
        'weighted_astar': search.weighted_astar,
    }[args.search_alg]

    def get_successors(lock):
        # don't deepcopy unless you want billions of copies of the actions list!
        return [(copy.copy(lock).apply_macro(diff=a), a) for a in actions]

    search_dict = {
        'start': start,
        'is_goal': lambda node: node.state == goal,
        'step_cost': lambda action: 1,
        'heuristic': lambda lock: sum(lock.summarize_effects(baseline=goal) > 0),
        'get_successors': get_successors,
        'max_transitions': args.max_transitions,
    }

    if search_fn == 'weighted_astar':
        assert (args.g_weight is not None
                and args.h_weight is not None), 'Must specify weights if using weighted A*.'
        gh_weights = (args.g_weight, args.h_weight)
        search_dict['gh_weights'] = gh_weights

    #%% Run the search
    search_results = search_fn(**search_dict)

    #%% Save the results
    tag = 'n_vars-{}/n_values-{}/entanglement-{}'
    tag = tag.format(args.n_vars, args.n_values, args.entanglement, args.max_transitions)

    results_dir = 'results/suitcaselock/{}/{}/'.format(args.search_alg, tag)
    os.makedirs(results_dir, exist_ok=True)
    with open(results_dir+'seed-{:03d}.pickle'.format(seed), 'wb') as file:
        pickle.dump(search_results, file)


if __name__ == '__main__':
    solve()
