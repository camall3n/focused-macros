import argparse
import copy
import os
import pickle
import random
import sys
from types import SimpleNamespace

import numpy as np

from domains.npuzzle import NPuzzle, macros
from experiments import search, iw, bfws

def parse_args():
    """Parse input arguments

    Use --help to see a pretty description of the arguments
    """
    if 'ipykernel' in sys.argv[0]:
        sys.argv = [sys.argv[0]]
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=15, choices=[8, 15, 24, 35, 48, 63, 80],
                        help='Number of tiles')
    parser.add_argument('--random_seed','-s', type=int, default=1,
                        help='Seed to use for RNGs')
    parser.add_argument('--macro_type','-m', type=str, default='primitive',
                        choices=['primitive','random','learned'],
                        help='Type of macro_list to consider during search')
    parser.add_argument('--search_alg', type=str, default='gbfs',
                        choices = ['astar', 'gbfs', 'weighted_astar','bfws_r0', 'bfws_rg'],
                        help='Search algorithm to run')
    parser.add_argument('--g_weight', type=float, default=None,
                        help='Weight for g-score in weighted A*')
    parser.add_argument('--h_weight', type=float, default=None,
                        help='Weight for h-score in weighted A*')
    parser.add_argument('--random_goal','-r', action='store_true', default=False,
                        help='Generate a random goal instead of the default solve configuration')
    parser.add_argument('--max_transitions', type=lambda x: int(float(x)), default=5e5,
                        help='Maximum number of state transitions')
    parser.add_argument('--bfws_precision', type=int, default=3,
                        help='The number of width values, w \in {1,...,P}, to use when the search algorithm is best-first width search')
    return parser.parse_args()


def solve():
    """Instantiate an N-Puzzle and solve with the specified macro-actions and search algorithm"""
    args = parse_args()
    #

    # Set up the scramble
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    start = NPuzzle(n=args.n).scramble(seed=args.random_seed)

    if args.random_goal:
        goal = NPuzzle(n=args.n).scramble(seed=args.random_seed+1000)
        print('Using goal pattern: {:03d}'.format(args.random_seed+1000))
    else:
        goal = NPuzzle(n=args.n)

    print('Using seed: {:03d}'.format(args.random_seed))
    print('Start:', start)
    print('Goal:', goal)
    print('Start:', ' '.join(map(str,list(start))))
    print('Goal: ', ' '.join(map(str,list(goal))))

    # Define the macros / models
    if args.macro_type == 'random':
        macros.generate_random_macro_set(args.random_seed)

    macro_namespace = {
        'primitive': SimpleNamespace(macros=[], models=[]),
        'random': macros.random,
        'learned': macros.learned,
    }[args.macro_type]
    macro_list = macro_namespace.macros
    model_list = macro_namespace.models

    # Set up the search problem
    search_fn = {
        'astar': search.astar,
        'gbfs': search.gbfs,
        'weighted_astar': search.weighted_astar,
        'bfws_r0': bfws.bfws,
        'bfws_rg': bfws.bfws,
    }[args.search_alg]

    def get_successors(puz):
        successors = [(copy.deepcopy(puz).transition(a), [a]) for a in puz.actions()]
        if args.macro_type != 'primitive':
            valid_macros = macro_list[puz.blank_idx]
            valid_models = model_list[puz.blank_idx]
            macro_successors = [(copy.deepcopy(puz).apply_macro(model=model), macro)
                                for (macro, model) in zip(valid_macros, valid_models)]
            successors += macro_successors
        return successors

    search_dict = {
        'start': start,
        'is_goal': lambda node: node.state == goal,
        'step_cost': lambda macro: 1,
        'heuristic': lambda puz: len(puz.summarize_effects(baseline=goal)[0]),
        'get_successors': get_successors,
        'max_transitions': args.max_transitions,
    }

    if args.search_alg == 'weighted_astar':
        assert (args.g_weight is not None
                and args.h_weight is not None), 'Must specify weights if using weighted A*.'
        gh_weights = (args.g_weight, args.h_weight)
        search_dict['gh_weights'] = gh_weights

    if 'bfws' in args.search_alg:
        search_dict['precision'] = args.bfws_precision
    if args.search_alg == 'bfws_rg':
        goal_fns = [(lambda x, i=i: x.state[i] == goal[i]) for i, _ in enumerate(goal)]
        relevant_atoms = iw.iw(1, start, get_successors, goal_fns)
        if not relevant_atoms:
            relevant_atoms = iw.iw(2, start, get_successors, goal_fns)
        if not relevant_atoms:
            relevant_atoms = start.all_atoms()
        search_dict['R'] = relevant_atoms

    #%% Run the search
    search_results = search_fn(**search_dict)

    #%% Save the results
    tag = '{}-puzzle/'.format(args.n)
    if args.random_goal:
        tag += 'random_goal/'
    else:
        tag += 'default_goal/'
    tag += args.macro_type

    results_dir = 'results/npuzzle/{}/{}/'.format(args.search_alg,tag)
    os.makedirs(results_dir, exist_ok=True)
    with open(results_dir+'seed-{:03d}.pickle'.format(args.random_seed), 'wb') as file:
        pickle.dump(search_results, file)


if __name__ == '__main__':
    solve()
