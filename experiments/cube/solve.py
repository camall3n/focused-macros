import argparse
import copy
import os
import pickle
import sys
from types import SimpleNamespace

from domains import cube
from domains.cube import macros, pattern, formula
from experiments import search, iw, bfws


def parse_args():
    """Parse input arguments

    Use --help to see a pretty description of the arguments
    """
    if 'ipykernel' in sys.argv[0]:
        sys.argv = [sys.argv[0]]
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed','-s', type=int, default=1,
                        help='Seed to use for initial scramble')
    parser.add_argument('--macro_type','-m', type=str, default='expert',
                        choices=['primitive','expert','random','learned'],
                        help='Type of macros to consider during search')
    parser.add_argument('--search_alg', type=str, default='gbfs',
                        choices=['astar','gbfs','weighted_astar', 'bfws_r0', 'bfws_rg'],
                        help='Search algorithm to run')
    parser.add_argument('--cost_mode', type=str, default='per-macro',
                        choices=['per-macro','per-action'],
                        help='How to measure the plan cost')
    parser.add_argument('--g_weight', type=float, default=None,
                        help='Weight for g-score in weighted A*')
    parser.add_argument('--h_weight', type=float, default=None,
                        help='Weight for h-score in weighted A*')
    parser.add_argument('--random_goal','-r', action='store_true', default=False,
                        help='Generate a random goal instead of the default solve configuration')
    parser.add_argument('--buchner2018', action='store_true', default=False,
                        help='Use the Büchner 2018 problems instead of generating a scramble (for comparing with SAS+ planners)')
    parser.add_argument('--max_transitions', type=lambda x: int(float(x)), default=1e5,
                        help='Maximum number of variables changed per primitive action')
    parser.add_argument('--bfws_precision', type=int, default=3,
                        help='The number of width values, w \in {1,...,P}, to use when the search algorithm is best-first width search')
    return parser.parse_args()


def solve():
    """Instantiate a Rubik's cube and solve with the specified macro-actions and search algorithm"""
    args = parse_args()

    # Set up the scramble
    if args.buchner2018:
        if args.random_goal:
            raise RuntimeError('--random_goal is incompatible with --buchner2018')
        scramble = pattern.buchner2018pattern(args.seed)
    else:
        scramble = pattern.scramble(args.seed)

    start = cube.Cube()
    start.apply(sequence=scramble)
    print('Using scramble: {:03d}'.format(args.seed))
    start.render()

    if args.random_goal:
        goal = cube.Cube().apply(sequence=pattern.scramble(args.seed+1000))
        print('Using goal pattern: {:03d}'.format(args.seed+1000))
    else:
        goal = cube.Cube()

    # Define the macros and models
    if args.macro_type == 'random':
        macros.generate_random_macro_set(args.seed)

    macro_namespace = {
        'primitive': SimpleNamespace(macros=[], models=[]),
        'expert': macros.expert,
        'random': macros.random,
        'learned': macros.learned,
    }[args.macro_type]
    macro_list = macros.primitive.actions + macro_namespace.macros
    model_list = macros.primitive.models + macro_namespace.models

    # Set up the search problem
    search_fn = {
        'astar': search.astar,
        'gbfs': search.gbfs,
        'weighted_astar': search.weighted_astar,
        'bfws_r0': bfws.bfws,
        'bfws_rg': bfws.bfws,
    }[args.search_alg]

    def get_successors(cube_):
        return [(copy.deepcopy(cube_).apply(swap_list=model), macro)
                for (macro, model) in zip(macro_list, model_list)]

    search_dict = {
        'start': start,
        'is_goal': lambda node: node.state == goal,
        'step_cost': lambda macro: len(macro) if args.cost_mode == 'per-action' else 1,
        'heuristic': lambda cube_: len(cube_.summarize_effects(baseline=goal)),
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
    tag = args.macro_type
    if args.random_goal:
        tag = 'random_goal/'+tag
    else:
        tag = 'default_goal/'+tag
    if args.search_alg == 'weighted_astar':
        args.search_alg += '-g_{}-h_{}'.format(*gh_weights)
    problem_name = 'cube' if not args.buchner2018 else 'cube-buchner2018'
    results_dir = 'results/{}/{}/{}/'.format(problem_name, args.search_alg, tag)
    os.makedirs(results_dir, exist_ok=True)
    with open(results_dir+'/seed-{:03d}.pickle'.format(args.seed), 'wb') as file:
        pickle.dump(search_results, file)


if __name__ == '__main__':
    solve()
