import argparse
import copy
import os
import pickle
import random
import sys

import gym
import pddlgym

from domains.npuzzle import npuzzle
from experiments import search


def main():
    """Run best-first-search to find macros for the specified PDDLGym domain

    Use --help to see a pretty description of the arguments
    """
    if 'ipykernel' in sys.argv[0]:
        sys.argv = [sys.argv[0]]
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='miconic',
                        help='Name of PDDL domain')
    parser.add_argument('--seed','-s', type=int, default=1,
                        help='Seed to use for RNGs and problem index')
    parser.add_argument('--max_transitions', type=lambda x: int(float(x)), default=100000,
                        help='Maximum number of simulator transitions')
    parser.add_argument('--save_best_n', type=int, default=1000,
                        help='Number of best macros to save')
    args = parser.parse_args()

    # Set up the domain
    random.seed(args.seed)
    env = gym.make("PDDLEnv-Gen-{}-v0".format(args.env_name.capitalize()))
    env._render = None
    env.fix_problem_index(args.seed)
    env.seed(args.seed)

    start, _ = env.reset()
    env.action_space.seed(args.seed)
    goal = start.goal

    print('Using seed: {:03d}'.format(args.seed))
    print('Objects:', sorted(list(start.objects)))
    print('Goal:', goal)

    tag = '{}/problem-{:02d}'.format(args.env_name, args.seed)

    #%% Configure the search
    def heuristic(state):
        orig_literals = start.literals
        current_literals = state.literals
        union_of_literals = frozenset.union(orig_literals, current_literals)
        changes = [lit for lit in union_of_literals
                   if (lit not in orig_literals or lit not in current_literals)]
        return len(changes)

    def restore_state(state):
        env.set_state(state)
        return env

    def get_successors(state):
        env.set_state(state)
        valid_actions = sorted(list(env.action_space.all_ground_literals(state)))
        random.shuffle(valid_actions)
        successors = [(restore_state(state).step(a)[0], [a]) for a in valid_actions]
        return successors

    #%% Run the search
    search_results = search.astar(start = start,
                                  is_goal = lambda node: False,
                                  step_cost = lambda action: 1,
                                  heuristic = heuristic,
                                  get_successors = get_successors,
                                  max_transitions = args.max_transitions,
                                  save_best_n = args.save_best_n)

    #%% Save the results
    results_dir = 'results/macros/pddlgym-gen/{}/'.format(tag)
    os.makedirs(results_dir, exist_ok=True)
    macros_filename = results_dir+'seed{:03d}-macros.pickle'.format(args.seed, tag)
    with open(macros_filename, 'wb') as file:
        pickle.dump(search_results, file)


if __name__ == '__main__':
    main()
