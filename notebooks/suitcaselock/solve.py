import argparse
import copy
import numpy as np
import pickle
import random
import os
import sys
from suitcaselock.suitcaselock import SuitcaseLock
from notebooks import search

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
parser.add_argument('--search_alg', type=str, default='astar', choices = ['astar', 'gbfs', 'weighted-astar'],
                    help='Search algorithm to run')
parser.add_argument('--g_weight', type=float, default=None,
                    help='Weight for g-score in weighted A*')
parser.add_argument('--h_weight', type=float, default=None,
                    help='Weight for h-score in weighted A*')
parser.add_argument('--max_transitions', type=lambda x: int(float(x)), default=1e5,
                    help='Maximum number of state transitions')
args = parser.parse_args()
#
seed = args.random_seed
cost_mode = 'per-skill'

# Set up the scramble
random.seed(seed)
np.random.seed(seed)

newlock = SuitcaseLock(n_vars=args.n_vars, n_values=args.n_values, entanglement=args.entanglement)
start = copy.deepcopy(newlock).scramble(seed=seed)
goal = copy.deepcopy(newlock).scramble(seed=seed+1000)

print('Using seed: {:03d}'.format(seed))
print('Start:', start)
print('Goal:', goal)

# Define the skills
skills = newlock.actions()
s = skills[0]

models = [copy.deepcopy(newlock).apply_macro(diff=s).summarize_effects(baseline=newlock) for s in skills]

# Set up the search problem
is_goal = lambda node: node.state == goal
heuristic = lambda lock: sum(lock.summarize_effects(baseline=goal) > 0)
step_cost = lambda skill: 1

def get_successors(lock):
    return [(copy.deepcopy(lock).apply_macro(diff=m), s) for s,m in zip(skills, models)]

#%% Run the search
if args.search_alg == 'astar':
    search_results = search.astar(start, is_goal, step_cost, heuristic, get_successors, args.max_transitions)
elif args.search_alg == 'gbfs':
    search_results = search.gbfs(start, is_goal, step_cost, heuristic, get_successors, args.max_transitions)
elif args.search_alg == 'weighted-astar':
    assert args.g_weight is not None and args.h_weight is not None, 'Must specify weights if using weighted A*.'
    gh_weights = args.g_weight, args.h_weight
    search_results = search.weighted_astar(start, is_goal, step_cost, heuristic, get_successors, args.max_transitions, gh_weights=gh_weights)

#%% Save the results
tag = 'n_vars-{}/n_values-{}/entanglement-{}'
tag = tag.format(args.n_vars, args.n_values, args.entanglement, args.max_transitions)

results_dir = 'results/suitcaselock/{}/{}/'.format(args.search_alg, tag)
os.makedirs(results_dir, exist_ok=True)
with open(results_dir+'seed-{:03d}.pickle'.format(seed), 'wb') as f:
    pickle.dump(search_results, f)
