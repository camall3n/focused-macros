import argparse
import copy
import numpy as np
import pickle
import random
import os
import sys
from suitcaselock.suitcaselock import SuitcaseLock
from notebooks import astar

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
search_results = astar.search(start, is_goal, step_cost, heuristic, get_successors, args.max_transitions)

#%% Save the results
tag = 'n_vars-{}/n_values-{}/entanglement-{}'
tag = tag.format(args.n_vars, args.n_values, args.entanglement, args.max_transitions)

results_dir = 'results/fixedsuitcaselock/{}/'.format(tag)
os.makedirs(results_dir, exist_ok=True)
with open(results_dir+'seed-{:03d}.pickle'.format(seed), 'wb') as f:
    pickle.dump(search_results, f)
