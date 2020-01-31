import argparse
import copy
import pickle
import os
import sys

from domains import cube
from notebooks import search
from domains.cube import macros, pattern
from matplotlib import pyplot as plt

if 'ipykernel' in sys.argv[0]:
    sys.argv = [sys.argv[0]]
parser = argparse.ArgumentParser()
parser.add_argument('--scramble_seed','-s', type=int, default=1,
                    help='Seed to use for initial scramble')
parser.add_argument('--macro_type','-m', type=str, default='expert',
                    choices=['primitive','expert','fixed_random','full_random','generated'],
                    help='Type of macros to consider during search')
parser.add_argument('--macro_version','-v', type=str, default='0.4',
                    choices=['0.1','0.2','0.3','0.4'],
                    help='Which version to use for generated macros')
parser.add_argument('--search_alg', type=str, default='gbfs', choices=['astar','gbfs','weighted_astar'],
                    help='Search algorithm to run')
parser.add_argument('--g_weight', type=float, default=None,
                    help='Weight for g-score in weighted A*')
parser.add_argument('--h_weight', type=float, default=None,
                    help='Weight for h-score in weighted A*')
parser.add_argument('--random-goal','-r', action='store_true', default=False,
                    help='Generate a random goal instead of the default solve configuration')
parser.add_argument('--max_transitions', type=lambda x: int(float(x)), default=1e5,
                    help='Maximum number of variables changed per primitive action')
args = parser.parse_args()

seed = args.scramble_seed
macro_type = args.macro_type
cost_mode = 'per-macro'
debug = False

# Set up the scramble
newcube = cube.Cube()
scramble = pattern.scramble(seed)

start = copy.deepcopy(newcube)
start.apply(sequence=scramble)
print('Using scramble: {:03d}'.format(seed))
start.render()

# Define the macros and models
if macro_type == 'primitive':
    macro_list = macros.primitive.actions
    model_list = macros.primitive.models
elif macro_type == 'expert':
    macro_list = macros.primitive.actions + macros.expert.macros
    model_list = macros.primitive.models + macros.expert.models
elif 'random' in macro_type:
    if macro_type == 'full_random':
        macros.generate_random_macro_set(seed)
    macro_list = macros.primitive.actions + macros.random.macros
    model_list = macros.primitive.models + macros.random.models
elif macro_type == 'generated':
    macros.load_learned_macros(args.macro_version)
    macro_list = macros.primitive.actions + macros.generated.macros
    model_list = macros.primitive.models + macros.generated.models

if args.random_goal:
    goal = cube.Cube().apply(sequence=pattern.scramble(seed+1000))
    print('Using goal pattern: {:03d}'.format(seed+1000))
else:
    goal = newcube

# Set up the search problem
def is_goal(node): node.state == goal
def heuristic(cube): len(cube.summarize_effects(baseline=goal))

if cost_mode == 'per-action':
    def step_cost(macro): len(macro)
elif cost_mode == 'per-macro':
    def step_cost(macro): 1

def get_successors(cube):
    return [(copy.deepcopy(cube).apply(swap_list=model), macro) for macro,model in zip(macro_list, model_list)]

#%% Run the search
search_alg = args.search_alg
if search_alg == 'astar':
    search_results = search.astar(start, is_goal, step_cost, heuristic, get_successors, args.max_transitions)
elif search_alg == 'gbfs':
    search_results = search.gbfs(start, is_goal, step_cost, heuristic, get_successors, args.max_transitions)
elif search_alg == 'weighted_astar':
    assert args.g_weight is not None and args.h_weight is not None, 'Must specify weights if using weighted A*.'
    gh_weights = args.g_weight, args.h_weight
    search_results = search.weighted_astar(start, is_goal, step_cost, heuristic, get_successors, args.max_transitions, gh_weights=gh_weights)

#%% Save the results
tag = macro_type
if macro_type == 'generated':
    tag += '-v{}'.format(args.macro_version)
if args.random_goal:
    tag = 'random_goal/'+tag
else:
    tag = 'default_goal/'+tag
if search_alg == 'weighted_astar':
    search_alg += '-g_{}-h_{}'.format(*gh_weights)
results_dir = 'results/cube/{}/{}/'.format(search_alg, tag)
os.makedirs(results_dir, exist_ok=True)
with open(results_dir+'/seed-{:03d}.pickle'.format(seed), 'wb') as f:
    pickle.dump(search_results, f)
