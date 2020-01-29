import argparse
import copy
import pickle
import os
import sys

from domains import cube
from notebooks import search
from domains.cube import options, pattern
from matplotlib import pyplot as plt

if 'ipykernel' in sys.argv[0]:
    sys.argv = [sys.argv[0]]
parser = argparse.ArgumentParser()
parser.add_argument('--scramble_seed','-s', type=int, default=1,
                    help='Seed to use for initial scramble')
parser.add_argument('--skill_mode','-m', type=str, default='expert',
                    choices=['primitive','expert','fixed_random','full_random','generated'],
                    help='Type of skills to consider during search')
parser.add_argument('--skill_version','-v', type=str, default='0.4',
                    choices=['0.1','0.2','0.3','0.4'],
                    help='Which version to use for generated skills')
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
skill_mode = args.skill_mode
cost_mode = 'per-skill'
debug = False

# Set up the scramble
newcube = cube.Cube()
scramble = pattern.scramble(seed)

start = copy.deepcopy(newcube)
start.apply(scramble)
print('Using scramble: {:03d}'.format(seed))
start.render()

# Define the skills
if skill_mode == 'primitive':
    skills = options.primitive.actions
    models = options.primitive.models
elif skill_mode == 'expert':
    skills = options.primitive.actions + options.expert.options
    models = options.primitive.models + options.expert.models
elif 'random' in skill_mode:
    if skill_mode == 'full_random':
        options.set_random_skill_seed(seed)
    skills = options.primitive.actions + options.random.options
    models = options.primitive.models + options.random.models
elif skill_mode == 'generated':
    options.load_generated_skills(args.skill_version)
    skills = options.primitive.actions + options.generated.options
    models = options.primitive.models + options.generated.models

if args.random_goal:
    goal = cube.Cube().apply(pattern.scramble(seed+1000))
    print('Using goal pattern: {:03d}'.format(seed+1000))
else:
    goal = newcube

# Set up the search problem
is_goal = lambda node: node.state == goal
heuristic = lambda cube: len(cube.summarize_effects(baseline=goal))

if cost_mode == 'per-action':
    step_cost = lambda skill: len(skill)
elif cost_mode == 'per-skill':
    step_cost = lambda skill: 1

def get_successors(cube):
    return [(copy.deepcopy(cube).apply(swap_list=m), s) for s,m in zip(skills, models)]

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
tag = skill_mode
if skill_mode == 'generated':
    tag += '-v{}'.format(args.skill_version)
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
