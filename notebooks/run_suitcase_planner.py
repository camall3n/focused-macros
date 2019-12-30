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
parser.add_argument('--skill_mode','-m', type=str, default='primitive',
                    choices=['primitive','expert','fixed_random','full_random','generated'],
                    help='Type of skills to consider during search')
parser.add_argument('--n_vars', type=int, default=4,
                    help='Number of variables')
parser.add_argument('--n_values', type=int, default=10,
                    help='Number of possible values for each variable')
parser.add_argument('--max_vars_per_action', type=int, default=1,
                    help='Maximum number of variables changed per primitive action')
parser.add_argument('--max_transitions', type=lambda x: int(float(x)), default=1e5,
                    help='Maximum number of variables changed per primitive action')
args = parser.parse_args()
#
seed = args.random_seed
skill_mode = args.skill_mode
cost_mode = 'per-skill'

# Set up the scramble
random.seed(seed)
np.random.seed(seed)

newlock = SuitcaseLock(n_vars=args.n_vars, n_values=args.n_values, max_vars_per_action=args.max_vars_per_action)
start = copy.deepcopy(newlock).scramble(seed=seed)
goal = copy.deepcopy(newlock).scramble(seed=seed+1000)

print('Using seed: {:03d}'.format(seed))
print('Start:', start)
print('Goal:', goal)

# Define the skills
if skill_mode == 'primitive':
    skills = newlock.actions()
    s = skills[0]

    models = [copy.deepcopy(newlock).apply_macro(diff=s).summarize_effects(baseline=newlock) for s in skills]
# elif skill_mode == 'expert':
#     skills = options.primitive.actions + options.expert.options
#     models = options.primitive.models + options.expert.models
# elif 'random' in skill_mode:
#     if skill_mode == 'full_random':
#         options.set_random_skill_seed(seed)
#     skills = options.primitive.actions + options.random.options
#     models = options.primitive.models + options.random.models
# elif skill_mode == 'generated':
#     options.load_generated_skills(args.skill_version)
#     skills = options.primitive.actions + options.generated.options
#     models = options.primitive.models + options.generated.models



# Set up the search problem
is_goal = lambda node: node.state == goal
start == goal
heuristic = lambda lock: sum(lock.summarize_effects(baseline=goal) > 0)

if cost_mode == 'per-action':
    step_cost = lambda skill: len(skill)
elif cost_mode == 'per-skill':
    step_cost = lambda skill: 1

def get_successors(lock):
    return [(copy.deepcopy(lock).apply_macro(diff=m), s) for s,m in zip(skills, models)]

#%% Run the search
search_results = astar.search(start, is_goal, step_cost, heuristic, get_successors, args.max_transitions)

#%% Save the results
tag = skill_mode
tag = 'max_vars_{}/'.format(args.max_vars_per_action) + tag
if skill_mode == 'generated':
    tag += '-v{}'.format(args.skill_version)
results_dir = 'results/suitcaselock/{}/'.format(tag)
os.makedirs(results_dir, exist_ok=True)
with open(results_dir+'seed-{:03d}.pickle'.format(seed), 'wb') as f:
    pickle.dump(search_results, f)
