import argparse
import copy
import numpy as np
import pickle
import random
import os
import sys
from npuzzle.npuzzle import NPuzzle
from notebooks import astar

if 'ipykernel' in sys.argv[0]:
    sys.argv = [sys.argv[0]]
parser = argparse.ArgumentParser()
parser.add_argument('-n', type=int, default=15, choices=[8, 15, 24, 35, 48, 63, 80],
                    help='Number of tiles')
parser.add_argument('--random_seed','-s', type=int, default=1,
                    help='Seed to use for RNGs')
parser.add_argument('--skill_mode','-m', type=str, default='primitive',
                    choices=['primitive','random','generated'],
                    help='Type of skills to consider during search')
parser.add_argument('--random-goal','-r', action='store_true', default=False,
                    help='Generate a random goal instead of the default solve configuration')
parser.add_argument('--max_transitions', type=lambda x: int(float(x)), default=1e5,
                    help='Maximum number of state transitions')
args = parser.parse_args()
#
seed = args.random_seed
cost_mode = 'per-skill'

# Set up the scramble
random.seed(seed)
np.random.seed(seed)

newpuz = NPuzzle(n=args.n)
start = copy.deepcopy(newpuz).scramble(seed=seed)
if args.random_goal:
    goal = copy.deepcopy(newpuz).scramble(seed=seed+1000)
    print('Using goal pattern: {:03d}'.format(seed+1000))
else:
    goal = newpuz

print('Using seed: {:03d}'.format(seed))
print('Start:', start)
print('Goal:', goal)

# Define the skills
if args.skill_mode == 'primitive':
    skills = []
    models = []
elif args.skill_mode == 'random':
    raise NotImplementedError
elif args.skill_mode == 'generated':
    raise NotImplementedError
    skills = []
    models = [copy.deepcopy(newpuz).apply_macro(diff=s).summarize_effects(baseline=newpuz) for s in skills]


# Set up the search problem
is_goal = lambda node: node.state == goal
heuristic = lambda puz: len(puz.summarize_effects(baseline=goal)[0])
step_cost = lambda skill: 1

def get_successors(puz):
    primitive_successors = [(copy.deepcopy(puz).transition(a), [a]) for a in puz.actions()]
    macro_successors = [(copy.deepcopy(puz).apply_macro(model=m), s) for s,m in zip(skills, models)]
    return primitive_successors+macro_successors

#%% Run the search
search_results = astar.search(start, is_goal, step_cost, heuristic, get_successors, args.max_transitions)

#%% Save the results
tag = '{}-puzzle/'.format(args.n)
if args.random_goal:
    tag += 'random_goal/'
else:
    tag += 'default_goal/'
tag += args.skill_mode
# if skill_mode == 'generated':
#     tag += '-v{}'.format(args.skill_version)

results_dir = 'results/npuzzle/{}/'.format(tag)
os.makedirs(results_dir, exist_ok=True)
with open(results_dir+'seed-{:03d}.pickle'.format(seed), 'wb') as f:
    pickle.dump(search_results, f)
