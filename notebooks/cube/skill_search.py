import argparse
import copy
import pickle
import os
import sys

from domains import cube
from notebooks import search
from domains.cube import options

version = '0.4'
cost_mode = 'per-skill'
max_transitions = 1e6
save_best_n = 1200

newcube = cube.Cube()
start = cube.Cube()

skills = options.primitive.actions
models = options.primitive.models

is_goal = lambda node: False

def heuristic(cube):
    effects = cube.summarize_effects()
    if len(effects) == 0:
        return float('inf')
    else:
        return len(effects)

if cost_mode == 'per-action':
    step_cost = lambda skill: len(skill)
elif cost_mode == 'per-skill':
    step_cost = lambda skill: 1

def get_successors(cube):
    return [(copy.deepcopy(cube).apply(swap_list=m), s) for s,m in zip(skills, models)]

#%% Run the search
search_results = search.astar(start, is_goal, step_cost, heuristic, get_successors, max_transitions, save_best_n)

#%% Save the results
os.makedirs('results/skillsearch/cube', exist_ok=True)
with open('results/skillsearch/cube/v{}-results.pickle'.format(version), 'wb') as f:
    pickle.dump(search_results, f)
