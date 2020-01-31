import argparse
import copy
import pickle
import os
import sys

from domains import cube
from notebooks import search
from domains.cube import macros

version = '0.4'
cost_mode = 'per-macro'
max_transitions = 1e6
save_best_n = 1200

newcube = cube.Cube()
start = cube.Cube()

actions = macros.primitive.actions
models = macros.primitive.models

def is_goal(node): False

def heuristic(cube):
    effects = cube.summarize_effects()
    if len(effects) == 0:
        return float('inf')
    else:
        return len(effects)

if cost_mode == 'per-action':
    def step_cost(macro): len(macro)
elif cost_mode == 'per-macro':
    def step_cost(macro): 1

def get_successors(cube):
    return [(copy.deepcopy(cube).apply(swap_list=m), a) for a,m in zip(actions, models)]

#%% Run the search
search_results = search.astar(start, is_goal, step_cost, heuristic, get_successors, max_transitions, save_best_n)

#%% Save the results
os.makedirs('results/macros/cube', exist_ok=True)
with open('results/macros/cube/v{}-results.pickle'.format(version), 'wb') as f:
    pickle.dump(search_results, f)
