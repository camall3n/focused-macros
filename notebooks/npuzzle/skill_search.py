import argparse
import copy
import pickle
import os
import sys
from npuzzle import npuzzle
from notebooks import astar
from cube import options

version = '0.1'
max_transitions = 1e6
save_best_n = 1200

newpuz = npuzzle.NPuzzle()
start = npuzzle.NPuzzle()

is_goal = lambda node: False
step_cost = lambda skill: 1
def heuristic(puz):
    swap_list, starting_blank_idx = puz.summarize_effects(baseline=newpuz)
    if len(swap_list) == 0:
        return float('inf')
    else:
        return len(swap_list)

def get_successors(puz):
    return [(copy.deepcopy(puz).transition(a) for a in puz.actions())]

#%% Run the search
search_results = astar.search(start, is_goal, step_cost, heuristic, get_successors, max_transitions, save_best_n)

#%% Save the results
os.makedirs('results/skillsearch', exist_ok=True)
with open('results/skillsearch/v{}-results.pickle'.format(version), 'wb') as f:
    pickle.dump(search_results, f)
