import copy
import pickle
import os

from domains import cube
from domains.cube import macros
from experiments import search


def main():
    """Search for disentangled macro-actions using A*"""
    cost_mode = 'per-macro'
    max_transitions = 1e6
    save_best_n = 1200

    newcube = cube.Cube()
    start = cube.Cube()

    actions = macros.primitive.actions
    models = macros.primitive.models

    is_goal = lambda node: False
    step_cost = lambda macro: len(macro) if cost_mode == 'per-action' else 1

    def heuristic(cube_):
        effects = cube_.summarize_effects(baseline=newcube)
        if len(effects) == 0:
            return float('inf')
        return len(effects)

    def get_successors(cube_):
        return [(copy.deepcopy(cube_).apply(swap_list=m), a) for a, m in zip(actions, models)]

    #%% Run the search
    search_results = search.astar(start=start,
                                  is_goal=is_goal,
                                  step_cost=step_cost,
                                  heuristic=heuristic,
                                  get_successors=get_successors,
                                  max_transitions=max_transitions,
                                  save_best_n=save_best_n)

    #%% Save the results
    os.makedirs('results/macros/cube', exist_ok=True)
    with open('results/macros/cube/macro_results.pickle', 'wb') as file:
        pickle.dump(search_results, file)

if __name__ == '__main__':
    main()
