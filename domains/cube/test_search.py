import copy

from domains import cube
from notebooks import search
from domains.cube.macros import primitive, expert, random
from domains.cube import pattern

def test():
    # Set up the scramble
    newcube = cube.Cube()
    scramble = pattern.SCRAMBLE_1

    start = copy.deepcopy(newcube)
    start.apply(scramble)
    start.render()

    #%% Run the search
    skills = primitive.actions
    models = primitive.models

    def is_goal(node): node.state == newcube
    def step_cost(skill): len(skill)
    def heuristic(cube): len(cube.summarize_effects())
    max_transitions = 3e3
    def debug_fn(cube): cube.render() if cube else None
    def get_successors(cube):
        return [(copy.deepcopy(cube).apply(swap_list=m), s) for s,m in zip(skills, models)]

    #%%
    states, actions, n_expanded, n_transitions = search.astar(start, is_goal, step_cost, heuristic, get_successors, max_transitions)[:4]

    #%%
    for s in states:
        s.render()
    #%%
    testcube = copy.deepcopy(newcube)
    testcube.apply(scramble)
    for a in actions:
        testcube.apply(a)
    n_errors = len(testcube.summarize_effects())
    #%%
    actions
    sum(len(a) for a in actions)
    n_expanded
    n_transitions
    n_errors

    #%%
    start = copy.deepcopy(newcube)
    start.apply(scramble)
    start.render()

    #%% Run the search
    skills = primitive.actions + expert.macros
    models = primitive.models + expert.models
    def step_cost(skill): 1

    #%%
    states, actions, n_expanded, n_transitions = search.astar(start, is_goal, step_cost, heuristic, get_successors, max_transitions)[:4]

    #%%
    for s in states:
        s.render()
    #%%
    testcube = copy.deepcopy(newcube)
    testcube.apply(scramble)
    for a in actions:
        testcube.apply(a)
    n_errors = len(testcube.summarize_effects())
    #%%
    actions
    sum(len(a) for a in actions)
    len(actions)
    n_expanded
    n_transitions
    n_errors

if __name__ == '__main__':
    test()