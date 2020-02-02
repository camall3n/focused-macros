import copy

from domains import cube
from domains.cube import pattern, macros
from notebooks import search

def test():
    """Test search functionality with Cube primitive actions and expert macro-actions"""
    # Set up the scramble
    newcube = cube.Cube()

    # Primitive action search
    print('Running primitive action search...')
    start = copy.deepcopy(newcube)
    start.apply(pattern.SCRAMBLE_1)

    skills = macros.primitive.actions
    models = macros.primitive.models

    is_goal = lambda node: node.state == newcube
    heuristic = lambda cube: len(cube.summarize_effects())
    max_transitions = 3e3
    def get_successors(cube_):
        return [(copy.deepcopy(cube_).apply(swap_list=model), macro)
                for (macro, model) in zip(skills, models)]

    search_results = search.astar(start=start,
                                  is_goal=is_goal,
                                  step_cost=len,
                                  heuristic=heuristic,
                                  get_successors=get_successors,
                                  max_transitions=max_transitions)
    actions = search_results[1]

    testcube = copy.deepcopy(newcube)
    testcube.apply(pattern.SCRAMBLE_1)
    n_starting_errors = len(testcube.summarize_effects())
    for action in actions:
        testcube.apply(action)
    n_remaining_errors_primitive = len(testcube.summarize_effects())
    assert n_remaining_errors_primitive < n_starting_errors


    # Expert macro-action search
    print('Running expert macro-action search...')
    start = copy.deepcopy(newcube)
    start.apply(pattern.SCRAMBLE_1)

    skills = macros.primitive.actions + macros.expert.macros
    models = macros.primitive.models + macros.expert.models

    search_results = search.astar(start=start,
                                  is_goal=is_goal,
                                  step_cost=lambda _: 1,
                                  heuristic=heuristic,
                                  get_successors=get_successors,
                                  max_transitions=max_transitions)
    actions = search_results[1]

    testcube = copy.deepcopy(newcube)
    testcube.apply(pattern.SCRAMBLE_1)
    n_starting_errors = len(testcube.summarize_effects())
    for action in actions:
        testcube.apply(action)
    n_remaining_errors_expert = len(testcube.summarize_effects())
    assert n_remaining_errors_expert < n_remaining_errors_primitive

    print('All tests passed.')

if __name__ == '__main__':
    test()
