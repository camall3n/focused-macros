import copy

from domains import cube
from notebooks import search
from domains.cube.options import primitive, expert, random
from domains.cube import pattern

# Set up the scramble
newcube = cube.Cube()
scramble = pattern.scramble1

start = copy.deepcopy(newcube)
start.apply(scramble)
start.render()

#%% Run the search
skills = primitive.actions
models = primitive.models

is_goal = lambda node: node.state == newcube
step_cost = lambda skill: len(skill)
heuristic = lambda cube: len(cube.summarize_effects())
max_transitions = 3e3
debug_fn = lambda cube: cube.render() if cube else None
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
skills = primitive.actions + expert.options
models = primitive.models + expert.models
step_cost = lambda skill: 1

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
