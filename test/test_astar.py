import copy
from cube import cube
from notebooks import astar

# Set up the scramble
newcube = cube.Cube()
scramble = 'R U L D F'.split()

start = copy.deepcopy(newcube)
start.apply(scramble)
start.render()

#%% Run the search
skills = [[a] for a in cube.Action.keys()]

is_goal = lambda node: node.state == newcube
step_cost = lambda skill: len(skill)
heuristic = lambda cube: len(cube.summarize_effects())
max_transitions = 3e4
debug_fn = lambda cube: cube.render() if cube else None
def get_successors(cube):
    return [(copy.deepcopy(cube).apply(s), s) for s in skills]

states, actions, n_expanded, n_transitions = astar.search(start, is_goal, step_cost, heuristic, get_successors, max_transitions)

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
n_expanded
n_transitions
n_errors
