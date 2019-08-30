import copy
from cube import cube
from notebooks import astar

start = cube.Cube()
newcube = copy.deepcopy(start)
newcube.transform('R')
newcube.render()
start.render()

is_goal = lambda cube: cube == cube.Cube()
heuristic = lambda cube: len(cube.summarize_effects())

def get_successors(cube):
    [copy.deepcopy(cube).transform(a) for a in cube.Action.keys]

astar.search(start, is_goal, 1, heuristic, get_successors)
