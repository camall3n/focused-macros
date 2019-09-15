import copy
from cube import cube
from notebooks import astar
from cube import options
from cube import pattern
from matplotlib import pyplot as plt

seed = 1
skill_mode = 'expert'
cost_mode = 'per-skill'
max_transitions = 3e5
debug = False

# Set up the scramble
newcube = cube.Cube()
scramble = pattern.scramble(seed)

start = copy.deepcopy(newcube)
start.apply(scramble)

# Define the skills
if skill_mode == 'primitive':
    skills = options.primitive.actions
    models = options.primitive.models
elif skill_mode == 'expert':
    skills = options.primitive.actions + options.expert.options
    models = options.primitive.models + options.expert.models
elif skill_mode == 'random':
    skills = options.primitive.actions + options.random.options
    models = options.primitive.models + options.random.models

# Set up the search problem
is_goal = lambda node: node.state == newcube
heuristic = lambda cube: len(cube.summarize_effects())

if cost_mode == 'per-action':
    step_cost = lambda skill: len(skill)
elif cost_mode == 'per-skill':
    step_cost = lambda skill: 1

def get_successors(cube):
    return [(copy.deepcopy(cube).apply(swap_list=m), s) for s,m in zip(skills, models)]

#%% Run the search
states, actions, n_expanded, n_transitions, candidates = astar.search(start, is_goal, step_cost, heuristic, get_successors, max_transitions)

n_errors = len(states[-1].summarize_effects())
x = [c for c,n in candidates]
y = [n.h_score for c,n in candidates]

#%%
x
y
sum(len(a) for a in actions)
len(actions)
n_expanded
n_transitions
n_errors

plt.plot(x,y)
