import copy
import random
from tqdm import tqdm

from cube import cube
from cube import skills
from cube.options import expert
from cube import pattern

c = cube.Cube()
c.apply(pattern.scramble1)
c.render()

#%%

mods = c.summarize_effects()

action_steps = ["L'", 'B', 'R', 'R', "B'", "D'", "B'", "F'", 'R', 'F', 'L', "D'", "D'", "L'"]
n_action_steps = len(action_steps)
action_experiences = 1359208

option_steps = [
    ['F', 'U', "B'", "U'", "L'", "B'", 'L', "F'", "L'", 'B', 'L', 'U', 'B', "U'"],
    ["B'", 'L', 'B', "R'", "B'", "L'", 'B', 'R'],
    ["F'", 'L', 'F', "R'", "F'", "L'", 'F', 'R'],
    ["L'", 'R', 'F', "L'", 'R', 'U', "L'", 'R', 'B', "L'", 'R', 'D', 'D', "L'", 'R', 'F', "L'", 'R', 'U', "L'", 'R', 'B', "L'", 'R'],
    ["L'", 'B', "L'", "B'", "L'", "B'", "L'", 'B', 'L', 'B', 'L', 'L'],
    ["D'", "D'", "L'", "D'", "L'", 'D', 'L', 'D', 'L', 'D', "L'", 'D'],
    ['F', "B'", "U'", 'F', "B'", "R'", 'F', "B'", "D'", 'F', "B'", "L'", "L'", 'F', "B'", "U'", 'F', "B'", "R'", 'F', "B'", "D'", 'F', "B'"],
    ['D', 'F', "U'", "F'", "L'", "U'", 'L', "D'", "L'", 'U', 'L', 'F', 'U', "F'"],
    ['U', 'R', "D'", "R'", "U'", 'R', 'D', "R'"],
    ["L'", "F'", 'R', 'F', 'D', 'R', "D'", 'L', 'D', "R'", "D'", "F'", "R'", 'F'],
    ['D', "R'", 'D', 'R', 'D', 'R', 'D', "R'", "D'", "R'", "D'", "D'"],
    ['F', "B'", "L'", 'F', "B'", "U'", 'F', "B'", "R'", 'F', "B'", "D'", "D'", 'F', "B'", "L'", 'F', "B'", "U'", 'F', "B'", "R'", 'F', "B'"],
    ["D'", 'U', 'F', "D'", 'U', 'L', "D'", 'U', 'B', "D'", 'U', 'R', 'R', "D'", 'U', 'F', "D'", 'U', 'L', "D'", 'U', 'B', "D'", 'U'],
    ['F', 'F', 'R', 'F', 'R', "F'", "R'", "F'", "R'", "F'", 'R', "F'"],
    ["B'", 'F', 'U', 'U', "F'", 'B', 'L', 'L'],
    ["R'", 'U', "R'", "U'", "R'", "U'", "R'", 'U', 'R', 'U', 'R', 'R'],
    ["U'", "U'", "B'", "U'", "B'", 'U', 'B', 'U', 'B', 'U', "B'", 'U'],
    ['R', "L'", "D'", "D'", 'L', "R'", "B'", "B'"],
    ['B', 'B', 'U', 'B', 'U', "B'", "U'", "B'", "U'", "B'", 'U', "B'"],
    ["R'", 'L', 'F', 'F', "L'", 'R', 'U', 'U'],
    ['U', "L'", "U'", 'R', 'U', 'L', "U'", "R'"],
    ["U'", "B'", 'D', 'B', 'U', "B'", "D'", 'B']
]
option_steps = [
    ["B'", 'L', 'D', "B'", "B'", 'F', "R'", 'D', 'D', 'L', 'D', "R'"],
    ['L', 'B', "U'", "L'", 'F', 'R', 'F', "L'", 'U', "D'"],
    ['R', "L'", 'B', "D'", "F'", "D'", 'B', 'L', "U'", "B'"],
    ["R'", 'L', "U'", 'F', 'D', 'F', "U'", "L'", 'B', 'U'],
    ["U'", "U'", 'B', "F'", "F'", 'U', 'R', 'D', 'R', "D'", "B'", "U'", 'B', "D'", 'R', 'R', "U'", 'D', 'D', "U'"],
    ['F', 'F', "D'", 'U', 'U', "F'", "R'", "B'", "R'", 'B', 'D', 'F', "D'", 'B', "R'", "R'", 'F', "B'", "B'", 'F'],
    ['R', 'R', "U'", 'D', 'D', "R'", "F'", "L'", "F'", 'L', 'U', 'R', "U'", 'L', "F'", "F'", 'R', "L'", "L'", 'R'],
    ['L', "R'", "R'", 'L', "F'", "F'", 'R', "U'", 'L', 'U', 'R', "F'", "R'", "F'", "L'", 'D', 'D', "U'", 'L', 'L'],
    ['R', "L'", 'B', "D'", "F'", "D'", 'B', 'L', "U'", "B'"],
    ["B'", "R'", 'D', 'B', "L'", "F'", "L'", 'B', "D'", 'U']
]


n_option_steps = sum([len(o) for o in option_steps])
option_experiences = 700368

a_cube = copy.deepcopy(c)
for step in action_steps:
    a_cube.apply([step])
a_cube.render()
print(n_action_steps)

#%%
o_cube = copy.deepcopy(c)
for step in option_steps:
    o_cube.apply(step)
o_cube.render()
print(n_option_steps)
