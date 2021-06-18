from collections import namedtuple

import numpy as np
import seaborn as sns

NAME = 'npuzzle'
DIR = NAME
PLOTS = ['learning_curves', 'planning_boxes']
SUMMARIES = ['macro_type', 'goal_type']
FIELDS = 'alg', 'puzzle_size', 'goal_type', 'macro_type', 'seed'
FIGSIZE = (4,3)
FONTSIZE = 12
HLINE = None
TRANSITION_CAP = 5e5
XLIM = [-100, 2e5]
YLIM = [0,16]

blue, orange, green, red, purple, brown, pink, gray, yellow, teal  = sns.color_palette('deep', n_colors=10)

PlotVars = namedtuple('PlotVars', ['macro_type', 'goal_type', 'color', 'zorder', 'tick_size'])
PLOT_VARS = [
    PlotVars('Random', 'default_goal', teal, 5, 100e3),
    PlotVars('Primitive', 'default_goal', blue, 10, 100e3),
    PlotVars('Focused', 'default_goal', red, 10, 100e3),
]


def get_goal(state, metadata):
    if metadata.goal_type == 'default_goal':
        goal = state.reset()
    else:
        goal = state.reset().scramble(seed=metadata.seed+1000)
    return goal

def heuristic(state, goal):
    return len(state.summarize_effects(baseline=goal)[0])

def get_primitive_steps(sequence):
    return len(np.concatenate(sequence))

def get_macro_steps(sequence):
    return len(sequence)

def get_macro_lengths(sequence):
    return list(map(len, sequence))
