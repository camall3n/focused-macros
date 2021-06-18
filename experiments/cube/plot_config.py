from collections import namedtuple

import numpy as np
import seaborn as sns

from domains.cube import pattern

NAME = 'cube'
DIR = NAME
PLOTS = ['learning_curves', 'alt_learning_curves', 'alt_planning_boxes']
SUMMARIES = ['macro_type', 'goal_type']
FIELDS = 'alg', 'goal_type', 'macro_type', 'seed'
FIGSIZE = (4,3)
FONTSIZE = 12
HLINE = 48
TRANSITION_CAP = 2e6
XLIM = [0,TRANSITION_CAP]
YLIM = [0,50]

blue, orange, green, red, purple, brown, pink, gray, yellow, teal  = sns.color_palette('deep', n_colors=10)

PlotVars = namedtuple('PlotVars', ['macro_type', 'goal_type', 'color', 'zorder', 'tick_size'])
PLOT_VARS = [
    PlotVars('Primitive', 'default_goal', blue, 15, 0.5e6),
    PlotVars('Random', 'default_goal', teal, 5, 0.5e6),
    PlotVars('Focused', 'default_goal', red, 10, 0.5e6),
    PlotVars('Expert', 'default_goal', orange, 20, 0.5e6),
]
PLOT_VARS_ALT = [
    PlotVars('focused', 'default_goal', red, 10, 0.5e6),
    PlotVars('focused', 'random_goal', purple, 20, 0.5e6),
]



def get_goal(state, metadata):
    if metadata.goal_type == 'default_goal':
        goal = state.reset()
    else:
        goal = state.reset().apply(sequence=pattern.scramble(seed=metadata.seed+1000))
    return goal

def heuristic(state, goal):
    return len(state.summarize_effects(baseline=goal))

def get_primitive_steps(sequence):
    return len(np.concatenate(sequence))

def get_macro_steps(sequence):
    return len(sequence)

def get_macro_lengths(sequence):
    return list(map(len, sequence))
