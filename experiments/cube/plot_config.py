from collections import namedtuple

import numpy as np

from domains.cube import pattern

NAME = 'cube'
DIR = NAME
PLOTS = ['learning_curves', 'alt_learning_curves', 'alt_planning_boxes']
SUMMARIES = ['macro_type', 'goal_type']
FIELDS = 'alg', 'goal_type', 'macro_type', 'seed'
FIGSIZE = (8,6)
FONTSIZE = 20
HLINE = 48
TRANSITION_CAP = 2e6
XLIM = [0,TRANSITION_CAP]
YLIM = [0,50]

PlotVars = namedtuple('PlotVars', ['macro_type', 'goal_type', 'color', 'zorder', 'tick_size'])
PLOT_VARS = [
    PlotVars('random', 'default_goal', 'C2', 5, 0.5e6),
    PlotVars('primitive', 'default_goal', 'C0', 15, 0.5e6),
    PlotVars('focused', 'default_goal', 'C3', 10, 0.5e6),
    PlotVars('expert', 'default_goal', 'C1', 20, 0.5e6),
]
PLOT_VARS_ALT = [
    PlotVars('focused', 'default_goal', 'C3', 10, 0.5e6),
    PlotVars('focused', 'random_goal', 'C4', 20, 0.5e6),
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
