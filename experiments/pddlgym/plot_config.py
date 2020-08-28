from collections import namedtuple

import numpy as np

NAME = 'pddlgym'
DIR = 'pddlgym/{}'
PLOTS = ['learning_curves', 'planning_boxes']
SUMMARIES = ['macro_type']
FIELDS = 'alg', 'pddl_env', 'problem_id', 'macro_type', 'seed'
FIGSIZE = (8,6)
FONTSIZE = 18
HLINE = None
TRANSITION_CAP = 1e5
XLIM = [-100,TRANSITION_CAP]
YLIM = [0,16]

PlotVars = namedtuple('PlotVars', ['macro_type', 'problem_id', 'color', 'zorder'])
PLOT_VARS = [
    PlotVars('primitive', 10, 'C0', 10),
    PlotVars('learned', 10, 'C3', 10),
]


def get_goal(state, metadata):
    return state.goal

def heuristic(state, goal):
    return len([lit for lit in goal.literals if lit not in state.literals])

def get_primitive_steps(sequence):
    return len(sequence)

def get_macro_steps(sequence):
    return len(sequence)

def get_macro_lengths(sequence):
    return list(map(lambda x: 1, sequence))
