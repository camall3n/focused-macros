from collections import namedtuple

NAME = 'suitcaselock'
DIR = NAME
PLOTS = ['entanglement_boxes']
SUMMARIES = []
FIELDS = 'alg', 'n_vars', 'n_values', 'entanglement', 'seed'
FIGSIZE = (8,6)
FONTSIZE = 22
TRANSITION_CAP = 1e8

PlotVars = namedtuple('PlotVars', ['n_vars', 'n_values'])
PLOT_VARS = [
    PlotVars(20, 2),
    PlotVars(10, 4),
]

def get_goal(state, metadata):
    return state.reset().scramble(seed=metadata.seed+1000)

def heuristic(state, goal):
    return sum(state.summarize_effects(baseline=goal)>0)

def get_primitive_steps(sequence):
    return len(sequence)

def get_macro_steps(sequence):
    return get_primitive_steps(sequence)

def get_macro_lengths(sequence):
    return [1]*len(sequence)
