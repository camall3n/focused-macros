import copy
import pickle
import random as pyrandom
from npuzzle import npuzzle

version='0.2'
results_dir = 'results/skillsearch/npuzzle/'
filename = results_dir+'v'+version+'-clean_skills.pickle'
with open(filename, 'rb') as f:
    _options = pickle.load(f)
_puzzles = {}
for start_blank in _options.keys():
    puzzle = npuzzle.NPuzzle(n=15)
    while start_blank[0] < puzzle.blank_idx[0]:
        puzzle.transition(puzzle.up())
    while start_blank[1] < puzzle.blank_idx[1]:
        puzzle.transition(puzzle.left())
    assert puzzle.blank_idx == start_blank
    _puzzles[start_blank] = copy.deepcopy(puzzle)
_models = {}
for blank_idx, sequences in _options.items():
    _models[blank_idx] = [copy.deepcopy(_puzzles[blank_idx]).apply_macro(o).summarize_effects(baseline=_puzzles[blank_idx]) for o in sequences]

class generated:
    options = _options
    models = _models
