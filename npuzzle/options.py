import copy
import numpy as np
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

def random_skill(start_blank, length):
    puzzle = npuzzle.NPuzzle(n=15)
    # start_blank = np.random.choice(4,2)
    start_row, start_col = start_blank
    while start_row < puzzle.blank_idx[0]:
        puzzle.transition(puzzle.up())
    while start_col < puzzle.blank_idx[1]:
        puzzle.transition(puzzle.left())
    assert puzzle.blank_idx == start_blank
    baseline = copy.deepcopy(puzzle)
    effect_size = 0
    while effect_size == 0:
        puzzle = copy.deepcopy(baseline)
        sequence = []
        for step in range(length):
            action = pyrandom.choice(puzzle.actions())
            puzzle.transition(action)
            sequence.append(action)
        model = puzzle.summarize_effects(baseline=baseline)
        effect_size = len(model[0])
    return sequence, model

class random:
    pass

def set_random_skill_seed(seed):
    py_st = pyrandom.getstate()
    np_st = np.random.get_state()
    pyrandom.seed(seed)
    np.random.seed(seed)

    _options = {}
    _models = {}

    for blank_idx, option_list in generated.options.items():
        skills = [random_skill(blank_idx, len(o)) for o in generated.options[blank_idx]]
        _options[blank_idx], _models[blank_idx] = zip(*skills)

    pyrandom.setstate(py_st)
    np.random.set_state(np_st)

    global random
    random.options = _options
    random.models = _models

set_random_skill_seed(0)

def test_random_seed():
    set_random_skill_seed(0)
    op0 = random.options[(0,0)][0]

    set_random_skill_seed(1)
    op1 = random.options[(0,0)][0]
    assert op0 != op1

    set_random_skill_seed(0)
    assert random.options[(0,0)][0] == op0

    set_random_skill_seed(1)
    assert random.options[(0,0)][0] == op1

if __name__ == '__main__':
    test_random_seed()
