import copy
import numpy as np
import pickle
import random as pyrandom
from domains import npuzzle

version='0.2'
results_dir = 'results/skillsearch/npuzzle/'
filename = results_dir+'v'+version+'-clean_skills.pickle'
with open(filename, 'rb') as f:
    _options = pickle.load(f)
_models = {}
for blank_idx, sequences in _options.items():
    puzzle = npuzzle.NPuzzle(n=15, start_blank=blank_idx)
    _models[blank_idx] = [copy.deepcopy(puzzle).apply_macro(o).summarize_effects(baseline=puzzle) for o in sequences]

class generated:
    options = _options
    models = _models

def random_skill(start_blank, length):
    baseline = npuzzle.NPuzzle(n=15, start_blank=start_blank)
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
    print('All tests passed.')
