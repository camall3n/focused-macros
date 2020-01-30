import copy
import numpy as np
import pickle
import random as pyrandom
from domains import npuzzle

version='0.2'
results_dir = 'results/skillsearch/npuzzle/'
filename = results_dir+'v'+version+'-clean_skills.pickle'
with open(filename, 'rb') as f:
    _macros = pickle.load(f)
_models = {}
for blank_idx, sequences in _macros.items():
    puzzle = npuzzle.NPuzzle(n=15, start_blank=blank_idx)
    _models[blank_idx] = [copy.deepcopy(puzzle).apply_macro(macro).summarize_effects(baseline=puzzle) for macro in sequences]

class learned:
    macros = _macros
    models = _models

def random_macro(start_blank, length):
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

def generate_random_macro_set(seed):
    py_st = pyrandom.getstate()
    np_st = np.random.get_state()
    pyrandom.seed(seed)
    np.random.seed(seed)

    _macros = {}
    _models = {}

    for blank_idx, macro_list in learned.macros.items():
        random_macros = [random_macro(blank_idx, len(macro)) for macro in macro_list]
        _macros[blank_idx], _models[blank_idx] = zip(*random_macros)

    pyrandom.setstate(py_st)
    np.random.set_state(np_st)

    global random
    random.macros = _macros
    random.models = _models

generate_random_macro_set(0)

def test():
    generate_random_macro_set(0)
    macro_0 = random.macros[(0,0)][0]

    generate_random_macro_set(1)
    macro_1 = random.macros[(0,0)][0]
    assert macro_0 != macro_1

    generate_random_macro_set(0)
    assert random.macros[(0,0)][0] == macro_0

    generate_random_macro_set(1)
    assert random.macros[(0,0)][0] == macro_1

    print('All tests passed.')

if __name__ == '__main__':
    test()
