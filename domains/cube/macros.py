import pickle
import random as pyrandom
from domains import cube
from domains.cube import formula

class primitive:
    alg_formulas = [[a] for a in cube.actions]
    actions = alg_formulas
    models = [cube.Cube().apply(a).summarize_effects() for a in actions]

class expert:
    alg_formulas = [
        formula.r_permutation,
        formula.swap_3_edges_face,
        formula.swap_3_edges_mid,
        formula.swap_3_corners,
        formula.orient_2_edges,
        formula.orient_2_corners,
    ]
    macros = [variation for f in alg_formulas for variation in formula.variations(f)]
    models = [cube.Cube().apply(macro).summarize_effects() for macro in macros]

class learned:
    pass

def load_learned_macros(version):
    results_dir = 'results/macros/cube/'
    filename = results_dir+'v'+version+'-clean_skills.pickle'
    try:
        with open(filename, 'rb') as f:
            _macros = pickle.load(f)
    except FileNotFoundError:
        warning('Failed to load learned macros from file {}'.format(filename))
        _macros = []

    _models = [cube.Cube().apply(macro).summarize_effects() for macro in _macros]

    global learned
    learned.macros = _macros
    learned.models = _models

load_learned_macros('0.4')

class random:
    pass

def generate_random_macro_set(seed):
    st = pyrandom.getstate()
    pyrandom.seed(seed)
    formulas = [formula.random_formula(len(a)) for a in expert.alg_formulas]
    pyrandom.setstate(st)

    _macros = [variation for f in formulas for variation in formula.variations(formula.simplify(f))]
    _models = [cube.Cube().apply(macro).summarize_effects() for macro in _macros]
    global random
    random.seed = seed
    random.alg_formulas = formulas
    random.macros = _macros
    random.models = _models

generate_random_macro_set(0)

def test():
    generate_random_macro_set(0)
    macro_0 = random.macros[0]

    generate_random_macro_set(1)
    macro_1 = random.macros[0]
    assert macro_0 != macro_1

    generate_random_macro_set(0)
    assert random.macros[0] == macro_0

    generate_random_macro_set(1)
    assert random.macros[0] == macro_1

    print('All tests passed.')

if __name__ == '__main__':
    test()
