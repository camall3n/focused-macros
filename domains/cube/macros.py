import pickle
import random as pyrandom
import warnings

from domains import cube
from domains.cube import formula

class primitive:  # pylint: disable=R0903,C0103
    """Namespace for primitive actions and their corresponding models"""
    alg_formulas = [[a] for a in cube.ACTIONS]
    actions = alg_formulas
    models = [cube.Cube().apply(sequence=a).summarize_effects() for a in actions]

class expert:  # pylint: disable=R0903,C0103
    """Namespace for expert macro-actions and their corresponding models"""
    alg_formulas = [
        formula.R_PERMUTATION,
        formula.SWAP_3_EDGES_FACE,
        formula.SWAP_3_EDGES_MID,
        formula.SWAP_3_CORNERS,
        formula.ORIENT_2_EDGES,
        formula.ORIENT_2_CORNERS,
    ]
    macros = [variation for f in alg_formulas for variation in formula.variations(f)]
    models = [cube.Cube().apply(sequence=macro).summarize_effects() for macro in macros]

class learned:  # pylint: disable=R0903,C0103
    """Namespace for learned macro-actions and their corresponding models"""

def load_learned_macros(version):
    """Load the set of learned macro-actions for a given version"""
    results_dir = 'results/macros/cube/'
    filename = results_dir+'v'+version+'-clean_macros.pickle'
    try:
        with open(filename, 'rb') as file:
            _macros = pickle.load(file)
    except FileNotFoundError:
        warnings.warn('Failed to load learned macros from file {}'.format(filename))
        _macros = []

    _models = [cube.Cube().apply(sequence=macro).summarize_effects() for macro in _macros]

    global learned  # pylint: disable=W0601,C0103
    learned.macros = _macros
    learned.models = _models

load_learned_macros('0.4')

class random:  # pylint: disable=R0903,C0103
    """Namespace for randomly generated macro-actions and their corresponding models"""

def generate_random_macro_set(seed):
    """Generate a new set of random macro-actions using the given random seed"""
    old_state = pyrandom.getstate()
    pyrandom.seed(seed)
    random_formulas = [formula.random_formula(len(alg)) for alg in expert.alg_formulas]
    pyrandom.setstate(old_state)

    _macros = [variation
               for formula_ in random_formulas
               for variation in formula.variations(formula.simplify(formula_))]
    _models = [cube.Cube().apply(sequence=macro).summarize_effects() for macro in _macros]

    global random  # pylint: disable=W0601,C0103
    random.seed = seed
    random.alg_formulas = random_formulas
    random.macros = _macros
    random.models = _models

generate_random_macro_set(0)

def test():
    """Test generating macros"""
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
