import copy
import random
import warnings

from domains.cube import cube
from domains.cube import Face

SWAP_2_EDGES_ADJ = "R U R' U R U U R' U".split()
R_PERMUTATION = "F F R' F' U' F' U F R F' U U F U U F' U'".split()
SWAP_3_EDGES_FACE = "R R U R U R' U' R' U' R' U R'".split()
SWAP_3_EDGES_MID = "L' R U U R' L F F".split()
SWAP_4_EDGES = "L L R R D D R' R' L' L' U U".split()
SWAP_5_EDGES = "F' R' F B' D B".split()
ORIENT_2_EDGES = "L R' F L R' D L R' B L R' U U L R' F L R' D L R' B L R'".split()
SWAP_3_CORNERS = "R U' R' D R U R' D'".split()
ORIENT_2_CORNERS = "R B' R' U' B' U F U' B U R B R' F'".split()

def _inverse_move(move):
    """Invert a move"""
    if '\'' in move:
        return move.strip('\'')
    return move+'\''

def _mirror_move(move, face=Face.R):
    table = {
        Face.R: {
            'L':   'R\'',
            'R':   'L\'',
            'U':   'U\'',
            'D':   'D\'',
            'F':   'F\'',
            'B':   'B\'',
            'L\'': 'R',
            'R\'': 'L',
            'U\'': 'U',
            'D\'': 'D',
            'F\'': 'F',
            'B\'': 'B',
        },
        Face.U: {
            'U':   'D\'',
            'D':   'U\'',
            'L':   'L\'',
            'R':   'R\'',
            'F':   'F\'',
            'B':   'B\'',
            'U\'': 'D',
            'D\'': 'U',
            'L\'': 'L',
            'R\'': 'R',
            'F\'': 'F',
            'B\'': 'B',
        },
        Face.F: {
            'F':   'B\'',
            'B':   'F\'',
            'L':   'L\'',
            'R':   'R\'',
            'U':   'U\'',
            'D':   'D\'',
            'F\'': 'B',
            'B\'': 'F',
            'L\'': 'L',
            'R\'': 'R',
            'U\'': 'U',
            'D\'': 'D',
        },
    }
    table[Face.L] = table[Face.R]
    table[Face.D] = table[Face.U]
    table[Face.B] = table[Face.F]
    return table[face][move]

def _rotate_move(move, axis, n=1):
    """Rotate a move clockwise about an axis

    The axis of rotation should correspond to a primitive rotation operation of
    a cube Face.
    """
    if n == 0:
        return move
    table = {
        Face.U: {
            'U':   'U',
            'D':   'D',
            'U\'': 'U\'',
            'D\'': 'D\'',
            'F':   'L',
            'L':   'B',
            'B':   'R',
            'R':   'F',
            'F\'': 'L\'',
            'L\'': 'B\'',
            'B\'': 'R\'',
            'R\'': 'F\'',
        },
        Face.D: {
            'U':   'U',
            'D':   'D',
            'U\'': 'U\'',
            'D\'': 'D\'',
            'L':   'F',
            'F':   'R',
            'R':   'B',
            'B':   'L',
            'L\'': 'F\'',
            'F\'': 'R\'',
            'R\'': 'B\'',
            'B\'': 'L\'',
        },
        Face.R: {
            'R':     'R',
            'L':     'L',
            'R\'':   'R\'',
            'L\'':   'L\'',
            'U':     'B',
            'B':     'D',
            'D':     'F',
            'F':     'U',
            'U\'':   'B\'',
            'B\'':   'D\'',
            'D\'':   'F\'',
            'F\'':   'U\'',
        },
        Face.L: {
            'R':     'R',
            'L':     'L',
            'R\'':   'R\'',
            'L\'':   'L\'',
            'U':     'F',
            'F':     'D',
            'D':     'B',
            'B':     'U',
            'U\'':   'F\'',
            'F\'':   'D\'',
            'D\'':   'B\'',
            'B\'':   'U\'',
        },
        Face.F: {
            'F':     'F',
            'B':     'B',
            'F\'':   'F\'',
            'B\'':   'B\'',
            'R':     'D',
            'D':     'L',
            'L':     'U',
            'U':     'R',
            'R\'':   'D\'',
            'D\'':   'L\'',
            'L\'':   'U\'',
            'U\'':   'R\'',
        },
        Face.B: {
            'F':     'F',
            'B':     'B',
            'F\'':   'F\'',
            'B\'':   'B\'',
            'R':     'U',
            'U':     'L',
            'L':     'D',
            'D':     'R',
            'R\'':   'U\'',
            'U\'':   'L\'',
            'L\'':   'D\'',
            'D\'':   'R\'',
        }
    }
    for _ in range(n):
        move = table[axis][move]
    return move

def inverse(formula):
    """Invert a formula"""
    result = copy.copy(formula)
    result.reverse()
    for i, move in enumerate(result):
        result[i] = _inverse_move(move)
    return result


def mirror(formula, face=Face.R):
    """Flip a formula left/right to use opposite face(s)."""
    result = copy.copy(formula)
    for i, move in enumerate(result):
        result[i] = _mirror_move(move, face)
    return result

def rotate(formula, axis, n=1):
    """Rotate a formula clockwise around the specified cube face."""
    result = copy.copy(formula)
    for i, move in enumerate(result):
        result[i] = _rotate_move(move, axis, n)
    return result

def simplify(formula):
    """Simplify a formula

    Repeatedly remove noops [F F']; replace triple rotations [F F F] with their
    corresponding inverse single rotation [F']; and find non-interacting
    sandwiches [F B F'] and rearrange them [F F' B] for future cancellation.

    """
    # To treat each action as a single character, replace inverses with lowercase
    string_form = ''.join(formula).strip()
    for move in 'FBLRUD':
        string_form = string_form.replace(move+'\'', move.lower())

    noops = ["Ff", "Bb", "Ll", "Rr", "Uu", "Dd"]
    noops += [op[::-1] for op in noops]
    triples = [op*3 for op in 'FBLRUDfblrud']
    singles = 'fblrudFBLRUD'
    # Sandwiches
    outers = ["Ff", "Bb", "Ll", "Rr", "Uu", "Dd", 'fF', 'bB', 'lL', 'rR', 'uU', 'dD']
    inners = ['B', 'F', 'R', 'L', 'D', 'U']*2
    while True:
        prev_string = string_form
        # Noops: [F F'] -> []
        for noop in noops:
            string_form = string_form.replace(noop, '')
        # Triples: [F F F] -> [F']
        for triple, single in zip(triples, singles):
            string_form = string_form.replace(triple, single)
        # Sandwiches: [F B F'] -> [B]; [L R R L'] -> [R R]
        for outer, inner in zip(outers, inners):
            sandwich1 = outer[0]+inner+outer[1]
            string_form = string_form.replace(sandwich1, inner)
            sandwich2 = outer[0]+inner.lower()+outer[1]
            string_form = string_form.replace(sandwich2, inner.lower())
            sandwich1 = outer[0]+inner*2+outer[1]
            string_form = string_form.replace(sandwich1, inner*2)
            sandwich2 = outer[0]+inner.lower()*2+outer[1]
            string_form = string_form.replace(sandwich2, inner.lower()*2)
        if string_form == prev_string:
            # No change; already simplified
            break

    # Convert back to normal form
    simplified = [move if move in 'FBLRUD' else (move.upper()+'\'') for move in string_form]
    return simplified

def variations(formula):
    """Generate all variations of formula, accounting for rotations, mirrors, and inverses"""
    # Consider each possible orientation of the cube
    formulas = []
    orient_f = formula
    orient_l = rotate(formula, Face.U, 1)
    orient_b = rotate(formula, Face.U, 2)
    orient_r = rotate(formula, Face.D, 1)
    orient_u = rotate(formula, Face.L, 1)
    orient_d = rotate(formula, Face.R, 1)
    formulas = [orient_f, orient_l, orient_b, orient_r, orient_u, orient_d]
    # Rotate the corresponding Face 0 to 3 times
    faces = [Face.F, Face.L, Face.B, Face.R, Face.D, Face.U]
    for n in range(4):
        for formula_, face in zip(formulas, faces):
            formula_ = rotate(formula_, face, n)
            formulas.append(formula_)

    formulas += [mirror(x) for x in formulas]# Add mirrored algs
    formulas += [inverse(x) for x in formulas]# Add inverse algs

    # Remove duplicate formulas
    formulas = [' '.join(x) for x in formulas]
    formulas = sorted(list(set(formulas)))
    formulas = [x.split() for x in formulas]
    return formulas

def random_formula(length=3):
    """Generate a random formula of a given length"""
    formula_ = [random.choice(list(cube.Action.keys())) for _ in range(length)]
    # formula_ = simplify(formula_)
    return formula_

def random_conjugate(prefix_length=1, body_length=1):
    """Generates a random formula of the form (X Y X')"""
    if prefix_length <= 0 or body_length <= 0:
        raise ValueError("Lengths ({}, {}) must be positive".format(prefix_length, body_length))
    prefix = random_formula(prefix_length)
    suffix = inverse(prefix)
    body = random_formula(body_length)
    formula_ = prefix + body + suffix
    formula_ = simplify(formula_)
    return formula_

def random_commutator(x_length=3, y_length=1):
    """Generates a random formula of the form (X Y X' Y')"""
    if x_length <= 0 and y_length <= 0:
        raise ValueError("Lengths ({}, {}) must be positive".format(x_length, y_length))
    x_part = random_formula(x_length)
    x_inv = inverse(x_part)
    y_part = random_formula(y_length)
    y_inv = inverse(y_part)
    formula_ = x_part + y_part + x_inv + y_inv
    formula_ = simplify(formula_)
    return formula_


def test_mirroring():
    """Test mirroring on some example formulas"""
    # A typical formula, whose basic and mirror versions have different effects
    formula_ = SWAP_3_CORNERS
    mirrored = mirror(formula_)
    assert len(mirrored) == len(formula_)
    assert mirror(mirrored) == formula_

    # A formula whose mirror version has the same net effect on the cube
    formula_ = ORIENT_2_CORNERS
    mirrored = mirror(formula_)
    assert len(mirrored) == len(formula_)
    assert mirror(mirrored) == formula_
    assert cube.Cube().apply(sequence=formula_) == cube.Cube().apply(sequence=mirrored)

def test_simplify():
    """Test simplify on some example formulas"""
    # Nested noops
    formula_ = "L R F F' R' U' D'".split()
    simplified = simplify(formula_)
    assert simplified == "L U' D'".split()

    # Trailing noops
    formula_ = "L D D'".split()
    simplified = simplify(formula_)
    assert simplified == "L".split()

    # A formula and its inverse should cancel completely
    formula_ = "L R F B U D L' R' F' B' U' D'".split()
    simplified = simplify(formula_ + inverse(formula_))
    alternate = simplify(inverse(formula_) + formula_)
    assert simplified == [] and alternate == []

    # An irreducible formula
    formula_ = "D' D'".split()
    simplfied = simplify(formula_)
    assert simplfied == formula_

    # A formula with a quadruple rotation
    formula_ = "L B' F' D D D D F B L".split()
    simplified = simplify(formula_)
    assert simplified == ['L', 'L']

    # A formula with non-interacting sandwiches of nested noops
    formula_ = "U F L' B U' D L' U' U' D' U D U L D' U B' L F' U'".split()
    simplified = simplify(formula_)
    assert simplified == []

    # Another sandwich
    formula_ = "L' R' L R".split()
    simplified = simplify(formula_)
    assert simplified == []

    # A double sandwich
    formula_ = "L' R' R' L R R".split()
    simplified = simplify(formula_)
    assert simplified == []

    # A sandwich of a quadruple and a noop
    formula_ = "D D U D D U'".split()
    simplified = simplify(formula_)
    assert simplified == []

    # TODO: sandwiches of non-interacting double rotations
    formula_ = "D D U U D D U U".split()
    simplified = simplify(formula_)
    if simplified != []:
        warnings.warn('Simplify failed to reduce {} to []'.format(formula_))

def test_rotate():
    """Test rotate on some example formulas"""

    # Rotating then un-rotating by using the opposite cube Face
    formula_ = ORIENT_2_CORNERS
    rotated = rotate(formula_, cube.Face.U)
    unrotated = rotate(rotated, cube.Face.D)
    assert unrotated == formula_

    # Rotating twice has the same effect regardless of direction
    cw_rotated = rotate(formula_, cube.Face.F, 2)
    ccw_rotated = rotate(formula_, cube.Face.B, 2)
    assert cw_rotated == ccw_rotated

    # Rotating clockwise three times is the same as rotating counter-clockwise once
    cw_rotated = ' '.join(rotate(formula_, cube.Face.L, 3))
    ccw_rotated = ' '.join(rotate(formula_, cube.Face.R, 1))
    assert ccw_rotated == cw_rotated

    # Rotating zero times is a noop
    formula_ = "R F B U L D".split()
    unrotated = rotate(formula_, cube.Face.U, n=0)
    assert formula_ == unrotated


def test_variations():
    """Test variations on some example formulas"""
    # The primitive actions are the variations of a single primitive action
    formula_ = ["R"]
    variations_ = [x[0] for x in variations(formula_)]
    assert sorted(variations_) == sorted(cube.actions)

    # A typical formula that has all 96 variations
    formula_ = ORIENT_2_CORNERS
    variations_ = variations(formula_)
    assert len(variations_) == 96

def test():
    """Run all tests"""
    test_mirroring()
    test_simplify()
    test_rotate()
    test_variations()
    print('All tests passed.')

if __name__ == '__main__':
    test()
