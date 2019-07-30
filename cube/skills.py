import random
import matplotlib.pyplot as plt
from tqdm import tqdm

from cube import cube

swap_3_edges = "L' R U U R' L F F".split()
swap_3_edges_alt = "L' R B B R' L U U".split()
swap_4_edges = "L L R R D D R' R' L' L' U U".split()
swap_5_edges = "F' R' F B' D B".split()
swap_5_edges_alt = "F' R F B' D' B".split()
swap_3_corners = "R U' R' D R U R' D'".split()
orient_2_corners = "R B' R' U' B' U F U' B U R B R' F'".split()

def random_skill(length=3):
    formula = [random.choice(list(cube.Action.keys())) for _ in range(length)]
    formula = cube.simplify_formula(formula)
    return formula

def random_conjugate(prefix_length=1, body_length=1):
    """Generates a random skill of the form (X Y X')"""
    assert prefix_length > 0 and body_length>0, "Lengths ({}, {}) must be positive".format(prefix_length, body_length)
    prefix = random_skill(prefix_length)
    suffix = cube.inverse_formula(prefix)
    body = random_skill(body_length)
    formula = prefix + body + suffix
    formula = cube.simplify_formula(formula)
    return formula

def random_commutator(x_length=3, y_length=1):
    """Generates a random skill of the form (X Y X' Y')"""
    assert x_length > 0 and y_length>0, "Lengths ({}, {}) must be positive".format(x_length, y_length)
    X = random_skill(x_length)
    Xinv = cube.inverse_formula(X)
    Y = random_skill(y_length)
    Yinv = cube.inverse_formula(Y)
    formula = X + Y + Xinv + Yinv
    formula = cube.simplify_formula(formula)
    return formula
