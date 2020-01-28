import random

from . import cube
from . import formula

swap_2_edges_adj = "R U R' U R U U R' U".split()
r_permutation = "F F R' F' U' F' U F R F' U U F U U F' U'".split()
swap_3_edges_face = "R R U R U R' U' R' U' R' U R'".split()
swap_3_edges_mid = "L' R U U R' L F F".split()
swap_4_edges = "L L R R D D R' R' L' L' U U".split()
swap_5_edges = "F' R' F B' D B".split()
orient_2_edges = "L R' F L R' D L R' B L R' U U L R' F L R' D L R' B L R'".split()
swap_3_corners = "R U' R' D R U R' D'".split()
orient_2_corners = "R B' R' U' B' U F U' B U R B R' F'".split()

def random_skill(length=3):
    f = [random.choice(list(cube.actions)) for _ in range(length)]
    # f = formula.simplify(f)
    return f

# def random_conjugate(prefix_length=1, body_length=1):
#     """Generates a random skill of the form (X Y X')"""
#     assert prefix_length > 0 and body_length>0, "Lengths ({}, {}) must be positive".format(prefix_length, body_length)
#     prefix = random_skill(prefix_length)
#     suffix = formula.inverse(prefix)
#     body = random_skill(body_length)
#     f = prefix + body + suffix
#     f = formula.simplify(f)
#     return f
#
# def random_commutator(x_length=3, y_length=1):
#     """Generates a random skill of the form (X Y X' Y')"""
#     assert x_length > 0 and y_length>0, "Lengths ({}, {}) must be positive".format(x_length, y_length)
#     X = random_skill(x_length)
#     Xinv = formula.inverse(X)
#     Y = random_skill(y_length)
#     Yinv = formula.inverse(Y)
#     f = X + Y + Xinv + Yinv
#     f = formula.simplify(f)
#     return f
