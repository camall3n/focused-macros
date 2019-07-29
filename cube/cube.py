import copy
import enum
import random

from .util import fill

class color: pass
color.R = 'R'
color.Y = 'Y'
color.G = 'G'
color.W = 'W'
color.O = 'O'
color.B = 'B'

class Pos(enum.IntEnum):
    NW = 0
    N  = 1
    NE = 2
    W  = 3
    M  = 4
    E  = 5
    SW = 6
    S  = 7
    SE = 8

class Face(enum.IntEnum):
    F = 0 # front
    B = 1 # back
    L = 2 # left
    R = 3 # right
    U = 4 # up
    D = 5 # down

Action = {# (face, inverse)
    'L':   (Face.L, False),
    'R':   (Face.R, False),
    'U':   (Face.U, False),
    'D':   (Face.D, False),
    'F':   (Face.F, False),
    'B':   (Face.B, False),
    'L\'': (Face.L, True),
    'R\'': (Face.R, True),
    'U\'': (Face.U, True),
    'D\'': (Face.D, True),
    'F\'': (Face.F, True),
    'B\'': (Face.B, True),
}

def inverse_move(move):
    if '\'' in move:
        return move.strip('\'')
    else:
        return move+'\''

def inverse_formula(formula):
    result = copy.copy(formula)
    result.reverse()
    for i, move in enumerate(result):
        result[i] = inverse_move(move)
    return result

def inverse_swaps(swap_list):
    start, end = zip(*swap_list)
    result = list(zip(end, start))
    return result

def mirror_move(move):
    opposite = {
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
    }
    return opposite[move]

def mirror_formula(formula):
    """Flip a formula left/right to use opposite face(s)."""
    result = copy.copy(formula)
    for i, move in enumerate(result):
        result[i] = mirror_move(move)
    return result

def rotate_move(move, axis, n=1):
    if n==0:
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
    for i in range(n):
        move = table[axis][move]
    return move

def rotate_formula(formula, axis, n=1):
    """Rotate a formula clockwise around the specified cube face."""
    result = copy.copy(formula)
    for i, move in enumerate(result):
        result[i] = rotate_move(move, axis, n)
    return result

def formula_collection(formula):
    collection = []
    f0 = formula
    l0 = rotate_formula(formula, Face.U, 1)
    b0 = rotate_formula(formula, Face.U, 2)
    r0 = rotate_formula(formula, Face.D, 1)
    u0 = rotate_formula(formula, Face.L, 1)
    d0 = rotate_formula(formula, Face.R, 1)
    formulas = [f0, l0, b0, r0, u0, d0]
    faces = [Face.F, Face.L, Face.B, Face.R, Face.D, Face.U]
    for n in range(4):
        for f, face in zip(formulas, faces):
            f = rotate_formula(f, face, n)
            g = mirror_formula(f)
            collection.append(f)
            collection.append(g)

    collection = [' '.join(x) for x in collection]
    collection = list(set(collection))
    collection = [x.split() for x in collection]
    return collection

def simplify_formula(formula):
    s = ''.join(formula).strip()
    for move in 'FBLRUD':
        s = s.replace(move+'\'', move.lower())

    noops = ["Ff","Bb","Ll","Rr","Uu","Dd"]
    noops += [op[::-1] for op in noops]
    triples = [op*3 for op in 'FBLRUDfblrud']
    singles = [op for op in 'fblrudFBLRUD']
    outers = ["Ff","Bb","Ll","Rr","Uu","Dd",'fF', 'bB', 'lL', 'rR', 'uU', 'dD']
    inners = ['B','F','R','L','D','U']*2
    while True:
        s_prev = s
        # Noops: [F F'] -> []
        for noop in noops:
            s = s.replace(noop, '')
        # Triples: [F F F] -> [F']
        for op3, op1 in zip(triples, singles):
            s = s.replace(op3, op1)
        # Sandwiches: [F B F'] -> [B]; [L R R L'] -> [R R]
        for outer, inner in zip(outers, inners):
            sandwich1 = outer[0]+inner+outer[1]
            s = s.replace(sandwich1, inner)
            sandwich2 = outer[0]+inner.lower()+outer[1]
            s = s.replace(sandwich2, inner.lower())
            sandwich1 = outer[0]+inner*2+outer[1]
            s = s.replace(sandwich1, inner*2)
            sandwich2 = outer[0]+inner.lower()*2+outer[1]
            s = s.replace(sandwich2, inner.lower()*2)
        if s == s_prev:
            break
    simplified = [move if move in 'FBLRUD' else (move.upper()+'\'') for move in s]
    return simplified

initial_colors = {
    "U": color.W,
    "D": color.Y,
    "L": color.G,
    "R": color.B,
    "F": color.R,
    "B": color.O
}

NW, N, NE, W, M, E, SW, S, SE = range(9)

def face_rotation(face):
    # rotate face CW w.r.t. 2-D plane in diagram
    swaps = [
        ((face, Pos.NW), (face, Pos.NE)),
        ((face, Pos.N),  (face, Pos.E)),
        ((face, Pos.NE), (face, Pos.SE)),
        ((face, Pos.E),  (face, Pos.S)),
        ((face, Pos.SE), (face, Pos.SW)),
        ((face, Pos.S),  (face, Pos.W)),
        ((face, Pos.SW), (face, Pos.NW)),
        ((face, Pos.W),  (face, Pos.N)),
    ]
    return swaps

def swap_list(move):
    # Returns [((StartFace, StartPos), (EndFace, EndPos)), ...],
    swaps = {
        Face.L: face_rotation(Face.L) + [
            # adjacent edges
            ((Face.U, Pos.SW), (Face.B, Pos.NW)),
            ((Face.U, Pos.W),  (Face.B, Pos.W)),
            ((Face.U, Pos.NW), (Face.B, Pos.SW)),
            ((Face.B, Pos.NW), (Face.D, Pos.NW)),
            ((Face.B, Pos.W),  (Face.D, Pos.W)),
            ((Face.B, Pos.SW), (Face.D, Pos.SW)),
            ((Face.D, Pos.NW), (Face.F, Pos.SW)),
            ((Face.D, Pos.W),  (Face.F, Pos.W)),
            ((Face.D, Pos.SW), (Face.F, Pos.NW)),
            ((Face.F, Pos.SW), (Face.U, Pos.SW)),
            ((Face.F, Pos.W),  (Face.U, Pos.W)),
            ((Face.F, Pos.NW), (Face.U, Pos.NW)),
        ],
        Face.R: face_rotation(Face.R) + [
            ((Face.U, Pos.SE), (Face.B, Pos.NE)),
            ((Face.U, Pos.E),  (Face.B, Pos.E)),
            ((Face.U, Pos.NE), (Face.B, Pos.SE)),
            ((Face.B, Pos.NE), (Face.D, Pos.NE)),
            ((Face.B, Pos.E),  (Face.D, Pos.E)),
            ((Face.B, Pos.SE), (Face.D, Pos.SE)),
            ((Face.D, Pos.NE), (Face.F, Pos.SE)),
            ((Face.D, Pos.E),  (Face.F, Pos.E)),
            ((Face.D, Pos.SE), (Face.F, Pos.NE)),
            ((Face.F, Pos.SE), (Face.U, Pos.SE)),
            ((Face.F, Pos.E),  (Face.U, Pos.E)),
            ((Face.F, Pos.NE), (Face.U, Pos.NE)),
        ],
        Face.U: face_rotation(Face.U) + [
            ((Face.L, Pos.NW), (Face.B, Pos.NW)),
            ((Face.L, Pos.N),  (Face.B, Pos.N)),
            ((Face.L, Pos.NE), (Face.B, Pos.NE)),
            ((Face.B, Pos.NW), (Face.R, Pos.NE)),
            ((Face.B, Pos.N),  (Face.R, Pos.N)),
            ((Face.B, Pos.NE), (Face.R, Pos.NW)),
            ((Face.R, Pos.NE), (Face.F, Pos.NE)),
            ((Face.R, Pos.N),  (Face.F, Pos.N)),
            ((Face.R, Pos.NW), (Face.F, Pos.NW)),
            ((Face.F, Pos.NE), (Face.L, Pos.NW)),
            ((Face.F, Pos.N),  (Face.L, Pos.N)),
            ((Face.F, Pos.NW), (Face.L, Pos.NE)),
        ],
        Face.D: face_rotation(Face.D) + [
            ((Face.L, Pos.SW), (Face.B, Pos.SW)),
            ((Face.L, Pos.S),  (Face.B, Pos.S)),
            ((Face.L, Pos.SE), (Face.B, Pos.SE)),
            ((Face.B, Pos.SW), (Face.R, Pos.SE)),
            ((Face.B, Pos.S),  (Face.R, Pos.S)),
            ((Face.B, Pos.SE), (Face.R, Pos.SW)),
            ((Face.R, Pos.SE), (Face.F, Pos.SE)),
            ((Face.R, Pos.S),  (Face.F, Pos.S)),
            ((Face.R, Pos.SW), (Face.F, Pos.SW)),
            ((Face.F, Pos.SE), (Face.L, Pos.SW)),
            ((Face.F, Pos.S),  (Face.L, Pos.S)),
            ((Face.F, Pos.SW), (Face.L, Pos.SE)),
        ],
        Face.F: face_rotation(Face.F) + [
            ((Face.U, Pos.SW), (Face.R, Pos.NW)),
            ((Face.U, Pos.S),  (Face.R, Pos.W)),
            ((Face.U, Pos.SE), (Face.R, Pos.SW)),
            ((Face.R, Pos.NW), (Face.D, Pos.SE)),
            ((Face.R, Pos.W),  (Face.D, Pos.S)),
            ((Face.R, Pos.SW), (Face.D, Pos.SW)),
            ((Face.D, Pos.SE), (Face.L, Pos.SW)),
            ((Face.D, Pos.S),  (Face.L, Pos.W)),
            ((Face.D, Pos.SW), (Face.L, Pos.NW)),
            ((Face.L, Pos.SW), (Face.U, Pos.SW)),
            ((Face.L, Pos.W),  (Face.U, Pos.S)),
            ((Face.L, Pos.NW), (Face.U, Pos.SE)),
        ],
        Face.B: face_rotation(Face.B) + [
            ((Face.U, Pos.NW), (Face.R, Pos.NE)),
            ((Face.U, Pos.N),  (Face.R, Pos.E)),
            ((Face.U, Pos.NE), (Face.R, Pos.SE)),
            ((Face.R, Pos.NE), (Face.D, Pos.NE)),
            ((Face.R, Pos.E),  (Face.D, Pos.N)),
            ((Face.R, Pos.SE), (Face.D, Pos.NW)),
            ((Face.D, Pos.NE), (Face.L, Pos.SE)),
            ((Face.D, Pos.N),  (Face.L, Pos.E)),
            ((Face.D, Pos.NW), (Face.L, Pos.NE)),
            ((Face.L, Pos.SE), (Face.U, Pos.NW)),
            ((Face.L, Pos.E),  (Face.U, Pos.N)),
            ((Face.L, Pos.NE), (Face.U, Pos.NE)),
        ]
    }
    return swaps[move]


class Cube:
    def __init__(self):
        self.formula = []
        self.faces = {
            Face.F: [initial_colors['F'] for _ in range(9)],
            Face.B: [initial_colors['B'] for _ in range(9)],
            Face.L: [initial_colors['L'] for _ in range(9)],
            Face.R: [initial_colors['R'] for _ in range(9)],
            Face.U: [initial_colors['U'] for _ in range(9)],
            Face.D: [initial_colors['D'] for _ in range(9)],
        }
        self.indices = {
            Face.F: [(Face.F, Pos(i)) for i in range(9)],
            Face.B: [(Face.B, Pos(i)) for i in range(9)],
            Face.L: [(Face.L, Pos(i)) for i in range(9)],
            Face.R: [(Face.R, Pos(i)) for i in range(9)],
            Face.U: [(Face.U, Pos(i)) for i in range(9)],
            Face.D: [(Face.D, Pos(i)) for i in range(9)],
        }

    def transform(self, move):
        face, inverse = Action[move]
        swaps = swap_list(face)
        need_flip = (face in [Face.B, Face.L, Face.D])
        if (need_flip and not inverse) or (inverse and not need_flip):
            swaps = inverse_swaps(swaps)
        self.apply(swap_list=swaps)

    def apply(self, formula=None, swap_list=None):
        assert formula is not None or swap_list is not None
        if swap_list:
            cube_copy = copy.deepcopy(self)
            for ((start_face, start_pos), (end_face, end_pos)) in swap_list:
                self.faces[end_face][end_pos] = cube_copy.faces[start_face][start_pos]
                self.indices[end_face][end_pos] = cube_copy.indices[start_face][start_pos]
        else:
            for move in formula:
                self.transform(move)
        if formula:
            self.formula += formula
        return self

    def scramble(self, n=30):
        formula = [random.choice(list(Action.keys())) for _ in range(n)]
        self.apply(formula)

    def __eq__(self, another):
        return self.faces == another.faces

    def __ne__(self, another):
        return not self.__eq__(another)

    def render(self, color=True):
        outline = """
                                          +------+------+------+
                                          |{B_NW}|{B__N}|{B_NE}|
                                          |{B_NW}|{B__N}|{B_NE}|
                                          +------+------+------+
                                          |{B__W}|{B__M}|{B__E}|
                                          |{B__W}|{B__M}|{B__E}|
      +           +------+------+------+  +------+------+------+
     /|          /{U_NW}/{U__N}/{U_NE}/|  |{B_SW}|{B__S}|{B_SE}|
    +?|         +------+------+------+?|  |{B_SW}|{B__S}|{B_SE}|
   /|?+        /{U__W}/{U__M}/{U__E}/|?+  +------+------+------+
  +?|/|       +------+------+------+?|/|
 /|?+?|      /{U_SW}/{U__S}/{U_SE}/|?+?|
+@|/|?+     +------+------+------+@|/|?+
|@+?|/|     |{F_NW}|{F__N}|{F_NE}|@+?|/|
|/|?+?|     |{F_NW}|{F__N}|{F_NE}|/|?+?|
+?|/|?+     +------+------+------+?|/|?+
|?+?|/      |{F__W}|{F__M}|{F__E}|?+?|/
|/|?+       |{F__W}|{F__M}|{F__E}|/|?+
+?|/        +------+------+------+?|/
|?+         |{F_SW}|{F__S}|{F_SE}|?+
|/          |{F_SW}|{F__S}|{F_SE}|/
+           +------+------+------+

                  +------+------+------+
                 /{D_NW}/{D__N}/{D_NE}/
                +------+------+------+
               /{D__W}/{D__M}/{D__E}/
              +------+------+------+
             /{D_SW}/{D__S}/{D_SE}/
            +------+------+------+""".replace('@','{}').replace('?','{}')
        # '@' signifies NW position on L and R faces

        diagram = outline.format(
            self.faces[Face.L][Pos.NE], self.faces[Face.R][Pos.NE],
            self.faces[Face.L][Pos.NE], self.faces[Face.R][Pos.NE],
            self.faces[Face.L][Pos.N],  self.faces[Face.R][Pos.N],
            self.faces[Face.L][Pos.N],  self.faces[Face.L][Pos.E],  self.faces[Face.R][Pos.N],  self.faces[Face.R][Pos.E],
            self.faces[Face.L][Pos.NW], self.faces[Face.L][Pos.E],  self.faces[Face.R][Pos.NW], self.faces[Face.R][Pos.E],
            self.faces[Face.L][Pos.NW], self.faces[Face.L][Pos.M],  self.faces[Face.R][Pos.NW], self.faces[Face.R][Pos.M],
            self.faces[Face.L][Pos.M],  self.faces[Face.L][Pos.SE], self.faces[Face.R][Pos.M],  self.faces[Face.R][Pos.SE],
            self.faces[Face.L][Pos.W],  self.faces[Face.L][Pos.SE], self.faces[Face.R][Pos.W],  self.faces[Face.R][Pos.SE],
            self.faces[Face.L][Pos.W],  self.faces[Face.L][Pos.S],  self.faces[Face.R][Pos.W],  self.faces[Face.R][Pos.S],
            self.faces[Face.L][Pos.S],  self.faces[Face.R][Pos.S],
            self.faces[Face.L][Pos.SW], self.faces[Face.R][Pos.SW],
            self.faces[Face.L][Pos.SW], self.faces[Face.R][Pos.SW],
            B_NW=self.faces[Face.B][Pos.NW]*6, B__N=self.faces[Face.B][Pos.N]*6, B_NE=self.faces[Face.B][Pos.NE]*6,
            B__W=self.faces[Face.B][Pos.W]*6,  B__M=self.faces[Face.B][Pos.M]*6, B__E=self.faces[Face.B][Pos.E]*6,
            B_SW=self.faces[Face.B][Pos.SW]*6, B__S=self.faces[Face.B][Pos.S]*6, B_SE=self.faces[Face.B][Pos.SE]*6,
            U_NW=self.faces[Face.U][Pos.NW]*6, U__N=self.faces[Face.U][Pos.N]*6, U_NE=self.faces[Face.U][Pos.NE]*6,
            U__W=self.faces[Face.U][Pos.W]*6,  U__M=self.faces[Face.U][Pos.M]*6, U__E=self.faces[Face.U][Pos.E]*6,
            U_SW=self.faces[Face.U][Pos.SW]*6, U__S=self.faces[Face.U][Pos.S]*6, U_SE=self.faces[Face.U][Pos.SE]*6,
            F_NW=self.faces[Face.F][Pos.NW]*6, F__N=self.faces[Face.F][Pos.N]*6, F_NE=self.faces[Face.F][Pos.NE]*6,
            F__W=self.faces[Face.F][Pos.W]*6,  F__M=self.faces[Face.F][Pos.M]*6, F__E=self.faces[Face.F][Pos.E]*6,
            F_SW=self.faces[Face.F][Pos.SW]*6, F__S=self.faces[Face.F][Pos.S]*6, F_SE=self.faces[Face.F][Pos.SE]*6,
            D_NW=self.faces[Face.D][Pos.NW]*6, D__N=self.faces[Face.D][Pos.N]*6, D_NE=self.faces[Face.D][Pos.NE]*6,
            D__W=self.faces[Face.D][Pos.W]*6,  D__M=self.faces[Face.D][Pos.M]*6, D__E=self.faces[Face.D][Pos.E]*6,
            D_SW=self.faces[Face.D][Pos.SW]*6, D__S=self.faces[Face.D][Pos.S]*6, D_SE=self.faces[Face.D][Pos.SE]*6,
        )
        if color:
            for letter in 'WYGBRO':
                diagram = diagram.replace(letter, fill[letter])
        print(diagram)

    def summarize_effects(self, baseline=None):
        if not baseline:
            baseline = Cube()
        src_indices = [idx for _,face in baseline.indices.items() for idx in face]
        dst_indices = [idx for _,face in self.indices.items() for idx in face]
        swap_list = list(zip(dst_indices, src_indices))
        swap_list = [swap for swap in swap_list if swap[0] != swap[1]]
        return swap_list
