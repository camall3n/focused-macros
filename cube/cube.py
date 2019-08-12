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
actions = list(Action.keys())

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

def inverse_swaps(swap_list):
    start, end = zip(*swap_list)
    result = list(zip(end, start))
    return result

def combine_swaps(*swaps):
    c = Cube()
    for s in swaps:
        c.apply(swap_list=s)
    return c.summarize_effects()

class Cube:
    def __init__(self):
        self.sequence = []
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
        return self

    def apply(self, sequence=None, swap_list=None):
        assert sequence is not None or swap_list is not None
        if swap_list:
            cube_copy = copy.deepcopy(self)
            for ((start_face, start_pos), (end_face, end_pos)) in swap_list:
                self.faces[end_face][end_pos] = cube_copy.faces[start_face][start_pos]
                self.indices[end_face][end_pos] = cube_copy.indices[start_face][start_pos]
        else:
            for move in sequence:
                self.transform(move)
        if sequence:
            self.sequence += sequence
        return self

    def scramble(self, n=30):
        f = [random.choice(list(Action.keys())) for _ in range(n)]
        self.apply(f)
        return self

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
