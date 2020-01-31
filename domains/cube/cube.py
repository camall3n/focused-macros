import copy
import enum
import random

class ANSIColor:
    """Namespace for ANSI color codes"""
ANSIColor.RED = '\x1b[41m'
ANSIColor.ORANGE = '\x1b[48;5;130m'
ANSIColor.GREEN = '\x1b[42m'
ANSIColor.YELLOW = '\x1b[43m'
ANSIColor.BLUE = '\x1b[44m'
ANSIColor.MAGENTA = '\x1b[45m'
ANSIColor.CYAN = '\x1b[46m'
ANSIColor.WHITE = '\x1b[47m'
ANSIColor.RESET = '\x1b[49m'

class CubeColor:
    """Namespace for Cube color codes"""
CubeColor.W = 'W'
CubeColor.Y = 'Y'
CubeColor.G = 'G'
CubeColor.B = 'B'
CubeColor.R = 'R'
CubeColor.O = 'O'

class Pos(enum.IntEnum):
    """Enum for representing the position of a sticker on a Face

    Positions are named for their corresponding compass directions, plus M for middle
    """
    NW = 0
    N = 1
    NE = 2
    W = 3
    M = 4
    E = 5
    SW = 6
    S = 7
    SE = 8

class Face(enum.IntEnum):
    """Enum for representing the faces of a Cube"""
    F = 0 # front
    B = 1 # back
    L = 2 # left
    R = 3 # right
    U = 4 # up
    D = 5 # down

# Mapping from action names to (face, is_inverted) tuples
ACTION_MAP = {
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
ACTIONS = sorted(list(ACTION_MAP.keys()))

# Mapping from Faces to CubeColors
INITIAL_COLORS = {
    Face.F: CubeColor.R,
    Face.B: CubeColor.O,
    Face.L: CubeColor.G,
    Face.R: CubeColor.B,
    Face.U: CubeColor.W,
    Face.D: CubeColor.Y,
}

def get_face_swaps(face):
    """Return the list of position swaps on a single face associated with rotating that face

    Each swap is a tuple of tuples: ((face, StartPos), (face, EndPos))

    The rotation for a given face is clockwise w.r.t. the 2-D plane in the render diagram
    """
    swaps = [
        ((face, Pos.NW), (face, Pos.NE)),
        ((face, Pos.N), (face, Pos.E)),
        ((face, Pos.NE), (face, Pos.SE)),
        ((face, Pos.E), (face, Pos.S)),
        ((face, Pos.SE), (face, Pos.SW)),
        ((face, Pos.S), (face, Pos.W)),
        ((face, Pos.SW), (face, Pos.NW)),
        ((face, Pos.W), (face, Pos.N)),
    ]
    return swaps

def get_position_swaps(action):
    """Return the list of position swaps associated with a given action

    Each swap is a tuple of tuples: ((StartFace, StartPos), (EndFace, EndPos))
    """
    swaps = {
        Face.L: get_face_swaps(Face.L) + [
            # plus adjacent edge swaps...
            ((Face.U, Pos.SW), (Face.B, Pos.NW)),
            ((Face.U, Pos.W), (Face.B, Pos.W)),
            ((Face.U, Pos.NW), (Face.B, Pos.SW)),
            ((Face.B, Pos.NW), (Face.D, Pos.NW)),
            ((Face.B, Pos.W), (Face.D, Pos.W)),
            ((Face.B, Pos.SW), (Face.D, Pos.SW)),
            ((Face.D, Pos.NW), (Face.F, Pos.SW)),
            ((Face.D, Pos.W), (Face.F, Pos.W)),
            ((Face.D, Pos.SW), (Face.F, Pos.NW)),
            ((Face.F, Pos.SW), (Face.U, Pos.SW)),
            ((Face.F, Pos.W), (Face.U, Pos.W)),
            ((Face.F, Pos.NW), (Face.U, Pos.NW)),
        ],
        Face.R: get_face_swaps(Face.R) + [
            # plus adjacent edge swaps...
            ((Face.U, Pos.SE), (Face.B, Pos.NE)),
            ((Face.U, Pos.E), (Face.B, Pos.E)),
            ((Face.U, Pos.NE), (Face.B, Pos.SE)),
            ((Face.B, Pos.NE), (Face.D, Pos.NE)),
            ((Face.B, Pos.E), (Face.D, Pos.E)),
            ((Face.B, Pos.SE), (Face.D, Pos.SE)),
            ((Face.D, Pos.NE), (Face.F, Pos.SE)),
            ((Face.D, Pos.E), (Face.F, Pos.E)),
            ((Face.D, Pos.SE), (Face.F, Pos.NE)),
            ((Face.F, Pos.SE), (Face.U, Pos.SE)),
            ((Face.F, Pos.E), (Face.U, Pos.E)),
            ((Face.F, Pos.NE), (Face.U, Pos.NE)),
        ],
        Face.U: get_face_swaps(Face.U) + [
            # plus adjacent edge swaps...
            ((Face.L, Pos.NW), (Face.B, Pos.NW)),
            ((Face.L, Pos.N), (Face.B, Pos.N)),
            ((Face.L, Pos.NE), (Face.B, Pos.NE)),
            ((Face.B, Pos.NW), (Face.R, Pos.NE)),
            ((Face.B, Pos.N), (Face.R, Pos.N)),
            ((Face.B, Pos.NE), (Face.R, Pos.NW)),
            ((Face.R, Pos.NE), (Face.F, Pos.NE)),
            ((Face.R, Pos.N), (Face.F, Pos.N)),
            ((Face.R, Pos.NW), (Face.F, Pos.NW)),
            ((Face.F, Pos.NE), (Face.L, Pos.NW)),
            ((Face.F, Pos.N), (Face.L, Pos.N)),
            ((Face.F, Pos.NW), (Face.L, Pos.NE)),
        ],
        Face.D: get_face_swaps(Face.D) + [
            # plus adjacent edge swaps...
            ((Face.L, Pos.SW), (Face.B, Pos.SW)),
            ((Face.L, Pos.S), (Face.B, Pos.S)),
            ((Face.L, Pos.SE), (Face.B, Pos.SE)),
            ((Face.B, Pos.SW), (Face.R, Pos.SE)),
            ((Face.B, Pos.S), (Face.R, Pos.S)),
            ((Face.B, Pos.SE), (Face.R, Pos.SW)),
            ((Face.R, Pos.SE), (Face.F, Pos.SE)),
            ((Face.R, Pos.S), (Face.F, Pos.S)),
            ((Face.R, Pos.SW), (Face.F, Pos.SW)),
            ((Face.F, Pos.SE), (Face.L, Pos.SW)),
            ((Face.F, Pos.S), (Face.L, Pos.S)),
            ((Face.F, Pos.SW), (Face.L, Pos.SE)),
        ],
        Face.F: get_face_swaps(Face.F) + [
            # plus adjacent edge swaps...
            ((Face.U, Pos.SW), (Face.R, Pos.NW)),
            ((Face.U, Pos.S), (Face.R, Pos.W)),
            ((Face.U, Pos.SE), (Face.R, Pos.SW)),
            ((Face.R, Pos.NW), (Face.D, Pos.SE)),
            ((Face.R, Pos.W), (Face.D, Pos.S)),
            ((Face.R, Pos.SW), (Face.D, Pos.SW)),
            ((Face.D, Pos.SE), (Face.L, Pos.SW)),
            ((Face.D, Pos.S), (Face.L, Pos.W)),
            ((Face.D, Pos.SW), (Face.L, Pos.NW)),
            ((Face.L, Pos.SW), (Face.U, Pos.SW)),
            ((Face.L, Pos.W), (Face.U, Pos.S)),
            ((Face.L, Pos.NW), (Face.U, Pos.SE)),
        ],
        Face.B: get_face_swaps(Face.B) + [
            # plus adjacent edge swaps...
            ((Face.U, Pos.NW), (Face.R, Pos.NE)),
            ((Face.U, Pos.N), (Face.R, Pos.E)),
            ((Face.U, Pos.NE), (Face.R, Pos.SE)),
            ((Face.R, Pos.NE), (Face.D, Pos.NE)),
            ((Face.R, Pos.E), (Face.D, Pos.N)),
            ((Face.R, Pos.SE), (Face.D, Pos.NW)),
            ((Face.D, Pos.NE), (Face.L, Pos.SE)),
            ((Face.D, Pos.N), (Face.L, Pos.E)),
            ((Face.D, Pos.NW), (Face.L, Pos.NE)),
            ((Face.L, Pos.SE), (Face.U, Pos.NW)),
            ((Face.L, Pos.E), (Face.U, Pos.N)),
            ((Face.L, Pos.NE), (Face.U, Pos.NE)),
        ]
    }
    return swaps[action]

def get_inverse_swaps(swap_list):
    """Invert a given list of position swaps

    ((StartFace, StartPos), (EndFace, EndPos)) -> ((EndFace, EndPos), (StartFace, StartPos))
    """
    start, end = zip(*swap_list)
    result = list(zip(end, start))
    return result

class Cube:
    """Rubik's cube puzzle"""
    def __init__(self):
        self.reset()

    def reset(self):
        """Reset the Cube to the canonical 'solved' state"""
        self.sequence = []
        self.faces = [
            [INITIAL_COLORS[Face.F] for _ in range(9)],
            [INITIAL_COLORS[Face.B] for _ in range(9)],
            [INITIAL_COLORS[Face.L] for _ in range(9)],
            [INITIAL_COLORS[Face.R] for _ in range(9)],
            [INITIAL_COLORS[Face.U] for _ in range(9)],
            [INITIAL_COLORS[Face.D] for _ in range(9)],
        ]
        self.indices = [
            [(Face.F, Pos(i)) for i in range(9)],
            [(Face.B, Pos(i)) for i in range(9)],
            [(Face.L, Pos(i)) for i in range(9)],
            [(Face.R, Pos(i)) for i in range(9)],
            [(Face.U, Pos(i)) for i in range(9)],
            [(Face.D, Pos(i)) for i in range(9)],
        ]

    def transform(self, action):
        """Transform the Cube with a single quarter-turn action"""
        face, is_inverted = ACTION_MAP[action]
        swaps = get_position_swaps(face)

        # Since swaps are implemented w.r.t the 2-D render diagram, half of them need to be flipped
        need_flip = (face in [Face.B, Face.L, Face.D])
        if (need_flip and not is_inverted) or (is_inverted and not need_flip):
            swaps = get_inverse_swaps(swaps)
        self.apply(swap_list=swaps)
        return self

    def apply(self, sequence=None, swap_list=None):
        """Apply a sequence of actions or a swap_list to transform the Cube"""
        assert sequence is not None or swap_list is not None
        if swap_list is not None:
            cube_copy = copy.deepcopy(self)
            for ((start_face, start_pos), (end_face, end_pos)) in swap_list:
                self.faces[end_face][end_pos] = cube_copy.faces[start_face][start_pos]
                self.indices[end_face][end_pos] = cube_copy.indices[start_face][start_pos]
        elif sequence is not None:
            for action in sequence:
                self.transform(action)
        if sequence:
            self.sequence += sequence
        return self

    def scramble(self, n=60):
        """Scramble the Cube with randomly selected actions

        For repeatable results, see domains.cube.pattern.scramble()
        """
        formula_ = [random.choice(ACTIONS) for _ in range(n)]
        self.apply(formula_)
        return self

    def __hash__(self):
        return hash(repr((self.faces, self.indices)))

    def __eq__(self, another):
        return self.faces == another.faces

    def __ne__(self, another):
        return not self.__eq__(another)

    def render(self, use_color=True):
        """Print an ASCII-art diagram of the Cube

        The default behavior uses ANSI color codes to produce a colored diagram
        """

        # The 'outline' variable below is just a giant format string.
        #
        # Named format placeholders in curly braces (e.g. {U_NW}) correspond to particular
        # (Face, Pos) pairs.
        #
        # The remaining positions are too small to have named format placeholders in the
        # diagram, so they are denoted with '@' and '?' characters. Both characters are
        # replaced with unnamed format placeholders '{}' before formatting. The '@' just
        # signifies the NW position on L and R faces, for convenience, and is otherwise
        # treated the same as '?'.
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
            +------+------+------+""".replace('@', '{}').replace('?', '{}')


        # We fill the unnamed placeholders first, in the order they appear in the
        # format string, followed by the named placeholders.
        #
        # The unnamed placeholders are each replaced with the appropriate CubeColor
        # code for the sticker at that position. The named placeholders are wider,
        # so we fill them with 6 repeated copies of their CubeColor code.
        diagram = outline.format(
            self.faces[Face.L][Pos.NE], self.faces[Face.R][Pos.NE],
            self.faces[Face.L][Pos.NE], self.faces[Face.R][Pos.NE],
            self.faces[Face.L][Pos.N], self.faces[Face.R][Pos.N],
            self.faces[Face.L][Pos.N], self.faces[Face.L][Pos.E],
            self.faces[Face.R][Pos.N], self.faces[Face.R][Pos.E],
            self.faces[Face.L][Pos.NW], self.faces[Face.L][Pos.E],
            self.faces[Face.R][Pos.NW], self.faces[Face.R][Pos.E],
            self.faces[Face.L][Pos.NW], self.faces[Face.L][Pos.M],
            self.faces[Face.R][Pos.NW], self.faces[Face.R][Pos.M],
            self.faces[Face.L][Pos.M], self.faces[Face.L][Pos.SE],
            self.faces[Face.R][Pos.M], self.faces[Face.R][Pos.SE],
            self.faces[Face.L][Pos.W], self.faces[Face.L][Pos.SE],
            self.faces[Face.R][Pos.W], self.faces[Face.R][Pos.SE],
            self.faces[Face.L][Pos.W], self.faces[Face.L][Pos.S],
            self.faces[Face.R][Pos.W], self.faces[Face.R][Pos.S],
            self.faces[Face.L][Pos.S], self.faces[Face.R][Pos.S],
            self.faces[Face.L][Pos.SW], self.faces[Face.R][Pos.SW],
            self.faces[Face.L][Pos.SW], self.faces[Face.R][Pos.SW],
            B_NW=self.faces[Face.B][Pos.NW]*6, B__N=self.faces[Face.B][Pos.N]*6,
            B_NE=self.faces[Face.B][Pos.NE]*6, B__W=self.faces[Face.B][Pos.W]*6,
            B__M=self.faces[Face.B][Pos.M]*6, B__E=self.faces[Face.B][Pos.E]*6,
            B_SW=self.faces[Face.B][Pos.SW]*6, B__S=self.faces[Face.B][Pos.S]*6,
            B_SE=self.faces[Face.B][Pos.SE]*6, U_NW=self.faces[Face.U][Pos.NW]*6,
            U__N=self.faces[Face.U][Pos.N]*6, U_NE=self.faces[Face.U][Pos.NE]*6,
            U__W=self.faces[Face.U][Pos.W]*6, U__M=self.faces[Face.U][Pos.M]*6,
            U__E=self.faces[Face.U][Pos.E]*6, U_SW=self.faces[Face.U][Pos.SW]*6,
            U__S=self.faces[Face.U][Pos.S]*6, U_SE=self.faces[Face.U][Pos.SE]*6,
            F_NW=self.faces[Face.F][Pos.NW]*6, F__N=self.faces[Face.F][Pos.N]*6,
            F_NE=self.faces[Face.F][Pos.NE]*6, F__W=self.faces[Face.F][Pos.W]*6,
            F__M=self.faces[Face.F][Pos.M]*6, F__E=self.faces[Face.F][Pos.E]*6,
            F_SW=self.faces[Face.F][Pos.SW]*6, F__S=self.faces[Face.F][Pos.S]*6,
            F_SE=self.faces[Face.F][Pos.SE]*6, D_NW=self.faces[Face.D][Pos.NW]*6,
            D__N=self.faces[Face.D][Pos.N]*6, D_NE=self.faces[Face.D][Pos.NE]*6,
            D__W=self.faces[Face.D][Pos.W]*6, D__M=self.faces[Face.D][Pos.M]*6,
            D__E=self.faces[Face.D][Pos.E]*6, D_SW=self.faces[Face.D][Pos.SW]*6,
            D__S=self.faces[Face.D][Pos.S]*6, D_SE=self.faces[Face.D][Pos.SE]*6,
        )

        # If we are using color, replace the CubeColor codes with their
        # corresponding ANSIColor codes.
        if use_color:
            fill = {
                'W': ANSIColor.WHITE+' '+ANSIColor.RESET,
                'Y': ANSIColor.YELLOW+' '+ANSIColor.RESET,
                'G': ANSIColor.GREEN+' '+ANSIColor.RESET,
                'B': ANSIColor.CYAN+' '+ANSIColor.RESET,
                'R': ANSIColor.RED+' '+ANSIColor.RESET,
                'O': ANSIColor.ORANGE+' '+ANSIColor.RESET,
            }
            for letter in 'WYGBRO':
                diagram = diagram.replace(letter, fill[letter])

        print(diagram)

    def summarize_effects(self, baseline=None):
        """Summarize the position changes in the Cube relative to a baseline Cube

        The default behavior compares the current Cube against a solved Cube.

        Returns:
            A list of position swaps ((StartFace, StartPos), (EndFace, EndPos))
        """
        if not baseline:
            baseline = Cube()
        src_indices = [idx for face in baseline.indices for idx in face]
        dst_indices = [idx for face in self.indices for idx in face]
        swap_list = list(zip(dst_indices, src_indices))
        swap_list = tuple([swap for swap in swap_list if swap[0] != swap[1]])
        return swap_list
