from .cube import Cube
import random

# action_map = {
#     'L': cube.Action.L,
#     'R': cube.Action.R,
#     'U': cube.Action.T,
#     'D': cube.Action.D,
#     'F': cube.Action.F,
#     'B': cube.Action.B,
#     'L\'': cube.Action.l,
#     'R\'': cube.Action.r,
#     'U\'': cube.Action.t,
#     'D\'': cube.Action.d,
#     'F\'': cube.Action.f,
#     'B\'': cube.Action.b,
# }

class CubeEnv:
    def __init__(self):
        self.cube = Cube()
        self.goal = Cube()
        self.formula = []
        self.action_meanings = {
            0:  'L',
            1:  'R',
            2:  'U',
            3:  'D',
            4:  'F',
            5:  'B',
            6:  'L\'',
            7:  'R\'',
            8:  'U\'',
            9:  'D\'',
            10: 'F\'',
            11: 'B\'',
        }
        self.n_actions = len(self.action_meanings)

    def random_action(self):
        return random.choice(self.n_actions)

    def scramble(self, n=30):
        self.formula = [self.action_meanings[random.randint(0,self.n_actions-1)] for _ in range(n)]
        for move in self.formula:
            self.cube.transform(move)

    def step(self, action):
        assert(action < self.n_actions)
        self.cube.transform(self.action_meanings[action])
        done = (self.cube == self.goal)
        r = 1000 if done else -1
        return self.cube, r, done

    def render(self):
        self.cube.render()
