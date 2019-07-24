from .cube import Cube
import random

class CubeEnv:
    def __init__(self):
        self.cube = Cube()
        self.goal = Cube()
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

    @property
    def formula(self):
        return self.cube.formula

    def reset(self, n=30):
        self.cube.scramble(n)

    def random_action(self):
        return random.choice(self.n_actions)

    def step(self, action):
        assert(action < self.n_actions)
        self.cube.transform(self.action_meanings[action])
        done = (self.cube == self.goal)
        r = 1000 if done else -1
        return self.cube, r, done

    def render(self):
        self.cube.render()
