from domains.cube import Cube
import random
import torch

class CubeEnv:
    def __init__(self):
        self.cube = Cube()
        self.goal = self.state
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
    def sequence(self):
        return self.cube.sequence

    @property
    def state(self):
        square_colors = [square for face in self.cube.faces for i, square in enumerate(face) if i != 4]
        color_codes = dict((c, i) for (i, c) in enumerate('WYGBRO'))
        square_codes = [color_codes[color] for color in square_colors]
        return square_codes

    def reset(self, scramble=0, sequence=None):
        self.cube.reset()
        if sequence is None:
            self.cube.scramble(scramble)
        else:
            self.cube.apply(sequence)
        return self.state

    def random_action(self):
        return random.choice(self.n_actions)

    def step(self, action):
        assert(action < self.n_actions)
        if type(action) is torch.Tensor:
            action = action.item()
        self.cube.transform(self.action_meanings[action])
        done = (self.state == self.goal)
        r = 1000.0 if done else -1.0
        return self.state, r, done, None

    def render(self):
        self.cube.render()


def test():
    from domains.cube.cubeenv import CubeEnv

    c = CubeEnv()
    c.reset()
    c.render()

if __name__ == '__main__':
    test()
