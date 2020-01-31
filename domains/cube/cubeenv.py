import random

from domains.cube import Cube

class CubeEnv:
    """An OpenAI gym-style wrapper for the Cube class

    Attributes
    ----------
    action_meanings : dict
        Mapping from actions (int) to action meanings ('str')
    action_lookup : dict
        Mapping from action meanings ('str') to actions (int)
    n_actions :
        The number of actions recognized by the environment (quarter-turn metric)

    """
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
        self.action_lookup = {meaning: id for id, meaning in self.action_meanings.items()}
        self.n_actions = len(self.action_meanings)

    @property
    def sequence(self):
        """:obj:`list` of int: List of actions that produced this Cube's initial pattern
        """
        return [self.action_lookup[a] for a in self.cube.sequence]

    @property
    def state(self):
        """:obj:`list` of :obj:`str`: List of color codes for all non-center squares
        """
        square_colors = [square
                         for face in self.cube.faces
                         for i, square in enumerate(face)
                         if i != 4]
        color_codes = dict((c, i) for (i, c) in enumerate('WYGBRO'))
        square_codes = [color_codes[color] for color in square_colors]
        return square_codes

    def reset(self, scramble_len=0, sequence=None):
        """Reset the cube

        Args:
            scramble_len (int, optional): Number of steps to scramble the cube
            sequence (:obj:`list` of int): List of actions to reproduce an initial pattern

        Returns:
            The resulting cube state

        """
        self.cube.reset()
        if sequence is None:
            self.cube.scramble(scramble_len)
        else:
            self.cube.apply(sequence=[self.action_meanings[a] for a in sequence])
        return self.state

    def random_action(self):
        """Generate a random action

        Returns:
            (int): The generated action

        """
        return random.choice(self.n_actions)

    def step(self, action):
        """Apply an action to the cube

        Args:
            action (int): The action to apply

        Returns:
            (state, reward, done)

        """
        assert action < self.n_actions
        self.cube.transform(self.action_meanings[action])
        done = (self.state == self.goal)
        reward = 1000.0 if done else -1.0
        return self.state, reward, done

    # Note: This function assumes your terminal supports the extended ANSI color codes
    def render(self):
        """Render the cube as colored ASCII art"""
        self.cube.render()


def test():
    """Test the CubeEnv"""
    env = CubeEnv()
    env.reset(scramble_len=50)
    # env.render()

    other_env = CubeEnv()
    other_env.reset(sequence=env.sequence)
    assert env.state == other_env.state

    print('All tests passed.')

if __name__ == '__main__':
    test()
