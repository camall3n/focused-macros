import time

from domains import cube
from domains import npuzzle

import gym
import pddlgym

class CPUTimer:
    def __enter__(self):
        self.start = time.time()
        self.end = self.start
        self.duration = 0.0
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.end = time.time()
            self.duration = self.end - self.start

n_steps = 1000
with CPUTimer() as timer:
    puzzle = cube.Cube()
    puzzle.scramble(length=n_steps)
print('Cube FPS:', n_steps / timer.duration)

with CPUTimer() as timer:
    puzzle = npuzzle.NPuzzle()
    puzzle.scramble(n_steps=n_steps)
print('15-puzzle FPS:', n_steps / timer.duration)

with CPUTimer() as timer:
    env_name = "hanoi_operator_actions"
    problem_index = 1
    render = False
    verbose = False
    seed = 0
    env = gym.make("PDDLEnv{}-v0".format(env_name.capitalize()))
    if not render: env._render = None
    env.fix_problem_index(problem_index)

    if seed is not None:
        env.seed(seed)
    obs, _ = env.reset()

    if seed is not None:
        env.action_space.seed(seed)

    for t in range(n_steps):
        action = env.action_space.sample(obs)
        obs, reward, done, _ = env.step(action)
        if done:
            obs = env.reset()
    env.close()
print('Hanoi (operator actions):', n_steps / timer.duration)
