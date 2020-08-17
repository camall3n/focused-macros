import time

from domains import cube
from domains import npuzzle

import gym
import pddlgym
from pddlgym.structs import LiteralConjunction

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


env = gym.make("PDDLEnv{}-v0".format(env_name.capitalize()))
if not render: env._render = None
env.fix_problem_index(problem_index)

if seed is not None:
    env.seed(seed)
obs0, _ = env.reset()

assert isinstance(obs.goal, LiteralConjunction)
heuristic = lambda obs: len([lit for lit in obs.goal.literals if lit not in obs.literals])
h0 = heuristic(obs0)

if seed is not None:
    env.action_space.seed(seed)

observations = []
actions = []
heuristic_values = []
for t in range(30):
    action = env.action_space.sample(obs)
    obs, reward, done, _ = env.step(action)
    actions.append(action)
    observations.append(obs)
    h = heuristic(obs)
    heuristic_values.append(h)
    if done:
        obs = env.reset()
env.set_state(obs0)
obs = env.get_state()
assert obs == obs0
for t in range(30):
    action = actions[t]
    obs, reward, done, _ = env.step(action)
    assert obs == observations[t]
    h = heuristic(obs)
    assert h == heuristic_values[t]
    print(h, "==", heuristic_values[t])
    if done:
        obs = env.reset()
print('All tests passed.')
