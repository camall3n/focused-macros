import argparse
import copy
import os
import pickle
import random
import sys
from types import SimpleNamespace

import gym
import pddlgym
from pddlgym.structs import LiteralConjunction
from pddlgym.utils import VideoWrapper

def parse_args():
    """Parse input arguments

    Use --help to see a pretty description of the arguments
    """
    if 'ipykernel' in sys.argv[0]:
        sys.argv = [sys.argv[0]]
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='hanoi_operator_actions',
                        help='Name of PDDL domain')
    parser.add_argument('--problem_index', type=int, default=None,
                        help='The index of the particular problem file to use')
    parser.add_argument('--steps','-s', type=int, default=100,
                        help='Number of steps')
    # parser.add_argument('--scramble_length', '-n', type=int, default=0,
    #                     help='Number of random actions to generate initial state.')
    return parser.parse_args()

def main():
    """Instantiate PDDL domain with PDDLGym and generate initial states"""
    args = parse_args()

    env = gym.make("PDDLEnv{}-v0".format(args.env_name.capitalize()))
    env.fix_problem_index(args.problem_index)
    initial_state, _ = env.reset()

    n_steps = 0
    seen_states = set()
    frontier = [initial_state]
    while frontier and n_steps < args.steps:
        state = frontier.pop()
        valid_actions = sorted(list(env.action_space.all_ground_literals(state)))
        for action in valid_actions:
            env.set_state(state)
            next_state = env.step(action)[0]
            n_steps += 1
            if next_state not in seen_states:
                seen_states.add(next_state)
                frontier.add(next_state)

    print(len(seen_states), 'unique states after', args.steps, 'steps')

if __name__ == "__main__":
    main()