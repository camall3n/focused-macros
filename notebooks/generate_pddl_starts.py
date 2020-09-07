import glob
import os
import random

import gym
import pddlgym
from pddlgym.structs import LiteralConjunction

from domains.pddlgym.pddlgymenv import all_scrambles

def remove_half_turns(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    output = []

    # Grab header up to first operator
    first_operator_pos = lines.index('begin_operator\n')
    output.extend(lines[:first_operator_pos])
    del lines[:first_operator_pos]

    output[-1] = output[-1].replace('18','12')# Modify the number of operators

    while lines[0].strip('\n') == 'begin_operator':
        current_operator_name = lines[1].strip('\n')
        current_operator_end = lines.index('end_operator\n')
        # print('Found operator:', current_operator_name)
        if '2_0' in current_operator_name:
            # print('Half-turn operator. Removing...')
            pass
        else:
            # print('Quarter-turn operator. Saving...')
            output.extend(lines[:current_operator_end+1])

        del lines[:current_operator_end+1]

    output.extend(lines)
    del lines

    new_dir = 'domains/cube/buchner2018/problems'
    new_filename = os.path.split(filename)[-1]
    new_filepath = os.path.join(new_dir,new_filename)
    with open(new_filepath, 'w') as file:
        file.writelines(output)
    print('Wrote modified domain file to:', new_filepath)

if __name__ == "__main__":
    problems = [
        ('depot', 3, True),
        ('doors', 9, True),
        ('elevator', 8, True),
        ('ferry', 4, True),
        ('gripper', 19, False),
        ('hanoi_operator_actions', 10, False),
    ]
    for env_name, problem_index, typed in problems:
        env = gym.make("PDDLEnv{}-v0".format(env_name.capitalize()))
        env.fix_problem_index(problem_index)
        initial_state, _ = env.reset()
        problem = env._problem
        print(problem.domain_name)
        output_dir = 'pddlgym/pddlgym/generated-pddl/'+problem.domain_name
        scrambles = all_scrambles(env)
        if len(scrambles) < 100:
            scrambles = all_scrambles(env, max_steps=6000)

        for seed in range(101):
            if seed == 0:
                start = initial_state
            elif seed <= len(scrambles):
                start = scrambles[seed-1]
            else:
                break

            new_prob_name = problem.problem_name+'_{:02d}'.format(seed)
            new_problem = problem.pddl_string(
                              problem.objects,
                              start.literals,
                              new_prob_name,
                              problem.domain_name,
                              problem.goal,
                              fast_downward_order=True,
                              typed=typed,
                          )
            new_filename = os.path.join(output_dir, 'problem{:03d}.pddl'.format(seed))
            with open(new_filename, 'w') as file:
                file.write(new_problem)
