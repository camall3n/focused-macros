import argparse

import gym, pddlgym

env_names = [
    'barman-sat14-strips',
    'blocks',
    'depot',
    'driverlog',
    'freecell',
    'grid',
    'gripper',
    'hiking-sat14-strips',
    'logistics00',
    'logistics98',
    'miconic',
    'movie',
    'mprime',
    'mystery',
    'rovers',
    'satellite',
    'thoughtful-sat14-strips',
    'tidybot-sat11-strips',
    'tpp',
    'visitall-sat11-strips',
    'visitall-sat14-strips',
    'zenotravel',
]

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default=None, choices=env_names,
                    help='Name of PDDL domain. If none supplied, loop through all IPC domains.')
args = parser.parse_args()

def test_env(env_name='barman-sat14-strips'):
    print('Testing {}... '.format(env_name), end='', flush=True)

    try:
        env = gym.make('PDDLEnv-IPC-{}-v0'.format(env_name.capitalize()))
        n_problems = len(env.problems)
        state, _ = env.reset()
        env.step(env.action_space.sample(state))
        env.close()
        print('Success!  Max problem ID: ', n_problems-1)
    except Exception as e:
        print('Failed.')
        print('  ', e)
        print()


if args.env_name is not None:
    env_names = [args.env_name]
for env_name in env_names:
    test_env(env_name)