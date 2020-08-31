import gym, pddlgym

def test_env(env_name='barman-sat14-strips'):
    print('Testing {}... '.format(env_name), end='', flush=True)

    try:
        env = gym.make('PDDLEnv-IPC-{}-v0'.format(env_name.capitalize()))
        state, _ = env.reset()
        env.step(env.action_space.sample(state))
        env.close()
    except Exception as e:
        print('Failed.')
        print('  ', e)
        print()
    else:
        print('Success!')

for env_name in [
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
]:
    test_env(env_name)