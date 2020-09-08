import os

import gym
from gym.envs.registration import register

def load_learned_macros(env, problem_index):
    """Load the set of learned macro-actions"""
    # Modify env name
    spec = env.spec
    split_point = len(spec._env_name)
    name, version = spec.id[:split_point], spec.id[split_point:]
    spec.id = name + '_macros' + version

    # Modify domain file to use macros
    if 'generated-pddl' in env._problem_dir:
        domain_macros_file = env._problem_dir+'/macros-gen.pddl'
    else:
        domain_macros_file = env._problem_dir+'/macros/'+env.problems[problem_index].problem_fname.split('/')[-1]
    spec._kwargs['domain_file'] = domain_macros_file

    # Register macro-augmented domain with gym
    if os.path.exists(domain_macros_file):
        register(
            id=spec.id,
            entry_point=spec.entry_point,
            kwargs=spec._kwargs,
        )
    else:
        raise FileNotFoundError('Macro-augmented domain file does not exist: {}'.format(domain_macros_file))

    # Make and return macro-augmented environment
    return gym.make(spec.id)
