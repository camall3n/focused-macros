import argparse
from collections import OrderedDict, namedtuple
import glob
import pickle
import os

import gym
import pddlgym
from pddlgym.utils import VideoWrapper
from tqdm import tqdm

Type = pddlgym.structs.Type
TypedEntity = pddlgym.structs.TypedEntity
Literal = pddlgym.structs.Literal
LiteralConjunction = pddlgym.structs.LiteralConjunction
Operator = pddlgym.parser.Operator

def bind_literal(literal, binding, use_typing=False):
    predicate = literal.predicate
    new_vars = [TypedEntity(binding[var], var.var_type) for var in literal.variables]
    return Literal(predicate, new_vars)

def build_macro_operator(name, macro, primitive_operators, use_typing=False):
    params = set()
    typed_params = set()
    net_preconds = set()
    net_effects = set()
    for step, action in enumerate(macro):
        operator = primitive_operators[action.predicate.name]
        binding = {param: '?'+var for (param, var) in zip(operator.params, action.pddl_variables())}
        typed_binding = {param: '?'+var for (param, var) in zip(operator.params, action.pddl_variables_typed())}
        for param in operator.params:
            params.add(binding[param])
            typed_params.add(typed_binding[param])
        for literal in operator.preconds.literals:
            bound_literal = bind_literal(literal, binding, use_typing=use_typing)
            if bound_literal.inverted_anti in net_effects:
                msg = ("Precondition not satisfied for action {}"
                        " in step {} of macro:\n {}\n"
                        "Literal {} is negated in macro effects: {}")
                msg = msg.format(action, step, macro_id, bound_literal, net_effects)
                raise RuntimeError(msg)
            if bound_literal not in net_effects and bound_literal not in net_preconds:
                net_preconds.add(bound_literal)
        for literal in operator.effects.literals:
            bound_literal = bind_literal(literal, binding)
            if bound_literal.inverted_anti in net_effects:
                net_effects.remove(bound_literal.inverted_anti)
            else:
                net_effects.add(bound_literal)

    if use_typing:
        params = sorted(list(typed_params))
    else:
        params = sorted(list(params))
    net_preconds = LiteralConjunction(sorted(list(net_preconds)))
    net_effects = LiteralConjunction(sorted(list(net_effects)))
    return Operator(name, params, net_preconds, net_effects)

def main():
    """Clean up PDDLGym macros"""

    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='hanoi_operator_actions',
                        help='Name of PDDL domain')
    parser.add_argument('--problem_index', type=int, default=10,
                        help='The index of the particular problem file to use')
    parser.add_argument('-n', type=int, default=16,
                        help='The number of (best) macros to output')
    args = parser.parse_args()

    results_dir = 'results/macros/pddlgym/{}/problem-{:02d}/'.format(args.env_name, args.problem_index)
    filenames = glob.glob(results_dir+'seed*-macros.pickle')
    assert(len(filenames)==1)
    filename = filenames[0]
    with open(filename, 'rb') as file:
        search_results = pickle.load(file)
    best_n = search_results[-1]
    best_n = [(score, [a[0] for a in macro]) for score, macro in sorted(best_n)]
    raw_macros = [macro for (_, macro) in best_n if len(macro) > 1]

    env = gym.make("PDDLEnv{}-v0".format(args.env_name.capitalize()))
    env.fix_problem_index(args.problem_index)
    operators = env.action_space._action_predicate_to_operators
    start, _ = env.reset()

    macros = []
    for macro_id, macro in enumerate(raw_macros[:args.n]):
        macro_name = 'macro{:04d}'.format(macro_id)
        macro_operator = build_macro_operator(macro_name, macro,
                                              primitive_operators=operators,
                                              use_typing=env.domain.uses_typing)
        macros.append(macro_operator)

    with open(env._domain_file, 'r') as file:
        domain_pddl = file.read()

    #%% Save the results
    # split domain file and insert macros just before final closing parenthetical
    insertion_point = domain_pddl.rfind(')')
    header, footer = domain_pddl[:insertion_point], domain_pddl[insertion_point:]
    body = ''.join([macro.pddl_str().replace('\t', '  ') for macro in macros])+'\n'
    macro_dir = env._domain_file.split('.')[0]+'/macros/'
    os.makedirs(macro_dir, exist_ok=True)
    problem_filename = env.problems[args.problem_index].problem_fname.split('/')[-1]
    macro_file = macro_dir+problem_filename
    with open(macro_file, 'w') as file:
        file.write(header+body+footer)
    print('Wrote macro-augmented domain to file: {}'.format(macro_file))

if __name__ == "__main__":
    main()