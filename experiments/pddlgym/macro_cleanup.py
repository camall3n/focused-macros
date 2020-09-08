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

def var_generator(prefix='var'):
    i = 0
    while True:
        yield prefix+'{:04d}'.format(i)
        i += 1

def equal_operators(op1, op2):
    return op1.params == op2.params and op1.effects == op2.effects and op1.preconds == op2.preconds

def build_macro_operator(name, macro, primitive_operators, use_typing=False):
    params = set()
    typed_params = set()
    net_preconds = set()
    net_effects = set()
    var_name_gen = var_generator(prefix='var')
    lifted = {}
    for step, action in enumerate(macro):
        grounded_var_names = action.pddl_variables()
        var_types = action.predicate.var_types
        lifted.update({name: next(var_name_gen) for name in grounded_var_names if name not in lifted})
        lifted_var_names = [lifted[name] for name in grounded_var_names]

        operator = primitive_operators[action.predicate.name]
        binding = {param: '?'+var for (param, var) in zip(operator.params, lifted_var_names)}
        typed_binding = {param: '?'+var+' - '+type_ for (param, var, type_) in zip(operator.params, lifted_var_names, var_types)}

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

        # clean operator effects:
        #   keep [(A)] -> [(A)]
        #   keep [(not A)] -> [(not A)]
        #   convert [(not A), (A)] -> [(A)]
        effect_literals = [
            literal for literal in operator.effects.literals
            if not literal.is_anti or (literal.inverted_anti not in operator.effects.literals)
        ]

        for literal in effect_literals:
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
    parser.add_argument('--env_name', type=str, default='miconic',
                        help='Name of PDDL domain')
    parser.add_argument('--seed', '-s', type=int, default=10,
                        help='The index of the particular problem file to use')
    parser.add_argument('-n', type=int, default=16,
                        help='The number of (best) macros to output')
    args = parser.parse_args()

    results_dir = 'results/macros/pddlgym-gen/{}/problem-{:02d}/'.format(args.env_name, args.seed)
    filenames = glob.glob(results_dir+'seed*-macros.pickle')
    assert(len(filenames)==1)
    filename = filenames[0]
    with open(filename, 'rb') as file:
        search_results = pickle.load(file)
    n_transitions = search_results[2]
    print('Macro search generated', n_transitions, 'states.')
    print()
    best_n = search_results[-1]
    best_n = [(score, [a[0] for a in macro]) for score, macro in sorted(best_n)]
    raw_macros = [macro for (_, macro) in best_n if len(macro) > 1]

    env = gym.make("PDDLEnv-Gen-{}-v0".format(args.env_name.capitalize()))
    env.fix_problem_index(args.seed)
    operators = env.action_space._action_predicate_to_operators
    start, _ = env.reset()

    # Generate unique macros
    macros = []
    for macro in raw_macros:
        macro_name = 'macro{:04d}'.format(len(macros))
        macro_operator = build_macro_operator(macro_name, macro,
                                              primitive_operators=operators,
                                              use_typing=env.domain.uses_typing)
        if not any([equal_operators(macro_operator, m) for m in macros]):
            macros.append(macro_operator)

    # Only save the best N macros
    macros = macros[:args.n]

    with open(env._domain_file, 'r') as file:
        domain_pddl = file.read()

    #%% Save the results
    # split domain file and insert macros just before final closing parenthetical
    insertion_point = domain_pddl.rfind(')')
    header, footer = domain_pddl[:insertion_point], domain_pddl[insertion_point:]
    body = ''.join([macro.pddl_str().replace('\t', '  ') for macro in macros])+'\n'
    macro_file = os.path.join(os.path.split(env._domain_file)[0],'macros-gen.pddl')
    # os.makedirs(macro_dir, exist_ok=True)
    # problem_filename = env.problems[args.seed].problem_fname.split('/')[-1]
    # macro_file = macro_dir+problem_filename
    with open(macro_file, 'w') as file:
        file.write(header+body+footer)
    print('Wrote macro-augmented domain to file: {}'.format(macro_file))

if __name__ == "__main__":
    main()