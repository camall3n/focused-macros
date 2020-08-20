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

def bind_literal(literal, binding):
    predicate = literal.predicate
    new_vars = [TypedEntity('?'+binding[var], var.var_type) for var in literal.variables]
    return Literal(predicate, new_vars)

def build_macro(*actions):
    parameters = frozenset.union(*[a.parameters for a in actions])
    net_precondition = set()
    net_effect = set()
    for action in actions:
        for literal in action.precondition:
            if (literal.negation) in net_effect:
                msg = 'Precondition {i} not satisfied (action = {}). Literal {} is negated in effects: {}'
                raise RuntimeError(msg.format(i, a, literal, effects))
            if literal not in net_effect and literal not in net_precondition:
                net_precondition.add(literal)
        for literal in action.effect:
            if (literal.negation) in net_effect:
                net_effect.remove(literal.negation)
            else:
                net_effect.add(literal)
    return action(parameters, net_precondition, net_effect)


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
    filenames = glob.glob(results_dir+'seed001-macros.pickle')
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
        params = set()
        net_preconds = set()
        net_effects = set()
        for step, action in enumerate(macro):
            operator = operators[action.predicate.name]
            binding = {param: var for (param, var) in zip(operator.params, action.pddl_variables())}
            for param in operator.params:
                params.add(binding[param])
            for literal in operator.preconds.literals:
                bound_literal = bind_literal(literal, binding)
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

        params = sorted(list(params))
        net_preconds = LiteralConjunction(sorted(list(net_preconds)))
        net_effects = LiteralConjunction(sorted(list(net_effects)))
        macro_operator = Operator('macro{:04d}'.format(macro_id), params, net_preconds, net_effects)

        print(macro_operator.pddl_str())
        print()

    #%% Save the results
    # with open(results_dir+'clean_macros.pickle', 'wb') as file:
    #     pickle.dump(macros, file)

if __name__ == "__main__":
    main()