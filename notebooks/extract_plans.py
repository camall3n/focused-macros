import argparse
import glob
import os
import pickle

# import domains.npuzzle
# import domains.cube
# import experiments.search
# sys.modules['npuzzle'] = domains.npuzzle
# sys.modules['cube'] = domains.cube
# sys.modules['notebooks.search'] = experiments.search
# experiments.search.Node = experiments.search.SearchNode

def parse_args():
    """Parse input arguments

    Use --help to see a pretty description of the arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str, choices=['pddlgym-gen'],
                        help='Name of experiment to plot')
    parser.add_argument('--pddl_env', type=str, required=True,
                        help='Name of pddl environment (used with PDDLGym)')
    parser.add_argument('--alg', type=str, default='gbfs',
                        choices=['gbfs', 'astar', 'weighted_astar', 'bfws_r0', 'bfws_rg'],
                        help='Search algorithm')
    parser.add_argument('--macro_type', '-m', type=str, required=True, choices=['primitive','learned'])
    return parser.parse_args()

def extract_plan(filename):
    with open(filename, 'rb') as file:
        search_results = pickle.load(file)

    states, plan, n_expanded, n_transitions, candidates = search_results[:5]

    action_list = [action for macro in plan for action in macro]
    return action_list

def write_plan_to_file(filename, plan):
    for i, action in enumerate(plan):
        name = action.predicate
        variables = [v.name for v in action.variables]
        plan[i] = '{}: ({} {})\n'.format(i, name, ' '.join(variables))
    with open(filename, 'w') as file:
        file.writelines(plan)


def main():
    args = parse_args()
    results_dir = 'results/{}/{}/{}/{}/'.format(args.name, args.alg, args.pddl_env, args.macro_type)
    print(results_dir)
    filenames = glob.glob(results_dir+'seed*.pickle')
    plans_dir = os.path.join(results_dir,'plans')
    if filenames:
        os.makedirs(plans_dir, exist_ok=True)
    for filename in filenames:
        plan = extract_plan(filename)
        output_filename = os.path.splitext(os.path.split(filename)[-1])[0]+'.plan'
        write_plan_to_file(os.path.join(plans_dir, output_filename), plan)

if __name__ == "__main__":
    main()