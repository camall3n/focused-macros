import argparse
from collections import OrderedDict
import glob
import pickle
import os

import gym
import pddlgym
from pddlgym.utils import VideoWrapper
from tqdm import tqdm

def main():
    """Visualize PDDLGym macros"""

    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='hanoi_operator_actions',
                        help='Name of PDDL domain')
    parser.add_argument('--problem_index', type=int, default=None,
                        help='The index of the particular problem file to use')
    args = parser.parse_args()

    results_dir = 'results/macros/pddlgym/{}/problem-{:02d}/'.format(args.env_name, args.problem_index)
    filenames = glob.glob(results_dir+'seed001-macros.pickle')
    assert(len(filenames)==1)
    filename = filenames[0]
    macros = OrderedDict()
    with open(filename, 'rb') as file:
        search_results = pickle.load(file)
    best_n = search_results[-1]
    best_n = [(score, [a[0] for a in macro]) for score, macro in sorted(best_n)]

    clean_macros = []
    for _, macro in best_n:
        if len(macro) > 1:
            clean_macros.append(macro)

    # Set up the domain
    video_dir = results_dir+'macro_vis/'
    os.makedirs(video_dir, exist_ok=True)
    for i, macro in enumerate(tqdm(clean_macros[:10])):
        env = gym.make("PDDLEnv{}-v0".format(args.env_name.capitalize()))
        env.fix_problem_index(args.problem_index)
        video_path = os.path.join(video_dir+'macro-{:03d}.mp4'.format(i))
        env = VideoWrapper(env, video_path, fps=3)
        obs, _ = env.reset()
        env.render()
        for action in macro:
            env.step(action)
            env.render()
        env.close()


if __name__ == '__main__':
    main()
