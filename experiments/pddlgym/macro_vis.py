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
    parser.add_argument('--env_name', type=str, default='hanoi',
                        help='Name of PDDL domain')
    parser.add_argument('--seed', type=int, default=1,
                        help='The index of the particular problem file to use')
    parser.add_argument('-n', type=int, default=8,
                        help='The number of (best) macros to output')
    args = parser.parse_args()

    results_dir = 'results/macros/pddlgym-gen1/{}/problem-{:02d}/'.format(args.env_name, args.seed)
    filenames = glob.glob(results_dir+'seed{:03d}-macros.pickle'.format(args.seed))
    assert(len(filenames)==1)
    filename = filenames[0]
    with open(filename, 'rb') as file:
        search_results = pickle.load(file)
    best_n = search_results[-1]
    best_n = [(score, [a[0] for a in macro]) for score, macro in sorted(best_n)]
    macros = [macro for (_, macro) in best_n if len(macro) > 1]

    # Set up the domain
    video_dir = results_dir+'macro_vis/'
    os.makedirs(video_dir, exist_ok=True)
    for i, macro in enumerate(tqdm(macros[:args.n])):
        env = gym.make("PDDLEnv-Gen-{}-v0".format(args.env_name.capitalize()))
        env.fix_problem_index(args.seed)
        video_path = os.path.join(video_dir+'macro-{:03d}.mp4'.format(i))
        env = VideoWrapper(env, video_path, fps=3)
        obs, _ = env.reset()
        env.render()
        for action in macro:
            env.step(action)
            env.render()
        env.close()
    print('Visualizations saved to:', video_dir)


if __name__ == '__main__':
    main()
