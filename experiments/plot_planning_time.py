import argparse
from collections import namedtuple
import glob
import os
import pickle
import sys

import matplotlib.pyplot as plt
from matplotlib.axes._axes import _log as matplotlib_axes_logger
import numpy as np
import pandas as pd
import seaborn as sns

import experiments.cube.plot_config as cube_cfg
import experiments.npuzzle.plot_config as npuzzle_cfg
import experiments.suitcaselock.plot_config as suitcaselock_cfg
import experiments.pddlgym.plot_config as pddlgym_cfg

def parse_args():
    """Parse input arguments

    Use --help to see a pretty description of the arguments
    """
    if 'ipykernel' in sys.argv[0]:
        sys.argv = [sys.argv[0]] + 'suitcaselock'.split(' ')
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str, choices=['cube', 'npuzzle', 'suitcaselock', 'pddlgym'],
                        help='Name of experiment to plot')
    parser.add_argument('--pddl_env', type=str, required=False,
                        help='Name of pddl environment (used with PDDLGym)')
    parser.add_argument('--pddl_problem_id', type=int, required=False,
                        help='ID of pddl problem (used with PDDLGym)')
    parser.add_argument('--alg', type=str, default='gbfs',
                        choices=['gbfs', 'astar', 'weighted_astar', 'bfws', 'bfwsr'],
                        help='Search algorithm')
    return parser.parse_args()


def parse_filepath(path, field_names, prefix):
    """Parse a results filepath into metadata

    Args:
        path (str):
            The filepath from the base directory of the project (must start with 'prefix')
        field_names (list of str):
            The ordered list of names of the fields to parse into metadata
        prefix (str):
            The path to the first directory with parsable contents

    Returns:
        A namedtuple of metadata, with a field for each of the field_names

    Example:
        parse_filepath(
            path = '/results/foo/var-3/tag-bar/seed-17.pickle',
            field_names = ['var', 'tag', 'seed'],
            prefix = '/results/foo/'
        ) -> namedtuple('MetaData', var=3, tag='bar')
    """
    assert path.startswith(prefix)
    path = path[len(prefix):]
    filename_sections = path.split('/')

    parsed_sections = []
    for text, field in zip (filename_sections, field_names):
        if field in ['alg', 'goal_type', 'macro_type', 'pddl_env']:
            parsed_sections.append(text)
        elif field == 'puzzle_size':
            text = text.split('-')[0]
            parsed_sections.append(int(text))
        elif field == 'seed':
            text = text.split('.')[0].split('-')[-1]
            parsed_sections.append(int(text))
        elif field in ['n_vars', 'n_values', 'entanglement', 'problem_id']:
            text = text.split('-')[-1]
            parsed_sections.append(int(text))

    return namedtuple('MetaData', field_names)(*parsed_sections)

def load_data(alg, pddl_env=None, pddl_problem_id=None):
    """Load all data in RESULTS_DIR matching the specified algorithm"""
    result_files = sorted(glob.glob(RESULTS_DIR+'/**', recursive=True))

    learning_curves = []
    # macro_data = []
    final_results = []
    for filepath in result_files:
        if not os.path.isfile(filepath) or os.path.splitext(filepath)[-1] != '.pickle':
            continue
        if 'archive' in filepath:
            continue
        metadata = parse_filepath(filepath, cfg.FIELDS, prefix=RESULTS_DIR)
        if metadata.alg != alg:
            continue
        if pddl_env is not None and metadata.pddl_env != pddl_env:
            continue
        if pddl_problem_id is not None and metadata.problem_id != pddl_problem_id:
            continue
        with open(filepath, 'rb') as file:
            search_results = pickle.load(file)
        states, actions, _, n_transitions, candidates = search_results[:5]
        goal = cfg.get_goal(states[0], metadata)
        n_errors = cfg.heuristic(states[-1], goal)

        sim_steps = [transitions for transitions, node in candidates]
        if metadata.alg not in ['bfws','bfwsr']:
            h_scores = [node.h_score for transitions, node in candidates]
        else:
            h_scores = [node.h_score[1] for transitions, node in candidates]

        # Extend final value to end of plot
        if n_errors > 0:
            sim_steps += [n_transitions]
            h_scores += [n_errors]

        # Save learning curves
        for sim_step, h_score in zip(sim_steps, h_scores):
            learning_curves.append({
                **metadata._asdict(),
                'transitions': sim_step,
                'n_errors': h_score,
            })

        # # Save macro data
        # for length in cfg.get_macro_lengths(actions):
        #     macro_data.append({
        #         **metadata._asdict(),
        #         'macro_length': length,
        #     })

        # Save final results
        final_results.append({
            **metadata._asdict(),
            'transitions': n_transitions,
            'n_errors': n_errors,
            'n_action_steps': cfg.get_primitive_steps(actions),
            'n_macro_steps': cfg.get_macro_steps(actions),
        })

    results = [learning_curves, final_results] #, macro_data
    return tuple(map(pd.DataFrame, results))


def _autoscale_ticks(set_fn, get_fn):
    """Automatically scale ticks and return a string for labeling the axis"""
    if 2000 < cfg.TRANSITION_CAP < 1e6:
        set_fn(map(int, np.asarray(get_fn(), dtype=int)//1e3))
        scale_str = ' (in thousands)'
    elif cfg.TRANSITION_CAP >= 1e6:
        set_fn(np.asarray(get_fn())/1e6)
        scale_str = ' (in millions)'
    else:
        scale_str = ''
    return scale_str

def autoscale_xticks(ax):
    """Automatically scale xticks and return a string for labeling the axis"""
    return _autoscale_ticks(ax.set_xticklabels, ax.get_xticks)

def autoscale_yticks(ax):
    """Automatically scale yticks and return a string for labeling the axis"""
    return _autoscale_ticks(ax.set_yticklabels, ax.get_yticks)


def plot_learning_curves(data, plot_var_list, category):
    """Plot n_errors vs. time, with hue according to the specified category"""

    if data.empty:
        return

    plt.rcParams.update({'font.size': cfg.FONTSIZE})
    _, ax = plt.subplots(figsize=cfg.FIGSIZE)

    lines = []
    names = []
    for plot_vars in plot_var_list:
        value = plot_vars._asdict()[category]
        if len(data.query(category+'==@value')) > 0:
            sns.lineplot(data=data.query(category+'==@value'),
                         x='transitions', y='n_errors',
                         legend=False, estimator=None, units='seed',
                         ax=ax, linewidth=2, alpha=.6,
                         color=plot_vars.color, zorder=plot_vars.zorder)
            lines.append(ax.get_lines()[-1])
            names.append(value)

    ax.legend(lines, names, framealpha=1, borderpad=0.7)
    ax.set_ylim(cfg.YLIM)
    ax.set_xlim(cfg.XLIM)
    ax.set_xlabel('Generated states' + autoscale_xticks(ax))
    ax.set_ylabel('Number of errors remaining')
    ax.set_axisbelow(False)
    # [i.set_linewidth(1) for i in ax.spines.values()]
    if cfg.HLINE:
        ax.hlines(cfg.HLINE, 0, cfg.TRANSITION_CAP, linestyles='dashed', linewidths=2)
    plt.savefig('results/plots/{}/{}_planning_curves_by_{}.png'.format(
        cfg.DIR, cfg.NAME, category), dpi=100)
    plt.show()


def plot_planning_boxes(data, plot_var_list, category):
    """Boxplot planning time, with hue according to the specified category"""

    if data.empty:
        return
    # In order to get the right legend on the plot, first make an empty plot with colored lines
    plt.figure()
    palette = []
    for plot_vars in plot_var_list:
        plt.plot(0, 0, c=plot_vars.color, label=plot_vars._asdict()[category], lw=3)
        palette.append(plot_vars.color)
    plt.legend()
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.show()
    plt.close()

    # Now we make the actual plot, since seaborn's catplot doesn't accept an 'ax' argument
    plt.rcParams.update({'font.size': cfg.FONTSIZE, 'figure.figsize': cfg.FIGSIZE})
    matplotlib_axes_logger.setLevel('ERROR')
    catplot = sns.catplot(data=data.query('n_errors==0'), y=category, x='transitions',
                          kind='boxen', palette=reversed(palette), orient='h', legend='True')
    catplot.despine(right=False, top=False)
    plt.ylabel('Macro-action type')
    plt.gcf().set_size_inches(*cfg.FIGSIZE)
    plt.tight_layout()
    plt.xlim(cfg.XLIM)
    ax = plt.gca()
    ax.invert_yaxis()
    ax.set_yticklabels([])
    ax.set_xlabel('Generated states' + autoscale_xticks(ax))
    plt.tight_layout()
    ax.legend(handles, labels, loc='lower right')
    plt.gcf().savefig('results/plots/{}/{}_planning_time_by_{}.png'.format(
        cfg.DIR, cfg.NAME, category), dpi=100)
    plt.show()


def plot_entanglement_boxes(data):
    """Boxplot planning time vs entanglement"""
    if data.empty:
        return
    plt.rcParams.update({'font.size': cfg.FONTSIZE})
    _, ax = plt.subplots(figsize=cfg.FIGSIZE)
    sns.boxplot(x='entanglement', y='transitions', data=data, color='C0', ax=ax)

    n_values = list(data['n_values'])[0]

    plt.xlabel('Variables modified per action')
    ax.set_ylabel('Generated states')
    ax.set_yscale('linear')
    ax.set_ylabel('Generated states' + autoscale_yticks(ax))
    plt.tight_layout()
    plt.savefig('results/plots/{}/{}_{}ary.png'.format(
        cfg.DIR, cfg.NAME, n_values), dpi=100)
    plt.show()


def get_summary(results, category):
    """Print a summary of the planning results broken down for the specified category"""
    summary = results.groupby([category], as_index=False).mean().round(1)
    summary['solves'] = [len(results.query(category+'==@tag and n_errors==0'))
                         for tag in summary[category]]
    summary['attempts'] = [len(results.query(category+'==@tag')) for tag in summary[category]]
    return str(summary)

def make_plots():
    """Make the plots and print summaries"""
    learning_curves, final_results = load_data(alg=args.alg,
                                               pddl_env=args.pddl_env,
                                               pddl_problem_id=args.pddl_problem_id)
    os.makedirs('results/plots/'+cfg.DIR+'/', exist_ok=True)
    if 'learning_curves' in cfg.PLOTS:
        try:
            data = learning_curves.query("goal_type=='default_goal'")
        except pd.core.computation.ops.UndefinedVariableError:
            data = learning_curves
        plot_learning_curves(data, cfg.PLOT_VARS, category='macro_type')
    if 'planning_boxes' in cfg.PLOTS:
        try:
            data = final_results.query("goal_type=='default_goal'")
        except pd.core.computation.ops.UndefinedVariableError:
            data = final_results
        plot_planning_boxes(data, cfg.PLOT_VARS, category='macro_type')

    if 'alt_learning_curves' in cfg.PLOTS:
        data = learning_curves.query("macro_type=='learned'")
        plot_learning_curves(data, cfg.PLOT_VARS_ALT, category='goal_type')
    if 'alt_planning_boxes' in cfg.PLOTS:
        data = final_results.query("macro_type=='learned'")
        plot_planning_boxes(data, cfg.PLOT_VARS_ALT, category='goal_type')

    if 'entanglement_boxes' in cfg.PLOTS:
        for plot_vars in cfg.PLOT_VARS: # pylint: disable=W0612
            plot_entanglement_boxes(final_results.query("n_vars==@plot_vars.n_vars"))

    summary_text = []
    if any([summary_type == 'macro_type' for summary_type in cfg.SUMMARIES]):
        try:
            results = final_results.query("goal_type=='default_goal'")
        except pd.core.computation.ops.UndefinedVariableError:
            results = final_results
        results = results[['macro_type', 'transitions', 'n_errors']]
        summary_text.append(get_summary(results, category='macro_type'))

    if any([summary_type == 'goal_type' for summary_type in cfg.SUMMARIES]):
        results = final_results.query("macro_type=='learned'")
        results = results[['goal_type', 'transitions', 'n_errors']]
        summary_text.append(get_summary(results, category='goal_type'))

    if summary_text:
        summary_text = '\n\n'.join(summary_text)
        print(summary_text)
        with open('results/plots/{}/{}_summary.txt'.format(cfg.DIR, cfg.NAME), 'w') as file:
            file.write(summary_text)

if __name__ == '__main__':
    args = parse_args()
    cfg = {
        'cube': cube_cfg,
        'npuzzle': npuzzle_cfg,
        'suitcaselock': suitcaselock_cfg,
        'pddlgym': pddlgym_cfg,
    }[args.name]

    RESULTS_DIR = 'results/' + cfg.NAME + '/'
    if args.name == 'pddlgym':
        cfg.DIR = cfg.DIR.format(args.pddl_env)
        cfg.NAME = args.pddl_env
    make_plots()
