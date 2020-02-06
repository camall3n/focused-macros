import glob
from itertools import groupby, count
import os

import numpy as np
import pandas as pd

from experiments.plot_planning_time import (
    parse_args, parse_filepath, cube_cfg, npuzzle_cfg, suitcaselock_cfg
)

# ------------------------------------------------------------------------------
# Adapted from https://codereview.stackexchange.com/a/5202
def as_range(iterable):
    list_ = list(iterable)
    if len(list_) > 1:
        return '{0}-{1}'.format(list_[0], list_[-1])
    return '{0}'.format(list_[0])

def ranges_to_string(range):
    """Convert a list of ranges to a comma-separated string of ranges"""
    return ','.join(as_range(g) for _, g in groupby(range, key=lambda n, c=count(): n-next(c)))
# ------------------------------------------------------------------------------

def show_missing(max_seed):
    """Show seeds in [1,max_seed] that are missing from the current results"""
    result_files = sorted(glob.glob(RESULTS_DIR+'/**', recursive=True))

    completed_runs = []
    for filepath in result_files:
        if not os.path.isfile(filepath):
            continue
        metadata = parse_filepath(filepath, cfg.FIELDS, prefix=RESULTS_DIR)
        if metadata.alg != args.alg:
            continue

        completed_runs.append({
            **metadata._asdict(),
        })

    completed_runs = pd.DataFrame(completed_runs)

    columns = [column for column in list(completed_runs.columns) if column != 'seed']
    keys = completed_runs[columns].values
    str_keys = list(map(lambda x: np.array(list(map(str, x))), keys))
    unique_idx = np.unique(np.stack(str_keys), axis=0, return_index=True)[1]

    results = []
    for key in keys[unique_idx]:
        all_matching_keys = np.all(keys==key, axis=1)
        completed_seeds = completed_runs['seed'][all_matching_keys]
        missing = [x for x in range(1, max_seed+1) if x not in list(completed_seeds)]
        missing_str = ranges_to_string(missing)
        if not missing:
            missing_str = 'N/A'
        results.append({
            **dict(zip(columns, key)),
            'seeds': missing_str,
        })

    results = pd.DataFrame(results)
    print('Missing:')

    pd.set_option('display.max_colwidth', 500)
    pd.set_option('display.width', 1000)
    print(results)

if __name__ == '__main__':
    args = parse_args()
    cfg = {
        'cube': cube_cfg,
        'npuzzle': npuzzle_cfg,
        'suitcaselock': suitcaselock_cfg,
    }[args.name]
    RESULTS_DIR = 'results/' + args.name + '/'

    show_missing(max_seed=100)
