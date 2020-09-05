import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tabulate
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=6, help='Number of dials')
    parser.add_argument('-v', type=int, default=2, help='Number of digits per dial')
    args = parser.parse_args()

    n = args.n
    v = args.v

    results_dir = 'results/heuristic/lock_{}x{}ary/'.format(n, v)

    os.makedirs('results/plots/heuristic/', exist_ok=True)
    base_filename = 'results/plots/heuristic/suitcaselock_heuristic_vs_true_distance_{}x{}ary'.format(n,v)

    # fig, ax = plt.subplots(figsize=(8,6))
    # sns.pointplot(data=data, x='distance', y='heuristic', hue='k', ci=None, dodge=0.25)
    # plt.savefig(base_filename+'.png')
    # plt.show()

    # with open(base_filename+'.txt', 'w') as file:
        # file.write('k,corr\n')
    print('k','pearson_r','spearman_r')
    results = []
    for i in range(1,n):
        results_files = glob.glob(results_dir+'k-{:02d}_*.csv'.format(i))
        data = []
        for results_file in results_files:
            data.append(pd.read_csv(results_file))
        k_data = pd.concat(data)

        pr = k_data['distance'].corr(k_data['heuristic'], 'pearson')
        sr = k_data['distance'].corr(k_data['heuristic'], 'spearman')
        s = '{},{},{}\n'.format(i, pr, sr)
        results.append([i, pr, sr])
        # file.write(s)
        print(s.replace(',', ' ').replace('\n',''))
    print(tabulate.tabulate(results, ['k','pearson_r','spearman_r']))