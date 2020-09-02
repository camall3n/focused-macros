import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=6, help='Number of dials')
    parser.add_argument('-v', type=int, default=2, help='Number of digits per dial')
    args = parser.parse_args()

    n = args.n
    v = args.v

    results_file = 'results/heuristic/data_{}x{}ary.csv'.format(n, v)
    data = pd.read_csv(results_file)

    base_filename = 'results/plots/suitcaselock_heuristic_vs_true_distance_{}x{}ary'.format(n,v)

    fig, ax = plt.subplots(figsize=(8,6))
    sns.pointplot(data=data, x='distance', y='heuristic', hue='k', ci='sd', dodge=0.25)
    plt.savefig(base_filename+'.png')
    plt.show()

    with open(base_filename+'.txt', 'w') as file:
        file.write('k,corr\n')
        print('k','corr')
        for i in range(1,n):
            s = '{},{:0.2f}\n'.format(i, data.query('k==@i')['distance'].corr(data['heuristic']).round(2))
            file.write(s)
            print(s.replace(',', ' ').replace('\n',''))
