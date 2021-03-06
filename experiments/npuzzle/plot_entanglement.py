import os

import matplotlib.pyplot as plt
import numpy as np

from domains.npuzzle import npuzzle
from domains.npuzzle import macros

def load_and_plot_macros():
    """Load and plot N-Puzzle macros"""
    filter_ = lambda x,y: zip(*[(x,y) for x,y in zip(x,y) if x!=1])
    rnd_macro_len = list(map(len,macros.random.macros[(0,0)]))
    rnd_macro_ent = list(map(lambda x: len(x[0]),macros.random.models[(0,0)]))
    rnd_macro_len, rnd_macro_ent = filter_(rnd_macro_len, rnd_macro_ent)

    gen_macro_len = list(map(len,macros.learned.macros[(0,0)]))
    gen_macro_ent = list(map(lambda x: len(x[0]),macros.learned.models[(0,0)]))
    gen_macro_len, gen_macro_ent = filter_(gen_macro_len, gen_macro_ent)

    noise = 0.
    offset = 0.1
    plt.rcParams.update({'font.size': 12})
    plt.subplots(figsize=(4,3))
    # plt.grid('on')
    x = [1-offset]
    y = [2]
    plt.scatter(x,y, c='C0', s=150, marker='o', label='Primitive')
    x = np.asarray(rnd_macro_len)+offset+np.random.normal(0,noise,len(rnd_macro_len))
    y = np.asarray(rnd_macro_ent)+np.random.normal(0,noise,len(rnd_macro_ent))
    plt.scatter(x,y, c='C2', s=150, marker='^', label='Random')
    x = np.asarray(gen_macro_len)+np.random.normal(0,noise,len(gen_macro_len))
    y = np.asarray(gen_macro_ent)+np.random.normal(0,noise,len(gen_macro_ent))
    plt.scatter(x,y, c='C3', s=150, marker='+', label='Learned')
    plt.xlabel('Number of steps per macro-action')
    plt.ylabel('Number of variables modified')
    plt.gcf().set_size_inches(8,6)
    plt.xlim([0,20])
    plt.ylim([1.5,9.5])
    plt.xticks(range(1,20,3))
    plt.yticks(range(2,10))
    plt.gca().set_axisbelow(True)
    plt.legend(loc='upper left')
    plt.tight_layout()
    # plt.title('Entanglement by macro-action type (15-puzzle)')
    os.makedirs('results/plots/npuzzle/', exist_ok=True)
    plt.savefig('results/plots/npuzzle/npuzzle_entanglement.png')
    plt.show()

def visualize():
    """Visualize some macros"""
    for blank_idx in [(3,3)]:#macros.learned.models.keys():
        macro_list = macros.learned.macros[blank_idx]
        model_list = macros.learned.models[blank_idx]
        for i in range(len(macro_list)):
            macro = macro_list[i]
            model = model_list[i]
            if len(model[0]) == 2 and len(macro) == 19:
                puz = npuzzle.NPuzzle(n=15, start_blank=blank_idx)
                print(puz)
                puz.apply_macro(model=model)
                print(macro)
                print(model)
                print(puz)
                print()

if __name__ == '__main__':
    load_and_plot_macros()
    # visualize()
