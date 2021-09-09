import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from domains import cube
from domains.cube import macros

def jitter(arr, sd=None):
    if sd is None:
        sd = 0
    return arr + np.random.randn(len(arr)) * sd

def load_and_plot_macros():
    """Load and plot Rubik's cube macros"""

    plt.rcParams.update({'font.size': 12})
    _, ax = plt.subplots(figsize=(4,3))
    macro_names = [
        'Random',
        'Primitive',
        'Focused',
        'Expert',
    ]
    macros.generate_random_macro_set(14)
    macro_types = [
        macros.random,
        macros.primitive,
        macros.learned,
        macros.expert,
    ]
    marker_styles = [
        '^',
        'o',
        '+',
        'x',
    ]
    blue, orange, green, red, purple, brown, pink, gray, yellow, teal  = sns.color_palette('deep', n_colors=10)
    colors = [
        teal,   # random
        blue,   # primitive
        red,    # focused
        orange,   # expert
    ]
    for i, macro_type in enumerate(macro_types):
        if macro_names[i] == 'Random':
            for j in range(1,11):
                macros.generate_random_macro_set(j)
                lengths = [len(macro) for macro in macros.random.macros]
                effects = [len(cube.Cube().apply(swap_list=model).summarize_effects())
                           for model in macros.random.models]
                plt.scatter(lengths, effects, c=colors[i], marker='^',
                            facecolor='none', s=75, linewidths=1)
            macros.generate_random_macro_set(0)
        try:
            lengths = [len(macro) for macro in macro_type.macros]
        except AttributeError:
            lengths = [len(a) for a in macro_type.actions]
        effects = [len(cube.Cube().apply(swap_list=model).summarize_effects())
                   for model in macro_type.models]
        label = macro_names[i]# + ' Macros' if macro_names[i] != 'Primitive' else 'Base Actions'
        # if macro_names[i] == 'primitive':
        #     jx, jy = None, None
        # else:
        #     jx, jy = 0.05, 0.25
        jx, jy = None, None
        plt.scatter(jitter(lengths,jx), jitter(effects,jy), c=colors[i], marker=marker_styles[i],
                    facecolor='none', s=75, label=label, linewidths=1)
    plt.hlines(48, 0, 26, linestyles='dashed', linewidths=1, colors='k')
    handles, labels = ax.get_legend_handles_labels()
    # handles = [handles[0], handles[1], handles[2], handles[3]]
    # labels = [labels[0], labels[1], labels[2], labels[3]]
    leg = plt.legend(handles, labels)
    leg.set_draggable(True)
    plt.ylim([0,50])
    plt.xlim([0,26])
    plt.xticks(range(1,26,4))
    plt.xlabel('Macro-action length')
    plt.ylabel('Effect size')
    plt.tight_layout()
    plt.subplots_adjust(top = .96, bottom = .19, right = .95, left = 0.14,
        hspace = 0, wspace = 0)
    os.makedirs('results/plots/cube-buchner2018/', exist_ok=True)
    # plt.savefig('results/plots/cube-buchner2018/cube_entanglement.png')
    plt.show()

if __name__ == '__main__':
    load_and_plot_macros()
