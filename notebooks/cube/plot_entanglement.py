import matplotlib.pyplot as plt

from domains import cube
from domains.cube import macros

#%%
plt.rcParams.update({'font.size': 18})
fig, ax = plt.subplots(figsize=(8,6))
macro_names = ['primitive', 'expert', 'random', 'learned']
macros.generate_random_macro_set(14)
macro_types = [macros.primitive, macros.expert, macros.random, macros.learned]
marker_styles = ['o','x','^','+']
for i, macro_type in enumerate(macro_types):
    if macro_names[i] == 'random':
        for j in range(1,11):
            macros.generate_random_macro_set(j)
            lengths = [len(macro) for macro in macros.random.macros]
            effects = [len(cube.Cube().apply(swap_list=model).summarize_effects())
                       for model in macros.random.models]
            plt.scatter(lengths, effects, c='C{}'.format(i), marker='^', facecolor='none', s=150, linewidths=1)
        macros.generate_random_macro_set(0)
    try:
        lengths = [len(macro) for macro in macro_type.macros]
    except AttributeError:
        lengths = [len(a) for a in macro_type.actions]
    effects = [len(cube.Cube().apply(swap_list=model).summarize_effects()) for model in macro_type.models]
    label = macro_names[i]
    plt.scatter(lengths, effects, c='C{}'.format(i), marker=marker_styles[i], facecolor='none', s=150, label=label, linewidths=1)
plt.hlines(48, 0, 25, linestyles='dotted', linewidths=2)
handles, labels = ax.get_legend_handles_labels()
handles = [handles[0], handles[1],handles[3], handles[2]]
labels = [labels[0], labels[1],labels[3], labels[2]]
plt.legend(handles, labels)
plt.ylim([0,50])
plt.xlim([0,25])
plt.xticks(range(1,25,3))
plt.xlabel('Number of steps per macro-action')
plt.ylabel('Number of variables modified')
plt.savefig('results/plots/cube/cube_entanglement.png')
plt.show()
