import matplotlib.pyplot as plt
from tqdm import tqdm

from cube import cube
from cube import skills as random_skills
from cube import options
from cube import formula

#%%
fig, ax = plt.subplots(figsize=(8,6))
option_names = ['primitive actions', 'expert skills', 'random skills', 'generated skills']
options.set_random_skill_seed(14)
option_types = [options.primitive, options.expert, options.random, options.generated]
marker_styles = ['o','x','^','+']
for i,option_type in enumerate(option_types):
    if option_names[i] == 'random skills':
        for j in range(1,10):
            options.set_random_skill_seed(j)
            lengths = [len(o) for o in options.random.options]
            effects = [len(cube.Cube().apply(swap_list=m).summarize_effects()) for m in options.random.models]
            label = 'random skills (multiple seeds)' if j == 1 else None
            plt.scatter(lengths, effects, marker='v', edgecolors='C{}'.format(i), facecolors='none', s=30, label=label, linewidths=1)
        options.set_random_skill_seed(0)
    try:
        lengths = [len(o) for o in option_type.options]
    except AttributeError:
        lengths = [len(a) for a in option_type.actions]
    effects = [len(cube.Cube().apply(swap_list=m).summarize_effects()) for m in option_type.models]
    label = option_names[i] if option_names[i] != 'random skills' else 'random skills (single seed)'
    plt.scatter(lengths, effects, c='C{}'.format(i), marker=marker_styles[i], facecolor='none', s=70, label=label, linewidths=1)
plt.hlines(48, 0, 25, linestyles='dotted')
handles, labels = ax.get_legend_handles_labels()
handles = [handles[0], handles[1],handles[3], handles[2], handles[4]]
labels = [labels[0], labels[1],labels[3], labels[2], labels[4]]
plt.legend(handles, labels)
plt.title('Skill Effect Size by Type')
plt.xlabel('Number of primitive actions per skill')
plt.ylabel('Number of state variables modified')
plt.ylim([0,50])
plt.xlim([0,25])
plt.xticks(range(1,25))
plt.savefig('results/plots/skill_effects_by_type.png')
plt.show()

# #%%
# effects = []
# lengths = []
# short_skills  = []
# for length in tqdm(range(1,25)):
#     n_trials = 1000
#     effect = 0
#     for trial in range(n_trials):
#         d = cube.Cube()
#         f = random_skills.random_skill(length)
#         d.apply(f)
#         effect = len(d.summarize_effects())
#         if effect < 20:
#             short_skills.append(f)
#             if f:
#                 tqdm.write('moves={}; effect={}; seq={}'.format(len(f),effect,' '.join(f)))
#         lengths.append(len(f))
#         effects.append(effect)
# lengths = [l-0.1 for l in lengths]
#
# plt.scatter(lengths, effects, marker='o', label='Random')
#
# #%%
# effects = []
# lengths = []
# for prefix_length in range(1,9):
#     for body_length in range(1,9):
#         n_trials = 100
#         effect = 0
#         for trial in range(n_trials):
#             d = cube.Cube()
#             f = random_skills.random_conjugate(prefix_length, body_length)
#             d.apply(f)
#             effect = len(d.summarize_effects())
#             lengths.append(len(f))
#             effects.append(effect)
# plt.scatter(lengths, effects, marker='^', label='Conjugates')
#
# effects = []
# lengths = []
# for x_length in range(1,7):
#     for y_length in range(1,7):
#         n_trials = 100
#         effect = 0
#         L = len(random_skills.random_commutator(x_length, y_length))
#         for trial in range(n_trials):
#             d = cube.Cube()
#             f = random_skills.random_commutator(x_length, y_length)
#             d.apply(f)
#             effect = len(d.summarize_effects())
#             lengths.append(len(f))
#             effects.append(effect)
# lengths = [l+0.1 for l in lengths]
#
# #%%
# plt.scatter(lengths, effects, marker='d', label='Commutators')
# plt.scatter([8, 8, 12, 14], [6, 9, 8, 6], marker='x', label='Expert skills')
# plt.hlines(48, 0, 25, linestyles='dotted')
# plt.legend(loc='lower right')
# plt.title('Number of squares modified by skills')
# plt.xlabel('Effective number of steps per skill')
# plt.ylim([0,50])
# plt.xlim([0,25])
# plt.xticks(range(1,25))
# plt.show()
