from tqdm import tqdm

from cube import cube, skills

#%%
effects = []
lengths = []
short_skills  = []
for length in tqdm(range(1,25)):
    n_trials = 1000
    effect = 0
    for trial in range(n_trials):
        d = cube.Cube()
        f = skills.random_skill(length)
        d.apply(f)
        effect = len(d.summarize_effects())
        if effect < 20:
            short_skills.append(f)
            if f:
                tqdm.write('moves={}; effect={}; seq={}'.format(len(f),effect,' '.join(f)))
        lengths.append(len(f))
        effects.append(effect)
lengths = [l-0.1 for l in lengths]

plt.scatter(lengths, effects, marker='o', label='Random')

#%%
effects = []
lengths = []
for prefix_length in range(1,9):
    for body_length in range(1,9):
        n_trials = 100
        effect = 0
        for trial in range(n_trials):
            d = cube.Cube()
            f = skills.random_conjugate(prefix_length, body_length)
            d.apply(f)
            effect = len(d.summarize_effects())
            lengths.append(len(f))
            effects.append(effect)
plt.scatter(lengths, effects, marker='^', label='Conjugates')

effects = []
lengths = []
for x_length in range(1,7):
    for y_length in range(1,7):
        n_trials = 100
        effect = 0
        L = len(skills.random_commutator(x_length, y_length))
        for trial in range(n_trials):
            d = cube.Cube()
            f = skills.random_commutator(x_length, y_length)
            d.apply(f)
            effect = len(d.summarize_effects())
            lengths.append(len(f))
            effects.append(effect)
lengths = [l+0.1 for l in lengths]

#%%
plt.scatter(lengths, effects, marker='d', label='Commutators')
plt.scatter([8, 8, 12, 14], [6, 9, 8, 6], marker='x', label='Expert skills')
plt.hlines(48, 0, 25, linestyles='dotted')
plt.legend(loc='lower right')
plt.title('Number of squares modified by skills')
plt.xlabel('Effective number of steps per skill')
plt.ylim([0,50])
plt.xlim([0,25])
plt.xticks(range(1,25))
plt.show()
