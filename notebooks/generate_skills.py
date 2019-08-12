import copy
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from collections import namedtuple

from cube import cube
from cube import formula
from cube import skills
from cube import options
from cube import pattern

c = cube.Cube()
c.render()

Skill = namedtuple('Skill',['seq','mdl'])

def combine_skills(skills, depth, prefix=None):
    assert depth > 0, 'Depth must be > 0'
    # skills = [s for s in random.sample(skills, len(skills))]
    if depth==1:
        seqs = [s.seq if prefix==None else formula.simplify(prefix.seq+s.seq) for s in skills]
        mdls = [s.mdl if prefix==None else cube.combine_swaps(prefix.mdl, s.mdl) for s in skills]
        return [Skill(s,tuple(m)) for (s, m) in zip(seqs, mdls) if s !=[] and m != []]
    else:
        new_prefix_seqs = [s.seq if prefix==None else formula.simplify(prefix.seq+s.seq) for s in skills]
        new_prefix_mdls = [s.mdl if prefix==None else cube.combine_swaps(prefix.mdl,s.mdl) for s in skills]
        new_prefixes = [Skill(s,tuple(m)) for (s,m) in list(zip(new_prefix_seqs, new_prefix_mdls)) if s !=[] and m != []]
        combo_skills = [combine_skills(skills, depth-1, prefix=p) for p in tqdm(new_prefixes)]
        result = [skill for entry in combo_skills for skill in entry]
        return result

skills = [Skill([a],tuple(cube.Cube().apply([a]).summarize_effects())) for a in cube.actions]
cached_effects = set([s.mdl for s in skills])
min_effect_size = 17
depth = 2
#%%
combos = combine_skills(skills, depth)
n_changes = [len(skill.mdl) for skill in combos]
rankings = sorted([(n, skill) for (n, skill) in list(zip(n_changes, combos)) if n > 0 and n <= min_effect_size])
for n, skill in rankings:
    if skill.mdl not in cached_effects:
        cached_effects.add(skill.mdl)
        skills.append(skill)
    min_effect_size = min(n, min_effect_size)

#%%
for l in range(20):
    print('{}:'.format(l), len([s for s in skills if len(s.seq) == l]))
print('total:',len(skills))
sorted([tuple(s.seq) for s in skills])

for l in range(17):
    print(l,len([s for s in skills if len(s.mdl) == 8 and len(s.seq)==l]))

sorted([s for s in skills if len(s.mdl) == 8 and len(s.seq)==16])

y = [len(s.mdl) for s in skills]
plt.scatter(x,y)

#%%
from cube import cube
from cube import skills as random_skills

fig, ax = plt.subplots(figsize=(10,6))
effects = []
lengths = []
short_skills  = []
for length in tqdm(range(1,25)):
    n_trials = 100
    effect = 0
    for trial in range(n_trials):
        d = cube.Cube()
        f = random_skills.random_skill(length)
        d.apply(f)
        effect = len(d.summarize_effects())
        if effect < 20:
            short_skills.append(f)
        lengths.append(len(f))
        effects.append(effect)
lengths = [l-0.1 for l in lengths]

plt.scatter(lengths, effects, marker='o', label='Random')

effects = []
lengths = []
for prefix_length in tqdm(range(1,9)):
    for body_length in range(1,9):
        n_trials = 100
        effect = 0
        for trial in range(n_trials):
            d = cube.Cube()
            f = random_skills.random_conjugate(prefix_length, body_length)
            d.apply(f)
            effect = len(d.summarize_effects())
            lengths.append(len(f))
            effects.append(effect)
plt.scatter(lengths, effects, marker='^', label='Conjugates')

effects = []
lengths = []
for x_length in tqdm(range(1,7)):
    for y_length in range(1,7):
        n_trials = 100
        effect = 0
        L = len(random_skills.random_commutator(x_length, y_length))
        for trial in range(n_trials):
            d = cube.Cube()
            f = random_skills.random_commutator(x_length, y_length)
            d.apply(f)
            effect = len(d.summarize_effects())
            lengths.append(len(f))
            effects.append(effect)
lengths = [l+0.1 for l in lengths]
plt.scatter(lengths, effects, marker='d', label='Commutators')

x = [len(s.seq)+0.2 for s in skills]
y = [len(s.mdl) for s in skills]
plt.scatter(x,y, marker='o', facecolors='none', edgecolors='C4', label='Generated')

from cube.options import expert
x = [len(o) for o in expert.options]
y = [len(m) for m in expert.models]
plt.scatter(x, y, marker='x', c='C3', label='Expert skills')

plt.hlines(48, 0, 25, linestyles='dotted')
plt.legend(loc='lower left')
plt.title('Number of squares modified by skills')
plt.xlabel('Effective number of steps per skill')
plt.ylim([0,50])
plt.xlim([0,25])
plt.xticks(range(1,25))
plt.show()
