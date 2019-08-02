import copy
import random
from tqdm import tqdm

from cube import cube
from cube import formula
from cube import skills
from cube import expert
from cube import pattern

c = cube.Cube()
c.apply(pattern.scramble1)
c.render()

#%%
mods = c.summarize_effects()
steps = []
experiences = 0
tqdm.write('experiences:{}--steps:{}--errors:{}'.format(experiences, len(steps),len(mods)))

max_depth = 2
def option_sequences(depth, prefix=None):
    assert depth > 0, 'Depth must be > 0'
    options = [o for o in random.sample(expert.options, len(expert.options))]
    if depth==1:
        return [[o] if prefix==None else prefix+[o] for o in options]
    else:
        new_prefixes = [[o] if prefix==None else prefix+[o] for o in options]
        result = []
        for p in new_prefixes:
            result += option_sequences(depth-1, prefix=p)
        return result
option_seq = [None]*max_depth
for i in range(max_depth):
    option_seq[i] = option_sequences(i+1)
option_seq[1][:4]
mdl = {}
for op, m in zip(expert.options, expert.models):
    mdl[tuple(op)] = m

for _ in range(100):
    good_sequences = []
    improvements = []

    # Iterative deepening random search
    for depth in tqdm(range(max_depth)):
        for seq in tqdm(random.sample(option_seq[depth], len(option_seq[depth]))):
            c_copy = copy.deepcopy(c)
            for op in seq:
                c_copy.apply(swap_list=mdl[tuple(op)])
                experiences += 1
            resulting_mods = c_copy.summarize_effects()
            improvement = len(mods) - len(resulting_mods)
            if improvement > 0:
                good_sequences.append(seq)
                improvements.append(improvement)
                if depth >= 1:
                    break
        if improvements != []:
            break
        else:
            continue
    if improvements == []:
        break
    else:
        rankings = sorted(list(zip(improvements, good_sequences)), reverse=True)
        best_impr = rankings[0][0]
        best_seqs = [op for impr, op in rankings if impr == best_impr]
        seq = random.choice(best_seqs)
        for op in seq:
            c.apply(swap_list=mdl[tuple(op)])
        mods = c.summarize_effects()
        steps += seq
        tqdm.write('experiences:{}--steps:{}--errors:{}'.format(experiences, len(steps),len(mods)))
        c.render()

print()
print()
print()
print('Experiences:', experiences)
print('Steps:', len(steps))
print(steps)
