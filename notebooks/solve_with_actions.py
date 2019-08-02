import copy
import random
from tqdm import tqdm

from cube import cube
from cube import formula
from cube import skills
from cube import pattern

c = cube.Cube()
c.apply(pattern.scramble1)
c.render()

#%%
mods = c.summarize_effects()
steps = []
experiences = 0
tqdm.write('experiences:{}--steps:{}--errors:{}'.format(experiences, len(steps),len(mods)))

max_depth = 5
def action_sequences(depth, prefix=None):
    assert depth > 0, 'Depth must be > 0'
    actions = [a for a in random.sample(cube.actions, len(cube.actions))]
    if depth==1:
        return [[a] if prefix==None else prefix+[a] for a in actions]
    else:
        new_prefixes = [[a] if prefix==None else prefix+[a] for a in actions]
        result = []
        for p in new_prefixes:
            result += action_sequences(depth-1, prefix=p)
        return result
action_seq = [None]*max_depth
for i in range(max_depth):
    action_seq[i] = action_sequences(i+1)

for _ in range(100):
    good_sequences = []
    improvements = []

    # Iterative deepening random search
    for depth in tqdm(range(max_depth)):
        for seq in tqdm(random.sample(action_seq[depth], len(action_seq[depth]))):
            c_copy = copy.deepcopy(c)
            c_copy.apply(seq)
            experiences += len(seq)
            resulting_mods = c_copy.summarize_effects()
            improvement = len(mods) - len(resulting_mods)
            if improvement > 0:
                good_sequences.append(seq)
                improvements.append(improvement)
                if depth >= 3:
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
        c.apply(seq)
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
