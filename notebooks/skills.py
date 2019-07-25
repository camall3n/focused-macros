import random
import matplotlib.pyplot as plt

from cube import cube

def random_skill(length=3):
    formula = [random.choice(list(cube.Action.keys())) for _ in range(length)]
    return formula

def random_conjugate(prefix_length=1, body_length=1):
    """Generates a random skill of the form (X Y X')"""
    assert prefix_length > 0 and body_length>0, "Lengths ({}, {}) must be positive".format(prefix_length, body_length)
    prefix = random_skill(prefix_length)
    suffix = cube.inverse_formula(prefix)
    body = random_skill(body_length)
    formula = prefix + body + suffix
    return formula

def random_commutator(x_length=3, y_length=1):
    """Generates a random skill of the form (X Y X' Y')"""
    assert x_length > 0 and y_length>0, "Lengths ({}, {}) must be positive".format(x_length, y_length)
    X = random_skill(x_length)
    Xinv = cube.inverse_formula(X)
    Y = random_skill(y_length)
    Yinv = cube.inverse_formula(Y)
    formula = X + Y + Xinv + Yinv
    return formula

' '.join(random_conjugate(5,2))
' '.join(random_commutator(3,2))

orient_corners = "R B' R' U' B' U F U' B U R B R' F'".split()
swap_corners = "R U' R' D R U R' D'".split()
len(orient_corners)

len(swap_corners)
c = cube.Cube()
c.apply(orient_corners)
print('orient_corners:',len(c.summarize_effects()))

c = cube.Cube()
c.apply(swap_corners)
print('swap_corners:',len(c.summarize_effects()))

#%%
effects = []
lengths = []
for length in range(1,25):
    n_trials = 100
    effect = 0
    for trial in range(n_trials):
        d = cube.Cube()
        d.apply(random_skill(length))
        effect = len(d.summarize_effects())
        lengths.append(length)
        effects.append(effect)
# print('len={}:'.format(length), avg_effects)
lengths = [l-0.1 for l in lengths]
plt.scatter(lengths, effects, marker='o', label='Random')

#%%
effects = []
lengths = []
for prefix_length in range(1,9):
    for body_length in range(1,9):
        n_trials = 100
        effect = 0
        L = len(random_conjugate(prefix_length, body_length))
        for trial in range(n_trials):
            d = cube.Cube()
            d.apply(random_conjugate(prefix_length, body_length))
            effect = len(d.summarize_effects())
            lengths.append(L)
            effects.append(effect)
plt.scatter(lengths, effects, marker='^', label='Conjugates')

effects = []
lengths = []
for x_length in range(1,7):
    for y_length in range(1,7):
        n_trials = 100
        effect = 0
        L = len(random_commutator(x_length, y_length))
        for trial in range(n_trials):
            d = cube.Cube()
            d.apply(random_commutator(x_length, y_length))
            effect = len(d.summarize_effects())
            lengths.append(L)
            effects.append(effect)
lengths = [l+0.1 for l in lengths]

#%%
plt.scatter(lengths, effects, marker='d', label='Commutators')
plt.scatter(8, 9, marker='x', label='Swap-3-corners')
plt.scatter(14, 6, marker='+', label='Orient-2-corners')
plt.scatter(8, 6, marker='3', label='Swap-3-edges')
plt.scatter(12, 8, marker='4', label='Swap-4-edges')
plt.hlines(48, 0, 25, linestyles='dotted')
plt.legend(loc='lower right')
plt.title('Number of squares modified by skills')
plt.xlabel('Number of steps per skill')
plt.ylim([0,50])
plt.xlim([0,25])
plt.xticks(range(1,25))
plt.show()
