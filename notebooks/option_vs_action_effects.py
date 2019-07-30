import random
import matplotlib.pyplot as plt
from tqdm import tqdm

from cube import cube
from cube import skills

algs = [
    skills.swap_3_edges_face,
    skills.swap_3_edges_mid,
    skills.swap_3_corners,
    skills.orient_2_corners,
]

options = []
for alg in algs:
    variations = cube.formula_collection(alg)
    options.extend(variations)

models = []
for o in options:
    m = cube.Cube().apply(o).summarize_effects()
    models.append(m)

def random_action_skill(length=3):
    formula = [random.choice(list(cube.Action.keys())) for _ in range(length)]
    formula = cube.simplify_formula(formula)
    return formula

def random_option_skill(length=3):
    idx_sequence = [random.choice(range(len(options))) for _ in range(length)]
    o_seq = [options[idx] for idx in idx_sequence]
    m_seq = [models[idx] for idx in idx_sequence]
    return o_seq, m_seq

def main():
    effects = []
    lengths = []
    for length in tqdm(range(1,41)):
        n_trials = 100
        effect = 0
        for trial in range(n_trials):
            d = cube.Cube()
            f = random_action_skill(length)
            d.apply(f)
            effect += len(d.summarize_effects())
        lengths.append(length)
        effects.append(effect/n_trials)
    lengths = [l-0.1 for l in lengths]
    plt.scatter(lengths, effects, marker='^', label='Primitive Actions')

    effects = []
    lengths = []
    for length in tqdm(range(1,41)):
        n_trials = 50
        effect = 0
        for trial in range(n_trials):
            d = cube.Cube()
            o_seq, m_seq = random_option_skill(length)
            for o,m in zip(o_seq, m_seq):
                d.apply(swap_list=m)
            effect += len(d.summarize_effects())
        lengths.append(length)
        effects.append(effect/n_trials)
    lengths = [l for l in lengths]
    plt.scatter(lengths, effects, marker='o', label='Options')

    plt.hlines(48, 0, 40, linestyles='dotted')
    plt.legend(loc='lower right')
    plt.title('Average number of squares modified by sequence')
    plt.xlabel('Effective number of steps per sequence')
    plt.ylim([0,50])
    plt.show()

if __name__ == '__main__':
    main()
