import random
import matplotlib.pyplot as plt
from tqdm import tqdm

from cube import cube
from cube import formula
from cube import skills
from cube import expert

def random_action_skill(length=3):
    f = [random.choice(list(cube.Action.keys())) for _ in range(length)]
    f = formula.simplify(f)
    return f

def random_option_skill(length=3):
    idx_sequence = [random.choice(range(len(expert.options))) for _ in range(length)]
    o_seq = [expert.options[idx] for idx in idx_sequence]
    m_seq = [expert.models[idx] for idx in idx_sequence]
    return o_seq, m_seq

def main(count_type=count_type):
    assert count_type in ['decisions', 'actions']

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
        n_actions = 0
        for trial in range(n_trials):
            d = cube.Cube()
            o_seq, m_seq = random_option_skill(length)
            for o,m in zip(o_seq, m_seq):
                n_actions += len(o)
                d.apply(swap_list=m)
            effect += len(d.summarize_effects())
        if count_type == 'actions':
            lengths.append(n_actions/n_trials)# count actions
        else:
            lengths.append(length)# count options
        effects.append(effect/n_trials)
    lengths = [l for l in lengths]
    plt.scatter(lengths, effects, marker='o', label='Options')

    x_max = 40 if count_type == 'decisions' else 500
    plt.hlines(48, 0, x_max, linestyles='dotted')
    plt.legend(loc='lower right')
    plt.title('Average number of squares modified by sequence')
    plt.xlabel('Number of {} per sequence'.format(count_type))
    plt.ylim([0,50])
    if count_type == 'actions':
        plt.xlim([0,500])
    plt.show()

if __name__ == '__main__':
    main(count_type='decisions')
