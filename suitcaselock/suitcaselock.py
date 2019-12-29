import copy
import random
import numpy as np
from tqdm import tqdm

class SuitcaseLock:
    def __init__(self, n_vars=4, n_values=10, max_vars_per_action=1):
        assert n_vars > 0, 'n_vars must be > 0'
        assert n_values > 0, 'n_values must be > 0'
        assert max_vars_per_action <= n_vars, 'max_vars_per_action must be <= n_vars'
        self.n_vars = n_vars
        self.n_values = n_values
        self.max_vars_per_action = max_vars_per_action
        self.state = np.zeros(n_vars, dtype=np.int)

    def actions(self):
        k = -self.max_vars_per_action
        up_actions = np.tri(self.n_vars) - np.tri(self.n_vars, k=k)
        down_actions = -1 * up_actions
        actions = np.concatenate((up_actions, down_actions))
        return list(actions.astype(int))

    def reset(self):
        self.state = np.zeros(self.n_vars, dtype=np.int)
        return self

    def scramble(self, seed=None):
        if seed is not None:
            st = np.random.get_state()
            np.random.seed(seed)
        self.state = np.random.randint(0, self.n_values, size=(self.n_vars,))
        if seed is not None:
            np.random.set_state(st)
        return self

    def transition(self, action):
        assert len(action) == self.n_vars
        assert np.sum(np.abs(action)) <= self.max_vars_per_action
        self._unchecked_transition(action)
        return self

    def _unchecked_transition(self, action):
        self.state = (self.state + action) % self.n_values

    def __repr__(self):
        return 'SuitcaseLock({})'.format(self.state)

    def __hash__(self):
        return hash(repr(self))

    def __eq__(self, another):
        assert self.n_vars == another.n_vars, 'Instances must have same n_var'
        assert self.n_values == another.n_values, 'Instances must have same n_values'
        return np.all(self.state == another.state)

    def __ne__(self, another):
        return not self.__eq__(another)

    def apply_macro(self, sequence=None, diff=None):
        assert sequence is not None or diff is not None
        if diff is not None:
            self._unchecked_transition(diff)
        elif sequence is not None:
            for move in sequence:
                self.transition(move)
        return self

    def summarize_effects(self, baseline=None):
        if baseline is None:
            baseline = copy.deepcopy(self).reset()
        diff = (self.state - baseline.state) % self.n_values
        return diff

def test():
    lock1 = SuitcaseLock(max_vars_per_action=2).scramble()

    lock2 = copy.deepcopy(lock1)
    actions = lock2.actions()
    seq_length = 20
    idx = np.random.choice(len(actions),seq_length)
    seq = list(np.stack(actions)[idx])
    lock2.apply_macro(sequence=seq)
    diff = lock2.summarize_effects(baseline=lock1)

    lock1.apply_macro(diff=diff)
    assert lock1 == lock2

    lock9 = SuitcaseLock(n_vars=9, n_values=4, max_vars_per_action=9)
    assert np.mean(np.abs(np.sum(np.stack(lock9.actions()),axis=1))) == 5

lock20 = SuitcaseLock(n_vars=20, n_values=10, max_vars_per_action=8)
actions = lock20.actions()
T = np.stack(actions[0:len(actions)//2])
np.linalg.matrix_rank(T)
