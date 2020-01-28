import copy
import gmpy
import random
import numpy as np
from tqdm import tqdm

from notebooks.rrank import rrank

class SuitcaseLock:
    def __init__(self, n_vars=4, n_values=10, entanglement=1):
        assert n_vars > 0, 'n_vars must be > 0'
        assert n_values > 0, 'n_values must be > 0'
        assert entanglement < n_vars, 'entanglement must be < n_vars'
        self.n_vars = n_vars
        self.n_values = n_values
        self.entanglement = entanglement
        self.state = np.zeros(n_vars, dtype=np.int)
        self._actions = None
        self.actions()

    def states(self):
        for i in range(self.n_values**self.n_vars):
            digits = gmpy.digits(i,self.n_values).zfill(self.n_vars)
            lock = copy.deepcopy(self)
            lock.state = np.asarray(list(map(int, digits)))
            yield lock

    def actions(self):
        if self._actions is None:
            N = self.n_vars
            k = self.entanglement
            if k == 1:
                up_actions = np.eye(N)
            elif k == N-1:
                up_actions = np.ones((N,N))-np.eye(N)
                # Have to break symmetry to ensure matrix is full rank modulo n_values
                up_actions[0,0] = 1
            else:
                for i in range(10000):
                    up_actions = np.random.choice([0,1], size=(N,N), p=[(1-k/N), k/N])
                    rank = rrank(up_actions)
                    if rank == N:
                        break
                if rank < N:
                    raise RuntimeError('Failed to find full-rank action matrix for N={}, k={}. This may just be bad luck with the random number generator. Try increasing the number of attempts in SuitcaseLock.actions().'.format(N,k))
            down_actions = -1 * up_actions
            actions = np.concatenate((up_actions, down_actions))
            self._actions = list(actions.astype(int))
        return self._actions

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
        assert np.sum(np.abs(action)) <= self.entanglement
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
    lock1 = SuitcaseLock(entanglement=2).scramble()

    lock2 = copy.deepcopy(lock1)
    actions = lock2.actions()
    seq_length = 20
    idx = np.random.choice(len(actions),seq_length)
    seq = list(np.stack(actions)[idx])
    lock2.apply_macro(sequence=seq)
    diff = lock2.summarize_effects(baseline=lock1)

    lock1.apply_macro(diff=diff)
    assert lock1 == lock2

    lock9 = SuitcaseLock(n_vars=9, n_values=4, entanglement=9)
    assert np.mean(np.abs(np.sum(np.stack(lock9.actions()),axis=1))) == 5

    lock20 = SuitcaseLock(n_vars=20, n_values=10, entanglement=8)
    actions = lock20.actions()
    T = np.stack(actions[0:len(actions)//2])
    np.linalg.matrix_rank(T)
