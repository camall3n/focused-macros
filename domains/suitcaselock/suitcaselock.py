import copy

import gmpy
import numpy as np


def reduce_mod2(A):
    """Reduce a square binary matrix to reduced row echelon form (modulo 2)"""
    n = len(A)
    for row in range(n):
        A = sorted(map(tuple, list(A)), reverse=True)
        row_contents = A[row]
        if np.all(row_contents == 0):
            continue
        first_nonzero_idx = (np.asarray(row_contents) != 0).argmax(axis=0)
        for other_row in range(n):
            if other_row != row and A[other_row][first_nonzero_idx] == 1:
                A[other_row] = tuple((np.array(A[other_row]) + np.array(row_contents))%2)
    return np.stack(A)


def rank_mod2(A):
    """Compute the rank of a square binary matrix (modulo 2)"""
    A = reduce_mod2(A)
    return np.linalg.matrix_rank(A)


class SuitcaseLock:
    """SuitcaseLock simulator"""
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

    def __len__(self):
        return self.n_vars

    def __getitem__(self, key):
        return self.state[key]

    def __iter__(self):
        return iter(self.state)

    def states(self):
        """Iterator for the possible states of the SuitcaseLock"""
        for i in range(self.n_values**self.n_vars):
            digits = gmpy.digits(i,self.n_values).zfill(self.n_vars)
            lock = copy.deepcopy(self)
            lock.state = np.asarray(list(map(int, digits)))
            yield lock

    def actions(self):
        """Return the list of actions for the SuitcaseLock"""
        # When a SuitcaseLock instance is created, this function generates a new
        # action set such that there are self.n_vars increment actions that are
        # capable of transitioning from any state to any other state, and the
        # same number of matching decrement actions.
        #
        # We repeatedly generate action sets with the desired mean entanglement
        # until we find one with full rank (modulo 2). The resulting action
        # sets are therefore different for each random seed, except when
        # k==1, where we always use the identity matrix I, and when k==(N-1),
        # where we use (1 - I) with an extra 1 added to the first diagonal element
        # to break symmetry. The decrement actions are always the negation of the
        # increment actions.
        if self._actions is None:
            N = self.n_vars
            k = self.entanglement
            if k == 1:
                up_actions = np.eye(N)
            elif k == N-1:
                up_actions = np.ones((N,N))-np.eye(N)
                # Have to break symmetry to ensure matrix is full rank
                up_actions[0,0] = 1
            else:
                for _ in range(10000):
                    up_actions = np.random.choice([0,1], size=(N,N), p=[(1-k/N), k/N])
                    rank = rank_mod2(up_actions)
                    if rank == N:
                        break
                if rank < N:
                    raise RuntimeError('Failed to find full-rank action matrix for N={}, k={}. '
                                       'This may just be bad luck with the random number '
                                       'generator. Try increasing the number of attempts in'
                                       'SuitcaseLock.actions().'.format(N, k))
            down_actions = -1 * up_actions
            actions = np.concatenate((up_actions, down_actions))
            self._actions = list(actions.astype(int))
        return self._actions

    def reset(self):
        """Reset the SuitcaseLock to the all-zeros state"""
        self.state = np.zeros(self.n_vars, dtype=np.int)
        return self

    def scramble(self, seed=None):
        """Scramble the SuitcaseLock with randomly selected actions

        Specify a random seed for repeatable results.
        """
        if seed is not None:
            np_st = np.random.get_state()
            np.random.seed(seed)
        self.state = np.random.randint(0, self.n_values, size=(self.n_vars,))
        if seed is not None:
            np.random.set_state(np_st)
        return self

    def transition(self, action):
        """Transform the SuitcaseLock with a single action

        Args:
            action (numpy.ndarray): A difference vector compatible with the SuitcaseLock shape
        """
        if not isinstance(action, np.ndarray):
            raise TypeError('Action must be of type numpy.ndarray')
        if action.shape != self.state.shape:
            message = 'Action shape {} incompatible with lock shape {}'
            raise ValueError(message.format(action.shape, self.state.shape))
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
        """Apply a sequence of actions or a difference vector to transform the SuitcaseLock"""
        assert sequence is not None or diff is not None
        if diff is not None:
            self.transition(diff)
        elif sequence is not None:
            for move in sequence:
                self.transition(move)
        return self

    def summarize_effects(self, baseline=None):
        """Summarize the changes in the SuitcaseLock relative to a baseline SuitcaseLock

        The default behavior compares the current SuitcaseLock against the all-zeros SuitcaseLock.

        Returns:
            (numpy.ndarray) A difference vector representing the net change to each state variable
        """
        if baseline is None:
            baseline = copy.deepcopy(self).reset()
        diff = (self.state - baseline.state) % self.n_values
        return diff


def test_binary_matrix_ops():
    """Test functionality of reduce_mod2 and rank_mod2"""
    # Identity matrix is already reduced
    A = np.eye(7)
    assert np.all(reduce_mod2(A) == A)
    assert rank_mod2(A) == 7

    # Matrix of all ones
    A = np.ones((6,6), dtype=int)
    B = np.asarray([
        [1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]])
    assert np.all(reduce_mod2(A) == B)
    assert rank_mod2(A) == 1

    # 6x6 matrix with 5 linearly independent rows (mod 2)
    A = np.asarray([
        [1, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 1, 1],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]])
    B = np.asarray([
        [1, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0]])
    assert np.all(reduce_mod2(A) == B)
    assert rank_mod2(A) == 5

    # Matrix with even number of 1s in each row
    A = np.asarray([
        [0, 0, 1, 1, 1, 1],
        [1, 1, 0, 0, 1, 1],
        [0, 1, 0, 0, 1, 0],
        [1, 0, 0, 1, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 1, 0, 1, 0, 0]])
    B = np.asarray([
        [1, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 1],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0]])
    assert np.all(reduce_mod2(A) == B)
    assert rank_mod2(A) == 5

    # 10x10 matrix with 9 linearly independent rows (mod 2)
    A = np.array([
        [0, 1, 1, 0, 0, 1, 1, 1, 0, 1],
        [0, 0, 1, 1, 0, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 1, 0, 0, 0, 1, 1],
        [1, 0, 0, 1, 1, 1, 0, 0, 0, 0],
        [1, 0, 1, 1, 1, 1, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 1, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 0, 1, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 0, 0, 0, 0, 1]])
    B = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    assert np.all(reduce_mod2(A) == B)
    assert rank_mod2(A) == 9

def test_suitcaselock():
    """Test core SuitcaseLock functionality"""
    lock1 = SuitcaseLock(entanglement=2).scramble()

    lock2 = copy.deepcopy(lock1)
    actions = lock2.actions()
    seq_length = 20
    idx = np.random.choice(len(actions), seq_length)
    seq = list(np.stack(actions)[idx])
    lock2.apply_macro(sequence=seq)
    diff = lock2.summarize_effects(baseline=lock1)

    lock1.apply_macro(diff=diff)
    assert lock1 == lock2

    lock20 = SuitcaseLock(n_vars=20, n_values=10, entanglement=8)
    actions = lock20.actions()
    action_matrix = np.stack(actions[0:len(actions)//2])
    assert np.linalg.matrix_rank(action_matrix) == 20

def test():
    """Test all SuitcaseLock functionality"""
    test_binary_matrix_ops()
    test_suitcaselock()
    print('All tests passed.')


if __name__ == '__main__':
    test()
