from collections import defaultdict
import unittest

class WidthAugmentedHeuristic:
    def __init__(self, n_variables, heuristic, R=set([]), precision=2):
        if precision < 2:
            raise ValueError('precision must be >= 2')
        if precision > 4:
            raise NotImplementedError('precision > 4 not yet supported')
        self.n_variables = n_variables
        self.heuristic = heuristic
        self.R = R
        self.precision = precision

        self.history = dict()
        self.history[1] = dict()
        self.history[2] = dict()
        self.history[3] = dict()
        for i in range(self.n_variables):
            self.history[1][i] = defaultdict(set)
            for j in range(i+1, self.n_variables):
                self.history[2][(i,j)] = defaultdict(set)
                for k in range(j+1, self.n_variables):
                    self.history[3][(i,j,k)] = defaultdict(set)

    def __call__(self, x, atoms=set([])):
        h = self.heuristic(x)
        if self.R is not None:
            r = len(atoms.intersection(self.R))
        else:
            r = len(atoms)
        w = self.get_width(x, (r, h))
        self.record(x, (r, h))
        return w, h

    def record(self, x, h):
        for i in range(self.n_variables):
            self.history[1][i][h].add(x[i])
            if self.precision > 2:
                for j in range(i+1, self.n_variables):
                    self.history[2][(i,j)][h].add((x[i],x[j]))
                    if self.precision > 3:
                        for k in range(j+1, self.n_variables):
                            self.history[3][(i,j,k)][h].add((x[i],x[j],x[k]))

    def get_width(self, x, h):
        # consider length 1 atoms
        for i in range(self.n_variables):
            if x[i] not in self.history[1][i][h]:
                return 1
        if self.precision == 2:
            return 2
        # consider length 2 atoms
        for i in range(self.n_variables):
            for j in range(i+1, self.n_variables):
                if (x[i], x[j]) not in self.history[2][(i,j)][h]:
                    return 2
        if self.precision == 3:
            return 3
        # consider length 4 atoms
        for i in range(self.n_variables):
            for j in range(i+1, self.n_variables):
                for k in range(j+1, self.n_variables):
                    if (x[i], x[j], x[k]) not in self.history[3][(i,j,k)][h]:
                        return 3
        if self.precision == 4:
            return 4
        raise NotImplementedError

class TestWidthAugmentedHeuristic(unittest.TestCase):
    def test_fixed_h(self):
        import numpy as np
        goal = np.array([4,4,4,4])
        heuristic = lambda *x: np.sum(goal-x != 0)
        h = WidthAugmentedHeuristic(4, heuristic, precision=4)
        def f(*x): return h(np.array([*x]))

        self.assertEqual(f(0,0,0,0), (1, 4))
        self.assertEqual(f(0,0,0,0), (4, 4))
        self.assertEqual(f(1,1,1,1), (1, 4))
        self.assertEqual(f(2,2,2,2), (1, 4))
        self.assertEqual(f(1,2,1,2), (2, 4))
        self.assertEqual(f(1,2,2,1), (2, 4))
        self.assertEqual(f(1,2,1,1), (3, 4))
        self.assertEqual(f(3,1,1,1), (1, 4))
        self.assertEqual(f(1,3,1,1), (1, 4))
        self.assertEqual(f(1,1,3,1), (1, 4))
        self.assertEqual(f(1,1,1,3), (1, 4))
        self.assertEqual(f(3,3,1,1), (2, 4))
        self.assertEqual(f(3,1,3,1), (2, 4))
        self.assertEqual(f(3,1,1,3), (2, 4))
        self.assertEqual(f(1,3,3,1), (2, 4))
        self.assertEqual(f(1,3,1,3), (2, 4))
        self.assertEqual(f(1,1,3,3), (2, 4))
        self.assertEqual(f(3,3,3,1), (3, 4))
        self.assertEqual(f(3,3,1,3), (3, 4))
        self.assertEqual(f(3,1,3,3), (3, 4))
        self.assertEqual(f(1,3,3,3), (3, 4))
        self.assertEqual(f(3,3,3,3), (4, 4))

    def test_variable_h(self):
        import numpy as np
        goal = np.array([1,2,3,4])
        heuristic = lambda *x: np.sum(goal-x != 0)
        h = WidthAugmentedHeuristic(4, heuristic, precision=4)
        def f(*x): return h(np.array([*x]))

        self.assertEqual(f(0,0,0,0), (1, 4))
        self.assertEqual(f(0,0,0,0), (4, 4))
        self.assertEqual(f(1,1,1,1), (1, 3))
        self.assertEqual(f(2,2,2,2), (1, 3))
        self.assertEqual(f(1,2,1,2), (1, 2))
        self.assertEqual(f(1,2,2,1), (1, 2))
        self.assertEqual(f(1,2,1,1), (2, 2))
        self.assertEqual(f(2,1,1,1), (1, 4))
        self.assertEqual(f(2,1,2,2), (1, 4))
        self.assertEqual(f(2,1,2,1), (2, 4))

if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2, exit=False)
