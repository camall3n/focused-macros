import copy
import random
import numpy as np

class NPuzzle:
    def __init__(self, n=8, start_blank=None):
        self.n = int(n)
        self.width = int(np.round(np.sqrt(n+1)))
        assert self.width**2-1 == self.n

        self.state = np.arange(self.n+1).reshape(self.width, self.width)
        self.blank_idx = (self.width-1, self.width-1)
        self.labels = list(range(1,self.n+1))+[0]

        self.sequence = []

        if start_blank is not None:
            while start_blank[0] < self.blank_idx[0]:
                self.transition(self.up())
            while start_blank[1] < self.blank_idx[1]:
                self.transition(self.left())
            assert self.blank_idx == start_blank

    def actions(self):
        directions = [self.up, self.down, self.left, self.right]
        return [d(self.blank_idx) for d in directions if d(self.blank_idx) is not None]

    def up(self, loc=None):
        if loc is None:
            loc = self.blank_idx
        row, col = loc
        row, col = row-1, col
        if row >= 0:
            return row, col
        return None
    def down(self, loc=None):
        if loc is None:
            loc = self.blank_idx
        row, col = loc
        row, col = row+1, col
        if row < self.width:
            return row, col
        return None
    def left(self, loc=None):
        if loc is None:
            loc = self.blank_idx
        row, col = loc
        row, col = row, col-1
        if col >= 0:
            return row, col
        return None
    def right(self, loc=None):
        if loc is None:
            loc = self.blank_idx
        row, col = loc
        row, col = row, col+1
        if col < self.width:
            return row, col
        return None

    def reset(self):
        self.state = np.arange(self.n+1).reshape(self.width, self.width)
        self.blank_idx = (self.width-1, self.width-1)
        return self

    def scramble(self, seed=None):
        if seed is not None:
            py_st = random.getstate()
            np_st = np.random.get_state()
            random.seed(seed)
            np.random.seed(seed)
        # need both even and odd n_steps for blank to reach every space
        n_steps = random.choice([self.n**2,self.n**2+1])
        for i in range(n_steps):
            a = random.choice(self.actions())
            self.transition(a)
            self.sequence.append(a)
        if seed is not None:
            random.setstate(py_st)
            np.random.set_state(np_st)
        return self

    def transition(self, tile_idx):
        t_row, t_col = tile_idx
        b_row, b_col = self.blank_idx
        # Within bounds
        assert t_row >= 0 and t_row < self.width
        assert t_col >= 0 and t_col < self.width
        # Adjacent tile
        assert sum([np.abs(b_row-t_row), np.abs(b_col-t_col)]) == 1
        self._unchecked_transition(tile_idx, self.blank_idx)
        self.blank_idx = tile_idx
        return self

    def _unchecked_transition(self, tile_idx, blank_idx):
        self.state[tile_idx], self.state[blank_idx] = self.state[blank_idx], self.state[tile_idx]

    def __repr__(self):
        return '{}-Puzzle(\n{})'.format(self.n, np.asarray(list(map(lambda x: self.labels[x], self.state.flatten()))).reshape(self.width, self.width))

    def __hash__(self):
        return hash(repr(self))

    def __eq__(self, another):
        assert self.n == another.n, 'Instances must have same n_var'
        assert self.width == another.width, 'Instances must have same n_values'
        return np.all(self.state == another.state) and np.all(self.blank_idx == another.blank_idx)

    def __ne__(self, another):
        return not self.__eq__(another)

    def apply_macro(self, sequence=None, model=None):
        assert sequence is not None or model is not None
        if model is not None:
            swap_list, starting_blank_idx = model
            if self.blank_idx == starting_blank_idx:
                old_state = self.state.flatten()
                new_state = self.state.flatten()
                for (src_idx, dst_idx) in swap_list:
                    new_state[dst_idx] = old_state[src_idx]
                self.state = new_state.reshape(self.width, self.width)
                self.blank_idx = tuple(np.argwhere(self.state==self.n)[0])
            else:# starting blanks don't line up
                pass # cannot execute macro
        elif sequence is not None:
            for move in sequence:
                self.transition(move)
        if sequence:
            self.sequence += sequence
        return self

    def summarize_effects(self, baseline=None):
        if baseline is None:
            baseline = copy.deepcopy(self).reset()
        src_indices = np.arange(self.n+1)
        src_tiles = baseline.state.flatten()
        src_dict = dict(zip(src_tiles, src_indices))
        dst_indices = [src_dict[tile] for tile in self.state.flatten()]
        swap_list = list(zip(dst_indices, src_indices))
        swap_list = tuple([swap for swap in swap_list if swap[0] != swap[1]])
        return swap_list, baseline.blank_idx

def test_default_baseline():
    puz = NPuzzle(15)
    puz.scramble()

    baseline = copy.deepcopy(puz).reset()
    assert baseline == NPuzzle(15)
    assert puz != baseline

    baseline.apply_macro(sequence=puz.sequence)
    assert baseline == puz

    baseline.reset()
    assert baseline != puz

    baseline.apply_macro(model=puz.summarize_effects())
    assert baseline == puz

def test_custom_baseline():
    puz = NPuzzle(15)
    puz.transition(puz.left())
    puz.transition(puz.left())
    puz.transition(puz.left())
    model = puz.summarize_effects()
    assert model == (((15, 12), (12, 13), (13, 14), (14, 15)), (3, 3))

    baseline = NPuzzle(15)
    baseline.scramble(seed=40)# Seed 40 has blank in lower right corner
    assert baseline.blank_idx == (3,3)

    newpuz = copy.deepcopy(baseline)
    newpuz.transition(newpuz.left())
    newpuz.transition(newpuz.left())
    newpuz.transition(newpuz.left())
    assert newpuz.blank_idx == puz.blank_idx

    new_model = newpuz.summarize_effects(baseline=baseline)
    assert new_model == model
    assert NPuzzle(15).apply_macro(model=model) == puz
    assert copy.deepcopy(baseline).apply_macro(model=new_model) == newpuz
    assert puz != newpuz

if __name__ == '__main__':
    test_default_baseline()
    test_custom_baseline()
    print('All tests passed.')
