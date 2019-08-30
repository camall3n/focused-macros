import numpy as np
import matplotlib.pyplot as plt

from gridworlds.domain.gridworld.gridworld import GridWorld, DiagGridWorld
from gridworlds.domain.gridworld.objects.depot import Depot

from . import cgrid

class SubgoalGridWorld(DiagGridWorld):
    def __init__(self, rows=10, cols=10, goal=None, idx=None, penalty=None, obstacles=0):
        super().__init__(rows, cols)
        if goal is None or idx is None:
            self.set_goal(*self.random_goal())
        else:
            self.set_goal(idx, goal)
        self.penalty = penalty
        for o in range(obstacles):
            self.add_wall()
        self.reset()

    def random_goal(self):
        idx = np.random.randint(2)
        N = self._rows if idx == 0 else self._cols
        goal = np.random.randint(0, N)
        return idx, goal

    def random_action(self):
        return np.random.randint(self.n_actions)

    def discrete2continuous(self, a):
        assert np.all(a >= 0) and np.all(a < self.n_actions)
        ax = np.zeros(len(a))
        ay = np.zeros(len(a))
        ax += (a % 2 - 0.5) * (a < 2)
        ay += -1*(a % 2 - 0.5) * (a >= 2)
        return 0.1*np.stack([ax, ay]).transpose()

    def set_goal(self, idx, goal):
        if self.goals:
            self.goals = []
        goal_size = self._cols if idx == 0 else self._rows
        for i in range(goal_size):
            position = i*np.ones(2)
            position[idx] = goal
            d = Depot(position, color='red')
            self.goals.append(d)
        self.idx, self.goal = idx, goal

    def reset(self):
        self.reset_agent()
        self.s0 = self.get_state()

    def step(self, a):
        s = self.get_state()
        sp, _, done = super().step(a)
        mask = np.arange(2) != self.idx
        r = -1
        if self.penalty == 'stepwise':
            r -= np.sum(np.abs(sp[mask] - s[mask]))
        elif self.penalty == 'start':
            r -= np.sum(np.abs(sp[mask] - self.s0[mask]))
        return sp, r, done

class SubgoalCGridWorld(cgrid.CGridWorld):
    def __init__(self, goal=None, idx=None, tol=0.1, discrete_actions=False, penalty=None):
        super().__init__(discrete_actions=discrete_actions)
        if goal is None or idx is None:
            self.set_goal(*self.random_goal())
        self.tol = tol
        self.reset()

    def random_goal(self):
        idx = np.random.randint(0, self.n_states)
        goal = np.random.uniform(0, 1)
        return idx, goal

    def set_goal(self, idx, goal):
        self.idx, self.goal = idx, goal

    def reset(self):
        self.state = self.random_state()
        at = lambda x, y: np.abs(x-y) < self.tol
        while (self.goal is not None) and at(self.state[self.idx], self.goal):
            self.state = self.random_state()
        self.s0 = self.get_state()
        assert not at(self.state[self.idx], self.goal)

    def step(self, a):
        s = self.get_state()
        sp, _, _ = super().step(a)
        mask = np.arange(len(sp)) != self.idx
        r = -1
        if self.penalty == 'stepwise':
            r -= np.sum(np.abs(sp[mask] - s[mask]))
        elif self.penalty == 'start':
            r -= np.sum(np.abs(sp[mask] - self.s0[mask]))
        done = ( np.abs(sp[self.idx] - self.goal) < self.tol )
        return sp, r, done

    def plot(self, ax=None):
        ax = super().plot(ax)
        line_positions = [self.goal-self.tol, self.goal+self.tol]
        if self.idx % 2 == 0:
            ax.vlines(line_positions, 0, 1, 'r')
        else:
            ax.hlines(line_positions, 0, 1, 'r')
        return ax

def main():
    env = SubgoalGridWorld()
    env.plot()
#%%
    a = 2
    env.step(a)
    env.discrete2continuous(a)
#%%
    #0 LEFT
    #1 RIGHT
    #2 UP
    #3 DOWN
    cgrid.run_agent(env, n_trials=5, n_samples=100, video=True)

if __name__ == '__main__':
    main()
