import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time

class CGridWorld:
    '''Continuous gridworld domain
    '''
    def __init__(self, n_dims=2, discrete_actions=False):
        if discrete_actions:
            self.n_actions = 9
        else:
            self.n_actions = n_dims
        self.discrete_actions = discrete_actions
        self.n_states = n_dims
        self.state = self.random_state()

    def reset(self):
        self.state = self.random_state()

    def random_state(self):
        return np.random.uniform(0, 1, size=self.n_states)

    def random_action(self):
        if self.discrete_actions:
            a = np.random.randint(self.n_actions)
        else:
            a = np.random.uniform(-0.1,0.1, size=self.n_actions)
        return a

    def discrete2continuous(self, a):
        assert np.all(a >= 0) and np.all(a < self.n_actions)
        ax = a % 3 - 1
        ay = -1*(a // 3 - 1)
        return 0.1*np.stack([ax, ay]).transpose()

    def step(self, action):
        if self.discrete_actions:
            action = self.discrete2continuous(action)
        assert len(action)==self.n_states
        self.state += action + np.random.normal(0, 0.01, size=self.n_states)
        self.state = np.clip(self.state,0,1)

        s = self.get_state()
        r = 0
        done = False
        return s, r, done

    def get_state(self):
        return np.copy(self.state)

    def plot(self, ax=None):
        n_subplots = self.n_states//2 + 3
        if ax is None:
            _, ax = plt.subplots(nrows=1, ncols=(self.n_states//2), figsize=(4,4))
        for i in range(self.n_states//2):
            ax.set_xlim([0,1])
            ax.set_ylim([0,1])
            ax.scatter(self.state[2*i],self.state[2*i+1])
            ax.set_xticks([])
            ax.set_yticks([])
            # ax.set_xlabel('XY({})'.format(i))
        return ax

def run_agent(env, n_trials=1, n_samples=100, video=False):
    if video:
        ax = env.plot()
        fig = plt.gcf()
        fig.show()
    states = [env.get_state()]
    actions = []
    for trial in range(n_trials):
        for sample in range(n_samples):
            a = env.random_action()
            _, _, done = env.step(a)
            actions.append(a)
            states.append(env.get_state())

            if video:
                ax.clear()
                env.plot(ax)
                fig.canvas.draw()
                fig.canvas.flush_events()

            if done:
                time.sleep(1)
                env.reset()
                break
    return np.stack(states,axis=0), np.stack(actions,axis=0)

#%%
if __name__ == '__main__':
    env = CGridWorld()
    run_agent(env, n_samples=100, video=True)
