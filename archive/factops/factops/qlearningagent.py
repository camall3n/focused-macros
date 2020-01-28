from collections import defaultdict
import numpy as np

class QLearningAgent():
    def __init__(self, n_actions, lr=0.01, epsilon=0.1, gamma=0.99):
        self.n_actions = n_actions
        self.lr = lr
        self.epsilon = epsilon
        self.n_steps_init = 2000
        self.decay_period = 8000
        self.gamma = gamma
        self.reset()

    def reset(self):
        self.n_training_steps = 0
        self.default_q = 0.0
        self.q_table = defaultdict(lambda : defaultdict(lambda: self.default_q))

    def get_epsilon(self):
        alpha = (self.n_training_steps - self.n_steps_init)/self.decay_period
        alpha = np.clip(alpha, 0, 1)
        return self.epsilon*alpha + 1*(1-alpha)

    def act(self, state):
        # Epsilon-greedy selection w.r.t. valid actions/skills
        if (self.n_training_steps < self.n_steps_init
            or np.random.uniform() < self.get_epsilon()):
                action = np.random.randint(0, self.n_actions)
        else:
            action = self.greedy_policy(state)
        return action

    def greedy_policy(self, state):
        if type(state) is np.ndarray and state.ndim > 1:
            result = np.asarray([self.greedy_policy(s) for s in state])
        else:
            result = max(range(self.n_actions), key=lambda a: self.Q(state, a))
        return result

    def Q(self, state, action):
        if type(state) is np.ndarray and state.ndim > 1:
            result = np.asarray([self.Q(s,a) for s in state])
        else:
            result = self.q_table[tuple(state)][action]
        return result

    def v(self, state):
        if type(state) is np.ndarray and state.ndim > 1:
            result = np.asarray([self.v(s) for s in state])
        else:
            result = max([self.Q(state, a) for a in range(self.n_actions)])
        return result

    def train(self, s, a, r, sp, done):
        self.n_training_steps += 1
        s = tuple(s)
        sp = tuple(sp)
        max_q_next = self.v(sp)
        q_sa = self.Q(s, a)
        bootstrap = 0 if done else self.gamma * max_q_next
        self.q_table[s][a] = (1-self.lr) * q_sa + self.lr * (r + bootstrap)
