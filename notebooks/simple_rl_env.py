import gym
from gym.spaces import Discrete, Tuple
import numpy as np

class SimpleGymEnv(gym.Env):
    def __init__(self, mdp):
        self.mdp = mdp
        self.state = mdp.init_state
        self.observation_space = Tuple((Discrete(mdp.width),Discrete(mdp.height)))
        self.action_space = Discrete(len(mdp.ACTIONS))
        self.action_map = mdp.ACTIONS
        super().__init__()

    def step(self, action):
        action = self.action_map[action]
        next_state = self.mdp.transition_func(self.state, action)
        reward = self.mdp.reward_func(self.state, action, next_state)
        done = self.mdp.is_goal_state(next_state)
        self.state = next_state
        return np.asarray(next_state), reward, done, None

    def reset(self):
        self.state = self.mdp.init_state
        return np.asarray(self.state)

    def render(self, policy=None):
        grid = np.zeros((self.mdp.width+2, self.mdp.height+2))
        grid[0,:] = 1
        grid[-1,:] = 1
        grid[:,0] = 1
        grid[:,-1] = 1
        for x,y in self.mdp.walls:
            grid[x, y] = 1

        for y, column in enumerate(grid):
            for x, cell in enumerate(column):
                if grid[x,y]:
                    print('#',end='')
                elif policy is not None:
                    action_arrow = {'up': '˄', 'down': '˅', 'left': '˂', 'right': '˃', 'term': '•'}[policy[x,y]]
                    print(action_arrow, end='')
                else:
                    print('O' if tuple(self.state)==(x,y) else '.',end='')
            print()

    def close(self):
        pass
#
# from simple_rl.tasks import FourRoomMDP
# env = SimpleGymEnv(FourRoomMDP(12,12,goal_locs=[(12,12)]))
# env.reset()
# env.render()
