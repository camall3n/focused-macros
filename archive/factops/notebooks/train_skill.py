import json
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import sys
from tqdm import tqdm

from gridworlds.domain.gridworld.grid import action_meanings
from gridworlds.domain.gridworld.gridworld import GridWorld, DiagGridWorld
from factops.subgoalgrid import SubgoalGridWorld, SubgoalCGridWorld
from gridworlds.utils import reset_seeds
from gridworlds.agents.dqnagent import DQNAgent
from factops.qlearningagent import QLearningAgent
from gridworlds.nn.nullabstraction import NullAbstraction
from gridworlds.sensors import OffsetSensor

#%%
class args: pass
args.seed = 0
args.n_trials = 1
args.n_episodes = 10000
args.max_steps = 100
args.video = True
args.subgoal = True
args.continuous = False
# args.penalty = None
# args.penalty = 'stepwise'
args.penalty = 'start'
args.obstacles = 5

reset_seeds(args.seed)

if args.continuous:
    env = SubgoalCGridWorld(discrete_actions=True)
    xy_offset = 0
    flipxy = False
    xy_lim = 1
else:
    if args.subgoal:
        env = SubgoalGridWorld(10,10, penalty=args.penalty, obstacles=args.obstacles)
    else:
        env = DiagGridWorld(10, 10)
    env.discrete2continuous = lambda a: np.asarray(list(map(lambda i: env.action_map[i], a.tolist())))*np.asarray([-1,1])
    env.reset_goals(1)
    flipxy = True
    xy_offset = 0.5
    xy_lim = 9


# agent = DQNAgent(2, env.n_actions, NullAbstraction(-1, 2), n_hidden_layers=2, lr=0.001)
agent = QLearningAgent(env.n_actions, lr=0.1)

if args.video:
    fig, ax = plt.subplots(2,2)
    ax = ax.flatten()
    fig.show()

    def plot_value_function(ax):
        n_bins = 10
        x, y = np.meshgrid(np.linspace(0,xy_lim,n_bins), np.linspace(0,xy_lim,n_bins))
        if flipxy:
            x, y = y, x
        s = np.stack([np.asarray(x), np.asarray(y)],axis=-1)
        v = agent.v(s).reshape(n_bins,n_bins)
        v.shape
        if flipxy:
            x, y = y, x
        ax.contourf(x+xy_offset, y+xy_offset, v, vmin=-10, vmax=0)

    def plot_policy(ax):
        n_bins = 10
        x, y = np.meshgrid(np.linspace(0,xy_lim,n_bins), np.linspace(0,xy_lim,n_bins))
        if flipxy:
            x, y = y, x
        s = np.stack([np.asarray(x), np.asarray(y)],axis=-1)
        # s = np.concatenate([x, y], axis=-1)
        a = agent.greedy_policy(s).reshape(-1)
        dir = env.discrete2continuous(a)
        # dir = list(map(lambda x: action_meanings[tuple(x)], dir.tolist()))
        dir = np.asarray(dir).reshape(n_bins, n_bins, 2)
        dir_x = dir[:,:,1]
        dir_y = dir[:,:,0]

        if flipxy:
            x, y = y, x
        ax.quiver(x+xy_offset, y+xy_offset, dir_x, dir_y)

    def plot_states(ax):
        data = pd.DataFrame(agent.replay.memory)
        data[['x.r','x.c']] = pd.DataFrame(data['x'].tolist(), index=data.index)
        data[['xp.r','xp.c']] = pd.DataFrame(data['xp'].tolist(), index=data.index)
        sns.scatterplot(data=data, x='x.c',y='x.r', hue='done', style='done', markers=True, size='done', size_order=[1,0], ax=ax, alpha=0.3, legend=False)
        ax.invert_yaxis()

for trial in tqdm(range(args.n_trials), desc='trials'):
    # env.set_goal(*env.random_goal())
    agent.reset()
    total_reward = 0
    total_steps = 0
    losses = []
    rewards = []
    value_fn = []
    for episode in tqdm(range(args.n_episodes), desc='episodes'):
        env.reset()
        ep_rewards = []
        for step in range(args.max_steps):
            s = env.get_state()
            a = agent.act(s)
            sp, r, done = env.step(a)
            ep_rewards.append(r)
            if args.video:
                value_fn.append(agent.v(s))
            total_reward += r

            loss = agent.train(s, a, r, sp, done)
            losses.append(loss)

            if done:
                break
        rewards.append(sum(ep_rewards))

        if args.video and episode % 500 == 0:
            [a.clear() for a in ax]
            plot_value_function(ax[0])
            env.plot(ax[0])
            ax[0].set_title('V(s)')
            env.plot(ax[1])
            plot_policy(ax[1])
            ax[1].set_title('Policy')
            ax[2].plot(rewards, c='C3')
            ax[2].set_title('Rewards')
            ax[3].plot(value_fn)
            ax[3].set_title('V(s) vs time')
            # plot_states(ax[3])
            # ax[1].set_ylim([-10,0])
            fig.canvas.draw()
            fig.canvas.flush_events()

        total_steps += step
        # score_info = {
        #     'trial': trial,
        #     'episode': episode,
        #     'reward': sum(ep_rewards),
        #     'total_reward': total_reward,
        #     'total_steps': total_steps,
        #     'steps': step
        # }
        # json_str = json.dumps(score_info)
        # log.write(json_str+'\n')
        # log.flush()
print('\n\n')

fig, ax = plt.subplots(2,2)
ax = ax.flatten()
[a.clear() for a in ax]
plot_value_function(ax[0])
env.plot(ax[0])
ax[0].set_title('V(s)')
env.plot(ax[1])
plot_policy(ax[1])
ax[1].set_title('Policy')
ax[2].plot(rewards, c='C3')
ax[2].set_title('Rewards')
ax[3].plot(value_fn)
ax[3].set_title('V(s) vs time')

# mode_str = '_penalty_' if args.continuous else ''
results_dir = 'results/discrete-obstacles/tabular-q'
os.makedirs(results_dir, exist_ok=True)
plt.savefig(results_dir+'/train_{}_penalty_{}.png'.format(args.seed, str(args.penalty).lower()))
