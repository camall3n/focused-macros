import matplotlib.pyplot as plt
import numpy as np
import torch

from learners.vic import VIC
from learners.diayn import DIAYN
from learners.common import reset_seeds

def get_policy(alg, mdp, skill):
    policy = np.empty((mdp.width+1, mdp.height+1), dtype=np.object)
    for x in range(1,mdp.width+1):
        for y in range(1,mdp.height+1):
            if not env.mdp.is_wall(x,y):
                s = torch.tensor((x,y)).float()
                action = torch.argmax(alg.action_distr(s, torch.tensor(skill)).logits).item()
                action_word = dict(enumerate(alg.env.action_map)).get(action, 'term')
                policy[x,y] = action_word
    return policy

plt.figure()

#%%
reset_seeds(0)
from simple_rl.tasks import FourRoomMDP
from notebooks.simple_rl_env import SimpleGymEnv
env = SimpleGymEnv(FourRoomMDP(12,12,goal_locs=[(12,12)]))
env.render()
s0 = torch.as_tensor(env.reset(), dtype=torch.float32)
ndim_s = len(env.observation_space)
n_actions = env.action_space.n
n_skills = 40
gamma = 0.99
max_steps_per_skill = 10
n_units = 32
lr = 1e-3
alg = VIC
# alg = DIAYN
alg = alg(env, ndim_s, n_actions, n_skills, gamma, max_steps_per_skill, n_units, lr)
start_policies = [get_policy(alg, env.mdp, s) for s in range(n_skills)]
#%%
discrim_losses, critic_losses, actor_losses = alg.train(20000)

#%%
fig, ax = plt.subplots(2,2)
ax = ax.flatten()
ax[0].plot(discrim_losses)
ax[0].set_title('discrim_losses')
ax[1].plot(critic_losses)
ax[1].set_title('critic_losses')
ax[2].plot(actor_losses)
ax[2].set_title('actor_losses')
ax[3].axis('off')
fig.tight_layout()
fig.show()

#%%
# final_policies = [get_policy(alg, env.mdp, s) for s in range(n_skills)]
#
# for skill in range(n_skills):
#     # env.render(start_policies[skill])
#     env.render(final_policies[skill])
#     print()

#%%
ncols = 3#int(np.floor(np.sqrt(n_skills)))
nrows = int(np.ceil(n_skills/ncols))
fig, ax = plt.subplots(nrows, ncols, figsize=(3*ncols,3*nrows))
ax = ax.flatten()
n_samples = 200
for skill in range(n_skills):
    final_states = np.zeros((env.mdp.width+1, env.mdp.height+1))
    for i in range(n_samples):
        alg.env.reset()
        (x,y), _ = alg.run_skill(torch.tensor(skill))
        final_states[x.int(),y.int()]+=1
    ax[skill].imshow(final_states, cmap='hot', interpolation='nearest', vmax=n_samples)
    ax[skill].set_xlim([0.5, env.mdp.width+0.5])
    ax[skill].set_ylim([0.5, env.mdp.height+0.5])
    ax[skill].set_xticks(range(1,env.mdp.width+1,2))
    ax[skill].set_yticks(range(1,env.mdp.height+1,2))
    ax[skill].invert_yaxis()
[a.axis('off') for a in ax[skill+1:]]
fig.tight_layout()
plt.savefig('{}-40sk-40k.png'.format(alg.__class__.__name__))
plt.show()

#%%
fig, ax = plt.subplots( figsize=(9,9))
n_samples = 200
final_states = np.zeros((env.mdp.width+1, env.mdp.height+1))
for skill in range(n_skills):
    for i in range(n_samples):
        alg.env.reset()
        (x,y), _ = alg.run_skill(torch.tensor(skill))
        final_states[x.int(),y.int()]+=1
ax.imshow(final_states, cmap='hot', interpolation='nearest', vmax=n_samples)
ax.set_xlim([0.5, env.mdp.width+0.5])
ax.set_ylim([0.5, env.mdp.height+0.5])
ax.set_xticks(range(1,env.mdp.width+1,2))
ax.set_yticks(range(1,env.mdp.height+1,2))
ax.invert_yaxis()
fig.tight_layout()
# plt.savefig('{}-40sk-40k-all.png'.format(alg.__class__.__name__))
plt.show()
