import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import sys
import torch
import torch.nn
import torch.nn.functional as F
from tqdm import tqdm

from gridworlds.nn.nnutils import Network, Reshape, extract
from gridworlds.utils import reset_seeds
import gridworlds.sensors as sensors

#%%
# seed = 10# good
seed = 1
reset_seeds(seed)

n_zdim = 4

class Squares:
    def __init__(self, nsquares=1, size=12, side=2):
        self.nsquares = nsquares
        self.size = size # size of the observation
        self.side = side # size of squares
        self.reset()

    def reset(self):
        self.state = [np.random.randint(0,self.size-self.side,2) for i in range(self.nsquares)]

    @property
    def nactions(self):
        # 0=right, 1=left, 2=up, 3=down
        return 4

    def get_state(self):
        return np.copy(self.state)

    def observe(self, s):
        x = np.zeros([self.size]*2, 'float32')
        for i in range(self.nsquares):
            x[s[i][0]:s[i][0]+self.side, s[i][1]:s[i][1]+self.side] = 1
        return x

    def step(self, action):
        delta = [(1,0),(-1,0),(0,1),(0,-1)]
        self.state = [p+delta[action[i]] for i,p in enumerate(self.state)]
        self.state = [np.minimum(np.maximum(p,0),self.size-self.side) for p in self.state]
        r = 0
        done = False
        return (self.state, r, done)

    def genRandomSample(self):
        """
        get a random (s,a,s') transition from the environment (assuming a uniform policy)

        returns (state, action, next state)
        """
        self.reset()

        s0 = self.state
        x0 = self.observe(s0)

        action = np.random.randint(0,self.nactions,self.nsquares)
        self.step(action)

        s1 = self.state
        x1 = self.observe(s1)

        return (x0.flatten(), action, x1.flatten(), s0, s1)

class ICFNet(Network):
    def __init__(self, input_shape=(12,12), n_hidden=32, n_zdim=4, n_actions=4):
        super().__init__()
        n_inputs = np.prod(input_shape)
        self.conv1 = torch.nn.Conv2d( 1,16,kernel_size=3,stride=2,padding=1)
        self.conv2 = torch.nn.Conv2d(16,16,kernel_size=3,stride=2,padding=1)
        self.flatten = Reshape(-1, 16*3*3)
        self.fc1 = torch.nn.Linear(16*3*3, n_hidden)
        self.fc2 = torch.nn.Linear(n_hidden, n_zdim)
        self.fc_pi = torch.nn.Linear(n_hidden, n_zdim*n_actions)
        self.regroup = Reshape(-1, n_zdim, 4)

        self.dec_fc2 = torch.nn.Linear(n_zdim, n_hidden)
        self.dec_fc1 = torch.nn.Linear(n_hidden, 16*3*3)
        self.unflatten = Reshape(-1, 16, 3, 3)
        self.deconv2 = torch.nn.ConvTranspose2d(16,16,kernel_size=3,stride=2,padding=1,output_padding=1)
        self.deconv1 = torch.nn.ConvTranspose2d(16, 1,kernel_size=3,stride=2,padding=1,output_padding=1)

    def encode(self, x):
        tmp = x
        tmp = self.conv1(tmp)
        tmp = F.relu(tmp)
        tmp = self.conv2(tmp)
        tmp = F.relu(tmp)
        tmp = self.flatten(tmp)
        tmp = self.fc1(tmp)
        tmp = F.relu(tmp)
        prev = tmp
        tmp = self.fc2(tmp)
        z = torch.tanh(tmp)

        tmp = self.fc_pi(prev)
        tmp = self.regroup(tmp)
        pi = F.softmax(tmp, dim=-1)
        return z, pi

    def decode(self, z):
        tmp = z
        tmp = self.dec_fc2(tmp)
        tmp = F.relu(tmp)
        tmp = self.dec_fc1(tmp)
        tmp = F.relu(tmp)
        tmp = self.unflatten(tmp)
        tmp = self.deconv2(tmp)
        tmp = F.relu(tmp)
        tmp = self.deconv1(tmp)
        x_hat = torch.tanh(tmp)
        return x_hat

    def forward(self, x):
        z, pi = self.encode(x)
        x = self.decode(z)
        return x, pi, z
    pass

class ICFTrainer():
    def __init__(self, model, beta=0.1, lr=0.0005):
        self.model = model
        self.beta = beta
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def reconstruction_loss(self, x, x_hat):
        return 0.5 * F.mse_loss(input=x_hat, target=x)

    def selectivity(self, z, zp):
        return (zp - z) / (1e-4 + torch.sum(F.relu(zp-z), dim=-1, keepdim=True))

    def train_batch(self, batch_s, batch_a, batch_sp):
        self.model.train()
        self.optimizer.zero_grad()

        s_hat, pi, z = self.model(batch_s)
        sp_hat, pi_p, zp = self.model(batch_sp)

        L_ae = self.reconstruction_loss(batch_s, s_hat)

        Pr_a_acted = extract(pi, idx=batch_a, idx_dim=-1)
        sel_acted = self.selectivity(z, zp)
        L_sel = torch.mean(torch.sum(Pr_a_acted * sel_acted, dim=1), dim=0)
        loss = L_ae - self.beta * L_sel

        loss.backward()
        self.optimizer.step()
        return loss, L_ae, L_sel
    pass

#%%
env = Squares()
sensor = sensors.SensorChain([
    env,
    sensors.TorchSensor(),
    sensors.UnsqueezeSensor(0),
])

plt.imshow(sensor.observe(env.get_state())[0])
plt.show()

#%%
replay_s = []
replay_a = []
replay_sp = []
replay_x = []
replay_y = []
for t in tqdm(range(40000)):
    env.reset()
    s = sensor.observe(env.get_state())
    a = np.random.randint(4, size=env.nsquares)
    env.step(a)
    x,y = env.get_state()[0]
    sp = sensor.observe(env.get_state())
    replay_x.append(x)
    replay_y.append(y)
    replay_s.append(s)
    replay_a.append(a)
    replay_sp.append(sp)
replay_x = torch.as_tensor(replay_x)
replay_y = torch.as_tensor(replay_y)
replay_s = torch.stack(replay_s)
replay_a = torch.as_tensor(replay_a)
replay_sp = torch.stack(replay_sp)

#%%
net = ICFNet(n_zdim=n_zdim)
trainer = ICFTrainer(net)

batch_size = 64
losses = []
L_aes = []
L_sels = []
for i in tqdm(range(500*20)):
    batch_idx = np.random.choice(np.arange(len(replay_a)), batch_size)
    batch_s = replay_s[batch_idx]
    batch_a = replay_a[batch_idx]
    batch_sp = replay_sp[batch_idx]
    loss, L_ae, L_sel = trainer.train_batch(batch_s, batch_a, batch_sp)
    losses.append(loss)
    L_aes.append(L_ae)
    L_sels.append(L_sel)

#%%
os.makedirs('results/seed_{}'.format(seed), exist_ok=True)

s_recons, pi, z = net(batch_s)
fig, ax = plt.subplots(3,2, figsize=(4,6))
ax = ax.flatten()
[a.axis('off') for a in ax]
ax[0].set_title('observed state')
ax[1].set_title('reconstruction')
ax[0].imshow(batch_s[0][0].detach().numpy(),vmin=0,vmax=1)
ax[1].imshow(s_recons[0][0].detach().numpy(),vmin=0,vmax=1)
ax[2].imshow(batch_s[15][0].detach().numpy(),vmin=0,vmax=1)
ax[3].imshow(s_recons[15][0].detach().numpy(),vmin=0,vmax=1)
ax[4].imshow(batch_s[31][0].detach().numpy(),vmin=0,vmax=1)
ax[5].imshow(s_recons[31][0].detach().numpy(),vmin=0,vmax=1)
plt.savefig('results/seed_{}/reconstruction.png'.format(seed))
plt.show()

#%%
plt.plot(losses)
plt.title('Loss vs Time')
plt.xlabel('Minibatches')
plt.savefig('results/seed_{}/loss.png'.format(seed))
plt.show()

#%%

pi = net(replay_s)[1].detach().numpy()
plt.imshow(np.mean(pi,0), vmin=0, vmax=1)
plt.xticks(range(4),['right','left','up','down'])
plt.yticks(range(n_zdim))
plt.xlabel(r'$a$')
plt.ylabel(r'$k$')
plt.colorbar()
plt.title(r'$E_s[\pi_k(a|s)]$')
plt.savefig('results/seed_{}/policy.png'.format(seed))
plt.show()

z = net.encode(replay_s)[0].detach().numpy()
zp = net.encode(replay_sp)[0].detach().numpy()
a = replay_a.detach().numpy()

r = np.zeros((n_zdim,2))
for k in range(n_zdim):
    r[k,0] = np.corrcoef(x=replay_x,y=z[:,k])[1][0]
    r[k,1] = np.corrcoef(x=replay_y,y=z[:,k])[1][0]
plt.imshow(r,cmap='coolwarm', vmin=-1, vmax=1)
plt.xticks([0,1],['x','y'])
plt.yticks(range(n_zdim))
plt.ylabel(r'$f_k$')
plt.title('Correlation with true state')
plt.colorbar()
plt.savefig('results/seed_{}/factors.png'.format(seed))
plt.show()
