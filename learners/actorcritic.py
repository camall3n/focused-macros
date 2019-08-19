from collections import namedtuple
import torch
import torch.nn as nn

Experience = namedtuple('Experience',['s','a','r','done','sp'])

class Network(nn.Module):
    def __init__(self, ndim_s, n_actions, n_units=32):
        super().__init__()
        self.ndim_s = ndim_s
        self.n_actions = n_actions
        self.phi = nn.Sequential(*[
            nn.Linear(ndim_s, n_units), nn.ReLU(),
            nn.Linear(n_units, n_units), nn.ReLU(),
        ])
        self.policy_head = nn.Linear(n_units, n_actions)
        self.q_head = nn.Linear(n_units, n_actions)

    def forward(self, s):
        features = self.phi(s)
        pi = torch.distributions.Categorical(logits=self.policy_head(features))
        q = self.q_head(features)
        v = torch.sum(pi.probs*q, dim=-1)
        return v, q, pi

    def action(self, s):
        with torch.no_grad():
            pi = self(s)[2]
            return pi.sample()

class Trainer:
    def __init__(self, net, discount):
        self.net = net
        self.discount = torch.as_tensor(discount)
        self.loss_fn = nn.MSELoss()
        self.actor_optimizer = torch.optim.Adam(net.parameters())
        self.critic_optimizer = torch.optim.Adam(net.parameters())

    def critic_loss(self, batch):
        a_onehot = nn.functional.one_hot(batch.a, self.net.n_actions)

        v_sp = self.net(batch.sp)[0]
        q_s = self.net(batch.s)[1]
        q_s_acted = q_s * a_onehot.float()

        with torch.no_grad():
            v_s_target = (batch.r + (1-batch.done).float() * self.discount * v_sp)
            q_s_target = q_s_acted.clone()
            idx = torch.arange(q_s_target.shape[0], dtype=torch.int64)
            q_s_target[idx, batch.a] = v_s_target

        return nn.functional.mse_loss(input=q_s_acted, target=q_s_target)

    def actor_loss(self, batch):
        v, q, pi = self.net(batch.s)

        idx = torch.arange(q.shape[0], dtype=torch.int64)
        q_acted = q[idx, batch.a]
        adv = q_acted - v

        log_p = pi.log_prob(batch.a)
        return torch.mean(-log_p * adv, dim=0)

    def train_actor(self, batch):
        self.actor_optimizer.zero_grad()
        loss = self.actor_loss(batch)
        loss.backward()
        self.actor_optimizer.step()
        return loss.detach()

    def train_critic(self, batch):
        self.critic_optimizer.zero_grad()
        loss = self.critic_loss(batch)
        loss.backward()
        self.critic_optimizer.step()
        return loss.detach()

    def train(self, batch):
        for _ in range(10):
            self.train_critic(batch)
        self.train_actor(batch)
