# Variational Intrinsic Control
# https://arxiv.org/pdf/1611.07507.pdf
from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from learners import actorcritic as a2c
from learners import classifier
from learners.common import get_batch

class VIC:
    def __init__(self, env, ndim_s, n_actions, n_skills, discount, max_timesteps=6, n_units=32, lr=1e-3):
        self.env = env
        self.n_actions = n_actions
        self.terminate_action = n_actions
        self.n_skills = n_skills
        self.discount = discount
        self.max_timesteps = max_timesteps
        self.skill_policies = [None]*n_skills
        for i in range(n_skills):
            net = a2c.Network(ndim_s, n_actions+1, n_units)
            self.skill_policies[i] = a2c.Trainer(net, discount, lr=lr)

        discrim_net = classifier.Network(ndim_s, n_skills, n_units)
        self.discriminator = classifier.Trainer(discrim_net, lr=lr)

        self.skill_prior = torch.distributions.Categorical(logits=torch.ones(n_skills))

    def get_intrinsic_reward(self, s0, sf, skill):
        with torch.no_grad():
            ds = sf - s0
            reward = self.discriminator.net(ds).log_prob(skill) - self.skill_prior.log_prob(skill)
        return reward

    def _get_experiences(self, trajectory, skill):
        s,a,rewards,dones,sp = trajectory
        #
        # skill_1hot = torch.nn.functional.one_hot(skill, self.n_skills).float()
        # s = [torch.cat((x, skill_1hot)) for x in s]
        # sp = [torch.cat((x, skill_1hot)) for x in sp]

        returns = []
        bootstrap = 0
        for reward, done in zip(rewards[::-1], dones[::-1]):
            bootstrap = reward + (self.discount*bootstrap if not done else 0)
            returns.insert(0, bootstrap)

        discounted_trajectory = a2c.Experience(s,a,returns,dones,sp)
        experiences = list(map(lambda x: a2c.Experience(*x),list(zip(*discounted_trajectory))))
        return experiences

    def get_env_state(self):
        return torch.as_tensor(tuple(self.env.state), dtype=torch.float32)

    def action_distr(self, state, skill, primitives_only=False):
        distr = self.skill_policies[skill].net.action_distr(state)
        if primitives_only:
            primitive_logits = distr.logits[:-1]
            distr = torch.distributions.Categorical(logits=primitive_logits)
        return distr

    def run_skill(self, skill):
        states = []
        actions = []
        intrinsic_rewards = []
        extrinsic_rewards = []
        dones = []

        done = False
        timestep = 0
        while True:
            state = self.get_env_state()
            states.append(state)

            is_first_timestep = (timestep==0)
            action = self.action_distr(state, skill, primitives_only=is_first_timestep).sample()
            if done or action == self.terminate_action or timestep >= self.max_timesteps:
                break

            next_state, extrinsic_reward, done, _ = self.env.step(action)
            timestep += 1

            actions.append(action)
            intrinsic_rewards.append(torch.tensor(0))
            extrinsic_rewards.append(torch.tensor(extrinsic_reward))
            dones.append(torch.tensor(done))
        if done:
            self.env.reset()

        final_state = states[-1]
        next_states = states[1:]
        states = states[:-1]
        # rewards = list(zip(intrinsic_rewards, extrinsic_rewards))
        rewards = intrinsic_rewards
        trajectory = a2c.Experience(states, actions, rewards, dones, next_states)
        return final_state, trajectory

    def train(self, n_episodes):
        discrim_losses = []
        critic_losses = []
        actor_losses = []
        for episode in tqdm(range(n_episodes)):
            self.env.reset()
            s0 = self.get_env_state()

            # Sample Ω ~ p^C(Ω|s0)
            skill = self.skill_prior.sample()

            # Follow policy π(a|Ω,s) till termination state s_f
            s_f, trajectory = self.run_skill(skill)

            # Regress q(Ω|s0,s_f) towards Ω
            batch = get_batch([classifier.Sample((s_f-s0).detach(), skill)])
            loss = self.discriminator.train(batch)
            discrim_losses.append(loss.item())

            # Calculate intrinsic reward r_I (and update trajectory experiences)
            r_I = self.get_intrinsic_reward(s0, s_f, skill)
            done = trajectory.done[-1]
            if not done:
                trajectory.s.append(s_f)
                trajectory.a.append(torch.tensor(self.terminate_action))
                trajectory.r.append(r_I)
                trajectory.done.append(torch.tensor(True))
                trajectory.sp.append(s_f)
            experiences = self._get_experiences(trajectory, skill)

            # Use an RL algorithm update for π(a|Ω,s) to maximize r_I
            critic_loss,actor_loss = self.skill_policies[skill].train(get_batch(experiences), critic_mode='montecarlo')
            critic_losses.append(critic_loss.item())
            actor_losses.append(actor_loss.item())

            # Reinforce option prior p^C(Ω|s0) based on r_I
            # TODO

        return discrim_losses, critic_losses, actor_losses
