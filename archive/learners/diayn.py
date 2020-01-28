import torch
from tqdm import tqdm

from learners.vic import VIC
from learners.common import get_batch
from learners import classifier

class DIAYN(VIC):
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
            batch = get_batch([classifier.Sample((s-s0).detach(), skill) for s in trajectory.sp])
            loss = self.discriminator.train(batch)
            discrim_losses.append(loss.item())

            # Calculate intrinsic rewards r_I (and update trajectory experiences)
            r_I = self.get_intrinsic_reward(s0, torch.stack(trajectory.sp), skill)
            r_If = self.get_intrinsic_reward(s0, s_f, skill)
            trajectory.r.clear()
            trajectory.r.extend(r_I)
            done = trajectory.done[-1]
            if not done:
                trajectory.s.append(s_f)
                trajectory.a.append(torch.tensor(self.terminate_action))
                trajectory.r.append(r_If)
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
