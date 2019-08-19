import gym
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from learners.actorcritic import Network, Trainer, Experience

def test_network_single():
    ndim_s = 48
    n_actions = 12
    n = Network(ndim_s, n_actions)
    state = torch.rand((ndim_s,),dtype=torch.float32)
    action = n.action(state)
    value, q_value, pi = n(state)
    assert value.shape == torch.Size([])
    assert q_value.shape == torch.Size([n_actions])
    assert pi.param_shape == torch.Size([n_actions])

def test_network_batch():
    batch_size = 100
    ndim_s = 48
    n_actions = 12
    n = Network(ndim_s, n_actions)
    states = torch.rand((batch_size,ndim_s),dtype=torch.float32)
    values, q_values, pi = n(states)
    assert values.shape == torch.Size([batch_size])
    assert q_values.shape == torch.Size([batch_size,n_actions])
    assert pi.param_shape == torch.Size([batch_size,n_actions])

def test_train_critic():
    batch_size = 100
    ndim_s = 48
    n_actions = 12
    n = Network(ndim_s, n_actions)
    t = Trainer(n, discount=0.9)

    s = torch.rand((batch_size, ndim_s), dtype=torch.float32)
    a = torch.randint(0, n_actions, size=(batch_size,))
    r = torch.rand((batch_size,), dtype=torch.float32)
    done = torch.randint(0, 2, size=(batch_size,))
    sp = torch.rand((batch_size, ndim_s), dtype=torch.float32)
    batch = Experience(s,a,r,done,sp)

    losses = []
    n_updates = 200
    for _ in range(n_updates):
        losses.append(t.train_critic(batch).item())
    assert losses[n_updates//2] < losses[0], '{} >= {}'.format(losses[n_updates//2], losses[0])
    assert losses[-1] < losses[n_updates//2], '{} >= {}'.format(losses[-1], losses[n_updates//2])

def test_train_actor():
    batch_size = 100
    ndim_s = 48
    n_actions = 12
    n = Network(ndim_s, n_actions)
    t = Trainer(n, discount=0.9)

    s = torch.rand((batch_size, ndim_s), dtype=torch.float32)
    a = torch.randint(0,n_actions,size=(batch_size,))
    r = torch.rand((batch_size,), dtype=torch.float32)
    done = torch.randint(0,2,size=(batch_size,))
    sp = torch.rand((batch_size, ndim_s), dtype=torch.float32)
    batch = Experience(s,a,r,done,sp)

    losses = []
    n_updates = 200
    for _ in range(n_updates):
        losses.append(t.train_actor(batch).item())
    assert losses[n_updates//2] < losses[0], '{} >= {}'.format(losses[n_updates//2], losses[0])
    assert losses[-1] < losses[n_updates//2], '{} >= {}'.format(losses[-1], losses[n_updates//2])

def get_batch(replay):
    batch = Experience(*map(lambda x: torch.stack(x, dim=0), zip(*replay)))
    return batch

def test_rl():
    batch_size = 100
    env = gym.make('CartPole-v0')
    ndim_s = 4
    n_actions = 2
    net = Network(ndim_s, n_actions)
    trainer = Trainer(net, discount=0.99)

    ep_rewards = []
    replay = []
    for episode in tqdm(range(1000)):
        state = torch.as_tensor(env.reset(), dtype=torch.float32)
        ep_reward = 0

        for t in range(1000):
            action = net.action(state)
            next_state, reward, done, _ = env.step(action.item())
            ep_reward += reward
            next_state = torch.as_tensor(next_state, dtype=torch.float32)
            reward = torch.as_tensor(reward, dtype=torch.float32)
            done = torch.as_tensor(done, dtype=torch.int64)

            experience = Experience(state, action, reward, done, next_state)
            replay.append(experience)
            state = next_state

            if len(replay) >= batch_size:
                batch = get_batch(replay)
                trainer.train(batch)
                replay = []

            if done:
                break

        ep_rewards.append(ep_reward)
    plt.plot(ep_rewards)

if __name__ == '__main__':
    test_network_single()
    test_network_batch()
    test_train_critic()
    test_train_actor()
    test_rl()
