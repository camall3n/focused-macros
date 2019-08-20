import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import datasets, transforms
from tqdm import tqdm

from learners.classifier import Network, Trainer, Sample

def test_network_single():
    ndim_x = 28*28
    n_actions = 10
    net = Network(ndim_x, n_actions)
    x = torch.rand((ndim_x,),dtype=torch.float32)
    y_preds = net(x)
    assert y_preds.param_shape == torch.Size([n_actions])

def test_network_batch():
    batch_size = 100
    ndim_x = 28*28
    n_actions = 10
    net = Network(ndim_x, n_actions)
    x = torch.rand((batch_size,ndim_x),dtype=torch.float32)
    y_preds = net(x)
    assert y_preds.param_shape == torch.Size([batch_size, n_actions])

def get_batch(replay):
    batch = Experience(*map(lambda x: torch.stack(x, dim=0), zip(*replay)))
    return batch

def test_train_mnist():
    batch_size = 100
    epochs = 1
    ndim_x = 28*28
    n_actions = 10

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)

    net = Network(ndim_x, n_actions)
    t = Trainer(net)

    accuracies = []
    for epoch in range(1, epochs + 1):
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            batch = Sample(x=torch.flatten(data, 1), y=target)
            t.train(batch)
            accuracy = t.accuracy(batch)
            accuracies.append(accuracy)

    fig, ax = plt.subplots()
    ax.plot(accuracies)
    plt.show()

if __name__ == '__main__':
    test_network_single()
    test_network_batch()
    test_train_mnist()
