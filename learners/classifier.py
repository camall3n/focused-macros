from collections import namedtuple
import torch
import torch.nn as nn

Sample = namedtuple('Sample',['x','y'])

class Network(nn.Module):
    def __init__(self, ndim_x, ndim_y, n_units=32):
        super().__init__()
        self.ndim_x = ndim_x
        self.ndim_y = ndim_y
        self.layers = nn.Sequential(*[
            nn.Linear(ndim_x, n_units), nn.ReLU(),
            nn.Linear(n_units, n_units), nn.ReLU(),
            nn.Linear(n_units, ndim_y)
        ])

    def forward(self, x):
        logits = self.layers(x)
        predictions = torch.distributions.Categorical(logits=logits)
        return predictions

class Trainer:
    def __init__(self, net, lr=1e-3):
        self.net = net
        self.optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    def classification_loss(self, batch):
        logits = self.net(batch.x).logits
        loss = nn.functional.cross_entropy(logits, batch.y)
        return loss

    def accuracy(self, batch):
        with torch.no_grad():
            logits = self.net(batch.x).logits
            pred = logits.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct = pred.eq(batch.y.view_as(pred)).sum().item()
        return correct / len(pred)

    def train(self, batch):
        self.optimizer.zero_grad()
        loss = self.classification_loss(batch)
        loss.backward()
        self.optimizer.step()
        return loss.detach()
