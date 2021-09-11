import os
import sys
root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)

from torch import nn

class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()
        self.config = config

        self.identification = nn.Sequential(
            nn.Linear(config['n_actor'], int(config['n_actor']/2)),
            nn.ReLU(inplace=True),
            nn.Linear(int(config['n_actor']/2), int(config['n_actor']/4)),
            nn.ReLU(inplace=True),
            nn.Linear(int(config['n_actor']/4), int(config['n_actor']/8)),
            nn.ReLU(inplace=True),
            nn.Linear(int(config['n_actor']/8), int(config['n_actor']/16)),
            nn.Tanh(),
        )

    def forward(self, actors_mod):
        # construct actor feature
        id = self.identification(actors_mod)

        return id