import os
import sys
root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)

from torch import Tensor, nn
from typing import Dict, List, Tuple, Union
import torch
from layers import LinearRes
from utils import gpu

class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()
        self.config = config

        self.decoder = Decoder(config)

    def forward(self, actors, actor_idcs_mod, actor_ctrs_mod):
        out = self.decoder(actors, actor_idcs_mod, actor_ctrs_mod)

        return out


class Decoder(nn.Module):
    """
    Final motion forecasting with Linear Residual block
    """

    def __init__(self, config):
        super(Decoder, self).__init__()
        self.config = config
        norm = "GN"
        ng = 1

        n_actor = config["n_actor"]

        self.decoder = LinearRes(n_actor, n_actor, norm=norm, ng=ng)
        self.generator = nn.Linear(n_actor, 2 * config["num_preds"])
        self.reconstructor = nn.Linear(n_actor, 2 * config["num_preds"])

    def forward(self, actors: Tensor, actor_idcs_mod: List[Tensor], actor_ctrs_mod: List[Tensor]) -> Dict[str, List[Tensor]]:
        preds = []
        recons = []

        hid = self.decoder(actors)
        preds.append(self.generator(hid))

        hid_for_ego = torch.cat([hid[x[0]:x[0+1]] for x in actor_idcs_mod])
        recons.append(self.reconstructor(hid_for_ego))

        reg = torch.cat([x.unsqueeze(1) for x in preds], 1)
        reg = reg.view(reg.size(0), reg.size(1), -1, 2)
        reconstruction = torch.cat([x.unsqueeze(1) for x in recons], 1)
        reconstruction = reconstruction.view(reconstruction.size(0), reconstruction.size(1), -1, 2)

        for i in range(len(actor_idcs_mod)):
            idcs = actor_idcs_mod[i]
            ctrs = actor_ctrs_mod[i].view(-1, 1, 1, 2)
            reg[idcs] = reg[idcs] + ctrs
            reconstruction[i] = reconstruction[i] + ctrs[0]

        out = dict()
        out["reconstruction"], out["reg"] = [], []
        for i in range(len(actor_idcs_mod)):
            idcs = actor_idcs_mod[i]
            out["reg"].append(reg[idcs])
            out['reconstruction'].append(reconstruction[i,0])
        return out
