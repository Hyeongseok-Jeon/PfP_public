# Copyright (c) 2020 Uber Technologies, Inc.
# Please check LICENSE for more detail

import numpy as np
import sys
import cv2
import os

import torch
from torch import optim


def index_dict(data, idcs):
    returns = dict()
    for key in data:
        returns[key] = data[key][idcs]
    return returns


def rotate(xy, theta):
    st, ct = torch.sin(theta), torch.cos(theta)
    rot_mat = xy.new().resize_(len(xy), 2, 2)
    rot_mat[:, 0, 0] = ct
    rot_mat[:, 0, 1] = -st
    rot_mat[:, 1, 0] = st
    rot_mat[:, 1, 1] = ct
    xy = torch.matmul(rot_mat, xy.unsqueeze(2)).view(len(xy), 2)
    return xy


def merge_dict(ds, dt):
    for key in ds:
        dt[key] = ds[key]
    return


class Logger(object):
    def __init__(self, log):
        self.terminal = sys.stdout
        self.log = open(log, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass


def load_pretrain(net, pretrain_dict):
    state_dict = net.state_dict()
    for key in pretrain_dict.keys():
        if key in state_dict and (pretrain_dict[key].size() == state_dict[key].size()):
            value = pretrain_dict[key]
            if not isinstance(value, torch.Tensor):
                value = value.data
            state_dict[key] = value
    net.load_state_dict(state_dict)


def gpu(data):
    """
    Transfer tensor in `data` to gpu recursively
    `data` can be dict, list or tuple
    """
    if isinstance(data, list) or isinstance(data, tuple):
        data = [gpu(x) for x in data]
    elif isinstance(data, dict):
        data = {key:gpu(_data) for key,_data in data.items()}
    elif isinstance(data, torch.Tensor):
        data = data.contiguous().cuda(non_blocking=True)
    return data


def to_long(data):
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = to_long(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [to_long(x) for x in data]
    if torch.is_tensor(data) and data.dtype == torch.int16:
        data = data.long()
    return data

class Optimizer(object):
    def __init__(self, params, config, coef=None):
        if not (isinstance(params, list) or isinstance(params, tuple)):
            params = [params]

        if coef is None:
            coef = [1.0] * len(params)
        else:
            if isinstance(coef, list) or isinstance(coef, tuple):
                assert len(coef) == len(params)
            else:
                coef = [coef] * len(params)
        self.coef = coef

        param_groups = []
        for param in params:
            param_groups.append({"params": param, "lr": 0})

        opt = config["opt"]
        assert opt == "sgd" or opt == "adam"
        if opt == "sgd":
            self.opt = optim.SGD(
                param_groups, momentum=config["momentum"], weight_decay=config["wd"]
            )
        elif opt == "adam":
            self.opt = optim.Adam(param_groups, weight_decay=0)

        self.lr_func = config["lr_func"]

        if "clip_grads" in config:
            self.clip_grads = config["clip_grads"]
            self.clip_low = config["clip_low"]
            self.clip_high = config["clip_high"]
        else:
            self.clip_grads = False

    def zero_grad(self):
        self.opt.zero_grad()

    def step(self, epoch):
        if self.clip_grads:
            self.clip()

        lr = self.lr_func(epoch)
        for i, param_group in enumerate(self.opt.param_groups):
            param_group["lr"] = lr * self.coef[i]
        self.opt.step()
        return lr

    def clip(self):
        low, high = self.clip_low, self.clip_high
        params = []
        for param_group in self.opt.param_groups:
            params += list(filter(lambda p: p.grad is not None, param_group["params"]))
        for p in params:
            mask = p.grad.data < low
            p.grad.data[mask] = low
            mask = p.grad.data > high
            p.grad.data[mask] = high

    def load_state_dict(self, opt_state):
        self.opt.load_state_dict(opt_state)


class StepLR:
    def __init__(self, lr, lr_epochs):
        assert len(lr) - len(lr_epochs) == 1
        self.lr = lr
        self.lr_epochs = lr_epochs

    def __call__(self, epoch):
        idx = 0
        for lr_epoch in self.lr_epochs:
            if epoch < lr_epoch:
                break
            idx += 1
        return self.lr[idx]


def feats_to_traj(feats, orig, rot):
    orig_ego_view = orig[0]
    feats_ego_view = feats[0, :, :, :2]
    has_val = feats[0,:,:,-1]
    rot_ego_view = rot[0]

    traj_out = []
    for i in range(feats_ego_view.shape[0]):
        target_feats = feats_ego_view[i]
        traj = np.matmul(np.linalg.inv(rot_ego_view), target_feats.T).T + orig_ego_view
        traj_out.append(traj[np.where(has_val[i] == 1)])


    return traj_out


def path_gen(init_pos, init_vel, end_pos, end_vel):
    init_pos_x = init_pos[0]
    init_vel_x = init_vel[0]
    end_pos_x = [end_pos[i][0] for i in range(len(end_pos))]
    end_vel_x = end_vel[0]

    init_pos_y = init_pos[1]
    init_vel_y = init_vel[1]
    end_pos_y = [end_pos[i][1] for i in range(len(end_pos))]
    end_vel_y = end_vel[1]

    d_x = init_pos_x
    c_x = init_vel_x
    b_x = ((end_pos_x - 3 * c_x - d_x) - (end_vel_x - c_x)) / 3
    a_x = (end_pos_x - d_x - 3 * c_x - 9 * b_x) / 27

    d_y = init_pos_y
    c_y = init_vel_y
    b_y = ((end_pos_y - 3 * c_y - d_y) - (end_vel_y - c_y)) / 3
    a_y = (end_pos_y - d_y - 3 * c_y - 9 * b_y) / 27

    traj = np.zeros(shape=(len(end_vel_x), 21, 2), dtype=np.float32)
    for i in range(21):
        t = 0.1 * i
        x = a_x * t ** 3 + b_x * t ** 2 + c_x * t + d_x
        y = a_y * t ** 3 + b_y * t ** 2 + c_y * t + d_y

        traj[:, i, 0] = x
        traj[:, i, 1] = y
    return traj

def deg_norm(deg):
    if deg < 0:
        out = deg + 360
    else:
        out = deg

    return out
