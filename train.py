from data import ArgoDataset as Dataset, collate_fn
import os

# os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.umask(0)
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import random
import sys
import json
root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)
import time
import shutil

from tqdm import tqdm
import torch
from torch.utils.data import Sampler, DataLoader
import horovod.torch as hvd
from torch.utils.data.distributed import DistributedSampler
from model.utils import Logger, load_pretrain
# from mpi4py import MPI
from bigmpi4py import MPI
from model.Net import get_model
from model.config import config

comm = MPI.COMM_WORLD
hvd.init()
torch.cuda.set_device(hvd.local_rank())


def main():
    seed = hvd.rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Import all settings for experiment.

    downstream_net, loss, post_process, opt = get_model(config)

    if config["horovod"]:
        opt.opt = hvd.DistributedOptimizer(
            opt.opt, named_parameters=downstream_net.named_parameters()
        )

    # Create log and copy all code
    if hvd.rank() == 0:
        save_dir = config["save_dir"]+'_0'

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        else:
            save_dir = save_dir[:-1] + '1'
            idx = 1
            while os.path.exists(save_dir):
                idx = idx + 1
                save_dir = save_dir[:-1] + str(idx)
            os.makedirs(save_dir)
        config["save_dir"] = save_dir

        print(save_dir)
        log = os.path.join(save_dir, "log")
        sys.stdout = Logger(log)


    # Data loader for training
    dataset = Dataset(config["train_split"], config, train=True)
    train_sampler = DistributedSampler(
        dataset, num_replicas=hvd.size(), rank=hvd.rank()
    )
    train_loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        num_workers=config["workers"],
        sampler=train_sampler,
        collate_fn=collate_fn,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )

    # Data loader for evaluation
    dataset = Dataset(config["val_split"], config, train=False)
    val_sampler = DistributedSampler(dataset, num_replicas=hvd.size(), rank=hvd.rank())
    val_loader = DataLoader(
        dataset,
        batch_size=config["val_batch_size"],
        num_workers=config["val_workers"],
        sampler=val_sampler,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    hvd.broadcast_parameters(downstream_net.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(opt.opt, root_rank=0)
    config["display_iters"] = len(train_loader.dataset.split)
    config["val_iters"] = len(train_loader.dataset.split) * 2
    epoch = config["epoch"]
    remaining_epochs = int(np.ceil(config["num_epochs"] - epoch))
    for i in range(remaining_epochs):
        train(epoch + i, config, train_loader, downstream_net, loss, post_process, opt, val_loader)


def worker_init_fn(pid):
    np_seed = hvd.rank() * 1024 + int(pid)
    np.random.seed(np_seed)
    random_seed = np.random.randint(2 ** 32 - 1)
    random.seed(random_seed)


def train(epoch, config, train_loader, downstream_net, loss, post_process, opt, val_loader=None):
    train_loader.sampler.set_epoch(int(epoch))
    downstream_net.train()

    num_batches = len(train_loader)
    epoch_per_batch = 1.0 / num_batches
    save_iters = int(np.ceil(config["save_freq"] * num_batches))
    display_iters = int(
        config["display_iters"] / (hvd.size() * config["batch_size"])
    )
    val_iters = int(config["val_iters"] / (hvd.size() * config["batch_size"]))

    start_time = time.time()
    metrics = dict()
    for i, data in tqdm(enumerate(train_loader), disable=hvd.rank()):
        epoch += epoch_per_batch
        data = dict(data)

        out, ids, gts, idcs = downstream_net(data)
        ego_fut_aug_idcs, actor_idcs_mod, actor_ctrs_mod = idcs
        pred_out = torch.cat([out['reg'][i[0]][1:,0,:,:] for i in ego_fut_aug_idcs])
        reconstruction_out = torch.cat([x.unsqueeze(dim=0) for x in out['reconstruction']])
        ids_hist, ids_fut = ids
        reconstruction_gt, pred_gt = gts

        # ids_hist = torch.rand(ids_hist.shape).cuda()
        # ids_fut = torch.rand(ids_fut.shape).cuda()

        loss_out, loss_orig = loss(pred_out, pred_gt, reconstruction_out, reconstruction_gt, ids_hist, ids_fut, ego_fut_aug_idcs, actor_idcs_mod)
        # print(loss_out)
        if torch.isnan(loss_out):
            print('nan loss')
            break

        post_out = post_process(out, ego_fut_aug_idcs, data)
        post_process.append(metrics, loss_out, loss_orig, post_out)


        opt.zero_grad()
        loss_out.backward()
        grad_check = 0
        for name, param in downstream_net.named_parameters():
            if not(torch.isfinite(param.grad).all()):
                print('nan grad')
                grad_check = 1
        if grad_check == 0:
            lr = opt.step(epoch)
        else:
            break


        num_iters = int(np.round(epoch * num_batches))
        if hvd.rank() == 0 and epoch >= 0 and (
                num_iters % save_iters == 0 or epoch >= config["num_epochs"]
        ):
            save_ckpt(downstream_net, opt, config["save_dir"], epoch)

        if num_iters % display_iters == 0:
            dt = time.time() - start_time
            metrics = sync(metrics)
            if hvd.rank() == 0:
                post_process.display(metrics, dt, epoch, lr)
            start_time = time.time()
            metrics = dict()

        if num_iters % val_iters == 0:
            val(config, val_loader, downstream_net, loss, post_process, epoch)

        if epoch >= config["num_epochs"]:
            val(config, val_loader, downstream_net, loss, post_process, epoch)
            return


def val(config, data_loader, downstream_net, loss, post_process, epoch):
    downstream_net.eval()

    start_time = time.time()
    metrics = dict()
    for i, data in enumerate(data_loader):
        data = dict(data)
        with torch.no_grad():
            out, ids, gts, idcs = downstream_net(data)
            ego_fut_aug_idcs, actor_idcs_mod, actor_ctrs_mod = idcs
            pred_out = torch.cat([out['reg'][i[0]][1:, 0, :, :] for i in ego_fut_aug_idcs])
            reconstruction_out = torch.cat([x.unsqueeze(dim=0) for x in out['reconstruction']])
            ids_hist, ids_fut = ids
            reconstruction_gt, pred_gt = gts
            loss_out, loss_orig = loss(pred_out, pred_gt, reconstruction_out, reconstruction_gt, ids_hist, ids_fut, ego_fut_aug_idcs, actor_idcs_mod)
            post_out = post_process(out, ego_fut_aug_idcs, data)
            post_process.append(metrics, loss_out, loss_orig, post_out)

    dt = time.time() - start_time
    metrics = sync(metrics)
    if hvd.rank() == 0:
        post_process.display(metrics, dt, epoch)


def save_ckpt(net, opt, save_dir, epoch):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    state_dict = net.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()

    save_name = "%3.3f.ckpt" % epoch
    torch.save(
        {"epoch": epoch, "state_dict": state_dict, "opt_state": opt.opt.state_dict()},
        os.path.join(save_dir, save_name),
    )

def sync(data):
    data_list = comm.allgather(data)
    data = dict()
    for key in data_list[0]:
        if isinstance(data_list[0][key], list):
            data[key] = []
        else:
            data[key] = 0
        for i in range(len(data_list)):
            data[key] += data_list[i][key]
    return data


if __name__ == "__main__":
    main()