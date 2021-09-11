import os
import sys
root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)

from torch import nn
from utils import gpu, Optimizer
import torch
from modules.state_encoder import Net as enc
from modules.interaction_embedder import Net as emb
from modules.decoder_reconstructor import Net as dec
from modules.identifier import Net as id
import numpy as np

class Net(nn.Module):
    def __init__(self, config, state_encoder, interaction_embedder, decoder, identifier):
        super(Net, self).__init__()
        self.config = config
        self.state_encoder = state_encoder
        self.interaction_embedder = interaction_embedder
        self.decoder = decoder
        self.identifier = identifier

    def forward(self, data):
        # construct actor feature
        batch_num = len(data['city'])
        data_reform = data_form(data)
        _, actor_idcs = actor_gather(gpu(data_reform["feats"]))

        # get interaction-aware hidden from historical observation
        actors, actor_idcs, actor_ctrs = self.state_encoder(data_reform)
        actors = self.interaction_embedder(actors, actor_idcs, actor_ctrs)

        # get hidden from future observation
        data_refrom = ego_fut_data_form(data)
        ego_fut_actors, ego_fut_actor_idcs, ego_fut_actor_ctrs = self.state_encoder(data_refrom)

        # get ID code from history
        actors_mod, actor_idcs_mod, actor_ctrs_mod = self.data_mod(actors, actor_idcs, ego_fut_actors, ego_fut_actor_idcs, data)
        ids_hist = self.identifier(actors_mod)
        ids_hist = torch.cat([ids_hist[x[1:]] for x in actor_idcs_mod])

        # get reactive hidden from history and future
        actors = self.interaction_embedder(actors_mod, actor_idcs_mod, actor_ctrs_mod)
        out = self.decoder(actors, actor_idcs_mod, actor_ctrs_mod)

        # output sorting
        rot, orig = gpu(data["rots"]), gpu(data["origs"])
        rot_mod = []
        orig_mod = []
        for i in range(batch_num):
            rot_tmp = rot[i][0]
            orig_tmp = orig[i][0]
            for _ in range(len(ego_fut_actor_idcs[i])):
                rot_mod.append(rot_tmp)
                orig_mod.append(orig_tmp)

        # transform prediction to world coordinates
        for i in range(len(out["reg"])):
            out['reconstruction'][i] = torch.matmul(out["reconstruction"][i], rot_mod[i]) + orig_mod[i]
            out["reg"][i] = torch.matmul(out["reg"][i], rot_mod[i]) + orig_mod[i].view(
                1, 1, 1, -1
            )

        # get reconstructed action input
        recon_gt = []
        for i in range(batch_num):
            tmp = gpu(torch.cat([data['gt_preds'][i][0:1], data['ego_aug'][i]['traj']]))
            recon_gt.append(tmp[~torch.isnan(tmp[:, 0, 0])])
        reconstruction_gt = torch.cat(recon_gt)

        # get ID code from future prediction
        fut_data = self.get_fut_data(out, reconstruction_gt, ego_fut_actor_idcs, batch_num, data)

        actors_fut, actor_idcs_fut, actor_ctrs_fut = self.state_encoder(fut_data)
        representation = self.interaction_embedder(actors, actor_idcs, actor_ctrs)
        ids_fut = self.identifier(representation)
        ids_fut = torch.cat([ids_fut[x[1:]] for x in actor_idcs_mod])

        # get GT for supervised prediction loss
        pred_gt = torch.cat([gpu(x[1:]) for x in data['gt_preds']])

        return out, [ids_hist, ids_fut], [reconstruction_gt, pred_gt], [ego_fut_actor_idcs, actor_idcs_mod, actor_ctrs_mod]

    def get_fut_data(self, out, reconstruction_gt, ego_fut_aug_idcs, batch_num, data):
        feats, ctrs, graph = [], [], []
        '''
        len(feats) : batch_num * ego_aug_num
        feats[i].shape : (veh_num , 20, 3)
        '''
        for i in range(batch_num):
            ego_recon = reconstruction_gt[ego_fut_aug_idcs[i]]
            for j in range(ego_recon.shape[0]):
                ego_recon_tmp = ego_recon[j:j + 1]
                sur_pred = out['reg'][ego_fut_aug_idcs[i][j]][1:, 0, :, :]
                trajs = torch.cat([ego_recon_tmp, sur_pred])
                feat = torch.matmul(gpu(data['rots'][i][0]), (trajs - gpu(data['origs'][i][0])).reshape(-1, 2).T).T.reshape(-1, 20, 2)
                feat = torch.cat([feat, torch.ones_like(feat[:, :, 0:1])], dim=2)
                feats.append(feat)

                ctrs.append(data['ctrs'][i][0, :, :])
                graph_tmp = dict()
                for key in ['num_nodes',
                            'turn',
                            'control',
                            'intersect',
                            'pre',
                            'suc',
                            'left',
                            'right']:
                    graph_tmp[key] = data['graph'][i][key]
                graph_tmp['feats'] = data['graph'][i]['feats_tot'][0, :, :]
                graph_tmp['ctrs'] = data['graph'][i]['ctrs_tot'][0, :, :]
                graph.append(graph_tmp)

        fut_data = dict()
        fut_data['feats'] = feats
        fut_data['ctrs'] = ctrs
        fut_data['graph'] = graph

        return fut_data

    def data_mod(self, actors, actor_idcs, ego_fut_actors, ego_fut_aug_idcs, data):
        batch_num = len(actor_idcs)
        actors_mod = []
        actor_idcs_mod = []
        actor_ctrs_mod = []
        for i in range(batch_num):
            actors_in_batch = actors[actor_idcs[i]]
            ego_num = len(ego_fut_aug_idcs[i])
            for j in range(ego_num):
                ego_fut = ego_fut_actors[ego_fut_aug_idcs[i][j]:ego_fut_aug_idcs[i][j] + 1]
                actors_mod_tmp = torch.cat([ego_fut, actors_in_batch[1:]])
                actors_mod.append(actors_mod_tmp)
                idcs = torch.arange(actors_mod_tmp.shape[0], device=actors_mod_tmp.device)
                if len(actor_idcs_mod) == 0:
                    actor_idcs_mod.append(idcs)
                else:
                    cnt = torch.max(torch.cat(actor_idcs_mod)) + 1
                    actor_idcs_mod.append(idcs + cnt)
                actor_ctrs_mod.append(gpu(data['ctrs'][i][0]))
        actors_mod = torch.cat(actors_mod)

        return actors_mod, actor_idcs_mod, actor_ctrs_mod

def actor_gather(actors):
    batch_size = len(actors)
    num_actors = [len(x) for x in actors]

    actors = [x.transpose(1, 2) for x in actors]
    actors = torch.cat(actors, 0)

    actor_idcs = []
    count = 0
    for i in range(batch_size):
        idcs = torch.arange(count, count + num_actors[i]).to(actors.device)
        actor_idcs.append(idcs)
        count += num_actors[i]
    return actors, actor_idcs

def data_form(data):
    batch_num = len(data['city'])
    feats, ctrs, graph = [], [], []
    # city, file_name, idxs, origs, thetas, rots, gt_preds, has_preds, idx, graph, hist_feats, fut_feats, ctrs = [], [], [], [], [], [], [], [], [], [], [], [], []
    data_spread = dict()
    for i in range(batch_num):
        j = 0
        feats.append(data['hist_feats'][i][j, :, :, :])
        ctrs.append(data['ctrs'][i][j, :, :])
        graph_tmp = dict()
        for key in ['num_nodes',
                    'turn',
                    'control',
                    'intersect',
                    'pre',
                    'suc',
                    'left',
                    'right']:
            graph_tmp[key] = data['graph'][i][key]
        graph_tmp['feats'] = data['graph'][i]['feats_tot'][j, :, :]
        graph_tmp['ctrs'] = data['graph'][i]['ctrs_tot'][j, :, :]
        graph.append(graph_tmp)

    data_spread['feats'] = feats
    data_spread['ctrs'] = ctrs
    data_spread['graph'] = graph

    return data_spread


def ego_fut_data_form(data):
    batch_num = len(data['city'])
    feats, ctrs, graph = [], [], []
    # city, file_name, idxs, origs, thetas, rots, gt_preds, has_preds, idx, graph, hist_feats, fut_feats, ctrs = [], [], [], [], [], [], [], [], [], [], [], [], []
    data_spread = dict()
    for i in range(batch_num):
        j = 0

        if type(data['ego_aug'][i]['traj']) == list:
            data['ego_aug'][i]['traj'] = torch.Tensor(size=(0, 20, 2))
        ego_aug_feats = torch.zeros(size=(data['ego_aug'][i]['traj'].shape[0] + 1, 20, 3), dtype=torch.float32)
        idx_chk = []
        for k in range(data['ego_aug'][i]['traj'].shape[0] + 1):
            if k == 0:
                ego_aug_feats[k, :, :] = data['fut_feats'][i][0, 0]
                idx_chk.append(k)
            else:
                if torch.isnan(data['ego_aug'][i]['traj'][k - 1]).any():
                    pass
                else:
                    ego_aug_feats[k, :, :2] = torch.matmul(data['rots'][i][0], (data['ego_aug'][i]['traj'][k - 1] - data['origs'][i][j].reshape(-1, 2)).T).T
                    ego_aug_feats[k, :, 2] = 1.0
                    idx_chk.append(k)

        ego_aug_feats = ego_aug_feats[idx_chk]
        feats.append(ego_aug_feats)
        ctrs.append(torch.repeat_interleave(data['ctrs'][i][j, 0:1, :], ego_aug_feats.shape[0], dim=0))
        graph_tmp = dict()
        for key in ['num_nodes',
                    'turn',
                    'control',
                    'intersect',
                    'pre',
                    'suc',
                    'left',
                    'right']:
            graph_tmp[key] = data['graph'][i][key]
        graph_tmp['feats'] = data['graph'][i]['feats_tot'][j, :, :]
        graph_tmp['ctrs'] = data['graph'][i]['ctrs_tot'][j, :, :]
        graph.append(graph_tmp)

    data_spread['feats'] = feats
    data_spread['ctrs'] = ctrs
    data_spread['graph'] = graph

    return data_spread


class PredLoss(nn.Module):
    def __init__(self, config):
        super(PredLoss, self).__init__()
        self.config = config

    def forward(self, pred_out, pred_gt):
        pred_gt_mod = pred_gt.view(-1, 2)
        pred_mod = pred_out.view(-1, 2)
        non_zero_idx = torch.nonzero(pred_gt_mod[:, 0])[:, 0]
        err = (pred_gt_mod - pred_mod)[non_zero_idx]
        mse_loss = torch.norm(err, dim=1)


        loss_out = dict()
        loss_out['reg_loss'] = torch.sum(mse_loss)
        loss_out['reg_loss_cnt'] = len(non_zero_idx)

        return loss_out


class ReconstLoss(nn.Module):
    def __init__(self, config):
        super(ReconstLoss, self).__init__()
        self.config = config

    def forward(self, reconstruction_out, reconstruction_gt):
        reconst_gt_mod = reconstruction_gt.view(-1, 2)
        reconst_mod = reconstruction_out.view(-1, 2)
        non_zero_idx = torch.nonzero(reconst_gt_mod[:, 0])[:,0]
        err = (reconst_gt_mod - reconst_mod)[non_zero_idx]
        mse_loss = torch.norm(err, dim=1)

        loss_out = dict()
        loss_out['reconst_loss'] = torch.sum(mse_loss)
        loss_out['reconst_loss_cnt'] = len(non_zero_idx)

        return loss_out

class IDloss(nn.Module):
    def __init__(self, config):
        super(IDloss, self).__init__()
        self.config = config
        self.cosine_sim = nn.CosineSimilarity()
        self.pi = torch.acos(torch.zeros(1)).item() * 2

    def forward(self, ids_fut, ids_hist, ego_fut_aug_idcs, actor_idcs_mod):
        batch_num = len(ego_fut_aug_idcs)
        loss_cnt = 0
        sim_loss_tot = 0
        for i in range(batch_num):
            vehicle_num = len(actor_idcs_mod[ego_fut_aug_idcs[i][0]]) - 1
            aug_num = len(ego_fut_aug_idcs[i])
            if aug_num > 1:
                for j in range(vehicle_num):
                    anchor_idx_hist = actor_idcs_mod[ego_fut_aug_idcs[i][0]][j+1]-(ego_fut_aug_idcs[i][0]+1).unsqueeze(dim=0)
                    positive_idx_fut = torch.cat([(actor_idcs_mod[ego_fut_aug_idcs[i][k]][j+1]-(ego_fut_aug_idcs[i][k]+1)).unsqueeze(dim=0) for k in range(len(ego_fut_aug_idcs[i])) if k > 0])
                    negative_idx_fut = torch.cat((actor_idcs_mod[ego_fut_aug_idcs[i][0]][1: j+1]-(ego_fut_aug_idcs[i][0]+1), actor_idcs_mod[ego_fut_aug_idcs[i][0]][j+2:]-(ego_fut_aug_idcs[i][0]+1)))

                    anchor = ids_hist[anchor_idx_hist.item():anchor_idx_hist.item()+1]
                    pos_sample = ids_fut[[positive_idx_fut[i].item() for i in range(len(positive_idx_fut))]]
                    neg_sample = ids_fut[[negative_idx_fut[i].item() for i in range(len(negative_idx_fut))]]

                    anc_pos = self.cosine_sim(anchor, pos_sample) + 1
                    anc_neg = self.cosine_sim(anchor, neg_sample) + 1

                    sim_loss = -torch.log(torch.sum(anc_pos)/(torch.sum(anc_pos) + torch.sum(anc_neg)))
                    sim_loss_tot = sim_loss_tot + sim_loss
                    loss_cnt = loss_cnt + 1

        loss_out = dict()
        loss_out['repres_loss'] = sim_loss_tot
        loss_out['repres_loss_cnt'] = loss_cnt

        return loss_out


class Loss(nn.Module):
    def __init__(self, config):
        super(Loss, self).__init__()
        self.config = config
        self.prediction_loss = PredLoss(config)
        self.reconstruction_loss = ReconstLoss(config)
        self.ID_loss = IDloss(config)

    def forward(self, pred_out, pred_gt, reconstruction_out, reconstruction_gt, ids_hist, ids_fut, ego_fut_aug_idcs, actor_idcs_mod):
        Predloss = self.prediction_loss(pred_out, pred_gt)
        Reconsloss = self.reconstruction_loss(reconstruction_out, reconstruction_gt)
        Represloss = self.ID_loss(ids_fut, ids_hist, ego_fut_aug_idcs, actor_idcs_mod)

        loss = dict()
        for key in Predloss:
            loss[key] = Predloss[key]
        for key in Reconsloss:
            loss[key] = Reconsloss[key]
        for key in Represloss:
            loss[key] = Represloss[key]

        loss_out = self.config["reg_coef"] * (loss['reg_loss'] / (loss["reg_loss_cnt"] + 1e-10)) + \
                   self.config["repres_coef"] * (loss['repres_loss'] / (loss["repres_loss_cnt"] + 1e-10)) + \
                   self.config["reconst_coef"] * (loss['reconst_loss'] / (loss["reconst_loss_cnt"] + 1e-10))

        return loss_out, loss


class PostProcess(nn.Module):
    def __init__(self, config):
        super(PostProcess, self).__init__()
        self.config = config

    def forward(self, out, ego_fut_aug_idcs, data):
        post_out = dict()
        post_out["preds"] = [out['reg'][x[0]][1].detach().cpu().numpy() for x in ego_fut_aug_idcs]
        post_out["gt_preds"] = [x[1:2].numpy() for x in data["gt_preds"]]
        post_out["has_preds"] = [x[1:2].numpy() for x in data["has_preds"]]
        return post_out

    def append(self, metrics, loss_out, loss_orig, post_out):
        if len(metrics.keys()) == 0:
            for key in loss_orig:
                metrics[key] = 0.0
            metrics['loss'] = 0.0
            for key in post_out:
                metrics[key] = []

        for key in loss_orig:
            if isinstance(loss_orig[key], torch.Tensor):
                metrics[key] += loss_orig[key].item()
            else:
                metrics[key] += loss_orig[key]

        for key in post_out:
            metrics[key] += post_out[key]
        metrics['loss'] += loss_out.item()
        return metrics

    def display(self, metrics, dt, epoch, lr=None):
        """Every display-iters print training/val information"""
        if lr is not None:
            print("Epoch %3.3f, lr %.5f, time %3.2f" % (epoch, lr, dt))
        else:
            print(
                "************************* Validation, time %3.2f *************************"
                % dt
            )

        reg = metrics["reg_loss"] / (metrics["reg_loss_cnt"] + 1e-10)
        reconst = metrics["reconst_loss"] / (metrics["reconst_loss_cnt"] + 1e-10)
        repres = metrics["repres_loss"] / (metrics["repres_loss_cnt"] + 1e-10)
        loss = self.config['reg_coef'] * reg + \
               self.config['reconst_coef'] * reconst + \
               self.config["repres_coef"] * repres

        preds = np.concatenate(metrics["preds"], 0)
        gt_preds = np.concatenate(metrics["gt_preds"], 0)
        has_preds = np.concatenate(metrics["has_preds"], 0)

        idx = []
        for i in range(has_preds.shape[0]):
            if has_preds[i].all():
                idx.append(i)

        ade1, fde1 = pred_metrics(preds[idx], gt_preds[idx], has_preds[idx])

        print(
            "loss %2.4f reg %2.4f reconst %2.4f repres %2.4f :: ade1 %2.4f, fde1 %2.4f"
            % (loss, reg, reconst, repres, ade1, fde1)
        )
        print()


def pred_metrics(preds, gt_preds, has_preds):
    assert has_preds.all()
    preds = np.asarray(preds, np.float32)
    gt_preds = np.asarray(gt_preds, np.float32)

    """batch_size x num_mods x num_preds"""
    err = np.sqrt(((preds - gt_preds) ** 2).sum(2))

    ade1 = err.mean()
    fde1 = err[:, -1].mean()
    return ade1, fde1

def get_model(config):
    enc_net = enc(config)
    emb_net = emb(config)
    dec_net = dec(config)
    id_net = id(config)

    net = Net(config, enc_net, emb_net, dec_net, id_net)
    net = net.cuda()

    loss = Loss(config).cuda()
    post_process = PostProcess(config).cuda()

    params = net.parameters()
    opt = Optimizer(params, config)

    return net, loss, post_process, opt