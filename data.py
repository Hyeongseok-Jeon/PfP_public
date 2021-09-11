# Copyright (c) 2020 Uber Technologies, Inc.
# Please check LICENSE for more detail


import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import sparse
import os
import copy
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap
from skimage.transform import rotate

class ArgoDataset(Dataset):
    def __init__(self, split, config, train=True):
        self.config = config
        self.train = train

        if 'preprocess' in config and config['preprocess']:
            if train:
                self.split = np.load(self.config['preprocess_train'], allow_pickle=True)
            else:
                self.split = np.load(self.config['preprocess_val'], allow_pickle=True)
        else:
            self.avl = ArgoverseForecastingLoader(split)
            self.avl.seq_list = sorted(self.avl.seq_list)
            self.am = ArgoverseMap()

    def __getitem__(self, idx):
        if 'preprocess' in self.config and self.config['preprocess']:
            data = self.split[idx]

            new_data = dict()
            veh_num = data['hist_feats'].shape[0]
            for key in [
                "city",
                "file_name",
                "idxs",
                "origs",
                "thetas",
                "rots",
                "gt_preds",
                "has_preds",
                "idx",
                "graph",
                'hist_feats',
                'fut_feats',
                'ctrs',
                'ego_aug',
                'ego_maneuver',
                'target_maneuver',
            ]:
                if key in data:
                    if veh_num > 20:
                        data_tmp = ref_copy(data[key])
                        if key in ["idxs", "origs", "thetas", "rots", "gt_preds", "has_preds"]:
                            new_data[key] = data_tmp[:20]
                        elif key in ["hist_feats","fut_feats","ctrs"]:
                            new_data[key] = data_tmp[:20, :20]
                        elif key in ["graph"]:
                            data_tmp['ctrs_tot'] = data_tmp['ctrs_tot'][:20]
                            data_tmp['feats_tot'] = data_tmp['feats_tot'][:20]
                            new_data[key] = data_tmp
                        else:
                            new_data[key] = data_tmp
                    else:
                        new_data[key] = ref_copy(data[key])
            data = new_data
            # data = self.get_feats(data)
            # data = self.get_graph_feats(data)
            return data

        else:
            print('please use preprocessed data')

    def __len__(self):
        if 'preprocess' in self.config and self.config['preprocess']:
            return len(self.split)
        else:
            return len(self.avl)

def ref_copy(data):
    if isinstance(data, list):
        return [ref_copy(x) for x in data]
    if isinstance(data, dict):
        d = dict()
        for key in data:
            d[key] = ref_copy(data[key])
        return d
    return data


def collate_fn(batch):
    batch = from_numpy(batch)
    return_batch = dict()
    # Batching by use a list for non-fixed size
    for key in batch[0].keys():
        return_batch[key] = [x[key] for x in batch]
    return return_batch


def from_numpy(data):
    """Recursively transform numpy.ndarray to torch.Tensor.
    """
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = from_numpy(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [from_numpy(x) for x in data]
    if isinstance(data, np.ndarray):
        """Pytorch now has bool type."""
        data = torch.from_numpy(data)
    return data


def cat(batch):
    if torch.is_tensor(batch[0]):
        batch = [x.unsqueeze(0) for x in batch]
        return_batch = torch.cat(batch, 0)
    elif isinstance(batch[0], list) or isinstance(batch[0], tuple):
        batch = zip(*batch)
        return_batch = [cat(x) for x in batch]
    elif isinstance(batch[0], dict):
        return_batch = dict()
        for key in batch[0].keys():
            return_batch[key] = cat([x[key] for x in batch])
    else:
        return_batch = batch
    return return_batch
