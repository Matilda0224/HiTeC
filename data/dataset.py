from typing import Optional
import os.path as osp
import pickle

import torch
from torch_scatter import scatter_add
from torch.utils.data import random_split


class BaseDataset(object):
    def __init__(self, name: str,  device: str):
        self.name = name
        self.device = device

        self.dataset_dir = f'tahg_datasets/{self.name}'
        self.split_dir = osp.join(self.dataset_dir, 'splits')
        
        self.load_dataset()
        self.preprocess_dataset()
      

    def load_dataset(self):
        
        with open(osp.join(self.dataset_dir, 'features.pt'), 'rb') as f:
            self.features = torch.load(f, weights_only = False)
   
        with open(osp.join(self.dataset_dir, 'hypergraph_dict.pt'), 'rb') as f:
            self.hypergraph = torch.load(f, weights_only = False)

        with open(osp.join(self.dataset_dir, 'labels.pt'), 'rb') as f:
            self.labels = torch.load(f, weights_only = False)

        with open(osp.join(self.dataset_dir, 'texts.pt'), 'rb') as f:
            self.texts = torch.load(f, weights_only = False)

    def load_splits(self, seed: int):
        with open(osp.join(self.split_dir, f'{seed}.pt'), 'rb') as f:
            splits = torch.load(f, weights_only = False)
        return splits

    def preprocess_dataset(self):
        edge_set = set(self.hypergraph.keys())
        num_nodes = len(self.texts)

        edge_to_num = {}
        num_to_edge = {}
        incidence_matrix = []
        processed_hypergraph = {}

        for new_edge_id, edge in enumerate(self.hypergraph.keys()):
            edge_to_num[edge] = new_edge_id
            num_to_edge[new_edge_id] = edge
            processed_hypergraph[new_edge_id] = self.hypergraph[edge]
            for node in self.hypergraph[edge]:
                incidence_matrix.append([node, new_edge_id])

        self.edge_to_num = edge_to_num
        self.num_to_edge = num_to_edge
        self.processed_hypergraph = processed_hypergraph
        self.hyperedge_index = torch.LongTensor(incidence_matrix).T.contiguous()
        self.num_nodes = num_nodes
        self.num_edges = len(self.hypergraph)

        self.features = self.features 
        self.labels = torch.LongTensor(self.labels)

        weight = torch.ones(self.num_edges)
        Dn = scatter_add(weight[self.hyperedge_index[1]], self.hyperedge_index[0], dim=0, dim_size=self.num_nodes)
        De = scatter_add(torch.ones(self.hyperedge_index.shape[1]), self.hyperedge_index[1], dim=0, dim_size=self.num_edges)

        self.to(self.device)

    def to(self, device: str):
        self.features = self.features.to(device) # features不计算
        self.hyperedge_index = self.hyperedge_index.to(device)
        self.labels = self.labels.to(device)
        self.device = device
        return self

    def generate_random_split(self, train_ratio: float = 0.1, val_ratio: float = 0.1,
                              seed: Optional[int] = None, use_stored_split: bool = True):
        if use_stored_split:
            splits = self.load_splits(seed)
            train_mask = torch.tensor(splits['train_mask'], dtype=torch.bool, device=self.device)
            val_mask = torch.tensor(splits['val_mask'], dtype=torch.bool, device=self.device)
            test_mask = torch.tensor(splits['test_mask'], dtype=torch.bool, device=self.device)

        else:
            num_train = int(self.num_nodes * train_ratio)
            num_val = int(self.num_nodes * val_ratio)
            num_test = self.num_nodes - (num_train + num_val)

            if seed is not None:
                generator = torch.Generator().manual_seed(seed)
            else:
                generator = torch.default_generator

            train_set, val_set, test_set = random_split(
                torch.arange(0, self.num_nodes), (num_train, num_val, num_test), 
                generator=generator)
            train_idx, val_idx, test_idx = \
                train_set.indices, val_set.indices, test_set.indices
            train_mask = torch.zeros((self.num_nodes,), device=self.device).to(torch.bool)
            val_mask = torch.zeros((self.num_nodes,), device=self.device).to(torch.bool)
            test_mask = torch.zeros((self.num_nodes,), device=self.device).to(torch.bool)

            train_mask[train_idx] = True
            val_mask[val_idx] = True
            test_mask[test_idx] = True

        return [train_mask, val_mask, test_mask]

class TAHGDataset(BaseDataset):
    def __init__(self, dataset_name,   device, **kwargs):
        super().__init__( dataset_name , device, **kwargs)

