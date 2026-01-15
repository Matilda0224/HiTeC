from .dataset import (
    TAHGDataset,
)
import os.path as osp
import torch

def load_splits(dataset_name, num_splits=20, device='cuda'):
        tahg_dir =  f'tahg_datasets/{dataset_name}'
        masks= []
        for seed in range(num_splits):
            split_path = osp.join(tahg_dir, 'splits', f'{seed}.pt')
            mask_dic = torch.load(split_path)
            mask = [mask_dic['train_mask'].to(device), mask_dic['val_mask'].to(device), mask_dic['test_mask'].to(device)]
            masks.append(mask)
        return masks

class DatasetLoader(object):
    def __init__(self):
        pass

    def load(self, dataset_name: str = 'cora',  device:str = 'cpu'):
        return TAHGDataset(dataset_name = dataset_name, device = device)
    
