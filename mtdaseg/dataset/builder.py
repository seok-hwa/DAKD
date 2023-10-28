import copy
import random
from collections import deque

from torch.utils import data

from .idd import IDDDataset
from .gta5 import GTA5Dataset
from .mapillary import MapillaryDataset
from .cityscapes import CityscapesDataset
from mtdaseg.utils import project_root


DATASET = {
    'IDD': 'IDDDataset',
    'GTA5': 'GTA5Dataset',
    'Mapillary': 'MapillaryDataset',
    'Cityscapes': 'CityscapesDataset',
}
DATA_DIR = {
    'IDD': 'data/IDD_Segmentation',
    'GTA5': 'data/GTA5',
    'Mapillary': 'data/Mapillary',
    'Cityscapes': 'data/Cityscapes',
}


def get_tgt_dataloader(tgt_loader, tgt_loaders, cfg):
    ''' Select a target dataloader in the mult-target dataloaders '''

    if tgt_loader: tgt_loaders.append(tgt_loader)
    tgt_loader = tgt_loaders.popleft()

    return tgt_loader, enumerate(tgt_loader), cfg.TRAIN.CUR_ITER


def get_mtda_loader(cfg, train=True):
    ''' Return the dataloaders for multi-target DA '''

    # Get source and target datasets
    src_dataset, tgt_datasets = get_mtda_datasets(cfg, train)

    # Make source dataloader
    src_loader = data.DataLoader(src_dataset,
                                 batch_size=cfg.DATALOADER.BATCH_SIZE_SOURCE if train else cfg.TEST.BATCH_SIZE,
                                 shuffle=True,
                                 num_workers=cfg.DATALOADER.NUM_WORKERS,
                                 pin_memory=True,
                                 drop_last=True)
    
    # Make target dataloaders
    tgt_loaders = deque()

    for tgt_dataset in tgt_datasets:
        tgt_loader = data.DataLoader(tgt_dataset,
                                    batch_size=cfg.DATALOADER.BATCH_SIZE_TARGET if train else cfg.TEST.BATCH_SIZE,
                                    shuffle=True,
                                    num_workers=cfg.DATALOADER.NUM_WORKERS,
                                    pin_memory=True,
                                    drop_last=True)
        tgt_loaders.append(tgt_loader)
    
    return src_loader, tgt_loaders


def get_mtda_datasets(cfg, train=True):
    ''' Return the datasets for multi-target DA '''
    
    # Get a source dataset
    source = eval(DATASET[cfg.DATASETS.SOURCE[0]])(
        root=project_root / DATA_DIR[cfg.DATASETS.SOURCE[0]],
        train=train,
        num_classes=cfg.DATASETS.NUM_CLASSES,
        crop_size=cfg.INPUT.INPUT_SIZE_SOURCE if train else cfg.TEST.INPUT_SIZE,
        mean=cfg.INPUT.IMG_MEAN,
    )
    
    # Get target datasets
    targets = []
    
    for idx, target_name in enumerate(cfg.DATASETS.TARGET):
        target = eval(DATASET[target_name])(
            root=project_root / DATA_DIR[target_name],
            train=train,
            num_classes=cfg.DATASETS.NUM_CLASSES,
            crop_size=cfg.INPUT.INPUT_SIZE_TARGET if train else cfg.TEST.INPUT_SIZE,
            mean=cfg.INPUT.IMG_MEAN,
            domain=idx,
        )
        targets.append(target)
        
    return source, targets