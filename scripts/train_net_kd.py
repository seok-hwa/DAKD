import os
import pprint
import random
import argparse
import warnings

import yaml
import torch
import gc
import numpy as np

from mtdaseg.utils import project_root
from mtdaseg.utils.config import cfg, cfg_from_file
from mtdaseg.dataset.builder import get_mtda_loader
from mtdaseg.model.deeplabv2 import get_deeplab_v2
from mtdaseg.core.train_kd import train_domain_adaptation


warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore")


def parse_args():
    ''' Get arguments from user '''
    
    parser = argparse.ArgumentParser(description="Code for domain adaptation (DA) training")
    parser.add_argument('--cfg', type=str, default=None,
                        help='optional config file', )
    parser.add_argument('--work-dir', type=str, default=None,
                        help='working directory path', )
    parser.add_argument('--ckp', type=str, default=None,
                        help='checkpoint file path', )
    parser.add_argument('--init-iter', type=int, default=0,
                        help='initial interation', )
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID', )
    parser.add_argument("--random-train", action="store_true",
                        help="not fixing random seed.")
    return parser.parse_args()


def main():
    ''' Main function for Training '''
    gc.collect()
    torch.cuda.empty_cache()
    # Load arguments
    args = parse_args()

    assert args.cfg is not None, 'Missing cfg file'
    cfg_from_file(args.cfg)

    # Set the GPU ID to use
    cfg.MODEL.GPU_ID = args.gpu

    # Set the initial iteration
    cfg.TRAIN.INIT_ITER = args.init_iter

    # Set the pre-trained model path
    if args.ckp:
        cfg.MODEL.RESTORE_FROM = str(project_root / args.ckp)
        assert os.path.exists(cfg.MODEL.RESTORE_FROM), 'Checkpoint file path is not valid.'

    # Set the working directory path for saving weight parameters
    assert args.work_dir, 'Working direcoty path is not valid.'
    cfg.OUTPUT_DIR.EXP = cfg.OUTPUT_DIR.ROOT / args.work_dir
    cfg.OUTPUT_DIR.SNAPSHOT = str(cfg.OUTPUT_DIR.EXP / 'snapshots')
    os.makedirs(cfg.OUTPUT_DIR.SNAPSHOT, exist_ok=True)

    print('Using config:')
    pprint.pprint(cfg)

    # Initialization
    if not args.random_train:
        torch.manual_seed(cfg.TRAIN.RANDOM_SEED)
        torch.cuda.manual_seed(cfg.TRAIN.RANDOM_SEED)
        np.random.seed(cfg.TRAIN.RANDOM_SEED)
        random.seed(cfg.TRAIN.RANDOM_SEED)

    # Load the segmentation model
    assert os.path.exists(cfg.MODEL.RESTORE_FROM), f'Missing init model {cfg.MODEL.RESTORE_FROM}'
    if cfg.MODEL.MODEL == 'DeepLabv2':
        model, st_model = get_deeplab_v2(num_classes=cfg.DATASETS.NUM_CLASSES, multi_level=cfg.MODEL.MULTI_LEVEL)
        # saved_state_dict = torch.load(cfg.MODEL.RESTORE_FROM, map_location='cuda:{}'.format(cfg.MODEL.GPU_ID))
        # if 'DeepLab_resnet_pretrained_imagenet' in cfg.MODEL.RESTORE_FROM:
        #     new_params = model.state_dict().copy()
        #     for i in saved_state_dict:
        #         i_parts = i.split('.')
        #         if not i_parts[1] == 'layer5':
        #             new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
        #     model.load_state_dict(new_params)
        # else:
        #     model.load_state_dict(saved_state_dict)
    else:
        raise NotImplementedError(f"Not yet supported {cfg.TRAIN.MODEL}")

    model = model.cuda(cfg.MODEL.GPU_ID)
    st_model = st_model.cuda(cfg.MODEL.GPU_ID)

    print('\nModel was loaded!')

    # Load the dataloaders
    src_loader, tgt_loader = get_mtda_loader(cfg, train=True)
    print('Dataloaders were loaded!')

    with open(os.path.join(cfg.OUTPUT_DIR.EXP, 'train_cfg.yml'), 'w') as yaml_file:
        yaml.dump(cfg, yaml_file, default_flow_style=False)

    # DA training
    print('Starting the training!\n')
    train_domain_adaptation(model, st_model, src_loader, tgt_loader, cfg)


if __name__ == '__main__':
    main()