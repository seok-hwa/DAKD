import os
import pprint
import argparse

import torch

from mtdaseg.utils import project_root
from mtdaseg.core.eval import eval_model
from mtdaseg.model.deeplabv2 import get_deeplab_v2
from mtdaseg.dataset.builder import get_mtda_loader
from mtdaseg.utils.config import cfg, cfg_from_file


def parse_args():
    ''' Get arguments from user '''
    
    parser = argparse.ArgumentParser(description="Code for domain adaptation (DA) training")
    parser.add_argument('--cfg', type=str, default=None,
                        help='optional config file', )
    parser.add_argument('--ckp', type=str, default=None,
                        help='checkpoint file path', )
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID', )
    return parser.parse_args()


def main():
    ''' Main function for test '''
    
    # Load arguments
    args = parse_args()

    assert args.cfg is not None, 'Missing cfg file'
    cfg_from_file(args.cfg)

    # Set the GPU ID to use
    cfg.MODEL.GPU_ID = args.gpu

    # Set the pre-trained model path
    cfg.MODEL.RESTORE_FROM = str(project_root / args.ckp)
    assert os.path.exists(cfg.MODEL.RESTORE_FROM), 'Checkpoint file path is not valid.'

    print('Using config:')
    pprint.pprint(cfg)

    # Load the segmentation model
    if cfg.MODEL.MODEL == 'DeepLabv2':
        model = get_deeplab_v2(num_classes=cfg.DATASETS.NUM_CLASSES, multi_level=cfg.MODEL.MULTI_LEVEL).to(cfg.MODEL.GPU_ID)
        model.load_state_dict(torch.load(cfg.MODEL.RESTORE_FROM, map_location='cuda:{}'.format(cfg.MODEL.GPU_ID)))
        model.eval()
    else:
        raise NotImplementedError(f"Not yet supported {cfg.TRAIN.MODEL}")

    # Get the dataloader
    dataloader = get_mtda_loader(cfg, train=False)[1]

    # Evaluate
    eval_model(cfg, model, dataloader)


if __name__ == '__main__':
    main()