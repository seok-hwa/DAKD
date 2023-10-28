import numpy as np
from easydict import EasyDict

from . import project_root
from .serialization import load_yaml


cfg = EasyDict()

# Configs for datasets
cfg.DATASETS = EasyDict()
cfg.DATASETS.SOURCE = ['GTA5']
cfg.DATASETS.TARGET = ['Cityscapes']
cfg.DATASETS.NUM_CLASSES = 19

# Configs for dataloader
cfg.DATALOADER = EasyDict()
cfg.DATALOADER.BATCH_SIZE_SOURCE = 4
cfg.DATALOADER.BATCH_SIZE_TARGET = 4
cfg.DATALOADER.NUM_WORKERS = 4

# Configs for input data
cfg.INPUT = EasyDict()
cfg.INPUT.IGNORE_LABEL = 255
cfg.INPUT.INPUT_SIZE_SOURCE = (1024, 512)
cfg.INPUT.INPUT_SIZE_TARGET = (1024, 512)
cfg.INPUT.IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

# Configs for models
cfg.MODEL = EasyDict()
cfg.MODEL.MODEL = 'DeepLabv2'
cfg.MODEL.MULTI_LEVEL = True
cfg.MODEL.GPU_ID = 0
cfg.MODEL.RESTORE_FROM = ''

# Configs for output directory
cfg.OUTPUT_DIR = EasyDict()
cfg.OUTPUT_DIR.ROOT = project_root / 'outputs'
cfg.OUTPUT_DIR.EXP = cfg.OUTPUT_DIR.ROOT / 'exp'
cfg.OUTPUT_DIR.SNAPSHOT = str(cfg.OUTPUT_DIR.EXP / 'snapshots')

# Configs for domain adaptation
cfg.DA = EasyDict()
cfg.DA.DA_TYPE = 'SourceOnly'
cfg.DA.DA_METHOD = ''
cfg.DA.BRIDGE_TYPE = ''
cfg.DA.LS_TYPE = 'class'
cfg.DA.CS_TYPE = 'random'
cfg.DA.REMOVE_BOUNDARY_ARTIFACT = False
cfg.DA.DOMAIN_LABEL = True

# Configs for data augmentations
cfg.AUG = EasyDict()
cfg.AUG.KERNEL_SIZE = 30
cfg.AUG.NUM_AUGMENTATION = 10
cfg.AUG.TRANSLATION_RANGE = [-50, 50]
cfg.AUG.SCALE_RANGE = [1.0, 1.5]

# Configs for training
cfg.TRAIN = EasyDict()
cfg.TRAIN.LEARNING_RATE = 2.5e-4
cfg.TRAIN.MOMENTUM = 0.9
cfg.TRAIN.WEIGHT_DECAY = 0.0005
cfg.TRAIN.POWER = 0.9
cfg.TRAIN.LAMBDA_SEG_MAIN = 1.0
cfg.TRAIN.LAMBDA_SEG_AUX = 0.1
cfg.TRAIN.LAMBDA_BRIDGE = 1.0
cfg.TRAIN.LAMBDA_REGULARIZATION = 1
cfg.TRAIN.CONF_THRESHOLD = 0.968
cfg.TRAIN.EMA_STEP = 1
cfg.TRAIN.KEEP_RATIO = 0.99
cfg.TRAIN.INIT_ITER = 0
cfg.TRAIN.CUR_ITER = 0
cfg.TRAIN.MAX_ITERS = 250000
cfg.TRAIN.EARLY_STOP = 120000
cfg.TRAIN.LOG_STEP = 100
cfg.TRAIN.SNAPSHOT_STEP = 2000
cfg.TRAIN.RANDOM_SEED = 1234

# Configs for testing
cfg.TEST = EasyDict()
cfg.TEST.EVAL_STEP = 2000
cfg.TEST.BATCH_SIZE = 1
cfg.TEST.INPUT_SIZE = (1024, 512)


def _merge_a_into_b(a, b):
    ''' Merge config directory a into config directory b '''
    
    assert isinstance(a, EasyDict), 'The type of config directory must be \'EasyDict\'.'
    
    for k, v in a.items():
        assert k in b.keys(), 'a must specify keys that are in b.'
        
        # The types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(f'Type mismatch ({type(b[k])} vs. {type(v)}) '
                                 f'for config key: {k}')
        
        # Merge dicts recursively
        if isinstance(v, EasyDict):
            try:
                _merge_a_into_b(a[k], b[k])
            except Exception:
                print(f'Error under config key: {k}')
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    ''' Load a config file and merge it into the default options '''
    
    yaml_cfg = EasyDict(load_yaml(filename))
    _merge_a_into_b(yaml_cfg, cfg)