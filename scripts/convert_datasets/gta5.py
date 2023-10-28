import argparse

import mmcv
import numpy as np
from PIL import Image

from mtdaseg.utils import project_root


CITYSCAPES_IDX_7 = {
    7: 0,
    8: 0,
    11: 1,
    12: 1,
    13: 1,
    17: 2,
    19: 2,
    20: 2,
    21: 3,
    22: 0,
    23: 4,
    24: 5,
    25: 5,
    26: 6,
    27: 6,
    28: 6,
    31: 6,
    32: 6,
    33: 6,
}
CITYSCAPES_IDX_19 = {
    7: 0,
    8: 1,
    11: 2,
    12: 3,
    13: 4,
    17: 5,
    19: 6,
    20: 7,
    21: 8,
    22: 9,
    23: 10,
    24: 11,
    25: 12,
    26: 13,
    27: 14,
    28: 15,
    31: 16,
    32: 17,
    33: 18,
}


def convert_to_trainids_7(file):
    ''' Convert annotations to TrainIds format with 7 classes '''
    
    label = np.asarray(Image.open(file))
    new_label = 255 * np.ones(label.shape, dtype=np.uint8)
    
    # Convert to Cityscapes TrainIds format
    for k, v in CITYSCAPES_IDX_7.items():
        mask = (label == k)
        new_label[mask] = v
        
    # Save a new label
    filename = str(file).replace('.png', '_labelTrainIds7.png')
    Image.fromarray(new_label, mode='L').save(filename)


def convert_to_trainids_19(file):
    ''' Convert annotations to TrainIds format with 19 classes '''
    
    label = np.asarray(Image.open(file))
    new_label = 255 * np.ones(label.shape, dtype=np.uint8)
    
    # Convert to Cityscapes TrainIds format
    for k, v in CITYSCAPES_IDX_19.items():
        mask = (label == k)
        new_label[mask] = v
        
    # Save a new label
    filename = str(file).replace('.png', '_labelTrainIds19.png')
    Image.fromarray(new_label, mode='L').save(filename)


def parse_args():
    ''' Get arguments from user '''
    
    parser = argparse.ArgumentParser(
        description='Convert GTA5 annotations to TrainIds')
    parser.add_argument(
        'data_path', help='GTA5 data path')
    parser.add_argument(
        '--num-classes', default=19, type=int, help='number of categories')
    parser.add_argument(
        '--nproc', default=8, type=int, help='number of process')
    args = parser.parse_args()
    
    return args


def main():
    ''' The main function for converting annotations to Cityscapes TrainIds format '''
    
    args = parse_args()
    assert args.num_classes in [7, 19], 'The number of categories must be 7 or 19.'
    
    # Get a list of ground-truth file paths
    data_path = project_root / args.data_path
    gt_paths = sorted(data_path.glob('labels/*.png'))
    gt_paths = [gt_path for gt_path in gt_paths if '_labelTrainIds' not in str(gt_path)]
    
    # Convert annotations to Cityscapes TrainIds format
    if args.nproc > 1:
        mmcv.track_parallel_progress(
            convert_to_trainids_19 if args.num_classes == 19 else convert_to_trainids_7,
            gt_paths,
            args.nproc,
        )
    else:
        mmcv.track_progress(
            convert_to_trainids_19 if args.num_classes == 19 else convert_to_trainids_7, 
            gt_paths,
        )


if __name__ == '__main__':
    main()