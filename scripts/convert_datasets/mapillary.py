import argparse

import cv2
import mmcv
import numpy as np
from PIL import Image

from mtdaseg.utils import project_root


CITYSCAPES_IDX_7 = np.array([
    255, 255,   1,   1,   1,   1,   1,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   1,   1,   1,   5,   5,   5, 
      5,   0,   0, 255, 255,   4, 255,   0,   3, 255, 255, 
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255,   2, 
      2,   2,   2,   2,   2,   2,   2, 255,   6,   6,   6, 
      6,   6,   6,   6,   6,   6,   6,   6, 255, 255, 255, 
])
CITYSCAPES_IDX_19 = np.array([
    255, 255, 255,   4, 255, 255,   3, 255, 255, 255, 255, 
    255, 255,   0, 255,   1, 255,   2, 255,  11,  12,  12, 
     12, 255, 255, 255, 255,  10, 255,   9,   8, 255, 255, 
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 
    255,   5, 255, 255,   6,   7,   7, 255,  18, 255,  15, 
     13, 255,  17,  16, 255, 255,  14, 255, 255, 255, 255,
])


def convert_to_trainids_7(file):
    ''' Convert annotations to TrainIds format with 7 classes '''
    
    # Convert to Cityscapes TrainIds format
    mask = np.asarray(Image.open(file))
    mask = CITYSCAPES_IDX_7[mask]
    
    # Save a mask
    filename =  str(file).replace('.png', '_labelTrainIds7.png')
    cv2.imwrite(filename, mask.astype(np.uint8))
    
    
def convert_to_trainids_19(file):
    ''' Convert annotations to TrainIds format with 19 classes '''
    
    # Convert to Cityscapes TrainIds format
    mask = np.asarray(Image.open(file))
    mask = CITYSCAPES_IDX_19[mask]
    
    # Save a mask
    filename =  str(file).replace('.png', '_labelTrainIds19.png')
    cv2.imwrite(filename, mask.astype(np.uint8))


def parse_args():
    ''' Get arguments from user '''
    
    parser = argparse.ArgumentParser(
        description='Convert Mapillary annotations to TrainIds')
    parser.add_argument(
        'data_path', help='Mapillary data path')
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
    gt_paths = sorted(data_path.glob('*/labels/*.png'))
    gt_paths = [gt_path for gt_path in gt_paths if '_labelTrainIds' not in str(gt_path)]
    
    # Convert json files to label images
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