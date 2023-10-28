import argparse

import cv2
import mmcv
import numpy as np
from PIL import Image, ImageDraw

from mtdaseg.utils import project_root
from mtdaseg.utils.serialization import load_json


CITYSCAPES_IDX_7 = {
    'unlabeled': 255, 
    'ego vehicle': 255, 
    'rectification border': 255, 
    'out of roi': 255, 
    'static': 255, 
    'dynamic': 255, 
    'ground': 255, 
    'road': 0, 
    'sidewalk': 0, 
    'parking': 255, 
    'rail track': 255, 
    'building': 1, 
    'wall': 1, 
    'fence': 1, 
    'guard rail': 255, 
    'bridge': 255, 
    'tunnel': 255, 
    'pole': 2, 
    'traffic light': 2, 
    'traffic sign': 2, 
    'vegetation': 3, 
    'terrain': 3, 
    'sky': 4, 
    'person': 5, 
    'rider': 5, 
    'car': 6,
    'truck': 6, 
    'bus': 6, 
    'caravan': 6, 
    'trailer': 6, 
    'train': 6, 
    'motorcycle': 6, 
    'bicycle': 6, 
    'license plate': 255, 
}
CITYSCAPES_IDX_19 = {
    'unlabeled': 255, 
    'ego vehicle': 255, 
    'rectification border': 255, 
    'out of roi': 255, 
    'static': 255, 
    'dynamic': 255, 
    'ground': 255, 
    'road': 0, 
    'sidewalk': 1, 
    'parking': 255, 
    'rail track': 255, 
    'building': 2, 
    'wall': 3, 
    'fence': 4, 
    'guard rail': 255, 
    'bridge': 255, 
    'tunnel': 255, 
    'pole': 5, 
    'traffic light': 6, 
    'traffic sign': 7, 
    'vegetation': 8, 
    'terrain': 9, 
    'sky': 10, 
    'person': 11, 
    'rider': 12, 
    'car': 13,
    'truck': 14, 
    'bus': 15, 
    'caravan': 255, 
    'trailer': 255, 
    'train': 16, 
    'motorcycle': 17, 
    'bicycle': 18, 
    'license plate': 255, 
}


def convert_to_trainids_7(file):
    ''' Convert annotations to TrainIds format with 7 classes '''

    # Get label information from a JSON file
    labels = load_json(file)
    
    width, height = labels['imgWidth'], labels['imgHeight']
    objects = labels['objects']
    
    # Create a new mask
    mask = Image.new(mode='I', size=(width, height), color=255)
    drawer = ImageDraw.Draw(mask)
    
    # Drow polygons on the mask
    for obj in objects:
        id = CITYSCAPES_IDX_7[obj['label'].replace('group', '')]
        polygon = [(xy[0], xy[1]) for xy in obj['polygon']]
        
        drawer.polygon(polygon, fill=id)
        
    # Save a mask
    mask = np.asarray(mask)
    mask_name = str(file).replace('_polygons.json', '_labelTrainIds7.png')
    cv2.imwrite(mask_name, mask.astype(np.uint8))
    
    # Release resorces
    del drawer
    
    
def convert_to_trainids_19(file):
    ''' Convert annotations to TrainIds format with 19 classes '''
    
    # Get label information from a JSON file
    labels = load_json(file)
    
    width, height = labels['imgWidth'], labels['imgHeight']
    objects = labels['objects']
    
    # Create a new mask
    mask = Image.new(mode='I', size=(width, height), color=255)
    drawer = ImageDraw.Draw(mask)
    
    # Drow polygons on the mask
    for obj in objects:
        id = CITYSCAPES_IDX_19[obj['label'].replace('group', '')]
        polygon = [(xy[0], xy[1]) for xy in obj['polygon']]
        
        drawer.polygon(polygon, fill=id)
        
    # Save a mask
    mask = np.asarray(mask)

    mask_name = str(file).replace('_polygons.json', '_labelTrainIds19.png')
    cv2.imwrite(mask_name, mask.astype(np.uint8))
    
    # Release resorces
    del drawer


def parse_args():
    ''' Get arguments from user '''
    
    parser = argparse.ArgumentParser(
        description='Convert IDD annotations to TrainIds')
    parser.add_argument(
        'data_path', help='IDD data path')
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
    gt_paths = sorted(data_path.glob('gtFine/*/*/*.json'))
    
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