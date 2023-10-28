import os
import numpy as np

from .base_dataset import BaseDataset


class IDDDataset(BaseDataset):
    ''' The dataset class for IDD '''

    def __init__(self,
                 root,
                 train=True,
                 num_classes=19,
                 crop_size=(321, 321),
                 label_size=None,
                 mean=(128, 128, 128),
                 domain=0):
        ''' Initialize the class '''

        super(IDDDataset, self).__init__(root, num_classes, crop_size, label_size, mean, domain)
        self.dataset_name = 'IDD'

        # Get a list of image and label paths
        self.split = 'train' if train else 'val'
        img_paths = sorted(self.root.glob('leftImg8bit/{}/*/*.png'.format(self.split)))
        
        for img_path in img_paths:
            img_path, lbl_path, img_name = self.get_metadata(img_path)
            self.files.append((img_path, lbl_path, img_name, self.domain))
        
    def get_metadata(self, img_path):
        ''' Get the metadata (e.g., image name, image path, label path) '''

        # Split an image path into (dir path, image name)
        *dir_path, img_name = str(img_path).split('/')
        
        # Get a label path
        dir_path = '/'.join(dir_path).replace('leftImg8bit', 'gtFine')
        lbl_name = img_name.replace('_leftImg8bit.png', '_gtFine_labelTrainIds{}.png'.format(self.num_classes))
        lbl_path = os.path.join(dir_path, lbl_name)
        
        return img_path, lbl_path, img_name
        
    def __getitem__(self, index):
        ''' Return a data from dataset using indexing '''
        
        img_path, lbl_path, img_name, domain = self.files[index % len(self.files)]
        label = self.get_label(lbl_path)
        image = self.get_image(img_path)
        image = self.preprocess(image)
        
        return image.copy(), label, np.array(image.shape), img_name, domain