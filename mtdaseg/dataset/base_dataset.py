import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils import data

from mtdaseg.utils import project_root
from mtdaseg.utils.serialization import load_json


CLASS7_INFO = project_root / 'mtdaseg/dataset/class_info/class7_info.json'
CLASS19_INFO = project_root / 'mtdaseg/dataset/class_info/class19_info.json'


class BaseDataset(data.Dataset):
    ''' The base class for datasets '''
    
    def __init__(self, root, num_classes, image_size, label_size, mean, domain):
        ''' Initialize the class '''
        
        self.root = Path(root)
        self.num_classes = num_classes
        self.image_size = image_size
        self.label_size = label_size if label_size else image_size
        self.mean = mean
        self.domain = domain
        self.files = []
        
        # Get the class information
        self.info = load_json(CLASS19_INFO) if self.num_classes == 19 else load_json(CLASS7_INFO)
        self.class_names = np.array(self.info['label'], dtype=np.str)
        self.palette = np.array(self.info['palette'], dtype=np.uint8)
        
    def get_metadata(self, img_path):
        ''' Get the metadata (e.g., image name, image path, label path) '''

        raise NotImplementedError
    
    def __len__(self):
        ''' Return the total number of data in the dataset '''
        
        return len(self.files)
    
    def __getitem__(self, index):
        ''' Return a data from dataset using indexing '''
        
        raise NotImplementedError
    
    def preprocess(self, image):
        ''' Perform preprocessing on the image before training and testing '''
        
        image = image[:, :, ::-1] # Convert to BGR
        image = image - self.mean
        image = image.transpose(2, 0, 1)
        return image
    
    def get_image(self, file):
        ''' Get the image using the file path '''
        
        return _load_img(file, self.image_size, Image.BICUBIC)
    
    def get_label(self, file):
        ''' Get the label using the file path '''
        
        return _load_img(file, self.label_size, Image.NEAREST, rgb=False)
    
    
def _load_img(file, size, interpolation, rgb=True):
    ''' Load an image using the file path '''
    
    img = Image.open(file)
    if rgb: 
        img = img.convert('RGB')
    img = img.resize(size, interpolation)
    return np.asarray(img, np.float32)