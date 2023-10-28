# Domain Adaptation via Knowledge Distillation
Domain Adaptation via Knowledge Distillation for Semantic Segmentation.


# Installation

## Environment

I conducted experiments in the environment below.
```
# Hardware
OS         : Ubuntu 20.04
CPU        : Intel(R) Xeon(R) Silver 4116
GPU        : NVIDIA RTX A5000
Memory     : 128 GB

# CUDA and cuDNN
CUDA       : 11.6
cuDNN      : 8.4.0

# Libraries
Python     : 3.8
PyTorch    : 1.12.1
Torchvision: 0.13.1
```
</br>

## Packages

1. Install PyTorch and Torchvision using `conda`:
```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 -c pytorch
``` 
</br>

2. Clone this repo:
```
git clone https://github.com/enpko52/MTDASeg.git
cd MTDASeg
```
</br>

3. Install this repo and the dependencies using `pip`:
```
pip install -e .
```
</br>

## Datasets

1. Make a data directory:
```
mkdir data
```
</br>

2. Download dataset files:

    GTA5
     - Access to [https://download.visinf.tu-darmstadt.de/data/from_games/](https://download.visinf.tu-darmstadt.de/data/from_games/)
     - Download all images and labels files
     - Move the files to `MTDASeg/data` directory and unzip all files

    Cityscapes
     - Access to [https://www.cityscapes-dataset.com/downloads/](https://www.cityscapes-dataset.com/downloads/)
     - Download `leftImg8bit_trainvaltest.zip` and `gtFine_trainvaltest.zip`
     - Move the files to `MTDASeg/data` directory and unzip all files
    
    IDD
     - Access to [https://idd.insaan.iiit.ac.in/dataset/download/](https://idd.insaan.iiit.ac.in/dataset/download/)
     - Download the `IDD Segmentation` dataset
     - Move the files to `MTDASeg/data` directory and decompress the file
    
    Mapillary
     - Access to [https://www.mapillary.com/dataset/vistas](https://www.mapillary.com/dataset/vistas)
     - Download `mapillary-vistas-dataset_public_v1.2.zip`
     - Move the files to `MTDASeg/data` directory and unzip the file

</br>

3. Organize the dataset as follows:
```
├── data
      ├── GTA5
            ├── images
                   ├── 00001.png
                   ├── 00002.png
                   ├── ...
            ├── labels
                   ├── 00001.png
                   ├── 00002.png
                   ├── ...
      ├── Cityscapes
            ├── leftImg8bit
                   ├── train
                   ├── val
            ├── gtFine
                   ├── train
                   ├── val
      ├── IDD_Segmentation
            ├── leftImg8bit
                   ├── train
                   ├── val
            ├── gtFine
                   ├── train
                   ├── val
      ├── Mapillary
            ├── training
                   ├── images
                   ├── labels
            ├── validation
                   ├── images
                   ├── labels
```
</br>

4. Convert labels to Cityscapes format:
```
cd scripts

# Protocol with 7 classes
python convert_datasets/gta5.py data/GTA5 --num-classes 7
python convert_datasets/cityscapes.py data/Cityscapes --num-classes 7
python convert_datasets/idd.py data/IDD_Segmentation --num-classes 7
python convert_datasets/mapillary.py data/Mapillary --num-classes 7

# Protocol with 19 classes
python convert_datasets/gta5.py data/GTA5 --num-classes 19
python convert_datasets/cityscapes.py data/Cityscapes --num-classes 19
python convert_datasets/idd.py data/IDD_Segmentation --num-classes 19
python convert_datasets/mapillary.py data/Mapillary --num-classes 19
```


## Training

You need to move to the `scripts` directory to train the models.
```
cd ./scrips
```


### 7 Classes Benchmarks (GTA5 &rarr; Cityscapes + IDD)

To train the source only:
 - `--cfg`: The config file path for training
 - `--work-dir`: The working directory path for saving logs
 - `--gpu`: The GPU ID to be used for training
```
python train_net.py --cfg configs/sourceonly/sourceonly_gta52city+idd_cls7.yml --work-dir <WORK_DIR_PATH> --gpu <GPU_ID>
```

To train the baseline:
```
python train_net.py --cfg configs/baseline/baseline_classmix_gta52city+idd_cls7.yml --work-dir <WORK_DIR_PATH> --gpu <GPU_ID>
```

To train the location selection:
```
python train_net.py --cfg configs/ls/ls_gta52city+idd_cls7.yml --work-dir <WORK_DIR_PATH> --gpu <GPU_ID>
```

### 19 Classes Benchmarks (GTA5 &rarr; Cityscapes + IDD)

To train the source only:
```
python train_net.py --cfg configs/sourceonly/sourceonly_gta52city+idd_cls19.yml --work-dir <WORK_DIR_PATH> --gpu <GPU_ID>
```

To train the baseline:
```
python train_net.py --cfg configs/baseline/baseline_classmix_gta52city_cls19.yml --work-dir <WORK_DIR_PATH> --gpu <GPU_ID>
```

To train the location selection:
```
python train_net.py --cfg configs/ls/ls_gta52city+idd_cls19.yml --work-dir <WORK_DIR_PATH> --gpu <GPU_ID>
```


## Testing

To test the models:
 - `--ckp`: The checkpoint file path for testing
```
python test_net.py --cfg <CONFIG_PATH> --ckp <CHECKPOINT_PATH> --gpu <GPU_ID>
```


# Acknowledgements

This codebase is heavily borrowed from [ADVENT](https://github.com/valeoai/ADVENT).
