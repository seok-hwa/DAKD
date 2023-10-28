from setuptools import find_packages
from setuptools import setup

setup(name='MTDASeg',
      install_requires=['mmcv==1.4.1','pyyaml', 
                        'easydict', 'setuptools',
                        'numpy==1.21.5', 'kornia'],
      packages=find_packages())