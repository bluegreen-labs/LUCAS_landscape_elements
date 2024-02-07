from setuptools import setup, find_packages

setup(
  name = 'mlenv',
  packages = find_packages(exclude=['data', 'models']),
  version = '0.1.0',
  license='GNU AFFERO GENERAL PUBLIC LICENSE',
  description = 'LUCAS single class segmentation example framework - pytorch',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/bluegreen-labs/LUCAS_landscape_elements.git',
  keywords = [
    'semantic segmentation',
    'LUCAS images',
  ],
  install_requires=[
    'pytest',
    'torch==2.1.0',
    'torchvision==0.6',
    'torchaudio==2.1.0',
    'tensorrt==8.6.1',
    'tqdm',
    'setuptools',
    'wheel',
    'numpy',
    'matplotlib',
    'pandas',
    'albumentations',
    'imgaug',
    'segmentation-models-pytorch',
    'pytorch-lightning==1.5.4',
    'tensorboardX'
  ],
)
