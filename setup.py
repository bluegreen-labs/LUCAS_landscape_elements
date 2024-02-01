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
    'einops>=0.7.0',
    'torch>=1.10',
    'torchvision'
  ],
  setup_requires=[
    'pytest-runner',
  ],
  tests_require=[
    'pytest',
    'torch==1.12.1',
    'torchvision==0.13.1'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)