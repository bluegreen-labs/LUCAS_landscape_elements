#!/usr/bin/env python

import os
import random
import numpy as np
import pandas as pd
import shutil # shell utilities
#from sklearn.model_selection import train_test_split

# set random seed
random.seed(1)

# set the data path, note that this is
# a relative path with respect to the
# directory holding this script
DATA_DIR = '../../data/raw/ml_data/'

# Split the data in a training, validation and testing dataset
df = pd.read_csv(DATA_DIR + 'lucas_ml_data.csv') 

# drop the extension (will be set for images and masks separately)
df['file'] = df['file'].str.rstrip('.png')

# Use numpy split to split things along rows, this assumes no
# grouping. For more complex splits, to account for class
# imbalances use sklearn methods train_test_split
# (note the array [int, int] specifies breaks in the dataset)
train, validation, test = np.split(df, [int(.6*len(df)), int(.8*len(df))])

# copy things over to the data processed directory
# for the final machine learning modelling

#--- Training data
if not os.path.exists('../../data/processed/train'):
    os.makedirs('../../data/processed/train')

for file in train['file']:
    shutil.copy(DATA_DIR + "images/" + file + ".jpg", '../../data/processed/train')

if not os.path.exists('../../data/processed/trainannot'):
    os.makedirs('../../data/processed/trainannot')

for file in train['file']:
    shutil.copy(DATA_DIR + "masks/" + file + ".png", '../../data/processed/trainannot')

#--- Validation data
if not os.path.exists('../../data/processed/val'):
    os.makedirs('../../data/processed/val')
    
for file in validation['file']:
    shutil.copy(DATA_DIR + "images/" + file + ".jpg", '../../data/processed/val')

if not os.path.exists('../../data/processed/valannot'):
    os.makedirs('../../data/processed/valannot')

for file in validation['file']:
    shutil.copy(DATA_DIR + "masks/" + file + ".png", '../../data/processed/valannot')

#--- Testing data
if not os.path.exists('../../data/processed/test'):
    os.makedirs('../../data/processed/test')
    
for file in test['file']:
    shutil.copy(DATA_DIR + "images/" + file + ".jpg", '../../data/processed/test')

if not os.path.exists('../../data/processed/testannot'):
    os.makedirs('../../data/processed/testannot')

for file in test['file']:
    shutil.copy(DATA_DIR + "masks/" + file + ".png", '../../data/processed/testannot')
    
    

