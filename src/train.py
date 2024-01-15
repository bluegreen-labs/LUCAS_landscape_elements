#!/usr/bin/env python

# general python setup
import os, json, argparse
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# general data wrangling and plotting libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt

# basic torch machine learning framework
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

# easy loading of model frameworks
# provides access to a zoo of segmentation
# models and weights
import segmentation_models_pytorch as smp

# wrappers for torch to make
# training schedules easier to code up
import lightning as pl

# image augmentation library
import imgaug
import albumentations as albu

parser = argparse.ArgumentParser(
        description = 'LUCAS segmentation model routine'
        )

parser.add_argument(
        '-p',
        '--path',
        help='path to training data (assumes a data.json file with the data split is present)',
        required = True
        )
        
group = parser.add_mutually_exclusive_group()
    
group.add_argument(
        '--train',
        action='store_true',
        help='Train'
        )
        
group.add_argument(
        '--test',
        action = 'store_true',
        help = 'Predict on test set'
        )
        
group.add_argument(
        '--predict',
        action='store_true',
        help='Predict on single file'
        )
        
parser.add_argument(
        '--filename',
        help='path to file'
        )
        
if __name__ == "__main__":

    args = parser.parse_args()
    
    
ENCODER = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

model = Model(
    "DeepLabV3Plus",
    "resnet34",
    in_channels = 3,
    out_classes = 1
)

CLASSES = ['car']
    

train_dataset = Dataset(
    data_dir = "./data/raw/ml_data/",
    split = "train",
    augmentation = get_training_augmentation(), 
    preprocessing = get_preprocessing(preprocessing_fn),
    classes = CLASSES,
)

valid_dataset = Dataset(
    data_dir = "./data/raw/ml_data/",
    split = "val",
    augmentation = get_validation_augmentation(), 
    preprocessing = get_preprocessing(preprocessing_fn),
    classes = CLASSES,
)

test_dataset = Dataset(
    data_dir = "./data/raw/ml_data/",
    split = "test",
    augmentation = get_validation_augmentation(), 
    preprocessing = get_preprocessing(preprocessing_fn),
    classes = CLASSES,
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=3,
    shuffle=True,
    num_workers=12
)

valid_dataloader = DataLoader(
    valid_dataset,
    batch_size=1, 
    shuffle=False,
    num_workers=4
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=4
)

trainer = pl.Trainer(
    accelerator = "gpu", # change to cpu 
    max_epochs = 5,
)

trainer.fit(
    model2, 
    train_dataloaders = train_dataloader,
    val_dataloaders = valid_dataloader,
)

# run validation dataset
valid_metrics = trainer.validate(
    model,
    dataloaders = valid_dataloader,
    verbose = False
)

pprint(valid_metrics)

# run test dataset
test_metrics = trainer.test(
    model,
    dataloaders = test_dataloader,
    verbose=False
)

pprint(test_metrics)



    if args.predict_on_test_set:
        predict_on_test_set(args)

    elif args.predict:
        if args.filename is None:
            raise Exception('missing --filename FILENAME')
        else:
            predict(args)

    elif args.train:
        print('Starting training')
        train(args)
    else:
        raise Exception('Unknown args') 
