#!/usr/bin/env python

# general python setup
import os, sys, json, argparse

# general data wrangling and plotting libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt

# basic torch machine learning framework
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

# easy loading of model frameworks
# provides access to segmentation models and weights
import segmentation_models_pytorch as smp

# wrappers for torch to make
# training schedules easier to code up
import lightning as pl

# image augmentation library
import imgaug
import albumentations as albu

# import

# appending a path
# sys.append doent like relative paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'dataloader'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from model import *
from dataloader import *
from utils import *

parser = argparse.ArgumentParser(
        description = 'LUCAS segmentation model routine'
        )

parser.add_argument(
        '-d',
        '--data',
        help='path to training data (with a valid data.json file)',
        required = True
        )
        
parser.add_argument(
        '-m',
        '--model',
        help='path to models',
        required = True
        )
        
group = parser.add_mutually_exclusive_group()

group.add_argument(
        '--train',
        action='store_true',
        help = 'Train'
        )
        
group.add_argument(
        '--test',
        action = 'store_true',
        help = 'Predict on test set'
        )

if __name__ == "__main__":
	
	# Parsing command line arguments
    args = parser.parse_args()
	
	# Model setup
    ENCODER = 'resnet34'
    ENCODER_WEIGHTS = 'imagenet'
    preprocessing_fn = smp.encoders.get_preprocessing_fn(
        ENCODER,
        ENCODER_WEIGHTS
    )
    
    # Initializing the segmentation model
    model = Model(
        "DeepLabV3Plus",
        "resnet34",
        in_channels = 3,
        out_classes = 1
    )
    
    # Defining the classes for segmentation
    CLASSES = ['tree']
    
    # Prepare data for training dataset
    train_dataset = Dataset(
        data_dir = args.data,
        split = "train",
        augmentation = get_training_augmentation(), 
        preprocessing = get_preprocessing(preprocessing_fn),
        classes = CLASSES,
    )
    
    #for validation dataset
    valid_dataset = Dataset(
        data_dir = args.data,
        split = "val",
        augmentation = get_validation_augmentation(), 
        preprocessing = get_preprocessing(preprocessing_fn),
        classes = CLASSES,
    )
    
    #for test dataset
    test_dataset = Dataset(
        data_dir = args.data,
        split = "test",
        augmentation = get_validation_augmentation(), 
        preprocessing = get_preprocessing(preprocessing_fn),
        classes = CLASSES,
    )
    
    #dataloaders for train validation 
    train_dataloader = DataLoader(
        train_dataset,
        batch_size = 2,
        shuffle = False,
        num_workers = 0,
        drop_last = True
    )
    
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size = 2,
        shuffle = False,
        num_workers = 0,
        drop_last = True
    )
    
    #--- train and validation routine ----

    if args.train:
        
        # callback for early stopping
        early_stopping = pl.pytorch.callbacks.EarlyStopping(
            "val_loss",
            patience = 10
        )
        
        # model checkpoint callback
        model_checkpoint = pl.pytorch.callbacks.ModelCheckpoint(
            monitor = "val_loss",
            dirpath = args.model,
            filename = "last_epoch"
        )
		
		# PyTorch Lightning Trainer
        trainer = pl.Trainer(
            accelerator = "gpu", # change to cpu 
            max_epochs = 1,
            callbacks = [early_stopping, model_checkpoint]
        )
        
        # Training the model
        trainer.fit(
            model,
            train_dataloaders = train_dataloader,
            val_dataloaders = valid_dataloader
        )

    #--- test routine ----
    
    if args.test:
            
        # load previous checkpoint and evaluate
        # all test data, return the test metrics
        model = Model.load_from_checkpoint(
            args.model + "last_epoch.ckpt"
            )
        
        test_dataloader = DataLoader(
            test_dataset,
            batch_size = 1,
            shuffle = False,
            drop_last = True
        )
        
        trainer = pl.Trainer(
            accelerator = "gpu"
            )
    
        # run test dataset
        test_metrics = trainer.test(
            model,
            dataloaders = test_dataloader,
            verbose = False
        )
        
        print(test_metrics)
 
