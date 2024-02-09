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
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping


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
        '-o',
        '--output',
        help='path to where we will save the checkpoints',
        required = True
        )    

parser.add_argument(
        '-m',
        '--model',
        help='path to the model configuration',
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

    # Load configuration file
    
    with open(args.model) as config_file:
        config = json.load(config_file)

    # Defining the classes for segmentation
    CLASSES = ['tree']
	
	  # Model setup
    ENCODER = config['encoder']
    ENCODER_WEIGHTS = config["encoder_weights"]
    preprocessing_fn = smp.encoders.get_preprocessing_fn(
        ENCODER,
        ENCODER_WEIGHTS
    )
    
    # Initializing the segmentation model
    model = Model(
        config["model_architecture"],
        ENCODER,
        in_channels = config["in_channels"],
        out_classes = config["out_classes"],
        learning_rate = config["learning_rate"]
    )

    # Model checkpoint callback with dynamic filename
    model_checkpoint = ModelCheckpoint(
        monitor="val_loss",
        dirpath=args.output,
        filename="{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,  
        mode="min",  
        save_last=True  
        )
      
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
        batch_size = config['batch_size'],
        shuffle = False,
        num_workers = 0,
        drop_last = True
    )
    
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size = config['batch_size'],
        shuffle = False,
        num_workers = 0,
        drop_last = True
    )
    
    #--- train and validation routine ----

    if args.train:
        
        # callback for early stopping
        early_stopping = EarlyStopping(
            "val_loss",
            patience = 10
        )
		
		# PyTorch Lightning Trainer
        trainer = pl.Trainer(
            accelerator = "gpu", # change to cpu 
            max_epochs = config["epochs"],
            callbacks = [model_checkpoint, early_stopping]
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
            args.output + "/last.ckpt"
            )
        
        test_dataloader = DataLoader(
            test_dataset,
            batch_size = config['batch_size'],
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
 
