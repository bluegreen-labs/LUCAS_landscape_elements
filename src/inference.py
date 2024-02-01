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

# appending a path
# sys.append doent like relative paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'dataloader'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from model import *
from dataloader import *
from utils import *

parser = argparse.ArgumentParser(
        description = 'LUCAS segmentation model inference routine'
        )

parser.add_argument(
        '-i',
        '--image',
        help = 'path to training data (with a valid data.json file)',
        required = True
        )
        
parser.add_argument(
        '-m',
        '--model',
        help='path to model checkpoint',
        required = True
        )
        
parser.add_argument(
        '-s',
        '--save',
        help='path where to save the generated mask',
        required = True
        )

parser.add_argument(
        '-v',
        '--visualize',
        help='visualize output',
        required = True
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
    
    # load previous checkpoint and evaluate
    # all test data, return the test metrics
    model = Model.load_from_checkpoint(
        args.model + "last_epoch.ckpt"
        )
    
    # Setting up the PyTorch Lightning trainer
    trainer = pl.Trainer(
        accelerator = "gpu"
        )

    # Performing inference on the test data
    mask = trainer.predict(
        model,
        dataloaders = test_dataloader,
        verbose = False
    )

	# Visualizing the results
    visualize(
        image=image_vis, 
        ground_truth_mask=gt_mask, 
        predicted_mask=pr_mask
    )
