#!/usr/bin/env python

# general python setup
import os, sys, json, argparse

# general data wrangling and plotting libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt

# basic torch machine learning framework
import torch

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
        '-c',
        '--checkpoint',
        help='Path to the model checkpoint.',
        required = True
        )

parser.add_argument(
        '-i',
        '--image',
        help = 'Path to the image for inference.',
        required = True
        )
parser.add_argument(
        '-m',
        '--model',
        help='path to model configuration file',
        required = True
        )

parser.add_argument(
        '-o',
        '--output',
        help='output folder to save the plots',
        required = True
        )

if __name__ == "__main__":
<<<<<<< HEAD
	  # Parsing command line arguments
    args = parser.parse_args()

    # Load configuration
    with open(args.model) as f:
        config = json.load(f)


    ENCODER = config['encoder']
    ENCODER_WEIGHTS = config["encoder_weights"]
    preprocessing_fn = get_preprocessing(smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS))

    # Load model
    model = load_model(args.checkpoint, config)

    # Prepare image
    image = prepare_image(args.image, preprocessing_fn)

    # Perform inference
    predictions = inference(model, image)

    # Post-processing and visualization
    # Assuming binary segmentation for simplicity; adjust as needed
    pred_mask = predictions.squeeze().cpu().numpy()
    pred_mask = (pred_mask > 0.5).astype(np.uint8)  # Threshold predictions

    original_image = cv2.imread(args.image)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Load and prepare the original mask for visualization
    mask_path = os.path.join(os.path.dirname(os.path.dirname(args.image)), "masks", \
        os.path.basename(args.image).split('.')[0]+'.png')
    original_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if original_mask is None:
        raise FileNotFoundError(f"Original mask file {mask_path} not found or is corrupted.")

    # Visualize the original image, the predicted mask, and the original mask
    visualize(f"{args.output}{os.path.basename(args.image).split('.')[0]}", original_image=original_image, predicted_mask=pred_mask, original_mask=original_mask)


=======
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
>>>>>>> 8834f8adba0f8a6b239725f9396a2f553dc2ffad
