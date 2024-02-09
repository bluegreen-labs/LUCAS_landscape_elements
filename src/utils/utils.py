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



def load_model(checkpoint_path, config):
    """
    Load the trained model from a checkpoint

    Parameters:
        checkpoint_path (str): Path to the model checkpoint
        config (dict): Configuration dictionary

    Returns:
        Model: Loaded PyTorch model
    """
    model = Model.load_from_checkpoint(checkpoint_path=checkpoint_path)
    return model

def prepare_image(image_path, preprocessing_fn):
    """
    Prepare an image for inference.

    Parameters:
        image_path (str): Path to the image
        preprocessing_fn: Preprocessing function
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    image = cv2.imread(image_path)
    if image is None:
        raise Exception(f"Failed to load image '{image_path}'. The file may be corrupt or not an image.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply preprocessing
    sample = preprocessing_fn(image=image)
    image = sample['image']

    # Convert to PyTorch tensor and add batch dimension
    image = torch.from_numpy(image).to(torch.float32)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

def inference(model, images):
    """
    Perform inference on an image or a batch of images

    Parameters:
        model (Model): Trained PyTorch model
        images (torch.Tensor): Batch of images

    Returns:
        torch.Tensor: Predicted masks.
    """
    model.eval()  # Set the model to evaluation mode
    print(model.device)
    images = images.to(model.device)
    with torch.no_grad():
        predictions = model(images)
    return predictions

# helper function for data visualization
def visualize(name_plot, **images):
    """
    Plot images in one row

    Parameters:
        **images: Keyword arguments where keys are image names and values are the images themselves
    """
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.savefig(name_plot+'.png')
    plt.show()


