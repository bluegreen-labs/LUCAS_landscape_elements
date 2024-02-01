# general python setup
import os, json

# general data wrangling and plotting libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt

# helper function for data visualization
def visualize(**images):
	"""
    Plot images in one row.

    Parameters:
        **images: Keyword arguments where keys are image names and values are the images themselves.
    """
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()
