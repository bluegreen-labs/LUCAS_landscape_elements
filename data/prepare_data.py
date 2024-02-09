#!/usr/bin/env python

# this generates a simpel train, validation, test split
# of the LUCAS segmentation data. Keep in mind that the
# data are not balanced using class coverage and this
# method should be improved upon to increase model
# accuracy (i.e. this is worked example only)
# 
# It also deals with reclassifying the data into
# consistently fewer classes (binning data) to improve
# predictive capacity.

import os, sys, glob, shutil
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image

def copy_images(df, image_path, split, out_path):
<<<<<<< HEAD
    """
    Copy images specified in a DataFrame from one location to another.

    Parameters:
    - df: DataFrame containing image information.
    - image_path: Path where the source images are located.
    - split: Split criteria for organizing images in the output path
          line train/test/val.
    - out_path: Root path for the output destination.

    Returns:
    None
    """
    for i,img in df.iterrows():
=======
	"""
	Copy images specified in a DataFrame from one location to another.

	Parameters:
	- df: DataFrame containing image information.
	- image_path: Path where the source images are located.
	- split: Split criteria for organizing images in the output path
				line train/test/val.
	- out_path: Root path for the output destination.

	Returns:
	None
	"""
   for i,img in df.iterrows():
>>>>>>> 8834f8adba0f8a6b239725f9396a2f553dc2ffad
      # match file names
      basename = os.path.basename(img[0])
      file = os.path.splitext(basename)[0] + ".jpg"
      path = os.path.join(image_path, file)
      out_file = os.path.join(output_path, split, "images", file)
      
      try:
      # copy stuff
        shutil.copyfile(
         path,
         out_file
        )
      except:
        pass

def convert_classes(df, split, labels, out_path):
<<<<<<< HEAD
    """
    Open mask files, relabel classes where necessary, and save them in the output directory.

    Parameters:
    - df: DataFrame containing mask file information.
    - split: Split criteria for organizing images in the output path
          line train/test/val.	
    - labels: DataFrame containing label codes and corresponding new codes.
    - out_path: Root path for the output destination.

    Returns:
    None
    """
   
    for i,r in df.iterrows():
=======
	"""
	Open mask files, relabel classes where necessary, and save them in the output directory.

	Parameters:
	- df: DataFrame containing mask file information.
	- split: Split criteria for organizing images in the output path
				line train/test/val.	
	- labels: DataFrame containing label codes and corresponding new codes.
	- out_path: Root path for the output destination.

	Returns:
	None
	"""
   
   for i,r in df.iterrows():
>>>>>>> 8834f8adba0f8a6b239725f9396a2f553dc2ffad
      
       # read in the mask data
       im = Image.open(r[0])
       data = np.array(im)
       
       # grab label codes, convert to numpy array
       # (inherits pandas indexed data frame data
       # which is not dealt with nicely)
       codes = labels['Mask code'].to_numpy()
       new_codes = labels['New code'].to_numpy()
       
       # Use searchsorted to find the indices of old_codes
       # that correspond to each element of array
       indices = np.searchsorted(codes, data).astype(np.uint8)
       new_data = new_codes[indices].astype(np.uint8)
       
       # save the array as an image in the output
       # directory
       im = Image.fromarray(new_data)
       
       filename = os.path.join(output_path, split, "masks", os.path.basename(r[0]))
       im.save(filename)


def getArgs():
<<<<<<< HEAD
    """
    Parse command line arguments for the main routine

    Returns:
    - args: Parsed command line arguments.
    """
=======
	"""
	Parse command line arguments for the main routine

	Returns:
	- args: Parsed command line arguments.
	"""
>>>>>>> 8834f8adba0f8a6b239725f9396a2f553dc2ffad

    parser = argparse.ArgumentParser(
    description = 'Generate a ML dataset for LUCAS data',
    epilog = 'post bugs to the github repository'
    )

    parser.add_argument(
    '-i',
    '--input_path',
    type = str,
    help = 'Input data path (top most directory)',
    required = True
    )
      
<<<<<<< HEAD
    parser.add_argument(
    '-l',
    '--labels',
    type = str,
    help = 'File with label data codes in csv format',
    required = True
    )

    return parser.parse_args()
 
if __name__ == '__main__':
   
    # parse arguments
    Args = getArgs()
    input_path = Args.input_path
    labels = Args.labels

    # paths
    path_to_img = os.path.join(input_path, "images")
    path_to_masks = os.path.join(input_path, "masks")

    # images
    images = glob.glob("{}/*.jpg".format(path_to_img))
    masks = glob.glob("{}/*.png".format(path_to_masks))

    # labels and recode values
    labels = pd.read_csv(labels)
    nr_classes = len(labels['Mask code'])

    # Split it in 0.6, 0.2, 0.2 (train, validation, test) -
    # Note: The image split is not distributed or
    # weighted by pixel coverage and an improved strategy
    # would consider the weights of pixel representation
    # in either training or during the train split routine
    df = pd.DataFrame(masks)
    df['png_file'] = df[0].transform(lambda x: os.path.basename(x))
    df['annotation'] = df['png_file'] #.transform(lambda x: os.path.join(output_path, x))
    df['image'] = df['png_file'].transform(lambda x: os.path.splitext(x)[0] + ".jpg")
    df.drop(df.columns[[0,1]], axis=1, inplace=True)

    train, val, test = np.split(
      df.sample(frac=1, random_state=1),
      [int(.6*len(df)),
      int(.8*len(df))]
    )

    json_data = {}
    json_data['train'] = train.to_dict('records')
    json_data['test'] = test.to_dict('records')
    json_data['val'] = val.to_dict('records')

    # the json file is saved in the input data folder

    with open(os.path.join(input_path, 'data.json'), 'w') as f:
      json.dump(json_data, f)

=======
   parser.add_argument(
   '-l',
   '--labels',
   type = str,
   help = 'File with label data codes in csv format',
   required = True
   )
   
   return parser.parse_args()
 
if __name__ == '__main__':
   
   # parse arguments
   Args = getArgs()
   input_path = Args.input_path
   labels = Args.labels
   
   # paths
   path_to_img = os.path.join(input_path, "images")
   path_to_masks = os.path.join(input_path, "masks")
   
   # images
   images = glob.glob("{}/*.jpg".format(path_to_img))
   masks = glob.glob("{}/*.png".format(path_to_masks))
   
   # labels and recode values
   labels = pd.read_csv(labels)
   nr_classes = len(labels['Mask code'])

   # Split it in 0.6, 0.2, 0.2 (train, validation, test) -
   # Note: The image split is not distributed or
   # weighted by pixel coverage and an improved strategy
   # would consider the weights of pixel representation
   # in either training or during the train split routine
   df = pd.DataFrame(masks)
   df['png_file'] = df[0].transform(lambda x: os.path.basename(x))
   df['annotation'] = df['png_file'] #.transform(lambda x: os.path.join(output_path, x))
   df['image'] = df['png_file'].transform(lambda x: os.path.splitext(x)[0] + ".jpg")
   df.drop(df.columns[[0,1]], axis=1, inplace=True)

   train, val, test = np.split(
    df.sample(frac=1, random_state=1),
    [int(.6*len(df)),
    int(.8*len(df))]
   )

   json_data = {}
   json_data['train'] = train.to_dict('records')
   json_data['test'] = test.to_dict('records')
   json_data['val'] = val.to_dict('records')
   
   # the json file is saved in the input data folder
   
   with open(os.path.join(input_path, 'data.json'), 'w') as f:
    json.dump(json_data, f)
   
>>>>>>> 8834f8adba0f8a6b239725f9396a2f553dc2ffad
