# general python setup
import os, json

# general data wrangling and plotting libraries
import numpy as np
import cv2

# import data loader
from torch.utils.data import Dataset as BaseDataset

# image augmentation libraries
import imgaug
import albumentations as albu

class Dataset(BaseDataset):
    """Read images, apply augmentation and preprocessing transformations.
    
    Args:
        data_dir (str): path to main image path
        split (str): which split to process ("train", "test", "val")
        classes (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. normalization, shape manipulation, etc.)

    By default a file with files as relative paths is loaded
    from data.json in the data_dir
    """
    
    # These are the fixed classes as described in the data paper
    # for this worked example we will only try to classify trees
    # in the field of view (if any) - a more dynamic approach
    # would load this information from the classes file provided
    # and can be considered an improvement
    CLASSES = ['Bridge','Path','Rail;Transport','Automobile','Person',
     'Road','Sky','Tree','background','Poles','Tower',
     'Traffic;Sign','Lucas;Marker','Dense;Woody;Features',
     'Grass','Terrain','Mountain','Plant;Bush','Cropfield'
     'Crop','Rock','Flowerfield','Animal','Stonewall','Flower',
     'Bark','Well','Ditch','Terrace','Building','Earth;Ground',
     'Wall','Orchard','Field;Margin','Fruit','waterbodies']
    
    # lower case 
    CLASSES = [item.lower() for item in CLASSES]
    
    def __init__(
            self,
            data_dir,
            split,
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        
        # read in list of files from JSON
        with open(os.path.join(data_dir,"data.json")) as f:
            self.full_dataset = json.load(f)

        dataset = self.full_dataset[split]
        
        self.images = []
        self.masks = []
                  
        for item in dataset:
            self.images.append(os.path.join(data_dir,"images", item['image']))
            self.masks.append(os.path.join(data_dir,"masks", item['annotation']))

        # assign id
        self.ids = np.arange(1, len(self.images))
                
        # convert str names to class values on masks
        # a single class is used in this example, but
        # can be adjusted for multi-class purposes
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks[i], 0)
        
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        # check if this shouldn't be type int
        mask = np.stack(masks, axis=-1).astype('float')
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.ids)


# To counter overfitting of large models some augmentation
# of the training data is required

def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
        albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        albu.RandomCrop(height=320, width=320, always_apply=True),
        albu.Perspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightnessContrast(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.Sharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomBrightnessContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.RandomCrop(height=320, width=320, always_apply=True)
        #albu.PadIfNeeded(384, 480)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


