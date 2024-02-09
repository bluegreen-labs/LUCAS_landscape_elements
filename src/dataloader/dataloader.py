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
    """
    Dataset class for reading images, applying augmentation, and preprocessing transformations.

    By default, a file with file paths as relative paths is loaded
    from data.json in the data_dir.
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
<<<<<<< HEAD
        """
        Constructor method for initializing the ImageProcessor class.

        Parameters:
          - data_dir (str): Path to the main image directory.
          - split (str): Which split to process ("train", "test", "val").
          - classes (list): Values of classes to extract from the segmentation mask.
          - augmentation (albumentations.Compose): Data transformation pipeline
          (e.g., flip, scale, etc.).
          - preprocessing (albumentations.Compose): Data preprocessing
          (e.g., normalization, shape manipulation, etc.).
        """
=======
		"""
		Constructor method for initializing the ImageProcessor class.

		Parameters:
			- data_dir (str): Path to the main image directory.
			- split (str): Which split to process ("train", "test", "val").
			- classes (list): Values of classes to extract from the segmentation mask.
			- augmentation (albumentations.Compose): Data transformation pipeline
			(e.g., flip, scale, etc.).
			- preprocessing (albumentations.Compose): Data preprocessing
			(e.g., normalization, shape manipulation, etc.).
		"""
>>>>>>> 8834f8adba0f8a6b239725f9396a2f553dc2ffad
        
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
<<<<<<< HEAD
        """
=======
		"""
>>>>>>> 8834f8adba0f8a6b239725f9396a2f553dc2ffad
        Method to get an item from the dataset.

        Parameters:
            -i (int): Index of the item.

        Returns:
            tuple: Image and corresponding mask.
        """
<<<<<<< HEAD
        # Attempt to read the image file
        try:
            image = cv2.imread(self.images[i])
            if image is None:
                raise FileNotFoundError(f"Image file {self.images[i]} not found or is corrupted.")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error reading image file {self.images[i]}: {e}")

        # Attempt to read the mask file
        try:
            mask = cv2.imread(self.masks[i], 0)
            if mask is None:
                raise FileNotFoundError(f"Mask file {self.masks[i]} not found or is corrupted.")
        except Exception as e:
            print(f"Error reading mask file {self.masks[i]}: {e}")

        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(f"Image and mask dimensions do not match for {self.images[i]} and {self.masks[i]}.")
=======
>>>>>>> 8834f8adba0f8a6b239725f9396a2f553dc2ffad
        
        # extract certain classes from mask (e.g. Trees)
        # we have to add +1 since the codes of this datasets starts with 1
        masks = [(mask == v+1) for v in self.class_values]

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
<<<<<<< HEAD
        """
=======
		"""
>>>>>>> 8834f8adba0f8a6b239725f9396a2f553dc2ffad
        Method to get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.ids)




def get_training_augmentation():
<<<<<<< HEAD
    """
=======
	"""
>>>>>>> 8834f8adba0f8a6b239725f9396a2f553dc2ffad
    Function to get data augmentation transformations for training.
    To counter overfitting of large models some augmentation
    of the training data is required

    Returns:
        albumentations.Compose: Data augmentation pipeline for training.
    """
    
	# List of augmentation transformations
    train_transform = [
        # Flip the image horizontally with a 50% probability
        albu.HorizontalFlip(p=0.5),
        
        # Apply random scaling, rotation, and shifting
        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
        
        # Pad the image if needed to meet minimum height and width requirements
        albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        
        # Randomly crop the image to the specified height and width
        albu.RandomCrop(height=320, width=320, always_apply=True),
        
        # Apply perspective transformation with a 50% probability
        albu.Perspective(p=0.5),

        # Randomly apply one of the contrast enhancement techniques
        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightnessContrast(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        # Randomly apply one of the sharpening or blurring techniques
        albu.OneOf(
            [
                albu.Sharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        # Randomly apply one of the brightness and contrast adjustments
        albu.OneOf(
            [
                albu.RandomBrightnessContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]

    # Return the composition of all augmentation transformations
    return albu.Compose(train_transform)


def get_validation_augmentation():
<<<<<<< HEAD
    """
=======
	"""
>>>>>>> 8834f8adba0f8a6b239725f9396a2f553dc2ffad
    Function to get data augmentation transformations for the validation split

    Returns:
        albumentations.Compose: Data augmentation pipeline for validation.
    """
    
    #Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.RandomCrop(height=320, width=320, always_apply=True)
        # Uncomment the line below to add paddings to make image shape divisible by 32
        #albu.PadIfNeeded(384, 480)
    ]
    return albu.Compose(test_transform)

def get_predict_augmentation(target_height=1200, target_width=1600):
    """
    Function to get data augmentation transformations for the validation split.

    Parameters:
        - target_height (int): Target height of the image.
        - target_width (int): Target width of the image.

    Returns:
        albumentations.Compose: Data augmentation pipeline for validation.
    """
    
    # Pad the image if needed to make it divisible by 32
    test_transform = [
        albu.PadIfNeeded(
            min_height=((target_height // 32) + 1) * 32,
            min_width=((target_width // 32) + 1) * 32,
            always_apply=True
        )
    ]
    return albu.Compose(test_transform)



def to_tensor(x, **kwargs):
<<<<<<< HEAD
    """
=======
	"""
>>>>>>> 8834f8adba0f8a6b239725f9396a2f553dc2ffad
    Convert the input array to a PyTorch tensor.

    Parameters:
        - x: Input array.

    Returns:
        numpy.ndarray: Transposed array converted to float32.
    """
    
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
<<<<<<< HEAD
    """
    Function to get preprocessing transformations.

    Parameters:
        - preprocessing_fn: Preprocessing function to be applied.

    Returns:
        albumentations.Compose: Preprocessing pipeline.
    """
=======
    """
    Function to get preprocessing transformations.

    Parameters:
        - preprocessing_fn: Preprocessing function to be applied.

    Returns:
        albumentations.Compose: Preprocessing pipeline.
    """
    
>>>>>>> 8834f8adba0f8a6b239725f9396a2f553dc2ffad
    # List of preprocessing transformations
    _transform = [
        # Apply the provided preprocessing function to the image
        albu.Lambda(image=preprocessing_fn),
        
        # Convert both image and mask to PyTorch tensors
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    
    return albu.Compose(_transform)
