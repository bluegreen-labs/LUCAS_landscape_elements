# general python setup
import os, json

# general data wrangling and plotting libraries
import numpy as np

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

class Model(pl.LightningModule):
    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
		"""
		Constructor method for initializing the Model class.

		Parameters:
			- arch: Architecture of the model.
			- encoder_name: Name of the encoder for feature extraction.
			- in_channels: Number of input channels.
			- out_classes: Number of output classes.
			- **kwargs: Additional keyword arguments for model creation.
		"""
        super().__init__()
        
        # load model architecture
        self.model = smp.create_model(
            arch,
            encoder_name = encoder_name,
            in_channels = in_channels,
            classes = out_classes,
            **kwargs
        )

        # preprocessing parameteres for images
        # i.e. the mean and std of the pretrained model
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # for image segmentation dice loss could be the best first choice
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits = True)
        
        # save hyperparameters in model
        self.save_hyperparameters()
    
    def configure_optimizers(self):
		"""
        Method to configure the optimizer
        in this case an Adam optimizer with LR 0.0005
        Returns:
            torch.optim.Optimizer: Optimizer for training.
        """
        
        optimizer = torch.optim.Adam(self.parameters(), lr = 0.0005)
        return optimizer
    
    def infer_batch(self, batch):
		"""
        Method to perform inference on a batch of data.

        Parameters:
            - batch: Input batch containing images and masks.

        Returns:
            tuple: Predicted mask and ground truth mask.
        """
        
        # load image, masks from list
        image = batch[0]
        mask = batch[1]
        
        # check image dimensions and
        # multiples of 32
        assert image.ndim == 4
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        # check image dimensions and the range of 
        # the mask values - should be between 0 and 1
        assert mask.ndim == 4
        assert mask.max() <= 1.0 and mask.min() >= 0
        
        x, y = self.prepare_batch(batch)
        y_hat = self.model(x)
        return y_hat, y

    def training_step(self, batch, batch_idx):
		"""
        Method to do the inference part of a training step and
        calculate the loss.

        Parameters:
            - batch: Input batch containing images and masks.
            - batch_idx: Index of the batch.

        Returns:
            torch.Tensor: Loss for the current step.
        """
        y_hat, y = self.infer_batch(batch)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss, prog_bar = True)
        return loss
    
    def validation_step(self, batch, batch_idx):
		"""
        Method to do the inference part of a validation step and
        calculate the loss.

        Parameters:
            - batch: Input batch containing images and masks.
            - batch_idx: Index of the batch.

        Returns:
            torch.Tensor: Loss for the current step.
        """
        y_hat, y = self.infer_batch(batch)
        loss = self.loss_fn(y_hat, y)
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
		"""
        Method to do the inference part of a test step and
        calculate the loss.

        Parameters:
            - batch: Input batch containing images and masks.
            - batch_idx: Index of the batch.

        Returns:
            torch.Tensor: Loss for the current step.
        """
        y_hat, y = self.infer_batch(batch)
        loss = self.loss_fn(y_hat, y)
        self.log('test_loss', loss)
        return loss
    
    def predict_step(self, batch, batch_idx):
		"""
        Method to perform inference on a batch 

        Parameters:
            - batch: Input batch containing images and masks.
            - batch_idx: Index of the batch.

        Returns:
            torch.Tensor: Predicted mask.
        """
        y_hat, y = self.infer_batch(batch)
        return y_hat

