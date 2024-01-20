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
        optimizer = torch.optim.Adam(self.parameters(), lr = 0.0005)
        return optimizer
    
    def infer_batch(self, batch):
        
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
        y_hat, y = self.infer_batch(batch)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss, prog_bar = True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        y_hat, y = self.infer_batch(batch)
        loss = self.loss_fn(y_hat, y)
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        y_hat, y = self.infer_batch(batch)
        loss = self.loss_fn(y_hat, y)
        self.log('test_loss', loss)
        return loss
    
    def predict_step(self, batch, batch_idx):
        y_hat, y = self.infer_batch(batch)
        return y_hat

