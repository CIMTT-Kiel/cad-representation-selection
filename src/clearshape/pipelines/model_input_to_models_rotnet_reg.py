import os
import shutil
import time

import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
import pandas as pd

from pathlib import Path
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

import clearshape.constants as constants
from clearshape.rotationnet.rotnet_classifier import RotationNetModel
from clearshape.dataset import FabwaveDataset

args = constants.ROTNET

def get_dataloaders():
    train_loader = DataLoader(FabwaveDataset(csv_file="/clear-shape/data/5_model_input/train.csv", classification=True, data_type="images"), batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(FabwaveDataset(csv_file="/clear-shape/data/5_model_input/validation.csv", classification=True, data_type="images"), batch_size=args.batch_size)

    return train_loader, validation_loader
 
def main():

    mlf_logger = MLFlowLogger(experiment_name="images-classification", tracking_uri="/clear-shape/src/clearshape/rotationnet/ml/mlruns", run_name="best-model")
    
    #criterion = nn.CrossEntropyLoss()#.cuda()
    # Resume from checkpoint if available
    model = RotationNetModel()
    
    train_loader, val_loader = get_dataloaders()

    early_stop_callback = EarlyStopping(
            monitor='f1_score',     # oder "val_acc", je nachdem
            patience=30,             # z. B. 5 Epochen ohne Verbesserung
            mode='max',             # "min" für loss, "max" für accuracy
            verbose=True
        )
    
    checkpoint_callback = ModelCheckpoint(
        monitor='f1_score',           # oder z. B. "val_acc"
        save_top_k=1,                 # nur das beste Modell speichern
        mode='max',                   # "min" für loss, "max" für acc
        dirpath=constants.PATHS.DATA_MODELS.as_posix(),
        filename='images-classifier',  # Dateinamenformat
        save_weights_only=False,      # speichert komplette Checkpoints
        verbose=True
    )

    trainer = Trainer(max_epochs=150, logger=mlf_logger, callbacks=[early_stop_callback, checkpoint_callback], enable_checkpointing=True, enable_model_summary=False, log_every_n_steps=1)
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()