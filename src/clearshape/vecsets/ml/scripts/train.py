from typing import Any, Dict, List, Optional, Tuple
import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger
from omegaconf import DictConfig
from pytorch_lightning import Callback
import mlflow, os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from pathlib import Path

import numpy as np

from clearshape.vecsets.ml.modules.vecset_datamodul import VecsetDataset
from clearshape.vecsets.ml.modules.trsfm_classificator import VecsetClassifierModule
#custom imports



@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    # Optional: Logger einbinden
    run_name = "clearshape_classification"
    experimnett_name = "clearshape_initial_classification"
    tracking_uri = "./../mlruns"

    mlflow.autolog()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experimnett_name)
    mlflow.start_run(run_name=run_name)

    mlflow_artifact_uri = mlflow.get_artifact_uri()
    #logger = MLFlowLogger(experiment_name=experimnett_name, tracking_uri=tracking_uri, run_name=run_name)

    # Instanziiere DataModule & Modell via Hydra
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    model = hydra.utils.instantiate(cfg.model)


    # Trainer
    trainer = hydra.utils.instantiate(cfg.trainer)


    trainer.fit(model, datamodule=datamodule)



    mlflow.end_run()




if __name__ == "__main__":
    main()