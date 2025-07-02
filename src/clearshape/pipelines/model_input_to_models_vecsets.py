import optuna
import mlflow
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger
import torch
from torch.utils.data import DataLoader, random_split, TensorDataset
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from clearshape.vecsets.ml.modules.trsfm_classificator import VecsetClassifierModule
from clearshape.dataset import FabwaveDataset
from clearshape.constants import PATHS


def get_dataloaders(batch_size=32):
    train_loader = DataLoader(FabwaveDataset(csv_file="/clear-shape/data/5_model_input/train.csv", classification=True, data_type="vecsets"), batch_size=batch_size)
    validation_loader = DataLoader(FabwaveDataset(csv_file="/clear-shape/data/5_model_input/validation.csv", classification=True, data_type="vecsets"), batch_size=batch_size)

    return train_loader, validation_loader

def objective(trial):

    # Hyperparameter-sampling
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    batch_size = 64


    # Transformer-specific hyperparameters
    nhead = trial.suggest_categorical("nhead", [4, 8])
    num_layers = trial.suggest_int("num_layers", 2, 8)

    # Feedforward network size
    dim_feedforward = trial.suggest_int("hidden_size", 512, 2024)


    train_loader, val_loader = get_dataloaders(batch_size=batch_size)

    model = VecsetClassifierModule( 
                lr=lr,
                input_dim=32, 
                d_model=1024, 
                nhead=nhead, 
                num_layers=num_layers, 
                num_classes=38, 
                dim_feedforward=dim_feedforward, 
                fc_layers=None,
                dropout=dropout, 
                use_pos_embedding=False, 
            )

    # MLflow Logger
    mlf_logger = MLFlowLogger(experiment_name="vecsets-classification", tracking_uri="file:./../vecsets/ml/mlruns")

    early_stop_callback = EarlyStopping(
            monitor='val_loss',     # oder "val_acc", je nachdem
            patience=10,             # z. B. 5 Epochen ohne Verbesserung
            mode='min',             # "min" für loss, "max" für accuracy
            verbose=True
        )

    trainer = Trainer(
        max_epochs=100,
        logger=mlf_logger,
        enable_checkpointing=False,
        enable_model_summary=False,
        log_every_n_steps=2,
        callbacks=[early_stop_callback] 
    )

    trainer.fit(model, train_loader, val_loader)
    val_loss = trainer.callback_metrics["val_loss"].item()

    # Hyperparameter und Metrik an MLflow loggen (über den Logger)
    mlf_logger.log_hyperparams(trial.params)
    mlf_logger.log_metrics({"val_loss": val_loss})

    return val_loss  


def main():
    mlflow.set_tracking_uri("file:./../invariants/ml/mlruns")  # Oder HTTP URI für Server
    mlflow.set_experiment("invariants-classification")

    mlf_logger = MLFlowLogger(experiment_name="vecsets-classification", tracking_uri="file:./../vecsets/ml/mlruns", run_name="best-model")

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    # Bestes Modell trainieren und speichern
    best_params = study.best_trial.params

    # #from previous runs
    # best_params ={
    #     "lr" : 2.53e-4,
    #     "nhead" : 8,
    #     "num_layers" : 4,
    #     "hidden_size" : 1211,
    #     "dropout" : 0.42,
    #     "fc_layers" : None,
    #     "batch_size" : 64

    # }

    model = VecsetClassifierModule( 
                lr = best_params["lr"],
                input_dim=32, 
                d_model=1024, 
                nhead=best_params["nhead"], 
                num_layers=best_params["num_layers"], 
                num_classes=38, 
                dim_feedforward=best_params["hidden_size"], 
                fc_layers=best_params["fc_layers"],
                dropout=best_params["dropout"], 
                use_pos_embedding=False, 
            )
    
    train_loader, val_loader = get_dataloaders(best_params["batch_size"])

    early_stop_callback = EarlyStopping(
            monitor='val_loss',     # oder "val_acc", je nachdem
            patience=10,             # z. B. 5 Epochen ohne Verbesserung
            mode='min',             # "min" für loss, "max" für accuracy
            verbose=True
        )
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',           # oder z. B. "val_acc"
        save_top_k=1,                 # nur das beste Modell speichern
        mode='min',                   # "min" für loss, "max" für acc
        dirpath=PATHS.DATA_MODELS.as_posix(),
        filename='vecsets_classification',  # Dateinamenformat
        save_weights_only=False,      # speichert komplette Checkpoints
        verbose=True
    )

    trainer = Trainer(max_epochs=500, logger=mlf_logger, callbacks=[early_stop_callback, checkpoint_callback], enable_checkpointing=True, enable_model_summary=False, log_every_n_steps=1)
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()