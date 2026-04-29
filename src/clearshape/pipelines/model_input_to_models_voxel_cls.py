import logging
import warnings

import mlflow
import optuna
import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader

from clearshape.constants import PATHS
from clearshape.dataset import FabwaveDataset
from clearshape.voxels.ml.modules.voxel_classificator import VoxelClassifier

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning_fabric").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


def get_num_classes():
    return len(pd.read_csv(PATHS.DATA_FEATURE / "fabwave_targets.csv")["class_id"].unique())


def get_dataloaders(batch_size):
    train_loader = DataLoader(
        FabwaveDataset(csv_file=PATHS.DATA_MODEL_INPUT / "train.csv", classification=True, data_type="voxel"),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        FabwaveDataset(csv_file=PATHS.DATA_MODEL_INPUT / "validation.csv", classification=True, data_type="voxel"),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, val_loader


def objective(trial):
    with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
        dropout = trial.suggest_float("dropout", 0.0, 0.3)
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 0.00001, 0.1, log=True)
        batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])

        mlflow.log_params({
            "dropout": dropout, "lr": lr, "weight_decay": weight_decay,
            "batch_size": batch_size, "trial_number": trial.number,
        })

        train_loader, val_loader = get_dataloaders(batch_size)
        num_classes = get_num_classes()
        mlflow.log_param("num_classes", num_classes)

        model = VoxelClassifier(num_classes=num_classes, dropout=dropout, lr=lr, weight_decay=weight_decay)

        mlf_logger = MLFlowLogger(
            experiment_name="voxel-classification",
            tracking_uri=PATHS.MLFLOW_TRACKING_URI.as_posix(),
            run_id=mlflow.active_run().info.run_id,
        )

        early_stop = EarlyStopping(monitor='val_f1_score', patience=20, mode='max', verbose=False)

        trainer = Trainer(
            max_epochs=100,
            logger=mlf_logger,
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=True,
            log_every_n_steps=10,
            callbacks=[early_stop],
            accelerator="cuda",
            devices=1,
            precision="bf16-mixed",
            default_root_dir="/tmp",
        )

        trainer.fit(model, train_loader, val_loader)

        val_f1 = trainer.callback_metrics.get("val_f1_score", 0.0)
        val_loss = trainer.callback_metrics.get("val_loss", float('inf'))
        val_acc = trainer.callback_metrics.get("val_acc", 0.0)

        if torch.is_tensor(val_f1):
            val_f1 = val_f1.item()
        if torch.is_tensor(val_loss):
            val_loss = val_loss.item()
        if torch.is_tensor(val_acc):
            val_acc = val_acc.item()

        mlflow.log_metrics({"val_f1_score": val_f1, "val_loss": val_loss, "val_acc": val_acc,
                            "final_epoch": trainer.current_epoch})

        print(f"Trial {trial.number}: F1={val_f1:.4f}, Loss={val_loss:.4f}, Acc={val_acc:.4f}")
        return val_f1


def main():
    mlflow.set_tracking_uri(PATHS.MLFLOW_TRACKING_URI.as_posix())
    mlflow.set_experiment("voxel-classification")

    with mlflow.start_run(run_name="hyperparameter_optimization"):
        print("Starting hyperparameter optimization for Voxel Classification...")
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=30)

        best_params = study.best_trial.params
        best_value = study.best_trial.value
        mlflow.log_params(best_params)
        mlflow.log_metric("best_val_f1_score", best_value)
        print(f"\nBest trial: {study.best_trial.number}")
        print(f"Best F1-Score: {best_value:.4f}")
        print(f"Best parameters: {best_params}")

    print("\nTraining final model with best parameters...")

    with mlflow.start_run(run_name="final_best_model"):
        mlflow.log_params(best_params)
        mlflow.log_param("model_type", "final_best_model")

        num_classes = get_num_classes()
        mlflow.log_param("num_classes", num_classes)

        train_loader, val_loader = get_dataloaders(batch_size=best_params["batch_size"])

        model = VoxelClassifier(
            num_classes=num_classes,
            dropout=best_params["dropout"],
            lr=best_params["lr"],
            weight_decay=best_params["weight_decay"],
        )

        mlf_logger = MLFlowLogger(
            experiment_name="voxel-classification",
            tracking_uri=PATHS.MLFLOW_TRACKING_URI.as_posix(),
            run_id=mlflow.active_run().info.run_id,
        )

        early_stop = EarlyStopping(monitor='val_f1_score', patience=50, mode='max', verbose=False)
        checkpoint = ModelCheckpoint(
            monitor='val_f1_score',
            save_top_k=1,
            mode='max',
            dirpath=PATHS.DATA_MODELS.as_posix(),
            filename='voxels-classifier',
            save_weights_only=False,
            verbose=False,
        )

        trainer = Trainer(
            max_epochs=500,
            logger=mlf_logger,
            callbacks=[early_stop, checkpoint],
            enable_checkpointing=True,
            enable_model_summary=False,
            enable_progress_bar=True,
            log_every_n_steps=10,
            accelerator="cuda",
            devices=1,
            precision="bf16-mixed",
            default_root_dir="/tmp",
        )

        trainer.fit(model, train_loader, val_loader)

        final_metrics = trainer.callback_metrics
        for key, value in final_metrics.items():
            if torch.is_tensor(value):
                mlflow.log_metric(f"final_{key}", value.item())

        best_model_path = checkpoint.best_model_path
        if best_model_path:
            mlflow.log_artifact(best_model_path, "model")
            print(f"Best model saved to: {best_model_path}")

        print("Final training completed!")


if __name__ == "__main__":
    main()
