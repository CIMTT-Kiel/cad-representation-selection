# standard library imports
import logging
import pickle
import warnings

# third-party imports
import mlflow
import optuna
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader

# custom imports
from clearshape.constants import PATHS
from clearshape.dataset import FabwaveDataset
from clearshape.voxels.ml.modules.voxel_regressor import VoxelRegressor

# Setup logging 
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning_fabric").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


def get_scaler():
    with open(PATHS.DATA_MODEL_INPUT / "log_scaler.pkl", "rb") as f:
        return pickle.load(f)


class LogTransformDataset(torch.utils.data.Dataset):
    """Wraps FabwaveDataset and applies log-scale transformation to regression targets."""

    def __init__(self, base_dataset, log_scaler):
        self.transformed_data = []
        for i in range(len(base_dataset)):
            x, y, meta = base_dataset[i]
            y_transformed = torch.FloatTensor(
                log_scaler.transform(y.numpy().reshape(1, -1)).flatten()
            )
            self.transformed_data.append((x, y_transformed, meta))

    def __len__(self):
        return len(self.transformed_data)

    def __getitem__(self, idx):
        return self.transformed_data[idx]


def get_dataloaders(batch_size, log_scaler):
    train_loader = DataLoader(
        LogTransformDataset(
            FabwaveDataset(csv_file=PATHS.DATA_MODEL_INPUT / "train.csv", regression=True, data_type="voxel"),
            log_scaler,
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        LogTransformDataset(
            FabwaveDataset(csv_file=PATHS.DATA_MODEL_INPUT / "validation.csv", regression=True, data_type="voxel"),
            log_scaler,
        ),
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
        use_target_heads = trial.suggest_categorical("use_target_heads", [True, False])
        batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])

        mlflow.log_params({
            "dropout": dropout, "lr": lr, "weight_decay": weight_decay,
            "use_target_heads": use_target_heads, "batch_size": batch_size,
            "trial_number": trial.number,
        })

        log_scaler = get_scaler()
        train_loader, val_loader = get_dataloaders(batch_size, log_scaler)

        model = VoxelRegressor(
            n_targets=4,
            dropout=dropout,
            use_target_specific_heads=use_target_heads,
            lr=lr,
            weight_decay=weight_decay,
            log_scaler=log_scaler,
            target_names=['VOLUME', 'FACES', 'EDGES', 'VERTICES'],
        )

        mlf_logger = MLFlowLogger(
            experiment_name="voxel-regression",
            tracking_uri=PATHS.MLFLOW_TRACKING_URI.as_posix(),
            run_id=mlflow.active_run().info.run_id,
        )

        early_stop = EarlyStopping(monitor='val_mse_original', patience=20, mode='min', verbose=False)

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

        val_mse = trainer.callback_metrics.get("val_mse_original", float('inf'))
        val_mae = trainer.callback_metrics.get("val_mae_original", float('inf'))
        val_loss = trainer.callback_metrics.get("val_loss", float('inf'))

        if torch.is_tensor(val_mse):
            val_mse = val_mse.item()
        if torch.is_tensor(val_mae):
            val_mae = val_mae.item()
        if torch.is_tensor(val_loss):
            val_loss = val_loss.item()

        mlflow.log_metrics({"val_mse_original": val_mse, "val_mae_original": val_mae,
                            "val_loss": val_loss, "final_epoch": trainer.current_epoch})

        print(f"Trial {trial.number}: MSE={val_mse:.4f}, MAE={val_mae:.4f}")
        return val_mse


def main():
    mlflow.set_tracking_uri(PATHS.MLFLOW_TRACKING_URI.as_posix())
    mlflow.set_experiment("voxel-regression")

    with mlflow.start_run(run_name="hyperparameter_optimization"):
        print("Starting hyperparameter optimization for Voxel Regression...")
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=30)

        best_params = study.best_trial.params
        best_value = study.best_trial.value
        mlflow.log_params(best_params)
        mlflow.log_metric("best_val_mse_original", best_value)
        print(f"\nBest trial: {study.best_trial.number}")
        print(f"Best MSE: {best_value:.4f}")
        print(f"Best parameters: {best_params}")

    print("\nTraining final model with best parameters...")

    with mlflow.start_run(run_name="final_best_model"):
        mlflow.log_params(best_params)
        mlflow.log_param("model_type", "final_best_model")

        log_scaler = get_scaler()
        train_loader, val_loader = get_dataloaders(best_params["batch_size"], log_scaler)

        model = VoxelRegressor(
            n_targets=4,
            dropout=best_params["dropout"],
            use_target_specific_heads=best_params["use_target_heads"],
            lr=best_params["lr"],
            weight_decay=best_params["weight_decay"],
            log_scaler=log_scaler,
            target_names=['VOLUME', 'FACES', 'EDGES', 'VERTICES'],
        )

        mlf_logger = MLFlowLogger(
            experiment_name="voxel-regression",
            tracking_uri=PATHS.MLFLOW_TRACKING_URI.as_posix(),
            run_id=mlflow.active_run().info.run_id,
        )

        early_stop = EarlyStopping(monitor='val_mse_original', patience=50, mode='min', verbose=False)
        checkpoint = ModelCheckpoint(
            monitor='val_mse_original',
            save_top_k=1,
            mode='min',
            dirpath=PATHS.DATA_MODELS.as_posix(),
            filename='voxels-regressor',
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


def evaluate_model(model_path=None, scaler_path=None):
    with mlflow.start_run(run_name="model_evaluation"):
        if model_path is None:
            model_path = PATHS.DATA_MODELS / "voxels-regressor.ckpt"
        if scaler_path is None:
            scaler_path = PATHS.DATA_MODEL_INPUT / "log_scaler.pkl"

        mlflow.log_param("model_path", str(model_path))

        log_scaler = get_scaler()
        model = VoxelRegressor.load_from_checkpoint(model_path, log_scaler=log_scaler)

        test_dataset = FabwaveDataset(
            csv_file=PATHS.DATA_MODEL_INPUT / "test.csv", regression=True, data_type="voxel"
        )
        test_loader = DataLoader(
            LogTransformDataset(test_dataset, log_scaler),
            batch_size=4,
            shuffle=False,
            num_workers=0,
        )

        mlf_logger = MLFlowLogger(
            experiment_name="voxel-regression",
            tracking_uri=PATHS.MLFLOW_TRACKING_URI.as_posix(),
            run_id=mlflow.active_run().info.run_id,
        )

        trainer = Trainer(logger=mlf_logger, enable_progress_bar=True,
                          accelerator="cuda", devices=1, precision="bf16-mixed",
                          default_root_dir="/tmp")
        results = trainer.test(model, test_loader)

        if results:
            for key, value in results[0].items():
                mlflow.log_metric(f"test_{key}", value)

        print("Model evaluation completed!")
        return results


if __name__ == "__main__":
    main()
