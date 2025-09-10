# Third party imports
import optuna
import mlflow
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger
import torch
from torch.utils.data import DataLoader, random_split, TensorDataset
import pandas as pd
import numpy as np

# Custom imports
from clearshape.invariants.ml.modules.invs_regressor import InvariantRegressor  # Ihr neues Regressionsmodul
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from clearshape.dataset import FabwaveDataset
from clearshape.constants import PATHS
from clearshape.scaler.custom_scalers import LogScaler  # Anpassen an Ihren Import-Pfad


def get_dataloaders_with_scaler(batch_size):
    """
    DataLoader mit Log-Transformation für Regression
    """
    # Datasets laden
    train_dataset = FabwaveDataset(
        csv_file="/clear-shape/data/5_model_input/train.csv", 
        classification=False,  # Regression!
        regression=True,
        data_type="invariants"
    )
    
    val_dataset = FabwaveDataset(
        csv_file="/clear-shape/data/5_model_input/validation.csv", 
        classification=False,  # Regression!
        regression=True,
        data_type="invariants"
    )
    
    # Log Scaler auf Trainingsdaten fitten
    # Annahme: Dataset gibt (X, y) zurück, wobei y die 4 Targets sind
    train_targets = []
    for i in range(len(train_dataset)):
        _, targets, _ = train_dataset[i]
        train_targets.append(targets.numpy())
    
    train_targets = np.array(train_targets)
    
    # Log Scaler erstellen und fitten
    log_scaler = LogScaler(epsilon=1e-8)
    log_scaler.fit(train_targets)
    
    # Transformierte Datasets erstellen
    train_loader = DataLoader(
        LogTransformDataset(train_dataset, log_scaler, fit_scaler=False), 
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        LogTransformDataset(val_dataset, log_scaler, fit_scaler=False), 
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, val_loader, log_scaler


class LogTransformDataset(torch.utils.data.Dataset):
    """
    Wrapper Dataset für Log-Transformation
    """
    def __init__(self, base_dataset, log_scaler, fit_scaler=False):
        self.base_dataset = base_dataset
        self.log_scaler = log_scaler
        
        # Alle Targets sammeln und transformieren
        self.transformed_data = []
        
        for i in range(len(base_dataset)):
            x, y, metadata = base_dataset[i]
            
            # Y transformieren
            y_np = y.numpy().reshape(1, -1)  # (1, n_targets) für scaler
            if fit_scaler and i == 0:
                y_transformed = self.log_scaler.fit_transform(y_np)
            else:
                y_transformed = self.log_scaler.transform(y_np)
            
            y_transformed = torch.FloatTensor(y_transformed.flatten())
            
            self.transformed_data.append((x, y_transformed, metadata))
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        return self.transformed_data[idx]


def objective(trial):
    """Optuna Objective für Hyperparameter-Tuning"""
    
    # Hyperparameter-Sampling für Regression
    dropout = trial.suggest_float("dropout", 0.05, 0.3)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    
    # Architektur-Parameter
    hidden_size = trial.suggest_int("hidden_size", 64, 1024)
    n_layers = trial.suggest_int("n_layers", 3, 6)
    use_target_heads = trial.suggest_categorical("use_target_heads", [True, False])
    
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
    
    # DataLoader mit Log-Transformation
    train_loader, val_loader, log_scaler = get_dataloaders_with_scaler(batch_size=batch_size)
    
    # Dynamische Layer-Konfiguration
    fc_layers = []
    current_size = hidden_size // 4
    for i in range(n_layers):
        fc_layers.append(current_size)
        if i < n_layers // 2:  # Erste Hälfte: vergrößern
            current_size = min(current_size * 2, hidden_size)
        else:  # Zweite Hälfte: verkleinern
            current_size = max(current_size // 2, hidden_size // 8)
    
    # Regressionsmodell erstellen
    model = InvariantRegressor(
        in_dim=16,
        n_targets=4,  # VOLUME, FACES, EDGES, VERTICES
        fc_layers=fc_layers,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
        use_target_specific_heads=use_target_heads,
        log_scaler=log_scaler,
        target_names=['VOLUME', 'FACES', 'EDGES', 'VERTICES']
    )
    
    # MLflow Logger
    mlf_logger = MLFlowLogger(
        experiment_name="invariants-regression", 
        tracking_uri="file:./../invariants/ml/mlruns"
    )
    
    # Early Stopping für Regression
    early_stop_callback = EarlyStopping(
        monitor='val_mse_original',  # MSE auf ursprünglicher Skala
        patience=30,
        mode='min',
        verbose=True
    )
    
    trainer = Trainer(
        max_epochs=200,
        logger=mlf_logger,
        enable_checkpointing=False,
        enable_model_summary=False,
        log_every_n_steps=1,
        callbacks=[early_stop_callback]
    )
    
    trainer.fit(model, train_loader, val_loader)
    
    # Metriken extrahieren
    val_mse = trainer.callback_metrics.get("val_mse_original", float('inf')).item()
    val_mae = trainer.callback_metrics.get("val_mae_original", float('inf')).item()
    val_loss = trainer.callback_metrics.get("val_loss", float('inf')).item()
    
    # Hyperparameter und Metriken loggen
    mlf_logger.log_hyperparams(trial.params)
    mlf_logger.log_metrics({
        "val_mse_original": val_mse,
        "val_mae_original": val_mae,
        "val_loss": val_loss
    })
    
    # Wir minimieren MSE auf ursprünglicher Skala
    return val_mse


def main():
    """Hauptfunktion für Hyperparameter-Tuning und finales Training"""
    
    mlflow.set_tracking_uri("file:./../invariants/ml/mlruns")
    mlflow.set_experiment("invariants-regression")
    
    # Optuna Study
    study = optuna.create_study(direction="minimize")  # Minimiere MSE
    study.optimize(objective, n_trials=50)  # Weniger Trials für Regression
    
    print("="*50)
    print("BEST HYPERPARAMETERS:")
    print("="*50)
    for key, value in study.best_trial.params.items():
        print(f"{key}: {value}")
    print(f"Best MSE: {study.best_trial.value:.6f}")
    print("="*50)
    
    # Bestes Modell mit finalen Parametern trainieren
    mlf_logger = MLFlowLogger(
        experiment_name="invariants-regression", 
        tracking_uri="file:./../invariants/ml/mlruns", 
        run_name="best-regression-model"
    )
    
    best_params = study.best_trial.params
    
    # DataLoader für finales Training
    train_loader, val_loader, log_scaler = get_dataloaders_with_scaler(
        batch_size=best_params["batch_size"]
    )
    
    # Beste Architektur rekonstruieren
    hidden_size = best_params["hidden_size"]
    n_layers = best_params["n_layers"]
    
    fc_layers = []
    current_size = hidden_size // 4
    for i in range(n_layers):
        fc_layers.append(current_size)
        if i < n_layers // 2:
            current_size = min(current_size * 2, hidden_size)
        else:
            current_size = max(current_size // 2, hidden_size // 8)
    
    # Finales Modell
    model = InvariantRegressor(
        in_dim=16,
        n_targets=4,
        fc_layers=fc_layers,
        dropout=best_params["dropout"],
        lr=best_params["lr"],
        weight_decay=best_params["weight_decay"],
        use_target_specific_heads=best_params["use_target_heads"],
        log_scaler=log_scaler,
        target_names=['VOLUME', 'FACES', 'EDGES', 'VERTICES']
    )
    
    # Callbacks für finales Training
    early_stop_callback = EarlyStopping(
        monitor='val_mse_original',
        patience=50,  # Mehr Geduld für finales Training
        mode='min',
        verbose=True
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_mse_original',
        save_top_k=1,
        mode='min',
        dirpath=PATHS.DATA_MODELS.as_posix(),
        filename='invariants-regressor-{epoch:02d}-{val_mse_original:.6f}',
        save_weights_only=False,
        verbose=True
    )
    
    # Finaler Trainer
    trainer = Trainer(
        max_epochs=500,  # Mehr Epochs für finales Training
        logger=mlf_logger,
        callbacks=[early_stop_callback, checkpoint_callback],
        enable_checkpointing=True
    )
    
    # Training
    trainer.fit(model, train_loader, val_loader)
    
    # Log Scaler separat speichern (wichtig für Deployment!)
    import joblib
    scaler_path = PATHS.DATA_MODELS / "log_scaler.joblib"
    joblib.dump(log_scaler, scaler_path)
    
    print("="*50)
    print("TRAINING COMPLETE!")
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")
    print(f"Log scaler saved at: {scaler_path}")
    print("="*50)


def evaluate_model():
    """Separate Funktion für Modell-Evaluation auf Test-Set"""
    
    # Bestes Modell und Scaler laden
    import joblib
    from pathlib import Path
    
    # Pfade anpassen
    model_path = "path/to/best/model.ckpt"  # Aus Training
    scaler_path = PATHS.DATA_MODELS / "log_scaler.joblib"
    
    log_scaler = joblib.load(scaler_path)
    model = InvariantRegressor.load_from_checkpoint(model_path, log_scaler=log_scaler)
    
    # Test Dataset
    test_dataset = FabwaveDataset(
        csv_file="/clear-shape/data/5_model_input/test.csv", 
        classification=False,
        data_type="invariants"
    )
    
    test_loader = DataLoader(
        LogTransformDataset(test_dataset, log_scaler, fit_scaler=False),
        batch_size=512,
        shuffle=False
    )
    
    # Evaluation
    trainer = Trainer()
    results = trainer.test(model, test_loader)
    
    print("="*50)
    print("TEST RESULTS:")
    print("="*50)
    for metric, value in results[0].items():
        print(f"{metric}: {value:.6f}")
    print("="*50)


if __name__ == "__main__":
    main()
    
    # Optional: Test-Evaluation
    # evaluate_model()